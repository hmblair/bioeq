from __future__ import annotations
from typing import Generator, Any
import itertools
import os
import torch
import torch.nn as nn
import dgl
import wigners as wi
import sphericart.torch as sc
from sphecerix import tesseral_wigner_D as wigner_D
from scipy.spatial.transform import Rotation as scipy_r
from enum import Enum


class MoleculeType(Enum):
    PROTEIN = 0
    RNA = 1
    DNA = 2


TM_SCORE_FACTOR = {
    MoleculeType.PROTEIN: lambda x: 1.24 * (x - 15) ** (1/3) - 1.8,
    MoleculeType.RNA: lambda x: 0.6 * (x - 0.5) ** (1/3) - 2.5,
}

POINT_CLOUD_DIM = 3
FEATURE_DIM = -2
REPR_DIM = -1
KERNEL = os.environ.get("BIOEQ_KERNEL", "0") == "1"


#
# Alignment and scoring
#


def get_kabsch_rotation_matrix(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Get the rotation matrix in SO(n) that best aligns the point cloud x with the
    point cloud y using the Kabsch algorithm. If y is None, then the point cloud
    x is aligned with the coordinate axes, which is equivalent to ... 

    Both x and y should be tensors of shape (..., N, 3). If weight is not None,
    then it should be a tensor of shape (..., N) containing the weights for each
    point in the point cloud.
    """

    # Calculate the covariance matrix
    if weight is not None:
        C = torch.einsum(
            '...ji,...j,...jk->...ik',
            x, weight, y,
        ) / weight.mean(-1)
    else:
        C = torch.einsum('...ji,...jk->...ik', x, y)
    # Calculate the singular value decomposition
    U, _, Vh = torch.linalg.svd(C)
    # Get the correction factor and apply it to the last column of V
    d = torch.linalg.det(U) * torch.linalg.det(Vh)
    Vh[:, -1] = Vh[:, -1] * d[..., None]
    # Calculate the optimal rotation matrix
    return U @ Vh


def kabsch_align(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align the point clouds in x with the point clouds in y using the 
    Kabsch algorithm, which finds the rotation matrix that aligns the two point
    clouds, such that the root mean square deviation between the two point
    clouds is minimized. The point clouds are also centered at the origin.

    The point clouds should be tensors of shape (..., N, 3), where N is the number
    of points in each point cloud. If weight is not None, then it should be a
    tensor of shape (..., N) containing the weights for each point in the point
    cloud.
    """

    # Center the point clouds
    x = x - x.mean(-2, keepdim=True)
    y = y - y.mean(-2, keepdim=True)
    # Get the optimal rotation matrix
    R = get_kabsch_rotation_matrix(x, y, weight)
    # Apply the rotation matrix to the first point cloud, and return it,
    # along with the zero-centered second point cloud
    return x @ R, y


class KabschLoss(nn.Module):
    """
    The Kabsch loss function, which calculates the mean square deviation
    between two point clouds after applying the Kabsch algorithm to optimally
    align the point clouds. The weight tensor is used during alignment only.
    """

    def __init__(
        self: KabschLoss,
        weight: torch.Tensor | None = None,
    ) -> None:

        super().__init__()
        self.weight: torch.Tensor | None
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(
        self: KabschLoss,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the mean square deviation between two point clouds, after
        applying the Kabsch algorithm to optimally align the point clouds.
        """

        # Align the point clouds using the Kabsch algorithm
        input, target = kabsch_align(input, target, self.weight)
        # Calculate the mean square deviation
        return ((input - target) ** 2).mean()


def tm_score(
    x: torch.Tensor,
    y: torch.Tensor,
    d0: float,
) -> torch.Tensor:
    """
    Compute the TM-score between two aligned point clouds x and y, which should 
    be tensors of shape (..., N, 3). The result is a tensor of shape (...,).
    """

    # Compute the squared distances between the points in the point clouds
    d = ((x - y) ** 2).sum(-1)
    # Compute the TM-score
    return (1 / (1 + d / (d0 ** 2))).mean(-1)


class TMScore(nn.Module):
    """
    For computing the TM score between two point clouds.
    """

    def __init__(
        self: TMScore,
        mtype: MoleculeType,
    ) -> None:

        self.normf = TM_SCORE_FACTOR[mtype]

    def forward(
        self: TMScore,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the TM score between the two point clouds after alignment.
        """

        # Align the molecules
        input, target = kabsch_align(input, target)
        # Get the normalisation factor
        factor = self.normf(input.size(0))
        # Compute the TM score
        return tm_score(input, target, factor)


def frame_align(
    x: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """
    Take a point cloud x in (R^d)^N and corresponding orientations R in SO(d)^N, and
    align the point cloud at each point to the global frame using the rotation 
    matrix at that point. The point cloud is also centered at the origin.
    """

    # Center the point cloud
    x = x - x.mean(-2, keepdim=True)
    # Invert the rotation matrices by transposing them
    R_inv = R.transpose(-2, -1)
    # align the point cloud at each point separately
    return torch.einsum('...lij,...kj->...lki', R_inv, x)


def get_local_frame(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the map f : R^6 -> SO(3) which is the inverse of the immersion of
    SO(3) into R^6 described in the 2019 paper On the Continuity of Rotation
    Representations in Neural Networks. The map applies the Gram-Schmidt
    process to the two vectors in the input tensor and computes the cross product
    of the resulting vectors to get the third vector.
    """

    # Reshape the input tensor and split it into two vectors
    x = x.view(*x.shape[:-1], 3, 2)
    x1 = x[..., :, 0]
    x2 = x[..., :, 1]
    # Normalize the first vector
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    # Subtract the projection of the first vector onto the second vector
    x2 = x2 - x1 * (x1 * x2).sum(dim=-1, keepdim=True)
    # Normalize the second vector
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # Get the third vector by taking the cross product of the first two
    x3 = torch.cross(x1, x2, dim=-1)
    # Stack the vectors to get the local frame
    return torch.stack([x1, x2, x3], dim=-2)


#
# Representation theory
#


class Irrep:
    """
    A helper class for getting the dimensions of the real
    irreducible representations of SO(3).
    """

    def __init__(
        self: Irrep,
        l: int,
        mult: int = 1,
    ) -> None:

        # Store the degree and the multiplicity
        self.l = l
        self.mult = mult

    def __eq__(
        self: Irrep,
        other: Any,
    ) -> bool:
        """
        Check if the representations have the same degree and multiplicity.
        """

        if not isinstance(other, Irrep):
            return False
        return self.l == other.l and self.mult == other.mult

    def dim(
        self: Irrep,
    ) -> int:
        """
        Return the dimension of the representation.
        """

        return 2 * self.l + 1

    def offset(
        self: Irrep,
    ) -> int:
        """
        Return the offset of this representation from zero.
        """

        return sum(
            Irrep(l).dim()
            for l in range(0, self.l)
        )

    def rot(
        self: Irrep,
        axis: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        """
        Return the Wigner-D matrix for this irrep with the given
        axis and angle.
        """

        # Get the scipy rotation object
        rot = scipy_r.from_rotvec(axis.numpy() * angle)
        # Compute the corresponding Wigner-D matrix
        return torch.tensor(
            wigner_D(self.l, rot),
            dtype=torch.float32,
        )

    def toreal(
        self: Irrep,
    ) -> torch.Tensor:
        """
        Get the converstion matrix from the complex to real
        irreducible representation of SO(3).
        """

        # Get the division factor
        SQRT2 = 2 ** -0.5
        # Initialise the conversion matrix
        q = torch.zeros(
            self.dim(),
            self.dim(),
            dtype=torch.complex128,
        )
        # Fill in the negative degrees
        for m in range(-self.l, 0):
            q[self.l + m, self.l + abs(m)] = SQRT2
            q[self.l + m, self.l - abs(m)] = -1j * SQRT2
            q[self.l, self.l] = 1
        # Fill in the positive degrees
        for m in range(1, self.l + 1):
            q[self.l + m, self.l + abs(m)] = (-1)**m * SQRT2
            q[self.l + m, self.l - abs(m)] = 1j * (-1)**m * SQRT2
        # Fill in the zero degree
        q = (-1j)**self.l * q
        return q

    def __str__(
        self: Irrep,
    ) -> str:
        """
        Return some useful info.
        """

        return f"Irrep\n    degree: {self.l}\n    multiplicity: {self.mult}."


class ProductIrrep:
    """
    A helper class for getting the decomposition of a tensor
    product of two real irreducible representations.
    """

    def __init__(
        self: ProductIrrep,
        rep1: Irrep,
        rep2: Irrep,
    ) -> None:

        self.rep1 = rep1
        self.rep2 = rep2

        self.lmin = abs(rep1.l - rep2.l)
        self.lmax = rep1.l + rep2.l

        self.reps = [
            Irrep(l)
            for l in range(self.lmin, self.lmax + 1)
        ]

    def dim(
        self: ProductIrrep,
    ) -> int:
        """
        Return the dimension of the representation.
        """
        return sum(rep.dim() for rep in self.reps)

    def maxdim(
        self: ProductIrrep,
    ) -> int:
        """
        Return the maximum dimension of all irreps in the representation.
        """
        return max(rep.dim() for rep in self.reps)

    def cumdims(
        self: ProductIrrep,
    ) -> list[int]:
        """
        Get a list of the cumulative dimensions of the
        irreducible representations in the tensor product.
        """

        return [
            sum(rep.dim() for rep in self.reps[:l])
            for l in range(self.nreps()+1)
        ]

    def offset(
        self: ProductIrrep,
    ) -> int:
        """
        Return the offset of this representation from zero.
        """
        return sum(
            Irrep(l).dim()
            for l in range(0, self.lmin)
        )

    def nreps(
        self: ProductIrrep,
    ) -> int:
        """
        Return the number of irreducible representations.
        """
        return len(self.reps)

    def coupling(
        self: ProductIrrep,
    ) -> torch.Tensor:
        """
        Return the coupling coefficient for the real irreducible
        decomposition of this tensor product. The terms correspdonding
        to different irreps are placed in separate dimensions.
        """

        # Initialise an array to store the coefficient
        coeff = torch.zeros(
            self.dim(),
            self.rep1.dim(),
            self.nreps(),
            self.rep2.dim(),
        )

        # Get two of the conversion matrices
        Q1 = self.rep1.toreal()
        Q2 = self.rep2.toreal()

        repnum = 0
        for rep, cdim in zip(self.reps, self.cumdims()):

            # Get the complex coupling coefficient
            c_coeff = torch.Tensor(
                wi.clebsch_gordan_array(
                    rep.l,
                    self.rep1.l,
                    self.rep2.l,
                )
            ).to(torch.complex128)

            # renormalize the coefficient
            c_coeff /= rep.dim()

            # Get the third conversion matrix
            Q = rep.toreal()

            # Convert to the real coefficient
            coeff[cdim:cdim+rep.dim(), :, repnum, :] = torch.einsum(
                'ij,kl,mn,ikm->jln',
                Q, Q1, torch.conj(Q2), c_coeff
            ).real
            # Advance the rep number
            repnum += 1

        # return the coefficient as a 32-bit float
        return coeff.to(torch.float32)

    def low_rank_coupling(
        self: ProductIrrep,
    ) -> torch.Tensor:
        """
        Return the low-rank coupling coefficient for the real
        irreducible decomposition of this tensor product.
        """

        # In the low-rank coupling, the coupling only occurs
        # if rep1 == rep2
        if self.rep1.l == self.rep2.l:
            return torch.eye(self.rep1.dim())
        else:
            return torch.zeros(
                self.rep1.dim(),
                self.rep2.dim(),
            )

    def __str__(
        self: ProductIrrep,
    ) -> str:
        """
        Return some useful info.
        """
        return f"A tensor product of irreps of degrees {self.rep1.l} and {self.rep2.l}."


class Repr:
    """
    Collect together a group of irreducible representations into
    a single representation.
    """

    def __init__(
        self: Repr,
        lvals: list[int],
        mult: int = 1,
    ) -> None:

        self.irreps = [
            Irrep(l, mult)
            for l in lvals
        ]
        self.lvals = [
            irrep.l
            for irrep in self.irreps
        ]
        self.mult = mult

    def nreps(
        self: Repr,
    ) -> int:
        """
        Return the number of irreducible representations.
        """
        return len(self.irreps)

    def __iter__(
        self: Repr,
    ) -> Generator[Irrep]:
        """
        Iterate over the irreducible representations.
        """
        yield from self.irreps

    def __eq__(
        self: Repr,
        other: Any,
    ) -> bool:
        """
        Check if the representations have the same degrees. The multiplicity
        is not checked.
        """
        if not isinstance(other, Repr):
            return False
        return self.irreps == other.irreps

    def dim(
        self: Repr,
    ) -> int:
        """
        Get the dimension of the representation as the sum
        of the irreducible representations.
        """

        return sum(irrep.dim() for irrep in self)

    def lmax(
        self: Repr,
    ) -> int:
        """
        Get the largest degree of all irreps in this
        representation.
        """

        return max(irrep.l for irrep in self)

    def cumdims(
        self: Repr,
    ) -> list[int]:
        """
        Get a list of the cumulative dimensions of the
        irreducible representations.
        """

        return [
            sum(rep.dim() for rep in self.irreps[:l])
            for l in range(self.nreps()+1)
        ]

    def offsets(
        self: Repr,
    ) -> list[int]:
        """
        Return the offset of each representation from zero.
        """

        return [
            rep.offset()
            for rep in self
        ]

    def indices(
        self: Repr,
    ) -> list[int]:
        """
        Return a list of size self.dim() containing the zero-based index
        to which each dimension corresponds.
        """

        return [
            repnum
            for repnum, irrep in enumerate(self)
            for _ in range(irrep.dim())
        ]

    def rot(
        self: Repr,
        axis: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        """
        Return the Wigner-D matrix for this representation with the given
        axis and angle.
        """

        # Initialse an empty array
        rot = torch.zeros(
            self.dim(),
            self.dim(),
        )
        # Fill in using the rotations of all irreps
        for irrep, cdim in zip(self, self.cumdims()):
            rot[
                cdim: cdim + irrep.dim(),
                cdim: cdim + irrep.dim(),
            ] = irrep.rot(axis, angle)
        return rot

    def __str__(
        self: Repr,
    ) -> str:
        """
        Return some useful info.
        """
        return f"A representation with degrees " + ', '.join([str(rep.l) for rep in self.irreps[:-1]]) + f', and {self.irreps[-1].l}.'


class ProductRepr:
    """
    Represent the tensor product of two representations, and
    allow for the computing of the irreducible decomposition.
    """

    def __init__(
        self: ProductRepr,
        rep1: Repr,
        rep2: Repr,
    ) -> None:

        self.rep1 = rep1
        self.rep2 = rep2

        self.reps = [
            ProductIrrep(irrep1, irrep2)
            for irrep1 in rep1
            for irrep2 in rep2
        ]
        self.offsets = itertools.product(
            rep1.cumdims()[:-1],
            rep2.cumdims()[:-1],
        )

    def dim(
        self: ProductRepr,
    ) -> int:
        """
        Get the dimension of the representation as the sum
        of the irreducible representations.
        """

        return sum(rep.dim() for rep in self.reps)

    def __eq__(
        self: ProductRepr,
        other: Any,
    ) -> bool:
        """
        Check if both reps are equal.
        """

        if not isinstance(other, ProductRepr):
            return False
        return self.rep1 == other.rep1 and self.rep2 == other.rep2

    def lmax(
        self: ProductRepr,
    ) -> int:
        """
        Get the largest degree of all irreps in this
        representation.
        """

        return max(rep.lmax for rep in self.reps)

    def maxdim(
        self: ProductRepr,
    ) -> int:
        """
        Get the dimension of the largest irreducible
        representation tensor product in the decomposition.
        """

        return max(rep.dim() + rep.lmin ** 2 for rep in self.reps)

    def nreps(
        self: ProductRepr,
    ) -> int:
        """
        Return the number of irreducible representations
        appearing in the tensor product.
        """
        return sum(rep.nreps() for rep in self.reps)

    def coupling(
        self: ProductRepr,
    ) -> torch.Tensor:
        """
        Return the coupling coefficient for the real irreducible
        decomposition of this tensor product.
        """

        if KERNEL:
            # Initialise an array to store the coefficient
            coeff = torch.zeros(
                self.maxdim(),
                self.rep1.dim(),
                self.lmax() + 1,
                self.rep2.dim(),
            )
            # Loop over the tensor products of the irreducible
            # representations, and fill in the corresponding
            # coupling coefficient
            for rep, (rep1o, rep2o) in zip(self.reps, self.offsets):
                coeff[
                    rep.offset(): rep.offset() + rep.dim(),
                    rep1o: rep1o + rep.rep1.dim(),
                    :rep.nreps(),
                    rep2o: rep2o + rep.rep2.dim(),
                ] = rep.coupling()

        else:
            # Initialise an array to store the coefficient
            coeff = torch.zeros(
                self.maxdim(),
                self.rep1.dim(),
                self.nreps(),
                self.rep2.dim(),
            )

            # Loop over the tensor products of the irreducible
            # representations, and fill in the corresponding
            # coupling coefficient
            repnum = 0
            for rep, (o1, o2) in zip(self.reps, self.offsets):
                coeff[
                    rep.offset(): rep.offset() + rep.dim(),
                    o1: o1 + rep.rep1.dim(),
                    repnum: repnum + rep.nreps(),
                    o2: o2 + rep.rep2.dim(),
                ] = rep.coupling()
                repnum += rep.nreps()

        # Reshape the coefficient for easier matrix multiplication with
        # spherical harmonics
        return coeff.view(
            self.maxdim(), -1
        ).contiguous()


#
# Equivariant maps and helper functions
#


class RepNorm(nn.Module):
    """
    For getting the norms of a spherical tensor associated with
    a given representation.
    """

    def __init__(
        self: RepNorm,
        repr: Repr,
    ) -> None:

        super().__init__()
        # Get the number of representations and their offsets
        self.nreps = repr.nreps()
        self.cdims = repr.cumdims()

    def forward(
        self: RepNorm,
        st: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take a spherical tensor and return the norm of each irrep.
        """

        # Initialse a tensor to store the norms
        norms = torch.zeros(
            st.shape[:-1] + (self.nreps,),
            device=st.device,
        )
        # Compute the norm of the representation of each degree
        for i in range(self.nreps):
            norms[..., i] = st[
                ..., self.cdims[i]:self.cdims[i+1],
            ].norm(dim=REPR_DIM)
        return norms


class SphericalHarmonic(nn.Module):
    """
    Given a set of coordinates, computes the values of the spherical harmonics
    at those coordinates up to the given maximum degree.
    """

    def __init__(
        self: SphericalHarmonic,
        lmax: int,
        normalized: bool = True,
    ) -> None:

        super().__init__()
        # Create the spherical harmonic calculator
        self.sh = sc.SphericalHarmonics(lmax, normalized)
        # Store the maximum degree
        self.lmax = lmax
        # Get the indices for permuting the input
        self.ix = torch.tensor(
            [2, 0, 1],
            dtype=torch.int64,
        )

    def forward(
        self: SphericalHarmonic,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take coordinates of shape (..., 3) and return the spherical harmonic
        features of shape (..., (lmax + 1) ** 2 ). Any nan values,
        which correspond to the case where the two points are the same, are set
        to zero.
        """

        # Flatten the batch and point dimensions
        *b, n, _ = x.shape
        x = x.view(-1, 3)
        # In order for the degree-1 spherical harmonics to be the identity
        # function (and for proper equivariance to hold), we need to permute
        # the coordinates
        x = x[:, self.ix]
        # Sphericart only supports float32 and float64
        half = False
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
            half = True
        # Compute and keep only the features for the requested degrees
        sh = self.sh.compute(x)
        # Convert the features back to half precision if necessary
        if half:
            sh = sh.half()
        # Remove any nan's that come from input zeros
        sh = torch.nan_to_num(sh, nan=0.0)
        # Return the features in the original shape
        return sh.view(*b, n, -1)

    def edgewise(
        self: SphericalHarmonic,
        x: torch.Tensor,
        graph: dgl.DGLGraph,
    ) -> torch.Tensor:
        """
        Compute the spherical harmonic features for the edgewise relative
        positions between points in the point cloud.
        """

        # Get the source and destination of each edge
        U, V = graph.edges()
        # Compute the edgewise relative positions
        x_pairwise = x[U] - x[V]
        # Compute the spherical harmonic features
        sh = self(x_pairwise)
        # Set the features corresponding to the same point to zero
        return torch.nan_to_num(sh, nan=0.0)


class EquivariantBasis(nn.Module):
    """
    Compute a basis for the space of equivariant maps from R^3 into the
    coupling space of the ProductRepr.
    """

    def __init__(
        self: EquivariantBasis,
        repr: ProductRepr,
    ) -> None:

        super().__init__()
        # Get the coupling coefficient for the product representation
        coupling = repr.coupling()
        self.register_buffer('coupling', coupling)
        if KERNEL:
            self.outdims = (
                repr.rep1.dim(),
                repr.rep2.dim() * (repr.lmax() + 1),
            )
        else:
            self.outdims = (
                repr.rep1.dim(),
                repr.rep2.dim() * repr.nreps(),
            )
        # Initialise the spherical harmonic calculator
        self.sh = SphericalHarmonic(repr.lmax())
        self.repr = repr

    def forward(
        self: EquivariantBasis,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take a set of equivariant edge features of shape (..., n, 3) and return
        the equivariant weight matrix W of shape (..., n, d_in, d_out, lmax).
        """

        # Get the spherical harmonic features
        sh = self.sh(x)
        # Reshape for multiplication
        *b, maxdim = sh.size()
        sh_r = sh.view(-1, maxdim)
        # Multiply and reshape for multiplication with the
        # input features
        return (sh_r @ self.coupling).view(*b, *self.outdims)

    def edgewise(
        self: EquivariantBasis,
        x: torch.Tensor,
        graph: dgl.DGLGraph,
    ) -> torch.Tensor:
        """
        The forward method applied to the relative position of nodes for
        each edge in the input graph.
        """

        # Get the spherical harmonic features
        sh = self.sh.edgewise(x, graph)
        # Reshape for multiplication
        *b, maxdim = sh.size()
        sh_r = sh.view(-1, maxdim)
        # Multiply and reshape for multiplication with the
        # input features in the convolution step
        return (sh_r @ self.coupling).view(*b, *self.outdims)


class EquivariantBases(nn.Module):
    """
    A variant of EquivariantBasis which uses the same spherical harmonic
    features to compute multiple equivariant maps for multiple ProductReps.
    Even if a long list of ProductReps is passed, this function only computes
    each unique matrix once.
    """

    def __init__(
        self: EquivariantBases,
        *reprs: ProductRepr,
    ) -> None:

        super().__init__()
        # To avoid redundant operations, we only keep the unique ProductReps
        # and keep track of the indices of each one
        self.unique_reprs = []
        self.repr_ix = []
        repr_count = -1
        for repr in reprs:
            if repr not in self.unique_reprs:
                self.unique_reprs.append(repr)
                repr_count += 1
            self.repr_ix.append(repr_count)

        # Get the maximum maxdim and lmax
        maxdim = max(repr.maxdim() for repr in self.unique_reprs)
        lmax = max(repr.lmax() for repr in self.unique_reprs)
        # Create matrices to store the expanded coupling coefficients
        self.couplings = [
            torch.zeros(
                maxdim,
                repr.rep1.dim(),
                repr.nreps(),
                repr.rep2.dim(),
            ).view(maxdim, -1)
            for repr in self.unique_reprs
        ]
        # Fill in the matrices
        for repr, coupling in zip(self.unique_reprs, self.couplings):
            coupling[:repr.maxdim()] = repr.coupling()
        # Store the output dimensions
        self.outdims = [(
            repr.rep1.dim(),
            repr.rep2.dim() * repr.nreps(),
        ) for repr in self.unique_reprs]
        # Initialise the spherical harmonic calculator
        self.sh = SphericalHarmonic(lmax)

    def _mm(
        self: EquivariantBases,
        sh: torch.Tensor,
        coupling: torch.Tensor,
        outdims: tuple[int, int],
    ) -> torch.Tensor:
        """
        A single of the reqiured matrix multiplication steps.
        """

        # Reshape for multiplication
        *b, maxdim = sh.size()
        sh_r = sh.view(-1, maxdim)
        # Multiply and reshape for multiplication with the
        # input features in the convolution step
        return (sh_r @ coupling).view(*b, *outdims)

    def forward(
        self: EquivariantBases,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Take a set of equivariant edge features of shape (..., n, 3) and return
        the equivariant weight matrix W of shape (..., n, d_in, d_out, lmax).
        """

        # Get the spherical harmonic features
        sh = self.sh(x)
        # Matrix multiplication
        ms = [
            self._mm(sh, coupling, outdims)
            for coupling, outdims in zip(self.couplings, self.outdims)
        ]
        # Expand to the required number of outputs without a new calculation
        return tuple(ms[ix] for ix in self.repr_ix)

    def edgewise(
        self: EquivariantBases,
        x: torch.Tensor,
        graph: dgl.DGLGraph,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward method applied to the relative position of nodes for
        each edge in the input graph.
        """

        # Get the spherical harmonic features
        sh = self.sh.edgewise(x, graph)
        # Matrix multiplication
        ms = [
            self._mm(sh, coupling, outdims)
            for coupling, outdims in zip(self.couplings, self.outdims)
        ]
        # Expand to the required number of outputs without a new calculation
        return tuple(ms[ix] for ix in self.repr_ix)
