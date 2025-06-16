from __future__ import annotations
from typing import Generator, Any
import itertools
import os
import torch
import torch.nn as nn
import dgl
import wigners as wi
import sphericart.torch as sc
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
# Simple geometric utils
#

def unit_dot_product(
    coordinates: torch.Tensor
) -> torch.Tensor:
    """
    Take a set of coordinates of shape (n, 3, 3), treating the last as the
    coordinate dim, and compute the unit dot product between the difference
    vectors.
    """

    # Get the difference vectors
    diff = coordinates[:, 1:] - coordinates[:, :-1]
    # Normalise the differences
    diff_unit = diff / torch.linalg.norm(
        diff,  dim=-1, keepdim=True
    )
    # Compute the dot product
    return (diff_unit[:, 0] * diff_unit[:, 1]).sum(-1)


def torsion_unit_dot_product(
    coordinates: torch.Tensor
) -> torch.Tensor:
    """
    Take a set of coordinates of shape (n, 4, 3), treating the last as the
    coordinate dim, and compute the torsion angle associated with the
    triplet.
    """

    # Compute bond vectors
    b = coordinates[:, 1:] - coordinates[:, :-1]
    # Compute the perpendicular vectors to the planes formed by b1, b2
    # and b2, b3
    n1 = torch.cross(b[:, 0], b[:, 1], dim=-1)
    n2 = torch.cross(b[:, 1], b[:, 2], dim=-1)
    # Compute the unit vectors of n1 and n2
    n1_unit = n1 / n1.norm(dim=-1, keepdim=True)
    n2_unit = n2 / n2.norm(dim=-1, keepdim=True)
    # Compute the torsion angle via dot product
    return (n1_unit * n2_unit).sum(dim=-1)


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
    x is aligned with the coordinate axes.
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


def _complete_subgraphs(sizes: torch.Tensor) -> dgl.DGLGraph:
    """
    Create a graph with sum(sizes) nodes consisting of complete subgraphs of the given sizes.
    """

    src = []
    dst = []

    start = 0
    for size in sizes:

        nodes = torch.arange(start, start + size)
        src_tensor, dst_tensor = torch.meshgrid(nodes, nodes, indexing='ij')

        ix = torch.triu_indices(src_tensor.shape[0], src_tensor.shape[1], 1)

        complete_src = src_tensor[ix[0], ix[1]].flatten()
        complete_dst = dst_tensor[ix[0], ix[1]].flatten()

        src.append(complete_src)
        dst.append(complete_dst)

        start += size

    src = torch.cat(src)
    dst = torch.cat(dst)

    return dgl.graph((src, dst))


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
        # The datatypes used internally
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex128

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

    def mvals(
        self: Irrep,
    ) -> list[int]:
        """
        Return a list of the m quantum numbers associated with this
        irrep, sorted in decreasing order.
        """

        return [m for m in range(-self.l, self.l + 1)]

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

    def raising(
        self: Irrep,
    ) -> torch.Tensor:
        """
        Get the raising operator associated with this irrep.
        """

        j = self.l
        m = torch.arange(-j, j)
        return torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

        mvals = torch.tensor(self.mvals())
        # Initialise an empty array of the correct size
        raising = torch.zeros(
            (self.dim(), self.dim())
        ).to(self.complex_dtype)
        # Fill in the off-diagonal
        for m in mvals[:-1]:
            raising[
                self.l - m, self.l - m - 1
            ] = self.l*(self.l+1) - m*(m+1)
        return torch.sqrt(raising)

    def lowering(
        self: Irrep,
    ) -> torch.Tensor:
        """
        Get the lowering operator associated with this irrep.
        """

        j = self.l
        m = torch.arange(-j + 1, j + 1)
        return torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)
        return self.raising().t().conj()

    def _generators(
        self: Irrep,
    ) -> torch.Tensor:
        """
        Return the image of the three generators of so(3) under the
        irrep specified by this object.
        """

        # Get the raising and lowering operators
        raising = self.raising()
        lowering = self.lowering()
        # Compute the x and y generators
        genx = 0.5 * (raising + lowering)
        geny = -0.5j * (raising - lowering)
        # Get the z generator as a diagonal matrix
        mvals = torch.tensor(
            self.mvals(),
            dtype=self.complex_dtype,
        )
        genz = 1j * torch.diag(mvals)
        # Stack the results along the first dimension
        gens = torch.stack(
            [genx, genz, geny],
            dim=0,
        )
        # Convert from complex to real representations
        Q = self.toreal()
        out = (Q.t().conj() @ gens @ Q)
        return out.real.to(self.real_dtype)

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
        q[self.l, self.l] = 1
        # Scale by the correct complex factor
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

    def __str__(
        self: ProductIrrep,
    ) -> str:
        """
        Return some useful info.
        """
        return f"A tensor product of irreps of degrees {self.rep1.l} and {self.rep2.l}."


class Repr(nn.Module):
    """
    Collect together a group of irreducible representations into
    a single representation.
    """

    def __init__(
        self: Repr,
        lvals: list[int] = [1],
        mult: int = 1,
    ) -> None:

        super().__init__()
        self.irreps = [
            Irrep(l, mult)
            for l in lvals
        ]
        self.lvals = [
            irrep.l
            for irrep in self.irreps
        ]
        self.mult = mult
        # Pre-compute the so(3) generators and reshape for downstream
        # multiplications
        self.register_buffer(
            'generators',
            self._generators().view(3, -1)
        )
        self.perm = self._reorder_generators()

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

    def verify(
        self: Repr,
        st: torch.Tensor,
    ) -> bool:
        """
        Check if the given spherical tensor has dimensions matchinig this
        representation.
        """

        correct_mult = st.size(-2) == self.mult
        correct_dim = st.size(-1) == self.dim()
        return correct_mult and correct_dim

    def find_scalar(
        self: Repr,
    ) -> tuple[int, list]:
        """
        Find all degree-zero representations and their locations.
        """

        nreps = 0
        locs = []
        cumdim = 0
        for repr in self.irreps:
            if repr.l == 0:
                nreps += 1
                locs.append(cumdim)
            cumdim += repr.dim()

        return nreps, locs

    def _generators(
        self: Repr,
    ) -> torch.Tensor:
        """
        Return the image of the three generators of so(3) under the
        irrep specified by this object.
        """

        # Initialise an appropriate array
        NUM_GENS = 3
        gens = torch.zeros(
            NUM_GENS, self.dim(), self.dim(),
        )
        # Fill in using the generators of each irrep
        cumdim = 0
        for irrep in self.irreps:
            gens[
                ...,
                cumdim: cumdim + irrep.dim(),
                cumdim: cumdim + irrep.dim(),
            ] = irrep._generators()
            cumdim += irrep.dim()
        # Rearrange so that all rank-1 terms appear first
        return gens

    def _reorder_generators(
        self: Repr,
    ) -> torch.Tensor:
        """
        Get a permutation matrix which will re-order the rotation
        generators so that all rank-1 and rank-2 equivariant matrices
        are contiguous.
        """

        # Initialise a trivial permutation matrix
        p = torch.eye(self.dim())
        perm1 = []
        perm2 = []
        ix = 0
        for irrep in self.irreps:
            perm1.append(
                ix + irrep.l
            )
            for jx in range(irrep.l):
                perm2.append(
                    ix + jx
                )
                perm2.append(
                    ix + irrep.dim() - jx - 1
                )
            ix += irrep.dim()

        perm = perm1 + perm2

        return p[perm]

    def rot(
        self: Repr,
        axis: torch.Tensor,
        angle: torch.Tensor,
        perm: bool = False,
    ) -> torch.Tensor:
        """
        Return the Wigner-D matrix for this irrep with the given
        axis and angle.
        """

        # Weight the generators of so(3) by the axis values
        *b, _ = axis.size()
        gens = (axis @ self.generators).view(
            *b, self.dim(), self.dim(),
        )
        # Multiply by the angle and exponentiate to move to SO(3)
        rot = torch.linalg.matrix_exp(
            angle[..., None, None] * gens
        )
        if perm:
            rot = rot @ self.perm.t()
        # Zero out any nan values, which come from invalid axes
        # TODO: we don't zero out the degree-0 features.
        return torch.nan_to_num(rot, 0.0)

    def basis(
        self: Repr,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the equivariant basis elements associated with the input points.
        """

        p = torch.tensor(
            [0, 1, 0],
            device=x.device,
            dtype=x.dtype,
        )
        # Find the axis to rotate about
        axis = torch.linalg.cross(x, p[None, :])
        axis = axis / torch.linalg.norm(
            axis, dim=-1, keepdim=True,
        )
        # Find the angle to rotate by
        angle = -torch.arccos(
            x[:, 1] / torch.linalg.norm(x, dim=-1)
        )
        # Compute the basis via the Wigner matrices
        return self.rot(axis, angle)

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
            dtype=st.dtype,
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
        * b, n, _ = x.shape
        x = x.view(-1, 3)
        # In order for the degree-1 spherical harmonics to be the identity
        # function (and for proper equivariance to hold), we need to permute
        # the coordinates
        x = x[:, self.ix]
        # Sphericart only supports float32 and float64
        dtype = x.dtype
        SPHERICART_DTYPES = [torch.float32, torch.float64]
        if dtype not in SPHERICART_DTYPES:
            x = x.to(torch.float32)
        # Compute and keep only the features for the requested degrees
        sh = self.sh.compute(x)
        # Convert the features back to their original precision
        sh = sh.to(dtype)
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
    Compute the equivariant weight matrices of shape (2l2+1)x(2l1+1).
    """

    def __init__(
        self: EquivariantBasis,
        repr: ProductRepr,
    ) -> None:

        super().__init__()
        self.outdims1 = (
            repr.rep1.dim(),
            repr.rep1.nreps(),
        )
        self.outdims2 = (
            repr.rep2.nreps(),
            repr.rep2.dim(),
        )
        # Initialise the spherical harmonic calculator
        self.sh = SphericalHarmonic(repr.lmax())
        self.repr = repr

    def forward(
        self: EquivariantBasis,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Take a set of equivariant edge features of shape (..., n, 3) and return
        the equivariant weight matrix W of shape (..., n, d_in, d_out, lmax).
        """

        # Get the spherical harmonic features
        sh = self.sh(x)

        # Set the features corresponding to the same point to zero
        sh = torch.nan_to_num(sh, nan=0.0)

        coeff1 = torch.zeros(x.size(0), *self.outdims1)
        coeff2 = torch.zeros(x.size(0), *self.outdims2)
        for j in range(self.repr.rep1.nreps()):
            low = j ** 2
            high = (j + 1) ** 2
            coeff1[..., low:high, j] = sh[..., low:high]

        for j in range(self.repr.rep2.nreps()):
            low = j ** 2
            high = (j + 1) ** 2
            coeff2[..., j, low:high] = sh[..., low:high]

        return coeff1, coeff2

    def edgewise(
        self: EquivariantBasis,
        x: torch.Tensor,
        graph: dgl.DGLGraph,
    ) -> torch.Tensor:
        """
        The forward method applied to the relative position of nodes for
        each edge in the input graph.
        """

        U, V = graph.edges()
        return self(x[U] - x[V])


class EquivariantBases(nn.Module):

    def __init__(
        self: EquivariantBases,
        *reprs: ProductRepr,
    ) -> None:

        super().__init__()

        # To avoid redundant operations, we only keep the unique ProductReps
        # and keep track of the indices of each one
        self.unique_reprs = []
        self.comps = []
        self.repr_ix = []
        repr_count = -1
        for repr in reprs:
            if repr not in self.unique_reprs:
                self.unique_reprs.append(repr)
                self.comps.append(EquivariantBasis(repr))
                repr_count += 1
            self.repr_ix.append(repr_count)

    def forward(
        self: EquivariantBases,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:

        # Compute the individual basis elements
        ms = [comp(x) for comp in self.comps]
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

        U, V = graph.edges()
        return self(x[U] - x[V])


class RadialBasisFunctions(nn.Module):
    """
    Compute one or more radial basis functions.
    """

    def __init__(
        self: RadialBasisFunctions,
        num_functions: int,
    ) -> None:

        super().__init__()
        # Create the mus and the sigmas
        self.mu = nn.Parameter(
            torch.randn(num_functions),
            requires_grad=True,
        )
        self.sigma = nn.Parameter(
            torch.randn(num_functions),
            requires_grad=True,
        )

    def forward(
        self: RadialBasisFunctions,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass each element of x though all the radial basis functions.
        """

        # Compute the exponent
        exp = (x[..., None] - self.mu) * self.sigma ** 2
        # Compute the radial basis function
        return torch.exp(-exp ** 2) * self.sigma
