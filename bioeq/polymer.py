from __future__ import annotations
from functools import cache
from bioeq._cpp import _connectedSubgraphs
from typing import Any, Union, Callable, Self
from numpy import poly
import torch
import torch.nn as nn
import torch.utils.data as tdata
import dgl
from torch_scatter import (
    scatter_sum as t_scatter_sum,
    scatter_mean as t_scatter_mean,
    scatter_max as t_scatter_max,
    scatter_min as t_scatter_min,
)
from .data import (
    StructureDataset,
    to_cif,
    padded_cumsum
)
from copy import copy
from ._index import (
    ATOM_PAIR_TO_ENUM,
    ALL_PAIR_TO_ENUM,
    TRIPLET_TO_ENUM,
    QUAD_TO_ENUM,
    Element,
    Donors,
    Acceptors,
    RibonucleicAcid,
    VALENCE,
    Residue,
    Property,
    Reduction,
    PARTIAL_CHARGE,
    VdW_SIGMA,
    VdW_EPSILON,
    COULOMB,
    BETA,
    Ribose,
    Adenine,
    Adenosine,
    Guanosine,
    Cytidine,
    Uridine,
)
from .geom import (
    unit_dot_product,
    torsion_unit_dot_product,
    _complete_subgraphs,
)


def lazyproperty(fn: Callable):
    """
    A property decorator that evaluates lazily if the corresponding private
    variable is None.
    """

    @property
    def wrapper(self):
        # Generate the private attribute name from the function name
        private_name = f"_{fn.__name__}"
        # Check if the private attribute is None, and if so, call the function to set it
        if getattr(self, private_name, None) is None:
            setattr(self, private_name, fn(self))
        # Return the cached value
        return getattr(self, private_name)

    return wrapper


ATOM_DIM = 0


def t_scatter_collate(
    features: torch.Tensor,
    indices: torch.Tensor,
    dim: int,
) -> list[torch.Tensor]:
    """
    Return a list of tensors, containing all values corresponding to index

    """

    return [
        features[indices == ix]
        for ix in range(indices.max() + 1)
    ]


REDUCTIONS = {
    Reduction.NONE: lambda features, indices, dim: features,
    Reduction.COLLATE: t_scatter_collate,
    Reduction.MEAN: t_scatter_mean,
    Reduction.SUM: t_scatter_sum,
    Reduction.MIN: t_scatter_min,
    Reduction.MAX: t_scatter_max,
}
_Reduction = Union[
    torch.Tensor,
    tuple[torch.Tensor, torch.LongTensor],
    list[torch.Tensor],
]


HBOND_DIST_CUTOFF = 2.5
HBOND_ANGLE_CUTOFF = -0.5


def allequal(*x: Any) -> bool:
    """
    Check if all input values are equal.
    """
    return len(set(x)) <= 1


class Polymer:
    """
    Store all purely spatial information about a polymer.
    """

    def __init__(
        self: Polymer,
        molecule_ids: list[str],
        chain_ids: list[str],
        coordinates: torch.Tensor,
        elements: torch.Tensor,
        residues: torch.Tensor,
        atom_names: torch.Tensor,
        graph: dgl.DGLGraph,
        residue_sizes: torch.Tensor,
        chain_sizes: torch.Tensor,
        molecule_sizes: torch.Tensor,
    ) -> None:

        # Store all attributes
        self.molecule_ids = molecule_ids
        self.chain_ids = chain_ids
        self.coordinates = coordinates
        self.elements = elements
        self.residues = residues
        self.graph = graph
        # self.graph = graph
        self.residue_sizes = residue_sizes.long()
        self.chain_sizes = chain_sizes.long()
        self.molecule_sizes = molecule_sizes.long()
        self.atom_names = atom_names
        # Keep track of the device of the Polymer
        self.device = coordinates.device
        # Store the number and size of each group
        self._nums = {
            Property.ATOM: self.num_atoms,
            Property.RESIDUE: self.num_residues,
            Property.CHAIN: self.num_chains,
            Property.MOLECULE: self.num_molecules,
            Property.ALL: 1,
        }
        self._sizes = {
            Property.ATOM: torch.ones(self.num_atoms).long(),
            Property.RESIDUE: self.residue_sizes,
            Property.CHAIN: self.chain_sizes,
            Property.MOLECULE: self.molecule_sizes,
            Property.ALL: torch.tensor([self.num_atoms]).long(),
        }
        self._atom_properties = {
            Property.ELEMENT: self.elements,
            Property.NAME: self.atom_names,
        }

        self._bonds = torch.stack(
            self.bonds(), dim=1,
        )
        # All the variables below are lazily computed if needed
        self._bond_types = None
        self._charge = None
        self._hops = None

    @property
    def num_edges(
        self: Polymer,
    ) -> int:
        """
        The number of edges in the polymer.
        """

        return self.graph.num_edges()

    @property
    def num_atoms(
        self: Polymer,
    ) -> int:
        """
        The number of atoms in the polymer.
        """

        return self.coordinates.size(0)

    @property
    def num_residues(
        self: Polymer,
    ) -> int:
        """
        The number of atoms in the polymer.
        """

        return self.residue_sizes.size(0)

    @property
    def num_chains(
        self: Polymer,
    ) -> int:
        """
        The number of atoms in the polymer.
        """

        return self.chain_sizes.size(0)

    @property
    def num_molecules(
        self: Polymer,
    ) -> int:
        """
        The number of atoms in the polymer.
        """

        return self.molecule_sizes.size(0)

    def split(
        self: Polymer,
        prop: Property
    ) -> tuple[Polymer, ...]:
        """
        Split the polymer based on the given property.
        """

        return tuple(
            self.select(torch.tensor([j]), prop)
            for j in range(self._nums[prop])
        )

    def bonds(
        self: Polymer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the edges of the underlying graph.
        """

        return self.graph.edges()

    def _bins(
        self: Polymer,
        prop: Property,
    ) -> torch.Tensor:
        """
        Return bin edges corresponding to each [scale].
        """

        if prop == Property.ELEMENT:
            return self.elements.unique()
        if prop == Property.NAME:
            return self.atom_names.unique()
        return padded_cumsum(
            self._sizes[prop]
        )

    def _hist(
        self: Polymer,
        indices: torch.Tensor,
        prop: Property,
    ) -> torch.Tensor:
        """
        Count the number of indices associated with each property.
        """

        bins = self._bins(prop)
        num, _ = torch.histogram(
            indices.to(torch.float32),
            bins.to(torch.float32),
        )
        return num.long()

    def num_bonds(
        self: Polymer,
        prop: Property,
        multiple: bool = False,
    ) -> torch.Tensor:
        """
        Count the number of bonds beginning from or terminating in each
        different property. If multiple is True, then double bounds are
        counted as two.
        """

        return self._hist(
            self._bonds.flatten(),
            prop,
        )

    @cache
    def subgraphs(
        self: Polymer,
        num_atoms: int,
    ) -> torch.Tensor:
        """
        Find all connected subgraphs with exactly the prescribed number
        of atoms.
        """

        # Call into the C++ function
        return _connectedSubgraphs(
            self._bonds,
            num_atoms,
        )

    def angles(
        self: Polymer,
    ) -> _Reduction:
        """
        Compute all bond angles in the polymer.
        """

        # Find all triples of atoms
        triples = self.subgraphs(3)
        # Get the corresponding coordinates
        triple_coords = self.coordinates[triples]
        # Compute the angles
        return unit_dot_product(triple_coords)

    def torsions(
        self: Polymer,
    ) -> _Reduction:
        """
        Compute all torsion angles in the polymer.
        """

        # Find all quadruplets of atoms
        quads = self.subgraphs(4)
        # Get the corresponding coordinates
        quad_coords = self.coordinates[quads]
        # Compute the angles
        return torsion_unit_dot_product(quad_coords)

    @cache
    def subgraph_types(
        self: Polymer,
        num_atoms: int,
    ) -> torch.Tensor:
        """
        Enumate all subgraphs in the polymer and associate each one
        with a unique integer based on the atom types in the subgraph.
        """

        # Get all subgraphs in the polymer
        sg = self.subgraphs(num_atoms)
        # Get the corresponding atom names
        names = self.atom_names[sg]
        # Get a unique index per triplet
        if num_atoms == 2:
            return ATOM_PAIR_TO_ENUM[
                *[names[:, i] for i in range(2)]
            ]
        elif num_atoms == 3:
            return TRIPLET_TO_ENUM[
                *[names[:, i] for i in range(3)]
            ]
        elif num_atoms == 4:
            return QUAD_TO_ENUM[
                *[names[:, i] for i in range(4)]
            ]
        else:
            raise ValueError("must be 3 or 4.")

    def _masked_size(
        self: Polymer,
        mask: torch.Tensor,
        prop: Property,
    ) -> torch.Tensor:
        """
        Get the number of atoms of each instance of the given property
        which are not masked by the mask.
        """

        # Get the sizes by reducing the mask
        sizes = self.reduce(
            mask.long(),
            prop,
            Reduction.SUM,
        )
        # Remove all zero-sized [scale]s
        nonzero = sizes > 0
        return nonzero, sizes[nonzero]

    def __getitem__(
        self: Polymer,
        ix: torch.Tensor,
    ) -> Polymer:
        """
        Return a polymer containing only the selected atoms.
        """

        # Get the size of each scale post-masking
        _, residue_sizes = self._masked_size(
            ix, Property.RESIDUE
        )
        chain_ix, chain_sizes = self._masked_size(
            ix, Property.CHAIN
        )
        molecule_ix, molecule_sizes = self._masked_size(
            ix, Property.MOLECULE
        )
        # Get the subgraph
        graph = dgl.node_subgraph(self.graph, ix)
        # Subset all relevant attributes
        coordinates = self.coordinates[ix]
        elements = self.elements[ix]
        residues = self.residues[ix]
        atom_names = self.atom_names[ix]
        molecule_ids = self.molecule_ids[molecule_ix.numpy()]
        chain_ids = self.chain_ids[chain_ix.numpy()]
        # Create and return a new Polymer object with the subsetted data
        return Polymer(
            molecule_ids,
            chain_ids,
            coordinates,
            elements,
            residues,
            atom_names,
            graph,
            residue_sizes,
            chain_sizes,
            molecule_sizes,
        )

    def expand(
        self: Polymer,
        features: torch.Tensor,
        prop: Property,
    ) -> torch.Tensor:
        """
        Expand the features so that there is one per atom, rather than
        one per [scale].
        """

        return features.repeat_interleave(
            self._sizes[prop], dim=0
        )

    def _mask(
        self: Polymer,
        ix: torch.Tensor,
        prop: Property,
    ) -> torch.Tensor:
        """
        Create a mask to select only the atoms with property contained
        in the provided indices.
        """

        if ix.dtype == torch.bool:
            ix = ix.nonzero().squeeze()

        # Get which [scale] each atom belongs to
        obj_ix = self.indices(prop)
        # Find any matches to the given indices
        return (obj_ix[:, None] == ix[None, :]).any(1)

    def select(
        self: Polymer,
        ix: torch.Tensor,
        prop: Property,
    ) -> Polymer:
        """
        Get all atoms where the value of the provided property is
        contained in the given indices.
        """

        return self[self._mask(ix, prop)]

    def indices(
        self: Polymer,
        prop: Property,
    ) -> torch.Tensor:
        """
        Compute an index tensor of shape (self.num_atoms,), indicating
        the value of the property at each atom.
        """

        if prop in self._nums:
            return torch.arange(
                self._nums[prop],
                device=self._sizes[prop].device,
            ).repeat_interleave(self._sizes[prop])
        return self._atom_properties[prop]

    def reduce(
        self: Polymer,
        features: torch.Tensor,
        prop: Property,
        rtype: Reduction = Reduction.MEAN,
    ) -> _Reduction:
        """
        Reduce the input values within each copy of the given object.
        MIN and MAX reductions return the indices too. A COLLATE
        reduction instead returns a list of tensors containing the
        values aligning with each specific index.
        """

        # Get the indices of the given scale and reduce
        return REDUCTIONS[rtype](
            features,
            self.indices(prop),
            dim=ATOM_DIM,
        )

    def sreduce(
        self: Polymer,
        features: torch.Tensor,
        num_atoms: int,
        rtype: Reduction = Reduction.MEAN,
    ) -> _Reduction:
        """
        Reduce among all subgraphs in the polymer with the given number
        of atoms.
        """

        # Reduce the features over the subgraphs
        return REDUCTIONS[rtype](
            features,
            self.subgraph_types(num_atoms),
            dim=ATOM_DIM,
        )

    def aggregate(
        self: Polymer,
        features: torch.Tensor,
        rtype: Reduction = Reduction.MEAN,
    ) -> torch.Tensor:
        """
        Aggregate the input features to the destination node.
        """

        # Reduce the features over the subgraphs
        U, _ = self.bonds()
        return REDUCTIONS[rtype](
            features,
            U,
            dim=ATOM_DIM,
        )

    def hbond(
        self: Polymer,
    ) -> torch.Tensor:
        """
        Add hydrogen bonds to the underlying graph.
        """

        Utot, Vtot = self.graph.edges()
        all_hydrogens = []
        all_acceptors = []

        sizes = padded_cumsum(
            self.molecule_sizes
        ).tolist()

        low = 0
        high = 0
        for s in range(self.num_molecules):

            print(s)

            low += sizes[s]
            high += sizes[s+1]

            U = Utot[low:high]
            V = Vtot[low:high]

            # Find all donor-hydrogen pairs
            donor_ix = (
                self.elements[U, None] == Donors.index()[None, :]
            ).any(1)
            hydrogen_ix = (
                self.elements[V] == Element.H.value
            )
            pair_ix = donor_ix & hydrogen_ix

            donors = U[pair_ix]
            hydrogens = V[pair_ix]

            # Find all acceptors
            acceptors = (
                self.elements[:, None] == Acceptors.index()[None, :]
            ).any(1).nonzero().squeeze()

            # Get all acceptors within an acceptable range
            pdist = torch.cdist(
                self.coordinates[hydrogens],
                self.coordinates[acceptors],
            )
            acc_pairs = (pdist < HBOND_DIST_CUTOFF).nonzero()

            donors = donors[acc_pairs[:, 0]]
            hydrogens = hydrogens[acc_pairs[:, 0]]
            acceptors = acceptors[acc_pairs[:, 1]]

            vec1 = self.coordinates[donors] - self.coordinates[hydrogens]
            vec2 = self.coordinates[acceptors] - self.coordinates[hydrogens]
            vec1 = vec1 / torch.linalg.norm(vec1, dim=-1, keepdim=True)
            vec2 = vec2 / torch.linalg.norm(vec2, dim=-1, keepdim=True)

            cos_ang = (vec1 * vec2).sum(-1)
            angle_ix = (cos_ang < HBOND_ANGLE_CUTOFF)

            all_hydrogens.append(
                hydrogens[angle_ix]
            )
            all_acceptors.append(
                acceptors[angle_ix]
            )

        return torch.cat(all_hydrogens), torch.cat(all_acceptors)

    def causal_mask(
        self: Polymer,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Create an edge mask so that information can only flow forward
        through the polymer, and not backward.
        """

        # Push the residue indices to the edges
        U, V = self.bonds()
        indices = self.indices(Property.RESIDUE)
        residue_U, residue_V = indices[U], indices[V]
        # Create a causal mask
        if reverse:
            return residue_U >= residue_V
        else:
            return residue_U <= residue_V

    def bond_mask(
        self: Polymer,
        mask: torch.Tensor,
        prop: Property,
    ) -> torch.Tensor:
        """
        Mask all information flowing from the specified objects.
        """

        # Push the indices to the edges
        U, _ = self.bonds()
        indices = self.indices(prop)
        residue_U = indices[U]
        # Get the nodes participating in the masking
        mask_U = torch.nonzero(mask).squeeze(1)
        # Find all edges beginning in a masked residue
        edge_mask = (residue_U[:, None] == mask_U[None, :])
        # It is generally faster to take the logical not at the end,
        # since the unmaksed residues will outnumber the masked
        # residues.
        return torch.logical_not(
            edge_mask.any(1)
        )

    def seqdist(
        self: Polymer,
        max: int,
    ) -> torch.Tensor:
        """
        Return a 1D tensor of length self.num_edges containing the
        residue-wise relative distance in the sequence between the source
        and destination of each edge.
        """

        # Get the source and destination nodes of each edge
        U, V = self.bonds()
        # Compute the distance in the sequence
        indices = self.indices(Property.RESIDUE)
        relpos = indices[V] - indices[U]
        # Clamp to the maximum distance
        return torch.clamp(
            relpos, -max, max
        ) + max

    def center(
        self: Self,
        prop: Property = Property.MOLECULE,
    ) -> Self:
        """
        Center the coordinates of each of the chosen object in the molecule.
        """

        # Get the coordinate means
        coord_means = self.reduce(
            self.coordinates,
            prop,
        )
        # Expand the means and subtract
        coord_means = self.expand(coord_means, prop)
        coordinates = self.coordinates - coord_means
        # Create a new Polymer with the centered coordinates
        centered_polymer = copy(self)
        centered_polymer.coordinates = coordinates
        return centered_polymer

    def relpos(
        self: Polymer,
    ) -> torch.Tensor:
        """
        Return the relative positions of each pair of nodes connected by an
        edge.
        """

        # Expand the coordinates to the edges
        exp_coords = self.coordinates[self._bonds]
        # Compute the relative positions
        return exp_coords[:, 0] - exp_coords[:, 1]

    def pdist(
        self: Polymer,
    ) -> torch.Tensor:
        """
        Get the pairwise distances between all coordinates which are
        connected by an edge.
        """

        # Compute the pairwise norms
        return torch.linalg.norm(
            self.relpos(),
            dim=-1,
        )

    def complete(self: Polymer, hops: int = 0) -> Polymer:
        """
        Connect all atoms in the same molecule with edges.
        """

        _graph = _complete_subgraphs(self.molecule_sizes)
        if (hops > 0):
            shortest_dist = self.shortest_distance()
            U, V = _graph.edges()
            mask = shortest_dist[U, V] > hops
            _graph = dgl.graph((U[mask], V[mask]))

        return Polymer(
            self.molecule_ids,
            self.chain_ids,
            self.coordinates,
            self.elements,
            self.residues,
            self.atom_names,
            _graph,
            self.residue_sizes,
            self.chain_sizes,
            self.molecule_sizes,
        )

    def _pc(
        self: Polymer,
        prop: Property,
    ) -> torch.Tensor:
        """
        Compute the principal components of all atoms for each instance
        of the given property.

        Since principal components are only defined up to sign,
        the outputs of this function may be unstable with respect to
        the coordinates. See the .align method for getting stable
        principal components.
        """

        # Get the coordinate covariance matrices at the given scale
        cov = polymer_covariance(
            self, self, prop=prop,
        )
        # Compute the eigenvectors of the covariance matrices
        _, Q = torch.linalg.eigh(cov)
        return Q.transpose(1, 2)

    def moment(
        self: Polymer,
        n: int,
        prop: Property,
    ) -> torch.Tensor:
        """
        Return the n-th (uncentered) moment of the coordinates of the
        polymer at the given scale.
        """

        return self.reduce(
            self.coordinates ** n,
            prop,
        )

    def rotate(
        self: Polymer,
        R: torch.Tensor,
    ) -> Polymer:
        """
        Rotate the coordinates of the polymer
        """

        out = copy(self)
        out.coordinates = out.coordinates @ R
        return out

    def align(
        self: Polymer,
        prop: Property,
    ) -> tuple[Polymer, torch.Tensor]:
        """
        Align the objects such that their covariance matrix is diagonal.
        Also return the principal components used to align the objects.
        """

        # Center and get the principal components
        al_polymer = self.center(prop)
        Q = al_polymer._pc(prop)
        # Expand the components and use them to rotate the coordinates
        Q_exp = al_polymer.expand(Q, prop)
        al_polymer.coordinates = (
            Q_exp @ al_polymer.coordinates[..., None]
        ).squeeze()

        # To ensure stability and uniqueness, we modify the coordinates
        # and Q so that the largest two third moments are positive. The
        # third is chosen such that the system is right-handed.
        signs = al_polymer.moment(3, prop).sign()
        # The smallest eigenvalue is always the first
        signs[:, 0] = signs[:, 1] * signs[:, 2] * torch.linalg.det(Q)
        signs_exp = al_polymer.expand(signs, prop)
        # Modify the signs and return
        al_polymer.coordinates = (
            al_polymer.coordinates * signs_exp
        )
        Q = Q * signs[..., None]
        return al_polymer, Q

    def steric_anomaly(
        self: Polymer,
        ideal_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the mean-squared error between the ideal and true bond
        lengths.
        """

        # Get the true bond lengths
        true = self.pdist()
        # Get the ideal bond lengths
        ideal = ideal_lengths[
            self.bond_types
        ]
        # Compute the mean-squared difference
        anomaly = torch.mean(
            (true - ideal) ** 2
        )
        return anomaly

    def connect(
        self: Polymer,
        num_neighbours: int,
    ) -> None:
        """
        Replace the existing graph with a knn graph based on the current
        coordinates.
        """

        # Call into the dgl engine
        self.graph = dgl.segmented_knn_graph(
            self.coordinates,
            num_neighbours,
            self.molecule_sizes.tolist(),
        )
        # Update the bonds
        U, V = self.bonds()
        self._bonds = torch.stack(
            [U, V], dim=1,
        )

    def to(
        self: Polymer,
        dest: str | torch.device | torch.dtype,
    ) -> Polymer:
        """
        Move all tensors to the given device.
        """

        # Move the coordinates to the object
        coordinates = self.coordinates.to(dest)
        # If the object is not a new dtype, then also move all the
        # indices to the object too
        graph = self.graph
        elements = self.elements
        residues = self.residues
        atom_names = self.atom_names
        residue_sizes = self.residue_sizes
        chain_sizes = self.chain_sizes
        molecule_sizes = self.molecule_sizes

        if not isinstance(dest, torch.dtype):
            elements = elements.to(dest)
            residues = residues.to(dest)
            atom_names = atom_names.to(dest)
            graph = graph.to(dest)
            residue_sizes = residue_sizes.to(dest)
            chain_sizes = chain_sizes.to(dest)
            molecule_sizes = molecule_sizes.to(dest)

        return self.__class__(
            coordinates,
            elements,
            residues,
            atom_names,
            graph,
            residue_sizes,
            chain_sizes,
            molecule_sizes,
        )

    def element_names(
        self: Polymer,
    ) -> list[str]:
        """
        Get a list of the names of the element of each atom.
        """

        elem_names = Element.revdict()
        return [
            elem_names[ix]
            for ix in self.elements.cpu().tolist()
        ]

    def sequence(
        self: Polymer,
    ) -> str:
        """
        Return the sequence of residues in the polymer.
        """

        # For now, we have to reduce the residues to get one per
        # residue
        residues = self.reduce(
            self.residues,
            Property.RESIDUE,
            Reduction.MEAN
        )

        residue_strs = Residue.revdict()
        return "".join([
            residue_strs[ix]
            for ix in residues.tolist()
        ])

    @lazyproperty
    def charge(
        self: Polymer,
    ) -> torch.Tensor:
        """
        Return the partial charges on each atom.
        """

        return PARTIAL_CHARGE[self.atom_names]

    def _coulomb(
        self: Polymer,
        rpd: torch.Tensor,
        prop: Property = Property.MOLECULE,
        params: torch.Tensor = PARTIAL_CHARGE,
    ) -> torch.Tensor:
        """
        Return the Coulomb energy of the molecule using the partial charges.
        """

        DIELECTRIC = 1 / 6

        # Get the partial charges
        pc = params[self.atom_names]
        # Compute the Coulomb energy
        energy = rpd * pc[:, None] * pc[None, :]
        # Zero out any atoms which are too close in bond-space
        # energy[self.hops < 4] = 0
        # Sum over all atoms in one dimension
        energy = energy.sum(-1)
        # Reduce to get the energy contributions
        return DIELECTRIC * COULOMB * self.reduce(
            energy,
            prop,
            Reduction.SUM,
        )

    def _coulomb_grad(
        self: Polymer,
        rpd: torch.Tensor,
        params: torch.Tensor = PARTIAL_CHARGE,
    ) -> torch.Tensor:
        """
        Return the Coulomb energy of the molecule using the partial charges.
        """

        DIELECTRIC = 1 / 6

        # Get the pairwise differenes
        diff = self.coordinates[:, None] - self.coordinates[None, :]
        # Get the partial charges
        pc = params[self.atom_names]
        # Compute the Coulomb energy
        energy = -2 * diff * (rpd[..., None] ** 3) * \
            pc[:, None, None] * pc[None, :, None]
        # Zero out any atoms which are too close in bond-space
        energy[self.hops < 4] = 0
        # Sum over all atoms in one dimension
        return DIELECTRIC * COULOMB * energy.sum(1)

    def _coulomb_hess(
        self: Polymer,
        rpd: torch.Tensor,
        params: torch.Tensor = PARTIAL_CHARGE,
    ) -> torch.Tensor:
        """
        Return the Coulomb energy of the molecule using the partial charges.
        """

        DIELECTRIC = 1 / 6

        # Get the pairwise differenes
        diff = self.coordinates[:, None] - self.coordinates[None, :]
        # Get the partial charges
        pc = params[self.atom_names]
        # Compute the Coulomb energy
        energy = -2 * (rpd[..., None] ** 3 - 3 * diff ** 2 * rpd[..., None] ** 5) * \
            pc[:, None, None] * pc[None, :, None]
        # Zero out any atoms which are too close in bond-space
        energy[self.hops < 4] = 0
        # Sum over all atoms in one dimension
        return DIELECTRIC * COULOMB * energy.sum(1)

    def shortest_distance(self: Polymer) -> torch.Tensor:
        """
        Get the shortest distance between each pair of atoms along the bond graph.
        """

        return dgl.shortest_dist(
            dgl.to_bidirected(self.graph)
        )

    @lazyproperty
    def hops(
        self: Polymer,
    ) -> torch.Tensor:
        """
        Return the number of hops bewteen all atoms in the polymer.
        """

        U, V = self.bonds()
        dists = self.shortest_distance()
        return dists[U, V]

    @lazyproperty
    def bond_types(self: Polymer) -> torch.Tensor:
        """
        Get a unique identifier for each pair of atoms in the molecule
        """

        U, V = self.bonds()

        return ATOM_PAIR_TO_ENUM[
            self.atom_names[U],
            self.atom_names[V],
        ]

    def all_bond_types(self: Polymer) -> torch.Tensor:
        """
        Get a unique identifier for each pair of atoms in the molecule
        """

        U, V = self.bonds()

        return ALL_PAIR_TO_ENUM[
            self.atom_names[U],
            self.atom_names[V],
        ]

    def _vdw(
        self: Polymer,
        rpd: torch.Tensor,
        prop: Property = Property.MOLECULE,
        sigma_params: torch.Tensor = VdW_SIGMA,
        epsilon_params: torch.Tensor = VdW_EPSILON,
    ) -> torch.Tensor:
        """
        Return the Lennard-Jones energy of the molecule.
        """

        # Get the VdW sigma and epsilon values
        sigma = sigma_params[self.atom_names]
        epsilon = epsilon_params[self.atom_names]
        # Use the Lorentz-Berthelot rules to get the pairwise values
        p_sigma = (sigma[None, :] + sigma[:, None]) / 2
        p_epsilon = torch.sqrt(epsilon[None, :] * epsilon[:, None])
        # Multiply by sigma
        rpd = rpd * p_sigma
        # Compute the LJ potential
        lj_energy = 4 * p_epsilon * (rpd ** 12 - rpd ** 6)
        # Zero out any atoms which are too close in bond-space
        lj_energy[self.hops < 4] = 0
        # Sum over all atoms
        lj_energy = lj_energy.sum(-1)
        # Reduce to get the energy contribution
        return self.reduce(
            lj_energy,
            prop,
            Reduction.SUM,
        )

    def _energy(
        self: Polymer,
        prop: Property = Property.MOLECULE,
        charge_params: torch.Tensor = PARTIAL_CHARGE,
        sigma_params: torch.Tensor = VdW_SIGMA,
        epsilon_params: torch.Tensor = VdW_EPSILON,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Coulomb and VdW energies of the polymer.
        """

        # Get the pairwise distances
        pdist = self.global_pdist()
        # Take the reciprocal and zero out the diagonal
        rpd = (1 / pdist).fill_diagonal_(0.0)
        # Compute the eneriges
        coulomb = self._coulomb(rpd, prop, charge_params)
        vdw = self._vdw(rpd, prop, sigma_params, epsilon_params)
        return coulomb, vdw

    def cif(
        self: Polymer,
        file: str,
        overwrite: bool = False,
    ) -> None:
        """
        Save the coordinates, elements, and residues into a .cif file.
        """

        residue_ix = torch.arange(
            self.num_residues
        ).repeat_interleave(self.residue_sizes.cpu())
        # Save to a cif file
        to_cif(
            self.coordinates.detach().cpu().numpy(),
            self.element_names(),
            residue_ix.numpy(),
            file,
            overwrite,
        )

    def __repr__(
        self: Polymer,
    ) -> str:
        """
        Return a little information.
        """

        return f"""{self.__class__.__name__}:
    molecules: {self.num_molecules}
    chains:    {self.num_chains}
    residues:  {self.num_residues}
    atoms:     {self.num_atoms}
    edges:     {self.num_edges}"""


class GeometricPolymer(Polymer):
    """
    A geometric graph which is divided into four levels of coarseness:
    molecule, chain, residue, and atom.
    """

    def __init__(
        self: GeometricPolymer,
        polymer: Polymer,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> None:

        super().__init__(
            polymer.molecule_ids,
            polymer.chain_ids,
            polymer.coordinates,
            polymer.elements,
            polymer.residues,
            polymer.atom_names,
            polymer.graph,
            polymer.residue_sizes,
            polymer.chain_sizes,
            polymer.molecule_sizes,
        )
        # Ensure that all node sizes match
        if not allequal(
            polymer.num_atoms,
            node_features.size(0),
        ):
            raise ValueError(
                "The node features must have the same first"
                " dimension as the number of atoms."
            )
        # Ensure that all edge sizes match
        if not allequal(
            polymer.graph.num_edges(),
            edge_features.size(0),
        ):
            raise ValueError(
                "The edge features must have their first dimension equal to"
                " the number of edges."
            )
        # Store the edge and node features
        self.node_features = node_features
        self.edge_features = edge_features

    def to(
        self: GeometricPolymer,
        dest: str | torch.device | torch.dtype,
    ) -> GeometricPolymer:
        """
        Move all tensors to the given device.
        """

        # Move the underlying polymer
        new_poly = super().to(dest)
        # Move the node and edge features too
        return self.__class__(
            new_poly,
            self.node_features.to(dest),
            self.edge_features.to(dest),
        )

    def __repr__(
        self: GeometricPolymer,
    ) -> str:
        """
        Return a little information.
        """
        return super().__repr__() + f"""
    node_features:
        repr dim:     {self.node_features.size(2)}
        multiplicity: {self.node_features.size(1)}
    edge_feature dim: {self.edge_features.size(1)}"""


def polymer_covariance(
    polymer1: Polymer,
    polymer2: Polymer,
    prop: Property = Property.MOLECULE,
) -> torch.Tensor:
    """
    Get the covariance matrices between the coordintes of the two polymers.
    """

    # Ensure that both polymers have the same number of molecules
    if not polymer1.num_molecules == polymer2.num_molecules:
        raise ValueError(
            "Both Polymers must have the same number of molecules."
        )
    # Center and extract the coordinates
    coords1 = polymer1.coordinates
    coords2 = polymer2.coordinates
    # Weight the coordinates if necessary
    # Get the outer product of the coordinates
    outer_prod = torch.multiply(
        coords1[:, None, :],
        coords2[:, :, None],
    )
    # Compute the per-molecule covariance matrices
    return polymer1.reduce(
        outer_prod,
        prop,
    )


def polymer_kabsch_align(
    polymer1: Polymer,
    polymer2: Polymer,
    weight: torch.Tensor | None = None,
    prop: Property = Property.MOLECULE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the aligned distance between the individual molecules in the polymers
    using the kabsch algorithm. The two polymers should have the same molecule
    indices. An optional weight can be provided to bias the alignment.
    """

    # Center the polymers
    polymer1_c = polymer1.center(prop)
    polymer2_c = polymer2.center(prop)
    # Weight the coordinates accordingly
    if weight is not None:
        weight = polymer1.num_atoms * weight / weight.mean()
        weight = weight[:, None]
        polymer1_c.coordinates = polymer1_c.coordinates * weight
        polymer2_c.coordinates = polymer2_c.coordinates * weight
    # Get the coordinate covariance matrices
    cov = polymer_covariance(
        polymer1_c,
        polymer2_c,
        prop,
    )
    # Calculate the singular value decomposition
    U, _, Vh = torch.linalg.svd(cov)
    # Get the correction factor and apply it to the last column of V
    d = torch.linalg.det(U) * torch.linalg.det(Vh)
    Vh[:, -1] = Vh[:, -1] * d[..., None]
    # Calculate the optimal rotation matrix
    rot = U @ Vh
    # Apply the rotation to the coordinates of the first polymer
    coords1 = polymer1.coordinates[:, None,
                                   :] @ rot[polymer1.indices(Property.MOLECULE)]
    return coords1.squeeze(1), polymer2.coordinates


def polymer_kabsch_distance(
    polymer1: Polymer,
    polymer2: Polymer,
    weight: torch.Tensor | None = None,
    prop: Property = Property.MOLECULE,
) -> torch.Tensor:
    """
    Get the aligned distance between the individual molecules in the polymers
    using the kabsch algorithm. The two polymers should have the same molecule
    indices. An optional weight can be provided to bias the alignment.
    """

    # Center the polymers
    polymer1_c = polymer1.center(prop)
    polymer2_c = polymer2.center(prop)
    # Weight the coordinates accordingly
    if weight is not None:
        weight = polymer1.num_atoms * weight / weight.mean()
        weight = weight[:, None]
        polymer1_c.coordinates = polymer1_c.coordinates * weight
        polymer2_c.coordinates = polymer2_c.coordinates * weight
    # Get the coordinate covariance matrices
    cov = polymer_covariance(
        polymer1_c,
        polymer2_c,
        prop,
    )
    # Get the mean the of singular values of the covariance matrices, reversing
    # the sign of the final one if necessary
    try:
        sigma = torch.linalg.svdvals(cov)
    except Exception:
        breakpoint()
    det = torch.linalg.det(cov)
    # Clone to preserve gradients
    sigma = sigma.clone()
    sigma[det < 0, -1] = - sigma[det < 0, -1]
    sigma = sigma.mean(-1)
    # Get the variances of the point clouds
    var1 = polymer1_c.moment(2, prop).mean(-1)
    var2 = polymer2_c.moment(2, prop).mean(-1)
    # Compute the kabsch distance
    return (var1 + var2 - 2 * sigma)


class PolymerDistance(nn.Module):
    """
    Get the aligned distance between the two GeometricPolymers using the kabsch
    algorithm.
    """

    def __init__(
        self: PolymerDistance,
        prop: Property = Property.MOLECULE,
    ) -> None:

        super().__init__()
        self.prop = prop

    def forward(
        self: PolymerDistance,
        polymer1: Polymer,
        polymer2: Polymer,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the mean aligned distance. The two polymers must have the same
        number of molecules and the same molecule indices.
        """

        return polymer_kabsch_distance(
            polymer1,
            polymer2,
            weight,
            self.prop,
        ).mean()


class PolymerDataset(tdata.Dataset):
    """
    Load polymers from a .nc file, alongside additional data specified
    by the user.
    """

    def __init__(
        self: PolymerDataset,
        file: str,
        device: str = 'cpu',
        atom_features: list[str] = [],
        residue_features: list[str] = [],
        chain_features: list[str] = [],
        molecule_features: list[str] = [],
        edge_features: list[str] = [],
    ) -> None:

        # Initialise the base dataset
        self.structures = StructureDataset(
            file,
            device,
            atom_features,
            residue_features,
            chain_features,
            molecule_features,
            edge_features,
        )
        # Store which device to move the outputs to
        self.device = device

    def __getitem__(
        self: PolymerDataset,
        ix: int | slice,
    ) -> tuple[Polymer, *tuple[torch.Tensor, ...]]:
        """
        Return the polymer alongside the requested features.
        """

        # Load the raw data
        (
            molecule_ids,
            chain_ids,
            coordinates,
            elements,
            residues,
            atom_names,
            edges,
            bond_strength,
            residue_sizes,
            chain_sizes,
            molecule_sizes,
            *user_features
        ) = self.structures[ix]
        # Construct the graph from the edges
        graph = dgl.graph(
            (edges[:, 0], edges[:, 1])
        )
        # Construct the polymer
        polymer = Polymer(
            molecule_ids,
            chain_ids,
            coordinates,
            elements,
            residues,
            atom_names,
            graph,
            residue_sizes,
            chain_sizes,
            molecule_sizes,
        )
        return polymer, *user_features

    def __len__(
        self: PolymerDataset,
    ) -> int:
        """
        Return the number of molecules in the dataset.
        """

        return len(self.structures)

    def __repr__(
        self: PolymerDataset,
    ) -> str:
        """
        Get the underlying StructureDataset's string.
        """

        return self.structures.__repr__()


def _ideal_bond_lengths(
    polymer: Polymer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the ideal bond lengths of each bond in the polymer. Return also
    the associated spring constants.
    """

    # Compute pairwise distances
    pdist = polymer.pdist()
    # Compute ideal distances as the mean
    mean_pdist = polymer.sreduce(pdist, 2)
    # Compute the spring constants
    std_pdist = polymer.sreduce(
        (pdist - mean_pdist[polymer.subgraph_types(2)]) ** 2,
        2
    )
    kb = 3 / (2 * BETA * std_pdist)
    return mean_pdist, kb


def _ideal_bond_angles(
    polymer: Polymer,
) -> torch.Tensor:
    """
    Get the ideas bond angles of each bonded triplet in the polymer.
    Return also the associated spring constants.
    """

    # Compute angles
    angles = polymer.angles()
    # Compute ideal angles as the mean
    mean_angles = polymer.sreduce(angles, 3)
    # Compute the spring constants
    std_angles = polymer.sreduce(
        (angles - mean_angles[polymer.subgraph_types(3)]) ** 2,
        3
    )
    ka = 1 / (2 * BETA * std_angles)
    return mean_angles, ka


def _ideal_torsion_angles(
    polymer: Polymer,
) -> torch.Tensor:
    """
    Get the ideas bond angles of each bonded triplet in the polymer.
    Return also the associated spring constants.
    """

    # Compute angles
    angles = polymer.torsions()
    # Compute ideal angles as the mean
    mean_angles = polymer.sreduce(angles, 4)
    # Compute the spring constants
    std_angles = polymer.sreduce(
        (angles - mean_angles[polymer.subgraph_types(4)]) ** 2,
        4,
    )
    kt = 1 / (2 * BETA * std_angles)
    kt[kt < 1000] = 0
    return mean_angles, kt


def _ideal_coordinates(
    polymer: Polymer,
) -> list[torch.Tensor]:
    """
    Return the ideal coordinates of each of the four bases in the polymer.
    """

    ad = polymer.select(
        Ribose.index(),
        Property.NAME,
    )
    ad = ad.center(Property.RESIDUE)
    ad, _ = ad.align(Property.RESIDUE)
    ad.cif('adenosine_tst.cif', True)
    coords = ad.reduce(
        ad.coordinates,
        Property.NAME,
    )
    return coords[Ribose.index()[:-1]]


def _steric_anomaly(
    polymer: Polymer,
    ideals: torch.Tensor,
    ks: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Compute the steric anomaly based on the given ideal values.
    """

    # Get an ideal value for each of the subgraph types
    ideals = ideals[polymer.subgraph_types(n)]
    ks = ks[polymer.subgraph_types(n)]

    if n == 2:
        return torch.mean(
            ks * (ideals - polymer.pdist()) ** 2
        )
    if n == 3:
        return torch.mean(
            ks * (ideals - polymer.angles()) ** 2
        )
    if n == 4:
        return torch.mean(
            ks * (ideals - polymer.torsions()) ** 2
        )
