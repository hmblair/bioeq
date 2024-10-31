from __future__ import annotations
from typing import Any
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
from .data import StructureDataset
from copy import copy

NODE_DIM = 0
UV_DIM = 1
REDUCTIONS = {
    'mean': t_scatter_mean,
    'sum': t_scatter_sum,
    'min': t_scatter_min,
    'max': t_scatter_max,
}


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
        coordinates: torch.Tensor,
        graph: dgl.DGLGraph,
        residue_ix: torch.Tensor,
        chain_ix: torch.Tensor,
        molecule_ix: torch.Tensor,
    ) -> None:

        # Store all attributes
        self.coordinates = coordinates
        self.graph = graph
        self.residue_ix = residue_ix
        self.chain_ix = chain_ix
        self.molecule_ix = molecule_ix
        # Compute the number of edges, atoms, residues, chains, and
        # molecules
        self.num_edges = graph.num_edges()
        self.num_atoms = coordinates.size(0)
        self.num_residues = self.residue_ix[-1].item() + 1
        self.num_chains = self.chain_ix[-1].item() + 1
        self.num_molecules = self.molecule_ix[-1].item() + 1
        # Compute the size of each molecule
        self.molecule_sizes = torch.bincount(
            self.molecule_ix
        )

        # Keep track of whether the molecule is centered
        self.is_centered = False

    def residue_reduce(
        self: Polymer,
        features: torch.Tensor,
        rtype: str = 'mean',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.LongTensor]:
        """
        Reduce the input values within each residue. Min and max reductions
        return the indices too.
        """

        return REDUCTIONS[rtype](
            features,
            self.residue_ix,
            dim=NODE_DIM,
        )

    def chain_reduce(
        self: Polymer,
        features: torch.Tensor,
        rtype: str = 'mean',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.LongTensor]:
        """
        Reduce the input values within each chain. Min and max reductions
        return the indices too.
        """

        return REDUCTIONS[rtype](
            features,
            self.chain_ix,
            dim=NODE_DIM,
        )

    def molecule_reduce(
        self: Polymer,
        features: torch.Tensor,
        rtype: str = 'mean',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.LongTensor]:
        """
        Reduce the input values within each molecule. Min and max reductions
        return the indices too.
        """

        return REDUCTIONS[rtype](
            features,
            self.molecule_ix,
            dim=NODE_DIM,
        )

    def causal_mask(
        self: Polymer,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Create an edge mask so that information can only flow forward
        through the polymer, and not backward.
        """

        # Push the residue indices to the edges
        U, V = self.graph.edges()
        residue_U, residue_V = self.residue_ix[U], self.residue_ix[V]
        # Create a causal mask
        if reverse:
            return residue_U >= residue_V
        else:
            return residue_U <= residue_V

    def bond_mask(
        self: Polymer,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask all information flowing from the specified nucleotides.
        """

        # Push the residue indices to the edges
        U, _ = self.graph.edges()
        residue_U = self.residue_ix[U]
        # Get the nodes participating in the masking
        mask_U = torch.nonzero(mask).squeeze(1)
        # Find all edges beginning in a masked residue
        edge_mask = (residue_U[:, None] == mask_U[None, :])
        return torch.logical_not(
            edge_mask.any(UV_DIM)
        )

    def relpos(
        self: Polymer,
        max: int | None = None,
    ) -> torch.Tensor:
        """
        Return a 1D tensor of length self.num_edges containing the
        residue-wise relative distance between the source and
        destination of each edge.
        """

        # Get the source and destination nodes of each edge
        U, V = self.graph.edges()
        # Compute the distance in the sequence
        relpos = self.residue_ix[V] - self.residue_ix[U]
        # Clamp if a maximum distance is given
        if max is not None:
            relpos = torch.clamp(
                relpos, -max, max
            )
        return relpos

    def center(
        self: Polymer,
    ) -> None:
        """
        Center the coordinates of each molecule in the polymer.
        """

        if not self.is_centered:
            # Get the coordinate means
            coord_means = self.molecule_reduce(self.coordinates)
            # Expand the means and subtract
            self.coordinates = self.coordinates - coord_means[self.molecule_ix]
            # We are now centered
            self.is_centered = True

    def pdist(
        self: Polymer,
    ) -> torch.Tensor:
        """
        Get the pairwise distances between all coordinates which are
        connected by an edge.
        """

        # Get the edges from the graph
        U, V = self.graph.edges()
        # Compute the pairwise norms
        return torch.linalg.norm(
            self.coordinates[U] - self.coordinates[V],
            dim=-1
        )

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
        # Update the number of edges
        self.num_edges = self.graph.num_edges()

    def to(
        self: Polymer,
        device: str | torch.device,
    ) -> Polymer:
        """
        Move all tensors to the given device.
        """

        # Copy so as not to overwrite
        new = copy(self)
        # Move all relevant tensors to the device
        new.coordinates = self.coordinates.to(device)
        new.graph = self.graph.to(device)
        new.residue_ix = self.residue_ix.to(device)
        new.chain_ix = self.chain_ix.to(device)
        new.molecule_ix = self.molecule_ix.to(device)
        return new

    def __repr__(
        self: Polymer,
    ) -> str:
        """
        Return a little information.
        """
        return f"""Polymer:
    num_molecules:    {self.num_molecules}
    num_chains:       {self.num_chains}
    num_residues:     {self.num_residues}
    num_atoms:        {self.num_atoms}
    num_edges:        {self.num_edges}"""


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
            polymer.coordinates,
            polymer.graph,
            polymer.residue_ix,
            polymer.chain_ix,
            polymer.molecule_ix,
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
        device: str | torch.device,
    ) -> GeometricPolymer:
        """
        Move all tensors to the given device.
        """

        # Copy so as not to overwrite
        new = copy(self)
        # Move all relevant tensors to the device
        new.coordinates = self.coordinates.to(device)
        new.graph = self.graph.to(device)
        new.residue_ix = self.residue_ix.to(device)
        new.chain_ix = self.chain_ix.to(device)
        new.molecule_ix = self.molecule_ix.to(device)
        new.node_features = self.node_features.to(device)
        new.edge_features = self.edge_features.to(device)
        return new

    def __repr__(
        self: GeometricPolymer,
    ) -> str:
        """
        Return a little information.
        """
        return f"""GeometricPolymer:
    num_molecules:    {self.num_molecules}
    num_chains:       {self.num_chains}
    num_residues:     {self.num_residues}
    num_atoms:        {self.num_atoms}
    num_edges:        {self.num_edges}
    node_features:
        repr dim:     {self.node_features.size(2)}
        multiplicity: {self.node_features.size(1)}
    edge_feature dim: {self.edge_features.size(1)}"""


def polymer_kabsch_distance(
    polymer1: GeometricPolymer,
    polymer2: GeometricPolymer,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Get the aligned distance between the individual molecules in the polymers
    using the kabsch algorithm. The two polymers should have the same molecule
    indices. An optional weight can be provided to bias the alignment.
    """

    # Ensure that both polymers have the same number of molecules
    if not polymer1.num_molecules == polymer2.num_molecules:
        raise ValueError(
            "Both GeometricPolymers must have the same number of molecules."
        )
    # Center the coordinates
    polymer1.center()
    polymer2.center()
    # Get the outer product of the coordinates
    outer_prod = torch.multiply(
        polymer1.coordinates[:, None, :],
        polymer2.coordinates[:, :, None],
    )
    # Weight the nodes if provided
    if weight is not None:
        outer_prod = outer_prod * weight[:, None]
    # Compute the per-molecule covariance matrices
    cov = t_scatter_mean(
        outer_prod,
        polymer1.molecule_ix,
        dim=0,
    )
    # Get the mean the of singular values of the covariance matrices, reversing
    # the sign of the final one if necessary
    sigma = torch.linalg.svdvals(cov)
    det = torch.linalg.det(cov)
    # Clone to preserve gradients
    sigma = sigma.clone()
    sigma[det < 0, -1] = - sigma[det < 0, -1]
    sigma = sigma.mean(-1)
    # Get the variances of the point clouds
    var1 = polymer1.molecule_reduce(
        polymer1.coordinates ** 2
    ).mean(-1)
    var2 = polymer2.molecule_reduce(
        polymer2.coordinates ** 2
    ).mean(-1)
    # Compute the kabsch distance
    return (var1 + var2 - 2 * sigma)


class PolymerDistance(nn.Module):
    """
    Get the aligned distance between the two GeometricPolymers using the kabsch
    algorithm.
    """

    def __init__(
        self: PolymerDistance,
        weight: torch.Tensor | None = None,
    ) -> None:

        super().__init__()
        self.weight = weight

    def forward(
        self: PolymerDistance,
        polymer1: GeometricPolymer,
        polymer2: GeometricPolymer,
    ) -> torch.Tensor:
        """
        Compute the mean aligned distance. The two polymers must have the same
        number of molecules and the same molecule indices.
        """

        return polymer_kabsch_distance(
            polymer1,
            polymer2,
            self.weight,
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
            coordinates,
            edges,
            residue_ix,
            chain_ix,
            molecule_ix,
            *user_features
        ) = self.structures[ix]
        # Construct the graph from the edges
        graph = dgl.graph(
            (edges[:, 0], edges[:, 1])
        )
        # Construct the polymer
        polymer = Polymer(
            coordinates,
            graph,
            residue_ix,
            chain_ix,
            molecule_ix,
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
