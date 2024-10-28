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
from data import read_pdb, PDB_SUFFIX
import os
from copy import copy
import numpy as np
import xarray as xr
from pathlib import Path

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


def batch(
    items: list[GeometricPolymer],
) -> GeometricPolymer:
    """
    Stack multiple geometric polymers into a single unit.
    """

    if not items:
        raise ValueError('At least one GeometricPolymer must be provided.')

    # Get the new coordinates, node and edge features
    coordinates = torch.cat(
        [poly.coordinates for poly in items]
    )
    node_features = torch.cat(
        [poly.node_features for poly in items]
    )
    edge_features = torch.cat(
        [poly.edge_features for poly in items]
    )
    # Get the new graph
    graph = dgl.batch(
        [poly.graph for poly in items]
    )
    # Get updated residue indices. To do so, we must offset the indices of
    # each item by the cumulative number of residues in the previous molecule.
    num_residues = torch.tensor(
        [0] + [poly.num_residues for poly in items[:-1]]
    )
    cum_residues = torch.cumsum(num_residues, dim=0)
    residue_ix = torch.cat(
        [poly.residue_ix + cumres
         for poly, cumres in zip(items, cum_residues)]
    )
    # Do the same for the chains
    num_chains = torch.tensor(
        [0] + [poly.num_chains for poly in items[:-1]]
    )
    cum_chains = torch.cumsum(num_chains, dim=0)
    chain_ix = torch.cat(
        [poly.chain_ix + cumchain
         for poly, cumchain in zip(items, cum_chains)]
    )
    # Create the batched GeometricPolymer
    return GeometricPolymer(
        coordinates,
        graph,
        node_features,
        edge_features,
        residue_ix,
        chain_ix,
    )


class GeometricPolymer:
    """
    A geometric graph which is divided into three levels of coarseness:
    molecule, residue, and atom.
    """

    def __init__(
        self: GeometricPolymer,
        coordinates: torch.Tensor,
        graph: dgl.DGLGraph,
        elements: torch.Tensor,
        residues: torch.Tensor,
        residue_ix: torch.Tensor,
        chain_ix: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> None:

        # Ensure that all node sizes match
        if not allequal(
            coordinates.size(0),
            elements.size(0),
            residues.size(0),
            residue_ix.size(0),
            chain_ix.size(0),
        ):
            raise ValueError(
                "The coordinates, node features, chain and residue indices"
                " must have the same first dimension."
            )

        # Ensure that all edge sizes match
        if not allequal(
            edge_features.size(0),
            graph.num_edges(),
        ):
            raise ValueError(
                "The edge features must have their first dimension equal to"
                " the number of nodes."
            )

        # Store all attributes
        self.coordinates = coordinates
        self.graph = graph
        self.node_features = None
        self.elements = elements
        self.residues = residues
        self.edge_features = edge_features
        self.residue_ix = residue_ix
        self.chain_ix = chain_ix
        # Store the number of atoms and edges
        self.num_atoms = coordinates.size(0)
        self.num_edges = graph.num_edges()
        # Compute the number of residues and their sizes
        self.num_residues = self.residue_ix[-1].item() + 1
        self.residue_sizes = torch.bincount(self.residue_ix)
        # Compute the number of chains and their sizes
        self.num_chains = self.chain_ix[-1].item() + 1
        self.chain_sizes = torch.bincount(self.chain_ix)
        # Get the number of molecules and their sizes
        self.num_molecules = graph.batch_size
        self.molecule_sizes = graph.batch_num_nodes()
        # Compute which molecule each node belongs to
        self.molecule_ix = torch.repeat_interleave(
            torch.arange(
                self.num_molecules,
                device=graph.device,
            ),
            self.molecule_sizes,
        )

    @classmethod
    def from_pdb(
        cls: type[GeometricPolymer],
        filename: str,
        edge_degree: int,
    ) -> GeometricPolymer:
        """
        Create a GeometricPolymer object from a PDB file.
        """

        # Read the PDB file
        (coordinates,
         bond_src,
         bond_dst,
         bond_type,
         elements,
         residues,
         residue_ix,
         chain_ix) = read_pdb(filename)
        # Construct a knn graph from the coordinates
        graph = dgl.knn_graph(
            coordinates,
            edge_degree,
        )
        # Use the pairwise distances as bond features
        U, V = graph.edges()
        edge_features = torch.norm(
            coordinates[U] - coordinates[V], dim=1
        )[:, None]
        return cls(
            coordinates,
            graph,
            elements,
            residues,
            edge_features,
            residue_ix,
            chain_ix,
        )

    def residue_reduce(
        self: GeometricPolymer,
        features: torch.Tensor,
        rtype: str = 'mean',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.LongTensor]:
        """
        Reduce the input features within each residue. Min and max reductions
        return the indices too.
        """

        return REDUCTIONS[rtype](
            features,
            self.residue_ix,
            dim=NODE_DIM,
        )

    def chain_reduce(
        self: GeometricPolymer,
        features: torch.Tensor,
        rtype: str = 'mean',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.LongTensor]:
        """
        Reduce the input features within each chain. Min and max reductions
        return the indices too.
        """

        return REDUCTIONS[rtype](
            features,
            self.chain_ix,
            dim=NODE_DIM,
        )

    def molecule_reduce(
        self: GeometricPolymer,
        features: torch.Tensor,
        rtype: str = 'mean',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.LongTensor]:
        """
        Reduce the input features within each molecule. Min and max reductions
        return the indices too.
        """

        return REDUCTIONS[rtype](
            features,
            self.molecule_ix,
            dim=NODE_DIM,
        )

    def bond_mask(
        self: GeometricPolymer,
        mask: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Convert a residue-wise mask into an atom-wise mask. If no mask is
        passed, then it is assumed to be a causal mask. If the mask is causal,
        causality can be reversed using the reverse bool.
        """

        # Push the residue indices to the edges
        U, V = self.graph.edges()
        residue_U, residue_V = self.residue_ix[U], self.residue_ix[V]
        # Create a causal mask if none is specified
        if mask is None:
            if reverse:
                return residue_U >= residue_V
            else:
                return residue_U <= residue_V
        # Get the nodes participating in the masking
        mask_U, mask_V = torch.nonzero(mask).chunk(2, dim=UV_DIM)
        mask_U = mask_U.squeeze(1)
        mask_V = mask_V.squeeze(1)
        # If position (i,j) is masked, then we need to prevent
        # resuidue i from passing messages to residue j. To do
        # this, we need to find all edges starting within residue
        # i and ending within residue j.
        src_mask = (residue_U[:, None] == mask_U[None, :])
        dst_mask = (residue_V[:, None] == mask_V[None, :])
        # Return true if any edge connects a masked pair of residues
        return (src_mask & dst_mask).any(UV_DIM)

    def relpos(
        self: GeometricPolymer,
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
        self: GeometricPolymer,
    ) -> None:
        """
        Center the coordinates of each molecule in the polymer.
        """

        # Get the coordinate means
        coord_means = self.molecule_reduce(self.coordinates)
        # Expand the means and subtract
        self.coordinates = self.coordinates - coord_means[self.molecule_ix]

    def connect(
        self: GeometricPolymer,
        num_neighbours: int,
    ) -> None:
        """
        Replace the existing graph with a knn graph based on the current
        coordinates. WARNING: if this changes the number of edges in the
        graph, then the class instance will become unusable.
        """

        # Call into the dgl engine
        self.graph = dgl.segmented_knn_graph(
            self.coordinates,
            num_neighbours,
            self.molecule_sizes.tolist(),
        )

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
        new.node_features = self.node_features.to(device)
        new.edge_features = self.edge_features.to(device)
        new.graph = self.graph.to(device)
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
    Load a single polymer from a directory of PDB files.
    """

    def __init__(
        self: PolymerDataset,
        directory: str,
        energy_file: str,
        edge_degree: int,
        suffix: str = '.pdb',
    ) -> None:

        # Get the names of all the PDB files
        self.files = [
            directory + '/' + file
            for file in os.listdir(directory)
            if file.endswith(suffix)
        ]
        if not self.files:
            raise RuntimeError(
                f"The direcotry {directory} does not contain"
                " any PDB files."
            )
        # Store the edge degree
        self.degree = edge_degree
        # Get the additional features
        self.ds = xr.load_dataset(energy_file)

    def shuffle(
        self: PolymerDataset,
    ) -> None:
        """
        Shuffle the files in the dataset.
        """

        ix = np.random.permutation(len(self))
        self.files = [self.files[i] for i in ix]
        self.ds = self.ds.isel(ix)

    def __len__(
        self: PolymerDataset,
    ) -> int:
        """
        The number of PDB files in the folder.
        """

        return len(self.files)

    def __getitem__(
        self: PolymerDataset,
        ix: int,
    ) -> tuple[GeometricPolymer, torch.Tensor]:
        """
        Get the PDB file at index ix and create a GeometricPolyemr
        from it.
        """

        polymer = GeometricPolymer.from_pdb(
            self.files[ix],
            self.degree,
        )

        file_number = int(Path(
            self.files[ix]
        ).stem)
        energy = torch.tensor(
            self.ds['energy'].values[file_number]
        )

        return polymer, energy
