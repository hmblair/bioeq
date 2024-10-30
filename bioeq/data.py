from __future__ import annotations
from typing import Iterable
import torch
from biotite.structure.io import load_structure
from biotite.structure import (
    connect_via_residue_names,
    connect_via_distances,
)
import numpy as np
import os
import xarray as xr
from pathlib import Path
from tqdm import tqdm

ELEMENT_IX = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "P": 4,
    "D": 0,
}
NUM_ELEMENTS = len(ELEMENT_IX)
NUCLEOTIDE_RES_IX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "U": 3,
}
NUM_NUCLEOTIDE_RES = len(NUCLEOTIDE_RES_IX)

ATOM_DIM = 'atom'
RESIDUE_DIM = 'residue'
CHAIN_DIM = 'chain'
MOLECULE_DIM = 'molecule'
EDGE_DIM = 'edge'
COORDINATE_DIM = 'axis'
SRCDST_DIM = 'loc'

COORDINATES_KEY = 'coordinates'
EDGES_KEY = 'edges'
RESIDUE_IX_KEY = 'residue_ix'
CHAIN_IX_KEY = 'chain_ix'
MOLECULE_IX_KEY = 'molecule_ix'
EDGE_IX_KEY = 'edge_ix'
ID_KEY = 'id'

INDEX_VARS = [
    RESIDUE_IX_KEY,
    CHAIN_IX_KEY,
    MOLECULE_IX_KEY,
    EDGE_IX_KEY,
]

DIMENSION_VARS = [
    ATOM_DIM,
    RESIDUE_DIM,
    CHAIN_DIM,
    MOLECULE_DIM,
    EDGE_DIM,
]


def bullets(
    strings: list[str],
    offset: int = 0,
) -> str:
    """
    Convert a list of strings to a bullet point list.
    """
    return " " * offset + ("\n" + " " * offset).join(f"- {s}" for s in strings)


def proportional_slices(
    proportions: Iterable[float],
    total_length: int,
) -> list[slice]:

    if not sum(proportions) <= 1:
        raise ValueError(
            "The proportions must have a sum of at most"
            " one."
        )

    counts = [
        int(p * total_length)
        for p in proportions
    ]

    # Adjust for any rounding differences
    remainder = total_length - sum(counts)
    for i in range(remainder):
        counts[i] += 1

    # Generate slices
    slices = []
    start = 0
    for count in counts:
        end = start + count
        slices.append(slice(start, end))
        start = end

    return slices


def read_structure(
    file: str,
    connect_via: str = 'residue_names',
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Read information from a PDB file.
    """

    # Load the structure into an AtomArray
    struct = load_structure(file)
    # Get the coordinates
    coordinates = struct.coord
    # Get the bonds and their types
    if connect_via == 'residue_names':
        bonds = connect_via_residue_names(
            struct
        ).as_array()
    elif connect_via == 'distances':
        bonds = connect_via_distances(
            struct
        ).as_array()
    else:
        raise ValueError(
            "'connect_via' must be either 'residue_names' or"
            " 'distances'."
        )
    bond_edges = bonds[:, 0:2]
    bond_types = bonds[:, 2]
    # Add self and reverse edges too
    self_edges = np.tile(
        np.arange(coordinates.shape[0])[:, None],
        (1, 2)
    )
    self_bond_types = np.zeros(
        coordinates.shape[0]
    )
    bond_edges = np.concatenate(
        [bond_edges,
         bond_edges[:, ::-1],
         self_edges],
        axis=0
    )
    bond_types = np.concatenate(
        [bond_types,
         bond_types,
         self_bond_types,
         ],
        axis=0
    )

    # Get the elements as integers
    elements = np.array([
        ELEMENT_IX.get(element, 0)
        for element in struct.element
    ]).astype(np.int64)
    # Get the residues as integers
    residues = np.array([
        NUCLEOTIDE_RES_IX.get(res, -1)
        for res in struct.res_name
    ]).astype(np.int64)
    # Get which residue each atom belongs to
    if all(not res for res in struct.res_id):
        residue_ix = np.zeros(
            len(struct.res_id),
        )
    else:
        residue_ix = struct.res_id
        residue_ix = residue_ix - residue_ix[0]
        # The residue indices are often screwed up in the PDB file. We
        # re-index them here.
        diffs = np.concatenate(
            (np.array([0]), residue_ix[1:] != residue_ix[:-1])
        )
        residue_ix = np.cumsum(diffs)
    # Get which chain each atom belongs to
    if all(not res for res in struct.chain_id):
        chain_ix = np.zeros(
            len(struct.chain_id),
        )
    else:
        _, chain_ix = np.unique(
            struct.chain_id,
            return_inverse=True,
        )
    return (
        coordinates,
        bond_edges,
        bond_types,
        elements,
        residues,
        residue_ix,
        chain_ix,
    )


def create_structure_dataset(
    directory: str,
    extension: str,
    out_file: str,
    connect_via: str = 'residue_names',
) -> None:

    ids = []

    coordinates_ls = []
    residues_ls = []
    bond_edges_ls = []
    bond_types_ls = []
    elements_ls = []

    residue_ix_ls = []
    chain_ix_ls = []
    molecule_ix_ls = []
    edge_molecule_ix_ls = []

    prev_residue_ix = 0
    prev_chain_ix = 0
    prev_molecule_ix = 0
    prev_edge_ix = 0
    prev_edge_molecule_ix = 0

    for file in tqdm(
        os.listdir(directory),
        desc=f'Reading {extension} files'
    ):

        if not file.endswith(extension):
            continue

        ids.append(
            Path(file).stem
        )

        (coordinates,
         bond_edges,
         bond_types,
         elements,
         residues,
         residue_ix,
         chain_ix) = read_structure(
            os.path.join(directory, file),
            connect_via,
        )

        coordinates_ls.append(coordinates)
        residues_ls.append(residues)
        elements_ls.append(elements)
        bond_edges_ls.append(
            bond_edges + prev_edge_ix
        )
        prev_edge_ix += coordinates.shape[0]
        bond_types_ls.append(bond_types)

        residue_ix_ls.append(
            residue_ix + prev_residue_ix
        )
        prev_residue_ix = residue_ix_ls[-1][-1] + 1
        chain_ix_ls.append(
            chain_ix + prev_chain_ix
        )
        prev_chain_ix = chain_ix_ls[-1][-1] + 1

        molecule_ix_ls.append(
            np.ones(coordinates.shape[0]) * prev_molecule_ix
        )
        prev_molecule_ix += 1

        edge_molecule_ix_ls.append(
            np.ones(bond_edges.shape[0]) * prev_edge_molecule_ix
        )
        prev_edge_molecule_ix += 1

    coordinates = np.concatenate(coordinates_ls)
    residues = np.concatenate(residues_ls).astype(np.int64)
    elements = np.concatenate(elements_ls).astype(np.int64)

    bond_edges = np.concatenate(bond_edges_ls).astype(np.int64)
    bond_types = np.concatenate(bond_types_ls).astype(np.int64)
    residue_ix = np.concatenate(residue_ix_ls).astype(np.int64)
    chain_ix = np.concatenate(chain_ix_ls).astype(np.int64)
    molecule_ix = np.concatenate(molecule_ix_ls).astype(np.int64)
    edge_ix = np.concatenate(edge_molecule_ix_ls).astype(np.int64)

    ds = xr.Dataset(
        {
            COORDINATES_KEY: ([ATOM_DIM, COORDINATE_DIM], coordinates),
            EDGES_KEY: ([EDGE_DIM, SRCDST_DIM], bond_edges),
            RESIDUE_IX_KEY: ([ATOM_DIM], residue_ix),
            CHAIN_IX_KEY: ([ATOM_DIM], chain_ix),
            MOLECULE_IX_KEY: ([ATOM_DIM], molecule_ix),
            EDGE_IX_KEY: ([EDGE_DIM], edge_ix),
            ID_KEY: ([MOLECULE_DIM], ids),
            'elements': ([ATOM_DIM], elements),
            'residues': ([ATOM_DIM], residues),
            'bond_types': ([EDGE_DIM], bond_types),
        }
    )

    ds.to_netcdf(out_file)


def padded_cumsum(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Return the cumulative sum, but shifted to the left by one.
    """

    return torch.cumsum(
        torch.cat([torch.tensor([0]), x]), dim=0
    )


class Slicer:
    def __init__(
        self: Slicer,
        arrays: dict[str, torch.Tensor],
    ) -> None:

        self.arrays = arrays

    def __call__(
        self: Slicer,
        start: int,
        stop: int,
    ) -> dict[str, slice]:

        return {
            name: slice(
                array[start],
                array[stop],
            ) for name, array in self.arrays.items()
        }


class StructureDataset:
    """
    Load polymers from a .nc file, alongside additional data specified
    by the user.
    """

    def __init__(
        self: StructureDataset,
        file: str,
        device: str = 'cpu',
        atom_features: list[str] = [],
        residue_features: list[str] = [],
        chain_features: list[str] = [],
        molecule_features: list[str] = [],
        edge_features: list[str] = [],
        dropped_features: list[str] = INDEX_VARS,
    ) -> None:

        # Store the device to use and the file
        self.device = device
        self.file = file
        # Store which features we will be loading
        self.user_features = (
            atom_features +
            residue_features +
            chain_features +
            molecule_features +
            edge_features
        )
        # Lazily open the dataset
        self.ds = xr.open_dataset(file)
        # Store the indices of the three atom groups
        self.residue_ix = torch.from_numpy(
            self.ds[RESIDUE_IX_KEY].values
        )
        self.chain_ix = torch.from_numpy(
            self.ds[CHAIN_IX_KEY].values
        )
        self.molecule_ix = torch.from_numpy(
            self.ds[MOLECULE_IX_KEY].values
        )
        self.edge_ix = torch.from_numpy(
            self.ds[EDGE_IX_KEY].values
        )
        # Drop variables from the dataset
        self.ds = self.ds.drop_vars(dropped_features)
        # Get the sizes of the groups of atoms
        self.residue_sizes = torch.bincount(
            self.residue_ix
        )
        self.chain_sizes = torch.bincount(
            self.chain_ix
        )
        self.atoms_per_molecule = torch.bincount(
            self.molecule_ix
        )
        self.edges_per_molecule = torch.bincount(
            self.edge_ix
        )

        # Get the number of residues and chains per molecule
        molecule_indices = torch.arange(self.molecule_ix.max().item() + 1)
        # Create a 2D tensor of (molecule, residue) pairs
        molecule_residue_pairs = torch.stack(
            [self.molecule_ix, self.residue_ix], dim=1)
        # Find unique pairs, then count unique residues per molecule
        unique_pairs = torch.unique(molecule_residue_pairs, dim=0)
        # Count residues per molecule by binning unique molecule indices
        self.residues_per_molecule = torch.bincount(
            unique_pairs[:, 0],
            minlength=molecule_indices.numel(),
        )
        # Create a 2D tensor of (molecule, chain) pairs
        molecule_chain_pairs = torch.stack(
            [self.molecule_ix, self.chain_ix], dim=1)
        # Find unique pairs, then count unique chains per molecule
        unique_pairs = torch.unique(molecule_chain_pairs, dim=0)
        # Count residues per molecule by binning unique molecule indices
        self.chains_per_molecule = torch.bincount(
            unique_pairs[:, 0],
            minlength=molecule_indices.numel(),
        )

        # Get the starting position of each atom group in the dataset
        self.atom_starts = padded_cumsum(
            self.atoms_per_molecule
        )
        self.residue_starts = padded_cumsum(
            self.residues_per_molecule
        )
        self.chain_starts = padded_cumsum(
            self.chains_per_molecule
        )
        self.edge_starts = padded_cumsum(
            self.edges_per_molecule
        )
        self.molecule_starts = torch.arange(len(self) + 1)
        # Construct a slicer object to slice the array
        self.slicer = Slicer({
            ATOM_DIM: self.atom_starts,
            EDGE_DIM: self.edge_starts,
        })
        if RESIDUE_DIM in self.ds.dims:
            self.slicer.arrays[RESIDUE_DIM] = self.residue_starts
        if CHAIN_DIM in self.ds.dims:
            self.slicer.arrays[CHAIN_DIM] = self.chain_starts
        if MOLECULE_DIM in self.ds.dims:
            self.slicer.arrays[MOLECULE_DIM] = self.molecule_starts

    def __getitem__(
        self: StructureDataset,
        ix: int | slice,
    ) -> tuple[torch.Tensor, ...]:
        """
        Load all data associated with the molecules with index/indices ix.
        """

        if isinstance(ix, slice):
            start = (
                max(ix.start, -len(self))
                if ix.start is not None
                else 0
            )
            stop = (
                min(ix.stop, len(self))
                if ix.stop is not None
                else len(self)
            )
        elif isinstance(ix, int):
            start = ix
            stop = ix + 1

        # Get the slice of data we want to load
        data_slice = self.slicer(start, stop)
        atom_slice = data_slice[ATOM_DIM]
        # Load the relevant data from the dataset
        data = self.ds.isel(data_slice)
        # Extract the coordinates and edges for constructing the
        # polymer
        coordinates = torch.Tensor(
            data[COORDINATES_KEY].values
        )
        edges = torch.Tensor(
            data[EDGES_KEY].values
        )
        # Get the relevant indices, and reset them to begin at zero
        residue_ix = self.residue_ix[atom_slice]
        residue_ix = residue_ix - residue_ix[0]
        chain_ix = self.chain_ix[atom_slice]
        chain_ix = chain_ix - chain_ix[0]
        molecule_ix = self.molecule_ix[atom_slice]
        molecule_ix = molecule_ix - molecule_ix[0]
        # Extract the additional feautures the user wants
        user_features = [
            torch.Tensor(data[feature].values)
            for feature in self.user_features
        ]
        # Move to the correct datatype and device
        coordinates = coordinates.to(
            dtype=torch.float32,
            device=self.device,
        )
        edges = edges.to(
            dtype=torch.long,
            device=self.device,
        )
        edges = edges - atom_slice.start
        user_features = [
            feature.to(self.device)
            for feature in user_features
        ]
        # Return, including the relevant indices as well
        return (
            coordinates,
            edges,
            residue_ix.long(),
            chain_ix.long(),
            molecule_ix.long(),
            *user_features,
        )

    def num_atoms(
        self: StructureDataset,
    ) -> int:
        """
        The number of atoms in the dataset.
        """

        return self.atoms_per_molecule.sum().item()

    def num_residues(
        self: StructureDataset,
    ) -> int:
        """
        The number of residues in the dataset.
        """

        return len(self.residue_sizes)

    def num_chains(
        self: StructureDataset,
    ) -> int:
        """
        The number of chains in the dataset.
        """

        return len(self.chain_sizes)

    def num_molecules(
        self: StructureDataset,
    ) -> int:
        """
        The number of molecules in the dataset.
        """

        return len(self.atoms_per_molecule)

    def __len__(
        self: StructureDataset,
    ) -> int:
        """
        An alias for num_molecules.
        """

        return self.num_molecules()

    def split(
        self: StructureDataset,
        props: list[float],
        filenames: list[str],
    ) -> list[xr.Dataset]:
        """
        Split the underlying dataset into len(props) datasets containing
        the respective portion of the data.
        """

        orig_slices = proportional_slices(
            props, len(self),
        )

        datasets = []
        for orig_slice in orig_slices:
            start = orig_slice.start
            stop = orig_slice.stop
            data_slice = self.slicer(start, stop)
            datasets.append(
                self.ds.isel(data_slice)
            )
        # Save the new datasets
        for ds, filename in zip(datasets, filenames):
            ds.to_netcdf(filename)

    def __repr__(
        self: StructureDataset,
    ) -> str:
        """
        Print the available variables in the dataset.
        """

        out_str = "\nPolymerDataset\n"
        out_str += f'  file: {self.file}\n'
        out_str += 'Available variables:\n'

        for dim in DIMENSION_VARS:
            out = [
                var for var in self.ds.data_vars
                if dim in self.ds[var].dims
            ]
            if out:
                out_str += ("  " + dim + '\n')
                out_str += bullets(out, 4) + '\n'

        return out_str
