from __future__ import annotations
from typing import Iterable
import torch
from biotite.structure.io import load_structure
from biotite.structure.io.pdbx import (
    CIFFile,
    set_structure,
)
from biotite.structure import (
    AtomArray,
    connect_via_residue_names,
    connect_via_distances,
)
import hydride
import numpy as np
import os
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from ._index import (
    Element,
    Residue,
    Backbone,
    RibonucleicAcid,
    ATOM_PAIR_TO_ENUM,
)
from bioeq._cpp._c import _partitionCount


def unprime(
    arr: np.ndarray,
) -> np.ndarray:
    """
    Convert all primes in the entires to ps.
    """

    return np.char.replace(arr, "'", "p")


ATOM_DIM = 'atom'
RESIDUE_DIM = 'residue'
CHAIN_DIM = 'chain'
MOLECULE_DIM = 'molecule'
EDGE_DIM = 'edge'
COORDINATE_DIM = 'axis'
SRCDST_DIM = 'loc'

COORDINATES_KEY = 'coordinates'
ELEMENTS_KEY = 'elements'
EDGES_KEY = 'edges'
RESIDUE_SIZES_KEY = 'residue_sizes'
CHAIN_SIZES_KEY = 'chain_sizes'
MOLECULE_SIZES_KEY = 'molecule_sizes'
EDGE_SIZES_KEY = 'edge_sizes'
ID_KEY = 'id'

INDEX_VARS = [
    RESIDUE_SIZES_KEY,
    CHAIN_SIZES_KEY,
    MOLECULE_SIZES_KEY,
    EDGE_SIZES_KEY,
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
    np.ndarray,
]:
    """
    Read information from a PDB file.
    """

    # Load the structure into an AtomArray
    struct = load_structure(file)
    # Remove any hydrogens
    struct = struct[
        struct.element != 'H'
    ]
    # Remove any deuteriums
    struct = struct[
        struct.element != 'D'
    ]
    # Remove any Ns
    struct = struct[
        struct.res_name != 'N'
    ]

    # Get the bonds and their types
    if connect_via == 'residue_names':
        struct.bonds = connect_via_residue_names(
            struct
        )
    elif connect_via == 'distances':
        struct.bonds = connect_via_distances(
            struct
        )
    else:
        raise ValueError(
            "'connect_via' must be either 'residue_names' or"
            " 'distances'."
        )
    # Add formal charges
    struct.charge = np.zeros(struct.shape[0])
    # Add in predicted hydrogens
    struct, _ = hydride.add_hydrogen(struct)
    # Get the atom names without primes
    struct.atom_name = unprime(struct.atom_name)
    # Add the residue name on to the atom name if it is not on the backbone
    for i, name in enumerate(struct.atom_name):
        if name not in Backbone.list():
            struct.atom_name[i] = struct.res_name[i] + \
                '_' + struct.atom_name[i]
    # Get rid of any weird atoms
    struct = struct[
        np.isin(struct.atom_name, RibonucleicAcid.list())
    ]
    # Get rid of any weird bases
    struct.res_name = np.char.replace(struct.res_name, "DU", "U")
    # Get the coordinates
    coordinates = struct.coord
    # Get the bonds
    bonds = struct.bonds.as_array()
    bond_edges = bonds[:, 0:2]
    bond_types = bonds[:, 2]

    # Get the atom names as integers
    atom_names = np.array([
        RibonucleicAcid[element].value
        for element in struct.atom_name
    ]).astype(np.int64)
    # Get the elements as integers
    elements = np.array([
        Element[element].value
        for element in struct.element
    ]).astype(np.int64)
    # Get the residues as integers
    residues = np.array([
        Residue[res].value
        for res in struct.res_name
    ]).astype(np.int64)

    # Remove any weird bonds
    bond_src = atom_names[bond_edges[:, 0]]
    bond_dst = atom_names[bond_edges[:, 1]]
    ix = ATOM_PAIR_TO_ENUM[bond_src, bond_dst] != -1
    bond_edges = bond_edges[ix]
    bond_types = bond_types[ix]

    # Get the size of each residue
    residue_sizes = np.bincount(
        struct.res_id - struct.res_id.min()
    )
    residue_sizes = residue_sizes[
        residue_sizes > 0
    ]
    # Get the size of each chain
    _, chain_ix = np.unique(
        struct.chain_id,
        return_inverse=True,
    )
    chain_sizes = np.bincount(chain_ix)
    chain_sizes = chain_sizes[
        chain_sizes > 0
    ]

    return (
        coordinates,
        bond_edges,
        bond_types,
        atom_names,
        elements,
        residues,
        residue_sizes,
        chain_sizes,
    )


def to_cif(
    coordinates: np.ndarray,
    elements: list,
    residue_ix: np.ndarray,
    file: str,
    overwrite: bool = False,
) -> None:
    """
    Save a structure to a PDB file.
    """

    # Check if the file exists already
    if os.path.exists(file) and not overwrite:
        raise OSError(
            f"The file {file} already exists and overwrite was not"
            " set."
        )
    # Create an AtomArray
    num_atoms = len(coordinates)
    atom_array = AtomArray(num_atoms)
    # Assign coordinates
    atom_array.coord = coordinates
    # Set element symbols and residue indices
    atom_array.element = np.array(elements, dtype="U2")
    atom_array.res_id = np.array(residue_ix)

    # Create a CIFFile and add the AtomArray structure
    cif_file = CIFFile()
    set_structure(cif_file, atom_array)
    # Save the CIF file
    with open(file, "w") as f:
        f.write(str(cif_file))


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
    atom_names_ls = []
    elements_ls = []

    residue_sizes_ls = []
    chain_sizes_ls = []
    molecule_sizes_ls = []
    edge_sizes_ls = []

    prev_edge_ix = 0

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
         atom_names,
         elements,
         residues,
         residue_sizes,
         chain_sizes) = read_structure(
            os.path.join(directory, file),
            connect_via,
        )

        coordinates_ls.append(coordinates)
        residues_ls.append(residues)
        elements_ls.append(elements)
        atom_names_ls.append(atom_names)
        bond_edges_ls.append(
            bond_edges + prev_edge_ix
        )
        prev_edge_ix += coordinates.shape[0]
        bond_types_ls.append(bond_types)

        residue_sizes_ls.append(
            residue_sizes
        )
        chain_sizes_ls.append(
            chain_sizes
        )
        molecule_sizes_ls.append(
            coordinates.shape[0]
        )
        edge_sizes_ls.append(
            bond_edges.shape[0]
        )

    coordinates = np.concatenate(coordinates_ls)
    residues = np.concatenate(residues_ls).astype(np.int64)
    elements = np.concatenate(elements_ls).astype(np.int64)
    atom_names = np.concatenate(atom_names_ls).astype(np.int64)

    bond_edges = np.concatenate(bond_edges_ls).astype(np.int64)
    bond_types = np.concatenate(bond_types_ls).astype(np.int64)
    residue_sizes = np.concatenate(residue_sizes_ls).astype(np.int64)
    chain_sizes = np.concatenate(chain_sizes_ls).astype(np.int64)
    molecule_sizes = np.array(molecule_sizes_ls).astype(np.int64)
    edge_sizes = np.array(edge_sizes_ls).astype(np.int64)

    ds = xr.Dataset(
        {
            COORDINATES_KEY: ([ATOM_DIM, COORDINATE_DIM], coordinates),
            EDGES_KEY: ([EDGE_DIM, SRCDST_DIM], bond_edges),
            RESIDUE_SIZES_KEY: ([RESIDUE_DIM], residue_sizes),
            CHAIN_SIZES_KEY: ([CHAIN_DIM], chain_sizes),
            MOLECULE_SIZES_KEY: ([MOLECULE_DIM], molecule_sizes),
            EDGE_SIZES_KEY: ([MOLECULE_DIM], edge_sizes),
            ID_KEY: ([MOLECULE_DIM], ids),
            ELEMENTS_KEY: ([ATOM_DIM], elements),
            'atom_names': ([ATOM_DIM], atom_names),
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
        # Open the dataset
        self.ds = xr.open_dataset(file)
        # Store the size of the residues, chains, and molecules
        self.residue_sizes = torch.from_numpy(
            self.ds[RESIDUE_SIZES_KEY].values
        ).long()
        self.chain_sizes = torch.from_numpy(
            self.ds[CHAIN_SIZES_KEY].values
        ).long()
        self.molecule_sizes = torch.from_numpy(
            self.ds[MOLECULE_SIZES_KEY].values
        ).long()
        # Store the number of edges per molecule
        self.edges_per_molecule = torch.from_numpy(
            self.ds[EDGE_SIZES_KEY].values
        ).long()

        # Drop variables from the dataset
        self.ds = self.ds.drop_vars(dropped_features)
        # Get the number of residues and chains per molecule
        self.residues_per_molecule = _partitionCount(
            self.residue_sizes, self.molecule_sizes,
        )
        self.chains_per_molecule = _partitionCount(
            self.chain_sizes, self.molecule_sizes,
        )

        # Get the starting position of each atom group in the dataset
        self.atom_starts = padded_cumsum(
            self.molecule_sizes
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

        self.slicer2 = Slicer({
            RESIDUE_DIM: self.residue_starts,
            CHAIN_DIM: self.chain_starts,
            MOLECULE_DIM: self.molecule_starts,
        })

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
        data_slice2 = self.slicer2(start, stop)

        residue_slice = data_slice2[RESIDUE_DIM]
        chain_slice = data_slice2[CHAIN_DIM]
        molecule_slice = data_slice2[MOLECULE_DIM]
        # Load the relevant data from the dataset
        data = self.ds.isel(data_slice)
        # Extract the coordinates, elements, atom_names, and edges for
        # constructing the polymer
        coordinates = torch.Tensor(
            data[COORDINATES_KEY].values
        )
        elements = torch.Tensor(
            data[ELEMENTS_KEY].values
        )
        residues = torch.Tensor(
            data['residues'].values
        )
        atom_names = torch.Tensor(
            data['atom_names'].values
        )
        edges = torch.Tensor(
            data[EDGES_KEY].values
        )
        # Get the sizes of the relevant objects
        residue_sizes = self.residue_sizes[residue_slice]
        chain_sizes = self.chain_sizes[chain_slice]
        molecule_sizes = self.molecule_sizes[molecule_slice]
        # Reset the edges to begin at zero too
        edges = edges - edges.min()
        # Extract the additional feautures the user wants
        user_features = [
            torch.Tensor(data[feature].values)
            for feature in self.user_features
        ]
        # Move to the correct datatype and device
        residue_sizes = residue_sizes.to(self.device)
        chain_sizes = chain_sizes.to(self.device)
        molecule_sizes = molecule_sizes.to(self.device)
        coordinates = coordinates.to(
            dtype=torch.float32,
            device=self.device,
        )
        elements = elements.to(
            dtype=torch.long,
            device=self.device,
        )
        residues = residues.to(
            dtype=torch.long,
            device=self.device,
        )
        atom_names = atom_names.to(
            dtype=torch.long,
            device=self.device,
        )
        edges = edges.to(
            dtype=torch.long,
            device=self.device,
        )
        user_features = [
            feature.to(self.device)
            for feature in user_features
        ]
        # Return, including the relevant indices as well
        return (
            coordinates,
            elements,
            residues,
            atom_names,
            edges,
            residue_sizes,
            chain_sizes,
            molecule_sizes,
            *user_features,
        )

    def num_atoms(
        self: StructureDataset,
    ) -> int:
        """
        The number of atoms in the dataset.
        """

        return self.molecule_sizes.sum().item()

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

        return len(self.molecule_sizes)

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
    ) -> None:
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
