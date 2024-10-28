from __future__ import annotations
import torch
from biotite.structure.io import load_structure
from biotite.structure import connect_via_residue_names
import numpy as np

ELEMENT_IX = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "P": 4,
    "D": 0,
}
NUM_ELEMENTS = len(ELEMENT_IX)
RES_IX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "U": 3,
}
NUM_RES = len(RES_IX)
PDB_SUFFIX = '.pdb'


def read_pdb(file: str) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Read information from a PDB file.
    """

    # Load the structure into an AtomArray
    struct = load_structure(file)
    # Get the coordinates
    coordinates = torch.from_numpy(
        struct.coord,
    ).to(torch.float32)
    # Get the bonds and their types
    bonds = connect_via_residue_names(
        struct
    ).as_array()
    bond_src = torch.from_numpy(
        bonds[:, 0]
    ).long()
    bond_dst = torch.from_numpy(
        bonds[:, 1]
    ).long()
    bond_type = torch.from_numpy(
        bonds[:, 2]
    ).long()
    # Get the elements as integers
    elements = torch.tensor([
        ELEMENT_IX[element]
        for element in struct.element
    ]).long()[:, None, None]
    # Get the residues as integers
    residues = torch.tensor([
        RES_IX[res]
        for res in struct.res_name
    ]).long()[:, None, None]
    # Get which residue each atom belongs to
    residue_ix = torch.from_numpy(
        struct.res_id,
    ).long()
    residue_ix = residue_ix - residue_ix[0]
    # The residue indices are often screwed up in the PDB file. We
    # re-index them here.
    diffs = torch.cat(
        (torch.tensor([0]), residue_ix[1:] != residue_ix[:-1])
    )
    residue_ix = torch.cumsum(diffs, dim=0)
    # Get which chain each atom belongs to
    _, chain_ix = np.unique(
        struct.chain_id,
        return_inverse=True,
    )
    chain_ix = torch.from_numpy(chain_ix).long()
    return (
        coordinates,
        bond_src,
        bond_dst,
        bond_type,
        elements,
        residues,
        residue_ix,
        chain_ix,
    )
