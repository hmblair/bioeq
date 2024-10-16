from __future__ import annotations
import torch
from biotite.structure.io import load_structure
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
PDB_SUFFIX = '.pdb'


def read_pdb(file: str) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
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
    # Get the elements as integers
    elements = torch.tensor([
        ELEMENT_IX[element]
        for element in struct.element
    ]).long()[:, None]
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
    chain_ix = torch.from_numpy(chain_ix)
    return (
        coordinates,
        elements,
        residue_ix,
        chain_ix,
    )
