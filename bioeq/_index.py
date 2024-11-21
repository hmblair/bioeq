from __future__ import annotations
from enum import Enum
import torch
import itertools


#
# Enumumerate atoms, elements, bonds
#


class IndexEnum(Enum):
    """
    An enum with the ability to collect all enum values into a tensor.
    """

    @classmethod
    def index(
        cls: type[IndexEnum],
    ) -> torch.Tensor:
        """
        Return a tensor of all the indices in the enum.
        """

        return torch.Tensor([
            atom.value for atom in cls
        ])

    @classmethod
    def list(
        cls: type[IndexEnum],
        modifier: str = ''
    ) -> list[str]:
        """
        Return the names in the enum as a list.
        """

        return [
            modifier + field.name
            for field in cls
        ]

    @classmethod
    def dict(
        cls: type[IndexEnum],
        modifier: str = ''
    ) -> dict[str, int]:
        """
        Return the enum as a dict.
        """

        return {
            modifier + field.name: field.value
            for field in cls
        }

    @classmethod
    def revdict(
        cls: type[IndexEnum],
        modifier: str = ''
    ) -> dict[int, str]:
        """
        Return the enum as a reversed dict.
        """

        return {
            field.value: modifier + field.name
            for field in cls
        }


class Residue(IndexEnum):
    A = 0
    C = 1
    G = 2
    U = 3
    T = 4
    N = 5


class Element(IndexEnum):
    H = 0
    C = 1
    N = 2
    O = 3
    P = 4
    S = 5


class Donors(IndexEnum):
    N = Element.N.value
    O = Element.O.value


class Acceptors(IndexEnum):
    N = Element.N.value
    O = Element.O.value


class Adenosine(IndexEnum):
    N1 = 0
    C2 = 1
    N3 = 2
    C4 = 3
    C5 = 4
    C6 = 5
    N6 = 6
    N7 = 7
    C8 = 8
    N9 = 9
    H2 = 10
    H8 = 11
    H61 = 12
    H62 = 13


class Cytidine(IndexEnum):
    N1 = 14
    C2 = 15
    O2 = 16
    N3 = 17
    C4 = 18
    N4 = 19
    C5 = 20
    C6 = 21
    H5 = 22
    H6 = 23
    H41 = 24
    H42 = 25


class Guanosine(IndexEnum):
    N1 = 26
    C2 = 27
    N2 = 28
    N3 = 29
    C4 = 30
    C5 = 31
    C6 = 32
    O6 = 33
    N7 = 34
    C8 = 35
    N9 = 36
    H1 = 37
    H8 = 38
    H21 = 39
    H22 = 40


class Uridine(IndexEnum):
    N1 = 41
    C2 = 42
    O2 = 43
    N3 = 44
    C4 = 45
    O4 = 46
    C5 = 47
    C6 = 48
    H3 = 49
    H5 = 50
    H6 = 51


class Ribose(IndexEnum):
    C1p = 52
    H1p = 53
    O1p = 54
    HO1p = 55
    C2p = 56
    H2p = 57
    O2p = 58
    HO2p = 59
    C3p = 60
    H3p = 61
    O3p = 62
    HO3p = 63
    C4p = 64
    H4p = 65
    O4p = 66
    C5p = 67
    H5p = 68
    H5pp = 69
    O5p = 70
    HO5p = 71


class Phosphate(IndexEnum):
    P = 72
    OP1 = 73
    OP2 = 74
    HOP2 = 75
    OP3 = 76
    HOP3 = 77


class CoarseGrainedTriplet(IndexEnum):
    P = Phosphate.P.value
    C5p = Ribose.C5p.value
    A_N9 = Adenosine.N9.value
    G_N9 = Guanosine.N9.value
    C_N1 = Cytidine.N1.value
    U_N1 = Uridine.N1.value


Purine = IndexEnum(
    "Purine", Cytidine.dict("C_") | Uridine.dict("U_")
)

Pyrimidine = IndexEnum(
    "Pyrimidine", Adenosine.dict("A_") | Guanosine.dict("G_")
)

Nucleotide = IndexEnum(
    "Nucleotide", Purine.dict() | Pyrimidine.dict()
)

Backbone = IndexEnum(
    "Backbone", Ribose.dict() | Phosphate.dict()
)

RibonucleicAcid = IndexEnum(
    "RibonucleicAcid", Nucleotide.dict() | Backbone.dict()
)

VALID_ATOMS = (
    Adenosine.list() +
    Cytidine.list() +
    Guanosine.list() +
    Uridine.list() +
    Backbone.list()
)


class _Bonds(list):
    """
    Store a set of pairs of atom enums, as well as their integer
    representation in a tensor.
    """

    def __init__(
        self: _Bonds,
        bonds: list[tuple[Enum, Enum]]
    ) -> None:

        super().__init__(bonds)
        self.ix = torch.tensor(
            [[atom1.value, atom2.value]
             for atom1, atom2 in self]
        )

    def __add__(
        self: _Bonds,
        other: list,
    ) -> _Bonds:

        return _Bonds(super().__add__(other))


_ADENOSINE_BONDS = _Bonds([
    (Adenosine.N9, Adenosine.C8),
    (Adenosine.C8, Adenosine.N7),
    (Adenosine.N7, Adenosine.C5),
    (Adenosine.C5, Adenosine.C6),
    (Adenosine.C6, Adenosine.N6),
    (Adenosine.C6, Adenosine.C4),
    (Adenosine.C4, Adenosine.N3),
    (Adenosine.C4, Adenosine.N9),
    (Adenosine.C4, Adenosine.C5),
    (Adenosine.N3, Adenosine.C2),
    (Adenosine.C2, Adenosine.N1),
    (Adenosine.N1, Adenosine.C6),
    (Adenosine.C2, Adenosine.H2),
    (Adenosine.C8, Adenosine.H8),
    (Adenosine.N6, Adenosine.H61),
    (Adenosine.N6, Adenosine.H62),
])

_GUANOSINE_BONDS = _Bonds([
    (Guanosine.N9, Guanosine.C8),
    (Guanosine.C8, Guanosine.N7),
    (Guanosine.N7, Guanosine.C5),
    (Guanosine.C5, Guanosine.C6),
    (Guanosine.C6, Guanosine.O6),
    (Guanosine.C6, Guanosine.C4),
    (Guanosine.C4, Guanosine.N3),
    (Guanosine.C4, Guanosine.N9),
    (Guanosine.C4, Guanosine.C5),
    (Guanosine.N3, Guanosine.C2),
    (Guanosine.C2, Guanosine.N2),
    (Guanosine.C2, Guanosine.N1),
    (Guanosine.N1, Guanosine.C6),
    (Guanosine.C8, Guanosine.H8),
    (Guanosine.N1, Guanosine.H1),
    (Guanosine.N2, Guanosine.H21),
    (Guanosine.N2, Guanosine.H22),
])

_CYTIDINE_BONDS = _Bonds([
    (Cytidine.N1, Cytidine.C2),
    (Cytidine.C2, Cytidine.O2),
    (Cytidine.C2, Cytidine.N3),
    (Cytidine.N3, Cytidine.C4),
    (Cytidine.C4, Cytidine.N4),
    (Cytidine.C4, Cytidine.C5),
    (Cytidine.C5, Cytidine.C6),
    (Cytidine.C6, Cytidine.N1),
    (Cytidine.C5, Cytidine.H5),
    (Cytidine.C6, Cytidine.H6),
    (Cytidine.N4, Cytidine.H41),
    (Cytidine.N4, Cytidine.H42),
])

_URIDINE_BONDS = _Bonds([
    (Uridine.N1, Uridine.C2),
    (Uridine.C2, Uridine.O2),
    (Uridine.C2, Uridine.N3),
    (Uridine.N3, Uridine.C4),
    (Uridine.C4, Uridine.O4),
    (Uridine.C4, Uridine.C5),
    (Uridine.C5, Uridine.C6),
    (Uridine.C6, Uridine.N1),
    (Uridine.N3, Uridine.H3),
    (Uridine.C5, Uridine.H5),
    (Uridine.C6, Uridine.H6),
])

_RIBOSE_BONDS = _Bonds([
    (Ribose.C1p, Ribose.C2p),
    (Ribose.C2p, Ribose.C3p),
    (Ribose.C3p, Ribose.C4p),
    (Ribose.C4p, Ribose.C5p),
    (Ribose.C1p, Ribose.O4p),
    (Ribose.C4p, Ribose.O4p),
    (Ribose.C3p, Ribose.O3p),
    (Ribose.C2p, Ribose.O2p),
    (Ribose.C5p, Ribose.O5p),
    (Ribose.C1p, Ribose.H1p),
    (Ribose.C2p, Ribose.H2p),
    (Ribose.O2p, Ribose.HO2p),
    (Ribose.C3p, Ribose.H3p),
    (Ribose.C4p, Ribose.H4p),
    (Ribose.C5p, Ribose.H5p),
    (Ribose.C5p, Ribose.H5pp),
    (Ribose.C1p, Ribose.O1p),
    (Ribose.O1p, Ribose.HO1p),
    (Ribose.O3p, Ribose.HO3p),
    (Ribose.O5p, Ribose.HO5p),
    # Bonds to the base
    (Ribose.C1p, Adenosine.N9),
    (Ribose.C1p, Guanosine.N9),
    (Ribose.C1p, Cytidine.N1),
    (Ribose.C1p, Uridine.N1),
])

_PHOSPHATE_BONDS = _Bonds([
    (Phosphate.P, Ribose.O5p),
    (Phosphate.P, Ribose.O3p),
    (Phosphate.P, Phosphate.OP1),
    (Phosphate.P, Phosphate.OP2),
    (Phosphate.P, Phosphate.OP3),
    (Phosphate.OP2, Phosphate.HOP2),
    (Phosphate.OP3, Phosphate.HOP3),
])

_AU_PAIRING_BONDS = _Bonds([
    (Uridine.H3, Adenosine.N1),
    (Adenosine.H61, Uridine.O4),
])

_GC_PAIRING_BONDS = _Bonds([
    (Guanosine.H1, Cytidine.N3),
    (Guanosine.H21, Cytidine.O2),
    (Cytidine.H41, Guanosine.O6),
])

_BONDS = (
    _ADENOSINE_BONDS +
    _CYTIDINE_BONDS +
    _GUANOSINE_BONDS +
    _URIDINE_BONDS +
    _RIBOSE_BONDS +
    _PHOSPHATE_BONDS +
    _AU_PAIRING_BONDS +
    _GC_PAIRING_BONDS
)


_NUM_ATOMS = (
    len(Adenosine) +
    len(Cytidine) +
    len(Guanosine) +
    len(Uridine) +
    len(Ribose) +
    len(Phosphate)
)
_NUM_BONDS = len(_BONDS)


#
# For converting between the different representations
#


_ADENOSINE_ATOM_TO_ENUM = {
    'A_' + value.name: value
    for value in Adenosine
}

_GUANOSINE_ATOM_TO_ENUM = {
    'G_' + value.name: value
    for value in Guanosine
}

_CYTIDINE_ATOM_TO_ENUM = {
    'C_' + value.name: value
    for value in Cytidine
}

_URIDINE_ATOM_TO_ENUM = {
    'U_' + value.name: value
    for value in Uridine
}

_RIBOSE_ATOM_TO_ENUM = {
    'R_' + value.name: value
    for value in Ribose
}

_PHOSPHATE_ATOM_TO_ENUM = {
    'P_' + value.name: value
    for value in Phosphate
}

ATOM_TO_ENUM = {
    **_ADENOSINE_ATOM_TO_ENUM,
    **_GUANOSINE_ATOM_TO_ENUM,
    **_CYTIDINE_ATOM_TO_ENUM,
    **_URIDINE_ATOM_TO_ENUM,
    **_RIBOSE_ATOM_TO_ENUM,
    **_PHOSPHATE_ATOM_TO_ENUM
}

ATOM_PAIR_TO_ENUM = torch.ones(
    _NUM_ATOMS, _NUM_ATOMS
).to(torch.long) * -1

ix = 0
for atom1, atom2 in _BONDS:
    ATOM_PAIR_TO_ENUM[
        atom1.value, atom2.value
    ] = ix
    ATOM_PAIR_TO_ENUM[
        atom2.value, atom1.value
    ] = ix
    ix += 1


#
# Some biochemical values of relevance
#

VALENCE_TABLE = {
    Element.H: 1,
    Element.C: 4,
    Element.N: 3,
    Element.O: 2,
    Element.P: 5,
    Element.S: 6,
}
VALENCE = torch.tensor(
    list(VALENCE_TABLE.values()),
    dtype=torch.long,
)

ELECTRONEGATIVITY_TABLE = {
    Element.H: 2.20,
    Element.C: 2.55,
    Element.N: 3.04,
    Element.O: 3.44,
    Element.P: 2.19,
    Element.S: 2.58,
}
ELECTRONEGATIVITY = torch.tensor(
    list(ELECTRONEGATIVITY_TABLE.values()),
    dtype=torch.float32,
)


#
# I don't know where else to put these for now
#


class Property(Enum):
    ELEMENT = 0
    NAME = 1
    ATOM = 2
    RESIDUE = 3
    CHAIN = 4
    MOLECULE = 5
    ALL = 6


class Reduction(Enum):
    NONE = 0
    COLLATE = 1
    MEAN = 2
    SUM = 3
    MIN = 4
    MAX = 5
