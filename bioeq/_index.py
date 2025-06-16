from __future__ import annotations
from enum import Enum
import torch
import itertools
from bioeq._cpp import _connectedSubgraphs


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
        ]).long()

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


class AtomEnum(IndexEnum):

    def __init__(
        self: AtomEnum,
        value: int,
        element: Element,
        orbital: Orbital,
    ):
        self._value_ = value
        self._orbital_ = orbital
        self._element_ = element

    @property
    def element(self):
        return self._element_

    @property
    def orbital(self):
        return self._orbital_

    @classmethod
    def elements(
        cls: type[AtomEnum],
    ) -> torch.Tensor:
        return torch.tensor([
            atom.element.value
            for atom in cls
        ])

    @classmethod
    def fulldict(
        cls: type[AtomEnum],
        modifier: str = ''
    ) -> dict[str, tuple[int, Element, Orbital]]:
        """
        Return the enum as a dict.
        """

        return {
            modifier + field.name: (field.value, field.element, field.orbital)
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
    H = 0, 1
    C = 1, 6
    N = 2, 7
    O = 3, 8
    P = 4, 15
    S = 5, 16

    def __init__(
        self: Element,
        value: int,
        z: int,
    ):
        self._value_ = value
        self._z_ = z

    @property
    def z(
        self: Element,
    ) -> int:
        return self._z_

    @classmethod
    def atomic_number(
        cls: type[Element],
    ) -> torch.Tensor:

        return torch.tensor(
            [element.z for element in cls]
        ).long()


class Orbital(IndexEnum):
    s = 0
    sp2 = 1
    sp3 = 2


class Donors(IndexEnum):
    N = Element.N.value
    O = Element.O.value


class Acceptors(IndexEnum):
    N = Element.N.value
    O = Element.O.value


class Adenine(AtomEnum):
    N1 = 0, Element.N, Orbital.sp2
    C2 = 1, Element.C, Orbital.sp2
    N3 = 2, Element.N, Orbital.sp2
    C4 = 3, Element.C, Orbital.sp2
    C5 = 4, Element.C, Orbital.sp2
    C6 = 5, Element.C, Orbital.sp2
    N6 = 6, Element.N, Orbital.sp3
    N7 = 7, Element.N, Orbital.sp2
    C8 = 8, Element.C, Orbital.sp2
    N9 = 9, Element.N, Orbital.sp2
    H2 = 10, Element.H, Orbital.s
    H8 = 11, Element.H, Orbital.s
    H61 = 12, Element.H, Orbital.s
    H62 = 13, Element.H, Orbital.s


class Cytosine(AtomEnum):
    N1 = 14, Element.N, Orbital.sp2
    C2 = 15, Element.C, Orbital.sp2
    O2 = 16, Element.O, Orbital.sp2
    N3 = 17, Element.N, Orbital.sp2
    C4 = 18, Element.C, Orbital.sp2
    N4 = 19, Element.N, Orbital.sp3
    C5 = 20, Element.C, Orbital.sp2
    C6 = 21, Element.C, Orbital.sp2
    H5 = 22, Element.H, Orbital.s
    H6 = 23, Element.H, Orbital.s
    H41 = 24, Element.H, Orbital.s
    H42 = 25, Element.H, Orbital.s


class Guanine(AtomEnum):
    N1 = 26, Element.N, Orbital.sp2
    C2 = 27, Element.C, Orbital.sp2
    N2 = 28, Element.N, Orbital.sp3
    N3 = 29, Element.N, Orbital.sp2
    C4 = 30, Element.C, Orbital.sp2
    C5 = 31, Element.C, Orbital.sp2
    C6 = 32, Element.C, Orbital.sp2
    O6 = 33, Element.O, Orbital.sp2
    N7 = 34, Element.N, Orbital.sp2
    C8 = 35, Element.C, Orbital.sp2
    N9 = 36, Element.N, Orbital.sp2
    H1 = 37, Element.H, Orbital.s
    H8 = 38, Element.H, Orbital.s
    H21 = 39, Element.H, Orbital.s
    H22 = 40, Element.H, Orbital.s


class Uracil(AtomEnum):
    N1 = 41, Element.N, Orbital.sp2
    C2 = 42, Element.C, Orbital.sp2
    O2 = 43, Element.O, Orbital.sp2
    N3 = 44, Element.N, Orbital.sp2
    C4 = 45, Element.C, Orbital.sp2
    O4 = 46, Element.O, Orbital.sp2
    C5 = 47, Element.C, Orbital.sp2
    C6 = 48, Element.C, Orbital.sp2
    H3 = 49, Element.H, Orbital.s
    H5 = 50, Element.H, Orbital.s
    H6 = 51, Element.H, Orbital.s


class Ribose(AtomEnum):
    C1p = 52, Element.C, Orbital.sp3
    H1p = 53, Element.H, Orbital.s
    C2p = 54, Element.C, Orbital.sp3
    H2p = 55, Element.H, Orbital.s
    O2p = 56, Element.O, Orbital.sp3
    HO2p = 57, Element.H, Orbital.s
    C3p = 58, Element.C, Orbital.sp3
    H3p = 59, Element.H, Orbital.s
    O3p = 60, Element.O, Orbital.sp3
    HO3p = 61, Element.H, Orbital.s
    C4p = 62, Element.C, Orbital.sp3
    H4p = 63, Element.H, Orbital.s
    O4p = 64, Element.O, Orbital.sp3
    C5p = 65, Element.C, Orbital.sp3
    H5p = 66, Element.H, Orbital.s
    H5pp = 67, Element.H, Orbital.s
    O5p = 68, Element.O, Orbital.sp3
    HO5p = 69, Element.H, Orbital.s


class Phosphate(AtomEnum):
    P = 70, Element.P, Orbital.sp3
    OP1 = 71, Element.O, Orbital.sp3
    OP2 = 72, Element.O, Orbital.sp3
    HOP2 = 73, Element.H, Orbital.s


class CoarseGrainedTriplet(IndexEnum):
    P = Phosphate.P.value
    C5p = Ribose.C5p.value
    A_N9 = Adenine.N9.value
    G_N9 = Guanine.N9.value
    C_N1 = Cytosine.N1.value
    U_N1 = Uracil.N1.value


class Glyc(IndexEnum):
    A_N9 = Adenine.N9.value
    G_N9 = Guanine.N9.value
    C_N1 = Cytosine.N1.value
    U_N1 = Uracil.N1.value


Adenosine = AtomEnum(
    "Adenosine", Adenine.fulldict() | Ribose.fulldict() | Phosphate.fulldict()
)

Guanosine = IndexEnum(
    "Guanosine", Guanine.dict() | Ribose.dict() | Phosphate.dict()
)

Cytidine = IndexEnum(
    "Cytidine", Cytosine.dict() | Ribose.dict() | Phosphate.dict()
)

Uridine = IndexEnum(
    "Uridine", Uracil.dict() | Ribose.dict() | Phosphate.dict()
)

Purine = IndexEnum(
    "Purine", Cytosine.dict("C_") | Uracil.dict("U_")
)

Pyrimidine = IndexEnum(
    "Pyrimidine", Adenine.dict("A_") | Guanine.dict("G_")
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
    Adenine.list() +
    Cytosine.list() +
    Guanine.list() +
    Uracil.list() +
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
    (Adenine.N9, Adenine.C8),
    (Adenine.C8, Adenine.N7),
    (Adenine.N7, Adenine.C5),
    (Adenine.C5, Adenine.C6),
    (Adenine.C6, Adenine.N6),
    (Adenine.C4, Adenine.N3),
    (Adenine.C4, Adenine.N9),
    (Adenine.C4, Adenine.C5),
    (Adenine.N3, Adenine.C2),
    (Adenine.C2, Adenine.N1),
    (Adenine.N1, Adenine.C6),
    (Adenine.C2, Adenine.H2),
    (Adenine.C8, Adenine.H8),
    (Adenine.N6, Adenine.H61),
    (Adenine.N6, Adenine.H62),
])

_GUANOSINE_BONDS = _Bonds([
    (Guanine.N9, Guanine.C8),
    (Guanine.C8, Guanine.N7),
    (Guanine.N7, Guanine.C5),
    (Guanine.C5, Guanine.C6),
    (Guanine.C6, Guanine.O6),
    (Guanine.C4, Guanine.N3),
    (Guanine.C4, Guanine.N9),
    (Guanine.C4, Guanine.C5),
    (Guanine.N3, Guanine.C2),
    (Guanine.C2, Guanine.N2),
    (Guanine.C2, Guanine.N1),
    (Guanine.N1, Guanine.C6),
    (Guanine.C8, Guanine.H8),
    (Guanine.N1, Guanine.H1),
    (Guanine.N2, Guanine.H21),
    (Guanine.N2, Guanine.H22),
])

_CYTIDINE_BONDS = _Bonds([
    (Cytosine.N1, Cytosine.C2),
    (Cytosine.C2, Cytosine.O2),
    (Cytosine.C2, Cytosine.N3),
    (Cytosine.N3, Cytosine.C4),
    (Cytosine.C4, Cytosine.N4),
    (Cytosine.C4, Cytosine.C5),
    (Cytosine.C5, Cytosine.C6),
    (Cytosine.C6, Cytosine.N1),
    (Cytosine.C5, Cytosine.H5),
    (Cytosine.C6, Cytosine.H6),
    (Cytosine.N4, Cytosine.H41),
    (Cytosine.N4, Cytosine.H42),
])

_URIDINE_BONDS = _Bonds([
    (Uracil.N1, Uracil.C2),
    (Uracil.C2, Uracil.O2),
    (Uracil.C2, Uracil.N3),
    (Uracil.N3, Uracil.C4),
    (Uracil.C4, Uracil.O4),
    (Uracil.C4, Uracil.C5),
    (Uracil.C5, Uracil.C6),
    (Uracil.C6, Uracil.N1),
    (Uracil.N3, Uracil.H3),
    (Uracil.C5, Uracil.H5),
    (Uracil.C6, Uracil.H6),
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
    (Ribose.O3p, Ribose.HO3p),
    (Ribose.O5p, Ribose.HO5p),

])

_GLYC_BONDS = _Bonds([
    (Ribose.C1p, Adenine.N9),
    (Ribose.C1p, Guanine.N9),
    (Ribose.C1p, Cytosine.N1),
    (Ribose.C1p, Uracil.N1),
])

_PHOSPHATE_BONDS = _Bonds([
    (Phosphate.P, Ribose.O5p),
    (Phosphate.P, Ribose.O3p),
    (Phosphate.P, Phosphate.OP1),
    (Phosphate.P, Phosphate.OP2),
    (Phosphate.OP2, Phosphate.HOP2),
])

_AU_PAIRING_BONDS = _Bonds([
    (Uracil.H3, Adenine.N1),
    (Adenine.H61, Uracil.O4),
])

_GC_PAIRING_BONDS = _Bonds([
    (Guanine.H1, Cytosine.N3),
    (Guanine.H21, Cytosine.O2),
    (Cytosine.H41, Guanine.O6),
])

_BONDS = (
    _ADENOSINE_BONDS +
    _CYTIDINE_BONDS +
    _GUANOSINE_BONDS +
    _URIDINE_BONDS +
    _RIBOSE_BONDS +
    _GLYC_BONDS +
    _PHOSPHATE_BONDS
)
_BI_BONDS = torch.cat(
    [_BONDS.ix, _BONDS.ix.flip(1)],
    dim=0,
)
_HBONDS = (
    _AU_PAIRING_BONDS +
    _GC_PAIRING_BONDS
)


_NUM_ATOMS = len(RibonucleicAcid)
_NUM_BONDS = len(_BONDS)


_ALL_PAIRS = []
for atom1, atom2 in itertools.product(RibonucleicAcid, RibonucleicAcid):
    if (atom1, atom2) not in _ALL_PAIRS and (atom2, atom1) not in _ALL_PAIRS:
        _ALL_PAIRS.append(
            (atom1, atom2)
        )


#
# For converting between the different representations
#


_ADENOSINE_ATOM_TO_ENUM = {
    'A_' + value.name: value
    for value in Adenine
}

_GUANOSINE_ATOM_TO_ENUM = {
    'G_' + value.name: value
    for value in Guanine
}

_CYTIDINE_ATOM_TO_ENUM = {
    'C_' + value.name: value
    for value in Cytosine
}

_URIDINE_ATOM_TO_ENUM = {
    'U_' + value.name: value
    for value in Uracil
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

ALL_PAIR_TO_ENUM = torch.ones(
    _NUM_ATOMS, _NUM_ATOMS
).to(torch.long) * -1

ix = 0
for atom1, atom2 in _ALL_PAIRS:
    ALL_PAIR_TO_ENUM[
        atom1.value, atom2.value
    ] = ix
    ALL_PAIR_TO_ENUM[
        atom2.value, atom1.value
    ] = ix
    ix += 1

num_pairs = ALL_PAIR_TO_ENUM.max().item() + 1
ENUM_TO_ALL_PAIR = torch.zeros(
    num_pairs, 2
).to(torch.long)
ix = 0
for atom1, atom2 in _ALL_PAIRS:
    ENUM_TO_ALL_PAIR[ix, 0] = atom1.value
    ENUM_TO_ALL_PAIR[ix, 1] = atom2.value
    ix += 1


#
# Get all connected triplets of atoms
#


_TRIPLETS = _connectedSubgraphs(
    _BI_BONDS, 3
)
# We need to exclude any bonds that include two glycostosidic atoms as these
# are artifacts of the triplet construction scheme
_TRIPLETS = _TRIPLETS[
    (_TRIPLETS[..., None] == Glyc.index()[None, None, :]).sum((1, 2)) < 2
]

TRIPLET_TO_ENUM = torch.ones(
    *([_NUM_ATOMS] * 3)
).to(torch.long) * -1

for ix, atoms in enumerate(_TRIPLETS.tolist()):
    for p_atoms in itertools.permutations(atoms):
        TRIPLET_TO_ENUM[*p_atoms] = ix

#
# Get all connected quadruplets of atoms
#


_QUADS = _connectedSubgraphs(
    _BI_BONDS, 4
)

# We need to exclude any bonds that include two glycostosidic atoms as these
# are artifacts of the triplet construction scheme
_QUADS = _QUADS[
    (_QUADS[..., None] == Glyc.index()[None, None, :]).sum((1, 2)) < 2
]

QUAD_TO_ENUM = torch.ones(
    *([_NUM_ATOMS] * 4)
).to(torch.long) * -1

for ix, atoms in enumerate(_QUADS.tolist()):
    for p_atoms in itertools.permutations(atoms):
        QUAD_TO_ENUM[*p_atoms] = ix


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


#
# I don't know where else to put these for now
#

COULOMB = 3.32063711E2
# Thermodynamic beta in mol/kcal at 37C
BETA = 1 / 1.7351


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


PARTIAL_CHARGE = torch.zeros(
    len(RibonucleicAcid),
    dtype=torch.float32,
)
VdW_SIGMA = torch.zeros(
    len(RibonucleicAcid),
    dtype=torch.float32,
)
VdW_EPSILON = torch.zeros(
    len(RibonucleicAcid),
    dtype=torch.float32,
)

VdW_SIGMA[Ribose.O5p.value] = 3.1569886976021064
VdW_EPSILON[Ribose.O5p.value] = 0.17
VdW_SIGMA[Ribose.C5p.value] = 3.3996695084235347
VdW_EPSILON[Ribose.C5p.value] = 0.1094
VdW_SIGMA[Ribose.H5p.value] = 2.471353044121301
VdW_EPSILON[Ribose.H5p.value] = 0.0157
VdW_SIGMA[Ribose.H5pp.value] = 2.471353044121301
VdW_EPSILON[Ribose.H5pp.value] = 0.0157
VdW_SIGMA[Ribose.C4p.value] = 3.3996695084235347
VdW_EPSILON[Ribose.C4p.value] = 0.1094
VdW_SIGMA[Ribose.H4p.value] = 2.471353044121301
VdW_EPSILON[Ribose.H4p.value] = 0.0157
VdW_SIGMA[Ribose.O4p.value] = 3.0000123434657784
VdW_EPSILON[Ribose.O4p.value] = 0.17
VdW_SIGMA[Ribose.C1p.value] = 3.3996695084235347
VdW_EPSILON[Ribose.C1p.value] = 0.1094
VdW_SIGMA[Ribose.H1p.value] = 2.2931733004932333
VdW_EPSILON[Ribose.H1p.value] = 0.0157
VdW_SIGMA[Ribose.C3p.value] = 3.3996695084235347
VdW_EPSILON[Ribose.C3p.value] = 0.1094
VdW_SIGMA[Ribose.H3p.value] = 2.471353044121301
VdW_EPSILON[Ribose.H3p.value] = 0.0157
VdW_SIGMA[Ribose.C2p.value] = 3.3996695084235347
VdW_EPSILON[Ribose.C2p.value] = 0.1094
VdW_SIGMA[Ribose.H2p.value] = 2.471353044121301
VdW_EPSILON[Ribose.H2p.value] = 0.0157
VdW_SIGMA[Ribose.O2p.value] = 3.0664733878390478
VdW_EPSILON[Ribose.O2p.value] = 0.2104
VdW_SIGMA[Ribose.HO2p.value] = 0.0
VdW_EPSILON[Ribose.HO2p.value] = 0.0
VdW_SIGMA[Ribose.O3p.value] = 3.1569886976021064
VdW_EPSILON[Ribose.O3p.value] = 0.17

VdW_SIGMA[Adenine.N9.value] = 3.2890020696561417
VdW_EPSILON[Adenine.N9.value] = 0.17
VdW_SIGMA[Adenine.C8.value] = 3.300405573248338
VdW_EPSILON[Adenine.C8.value] = 0.0636
VdW_SIGMA[Adenine.H8.value] = 2.421462715905442
VdW_EPSILON[Adenine.H8.value] = 0.015
VdW_SIGMA[Adenine.N7.value] = 3.2890020696561417
VdW_EPSILON[Adenine.N7.value] = 0.17
VdW_SIGMA[Adenine.C5.value] = 3.300405573248338
VdW_EPSILON[Adenine.C5.value] = 0.0636
VdW_SIGMA[Adenine.C6.value] = 3.300405573248338
VdW_EPSILON[Adenine.C6.value] = 0.0636
VdW_SIGMA[Adenine.N6.value] = 3.3507057148745414
VdW_EPSILON[Adenine.N6.value] = 0.17
VdW_SIGMA[Adenine.H61.value] = 0.0
VdW_EPSILON[Adenine.H61.value] = 0.0
VdW_SIGMA[Adenine.H62.value] = 0.0
VdW_EPSILON[Adenine.H62.value] = 0.0
VdW_SIGMA[Adenine.N1.value] = 3.2890020696561417
VdW_EPSILON[Adenine.N1.value] = 0.17
VdW_SIGMA[Adenine.C2.value] = 3.300405573248338
VdW_EPSILON[Adenine.C2.value] = 0.0636
VdW_SIGMA[Adenine.H2.value] = 2.421462715905442
VdW_EPSILON[Adenine.H2.value] = 0.015
VdW_SIGMA[Adenine.N3.value] = 3.2890020696561417
VdW_EPSILON[Adenine.N3.value] = 0.17
VdW_SIGMA[Adenine.C4.value] = 3.300405573248338
VdW_EPSILON[Adenine.C4.value] = 0.0636

VdW_SIGMA[Cytosine.N1.value] = 3.3507057148745414
VdW_EPSILON[Cytosine.N1.value] = 0.17
VdW_SIGMA[Cytosine.C6.value] = 3.28499302542451
VdW_EPSILON[Cytosine.C6.value] = 0.0538
VdW_SIGMA[Cytosine.H6.value] = 2.5105525877194763
VdW_EPSILON[Cytosine.H6.value] = 0.015
VdW_SIGMA[Cytosine.C5.value] = 3.28499302542451
VdW_EPSILON[Cytosine.C5.value] = 0.0538
VdW_SIGMA[Cytosine.H5.value] = 2.59964245953351
VdW_EPSILON[Cytosine.H5.value] = 0.015
VdW_SIGMA[Cytosine.C4.value] = 3.28499302542451
VdW_EPSILON[Cytosine.C4.value] = 0.0538
VdW_SIGMA[Cytosine.N4.value] = 3.3507057148745414
VdW_EPSILON[Cytosine.N4.value] = 0.17
VdW_SIGMA[Cytosine.H41.value] = 0.0
VdW_EPSILON[Cytosine.H41.value] = 0.0
VdW_SIGMA[Cytosine.H42.value] = 0.0
VdW_EPSILON[Cytosine.H42.value] = 0.0
VdW_SIGMA[Cytosine.N3.value] = 3.3507057148745414
VdW_EPSILON[Cytosine.N3.value] = 0.17
VdW_SIGMA[Cytosine.C2.value] = 3.28499302542451
VdW_EPSILON[Cytosine.C2.value] = 0.0538
VdW_SIGMA[Cytosine.O2.value] = 2.959921901149463
VdW_EPSILON[Cytosine.O2.value] = 0.21


VdW_SIGMA[Phosphate.P.value] = 3.741774616189425
VdW_EPSILON[Phosphate.P.value] = 0.2
VdW_SIGMA[Phosphate.OP1.value] = 3.116898255285791
VdW_EPSILON[Phosphate.OP1.value] = 0.21
VdW_SIGMA[Phosphate.OP2.value] = 3.116898255285791
VdW_EPSILON[Phosphate.OP2.value] = 0.21

VdW_SIGMA[Guanine.N9.value] = 3.2890020696561417
VdW_EPSILON[Guanine.N9.value] = 0.17
VdW_SIGMA[Guanine.C8.value] = 3.28499302542451
VdW_EPSILON[Guanine.C8.value] = 0.0538
VdW_SIGMA[Guanine.H8.value] = 2.421462715905442
VdW_EPSILON[Guanine.H8.value] = 0.015
VdW_SIGMA[Guanine.N7.value] = 3.2890020696561417
VdW_EPSILON[Guanine.N7.value] = 0.17
VdW_SIGMA[Guanine.C5.value] = 3.28499302542451
VdW_EPSILON[Guanine.C5.value] = 0.0538
VdW_SIGMA[Guanine.C6.value] = 3.28499302542451
VdW_EPSILON[Guanine.C6.value] = 0.0538
VdW_SIGMA[Guanine.O6.value] = 2.959921901149463
VdW_EPSILON[Guanine.O6.value] = 0.21
VdW_SIGMA[Guanine.N1.value] = 3.3507057148745414
VdW_EPSILON[Guanine.N1.value] = 0.17
VdW_SIGMA[Guanine.H1.value] = 0.0
VdW_EPSILON[Guanine.H1.value] = 0.0
VdW_SIGMA[Guanine.C2.value] = 3.28499302542451
VdW_EPSILON[Guanine.C2.value] = 0.0538
VdW_SIGMA[Guanine.N2.value] = 3.3507057148745414
VdW_EPSILON[Guanine.N2.value] = 0.17
VdW_SIGMA[Guanine.H21.value] = 0.0
VdW_EPSILON[Guanine.H21.value] = 0.0
VdW_SIGMA[Guanine.H22.value] = 0.0
VdW_EPSILON[Guanine.H22.value] = 0.0
VdW_SIGMA[Guanine.N3.value] = 3.3507057148745414
VdW_EPSILON[Guanine.N3.value] = 0.17
VdW_SIGMA[Guanine.C4.value] = 3.28499302542451
VdW_EPSILON[Guanine.C4.value] = 0.0538

VdW_SIGMA[Uracil.N1.value] = 3.3507057148745414
VdW_EPSILON[Uracil.N1.value] = 0.17
VdW_SIGMA[Uracil.C6.value] = 3.28499302542451
VdW_EPSILON[Uracil.C6.value] = 0.0538
VdW_SIGMA[Uracil.H6.value] = 2.5105525877194763
VdW_EPSILON[Uracil.H6.value] = 0.015
VdW_SIGMA[Uracil.C5.value] = 3.28499302542451
VdW_EPSILON[Uracil.C5.value] = 0.0538
VdW_SIGMA[Uracil.H5.value] = 2.59964245953351
VdW_EPSILON[Uracil.H5.value] = 0.015
VdW_SIGMA[Uracil.C4.value] = 3.28499302542451
VdW_EPSILON[Uracil.C4.value] = 0.0538
VdW_SIGMA[Uracil.O4.value] = 2.959921901149463
VdW_EPSILON[Uracil.O4.value] = 0.21
VdW_SIGMA[Uracil.N3.value] = 3.3507057148745414
VdW_EPSILON[Uracil.N3.value] = 0.17
VdW_SIGMA[Uracil.H3.value] = 0.0
VdW_EPSILON[Uracil.H3.value] = 0.0
VdW_SIGMA[Uracil.C2.value] = 3.28499302542451
VdW_EPSILON[Uracil.C2.value] = 0.0538
VdW_SIGMA[Uracil.O2.value] = 2.959921901149463
VdW_EPSILON[Uracil.O2.value] = 0.21

PARTIAL_CHARGE[Ribose.O5p.value] = -0.4989
PARTIAL_CHARGE[Ribose.C5p.value] = 0.0558
PARTIAL_CHARGE[Ribose.H5p.value] = 0.0679
PARTIAL_CHARGE[Ribose.H5pp.value] = 0.0679
PARTIAL_CHARGE[Ribose.C4p.value] = 0.1065
PARTIAL_CHARGE[Ribose.H4p.value] = 0.1174
PARTIAL_CHARGE[Ribose.O4p.value] = -0.3548
PARTIAL_CHARGE[Ribose.C1p.value] = 0.0394
PARTIAL_CHARGE[Ribose.H1p.value] = 0.2007
PARTIAL_CHARGE[Ribose.C3p.value] = 0.2022
PARTIAL_CHARGE[Ribose.H3p.value] = 0.0615
PARTIAL_CHARGE[Ribose.C2p.value] = 0.067
PARTIAL_CHARGE[Ribose.H2p.value] = 0.0972
PARTIAL_CHARGE[Ribose.O2p.value] = -0.6139
PARTIAL_CHARGE[Ribose.HO2p.value] = 0.4186
PARTIAL_CHARGE[Ribose.O3p.value] = -0.5246

PARTIAL_CHARGE[Adenine.N9.value] = -0.0251
PARTIAL_CHARGE[Adenine.C8.value] = 0.2006
PARTIAL_CHARGE[Adenine.H8.value] = 0.1553
PARTIAL_CHARGE[Adenine.N7.value] = -0.6073
PARTIAL_CHARGE[Adenine.C5.value] = 0.0515
PARTIAL_CHARGE[Adenine.C6.value] = 0.7009
PARTIAL_CHARGE[Adenine.N6.value] = -1.0088
PARTIAL_CHARGE[Adenine.H61.value] = 0.4738
PARTIAL_CHARGE[Adenine.H62.value] = 0.4738
PARTIAL_CHARGE[Adenine.N1.value] = -0.7969
PARTIAL_CHARGE[Adenine.C2.value] = 0.5875
PARTIAL_CHARGE[Adenine.H2.value] = 0.065
PARTIAL_CHARGE[Adenine.N3.value] = -0.6997
PARTIAL_CHARGE[Adenine.C4.value] = 0.3053

PARTIAL_CHARGE[Cytosine.N1.value] = -0.0484
PARTIAL_CHARGE[Cytosine.C6.value] = 0.0053
PARTIAL_CHARGE[Cytosine.H6.value] = 0.1958
PARTIAL_CHARGE[Cytosine.C5.value] = -0.5215
PARTIAL_CHARGE[Cytosine.H5.value] = 0.1928
PARTIAL_CHARGE[Cytosine.C4.value] = 0.8185
PARTIAL_CHARGE[Cytosine.N4.value] = -0.8716
PARTIAL_CHARGE[Cytosine.H41.value] = 0.3827
PARTIAL_CHARGE[Cytosine.H42.value] = 0.3827
PARTIAL_CHARGE[Cytosine.N3.value] = -0.7584
PARTIAL_CHARGE[Cytosine.C2.value] = 0.7538
PARTIAL_CHARGE[Cytosine.O2.value] = -0.6252

PARTIAL_CHARGE[Phosphate.P.value] = 1.1662
PARTIAL_CHARGE[Phosphate.OP1.value] = -0.776
PARTIAL_CHARGE[Phosphate.OP2.value] = -0.776
PARTIAL_CHARGE[Phosphate.HOP2.value] = 0.0

PARTIAL_CHARGE[Guanine.N9.value] = 0.0492
PARTIAL_CHARGE[Guanine.C8.value] = 0.1374
PARTIAL_CHARGE[Guanine.H8.value] = 0.164
PARTIAL_CHARGE[Guanine.N7.value] = -0.5709
PARTIAL_CHARGE[Guanine.C5.value] = 0.1744
PARTIAL_CHARGE[Guanine.C6.value] = 0.477
PARTIAL_CHARGE[Guanine.O6.value] = -0.5597
PARTIAL_CHARGE[Guanine.N1.value] = -0.5606
PARTIAL_CHARGE[Guanine.H1.value] = 0.4243
PARTIAL_CHARGE[Guanine.C2.value] = 0.7657
PARTIAL_CHARGE[Guanine.N2.value] = -1.0158
PARTIAL_CHARGE[Guanine.H21.value] = 0.4607
PARTIAL_CHARGE[Guanine.H22.value] = 0.4607
PARTIAL_CHARGE[Guanine.N3.value] = -0.6323
PARTIAL_CHARGE[Guanine.C4.value] = 0.1222

PARTIAL_CHARGE[Uracil.N1.value] = 0.0418
PARTIAL_CHARGE[Uracil.C6.value] = -0.1126
PARTIAL_CHARGE[Uracil.H6.value] = 0.2188
PARTIAL_CHARGE[Uracil.C5.value] = -0.3635
PARTIAL_CHARGE[Uracil.H5.value] = 0.1811
PARTIAL_CHARGE[Uracil.C4.value] = 0.5952
PARTIAL_CHARGE[Uracil.O4.value] = -0.5761
PARTIAL_CHARGE[Uracil.N3.value] = -0.3913
PARTIAL_CHARGE[Uracil.H3.value] = 0.3518
PARTIAL_CHARGE[Uracil.C2.value] = 0.4687
PARTIAL_CHARGE[Uracil.O2.value] = -0.5477
