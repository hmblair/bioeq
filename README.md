# Overview
`bioeq` is a set of tools for doing geometric deep learning with a focus on applications to structural biology. The core datatype it exposes is the `GeometricPolymer`, which contains node and edge information at varying levels of coarseness. A set of equivariant deep learning layers, including an equivariant transformer, allow for transformations of these polymers to be learned.

# Installation
To install, clone the repo and install with `pip`.
```
git clone https://github.com/hmblair/bioeq
cd bioeq
pip install .
```

# Usage
Import the `GeometricPolymer` class, and load the example PDB with a nearest-neighbour edge degree of 16.
```
from bioeq.polymer import GeometricPolymer

file = 'examples/6xrz.pdb'
edge_degree = 16
polymer = GeometricPolymer.from_pdb(file, edge_degree)
```

Print the polymer;
```
GeometricPolymer:
    num_molecules:    1
    num_chains:       1
    num_residues:     88
    num_atoms:        2808
    num_edges:        44928
    node_features:
        repr dim:     1
        multiplicity: 1
    edge_feature dim: 1
```
By default, the node features are initialised to a single degree-0 feature, which is the element type. Likewise, the edge features are initialised using the pairwise distances.

Create a single degree-0 representation for our input features, and a degree-1 representation with multiplicity 4 as an example output. Also, create a hidden representation with degrees 0 and 1, and multiplicity 16.
```
from bioeq.geom import Repr

in_repr = Repr([0], 1)
out_repr = Repr([1], 4)
hidden_repr = Repr([0, 1], 16)
```

Instantiaze an equivariant transformer with the single default edge feature as input, and a hidden size in the radial weight computation of 16. Set the hidden layers to 2.
```
from bioeq.modules import EquivariantTransformer

edge_dim = 1
edge_hidden_dim = 16
nlayers = 2

transformer = EquivariantTransformer(
   in_repr,
   out_repr,
   hidden_repr,
   nlayers,
   edge_dim,
   edge_hidden_dim,
)
```

Pass the polymer to the equivariant transformer to get a new set of node features.
```
out_polymer = transformer.polymer(polymer)
```

Print the output polymer;
```
GeometricPolymer:
    num_molecules:    1
    num_chains:       1
    num_residues:     88
    num_atoms:        2808
    num_edges:        44928
    node_features:
        repr dim:     3
        multiplicity: 4
    edge_feature dim: 1
```

Note how the node features have changed in degree and multiplicity, but the structure of the molecule has remained teh same, as have the edge features.
