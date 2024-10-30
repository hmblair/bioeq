# Overview
`bioeq` is a set of tools for doing geometric deep learning with a focus on applications to structural biology. The core datatype it exposes is the `GeometricPolymer`, which contains node and edge information at varying levels of coarseness. A set of equivariant deep learning layers, including an equivariant transformer, allow for transformations of these polymers to be learned.

# Installation
To install, clone the repo and install with `pip`.
```
git clone https://github.com/hmblair/bioeq
cd bioeq
pip3 install .
```

# Usage

A `Polymer` is a class which contains the coordinates and edges of a molecule or a set of molecules,as well as information on which chain and residue each molecule belongs to. The `PolymerDataset` will load `Polymer`s from a valid `.nc` file that was created with `create_structure_dataset`. It will also load any other features specified at instantiation.
```
from bioeq.polymer import PolymerDataset

# Create a PolymerDataset class, and request that it load the 'elements'
# and 'residues' features too
file = 'examples/polymers.nc'
dataset = PolymerDataset(
    file=file,
    atom_features=['elements', 'residues']
)

# Load the first five polymers and associated data
polymer, elements, residues = dataset[0:5]
```

Print the polymer:
```
Polymer:
    num_molecules:    1
    num_chains:       1
    num_residues:     88
    num_atoms:        2808
    num_edges:        44928
```

The easiest way to train a model with this information as input is to create a `GeometricPolymer` object. This is a `Polymer` object in combination with 
    1. node features, and
    2. edge features.
To create it, we need to compute some initial node and edge features.

The node features we can create by passing each of the elements and residues through an embedding layer.
```
import torch.nn

elem_embedding = nn.Embedding(5, 4)
residue_embedding = nn.Embedding(4, 4)

elements = elem_embedding(elements)
residues = elem_embedding(residues)

node_features = torch.cat([elements, residues], dim=-1)
```

For the edge features, we can compute the distance between all atoms which are connected by an edge.
```
edge_features = polymer.pdist()
```

We can now create our `GeometricPolymer`.
```
from bioeq.polymer import GeometricPolymer

geom_polymer = GeometricPolymer(
    polymer,
    node_features,
    edge_features,
)
```

Print the geometric polymer;
```
GeometricPolymer:
    num_molecules:    1
    num_chains:       1
    num_residues:     88
    num_atoms:        2808
    num_edges:        44928
    node_features:
        repr dim:     1
        multiplicity: 8
    edge_feature dim: 1
```

Create a degree-0 representation of multiplicity 8 for our input features, and a degree-1 representation with multiplicity 4 as an example output. Also, create a hidden representation with degrees 0 and 1, and multiplicity 16.
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
hidden_layers = 2

transformer = EquivariantTransformer(
   in_repr,
   out_repr,
   hidden_repr,
   hidden_layers,
   edge_dim,
   edge_hidden_dim,
)
```

Pass the polymer to the equivariant transformer to get a new set of node features.
```
out_geom_polymer = transformer.polymer(geom_polymer)
```

Print the output geometric polymer;
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

Note how the node features have changed in degree and multiplicity, but the structure of the molecule has remained the same, as have the edge features.
