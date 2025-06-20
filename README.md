# Overview
`bioeq` is a set of tools for doing geometric deep learning with a focus on applications to structural biology. The core datatypes it exposes are the `Polymer` and `GeometricPolymer`, which contain information on the position and connectivity of atoms in one or more molecules, and a suite of tools for computing data at the atom, residue, chain, and molecule level. The latter also has node and edge features, suitable for processing by a geometric deep learning model.

A set of equivariant deep learning layers, including an equivariant transformer, allow for transformations of these polymers to be learned.

# Installation
To install, clone the repo and install with `pip`.
```
git clone https://github.com/hmblair/bioeq
cd bioeq
pip3 install .
```

# Usage

A `Polymer` is a class which contains the coordinates and edges of a molecule or a set of molecules, as well as information on which chain and residue each molecule belongs to. The `PolymerDataset` will load `Polymer`s from a valid `.nc` file that was created with `create_structure_dataset`. It will also load any other features specified at instantiation.
```
from bioeq.polymer import PolymerDataset

# Create a PolymerDataset class, and request that it load the 'elements'
# and 'residues' features too
file = 'examples/polymers.nc'
dataset = PolymerDataset(
    file=file,
    atom_features=['elements', 'residues']
)

# Load the first five polymers and associated data. Notice how the elements
# are loaded in the same order that they were passed to the constructor.
polymer, elements, residues = dataset[0:5]
```

Print the polymer:
```
Polymer:
    num_molecules:    5
    num_chains:       7
    num_residues:     153
    num_atoms:        3258
    num_edges:        10532
```

The easiest way to train a model with this information as input is to create a `GeometricPolymer` object. This is a `Polymer` object in combination with 

* node features, of shape (num_atoms, num_features, degrees)
* edge features, of shape (num_edge, num_features).

To create it, we need to compute some initial node and edge features.

The node features we can create by passing each of the elements and residues through an embedding layer.
```
import torch
import torch.nn as nn
from bioeq._index import Element, Residue

# Create a separate embedding for the elements and the residues, both of size
# four
elem_embedding = nn.Embedding(len(Element), 4)
residue_embedding = nn.Embedding(len(Residue), 4)

# Pass through the embeddings and concatenate to get the node features
elements = elem_embedding(elements.long())
residues = residue_embedding(residues.long())
node_features = torch.cat([elements, residues], dim=-1)
```

For the edge features, we can compute the distance between all atoms which are connected by an edge.
```
edge_features = polymer.pdist()
```

Unsqueeze the final dimension, since the model will expect three dimensions for the node features and two for the edge features
```
node_features = node_features[..., None]
edge_features = edge_features[..., None]
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

Print the geometric polymer:
```
GeometricPolymer:
    num_molecules:    5
    num_chains:       7
    num_residues:     153
    num_atoms:        3258
    num_edges:        10532
    node_features:
        repr dim:     1
        multiplicity: 8
    edge_feature dim: 1
```

Create a degree-0 representation of multiplicity 8 for our input features, and a degree-1 representation with multiplicity 4 as an example output. Also, create a hidden representation with degrees 0 and 1, and multiplicity 16.
```
from bioeq.geom import Repr

in_repr = Repr([0], 8)
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
    num_molecules:    5
    num_chains:       7
    num_residues:     153
    num_atoms:        3258
    num_edges:        10532
    node_features:
        repr dim:     3
        multiplicity: 4
    edge_feature dim: 1
```

Note how the node features have changed in degree and multiplicity, but the structure of the molecule has remained the same, as have the edge features.

# Working with Polymers

Atoms in a polymer have five different properties associated with them:
* their element,
* their name (e.g. C5'),
* which residue they belong to,
* which chain they belong to, and
* which molecule they belong to.

The `select` method allows us to select atoms based on their properties. The different properties are stored in the `Property` enum. Moreover, the elements and residues are stored in the `Element` and `Residue` enums respectively. For example,
```
from bioeq._index import(
    Property,
    Element,
    Residue,
    Adenosine,
    Guanosine,
    Cytidine,
    Uridine,
    Reduction,
)

# Select the first molecule
ix = torch.tensor([0])
polymer.select(ix, Property.MOLECULE)

# Select residues 5 through 12
ix = torch.arange(5, 12)
polymer.select(ix, Property.RESIDUE)

# Select all hydrogens
ix = torch.tensor([Element.H.value])
polymer.select(ix, Property.ELEMENT)

# Select the glycosidic atoms
ix = torch.tensor([
    Adenosine.N9.value,
    Guanosine.N9.value,
    Cytidine.N1.value,
    Uridine.N1.value,
])
polymer.select(ix, Property.NAME)
```

The `reduce` method works analogously, but instead of selecting the atoms, it reduces the input features across all atoms sharing the same property. The choice of reduction is stored in the `Reduction` enum.

```
# Compute the mean position of each residue
# Note that reduce uses Reduction.MEAN by default
polymer.reduce(
    polymer.coordinates,
    Property.RESIDUE,
)

# Compute the maximum partial charge among all oxygen atoms
# Note that MAX and MIN reductions return the argmax or argmin as a second argument
max_pc, _ polymer.reduce(
    polymer.partial_charge(),
    Property.ELEMENT,
    Reduction.MAX,
)
oxygen_max_pc = pc[Element.O.value]
```

Some other features that may be useful to you are 

```
# Get the orientation-corrected principal components of all residues in the polymer
_, Q = polymer.align(Property.RESIDUE)

# Center the polymer
centered = polymer.center()

# Save the coordinates, elememts, and residues to a .cif file
polymer.cif('example.cif')

# Get the average bond length of all unique bonds (e.g. C5'-N1) in the molecule
# by reducing over bond types
lengths = polymer.breduce(
    polymer.pdist()
)
```
