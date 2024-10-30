import torch
import torch.nn as nn
from bioeq.polymer import PolymerDataset, GeometricPolymer
from bioeq.geom import Repr
from bioeq.modules import EquivariantTransformer

# Create a PolymerDataset class, and request that it load the 'elements'
# and 'residues' features too
file = 'polymers.nc'
dataset = PolymerDataset(
    file=file,
    atom_features=['elements', 'residues']
)

# Load the first five polymers and associated data
polymer, elements, residues = dataset[0:5]

# Print the polymer
print(polymer)


# Create a separate embedding for the elements and the residues, both of size
# four
elem_embedding = nn.Embedding(5, 4)
residue_embedding = nn.Embedding(4, 4)

# Pass through the embeddings and concatenate to get the node features
elements = elem_embedding(elements.long())
residues = elem_embedding(residues.long())
node_features = torch.cat([elements, residues], dim=-1)

# Compute the distances between all atoms connected by an edge
edge_features = polymer.pdist()

# Unsqueeze the final dimension, since the model will expect three dimensions
# for the node features and two for the edge features
node_features = node_features[..., None]
edge_features = edge_features[..., None]

# Create out geometric polymer
geom_polymer = GeometricPolymer(
    polymer,
    node_features,
    edge_features,
)

# Print the geometric polymer
print(geom_polymer)

# Create our input, output, and hidden representaiton
in_repr = Repr([0], 8)
out_repr = Repr([1], 4)
hidden_repr = Repr([0, 1], 16)

# Other relevant hyperparameters
edge_dim = 1
edge_hidden_dim = 16
hidden_layers = 2

# Instantiase the model
transformer = EquivariantTransformer(
    in_repr,
    out_repr,
    hidden_repr,
    hidden_layers,
    edge_dim,
    edge_hidden_dim,
)

# Get the model output and print it
out_geom_polymer = transformer.polymer(geom_polymer)
print(out_geom_polymer)
