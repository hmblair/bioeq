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
The `GeometricPolymer` class is exposed in the `bioeq.polymer` module. Its three main attributes are its `.node_features`, `.edge_features`, and `.graph`. The method `.from_pdb` loads the coordinates and elements from a PDB file and creates a `GeometricPolymer` with the element indices as node featuers and relative positions as edge features. Alternatively, the `PolymerDataset` will lazily load from a folder of PDB files.

The `bioeq.modules` module contains the `EquivariantTransformer` class. This has a `.polymer` method which takes as input a `GeometricPolymer` and updates its ndoe features using the transformer layers. Each layer, and hence the entire module, is equivariant to rotations.

To specify an input, you should use a `Repr` object from `bioeq.geom`. This represents a representation of SO(3), and thus takes as input a set of degrees. Recall that
 - degree-0 features are the 1D invariant features, and
 - degree-1 features are the 3D equivariant features.

The `Repr` class also takes in a multiplicity, which specifies the number of copies of each degree. The same goes for the output and hidden layers, each of which have their own associated representation.

When passing the tensors to the models, it is always expected that they are of shape `(..., n, d)`, where `n` is the multiplicity of the representation and `d` is the dimension. For example, `Repr([1], 32)` creates 32 degree-1 representations, which corresponds to an input/output tensor of shape `(..., 32, 3)`.
