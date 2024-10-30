
from __future__ import annotations
from .geom import (
    Repr,
    ProductRepr,
    RepNorm,
    FEATURE_DIM,
    REPR_DIM,
    EquivariantBases,
)
from .polymer import GeometricPolymer
from .kernel import CUDA_AVAILABLE
if CUDA_AVAILABLE:
    from .kernel._mm import EquivariantMatmulKernel
import torch
import torch.nn as nn
import dgl
from copy import copy, deepcopy
import itertools
import os


KERNEL = os.environ.get("BIOEQ_KERNEL", "0") == "1"
if KERNEL and not CUDA_AVAILABLE:
    raise RuntimeError('The kernel was not compiled.')


class EquivariantLinear(nn.Module):
    """
    Applies a linear layer to each degree of a spherical tensor.
    """

    def __init__(
            self: EquivariantLinear,
            repr: Repr,
            out_repr: Repr,
            dropout: float = 0.0,
    ) -> None:

        super().__init__()
        # Ensure that the representations have the same degrees
        if not repr.lvals == out_repr.lvals:
            raise ValueError(
                "An EquivariantLinear layer cannot modify the degrees of a \
                representation."
            )
        # Store the representation
        self.repr = repr
        # Initialize the weight matrix
        self.weight = nn.Parameter(
            torch.randn(
                repr.nreps() * out_repr.mult,
                repr.mult,
            )
        )
        # Get the indices which will retrieve the correct
        # degrees for the output
        indices = torch.Tensor(
            repr.indices(),
        ).long()
        self.register_buffer('indices', indices)
        # Store the dimensions used in the indexing
        self.expanddims = (
            1,
            repr.mult,
            repr.dim(),
        )
        self.outdims = (
            repr.nreps(),
            repr.mult,
            repr.dim(),
        )
        # Create a dropout object
        self.dropout = nn.Dropout(dropout)

    def forward(
            self: EquivariantLinear,
            f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take a spherical tensor and apply a linear layer to each irrep
        separately.
        """

        # TODO: check if the einsum is faster on GPU or not,
        # because it is on CPU

        GATHER_DIM = -3
        # Get the batch size
        *b, _, _ = f.shape
        # Apply the linear layers to each degree
        out = (self.weight @ f).view(*b, *self.outdims)
        # out = torch.einsum('lij,...jk->...lik', self.weight, f)
        # Gather the components corresponding to the same degree
        ix = self.indices.expand(*b, *self.expanddims)
        return out.gather(
            dim=GATHER_DIM, index=ix,
        ).squeeze(GATHER_DIM)


class EquivariantGating(nn.Module):
    """
    Compute norm-wise gating for a spherical tensor.
    """

    def __init__(
        self: EquivariantGating,
        repr: Repr,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()
        # Create an object for computing the irrep norms
        self.repr = repr
        self.norm = RepNorm(repr)
        # Create a linear layer for processing the norms
        self.linear = nn.Linear(
            repr.nreps() * repr.mult,
            repr.nreps() * repr.mult,
        )
        # Create the activation
        self.activation = nn.Sigmoid()
        # Create a dropout object
        self.dropout = nn.Dropout(dropout)
        # Get the indices to which each norm corresponds
        self.ix = torch.Tensor(
            repr.indices(),
        ).long()
        # Get the shape of the output dimensions
        self.outdims = (
            repr.mult,
            repr.nreps(),
        )

    def forward(
        self: EquivariantGating,
        st: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update the irrep norms based on the linear layer.
        """

        # Get the norms of the input tensor
        norms = self.norm(st)
        # Reshape for multiplication
        *b, _, _ = norms.size()
        norms = norms.flatten(-2, -1)
        # Pass the norms through the linear layer and reshape back
        norms = self.linear(norms).view(*b, *self.outdims)
        # Apply the activation
        norms = self.activation(norms)
        # Apply dropout
        norms = self.dropout(norms)
        # Expand and multiply by the norms
        return st * norms[..., self.ix]


class RadialWeight(nn.Module):
    """
    Computes weights from invariant edge features via a two-layer neural
    network.
    """

    def __init__(
        self: RadialWeight,
        repr: ProductRepr,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()
        # Store the representation
        self.repr = repr
        # Get the output dimension shape
        if KERNEL:
            self.outdims = (
                repr.nreps(),
                repr.rep1.mult,
                repr.rep2.mult,
            )
        else:
            self.outdims = (
                repr.rep2.mult,
                repr.rep1.mult * repr.nreps(),
            )
        # Create the layers
        tmp_out_dim = repr.nreps() * repr.rep1.mult * repr.rep2.mult
        self.layer1 = nn.Linear(
            edge_dim,
            hidden_dim,
        )
        self.layer2 = nn.Linear(
            hidden_dim,
            tmp_out_dim,
        )
        # Create the activation
        self.activation = nn.ReLU()
        # Create a dropout object
        self.dropout = nn.Dropout(dropout)

    def forward(
        self: RadialWeight,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the network.
        """

        # Get the batch size of the input
        *b, _ = x.size()
        # Apply the first layer and activation
        x = self.layer1(x)
        x = self.activation(x)
        # Apply dropout
        x = self.dropout(x)
        # Apply the second layer and reshape for later steps
        return self.layer2(x).view(*b, *self.outdims)


class EquivariantConvolution(nn.Module):
    """
    An SE(3)-equivariant convolution. This is missing the final
    reduction step, so it can be used for a true convolution or
    can also be passed to a graph attention layer.
    """

    def __init__(
        self: EquivariantConvolution,
        repr: ProductRepr,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()
        self.rwlin = RadialWeight(
            repr,
            edge_dim,
            hidden_dim,
            dropout
        )
        self.outdim = repr.rep2.dim()
        if KERNEL:
            self._kernel_mm = EquivariantMatmulKernel(repr)
        else:
            self._kernel_mm = None

    def _mm(
        self: EquivariantConvolution,
        g: dgl.DGLGraph,
        basis: torch.Tensor,
        rw: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform an SE(3)-equivariant convolution between the given basis
        elements, radial weights, and features.
        """

        # Get the batch size and the dimension of the output representations
        b, *_ = basis.size()
        # Get the sources of the edges from the graph
        U, _ = g.edges()
        # Compute the convolution in two steps. This implementation is
        # based on the one from the NVIDIA SE(3)-transformer.
        tmp = (f[U] @ basis).view(b, -1, self.outdim)
        return rw @ tmp

    def forward(
        self: EquivariantConvolution,
        g: dgl.DGLGraph,
        basis: torch.Tensor,
        edge_feats: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the radial weights associated with the given edge features.
        Then, perform an SE(3)-equivariant convolution between the basis
        elements, radial weights, and features.
        """

        # Compute the radial weights from the edge features
        rw = self.rwlin(edge_feats)
        # Pass the parameters to the matrix multiplication kernel
        if KERNEL:
            return self._kernel_mm(g, basis, rw, f)
        else:
            return self._mm(g, basis, rw, f)


class GraphAttention(nn.Module):
    """
    Perform graph attention operation on the inputs.
    """

    def __init__(
        self: GraphAttention,
        hidden_size: int,
        nheads: int = 1,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()
        # Verify the number of heads
        if hidden_size % nheads != 0:
            raise ValueError(
                "The hidden size must be divisible by the number of heads."
            )
        self.hidden_size = hidden_size
        # The size we reshape to for multi-head attention
        self.tmpsize = (
            nheads, hidden_size // nheads
        )
        # Attention dropout
        self.dropout = nn.Dropout1d(dropout)
        # The temperature
        self.temp = self.hidden_size ** -0.5
        # Leaky ReLU for the scores
        self.lrelu = nn.LeakyReLU(0.2)

    def _reshape(
        self: GraphAttention,
        graph: dgl.DGLGraph,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape the keys, queries, and values for multi-head attention.
        """

        # Reshape appropriately
        keys = keys.view(
            graph.num_edges(), *self.tmpsize,
        )
        queries = queries.view(
            graph.num_edges(), *self.tmpsize,
        )
        values = values.view(
            graph.num_edges(), *self.tmpsize,
        )
        return keys, queries, values

    def forward(
        self: GraphAttention,
        graph: dgl.DGLGraph,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Take a set of keys, queries and values of shape (e, f), (e, f) and
        (n, k) respectively, and a graph with e edges and n nodes, and return
        a tensor of shape (n, k) representing the attention-weighted values.
        """

        # Reshape based on the number of heads
        keys, queries, values = self._reshape(
            graph, keys, queries, values,
        )
        # Compute the scaled dot product of the keys and queries
        scores = (keys * queries).sum(-1) * self.temp
        # Apply the leaky ReLU
        scores = self.lrelu(scores)
        # Compute the attention weights
        weights = dgl.ops.edge_softmax(graph, scores)
        weights = self.dropout(weights)
        # Mask the values if a mask is provided
        if mask is not None:
            weights = weights * mask[:, None]
        # Compute the attention-weighted values
        values = weights[..., None] * values
        # Reduce the values to get the output
        output = dgl.ops.copy_e_sum(graph, values)
        # Return after fusing the heads again
        return output.view(
            graph.num_nodes(),
            self.hidden_size,
        )


class EquivariantAttention(nn.Module):
    """
    Perform an equivariant convolution and then pipe the outputs into a
    graph attention layer.
    """

    def __init__(
        self: EquivariantAttention,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:

        super().__init__()
        # Create a new representation object with a larger
        # output multiplicity (for keys, queries, and values)
        repr_h = deepcopy(repr)
        repr_h.rep2.mult = 3 * repr.rep2.mult
        self.conv = EquivariantConvolution(
            repr_h,
            edge_dim,
            edge_hidden_dim,
            dropout,
        )
        # Create a graph attention object
        self.attn = GraphAttention(
            repr.rep2.mult * repr.rep2.dim(),
            nheads,
            attn_dropout,
        )
        # Store the output dimensions
        self.outdims = (
            repr.rep2.mult,
            repr.rep2.dim(),
        )

    def forward(
        self: EquivariantAttention,
        graph: dgl.DGLGraph,
        basis: torch.Tensor,
        edge_feats: torch.Tensor,
        f: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass.
        """

        # Equivariant convolution to get keys, values, and queries
        conv = self.conv(graph, basis, edge_feats, f)
        k, q, v = torch.chunk(conv, 3, FEATURE_DIM)
        # Graph attention
        attn_out = self.attn(graph, k, q, v, mask)
        # Undo any reshapes performed by the attention layer
        return attn_out.view(graph.num_nodes(), *self.outdims)


class EquivariantLayerNorm(nn.Module):
    """
    An equivariant layernorm.
    """

    EPSILON = 1E-8

    def __init__(
        self: EquivariantLayerNorm,
        repr: Repr,
        epsilon: float = EPSILON,
    ) -> None:

        super().__init__()
        # For computing the norm of each irrep
        self.norm = RepNorm(repr)
        # The actual layernorm
        self.lnorm = nn.LayerNorm(repr.mult)
        # The nonlinearity applied to the norms
        self.nonlinearity = nn.ReLU()
        # To prevent division by zero
        self.epsilon = epsilon
        # Get the indices to which each norm corresponds
        self.ix = torch.Tensor(
            repr.indices(),
        ).long()

    def forward(
        self: EquivariantLayerNorm,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take a spherical tensor and apply a layernorm to the norm
        of each irrep.
        """

        # Compute the norms of the features
        norms = self.norm(f)
        *b, h, d = norms.size()
        # Apply the layernorm to each degree
        lnorms = self.lnorm(
            norms.view(-1, d, h)
        ).view(*b, h, d)
        # Apply the nonlinearity. this also ensures that the norms are
        # nonnegative, which is necessary to preserve chirality
        lnorms = self.nonlinearity(lnorms)
        # Expand the norms to the original shape and renormalize the features
        norms_r = lnorms / (norms + self.epsilon)
        return f * norms_r[..., self.ix]


class EquivariantTransformerBlock(nn.Module):
    """
    A single block of an equivariant transformer.
    """

    def __init__(
        self: EquivariantTransformerBlock,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:

        super().__init__()
        # The attention block
        self.attn = EquivariantAttention(
            repr,
            edge_dim,
            edge_hidden_dim,
            nheads,
            dropout,
            attn_dropout,
        )
        # The layernorm
        self.ln = EquivariantLayerNorm(repr.rep1)
        # The linear projection for the output
        self.proj = EquivariantLinear(
            repr.rep2,
            repr.rep2,
            dropout,
        )

    def forward(
        self: EquivariantTransformerBlock,
        graph: dgl.DGLGraph,
        basis: torch.Tensor,
        features: torch.Tensor,
        edge_feats: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Store for skip connection
        # features_tmp = features
        # Apply the first equivariant layernorm (we use the pre-ln transformer
        # variant)
        features = self.ln(features)
        # Apply the equivariant attention
        features = self.attn(
            graph,
            basis,
            edge_feats,
            features,
            mask,
        )
        # Apply the linear projection
        return self.proj(features)


class EquivariantTransformer(nn.Module):
    """
    An equivariant transformer.
    """

    def __init__(
        self: EquivariantTransformer,
        in_repr: Repr,
        out_repr: Repr,
        hidden_repr: Repr,
        hidden_layers: int,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:

        super().__init__()
        # Store all attributes
        self.edge_dim = edge_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.nheads = nheads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        # Store the representations
        self.in_repr = in_repr
        self.out_repr = out_repr
        self.hidden_repr = hidden_repr

        # Get the sequence of reprs the model passes through
        reprs = [in_repr] + [hidden_repr] * hidden_layers + [out_repr]
        # Create the layers to move between these representations. Store
        # the product reprs involved
        layers = []
        preprs = []
        for repr1, repr2 in itertools.pairwise(reprs):
            prepr = ProductRepr(
                deepcopy(repr1),
                deepcopy(repr2),
            )
            preprs.append(prepr)
            layers.append(
                self._construct_layer(prepr)
            )
        self.layers = nn.ModuleList(layers)

        # Create an equivariant map into the space of appropriately-sized
        # matrices.
        self.bases = EquivariantBases(*preprs)

    def _construct_layer(
        self: EquivariantTransformer,
        prep: ProductRepr,
    ) -> EquivariantTransformerBlock:
        """
        Construct a single layer based on the given product representation.
        """

        return EquivariantTransformerBlock(
            prep,
            self.edge_dim,
            self.edge_hidden_dim,
            self.nheads,
            self.dropout,
            self.attn_dropout,
        )

    def polymer(
        self: EquivariantTransformer,
        polymer: GeometricPolymer,
        mask: torch.Tensor | None = None,
    ) -> GeometricPolymer:
        """
        Apply the forward method to a geometric polymer.
        """

        # Center the coordinates as is necessary for translational
        # invariance
        polymer.center()
        # Copy the polymer so as not to overwrite the input
        polymer = copy(polymer)
        # Update the node features
        polymer.node_features = self(
            polymer.graph,
            polymer.coordinates,
            polymer.node_features,
            polymer.edge_features,
            mask,
        )
        return polymer

    def forward(
        self: EquivariantTransformer,
        graph: dgl.DGLGraph,
        coordinates: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply the equivariant transformer to the input geometric graph.
        """

        if not self.in_repr.verify(node_features):
            raise ValueError(
                "The given node features have shape"
                f" {node_features.size()}, which does not"
                " match the input representation."
            )

        # Get the basis elements
        bases = self.bases.edgewise(
            coordinates,
            graph,
        )
        # Pass through the layers
        for layer, basis in zip(self.layers, bases):
            node_features = layer(
                graph,
                basis,
                node_features,
                edge_features,
                mask,
            )
        return node_features


class CoordinateUpdate(nn.Module):
    """
    A helper class to construct a set of coordinates from a set of node
    features using an equivariant transformer block.
    """

    def __init__(
        self: CoordinateUpdate,
        repr: Repr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:

        # Get a single degree-one representation to output coordinates
        out_repr = Repr([1])
        # Construct a product repr
        prep = ProductRepr(repr, out_repr)
        # Construct the equivariant transformer
        self.tf = EquivariantTransformerBlock(
            prep,
            edge_dim,
            edge_hidden_dim,
            nheads,
            dropout,
            attn_dropout,
        )

    def forward(
        self: CoordinateUpdate,
        polymer: GeometricPolymer,
    ) -> torch.Tensor:
        """
        Get a set of coordinates from a GeometricPolymer.
        """

        # Copy the polymer
        polymer = copy(polymer)
        # Update the node features
        coordinates = self.tf(
            polymer.graph,
            polymer.coordinates,
            polymer.node_features,
            polymer.edge_features,
        )
        # Squeeze the feature dimension
        return coordinates.squeeze(FEATURE_DIM)
