from __future__ import annotations
import itertools
import torch
import torch.nn as nn


#
# Feedforward layers
#


class DenseNetwork(nn.Module):
    """
    A series of feedforward layers.
    """

    def __init__(
        self: DenseNetwork,
        in_size: int,
        out_size: int,
        hidden_sizes: list[int] = [],
        dropout: float = 0.0,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
    ) -> None:

        super().__init__()
        # Define the sizes of the network
        features = [in_size] + hidden_sizes + [out_size]
        # Construct the layers
        layers = []
        for l1, l2 in itertools.pairwise(features):
            layers.append(
                nn.Linear(l1, l2, bias)
            )
        # Store the layers and other relevant attributes
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(
        self: DenseNetwork,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        The forward pass of the model, taking an input of shape
        (*b, in_size) and returning an output of shape (*b, out_size).
        """

        # Apply each layer, save for the last, and corresponding
        # dropoput and activation
        for layer in self.layers[:-1]:
            x = self.dropout(
                self.activation(layer(x))
            )
        # Apply the final layer, with no activation or dropout
        return self.layers[-1](x)


#
# Positional encodings
#


def sinusoidal_embedding(
    batch_shape: torch.Size,
    seq_len: int,
    embedding_dim: int,
    device: str | torch.device,
    k: int = 10000,
) -> torch.Tensor:
    """
    Return the sinusoidal embedding of a sequence of length seq_len and
    dimension embedding_dim.
    """

    # create the positions
    ix = torch.arange(
        end=seq_len,
        dtype=torch.float32,
        device=device,
    ).expand(*batch_shape, seq_len)

    # create the dimensions
    dimensions = torch.arange(
        end=embedding_dim,
        dtype=torch.float32,
        device=device,
    ).expand(*ix.size(), embedding_dim)

    # calculate the angles
    angles = ix[..., None] / (k ** (2 * (dimensions // 2) / embedding_dim))

    # calculate the sinusoidal embedding
    sinusoidal_embedding = torch.zeros(
        *dimensions.size(),
        device=device,
    )
    sinusoidal_embedding[..., 0::2] = torch.sin(angles[..., 0::2])
    sinusoidal_embedding[..., 1::2] = torch.cos(angles[..., 1::2])

    return sinusoidal_embedding


class FixedSinusoidalEmbedding(nn.Module):
    """
    Implements a sinusoidal positional embedding with a fixed maximum input
    index. The forward method returns the value of the positional encoding
    at the specified indices.
    """

    def __init__(
        self: FixedSinusoidalEmbedding,
        max_index: int,
        embedding_dim: int,
    ) -> None:

        super().__init__()
        # Store the maximum index and the embedding dimension
        self.max_index = max_index
        self.embedding_dim = embedding_dim
        # Create the positional encoding
        encoding = sinusoidal_embedding(
            batch_shape=(),
            seq_len=max_index,
            embedding_dim=embedding_dim,
            device='cpu',
        )
        # Register the positional encoding with the module
        self.encoding: torch.Tensor
        self.register_buffer('encoding', encoding)

    def forward(
        self: FixedSinusoidalEmbedding,
        ix: torch.Tensor,
        shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """
        Return the positional encoding at the specified index.
        """

        return self.encoding[ix].expand(
            *ix.size(),
            *shape,
            self.embedding_dim,
        )
