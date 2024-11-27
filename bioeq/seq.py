from __future__ import annotations
import torch
import torch.nn as nn


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
