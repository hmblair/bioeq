from __future__ import annotations
import torch


def sinusoidal_embedding(
    batch_shape: torch.Size | tuple[int, ...],
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
