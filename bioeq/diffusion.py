from __future__ import annotations
import torch
from torch import nn
from bioeq.seq import (
    FixedSinusoidalEmbedding,
    DenseNetwork,
)
from bioeq.modules import EquivariantTransformerBlock


class NoiseSchedule(nn.Module):
    """
    A noise schedule for a generative diffusion process.
    """

    def __init__(
        self: NoiseSchedule,
        betas: torch.Tensor,
    ) -> None:
        super().__init__()

        # Ensure that the betas are in the correct range
        if (betas < 0).any() or (betas >= 1).any():
            raise ValueError('The betas must be in the range [0, 1).')
        # Type hints
        self._beta: torch.Tensor
        self._betatilde: torch.Tensor
        self._alpha: torch.Tensor
        self._alphabar: torch.Tensor
        # Compute the alphas and the product of the alphas
        _alpha = 1 - betas
        _alphabar = torch.cumprod(_alpha, 0)
        _betatilde = (1 - _alphabar[:-1]) / (1 - _alphabar[1:]) * betas[1:]
        _betatilde = torch.cat([torch.zeros_like(betas[:1]), _betatilde])
        # Store the values as buffers so they move to the correct device
        self.register_buffer('_beta', betas)
        self.register_buffer('_alpha', _alpha)
        self.register_buffer('_alphabar', _alphabar)
        self.register_buffer('_betatilde', _betatilde)

    def alpha(
        self: NoiseSchedule,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the value of alpha at the specified timesteps.
        """

        return self._alpha[t]

    def alphabar(
        self: NoiseSchedule,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the value of the product of the alphas at the specified
        timesteps.
        """

        return self._alphabar[t]

    def beta(
        self: NoiseSchedule,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the value of beta at the specified timesteps.
        """

        return self._beta[t]

    def betatilde(
        self: NoiseSchedule,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the value of beta-tilde at the specified timesteps.
        """

        return self._betatilde[t]

    def sigma(
        self: NoiseSchedule,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the value of sigma at the specified timesteps.
        """

        return torch.sqrt(self.beta(t))

    def __len__(
        self: NoiseSchedule,
    ) -> int:
        """
        Returns the number of timesteps in the noise schedule.
        """

        return len(self._beta)

    def timesteps(
            self: NoiseSchedule,
    ) -> torch.Tensor:
        """
        Returns the timesteps in the noise schedule, in reverse order for
        sampling.
        """

        return torch.arange(
            len(self)
        ).flip(0)

    def random_timestep(
        self: NoiseSchedule,
        size: tuple | torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """
        Returns random timesteps from the noise schedule.
        """

        return torch.randint(0, len(self), size)


class LinearNoiseSchedule(NoiseSchedule):
    """
    A linear noise schedule for the diffusion process, which interpolates
    between the two values of beta specified in the range.
    """

    LINEAR_NOISE_RANGE = (1E-4, 2E-2)

    def __init__(
        self: LinearNoiseSchedule,
        num_timesteps: int,
        t_range: tuple[float, float] = LINEAR_NOISE_RANGE,
    ) -> None:

        # Compute the betas
        betas = torch.linspace(
            t_range[0],
            t_range[1],
            num_timesteps,
        )
        # Initialise the noise schedule
        super().__init__(betas)


class CosineNoiseSchedule(NoiseSchedule):
    """
    A cosine noise schedule for the diffusion process, which interpolates
    between the two values of beta specified in the range.
    """

    EPSILON = 0.008
    BETA_CLIP = 0.999

    def __init__(
        self: CosineNoiseSchedule,
        num_timesteps: int,
        epsilon: float = EPSILON,
        beta_clip: float = BETA_CLIP,
    ) -> None:

        # Store the range and number of timesteps
        self.num_timesteps = num_timesteps
        # Precompute the alphabars
        timesteps = torch.arange(0, num_timesteps) / num_timesteps
        _alphabar = torch.cos(
            (timesteps + epsilon)/(timesteps[-1] + epsilon) * torch.pi / 2
        ) ** 2
        _alphabar = _alphabar / _alphabar[0]

        _alpha = _alphabar[1:] / _alphabar[:-1]
        _alpha = torch.cat([_alphabar[:1], _alpha])

        betas = torch.clamp(1 - _alpha, 0, beta_clip)

        # Initialise the noise schedule
        super().__init__(betas)


class DiffusionProcess(nn.Module):
    """
    Implement forward and reverse diffusion processes.
    """

    def __init__(
        self: DiffusionProcess,
        schedule: NoiseSchedule,
    ) -> None:

        super().__init__()
        # Store the schedule
        self.schedule = schedule

    def __len__(
        self: DiffusionProcess,
    ) -> int:
        """
        Returns the number of timesteps in the diffusion process.
        """

        return len(self.schedule)

    def timesteps(
        self: DiffusionProcess,
    ) -> torch.Tensor:
        """
        Returns the timesteps in the diffusion process, in reverse order
        for sampling.
        """

        return self.schedule.timesteps()

    def forward_diffusion(
        self: DiffusionProcess,
        x: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies t steps of the forward diffusion process to the input variable,
        and returns the noise that was added to the variable, along
        with the variable with the added noise.
        """

        # Sample standard normal noise
        z = torch.randn_like(x)
        # Get the alpha value for the current timestep
        alphabar = self.schedule.alphabar(timestep)
        # Apply the forward diffusion process
        mu = torch.einsum(
            'i, i... -> i...',
            torch.sqrt(alphabar), x,
        )
        sigma = torch.einsum(
            'i, i... -> i...',
            torch.sqrt(1 - alphabar), z,
        )
        diffuse_x = mu + sigma
        return z, diffuse_x

    def reverse_diffusion(
        self: DiffusionProcess,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies a single step of the reverse diffusion process to the
        input variable, and returns the denoised variable. The input
        x_hat is the predicted noise from a model.
        """

        # Sample standard normal noise at all steps except the final step
        # (which is the first step of the forward diffusion process)
        z = 0 if timestep == 0 else torch.randn_like(x)
        # Get the values of alpha and beta for the current timestep
        alpha = self.schedule.alpha(timestep)
        alphabar = self.schedule.alphabar(timestep)
        beta = self.schedule.beta(timestep)
        sigma = self.schedule.sigma(timestep)
        # Apply the reverse diffusion process
        undiffuse_x = 1 / torch.sqrt(alpha) * (
            x - beta / torch.sqrt(1 - alphabar) * x_hat
        ) + sigma * z
        return undiffuse_x


class TimestepEmbedding(nn.Module):
    """
    A sinusoidal positional embedding with a fixed maximum index, which is
    then passed through a feedforward layer.
    """

    def __init__(
        self: TimestepEmbedding,
        max_index: int,
        embedding_dim: int,
    ) -> None:

        super().__init__()
        # The embedding is based on a fixed sinusoidal embedding,
        self.embedding = FixedSinusoidalEmbedding(
            max_index=max_index,
            embedding_dim=embedding_dim,
        )
        # which is passed through a dense network.
        self.dense = DenseNetwork(
            in_size=embedding_dim,
            hidden_sizes=[4 * embedding_dim],
            out_size=embedding_dim,
        )

    def forward(
        self: TimestepEmbedding,
        ix: int | torch.Tensor,
        shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """
        Return the positional encoding at the specified index, passed
        through a feedforward layer.
        """

        return self.dense(
            self.embedding(ix, shape)
        )
