from __future__ import annotations
from typing import Iterator
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import wandb


class LinearWarmupSqrtDecay(_LRScheduler):
    """
    A learning rate scheduler with a linear warmup stage and then a square-root
    decay stage.
    """

    def __init__(
        self: LinearWarmupSqrtDecay,
        optimizer: Optimizer,
        warmup_steps: int,
        *args, **kwargs,
    ) -> None:

        self.warmup_steps = warmup_steps
        super().__init__(optimizer, *args, **kwargs)

    def get_lr(self) -> list[float]:
        """
        Get the learning rate for the current epoch (or step).
        """
        if self.last_epoch == -1:
            return [
                group['lr']
                for group in self.optimizer.param_groups
            ]
        elif self.last_epoch < self.warmup_steps:
            return self._warmup_step()
        else:
            return self._decay_step()

    def _warmup_step(self) -> list[float]:
        """
        A linear warmup step for the learning rate.
        """
        scale = (self.last_epoch + 1) / self.warmup_steps
        return [
            group['initial_lr'] * scale
            for group in self.optimizer.param_groups
        ]

    def _decay_step(self) -> list[float]:
        """
        An inverse square root decay step for the learning rate.
        """
        scale = (self.last_epoch + 2) / (self.last_epoch + 1)
        return [
            group['lr'] * (scale ** (-1/2))
            for group in self.optimizer.param_groups
        ]


class ProgressBar:
    """
    Dispaly the progress of the current epoch. Track the current and average
    loss.
    """

    def __init__(
        self: ProgressBar,
        data: DataLoader,
        name: str,
    ) -> None:

        # The name of this dataset
        self.name = name
        # The data we iterate over
        self.data = data
        # Keep track of the current epoch and step
        self.curr_epoch = -1
        self.curr_step = -1
        # The current loss and a moving average
        self.loss = 0
        self.av_loss = 0
        # Initialise the progress bar
        self.pbar = None

    def pbar_str(
        self: ProgressBar,
    ) -> str:
        """
        The progress bar description.
        """

        return f'Epoch {self.curr_epoch}; loss {self.loss:.2E}; average_loss {self.av_loss:.2E}'

    def update(
        self: ProgressBar,
        loss: torch.Tensor | None = None,
    ) -> None:
        """
        Increment the step. Update the loss and the average loss if a
        loss is provided.
        """

        if self.pbar is None:
            raise RuntimeError(
                "The epoch method must be called before"
                " updating the progress bar."
            )

        self.curr_step += 1
        if loss is not None:
            self.loss = loss
            self.av_loss = (
                self.curr_step * self.av_loss + self.loss
            ) / (self.curr_step + 1)
            self.pbar.set_description(self.pbar_str())
            wandb.log(
                {f'{self.name} loss': self.loss}
            )

    def epoch(
        self: ProgressBar,
    ) -> None:
        """
        Increment the epoch counter and reset the step counter and average loss.
        """

        # Create a new progress bar
        self.pbar = tqdm(
            self.data,
            desc=self.pbar_str()
        )

        self.curr_epoch += 1
        self.curr_step = -1
        self.av_loss = 0

    def __iter__(
        self: ProgressBar,
    ) -> Iterator:
        """
        Iterate over the dataloader.
        """

        if self.pbar is None:
            raise RuntimeError(
                "The epoch method must be called before iteration."
            )
        return self.pbar.__iter__()
