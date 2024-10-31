from __future__ import annotations
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class LinearWarmupSqrtDecay(_LRScheduler):
    """
    A learning rate scheduler with a linear warmup stage and then a square-root decay stage.
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
