from typing import Callable, Optional, Generator

import torch
from torch import nn

# Learning Rate Scheduler
from torch.optim import lr_scheduler

# Optimization Algorithms
from torch.optim import Optimizer


def build_scheduler(
    opts, optimizer: Optimizer, scheduler_mode: str, hidden_size: int = 0
) -> (Optional[lr_scheduler._LRScheduler], Optional[str]):
    """
    Create a learning rate scheduler if specified in config and
    determine when a scheduler step should be executed.

    Current options:
        - "plateau": see `torch.optim.lr_scheduler.ReduceLROnPlateau`
        - "decaying": see `torch.optim.lr_scheduler.StepLR`
        - "exponential": see `torch.optim.lr_scheduler.ExponentialLR`
        - "noam": see `joeynmt.transformer.NoamScheduler`

    If no scheduler is specified, returns (None, None) which will result in
    a constant learning rate.

    :param config: training configuration
    :param optimizer: optimizer for the scheduler, determines the set of
        parameters which the scheduler sets the learning rate for
    :param scheduler_mode: "min" or "max", depending on whether the validation
        score should be minimized or maximized.
        Only relevant for "plateau".
    :param hidden_size: encoder hidden size (required for NoamScheduler)
    :return:
        - scheduler: scheduler object,
        - scheduler_step_at: either "validation" or "epoch"
    """
    scheduler_name = opts.scheduler_name   # ["scheduling"].lower()

    if scheduler_name == "plateau":
        # learning rate scheduler
        return (
            lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=scheduler_mode,
                verbose=False,
                threshold_mode="abs",
                factor=opts.decrease_factor,          # config.get("decrease_factor", 0.1),
                patience=opts.patience  # config.get("patience", 10),
            ),
            "validation",
        )
    elif scheduler_name == "cosineannealing":
        return (
            lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=config.get("eta_min", 0),
                T_max=config.get("t_max", 20),
            ),
            "epoch",
        )
    elif scheduler_name == "cosineannealingwarmrestarts":
        return (
            lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=config.get("t_init", 10),
                T_mult=config.get("t_mult", 2),
            ),
            "step",
        )
    elif scheduler_name == "decaying":
        return (
            lr_scheduler.StepLR(
                optimizer=optimizer, step_size=config.get("decaying_step_size", 1)
            ),
            "epoch",
        )
    elif scheduler_name == "exponential":
        return (
            lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=config.get("decrease_factor", 0.99)
            ),
            "epoch",
        )
    elif scheduler_name == "noam":
        factor = config.get("learning_rate_factor", 1)
        warmup = config.get("learning_rate_warmup", 4000)
        return (
            NoamScheduler(
                hidden_size=hidden_size,
                factor=factor,
                warmup=warmup,
                optimizer=optimizer,
            ),
            "step",
        )
    elif scheduler_name == "warmupexponentialdecay":
        min_rate = config.get("learning_rate_min", 1.0e-5)
        decay_rate = config.get("learning_rate_decay", 0.1)
        warmup = config.get("learning_rate_warmup", 4000)
        peak_rate = config.get("learning_rate_peak", 1.0e-3)
        decay_length = config.get("learning_rate_decay_length", 10000)
        return (
            WarmupExponentialDecayScheduler(
                min_rate=min_rate,
                decay_rate=decay_rate,
                warmup=warmup,
                optimizer=optimizer,
                peak_rate=peak_rate,
                decay_length=decay_length,
            ),
            "step",
        )
    else:
        raise ValueError("Unknown learning scheduler {}.".format(scheduler_name))


class NoamScheduler:
    """
    The Noam learning rate scheduler used in "Attention is all you need"
    See Eq. 3 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        optimizer: torch.optim.Optimizer,
        factor: float = 1,
        warmup: int = 4000,
    ):
        """
        Warm-up, followed by learning rate decay.
        :param hidden_size:
        :param optimizer:
        :param factor: decay factor
        :param warmup: number of warmup steps
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        return self.factor * (
            self.hidden_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    # pylint: disable=no-self-use
    def state_dict(self):
        return None


class WarmupExponentialDecayScheduler:
    """
    A learning rate scheduler similar to Noam, but modified:
    Keep the warm up period but make it so that the decay rate can be tuneable.
    The decay is exponential up to a given minimum rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_rate: float = 1.0e-3,
        decay_length: int = 10000,
        warmup: int = 4000,
        decay_rate: float = 0.5,
        min_rate: float = 1.0e-5,
    ):
        """
        Warm-up, followed by exponential learning rate decay.
        :param peak_rate: maximum learning rate at peak after warmup
        :param optimizer:
        :param decay_length: decay length after warmup
        :param decay_rate: decay rate after warmup
        :param warmup: number of warmup steps
        :param min_rate: minimum learning rate
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.decay_length = decay_length
        self.peak_rate = peak_rate
        self._rate = 0
        self.decay_rate = decay_rate
        self.min_rate = min_rate

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        warmup = self.warmup

        if step < warmup:
            rate = step * self.peak_rate / warmup
        else:
            exponent = (step - warmup) / self.decay_length
            rate = self.peak_rate * (self.decay_rate ** exponent)
        return max(rate, self.min_rate)

    # pylint: disable=no-self-use
    def state_dict(self):
        return None