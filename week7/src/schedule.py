"""Linear variance schedule of the learning sheet (T = 1000 steps)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T = 1000
BETA_START = 1e-4
BETA_END = 0.02


@dataclass(frozen=True)
class Schedule:
    """Per-step variances and their cumulative products.

    ``betas[t-1]`` is :math:`\\beta_t`, ``alphas[t-1] = 1 - \\beta_t` and
    ``abars[t] = \\prod_{s \\le t} \\alpha_s`` with ``abars[0] = 1``.
    """

    betas: np.ndarray
    alphas: np.ndarray
    abars: np.ndarray

    def __len__(self) -> int:
        return int(self.betas.shape[0])


def linear_schedule(T: int = T, beta_start: float = BETA_START, beta_end: float = BETA_END) -> Schedule:
    """``beta_t`` increasing linearly from ``beta_start`` to ``beta_end``."""
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    alphas = 1.0 - betas
    abars = np.concatenate([[1.0], np.cumprod(alphas)])
    return Schedule(betas=betas, alphas=alphas, abars=abars)
