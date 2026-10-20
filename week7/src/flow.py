"""Flow matching on straight paths and its exact marginal velocity."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import mixture


def path_point(x0, z, s: float):
    """Equation (13): ``x_s = (1 - s) x_0 + s z``."""
    return (1.0 - s) * x0 + s * z


def path_velocity(x0, z):
    """The known straight-path velocity ``z - x_0``."""
    return z - x0


def exact_marginal_velocity(x, s: float) -> jnp.ndarray:
    """``E[z - x_0 | x_s = x]`` for the Gaussian mixture (equation 15)."""
    if s <= 0.0:
        raise ValueError("evaluate the marginal velocity at s > 0")
    x = jnp.asarray(x)
    a = 1.0 - s
    u = a * a * mixture.VAR + s * s
    centres = jnp.asarray(mixture.CENTERS)
    d2 = ((x[:, None, :] - a * centres[None, :, :]) ** 2).sum(-1)
    logits = -d2 / (2.0 * u)
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = jnp.exp(logits)
    w = w / w.sum(axis=-1, keepdims=True)
    x0_means = centres[None, :, :] + (a * mixture.VAR / u) * (x[:, None, :] - a * centres[None, :, :])
    v_k = (x[:, None, :] - x0_means) / s
    return (w[..., None] * v_k).sum(axis=1)


def exact_velocity_fn():
    def velocity(x, s):
        return exact_marginal_velocity(x, float(s))

    return velocity


def euler_sample(velocity_fn, x_1, steps: int, s_start: float = 1.0, s_end: float = 0.0):
    """Equation (17): Euler steps from ``s_start`` down to ``s_end``."""
    x = jnp.asarray(x_1)
    ds = (s_start - s_end) / steps
    for i in range(steps):
        s = s_start - i * ds
        x = x - ds * velocity_fn(x, s)
    return x


def velocity_relative_rms(learned, exact) -> float:
    """RMS of the difference divided by the RMS of the exact velocity."""
    learned = np.asarray(learned, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    num = float(np.sqrt(np.mean(np.sum((learned - exact) ** 2, axis=-1))))
    den = float(np.sqrt(np.mean(np.sum(exact**2, axis=-1))))
    return num / den
