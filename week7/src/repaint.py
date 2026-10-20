"""RePaint inpainting sampler (learning sheet equations 19-22).

Jump over the levels ``1000, 980, ..., 20, 0``.  Each jump proposes hidden
pixels from the clipped Tweedie estimate, merges the forward-noised
observations at the destination level, and is retried once (after forward
noising back to the source level) except for the final jump to zero.  That is
``49 * 2 + 1 = 99`` network evaluations per completion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def repaint_levels(step: int = 20, T: int = 1000) -> list[int]:
    """``1000, 980, ..., 20, 0``: 50 jumps from the noisiest level to zero."""
    return list(range(T, 0, -step)) + [0]


def repaint_complete(eps_fn, sched, observed, mask, x_T, key, levels, attempts: int = 2,
                     snapshots=None):
    """Fill ``mask == 0`` while overwriting ``mask == 1`` with noised observations.

    ``observed`` holds the known clean values at observed entries (any value at
    hidden entries is ignored).  ``x_T`` and every noise draw are the sampler's
    only randomness.
    """
    x = jnp.asarray(x_T, dtype=jnp.float32)
    observed = jnp.asarray(observed, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    if snapshots is not None:
        snapshots[levels[0]] = np.asarray(x)
    key_stream = key
    for t, t_next in zip(levels[:-1], levels[1:]):
        abar_t = float(sched.abars[t])
        abar_next = float(sched.abars[t_next])
        a = abar_t / abar_next
        key_stream, k_eta = jax.random.split(key_stream)
        eta = jax.random.normal(k_eta, x.shape)
        y_noised = np.sqrt(abar_next) * observed + np.sqrt(1.0 - abar_next) * eta
        attempts_here = attempts if t_next > 0 else 1
        for attempt in range(attempts_here):
            if attempt > 0:
                key_stream, k_forward = jax.random.split(key_stream)
                f = jax.random.normal(k_forward, x.shape)
                x = np.sqrt(a) * x + np.sqrt(1.0 - a) * f
            eps = eps_fn(x, t)
            x0_hat = (x - np.sqrt(1.0 - abar_t) * eps) / np.sqrt(abar_t)
            x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
            mean = (
                np.sqrt(abar_next) * (1.0 - a) / (1.0 - abar_t) * x0_hat
                + np.sqrt(a) * (1.0 - abar_next) / (1.0 - abar_t) * x
            )
            variance = (1.0 - abar_next) * (1.0 - a) / (1.0 - abar_t)
            if variance > 0.0:
                key_stream, k_step = jax.random.split(key_stream)
                z = jax.random.normal(k_step, x.shape)
                proposal = mean + np.sqrt(variance) * z
            else:
                proposal = mean
            x = mask * y_noised + (1.0 - mask) * proposal
        if snapshots is not None:
            snapshots[t_next] = np.asarray(x)
    return x


def square_mask(shape, start: int = 6, stop: int = 26):
    """Mask that hides rows and columns ``start:stop`` in every channel."""
    mask = np.ones(shape, dtype=np.float32)
    mask[..., start:stop, start:stop, :] = 0.0
    return mask


def hidden_mse(completion, truth, mask) -> float:
    hidden = np.asarray(mask) == 0.0
    hidden = np.broadcast_to(hidden, completion.shape)
    diff = np.asarray(completion) - np.asarray(truth)
    return float(np.mean(diff[hidden] ** 2))
