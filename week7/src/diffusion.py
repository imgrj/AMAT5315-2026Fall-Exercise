"""Forward noising, DDPM and DDIM reverse updates (learning sheet equations 4-12)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def direct_noise(x0, abar: float, eps):
    """Equation (5): jump from the clean point straight to level ``t``."""
    return math.sqrt(abar) * jnp.asarray(x0) + math.sqrt(1.0 - abar) * jnp.asarray(eps)


def forward_step(x, sched, t: int, key, sample_index: int = 0):
    """Equation (4): one forward noising step ``x_{t-1} -> x_t`` using ``z``.

    ``key`` of shape ``(n_levels, n, 2)`` is indexed at ``sample_index`` so a
    movie can follow the same samples with the same noise realisation.
    """
    beta = float(sched.betas[t - 1])
    z = jax.random.normal(jax.random.fold_in(key, sample_index), x.shape)
    return jnp.sqrt(1.0 - beta) * x + jnp.sqrt(beta) * z


def ddpm_mean(eps_fn, x, sched, t: int, t_next: int | None = None):
    """Reverse mean for the jump ``t -> t_next`` (equation 8, generalised by 12)."""
    abar_t = float(sched.abars[t])
    eps = eps_fn(x, t)
    if t_next is None:
        t_next = t - 1
    if t_next == t - 1:
        alpha_t = float(sched.alphas[t - 1])
        beta_t = float(sched.betas[t - 1])
        return (x - beta_t / math.sqrt(1.0 - abar_t) * eps) / math.sqrt(alpha_t)
    a = abar_t / float(sched.abars[t_next])
    return (x - (1.0 - a) / math.sqrt(1.0 - abar_t) * eps) / math.sqrt(a)


def ddpm_step(eps_fn, x, sched, t: int, key):
    """One adjacent DDPM step ``t -> t-1`` with variance ``beta_t`` (equation 9).

    The final step (``t = 1``) returns the mean: no noise is added.
    """
    mean = ddpm_mean(eps_fn, x, sched, t, t - 1)
    if t == 1:
        return mean
    beta_t = float(sched.betas[t - 1])
    z = jax.random.normal(key, x.shape)
    return mean + math.sqrt(beta_t) * z


def ddpm_skipped_mean(eps_fn, x, sched, t: int, t_next: int):
    """Mean of a DDPM jump over skipped levels (equation 12)."""
    return ddpm_mean(eps_fn, x, sched, t, t_next)


def ddpm_skipped_step(eps_fn, x, sched, t: int, t_next: int, key):
    """Equation (12) with Gaussian noise of variance ``1 - a`` when ``t_next > 0``."""
    mean = ddpm_skipped_mean(eps_fn, x, sched, t, t_next)
    if t_next == 0:
        return mean
    a = float(sched.abars[t]) / float(sched.abars[t_next])
    z = jax.random.normal(key, x.shape)
    return mean + math.sqrt(1.0 - a) * z


def tweedie_denoise(eps_fn, x, sched, t: int):
    """Equation (10): estimate the clean point from the predicted noise."""
    abar_t = float(sched.abars[t])
    eps = eps_fn(x, t)
    return (x - math.sqrt(1.0 - abar_t) * eps) / math.sqrt(abar_t)


def ddim_step(eps_fn, x, sched, t: int, t_next: int, key=None):
    """Deterministic DDIM update ``t -> t_next`` (equation 11); ``key`` is unused."""
    abar_t = float(sched.abars[t])
    abar_next = float(sched.abars[t_next])
    eps = eps_fn(x, t)
    x0_hat = (x - math.sqrt(1.0 - abar_t) * eps) / math.sqrt(abar_t)
    return math.sqrt(abar_next) * x0_hat + math.sqrt(1.0 - abar_next) * eps


def even_levels(steps: int, T: int = 1000) -> list[int]:
    """``steps + 1`` evenly spaced levels from ``T`` down to ``0``."""
    levels = [int(round(T * (steps - i) / steps)) for i in range(steps + 1)]
    levels[0], levels[-1] = T, 0
    return levels


def _split_steps(levels):
    return list(zip(levels[:-1], levels[1:]))


def ddpm_sample(eps_fn, x_T, sched, levels, key=None, noise=None):
    """Run DDPM over ``levels`` (descending, ending at 0).

    ``noise`` may supply precomputed standard normals of shape
    ``(len(levels) - 1, n, 2)`` so several runs share the same step noise.
    """
    x = jnp.asarray(x_T)
    steps = _split_steps(levels)
    keys = jax.random.split(key, len(steps)) if key is not None else None
    for i, (t, t_next) in enumerate(steps):
        if noise is not None:
            mean = ddpm_skipped_mean(eps_fn, x, sched, t, t_next)
            x = mean if t_next == 0 else mean + jnp.asarray(noise[i]) * math.sqrt(
                1.0 - float(sched.abars[t]) / float(sched.abars[t_next])
            )
        else:
            x = ddpm_skipped_step(eps_fn, x, sched, t, t_next, keys[i])
    return x


def ddim_sample(eps_fn, x_T, sched, levels):
    """Run deterministic DDIM over ``levels`` (descending, ending at 0)."""
    x = jnp.asarray(x_T)
    for t, t_next in _split_steps(levels):
        x = ddim_step(eps_fn, x, sched, t, t_next)
    return x


def ddpm_sample_path(eps_fn, x_T, sched, levels, key=None, noise=None):
    """DDPM over ``levels``, returning ``{level: state}`` for every visited level."""
    x = jnp.asarray(x_T)
    steps = _split_steps(levels)
    keys = jax.random.split(key, len(steps)) if key is not None else None
    path = {levels[0]: x}
    for i, (t, t_next) in enumerate(steps):
        mean = ddpm_skipped_mean(eps_fn, x, sched, t, t_next)
        if t_next == 0:
            x = mean
        else:
            a = float(sched.abars[t]) / float(sched.abars[t_next])
            z = jnp.asarray(noise[i]) if noise is not None else jax.random.normal(keys[i], x.shape)
            x = mean + z * math.sqrt(1.0 - a)
        path[t_next] = x
    return path


def ddim_sample_path(eps_fn, x_T, sched, levels):
    """Deterministic DDIM over ``levels``, returning ``{level: state}``."""
    x = jnp.asarray(x_T)
    path = {levels[0]: x}
    for t, t_next in _split_steps(levels):
        x = ddim_step(eps_fn, x, sched, t, t_next)
        path[t_next] = x
    return path


def forward_movie_states(x0, sched, levels, key):
    """Follow the same samples through the forward steps at ``levels``."""
    import jax.numpy as jnp_  # local alias kept for readability

    states = [jnp_.asarray(x0)]
    x = jnp_.asarray(x0)
    current = 0
    for level in levels:
        if level == 0:
            continue
        for t in range(current + 1, level + 1):
            x = forward_step(x, sched, t, key, sample_index=t)
        current = level
        states.append(x)
    return states
