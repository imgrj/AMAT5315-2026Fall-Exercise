"""The four-cluster Gaussian mixture and its exact conditional quantities.

The target is ``p_data(x) = sum_k (1/4) N(x; mu_k, 0.09 I)`` with centres
``(3,0), (0,3), (-3,0), (0,-3)``.  Because every component is Gaussian, the
noise-prediction target, the denoiser and the marginal velocity all have exact
closed forms; they are the reference for the learned networks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

CENTERS = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]], dtype=np.float32)
VAR = 0.09
STD = float(np.sqrt(VAR))  # 0.3
COMPONENTS = 4


def sample(key, count: int) -> jnp.ndarray:
    """Draw ``count`` exact samples from the target."""
    key_comp, key_noise = jax.random.split(key)
    comp = jax.random.randint(key_comp, (count,), 0, COMPONENTS)
    z = jax.random.normal(key_noise, (count, 2), dtype=jnp.float32)
    return jnp.asarray(CENTERS)[comp] + STD * z


def nearest_component(x) -> np.ndarray:
    """Assign every point to its nearest target centre."""
    x = np.asarray(x, dtype=np.float64)
    d2 = ((x[:, None, :] - CENTERS[None, :, :].astype(np.float64)) ** 2).sum(-1)
    return d2.argmin(-1)


def _abar_terms(abar):
    """``sqrt(abar)`` and the noisy coordinate variance ``0.09 abar + 1 - abar``."""
    abar = jnp.asarray(abar, dtype=jnp.float32)
    return jnp.sqrt(abar), VAR * abar + (1.0 - abar)


def component_weights(x, abar) -> jnp.ndarray:
    """Posterior weights ``r_k(x_t)`` proportional to the noisy component densities."""
    x = jnp.asarray(x)
    a, v = _abar_terms(abar)
    centres = jnp.asarray(CENTERS)
    d2 = ((x[:, None, :] - a * centres[None, :, :]) ** 2).sum(-1)
    logits = -d2 / (2.0 * v)
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = jnp.exp(logits)
    return w / w.sum(axis=-1, keepdims=True)


def exact_posterior_mean(x, abar) -> jnp.ndarray:
    """``E[x_0 | x_t]`` for the mixture (equation 23-25)."""
    x = jnp.asarray(x)
    a, v = _abar_terms(abar)
    centres = jnp.asarray(CENTERS)
    w = component_weights(x, abar)
    means = centres[None, :, :] + (VAR * a / v) * (x[:, None, :] - a * centres[None, :, :])
    return (w[..., None] * means).sum(axis=1)


def exact_conditional_noise(x, abar) -> jnp.ndarray:
    """``E[eps | x_t]``, the optimal squared-error noise prediction."""
    x = jnp.asarray(x)
    a, _ = _abar_terms(abar)
    b = jnp.sqrt(1.0 - jnp.asarray(abar, dtype=jnp.float32))
    return (x - a * exact_posterior_mean(x, abar)) / b


def exact_posterior_variance(x, abar):
    """Total per-point posterior variance of ``x_0 | x_t`` (both sources)."""
    x = jnp.asarray(x)
    a, v = _abar_terms(abar)
    centres = jnp.asarray(CENTERS)
    w = component_weights(x, abar)
    means = centres[None, :, :] + (VAR * a / v) * (x[:, None, :] - a * centres[None, :, :])
    mixture_mean = (w[..., None] * means).sum(axis=1, keepdims=True)
    within = VAR * (1.0 - abar) / v
    between = ((means - mixture_mean) ** 2).sum(axis=-1)
    return (w * (within + between)).sum(axis=-1)


def exact_posterior_std(x, abar) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(exact_posterior_variance(x, abar), 0.0))

