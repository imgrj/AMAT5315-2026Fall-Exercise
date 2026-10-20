"""Training loops: noise prediction (Part 1) and velocity prediction (Part 2)."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from . import mixture, model, schedule

DEFAULT_UPDATES = 25000
DEFAULT_BATCH = 512
DEFAULT_LR = 1e-3
DEFAULT_STAGES = (0, 100, 300, 1000, 3000, 6000, 10000, 15000, 20000, 25000)


def _zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _learning_rate(base_lr: float, update: int, updates: int) -> float:
    """Cosine decay from ``base_lr`` to zero at the final update."""
    return float(base_lr * 0.5 * (1.0 + np.cos(np.pi * update / max(updates, 1))))


def _adam_update(params, grads, state, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1 - beta2) * g * g, state["v"], grads)
    step = state["step"] + 1
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - beta2**step), v)
    params = jax.tree_util.tree_map(lambda p, m_, v_: p - lr * m_ / (jnp.sqrt(v_) + eps), params, m_hat, v_hat)
    return params, {"m": m, "v": v, "step": step}


def _noise_batch(key, abars, T: int, batch: int):
    k0, k1, k2 = jax.random.split(key, 3)
    x0 = mixture.sample(k0, batch)
    t = jax.random.randint(k1, (batch,), 1, T + 1)
    eps = jax.random.normal(k2, x0.shape)
    abar = jnp.asarray(abars, dtype=jnp.float32)[t]
    x_t = jnp.sqrt(abar)[:, None] * x0 + jnp.sqrt(1.0 - abar)[:, None] * eps
    return x_t, t, eps


def noise_loss(params, key, abars, T: int, batch: int):
    x_t, t, eps = _noise_batch(key, abars, T, batch)
    pred = model.apply(params, x_t, t.astype(jnp.float32) / float(T))
    return 0.5 * jnp.mean(jnp.sum((pred - eps) ** 2, axis=-1))


def _velocity_batch(key, batch: int):
    k0, k1, k2 = jax.random.split(key, 3)
    x0 = mixture.sample(k0, batch)
    z = jax.random.normal(k1, x0.shape)
    s = jax.random.uniform(k2, (batch,))
    x_s = (1.0 - s)[:, None] * x0 + s[:, None] * z
    return x_s, s, z - x0


def velocity_loss(params, key, batch: int):
    x_s, s, target = _velocity_batch(key, batch)
    pred = model.apply(params, x_s, s)
    return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))


def train_noise_predictor(
    key,
    updates: int = DEFAULT_UPDATES,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    stages=DEFAULT_STAGES,
    sched=None,
    log_every: int = 25,
):
    """Minimise equation (6) and return the saved stages plus the loss history."""
    sched = sched or schedule.linear_schedule()
    params = model.init_predictor(key)
    state = {"m": _zeros_like(params), "v": _zeros_like(params), "step": 0}
    grad_fn = jax.jit(jax.value_and_grad(noise_loss), static_argnames=("T", "batch"))
    history = {"updates": [], "loss": []}
    saved = {}
    stages = sorted(set(int(s) for s in stages))
    started = time.time()
    key_stream = jax.random.fold_in(key, 1)
    for update in range(updates + 1):
        if update in stages:
            saved[update] = jax.tree_util.tree_map(lambda a: jnp.array(a), params)
        if update == updates:
            break
        key_stream, step_key = jax.random.split(key_stream)
        loss, grads = grad_fn(params, step_key, jnp.asarray(sched.abars, dtype=jnp.float32), T=len(sched), batch=batch)
        params, state = _adam_update(params, grads, state, _learning_rate(lr, update, updates))
        if update % log_every == 0 or update + 1 == updates:
            history["updates"].append(update + 1)
            history["loss"].append(float(loss))
    history["seconds"] = time.time() - started
    return saved, history


def train_velocity_network(
    key,
    updates: int = DEFAULT_UPDATES,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    stages=DEFAULT_STAGES,
    log_every: int = 25,
):
    """Minimise equation (14) from scratch and return the saved stages."""
    params = model.init_predictor(key)
    state = {"m": _zeros_like(params), "v": _zeros_like(params), "step": 0}
    grad_fn = jax.jit(jax.value_and_grad(velocity_loss), static_argnames=("batch",))
    history = {"updates": [], "loss": []}
    saved = {}
    stages = sorted(set(int(s) for s in stages))
    started = time.time()
    key_stream = jax.random.fold_in(key, 1)
    for update in range(updates + 1):
        if update in stages:
            saved[update] = jax.tree_util.tree_map(lambda a: jnp.array(a), params)
        if update == updates:
            break
        key_stream, step_key = jax.random.split(key_stream)
        loss, grads = grad_fn(params, step_key, batch=batch)
        params, state = _adam_update(params, grads, state, _learning_rate(lr, update, updates))
        if update % log_every == 0 or update + 1 == updates:
            history["updates"].append(update + 1)
            history["loss"].append(float(loss))
    history["seconds"] = time.time() - started
    return saved, history


