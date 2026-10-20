"""Time-conditioned fully connected predictor used for noise and velocity."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

TIME_DIM = 32
HIDDEN = 128
DEPTH = 3


def sinusoidal_features(u) -> jnp.ndarray:
    """Numerical time features for a normalised level ``u in [0, 1]``."""
    u = jnp.asarray(u).reshape(-1)
    half = TIME_DIM // 2
    freqs = 2.0 * np.pi * (2.0 ** np.arange(half))
    args = u[:, None] * jnp.asarray(freqs, dtype=jnp.float32)[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def init_predictor(key, in_dim: int = 2, out_dim: int = 2, hidden: int = HIDDEN, depth: int = DEPTH):
    """He-initialised MLP with ``depth`` hidden layers and SiLU activations."""
    sizes = [in_dim + TIME_DIM] + [hidden] * depth + [out_dim]
    keys = jax.random.split(key, len(sizes) - 1)
    weights, biases = [], []
    for k, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:]):
        weights.append(jax.random.normal(k, (fan_in, fan_out), dtype=jnp.float32) * np.sqrt(2.0 / fan_in))
        biases.append(jnp.zeros((fan_out,), dtype=jnp.float32))
    return {"weights": weights, "biases": biases}


def apply(params, x, u) -> jnp.ndarray:
    """Evaluate the network at positions ``x`` and normalised level ``u``."""
    x = jnp.asarray(x, dtype=jnp.float32)
    u = jnp.asarray(u).reshape(-1)
    if u.shape[0] == 1 and x.shape[0] != 1:
        u = jnp.broadcast_to(u, (x.shape[0],))
    h = jnp.concatenate([x, sinusoidal_features(u)], axis=-1)
    n_layers = len(params["weights"])
    for i in range(n_layers - 1):
        h = jax.nn.silu(h @ params["weights"][i] + params["biases"][i])
    return h @ params["weights"][-1] + params["biases"][-1]


def noise_predictor(params, T: int = 1000):
    """Wrap the MLP as ``eps_theta(x_t, t)`` with an integer level ``t``."""

    def predict(x_t, t):
        x_t = jnp.asarray(x_t, dtype=jnp.float32)
        u = jnp.asarray(t, dtype=jnp.float32) / float(T)
        return apply(params, x_t, u)

    return predict


def velocity_predictor(params):
    """Wrap the MLP as ``v_theta(x_s, s)`` with a continuous time ``s in [0, 1]``."""

    def predict(x_s, s):
        x_s = jnp.asarray(x_s, dtype=jnp.float32)
        return apply(params, x_s, jnp.asarray(s, dtype=jnp.float32))

    return predict


def parameter_count(params) -> int:
    return int(sum(np.size(np.asarray(w)) for w in params["weights"]) + sum(np.size(np.asarray(b)) for b in params["biases"]))


def save_params(path, params):
    np.savez(
        path,
        **{f"w{i}": np.asarray(w) for i, w in enumerate(params["weights"])},
        **{f"b{i}": np.asarray(b) for i, b in enumerate(params["biases"])},
    )


def load_params(path):
    data = np.load(path)
    weights = [jnp.asarray(data[k]) for k in sorted((k for k in data.files if k.startswith("w")), key=lambda s: int(s[1:]))]
    biases = [jnp.asarray(data[k]) for k in sorted((k for k in data.files if k.startswith("b")), key=lambda s: int(s[1:]))]
    return {"weights": weights, "biases": biases}

