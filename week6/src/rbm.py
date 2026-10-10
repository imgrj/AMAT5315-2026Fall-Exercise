import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

def init_rbm_params(key, num_visible=784, num_hidden=128):
    """
    Initialize RBM parameters:
    W ~ N(0, 0.01^2), a = 0, b = 0.
    """
    kw, ka, kb = jax.random.split(key, 3)
    W = jax.random.normal(kw, (num_visible, num_hidden), dtype=jnp.float32) * 0.01
    a = jnp.zeros(num_visible, dtype=jnp.float32)
    b = jnp.zeros(num_hidden, dtype=jnp.float32)
    return {"W": W, "a": a, "b": b}

@jax.jit
def energy(params, v, h):
    """E(v, h) = -a^T v - b^T h - v^T W h"""
    return -jnp.dot(params["a"], v) - jnp.dot(params["b"], h) - jnp.dot(v, params["W"] @ h)

@jax.jit
def visible_free_energy_single(params, v):
    """F_theta(v) = -a^T v - sum_j log(1 + exp(b_j + (v^T W)_j))"""
    vW_b = params["b"] + jnp.dot(v, params["W"])
    return -jnp.dot(params["a"], v) - jnp.sum(jax.nn.softplus(vW_b))

@jax.jit
def visible_free_energy(params, v_batch):
    """Vectorized visible free energy over batch of configurations."""
    return jax.vmap(visible_free_energy_single, in_axes=(None, 0))(params, v_batch)

@jax.jit
def hidden_probs(params, v_batch):
    """p(h_j = 1 | v) = sigma(b_j + v^T W)"""
    return jax.nn.sigmoid(params["b"] + v_batch @ params["W"])

@jax.jit
def visible_probs(params, h_batch):
    """p(v_i = 1 | h) = sigma(a_i + h^T W^T)"""
    return jax.nn.sigmoid(params["a"] + h_batch @ params["W"].T)

@jax.jit
def sample_bernoulli(probs, key):
    return (jax.random.uniform(key, probs.shape) < probs).astype(jnp.float32)

@jax.jit
def gibbs_sweep(params, v_batch, key):
    """One full alternating Block Gibbs sweep: v -> h -> v'"""
    kh, kv = jax.random.split(key)
    p_h = hidden_probs(params, v_batch)
    h_sample = sample_bernoulli(p_h, kh)
    p_v = visible_probs(params, h_sample)
    v_sample = sample_bernoulli(p_v, kv)
    return v_sample, h_sample

def gibbs_k_sweeps(params, v_batch, key, k):
    """Run k alternating Block Gibbs sweeps."""
    v = v_batch
    curr_key = key
    for _ in range(k):
        curr_key, subkey = jax.random.split(curr_key)
        v, _ = gibbs_sweep(params, v, subkey)
    return v

@jax.jit
def cd_surrogate_loss(params, v0_batch, vk_batch):
    """
    CD surrogate: E_batch[F_theta(v0) - F_theta(sg(vk))].
    Automatic differentiation gives exact CD gradient.
    """
    f0 = visible_free_energy(params, v0_batch)
    fk = visible_free_energy(params, lax.stop_gradient(vk_batch))
    return jnp.mean(f0 - fk)

@jax.jit
def rbm_train_step(params, v0_batch, vk_batch, lr=0.1):
    """One gradient step on CD surrogate loss."""
    loss, grads = jax.value_and_grad(cd_surrogate_loss)(params, v0_batch, vk_batch)
    new_params = {
        "W": params["W"] - lr * grads["W"],
        "a": params["a"] - lr * grads["a"],
        "b": params["b"] - lr * grads["b"],
    }
    return new_params, loss

def evaluate_reconstruction_error(params, v_test, key, batch_size=1000):
    """
    Measure held-out one-step reconstruction error (Hamming fraction):
    Fraction of changed pixels after one Gibbs sweep from test images.
    """
    num_samples = v_test.shape[0]
    total_diff = 0.0
    curr_key = key
    for i in range(0, num_samples, batch_size):
        curr_key, subkey = jax.random.split(curr_key)
        batch = v_test[i:i + batch_size]
        v_rec, _ = gibbs_sweep(params, batch, subkey)
        total_diff += float(jnp.sum(jnp.abs(batch - v_rec)))
    return total_diff / (num_samples * v_test.shape[1])

def clamped_gibbs_step(params, v_batch, v_obs, mask, key):
    """
    Clamped Gibbs step for image completion:
    resample missing pixels while holding observed pixels fixed.
    mask: True for observed pixels, False for missing pixels.
    """
    v_next, _ = gibbs_sweep(params, v_batch, key)
    # Clamp observed bits
    v_clamped = jnp.where(mask, v_obs, v_next)
    return v_clamped

def calculate_ink_overlap(completion, target, missing_mask):
    """
    Ink overlap: 2*|A n B| / (|A| + |B|) on missing pixels.
    A: inked missing pixels of completion (v=1)
    B: inked missing pixels of target (v=1)
    """
    comp_missing = (completion * (~missing_mask)) > 0.5
    targ_missing = (target * (~missing_mask)) > 0.5
    intersection = np.sum(comp_missing & targ_missing, axis=-1)
    sum_ink = np.sum(comp_missing, axis=-1) + np.sum(targ_missing, axis=-1)
    # If both have 0 ink, overlap is 1.0; if only one has 0, overlap is 0.0
    overlap = np.where(sum_ink == 0, 1.0, 2.0 * intersection / np.maximum(sum_ink, 1e-8))
    return overlap
