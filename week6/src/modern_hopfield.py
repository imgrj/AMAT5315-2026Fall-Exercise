import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def modern_hopfield_update(X, x, beta):
    """
    Modern Hopfield update: x <- X @ softmax(beta * X^T @ x)
    X: shape (N, P), stored patterns as columns
    x: shape (N,), query state
    beta: inverse temperature
    """
    scores = beta * (X.T @ x)
    weights = jax.nn.softmax(scores)
    return X @ weights

@jax.jit
def modern_hopfield_attention(X, x, beta):
    """
    Compute update using JAX's softmax attention:
    query = x (1, 1, N), keys = X^T (P, 1, N), values = X^T (P, 1, N), scale = beta.
    """
    N, P = X.shape
    q = x.reshape(1, 1, N)
    k = X.T.reshape(P, 1, N)
    v = X.T.reshape(P, 1, N)
    out = jax.nn.dot_product_attention(q, k, v, scale=beta)
    return out.reshape(N)

def modern_hopfield_weights(X, x, beta):
    """Softmax weights over stored patterns: softmax(beta * X^T @ x)."""
    scores = beta * (X.T @ x)
    return jax.nn.softmax(scores)

def check_retrieval_success(retrieved_x, target_pattern, X_all):
    """
    Success criterion:
    1. Overlap of sign(retrieved_x) with target >= 0.95 (at most 19 wrong pixels of 784).
    2. Target is the nearest stored digit: argmax_mu (sign(retrieved_x) @ X[:, mu]) == target_idx.
    """
    retrieved_spins = np.sign(retrieved_x)
    retrieved_spins[retrieved_spins == 0] = 1 # handle zero tie
    N, P = X_all.shape
    
    # Overlap with all stored patterns: shape (P,)
    all_overlaps = (X_all.T @ retrieved_spins) / float(N)
    target_overlap = float(np.dot(target_pattern, retrieved_spins)) / float(N)
    
    is_nearest = (np.argmax(all_overlaps) == np.where((X_all.T == target_pattern).all(axis=1))[0][0]) if (X_all.T == target_pattern).all(axis=1).any() else False
    success = (target_overlap >= 0.95) and is_nearest
    return success, target_overlap
