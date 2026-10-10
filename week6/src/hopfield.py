import numpy as np
from scipy.special import erf

def create_hebbian_couplings(patterns):
    """
    Compute Hebbian coupling matrix:
    J_ij = (1/N) sum_mu xi_i^mu xi_j^mu for i != j, J_ii = 0.
    patterns: shape (P, N) with entries in {-1, +1}.
    """
    P, N = patterns.shape
    # J = (1/N) * (patterns.T @ patterns - P * I)
    J = (patterns.T @ patterns).astype(np.float64) / float(N)
    np.fill_diagonal(J, 0.0)
    return J

def calculate_unstable_bits(J, patterns):
    """
    Fraction of unstable bits: those with xi_i^mu * h_i < 0,
    where h_i = sum_j J_ij xi_j^mu.
    """
    # local fields: shape (P, N)
    fields = patterns @ J.T
    alignment = patterns * fields
    unstable = (alignment < 0.0)
    return np.mean(unstable)

def standard_normal_cdf(x):
    """Standard normal cumulative distribution function Phi(x)."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def crosstalk_prediction_large_n(alpha):
    """Large-N crosstalk prediction: Phi(-1 / sqrt(alpha))."""
    return standard_normal_cdf(-1.0 / np.sqrt(alpha))

def crosstalk_prediction_finite_n(N, P):
    """Finite-N crosstalk prediction: Phi(-sqrt((N - 1) / (P - 1)))."""
    if P <= 1:
        return 0.0
    arg = -np.sqrt(float(N - 1) / float(P - 1))
    return standard_normal_cdf(arg)

def asynchronous_recall(J, s_init, max_sweeps=100, rng=None):
    """
    Zero-temperature asynchronous update in random order:
    s_i <- sign(h_i) where h_i = sum_j J_ij s_j, leaving unchanged if h_i == 0.
    Terminates when a sweep flips no spin (local minimum).
    """
    if rng is None:
        rng = np.random.default_rng()
    N = len(s_init)
    s = s_init.copy().astype(np.int32)
    indices = np.arange(N)

    for sweep in range(max_sweeps):
        flips = 0
        rng.shuffle(indices)
        for i in indices:
            h_i = np.dot(J[i], s)
            if h_i > 0:
                new_s = 1
            elif h_i < 0:
                new_s = -1
            else:
                new_s = s[i]
            if new_s != s[i]:
                s[i] = new_s
                flips += 1
        if flips == 0:
            break
    return s

def calculate_overlap(s, pattern):
    """Overlap m = (1/N) sum_i xi_i * s_i."""
    return float(np.dot(s, pattern)) / float(len(s))
