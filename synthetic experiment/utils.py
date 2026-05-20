import numpy as np
from scipy.fftpack import dct
from matplotlib import pyplot as plt
def l1(x):
    return np.sum(np.abs(x))

def prox_l1(x, alpha):
    # Soft-thresholding (L1 prox)
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0.0)


def l2(x):
    return np.sqrt(np.sum(x**2))

def prox_l2_X(X, alpha):
    # L2 prox for array of points
    dim_X = len(X.shape)
    if dim_X == 1:
        return np.maximum(1-alpha/np.sqrt(np.sum(X**2)), 0.0) * X
    else:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        scales = np.maximum(1 - alpha / norms, 0.0)
        return X * scales

def prox_l2(x, alpha):
    # L2 prox
    return np.maximum(1-alpha/np.sqrt(np.sum(x**2)), 0.0) * x


def prox_l1_minus_l2(y, alpha, epsilon=1.0):
    """
    Prox of r_alpha(x) = ||x||_1 - alpha * ||x||_2 at y, parameter lam > 0.
    Implements Lemma 1 (Lou & Yan, 2016).
    """
    y = np.asarray(y, dtype=float)
    amax = np.max(np.abs(y))
    if amax <= (1 - epsilon) * alpha:
        return np.zeros_like(y)

    if amax > alpha:
        z = prox_l1(y, alpha)
        nz = np.linalg.norm(z)
        # nz > 0 because amax > lam
        scale = (nz + epsilon * alpha) / nz
        return scale * z

    # (1 - alpha)*lam < amax <= lam  
    i = np.argmax(np.abs(y))
    mag = amax + (epsilon - 1) * alpha           # > 0 here
    x = np.zeros_like(y)
    x[i] = np.sign(y[i]) * mag
    return x

def compute_rela_err(x_est, x_true):  # smaller is better
    num = np.linalg.norm(x_true - x_est)**2
    den = np.linalg.norm(x_true)**2
    return num / den

def partial_dct_matrix(m, n):
    r = np.random.rand(m)  # m random frequencies in [0,1]
    A = np.zeros((m, n))
    for i in range(n):
        A[:, i] = (1 / np.sqrt(m)) * np.cos(2 * (i + 1) * np.pi * r)
    return A

def oversampled_dct_matrix(m, n, F):
    r = np.random.rand(m)  # m random frequencies in [0,1]
    A = np.zeros((m, n))
    for i in range(n):
        A[:, i] = (1 / np.sqrt(m)) * np.cos(2 * (i + 1) * np.pi * r / F)
    return A


def generate_sparse_vector(n, k, L):
    """
    Generate a k-sparse vector x in R^n with minimum separation L between spikes.

    Parameters
    ----------
    n : int
        Length of the vector.
    k : int
        Number of nonzero entries (sparsity level).
    L : int
        Minimum separation between nonzero indices.

    Returns
    -------
    x : np.ndarray
        The k-sparse vector of length n.
    support : np.ndarray
        Indices of nonzero elements.
    """
        
    if (k - 1) * L + 1 > n:
        raise ValueError("Infeasible: vector too short for given k and L.")
        
    # Available positions
    available = list(range(n))
    support = []
    
    while len(support) < k:
        idx = np.random.choice(available)
        support.append(idx)
        # Remove indices within L distance
        available = [i for i in available if abs(i - idx) >= L]
    
    x = np.zeros(n)
    x[support] = np.random.randn(k)
    support.sort()
    return x, np.array(support)

# to check------
def prox_sigma_q(y, alpha, q):
    """
    Proximal operator of alpha * ||.||_{sigma_q},
    where ||x||_{sigma_q} = sum of q largest magnitudes of x.
    """
    y = np.asarray(y, dtype=float)
    abs_y = np.abs(y)
    # Indices of q largest entries
    idx = np.argsort(-abs_y)[:q]
    x = np.array(y, copy=True)
    # Apply soft-thresholding only on top-q entries
    x[idx] = np.sign(y[idx]) * np.maximum(abs_y[idx] - alpha, 0.0)
    return x


def prox_l1_minus_sigma_q(y, alpha, q):
    """
    Proximal operator of alpha * (||.||_1 - ||.||_{sigma_q}).
    Equivalent to soft-thresholding everywhere except we keep q indices
    (with largest scores) unpenalized.
    """
    y = np.asarray(y, dtype=float)
    abs_y = np.abs(y)

    # Score function: cost saved by leaving i unpenalized
    scores = np.where(
        abs_y <= alpha,
        0.5 * abs_y**2,
        alpha * abs_y - 0.5 * alpha**2
    )

    # Pick q indices with largest scores (unpenalized)
    idx = np.argsort(-scores)[:q]

    # Apply soft-thresholding everywhere
    x = prox_l1(y, alpha)
    # Restore unpenalized coordinates to original values
    x[idx] = y[idx]
    return x

def plot_ci_comparison(samples_dict, x_true, indices, save_path, title_suffix=""):
    """
    Plot 95% confidence intervals across given indices
    for multiple samplers side by side.
    
    samples_dict: dict of {label: samples}
    x_true: true vector
    indices: list/array of dimensions to plot
    """
    dims = np.arange(len(indices))

    fig, axes = plt.subplots(1, len(samples_dict), figsize=(18, 5), sharey=True)

    for ax, (label, samples) in zip(axes, samples_dict.items()):
        lower = np.percentile(samples[:, indices], 2.5, axis=0)
        upper = np.percentile(samples[:, indices], 97.5, axis=0)
        mean_est = np.mean(samples[:, indices], axis=0)
        true_vals = x_true[indices]

        ax.fill_between(dims, lower, upper, color="skyblue", alpha=0.4, label="95% CI")
        ax.plot(dims, mean_est, "b--", label="Posterior mean")
        ax.plot(dims, true_vals, "r-", label="True x*")

        ax.set_title(f"{label} {title_suffix}")
        ax.set_xlabel("Index within selected set")

    axes[0].set_ylabel("Value")
    axes[0].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def median_heuristic_bandwidth(X: np.ndarray) -> float:
    """
    Median heuristic bandwidth for an RBF kernel.
    Returns sigma (the bandwidth, not sigma^2).
    """
    X = np.atleast_2d(X)
    # pairwise squared distances
    diffs = X[:, None, :] - X[None, :, :]
    sq_dists = np.sum(diffs**2, axis=-1)
    # take median of off-diagonal distances
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples for median heuristic.")
    off_diag = sq_dists[~np.eye(n, dtype=bool)]
    # Avoid zero bandwidth in degenerate cases
    med = np.median(off_diag)
    sigma = np.sqrt(0.5 * med) if med > 0.0 else 1.0
    return float(sigma)

def ksd_rbf(
    X: np.ndarray,
    score_fn,
    bandwidth: float | str = "median",
    unbiased: bool = False,
) -> float:
    """
    Kernelized Stein Discrepancy (KSD) with RBF kernel, as in
    Liu, Lee, Jordan (2016), "A Kernelized Stein Discrepancy for Goodness-of-fit Tests".

    KSD^2 = E_{i,j}[ u_p(x_i, x_j) ]  where
      u_p(x, x') =
        s_p(x)^T k(x,x') s_p(x')
        + s_p(x)^T ∇_{x'} k(x,x')
        + s_p(x')^T ∇_{x} k(x,x')
        + tr( ∇_{x} ∇_{x'} k(x,x') ),
    with s_p(x) = ∇_x log p(x).  We estimate the expectation via the V- or U-statistic.

    Args
    ----
    X : (n, d) ndarray
        Samples x_i ~ q(x) (the distribution you want to test against p).
    score_fn : callable
        Function score_fn(X) -> (n, d) giving s_p(x) = ∇_x log p(x) evaluated at X.
        You provide this for the target density p (up to a normalization constant is fine).
    bandwidth : float or "median", default "median"
        RBF kernel bandwidth sigma. If "median", uses the median heuristic.
    unbiased : bool, default False
        If True, compute the U-statistic (exclude diagonal terms) with normalization n*(n-1).
        If False, compute the V-statistic (include diagonal terms) with normalization n^2.

    Returns
    -------
    ksd : float
        The (nonnegative) scalar KSD value.

    Notes
    -----
    For the RBF kernel k(x,x') = exp(-||x - x'||^2 / (2 σ^2)):
      ∇_x k = -k * (x - x') / σ^2
      ∇_{x'} k =  k * (x - x') / σ^2
      tr(∇_x ∇_{x'} k) = k * ( d/σ^2 - ||x - x'||^2 / σ^4 )
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n, d).")
    n, d = X.shape
    if n < 2:
        raise ValueError("Need at least two samples to estimate KSD.")

    # Scores under the target p
    S = np.asarray(score_fn(X), dtype=float)  # shape (n, d)
    if S.shape != (n, d):
        raise ValueError(f"score_fn(X) must return array of shape {(n, d)}, got {S.shape}.")

    # Bandwidth selection
    if isinstance(bandwidth, str):
        if bandwidth.lower() != "median":
            raise ValueError('bandwidth must be a float or the string "median".')
        sigma = median_heuristic_bandwidth(X)
    else:
        sigma = float(bandwidth)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("bandwidth must be a positive finite float.")

    sigma2 = sigma**2
    inv_sigma2 = 1.0 / sigma2
    inv_sigma4 = inv_sigma2**2

    # Pairwise differences and kernel
    diffs = X[:, None, :] - X[None, :, :]                      # (n, n, d) = x_i - x_j
    sq_dists = np.sum(diffs**2, axis=-1)                       # (n, n)
    K = np.exp(-0.5 * sq_dists * inv_sigma2)                   # (n, n)

    # Terms in u_p(x_i, x_j)
    # 1) s(x)^T k s(x')
    SS = S @ S.T                                               # (n, n)
    term1 = SS * K

    # 2) s(x)^T ∇_{x'} k = k * (s_i · (x_i - x_j)) / σ^2
    Si_dot_diffs = np.einsum("id,ijd->ij", S, diffs)           # (n, n)
    term2 = K * (Si_dot_diffs * inv_sigma2)

    # 3) s(x')^T ∇_x k = -k * (s_j · (x_i - x_j)) / σ^2
    Sj_dot_diffs = np.einsum("jd,ijd->ij", S, diffs)           # (n, n)
    term3 = -K * (Sj_dot_diffs * inv_sigma2)

    # 4) tr(∇_x ∇_{x'} k) = k * ( d/σ^2 - ||x_i - x_j||^2 / σ^4 )
    term4 = K * (d * inv_sigma2 - sq_dists * inv_sigma4)

    U = term1 + term2 + term3 + term4                          # (n, n)
    if unbiased:
        # Exclude diagonal, normalize by n*(n-1)
        np.fill_diagonal(U, 0.0)
        ksd2 = U.sum() / (n * (n - 1))
    else:
        # V-statistic normalization by n^2
        ksd2 = U.mean()

    # Numerical guard: KSD should be >= 0
    return float(np.sqrt(max(ksd2, 0.0)))