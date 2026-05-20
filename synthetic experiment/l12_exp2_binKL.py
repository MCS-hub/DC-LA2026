import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import pickle
import matplotlib.colors as mcolors
from joblib import Parallel, delayed
from utils import l1, prox_l1, l2, prox_l2, prox_l1_minus_l2
from sampling_algs import DC_LA, ULA_s, PSGLA, ULA_ns
np.random.seed(42)
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 14,          # base font size
    "axes.titlesize": 14,     # subplot titles
    "axes.labelsize": 14,     # x/y labels
    "xtick.labelsize": 14,    # tick labels
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14
})

# multiple chains
def main():
    d = 2
    mu_x_list = [0, 1, 2, 3]
    Sigma_x_list = [np.array([[1, 0.0], [0.0, 1]]), np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 2]])]
    tau_list = [10]

    lam, gamma = 0.01, 0.005
    n_samples = 1000
    burn_in = 0
    n_chains = 5000
    for tau in tau_list:
        for mu_x in mu_x_list:
            for Sigma_x in Sigma_x_list:
                print("mu_x:", mu_x, "Sigma_x:", Sigma_x, "tau:", tau)
                def r1(x):
                    return tau * l1(x)
                def r1_grad(x):
                    return tau * np.sign(x)
                def r2_grad(x, eps=1e-12):
                    norm = np.linalg.norm(x, ord=2)
                    if norm > eps:
                        return tau * x / norm
                    else:
                        return np.zeros_like(x)
                def prox_r1(x, alpha):
                    return prox_l1(x, tau * alpha)
                def r2(x):
                    return tau * l2(x)
                def prox_r2(x, alpha):
                    return prox_l2(x, tau * alpha)
                def prox_r(x, alpha):
                    return prox_l1_minus_l2(x, tau * alpha)
                def f(x):
                    return 0.5 * (x-mu_x).T @ Sigma_x @ (x-mu_x)  
                def grad_f(x):
                    return Sigma_x @ (x -mu_x) 
                def V(x):
                    return f(x) + r1(x) - r2(x)
                
                samples_ulas_last = []
                samples_dcla_last = []
                samples_psgla_last = []
                samples_ulans_last = []

                def run_chain_once(d, n_samples, burn_in, lam, gamma, grad_f, prox_r1, prox_r2, prox_r, grad_r1, grad_r2):
                    X0 = np.random.randn(d)
                    samples_ulas  = ULA_s(X0, n_samples, burn_in, lam, gamma, d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)
                    samples_dcla = DC_LA(X0, n_samples, burn_in, lam, gamma, d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)
                    samples_psgla = PSGLA(X0, n_samples, burn_in, gamma, d, grad_f=grad_f, prox_r=prox_r)
                    samples_ulans = ULA_ns(X0, n_samples, burn_in, gamma, d, grad_f=grad_f, grad_r1=grad_r1, grad_r2=grad_r2)
                    
                    return samples_ulas[-1], samples_dcla[-1], samples_psgla[-1], samples_ulans[-1]

                # Run chains in parallel
                results = Parallel(n_jobs=-1)(   # -1 = use all available cores
                    delayed(run_chain_once)(d, n_samples, burn_in, lam, gamma, grad_f, prox_r1, prox_r2, prox_r, r1_grad, r2_grad)
                    for _ in range(n_chains)
                )

                # Unpack results
                samples_ulas_last, samples_dcla_last, samples_psgla_last, samples_ulans_last = zip(*results)
                samples_ulas_last = np.array(samples_ulas_last)
                samples_dcla_last = np.array(samples_dcla_last)
                samples_psgla_last = np.array(samples_psgla_last)
                samples_ulans_last = np.array(samples_ulans_last)

                # target density
                def pi_unnormalized(x):
                    return np.exp(-V(x))

                def pi_unnormalized_2d(x1, x2):
                    x = np.array([x1, x2])
                    return pi_unnormalized(x)

                # Integral 
                val, err = dblquad(lambda x2, x1: pi_unnormalized_2d(x1, x2),
                                -np.inf, np.inf,  # x1 range
                                lambda _: -np.inf, lambda _: np.inf)  # x2 range
                print(f"Integral of unnormalized density: {val:.6f} ± {err:.4e}")
                print('error in integral estimate:', err)

                def pi_normalized(x):
                    return np.exp(-V(x))/val
                
                                # ----- Bin-KL comparisons -----
                # Settings
                EPS = 1e-12  # smoothing to avoid log(0)

                # Build bin edges from combined samples with a small padding
                all_samples = np.vstack([samples_ulas_last, samples_psgla_last, samples_dcla_last, samples_ulans_last])
                
                x_q01, x_q99 = np.quantile(all_samples[:, 0], [0.01, 0.99])
                y_q01, y_q99 = np.quantile(all_samples[:, 1], [0.01, 0.99])
                pad_x = 0.05 * (x_q99 - x_q01 + EPS)
                pad_y = 0.05 * (y_q99 - y_q01 + EPS)
                x_min, x_max = x_q01 - pad_x, x_q99 + pad_x
                y_min, y_max = y_q01 - pad_y, y_q99 + pad_y

                NBINS_LIST = [20, 30, 40, 50]
                ulas_binKL_list = []
                psgla_binKL_list = []
                dcla_binKL_list = []
                ulans_binKL_list = []
                for NBINS in NBINS_LIST:
                    NBINS_X = NBINS
                    NBINS_Y = NBINS

                    x_edges = np.linspace(x_min, x_max, NBINS_X + 1)
                    y_edges = np.linspace(y_min, y_max, NBINS_Y + 1)

                    # Integrate pi over each bin using dblquad (parallelized)
                    def integrate_bin(xa, xb, ya, yb):
                        # dblquad integrates over y first, then x: integrand(y, x)
                        val, err = dblquad(lambda y, x: pi_normalized(np.array([x, y])),
                                        xa, xb,
                                        lambda _: ya, lambda _: yb)
                        return val

                    # Prepare bin rectangles
                    bin_rects = []
                    for i in range(NBINS_X):
                        for j in range(NBINS_Y):
                            xa, xb = x_edges[i], x_edges[i+1]
                            ya, yb = y_edges[j], y_edges[j+1]
                            bin_rects.append((xa, xb, ya, yb))
                    # Compute target bin probabilities
                    target_probs_flat = Parallel(n_jobs=-1)(
                        delayed(integrate_bin)(xa, xb, ya, yb) for (xa, xb, ya, yb) in bin_rects
                    )
                    #print('target_probs_flat[:10]:', target_probs_flat[:10])
                    target_probs = np.array(target_probs_flat).reshape(NBINS_X, NBINS_Y)
                    
                    # Renormalize to exactly 1 (tiny numerical drift)
                    target_probs = np.maximum(target_probs, 0.0)
                    tp_sum = target_probs.sum()
                    if tp_sum <= 0:
                        raise RuntimeError("Target bin probability sum is non-positive. Check bin domain.")
                    target_probs /= tp_sum

                    # Empirical histograms for each sampler
                    def hist2d_probs(samples):
                        H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1],
                                                bins=[x_edges, y_edges])
                        H = H.astype(float)
                        H_sum = H.sum()
                        if H_sum == 0:
                            # Degenerate case
                            return np.ones_like(H) / H.size
                        return H / H_sum

                    p_ulas   = hist2d_probs(samples_ulas_last)
                    p_psgla = hist2d_probs(samples_psgla_last)
                    p_dcla  = hist2d_probs(samples_dcla_last)
                    p_ulans = hist2d_probs(samples_ulans_last)
                    # KL utilities
                    def kl_pq(p, q, eps=EPS):
                        # KL(p||q) where p and q are 2D arrays summing to 1
                        p = p.ravel().astype(float)
                        q = q.ravel().astype(float)
                        p = np.clip(p, eps, 1.0)
                        q = np.clip(q, eps, 1.0)
                        p /= p.sum()
                        q /= q.sum()
                        return float(np.sum(p * np.log(p / q)))

                    # Compute bin-KL (empirical || target)
                    ulas_binKL = kl_pq(p_ulas, target_probs)
                    psgla_binKL = kl_pq(p_psgla, target_probs)
                    dcla_binKL = kl_pq(p_dcla, target_probs)
                    ulans_binKL = kl_pq(p_ulans, target_probs)
                    ulas_binKL_list.append(ulas_binKL)
                    psgla_binKL_list.append(psgla_binKL)
                    dcla_binKL_list.append(dcla_binKL)
                    ulans_binKL_list.append(ulans_binKL)

                    # print results
                    header = f"[bin-KL @ NBINS={NBINS}, mu={mu_x}, Sigma12={Sigma_x[0,1]:.2f}, tau={tau}]"
                    print("\n" + "=" * len(header))
                    print(header)
                    print("=" * len(header))
                    results = {
                        "Moreau ULA": ulas_binKL,
                        "PSGLA": psgla_binKL,
                        "DC-LA": dcla_binKL,
                        "ULA": ulans_binKL
                    }
                    for name in ["ULA", "PSGLA", "DC-LA", "Moreau ULA"]:
                        print(f"{name:6s}  KL(emp||tar) = {results[name]:.6f}")
                    print("")
                
                # plot
                plt.figure(figsize=(8,6))
                plt.plot(NBINS_LIST, ulans_binKL_list, marker='d', label='ULA')
                plt.plot(NBINS_LIST, ulas_binKL_list, marker='o', label='Moreau ULA')
                plt.plot(NBINS_LIST, psgla_binKL_list, marker='^', label='PSGLA')
                plt.plot(NBINS_LIST, dcla_binKL_list, marker='s', label='DC-LA')
                plt.xlabel('Number of Bins per Dimension')
                plt.ylabel('Bin KL divergence')
                plt.legend()
                #plt.title(f'Bin KL Divergence vs Number of Bins (mu={mu_x}, Sigma12={Sigma_x[0,1]:.2f}, tau={tau})', fontsize=16)
                plt.grid(True, which='both', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f"synth/l12/exp2/binKL_mu{mu_x}_Sigma{Sigma_x[0,1]}_tau{tau}.png", dpi=600)
                plt.close()
if __name__ == "__main__":
    main()