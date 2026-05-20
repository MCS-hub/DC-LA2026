import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from joblib import Parallel, delayed
import pickle
import os

from utils import l1, prox_l1, l2, prox_l2
from sampling_algs import DC_LA

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

# -----------------------------
# Utility: KL(emp || target)
# -----------------------------
EPS = 1e-12

def kl_pq(p, q, eps=EPS):
    p = p.ravel().astype(float)
    q = q.ravel().astype(float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def hist2d_probs(samples, x_edges, y_edges):
    H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[x_edges, y_edges])
    H = H.astype(float)
    H_sum = H.sum()
    if H_sum <= 0:
        return np.ones_like(H) / H.size
    return H / H_sum


def compute_target_bin_probs(pi_normalized, x_edges, y_edges, n_jobs=-1):
    NBINS_X = len(x_edges) - 1
    NBINS_Y = len(y_edges) - 1


    def integrate_bin(xa, xb, ya, yb):
        # dblquad integrates y first then x: integrand(y, x)
        val, err = dblquad(
            lambda y, x: pi_normalized(np.array([x, y])),
            xa, xb,
            lambda _: ya, lambda _: yb
        )
        return val

    bin_rects = []
    for i in range(NBINS_X):
        for j in range(NBINS_Y):
            xa, xb = x_edges[i], x_edges[i + 1]
            ya, yb = y_edges[j], y_edges[j + 1]
            bin_rects.append((xa, xb, ya, yb))


    probs_flat = Parallel(n_jobs=n_jobs)(
        delayed(integrate_bin)(xa, xb, ya, yb) for (xa, xb, ya, yb) in bin_rects
    )
    target_probs = np.array(probs_flat).reshape(NBINS_X, NBINS_Y)

    # Renormalize (numerical safety)
    target_probs = np.maximum(target_probs, 0.0)
    s = target_probs.sum()
    if s <= 0:
        raise RuntimeError("Target bin probability sum is non-positive. Check domain.")
    target_probs /= s
    return target_probs

# -----------------------------
# run DC-LA chains
# -----------------------------
def run_dcla_last_samples(d, n_samples, burn_in, lam, gamma, n_chains, grad_f, prox_r1, prox_r2, n_jobs=-1):
    def run_chain_once(_):
        X0 = np.random.randn(d)
        samples = DC_LA(X0, n_samples, burn_in, lam, gamma, d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)
        return samples[-1]

    results = Parallel(n_jobs=n_jobs)(delayed(run_chain_once)(i) for i in range(n_chains))
    return np.array(results)


# -----------------------------
# Main ablation
# -----------------------------
def main():
    os.makedirs("synth/l12/ablation", exist_ok=True)

    d = 2
    mu_x_list = [0, 1, 2, 3]
    Sigma_x_list = [
        np.array([[1, 0.0], [0.0, 1.0]]),
        np.array([[1, 0.8], [0.8, 1.0]]),
        np.array([[1, -0.8], [-0.8, 2.0]])
    ]
    tau_list = [10]

    # --- Ablation grid ---
    lam_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    gamma_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    # --- Sampling budget ---
    n_samples = 1000
    burn_in = 0
    n_chains = 3000      

    # --- Heatmap resolution ---
    NBINS = 50          
    # --- Domain estimation for bins (pilot) ---
    pilot_lam = 0.01
    pilot_gamma = 0.005
    pilot_chains = 3000

    # Storage for aggregate over targets
    all_targets_scores = []  # list of (len(gamma_list), len(lam_list)) arrays

    for tau in tau_list:
        for mu_x in mu_x_list:
            for Sigma_x in Sigma_x_list:
                print("\n===================================================")
                print("Target: mu_x:", mu_x, "Sigma_x:\n", Sigma_x, "tau:", tau)
                print("===================================================")

                # --- Define objective pieces ---
                def r1(x): return tau * l1(x)
                def r2(x): return tau * l2(x)

                def prox_r1(x, alpha): return prox_l1(x, tau * alpha)
                def prox_r2(x, alpha): return prox_l2(x, tau * alpha)

                def f(x): return 0.5 * (x - mu_x).T @ Sigma_x @ (x - mu_x)
                def grad_f(x): return Sigma_x @ (x - mu_x)
                def V(x): return f(x) + r1(x) - r2(x)

                # --- Normalize constant ---
                def pi_unnormalized_2d(x1, x2):
                    x = np.array([x1, x2])
                    return np.exp(-V(x))

                valZ, errZ = dblquad(
                    lambda x2, x1: pi_unnormalized_2d(x1, x2),
                    -np.inf, np.inf,
                    lambda _: -np.inf, lambda _: np.inf
                )
                print(f"Z (integral): {valZ:.6f} ± {errZ:.4e}")

                def pi_normalized(x):
                    return np.exp(-V(x)) / valZ

                # --- Pilot run to set bin domain ---
                print(f"Pilot to set domain: lam={pilot_lam}, gamma={pilot_gamma}, chains={pilot_chains}")
                pilot_samples = run_dcla_last_samples(
                    d, n_samples, burn_in, pilot_lam, pilot_gamma, pilot_chains,
                    grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2, n_jobs=-1
                )
                

                x_q01, x_q99 = np.quantile(pilot_samples[:, 0], [0.01, 0.99])
                y_q01, y_q99 = np.quantile(pilot_samples[:, 1], [0.01, 0.99])
                pad_x = 0.10 * (x_q99 - x_q01 + EPS)
                pad_y = 0.10 * (y_q99 - y_q01 + EPS)
                x_min, x_max = x_q01 - pad_x, x_q99 + pad_x
                y_min, y_max = y_q01 - pad_y, y_q99 + pad_y

                x_edges = np.linspace(x_min, x_max, NBINS + 1)
                y_edges = np.linspace(y_min, y_max, NBINS + 1)

                # --- Precompute target bin probabilities ONCE ---
                print(f"Precomputing target bin probs for NBINS={NBINS}...")
                target_probs = compute_target_bin_probs(pi_normalized, x_edges, y_edges, n_jobs=-1)

                # --- Run ablation grid: compute Bin-KL for DC-LA ---
                scores = np.full((len(gamma_list), len(lam_list)), np.nan, dtype=float)

                for gi, gamma in enumerate(gamma_list):
                    for li, lam in enumerate(lam_list):
                        print(f"Running DC-LA: gamma={gamma:.3e}, lam={lam:.3e}, chains={n_chains}")
                        samples = run_dcla_last_samples(
                            d, n_samples, burn_in, lam, gamma, n_chains,
                            grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2, n_jobs=-1
                        )
                        

                        p_emp = hist2d_probs(samples, x_edges, y_edges)
                        binKL = kl_pq(p_emp, target_probs)
                        scores[gi, li] = binKL
                        print(f"  Bin-KL(emp||tar) = {binKL:.6f}")

                # Save per-target results
                tag = f"mu{mu_x}_S12{Sigma_x[0,1]:+.2f}_S22{Sigma_x[1,1]:.2f}_tau{tau}"
                out_pkl = f"synth/l12/ablation/ablation_{tag}.pkl"
                with open(out_pkl, "wb") as fpk:
                    pickle.dump({
                        "mu_x": mu_x,
                        "Sigma_x": Sigma_x,
                        "tau": tau,
                        "gamma_list": gamma_list,
                        "lam_list": lam_list,
                        "NBINS": NBINS,
                        "domain": (x_min, x_max, y_min, y_max),
                        "scores_binKL": scores,
                    }, fpk)
                print("Saved:", out_pkl)

                # Plot per-target heatmap
                plt.figure(figsize=(8, 6))
                # imshow expects [rows, cols] => [gamma, lambda]
                plt.imshow(scores, aspect="auto", origin="lower")
                plt.colorbar(label="Bin-KL(emp||target)")
                plt.xticks(np.arange(len(lam_list)), [f"{x:.0e}" for x in lam_list], rotation=45)
                plt.yticks(np.arange(len(gamma_list)), [f"{g:.0e}" for g in gamma_list])
                plt.xlabel("$\\lambda$")
                plt.ylabel("$\\gamma$")
                #plt.title(f"DC-LA ablation (Bin-KL)  {tag}")
                plt.tight_layout()
                out_png = f"synth/l12/ablation/heatmap_{tag}.png"
                plt.savefig(out_png, dpi=300)
                plt.close()
                print("Saved:", out_png)

                all_targets_scores.append(scores)

    # --- Aggregate across targets (median heatmap) ---
    if len(all_targets_scores) > 0:
        stack = np.stack(all_targets_scores, axis=0)  
        median_scores = np.median(stack, axis=0)
        q25 = np.quantile(stack, 0.25, axis=0)
        q75 = np.quantile(stack, 0.75, axis=0)
        iqr_scores = q75 - q25

        with open("synth/l12/ablation/ablation_AGG.pkl", "wb") as fpk:
            pickle.dump({
                "gamma_list": gamma_list,
                "lam_list": lam_list,
                "median_scores": median_scores,
                "iqr_scores": iqr_scores,
            }, fpk)

        plt.figure(figsize=(8, 6))
        plt.imshow(median_scores, aspect="auto", origin="lower")
        plt.colorbar(label="Median Bin-KL across targets")
        plt.xticks(np.arange(len(lam_list)), [f"{x:.0e}" for x in lam_list], rotation=45)
        plt.yticks(np.arange(len(gamma_list)), [f"{g:.0e}" for g in gamma_list])
        plt.xlabel("$\\lambda$")
        plt.ylabel("$\\gamma$")
        #plt.title("DC-LA ablation (median Bin-KL across targets)")
        plt.tight_layout()
        plt.savefig("synth/l12/ablation/heatmap_AGG_median.png", dpi=600)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.imshow(iqr_scores, aspect="auto", origin="lower")
        plt.colorbar(label="IQR Bin-KL across targets")
        plt.xticks(np.arange(len(lam_list)), [f"{x:.0e}" for x in lam_list], rotation=45)
        plt.yticks(np.arange(len(gamma_list)), [f"{g:.0e}" for g in gamma_list])
        plt.xlabel("$\\lambda$")
        plt.ylabel("$\\gamma$")
        #plt.title("DC-LA ablation variability (IQR across targets)")
        plt.tight_layout()
        plt.savefig("synth/l12/ablation/heatmap_AGG_iqr.png", dpi=600)
        plt.close()

        print("Saved aggregate heatmaps to synth/l12/ablation/")

if __name__ == "__main__":
    main()
