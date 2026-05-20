import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors
from scipy.integrate import dblquad
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

# single chain
def main():
    d = 2
    mu_x_list = [0, 1, 2, 3]
    Sigma_x_list = [np.array([[1,0],[0,1]]), np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 2]])]
    tau_list = [10, 15]

    lam, gamma = 0.01, 0.005
    n_samples = 10000
    burn_in = 500
    X0 = np.random.randn(d)

    for tau in tau_list:
        for mu_x in mu_x_list:
            for Sigma_x in Sigma_x_list:
                print("mu_x:", mu_x, "Sigma_x:", Sigma_x, "tau:", tau)
                figname = f"synth/l12/exp1/dcla_mu{mu_x}_Sigma{Sigma_x[0,1]}_tau{tau}.png"
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
                samples_ulas = ULA_s(X0, n_samples, burn_in, lam, gamma, d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)
                samples_dcla = DC_LA(X0, n_samples, burn_in, lam, gamma, d, grad_f=grad_f, prox_r1=prox_r1, prox_r2=prox_r2)
                samples_psgla = PSGLA(X0, n_samples, burn_in, gamma, d, grad_f=grad_f, prox_r=prox_r)
                samples_ulans = ULA_ns(X0, n_samples, burn_in, gamma, d, grad_f=grad_f, grad_r1=r1_grad, grad_r2=r2_grad)
                # target density
                def pi_unnormalized(x):
                    return np.exp(-V(x))

                def pi_unnormalized_2d(x1, x2):
                    x = np.array([x1, x2])
                    return pi_unnormalized(x)

                xlim1 = 3
                xlim2 = 3
                if mu_x > 1.5:
                    xlim1 = 1
                    xlim2 = 5

                # Integral 
                val, err = dblquad(lambda x2, x1: pi_unnormalized_2d(x1, x2),
                                -xlim1, xlim2,  # x1 range
                                lambda _: -xlim1, lambda _: xlim2)  # x2 range


                def pi_normalized(x):
                    return np.exp(-V(x))/val

                # plot
                # Meshgrid
                x1 = np.linspace(-xlim1, xlim2, 300)
                x2 = np.linspace(-xlim1, xlim2, 300)
                X1, X2 = np.meshgrid(x1, x2)
                f_vec = np.vectorize(lambda x, y: pi_normalized(np.array([x, y])))
                Z = f_vec(X1, X2)

            
                # Bins 
                xbin = np.linspace(-xlim1, xlim2, 50)
                ybin = np.linspace(-xlim1, xlim2, 50)

                # --- common vmin/vmax across all panels ---
                vmin = float(np.min(Z))
                vmax = float(np.max(Z))

                H_dcla, _, _ = np.histogram2d(samples_dcla[:,0], samples_dcla[:,1],
                                                bins=[xbin, ybin], density=True)
                H_ulas,  _,  _  = np.histogram2d(samples_ulas[:,0],  samples_ulas[:,1],
                                                bins=[xbin, ybin], density=True)
                H_psgla,_,  _  = np.histogram2d(samples_psgla[:,0], samples_psgla[:,1],
                                                bins=[xbin, ybin], density=True)
                H_ulans,_,  _  = np.histogram2d(samples_ulans[:,0], samples_ulans[:,1],
                                                bins=[xbin, ybin], density=True)
        
                vmin = min(vmin, H_dcla.min(), H_ulas.min(), H_psgla.min(), H_ulans.min())
                vmax = max(vmax, H_dcla.max(), H_ulas.max(), H_psgla.max(), H_ulans.max())
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = 'viridis'

                fig, axs = plt.subplots(2, 3, figsize=(12, 8))

                # --- Subplot 1: Contour ---
                #levels = np.linspace(vmin, vmax, 8)
                cs = axs[0,0].contourf(X1, X2, Z, norm=norm, cmap=cmap)  # levels enforce same scale
                axs[0,0].set_title("Target Distribution")
                #fig.colorbar(cs, ax=axs[0,0])

                # --- Subplot 2: DC-LA ---
                H, xedges, yedges, im1 = axs[0,1].hist2d(samples_dcla[:,0], samples_dcla[:,1],
                                                        bins=[xbin, ybin], density=True,
                                                        cmap=cmap, norm=norm)
                axs[0,1].set_title("DC-LA")
                #fig.colorbar(im1, ax=axs[0,1])

                # --- Subplot 3: PSGLA ---
                H, xedges, yedges, im3 = axs[0,2].hist2d(samples_psgla[:,0], samples_psgla[:,1],
                                                        bins=[xbin, ybin], density=True,
                                                        cmap=cmap, norm=norm)
                axs[0,2].set_title("PSGLA")

                # --- Subplot 4: ULA ---
                H, xedges, yedges, im2 = axs[1,0].hist2d(samples_ulas[:,0], samples_ulas[:,1],
                                                        bins=[xbin, ybin], density=True,
                                                        cmap=cmap, norm=norm)
                axs[1,0].set_title("Moreau ULA")
                #fig.colorbar(im2, ax=axs[1,0])
                # --- Subplot 5: ULA-NS ---
                H, xedges, yedges, im4 = axs[1,1].hist2d(samples_ulans[:,0], samples_ulans[:,1],
                                                        bins=[xbin, ybin], density=True,
                                                        cmap=cmap, norm=norm)
                axs[1,1].set_title("ULA")

                axs[1, 2].axis("off")

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im3, cax=cbar_ax)

                # plt.tight_layout()

                # Layout adjustments

                plt.savefig(figname, bbox_inches="tight", dpi=600)
                plt.close()


if __name__ == "__main__":
    main()