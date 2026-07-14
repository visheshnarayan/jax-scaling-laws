import sys 
from pathlib import Path 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RESULT_DIR = Path(__file__).resolve().parent.parent / "results"
PLOT_DIR = RESULT_DIR / "plots"


def chinchilla_law(X, E, A, alpha, B, beta):
    """L(N, D) = E + A/N^alpha + B/D^beta"""
    N, D = X
    return E + A / (N ** alpha) + B / (D ** beta)


def plot_loss_curves(results):
    """Plot training loss curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        steps = [entry["step"] for entry in data["log"]]
        val_losses = [entry["val_loss"] for entry in data["log"]]
        ax.plot(steps, val_losses, label=f'{name} ({data["n_params"]:,} params)')
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Loss Curves Across Model Family")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(PLOT_DIR / "loss_curves.png", dpi=150, bbox_inches="tight")
    print(f"Saved loss_curves.png")


def plot_scaling_law(N, D, L, names, coeffs):
    """Plot loss vs params on log-log axes with fitted curve."""
    E, A, alpha, B, beta = coeffs["E"], coeffs["A"], coeffs["alpha"], coeffs["B"], coeffs["beta"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Loss vs N (params) on log-log
    ax1.scatter(N, L, s=80, zorder=5, color="tab:blue")
    for i, name in enumerate(names):
        ax1.annotate(name, (N[i], L[i]), textcoords="offset points", xytext=(8, 4), fontsize=9)

    N_smooth = np.logspace(np.log10(N.min() * 0.5), np.log10(N.max() * 2), 200)
    D_ratio = D[0] / N[0]  # keep same tokens/param ratio
    L_pred = chinchilla_law((N_smooth, N_smooth * D_ratio), E, A, alpha, B, beta)
    ax1.plot(N_smooth, L_pred, "--", color="tab:red", label="Fitted law")
    ax1.set_xscale("log")
    ax1.set_xlabel("Parameters (N)")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Scaling Law: Loss vs Parameters")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss vs Tokens on log-log
    ax2.scatter(D, L, s=80, zorder=5, color="tab:green")
    for i, name in enumerate(names):
        ax2.annotate(name, (D[i], L[i]), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax2.set_xscale("log")
    ax2.set_xlabel("Training Tokens (D)")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Scaling Law: Loss vs Tokens")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"L(N,D) = {E:.3f} + {A:.1f}/N^{alpha:.3f} + {B:.1f}/D^{beta:.3f}", fontsize=11, y=1.02)
    fig.savefig(PLOT_DIR / "scaling_law.png", dpi=150, bbox_inches="tight")
    print(f"Saved scaling_law.png")


def plot_compute_optimal(N, D, L, coeffs):
    """Plot compute-optimal frontier: for a given FLOP budget, what's the best N/D split."""
    E, A, alpha, B, beta = coeffs["E"], coeffs["A"], coeffs["alpha"], coeffs["B"], coeffs["beta"]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Compute C = 6*N*D (approximate FLOPs)
    C = 6 * N * D
    ax.scatter(C, L, s=80, zorder=5, color="tab:purple")

    C_smooth = np.logspace(np.log10(C.min() * 0.5), np.log10(C.max() * 2), 200)
    # For each C, find optimal N: minimize L(N, C/(6N))
    opt_losses = []
    opt_N = []
    for c in C_smooth:
        N_candidates = np.logspace(4, 9, 1000)
        D_candidates = c / (6 * N_candidates)
        losses = chinchilla_law((N_candidates, D_candidates), E, A, alpha, B, beta)
        best_idx = np.argmin(losses)
        opt_losses.append(losses[best_idx])
        opt_N.append(N_candidates[best_idx])
    ax.plot(C_smooth, opt_losses, "--", color="tab:red", label="Compute-optimal frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Compute (FLOPs, approx 6ND)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Compute-Optimal Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(PLOT_DIR / "compute_optimal.png", dpi=150, bbox_inches="tight")
    print(f"Saved compute_optimal.png")


def fit_scaling_law():
    """Fit Chinchilla parametric form to sweep results and generate plots."""
    with open(RESULT_DIR / "sweep_results.json") as f:
        results = json.load(f)

    names = list(results.keys())
    N = np.array([results[n]["n_params"] for n in names], dtype=np.float64)
    D = np.array([results[n]["tokens"] for n in names], dtype=np.float64)
    L = np.array([results[n]["final_val_loss"] for n in names], dtype=np.float64)

    # Fit
    p0 = [2.0, 100.0, 0.4, 100.0, 0.4]
    bounds = ([0, 0, 0.01, 0, 0.01], [10, 1e6, 2.0, 1e6, 2.0])
    popt, pcov = curve_fit(chinchilla_law, (N, D), L, p0=p0, bounds=bounds, maxfev=10000)

    E, A, alpha, B, beta = popt
    print(f"Fitted scaling law:")
    print(f"    L(N,D) = {E:.4f} + {A:.2f}/N^{alpha:.4f} + {B:.2f}/D^{beta:.4f}")
    print(f"\nExponents: alpha={alpha:.4f}, beta={beta:.4f}")

    coeffs = {"E": E, "A": A, "alpha": alpha, "B": B, "beta": beta}
    with open(RESULT_DIR / "scaling_law_coeffs.json", "w") as f:
        json.dump(coeffs, f, indent=2)

    # Generate plots
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(results)
    plot_scaling_law(N, D, L, names, coeffs)
    plot_compute_optimal(N, D, L, coeffs)

    return coeffs


if __name__ == "__main__":
    fit_scaling_law()