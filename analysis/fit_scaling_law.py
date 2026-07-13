import sys 
from pathlib import Path 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json 
import numpy as np 
from scipy.optimize import curve_fit

RESULT_DIR = Path(__file__).resolve().parent.parent / "results"

def chinchila_law(X, E, A, alpha, B, beta): 
    """L(N, D) = E + A/N^alpha + B/D^alpha"""
    N, D = X
    return E + A / (N**alpha) + B / (D**beta)

def fit_scaling_law(): 
    """Fit Chinchila parametric form to sweep results"""
    with open(RESULT_DIR / "sweep_results.json") as f: 
        results = json.load(f)

    # extract data points 
    N_list = [] 
    D_list = [] 
    L_list = [] 

    for name, data in results.items(): 
        N_list.append(data["n_params"])
        D_list.append(data["tokens"])
        L_list.append(data["final_val_loss"])

    N = np.array(N_list, dtype=np.float64)
    D = np.array(D_list, dtype=np.float64)
    L = np.array(L_list, dtype=np.float64)

    # fit into log space 
    p0 = [2.0, 100.0, 0.4, 100.0, 0.4]
    bounds = ([0, 0, 0.01, 0, 0.01], [10, 1e6, 2.0, 1e6, 2.0])

    popt, pcov = curve_fit(chinchila_law, (N, D), L, p0=p0, bounds=bounds, maxfev=10000)

    E, A, alpha, B, beta = popt
    print(f"Fitted scaling law:")
    print(f"    L(N,D) = {E:.4f} + {A:.2f}/N^{alpha:.4f} + {B:.2f}/D^{beta:.4f}")
    print(f"\nExponents: alpha={alpha:.4f}, beta={beta:.4f}")

    coeffs = {"E": E, "A": A, "alpha":alpha, "B":B, "beta":beta}
    with open(RESULT_DIR / "scaling_law_coeffs.json", "w") as f: 
        json.dump(coeffs, f, indent=2)

if __name__ == "__main__":
    fit_scaling_law()