import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json 
import numpy as np 

from sweep.model_family import MODEL_FAMILY
from model.init import count_params
from train.run_training import train 

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def pick_batch_size(d_model: int) -> int:
    """Scale batch size down for larger models to fit in GPU memory."""
    if d_model <= 128:
        return 32
    elif d_model <= 256:
        return 16
    elif d_model <= 512:
        return 8
    else:
        return 4


def run_sweep(tokens_per_param: int = 20):
    """Train all models in the family and save loss logs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results to skip completed models
    results_path = RESULTS_DIR / "sweep_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded existing results for: {list(results.keys())}")
    else:
        results = {}

    for name, cfg in MODEL_FAMILY.items():
        if name in results:
            print(f"\nSkipping {name} (already completed, val_loss={results[name]['final_val_loss']:.4f})")
            continue

        n_params = count_params(cfg)
        token_budget = tokens_per_param * n_params
        batch_size = pick_batch_size(cfg.d_model)

        print(f"\n{'-'*60}")
        print(f"Training model: {name} ({n_params:,} params, {token_budget:,} tokens, bs={batch_size})")
        print(f"{'-'*60}")

        params, log = train(cfg, token_budget, batch_size, model_name=name)

        final_val_loss = log[-1]["val_loss"]
        results[name] = {
            "n_params": n_params,
            "tokens": token_budget,
            "final_val_loss": final_val_loss,
            "log": log,
        }

        # Save after each model completes
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f" -> final val loss: {final_val_loss:.4f}")

    return results


if __name__ == "__main__":
    run_sweep()
