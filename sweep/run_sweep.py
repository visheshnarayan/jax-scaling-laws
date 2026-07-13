import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json 
import numpy as np 

from sweep.model_family import MODEL_FAMILY
from model.init import count_params
from train.run_training import train 

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def run_sweep(tokens_per_param: int = 20, batch_size: int = 64):
    """Train all models in the family and save loss logs"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {} 
    for name, cfg in MODEL_FAMILY.items(): 
        n_params = count_params(cfg)
        token_budget = tokens_per_param * n_params

        print(f"\n{'-'*60}")
        print(f"Training model: {name} ({n_params:,} params, {token_budget:,} tokens)")
        print(f"\n{'-'*60}")

        params, log = train(cfg, token_budget, batch_size)

        #record final val loss for scaling law fit 
        final_val_loss = log[-1]["val_loss"]
        results[name] = {
            "n_params": n_params, 
            "tokens": token_budget, 
            "final_val_loss": final_val_loss, 
            "log": log, 
        }

        # save incrementally 
        with open(RESULTS_DIR / "sweep_results.json", "w") as f: 
            json.dump(results, f, indent=2)
        print(f" -> final val loss: {final_val_loss:.4f}")
    
    return results 

if __name__ == "__main__":
    run_sweep()

    