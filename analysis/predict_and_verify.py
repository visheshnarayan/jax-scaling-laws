import sys 
from pathlib import Path 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json 
import numpy as np 

from sweep.model_family import HELD_OUT
from model.init import count_params
from train.run_training import train 

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def predict_and_verify(tokens_per_param: int = 20): 
    # load fitted law 
    with open(RESULTS_DIR / "scaling_law_coeffs.json") as f: 
        coeffs = json.load(f)
    
    E, A, alpha, B, beta = coeffs["E"], coeffs["A"], coeffs["alpha"], coeffs["B"], coeffs["beta"]

    # predict loss for 150m model 
    cfg = HELD_OUT["150M"]
    n_params = count_params(cfg)
    token_budget = tokens_per_param * n_params

    predicted_loss = E + A / (n_params ** alpha) + B / (token_budget ** beta)
    print(f"Predicted val loss for 150M model: {predicted_loss:.4f}")
    print(f"    (N={n_params:,}, D={token_budget:,})")

    # train and verify loss 
    print(f"\nTraining 150M model to verify...")
    params, log = train(cfg, token_budget, batch_size=32, use_bf16=True)
    actual_loss = log[-1]["val_loss"]

    pct_error = abs(predicted_loss - actual_loss) / actual_loss * 100
    print(f"\n{'-'*60}")
    print(f"Predicted:  {predicted_loss:.4f}")
    print(f"Actual:     {actual_loss:.4f}")
    print(f"Error:      {pct_error:.2f}%")
    print(f"{'-'*60}")

    result = {
        "predicted": predicted_loss, 
        "actual": actual_loss, 
        "pct_error": pct_error, 
        "n_params": n_params, 
        "tokens": token_budget
    }

    with open(RESULTS_DIR / "prediction_result.json", "w") as f: 
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    predict_and_verify()