#!/bin/bash
# Cloud GPU setup script for RunPod / Vast.ai / any SSH-accessible instance
# Usage: bash setup_cloud.sh

set -e

echo "=== Scaling Laws: Cloud Setup ==="

# install JAX with CUDA support + project deps
pip install -q "jax[cuda12]" flax optax tiktoken datasets scipy numpy matplotlib

# verify GPU setup
python -c "
import jax
devices = jax.devices()
print(f'Devices detected: {len(devices)}')
for d in devices:
    print(f'  {d}')
"

# prepare data
echo ""
echo "=== Preparing data ==="
python -m data.prepare_data

# run the 150M prediction + verification
echo ""
echo "=== Training 150M held-out model ==="
python -m analysis.predict_and_verify

# regenerate plots (in case we want updated ones)
echo ""
echo "=== Fitting scaling law + plots ==="
python -m analysis.fit_scaling_law

echo ""
echo "=== Done ==="
echo "Results saved to results/"
echo "  prediction_result.json  -> predicted vs actual loss"
echo "  scaling_law_coeffs.json -> fitted coefficients"
echo "  plots/                  -> all figures"
echo ""
echo "To download results to your local machine:"
echo "  scp -r root@<pod-ip>:~/jax-scaling-laws/results/ ./results/"
