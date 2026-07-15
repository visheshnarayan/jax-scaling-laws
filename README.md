# JAX Scaling Laws

Replicating Chinchilla-style compute-optimal scaling laws from scratch in pure JAX/Flax. Trains a family of decoder-only transformers (3M–110M parameters), fits the parametric scaling law `L(N,D) = E + A/N^α + B/D^β`, then predicts the validation loss of a held-out 150M-parameter model before training it.

Built as a from-scratch exercise in scaling-law literacy: the entire stack — data pipeline, model, training loop, hyperparameter sweep, curve fitting, and prediction — is written in JAX with no PyTorch dependencies.

## Results

### Fitted Scaling Law

```
L(N, D) = 2.0279 + 25745.32/N^0.5598 + 5104.58/D^0.5595
```

| Coefficient | Value | Interpretation |
|---|---|---|
| E (irreducible loss) | 2.0279 | Entropy floor of the data |
| A | 25,745.32 | Parameter-scaling prefactor |
| α | 0.5598 | Parameter-scaling exponent |
| B | 5,104.58 | Data-scaling prefactor |
| β | 0.5595 | Data-scaling exponent |

The near-equal exponents (α ≈ β ≈ 0.56) suggest parameters and data contribute roughly symmetrically to loss reduction at this scale — consistent with Chinchilla's core finding that models and data should be scaled in tandem.

![Scaling Law: Loss vs Parameters and Tokens](results/plots/scaling_law.png)

### Sweep Results

| Config | Layers | d_model | Heads | d_ff | Params | Tokens | Val Loss |
|--------|--------|---------|-------|------|--------|--------|----------|
| 3M | 2 | 64 | 2 | 256 | 6,598,528 | 131.9M | 6.0118 |
| 6M | 4 | 96 | 3 | 384 | 10,195,200 | 203.9M | 5.2941 |
| 10M | 4 | 128 | 4 | 512 | 13,790,208 | 275.8M | 4.8865 |
| 17M | 6 | 192 | 6 | 768 | 22,164,864 | 443.3M | 4.0302 |
| 30M | 8 | 256 | 8 | 1024 | 32,312,320 | 646.2M | 3.3762 |
| 50M | 8 | 384 | 6 | 1536 | 53,187,072 | 1.06B | 3.4718 |
| 80M | 10 | 512 | 8 | 2048 | 83,512,320 | 1.67B | 3.0712 |
| 110M | 12 | 640 | 10 | 2560 | 124,067,840 | 2.48B | 2.7787 |
| **150M** (held-out) | 12 | 768 | 12 | 3072 | ~150M | — | pending |

Each model is trained on `20 × N` tokens following the Chinchilla compute allocation heuristic.

### Loss Curves

![Loss curves across model family](results/plots/loss_curves.png)

### Compute-Optimal Frontier

![Compute-optimal frontier](results/plots/compute_optimal.png)

For a given compute budget C ≈ 6ND (FLOPs), the fitted law yields an optimal parameter count N\* and token count D\* that minimizes loss. This is the core finding of Hoffmann et al. (2022) — larger models trained on proportionally more data outperform smaller models trained longer.

### Held-Out Prediction (150M)

| Metric | Parameters | Tokens | Val Loss |
|---|---|---|---|
| **Predicted** | ~150M | — | — |
| **Actual** | ~150M | — | — |
| **Error** | | | —% |

Pending — run `python -m analysis.predict_and_verify` to train the 150M model and fill in these values.

## Design Choices

### Why JAX/Flax/Optax

The entire training stack uses the JAX ecosystem — no PyTorch. This was a deliberate choice:

- **JAX** — functional transformations (`jit`, `grad`, `vmap`) compose cleanly, making the training step a pure function. This matters for scaling work where you need to reason precisely about compute.
- **Flax Linen** — provides the module abstraction (`nn.compact`) without hiding the parameter tree. Model params are explicit pytrees, which makes checkpointing and param counting trivial (`jax.tree.leaves`).
- **Optax** — composable optimizer + schedule. We use `adamw` with `warmup_cosine_decay_schedule`, matching the Chinchilla training recipe.

### Model Architecture

GPT-2-style decoder-only transformer with choices that keep the scaling signal clean:

- **Pre-LayerNorm** — LayerNorm before attention and MLP (more stable training than post-LN, especially at smaller scales)
- **No dropout** (`dropout=0.0`) — dropout adds noise to the loss signal; for scaling law fitting we want the cleanest possible loss curves
- **Fused QKV projection** — single `Dense(3 * d_model)` then split, standard efficient pattern
- **GELU activation** in MLP — matches GPT-2/Chinchilla
- **Learned positional embeddings** — simple and sufficient at these context lengths (1024)
- **No weight tying** — embedding and output projection are separate

### Training Optimization

- **AdamW** with weight decay 0.1 — standard for transformer pretraining
- **Cosine decay with warmup** — 5% warmup steps, decay from 3e-4 to 3e-5. Consistent schedule across all model sizes to avoid LR being a confound in the scaling fit.
- **Closure-based JIT** — `make_train_step(model, tx)` returns a `@jax.jit`-compiled closure. This avoids passing unhashable Flax modules as JIT arguments while keeping the train step a pure function.
- **`jax.value_and_grad`** — single forward+backward pass returns both loss and gradients
- **Batch size scaling** — smaller batch sizes for larger models (32 → 4) to avoid OOM on consumer GPUs

### Data Pipeline

- **WikiText-103** tokenized with GPT-2 BPE via `tiktoken` (50,257 vocab)
- **Flat `.npy` shards** with `mmap_mode="r"` — zero-copy reads, no memory overhead for the full dataset
- **Block-aligned train/val split** — 95/5 split aligned to 1024-token block boundaries
- **Random batch sampling** — offsets drawn uniformly from the token array; no epoch structure (standard for LM pretraining at these scales)

### Scaling Law Fit

- **Chinchilla parametric form:** `L(N, D) = E + A/N^α + B/D^β`
- **`scipy.optimize.curve_fit`** with bounded parameters — E ∈ [0, 10], exponents ∈ [0.01, 2.0]
- **Compute-optimal frontier** derived by grid search: for each FLOP budget C, sweep N candidates and find argmin L(N, C/6N)

### Robustness

- **Checkpointing every 500 steps** — full training state (params, optimizer state, RNG, log) serialized via pickle. Supports resume on crash or Colab disconnects.
- **Incremental sweep results** — `sweep_results.json` is saved after each model completes, so partial sweeps are not lost
- **Deterministic RNG** — NumPy `default_rng` with fixed seed, state saved/restored in checkpoints for exact reproducibility

## Project Structure

```
scaling-laws/
├── data/
│   ├── prepare_data.py       # download WikiText-103, tokenize with GPT-2 BPE, save as .npy
│   └── loader.py             # mmap loader + random batch sampler
├── model/
│   ├── transformer.py        # decoder-only transformer in Flax Linen
│   └── init.py               # exact param counting via dummy forward pass
├── train/
│   ├── train_step.py         # jitted train/eval steps (cross-entropy loss)
│   └── run_training.py       # training loop, AdamW + cosine schedule, checkpointing
├── sweep/
│   ├── model_family.py       # 8 model configs (3M–110M) + held-out 150M
│   └── run_sweep.py          # run full sweep, save results incrementally
├── analysis/
│   ├── fit_scaling_law.py    # fit Chinchilla L(N,D), generate scaling plots
│   └── predict_and_verify.py # predict held-out 150M loss, train, compare
└── results/
    ├── sweep_results.json    # per-model loss logs and final val losses
    ├── scaling_law_coeffs.json  # fitted {E, A, α, B, β}
    └── plots/
        ├── loss_curves.png
        ├── scaling_law.png
        └── compute_optimal.png
```

## Reproducing

```bash
pip install jax[cuda12] flax optax tiktoken datasets scipy numpy matplotlib

# 1. Prepare data (downloads WikiText-103, tokenizes, saves to cache/)
python -m data.prepare_data

# 2. Train all 8 models (saves results incrementally, supports resume)
python -m sweep.run_sweep

# 3. Fit scaling law and generate plots
python -m analysis.fit_scaling_law

# 4. Predict 150M loss and verify by training
python -m analysis.predict_and_verify
```

For CPU-only or Colab, replace `jax[cuda12]` with `jax`.

## References

- Hoffmann et al. 2022, [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)
- Kaplan et al. 2020, [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Scaling Book](https://jax-ml.github.io/scaling-book/) — JAX-specific scaling guide
