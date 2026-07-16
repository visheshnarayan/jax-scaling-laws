import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path

from model.transformer import Transformer, TransformerConfig
from model.init import count_params
from data.loader import load_shard, get_batch
from train.train_step import make_train_step, make_eval_step, make_grad_step, make_update_step


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints"


def create_optimizer(config: TransformerConfig, total_steps: int):
    """AdamW with cosine decay + warmup."""
    warmup_steps = int(total_steps * 0.05)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=3e-5,
    )
    return optax.adamw(learning_rate=schedule, weight_decay=0.1)


def save_checkpoint(model_name, step, params, opt_state, log, rng_state):
    """Save training state to disk."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{model_name}.pkl"
    state = {
        "step": step,
        "params": jax.tree.map(np.array, params),
        "opt_state": jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, opt_state),
        "log": log,
        "rng_state": rng_state.__getstate__(),
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_checkpoint(model_name):
    """Load training state from disk. Returns None if no checkpoint."""
    path = CHECKPOINT_DIR / f"{model_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["params"] = jax.tree.map(jnp.array, state["params"])
    rng = np.random.default_rng()
    rng.__setstate__(state["rng_state"])
    state["rng"] = rng
    return state


def _replicate(tree):
    """Replicate a pytree across all local devices (for pmap)."""
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (jax.local_device_count(),) + x.shape), tree)


def _unreplicate(tree):
    """Extract single copy from a replicated pytree."""
    return jax.tree.map(lambda x: x[0], tree)


def train(config: TransformerConfig, token_budget: int, batch_size: int = 64,
          eval_every: int = 100, checkpoint_every: int = 500,
          seed: int = 42, model_name: str = "model", use_bf16: bool = False,
          grad_accum_steps: int = 1):
    """Train a single model with checkpointing, resume, and auto multi-GPU support.

    grad_accum_steps: number of microbatches to accumulate before updating.
    Effective batch = batch_size * grad_accum_steps (per device if multi-GPU).
    """
    n_devices = jax.local_device_count()
    multi_device = n_devices > 1

    effective_batch = batch_size * grad_accum_steps
    if multi_device:
        effective_batch = effective_batch * n_devices
        print(f"Multi-GPU: {n_devices} devices detected")

    print(f"Micro-batch: {batch_size}, grad accum: {grad_accum_steps}, effective batch: {effective_batch}")

    model = Transformer(config)

    tokens_per_step = effective_batch * config.block_size
    total_steps = token_budget // tokens_per_step

    tx = create_optimizer(config, total_steps)

    # Try to resume from checkpoint
    ckpt = load_checkpoint(model_name)
    if ckpt is not None:
        start_step = ckpt["step"] + 1
        params = ckpt["params"]
        opt_state = ckpt["opt_state"]
        log = ckpt["log"]
        rng = ckpt["rng"]
        print(f"Resumed from checkpoint at step {ckpt['step']}")
    else:
        start_step = 0
        rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed)
        dummy = jnp.ones((1, config.block_size), dtype=jnp.int32)
        params = model.init(key, dummy)
        opt_state = tx.init(params)
        log = []

    # cast params to bfloat16 for faster compute
    if use_bf16:
        params = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, params)
        print("Using bfloat16 mixed precision")

    train_tokens = load_shard("train")
    val_tokens = load_shard("val")

    n_params = count_params(config)
    print(f"Training {n_params:,} param model for {total_steps:,} steps ({token_budget:,} tokens)")
    if start_step > 0:
        print(f"  Starting from step {start_step}")

    if multi_device and grad_accum_steps == 1:
        # standard pmap path (no gradient accumulation)
        train_step = make_train_step(model, tx)
        eval_step = make_eval_step(model)
        params = _replicate(params)
        opt_state = _replicate(opt_state)

        for step in range(start_step, total_steps):
            x, y = get_batch(train_tokens, effective_batch, rng, config.block_size)
            x, y = jnp.array(x), jnp.array(y)
            x = x.reshape(n_devices, batch_size, -1)
            y = y.reshape(n_devices, batch_size, -1)

            params, opt_state, loss = train_step(params, x, y, opt_state)
            tokens_seen = (step + 1) * tokens_per_step

            if step % eval_every == 0 or step == total_steps - 1:
                vx, vy = get_batch(val_tokens, effective_batch, rng, config.block_size)
                vx, vy = jnp.array(vx), jnp.array(vy)
                vx = vx.reshape(n_devices, batch_size, -1)
                vy = vy.reshape(n_devices, batch_size, -1)
                val_loss = eval_step(params, vx, vy)
                loss_val = float(loss[0])
                vloss_val = float(val_loss[0])
                print(f"    step {step:>6d} | train_loss {loss_val:.4f} | val_loss {vloss_val:.4f} | tokens {tokens_seen:,}")
                log.append({"step": step, "tokens_seen": tokens_seen, "train_loss": loss_val, "val_loss": vloss_val})

            if step % checkpoint_every == 0 and step > 0:
                save_checkpoint(model_name, step, _unreplicate(params), _unreplicate(opt_state), log, rng)

        final_params = _unreplicate(params)
        final_opt = _unreplicate(opt_state)

    else:
        # gradient accumulation path (works with single or multi-GPU)
        grad_step = make_grad_step(model)
        update_step = make_update_step(tx)
        eval_step_fn = make_eval_step(model)

        for step in range(start_step, total_steps):
            # accumulate gradients over microbatches
            acc_grads = None
            acc_loss = 0.0

            for _ in range(grad_accum_steps):
                x, y = get_batch(train_tokens, batch_size, rng, config.block_size)
                x, y = jnp.array(x), jnp.array(y)
                loss, grads = grad_step(params, x, y)

                if acc_grads is None:
                    acc_grads = grads
                else:
                    acc_grads = jax.tree.map(lambda a, g: a + g, acc_grads, grads)
                acc_loss += float(loss)

            # average gradients
            acc_grads = jax.tree.map(lambda g: g / grad_accum_steps, acc_grads)
            acc_loss /= grad_accum_steps

            params, opt_state = update_step(params, acc_grads, opt_state)
            tokens_seen = (step + 1) * tokens_per_step

            if step % eval_every == 0 or step == total_steps - 1:
                vx, vy = get_batch(val_tokens, batch_size, rng, config.block_size)
                vx, vy = jnp.array(vx), jnp.array(vy)
                val_loss = float(eval_step_fn(params, vx, vy))
                print(f"    step {step:>6d} | train_loss {acc_loss:.4f} | val_loss {val_loss:.4f} | tokens {tokens_seen:,}")
                log.append({"step": step, "tokens_seen": tokens_seen, "train_loss": acc_loss, "val_loss": val_loss})

            if step % checkpoint_every == 0 and step > 0:
                save_checkpoint(model_name, step, params, opt_state, log, rng)

        final_params = params
        final_opt = opt_state

    # Final checkpoint
    save_checkpoint(model_name, total_steps - 1, final_params, final_opt, log, rng)

    return final_params, log
