import jax 
import jax.numpy as jnp 
import numpy as np 
import optax
from pathlib import Path

from model.transformer import Transformer, TransformerConfig
from model.init import count_params
from data.loader import load_shard, get_batch
from train.train_step import make_train_step, make_eval_step


def create_optimizer(config: TransformerConfig, total_steps: int): 
    """AdamW with cosine decay + warmup"""
    warmup_steps = int(total_steps * 0.05) # 5% warmup 
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, 
        peak_value=3e-4, 
        warmup_steps=warmup_steps,
        decay_steps=total_steps, 
        end_value=3e-5,
    )
    return optax.adamw(learning_rate=schedule, weight_decay=0.1)

def train(config: TransformerConfig, token_budget: int, batch_size: int=64, eval_every: int = 100, seed: int = 42):
    """Train single model to give token budget
    
    Args: 
        config: model config 
        token_budget: total tokens to train on ( e.g. 20 * n_params ) 
        batch_size: sequences per step 
        eval_every: steps between val loss evaluations
    """
    model = Transformer(config)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    dummy = jnp.ones((1, config.block_size), dtype=jnp.int32)
    params = model.init(key, dummy)

    # toks per step batch * block size 
    tokens_per_step = batch_size * config.block_size
    total_steps = token_budget // tokens_per_step

    # optimizer 
    tx = create_optimizer(config, total_steps)
    opt_state = tx.init(params) 

    train_tokens = load_shard("train")
    val_tokens = load_shard("val")

    # create jitted step functions (closed over model and optimizer)
    train_step = make_train_step(model, tx)
    eval_step = make_eval_step(model)

    # training loop
    n_params = count_params(config)
    print(f"Training {n_params:,} param model for {total_steps:,} steps ({token_budget:,} tokens)")

    log = []
    for step in range(total_steps):
        # get batch
        x, y = get_batch(train_tokens, batch_size, rng, config.block_size)
        x, y = jnp.array(x), jnp.array(y)

        # train step
        params, opt_state, loss = train_step(params, x, y, opt_state)

        tokens_seen = (step + 1) * tokens_per_step

        if step%eval_every==0 or step == total_steps - 1:
            vx, vy = get_batch(val_tokens, batch_size, rng, config.block_size)
            vx, vy = jnp.array(vx), jnp.array(vy)
            val_loss = eval_step(params, vx, vy)

            print(f"    step {step:>6d} | train_loss {loss:.4f} | val_loss {val_loss:.4f} | tokens {tokens_seen:,}")
            log.append({"step":step, "tokens_seen":tokens_seen, "train_loss": float(loss), "val_loss": float(val_loss)})

    return params, log 