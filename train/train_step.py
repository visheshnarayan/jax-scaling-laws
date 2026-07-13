import functools

import jax
import jax.numpy as jnp
import optax


def cross_entropy_loss(logits, targets):
    """Standard next-token prediction loss"""
    # logits: (B, T, vocab_size), targets: (B, T)
    # flatten to (B * T, vocab_size) and (B*T,)
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return loss.mean()


@functools.partial(jax.jit, static_argnums=(4, 5))
def train_step(params, x, y, opt_state, model, tx):
    """Single training step: forward, loss, grad, update"""

    def loss_fn(params):
        logits = model.apply(params, x)
        return cross_entropy_loss(logits, y)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@functools.partial(jax.jit, static_argnums=(3,))
def eval_step(params, x, y, model):
    """Compute loss without grad (for validation)"""
    logits = model.apply(params, x)
    return cross_entropy_loss(logits, y)

