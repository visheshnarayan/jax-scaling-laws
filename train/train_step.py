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


def make_train_step(model, tx):
    """Create a jitted train step closed over model and optimizer."""

    @jax.jit
    def train_step(params, x, y, opt_state):
        def loss_fn(params):
            logits = model.apply(params, x)
            return cross_entropy_loss(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state_new, loss

    return train_step


def make_eval_step(model):
    """Create a jitted eval step closed over model."""

    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply(params, x)
        return cross_entropy_loss(logits, y)

    return eval_step

