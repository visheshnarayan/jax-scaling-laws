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


def _num_devices():
    return jax.local_device_count()


def make_grad_step(model):
    """Create a jitted function that returns loss and grads for one microbatch."""
    @jax.jit
    def grad_step(params, x, y):
        def loss_fn(params):
            logits = model.apply(params, x)
            return cross_entropy_loss(logits, y)
        return jax.value_and_grad(loss_fn)(params)
    return grad_step


def make_train_step(model, tx):
    """Create a train step: pmap across devices if >1 available, otherwise jit."""
    n_devices = _num_devices()

    if n_devices > 1:
        def _train_step(params, x, y, opt_state):
            def loss_fn(params):
                logits = model.apply(params, x)
                return cross_entropy_loss(logits, y)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            loss = jax.lax.pmean(loss, axis_name='batch')
            updates, opt_state_new = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state_new, loss

        train_step = jax.pmap(_train_step, axis_name='batch')
    else:
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


def make_update_step(tx):
    """Create a jitted optimizer update step."""
    @jax.jit
    def update_step(params, grads, opt_state):
        updates, opt_state_new = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state_new
    return update_step


def make_eval_step(model):
    """Create an eval step: pmap if >1 device, otherwise jit."""
    n_devices = _num_devices()

    if n_devices > 1:
        def _eval_step(params, x, y):
            logits = model.apply(params, x)
            loss = cross_entropy_loss(logits, y)
            return jax.lax.pmean(loss, axis_name='batch')

        eval_step = jax.pmap(_eval_step, axis_name='batch')
    else:
        @jax.jit
        def eval_step(params, x, y):
            logits = model.apply(params, x)
            return cross_entropy_loss(logits, y)

    return eval_step
