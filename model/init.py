import jax 
import jax.numpy as jnp 
from model.transformer import Transformer, TransformerConfig

def count_params(config: TransformerConfig) -> int: 
    """Count total params by initializing model with dummy input"""
    model = Transformer(config)
    key = jax.random.PRNGKey(0) # init key at pass for randomization -> jax is stateless, rng gen is required at each passing 
    dummy = jnp.ones((1, config.block_size), dtype=jnp.int32)
    params = model.init(key, dummy)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    return n_params