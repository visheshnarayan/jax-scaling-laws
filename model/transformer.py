from dataclasses import dataclass
import jax 
import jax.numpy as jnp 
import flax.linen as nn 

@dataclass 
class TransformerConfig: 
    vocab_size: int = 50257 # gpt2 vocab size 
    block_size: int = 1024 
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.0 # no dropout for scaling law runs (cleaner signal) 

class CasualSelfAttention(nn.Module): 
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        cfg = self.config 
        B, T, C = x.shape 
        head_dim = cfg.d_model // cfg.n_heads # how many dims per head 

        # project to q,k,v 
        qkv = nn.Dense(3 * cfg.d_model)(x) 
        # q,k,v is stacked, split 
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # reshape to (B, n_heads, T, head_dim)
        q = q.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)

        # scale dot product attention with casual mask 
        scale = jnp.sqrt(head_dim).astype(x.dtype) # denom of attention under root 
        attn = (q @ k.transpose(0,1,3,2)) / scale  # matmul of the info from keys to take given our query -> which vec is maximizing answer space 

        mask = jnp.tril(jnp.ones((T,T)))[None, None, :, :]
        attn = jnp.where(mask == 0, -1e9, attn)
        attn = nn.softmax(attn, axis=-1)

        # apply attention to value vecs (where our info was maximized, extract scaled vals from value vecs)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)

        out = nn.Dense(cfg.d_model)(out)
        return out 
    

class MLP(nn.Module): 
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        cfg = self.config 
        x = nn.Dense(cfg.d_ff)(x)
        x = nn.gelu(x)
        x = nn.Dense(cfg.d_model)(x)
        return x 
    
class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        # pre ln architecture
        x = x + CasualSelfAttention(self.config)(nn.LayerNorm()(x), deterministic)
        x = x + MLP(self.config)(nn.LayerNorm()(x), deterministic)
        return x 
    
class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, idx, deterministic=True):
        cfg = self.config 
        B, T = idx.shape

        # token and positional embedding 
        tok_emb = nn.Embed(cfg.vocab_size, cfg.d_model)(idx)
        pos_emb = nn.Embed(cfg.block_size, cfg.d_model)(jnp.arange(T))
        x = tok_emb + pos_emb

        # transformer blocks 
        for _ in range(cfg.n_layers):
            x = TransformerBlock(cfg)(x, deterministic)

        # final layer norm + project to vocab
        x = nn.LayerNorm()(x)
        logits = nn.Dense(cfg.vocab_size, use_bias=False)(x)

        return logits 
