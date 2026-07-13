from pathlib import Path
import numpy as np 

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

def load_shard(split: str="train"): 
    path = CACHE_DIR / f"{split}.npy"
    return np.load(path, mmap_mode="r") # open in reach (no mem over head for copy on write)


def get_batch(tokens, bactch_size, rng, block_size):
    offsets = rng.integers(low=0, high=len(tokens) - block_size, size=bactch_size)
    # numpy stores as uint16 but jax needs int32
    x = np.stack([tokens[i : i + block_size].astype(np.int32) for i in offsets])
    y = np.stack([tokens[i + 1: i + 1 + block_size].astype(np.int32) for i in offsets])
    return x, y
    
