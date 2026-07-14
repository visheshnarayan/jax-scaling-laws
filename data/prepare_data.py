"""
Data preparation pipeline for scaling-laws project.
Downloads OpenWebText subset from HuggingFace, tokenizes with GPT-2 BPE,
and stores as flat numpy arrays for fast random-access during training.
"""
from pathlib import Path

from datasets import load_dataset
import tiktoken
import numpy as np


CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
BLOCK_LENGTH = 1024
VAL_FRACTION = 0.05  # 5% held out for validation


def download_dataset():
    """Download WikiText-103 from HuggingFace."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    # Filter out empty lines
    ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
    return ds


def tokenize_and_concatenate(ds, enc) -> np.ndarray:
    """Tokenize all documents and concatenate into one flat token array."""
    all_tokens = []
    for row in ds:
        tokens = enc.encode(row["text"])
        all_tokens.extend(tokens)
    # uint16 is fine -> gpt2 vocab is 50257 (fits in uint16 max 65535)
    return np.array(all_tokens, dtype=np.uint16)


def prepare():
    """Main pipeline to download, tokenize, split, save data"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    print("Downloading dataset...")
    ds = download_dataset()

    print(f"Tokenizing {len(ds)} documents...")
    tokens = tokenize_and_concatenate(ds, enc)

    # Truncate to a multiple of block length (discard partial trailing block)
    n_tokens = (len(tokens) // BLOCK_LENGTH) * BLOCK_LENGTH
    tokens = tokens[:n_tokens]

    # Train/val split
    n_val = int(n_tokens * VAL_FRACTION)
    n_val = (n_val // BLOCK_LENGTH) * BLOCK_LENGTH  # align to block boundary
    n_train = n_tokens - n_val

    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:]

    # Save as flat .npy files — loader indexes into these at [i*1024 : (i+1)*1024]
    np.save(CACHE_DIR / "train.npy", train_tokens)
    np.save(CACHE_DIR / "val.npy", val_tokens)

    print(f"Done. {n_train:,} train tokens, {n_val:,} val tokens.")
    print(f"  = {n_train // BLOCK_LENGTH:,} train blocks, {n_val // BLOCK_LENGTH:,} val blocks")
    print(f"Saved to {CACHE_DIR}")


if __name__ == "__main__":
    prepare()