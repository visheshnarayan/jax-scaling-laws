"""
Data preparation pipeline for scaling-laws project.
Downloads WikiText-103, tokenizes with GPT-2 BPE,
and stores as flat numpy arrays for fast random-access during training.
"""
import io
import zipfile
import urllib.request
from pathlib import Path

import tiktoken
import numpy as np


CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
BLOCK_LENGTH = 1024
VAL_FRACTION = 0.05


def download_raw_text() -> str:
    """Download WikiText-103 raw text directly (no HuggingFace)."""
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    print(f"Downloading from {url} ...")
    data, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(data) as zf:
        with zf.open("wikitext-103-raw/wiki.train.raw") as f:
            text = f.read().decode("utf-8")
    return text


def tokenize_text(text: str, enc) -> np.ndarray:
    """Tokenize raw text into one flat token array."""
    tokens = enc.encode(text)
    return np.array(tokens, dtype=np.uint16)


def prepare():
    """Main pipeline: download, tokenize, split, save."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    print("Downloading dataset...")
    text = download_raw_text()

    print("Tokenizing...")
    tokens = tokenize_text(text, enc)

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