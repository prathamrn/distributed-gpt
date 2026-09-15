"""Character-level data pipeline for Tiny Shakespeare.

- `prepare()` tokenizes input.txt once into train.bin / val.bin (uint16) + meta.json.
- `Dataset` loads those and serves random training batches or a deterministic
  full pass over the validation split.
- Sharding for the distributed runs (PRD 7.8) lives here too so the baseline
  and workers share one tokenization.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

# Where datasets live. Repo checkout: ./data next to this file. Installed worker: $DGPT_DATA_DIR
# (default ~/.cache/dgpt/data), filled by fetching from the coordinator on first use.
_REPO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# repo checkout: ./data next to this file; installed package (no ./data): ~/.cache/dgpt/data
ROOT = os.environ.get("DGPT_DATA_DIR") or (_REPO_DATA if os.path.isdir(_REPO_DATA) else os.path.expanduser("~/.cache/dgpt/data"))
DATASET_FILES = ("meta.json", "train.bin", "val.bin")

# name -> (raw file, train fraction). Splits are contiguous cuts, tokenization is a per-char lookup.
DATASETS = {
    "tinyshakespeare": {"raw": "input.txt", "train_frac": 0.9},
    "text8": {"raw": "text8", "train_frac": 0.9},       # 100M chars of Wikipedia, vocab 27; 90M train / 10M val
}
DEFAULT_DATASET = "tinyshakespeare"
VAL_MAX_CHARS = 500_000        # per-round eval must stay ~1 s: score a fixed prefix of the val split


def _paths(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    d = os.path.join(ROOT, name)
    return {"dir": d, "raw": os.path.join(d, DATASETS[name]["raw"]), "train": os.path.join(d, "train.bin"),
            "val": os.path.join(d, "val.bin"), "meta": os.path.join(d, "meta.json")}


# backwards-compatible module constants (tinyshakespeare)
DATA_DIR = _paths(DEFAULT_DATASET)["dir"]
RAW_PATH = _paths(DEFAULT_DATASET)["raw"]
TRAIN_PATH = _paths(DEFAULT_DATASET)["train"]
VAL_PATH = _paths(DEFAULT_DATASET)["val"]
META_PATH = _paths(DEFAULT_DATASET)["meta"]


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def prepare(name: str = DEFAULT_DATASET, force: bool = False) -> dict:
    """Tokenize the raw text once. Split is a contiguous cut (90/10), as in nanoGPT."""
    p = _paths(name)
    if not force and all(os.path.exists(p[k]) for k in ("train", "val", "meta")):
        with open(p["meta"]) as f:
            return json.load(f)
    if not os.path.exists(p["raw"]):
        raise FileNotFoundError(f"{p['raw']} missing: download the {name} corpus first")
    with open(p["raw"], "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    # vectorized lookup (100M-char corpora): map each unicode code point to its vocab index
    codes = np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
    table = np.zeros(int(codes.max()) + 1, dtype=np.uint16)
    for i, ch in enumerate(chars):
        table[ord(ch)] = i
    ids = table[codes]
    n = len(ids)
    n_train = int(n * DATASETS[name]["train_frac"])
    ids[:n_train].tofile(p["train"])
    ids[n_train:].tofile(p["val"])
    meta = {"name": name, "vocab_size": len(chars), "chars": chars, "n_total": int(n), "n_train": int(n_train), "n_val": int(n - n_train)}
    with open(p["meta"], "w") as f:
        json.dump(meta, f)
    return meta


def ensure_dataset(name: str, fetch_from: str | None = None, progress=print, headers=None) -> None:
    """`headers` may be a dict or a callable (method, url) -> dict (for per-request signatures)."""
    """Make the tokenized files for `name` exist locally. If missing and `fetch_from` (a coordinator URL)
    is given, download them from GET {fetch_from}/data/{name}/{file}. Falls back to tokenizing a local raw file."""
    p = _paths(name)
    if all(os.path.exists(p[k]) for k in ("train", "val", "meta")):
        return
    if fetch_from:
        import urllib.request
        os.makedirs(p["dir"], exist_ok=True)
        for fn in DATASET_FILES:
            dst = os.path.join(p["dir"], fn)
            if os.path.exists(dst):
                continue
            url = f"{fetch_from.rstrip('/')}/data/{name}/{fn}"
            progress(f"downloading {url} -> {dst}")
            h = headers(method="GET", url=url) if callable(headers) else (headers or {})
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dst + ".part", dst)
        return
    prepare(name)


@dataclass
class Dataset:
    train: np.ndarray
    val: np.ndarray
    tokenizer: CharTokenizer
    name: str = DEFAULT_DATASET

    @classmethod
    def load(cls, name: str = DEFAULT_DATASET, fetch_from: str | None = None, headers=None) -> "Dataset":
        ensure_dataset(name, fetch_from, headers=headers)
        meta = prepare(name)
        p = _paths(name)
        train = np.fromfile(p["train"], dtype=np.uint16)
        val = np.fromfile(p["val"], dtype=np.uint16)[:VAL_MAX_CHARS]       # fixed eval prefix, same for every run
        return cls(train=train, val=val, tokenizer=CharTokenizer(meta["chars"]), name=name)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def _split(self, split: str) -> np.ndarray:
        return self.train if split == "train" else self.val

    def get_batch(self, split: str, batch_size: int, block_size: int, rng: np.random.Generator, device: str = "cpu"):
        """Random contiguous windows, nanoGPT-style. Seeded via `rng` for reproducibility."""
        d = self._split(split)
        ix = rng.integers(0, len(d) - block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(d[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    def iter_val_full(self, batch_size: int, block_size: int, device: str = "cpu"):
        """Deterministic single pass over the whole validation split in
        non-overlapping windows. Every eval sees exactly the same tokens, so
        numbers are comparable across runs and across machines."""
        d = self.val
        n_windows = (len(d) - 1) // block_size
        xs = np.stack([d[i * block_size:(i + 1) * block_size] for i in range(n_windows)]).astype(np.int64)
        ys = np.stack([d[i * block_size + 1:(i + 1) * block_size + 1] for i in range(n_windows)]).astype(np.int64)
        for s in range(0, n_windows, batch_size):
            x = torch.from_numpy(xs[s:s + batch_size]).to(device)
            y = torch.from_numpy(ys[s:s + batch_size]).to(device)
            yield x, y

    # ---- sharding (PRD 7.8); used by workers in later steps -----------------
    def shard_bounds(self, n_shards: int) -> list[tuple[int, int]]:
        n = len(self.train)
        edges = np.linspace(0, n, n_shards + 1, dtype=np.int64)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(n_shards)]

    def get_shard_batch(self, shard_id: int, n_shards: int, offset: int, batch_size: int, block_size: int, device: str = "cpu"):
        """Sequential batches from one shard, wrapping at the shard end.
        Returns (x, y, new_offset)."""
        lo, hi = self.shard_bounds(n_shards)[shard_id]
        d = self.train
        span = hi - lo
        assert span > block_size + 1, "shard too small for block_size"
        xs, ys = [], []
        for _ in range(batch_size):
            start = lo + (offset % (span - block_size - 1))
            xs.append(torch.from_numpy(d[start:start + block_size].astype(np.int64)))
            ys.append(torch.from_numpy(d[start + 1:start + 1 + block_size].astype(np.int64)))
            offset += block_size
        return torch.stack(xs).to(device), torch.stack(ys).to(device), offset


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    meta = prepare(name, force=True)
    print({k: v for k, v in meta.items() if k != "chars"})
