"""Evaluation: exact validation loss over the full held-out split + a text sample.

`evaluate_full` is the number every experiment reports. It is deterministic
(fixed non-overlapping windows) so baseline and distributed runs are directly
comparable.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from dgpt.data import Dataset
from dgpt.model import GPT, GPTConfig


@torch.no_grad()
def evaluate_full(model: GPT, ds: Dataset, device: str, batch_size: int = 256) -> dict:
    model.eval()
    total_nll, total_tok = 0.0, 0
    for x, y in ds.iter_val_full(batch_size, model.cfg.block_size, device):
        _, loss = model(x, y)
        total_nll += float(loss) * y.numel()
        total_tok += y.numel()
    model.train()
    nll = total_nll / total_tok
    return {
        "val_loss": nll,
        "val_bpc": nll / math.log(2),
        "val_ppl": math.exp(nll),
        "val_tokens": total_tok,
    }


@torch.no_grad()
def sample(model: GPT, ds: Dataset, device: str, prompt: str = "\n", n: int = 400, seed: int = 0, temperature: float = 0.8, top_k: int = 40) -> str:
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    idx = torch.tensor([ds.tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, n, temperature=temperature, top_k=top_k)
    model.train()
    return ds.tokenizer.decode(out[0].tolist())


def load_checkpoint(path: str, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample-chars", type=int, default=400)
    ap.add_argument("--dataset", default=None, help="default: the checkpoint's train_config dataset")
    args = ap.parse_args()
    ckpt0 = torch.load(args.ckpt, map_location="cpu")
    ds = Dataset.load(args.dataset or ckpt0.get("train_config", {}).get("dataset", "tinyshakespeare"))
    model, ckpt = load_checkpoint(args.ckpt, args.device)
    metrics = evaluate_full(model, ds, args.device)
    metrics["step"] = ckpt.get("step")
    metrics["tokens"] = ckpt.get("tokens")
    print(json.dumps(metrics, indent=2))
    print("---- sample ----")
    print(sample(model, ds, args.device, n=args.sample_chars))


if __name__ == "__main__":
    main()
