"""Evaluate a checkpoint on the held-out TEST region of a dataset: the last 5M characters of the val split, which
no training or validation pass ever reads (Dataset.load keeps only the first VAL_MAX_CHARS of val.bin).
Works for coordinator checkpoints (results/<run>/ckpt.pt, key "weights") and baseline ones (key "model").

    python3 scripts/test_eval.py results/fil9_full/ckpt.pt [--device mps]
"""
import argparse
import dataclasses
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dgpt.config import TrainConfig
from dgpt.data import Dataset, _paths
from dgpt.evaluate import evaluate_full
from dgpt.model import GPT, GPTConfig

TEST_CHARS = 5_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--dataset", default=None, help="override; default: from the checkpoint's train_config")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if "weights" in ck:                      # coordinator checkpoint
        fields = {f.name for f in dataclasses.fields(TrainConfig)}
        tc = TrainConfig(**{k: v for k, v in ck["train_config"].items() if k in fields})
        cfg, weights, dataset = tc.model_config(), ck["weights"], tc.dataset
    else:                                    # baseline checkpoint
        cfg, weights = GPTConfig(**ck["model_config"]), ck["model"]
        dataset = ck.get("train_config", {}).get("dataset") or a.dataset
    dataset = a.dataset or dataset
    model = GPT(cfg); model.load_state_dict(weights); model.to(a.device)
    ds = Dataset.load(dataset)
    full_val = np.fromfile(_paths(dataset)["val"], dtype=np.uint16)
    test = Dataset(train=ds.train, val=full_val[-TEST_CHARS:], tokenizer=ds.tokenizer, name=dataset)
    with torch.no_grad():
        r = evaluate_full(model, test, a.device)
    out = {"ckpt": a.ckpt, "dataset": dataset, "test_chars": TEST_CHARS, "test_loss": r["val_loss"], "test_bpc": r["val_bpc"], "test_ppl": r["val_ppl"]}
    print(json.dumps(out))
    with open(os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "test_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
