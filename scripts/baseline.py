"""Control experiment: one process, one model, standard AdamW training."""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, so `dgpt` imports work without installing

import argparse
import json
import os
import time

import numpy as np
import torch

from dgpt.config import TrainConfig, lr_at
from dgpt.data import Dataset
from dgpt.evaluate import evaluate_full, sample
from dgpt.model import GPT


def pick_device(name: str) -> str:
    """- Resolve --device: explicit name passes through, 'auto' prefers Apple MPS over CPU.
        - No CUDA branch: this is a Mac-side script and dgpt/worker.py owns full cuda/mps/cpu detection."""
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def quick_val(model: GPT, ds: Dataset, cfg: TrainConfig, device: str) -> float:
    """- Cheap in-training eval over cfg.eval_batches fixed random val windows.
    - The RNG is rebuilt from cfg.seed + 1 every call, so every point on the curve scores the same windows."""
    model.eval()
    rng = np.random.default_rng(cfg.seed + 1)
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = ds.get_batch("val", cfg.batch_size, cfg.block_size, rng, device)
        _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def train(cfg: TrainConfig, device: str, out_dir: str, log_every: int = 50) -> dict:
    """- Train one model to cfg.max_steps and write the run directory the rest of the project reads.
        - Uses exactly the dgpt/config.py recipe the workers use (same AdamW, warmup-then-cosine, batch)"""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    ds = Dataset.load(cfg.dataset)
    cfg.vocab_size = ds.vocab_size
    model = GPT(cfg.model_config()).to(device)
    opt = model.make_optimizer(cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2))
    n_params = model.num_params()
    print(f"device={device} params={n_params:,} steps={cfg.max_steps} tokens/step={cfg.tokens_per_step} total_tokens={cfg.total_tokens:,}")

    log_path = os.path.join(out_dir, "log.jsonl")
    logf = open(log_path, "w")

    def log(row: dict):
        logf.write(json.dumps(row) + "\n")
        logf.flush()

    t0 = time.time()
    train_loss_ema = None
    for step in range(cfg.max_steps + 1):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            vl = quick_val(model, ds, cfg, device)
            row = {
                "step": step, "tokens": step * cfg.tokens_per_step,
                "train_loss": train_loss_ema, "val_loss": vl, "lr": lr,
                "wall_time": time.time() - t0, "bytes_sent": 0,
            }
            log(row)
            print(f"step {step:5d} | tokens {row['tokens']:>10,} | train {train_loss_ema if train_loss_ema is not None else float('nan'):.4f} | val {vl:.4f} | lr {lr:.2e} | {row['wall_time']:.1f}s")
        if step == cfg.max_steps:
            break

        x, y = ds.get_batch("train", cfg.batch_size, cfg.block_size, rng, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        l = loss.item()
        train_loss_ema = l if train_loss_ema is None else 0.9 * train_loss_ema + 0.1 * l

    wall = time.time() - t0
    logf.close()

    # ---- final, exact evaluation on the full val split ----------------------
    metrics = evaluate_full(model, ds, device)
    metrics.update({
        "run": "baseline",
        "device": device,
        "params": n_params,
        "steps": cfg.max_steps,
        "tokens": cfg.total_tokens,
        "wall_time_s": wall,
        "steps_per_s": cfg.max_steps / wall,
        "tokens_per_s": cfg.total_tokens / wall,
        "final_train_loss_ema": train_loss_ema,
        "bytes_sent": 0,
        "config": cfg.to_dict(),
    })
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save({
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_config": cfg.model_config().to_dict(),
        "train_config": cfg.to_dict(),
        "step": cfg.max_steps,
        "tokens": cfg.total_tokens,
    }, os.path.join(out_dir, "ckpt.pt"))

    txt = sample(model, ds, device)
    with open(os.path.join(out_dir, "sample.txt"), "w") as f:
        f.write(txt)
    return metrics


def main():
    """- Parse the CLI, build a TrainConfig, run one control into --out.
    - Flags mirror the coordinator's --train-set, so a control and a pool run share one recipe."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu | mps | auto")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], help="override TrainConfig, e.g. --set batch_size=256 --set lr=2e-3")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    cfg = TrainConfig()
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.seed is not None:
        cfg.seed = args.seed
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(float(v)) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    m = train(cfg, device, args.out)
    print(json.dumps({k: v for k, v in m.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
