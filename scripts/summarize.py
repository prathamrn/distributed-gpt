"""Aggregate metrics.json files from a set of runs into one table + JSON. python3 scripts/summarize.py python3

  python3 scripts/summarize.py results/baseline*          # prints a table, writes results/baseline_summary.json"""
from __future__ import annotations

import glob
import json
import math
import os
import sys


def load_runs(patterns: list[str]) -> list[dict]:
    """- Load metrics.json for every run directory matching the glob patterns.
    - Directories without metrics.json are skipped, so an unfinished run never counts as a seed."""
    runs = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            mp = os.path.join(d, "metrics.json")
            if os.path.isdir(d) and os.path.exists(mp):
                with open(mp) as f:
                    m = json.load(f)
                m["_dir"] = d
                runs.append(m)
    return runs


def mean_std(xs: list[float]) -> tuple[float, float]:
    """- Mean and sample standard deviation (n-1) of xs; std 0.0 for a single value."""
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return mu, sd


def main():
    """- Print the per-seed table and write results/baseline_summary.json, the control every run is judged against.
        - Default pattern averages every baseline* directory into one mean"""
    pats = sys.argv[1:] or ["results/baseline*"]
    runs = load_runs(pats)
    if not runs:
        sys.exit("no runs found")
    print(f"{'run':<24}{'seed':>6}{'val_loss':>10}{'val_bpc':>9}{'val_ppl':>9}{'train_ema':>11}{'steps':>7}{'tokens':>12}{'wall_s':>8}{'steps/s':>9}")
    for m in runs:
        print(f"{os.path.basename(m['_dir']):<24}{m['config']['seed']:>6}{m['val_loss']:>10.4f}{m['val_bpc']:>9.3f}{m['val_ppl']:>9.3f}"
              f"{m['final_train_loss_ema']:>11.4f}{m['steps']:>7}{m['tokens']:>12,}{m['wall_time_s']:>8.1f}{m['steps_per_s']:>9.1f}")
    vl = [m["val_loss"] for m in runs]
    mu, sd = mean_std(vl)
    print(f"\nval_loss mean ± std over {len(runs)} seed(s): {mu:.4f} ± {sd:.4f}   (min {min(vl):.4f}, max {max(vl):.4f})")
    print(f"5% band around mean (PRD criterion 1): {mu:.4f} – {mu*1.05:.4f}")
    summary = {
        "n_runs": len(runs),
        "val_loss_mean": mu, "val_loss_std": sd, "val_loss_min": min(vl), "val_loss_max": max(vl),
        "val_bpc_mean": mu / math.log(2), "val_ppl_at_mean": math.exp(mu),
        "within_5pct_threshold": mu * 1.05,
        "steps": runs[0]["steps"], "tokens": runs[0]["tokens"], "params": runs[0]["params"],
        "runs": [{"dir": m["_dir"], "seed": m["config"]["seed"], "val_loss": m["val_loss"],
                  "train_loss_ema": m["final_train_loss_ema"], "wall_time_s": m["wall_time_s"]} for m in runs],
    }
    out = "results/baseline_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
