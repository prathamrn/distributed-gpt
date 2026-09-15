"""Run distributed experiments back to back: a coordinator plus N local worker processes each.

    python3 experiments.py --list
    python3 experiments.py diag_interleaved diag_plain_avg        # run some
    python3 experiments.py all                                    # run every experiment not yet done
    python3 experiments.py --table                                # summarize results/*/metrics.json

Each experiment is a dict: workers (N), threads per worker (`cpus`), device, coordinator --set
overrides, --train-set overrides. Workers are `worker.py` subprocesses on this machine (CPU by
default so N of them can share the box). Results land in results/<name>/ like any other run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import requests

PORT = 8000
BASE = dict(workers=4, cpus=2.0, device="cpu", sets=["local_steps=25", "total_steps=3000", "n_shards=4"], train=["lr=4e-3"])


def exp(**kw) -> dict:
    e = {k: (list(v) if isinstance(v, list) else v) for k, v in BASE.items()}
    for k, v in kw.items():
        if k in ("sets", "train"):
            e[k] = e[k] + list(v)
        else:
            e[k] = v
    return e


REASONS = {
    "diag_interleaved":    "Is the pool-vs-sync gap caused by non-IID contiguous shards? (IID shards, same everything else)",
    "diag_full_overlap":   "Does data overlap help? (no sharding; every worker samples the whole text)",
    "diag_plain_avg":      "Is DiLoCo's Nesterov outer step hurting with only 30 rounds? (plain averaging instead)",
    "diag_k5":             "Does the gap shrink toward sync as K -> 1? (K=5, DiLoCo outer)",
    "diag_lr2e-3":         "Is inner lr 4e-3 too hot for batch-64 local steps? (sqrt scaling instead of linear)",
    "diag_n2":             "Does the gap scale with worker count? (N=2 vs its own sync reference)",
    "diag_n3":             "Does the gap scale with worker count? (N=3 vs its own sync reference)",
    "diag_k5_plain":       "Was the K=5 collapse the outer momentum? (K=5 with plain averaging)",
    "diag_k5_reset":       "Was the K=5 collapse stale inner-Adam state across weight replacement? (reset Adam each round)",
    "diag_k5_plain_reset": "Both fixes together at K=5",
    "diag_k25_plain_reset":"Does inner-Adam reset help at K=25 with plain averaging?",
    "diag_k25_lr03":       "Is Nesterov's problem step size? (outer lr 0.3 instead of 0.7, momentum kept)",
    "sweep_k5":            "Headline K-sweep point, equal tokens (750 steps/worker), plain averaging",
    "sweep_k25":           "Headline K-sweep point, equal tokens (750 steps/worker), plain averaging",
    "sweep_k100":          "Headline K-sweep point, equal tokens (750 steps/worker), plain averaging; K=100 is the PRD's 100x bandwidth target",
    "long_k25":            "Equal wall-clock (3000 steps/worker, 49M tokens): does the pool beat one machine that trained as long? K=25 = 120 rounds",
    "long_k100":           "Equal wall-clock, K=100 = 30 rounds",
    "long_k500":           "Equal wall-clock, K=500 = 6 rounds, plain averaging",
    "long_k500_diloco":    "Does DiLoCo's Nesterov outer step work at its own K=500? (vs long_k500)",
    "nodes_n2":            "Node-count sweep under the new default: N=2, lr 2e-3 (linear scaling), vs its sync reference (batch 128)",
    "nodes_n3":            "Node-count sweep: N=3, lr 3e-3, vs sync reference (batch 192)",
    "nodes_n4":            "Node-count sweep: N=4, lr 4e-3 (same as diag_plain_avg; re-run as the seed check)",
    "nodes_n8":            "Node-count sweep: N=8 (1 CPU each), lr 8e-3 (linear scaling), vs sync reference (batch 512)",
    "nodes_n8_lr4e-3":     "N=8 with lr 4e-3: is linear scaling to 8e-3 too hot at batch-64 local steps?",
    "outer_mu05":          "As low as possible: is there a useful middle ground between plain averaging and DiLoCo's momentum? (mu 0.5)",
    "outer_mu03":          "Outer momentum sweep: mu 0.3 (mu 0.5 beat both 0 and 0.9; where is the optimum?)",
    "outer_mu07":          "Outer momentum sweep: mu 0.7",
    "k5_mu05":             "Does mu 0.5 also help at K=5 (150 rounds), where mu 0.9 collapsed?",
    "big_k25":             "Larger-scale run: text8 (100M chars), 10.7M-param model (6L/6H/384d), 4 workers, K=25, 12.3M tokens; does the gap to sync shrink with scale?",
    "big_k100":            "Larger-scale run on text8 at K=100 (PRD's 100x bandwidth target) on the 10.7M model",
    "big_long_k100":       "Larger-scale run at equal wall-clock: text8, 10.7M model, 4 workers x 3000 steps (49M tokens), K=100; vs the 12k-step single machine",
}

EXPERIMENTS = {
    # ---- diagnostics for the local-SGD gap (roadblocks R6) ----
    "diag_interleaved": exp(sets=["shard_mode=interleaved"]),
    "diag_full_overlap": exp(sets=["shard_mode=full"]),
    "diag_plain_avg":   exp(sets=["outer_lr=1.0", "outer_momentum=0"]),
    "diag_k5":          exp(sets=["local_steps=5"]),
    "diag_lr2e-3":      exp(train=["lr=2e-3"]),
    "diag_n2":          exp(workers=2, sets=["n_shards=2"]),
    "diag_n3":          exp(workers=3, sets=["n_shards=3"]),
    # ---- follow-ups: outer momentum and inner-Adam staleness (why K=5 was so bad) ----
    "diag_k5_plain":        exp(sets=["local_steps=5", "outer_lr=1.0", "outer_momentum=0"]),
    "diag_k5_reset":        exp(sets=["local_steps=5", "reset_inner_opt=true"]),
    "diag_k5_plain_reset":  exp(sets=["local_steps=5", "outer_lr=1.0", "outer_momentum=0", "reset_inner_opt=true"]),
    "diag_k25_plain_reset": exp(sets=["outer_lr=1.0", "outer_momentum=0", "reset_inner_opt=true"]),
    "diag_k25_lr03":        exp(sets=["outer_lr=0.3"]),
    # ---- headline K-sweep, equal tokens to the control (750 steps/worker), plain averaging ----
    "sweep_k5":    exp(sets=["local_steps=5",   "outer_lr=1.0", "outer_momentum=0"]),
    "sweep_k25":   exp(sets=["local_steps=25",  "outer_lr=1.0", "outer_momentum=0"]),
    "sweep_k100":  exp(sets=["local_steps=100", "outer_lr=1.0", "outer_momentum=0"]),
    # ---- equal wall-clock: 3000 steps/worker (pool 12,000 steps, 49M tokens); reference = 12k-step single machine ----
    "long_k25":          exp(sets=["local_steps=25",  "total_steps=12000", "outer_lr=1.0", "outer_momentum=0", "round_timeout_initial_s=600"], train=["max_steps=12000"]),
    "long_k100":         exp(sets=["local_steps=100", "total_steps=12000", "outer_lr=1.0", "outer_momentum=0", "round_timeout_initial_s=600"], train=["max_steps=12000"]),
    "long_k500":         exp(sets=["local_steps=500", "total_steps=12000", "outer_lr=1.0", "outer_momentum=0", "round_timeout_initial_s=900"], train=["max_steps=12000"]),
    "long_k500_diloco":  exp(sets=["local_steps=500", "total_steps=12000", "outer_lr=0.7", "outer_momentum=0.9", "round_timeout_initial_s=900"], train=["max_steps=12000"]),
    # ---- node-count sweep under the new default (plain averaging, K=25, equal tokens); lr scales linearly with N ----
    "nodes_n2":   exp(workers=2, cpus=2.0, sets=["n_shards=2", "outer_lr=1.0", "outer_momentum=0"], train=["lr=2e-3"]),
    "nodes_n3":   exp(workers=3, cpus=2.0, sets=["n_shards=3", "outer_lr=1.0", "outer_momentum=0"], train=["lr=3e-3"]),
    "nodes_n4":   exp(workers=4, cpus=2.0, sets=["n_shards=4", "outer_lr=1.0", "outer_momentum=0"], train=["lr=4e-3"]),
    "nodes_n8":   exp(workers=8, cpus=1.0, sets=["n_shards=8", "outer_lr=1.0", "outer_momentum=0"], train=["lr=8e-3"]),
    "nodes_n8_lr4e-3": exp(workers=8, cpus=1.0, sets=["n_shards=8", "outer_lr=1.0", "outer_momentum=0"], train=["lr=4e-3"]),
    # ---- outer momentum sweep (lr 1.0, Nesterov) ----
    "outer_mu05": exp(sets=["outer_lr=1.0", "outer_momentum=0.5"]),
    "outer_mu03": exp(sets=["outer_lr=1.0", "outer_momentum=0.3"]),
    "outer_mu07": exp(sets=["outer_lr=1.0", "outer_momentum=0.7"]),
    "k5_mu05":    exp(sets=["local_steps=5", "outer_lr=1.0", "outer_momentum=0.5"]),
    # ---- larger-scale run: nanoGPT's 6L/6H/384d (10.7M params), 4 workers, plain averaging ----
    "big_k25":    exp(sets=["outer_lr=1.0", "outer_momentum=0", "round_timeout_initial_s=600"],
                      train=["dataset=text8", "lr=4e-3", "n_layer=6", "n_head=6", "n_embd=384"]),
    "big_k100":   exp(sets=["local_steps=100", "outer_lr=1.0", "outer_momentum=0", "round_timeout_initial_s=900"],
                      train=["dataset=text8", "lr=4e-3", "n_layer=6", "n_head=6", "n_embd=384"]),
    "big_long_k100": exp(sets=["local_steps=100", "total_steps=12000", "outer_lr=1.0", "outer_momentum=0", "round_timeout_initial_s=1200"],
                         train=["dataset=text8", "lr=4e-3", "n_layer=6", "n_head=6", "n_embd=384", "max_steps=12000"]),
}

# single-machine runs made with baseline.py (not through this runner), with their reasons
SINGLE_RUNS = {
    "baseline":              "PRD control: one machine, batch 64, 3000 steps, lr 1e-3 (seed 1337)",
    "baseline_seed1":        "Control, seed 1: run-to-run variance of the control",
    "baseline_seed2":        "Control, seed 2: run-to-run variance of the control",
    "baseline_bs256_lr1e-3": "Synchronous reference for N=4 at the pool's original lr: what DDP would get at equal tokens",
    "baseline_bs256_lr2e-3": "Synchronous reference for N=4, sqrt-scaled lr",
    "baseline_bs256_lr4e-3": "Synchronous reference for N=4, linear-scaled lr (the fair bar for 4-worker pools)",
    "baseline_bs128_lr4e-3": "Synchronous reference for N=2 (batch 128, 1500 steps)",
    "baseline_bs192_lr4e-3": "Synchronous reference for N=3 (batch 192, 1000 steps)",
    "baseline_12k":          "Equal-tokens reference for the long runs: one machine, batch 64, 12,000 steps (49M tokens)",
    "baseline_bs256_12k_lr4e-3": "Synchronous reference for the long runs: batch 256, 3000 steps, lr 4e-3 (49M tokens)",
    "baseline_bs512_lr8e-3": "Synchronous reference for N=8 (batch 512, 375 steps, lr 8e-3, linear scaling)",
    "baseline_bs512_lr4e-3": "Synchronous reference for N=8 at lr 4e-3 (is 8e-3 too hot even for sync?)",
    "baseline_big":          "Control for text8 + 10.7M model: one machine, batch 64, 3000 steps, lr 1e-3",
    "baseline_big_bs256_lr4e-3": "Synchronous reference for text8 + 10.7M model at N=4 (batch 256, 750 steps, lr 4e-3)",
    "baseline_big_12k":      "Equal-tokens reference for big_long_k100: text8, 10.7M model, one machine, batch 64, 12,000 steps (49M tokens)",
}


def start_workers(n: int, threads: int, device: str, out_dir: str) -> list[subprocess.Popen]:
    procs = []
    for i in range(1, n + 1):
        log = open(os.path.join(out_dir, f"worker-{i}.out"), "a")
        procs.append(subprocess.Popen([sys.executable, "worker.py", "--name", f"worker-{i}", "--coordinator", f"http://127.0.0.1:{PORT}",
                                       "--threads", str(threads), "--device", device, "--out-dir", out_dir], stdout=log, stderr=subprocess.STDOUT))
    return procs


def last_round_line(path: str) -> str:
    try:
        lines = [l for l in open(path) if "] round" in l or "DONE" in l]
        return lines[-1].strip()[:120] if lines else "(starting)"
    except FileNotFoundError:
        return "(no log yet)"


def run_one(name: str, e: dict) -> dict | None:
    threads = max(1, int(round(float(e.get("cpus", 2.0)))))
    device = e.get("device", "cpu")
    print(f"\n=== {name}: {e['workers']} workers x {threads} threads ({device}), sets={e['sets']} train={e['train']} ===", flush=True)
    for _ in range(30):                                   # the previous coordinator must be gone and the port free
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1); time.sleep(2)
        except requests.ConnectionError:
            break
    else:
        sys.exit(f"port {PORT} still busy; is an old coordinator running?")
    out_dir = f"results/{name}"
    shutil.rmtree(out_dir, ignore_errors=True); os.makedirs(out_dir, exist_ok=True)
    log = open(f"results/{name}_coord.out", "w")
    args = [sys.executable, "coordinator.py", "--run-name", name, "--port", str(PORT), "--exit-when-done"]
    for s in e["sets"]:
        args += ["--set", s]
    for t in e["train"]:
        args += ["--train-set", t]
    coord = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).ok:
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        coord.kill(); sys.exit("coordinator did not come up")
    workers = start_workers(e["workers"], threads, device, out_dir)
    t0 = time.time()
    try:
        while coord.poll() is None:
            time.sleep(15)
            print(f"  [{time.time()-t0:5.0f}s] {last_round_line(f'results/{name}_coord.out')}", flush=True)
        for w in workers:                                 # workers exit on their own after fetching the final weights
            try:
                w.wait(timeout=60)
            except subprocess.TimeoutExpired:
                w.kill()
    except KeyboardInterrupt:
        print("interrupted: stopping workers and coordinator")
        for w in workers:
            w.kill()
        coord.kill()
        raise
    mp = f"results/{name}/metrics.json"
    if not os.path.exists(mp):
        print(f"  !! {name}: no metrics.json (see results/{name}_coord.out)")
        return None
    m = json.load(open(mp))
    print(f"  -> {name}: val {m['val_loss']:.4f}  rounds {m['rounds']}  {m['wall_time_s']:.0f}s  {m['bytes_total']/1e6:.0f} MB  stale {m['stale_total']}", flush=True)
    return m


def table(names: list[str] | None = None):
    b = json.load(open("results/baseline_summary.json")) if os.path.exists("results/baseline_summary.json") else None
    print(f"{'run':<22}{'val':>8}{'vs ctrl':>9}{'rounds':>7}{'wall_s':>8}{'MB':>8}{'stale':>6}  config")
    for mp in sorted(glob.glob("results/*/metrics.json")):
        m = json.load(open(mp)); name = os.path.basename(os.path.dirname(mp))
        if names and name not in names:
            continue
        if "run_config" not in m:
            continue
        rc = m["run_config"]; tc = m["train_config"]
        cfg = f"N={len([1 for _ in glob.glob(os.path.dirname(mp)+'/worker-*.jsonl')])} K={rc['local_steps']} lr={tc['lr']:g} outer=({rc['outer_lr']},{rc['outer_momentum']}) shards={rc['shard_mode']}"
        vs = f"{100*(m['val_loss']/b['val_loss_mean']-1):+.1f}%" if b else ""
        print(f"{name:<22}{m['val_loss']:>8.4f}{vs:>9}{m['rounds']:>7}{m['wall_time_s']:>8.0f}{m['bytes_total']/1e6:>8.0f}{m['stale_total']:>6}  {cfg}")


def ledger(path: str = "evals.md"):
    """Regenerate the eval ledger: every run with its reason, config, result, and verdict vs references."""
    import datetime
    b = json.load(open("results/baseline_summary.json")) if os.path.exists("results/baseline_summary.json") else None
    ctrl = b["val_loss_mean"] if b else None
    lines = ["# Eval ledger", "", "Every training run, why it was run, and what it showed. Regenerated by",
             "`python3 experiments.py --ledger`; reasons live in `experiments.py` (REASONS / SINGLE_RUNS).",
             f"Control = {ctrl:.4f} (mean of 3 seeds). 'vs ctrl' is relative to it. Updated {datetime.date.today()}.", "",
             "## Single-machine runs (baseline.py)", "",
             "| run | val loss | vs ctrl | batch | steps | tokens | lr | reason |", "|---|---|---|---|---|---|---|---|"]
    for name, reason in SINGLE_RUNS.items():
        mp = f"results/{name}/metrics.json"
        if not os.path.exists(mp):
            lines.append(f"| {name} | (running / not yet) | | | | | | {reason} |"); continue
        m = json.load(open(mp)); c = m["config"]
        vs = f"{100*(m['val_loss']/ctrl-1):+.1f}%" if ctrl else ""
        lines.append(f"| {name} | {m['val_loss']:.4f} | {vs} | {c['batch_size']} | {c['max_steps']} | {m['tokens']/1e6:.1f}M | {c['lr']:g} | {reason} |")
    lines += ["", "## Distributed runs (experiments.py)", "",
              "| run | val loss | vs ctrl | N | K | inner lr | outer (lr, mu) | shards | steps | rounds | MB | stale | reason |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in EXPERIMENTS.items():
        reason = REASONS.get(name, "")
        mp = f"results/{name}/metrics.json"
        if not os.path.exists(mp):
            lines.append(f"| {name} | (not run) | | {e['workers']} | | | | | | | | | {reason} |"); continue
        m = json.load(open(mp)); rc, tc = m["run_config"], m["train_config"]
        n = len(glob.glob(f"results/{name}/worker-*.jsonl"))
        vs = f"{100*(m['val_loss']/ctrl-1):+.1f}%" if ctrl else ""
        lines.append(f"| {name} | {m['val_loss']:.4f} | {vs} | {n} | {rc['local_steps']} | {tc['lr']:g} | ({rc['outer_lr']}, {rc['outer_momentum']}) | {rc['shard_mode']}"
                     f" | {m['steps']} | {m['rounds']} | {m['bytes_total']/1e6:.0f} | {m['stale_total']} | {reason} |")
    # manual runs not in EXPERIMENTS (first two pool runs)
    extra = {"k25_uniform": "First real 4-worker run: K=25, inner lr 1e-3 (control's lr), DiLoCo outer. Missed control by 21%.",
             "k25_lr4e-3": "Retune: inner lr 4e-3 (linear scaling), DiLoCo outer. Closed the batch-size part of the gap.",
             "wan_docker_honeydew": "Real internet: Docker CPU worker + honeydew GPU worker via ngrok, signed auth, adaptive K, contiguous shards. Fast worker did 95% of steps on half the text.",
             "wan_interleaved": "Same, interleaved (block-aligned) shards: fast worker cycled 7.8K fixed windows 23x.",
             "wan_full": "Same, full random-offset sampling: the control's data path. Ties the best single machine.",
             "mac_honeydew_big": "text8, 10.7M model: Mac GPU worker (local) + honeydew GPU via Cloudflare tunnel, adaptive K, bf16 deltas, full sampling. Within 2% of the control."}
    lines += ["", "## Earlier manual pool runs", "", "| run | val loss | vs ctrl | reason |", "|---|---|---|---|"]
    for name, reason in extra.items():
        mp = f"results/{name}/metrics.json"
        if os.path.exists(mp):
            m = json.load(open(mp)); vs = f"{100*(m['val_loss']/ctrl-1):+.1f}%" if ctrl else ""
            lines.append(f"| {name} | {m['val_loss']:.4f} | {vs} | {reason} |")
    lines += ["", "See `roadblocks.md` for the narrative and conclusions, `qna.md` for decisions."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--ledger", action="store_true", help="regenerate evals.md")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    args = ap.parse_args()
    if args.ledger:
        ledger(); return
    if args.list:
        for n, e in EXPERIMENTS.items():
            print(f"{n:<22} N={e['workers']} cpus={e['cpus']} sets={e['sets']} train={e['train']}")
        return
    if args.table:
        table(args.names or None); return
    names = list(EXPERIMENTS) if args.names == ["all"] else args.names
    for n in names:
        if n not in EXPERIMENTS:
            sys.exit(f"unknown experiment {n}; --list to see them")
        if os.path.exists(f"results/{n}/metrics.json") and not args.force:
            print(f"skip {n}: results exist (use --force)"); continue
        run_one(n, EXPERIMENTS[n])
    table(names)
    ledger()


if __name__ == "__main__":
    main()
