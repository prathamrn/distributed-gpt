"""Consolidate a set of pool runs (and a baseline) into one table: loss, test loss, wall clock

    python3 scripts/summarize_pool.py --baseline results/baseline_fil9_40m results/fil9_full results/fil9_interleaved ..."""
import argparse
import collections
import json
import os


def load_run(d: str) -> dict:
    """- Reduce one pool run's coordinator.jsonl (plus its metrics files) to the row this table prints.
    - Read from the event log, not a coordinator summary, because the interesting quantities are per-worker and"""
    rows = [json.loads(l) for l in open(os.path.join(d, "coordinator.jsonl"))]
    starts = [i for i, r in enumerate(rows) if r["event"] == "start" and not r.get("resumed")]
    rows = rows[starts[-1]:] if starts else rows          # only the last fresh start of this run name
    merges = [r for r in rows if r["event"] == "merge"]
    done = next((r for r in rows if r["event"] == "done"), None)
    metrics = json.load(open(os.path.join(d, "metrics.json"))) if os.path.exists(os.path.join(d, "metrics.json")) else {}
    test = json.load(open(os.path.join(d, "test_metrics.json"))) if os.path.exists(os.path.join(d, "test_metrics.json")) else {}
    steps = collections.Counter()
    for m in merges:
        for w, n in m["steps_per_worker"].items():
            steps[w] += n
    tokens_per_step = metrics.get("train_config", {}).get("tokens_per_step") or merges[-1]["tokens"] / max(1, merges[-1]["global_step"])
    speed, overhead = {}, {}
    if merges:
        for w, t in merges[-1].get("worker_timing", {}).items():
            speed[w] = t.get("steps_per_s"); overhead[w] = t.get("overhead_s")
    t_start = merges[0]["wall_time"] - merges[0]["round_wall_s"] if merges else None
    t_end = done["wall_time"] if done else (merges[-1]["wall_time"] if merges else None)
    stale = sum(1 for r in rows if r["event"] == "stale_delta")
    timeouts = sum(1 for m in merges if m["close_reason"].startswith("timeout"))
    return {
        "name": os.path.basename(d.rstrip("/")), "shard_mode": metrics.get("run_config", {}).get("shard_mode"),
        "val_loss": metrics.get("val_loss"), "test_loss": test.get("test_loss"),
        "wall_s": (t_end - t_start) if (t_start is not None and t_end is not None) else None,
        "steps": sum(steps.values()), "tokens": sum(steps.values()) * tokens_per_step,
        "share": {w: steps[w] / max(1, sum(steps.values())) for w in steps},
        "rounds": len(merges), "timeouts": timeouts, "stale": stale,
        "bytes_gb": (metrics.get("bytes_total") or (merges[-1]["bytes_total"] if merges else 0)) / 1e9,
        "speed": speed, "overhead": overhead,
    }


def load_baseline(d: str) -> dict:
    """- Load a single-machine control into the same row shape, with the pool-only fields zeroed.
    - Same columns as a pool row is what makes "vs control" computable in one pass"""
    m = json.load(open(os.path.join(d, "metrics.json")))
    test = json.load(open(os.path.join(d, "test_metrics.json"))) if os.path.exists(os.path.join(d, "test_metrics.json")) else {}
    return {"name": os.path.basename(d.rstrip("/")), "shard_mode": "(single machine)", "val_loss": m["val_loss"], "test_loss": test.get("test_loss"),
            "wall_s": m.get("wall_time_s"), "steps": m["steps"], "tokens": m["tokens"], "share": {m.get("device", "one"): 1.0},
            "rounds": 0, "timeouts": 0, "stale": 0, "bytes_gb": 0.0, "speed": {}, "overhead": {}}


def fmt(x, nd=4):
    """- Format a cell: fixed decimals for floats, str() for anything else, and an empty cell for None.
    - Missing values are meaningful here (no test pass, a node that never reported a speed)"""
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    """- Build the Markdown comparison table for a set of runs and write it next to the run directories.
    - The campaign-level view per-run metrics.json cannot give: several pools differing in one setting"""
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    rows = ([load_baseline(a.baseline)] if a.baseline else []) + [load_run(d) for d in a.runs]
    nodes = []
    for r in rows:
        for w in r["share"]:
            if w not in nodes:
                nodes.append(w)
    base = rows[0]["val_loss"] if a.baseline else None
    hdr = ["run", "sharding", "val loss", "vs control", "test loss", "wall clock", "steps", "tokens (M)", "rounds", "timeout closes", "stale"] + \
          [f"% steps {w}" for w in nodes] + [f"steps/s {w}" for w in nodes] + ["GB moved"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        vs = "" if (base is None or r["val_loss"] is None) else f"{100 * (r['val_loss'] / base - 1):+.1f}%"
        wall = "" if r["wall_s"] is None else f"{r['wall_s'] / 60:.1f} min"
        cells = [r["name"], r["shard_mode"] or "", fmt(r["val_loss"]), vs, fmt(r["test_loss"]), wall, str(r["steps"]), f"{r['tokens'] / 1e6:.1f}",
                 str(r["rounds"]), str(r["timeouts"]), str(r["stale"])] + \
                [f"{100 * r['share'].get(w, 0):.1f}" if w in r["share"] else "" for w in nodes] + \
                [fmt(r["speed"].get(w), 2) if r["speed"].get(w) else "" for w in nodes] + [f"{r['bytes_gb']:.2f}"]
        out.append("| " + " | ".join(cells) + " |")
    table = "\n".join(out)
    print(table)
    dst = os.path.join(os.path.dirname(os.path.abspath(a.runs[0].rstrip("/"))), "pool_summary.md")
    with open(dst, "w") as f:
        f.write(table + "\n")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
