"""Loss vs wall clock at constant N, one panel per dataset: how K trades time against loss. python3

    python3 scripts/plot_k_tradeoff.py            -> figures/17_loss_vs_time_by_K.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")


def pool_curve(run):
    """- Return (minutes, val_loss, metrics) for one pool run, with time measured from the first fetch. x values are
    - Read from coordinator.jsonl, the same file scripts/plots.py uses, so the two views cannot disagree."""
    rows = [json.loads(l) for l in open(os.path.join(R, run, "coordinator.jsonl"))]
    starts = [i for i, r in enumerate(rows) if r["event"] == "start" and not r.get("resumed")] or [0]
    segs = [rows[a:b] for a, b in zip(starts, starts[1:] + [len(rows)])]
    done = [g for g in segs if any(r["event"] == "done" for r in g)]
    rows = done[-1] if done else max(segs, key=lambda g: sum(r["event"] == "merge" for r in g))
    m = [r for r in rows if r["event"] == "merge" and r.get("val_loss") is not None]
    t0 = m[0]["wall_time"] - m[0]["round_wall_s"]           # first fetch, not process start (drops barrier waiting)
    xs, ys = [], []
    for r in m:
        if not ys or r["val_loss"] != ys[-1]:                # eval_every_rounds>1 repeats the last value; keep real evals
            xs.append((r["wall_time"] - t0) / 60); ys.append(r["val_loss"])
    met = json.load(open(os.path.join(R, run, "metrics.json")))
    return xs, ys, met


def single_curve(run):
    """- Return (minutes, val_loss, metrics) for a single-machine control
    - The final point comes from metrics.json so the run ends on the full-pass number quoted elsewhere"""
    rows = [json.loads(l) for l in open(os.path.join(R, run, "log.jsonl"))]
    rows = [r for r in rows if r.get("val_loss") is not None and r["step"] > 0]
    met = json.load(open(os.path.join(R, run, "metrics.json")))
    xs = [r["wall_time"] / 60 for r in rows]; ys = [r["val_loss"] for r in rows]
    xs.append(met["wall_time_s"] / 60); ys.append(met["val_loss"])   # final full eval
    return xs, ys, met


PANELS = [
    {"title": "Tiny Shakespeare, 0.8M model\nN = 4 CPU workers on one Mac, 12k steps",
     "single": ("baseline_12k", "1 machine, 12k steps"),
     "pools": [("long_k25", "K = 25"), ("long_k100", "K = 100"), ("long_k500", "K = 500")],
     "note": "pool workers share one machine: transfer is free, so time differences are CPU contention, not K"},
    {"title": "text8, 10.7M model\nN = 3 workers over the internet",
     "single": ("baseline_big", "1 machine (Mac CPU), 3000 steps"),
     "pools": [("text8_three", "K = 100 (Mac GPU + lab GPU + VM)"), ("wan_big3", "K = 500 (2 CPU containers + lab GPU)")],
     "note": "node mixes differ between the two pools; text8_three includes two coordinator stalls"},
    {"title": "fil9, 40M model\nN = 2 lab GPUs, coordinator on the Mac",
     "single": ("baseline_fil9_40m", "1 GPU, 5000 steps"),
     "pools": [("fil9_2gpu", "K = 300, 2 GPUs")],
     "note": "same GPUs for both; 45% of each pool round is transfer through the Mac's uplink"},
]

COLORS = ["#0F6E74", "#C46A1C", "#7A3E9D", "#2F7D4F"]


def main():
    """- Draw the three panels into figures/17_loss_vs_time_by_K.png, the project's one loss-vs-time chart.
    - Single machine always dark, pools coloured, so each panel answers directly: did N machines reach a given loss"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, p in zip(axes, PANELS):
        xs, ys, met = single_curve(p["single"][0])
        ax.plot(xs, ys, color="#333333", lw=2.2, label=f"{p['single'][1]}: {met['val_loss']:.3f}")
        ax.axhline(met["val_loss"], color="#333333", lw=0.8, ls=":", alpha=0.6)
        for (run, label), c in zip(p["pools"], COLORS):
            xs, ys, met = pool_curve(run)
            ax.plot(xs, ys, color=c, lw=2, marker="o", ms=3.5, label=f"{label}: {met['val_loss']:.3f} in {xs[-1]:.0f} min")
        ax.set_title(p["title"], fontsize=10.5, loc="left")
        ax.set_xlabel("wall clock (minutes)"); ax.set_ylabel("validation loss (nats / char)")
        lo = min(min(l.get_ydata()) for l in ax.get_lines())
        ax.set_ylim(lo - 0.05, lo + 0.9)
        ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="upper right")
        ax.text(0.01, 0.02, p["note"], transform=ax.transAxes, fontsize=8, color="#555555", wrap=True)
    fig.suptitle("Loss vs time at constant N: larger K spends fewer bytes per token but averages more drift", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "figures", "17_loss_vs_time_by_K.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
