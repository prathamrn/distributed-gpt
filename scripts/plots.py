"""Turn JSONL training logs into figures (PRD 9, 11).

  python3 scripts/plots.py baseline                 # control figure from results/baseline*/log.jsonl
  python3 scripts/plots.py curves A=path/a.jsonl B=path/b.jsonl --out results/x.png --title "..." """
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- palette (light mode) ---------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = {"family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]}


def read_log(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def style_axes(ax, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(True, axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def millions(x, _):
    return f"{x/1e6:g}M" if x else "0"


def plot_curves(ax, series: list[tuple[str, list[float], list[float]]], end_labels: bool = True):
    """- series: list of (label, xs, ys).
    - Hue follows list order (fixed slots)."""
    for i, (label, xs, ys) in enumerate(series):
        c = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, color=c, linewidth=2, solid_joinstyle="round", solid_capstyle="round", label=label, zorder=3)
        # end marker: >= 8px with 2px surface ring
        ax.plot(xs[-1], ys[-1], marker="o", markersize=7, markerfacecolor=c,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    if end_labels and len(series) <= 4:
        # Direct end labels in ink. If ends converge, fan the labels apart and
        # connect each to its line-end with a thin leader (never stack blindly).
        ends = sorted([(ys[-1], label, xs[-1]) for label, xs, ys in series])
        ymin, ymax = ax.get_ylim()
        min_gap = (ymax - ymin) * 0.05
        placed = []
        for y, label, x in ends:
            yy = y if not placed else max(y, placed[-1] + min_gap)
            placed.append(yy)
        # centre the fan on the true ends so labels don't drift upward
        shift = (sum(p for p in placed) / len(placed)) - (sum(e[0] for e in ends) / len(ends))
        placed = [p - shift for p in placed]
        x0, x1 = ax.get_xlim()
        dx = (x1 - x0) * 0.025
        for (y, label, x), yy in zip(ends, placed):
            ax.annotate(label, xy=(x, y), xytext=(x + dx, yy), textcoords="data",
                        va="center", ha="left", fontsize=9, color=INK2, zorder=5,
                        arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=0, shrinkB=4))
        ax.set_xlim(x0, x1 + (x1 - x0) * 0.12)
    if len(series) >= 2:
        leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
        for t in leg.get_texts():
            t.set_color(INK2)


def fig_baseline(out_png: str):
    """- Control figure: val loss vs tokens for every baseline seed, plus the train/val gap for the primary seed."""
    runs = {}
    for d in sorted(glob.glob("results/baseline*")):
        lp = os.path.join(d, "log.jsonl")
        mp = os.path.join(d, "metrics.json")
        if os.path.isdir(d) and os.path.exists(lp) and os.path.exists(mp):
            with open(mp) as f:
                m = json.load(f)
            runs[d] = (read_log(lp), m)
    if not runs:
        sys.exit("no baseline runs found under results/")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=PAGE)
    plt.rcParams["font.family"] = FONT["family"]

    # left: val loss vs tokens, one line per seed
    series = []
    for d, (rows, m) in runs.items():
        seed = m["config"]["seed"]
        xs = [r["tokens"] for r in rows if r["step"] > 0]
        ys = [r["val_loss"] for r in rows if r["step"] > 0]
        series.append((f"seed {seed}", xs, ys))
    ax = axes[0]
    style_axes(ax, "training tokens", "validation loss (nats/char)")
    plot_curves(ax, series)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_ylim(1.4, 2.6)
    finals = [m["val_loss"] for _, m in runs.values()]
    ax.set_title(f"Single-machine control, {len(runs)} seed{'s' if len(runs)>1 else ''}: final full-val loss "
                 f"{min(finals):.3f}–{max(finals):.3f}", color=INK, fontsize=11, loc="left", pad=12)

    # right: train vs val for the primary seed (or first)
    primary = "results/baseline" if "results/baseline" in runs else next(iter(runs))
    rows, m = runs[primary]
    xs = [r["tokens"] for r in rows if r["step"] > 0]
    ax = axes[1]
    style_axes(ax, "training tokens", "loss (nats/char)")
    plot_curves(ax, [
        ("val", xs, [r["val_loss"] for r in rows if r["step"] > 0]),
        ("train (EMA)", xs, [r["train_loss"] for r in rows if r["step"] > 0]),
    ])
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_ylim(1.2, 2.6)
    ax.set_title(f"Train vs val, seed {m['config']['seed']}: gap {m['val_loss']-m['final_train_loss_ema']:.2f} at end",
                 color=INK, fontsize=11, loc="left", pad=12)

    fig.text(0.01, 0.01,
             f"{m['params']:,} params · {m['config']['n_layer']}L/{m['config']['n_head']}H/{m['config']['n_embd']}d · block {m['config']['block_size']} · "
             f"batch {m['config']['batch_size']} · {m['steps']:,} steps · {m['tokens']/1e6:.1f}M tokens · AdamW lr {m['config']['lr']:g}→{m['config']['min_lr']:g} cosine · CPU",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, facecolor=PAGE)
    print("wrote", out_png)


def fig_curves(specs: list[str], out_png: str, title: str, ykey: str = "val_loss"):
    series = []
    for spec in specs:
        label, path = spec.rsplit("=", 1)
        rows = read_log(path)
        series.append((label, [r["tokens"] for r in rows if r["step"] > 0], [r[ykey] for r in rows if r["step"] > 0]))
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=PAGE)
    style_axes(ax, "training tokens", "validation loss (nats/char)")
    plot_curves(ax, series)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, facecolor=PAGE)
    print("wrote", out_png)


def read_run(path_or_dir: str) -> tuple[list[float], list[float], dict | None]:
    """- (tokens, val_loss, metrics) for either a baseline dir/log.jsonl or a distributed dir/coordinator.jsonl."""
    d = path_or_dir if os.path.isdir(path_or_dir) else os.path.dirname(path_or_dir)
    metrics = None
    mp = os.path.join(d, "metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            metrics = json.load(f)
    cj = os.path.join(d, "coordinator.jsonl")
    if os.path.exists(cj):
        rows = [r for r in read_log(cj) if r.get("event") == "merge" and r.get("val_loss") is not None]
        return [r["tokens"] for r in rows], [r["val_loss"] for r in rows], metrics
    rows = [r for r in read_log(os.path.join(d, "log.jsonl")) if r["step"] > 0]
    return [r["tokens"] for r in rows], [r["val_loss"] for r in rows], metrics


def fig_compare(specs: list[str], out_png: str, title: str, baseline_band: str | None = None):
    """- Distributed runs vs baseline, val loss vs tokens. specs: label=results/<run>."""
    series = []
    for spec in specs:
        label, path = spec.rsplit("=", 1)
        xs, ys, m = read_run(path)
        if m and "val_loss" in m:
            label = f"{label} (final {m['val_loss']:.3f})"
        series.append((label, xs, ys))
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=PAGE)
    style_axes(ax, "training tokens (pool total)", "validation loss (nats/char)")
    if baseline_band and os.path.exists(baseline_band):
        with open(baseline_band) as f:
            b = json.load(f)
        ax.axhspan(b["val_loss_mean"], b["within_5pct_threshold"], color=SERIES[0], alpha=0.06, zorder=1)
        ax.axhline(b["val_loss_mean"], color=AXIS, linewidth=1, zorder=2)
        ax.annotate(f"control {b['val_loss_mean']:.3f} ± {b['val_loss_std']:.3f} (shaded: within 5%)",
                    (0.99, b["within_5pct_threshold"]), xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", fontsize=8, color=INK2, va="bottom", ha="right")
    plot_curves(ax, series)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    lo = min(min(ys) for _, _, ys in series)
    ax.set_ylim(max(0.9, lo - 0.08), 3.0)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, facecolor=PAGE)
    print("wrote", out_png)


def fig_sweep(specs: list[str], out_png: str, title: str, refs: list[str], baseline_band: str | None = None):
    """- Headline figure: left = val loss vs tokens, one line per K (plus reference runs in gray)"""
    series, bars = [], []
    for spec in specs:
        label, path = spec.rsplit("=", 1)
        xs, ys, m = read_run(path)
        series.append((label, xs, ys))
        if m:
            bars.append((label, m["bytes_total"], m["val_loss"], m.get("steps", 0), m.get("run_config", {}).get("local_steps")))
    ref_series = []
    for spec in refs:
        label, path = spec.rsplit("=", 1)
        xs, ys, m = read_run(path)
        if m and "val_loss" in m:
            label = f"{label} ({m['val_loss']:.3f})"
        ref_series.append((label, xs, ys))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=PAGE, gridspec_kw={"width_ratios": [1.6, 1]})
    style_axes(ax, "training tokens (pool total)", "validation loss (nats/char)")
    for i, (label, xs, ys) in enumerate(ref_series):   # references: recessive gray, dashed, no markers, legend only
        ax.plot(xs, ys, color=AXIS if i == 0 else MUTED, linewidth=1.5, linestyle=(0, (4, 3)) if i == 0 else (0, (1, 2)),
                zorder=2, label=label)
    plot_curves(ax, series)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    lo = min(min(ys) for _, _, ys in series + ref_series)
    ax.set_ylim(max(1.3, lo - 0.1), 3.0)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)

    # bytes: one bar per K (+ K=1 theoretical), log scale, direct labels
    style_axes(bx, "", "bytes moved per run")
    bx.set_yscale("log")
    labels, vals, colors = [], [], []
    if bars:
        n_workers = 4
        steps = bars[0][3]
        k1 = (steps / n_workers) * n_workers * 2 * 3.25e6
        labels.append("K=1 (theory)"); vals.append(k1); colors.append(AXIS)
    for i, (label, b, v, _, k) in enumerate(bars):
        labels.append(label); vals.append(b); colors.append(SERIES[i % len(SERIES)])
    x = range(len(labels))
    bx.bar(x, vals, color=colors, width=0.6, zorder=3)
    for xi, v in zip(x, vals):
        s = f"{v/1e9:.1f} GB" if v >= 1e9 else f"{v/1e6:.0f} MB"
        bx.annotate(s, (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    bx.set_xticks(list(x)); bx.set_xticklabels(labels, fontsize=8, color=MUTED)
    bx.grid(True, axis="y", which="major", color=GRID, linewidth=1)
    bx.set_title("Communication per run", color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, facecolor=PAGE)
    print("wrote", out_png)


def fig_nodes(specs: list[str], out_png: str, title: str, control: str | None = "results/baseline_summary.json"):
    """- Worker-count figure: final val loss vs N for the pool and for its synchronous reference at the same tokens."""
    ns, pool, sync = [], [], []
    for spec in specs:
        parts = dict(kv.split("=", 1) for kv in spec.split(":"))
        n = int(parts["N"]); ns.append(n)
        pool.append(read_run(parts["pool"])[2]["val_loss"])
        sync.append(read_run(parts["sync"])[2]["val_loss"] if parts.get("sync") and os.path.exists(parts["sync"]) else None)
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=PAGE)
    style_axes(ax, "workers (N)", "final validation loss (nats/char)")
    if control and os.path.exists(control):
        with open(control) as f:
            b = json.load(f)
        ax.axhline(b["val_loss_mean"], color=AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(f"single-machine control {b['val_loss_mean']:.3f}", (0.99, b["val_loss_mean"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK2)
    series = [("pool (local SGD, K=25)", ns, pool)]
    if any(s is not None for s in sync):
        xs = [n for n, s in zip(ns, sync) if s is not None]; ys = [s for s in sync if s is not None]
        series.append(("synchronous, same tokens & lr", xs, ys))
    for i, (label, xs, ys) in enumerate(series):
        c = SERIES[i]
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 8 if i == 0 else -14), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(ns)
    ax.margins(y=0.15)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, facecolor=PAGE)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    nd = sub.add_parser("nodes", help="'N=2:pool=results/nodes_n2:sync=results/baseline_bs128_lr4e-3' ...")
    nd.add_argument("specs", nargs="+")
    nd.add_argument("--out", required=True)
    nd.add_argument("--title", default="")
    s = sub.add_parser("sweep", help="headline K figure: 'K=25=results/run' ... --ref 'control=results/baseline'")
    s.add_argument("specs", nargs="+")
    s.add_argument("--ref", action="append", default=[])
    s.add_argument("--out", required=True)
    s.add_argument("--title", default="")
    b = sub.add_parser("baseline")
    b.add_argument("--out", default="results/baseline_loss.png")
    c = sub.add_parser("curves")
    c.add_argument("specs", nargs="+", help="label=path/to/log.jsonl")
    c.add_argument("--out", required=True)
    c.add_argument("--title", default="")
    k = sub.add_parser("compare", help="distributed run(s) vs baseline: label=results/<run> ...")
    k.add_argument("specs", nargs="+")
    k.add_argument("--out", required=True)
    k.add_argument("--title", default="")
    k.add_argument("--baseline-band", default="results/baseline_summary.json")
    args = ap.parse_args()
    if args.cmd == "nodes":
        fig_nodes(args.specs, args.out, args.title)
    elif args.cmd == "sweep":
        fig_sweep(args.specs, args.out, args.title, args.ref)
    elif args.cmd == "baseline":
        fig_baseline(args.out)
    elif args.cmd == "curves":
        fig_curves(args.specs, args.out, args.title)
    else:
        fig_compare(args.specs, args.out, args.title, args.baseline_band)


if __name__ == "__main__":
    main()
