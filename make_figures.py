"""All loss curves, grouped by the question each group answers. Writes figures/*.png + figures/README.md.

    python3 make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plots import AXIS, INK, INK2, MUTED, PAGE, SERIES, millions, plot_curves, read_run, style_axes

OUT = "figures"
CONTROL = json.load(open("results/baseline_summary.json"))


def curve(label: str, path: str, final: bool = True):
    xs, ys, m = read_run(path)
    if final and m and "val_loss" in m:
        label = f"{label} ({m['val_loss']:.3f})"
    elif final and ys:
        label = f"{label} ({ys[-1]:.3f}, stopped)"
    return label, xs, ys


def fig(name: str, title: str, series: list, refs: list = (), control_line: bool = True, ylim=None, note: str = ""):
    """series: colored curves (<= 6); refs: gray dashed curves (references)."""
    assert len(series) <= 6, name
    f, ax = plt.subplots(figsize=(9, 5), facecolor=PAGE)
    style_axes(ax, "training tokens (pool total)", "validation loss (nats/char)")
    styles = [(0, (4, 3)), (0, (1, 2)), (0, (6, 2, 1, 2))]
    for i, (label, xs, ys) in enumerate(refs):
        ax.plot(xs, ys, color=AXIS if i == 0 else MUTED, linewidth=1.5, linestyle=styles[i % 3], zorder=2, label=label)
    if control_line:
        ax.axhline(CONTROL["val_loss_mean"], color=AXIS, linewidth=0.8, zorder=1)
        ax.annotate(f"control {CONTROL['val_loss_mean']:.3f}", (0.995, CONTROL["val_loss_mean"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=MUTED)
    all_y = [y for _, _, ys in series + list(refs) for y in ys]
    ax.set_ylim(*(ylim or (max(0.9, min(all_y) - 0.06), min(3.2, max(all_y) + 0.05))))
    plot_curves(ax, series, end_labels=len(series) <= 4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=12)
    if note:
        f.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    f.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    path = os.path.join(OUT, name + ".png")
    f.savefig(path, dpi=180, facecolor=PAGE); plt.close(f)
    print("wrote", path)
    return path


R = "results/"
index = []

# 1. single-machine references
index.append((fig("01_single_machine_references", "Single-machine runs at 12.3M tokens: the control (3 seeds) and better-tuned machines",
    [curve("control seed 1337, batch 64, lr 1e-3", R+"baseline"), curve("control seed 1", R+"baseline_seed1"), curve("control seed 2", R+"baseline_seed2"),
     curve("batch 128, lr 4e-3, 1500 steps", R+"baseline_bs128_lr4e-3"), curve("batch 192, lr 4e-3, 1000 steps", R+"baseline_bs192_lr4e-3"),
     curve("batch 256, lr 4e-3, 750 steps", R+"baseline_bs256_lr4e-3")], control_line=False),
    "The PRD control and the synchronous references for N=2/3/4 (batch N×64, lr N×1e-3). Seed spread is 0.004; the tuned batch-128 and batch-192 machines beat the control."))

index.append((fig("02_large_batch_lr_scaling", "Large batch needs a larger learning rate: one machine at batch 256 and 512, 12.3M tokens",
    [curve("batch 256, lr 1e-3", R+"baseline_bs256_lr1e-3"), curve("batch 256, lr 2e-3", R+"baseline_bs256_lr2e-3"), curve("batch 256, lr 4e-3", R+"baseline_bs256_lr4e-3"),
     curve("batch 512, lr 4e-3", R+"baseline_bs512_lr4e-3"), curve("batch 512, lr 8e-3", R+"baseline_bs512_lr8e-3")],
    refs=[curve("control, batch 64, lr 1e-3", R+"baseline")]),
    "Why the first pool run missed: at the control's lr, batch 256 is +15%; linear scaling (4e-3) brings it to +2%. At batch 512 even the best lr is +14% and 8e-3 diverges."))

# 2. the first run and the lr fix
index.append((fig("03_first_pool_runs_and_lr", "The first 4-worker runs (K=25, DiLoCo outer): inner learning rate is most of the gap",
    [curve("pool, inner lr 1e-3", R+"k25_uniform"), curve("pool, inner lr 2e-3", R+"diag_lr2e-3"), curve("pool, inner lr 4e-3", R+"k25_lr4e-3")],
    refs=[curve("control", R+"baseline"), curve("synchronous, batch 256, lr 1e-3", R+"baseline_bs256_lr1e-3"), curve("synchronous, batch 256, lr 4e-3", R+"baseline_bs256_lr4e-3")]),
    "The pool at lr 1e-3 (1.901) tracks synchronous training at lr 1e-3 (1.805); scaling the inner lr to 4e-3 closes most of it (1.783). Remaining gap to sync 4e-3 (1.600) is the optimizer."))

# 3. data layout
index.append((fig("04_data_layout", "Data layout does not matter: contiguous vs interleaved shards vs no sharding (K=25, DiLoCo outer, lr 4e-3)",
    [curve("contiguous shards (non-IID)", R+"k25_lr4e-3"), curve("interleaved shards (IID)", R+"diag_interleaved"), curve("full overlap (no sharding)", R+"diag_full_overlap")],
    refs=[curve("synchronous, batch 256, lr 4e-3", R+"baseline_bs256_lr4e-3")]),
    "Three data layouts within 0.006 of each other. Worker disagreement comes from batch noise, not from which plays each worker holds."))

# 4. outer optimizer at K=25
index.append((fig("05_outer_optimizer_k25", "Outer optimizer at K=25, 30 rounds: DiLoCo's momentum vs plain averaging vs mild momentum",
    [curve("DiLoCo: lr 0.7, momentum 0.9", R+"k25_lr4e-3"), curve("lr 0.3, momentum 0.9", R+"diag_k25_lr03"), curve("plain averaging: lr 1, no momentum", R+"diag_plain_avg"),
     curve("lr 1, momentum 0.3", R+"outer_mu03"), curve("lr 1, momentum 0.5", R+"outer_mu05"), curve("lr 1, momentum 0.7", R+"outer_mu07")],
    refs=[curve("synchronous, batch 256, lr 4e-3", R+"baseline_bs256_lr4e-3")]),
    "Momentum 0.9 loses; 0.5 is the optimum at 30 rounds (1.671); plain averaging (1.721) is the safe default."))

# 5. K=5 variants
index.append((fig("06_k5_variants", "K=5 (150 rounds): momentum collapses, resetting the inner Adam is worse still",
    [curve("DiLoCo outer", R+"diag_k5"), curve("plain averaging", R+"diag_k5_plain"), curve("lr 1, momentum 0.5", R+"k5_mu05"),
     curve("DiLoCo + reset inner Adam", R+"diag_k5_reset"), curve("plain + reset inner Adam", R+"diag_k5_plain_reset")],
    refs=[curve("synchronous, batch 256, lr 4e-3", R+"baseline_bs256_lr4e-3")]),
    "With 150 rounds any outer momentum overshoots (2.138 at 0.9, 1.782 at 0.5) and plain averaging wins (1.667, +4% vs sync). Resetting Adam each round costs ~0.5 nats."))

# 6. reset inner Adam
index.append((fig("07_reset_inner_adam", "Resetting the worker's Adam state each round hurts at every K (plain averaging)",
    [curve("K=25, Adam persists", R+"diag_plain_avg"), curve("K=25, Adam reset each round", R+"diag_k25_plain_reset"),
     curve("K=5, Adam persists", R+"diag_k5_plain"), curve("K=5, Adam reset each round", R+"diag_k5_plain_reset")]),
    "The first steps after a reset have no second-moment estimate; at K=5 every step is one of those."))

# 7. K sweep equal tokens
index.append((fig("08_k_sweep_equal_tokens", "K sweep at equal tokens (4 workers × 750 steps, plain averaging, lr 4e-3)",
    [curve("K=5 (3.9 GB moved)", R+"diag_k5_plain"), curve("K=25 (781 MB)", R+"diag_plain_avg"), curve("K=100 (208 MB)", R+"sweep_k100")],
    refs=[curve("control, batch 64", R+"baseline"), curve("synchronous, batch 256, lr 4e-3", R+"baseline_bs256_lr4e-3")]),
    "Each 4× in K costs ~0.04 nats and saves 4× the bytes. K=1 would move 19.5 GB."))

# 8. K sweep equal wall-clock
index.append((fig("09_k_sweep_equal_wallclock", "K sweep at equal wall-clock (4 workers × 3000 steps, 49M tokens)",
    [curve("K=25 (3.1 GB)", R+"long_k25"), curve("K=100 (852 MB)", R+"long_k100"), curve("K=500 (182 MB)", R+"long_k500"), curve("K=500, DiLoCo outer (241 MB)", R+"long_k500_diloco")],
    refs=[curve("1 machine, 12,000 steps, same tokens", R+"baseline_12k"), curve("synchronous, batch 256, 3000 steps, same tokens", R+"baseline_bs256_12k_lr4e-3")]),
    "Given the control's wall-clock, the pool beats the control (1.567) and the single machine at the same 49M tokens (1.641, overfit). DiLoCo's momentum overshoots visibly mid-run."))

# 9. worker count
index.append((fig("10_worker_count_curves", "Worker count at equal tokens (K=25, plain averaging, inner lr = N × 1e-3)",
    [curve("N=2, lr 2e-3", R+"nodes_n2"), curve("N=3, lr 3e-3", R+"nodes_n3"), curve("N=4, lr 4e-3", R+"diag_plain_avg"),
     curve("N=8, lr 4e-3", R+"nodes_n8_lr4e-3"), curve("N=8, lr 8e-3", R+"nodes_n8")],
    refs=[curve("control, batch 64", R+"baseline")]),
    "More workers at equal tokens = fewer effective optimizer steps. N=8 is past the useful batch size for this model; see 11 for the synchronous comparison."))

index.append((fig("11_worker_count_vs_sync", "Worker count: pool vs synchronous training at the same tokens and learning rate",
    [curve("N=2 pool", R+"nodes_n2"), curve("N=3 pool", R+"nodes_n3"), curve("N=4 pool", R+"diag_plain_avg"), curve("N=8 pool", R+"nodes_n8_lr4e-3")],
    refs=[curve("N=2 sync (batch 128)", R+"baseline_bs128_lr4e-3"), curve("N=3 sync (batch 192)", R+"baseline_bs192_lr4e-3"), curve("N=4 sync (batch 256)", R+"baseline_bs256_lr4e-3")]),
    "Pool minus synchronous: +4.8% (N=2), +7.8% (N=3), +7.5% (N=4), +11% (N=8, sync 1.790 not drawn). The gap to synchronous grows slowly; the gap to the control grows fast."))

# 10. text8
index.append((fig("12_text8_10M_model", "text8, 10.7M parameters: equal tokens (12.3M) and the interrupted equal-wall-clock run",
    [curve("pool, 4 workers, K=25, 12.3M tokens", R+"big_k25"), curve("pool, 4 workers, K=100, to 49M tokens", R+"big_long_k100")],
    refs=[curve("1 machine, batch 64, 3000 steps", R+"baseline_big"), curve("synchronous, batch 256, 750 steps", R+"baseline_big_bs256_lr4e-3"), curve("1 machine, batch 64, 12,000 steps", R+"baseline_big_12k")],
    control_line=False),
    "Different dataset: not comparable to the Shakespeare charts. At 12.3M tokens the 10.7M model is step-limited (sync +30% vs control); the pool sits ~9% above sync. The K=100 wall-clock run was stopped at round 5 of 30."))

# README
with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write("# Figures\n\nEvery loss curve from the project, grouped by the question each chart answers. All Shakespeare charts share the\n"
            "same axes (validation nats/char vs pool tokens) and the thin line marks the control (1.567). Regenerate with `python3 make_figures.py`.\n\n")
    for path, what in index:
        f.write(f"## {os.path.basename(path)}\n\n![]({os.path.basename(path)})\n\n{what}\n\n")
print("wrote figures/README.md")


# ============ effect of K and N: ordered single-hue ramps so the ordering reads at a glance ============
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]     # sequential ramp steps 250/350/450/550/700 (light mode)


def ramp_fig(name, title, groups, ylabel="validation loss (nats/char)", note=""):
    """groups: list of panels; each panel = (panel_title, [(label, xs, ys)...] ordered light->dark, refs)."""
    n = len(groups)
    f, axes = plt.subplots(1, n, figsize=(9 * n if n == 1 else 6.4 * n, 5), facecolor=PAGE, squeeze=False)
    for ax, (ptitle, series, refs) in zip(axes[0], groups):
        style_axes(ax, "training tokens (pool total)", ylabel)
        styles = [(0, (4, 3)), (0, (1, 2)), (0, (6, 2, 1, 2))]
        for i, (label, xs, ys) in enumerate(refs):
            ax.plot(xs, ys, color=AXIS if i == 0 else MUTED, linewidth=1.5, linestyle=styles[i % 3], zorder=2, label=label)
        cols = BLUES[-len(series):] if len(series) <= len(BLUES) else BLUES
        for (label, xs, ys), c in zip(series, cols):
            ax.plot(xs, ys, color=c, linewidth=2.2, solid_capstyle="round", label=label, zorder=3)
            ax.plot(xs[-1], ys[-1], marker="o", markersize=7, markerfacecolor=c, markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=4)
        all_y = [y for _, _, ys in series + list(refs) for y in ys]
        ax.set_ylim(max(0.9, min(all_y) - 0.06), min(3.2, max(all_y) + 0.05))
        ax.xaxis.set_major_formatter(FuncFormatter(millions))
        ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
        ax.set_title(ptitle, color=INK, fontsize=11, loc="left", pad=10)
    f.suptitle(title, color=INK, fontsize=12.5, x=0.01, ha="left")
    if note:
        f.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    f.tight_layout(rect=(0, 0.03 if note else 0, 1, 0.95))
    path = os.path.join(OUT, name + ".png"); f.savefig(path, dpi=180, facecolor=PAGE); plt.close(f); print("wrote", path)
    return path


index.append((ramp_fig("13_effect_of_K", "Effect of K (steps between syncs): darker = larger K = fewer syncs = fewer bytes",
    [("Equal tokens: 4 workers × 750 steps (12.3M)",
      [curve("K=5, 3.9 GB", R+"diag_k5_plain"), curve("K=25, 781 MB", R+"diag_plain_avg"), curve("K=100, 208 MB", R+"sweep_k100")],
      [curve("control", R+"baseline"), curve("synchronous, batch 256", R+"baseline_bs256_lr4e-3")]),
     ("Equal wall-clock: 4 workers × 3000 steps (49M)",
      [curve("K=25, 3.1 GB", R+"long_k25"), curve("K=100, 852 MB", R+"long_k100"), curve("K=500, 182 MB", R+"long_k500")],
      [curve("1 machine, 12k steps", R+"baseline_12k"), curve("synchronous, batch 256, 3k steps", R+"baseline_bs256_12k_lr4e-3")])],
    note="Plain averaging, inner lr 4e-3. Larger K costs a little loss and saves bytes in proportion (K=1 would move 19.5 GB / 78 GB)."),
    "The K effect in one picture, both budgets. Larger K = darker line = worse loss by a small, smooth amount, and 4× fewer bytes per 4× K."))

index.append((ramp_fig("14_effect_of_N", "Effect of N (number of workers) at equal tokens: darker = more workers",
    [("Pool, K=25, plain averaging, inner lr = N × 1e-3 (N=8 at 4e-3)",
      [curve("N=2", R+"nodes_n2"), curve("N=3", R+"nodes_n3"), curve("N=4", R+"diag_plain_avg"), curve("N=8", R+"nodes_n8_lr4e-3")],
      [curve("control (N=1), batch 64", R+"baseline")]),
     ("Synchronous training at the same tokens and lr (what DDP would get)",
      [curve("N=2: batch 128", R+"baseline_bs128_lr4e-3"), curve("N=3: batch 192", R+"baseline_bs192_lr4e-3"), curve("N=4: batch 256", R+"baseline_bs256_lr4e-3"), curve("N=8: batch 512", R+"baseline_bs512_lr4e-3")],
      [curve("control (N=1), batch 64", R+"baseline")])],
    note="Same tokens split N ways = N× fewer optimizer steps. Left: the pool. Right: perfect synchronous training pays most of the same price."),
    "The N effect: more workers at equal tokens means fewer effective optimizer steps. The right panel shows synchronous training pays most of the same price, so most of the N effect is data parallelism itself, not the protocol."))


# ============ time and compute cost ============
def bars_fig(name, title, panels, note=""):
    """panels: list of (ptitle, ylabel, labels, values, colors, fmt, log)."""
    f, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), facecolor=PAGE, squeeze=False)
    for ax, (ptitle, ylabel, labels, values, colors, fmt, log) in zip(axes[0], panels):
        style_axes(ax, "", ylabel)
        x = range(len(labels))
        if log == "dots":                       # differences matter more than the zero baseline: points on a zoomed scale
            ax.plot(list(x), values, color=AXIS, linewidth=1, zorder=2)
            for xi, v, c in zip(x, values, colors):
                ax.plot(xi, v, marker="o", markersize=9, markerfacecolor=c, markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=3)
                ax.annotate(fmt(v), (xi, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
            span = max(values) - min(values)
            ax.set_ylim(min(values) - 0.25 * span, max(values) + 0.35 * span)
        else:
            ax.bar(x, values, color=colors, width=0.62, zorder=3)
            if log:
                ax.set_yscale("log")
            for xi, v in zip(x, values):
                ax.annotate(fmt(v), (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
            if not log:
                ax.set_ylim(0, max(values) * 1.18)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9, color=MUTED)
        ax.set_title(ptitle, color=INK, fontsize=11, loc="left", pad=10)
    f.suptitle(title, color=INK, fontsize=12.5, x=0.01, ha="left")
    if note:
        f.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    f.tight_layout(rect=(0, 0.04 if note else 0, 1, 0.94))
    path = os.path.join(OUT, name + ".png"); f.savefig(path, dpi=180, facecolor=PAGE); plt.close(f); print("wrote", path)
    return path


def metrics(run):
    return json.load(open(f"results/{run}/metrics.json"))

CONTAINER_2CPU_STEPS_PER_S = 3.5      # measured: 2-CPU container, 0.8M model, uniform roster (3.2–3.6 steps/s)
single_est_s = 3000 / CONTAINER_2CPU_STEPS_PER_S
runs_n = [("1 (est.)", None, 1, 2.0), ("2", "nodes_n2", 2, 2.0), ("3", "nodes_n3", 3, 2.0), ("4", "diag_plain_avg", 4, 2.0), ("8", "nodes_n8_lr4e-3", 8, 1.0)]
labels, wall, cost, loss, colors = [], [], [], [], []
for lab, run, n, cpus in runs_n:
    m = metrics(run) if run else None
    w = m["wall_time_s"] if m else single_est_s
    labels.append(f"N={lab}"); wall.append(w / 60); cost.append(n * cpus * w / 3600); loss.append(m["val_loss"] if m else CONTROL["val_loss_mean"])
    colors.append(AXIS if not run else SERIES[0])
index.append((bars_fig("15_time_and_cost_vs_N", "Worker count at equal tokens (12.3M): wall-clock falls, compute cost rises, loss rises",
    [("Wall-clock per run", "minutes", labels, wall, colors, lambda v: f"{v:.1f} min", False),
     ("Compute cost per run", "CPU-hours (workers × CPUs × time)", labels, cost, colors, lambda v: f"{v:.2f}", False),
     ("Final validation loss (vs control)", "nats/char", labels, loss, colors, lambda v: f"{v:.3f} ({100*(v/CONTROL['val_loss_mean']-1):+.0f}%)", "dots")],
    note=f"Containers: 2 CPUs each (N=8: 1 CPU each). N=1 is estimated from the measured container speed ({CONTAINER_2CPU_STEPS_PER_S} steps/s → {single_est_s/60:.1f} min); "
         "the host baseline uses a different BLAS and is not time-comparable. Wall-clock includes round overheads and host contention."),
    "Time and cost vs N at equal tokens. Four workers finish 3.2× faster than one for ~25% more total CPU time and +9.8% loss. Cost scales with N; wall-clock does not scale perfectly because rounds wait for the slowest worker."))

runs_k = [("K=5", "diag_k5_plain"), ("K=25", "diag_plain_avg"), ("K=100", "sweep_k100")]
labels = [k for k, _ in runs_k]; ms = [metrics(r) for _, r in runs_k]
k1_bytes = 750 * 4 * 2 * 3.25e6
index.append((bars_fig("16_time_bytes_vs_K", "K at equal tokens (4 workers × 750 steps): bytes fall with K, time barely moves on a fast network, loss rises slowly",
    [("Bytes moved per run", "bytes (log)", ["K=1 (theory)"] + labels, [k1_bytes] + [m["bytes_total"] for m in ms], [AXIS] + [SERIES[0]] * 3, lambda v: f"{v/1e9:.1f} GB" if v >= 1e9 else f"{v/1e6:.0f} MB", True),
     ("Wall-clock per run", "minutes", labels, [m["wall_time_s"] / 60 for m in ms], [SERIES[0]] * 3, lambda v: f"{v:.1f} min", False),
     ("Final validation loss (vs control)", "nats/char", labels, [m["val_loss"] for m in ms], [SERIES[0]] * 3, lambda v: f"{v:.3f} ({100*(v/CONTROL['val_loss_mean']-1):+.0f}%)", "dots")],
    note="On Docker's internal network a 3.25 MB delta uploads in ~30 ms, so K does not change wall-clock here. On a 20 Mbit home uplink each upload is 1.3 s: K=5 would spend 40% of its time uploading, K=100 under 3%."),
    "Time, bytes, and loss vs K. Bytes scale as 1/K exactly. Wall-clock is flat on a fast network; the note gives the home-uplink arithmetic where K decides everything."))

with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write("# Figures\n\nEvery loss curve from the project, grouped by the question each chart answers. All Shakespeare charts share the\n"
            "same axes (validation nats/char vs pool tokens) and the thin line marks the control (1.567). Regenerate with `python3 make_figures.py`.\n\n")
    for path, what in index:
        f.write(f"## {os.path.basename(path)}\n\n![]({os.path.basename(path)})\n\n{what}\n\n")
print("wrote figures/README.md")
