"""Pure functions for the coordinator: aggregation, outer optimizer, round rule.

No I/O, no FastAPI, no globals. Everything here is unit-tested in tests/.
State dicts are plain {name: float32 tensor} mappings.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import torch

StateDict = dict[str, torch.Tensor]


# ---- aggregation -------------------------------------------------------------

def weighted_average(deltas: list[StateDict], weights: list[float]) -> StateDict:
    """Sum_i w_i * delta_i / Sum_i w_i, computed in float32."""
    assert deltas and len(deltas) == len(weights)
    total = float(sum(weights))
    assert total > 0, "weights must sum to > 0"
    out: StateDict = {}
    for k in deltas[0]:
        acc = torch.zeros_like(deltas[0][k], dtype=torch.float32)
        for d, w in zip(deltas, weights):
            acc.add_(d[k].to(torch.float32), alpha=w / total)
        out[k] = acc
    return out


def coordinate_median(deltas: list[StateDict]) -> StateDict:
    """Coordinate-wise median. One outlier cannot move it (PRD 7.11)."""
    return {k: torch.stack([d[k].to(torch.float32) for d in deltas]).median(dim=0).values for k in deltas[0]}


def trimmed_mean(deltas: list[StateDict], trim_fraction: float = 0.1) -> StateDict:
    """Coordinate-wise mean after dropping the top and bottom `trim_fraction` of values."""
    n = len(deltas)
    k = int(n * trim_fraction)
    out: StateDict = {}
    for name in deltas[0]:
        s = torch.stack([d[name].to(torch.float32) for d in deltas]).sort(dim=0).values
        out[name] = s[k:n - k].mean(dim=0) if n - 2 * k > 0 else s.mean(dim=0)
    return out


def aggregate(deltas: list[StateDict], weights: list[float], method: str = "mean", trim_fraction: float = 0.1) -> StateDict:
    if method == "mean":
        return weighted_average(deltas, weights)
    if method == "median":
        return coordinate_median(deltas)
    if method == "trimmed":
        return trimmed_mean(deltas, trim_fraction)
    raise ValueError(f"unknown aggregation {method}")


# ---- outer optimizer (DiLoCo: SGD + Nesterov momentum on the averaged delta) --

@dataclass
class OuterOptimizer:
    """W_{t+1} = W_t - lr * step, where the averaged delta plays the role of a gradient.

    momentum=0, lr=1 with a single worker reproduces that worker's local weights
    exactly (tested). Nesterov: v = mu*v + g ; step = g + mu*v. Plain: step = v.
    """
    lr: float = 0.7
    momentum: float = 0.9
    nesterov: bool = True
    buf: StateDict = field(default_factory=dict)

    def step(self, weights: StateDict, avg_delta: StateDict) -> StateDict:
        new: StateDict = {}
        for k, w in weights.items():
            g = avg_delta[k].to(torch.float32)
            if self.momentum > 0:
                v = self.buf.get(k)
                v = g.clone() if v is None else v.mul(self.momentum).add_(g)
                self.buf[k] = v
                s = g + self.momentum * v if self.nesterov else v
            else:
                s = g
            new[k] = w.to(torch.float32) - self.lr * s
        return new

    def state_dict(self) -> dict:
        return {"lr": self.lr, "momentum": self.momentum, "nesterov": self.nesterov,
                "buf": {k: v.clone() for k, v in self.buf.items()}}

    def load_state_dict(self, d: dict) -> None:
        self.lr, self.momentum, self.nesterov = d["lr"], d["momentum"], d["nesterov"]
        self.buf = {k: v.clone() for k, v in d["buf"].items()}


def delta_of(start: StateDict, local: StateDict) -> StateDict:
    """delta = W_start - W_local (gradient sign convention)."""
    return {k: start[k].to(torch.float32) - local[k].to(torch.float32) for k in start}


# ---- round closing rule (PRD 7.6 / 7.9) ---------------------------------------

@dataclass
class RoundView:
    """Everything the closing rule needs, with no reference to live state."""
    alive: set[str]               # registered workers with a recent heartbeat
    reported: set[str]            # workers whose delta was accepted this round
    elapsed_s: float              # since the round opened
    min_workers: int
    timeout_s: float


def should_close_round(v: RoundView) -> tuple[bool, str]:
    """PRD 7.6/7.9: close when every alive worker reported, or on timeout with >= min_workers.
    A worker still training an older version is alive and NOT reported, so the round
    waits for it until the timeout; the heartbeat tells it the deadline so it can upload
    a partial delta in time. Never closes with zero deltas."""
    if not v.reported:
        return False, "no deltas yet"
    waiting_on = v.alive - v.reported
    if not waiting_on:
        return True, "all alive workers reported"
    if v.elapsed_s >= v.timeout_s and len(v.reported) >= v.min_workers:
        return True, f"timeout ({v.elapsed_s:.0f}s >= {v.timeout_s:.0f}s), merged without {sorted(waiting_on)}"
    return False, f"waiting on {sorted(waiting_on)}"


def round_timeout(recent_round_times: list[float], factor: float, floor_s: float, initial_s: float) -> float:
    """factor * median of recent full-round durations, with a floor; `initial_s` until there is history."""
    if len(recent_round_times) < 2:
        return initial_s
    return max(floor_s, factor * statistics.median(recent_round_times))


def adaptive_local_steps(base_k: int, worker_steps_per_s: dict[str, float], worker_id: str, k_min: int = 1,
                         overhead_s: dict[str, float] | None = None, min_frac: float = 0.1) -> int:
    """Steps for one worker this round, chosen so every worker's *cycle* (download + train + upload) lands on the
    same target (PRD 7.9 straggler adaptation, made transfer-aware).

    train_target = base_k / median_speed          seconds the median worker spends training K steps
    cycle_target = train_target + median_overhead the median worker's whole cycle
    budget_i     = cycle_target - overhead_i      training time left for worker i after its own transfers
    K_i          = speed_i * budget_i

    With zero overhead this is the PRD formula K * speed_i / median_speed. A worker whose transfers eat the
    whole budget still gets `min_frac` of the training target so it keeps contributing (and keeps being measured)
    instead of being cut to one step and arriving stale. Workers without a measured speed get base_k."""
    speeds = [s for s in worker_steps_per_s.values() if s and s > 0]
    mine = worker_steps_per_s.get(worker_id)
    if not speeds or not mine or mine <= 0:
        return base_k
    train_target = base_k / statistics.median(speeds)
    overheads = {w: o for w, o in (overhead_s or {}).items() if o is not None and w in worker_steps_per_s}
    med_overhead = statistics.median(overheads.values()) if overheads else 0.0
    budget = train_target + med_overhead - overheads.get(worker_id, 0.0)
    budget = max(budget, min_frac * train_target)
    return max(k_min, int(round(mine * budget)))
