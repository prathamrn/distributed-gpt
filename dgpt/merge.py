"""Pure functions for the coordinator: aggregation, outer optimizer, round rule."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import torch

StateDict = dict[str, torch.Tensor]


# ---- aggregation -------------------------------------------------------------

def weighted_average(deltas: list[StateDict], weights: list[float]) -> StateDict:
    """- Sum_i w_i*delta_i / Sum_i w_i in float32; the default aggregation rule.
        - w_i is the step count, so 490 steps counts ~4.5x 110 steps and a straggler's partial still contributes.
        - fp32 accumulation keeps a bf16 pool coherent; called by aggregate(), feeds OuterOptimizer.step."""
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
    """- Coordinate-wise median of the deltas: a minority sending garbage cannot move it (PRD 7.11).
        - Selected with --set aggregation=median; it discards the step weights, so it discounts the busiest worker.
        - Reached through aggregate(); robustness here is a policy choice, not a free upgrade over the mean."""
    return {k: torch.stack([d[k].to(torch.float32) for d in deltas]).median(dim=0).values for k in deltas[0]}


def trimmed_mean(deltas: list[StateDict], trim_fraction: float = 0.1) -> StateDict:
    """- Coordinate-wise mean after dropping int(n*trim_fraction) values at each end; --set aggregation=trimmed.
        - The middle ground between mean and median: resists a few bad deltas, still ignores the step weights.
        - Falls back to the full mean when trimming would remove everything; reached through aggregate()."""
    n = len(deltas)
    k = int(n * trim_fraction)
    out: StateDict = {}
    for name in deltas[0]:
        s = torch.stack([d[name].to(torch.float32) for d in deltas]).sort(dim=0).values
        out[name] = s[k:n - k].mean(dim=0) if n - 2 * k > 0 else s.mean(dim=0)
    return out


def aggregate(deltas: list[StateDict], weights: list[float], method: str = "mean", trim_fraction: float = 0.1) -> StateDict:
    """- Dispatch to the rule named by RunConfig.aggregation; the coordinator's one entry point into this file.
        - One name behind three rules makes trusting-vs-robust a one-flag experiment; an unknown name raises here.
        - Called from Coordinator._merge; the returned delta goes straight into OuterOptimizer.step."""
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
    """- W_{t+1} = W_t - lr*step, with the averaged delta playing the role of a gradient (DiLoCo's outer SGD).
        - Nesterov: v = mu*v + g, step = g + mu*v; plain: step = v. lr 1, momentum 0, one worker => its weights.
        - RunConfig defaults to plain averaging: DiLoCo's (0.7, 0.9) lost at every K (R6: 1.783 vs 1.721 at K=25)."""
    lr: float = 0.7
    momentum: float = 0.9
    nesterov: bool = True
    buf: StateDict = field(default_factory=dict)

    def step(self, weights: StateDict, avg_delta: StateDict) -> StateDict:
        """- Apply one outer step; returns the next version's weights, which the coordinator hashes and serves.
                - Computed in float32 and out of place, so a crash mid-merge cannot leave a half-updated model.
                - With momentum 0 the velocity branch is skipped and the step is the averaged delta (the default)."""
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
        """- Everything needed to resume the outer optimizer; goes into results/<run>/ckpt.pt.
                - The buffer is cloned so a later step cannot mutate the saved copy.
                - The coordinator auto-resumes after a crash; forgetting the velocity would change the trajectory."""
        return {"lr": self.lr, "momentum": self.momentum, "nesterov": self.nesterov,
                "buf": {k: v.clone() for k, v in self.buf.items()}}

    def load_state_dict(self, d: dict) -> None:
        """- Inverse of state_dict, used when the coordinator resumes a run from ckpt.pt.
                - The outer hyperparameters come from the checkpoint too, so a resumed run keeps its original settings."""
        self.lr, self.momentum, self.nesterov = d["lr"], d["momentum"], d["nesterov"]
        self.buf = {k: v.clone() for k, v in d["buf"].items()}


def delta_of(start: StateDict, local: StateDict) -> StateDict:
    """- delta = W_start - W_local (gradient sign convention), in float32.
        - The worker calls this at the end of a round and packs the result into POST /delta.
        - This sign lets the average feed a standard subtract-lr-times-g rule unchanged (the anchor invariant)."""
    return {k: start[k].to(torch.float32) - local[k].to(torch.float32) for k in start}


# ---- round closing rule (PRD 7.6 / 7.9) ---------------------------------------

@dataclass
class RoundView:
    """- Snapshot of everything should_close_round needs, with no reference to live server state.
        - Built under the coordinator's lock each 250 ms tick, so every closing case is testable without FastAPI.
        - `alive` means heartbeating, not "fetched the current version"."""
    alive: set[str]               # registered workers with a recent heartbeat
    reported: set[str]            # workers whose delta was accepted this round
    elapsed_s: float              # since the round opened
    min_workers: int
    timeout_s: float


def should_close_round(v: RoundView) -> tuple[bool, str]:
    """- Close when every alive worker reported, or on timeout with >= min_workers (PRD 7.6/7.9); never on zero.
        - Waits on liveness, not participation: waiting on fetchers alone let one worker close every round (R2).
        - Returns (close?, reason); the reason lands in the merge row of the run's JSONL log."""
    if not v.reported:
        return False, "no deltas yet"
    waiting_on = v.alive - v.reported
    if not waiting_on:
        return True, "all alive workers reported"
    if v.elapsed_s >= v.timeout_s and len(v.reported) >= v.min_workers:
        return True, f"timeout ({v.elapsed_s:.0f}s >= {v.timeout_s:.0f}s), merged without {sorted(waiting_on)}"
    return False, f"waiting on {sorted(waiting_on)}"


def round_timeout(recent_round_times: list[float], factor: float, floor_s: float, initial_s: float) -> float:
    """- factor * median of recent full-cycle (download+train+upload) times, floored; initial_s until history.
        - Median not mean: the 100x-off worker is exactly what this bounds, and a mean would be dragged by it.
        - Recomputed each manager tick and sent as round_closes_in_s; whole cycles, not train time, after R16."""
    if len(recent_round_times) < 2:
        return initial_s
    return max(floor_s, factor * statistics.median(recent_round_times))


def adaptive_local_steps(base_k: int, worker_steps_per_s: dict[str, float], worker_id: str, k_min: int = 1,
                         overhead_s: dict[str, float] | None = None, min_frac: float = 0.1) -> int:
    """- Steps for one worker this round, sized so every worker's whole cycle hits the same target (PRD 7.9).
        - budget_i = base_k/median_speed + median_overhead - overhead_i; K_i = round(speed_i*budget_i) >= k_min.
        - Called from GET /weights when adaptive_k is on; transfer-aware after R16; no measured speed => base_k."""
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
