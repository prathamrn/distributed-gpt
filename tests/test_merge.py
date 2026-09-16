import math

import pytest
import torch

from dgpt.merge import (OuterOptimizer, RoundView, adaptive_local_steps, aggregate, coordinate_median,
                   delta_of, round_timeout, should_close_round, trimmed_mean, weighted_average)


def sd(*vals):
    return {"a": torch.tensor(vals, dtype=torch.float32), "b": torch.tensor([[vals[0]]], dtype=torch.float32)}


def test_weighted_average_equal_weights_is_mean():
    out = weighted_average([sd(1.0, 2.0), sd(3.0, 4.0)], [1, 1])
    assert torch.allclose(out["a"], torch.tensor([2.0, 3.0]))


def test_weighted_average_by_steps():
    # 500-step worker counts 5x a 100-step worker
    out = weighted_average([sd(0.0, 0.0), sd(6.0, 6.0)], [500, 100])
    assert torch.allclose(out["a"], torch.tensor([1.0, 1.0]))


def test_weighted_average_casts_bf16_to_fp32():
    d = {"a": torch.tensor([1.0, 2.0]).to(torch.bfloat16), "b": torch.tensor([[1.0]]).to(torch.bfloat16)}
    out = weighted_average([d, d], [1, 1])
    assert out["a"].dtype == torch.float32


def test_median_ignores_single_outlier():
    good = [sd(1.0, 1.0), sd(1.1, 0.9), sd(0.9, 1.1)]
    bad = sd(1000.0, -1000.0)
    out = coordinate_median(good + [bad])
    assert out["a"].abs().max() < 1.2


def test_trimmed_mean_drops_extremes():
    ds = [sd(float(v), 0.0) for v in [1, 2, 3, 4, 100]]
    out = trimmed_mean(ds, trim_fraction=0.2)   # drops 1 low + 1 high => mean(2,3,4)
    assert math.isclose(out["a"][0].item(), 3.0)


def test_aggregate_dispatch():
    ds = [sd(1.0, 1.0), sd(3.0, 3.0)]
    assert torch.allclose(aggregate(ds, [1, 1], "mean")["a"], torch.tensor([2.0, 2.0]))
    with pytest.raises(ValueError):
        aggregate(ds, [1, 1], "nope")


def test_outer_lr1_no_momentum_single_worker_reproduces_local_weights():
    """The invariant from the interview: one worker, lr=1, no momentum => global == W_local."""
    start = sd(0.5, -0.25)
    local = sd(0.1, 0.3)
    opt = OuterOptimizer(lr=1.0, momentum=0.0)
    new = opt.step(start, delta_of(start, local))
    for k in start:
        assert torch.allclose(new[k], local[k])


def test_outer_momentum_amplifies_consistent_direction():
    """Same delta every round with mu=0.9: the step grows toward g/(1-mu)."""
    w = sd(0.0, 0.0)
    g = sd(1.0, 1.0)
    opt = OuterOptimizer(lr=1.0, momentum=0.9, nesterov=False)
    prev = w["a"].clone()
    steps = []
    for _ in range(30):
        w = opt.step(w, g)
        steps.append((prev - w["a"])[0].item()); prev = w["a"].clone()
    assert steps[0] == pytest.approx(1.0)
    assert steps[-1] > 9.0 and steps[-1] < 10.0        # -> 1/(1-0.9) = 10


def test_outer_momentum_damps_alternating_direction():
    """Flip-flopping deltas: steady-state step magnitude -> 1/(1+mu) = 0.53, i.e. damped below 1,
    whereas a consistent direction is amplified toward 10 (previous test)."""
    w = sd(0.0, 0.0)
    opt = OuterOptimizer(lr=1.0, momentum=0.9, nesterov=False)
    prev = w["a"].clone(); steps = []
    for i in range(60):
        w = opt.step(w, sd(1.0 if i % 2 == 0 else -1.0, 0.0))
        steps.append(abs((prev - w["a"])[0].item())); prev = w["a"].clone()
    assert max(steps[-10:]) < 0.6 and min(steps[-10:]) > 0.45


def test_outer_state_roundtrip():
    opt = OuterOptimizer(lr=0.7, momentum=0.9)
    opt.step(sd(0.0, 0.0), sd(1.0, 2.0))
    opt2 = OuterOptimizer(); opt2.load_state_dict(opt.state_dict())
    w1 = opt.step(sd(0.0, 0.0), sd(1.0, 2.0)); w2 = opt2.step(sd(0.0, 0.0), sd(1.0, 2.0))
    assert torch.allclose(w1["a"], w2["a"])


def test_delta_sign_convention():
    d = delta_of(sd(1.0, 1.0), sd(0.0, 2.0))
    assert torch.allclose(d["a"], torch.tensor([1.0, -1.0]))


# ---- round closing rule -------------------------------------------------------

def view(**kw):
    base = dict(alive={"w1", "w2", "w3"}, reported=set(), elapsed_s=0.0, min_workers=1, timeout_s=60.0)
    base.update(kw)
    return RoundView(**base)


def test_never_closes_with_zero_deltas():
    assert should_close_round(view(elapsed_s=999))[0] is False


def test_closes_when_all_alive_reported():
    assert should_close_round(view(reported={"w1", "w2", "w3"}))[0] is True


def test_dead_worker_does_not_block_close():
    assert should_close_round(view(reported={"w1", "w2"}, alive={"w1", "w2"}))[0] is True


def test_slow_alive_worker_blocks_until_timeout():
    """A straggler still on an older version is alive and unreported: the round waits for it."""
    ok, why = should_close_round(view(reported={"w1", "w2"}, elapsed_s=10))
    assert ok is False and "w3" in why
    ok, why = should_close_round(view(reported={"w1", "w2"}, elapsed_s=61))
    assert ok is True and "w3" in why


def test_timeout_respects_min_workers():
    assert should_close_round(view(reported={"w1"}, elapsed_s=61, min_workers=1))[0] is True
    assert should_close_round(view(reported={"w1"}, elapsed_s=61, min_workers=2))[0] is False


def test_round_timeout_uses_median_with_floor_and_initial():
    assert round_timeout([], 1.5, 5, 120) == 120
    assert round_timeout([10.0], 1.5, 5, 120) == 120
    assert round_timeout([10.0, 12.0, 100.0], 1.5, 5, 120) == pytest.approx(18.0)
    assert round_timeout([1.0, 1.0], 1.5, 5, 120) == 5


def test_adaptive_k():
    speeds = {"fast": 4.0, "mid": 2.0, "slow": 0.5}
    assert adaptive_local_steps(100, speeds, "mid") == 100
    assert adaptive_local_steps(100, speeds, "fast") == 200
    assert adaptive_local_steps(100, speeds, "slow") == 25
    assert adaptive_local_steps(100, speeds, "unknown") == 100
    assert adaptive_local_steps(100, {}, "fast") == 100


def test_adaptive_k_subtracts_transfer_overhead():
    # median worker trains K=100 at 2 steps/s = 50 s; a fast worker whose WAN transfers cost 25 s more than the
    # median's gets 25 s less training, so its whole cycle still lands on the same target
    speeds = {"fast": 4.0, "mid": 2.0, "slow": 0.5}
    overhead = {"fast": 30.0, "mid": 5.0, "slow": 5.0}
    assert adaptive_local_steps(100, speeds, "mid", overhead_s=overhead) == 100
    assert adaptive_local_steps(100, speeds, "fast", overhead_s=overhead) == 100        # 4 * (50 + 5 - 30)
    assert adaptive_local_steps(100, speeds, "slow", overhead_s=overhead) == 25
    # transfers that eat the whole budget: still min_frac of the training target, never 1 step
    assert adaptive_local_steps(100, speeds, "fast", overhead_s={"fast": 80.0, "mid": 5.0, "slow": 5.0}) == 20  # 4 * 0.1 * 50
    # unmeasured overhead counts as zero; the PRD formula is unchanged
    assert adaptive_local_steps(100, speeds, "fast", overhead_s={"mid": None}) == 200
