"""Round timing is transfer-aware: the coordinator measures each worker's cycle (download + train + upload) from
the moment it served the weights, learns speed and overhead even from stale deltas, and sizes the timeout from
cycle times rather than training times."""
import os
import pytest
import torch

from dgpt.config import RunConfig, TrainConfig
from dgpt.coordinator import Coordinator
from dgpt.protocol import DeltaMeta, pack

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "data", "tinyshakespeare", "train.bin")), reason="dataset not tokenized")


class Clock:
    def __init__(self, t=1000.0):
        self.t = t
    def __call__(self):
        return self.t


@pytest.fixture
def coord(tmp_path, monkeypatch):
    clock = Clock()
    import dgpt.coordinator as C
    monkeypatch.setattr(C.time, "time", clock)
    c = Coordinator(RunConfig(run_name="timing", checkpoint_dir=str(tmp_path), total_steps=10_000, local_steps=100,
                              adaptive_k=True, n_shards=2, round_timeout_factor=2, round_timeout_floor_s=5,
                              round_timeout_initial_s=90, loss_check=False, dead_after_s=1e9), TrainConfig())
    c.clock = clock
    return c


def register(c, name):
    from dgpt.protocol import RegisterRequest
    c.register(RegisterRequest(worker_id=name, gpt_config_hash=None))


def fetch(c, name):
    from dgpt.protocol import unpack
    _, meta = unpack(c.weights_body(name, None))
    return meta


def send(c, name, meta, n_steps, train_s):
    delta = {k: torch.zeros_like(v) for k, v in c.weights.items()}
    body = pack(delta, DeltaMeta(worker_id=name, version=meta["version"], weights_hash=meta["weights_hash"], n_steps=n_steps,
                                 n_tokens=n_steps, shard_id=0, round_wall_s=train_s).model_dump(), "float32")
    return c.delta(body).status


def test_cycle_time_feeds_timeout_and_k(coord):
    c, clock = coord, coord.clock
    register(c, "mac"); register(c, "wan")
    m_mac = fetch(c, "mac"); m_wan = fetch(c, "wan")
    assert m_mac["local_steps"] == 100 and m_wan["local_steps"] == 100      # no speeds yet: base K
    clock.t += 10                                                           # mac: 100 steps in 10 s, no transfer time
    assert send(c, "mac", m_mac, 100, 10.0) == "accepted"
    clock.t += 24                                                           # wan: 34 s cycle, only 4 s of it training
    assert send(c, "wan", m_wan, 100, 4.0) == "accepted"
    w = c.workers
    assert w["mac"]["steps_per_s"] == 10 and w["mac"]["overhead_s"] == pytest.approx(0.0)
    assert w["wan"]["steps_per_s"] == 25 and w["wan"]["overhead_s"] == pytest.approx(30.0)
    assert sorted(c.round_times) == [10.0, 34.0]                            # cycle times, not training times
    assert c._timeout_s() == pytest.approx(2 * 22.0)                        # factor * median cycle
    # next round: the WAN worker's K is cut by its transfer overhead, the mac's grows by the median overhead
    with c.lock:
        c._merge("test")
    m_mac = fetch(c, "mac"); m_wan = fetch(c, "wan")
    # train target = 100 / median(10, 25) = 5.7 s; median overhead 15 s; cycle target 20.7 s
    assert m_wan["local_steps"] == round(25 * max(20.7 - 30, 0.1 * 5.7))    # 14 steps, not 1
    assert m_mac["local_steps"] == round(10 * 20.7)


def test_stale_delta_still_teaches_speed_and_overhead(coord):
    c, clock = coord, coord.clock
    register(c, "mac"); register(c, "slow")
    m_mac = fetch(c, "mac"); m_slow = fetch(c, "slow")
    clock.t += 10
    assert send(c, "mac", m_mac, 100, 10.0) == "accepted"
    with c.lock:
        c._merge("test")                                                    # round closed without `slow`
    clock.t += 30
    assert send(c, "slow", m_slow, 8, 28.0) == "stale"                      # too late, 8 of 100 steps
    s = c.workers["slow"]
    assert s["steps_per_s"] == pytest.approx(8 / 28) and s["overhead_s"] == pytest.approx(12.0)
    assert s["assigned_k"] == 100 and s["served_at"] is None
    # its planned cycle (overhead + K / speed = 362 s) enters the timeout history so the next timeout is honest
    assert max(c.round_times) == pytest.approx(12 + 100 / (8 / 28))
    # and its next K is sized to its speed and overhead instead of the base K
    m_slow = fetch(c, "slow")
    assert 1 <= m_slow["local_steps"] < 100


def test_tiny_partial_does_not_overwrite_a_real_speed(coord):
    c, clock = coord, coord.clock
    register(c, "w")
    m = fetch(c, "w"); clock.t += 10
    assert send(c, "w", m, 100, 10.0) == "accepted"
    with c.lock:
        c._merge("test")
    m = fetch(c, "w"); clock.t += 1
    assert send(c, "w", m, 1, 0.01) == "accepted"
    assert c.workers["w"]["steps_per_s"] == 10
