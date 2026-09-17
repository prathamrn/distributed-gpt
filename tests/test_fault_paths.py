"""Fault-tolerance paths exercised directly on the Coordinator object (no HTTP, no processes"""
import os
import pytest
import torch

from dgpt.config import RunConfig, TrainConfig
from dgpt.coordinator import Coordinator
from dgpt.protocol import DeltaMeta, RegisterRequest, pack, unpack

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "data", "tinyshakespeare", "train.bin")), reason="dataset not tokenized")


class Clock:
    """- A callable fake clock the tests advance by hand, monkeypatched over the coordinator's time.time.
        - Liveness is defined in seconds (a worker silent for dead_after_s is dead)"""
    def __init__(self):
        self.t = 1000.0
    def __call__(self):
        """- Return the current fake time; tests move it forward by assigning to .t."""
        return self.t


@pytest.fixture
def coord(tmp_path, monkeypatch):
    """- A real Coordinator on a fake clock, checkpointing into tmp_path, with the round timeout effectively off."""
    clock = Clock()
    import dgpt.coordinator as C
    monkeypatch.setattr(C.time, "time", clock)
    c = Coordinator(RunConfig(run_name="paths", checkpoint_dir=str(tmp_path), total_steps=10_000, local_steps=5, n_shards=3,
                              round_timeout_initial_s=1e9, loss_check=True, loss_check_margin=0.05), TrainConfig())
    c.clock = clock
    return c


def register(c, name):
    """- Register a worker by name and return the shard it was given, skipping the HTTP layer."""
    return c.register(RegisterRequest(worker_id=name, gpt_config_hash=None)).shard_id


def fetch(c, name):
    """- Fetch the current weights as a worker would and return just the metadata (version and weights hash)."""
    _, meta = unpack(c.weights_body(name, None))
    return meta


def send(c, name, meta, delta, n_steps=5):
    """- Upload a delta the way a worker does: packed tensors plus a DeltaMeta quoting the version it trained from."""
    body = pack(delta, DeltaMeta(worker_id=name, version=meta["version"], weights_hash=meta["weights_hash"], n_steps=n_steps,
                                 n_tokens=n_steps, shard_id=0, round_wall_s=1.0).model_dump(), "float32")
    return c.delta(body)


def zeros(c):
    """- An all-zero delta with the model's exact layout: a well-formed upload that changes nothing."""
    return {k: torch.zeros_like(v) for k, v in c.weights.items()}


# ---- shards ---------------------------------------------------------------------------------------

def test_shards_are_handed_out_freed_and_reused(coord):
    """- A dead worker's shard must be released and handed to the next machine that joins"""
    c = coord
    assert [register(c, w) for w in ("a", "b", "c")] == [0, 1, 2]
    # b dies: its shard is freed and the next joiner gets it
    c.clock.t += 100
    c.workers["a"]["last_seen"] = c.workers["c"]["last_seen"] = c.clock.t
    with c.lock:
        c._reap_dead()
    assert "b" not in c.workers and c.shard_owner[1] is None
    assert register(c, "d") == 1


def test_fourth_worker_shares_the_least_loaded_shard(coord):
    """- More volunteers than shards is the normal case in an open pool"""
    c = coord
    assert [register(c, w) for w in ("a", "b", "c")] == [0, 1, 2]
    assert register(c, "d") in (0, 1, 2)            # every shard taken: share one
    load = {}
    for w, i in c.workers.items():
        load[i["shard"]] = load.get(i["shard"], 0) + 1
    assert sorted(load.values()) == [1, 1, 2]
    assert register(c, "e") != c.workers["d"]["shard"]   # the next one goes to a shard with a single owner


def test_reregistering_worker_keeps_its_shard_after_coordinator_restart(coord, tmp_path):
    """- After a resume every shard is unowned, but a returning worker gets its old shard back."""
    c = coord
    register(c, "a"); register(c, "b")
    with c.lock:
        c._checkpoint()
    c2 = Coordinator(RunConfig(run_name="paths", checkpoint_dir=str(tmp_path), n_shards=3), TrainConfig(), resume=True)
    assert c2.restarts == 1 and all(o is None for o in c2.shard_owner.values())
    assert c2.register(RegisterRequest(worker_id="b", gpt_config_hash=None)).shard_id == 1   # its old shard, not shard 0


# ---- delta admission ------------------------------------------------------------------------------

def test_malicious_delta_is_rejected_by_the_loss_check(coord):
    """- With loss_check on, a noise delta that worsens the held-out batch is rejected; an honest one is not."""
    c = coord
    register(c, "evil"); meta = fetch(c, "evil")
    scale = float(torch.cat([v.flatten() for v in c.weights.values()]).std()) * 20
    garbage = {k: torch.randn_like(v) * scale for k, v in c.weights.items()}   # exactly what --malicious sends
    r = send(c, "evil", meta, garbage)
    assert r.status == "rejected" and "held-out loss" in r.detail
    assert c.rejected_count == 1 and "evil" not in c.round["deltas"]
    assert send(c, "evil", meta, zeros(c)).status == "accepted"                # an honest delta from the same worker is fine


def test_wrong_layout_and_unknown_worker_and_stale_are_rejected(coord):
    """- Wrong tensor layout, unknown worker and stale version each get their own rejection status."""
    c = coord
    register(c, "a"); meta = fetch(c, "a")
    bad = zeros(c); k = next(iter(bad)); bad[k] = torch.zeros(3)
    assert send(c, "a", meta, bad).status == "bad_layout"
    assert send(c, "ghost", meta, zeros(c)).status == "unknown_worker"
    assert send(c, "a", meta, zeros(c)).status == "accepted"
    with c.lock:
        c._merge("test")
    assert send(c, "a", meta, zeros(c)).status == "stale"                      # computed from the previous version
    assert c.stale_count == 1


def test_delta_signed_by_one_worker_but_claiming_another_is_rejected(coord):
    """- The worker_id inside a delta must match the signer, or the delta is rejected."""
    c = coord
    register(c, "a"); register(c, "b"); meta = fetch(c, "a")
    body = pack(zeros(c), DeltaMeta(worker_id="b", version=meta["version"], weights_hash=meta["weights_hash"], n_steps=5,
                                    n_tokens=5, shard_id=0).model_dump(), "float32")
    assert c.delta(body, auth_worker="a").status == "rejected"


# ---- liveness --------------------------------------------------------------------------------------

def test_dead_worker_is_told_to_reregister(coord):
    """- A worker that was reaped while frozen must learn about it from the heartbeat reply"""
    from dgpt.protocol import HeartbeatRequest
    c = coord
    register(c, "a")
    assert c.heartbeat(HeartbeatRequest(worker_id="a")).registered is True
    c.clock.t += 100
    with c.lock:
        c._reap_dead()
    assert c.heartbeat(HeartbeatRequest(worker_id="a")).registered is False
    assert register(c, "a") == 0                                                # and re-registering works


# ---- merging ---------------------------------------------------------------------------------------

def test_partial_deltas_are_weighted_by_steps(coord):
    """- Two deltas with 3 and 1 steps merge as a 3:1 weighted average, and global_step advances by 4."""
    c = coord
    register(c, "a"); register(c, "b")
    ma, mb = fetch(c, "a"), fetch(c, "b")
    before = {k: v.clone() for k, v in c.weights.items()}
    c.run.loss_check = False
    assert send(c, "a", ma, {k: torch.full_like(v, 1e-3) for k, v in c.weights.items()}, n_steps=3).status == "accepted"
    assert send(c, "b", mb, {k: torch.full_like(v, 7e-3) for k, v in c.weights.items()}, n_steps=1).status == "accepted"
    with c.lock:
        c._merge("test")
    # plain averaging weighted by steps: (3*1e-3 + 1*7e-3) / 4 = 2.5e-3, applied as W - avg
    for k in before:
        assert torch.allclose(c.weights[k], before[k] - 2.5e-3, atol=1e-6)
    assert c.global_step == 4 and c.version == 1
