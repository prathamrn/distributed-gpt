"""A second instance registering with the same worker id supersedes the first: the old session gets 409."""
import os
import pytest
from fastapi.testclient import TestClient

from dgpt.config import RunConfig, TrainConfig
from dgpt.coordinator import Coordinator, build_app

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "data", "tinyshakespeare", "train.bin")), reason="dataset not tokenized")


def test_newer_instance_supersedes_older(tmp_path):
    coord = Coordinator(RunConfig(run_name="fence_test", checkpoint_dir=str(tmp_path), total_steps=10, local_steps=2, n_shards=1), TrainConfig())
    app = build_app(coord)
    with TestClient(app) as c:
        a = c.post("/register", json={"worker_id": "alice", "gpt_config_hash": None}).json()
        b = c.post("/register", json={"worker_id": "alice", "gpt_config_hash": None}).json()
        assert a["session"] and b["session"] and a["session"] != b["session"]
        old = {"X-Session": a["session"], "X-Worker": "alice"}
        new = {"X-Session": b["session"], "X-Worker": "alice"}
        r = c.post("/heartbeat", json={"worker_id": "alice"}, headers=old)
        assert r.status_code == 409 and "superseded" in r.text
        assert c.post("/heartbeat", json={"worker_id": "alice"}, headers=new).status_code == 200
        assert c.get("/weights", params={"worker_id": "alice"}, headers=old).status_code == 409
        assert c.get("/weights", params={"worker_id": "alice"}, headers=new).status_code == 200
        # a legacy worker that sends no session header is not fenced
        assert c.post("/heartbeat", json={"worker_id": "alice"}).status_code == 200
