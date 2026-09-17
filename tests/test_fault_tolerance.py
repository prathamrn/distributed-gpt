"""End-to-end fault tolerance on a free port with real coordinator/worker processes (small model the run

     the run completes, and no version regression is served;
     re-registers by itself when it wakes;
     retries with backoff, is declared dead, and rejoins when the link returns; the run completes."""
import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest
import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "data", "tinyshakespeare", "train.bin")),
                                reason="tokenize the dataset first: python3 -m dgpt.data")


def free_port() -> int:
    """- Ask the OS for an unused port by binding to 0 and reading back what it assigned, then releasing it."""
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def start_coordinator(run, port, extra=()):
    """- Launch a real coordinator subprocess for this run and return once /health answers."""
    cmd = [sys.executable, "-m", "dgpt.coordinator", "--run-name", run, "--port", str(port), "--set", "local_steps=5",
           "--set", "total_steps=80", "--set", "n_shards=2", "--set", "round_timeout_initial_s=20", "--exit-when-done", *extra]
    p = subprocess.Popen(cmd, cwd=HERE, stdout=open(os.path.join(HERE, "results", f"{run}_coord.out"), "a"), stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
                return p
        except requests.ConnectionError:
            time.sleep(0.5)
    p.kill(); raise RuntimeError("coordinator did not start")


def start_worker(name, port):
    """- Launch a real worker subprocess against the coordinator, capped at one CPU thread."""
    return subprocess.Popen([sys.executable, "-m", "dgpt.worker", "--name", name, "--coordinator", f"http://127.0.0.1:{port}", "--threads", "1", "--device", "cpu"],
                            cwd=HERE, stdout=open(os.path.join(HERE, "results", f"ft_{name}.out"), "a"), stderr=subprocess.STDOUT)


def wait_for(pred, timeout, what):
    """- Poll a predicate twice a second until it holds, or fail with a message naming what was being waited for."""
    for _ in range(int(timeout * 2)):
        if pred():
            return
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what}")


def merges(run):
    """- Return the merge rows of a run's coordinator.jsonl, or an empty list if it does not exist yet."""
    p = os.path.join(HERE, "results", run, "coordinator.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if '"merge"' in l]


def test_worker_death_and_coordinator_restart():
    """- Two faults in one run: SIGKILL a worker, then SIGKILL the coordinator itself and relaunch it."""
    run = "ft_test"
    subprocess.run(["rm", "-rf", os.path.join(HERE, "results", run)])
    port = free_port()
    coord = start_coordinator(run, port, ["--fresh"])
    w1, w2 = start_worker("ft-w1", port), start_worker("ft-w2", port)
    try:
        # both workers contribute
        wait_for(lambda: len(merges(run)) >= 2, 40, "first two merges")
        assert merges(run)[-1]["n_deltas"] == 2

        # 1. kill a worker mid-run: the pool must keep merging without it and free its shard
        w2.kill(); w2.wait()
        n_before = len(merges(run))
        wait_for(lambda: len(merges(run)) >= n_before + 3, 40, "merges after worker death")
        dead = [json.loads(l) for l in open(os.path.join(HERE, "results", run, "coordinator.jsonl")) if '"worker_dead"' in l]
        assert dead and dead[0]["worker"] == "ft-w2"
        assert merges(run)[-1]["n_deltas"] == 1

        # 2. SIGKILL the coordinator mid-run and relaunch it: it must resume, not restart from 0
        v_before = merges(run)[-1]["version"]
        coord.send_signal(signal.SIGKILL); coord.wait()
        time.sleep(3)
        coord = start_coordinator(run, port)                  # no --fresh: auto-resume from the checkpoint
        st = requests.get(f"http://127.0.0.1:{port}/health", timeout=2).json()
        assert st["version"] >= v_before - 1                  # at most the in-flight round is lost

        # the surviving worker reconnects on its own and the run completes
        wait_for(lambda: os.path.exists(os.path.join(HERE, "results", run, "metrics.json")), 90, "run completion after restart")
        m = json.load(open(os.path.join(HERE, "results", run, "metrics.json")))
        assert m["steps"] >= 80 and m["restarts"] == 1
        versions = [r["version"] for r in merges(run)]
        assert versions == sorted(versions), "versions must never go backwards across a restart"
        w1.wait(timeout=30)
        assert w1.returncode == 0
    finally:
        for p in (w1, w2, coord):
            if p.poll() is None:
                p.kill()


def start_worker_via(name, port, extra=()):
    """- Launch a worker pointed at an arbitrary port, which is how one worker is routed through the chaos proxy."""
    return subprocess.Popen([sys.executable, "-m", "dgpt.worker", "--name", name, "--coordinator", f"http://127.0.0.1:{port}",
                             "--threads", "1", "--device", "cpu", *extra],
                            cwd=HERE, stdout=open(os.path.join(HERE, "results", f"ft_{name}.out"), "a"), stderr=subprocess.STDOUT)


def events(run, kind):
    """- Return the rows of one event kind (register, worker_dead, ...) from a run's coordinator.jsonl."""
    p = os.path.join(HERE, "results", run, "coordinator.jsonl")
    return [json.loads(l) for l in open(p) if f'"{kind}"' in l] if os.path.exists(p) else []


def test_paused_worker_is_reaped_and_rejoins():
    """- A SIGSTOPped worker is declared dead, the pool continues, and it re-registers and contributes after SIGCONT."""
    run = "ft_pause"
    subprocess.run(["rm", "-rf", os.path.join(HERE, "results", run)])
    port = free_port()
    coord = start_coordinator(run, port, ["--fresh", "--set", "total_steps=200"])
    w1, w2 = start_worker("ft-p1", port), start_worker("ft-p2", port)
    try:
        wait_for(lambda: len(merges(run)) >= 2, 40, "first two merges")
        w2.send_signal(signal.SIGSTOP)                       # frozen: no heartbeats, no training
        wait_for(lambda: any(d["worker"] == "ft-p2" for d in events(run, "worker_dead")), 30, "ft-p2 declared dead")
        n = len(merges(run))
        wait_for(lambda: len(merges(run)) >= n + 2, 40, "pool continues with one worker")
        assert merges(run)[-1]["n_deltas"] == 1
        w2.send_signal(signal.SIGCONT)
        regs = lambda: [r for r in events(run, "register") if r["worker"] == "ft-p2"]
        wait_for(lambda: len(regs()) >= 2, 40, "ft-p2 re-registers after waking")
        wait_for(lambda: os.path.exists(os.path.join(HERE, "results", run, "metrics.json")), 120, "run completion")
        assert any(m["n_deltas"] == 2 for m in merges(run)[-6:]), "ft-p2 should contribute again after rejoining"
    finally:
        for p in (w1, w2, coord):
            if p.poll() is None:
                p.kill()


def test_connection_cut_through_proxy():
    """- The failure that actually happened in this project, reproduced: the link dies while the process lives."""
    run = "ft_cut"
    subprocess.run(["rm", "-rf", os.path.join(HERE, "results", run)])
    port, via, ctl = free_port(), free_port(), free_port()
    coord = start_coordinator(run, port, ["--fresh", "--set", "total_steps=200"])
    proxy = subprocess.Popen([sys.executable, "scripts/chaos_proxy.py", "--upstream", f"127.0.0.1:{port}", "--via", f"ft-c2={via}", "--control", str(ctl)],
                             cwd=HERE, stdout=open(os.path.join(HERE, "results", "ft_proxy.out"), "a"), stderr=subprocess.STDOUT)
    wait_for(lambda: requests.get(f"http://127.0.0.1:{ctl}/status", timeout=1).ok if _up(ctl) else False, 15, "proxy control API")
    w1, w2 = start_worker("ft-c1", port), start_worker_via("ft-c2", via)
    try:
        wait_for(lambda: len(merges(run)) >= 2 and merges(run)[-1]["n_deltas"] == 2, 60, "both workers merging through the proxy")
        st = requests.get(f"http://127.0.0.1:{ctl}/status", timeout=2).json()["ft-c2"]
        assert st["bytes_down"] > 0 and st["bytes_up"] > 0
        # cut the link: the in-flight request fails, retries are refused, the worker is declared dead
        requests.get(f"http://127.0.0.1:{ctl}/set", params={"name": "ft-c2", "mode": "cut"}, timeout=2)
        wait_for(lambda: any(d["worker"] == "ft-c2" for d in events(run, "worker_dead")), 30, "ft-c2 declared dead")
        n = len(merges(run))
        wait_for(lambda: len(merges(run)) >= n + 2, 40, "pool continues with one worker")
        requests.get(f"http://127.0.0.1:{ctl}/set", params={"name": "ft-c2", "mode": "normal"}, timeout=2)
        regs = lambda: [r for r in events(run, "register") if r["worker"] == "ft-c2"]
        wait_for(lambda: len(regs()) >= 2, 40, "ft-c2 re-registers once the link is back")
        wait_for(lambda: os.path.exists(os.path.join(HERE, "results", run, "metrics.json")), 120, "run completion")
        log = open(os.path.join(HERE, "results", "ft_ft-c2.out")).read()
        assert "retry in" in log, "the worker should have retried through the outage, not crashed"
        assert w2.poll() is None or w2.returncode == 0
    finally:
        for p in (w1, w2, proxy, coord):
            if p.poll() is None:
                p.kill()


def _up(port):
    """- True if the proxy's control API on this port answers, used to wait for it before starting the worker."""
    try:
        return requests.get(f"http://127.0.0.1:{port}/status", timeout=1).ok
    except requests.ConnectionError:
        return False
