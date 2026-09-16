"""Inject failures into a running pool on a schedule (PRD 8): kill, pause/unpause, restart workers, add a
late joiner, restart the coordinator. Workers are local `dgpt.worker` processes found by their --name.
Logs every event to results/<run>/chaos.jsonl so plots can mark them.

    python3 scripts/chaos.py --run k25 --kill worker-3@60 --pause worker-1@90:20 --restart-coordinator@150 --late-join worker-5@200

Actions (time is seconds after chaos.py starts):
    --kill NAME@T                SIGKILL the worker (the coordinator reaps it after 15 s; its shard is freed)
    --pause NAME@T:DURATION      SIGSTOP for DURATION s, then SIGCONT (stalls heartbeats + training; it is declared
                                 dead if DURATION > 15 s and re-registers when it wakes)
    --restart NAME@T             SIGKILL, then relaunch the worker with its own command line
    --late-join NAME@T           start a new worker NAME against --coordinator (add --token if the pool needs one)
    --restart-coordinator@T      SIGKILL the coordinator and relaunch it with the command in results/<run>/coordinator.cmd
                                 (it auto-resumes from its last checkpoint; workers reconnect on their own)

Connection-level faults need the worker to talk through scripts/chaos_proxy.py (one port per worker); pass the
proxy's control URL with --proxy and the worker's proxy NAME:
    --cut NAME@T:DURATION        reset every open connection of NAME and refuse new ones for DURATION s
                                 (in-flight fetch/upload fails; the worker retries with backoff; > 15 s and it is
                                 declared dead and re-registers when the link returns)
    --lag NAME@T:DURATION:MS     add MS milliseconds of latency to every chunk for DURATION s
    --throttle NAME@T:DURATION:KBS  cap NAME's link at KBS kB/s for DURATION s (a slow tunnel: transfers dominate,
                                 adaptive K should shrink its step budget rather than cut it off)

    python3 scripts/chaos_proxy.py --upstream 127.0.0.1:8000 --via w1=8001 --control 8100
    python3 -m dgpt.worker --name w1 --coordinator http://127.0.0.1:8001
    python3 scripts/chaos.py --run k25 --proxy http://127.0.0.1:8100 --cut w1@30:20 --throttle w1@90:60:300
"""
from __future__ import annotations

import argparse
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import signal
import subprocess
import sys
import time


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def worker_pid(name: str) -> int | None:
    for pid in sh(["pgrep", "-f", "worker"]).split():
        args = sh(["ps", "-o", "args=", "-p", pid])
        if any(k in args for k in ("dgpt.worker", "worker.py", "dgpt-worker")) and f"--name {name}" in args and "chaos" not in args:
            return int(pid)
    return None


def worker_cmd(pid: int) -> list[str]:
    return sh(["ps", "-o", "args=", "-p", str(pid)]).split()


def coordinator_pid(run: str) -> int | None:
    for pid in sh(["pgrep", "-f", "coordinator"]).split():
        args = sh(["ps", "-o", "args=", "-p", pid])
        if any(k in args for k in ("dgpt.coordinator", "coordinator.py", "dgpt-coordinator")) and f"--run-name {run}" in args and "chaos" not in args:
            return int(pid)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--coordinator", default="http://127.0.0.1:8000", help="for --late-join")
    ap.add_argument("--token", default=None, help="for --late-join")
    ap.add_argument("--kill", action="append", default=[], metavar="NAME@T")
    ap.add_argument("--pause", action="append", default=[], metavar="NAME@T:DUR")
    ap.add_argument("--restart", action="append", default=[], metavar="NAME@T")
    ap.add_argument("--late-join", action="append", default=[], metavar="NAME@T")
    ap.add_argument("--restart-coordinator", action="append", default=[], metavar="@T")
    ap.add_argument("--proxy", default="http://127.0.0.1:8100", help="chaos_proxy.py control URL for --cut/--lag/--throttle")
    ap.add_argument("--cut", action="append", default=[], metavar="NAME@T:DUR")
    ap.add_argument("--lag", action="append", default=[], metavar="NAME@T:DUR:MS")
    ap.add_argument("--throttle", action="append", default=[], metavar="NAME@T:DUR:KBS")
    args = ap.parse_args()

    events = []
    for spec in args.kill:
        n, t = spec.split("@"); events.append((float(t), "kill", n, None))
    for spec in args.pause:
        n, rest = spec.split("@"); t, d = rest.split(":"); events.append((float(t), "pause", n, float(d)))
    for spec in args.restart:
        n, t = spec.split("@"); events.append((float(t), "restart", n, None))
    for spec in args.late_join:
        n, t = spec.split("@"); events.append((float(t), "late-join", n, None))
    for spec in args.restart_coordinator:
        events.append((float(spec.lstrip("@")), "restart-coordinator", "coordinator", None))
    for spec in args.cut:
        n, rest = spec.split("@"); t, d = rest.split(":"); events.append((float(t), "cut", n, (float(d), None)))
    for spec in args.lag:
        n, rest = spec.split("@"); t, d, ms = rest.split(":"); events.append((float(t), "lag", n, (float(d), float(ms))))
    for spec in args.throttle:
        n, rest = spec.split("@"); t, d, kbs = rest.split(":"); events.append((float(t), "throttle", n, (float(d), float(kbs))))
    events.sort()
    if not events:
        sys.exit("nothing scheduled")

    out_dir = os.path.join("results", args.run)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "chaos.jsonl"), "a")
    t0 = time.time()

    def log(action, target, **kw):
        row = {"t": round(time.time() - t0, 1), "wall_clock": time.time(), "action": action, "target": target, **kw}
        logf.write(json.dumps(row) + "\n"); logf.flush()
        print(f"[chaos +{row['t']:6.1f}s] {action} {target} {kw if kw else ''}", flush=True)

    def launch(cmd: list[str], logname: str) -> int:
        with open(os.path.join(out_dir, logname), "a") as f:
            return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT).pid

    def proxy_set(name: str, **params) -> dict | None:
        import urllib.parse, urllib.request
        url = f"{args.proxy.rstrip('/')}/set?" + urllib.parse.urlencode({"name": name, **params})
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return json.loads(r.read())
        except Exception as e:
            log("proxy-error", name, error=f"{type(e).__name__}: {e}", url=url)
            return None

    print("[chaos] schedule: " + ", ".join(f"{a} {n}@{t:.0f}s" for t, a, n, _ in events), flush=True)
    for t, action, name, extra in events:
        while time.time() - t0 < t:
            time.sleep(0.5)
        if action in ("kill", "pause", "restart"):
            pid = worker_pid(name)
            if pid is None:
                log(action, name, error="no such worker process"); continue
        if action == "kill":
            os.kill(pid, signal.SIGKILL); log("kill", name, pid=pid)
        elif action == "pause":
            os.kill(pid, signal.SIGSTOP); log("pause", name, pid=pid, duration_s=extra)
            time.sleep(extra)
            os.kill(pid, signal.SIGCONT); log("unpause", name, pid=pid)
        elif action == "restart":
            cmd = worker_cmd(pid)
            os.kill(pid, signal.SIGKILL); time.sleep(1)
            log("restart", name, old_pid=pid, new_pid=launch(cmd, f"{name}_restart.out"))
        elif action == "late-join":
            cmd = [sys.executable, "-m", "dgpt.worker", "--name", name, "--coordinator", args.coordinator, "--out-dir", out_dir]
            if args.token:
                cmd += ["--token", args.token]
            log("late-join", name, pid=launch(cmd, f"{name}.out"))
        elif action in ("cut", "lag", "throttle"):
            dur, val = extra
            params = {"mode": "cut"} if action == "cut" else {"lag_ms": val} if action == "lag" else {"rate_kb_s": val}
            if proxy_set(name, **params) is None:
                continue
            log(action, name, duration_s=dur, **({} if action == "cut" else params))
            time.sleep(dur)
            restore = {"mode": "normal"} if action == "cut" else {"lag_ms": 0} if action == "lag" else {"rate_kb_s": 0}
            proxy_set(name, **restore); log(f"un{action}", name)
        elif action == "restart-coordinator":
            pid = coordinator_pid(args.run)
            cmd_file = os.path.join(out_dir, "coordinator.cmd")
            if pid is None or not os.path.exists(cmd_file):
                log("restart-coordinator", "coordinator", error="coordinator pid or coordinator.cmd not found"); continue
            os.kill(pid, signal.SIGKILL); log("kill-coordinator", "coordinator", pid=pid)
            time.sleep(2)
            cmd = open(cmd_file).read().split()
            if cmd[0].endswith(("python", "python3")):
                cmd[0] = sys.executable
            log("restart-coordinator", "coordinator", new_pid=launch(cmd, "coordinator_restart.out"), cmd=" ".join(cmd))
    print("[chaos] done", flush=True)


if __name__ == "__main__":
    main()
