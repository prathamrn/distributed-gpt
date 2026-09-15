"""Inject failures into a running pool on a schedule (PRD 8): kill, pause/unpause, restart workers, add a
late joiner, restart the coordinator. Workers are local `worker.py` processes found by their --name.
Logs every event to results/<run>/chaos.jsonl so plots can mark them.

    python3 chaos.py --run k25 --kill worker-3@60 --pause worker-1@90:20 --restart-coordinator@150 --late-join worker-5@200

Actions (time is seconds after chaos.py starts):
    --kill NAME@T                SIGKILL the worker (the coordinator reaps it after 15 s; its shard is freed)
    --pause NAME@T:DURATION      SIGSTOP for DURATION s, then SIGCONT (stalls heartbeats + training; it is declared
                                 dead if DURATION > 15 s and re-registers when it wakes)
    --restart NAME@T             SIGKILL, then relaunch the worker with its own command line
    --late-join NAME@T           start a new worker NAME against --coordinator (add --token if the pool needs one)
    --restart-coordinator@T      SIGKILL the coordinator and relaunch it with the command in results/<run>/coordinator.cmd
                                 (it auto-resumes from its last checkpoint; workers reconnect on their own)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def worker_pid(name: str) -> int | None:
    for pid in sh(["pgrep", "-f", "worker.py"]).split():
        args = sh(["ps", "-o", "args=", "-p", pid])
        if "worker.py" in args and f"--name {name}" in args and "chaos.py" not in args:
            return int(pid)
    return None


def worker_cmd(pid: int) -> list[str]:
    return sh(["ps", "-o", "args=", "-p", str(pid)]).split()


def coordinator_pid(run: str) -> int | None:
    for pid in sh(["pgrep", "-f", "coordinator.py"]).split():
        args = sh(["ps", "-o", "args=", "-p", pid])
        if "coordinator.py" in args and f"--run-name {run}" in args and "chaos.py" not in args:
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
            cmd = [sys.executable, "worker.py", "--name", name, "--coordinator", args.coordinator, "--out-dir", out_dir]
            if args.token:
                cmd += ["--token", args.token]
            log("late-join", name, pid=launch(cmd, f"{name}.out"))
        elif action == "restart-coordinator":
            pid = coordinator_pid(args.run)
            cmd_file = os.path.join(out_dir, "coordinator.cmd")
            if pid is None or not os.path.exists(cmd_file):
                log("restart-coordinator", "coordinator", error="coordinator pid or coordinator.cmd not found"); continue
            os.kill(pid, signal.SIGKILL); log("kill-coordinator", "coordinator", pid=pid)
            time.sleep(2)
            cmd = open(cmd_file).read().split()
            if not cmd[0].endswith("coordinator.py"):
                cmd = cmd[1:] if cmd and cmd[0].endswith(("python3", "python")) else cmd
            cmd = [sys.executable] + cmd if not cmd[0].endswith("python") else cmd
            log("restart-coordinator", "coordinator", new_pid=launch(cmd, "coordinator_restart.out"), cmd=" ".join(cmd))
    print("[chaos] done", flush=True)


if __name__ == "__main__":
    main()
