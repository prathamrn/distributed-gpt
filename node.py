"""Placeholder node process: proves the container plumbing works.

Serves GET /health on NODE_PORT with everything the container can see about
itself: name, IP, effective CPU quota (from the cgroup, not os.cpu_count),
torch threads, dtype, and the netem qdisc actually applied to eth0. Also
probes the host-side coordinator once at startup (non-fatal).

worker.py replaces this once the training protocol exists.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

NAME = os.environ.get("NODE_NAME", socket.gethostname())
PORT = int(os.environ.get("NODE_PORT", "8001"))
COORD = os.environ.get("COORDINATOR_URL", "")
STARTED = time.time()


def cpu_quota() -> float | None:
    """Effective CPU limit from cgroup v2 (`cpu.max`) or v1; None if unlimited."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()
        return None if q == "max" else int(q) / int(p)
    except FileNotFoundError:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read())
        return None if q < 0 else q / p
    except FileNotFoundError:
        return None


def netem() -> str:
    try:
        out = subprocess.run(["tc", "qdisc", "show", "dev", "eth0"], capture_output=True, text=True, timeout=2).stdout.strip()
        return out or "(none)"
    except Exception as e:  # tc missing, etc.
        return f"(unavailable: {e})"


def status() -> dict:
    return {
        "node": NAME,
        "role": os.environ.get("NODE_ROLE", "worker"),
        "ip": os.environ.get("NODE_IP"),
        "hostname": socket.gethostname(),
        "uptime_s": round(time.time() - STARTED, 1),
        "cpus_configured": float(os.environ.get("CPUS", "0")) or None,
        "cpus_effective": cpu_quota(),
        "host_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "dtype": os.environ.get("DTYPE", "float32"),
        "speed_hint": os.environ.get("SPEED_HINT", ""),
        "net": {
            "latency_ms": float(os.environ.get("NET_LATENCY_MS", 0)),
            "jitter_ms": float(os.environ.get("NET_JITTER_MS", 0)),
            "bandwidth_mbit": float(os.environ.get("NET_BANDWIDTH_MBIT", 0)),
            "loss_pct": float(os.environ.get("NET_LOSS_PCT", 0)),
            "qdisc": netem(),
        },
        "coordinator_url": COORD,
        "malicious": os.environ.get("MALICIOUS") == "1",
        "join_delay_s": float(os.environ.get("JOIN_DELAY_S", 0)),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            body = json.dumps(status(), indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):  # keep container logs quiet
        pass


def probe_coordinator() -> None:
    if not COORD:
        return
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"{COORD.rstrip('/')}/health", timeout=3) as r:
            print(f"[{NAME}] coordinator {COORD} reachable: HTTP {r.status} in {1000*(time.time()-t0):.0f} ms")
    except Exception as e:
        print(f"[{NAME}] coordinator {COORD} not reachable yet ({type(e).__name__}); fine until coordinator.py exists")


def main():
    threads = int(os.environ.get("TORCH_THREADS", "1"))
    torch.set_num_threads(threads)
    st = status()
    print(f"[{NAME}] up. ip={st['ip']} cpus={st['cpus_effective']} threads={st['torch_threads']} "
          f"dtype={st['dtype']} netem='{st['net']['qdisc']}'")
    probe_coordinator()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[{NAME}] serving /health on :{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
