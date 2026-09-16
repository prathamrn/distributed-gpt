"""Worker: register, fetch weights, train K steps on a shard, upload delta, repeat.

Runs on any machine (flags or env vars):
    python3 -m dgpt.worker --name w1 --coordinator http://127.0.0.1:8000 --threads 2

Fault handling (PRD 7.9): every request retries with exponential backoff; a stale
delta means "refetch and start over"; if the coordinator forgets us (restart or
we were declared dead) we re-register; if the coordinator restarts and lost our
accepted delta we re-upload it (idempotent). A heartbeat thread runs throughout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np
import requests
import torch

from dgpt.config import RunConfig, TrainConfig, lr_at
from dgpt.data import Dataset
from dgpt.merge import delta_of
from dgpt.model import GPT, GPTConfig
from dgpt.protocol import DeltaMeta, config_hash, pack, unpack
from dgpt.auth import parse_invite, request_headers, sign_body


class HmacAuth(requests.auth.AuthBase):
    """Signs every request with the worker's secret (auth.py). The secret never leaves this process."""
    def __init__(self, worker_id: str, secret: bytes):
        self.worker_id, self.secret = worker_id, secret

    def __call__(self, r):
        body = r.body.encode() if isinstance(r.body, str) else r.body
        r.headers.update(request_headers(self.worker_id, self.secret, r.method, r.url, body))
        return r

BACKOFF_MAX_S = 30.0


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def socket_hostname() -> str:
    import socket
    return socket.gethostname().split(".")[0]


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Worker:
    def __init__(self, name: str, coordinator: str, dtype: str, threads: int, speed_hint: str, cpus: float | None,
                 malicious: bool, out_dir: str | None, device: str = "cpu", token: str | None = None):
        self.name, self.base, self.dtype = name, coordinator.rstrip("/"), dtype
        self.speed_hint, self.cpus, self.malicious = speed_hint, cpus, malicious
        self.device = pick_device(device)
        torch.set_num_threads(threads)
        self.http = requests.Session()
        self.http.headers["ngrok-skip-browser-warning"] = "1"     # harmless elsewhere; skips ngrok's free-tier interstitial
        self.secret: bytes | None = None
        if token:
            inv = parse_invite(token)
            if inv:                                   # per-worker credential: identity comes from it
                self.name, self.secret = inv[0], inv[1]
                if name and name != self.name:
                    print(f"[{self.name}] note: --name {name!r} ignored; identity is fixed by the invite token", flush=True)
                self.http.auth = HmacAuth(self.name, self.secret)
            else:                                     # legacy shared token
                self.http.headers["X-Token"] = token
        self.ds: Dataset | None = None          # loaded after registration, once we know which dataset the run uses
        self.run: RunConfig | None = None
        self.train: TrainConfig | None = None
        self.model: GPT | None = None
        self.opt = None
        self.shard = self.n_shards = None
        self.out_dir_override = out_dir
        self.logf = None
        self.stop = threading.Event()
        self.need_register = threading.Event()
        self.hb_version: int | None = None
        self.hb_have_delta = False
        self.hb_time = 0.0                 # when the last heartbeat reply arrived
        self.registered = False
        self.hb_thread_started = False
        self.hb_deadline = None            # local clock time when the round may close on timeout
        self.fetch_time = 0.0
        self.step_delay = float(os.environ.get("STEP_DELAY_S", "0"))   # debug: simulate a slow machine on the host
        # gradient accumulation: a low-memory node runs the same batch as micro-batches of this size (0 = whole batch).
        # Same tokens per step, same gradient (up to fp rounding), so the coordinator sees an identical contribution.
        self.micro_batch = int(os.environ.get("MICRO_BATCH", "0"))
        self.true_bf16 = os.environ.get("TRUE_BF16", "0") == "1"       # real autocast (5x slower on CPU); default is emulation
        self.upload_done_time = 0.0
        self.hb_done = False
        self.status = "starting"
        self.local_step = 0
        self.pending: tuple[int, bytes] | None = None      # (version, body) of our last accepted delta
        self.bytes_up = self.bytes_down = 0
        self.t0 = time.time()
        self.step_s = 0.5
        self.upload_s = 1.0
        self.download_s = 0.0

    # ---- logging / http -----------------------------------------------------------
    def log(self, **row):
        row.setdefault("wall_time", time.time() - self.t0)
        row.setdefault("worker", self.name)
        if self.logf:
            self.logf.write(json.dumps(row) + "\n"); self.logf.flush()

    def say(self, msg: str):
        print(f"[{self.name}] {msg}", flush=True)

    def _retry(self, fn, what: str):
        """Call fn() until it succeeds. Network errors, truncated responses and 5xx (a tunnel or proxy hiccup)
        are retried forever with exponential backoff; a 401/403 is fatal with the server's reason shown."""
        delay = 1.0
        while not self.stop.is_set():
            try:
                return fn()
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code in (401, 403):
                    sys.exit(f"[{self.name}] {what}: rejected by coordinator ({code}): {e.response.text.strip()[:200]}")
                if code == 409 and "superseded" in e.response.text:
                    sys.exit(f"[{self.name}] stopping: another instance registered with my token (the newer one wins)")
                self.say(f"{what}: HTTP {code} from coordinator/proxy; retry in {delay:.0f}s")
            except requests.RequestException as e:          # ConnectionError, Timeout, ChunkedEncodingError, ...
                self.say(f"{what}: {type(e).__name__}; retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(BACKOFF_MAX_S, delay * 2)
        return None

    # ---- protocol steps --------------------------------------------------------------
    def register(self):
        # First contact: adopt whatever model config the coordinator serves (so bigger models need no client change).
        # Re-registration: send the hash of the config we already built, so a mismatch is caught.
        gpt_hash = config_hash(self.model.cfg.to_dict()) if self.model is not None else None
        def go():
            r = self.http.post(f"{self.base}/register", json={
                "worker_id": self.name, "speed_hint": self.speed_hint, "dtype": self.dtype, "cpus": self.cpus,
                "gpt_config_hash": gpt_hash}, timeout=30)
            if r.status_code == 409:
                sys.exit(f"[{self.name}] rejected: {r.text}")
            if r.status_code == 401:
                sys.exit(f"[{self.name}] rejected by coordinator: {r.text.strip() or 'a join token / invite is required (--token)'}")
            if r.status_code == 403:
                sys.exit(f"[{self.name}] rejected: {r.text}")
            r.raise_for_status()
            self._check_response_signature(r)
            return r.json()
        resp = self._retry(go, "register")
        if resp is None:
            return
        self.run = RunConfig.from_dict(resp["run_config"])
        self.train = TrainConfig(**{k: v for k, v in resp["train_config"].items() if k in TrainConfig.__dataclass_fields__})
        self.shard, self.n_shards = resp["shard_id"], resp["n_shards"]
        self.http.headers["X-Session"] = resp.get("session", "")       # fencing token; also on X-Worker for /delta
        self.http.headers["X-Worker"] = self.name
        self.registered = True
        if not self.hb_thread_started:          # keep heartbeating during the dataset download (can take minutes over a WAN)
            self.hb_thread_started = True
            threading.Thread(target=self.heartbeat_loop, daemon=True, name="heartbeat").start()
        if self.ds is None or self.ds.name != self.train.dataset:
            def hdrs(method: str, url: str) -> dict:
                h = dict(self.http.headers)
                if self.secret is not None:
                    h.update(request_headers(self.name, self.secret, method, url, None))
                return h
            self.ds = Dataset.load(self.train.dataset, fetch_from=self.base, headers=hdrs)   # downloads if missing
        if self.model is None:
            self.model = GPT(GPTConfig(**resp["gpt_config"])).to(self.device)
            self.opt = self.model.make_optimizer(self.train.lr, self.train.weight_decay, (self.train.beta1, self.train.beta2))
        if self.logf is None:
            out_dir = self.out_dir_override or os.path.join("results", self.run.run_name)
            os.makedirs(out_dir, exist_ok=True)
            self.logf = open(os.path.join(out_dir, f"{self.name}.jsonl"), "a")
        self.need_register.clear()
        self.say(f"registered: shard {self.shard}/{self.n_shards}, K={self.run.local_steps}, dtype={self.dtype}, version {resp['version']}")
        self.log(event="register", shard=self.shard, version=resp["version"])

    def heartbeat_loop(self):
        while not self.stop.is_set():
            try:
                r = self.http.post(f"{self.base}/heartbeat", json={"worker_id": self.name, "status": self.status,
                                                                    "local_step": self.local_step}, timeout=10)
                if r.status_code == 409 and "superseded" in r.text:
                    print(f"[{self.name}] stopping: another instance registered with my token (the newer one wins)", flush=True)
                    os._exit(3)
                if r.ok:
                    d = r.json()
                    self.hb_version, self.hb_have_delta, self.hb_done = d["version"], d["have_delta_from_you"], d["done"]
                    self.hb_time = time.time()
                    self.hb_deadline = self.hb_time + d["round_closes_in_s"] if d.get("round_closes_in_s") is not None else None
                    if not d["registered"] and self.registered:
                        self.registered = False
                        self.need_register.set()
            except (requests.ConnectionError, requests.Timeout):
                pass
            self.stop.wait(self.run.heartbeat_interval_s if self.run else 5.0)

    def fetch_weights(self, since: int | None):
        def go():
            params = {"worker_id": self.name}
            if since is not None:
                params["since"] = since
            r = self.http.get(f"{self.base}/weights", params=params, timeout=60)
            if r.status_code == 204:
                return None
            r.raise_for_status()
            self._check_response_signature(r)
            return r.content
        t = time.time()
        body = self._retry(go, "fetch weights")
        if body is None:
            return None, None
        self.download_s = time.time() - t
        self.fetch_time = time.time()
        self.hb_deadline = None                 # any deadline heard before this fetch belonged to an older round clock
        self.bytes_down += len(body)
        tensors, meta = unpack(body)
        return tensors, meta

    def _check_response_signature(self, r) -> None:
        """With per-worker auth the coordinator signs weights with our secret; an impostor cannot."""
        if self.secret is None:
            return
        sig = r.headers.get("X-Signature")
        if not sig or sig != sign_body(self.secret, r.content):
            sys.exit(f"[{self.name}] FATAL: response from {self.base} is not signed with my credential; "
                     f"wrong coordinator or a man-in-the-middle. Refusing to train on it.")

    def upload_delta(self, body: bytes) -> dict | None:
        def go():
            r = self.http.post(f"{self.base}/delta", data=body, headers={"Content-Type": "application/octet-stream"}, timeout=300)
            r.raise_for_status()
            return r.json()
        t = time.time()
        resp = self._retry(go, "upload delta")
        self.upload_s = time.time() - t
        self.upload_done_time = time.time()
        self.bytes_up += len(body)
        return resp

    # ---- data ------------------------------------------------------------------------
    def batch(self, rng: np.random.Generator):
        bs, T = self.train.batch_size, self.train.block_size
        d = self.ds.train
        if self.run.shard_mode == "interleaved":          # start positions congruent to shard mod N: disjoint, IID, full offset diversity
            starts = rng.integers(0, len(d) - T - 1, size=bs)
            starts = starts - (starts % self.n_shards) + self.shard
        elif self.run.shard_mode == "full":               # no sharding: every worker samples the whole train split (max overlap)
            starts = rng.integers(0, len(d) - T - 1, size=bs)
        else:                                             # contiguous: worker owns one slice of the text
            lo, hi = self.ds.shard_bounds(self.n_shards)[self.shard]
            starts = rng.integers(lo, hi - T - 1, size=bs)
        x = torch.stack([torch.from_numpy(d[s:s + T].astype(np.int64)) for s in starts])
        y = torch.stack([torch.from_numpy(d[s + 1:s + 1 + T].astype(np.int64)) for s in starts])
        return x.to(self.device), y.to(self.device)

    # ---- one round ---------------------------------------------------------------------
    def train_round(self, start: dict, meta: dict) -> tuple[dict, int, float, float]:
        """Train up to meta['local_steps'] steps from `start`. Returns (delta, n_steps, mean_loss, train_s)."""
        K, g0, n_alive, version = meta["local_steps"], meta["global_step"], meta["n_workers_alive"], meta["version"]
        self.model.load_state_dict(start)
        if getattr(self.run, "reset_inner_opt", False):
            # fresh Adam state each round: no stale momentum from the previous local trajectory
            self.opt = self.model.make_optimizer(self.train.lr, self.train.weight_decay, (self.train.beta1, self.train.beta2))
        seed = int(hashlib.sha256(f"{self.run.seed}:{self.shard}:{version}:{self.name}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        # bf16 workers: EMULATED by default (decision 2026-09-13, see roadblocks.md R1): compute in fp32 at full
        # speed, then snap the weights to the bf16 grid after every step so the worker holds exactly the numbers a
        # bf16 machine would hold. TRUE_BF16=1 uses real autocast instead.
        emulate_bf16 = self.dtype == "bfloat16" and not self.true_bf16
        use_bf16 = self.dtype == "bfloat16" and self.true_bf16
        losses = []
        t0 = time.time()
        self.status = "training"
        n = 0
        for i in range(K):
            # a straggler learns the round closed from the heartbeat and stops wasting effort
            if self.hb_version is not None and self.hb_version > version:
                self.say(f"round v{version} already closed (coordinator at v{self.hb_version}); stopping after {n} steps")
                break
            # ... or learns the round is about to close: stop early and upload a partial delta in time
            if n > 0 and self.hb_deadline is not None and self.hb_version == version and self.hb_time > self.fetch_time:
                remaining = self.hb_deadline - time.time()
                if remaining < 2 * self.step_s + self.upload_s + 1.0:
                    self.say(f"round v{version} closes in {remaining:.1f}s; uploading partial delta after {n}/{K} steps")
                    break
            if self.step_delay:
                time.sleep(self.step_delay)
            lr = lr_at(g0 + i * n_alive, self.train)
            for g in self.opt.param_groups:
                g["lr"] = lr
            x, y = self.batch(rng)
            self.opt.zero_grad(set_to_none=True)
            bs = x.shape[0]
            mb = self.micro_batch if 0 < self.micro_batch < bs else bs
            loss_total = 0.0
            for j in range(0, bs, mb):
                xj, yj = x[j:j + mb], y[j:j + mb]
                with torch.autocast(self.device if self.device != "mps" else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                    _, loss_j = self.model(xj, yj)
                (loss_j * (xj.shape[0] / bs)).backward()          # mean over the whole batch, accumulated per micro-batch
                loss_total += loss_j.item() * xj.shape[0] / bs
            loss = torch.tensor(loss_total)
            if self.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train.grad_clip)
            self.opt.step()
            if emulate_bf16:
                with torch.no_grad():
                    for p in self.model.parameters():
                        p.copy_(p.to(torch.bfloat16).to(torch.float32))
            losses.append(loss.item()); n += 1
            self.local_step = n
            self.step_s = 0.8 * self.step_s + 0.2 * (time.time() - t0) / n
        train_s = time.time() - t0
        local = {k: v.detach().float().cpu() for k, v in self.model.state_dict().items()}
        delta = delta_of(start, local)
        if self.malicious:
            scale = float(torch.cat([v.flatten() for v in delta.values()]).std()) * 20
            delta = {k: torch.randn_like(v) * scale for k, v in delta.items()}
        return delta, n, (float(np.mean(losses)) if losses else float("nan")), train_s

    # ---- main loop -------------------------------------------------------------------------
    def run_forever(self):
        """Never let an unexpected exception leave the worker alive-but-idle: log it and go around again."""
        backoff = 2.0
        while not self.stop.is_set():
            try:
                self._run_loop()
                return
            except SystemExit:
                raise
            except Exception:
                import traceback
                self.say(f"unexpected error (worker keeps going, retry in {backoff:.0f}s):\n" + traceback.format_exc().strip())
                self.log(event="error", error=traceback.format_exc()[-500:])
                time.sleep(backoff); backoff = min(60.0, backoff * 2)
                self.need_register.set()

    def _run_loop(self):
        if self.run is None:
            self.register()                   # starts the heartbeat thread itself
        waiting_since: int | None = None
        while not self.stop.is_set():
            if self.need_register.is_set():
                self.say("coordinator does not know me (restart or declared dead); re-registering")
                self.register(); waiting_since = None; self.pending = None
            # coordinator restarted and lost our accepted delta for this version -> re-upload
            # only trust "have_delta_from_you=False" from a heartbeat that arrived AFTER our upload finished
            if (self.pending and waiting_since is not None and self.hb_version == self.pending[0]
                    and not self.hb_have_delta and self.hb_time > self.upload_done_time + 1.0):
                self.say(f"coordinator lost my delta for v{self.pending[0]}; re-uploading")
                resp = self.upload_delta(self.pending[1])
                if resp and resp["status"] != "accepted":
                    self.pending = None; waiting_since = None
                continue
            self.status = "waiting" if waiting_since is not None else "fetching"
            tensors, meta = self.fetch_weights(waiting_since)
            if tensors is None:
                continue                                    # long-poll timed out; loop
            if meta["done"]:
                self.say(f"run complete at version {meta['version']}; exiting")
                self.log(event="done", version=meta["version"], bytes_up=self.bytes_up, bytes_down=self.bytes_down)
                break
            if waiting_since is not None and meta["version"] <= waiting_since:
                continue
            self.pending = None
            version = meta["version"]
            t_round = time.time()
            delta, n, mean_loss, train_s = self.train_round(tensors, meta)
            if n == 0:
                waiting_since = None; continue
            self.status = "uploading"
            up_dtype = "bfloat16" if (self.dtype == "bfloat16" or self.run.delta_dtype == "bfloat16") else self.run.delta_dtype
            body = pack(delta, DeltaMeta(worker_id=self.name, version=version, weights_hash=meta["weights_hash"], n_steps=n,
                                         n_tokens=n * self.train.tokens_per_step, shard_id=self.shard, train_loss=mean_loss,
                                         round_wall_s=train_s, dtype=up_dtype).model_dump(), up_dtype)
            resp = self.upload_delta(body)
            if resp is None:
                break
            self.log(event="round", version=version, K=meta["local_steps"], n_steps=n, train_loss=mean_loss, train_s=train_s,
                     upload_s=self.upload_s, download_s=self.download_s, round_s=time.time() - t_round, status=resp["status"], upload_bytes=len(body),
                     bytes_up=self.bytes_up, bytes_down=self.bytes_down, steps_per_s=n / train_s if train_s else None)
            self.say(f"v{version}: {n}/{meta['local_steps']} steps, loss {mean_loss:.4f}, {n/train_s:.2f} steps/s, "
                     f"download {self.download_s:.1f}s, upload {len(body)/1e6:.2f}MB in {self.upload_s:.2f}s -> {resp['status']}")
            if resp["status"] == "accepted":
                self.pending = (version, body); waiting_since = version
            elif resp["status"] == "unknown_worker":
                self.need_register.set(); waiting_since = None
            elif resp["status"] == "bad_layout":
                sys.exit(f"[{self.name}] fatal: {resp['detail']}")
            else:                                            # stale or rejected: refetch and go again
                waiting_since = None
        self.stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=env("NODE_NAME", "") or f"{socket_hostname()}-{os.getpid() % 1000}")
    ap.add_argument("--coordinator", default=env("COORDINATOR_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--dtype", default=env("DTYPE", "float32"))
    ap.add_argument("--threads", type=int, default=int(env("TORCH_THREADS", "0")) or max(1, (os.cpu_count() or 2) - 1),
                    help="CPU threads (default: all but one core, or $TORCH_THREADS)")
    ap.add_argument("--device", default=env("DEVICE", "auto"), help="auto | cpu | cuda | mps")
    ap.add_argument("--token", default=env("DGPT_TOKEN", "") or None, help="join token if the coordinator requires one")
    ap.add_argument("--data-dir", default=None, help="where datasets are cached (default $DGPT_DATA_DIR or ./data)")
    ap.add_argument("--speed-hint", default=env("SPEED_HINT", ""))
    ap.add_argument("--cpus", type=float, default=float(env("CPUS", "0")) or None)
    ap.add_argument("--join-delay", type=float, default=float(env("JOIN_DELAY_S", "0")))
    ap.add_argument("--malicious", action="store_true", default=env("MALICIOUS", "0") == "1")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--micro-batch", type=int, default=int(env("MICRO_BATCH", "0")),
                    help="gradient-accumulation micro-batch size for low-memory nodes (0 = whole batch)")
    args = ap.parse_args()
    if args.join_delay > 0:
        print(f"[{args.name}] joining late: sleeping {args.join_delay}s", flush=True)
        time.sleep(args.join_delay)
    if args.data_dir:
        os.environ["DGPT_DATA_DIR"] = args.data_dir
        import dgpt.data as _data
        _data.ROOT = args.data_dir
    os.environ["MICRO_BATCH"] = str(args.micro_batch)
    w = Worker(args.name, args.coordinator, args.dtype, args.threads, args.speed_hint, args.cpus, args.malicious, args.out_dir,
               device=args.device, token=args.token)
    w.say(f"starting: coordinator={args.coordinator} device={w.device} dtype={args.dtype} threads={args.threads}" + (" MALICIOUS" if args.malicious else ""))
    w.run_forever()


if __name__ == "__main__":
    main()
