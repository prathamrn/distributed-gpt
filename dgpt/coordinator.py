"""Coordinator: owns the global model, runs rounds, merges deltas, checkpoints.

Runs on one machine; workers anywhere connect outbound to it over HTTP.

    python3 -m dgpt.coordinator --run-name k25 --set local_steps=25 --set total_steps=3000
    dgpt-coordinator --run-name k25            # same thing once installed; resumes from results/k25/ckpt.pt if present

Endpoints (see protocol.py for schemas):
    POST /register     JSON  -> shard assignment + configs
    GET  /weights      -> packed weights (long-polls with ?since=<version> until a newer one exists)
    POST /delta        packed delta -> accepted | stale | unknown_worker | bad_layout | rejected
    POST /heartbeat    JSON  -> version, whether we hold your delta, whether you are registered
    GET  /status       JSON summary for humans and chaos.py
    GET  /health
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response

from dgpt.config import RunConfig, TrainConfig
from dgpt.data import Dataset
from dgpt.evaluate import evaluate_full
from dgpt.merge import (OuterOptimizer, RoundView, adaptive_local_steps, aggregate, round_timeout,
                   should_close_round)
from dgpt.model import GPT
from dgpt.protocol import (DeltaMeta, DeltaResponse, HeartbeatRequest, HeartbeatResponse, RegisterRequest,
                      RegisterResponse, WeightsMeta, check_layout, config_hash, fingerprint, layout_of, pack,
                      unpack)

LONG_POLL_S = 25.0
UPLOAD_GRACE_S = 10.0     # don't close on timeout while a straggler says it is uploading (bounded)


class Coordinator:
    def __init__(self, run: RunConfig, train: TrainConfig, resume: bool = False):
        self.run, self.train = run, train
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.out_dir = run.checkpoint_dir or os.path.join("results", run.run_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.out_dir, "ckpt.pt")

        self.ds = Dataset.load(train.dataset)
        train.vocab_size = self.ds.vocab_size
        self.gpt_cfg = train.model_config().to_dict()
        self.gpt_cfg_hash = config_hash(self.gpt_cfg)
        torch.manual_seed(run.seed)
        self.model = GPT(train.model_config())
        self.weights = {k: v.detach().clone().float() for k, v in self.model.state_dict().items()}
        self.layout = layout_of(self.weights)
        self.outer = OuterOptimizer(run.outer_lr, run.outer_momentum, run.outer_nesterov)

        self.version = 0
        self.global_step = 0
        self.rounds_merged = 0
        self.done = False
        self.finished = False         # metrics.json written, final log row flushed
        self.workers: dict[str, dict] = {}
        self.shard_owner: dict[int, str | None] = {i: None for i in range(run.n_shards)}
        self.round_times: collections.deque[float] = collections.deque(maxlen=10)
        self.bytes_in = self.bytes_out = 0
        self.stale_count = self.rejected_count = 0
        self.last_val_loss: float | None = None
        self.t0 = time.time()
        self.wall_offset = 0.0
        self.restarts = 0

        # held-out batch for the optional loss check (PRD 7.11)
        rng = np.random.default_rng(run.seed + 99)
        self.check_batch = self.ds.get_batch("val", 128, train.block_size, rng)

        if resume and os.path.exists(self.ckpt_path):
            self._load_checkpoint()
        else:
            self.last_val_loss = self._evaluate()
            self._checkpoint()
        self.weights_hash = fingerprint(self.weights)
        self._open_round()
        self.logf = open(os.path.join(self.out_dir, "coordinator.jsonl"), "a")
        self._log_event("start", resumed=resume and self.restarts > 0, version=self.version, global_step=self.global_step)
        threading.Thread(target=self._manager, daemon=True, name="round-manager").start()

    # ---- helpers ------------------------------------------------------------
    def _wall(self) -> float:
        return self.wall_offset + time.time() - self.t0

    def _log(self, row: dict) -> None:
        row.setdefault("wall_time", self._wall())
        self.logf.write(json.dumps(row) + "\n"); self.logf.flush()

    def _log_event(self, event: str, **kw) -> None:
        self._log({"event": event, **kw})

    def _evaluate(self) -> float:
        self.model.load_state_dict(self.weights)
        return evaluate_full(self.model, self.ds, "cpu")["val_loss"]

    def _alive(self) -> set[str]:
        now = time.time()
        return {w for w, info in self.workers.items() if now - info["last_seen"] <= self.run.dead_after_s}

    def _open_round(self) -> None:
        self.round = {"version": self.version, "opened_at": time.time(), "participants": set(), "deltas": {}}

    def _free_shard(self, prefer: int | None = None) -> int:
        if prefer is not None and self.shard_owner.get(prefer) is None:
            return prefer
        free = [s for s, o in self.shard_owner.items() if o is None]
        if free:
            return free[0]
        # every shard taken: share the least-loaded one (n_shards < n_workers)
        load = collections.Counter(info["shard"] for info in self.workers.values())
        return min(self.shard_owner, key=lambda s: load.get(s, 0))

    def _timeout_s(self) -> float:
        return round_timeout(list(self.round_times), self.run.round_timeout_factor,
                             self.run.round_timeout_floor_s, self.run.round_timeout_initial_s)

    # ---- checkpoint --------------------------------------------------------
    def _checkpoint(self) -> None:
        tmp = self.ckpt_path + ".tmp"
        torch.save({
            "weights": self.weights, "version": self.version, "global_step": self.global_step,
            "rounds_merged": self.rounds_merged, "outer": self.outer.state_dict(),
            "shard_owner": self.shard_owner, "round_times": list(self.round_times),
            "bytes_in": self.bytes_in, "bytes_out": self.bytes_out, "stale_count": self.stale_count,
            "rejected_count": self.rejected_count, "last_val_loss": self.last_val_loss, "done": self.done,
            "run_config": self.run.to_dict(), "train_config": self.train.to_dict(), "gpt_config": self.gpt_cfg,
            "wall": self._wall(), "restarts": self.restarts,
        }, tmp)
        os.replace(tmp, self.ckpt_path)   # atomic: never a half-written checkpoint

    def _load_checkpoint(self) -> None:
        ck = torch.load(self.ckpt_path, map_location="cpu")   # our own file: trusted
        self.weights = {k: v.float() for k, v in ck["weights"].items()}
        self.version, self.global_step = ck["version"], ck["global_step"]
        self.rounds_merged = ck.get("rounds_merged", 0)
        self.outer.load_state_dict(ck["outer"])
        self.round_times = collections.deque(ck["round_times"], maxlen=10)
        self.bytes_in, self.bytes_out = ck["bytes_in"], ck["bytes_out"]
        self.stale_count, self.rejected_count = ck["stale_count"], ck["rejected_count"]
        self.last_val_loss, self.done = ck["last_val_loss"], ck["done"]
        self.wall_offset = ck.get("wall", 0.0)
        self.restarts = ck.get("restarts", 0) + 1
        # shards: remember who had what so a re-registering worker gets its old shard, but nobody owns anything yet
        self.previous_owner = dict(ck["shard_owner"])
        self.shard_owner = {i: None for i in range(self.run.n_shards)}
        print(f"[coordinator] resumed from {self.ckpt_path}: version {self.version}, global_step {self.global_step}, restart #{self.restarts}")

    # ---- round manager (background thread) ------------------------------------
    def _manager(self) -> None:
        while True:
            time.sleep(0.25)
            with self.lock:
                if self.done:
                    continue
                self._reap_dead()
                alive = self._alive()
                r = self.round
                view = RoundView(alive=alive, reported=set(r["deltas"]), elapsed_s=time.time() - r["opened_at"],
                                 min_workers=self.run.min_workers, timeout_s=self._timeout_s())
                close, why = should_close_round(view)
                if close and why.startswith("timeout"):
                    # bounded grace for a straggler that is mid-upload
                    waiting = alive - view.reported
                    uploading = [w for w in waiting if self.workers[w].get("status") == "uploading"
                                 and time.time() - self.workers[w]["last_seen"] < UPLOAD_GRACE_S]
                    if uploading and view.elapsed_s < view.timeout_s + UPLOAD_GRACE_S:
                        continue
                if close:
                    self._merge(why)

    def _reap_dead(self) -> None:
        now = time.time()
        for wid, info in list(self.workers.items()):
            if now - info["last_seen"] > self.run.dead_after_s:
                if self.shard_owner.get(info["shard"]) == wid:
                    self.shard_owner[info["shard"]] = None
                del self.workers[wid]
                self._log_event("worker_dead", worker=wid, shard=info["shard"], version=self.version,
                                had_delta=wid in self.round["deltas"])
                print(f"[coordinator] worker {wid} declared dead (no heartbeat for {self.run.dead_after_s}s); shard {info['shard']} freed")

    def _merge(self, why: str) -> None:
        r = self.round
        t_close = time.time()
        deltas = [d for d, _ in r["deltas"].values()]
        metas = [m for _, m in r["deltas"].values()]
        weights = [max(1, m["n_steps"]) for m in metas]
        avg = aggregate(deltas, weights, self.run.aggregation, self.run.trim_fraction)
        self.weights = self.outer.step(self.weights, avg)
        steps_merged = sum(m["n_steps"] for m in metas)
        self.global_step += steps_merged
        self.version += 1
        self.rounds_merged += 1
        self.weights_hash = fingerprint(self.weights)
        if self.global_step >= self.run.total_steps:
            self.done = True
        if self.done or self.rounds_merged % self.run.eval_every_rounds == 0:
            self.last_val_loss = self._evaluate()
        self._checkpoint()           # checkpoint BEFORE serving the new version
        row = {
            "event": "merge", "round": self.rounds_merged, "version": self.version, "global_step": self.global_step,
            "tokens": self.global_step * self.train.tokens_per_step, "val_loss": self.last_val_loss,
            "n_deltas": len(deltas), "steps_merged": steps_merged,
            "steps_per_worker": {m["worker_id"]: m["n_steps"] for m in metas},
            "participants": sorted(r["participants"]), "alive": sorted(self._alive()), "close_reason": why,
            "round_wall_s": t_close - r["opened_at"], "timeout_s": self._timeout_s(),
            "worker_timing": {w: {"steps_per_s": i["steps_per_s"], "overhead_s": i["overhead_s"], "assigned_k": i["assigned_k"]}
                              for w, i in self.workers.items()},
            "bytes_in": self.bytes_in, "bytes_out": self.bytes_out, "bytes_total": self.bytes_in + self.bytes_out,
            "stale_total": self.stale_count, "rejected_total": self.rejected_count, "done": self.done,
        }
        self._log(row)
        print(f"[coordinator] round {self.rounds_merged:4d} v{self.version:<4d} step {self.global_step:5d}/{self.run.total_steps} "
              f"deltas={len(deltas)} val={self.last_val_loss:.4f} bytes={(self.bytes_in+self.bytes_out)/1e6:8.1f}MB "
              f"round={row['round_wall_s']:5.1f}s ({why})")
        self._open_round()
        self.cond.notify_all()
        if self.done:
            self._finish()

    def _finish(self) -> None:
        m = evaluate_full(self.model, self.ds, "cpu")
        m.update({"run": self.run.run_name, "steps": self.global_step, "tokens": self.global_step * self.train.tokens_per_step,
                  "rounds": self.rounds_merged, "version": self.version, "wall_time_s": self._wall(),
                  "bytes_in": self.bytes_in, "bytes_out": self.bytes_out, "bytes_total": self.bytes_in + self.bytes_out,
                  "stale_total": self.stale_count, "rejected_total": self.rejected_count, "restarts": self.restarts,
                  "params": sum(v.numel() for v in self.weights.values()), "run_config": self.run.to_dict(),
                  "train_config": self.train.to_dict()})
        with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        self._log_event("done", **{k: v for k, v in m.items() if k not in ("run_config", "train_config")})
        print(f"[coordinator] DONE: val_loss {m['val_loss']:.4f} after {self.rounds_merged} rounds, "
              f"{m['bytes_total']/1e6:.1f} MB moved, {m['wall_time_s']:.0f}s")
        self.finished = True          # the exit watcher waits for this, not for `done` (final evals can take >8 s)

    # ---- request handlers ------------------------------------------------------
    def register(self, req: RegisterRequest) -> RegisterResponse:
        if req.gpt_config_hash is not None and req.gpt_config_hash != self.gpt_cfg_hash:
            raise HTTPException(409, f"model config mismatch: expected {self.gpt_cfg_hash}")
        with self.lock:
            prev = getattr(self, "previous_owner", {})
            old = next((s for s, o in prev.items() if o == req.worker_id), None)
            import secrets as _secrets
            session = _secrets.token_hex(8)
            if req.worker_id in self.workers:
                shard = self.workers[req.worker_id]["shard"]
                prev = self.workers[req.worker_id].get("session")
                if prev and time.time() - self.workers[req.worker_id]["last_seen"] <= self.run.dead_after_s:
                    # a live instance already holds this identity: the new one takes over, the old one is fenced out
                    self._log_event("takeover", worker=req.worker_id, old_session=prev, new_session=session, version=self.version)
                    print(f"[coordinator] {req.worker_id}: a new instance registered with the same token; the old one will be told to stop")
            else:
                shard = self._free_shard(prefer=old)
                self.shard_owner[shard] = req.worker_id if self.shard_owner[shard] is None else self.shard_owner[shard]
            self.workers[req.worker_id] = {
                "shard": shard, "speed_hint": req.speed_hint, "dtype": req.dtype, "cpus": req.cpus,
                "last_seen": time.time(), "registered_at": time.time(), "fetched_version": None,
                "steps_per_s": None, "overhead_s": None, "served_at": None, "assigned_k": None,
                "status": "registered", "local_step": 0, "session": session,
            }
            self._log_event("register", worker=req.worker_id, shard=shard, speed_hint=req.speed_hint, dtype=req.dtype, version=self.version)
            if len(self.workers) >= self.run.start_workers:
                self.cond.notify_all()            # release anyone waiting at the start barrier
            else:
                print(f"[coordinator] waiting for {self.run.start_workers - len(self.workers)} more worker(s) before serving weights")
            print(f"[coordinator] {req.worker_id} registered ({req.speed_hint}, {req.dtype}) -> shard {shard}")
            return RegisterResponse(worker_id=req.worker_id, session=session, shard_id=shard, n_shards=self.run.n_shards, version=self.version,
                                    run_config=self.run.to_dict(), train_config=self.train.to_dict(), gpt_config=self.gpt_cfg)

    def weights_body(self, worker_id: str | None, since: int | None) -> bytes | None:
        with self.cond:
            # start barrier: hold every weights request until start_workers have registered (long-poll style)
            deadline = time.time() + LONG_POLL_S
            while len(self.workers) < self.run.start_workers and not self.done:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None                       # 204: the worker asks again; heartbeats keep it alive meanwhile
                self.cond.wait(remaining)
            if since is not None and not self.done:
                deadline = time.time() + LONG_POLL_S
                while self.version <= since and not self.done:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                    self.cond.wait(remaining)
            k = self.run.local_steps
            if worker_id and worker_id in self.workers:
                info = self.workers[worker_id]
                info["fetched_version"] = self.version
                info["last_seen"] = time.time()
                if not self.round["participants"]:
                    self.round["opened_at"] = time.time()     # round clock starts at the first fetch, not at creation
                self.round["participants"].add(worker_id)
                if self.run.adaptive_k:
                    k = adaptive_local_steps(self.run.local_steps, {w: i["steps_per_s"] for w, i in self.workers.items()}, worker_id,
                                             overhead_s={w: i["overhead_s"] for w, i in self.workers.items()})
                info["served_at"] = time.time()               # the worker's cycle clock: download + train + upload
                info["assigned_k"] = k
            meta = WeightsMeta(version=self.version, weights_hash=self.weights_hash, global_step=self.global_step,
                               n_workers_alive=max(1, len(self._alive())), local_steps=k, done=self.done,
                               dtype=self.run.weights_dtype).model_dump()
            body = pack(self.weights, meta, self.run.weights_dtype)
            self.bytes_out += len(body)
            return body

    def delta(self, body: bytes, auth_worker: str | None = None) -> DeltaResponse:
        self.bytes_in += len(body)
        try:
            tensors, meta = unpack(body)
            m = DeltaMeta(**meta)
        except Exception as e:
            self.rejected_count += 1
            return DeltaResponse(status="bad_layout", version=self.version, detail=f"unparseable: {e}")
        if auth_worker is not None and m.worker_id != auth_worker:
            self.rejected_count += 1
            self._log_event("rejected_delta", worker=auth_worker, version=self.version, reason=f"claimed to be {m.worker_id}")
            return DeltaResponse(status="rejected", version=self.version, detail=f"signed by {auth_worker!r}, claims {m.worker_id!r}")
        with self.lock:
            if m.worker_id not in self.workers:
                return DeltaResponse(status="unknown_worker", version=self.version, detail="re-register")
            info = self.workers[m.worker_id]
            info["last_seen"] = time.time()
            self._learn_timing(info, m)
            if m.version != self.version or m.weights_hash != self.weights_hash:
                self.stale_count += 1
                self._log_event("stale_delta", worker=m.worker_id, delta_version=m.version, current_version=self.version, n_steps=m.n_steps)
                return DeltaResponse(status="stale", version=self.version, detail=f"delta for v{m.version}, current v{self.version}")
            why = check_layout(tensors, self.layout)
            if why:
                self.rejected_count += 1
                return DeltaResponse(status="bad_layout", version=self.version, detail=why)
            if self.run.loss_check:
                bad = self._loss_check(tensors)
                if bad:
                    self.rejected_count += 1
                    self._log_event("rejected_delta", worker=m.worker_id, version=self.version, reason=bad)
                    return DeltaResponse(status="rejected", version=self.version, detail=bad)
            self.round["deltas"][m.worker_id] = (tensors, m.model_dump())
            self._log_event("delta", worker=m.worker_id, version=m.version, n_steps=m.n_steps, train_loss=m.train_loss,
                            round_wall_s=m.round_wall_s, overhead_s=info["overhead_s"], bytes=len(body), dtype=m.dtype)
            return DeltaResponse(status="accepted", version=self.version)

    def _loss_check(self, delta: dict) -> str | None:
        """Reject a delta that makes the held-out batch worse by more than the margin (PRD 7.11)."""
        x, y = self.check_batch
        with torch.no_grad():
            self.model.load_state_dict(self.weights); _, base = self.model(x, y)
            self.model.load_state_dict({k: self.weights[k] - delta[k] for k in self.weights}); _, after = self.model(x, y)
        if float(after) > float(base) + self.run.loss_check_margin:
            return f"held-out loss {float(base):.3f} -> {float(after):.3f} exceeds margin {self.run.loss_check_margin}"
        return None

    def _learn_timing(self, info: dict, m: DeltaMeta) -> None:
        """Per-worker speed and transfer overhead, measured on the coordinator's clock from the moment the weights
        were served to the moment the delta arrived. Stale and partial deltas count too: the round they missed
        is exactly when we need to learn why. Feeds adaptive K and the round timeout (see merge.adaptive_local_steps)."""
        if not m.round_wall_s or m.n_steps <= 0 or info.get("fetched_version") != m.version or not info.get("served_at"):
            return
        speed = m.n_steps / m.round_wall_s
        if m.n_steps >= 3 or not info["steps_per_s"]:      # a 1-2 step partial is too noisy to overwrite a real measurement
            info["steps_per_s"] = speed
        cycle = time.time() - info["served_at"]
        overhead = max(0.0, cycle - m.round_wall_s)         # download + upload + anything that was not training
        info["overhead_s"] = overhead if info["overhead_s"] is None else 0.5 * info["overhead_s"] + 0.5 * overhead
        # timeout history: what this worker's full cycle takes (or would have taken, if it was cut short)
        assigned = info.get("assigned_k") or self.run.local_steps
        planned = info["overhead_s"] + assigned / info["steps_per_s"]
        self.round_times.append(cycle if m.n_steps >= assigned else planned)
        info["served_at"] = None                            # one measurement per fetch

    def heartbeat(self, hb: HeartbeatRequest) -> HeartbeatResponse:
        with self.lock:
            info = self.workers.get(hb.worker_id)
            if info is None:
                return HeartbeatResponse(ok=True, version=self.version, have_delta_from_you=False, registered=False, done=self.done)
            info["last_seen"] = time.time(); info["status"] = hb.status; info["local_step"] = hb.local_step
            closes_in = max(0.0, self._timeout_s() - (time.time() - self.round["opened_at"]))
            return HeartbeatResponse(ok=True, version=self.version, have_delta_from_you=hb.worker_id in self.round["deltas"],
                                     registered=True, done=self.done, round_closes_in_s=closes_in)

    def status(self) -> dict:
        with self.lock:
            alive = self._alive()
            return {
                "run": self.run.run_name, "version": self.version, "global_step": self.global_step, "total_steps": self.run.total_steps,
                "rounds": self.rounds_merged, "done": self.done, "val_loss": self.last_val_loss,
                "round": {"opened_s_ago": time.time() - self.round["opened_at"], "participants": sorted(self.round["participants"]),
                          "reported": sorted(self.round["deltas"]), "timeout_s": self._timeout_s()},
                "workers": {w: {"shard": i["shard"], "alive": w in alive, "status": i["status"], "local_step": i["local_step"],
                                "steps_per_s": i["steps_per_s"], "overhead_s": i["overhead_s"], "assigned_k": i["assigned_k"],
                                "dtype": i["dtype"], "last_seen_s_ago": time.time() - i["last_seen"]}
                            for w, i in self.workers.items()},
                "shards": self.shard_owner, "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
                "stale_total": self.stale_count, "rejected_total": self.rejected_count, "restarts": self.restarts,
                "wall_time": self._wall(),
            }


# ---- FastAPI wiring -------------------------------------------------------------

def build_app(coord: Coordinator, token: str | None = None, registry=None) -> FastAPI:
    """Auth modes: none (LAN/dev), shared token (X-Token), or per-worker signed requests (registry, see auth.py)."""
    app = FastAPI(title="decentralized-gpt coordinator")
    verifier = None
    if registry is not None:
        from dgpt.auth import AuthError, Verifier
        verifier = Verifier(registry)

        PUBLIC = {"/health", "/install.sh", "/install.ps1"}

        @app.middleware("http")
        async def require_signature(request: Request, call_next):
            if request.url.path in PUBLIC or request.url.path.startswith("/wheels/"):
                return await call_next(request)
            body = await request.body()
            pq = request.url.path + (f"?{request.url.query}" if request.url.query else "")
            try:
                request.state.worker_id = verifier.verify(dict(request.headers), request.method, pq, body)
            except AuthError as e:
                print(f"[coordinator] auth rejected {request.method} {request.url.path} from {request.headers.get('x-worker')!r}: {e}")
                return Response(status_code=401, content=f"auth: {e}")
            return await call_next(request)
    elif token:
        @app.middleware("http")
        async def require_token(request: Request, call_next):
            if request.url.path not in ("/health", "/install.sh", "/install.ps1") and not request.url.path.startswith("/wheels/") and request.headers.get("x-token") != token:
                return Response(status_code=401, content="missing or wrong X-Token")
            return await call_next(request)

    def bind(request: Request, claimed: str | None) -> None:
        """With per-worker auth, the identity inside a message must be the one that signed it."""
        wid = getattr(request.state, "worker_id", None)
        if wid is not None and claimed is not None and claimed != wid:
            raise HTTPException(403, f"message claims worker {claimed!r} but was signed by {wid!r}")

    def fence(worker_id: str | None, session: str | None) -> None:
        """A worker instance whose session was superseded by a newer registration must stop (409)."""
        if not worker_id or not session:
            return                                    # legacy worker without a session header: not fenced
        with coord.lock:                              # same lock as every other read of coord.workers (RLock: handlers may hold it)
            info = coord.workers.get(worker_id)
            superseded = info is not None and info.get("session") and info["session"] != session
        if superseded:
            raise HTTPException(409, "superseded: another instance registered with this token; this one must stop")

    def signed(request: Request, body: bytes, media_type: str) -> Response:
        headers = {}
        wid = getattr(request.state, "worker_id", None)
        if wid is not None:
            from dgpt.auth import sign_body
            headers["X-Signature"] = sign_body(registry.secret_for(wid), body)
        return Response(content=body, media_type=media_type, headers=headers)

    @app.get("/health")
    def health():
        return {"ok": True, "role": "coordinator", "version": coord.version, "done": coord.done,
                "dataset": coord.train.dataset, "auth": "signed" if verifier else ("token" if token else "none")}

    HERE = os.path.dirname(os.path.abspath(__file__))

    def _wheel_path() -> str | None:
        import glob as _glob
        w = sorted(_glob.glob(os.path.join(HERE, "..", "dist", "dgpt-*.whl")) + _glob.glob(os.path.join(os.getcwd(), "dist", "dgpt-*.whl")))
        return w[-1] if w else None

    def _public_base(request: Request) -> str:
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        return f"{scheme}://{request.headers.get('host', request.url.netloc)}"

    def _wheel_url(request: Request) -> str:
        p = _wheel_path()
        return f"{_public_base(request)}/wheels/{os.path.basename(p)}" if p else "dgpt"

    @app.get("/install.sh")
    def install_sh(request: Request):
        """The one-line installer, pointed at this coordinator's own wheel: curl -fsSL <url>/install.sh | sh"""
        text = open(os.path.join(HERE, "install", "install.sh")).read()
        text = text.replace('SRC="${DGPT_SRC:-dgpt @ git+https://github.com/YOUR_ORG/distribute}"',
                            f'SRC="${{DGPT_SRC:-{_wheel_url(request)}}}"')
        text = text.replace("http://HOST:8000", _public_base(request))
        return Response(content=text, media_type="text/x-shellscript")

    @app.get("/install.ps1")
    def install_ps1(request: Request):
        text = open(os.path.join(HERE, "install", "install.ps1")).read()
        text = text.replace('"dgpt @ git+https://github.com/YOUR_ORG/distribute"', f'"{_wheel_url(request)}"')
        text = text.replace("http://HOST:8000", _public_base(request))
        return Response(content=text, media_type="text/plain")

    @app.get("/wheels/{fn}")
    def wheel(fn: str):
        """Served under its real filename: uv/pip need the version in the name."""
        from fastapi.responses import FileResponse
        p = _wheel_path()
        if not p or fn != os.path.basename(p):
            raise HTTPException(404, f"no such wheel; current: {os.path.basename(p) if p else 'none built'}")
        return FileResponse(p, media_type="application/zip", filename=fn)

    @app.get("/data/{name}/{fn}")
    def data_file(name: str, fn: str):
        """Serve the tokenized dataset to workers that do not have it (installed donors)."""
        from dgpt.data import DATASET_FILES, _paths, ensure_dataset
        if fn not in DATASET_FILES or name != coord.train.dataset:
            raise HTTPException(404, "not served")
        ensure_dataset(name)
        path = os.path.join(_paths(name)["dir"], fn)
        from fastapi.responses import FileResponse
        return FileResponse(path, media_type="application/octet-stream", filename=fn)

    @app.post("/register")
    def register(req: RegisterRequest, request: Request):
        bind(request, req.worker_id)
        out = coord.register(req).model_dump_json().encode()
        return signed(request, out, "application/json")

    @app.get("/weights")
    def weights(request: Request, worker_id: str | None = None, since: int | None = None):
        bind(request, worker_id)
        fence(worker_id, request.headers.get("x-session"))
        body = coord.weights_body(worker_id, since)
        if body is None:
            return Response(status_code=204)
        return signed(request, body, "application/octet-stream")

    @app.post("/delta", response_model=DeltaResponse)
    async def delta(request: Request):
        body = await request.body()
        fence(request.headers.get("x-worker"), request.headers.get("x-session"))
        return coord.delta(body, auth_worker=getattr(request.state, "worker_id", None))

    @app.post("/heartbeat", response_model=HeartbeatResponse)
    def heartbeat(hb: HeartbeatRequest, request: Request):
        bind(request, hb.worker_id)
        fence(hb.worker_id, request.headers.get("x-session"))
        return coord.heartbeat(hb)

    @app.get("/status")
    def status():
        return coord.status()

    return app


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--set", action="append", default=[], help="override RunConfig, e.g. --set local_steps=25")
    ap.add_argument("--train-set", action="append", default=[], help="override TrainConfig, e.g. --train-set max_steps=3000")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resume", action="store_true", help="(default when results/<run>/ckpt.pt exists)")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start from version 0")
    ap.add_argument("--with-local-worker", action="store_true", help="also run a worker on this machine against localhost (uses the GPU if there is one)")
    ap.add_argument("--local-threads", type=int, default=4, help="CPU threads for the local worker")
    ap.add_argument("--exit-when-done", action="store_true")
    ap.add_argument("--token", default=os.environ.get("DGPT_TOKEN"), help="shared secret workers must send (X-Token); LAN/dev only")
    ap.add_argument("--auth", default=os.environ.get("DGPT_AUTH"), help="per-worker credential registry (workers.json); enables signed requests")
    ap.add_argument("--invite", metavar="WORKER_ID", help="with --auth: create a credential, print the invite token, exit")
    ap.add_argument("--revoke", metavar="WORKER_ID", help="with --auth: revoke a worker's credential, exit")
    return ap.parse_args(argv)


def apply_overrides(cfg, overrides: list[str]):
    for kv in overrides:
        k, v = kv.split("=", 1)
        if not hasattr(cfg, k):
            sys.exit(f"unknown config key {k}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        setattr(cfg, k, v)
    return cfg


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    registry = None
    if args.auth:
        from dgpt.auth import Registry
        registry = Registry(args.auth)
        if args.invite:
            print(registry.invite(args.invite)); return
        if args.revoke:
            print("revoked" if registry.revoke(args.revoke) else "no such worker"); return
        if not registry.workers:
            sys.exit(f"{args.auth} has no workers yet: create one with --invite NAME")
    run = apply_overrides(RunConfig(run_name=args.run_name), args.set)
    train = apply_overrides(TrainConfig(), args.train_set)
    ckpt = os.path.join(run.checkpoint_dir or os.path.join("results", run.run_name), "ckpt.pt")
    resume = (args.resume or os.path.exists(ckpt)) and not args.fresh
    if resume and os.path.exists(ckpt):
        print(f"[coordinator] checkpoint found at {ckpt}: resuming (use --fresh to start over)")
    coord = Coordinator(run, train, resume=resume)
    with open(os.path.join(coord.out_dir, "coordinator.cmd"), "w") as f:      # so chaos.py can restart us identically
        f.write(" ".join([sys.executable, "-m", "dgpt.coordinator", *sys.argv[1:]]) + "\n")
    print(f"[coordinator] run={run.run_name} K={run.local_steps} total_steps={run.total_steps} shards={run.n_shards} "
          f"outer=(lr {run.outer_lr}, mu {run.outer_momentum}, nesterov {run.outer_nesterov}) delta_dtype={run.delta_dtype} "
          f"-> {coord.out_dir}")
    app = build_app(coord, args.token, registry)
    if args.with_local_worker:
        import subprocess
        wcmd = [sys.executable, "-m", "dgpt.worker",
                "--coordinator", f"http://127.0.0.1:{args.port}", "--name", "local", "--threads", str(args.local_threads),
                "--out-dir", coord.out_dir]
        if registry is not None:
            wcmd += ["--token", registry.invite("local")]
        elif args.token:
            wcmd += ["--token", args.token]
        def start_local():
            time.sleep(3)          # let uvicorn bind first; the worker retries anyway
            with open(os.path.join(coord.out_dir, "local_worker.out"), "a") as f:
                subprocess.Popen(wcmd, stdout=f, stderr=subprocess.STDOUT)
            print(f"[coordinator] local worker started (log: {coord.out_dir}/local_worker.out)")
        threading.Thread(target=start_local, daemon=True).start()
    if registry is not None:
        print(f"[coordinator] auth: signed requests, {sum(1 for w in registry.workers.values() if not w['revoked'])} active credential(s) in {args.auth}")
    if args.exit_when_done:
        def watch():
            while not coord.finished:      # NOT coord.done: the final evaluations/metrics must complete first (R11)
                time.sleep(1)
            time.sleep(8)      # let workers fetch the final weights and exit
            os._exit(0)
        threading.Thread(target=watch, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
