"""Wire format and message schemas: [8-byte LE header length][JSON header][raw tensor bytes]."""
from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import torch
from pydantic import BaseModel, Field

DTYPES = {
    "float32": (np.dtype("<f4"), torch.float32),
    "bfloat16": (None, torch.bfloat16),   # numpy has no bf16; stored as uint16 bit patterns
    "float16": (np.dtype("<f2"), torch.float16),
}


# ---- JSON messages ------------------------------------------------------------

class RegisterRequest(BaseModel):
    """- POST /register, the worker's first call: announce itself, ask for a shard and the run's configs.
        - speed_hint, dtype and cpus are advisory; gpt_config_hash is the field that can fail the call with 409.
        - None means the worker has no model yet and adopts the coordinator's architecture."""
    worker_id: str
    speed_hint: str = ""
    dtype: str = "float32"
    cpus: float | None = None
    gpt_config_hash: str | None = None


class RegisterResponse(BaseModel):
    """- The coordinator's reply to /register: identity, data assignment and the whole recipe in one message.
        - The coordinator dictates model and run config, because averaging is weight-for-weight by tensor name.
        - session is the fencing token that makes one invite mean one running instance (the 409 path)."""
    worker_id: str
    session: str = ""               # fencing token: a later registration with the same worker id supersedes this one
    shard_id: int
    n_shards: int
    version: int
    run_config: dict
    train_config: dict
    gpt_config: dict


class HeartbeatRequest(BaseModel):
    """- POST /heartbeat every 5 s: liveness, plus what the worker is currently doing.
        - This is what puts the worker in the coordinator's `alive` set, which the round closing rule waits on.
        - Silence for dead_after_s reaps the worker and frees its shard; the rest feeds the operator's /status."""
    worker_id: str
    status: str = "idle"            # idle | training | uploading | waiting
    local_step: int = 0
    version: int | None = None      # version the worker is currently training from


class HeartbeatResponse(BaseModel):
    """- The only channel the coordinator has to steer a worker mid-round, so it carries the control fields.
        - round_closes_in_s makes a straggler a partial contributor; remaining seconds, so no clocks must agree.
        - registered False => re-register, have_delta_from_you False => re-upload, done => the run is over."""
    ok: bool
    version: int
    have_delta_from_you: bool       # for the current round; False => re-upload if you already sent one
    registered: bool                # False => the coordinator forgot you (restart/death); re-register
    done: bool
    round_closes_in_s: float | None = None   # seconds until the round may close on timeout; stop early and upload


class DeltaResponse(BaseModel):
    """- The answer to POST /delta, carrying the recovery action rather than leaving the worker to guess.
        - accepted => wait; stale/rejected => refetch and retry; unknown_worker => re-register; bad_layout is fatal.
        - The outcome is in the body, not the HTTP code, because tunnels and proxies rewrite status codes."""
    status: str                     # accepted | stale | unknown_worker | bad_layout | rejected
    version: int
    detail: str = ""


class DeltaMeta(BaseModel):
    """- Metadata riding in the packed header of a POST /delta body: the delta's claim about itself.
        - version and weights_hash must match the served weights or it is stale; n_steps is its weight in the mean.
        - train_loss and round_wall_s feed the run log, the round timeout and adaptive K."""
    worker_id: str
    version: int
    weights_hash: str
    n_steps: int
    n_tokens: int
    shard_id: int
    train_loss: float | None = None
    round_wall_s: float | None = None
    dtype: str = "float32"


class WeightsMeta(BaseModel):
    """- Metadata in the packed header of a GET /weights body.
        - One download says where to start, how many local steps to take, and whether the run has finished.
        - So there is no separate "what should I do now" route; local_steps may be adapted to this machine."""
    version: int
    weights_hash: str
    global_step: int
    n_workers_alive: int
    local_steps: int                # K for this worker this round (may be adapted)
    done: bool
    dtype: str = "float32"


# ---- packing ----------------------------------------------------------------

def _to_bytes(t: torch.Tensor, dtype: str) -> bytes:
    """- One tensor to little-endian wire bytes at the requested precision; used by pack and fingerprint.
        - Detached and made contiguous first, so the bytes are row-major values, not whatever stride it had.
        - bfloat16 goes via view(uint16) because numpy has no such dtype; that moves the bits exactly."""
    t = t.detach().contiguous()
    if dtype == "float32":
        return t.to(torch.float32).numpy().astype("<f4", copy=False).tobytes()
    if dtype == "float16":
        return t.to(torch.float16).numpy().astype("<f2", copy=False).tobytes()
    if dtype == "bfloat16":
        return t.to(torch.bfloat16).view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    raise ValueError(f"unsupported dtype {dtype}")


def _from_bytes(b: bytes, shape: list[int], dtype: str) -> torch.Tensor:
    """- Inverse of _to_bytes: reinterpret the bytes, then copy. Every path returns float32.
        - So nothing downstream ever sees half precision and a mixed-precision pool needs no promotion rule.
        - The .copy() is load-bearing: frombuffer returns a read-only view that pins the whole HTTP body."""
    if dtype == "float32":
        arr = np.frombuffer(b, dtype="<f4").reshape(shape)
        return torch.from_numpy(arr.copy())
    if dtype == "float16":
        arr = np.frombuffer(b, dtype="<f2").reshape(shape)
        return torch.from_numpy(arr.copy()).to(torch.float32)
    if dtype == "bfloat16":
        arr = np.frombuffer(b, dtype="<u2").reshape(shape)
        return torch.from_numpy(arr.copy()).view(torch.bfloat16).to(torch.float32)
    raise ValueError(f"unsupported dtype {dtype}")


def pack(tensors: dict[str, torch.Tensor], meta: dict, dtype: str = "float32") -> bytes:
    """- Serialize a state-dict-like mapping plus metadata into one self-describing body; tensors cast to dtype.
        - Tensors are walked in sorted name order, so the layout is deterministic and fingerprint agrees pool-wide.
        - Coordinator uses it for GET /weights, worker for POST /delta; the signature covers header and data."""
    manifest, chunks, offset = [], [], 0
    for name in sorted(tensors):
        b = _to_bytes(tensors[name], dtype)
        manifest.append({"name": name, "shape": list(tensors[name].shape), "offset": offset, "length": len(b)})
        chunks.append(b)
        offset += len(b)
    header = json.dumps({"dtype": dtype, "manifest": manifest, "meta": meta}).encode()
    return struct.pack("<Q", len(header)) + header + b"".join(chunks)


def unpack(body: bytes) -> tuple[dict[str, torch.Tensor], dict]:
    """- Inverse of pack; returns (tensors as float32, meta). Never executes code.
        - Runs on input from strangers, so every failure is a clean ValueError the caller turns into bad_layout.
        - The bounds test on each manifest entry is the security check: the manifest is attacker-controlled."""
    if len(body) < 8:
        raise ValueError("body too short")
    (hlen,) = struct.unpack("<Q", body[:8])
    header = json.loads(body[8:8 + hlen].decode())
    dtype = header["dtype"]
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype}")
    base = 8 + hlen
    tensors = {}
    for m in header["manifest"]:
        s, e = base + m["offset"], base + m["offset"] + m["length"]
        if e > len(body):
            raise ValueError("manifest points past end of body")
        tensors[m["name"]] = _from_bytes(body[s:e], m["shape"], dtype)
    return tensors, header["meta"]


def layout_of(tensors: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    """- Name -> shape map for a state dict.
        - The coordinator takes one from its own model at startup as the expected layout for check_layout."""
    return {k: list(v.shape) for k, v in tensors.items()}


def check_layout(tensors: dict[str, torch.Tensor], expected: dict[str, list[int]]) -> str | None:
    """- Return None if names and shapes match `expected`, else a short reason.
        - Last layer after the config hash (wrong architecture) and version/weights_hash (stale starting point).
        - Without it a missing name would KeyError inside aggregation and take the round down for the whole pool."""
    got = layout_of(tensors)
    if set(got) != set(expected):
        missing = sorted(set(expected) - set(got))[:3]
        extra = sorted(set(got) - set(expected))[:3]
        return f"name mismatch (missing {missing}, extra {extra})"
    for k in expected:
        if got[k] != expected[k]:
            return f"shape mismatch at {k}: {got[k]} != {expected[k]}"
    return None


def fingerprint(tensors: dict[str, torch.Tensor]) -> str:
    """- Content hash of a state dict: SHA-256 over fp32 bytes in sorted name order, first 16 hex chars.
        - A counter cannot do this: a coordinator resumed from the wrong checkpoint still calls its weights v7.
        - Shipped as weights_hash and echoed in DeltaMeta; an integrity check, not trust - that is auth.py's job."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        h.update(name.encode())
        h.update(_to_bytes(tensors[name], "float32"))
    return h.hexdigest()[:16]


def config_hash(d: dict) -> str:
    """- Fingerprint of a config dict, comparing a worker's GPTConfig with the coordinator's at /register.
        - sort_keys makes the digest independent of dict ordering, so both sides hash the same settings alike.
        - A mismatch is answered 409 at registration rather than after a wasted round and a layout failure."""
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
