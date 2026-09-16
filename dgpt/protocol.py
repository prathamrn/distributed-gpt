"""Wire format and message schemas.

Tensors never travel as pickles (see qna.md, security note). A "packed state"
is one binary body:

    [8 bytes: header length, little-endian uint64][header: JSON][raw tensor bytes]

The JSON header carries the tensor manifest (name, shape, dtype, byte offset,
byte length) plus arbitrary message metadata (version, n_steps, ...). The
receiver rebuilds tensors with numpy.frombuffer, which cannot execute code.
"""
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
    worker_id: str
    speed_hint: str = ""
    dtype: str = "float32"
    cpus: float | None = None
    gpt_config_hash: str | None = None   # hash of the client's GPTConfig if it has one; mismatch => 409. None = adopt server's


class RegisterResponse(BaseModel):
    worker_id: str
    session: str = ""               # fencing token: a later registration with the same worker id supersedes this one
    shard_id: int
    n_shards: int
    version: int
    run_config: dict
    train_config: dict
    gpt_config: dict


class HeartbeatRequest(BaseModel):
    worker_id: str
    status: str = "idle"            # idle | training | uploading | waiting
    local_step: int = 0
    version: int | None = None      # version the worker is currently training from


class HeartbeatResponse(BaseModel):
    ok: bool
    version: int
    have_delta_from_you: bool       # for the current round; False => re-upload if you already sent one
    registered: bool                # False => the coordinator forgot you (restart/death); re-register
    done: bool
    round_closes_in_s: float | None = None   # seconds until the round may close on timeout; stop early and upload


class DeltaResponse(BaseModel):
    status: str                     # accepted | stale | unknown_worker | bad_layout | rejected
    version: int
    detail: str = ""


class DeltaMeta(BaseModel):
    """Metadata that rides in the packed header of a POST /delta body."""
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
    """Metadata in the packed header of a GET /weights body."""
    version: int
    weights_hash: str
    global_step: int
    n_workers_alive: int
    local_steps: int                # K for this worker this round (may be adapted)
    done: bool
    dtype: str = "float32"


# ---- packing ----------------------------------------------------------------

def _to_bytes(t: torch.Tensor, dtype: str) -> bytes:
    t = t.detach().contiguous()
    if dtype == "float32":
        return t.to(torch.float32).numpy().astype("<f4", copy=False).tobytes()
    if dtype == "float16":
        return t.to(torch.float16).numpy().astype("<f2", copy=False).tobytes()
    if dtype == "bfloat16":
        # view the bf16 bit pattern as uint16 so numpy can carry it
        return t.to(torch.bfloat16).view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    raise ValueError(f"unsupported dtype {dtype}")


def _from_bytes(b: bytes, shape: list[int], dtype: str) -> torch.Tensor:
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
    """Serialize a state-dict-like mapping plus metadata. Tensors are cast to `dtype`."""
    manifest, chunks, offset = [], [], 0
    for name in sorted(tensors):
        b = _to_bytes(tensors[name], dtype)
        manifest.append({"name": name, "shape": list(tensors[name].shape), "offset": offset, "length": len(b)})
        chunks.append(b)
        offset += len(b)
    header = json.dumps({"dtype": dtype, "manifest": manifest, "meta": meta}).encode()
    return struct.pack("<Q", len(header)) + header + b"".join(chunks)


def unpack(body: bytes) -> tuple[dict[str, torch.Tensor], dict]:
    """Inverse of pack. Returns (tensors as float32, meta). Never executes code."""
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
    return {k: list(v.shape) for k, v in tensors.items()}


def check_layout(tensors: dict[str, torch.Tensor], expected: dict[str, list[int]]) -> str | None:
    """Return None if names and shapes match `expected`, else a short reason."""
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
    """Content hash of a state dict (fp32 bytes, sorted by name). First 16 hex chars of SHA-256."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        h.update(name.encode())
        h.update(_to_bytes(tensors[name], "float32"))
    return h.hexdigest()[:16]


def config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
