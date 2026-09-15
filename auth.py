"""Per-worker credentials and request signing (see SECURITY.md).

Invite token (given to one donor, once):   dgpt1.<worker_id>.<secret_b64url>
The secret never travels again. Every request carries:
    X-Worker: <worker_id>
    X-Timestamp: <unix seconds>
    X-Nonce: <random hex>
    X-Signature: HMAC-SHA256(secret, "<METHOD>\\n<path?query>\\n<timestamp>\\n<nonce>\\n<sha256(body)>")
The coordinator rejects unknown/revoked workers, bad signatures, timestamps outside the window, and
reused nonces, and binds the authenticated identity to the worker_id inside each message. Responses
that matter (weights, register) are signed back with the same secret so a worker can detect an impostor
coordinator. Confidentiality is NOT provided: put the coordinator behind TLS for that.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from urllib.parse import urlparse

PREFIX = "dgpt1"
WINDOW_S = 120          # accepted clock skew for X-Timestamp
NONCE_TTL_S = 300       # remember nonces this long (must be > WINDOW_S)


# ---- registry (coordinator side) --------------------------------------------------------------

class Registry:
    """{worker_id: {"secret": str, "created": ts, "revoked": bool}} stored as JSON (chmod 600)."""

    def __init__(self, path: str):
        self.path = path
        self.workers: dict[str, dict] = {}
        self._mtime = 0.0
        self.reload_if_changed()

    def reload_if_changed(self) -> None:
        """Pick up invites/revocations made by another process (e.g. `--revoke` while the coordinator runs)."""
        try:
            m = os.stat(self.path).st_mtime
        except FileNotFoundError:
            return
        if m != self._mtime:
            with open(self.path) as f:
                self.workers = json.load(f)
            self._mtime = m

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.workers, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)
        self._mtime = os.stat(self.path).st_mtime

    def invite(self, worker_id: str) -> str:
        """Create (or rotate) a credential for worker_id; returns the invite token to hand to the donor."""
        if not worker_id or "." in worker_id or "/" in worker_id:
            raise ValueError("worker_id must be non-empty and contain no '.' or '/'")
        secret = secrets.token_bytes(32)
        self.workers[worker_id] = {"secret": base64.urlsafe_b64encode(secret).decode().rstrip("="),
                                   "created": time.time(), "revoked": False}
        self.save()
        return f"{PREFIX}.{worker_id}.{self.workers[worker_id]['secret']}"

    def revoke(self, worker_id: str) -> bool:
        if worker_id in self.workers:
            self.workers[worker_id]["revoked"] = True
            self.save()
            return True
        return False

    def secret_for(self, worker_id: str) -> bytes | None:
        self.reload_if_changed()
        w = self.workers.get(worker_id)
        if not w or w.get("revoked"):
            return None
        return _b64d(w["secret"])


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def parse_invite(token: str) -> tuple[str, bytes] | None:
    """'dgpt1.<id>.<secret>' -> (worker_id, secret bytes); None if this is not an invite token."""
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != PREFIX:
        return None
    return parts[1], _b64d(parts[2])


# ---- signing ------------------------------------------------------------------------------------

def _canonical(method: str, path_query: str, ts: str, nonce: str, body: bytes | None) -> bytes:
    body_hash = hashlib.sha256(body or b"").hexdigest()
    return "\n".join([method.upper(), path_query, ts, nonce, body_hash]).encode()


def sign(secret: bytes, method: str, path_query: str, ts: str, nonce: str, body: bytes | None) -> str:
    return hmac.new(secret, _canonical(method, path_query, ts, nonce, body), hashlib.sha256).hexdigest()


def sign_body(secret: bytes, body: bytes) -> str:
    """Response signature: HMAC over the response body only."""
    return hmac.new(secret, hashlib.sha256(body).digest(), hashlib.sha256).hexdigest()


def request_headers(worker_id: str, secret: bytes, method: str, url: str, body: bytes | None) -> dict[str, str]:
    u = urlparse(url)
    path_query = u.path + (f"?{u.query}" if u.query else "")
    ts, nonce = str(int(time.time())), secrets.token_hex(16)
    return {"X-Worker": worker_id, "X-Timestamp": ts, "X-Nonce": nonce,
            "X-Signature": sign(secret, method, path_query, ts, nonce, body)}


class AuthError(Exception):
    pass


class Verifier:
    """Coordinator side: verify a signed request. Keeps a nonce set for replay protection."""

    def __init__(self, registry: Registry):
        self.registry = registry
        self.nonces: dict[str, float] = {}

    def verify(self, headers: dict, method: str, path_query: str, body: bytes | None, now: float | None = None) -> str:
        now = time.time() if now is None else now
        h = {k.lower(): v for k, v in headers.items()}
        wid, ts, nonce, sig = h.get("x-worker"), h.get("x-timestamp"), h.get("x-nonce"), h.get("x-signature")
        if not all((wid, ts, nonce, sig)):
            raise AuthError("missing auth headers (X-Worker, X-Timestamp, X-Nonce, X-Signature)")
        secret = self.registry.secret_for(wid)
        if secret is None:
            raise AuthError(f"unknown or revoked worker {wid!r}")
        try:
            skew = abs(now - int(ts))
        except ValueError:
            raise AuthError("bad timestamp")
        if skew > WINDOW_S:
            raise AuthError(f"timestamp outside the {WINDOW_S}s window (skew {skew:.0f}s); check the clock")
        expected = sign(secret, method, path_query, ts, nonce, body)
        if not hmac.compare_digest(expected, sig):
            raise AuthError("bad signature")
        self._prune(now)
        if nonce in self.nonces:
            raise AuthError("replayed nonce")
        self.nonces[nonce] = now
        return wid

    def _prune(self, now: float) -> None:
        if len(self.nonces) > 10000 or (self.nonces and now - min(self.nonces.values()) > NONCE_TTL_S):
            self.nonces = {n: t for n, t in self.nonces.items() if now - t <= NONCE_TTL_S}
