"""Per-worker credentials and request signing (see SECURITY.md).

    X-Worker: <worker_id>
    X-Timestamp: <unix seconds>
    X-Nonce: <random hex>"""
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
    """- {worker_id: {secret, created, revoked}} stored as JSON with mode 600.
    - A plain file, not a database: --invite/--revoke run in another process and the coordinator hot-reloads it.
    - Read by Verifier.secret_for on every request; written by the coordinator's --invite/--revoke CLI."""

    def __init__(self, path: str):
        """- Load the registry from disk, tolerating a missing file so `--invite` can create the first entry."""
        self.path = path
        self.workers: dict[str, dict] = {}
        self._mtime = 0.0
        self.reload_if_changed()

    def reload_if_changed(self) -> None:
        """- Pick up invites and revocations made by another process by re-reading workers.json when its mtime changes.
        - One stat() per call keeps the common case cheap; a revoke lands on a live coordinator without a restart.
        - Called from secret_for, so it runs on every signature check."""
        try:
            m = os.stat(self.path).st_mtime
        except FileNotFoundError:
            return
        if m != self._mtime:
            with open(self.path) as f:
                self.workers = json.load(f)
            self._mtime = m

    def save(self) -> None:
        """- Write the registry atomically: temp file, chmod 600, then os.replace onto the real path.
        - Chmod before the rename so the secrets are never world-readable at the final path, not even briefly.
        - Called by invite and revoke."""
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.workers, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)
        self._mtime = os.stat(self.path).st_mtime

    def invite(self, worker_id: str) -> str:
        """- Create (or rotate) a credential for worker_id and return the token dgpt1.<id>.<secret> to hand to the donor.
        - The id travels inside the token, which is why --name is ignored in signed mode: identity is not self-declared.
        - Called by dgpt-coordinator --invite."""
        if not worker_id or "." in worker_id or "/" in worker_id:
            raise ValueError("worker_id must be non-empty and contain no '.' or '/'")
        secret = secrets.token_bytes(32)
        self.workers[worker_id] = {"secret": base64.urlsafe_b64encode(secret).decode().rstrip("="),
                                   "created": time.time(), "revoked": False}
        self.save()
        return f"{PREFIX}.{worker_id}.{self.workers[worker_id]['secret']}"

    def revoke(self, worker_id: str) -> bool:
        """- Flag a worker as revoked and persist it; returns False for an unknown id so the CLI can say so.
        - Flagging rather than deleting keeps a record that the id existed; the save bumps mtime so a live coordinator notices.
        - Called by dgpt-coordinator --revoke."""
        if worker_id in self.workers:
            self.workers[worker_id]["revoked"] = True
            self.save()
            return True
        return False

    def secret_for(self, worker_id: str) -> bytes | None:
        """- The verifier's lookup: this worker's signing key, or None if it is unknown or revoked.
                - The hot-reload call is here rather than in the middleware so no route can forget it."""
        self.reload_if_changed()
        w = self.workers.get(worker_id)
        if not w or w.get("revoked"):
            return None
        return _b64d(w["secret"])


def _b64d(s: str) -> bytes:
    """- Decode a base64url secret whose '=' padding was stripped for the token, restoring it first."""
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def parse_invite(token: str) -> tuple[str, bytes] | None:
    """- 'dgpt1.<id>.<secret>' -> (worker_id, secret bytes); None if this is not an invite token.
        - The worker calls this once on the value of `--token` at startup"""
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != PREFIX:
        return None
    return parts[1], _b64d(parts[2])


# ---- signing ------------------------------------------------------------------------------------

def _canonical(method: str, path_query: str, ts: str, nonce: str, body: bytes | None) -> bytes:
    """- Build the exact string both sides sign.
        - Each field closes one attack, which is why all five are here."""
    body_hash = hashlib.sha256(body or b"").hexdigest()
    return "\n".join([method.upper(), path_query, ts, nonce, body_hash]).encode()


def sign(secret: bytes, method: str, path_query: str, ts: str, nonce: str, body: bytes | None) -> str:
    """- HMAC-SHA256 of the canonical string: what the worker sends as X-Signature and what Verifier recomputes.
        - HMAC rather than a public-key signature because the coordinator is the only verifier"""
    return hmac.new(secret, _canonical(method, path_query, ts, nonce, body), hashlib.sha256).hexdigest()


def sign_body(secret: bytes, body: bytes) -> str:
    """- Response signature: HMAC over the response body only.
        - Since the two already share a secret, proving origin costs one HMAC over a hash instead of a PKI."""
    return hmac.new(secret, hashlib.sha256(body).digest(), hashlib.sha256).hexdigest()


def request_headers(worker_id: str, secret: bytes, method: str, url: str, body: bytes | None) -> dict[str, str]:
    """- The worker's side of every signed call: four headers, freshly minted per request.
        - Hooked into the worker's HTTP session, so /register, /weights, /delta"""
    u = urlparse(url)
    path_query = u.path + (f"?{u.query}" if u.query else "")
    ts, nonce = str(int(time.time())), secrets.token_hex(16)
    return {"X-Worker": worker_id, "X-Timestamp": ts, "X-Nonce": nonce,
            "X-Signature": sign(secret, method, path_query, ts, nonce, body)}


class AuthError(Exception):
    """- Any reason a request failed verification.
        - The coordinator's middleware turns it into a 401 and logs the message"""
    pass


class Verifier:
    """- Coordinator side: verify a signed request and return the worker id it proves.
    - Nonce set is in memory only: the 120 s timestamp window already bounds what a restart could let through.
    - Used by the signature middleware in build_app; a failure becomes a 401 with the reason logged."""

    def __init__(self, registry: Registry):
        """- Hold the registry (looked up per request, so revocations apply live) and start an empty nonce set."""
        self.registry = registry
        self.nonces: dict[str, float] = {}

    def verify(self, headers: dict, method: str, path_query: str, body: bytes | None, now: float | None = None) -> str:
        """- Check one request (window, signature, nonce, revocation) and return the authenticated worker id, or raise AuthError.
        - Order is cheapest first; `now` is injectable for tests.
        - The returned id is what handlers bind to the worker_id inside each message (bind() in build_app)."""
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
        """- Drop expired nonces, but only when the set is large or the oldest entry is past NONCE_TTL_S.
        - Pruning on every request would be O(n) per request for a set that is almost always small."""
        if len(self.nonces) > 10000 or (self.nonces and now - min(self.nonces.values()) > NONCE_TTL_S):
            self.nonces = {n: t for n, t in self.nonces.items() if now - t <= NONCE_TTL_S}
