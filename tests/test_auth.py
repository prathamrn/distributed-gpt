import os
import time

import pytest

from auth import AuthError, Registry, Verifier, parse_invite, request_headers, sign_body


@pytest.fixture
def reg(tmp_path):
    return Registry(str(tmp_path / "workers.json"))


def test_invite_roundtrip_and_file_perms(reg):
    tok = reg.invite("alice")
    wid, secret = parse_invite(tok)
    assert wid == "alice" and reg.secret_for("alice") == secret
    assert oct(os.stat(reg.path).st_mode & 0o777) == "0o600"
    assert parse_invite("just-a-shared-token") is None


def test_signed_request_verifies_and_binds_identity(reg):
    tok = reg.invite("alice"); wid, secret = parse_invite(tok)
    body = b"\x00\x01payload"
    h = request_headers(wid, secret, "POST", "http://c:8000/delta?x=1", body)
    v = Verifier(reg)
    assert v.verify(h, "POST", "/delta?x=1", body) == "alice"


def test_wrong_secret_rejected(reg):
    reg.invite("alice"); tok_b = reg.invite("bob"); _, secret_b = parse_invite(tok_b)
    h = request_headers("alice", secret_b, "POST", "http://c/delta", b"x")     # bob signing as alice
    with pytest.raises(AuthError, match="bad signature"):
        Verifier(reg).verify(h, "POST", "/delta", b"x")


def test_tampered_body_or_path_rejected(reg):
    _, secret = parse_invite(reg.invite("alice"))
    h = request_headers("alice", secret, "POST", "http://c/delta", b"good")
    v = Verifier(reg)
    with pytest.raises(AuthError):
        v.verify(h, "POST", "/delta", b"evil")
    with pytest.raises(AuthError):
        v.verify(h, "POST", "/register", b"good")


def test_replay_rejected(reg):
    _, secret = parse_invite(reg.invite("alice"))
    h = request_headers("alice", secret, "GET", "http://c/weights", None)
    v = Verifier(reg)
    v.verify(h, "GET", "/weights", None)
    with pytest.raises(AuthError, match="replayed"):
        v.verify(h, "GET", "/weights", None)


def test_stale_timestamp_rejected(reg):
    _, secret = parse_invite(reg.invite("alice"))
    h = request_headers("alice", secret, "GET", "http://c/weights", None)
    with pytest.raises(AuthError, match="window"):
        Verifier(reg).verify(h, "GET", "/weights", None, now=time.time() + 1000)


def test_revoked_and_unknown_rejected(reg):
    _, secret = parse_invite(reg.invite("alice"))
    h = request_headers("alice", secret, "GET", "http://c/weights", None)
    v = Verifier(reg)
    reg.revoke("alice")
    with pytest.raises(AuthError, match="revoked"):
        v.verify(h, "GET", "/weights", None)
    h2 = request_headers("mallory", secret, "GET", "http://c/weights", None)
    with pytest.raises(AuthError, match="unknown"):
        v.verify(h2, "GET", "/weights", None)


def test_response_signature(reg):
    _, secret = parse_invite(reg.invite("alice"))
    assert sign_body(secret, b"weights") == sign_body(secret, b"weights")
    assert sign_body(secret, b"weights") != sign_body(secret, b"weightz")


def test_revocation_from_another_process_is_picked_up(tmp_path):
    """The coordinator holds a Registry in memory; a `--revoke` run elsewhere edits the file. It must take effect."""
    path = str(tmp_path / "workers.json")
    running = Registry(path); _, secret = parse_invite(running.invite("alice"))
    other = Registry(path); other.revoke("alice")          # separate process, same file
    h = request_headers("alice", secret, "GET", "http://c/weights", None)
    with pytest.raises(AuthError, match="revoked"):
        Verifier(running).verify(h, "GET", "/weights", None)
