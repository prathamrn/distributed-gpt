# Security model

The pool runs over the open internet with machines you do not control. This page says what is
protected, how, and what is deliberately not.

## Threats and what covers them

| Threat | Covered by | Notes |
|---|---|---|
| Anyone who finds the port joins the pool | **Per-worker invites** (`--auth workers.json`). Only workers with a credential can register or send anything | `dgpt-coordinator --auth workers.json --invite alice` prints a one-time token for one donor |
| Someone poses as an existing worker | **Signed requests + identity binding.** Every request carries `X-Worker`, `X-Timestamp`, `X-Nonce`, `X-Signature = HMAC-SHA256(secret, method + path + timestamp + nonce + sha256(body))`. The worker id inside the message must match the identity that signed it, or the request is refused (403) | the secret never travels; a donor cannot leak it by sending it |
| Replay of a captured request (e.g. re-sending an old delta) | **Nonce + timestamp.** Nonces are remembered for 5 minutes; timestamps must be within ±120 s | requires roughly synchronized clocks |
| Tampering with a request in transit | the **body hash is inside the signature**; any change invalidates it | integrity, not confidentiality |
| An impostor coordinator feeding a donor bad weights or data | **Signed responses.** Weights and the registration reply are signed back with the donor's own secret; the worker refuses to train on unsigned or mis-signed weights | a real coordinator that lost the registry file is indistinguishable from an impostor, by design |
| A donor misbehaving after being admitted | **Revocation** (`--revoke alice`, takes effect on the running coordinator within one request) | |
| A credentialed worker sending poisoned deltas | **Content defenses**, independent of identity: version + weights-hash check (stale/foreign deltas discarded), tensor layout check, optional held-out **loss check** (`--set loss_check=true`), **robust aggregation** (`--set aggregation=median` or `trimmed`) | authentication says *who* may participate; these say *what* gets in. Both layers are needed |
| Arbitrary code execution through uploaded tensors | **No pickles on the wire.** Deltas are raw floats + a JSON manifest, rebuilt with `numpy.frombuffer` | the reason `torch.save` is not used for anything that crosses the network |
| Eavesdropping on weights, deltas, or the dataset | **Not covered here.** Run the coordinator behind TLS (a `cloudflared` tunnel gives HTTPS for free; or caddy/nginx in front of port 8000) | for open training the traffic is not secret; for private data it is |
| Flooding with fake registrations (Sybil) | invites cap who can join; there is no rate limiting beyond that | fine for a pool of people you invited; not for an open sign-up page |
| A malicious coordinator harming a donor's machine | the worker only trains and uploads floats; it downloads a dataset to a cache directory you choose. It runs no code from the coordinator | read `worker.py`; it is short |

## Modes

| Mode | Flag | Use when |
|---|---|---|
| none | (default) | your own LAN or a single-machine experiment |
| shared token | `--token XYZ` on both sides | quick experiments across the internet with people you trust; one secret for everyone, cannot revoke one person, travels in the clear |
| per-worker signed | `--auth workers.json` on the coordinator; each donor runs `dgpt-worker --token dgpt1.<id>.<secret>` | anything public. Recommended together with TLS |

## Operating it

```sh
dgpt-coordinator --auth workers.json --invite alice     # -> dgpt1.alice.<secret>   (send to alice, once)
dgpt-coordinator --auth workers.json --revoke alice     # takes effect immediately on the running coordinator
dgpt-coordinator --auth workers.json --run-name pool1 ...  # normal start; refuses to start with an empty registry
```
`workers.json` holds the secrets and is written with mode 600. Back it up: without it, nobody can
rejoin after a coordinator restart and every donor needs a new invite. A worker's `--name` is ignored in
signed mode; its identity is the invite's id.

## What a reviewer should check

- `dgpt/auth.py` is ~130 lines: `sign`, `Verifier.verify`, `Registry`. Tests in `tests/test_auth.py` cover
  wrong secret, tampered body and path, replay, stale timestamp, revocation (including from another
  process), and identity binding.
- The coordinator applies verification in one middleware for every route except `/health`, and binds
  identity in each handler (`bind(...)`) plus inside `/delta` after the body is parsed.
- The worker signs through a `requests` auth hook, so every call from `worker.py` is covered without
  each call site remembering to.
