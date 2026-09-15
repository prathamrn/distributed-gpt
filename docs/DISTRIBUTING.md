# Running a node on any computer

A node needs three things: the `dgpt-worker` program, the coordinator's address, and (if the pool
requires one) a join token. It downloads the dataset from the coordinator on first run, picks the best
device it finds (NVIDIA GPU, Apple GPU, or CPU), and exits when the run is over. Nothing else to configure.

## For a donor (someone running a node)

macOS / Linux:
```sh
curl -fsSL https://<host>/install.sh | sh          # installs uv (+ Python if needed) and dgpt-worker
dgpt-worker --coordinator http://HOST:8000          # add --token XYZ if you were given one
```
Windows (PowerShell):
```powershell
irm https://<host>/install.ps1 | iex
dgpt-worker --coordinator http://HOST:8000
```
Given a wheel file instead of a URL: `DGPT_SRC=dgpt-0.1.0-py3-none-any.whl sh install.sh`, or directly
`uv tool install dgpt-0.1.0-py3-none-any.whl` (`pipx install` works too).

Useful flags: `--device cpu` (don't use the GPU), `--threads 4` (cap CPU use; default is all cores but
one), `--name mybox` (how you appear in the coordinator's logs), `--data-dir` (cache location, default
`$DGPT_DATA_DIR` or `./data`). Stop with Ctrl-C at any time; the pool carries on without you and your
unfinished round is simply not counted.

What leaves your machine: model updates (a few MB per round for the small model, 43 MB for the 10.7M one)
and a heartbeat every 5 s. Nothing about your files. What arrives: the current model each round and the
tokenized dataset once (2 MB for Tiny Shakespeare, 200 MB for text8).

## For the host (running the coordinator)

```sh
uv tool install dgpt-0.1.0-py3-none-any.whl          # or run from the repo: python3 -m dgpt.coordinator ...
dgpt-coordinator --auth workers.json --invite alice   # one invite token per donor; send it to them once
dgpt-coordinator --auth workers.json --run-name pool1 --set local_steps=25 --set total_steps=3000 \
    --train-set dataset=text8 --train-set lr=4e-3
```
Each donor then runs `dgpt-worker --coordinator http://HOST:8000 --token dgpt1.alice.<secret>`.
The coordinator needs the raw dataset once (`data/text8/text8` or `data/tinyshakespeare/input.txt`); it
tokenizes it and serves the result to workers. `--auth` gives every donor their own revocable credential
and signs every message (see `SECURITY.md`); `--token XYZ` is the quick shared-secret alternative for
people you trust; neither is needed on your own LAN. `--exit-when-done` stops it after the run.

## Networking: only the coordinator has to be reachable

Workers never talk to each other and never accept connections, so there is no LAN or VPN to build between
nodes. Exactly one inbound port on one machine must be reachable by every worker:

| Situation | What to do |
|---|---|
| Everyone on the same Wi-Fi / office LAN | give workers the coordinator's LAN address, e.g. `http://192.168.1.20:8000` |
| Coordinator at home, workers anywhere | forward port 8000 on the router to the coordinator machine, or run `sh scripts/tunnel.sh`: a Cloudflare quick tunnel (free, no account, no bandwidth cap; no uptime guarantee, a new URL each start, and the hostname takes ~30-60 s to appear in DNS). `TUNNEL=ngrok sh scripts/tunnel.sh` uses ngrok instead (free tier caps monthly transfer; a K=25 run moves ~800 MB) |
| Worker machine you can SSH into (a lab server) | `ssh -N -R 8000:localhost:8000 server` from the coordinator machine; the worker there uses `--coordinator http://127.0.0.1:8000`. No third party, no cap, encrypted |
| You have Tailscale | install it on the coordinator and on each worker; use the coordinator's Tailscale address. No port-forwarding, encrypted, and the closest thing to a "virtual LAN", but only the coordinator needs to be reachable |
| You have an always-on server with a stable address (e.g. a lab box) | run the coordinator there; it is the natural home for it. Workers anywhere connect outbound. If its firewall blocks 8000, an SSH tunnel from the coordinator machine works: `ssh -R 8000:localhost:8000 server` exposes a coordinator running on your laptop at `server:8000` |

Bandwidth planning for the small model: each worker moves 6.5 MB per round (3.25 MB each way, 1.6 MB each
way with `--set delta_dtype=bfloat16`). At K=25 that is ~800 MB for a whole 4-worker run.

## Building the wheel and hosting the installer

```sh
uv build --wheel                # -> dist/dgpt-0.1.0-py3-none-any.whl
```
Put the wheel and `dgpt/install/install.sh` / `install.ps1` anywhere workers can fetch them (a GitHub release, an S3
bucket, the coordinator machine via `python3 -m http.server`), and set `DGPT_SRC` in the installers to
that URL. Torch is the only large dependency (~100 MB CPU wheel; the installer picks the CPU build on
Linux machines without an NVIDIA GPU so nobody downloads 2 GB of CUDA by accident).

## What the worker does, so you can trust it

`dgpt/worker.py` is ~400 lines. It registers, downloads weights, trains K steps on its assigned slice of the
data, uploads `W_start − W_local` as raw floats (never pickles), heartbeats, and repeats. See `dgpt/protocol.py` for the exact bytes on the wire.
