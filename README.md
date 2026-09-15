# dgpt: volunteer-compute training for a small GPT

Several machines train one character-level GPT together over plain HTTP. Each worker downloads the
current weights, trains K steps on its slice of the data, uploads the difference, and the coordinator
averages the differences into the next version. Syncing every K steps instead of every step cuts
network traffic by roughly K× at a small cost in loss. Workers connect outbound only, so they need no
open ports; the coordinator survives dead workers, stragglers, and its own restarts.

## Setting up the orchestrator (coordinator)

Runs on any machine every worker can reach. Needs Python 3.10+.

```sh
pip install -e .                                   # or: uv tool install .
python3 data.py tinyshakespeare                    # tokenize the bundled dataset (text8: download to data/text8/text8 first)

dgpt-coordinator --auth workers.json --invite alice   # one credential per worker; send the printed token to them
dgpt-coordinator --auth workers.json --run-name pool1 \
    --set local_steps=25 --set total_steps=3000 --train-set lr=4e-3
```

`--set key=value` overrides the run settings (`local_steps` = K, `total_steps` = pool budget,
`n_shards`, `outer_lr`, `outer_momentum`, `delta_dtype`, ...), `--train-set` the model and training
recipe (`dataset`, `lr`, `n_layer`, `n_embd`, ...). Drop `--auth` on a private LAN. Results, the
per-round log, and a resumable checkpoint land in `results/<run-name>/`; restart with `--resume`.
Only port 8000 on this machine has to be reachable (LAN address, port-forward, tunnel, or Tailscale).
See `SECURITY.md` for the credential model.

## Setting up a worker

Any computer with Python 3.10+, or any computer at all via the installer, which brings its own Python.

```sh
curl -fsSL https://<where-you-host-it>/install.sh | sh        # macOS / Linux
irm https://<where-you-host-it>/install.ps1 | iex             # Windows PowerShell

dgpt-worker --coordinator http://HOST:8000 --token dgpt1.alice.<secret>
```

The worker fetches the dataset from the coordinator, uses the GPU if it finds one (CUDA or Apple),
otherwise all but one CPU core, and exits when the run is done. Useful flags: `--device cpu`,
`--threads N`, `--name`. From a checkout: `python3 worker.py --coordinator ...`. Details in
`DISTRIBUTING.md`.

## A whole pool on one machine

For experiments, run the coordinator with a local worker and add more worker processes:

```sh
python3 coordinator.py --run-name test --with-local-worker --set local_steps=25 --set total_steps=3000
python3 worker.py --name w2 --coordinator http://127.0.0.1:8000 --device cpu --threads 2   # as many as you like
python3 experiments.py --list        # named experiment queues (K sweeps, worker-count sweeps, ...)
python3 experiments.py nodes_n4      # runs the coordinator plus N local CPU worker processes, then writes evals.md
```

## Fault tolerance

Workers heartbeat every 5 s; a worker silent for 15 s is reaped and its data shard freed. Stragglers upload
partial deltas before the round deadline; stale deltas are discarded and the worker refetches. The
coordinator checkpoints before serving each new version and **resumes automatically** when relaunched
with the same command (`--fresh` starts over); workers retry with backoff and re-register on their own.
`chaos.py` injects failures on a schedule against a running pool:

```sh
python3 chaos.py --run k25 --kill worker-3@60 --pause worker-1@90:20 --restart-coordinator@150
```

## Tests

```sh
python3 -m pytest tests -q                       # merge, protocol, auth: ~1 s
python3 -m pytest tests/test_fault_tolerance.py  # real processes: kill a worker, SIGKILL + relaunch the coordinator: ~1 min
```
