# dgpt: volunteer-compute training for a small GPT

Several machines train one character-level GPT together over plain HTTP. Each worker downloads the
current weights, trains K steps on its slice of the data, uploads the difference, and the coordinator
averages the differences into the next version. Syncing every K steps instead of every step cuts
network traffic by roughly K× at a small cost in loss. Workers connect outbound only, so they need no
open ports; the coordinator survives dead workers, stragglers, and its own restarts.

## Layout

| | |
|---|---|
| `dgpt/` | the installable package: model, data, config, merge, protocol, auth, coordinator, worker, evaluate, and the installer scripts it serves |
| `scripts/` | things you run: the single-machine control (`baseline.py`), experiment queues, failure injection, plots, the tunnel |
| `tests/` | unit tests for the merge math, wire format, and auth; an end-to-end fault-tolerance test |
| `docs/` | `DISTRIBUTING.md` (running a node anywhere), `SECURITY.md` (credential model) |

## Setting up the orchestrator (coordinator)

Runs on any machine every worker can reach. Needs Python 3.10+.

```sh
pip install -e .                                   # or: uv tool install .
python3 -m dgpt.data tinyshakespeare                    # tokenize the bundled dataset (text8: download to data/text8/text8 first)

dgpt-coordinator --auth workers.json --invite alice   # one credential per worker; send the printed token to them
dgpt-coordinator --auth workers.json --run-name pool1 \
    --set local_steps=25 --set total_steps=3000 --train-set lr=4e-3
```

`--set key=value` overrides the run settings (`local_steps` = K, `total_steps` = pool budget,
`n_shards`, `outer_lr`, `outer_momentum`, `delta_dtype`, ...), `--train-set` the model and training
recipe (`dataset`, `lr`, `n_layer`, `n_embd`, ...). Drop `--auth` on a private LAN. Workers start training as soon as they register; add
`--set start_workers=N` to hold the start until N have joined. With `--set adaptive_k=true` each
worker gets a per-round step count from its measured speed *and* its measured transfer time (download +
upload), so a fast GPU behind a slow tunnel trains less but still merges; rounds close when everyone has
reported or at `round_timeout_factor` × the median cycle time. Results, the
per-round log, and a resumable checkpoint land in `results/<run-name>/`; restart with `--resume`.
Only port 8000 on this machine has to be reachable (LAN address, port-forward, tunnel, or Tailscale).
See `docs/SECURITY.md` for the credential model.

### Every flag on a real launch

The three-machine text8 run was started like this:

```sh
python3 -m dgpt.coordinator --run-name text8_four --port 8000 --auth workers.json \
  --with-local-worker --local-threads 4 \
  --set start_workers=3 --set local_steps=100 --set total_steps=3000 --set n_shards=3 \
  --set shard_mode=full --set adaptive_k=true --set delta_dtype=bfloat16 \
  --set round_timeout_floor_s=20 --set round_timeout_factor=2 --set round_timeout_initial_s=90 \
  --train-set dataset=text8 --train-set lr=1e-3 --train-set n_layer=6 --train-set n_head=6 --train-set n_embd=384 --fresh
```

**Coordinator flags** (process-level; `python3 -m dgpt.coordinator` and `dgpt-coordinator` are the same program)

| flag | what it does |
|---|---|
| `--run-name text8_four` | Names the run. Everything it produces goes to `results/text8_four/`: `coordinator.jsonl` (one row per merge, delta, register, death), `ckpt.pt` (weights + version + step, rewritten atomically after every merge), `metrics.json` (final val loss, bytes moved, stale count), `coordinator.cmd` (this command, so a crashed coordinator can be relaunched verbatim). |
| `--port 8000` | TCP port the HTTP API listens on (`--host` defaults to `0.0.0.0`, all interfaces). Workers connect here directly, or through the tunnel that forwards to it. |
| `--auth workers.json` | Per-worker signed credentials. The file holds one secret per invited worker; every request must carry a valid HMAC signature and identity, weights are signed on the way back, and one token can only be used by one instance at a time. Create entries with `--invite NAME`, remove with `--revoke NAME`. Omit on a trusted LAN. |
| `--with-local-worker` | Also start a worker process on this machine (named `local`, on the GPU if there is one) against `127.0.0.1`. Makes the coordinator's machine contribute compute instead of only merging. |
| `--local-threads 4` | CPU thread pool for that local worker (`torch.set_num_threads`). Nearly irrelevant on a GPU worker; keeps it from competing with the coordinator's CPU-side eval and merge. |
| `--fresh` | Ignore an existing `results/text8_four/ckpt.pt` and start at version 0. Without it a coordinator that finds a checkpoint resumes from it (that is how crash recovery works). |
| `--exit-when-done` | Exit the process after the final merge instead of keeping serving (used by the experiment runner). |

**`--set key=value`: run settings** (the distributed-training mechanism; the coordinator hands the whole set to every worker at registration, so all workers agree)

| setting | what it does |
|---|---|
| `start_workers=3` | Start barrier. Weights are not served until 3 workers have registered, so nobody trains alone for the first rounds. Default 1: start as soon as anyone shows up. |
| `local_steps=100` | K, the heart of local SGD: how many optimizer steps a worker takes on its own copy before uploading a weight delta and syncing. Larger K = fewer syncs and less bandwidth (a round moves ~64 MB per worker on this model) but the local copies drift further apart before averaging. 100 was the PRD's bandwidth target. |
| `total_steps=3000` | Pool budget, counted in steps merged across all workers. 3000 × 64 × 64 = 12.3M tokens, the same budget as the single-machine control, so val losses are comparable at equal tokens. The learning-rate schedule is also keyed on this global step. |
| `n_shards=3` | How many data shards the coordinator hands out, one per worker (freed when a worker dies, reused by a late joiner). Only matters for `shard_mode=contiguous` / `interleaved`. |
| `shard_mode=full` | How a worker samples training windows. `contiguous`: its shard is one slice of the text. `interleaved`: disjoint start positions spread over the whole text. `full`: no sharding, every worker samples random windows from the whole training split, exactly like the control. Required with adaptive K: otherwise the fastest worker's slice dominates the merged model (roadblocks R12). |
| `adaptive_k=true` | Per-worker K. Each worker gets a step count from its measured speed and its measured transfer time (download + train + upload), sized so every worker's whole cycle lands on the median worker's, `K_i = speed_i × (cycle_target − overhead_i)`. Off: every worker gets exactly K and slow ones get cut off by the timeout. |
| `delta_dtype=bfloat16` | Precision of the uploaded delta. bf16 halves the upload (21 MB instead of 43 MB on this model) at negligible loss cost, because a delta is a small difference and the average of many deltas smooths the rounding. `weights_dtype` is the download-side equivalent (kept fp32 here). |
| `round_timeout_factor=2` | A round closes when every alive worker has reported, or when it has been open for `factor × median recent cycle time` and at least `min_workers` deltas are in. 2× gives a slow worker up to twice the median cycle before the round goes on without it. |
| `round_timeout_floor_s=20` | The timeout never drops below 20 s, however fast recent rounds were. Protects a worker whose one slow transfer would otherwise miss a very short round. |
| `round_timeout_initial_s=90` | The timeout for the first rounds, before there is any cycle-time history. Set it to what the *fast* worker needs; 600 s once made round 1 wait ten minutes for two slow containers. |

Other run settings you did not set here, with their defaults: `min_workers=1` (deltas needed before a timeout close counts), `heartbeat_interval_s=5` / `dead_after_s=15` (a worker silent for 15 s is dead and its shard freed), `outer_lr=1.0` / `outer_momentum=0.0` (plain averaging of deltas; `0.7`/`0.9` gives DiLoCo's Nesterov outer step, which lost in our sweeps), `aggregation=mean` (`median` or `trimmed` for robustness to bad deltas), `loss_check=false` (reject a delta that worsens a held-out batch), `reset_inner_opt=false` (keep each worker's AdamW state across rounds), `eval_every_rounds=1` (full validation pass after every merge; ~8 s on the 10.7M model, raise it to speed up rounds), `seed=1337`.

**`--train-set key=value`: model and optimizer recipe** (identical for the control and every pool run, so losses are comparable)

| setting | what it does |
|---|---|
| `dataset=text8` | 100M characters of Wikipedia, vocab 27 (the vocab size follows the dataset). Default is Tiny Shakespeare (1.1M chars, vocab 65), which a 3000-step pool overfits. |
| `lr=1e-3` | Peak AdamW learning rate of the worker's inner optimizer. Linear warmup over `warmup_steps=100`, cosine decay to `min_lr=1e-4` by `max_steps`, keyed on the global step. Our rule of thumb: scale it linearly with the number of workers when they all train on the same data at once (4e-3 for 4 workers on Shakespeare); 1e-3 here because this pool is dominated by one or two workers. |
| `n_layer=6`, `n_head=6`, `n_embd=384` | Transformer depth, attention heads, and width: a 10.7M-parameter GPT (default 4/4/128 = 0.8M). Width must be divisible by heads. |

Other recipe settings, defaults: `block_size=64` (context length), `batch_size=64` (per worker per step; tokens per step = 4096), `weight_decay=0.1`, `beta1=0.9`, `beta2=0.95`, `grad_clip=1.0`, `dropout=0.0`, `max_steps=3000` (the schedule's horizon, keep equal to `total_steps`).

## Setting up a worker

Any computer with Python 3.10+, or any computer at all via the installer, which brings its own Python.

```sh
curl -fsSL https://<where-you-host-it>/install.sh | sh        # macOS / Linux
irm https://<where-you-host-it>/install.ps1 | iex             # Windows PowerShell

dgpt-worker --coordinator http://HOST:8000 --token dgpt1.alice.<secret>
```

The worker fetches the dataset from the coordinator, uses the GPU if it finds one (CUDA or Apple),
otherwise all but one CPU core, and exits when the run is done. Useful flags: `--device cpu`,
`--threads N`, `--name`. From a checkout: `python3 -m dgpt.worker --coordinator ...`. Details in
`docs/DISTRIBUTING.md`.

## A whole pool on one machine

For experiments, run the coordinator with a local worker and add more worker processes:

```sh
python3 -m dgpt.coordinator --run-name test --with-local-worker --set local_steps=25 --set total_steps=3000
python3 -m dgpt.worker --name w2 --coordinator http://127.0.0.1:8000 --device cpu --threads 2   # as many as you like
python3 scripts/experiments.py --list        # named experiment queues (K sweeps, worker-count sweeps, ...)
python3 scripts/experiments.py nodes_n4      # runs the coordinator plus N local CPU worker processes, then writes evals.md
```

## Fault tolerance

Workers heartbeat every 5 s; a worker silent for 15 s is reaped and its data shard freed. Stragglers upload
partial deltas before the round deadline; stale deltas are discarded and the worker refetches. The
coordinator checkpoints before serving each new version and **resumes automatically** when relaunched
with the same command (`--fresh` starts over); workers retry with backoff and re-register on their own.
`scripts/chaos.py` injects failures on a schedule against a running pool:

```sh
python3 scripts/chaos.py --run k25 --kill worker-3@60 --pause worker-1@90:20 --restart-coordinator@150
```

To break a *connection* rather than a process, put the worker behind `scripts/chaos_proxy.py` and let chaos.py
cut, lag or throttle that link on a schedule (works for any worker that can reach the proxy port; signed
requests pass through untouched):

```sh
python3 scripts/chaos_proxy.py --upstream 127.0.0.1:8000 --via w1=8001 --control 8100
python3 -m dgpt.worker --name w1 --coordinator http://127.0.0.1:8001
python3 scripts/chaos.py --run k25 --proxy http://127.0.0.1:8100 --cut w1@30:20 --lag w1@90:30:500 --throttle w1@150:60:300
```

What each fault should look like in the coordinator log: a killed or cut-off worker is `worker_dead` after 15 s
and merges continue with `n_deltas` one lower; a paused or cut worker re-`register`s by itself when it returns;
a restarted coordinator logs `start` with `resumed: true` and the next `merge` has the next version number.

## Tests

```sh
python3 -m pytest tests -q --ignore=tests/test_fault_tolerance.py   # merge, protocol, auth, session fencing, round timing,
                                                                   # fault paths (shards, malicious/stale/unknown deltas, liveness): ~10 s
python3 -m pytest tests/test_fault_tolerance.py                    # real processes: kill a worker, SIGKILL + relaunch the coordinator,
                                                                   # freeze a worker, cut a worker's connection through the proxy: ~2.5 min
```
