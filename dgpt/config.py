"""Single source of truth for the training recipe.

The baseline (control) and every distributed run must use the same model and
the same total token budget so loss curves are comparable at equal tokens
(PRD success criterion 1). Change numbers here, nowhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from dgpt.model import GPTConfig


@dataclass
class TrainConfig:
    # data
    dataset: str = "tinyshakespeare"   # tinyshakespeare | text8 (see data.DATASETS); vocab_size follows the dataset
    # model
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    # optimizer (inner / AdamW)
    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    # schedule / budget
    batch_size: int = 64
    max_steps: int = 3000          # 3000 * 64 * 64 = 12.3M tokens (~11 epochs of train split)
    eval_interval: int = 100
    eval_batches: int = 40         # quick eval during training (fixed seed => same batches)
    seed: int = 1337

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.block_size

    @property
    def total_tokens(self) -> int:
        return self.max_steps * self.tokens_per_step

    def model_config(self) -> GPTConfig:
        return GPTConfig(
            vocab_size=self.vocab_size, block_size=self.block_size, n_layer=self.n_layer,
            n_head=self.n_head, n_embd=self.n_embd, dropout=self.dropout,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens_per_step"] = self.tokens_per_step
        d["total_tokens"] = self.total_tokens
        return d


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr, keyed on the *global* step so
    workers in the distributed run can reproduce the same schedule."""
    import math
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    frac = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (cfg.lr - cfg.min_lr)


@dataclass
class RunConfig:
    """Knobs for one distributed run. The coordinator owns this and hands it to
    workers at registration. Edit here or override with --set key=value."""
    run_name: str = "run"
    # local SGD
    local_steps: int = 25            # K: steps a worker trains alone between syncs
    total_steps: int = 3000          # pool-cumulative steps; equals the baseline's budget => equal tokens
    adaptive_k: bool = False         # K_i = K * speed_i / median_speed (PRD 7.9 straggler adaptation)
    reset_inner_opt: bool = False    # re-create the worker's AdamW at every round (FedAvg-style) instead of persisting it
    # rounds
    min_workers: int = 1             # a round may close on timeout only once this many deltas arrived
    round_timeout_factor: float = 1.5   # timeout = factor * median of recent full-round times
    round_timeout_floor_s: float = 5.0
    round_timeout_initial_s: float = 120.0   # before any round-time history exists
    # liveness
    heartbeat_interval_s: float = 5.0
    dead_after_s: float = 15.0
    # outer optimizer (coordinator)
    # Default is plain averaging (decision 2026-09-13, roadblocks R6): DiLoCo's Nesterov (0.7, 0.9) lost at every
    # K we tried over 30-150 rounds. Set outer_lr=0.7 outer_momentum=0.9 to get DiLoCo's outer step back.
    outer_lr: float = 1.0
    outer_momentum: float = 0.0
    outer_nesterov: bool = True
    aggregation: str = "mean"        # mean | median | trimmed
    trim_fraction: float = 0.1       # for aggregation=trimmed
    loss_check: bool = False         # PRD 7.11: reject deltas that worsen a held-out batch
    loss_check_margin: float = 0.05
    # data
    n_shards: int = 4
    shard_mode: str = "contiguous"   # contiguous | interleaved | full (no sharding, max overlap)
    # wire
    delta_dtype: str = "float32"     # float32 | bfloat16 : precision of uploaded deltas
    weights_dtype: str = "float32"   # precision of downloaded weights
    # bookkeeping
    eval_every_rounds: int = 1
    checkpoint_dir: str = ""         # default results/<run_name>
    seed: int = 1337

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
