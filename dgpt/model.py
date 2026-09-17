"""Tiny character-level GPT, adapted from nanoGPT (Karpathy)."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """- The architecture, and the one thing every participant in a pool must agree on exactly.
        - Built by the coordinator from TrainConfig, published at /register"""
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False

    def to_dict(self) -> dict:
        """- Plain dict of the architecture: shipped to workers at /register and hashed by config_hash."""
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """- Multi-head causal self-attention; about a third of the parameters and so a third of every delta.
    - Heads are views of one projection, not separate modules, so the state dict stays small and flat."""

    def __init__(self, cfg: GPTConfig):
        """- Build the fused qkv projection and the output projection; heads must divide the embedding evenly."""
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """- Token ids in, (logits, loss) out; loss is None without targets, as during generation.
        - The single forward used by the control, by a worker's local steps, and by the coordinator's eval."""
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """- Position-wise feed-forward: 4x expansion, GELU, projection back, as in GPT-2.
    - Most of the model lives here (524K of the 804K default parameters), so most of each delta's bytes too."""

    def __init__(self, cfg: GPTConfig):
        """- Two Linears with the standard 4x hidden width; dropout is a no-op at the project's default of 0."""
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """- Token ids in, (logits, loss) out; loss is None without targets, as during generation.
        - The single forward used by the control, by a worker's local steps, and by the coordinator's eval."""
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """- One pre-LayerNorm transformer block: normalize, sublayer, add to the residual stream, twice.
    - Pre-norm rather than post-norm because it trains without warmup tricks at this scale."""

    def __init__(self, cfg: GPTConfig):
        """- Two LayerNorms, attention and MLP; n_layer of these are stacked inside GPT."""
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """- Token ids in, (logits, loss) out; loss is None without targets, as during generation.
        - The single forward used by the control, by a worker's local steps, and by the coordinator's eval."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """- The whole model.
        - Its state_dict is the unit of everything the pool does: serve, train, diff, average."""

    def __init__(self, cfg: GPTConfig):
        """- Assemble embeddings, the block stack, the final norm and the head, then apply the GPT-2 init."""
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying, as in nanoGPT / GPT-2
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """- Base init applied to every submodule by self.apply: normal(0, 0.02), zero biases where they exist."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """- Parameter count reported at startup and in metrics.json (804,096 at the defaults).
        - Doubles as a bandwidth figure: fp32 weights are 4 bytes each."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """- Token ids in, (logits, loss) out; loss is None without targets, as during generation.
        - The single forward used by the control, by a worker's local steps, and by the coordinator's eval."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def make_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)) -> torch.optim.AdamW:
        """- AdamW with weight decay on matrices only (nanoGPT convention).
        - Built once per worker at registration and kept across rounds: resetting its moments each round cost 20%."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """- Autoregressive sampling, used only by evaluate.sample.
        - Each step crops the context to block_size tokens because that is all the position table covers."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
