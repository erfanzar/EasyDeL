# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal GPT-2 written from scratch against the :mod:`spectrax.nn` surface.

Self-contained (no imports from :mod:`examples.models`) so the
file reads as a **complete** recipe for porting a classic
architecture to :mod:`spectrax`. Covers the defining GPT-2
choices that set it apart from Llama/Qwen:

* learned positional embeddings rather than RoPE;
* :class:`spectrax.nn.LayerNorm` with affine bias (not RMSNorm);
* post-GELU dense FFN (no SwiGLU gate);
* multi-head self-attention with a causal mask.

The implementation uses :class:`spectrax.nn.CausalSelfAttention`
for the attention block and hand-rolls the rest so the reader can
see exactly how the Transformer pieces plug together.

Run::

    python -m examples.02_implementation_guide.03_gpt2
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax import nn


@dataclass
class GPT2Config:
    """Small GPT-2 hyperparameters for CPU-friendly experiments."""

    vocab: int = 256
    max_seq: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4


class GPT2Block(spx.Module):
    """Pre-LN block: causal attention residual, then GELU MLP residual."""

    def __init__(self, cfg: GPT2Config, *, rngs: spx.Rngs):
        """Build the two LayerNorms, the causal attention, and the two FFN Linears."""
        super().__init__()
        d = cfg.d_model
        ffn = cfg.d_model * cfg.ffn_mult
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.CausalSelfAttention(d, num_heads=cfg.n_heads, rngs=rngs)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn, sharding=("fsdp", "tp"), rngs=rngs)
        self.fc2 = nn.Linear(ffn, d, sharding=("tp", "fsdp"), rngs=rngs)

    def forward(self, x):
        """Run ``attn(ln1(x)) + x`` then ``mlp(ln2(x)) + x``."""
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(jax.nn.gelu(self.fc1(self.ln2(x))))


class GPT2(spx.Module):
    """Token + learned positional embeddings → GPT-2 blocks → LM head."""

    def __init__(self, cfg: GPT2Config, *, rngs: spx.Rngs):
        """Build token and position embeddings, the block stack, the final LN, and the head."""
        super().__init__()
        self.tok = nn.Embed(cfg.vocab, cfg.d_model, rngs=rngs)
        self.pos = nn.Embed(cfg.max_seq, cfg.d_model, rngs=rngs)
        self.blocks = nn.ModuleList([GPT2Block(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, use_bias=False, rngs=rngs)

    def forward(self, ids):
        """Run ``tok + pos -> blocks -> final LN -> head`` on a batch of ids."""
        _, t = ids.shape
        x = self.tok(ids) + self.pos(jnp.arange(t))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def main():
    """Build a tiny GPT-2, run one forward pass, and print the output shape."""
    cfg = GPT2Config()
    model = GPT2(cfg, rngs=spx.Rngs(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (4, 32), 0, cfg.vocab)
    logits = model(ids)
    jax.block_until_ready(logits)
    parameter_count = sum(v.value.size for _, v in spx.iter_variables(model) if isinstance(v, spx.Parameter))
    print(f"gpt2 logits shape: {logits.shape}")
    print(f"gpt2 parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
