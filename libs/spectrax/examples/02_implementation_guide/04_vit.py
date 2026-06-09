# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal Vision Transformer (ViT) written from scratch.

Self-contained — nothing imported from :mod:`examples.models`.
Demonstrates the canonical ViT pipeline end to end:

* **patch embed** via a :math:`p \\times p` strided
  :class:`spectrax.nn.Conv2d` that folds spatial patches into a
  token sequence;
* a learned ``[CLS]`` token prepended to the sequence;
* additive learned positional embeddings across the patch grid;
* a stack of pre-LN Transformer encoder blocks with
  :class:`spectrax.nn.MultiheadAttention` + a GELU MLP;
* final :class:`spectrax.nn.LayerNorm` and a classifier head that
  reads the ``[CLS]`` row.

The sizing is deliberately tiny (image 32x32, patch 8, dim 128)
so the example runs on a laptop CPU.

Run::

    python -m examples.02_implementation_guide.04_vit
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
class ViTConfig:
    """Tiny ViT hyperparameters suitable for CPU demonstrations."""

    image_size: int = 32
    patch_size: int = 8
    in_channels: int = 3
    num_classes: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4

    @property
    def num_patches(self) -> int:
        """Count of non-overlapping patches per image."""
        side = self.image_size // self.patch_size
        return side * side


class ViTBlock(spx.Module):
    """Pre-LN encoder block: self-attention residual + GELU MLP residual."""

    def __init__(self, cfg: ViTConfig, *, rngs: spx.Rngs):
        """Build the two LayerNorms, the self-attention, and the two FFN Linears."""
        super().__init__()
        d = cfg.d_model
        ffn = cfg.d_model * cfg.ffn_mult
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads=cfg.n_heads, rngs=rngs)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn, sharding=("fsdp", "tp"), rngs=rngs)
        self.fc2 = nn.Linear(ffn, d, sharding=("tp", "fsdp"), rngs=rngs)

    def forward(self, x):
        """Run ``attn(ln1(x)) + x`` then ``mlp(ln2(x)) + x`` with bidirectional attention."""
        h = self.ln1(x)
        x = x + self.attn(h, h, h)
        return x + self.fc2(jax.nn.gelu(self.fc1(self.ln2(x))))


class ViT(spx.Module):
    """Vision Transformer: patches → CLS + pos → encoder blocks → classifier."""

    def __init__(self, cfg: ViTConfig, *, rngs: spx.Rngs):
        """Build the patch convolution, learned tokens, encoder stack, and classifier."""
        super().__init__()
        self.d_model = int(cfg.d_model)
        self.patch = nn.Conv2d(
            cfg.in_channels,
            cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            rngs=rngs,
        )
        n = cfg.num_patches + 1
        self.cls = spx.Parameter(jnp.zeros((1, 1, cfg.d_model), dtype=jnp.float32))
        self.pos = spx.Parameter(jnp.zeros((1, n, cfg.d_model), dtype=jnp.float32))
        self.blocks = nn.ModuleList([ViTBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes, rngs=rngs)

    def forward(self, images):
        """Classify a batch of ``(b, h, w, c)`` images; returns ``(b, num_classes)`` logits."""
        b = images.shape[0]
        patches = self.patch(images).reshape(b, -1, self.d_model)
        cls = jnp.broadcast_to(self.cls.value, (b, 1, self.d_model))
        x = jnp.concatenate([cls, patches], axis=1) + self.pos.value
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x)[:, 0])


def main():
    """Build a tiny ViT, run one forward pass on random images, and print the output shape."""
    cfg = ViTConfig()
    model = ViT(cfg, rngs=spx.Rngs(0))
    images = jax.random.normal(jax.random.PRNGKey(1), (4, cfg.image_size, cfg.image_size, cfg.in_channels))
    logits = model(images)
    jax.block_until_ready(logits)
    parameter_count = sum(v.value.size for _, v in spx.iter_variables(model) if isinstance(v, spx.Parameter))
    print(f"vit logits shape: {logits.shape}")
    print(f"vit parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
