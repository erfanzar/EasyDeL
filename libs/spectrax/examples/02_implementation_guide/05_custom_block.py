# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Custom :class:`spectrax.Module` transformer block written from first principles.

Walkthrough for authors who want to understand the full
``Module`` API rather than lean on
:class:`examples.models.llama.Llama3Block`. The block follows the
canonical modern-Transformer recipe:

    x -> RMSNorm -> GQA attention -> residual
      -> RMSNorm -> SwiGLU FFN   -> residual

Every :class:`spectrax.nn.Linear` inside carries an explicit
``sharding=("fsdp", "tp")`` (or its row-parallel mirror) so the
block composes under :func:`spectrax.sharding.logical_axis_rules`
the same way the production Llama/Qwen blocks do.

Run::

    python -m examples.02_implementation_guide.05_custom_block
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax import nn


class MyBlock(spx.Module):
    """Pre-norm GQA + SwiGLU block built by hand from :mod:`spectrax.nn` primitives.

    The projections follow the standard column/row-parallel pattern:
    q/k/v and the two FFN up-projections are column-parallel
    (``sharding=("fsdp", "tp")``) while the o and down projections
    are row-parallel (``sharding=("tp", "fsdp")``).
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, ffn: int, *, rngs: spx.Rngs):
        """Allocate the two RMSNorms and the six projection Linears."""
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        kv_d = n_kv_heads * self.head_dim
        self.norm1 = nn.RMSNorm(d_model)
        self.q = nn.Linear(d_model, d_model, use_bias=False, sharding=("fsdp", "tp"), rngs=rngs)
        self.k = nn.Linear(d_model, kv_d, use_bias=False, sharding=("fsdp", "tp"), rngs=rngs)
        self.v = nn.Linear(d_model, kv_d, use_bias=False, sharding=("fsdp", "tp"), rngs=rngs)
        self.o = nn.Linear(d_model, d_model, use_bias=False, sharding=("tp", "fsdp"), rngs=rngs)
        self.norm2 = nn.RMSNorm(d_model)
        self.gate = nn.Linear(d_model, ffn, use_bias=False, sharding=("fsdp", "tp"), rngs=rngs)
        self.up = nn.Linear(d_model, ffn, use_bias=False, sharding=("fsdp", "tp"), rngs=rngs)
        self.down = nn.Linear(ffn, d_model, use_bias=False, sharding=("tp", "fsdp"), rngs=rngs)

    def forward(self, x):
        """Run ``attn(norm1(x)) + x`` then ``swiglu(norm2(x)) + x``."""
        b, t, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).reshape(b, t, self.n_heads, self.head_dim)
        k = self.k(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        v = self.v(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        attn = jax.nn.dot_product_attention(q, k, v, scale=1.0 / jnp.sqrt(self.head_dim), is_causal=True).reshape(
            b, t, self.d_model
        )
        x = x + self.o(attn)
        h = self.norm2(x)
        return x + self.down(jax.nn.silu(self.gate(h)) * self.up(h))


def main():
    """Build one :class:`MyBlock`, run a forward pass, and print the output shape."""
    block = MyBlock(d_model=128, n_heads=4, n_kv_heads=2, ffn=256, rngs=spx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 32, 128))
    y = block(x)
    jax.block_until_ready(y)
    parameter_count = sum(v.value.size for _, v in spx.iter_variables(block) if isinstance(v, spx.Parameter))
    print(f"custom block input shape : {x.shape}")
    print(f"custom block output shape: {y.shape}")
    print(f"custom block parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
