# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qwen 2.5 — written **once**, runs under SPMD or MPMD via :func:`spectrax.run`.

Mirrors :mod:`examples.models.llama` in layout: a single :class:`Qwen`
top-level model assembled from :class:`QwenEmbed`, a list of
:class:`QwenBlock` layers, and :class:`QwenLMHead`. The block,
embedding, and head each expose a single-argument ``forward(x)`` so
the chain composes cleanly with
:func:`spectrax.runtime.primitives.split.auto_split`.

Qwen 2.5 differs from Llama 3 in three architectural details that
are exercised here:

* **QKV bias** — the q/k/v projections carry a learned bias while
  the output projection and the FFN projections remain bias-free.
* **RoPE theta** — Qwen 2.5 uses a larger rotary base
  (``theta=1_000_000``) than Llama 3's ``500_000``.
* **Tied embeddings** — Qwen 2.5 ties input/output embeddings for
  the smaller sizes and leaves them untied for the larger ones; for
  simplicity (and to keep parity with
  :class:`examples.models.llama.Llama3`) this file keeps them
  untied unconditionally.

Every :class:`spectrax.nn.Linear` in this file is annotated with
role-specific FSDP+TP logical sharding:

* column-parallel  (q/k/v/gate/up):    ``("fsdp", "tp_*")``
* row-parallel     (o/down):           ``("tp_*", "fsdp")``

The module also exports :data:`FSDP_TP_RULES`, the same
logical-to-physical axis-rules list used by :mod:`examples.models.llama`,
intended for use inside a
:func:`spectrax.sharding.logical_axis_rules` context.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax import nn

__all__ = [
    "FSDP_TP_RULES",
    "Qwen",
    "QwenBlock",
    "QwenConfig",
    "QwenEmbed",
    "QwenLMHead",
]


FSDP_TP_RULES: list[tuple[str, str | None]] = [
    ("tp_head", "tp"),
    ("tp_ffn", "tp"),
    ("tp_vocab", "tp"),
    ("fsdp", "fsdp"),
    ("vocab", "tp"),
    ("embed", "fsdp"),
    ("features", None),
    ("in", "fsdp"),
    ("out", "tp"),
]


@dataclass
class QwenConfig:
    """Qwen 2.5-shaped hyperparameters.

    Holds the knobs shared by :class:`QwenEmbed`, :class:`QwenBlock`,
    and :class:`QwenLMHead`: vocabulary size, model dimension, head
    counts (including GQA ``n_kv_heads``), FFN width, layer count,
    RoPE base, and parameter dtype.
    """

    vocab: int = 151_936
    d_model: int = 1024
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn: int = 2_816
    n_layers: int = 4
    rope_theta: float = 1_000_000.0
    dtype: jnp.dtype = jnp.float32

    @property
    def head_dim(self) -> int:
        """Per-head hidden size, i.e. ``d_model // n_heads``."""
        return self.d_model // self.n_heads


def _rope_freqs(seq_len: int, head_dim: int, theta: float, dtype: jnp.dtype):
    """Return ``(cos, sin)`` rotary-position tables for the given sequence length.

    Computed in float32 for numerical stability and cast to ``dtype``
    on the way out.
    """
    half = head_dim // 2
    inv = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv)
    return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def _apply_rope(x, cos, sin):
    """Apply rotary-position embeddings to a ``(b, t, h, d)`` tensor."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class QwenBlock(spx.Module):
    """One Qwen 2.5 transformer block — RoPE + GQA attention (with QKV bias) + SwiGLU FFN.

    ``forward(x)`` is single-argument (RoPE tables are computed
    inside the block from ``x.shape``), so the block composes
    cleanly with :func:`spectrax.runtime.primitives.split.auto_split`'s
    default ``x -> x`` chain.

    The attention uses :func:`jax.nn.dot_product_attention`, which
    is a flash-style fused attention that never materializes a
    ``(b, h, s, s)`` score matrix. Grouped-query attention is
    supported natively (``n_kv_heads < n_heads``).

    Unlike :class:`examples.models.llama.Llama3Block`, the q/k/v
    projections carry a learned bias — this is the defining knob of
    Qwen 2's attention. The output projection and both FFN
    projections remain bias-free.
    """

    def __init__(self, cfg: QwenConfig, *, rngs: spx.Rngs):
        """Build the block's RMSNorms and attention / FFN Linears from ``cfg``."""
        super().__init__()
        self.d_model = int(cfg.d_model)
        self.n_heads = int(cfg.n_heads)
        self.n_kv_heads = int(cfg.n_kv_heads)
        self.head_dim = int(cfg.head_dim)
        self.rope_theta = float(cfg.rope_theta)
        self.dtype_str = jnp.dtype(cfg.dtype).name
        d, ffn = cfg.d_model, cfg.ffn
        kv_d = cfg.n_kv_heads * cfg.head_dim
        lin_bias = dict(use_bias=True, dtype=cfg.dtype)
        lin_nobias = dict(use_bias=False, dtype=cfg.dtype)
        self.norm1 = nn.RMSNorm(d, dtype=cfg.dtype)
        self.q = nn.Linear(d, d, sharding=("fsdp", "tp_head"), rngs=rngs, **lin_bias)
        self.k = nn.Linear(d, kv_d, sharding=("fsdp", "tp_head"), rngs=rngs, **lin_bias)
        self.v = nn.Linear(d, kv_d, sharding=("fsdp", "tp_head"), rngs=rngs, **lin_bias)
        self.o = nn.Linear(d, d, sharding=("tp_head", "fsdp"), rngs=rngs, **lin_nobias)
        self.norm2 = nn.RMSNorm(d, dtype=cfg.dtype)
        self.gate = nn.Linear(d, ffn, sharding=("fsdp", "tp_ffn"), rngs=rngs, **lin_nobias)
        self.up = nn.Linear(d, ffn, sharding=("fsdp", "tp_ffn"), rngs=rngs, **lin_nobias)
        self.down = nn.Linear(ffn, d, sharding=("tp_ffn", "fsdp"), rngs=rngs, **lin_nobias)

    def forward(self, x):
        """Run the block: pre-norm + GQA attention residual, then pre-norm + SwiGLU FFN residual."""
        b, t, _ = x.shape
        cos, sin = _rope_freqs(t, self.head_dim, self.rope_theta, jnp.dtype(self.dtype_str))
        h = self.norm1(x)
        q = self.q(h).reshape(b, t, self.n_heads, self.head_dim)
        k = self.k(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        v = self.v(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        h = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / jnp.sqrt(self.head_dim),
            is_causal=True,
        ).reshape(b, t, self.d_model)
        x = x + self.o(h)
        h = self.norm2(x)
        h = self.down(jax.nn.silu(self.gate(h)) * self.up(h))
        return x + h


class QwenEmbed(spx.Module):
    """Token-id to embedding lookup.

    The vocab axis is sharded on TP and the hidden axis on FSDP
    under the standard :data:`FSDP_TP_RULES` logical mapping.
    """

    def __init__(self, cfg: QwenConfig, *, rngs: spx.Rngs):
        """Build the embedding table sized ``(cfg.vocab, cfg.d_model)``."""
        super().__init__()
        self.embed = nn.Embed(cfg.vocab, cfg.d_model, rngs=rngs, dtype=cfg.dtype)

    def forward(self, ids):
        """Look up embeddings for a batch of token ids."""
        return self.embed(ids)


class QwenLMHead(spx.Module):
    """Final RMSNorm + projection to vocab logits.

    Uses a column-parallel projection ``(fsdp, tp_vocab)`` so the
    vocabulary dimension is sharded across TP ranks. Input and
    output embeddings are kept untied for parity with
    :class:`examples.models.llama.Llama3LMHead`.
    """

    def __init__(self, cfg: QwenConfig, *, rngs: spx.Rngs):
        """Build the output RMSNorm and the vocab projection Linear."""
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model, dtype=cfg.dtype)
        self.proj = nn.Linear(
            cfg.d_model,
            cfg.vocab,
            sharding=("fsdp", "tp_vocab"),
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
        )

    def forward(self, x):
        """Apply the final norm and project to vocab-sized logits."""
        return self.proj(self.norm(x))


class Qwen(spx.Module):
    """Qwen 2.5 transformer — ``embed -> blocks[*] -> head``.

    Top-level model written **once** and used everywhere. The same
    class runs under SPMD or MPMD, for training or inference — no
    pipeline-aware code lives in this file.

    :func:`spectrax.run` auto-splits :attr:`blocks` across pipeline
    ranks and folds :attr:`embed` onto rank 0 and :attr:`head` onto
    rank n-1 when the active mesh has an MPMD axis.
    """

    def __init__(self, cfg: QwenConfig, *, rngs: spx.Rngs):
        """Build the embedding, the list of :class:`QwenBlock` layers, and the LM head."""
        super().__init__()
        self.embed = QwenEmbed(cfg, rngs=rngs)
        self.blocks = nn.ModuleList([QwenBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.head = QwenLMHead(cfg, rngs=rngs)

    def forward(self, ids):
        """Run ``embed -> blocks -> head`` on a batch of token ids, returning vocab logits."""
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)
