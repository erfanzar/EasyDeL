# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Llama 3 — written **once**, runs under SPMD or MPMD via :func:`spectrax.run`.

A single :class:`Llama3` model class, reused across all the
``examples/`` scripts. Use it for:

* training      — ``spx.run(model, inputs=ids, targets=labels, mode='train', ...)``
* prefill       — ``spx.run(model, inputs=ids, mode='forward', ...)``
* generation    — ``spx.run(model, inputs=token, mode='generate', state=cache, ...)``

For pipeline parallelism, pass an MPMD-tagged mesh
(``mpmd_axis='pp'``) and :func:`spectrax.run` auto-splits
:attr:`Llama3.blocks` across pipeline ranks — :class:`Llama3Embed`
lands on rank 0 and :class:`Llama3LMHead` on rank n-1.

Every :class:`spectrax.nn.Linear` in this file is annotated with
role-specific FSDP+TP logical sharding:

* column-parallel  (q/k/v/gate/up):    ``("fsdp", "tp_*")``
* row-parallel     (o/down):           ``("tp_*", "fsdp")``

The :class:`Llama3Block` carries its RoPE machinery internally, so
the standard ``forward(x)`` chain ``embed -> for blk in blocks:
blk(x) -> head`` composes cleanly under
:func:`spectrax.runtime.primitives.split.auto_split`.

The module also exports :data:`FSDP_TP_RULES`, a logical-to-physical
axis-rules list intended for use inside a
:func:`spectrax.sharding.logical_axis_rules` context. Beyond the
role-specific TP/FSDP keys used by this model, it also maps the
built-in axis names emitted by :class:`spectrax.nn.Embed`,
:class:`spectrax.nn.Linear`, and :class:`spectrax.nn.RMSNorm` so
that a Linear built without an explicit role-specific ``sharding=``
still gets sharded sensibly.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

import spectrax as spx
from spectrax import nn

__all__ = [
    "FSDP_TP_RULES",
    "Llama3",
    "Llama3Block",
    "Llama3Config",
    "Llama3Embed",
    "Llama3LMHead",
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
class Llama3Config:
    """Llama 3-shaped hyperparameters.

    Holds the knobs shared by :class:`Llama3Embed`, :class:`Llama3Block`,
    and :class:`Llama3LMHead`: vocabulary size, model dimension, head
    counts (including GQA ``n_kv_heads``), FFN width, layer count,
    RoPE base, and parameter dtype.
    """

    vocab: int = 32_000
    d_model: int = 1024
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn: int = 2_752
    n_layers: int = 4
    rope_theta: float = 500_000.0
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


class Llama3Block(spx.Module):
    """One Llama 3 transformer block — RoPE + GQA attention + SwiGLU FFN.

    ``forward(x)`` is single-argument (RoPE tables are computed
    inside the block from ``x.shape``), so the block composes
    cleanly with :func:`spectrax.runtime.primitives.split.auto_split`'s
    default ``x -> x`` chain.

    The attention uses :func:`jax.nn.dot_product_attention`, which
    is a flash-style fused attention that never materializes a
    ``(b, h, s, s)`` score matrix. Grouped-query attention is
    supported natively (``n_kv_heads < n_heads``) — no need to tile
    k/v with :func:`jax.numpy.repeat`. The expected layout
    ``(b, t, h, d)`` matches what this block produces, so no
    transpose is needed.

    Sharding annotations implement column-parallel projections for
    q/k/v/gate/up and row-parallel projections for o/down under the
    ``FSDP_TP_RULES`` logical-axis mapping.
    """

    def __init__(self, cfg: Llama3Config, *, rngs: spx.Rngs):
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
        lin = dict(use_bias=False, dtype=cfg.dtype)
        self.norm1 = nn.RMSNorm(d, dtype=cfg.dtype)
        self.q = nn.Linear(d, d, sharding=("fsdp", "tp_head"), rngs=rngs, **lin)
        self.k = nn.Linear(d, kv_d, sharding=("fsdp", "tp_head"), rngs=rngs, **lin)
        self.v = nn.Linear(d, kv_d, sharding=("fsdp", "tp_head"), rngs=rngs, **lin)
        self.o = nn.Linear(d, d, sharding=("tp_head", "fsdp"), rngs=rngs, **lin)
        self.norm2 = nn.RMSNorm(d, dtype=cfg.dtype)
        self.gate = nn.Linear(d, ffn, sharding=("fsdp", "tp_ffn"), rngs=rngs, **lin)
        self.up = nn.Linear(d, ffn, sharding=("fsdp", "tp_ffn"), rngs=rngs, **lin)
        self.down = nn.Linear(ffn, d, sharding=("tp_ffn", "fsdp"), rngs=rngs, **lin)

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

    def decode_step(self, x, k_cache, v_cache, pos, cos, sin):
        """Single-token decode with KV cache update.

        Args:
            x: ``(b, 1, d)`` — current token hidden state.
            k_cache: ``(b, max_seq, n_kv, hd)`` — key cache.
            v_cache: ``(b, max_seq, n_kv, hd)`` — value cache.
            pos: scalar int — current position in the sequence.
            cos, sin: full-length RoPE tables ``(max_seq, hd//2)``.

        Returns:
            ``(x, k_cache, v_cache)`` with cache updated at ``pos``.
        """
        from jax import lax

        b = x.shape[0]
        half = self.head_dim // 2
        h = self.norm1(x)
        q = self.q(h).reshape(b, 1, self.n_heads, self.head_dim)
        k = self.k(h).reshape(b, 1, self.n_kv_heads, self.head_dim)
        v = self.v(h).reshape(b, 1, self.n_kv_heads, self.head_dim)

        c = lax.dynamic_slice(cos, (pos, 0), (1, half))
        s = lax.dynamic_slice(sin, (pos, 0), (1, half))
        q = _apply_rope(q, c, s)
        k = _apply_rope(k, c, s)

        k_cache = lax.dynamic_update_slice(k_cache, k, (0, pos, 0, 0))
        v_cache = lax.dynamic_update_slice(v_cache, v, (0, pos, 0, 0))

        rep = self.n_heads // self.n_kv_heads
        max_len = k_cache.shape[1]
        qg = q.reshape(b, self.n_kv_heads, rep, 1, self.head_dim)
        kg = k_cache.transpose(0, 2, 1, 3)[:, :, None, :, :]
        vg = v_cache.transpose(0, 2, 1, 3)[:, :, None, :, :]
        attn = jnp.einsum("bgrqd,bgrkd->bgrqk", qg, kg) / jnp.sqrt(jnp.float32(self.head_dim))
        mask = jnp.arange(max_len) <= pos
        attn = jnp.where(mask[None, None, None, None, :], attn, -1e9)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        h = jnp.einsum("bgrqk,bgrkd->bgrqd", attn, vg).reshape(b, 1, self.d_model)

        x = x + self.o(h)
        h = self.norm2(x)
        h = self.down(jax.nn.silu(self.gate(h)) * self.up(h))
        return x + h, k_cache, v_cache


class Llama3Embed(spx.Module):
    """Token-id to embedding lookup.

    The vocab axis is sharded on TP and the hidden axis on FSDP
    under the standard :data:`FSDP_TP_RULES` logical mapping.
    """

    def __init__(self, cfg: Llama3Config, *, rngs: spx.Rngs):
        """Build the embedding table sized ``(cfg.vocab, cfg.d_model)``."""
        super().__init__()
        self.embed = nn.Embed(cfg.vocab, cfg.d_model, rngs=rngs, dtype=cfg.dtype)

    def forward(self, ids):
        """Look up embeddings for a batch of token ids."""
        return self.embed(ids)


class Llama3LMHead(spx.Module):
    """Final RMSNorm + projection to vocab logits.

    Uses a column-parallel projection ``(fsdp, tp_vocab)`` so the
    vocabulary dimension is sharded across TP ranks.
    """

    def __init__(self, cfg: Llama3Config, *, rngs: spx.Rngs):
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


class Llama3(spx.Module):
    """Llama 3 transformer — ``embed -> blocks[*] -> head``.

    One model, one ``forward``. Handles training, inference, AND
    generation — no separate methods, no mode flags. What ``forward``
    does depends on what you pass:

    * ``forward(ids)`` → logits ``(b, seq, vocab)`` — training / prefill.
    * ``forward(ids, k_caches, v_caches, start_pos)`` → ``(logits,
      k_caches, v_caches)`` — single decode step with KV cache update.

    For autoregressive generation, :meth:`generate` wraps ``forward``
    in a ``lax.fori_loop`` — but the actual computation is always
    ``forward``. ``generate`` is just the loop driver.
    """

    def __init__(self, cfg: Llama3Config, *, rngs: spx.Rngs):
        """Build the embedding, the list of :class:`Llama3Block` layers, and the LM head."""
        super().__init__()
        self.embed = Llama3Embed(cfg, rngs=rngs)
        self.blocks = nn.ModuleList([Llama3Block(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.head = Llama3LMHead(cfg, rngs=rngs)

    def forward(self, ids, k_caches=None, v_caches=None, start_pos=0):
        """Unified forward: prefill or cached decode based on args.

        Args:
            ids: ``(b, seq)`` token ids for prefill, or ``(b, 1)`` for
                a single decode step.
            k_caches: ``None`` for prefill (no cache), or
                ``(n_layers, b, max_seq, n_kv, hd)`` stacked KV cache
                for decode.
            v_caches: same as ``k_caches``.
            start_pos: int position offset for RoPE during decode.
                Ignored during prefill (``k_caches is None``).

        Returns:
            * Prefill (``k_caches is None``): logits ``(b, seq, vocab)``.
            * Decode (``k_caches`` provided): ``(logits, k_caches,
              v_caches)`` with caches updated at ``start_pos``.
        """

        x = self.embed(ids)

        if k_caches is None:
            for blk in self.blocks:
                x = blk(x)
            return self.head(x)

        head_dim = self.blocks[0].head_dim
        theta = self.blocks[0].rope_theta
        dtype = jnp.dtype(self.blocks[0].dtype_str)
        max_seq = k_caches.shape[2]
        cos, sin = _rope_freqs(max_seq, head_dim, theta, dtype)

        for i, blk in enumerate(self.blocks):
            x, kc_i, vc_i = blk.decode_step(
                x,
                k_caches[i],
                v_caches[i],
                start_pos,
                cos,
                sin,
            )
            k_caches = lax.dynamic_update_slice(k_caches, kc_i[None], (i, 0, 0, 0, 0))
            v_caches = lax.dynamic_update_slice(v_caches, vc_i[None], (i, 0, 0, 0, 0))

        return self.head(x), k_caches, v_caches
