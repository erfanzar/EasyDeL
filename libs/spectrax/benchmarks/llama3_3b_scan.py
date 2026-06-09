# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Llama 3B benchmark — NNX scan vs SPX scan / loop / fori_loop.

Compares layer-stack iteration patterns for a ~2.4B-parameter transformer
body (24 layers, 3072 dim, 24 heads, 8 KV heads, 8192 FFN):

* **NNX scan** — Flax NNX ``nnx.vmap`` to create blocks + ``nnx.scan`` to
  iterate, compiling one layer and running it 24 times.
* **SPX scan** — spectrax ``ModuleList.scan`` (export/stack/scan).
* **SPX loop** — plain Python ``for`` over a ``nn.ModuleList``.
* **NNX loop** — plain Python ``for`` over a list of NNX blocks.
* **SPX fori_loop** — ``spx.fori_loop`` over a ``nn.ModuleList`` (trace-aware unroll).

All SPX variants use end-to-end framework transforms (``spx.jit`` +
``spx.value_and_grad``); the NNX variant uses ``nnx.jit`` +
``nnx.value_and_grad``. No manual ``spx.export`` / ``spx.bind`` appears in
model code.

Usage::

    python -m benchmarks.llama3_3b_scan --iters 3
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from flax import nnx

import spectrax as spx
from spectrax import nn


def _rope_freqs(seq_len: int, head_dim: int, theta: float = 500_000.0):
    """Precompute RoPE cos/sin tables of shape ``(seq_len, head_dim // 2)``."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv_freq)
    return jnp.cos(freqs).astype(jnp.bfloat16), jnp.sin(freqs).astype(jnp.bfloat16)


def _apply_rope(x, cos, sin):
    """Rotate the last-dim pairs of ``x`` via cos/sin tables."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


@dataclass
class Llama3Config:
    """Llama 3B-like transformer body (no embed / lm_head)."""

    n_layers: int = 24
    d_model: int = 3072
    n_heads: int = 24
    n_kv_heads: int = 8
    ffn: int = 8192
    batch: int = 2
    seq_len: int = 128

    @property
    def total_params(self) -> int:
        """Approximate transformer-body parameter count."""
        head_dim = self.d_model // self.n_heads
        kv_d = self.n_kv_heads * head_dim
        attn = self.d_model * self.d_model + self.d_model * kv_d + self.d_model * kv_d + self.d_model * self.d_model
        ffn = 3 * self.d_model * self.ffn
        return self.n_layers * (attn + ffn)


class Llama3BlockNNX(nnx.Module):
    """One Llama 3 transformer block in Flax NNX."""

    def __init__(self, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize GQA attention + SwiGLU FFN sublayers.

        Args:
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads (GQA).
            rngs: PRNG source for parameter init.
        """
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d // n_heads
        kv_d = n_kv_heads * self.head_dim
        lin = dict(use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        self.norm1 = nnx.RMSNorm(d, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs)
        self.q = nnx.Linear(d, d, rngs=rngs, **lin)
        self.k = nnx.Linear(d, kv_d, rngs=rngs, **lin)
        self.v = nnx.Linear(d, kv_d, rngs=rngs, **lin)
        self.o = nnx.Linear(d, d, rngs=rngs, **lin)
        self.norm2 = nnx.RMSNorm(d, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs)
        self.gate = nnx.Linear(d, ffn, rngs=rngs, **lin)
        self.up = nnx.Linear(d, ffn, rngs=rngs, **lin)
        self.down = nnx.Linear(ffn, d, rngs=rngs, **lin)

    def __call__(self, x, cos, sin):
        """Run pre-norm GQA+RoPE attention followed by pre-norm SwiGLU FFN.

        Args:
            x: Input tensor ``(b, t, d)``.
            cos: RoPE cosine table.
            sin: RoPE sine table.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        b, t, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).reshape(b, t, self.n_heads, self.head_dim)
        k = self.k(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        v = self.v(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv_heads
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        attn = (q @ k.swapaxes(-1, -2)) / jnp.sqrt(self.head_dim).astype(x.dtype)
        attn = jax.nn.softmax(attn, axis=-1)
        h = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, self.d)
        h = self.o(h)
        x = x + h
        h = self.norm2(x)
        h = self.down(jax.nn.silu(self.gate(h)) * self.up(h))
        return x + h


class Llama3ModelNNXScan(nnx.Module):
    """NNX Llama 3B using ``nnx.vmap`` + ``nnx.scan`` over blocks."""

    def __init__(self, n_layers, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Stack ``n_layers`` blocks via vmap and scan them over the input.

        Args:
            n_layers: Number of transformer blocks.
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads.
            rngs: PRNG source for parameter init.
        """
        self.head_dim = d // n_heads

        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_block(rngs):
            """Create one Llama3BlockNNX under split RNGs."""
            return Llama3BlockNNX(d, ffn, n_heads, n_kv_heads, rngs=rngs)

        self.blocks = create_block(rngs)

    def __call__(self, x):
        """Run the scanned block stack on ``x``.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        _b, t, _d = x.shape
        cos, sin = _rope_freqs(t, self.head_dim)

        @nnx.scan(in_axes=(nnx.Carry, None, None, 0), out_axes=nnx.Carry)
        def forward(x, cos, sin, block):
            """Single scan step: apply one block."""
            return block(x, cos, sin)

        return forward(x, cos, sin, self.blocks)


class Llama3ModelNNXLoop(nnx.Module):
    """NNX Llama 3B using a plain Python ``for`` loop over blocks."""

    def __init__(self, n_layers, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Create ``n_layers`` blocks in an :class:`nnx.List`.

        Args:
            n_layers: Number of transformer blocks.
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads.
            rngs: PRNG source for parameter init.
        """
        self.head_dim = d // n_heads
        self.blocks = nnx.List([Llama3BlockNNX(d, ffn, n_heads, n_kv_heads, rngs=nnx.Rngs(i)) for i in range(n_layers)])

    def __call__(self, x):
        """Run every block sequentially with shared RoPE tables.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        _b, t, _d = x.shape
        cos, sin = _rope_freqs(t, self.head_dim)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x


class Llama3BlockSPX(spx.Module):
    """One Llama 3 transformer block in spectrax."""

    def __init__(self, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize GQA attention + SwiGLU FFN sublayers.

        Args:
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads (GQA).
            rngs: PRNG source for parameter init.
        """
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d // n_heads
        kv_d = n_kv_heads * self.head_dim
        lin = dict(use_bias=False, rngs=rngs, dtype=jnp.bfloat16)
        self.norm1 = nn.RMSNorm(d)
        self.q = nn.Linear(d, d, **lin)
        self.k = nn.Linear(d, kv_d, **lin)
        self.v = nn.Linear(d, kv_d, **lin)
        self.o = nn.Linear(d, d, **lin)
        self.norm2 = nn.RMSNorm(d)
        self.gate = nn.Linear(d, ffn, **lin)
        self.up = nn.Linear(d, ffn, **lin)
        self.down = nn.Linear(ffn, d, **lin)

    def forward(self, x, cos, sin):
        """Run pre-norm GQA+RoPE attention followed by pre-norm SwiGLU FFN.

        Args:
            x: Input tensor ``(b, t, d)``.
            cos: RoPE cosine table.
            sin: RoPE sine table.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        b, t, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).reshape(b, t, self.n_heads, self.head_dim)
        k = self.k(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        v = self.v(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv_heads
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        attn = (q @ k.swapaxes(-1, -2)) / jnp.sqrt(self.head_dim).astype(x.dtype)
        attn = jax.nn.softmax(attn, axis=-1)
        h = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, self.d)
        h = self.o(h)
        x = x + h
        h = self.norm2(x)
        h = self.down(jax.nn.silu(self.gate(h)) * self.up(h))
        return x + h


class Llama3ModelSPXLoop(spx.Module):
    """SPX Llama 3B using a plain Python ``for`` loop (baseline)."""

    def __init__(self, n_layers, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Create ``n_layers`` blocks in a :class:`nn.ModuleList`.

        Args:
            n_layers: Number of transformer blocks.
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads.
            rngs: PRNG source for parameter init.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [Llama3BlockSPX(d, ffn, n_heads, n_kv_heads, rngs=spx.Rngs(i)) for i in range(n_layers)]
        )

    def forward(self, x):
        """Run every block sequentially with shared RoPE tables.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        _b, t, _d = x.shape
        head_dim = self.blocks[0].head_dim
        cos, sin = _rope_freqs(t, head_dim)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x


class Llama3ModelSPXForiLoop(spx.Module):
    """SPX Llama 3B using ``spx.fori_loop`` with ``m.blocks[i]``."""

    def __init__(self, n_layers, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Create ``n_layers`` blocks in a :class:`nn.ModuleList`.

        Args:
            n_layers: Number of transformer blocks.
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads.
            rngs: PRNG source for parameter init.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [Llama3BlockSPX(d, ffn, n_heads, n_kv_heads, rngs=spx.Rngs(i)) for i in range(n_layers)]
        )

    def forward(self, x):
        """Run blocks via :func:`spx.fori_loop` with shared RoPE tables.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        _b, t, _d = x.shape
        head_dim = self.blocks[0].head_dim
        cos, sin = _rope_freqs(t, head_dim)

        def body(i, m, x):
            """One fori_loop step: apply block ``i``."""
            return m.blocks[i](x, cos, sin)

        return spx.fori_loop(0, len(self.blocks), body, self, x, mutable=())


def _dummy_batch(cfg: Llama3Config, step: int):
    """Return a deterministic ``(x, y)`` bf16 batch for ``step``."""
    key = jax.random.fold_in(jax.random.PRNGKey(0), step)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=jnp.bfloat16)
    y = jax.random.normal(ky, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=jnp.bfloat16)
    return x, y


def _median(xs: list[float]) -> float:
    """Return the median of ``xs``."""
    return sorted(xs)[len(xs) // 2]


def _loss_fn(out, y):
    """MSE loss between ``out`` and ``y`` computed in float32."""
    return ((out.astype(jnp.float32) - y.astype(jnp.float32)) ** 2).mean()


def bench_nnx_scan(cfg: Llama3Config, iters: int):
    """Time ``iters`` training steps of the NNX scan model."""
    rngs = nnx.Rngs(0)
    model = Llama3ModelNNXScan(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)

    def loss_fn(model, x, y):
        """MSE loss between ``model(x)`` and ``y``."""
        return _loss_fn(model(x), y)

    @nnx.jit
    def step(model, x, y):
        """One jitted training step: forward, loss, grad.

        Args:
            model: NNX model.
            x: Input tensor.
            y: Target tensor.

        Returns:
            ``(loss, grads)`` tuple.
        """
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        return loss, grads

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = step(model, x, y)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = step(model, x, y)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def bench_nnx_loop(cfg: Llama3Config, iters: int):
    """Time ``iters`` training steps of the NNX loop model."""
    rngs = nnx.Rngs(0)
    model = Llama3ModelNNXLoop(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)

    def loss_fn(model, x, y):
        """MSE loss between ``model(x)`` and ``y``."""
        return _loss_fn(model(x), y)

    @nnx.jit
    def step(model, x, y):
        """One jitted training step: forward, loss, grad.

        Args:
            model: NNX model.
            x: Input tensor.
            y: Target tensor.

        Returns:
            ``(loss, grads)`` tuple.
        """
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        return loss, grads

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = step(model, x, y)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = step(model, x, y)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


class Llama3ModelSPXScan(spx.Module):
    """SPX Llama 3B using ``ModuleList.scan`` inside ``forward``."""

    def __init__(self, n_layers, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Create ``n_layers`` blocks in a :class:`nn.ModuleList`.

        Args:
            n_layers: Number of transformer blocks.
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads.
            rngs: PRNG source for parameter init.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [Llama3BlockSPX(d, ffn, n_heads, n_kv_heads, rngs=spx.Rngs(i)) for i in range(n_layers)]
        )
        self.head_dim = d // n_heads

    def forward(self, x):
        """Run blocks via :meth:`ModuleList.scan` with shared RoPE tables.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        _b, t, _d = x.shape
        cos, sin = _rope_freqs(t, self.head_dim)
        return self.blocks.scan(lambda blk, x: blk(x, cos, sin), x)


def bench_spx_scan(cfg: Llama3Config, iters: int):
    """Time ``iters`` training steps of the SPX scan model (``spx.jit``)."""
    rngs = spx.Rngs(0)
    model = Llama3ModelSPXScan(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)

    @spx.jit
    def step(model, x, y):
        """One jitted training step: forward, loss, grad.

        Args:
            model: spectrax model.
            x: Input tensor.
            y: Target tensor.

        Returns:
            ``(loss, grads)`` tuple.
        """

        def loss_fn(m, x, y):
            """MSE loss between ``m(x)`` and ``y``."""
            return _loss_fn(m(x), y)

        return spx.value_and_grad(loss_fn)(model, x, y)

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = step(model, x, y)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = step(model, x, y)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def bench_spx_loop(cfg: Llama3Config, iters: int):
    """Time ``iters`` training steps of the SPX for-loop model (``spx.jit``)."""
    rngs = spx.Rngs(0)
    model = Llama3ModelSPXLoop(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)

    @spx.jit
    def step(model, x, y):
        """One jitted training step: forward, loss, grad.

        Args:
            model: spectrax model.
            x: Input tensor.
            y: Target tensor.

        Returns:
            ``(loss, grads)`` tuple.
        """

        def loss_fn(m, x, y):
            """MSE loss between ``m(x)`` and ``y``."""
            return _loss_fn(m(x), y)

        return spx.value_and_grad(loss_fn)(model, x, y)

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = step(model, x, y)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = step(model, x, y)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def bench_spx_fori_loop(cfg: Llama3Config, iters: int):
    """Time ``iters`` training steps using ``spx.fori_loop`` + ``m.blocks[i]``."""
    rngs = spx.Rngs(0)
    model = Llama3ModelSPXForiLoop(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)

    @spx.jit
    def step(model, x, y):
        """One jitted training step: forward, loss, grad.

        Args:
            model: spectrax model.
            x: Input tensor.
            y: Target tensor.

        Returns:
            ``(loss, grads)`` tuple.
        """

        def loss_fn(m, x, y):
            """MSE loss between ``m(x)`` and ``y``."""
            return _loss_fn(m(x), y)

        return spx.value_and_grad(loss_fn)(model, x, y)

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = step(model, x, y)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = step(model, x, y)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — build models, run benchmark, print table."""
    parser = argparse.ArgumentParser(description="Llama 3B scan benchmark")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--device", default="tpu")
    args = parser.parse_args(argv)

    cfg = Llama3Config(batch=args.batch, seq_len=args.seq_len)

    print("Llama 3B scan benchmark")
    print(f"  device       : {args.device} ({len(jax.devices())} devices)")
    print(f"  total params : ~{cfg.total_params / 1e9:.2f}B (transformer body only)")
    print(f"  n_layers     : {cfg.n_layers}  d/heads {cfg.d_model}/{cfg.n_heads} (kv {cfg.n_kv_heads})")
    print(f"  ffn (SwiGLU) : {cfg.ffn}")
    print(f"  batch / seq  : {cfg.batch} x {cfg.seq_len}")
    print()

    header = f"{'variant':<18} {'compile_ms':>11} {'step_ms':>10} {'loss':>10}"
    print(header)
    print("-" * len(header))

    variants = {
        "nnx_loop": bench_nnx_loop,
        "nnx_scan": bench_nnx_scan,
        "spx_loop": bench_spx_loop,
        "spx_scan": bench_spx_scan,
        "spx_fori_loop": bench_spx_fori_loop,
    }

    for name, runner in variants.items():
        try:
            compile_ms, step_ms, loss = runner(cfg, args.iters)
            print(f"{name:<18} {compile_ms:>11.1f} {step_ms:>10.2f} {loss:>10.4f}")
        except Exception as e:
            print(f"{name:<18} ERROR: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
