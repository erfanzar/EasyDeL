# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Llama 3 8B benchmark — Flax NNX, plain ``jax.jit``.

Reference baseline matching :mod:`benchmarks.llama3_8b`'s
``--modes jit`` row, written against ``flax.nnx`` instead of
spectrax. Same architecture (32 x Llama3Block, GQA + RoPE +
SwiGLU + RMSNorm, bf16) so step times are directly comparable.

Usage::

    python -m benchmarks.llama3_8b_nnx --iters 2 --device tpu
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


class Llama3BlockNNX(nnx.Module):
    """One Llama 3 transformer block in NNX (matches spectrax version).

    ``nnx.Linear`` defaults to ``use_bias=True``; we turn it off to
    match the Llama 3 convention.
    """

    def __init__(self, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize GQA attention + SwiGLU FFN sublayers."""
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d // n_heads
        kv_d = n_kv_heads * self.head_dim
        lin_kwargs = dict(use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        self.norm1 = nnx.RMSNorm(d, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs)
        self.q = nnx.Linear(d, d, rngs=rngs, **lin_kwargs)
        self.k = nnx.Linear(d, kv_d, rngs=rngs, **lin_kwargs)
        self.v = nnx.Linear(d, kv_d, rngs=rngs, **lin_kwargs)
        self.o = nnx.Linear(d, d, rngs=rngs, **lin_kwargs)
        self.norm2 = nnx.RMSNorm(d, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=rngs)
        self.gate = nnx.Linear(d, ffn, rngs=rngs, **lin_kwargs)
        self.up = nnx.Linear(d, ffn, rngs=rngs, **lin_kwargs)
        self.down = nnx.Linear(ffn, d, rngs=rngs, **lin_kwargs)

    def __call__(self, x, cos, sin):
        """Run pre-norm GQA+RoPE attention followed by pre-norm SwiGLU FFN."""
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


class Llama3ModelNNX(nnx.Module):
    """All 32 Llama 3 blocks chained together (no embedding / lm_head).

    ``nnx.List`` makes the per-block parameters a proper pytree — a
    plain Python list confuses nnx's split/merge.
    """

    def __init__(self, n_layers, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Build ``n_layers`` NNX Llama 3 blocks inside an :class:`nnx.List`."""
        self.head_dim = d // n_heads
        self.blocks = nnx.List(Llama3BlockNNX(d, ffn, n_heads, n_kv_heads, rngs=rngs) for _ in range(n_layers))

    def __call__(self, x):
        """Run every block with shared RoPE tables computed once per call."""
        _b, t, _d = x.shape
        cos, sin = _rope_freqs(t, self.head_dim)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x


@dataclass
class Llama3Config:
    """Llama 3 8B shape (transformer body only) for the NNX benchmark."""

    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn: int = 14336
    batch: int = 4
    seq_len: int = 128

    @property
    def total_params(self) -> int:
        """Approximate transformer-body parameter count."""
        attn = 2 * self.d_model * self.d_model + 2 * self.d_model * (self.d_model * self.n_kv_heads // self.n_heads)
        ffn = 3 * self.d_model * self.ffn
        return self.n_layers * (attn + ffn)


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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — build the NNX model, run ``--iters`` jitted steps."""
    parser = argparse.ArgumentParser(description="Llama 3 8B benchmark — Flax NNX, jit")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--device", default="tpu")
    args = parser.parse_args(argv)

    cfg = Llama3Config(batch=args.batch, seq_len=args.seq_len)

    print("nnx Llama 3 8B benchmark (jax.jit)")
    print(f"  device       : {args.device} ({len(jax.devices())} devices)")
    print(f"  total params : ~{cfg.total_params / 1e9:.2f}B")
    print(f"  n_layers     : {cfg.n_layers}  d/heads {cfg.d_model}/{cfg.n_heads} (kv {cfg.n_kv_heads})")
    print(f"  ffn (SwiGLU) : {cfg.ffn}")
    print(f"  batch / seq  : {cfg.batch} x {cfg.seq_len}")
    print()

    rngs = nnx.Rngs(0)
    model = Llama3ModelNNX(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)

    def loss_fn(model, x, y):
        """MSE loss between ``model(x)`` and ``y`` computed in float32."""
        out = model(x)
        return ((out.astype(jnp.float32) - y.astype(jnp.float32)) ** 2).mean()

    @nnx.jit
    def step(model, x, y):
        """Jitted training step: return ``(loss, grads)``."""
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        return loss, grads

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = step(model, x, y)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, args.iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = step(model, x, y)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    print(f"compile_ms : {compile_ms:>10.1f}")
    print(f"step_ms    : {_median(step_times):>10.2f}  (median of {args.iters})")
    print(f"loss       : {float(loss):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
