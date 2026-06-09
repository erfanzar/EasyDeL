# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three-way Llama transform benchmark: raw JAX vs Flax NNX vs spectrax.

This benchmark builds the same compact Llama-style transformer body in
three ways:

* raw JAX: explicit parameter pytrees + pure apply functions
* Flax NNX: live :class:`nnx.Module` objects
* spectrax: live :class:`spectrax.Module` objects

It measures the common transform surface that can be compared fairly
across all three stacks:

* ``eval_shape``
* ``jit`` forward
* ``grad`` training step
* ``value_and_grad`` training step
* ``jvp`` of the training loss
* ``vjp`` of the training loss
* ``vmap`` forward
* ``cond`` forward
* ``switch`` forward
* ``fori_loop`` forward
* ``while_loop`` forward
* ``scan`` training step
* ``remat`` training step

The model is the Llama transformer *body only* (RoPE + GQA attention +
SwiGLU FFN + RMSNorm), excluding token embeddings and LM head so that
the transform wiring remains visible instead of being dominated by the
vocabulary projection.

Usage::

    python -m benchmarks.llama_transforms_3way --preset tiny --quick
    python -m benchmarks.llama_transforms_3way --preset small --iters 20 --warmup 5
    python -m benchmarks.llama_transforms_3way --preset 3b --device gpu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_EARLY = argparse.ArgumentParser(add_help=False)
_EARLY.add_argument("--device", choices=["cpu", "gpu", "tpu"], default=None)
_EARLY_ARGS, _ = _EARLY.parse_known_args(sys.argv[1:])
if _EARLY_ARGS.device is not None:
    os.environ["JAX_PLATFORMS"] = _EARLY_ARGS.device
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

import spectrax as spx  # noqa: E402
from spectrax import nn  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

OP_ORDER = [
    "eval_shape",
    "jit_forward",
    "grad_train",
    "value_and_grad_train",
    "jvp_loss",
    "vjp_loss",
    "vmap_forward",
    "cond_forward",
    "switch_forward",
    "fori_loop_forward",
    "while_loop_forward",
    "scan_train",
    "remat_train",
]

PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {
        "n_layers": 4,
        "d_model": 256,
        "n_heads": 8,
        "n_kv_heads": 4,
        "ffn": 768,
        "batch": 2,
        "seq_len": 64,
        "dtype_name": "float32",
    },
    "small": {
        "n_layers": 8,
        "d_model": 512,
        "n_heads": 8,
        "n_kv_heads": 4,
        "ffn": 1536,
        "batch": 2,
        "seq_len": 128,
        "dtype_name": "float32",
    },
    "3b": {
        "n_layers": 24,
        "d_model": 3072,
        "n_heads": 24,
        "n_kv_heads": 8,
        "ffn": 8192,
        "batch": 2,
        "seq_len": 128,
        "dtype_name": "bfloat16",
    },
    "8b": {
        "n_layers": 32,
        "d_model": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "ffn": 14336,
        "batch": 4,
        "seq_len": 128,
        "dtype_name": "bfloat16",
    },
}


@dataclass(frozen=True)
class LlamaBenchConfig:
    """Llama-body benchmark shape."""

    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    ffn: int
    batch: int
    seq_len: int
    dtype_name: str = "float32"
    rope_theta: float = 500_000.0

    @property
    def dtype(self) -> jnp.dtype:
        """Return the JAX dtype object."""
        return jnp.dtype(self.dtype_name)

    @property
    def head_dim(self) -> int:
        """Per-head hidden size."""
        return self.d_model // self.n_heads


def _cfg_from_args(args: argparse.Namespace) -> LlamaBenchConfig:
    """Build the final config by overlaying CLI overrides on a preset."""
    base = dict(PRESETS[args.preset])
    for name in ("n_layers", "d_model", "n_heads", "n_kv_heads", "ffn", "batch", "seq_len"):
        value = getattr(args, name)
        if value is not None:
            base[name] = value
    if args.dtype is not None:
        base["dtype_name"] = args.dtype
    return LlamaBenchConfig(**base)


def _rope_freqs(seq_len: int, head_dim: int, theta: float, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    """Precompute RoPE tables shaped ``(seq_len, head_dim // 2)``."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv)
    return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def _apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply rotary embeddings to ``(b, t, h, d)`` tensors."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _rms_norm(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm in pure JAX."""
    inv = jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + eps)
    return (x * inv.astype(x.dtype)) * scale


def _linear(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Bias-free linear projection."""
    return jnp.einsum("...d,df->...f", x, weight)


def _loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Float32 MSE used by every training-style case."""
    return jnp.mean(jnp.square(pred.astype(jnp.float32) - target.astype(jnp.float32)))


def _cond_pred(x: jax.Array) -> jax.Array:
    """Dynamic predicate used by the control-flow benchmark cases."""
    return jnp.sum(x.astype(jnp.float32)) > 0


def _switch_index(x: jax.Array) -> jax.Array:
    """Dynamic branch index in ``{0, 1, 2}`` used by ``switch`` cases."""
    return jnp.int32(jnp.mod(jnp.abs(jnp.sum(x.astype(jnp.float32))), 3.0))


def _dummy_batch(cfg: LlamaBenchConfig, seed: int = 0) -> tuple[jax.Array, jax.Array]:
    """Deterministic input / target tensors for the benchmark."""
    key = jax.random.PRNGKey(seed)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=cfg.dtype)
    y = jax.random.normal(ky, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=cfg.dtype)
    return x, y


def _tree_param_count(tree: Any) -> int:
    """Count scalar values across a pytree."""
    total = 0
    for leaf in jax.tree.leaves(tree):
        shape = getattr(leaf, "shape", ())
        try:
            n = 1
            for dim in shape:
                n *= int(dim)
            total += n
        except Exception:
            continue
    return total


def _block_all(tree: Any) -> Any:
    """Block on every array leaf in a pytree."""
    for leaf in jax.tree.leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return tree


def _time_case(fn: Any, warmup: int, iters: int) -> dict[str, float]:
    """Measure first-call and steady-state wall times."""
    t0 = time.perf_counter_ns()
    out = fn()
    _block_all(out)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    for _ in range(warmup):
        _block_all(fn())

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = fn()
        _block_all(out)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    times.sort()
    n = len(times)

    def pct(p: float) -> float:
        """Return the ``p``-th percentile of the sorted times in milliseconds.

        Args:
            p: Percentile in ``[0, 1]``.

        Returns:
            Timing value at the requested percentile.
        """
        idx = min(n - 1, max(0, int(p * n)))
        return times[idx]

    return {
        "compile_ms": compile_ms,
        "median_ms": times[n // 2],
        "p05_ms": pct(0.05),
        "p95_ms": pct(0.95),
        "warmup": float(warmup),
        "iters": float(iters),
    }


def _make_markdown(
    cfg: LlamaBenchConfig,
    device: str,
    rows: list[dict[str, Any]],
) -> str:
    """Render a compact summary table with ratios against raw JAX."""
    by_op: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_op.setdefault(row["op"], {})[row["library"]] = row

    lines = [
        "# Llama Transform Benchmark",
        "",
        f"- device: {device}",
        f"- config: layers={cfg.n_layers} d_model={cfg.d_model} heads={cfg.n_heads} kv_heads={cfg.n_kv_heads} ffn={cfg.ffn} batch={cfg.batch} seq_len={cfg.seq_len} dtype={cfg.dtype_name}",
        "",
        "| op | jax compile | nnx compile | spx compile | nnx/jax compile | spx/jax compile | jax median | nnx median | spx median | nnx/jax | spx/jax |",
        "|----|------------:|------------:|------------:|----------------:|----------------:|-----------:|-----------:|-----------:|--------:|--------:|",
    ]
    for op in OP_ORDER:
        trio = by_op.get(op)
        if trio is None or {"jax", "nnx", "spx"} - set(trio):
            continue
        jax_row = trio["jax"]
        nnx_row = trio["nnx"]
        spx_row = trio["spx"]
        lines.append(
            f"| {op} | "
            f"{jax_row['compile_ms']:.3f} | {nnx_row['compile_ms']:.3f} | {spx_row['compile_ms']:.3f} | "
            f"{nnx_row['compile_vs_jax']:.3f} | {spx_row['compile_vs_jax']:.3f} | "
            f"{jax_row['median_ms']:.3f} | {nnx_row['median_ms']:.3f} | {spx_row['median_ms']:.3f} | "
            f"{nnx_row['median_vs_jax']:.3f} | {spx_row['median_vs_jax']:.3f} |"
        )
    return "\n".join(lines)


def _init_weight(key: jax.Array, in_dim: int, out_dim: int, dtype: jnp.dtype) -> jax.Array:
    """Simple fan-in scaled init shared by the raw JAX baseline."""
    scale = 1.0 / math.sqrt(max(in_dim, 1))
    w = jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * scale
    return w.astype(dtype)


def _init_jax_block(key: jax.Array, cfg: LlamaBenchConfig) -> dict[str, jax.Array]:
    """Initialize one raw-JAX Llama block."""
    kv_d = cfg.n_kv_heads * cfg.head_dim
    kq, kk, kv, ko, kg, ku, kd = jax.random.split(key, 7)
    return {
        "norm1": jnp.ones((cfg.d_model,), dtype=cfg.dtype),
        "q": _init_weight(kq, cfg.d_model, cfg.d_model, cfg.dtype),
        "k": _init_weight(kk, cfg.d_model, kv_d, cfg.dtype),
        "v": _init_weight(kv, cfg.d_model, kv_d, cfg.dtype),
        "o": _init_weight(ko, cfg.d_model, cfg.d_model, cfg.dtype),
        "norm2": jnp.ones((cfg.d_model,), dtype=cfg.dtype),
        "gate": _init_weight(kg, cfg.d_model, cfg.ffn, cfg.dtype),
        "up": _init_weight(ku, cfg.d_model, cfg.ffn, cfg.dtype),
        "down": _init_weight(kd, cfg.ffn, cfg.d_model, cfg.dtype),
    }


def _jax_block_apply(
    params: dict[str, jax.Array],
    x: jax.Array,
    cfg: LlamaBenchConfig,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply one raw-JAX Llama block."""
    b, t, _ = x.shape
    h = _rms_norm(x, params["norm1"])
    q = _linear(h, params["q"]).reshape(b, t, cfg.n_heads, cfg.head_dim)
    k = _linear(h, params["k"]).reshape(b, t, cfg.n_kv_heads, cfg.head_dim)
    v = _linear(h, params["v"]).reshape(b, t, cfg.n_kv_heads, cfg.head_dim)
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)
    h = jax.nn.dot_product_attention(
        q,
        k,
        v,
        scale=1.0 / math.sqrt(cfg.head_dim),
        is_causal=True,
    ).reshape(b, t, cfg.d_model)
    x = x + _linear(h, params["o"])
    h = _rms_norm(x, params["norm2"])
    h = _linear(jax.nn.silu(_linear(h, params["gate"])) * _linear(h, params["up"]), params["down"])
    return x + h


def _init_jax_loop_model(cfg: LlamaBenchConfig, seed: int = 0) -> dict[str, list[dict[str, jax.Array]]]:
    """Initialize the raw-JAX loop model."""
    keys = jax.random.split(jax.random.PRNGKey(seed), cfg.n_layers)
    return {"blocks": [_init_jax_block(key, cfg) for key in keys]}


def _init_jax_scan_model(cfg: LlamaBenchConfig, seed: int = 0) -> Any:
    """Initialize the raw-JAX scan model with leading layer axes stacked."""
    blocks = _init_jax_loop_model(cfg, seed=seed)["blocks"]
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *blocks)


def _jax_loop_forward(params: dict[str, list[dict[str, jax.Array]]], x: jax.Array, cfg: LlamaBenchConfig) -> jax.Array:
    """Run the raw-JAX Llama body with a Python loop over layers."""
    cos, sin = _rope_freqs(x.shape[1], cfg.head_dim, cfg.rope_theta, x.dtype)
    for block in params["blocks"]:
        x = _jax_block_apply(block, x, cfg, cos, sin)
    return x


def _jax_scan_forward(params: Any, x: jax.Array, cfg: LlamaBenchConfig) -> jax.Array:
    """Run the raw-JAX Llama body with ``jax.lax.scan`` over stacked layers."""
    cos, sin = _rope_freqs(x.shape[1], cfg.head_dim, cfg.rope_theta, x.dtype)

    def body(carry: jax.Array, layer_params: Any) -> tuple[jax.Array, None]:
        """Single scan step: apply one layer block.

        Args:
            carry: Input activation tensor.
            layer_params: Parameter dict for this layer.

        Returns:
            ``(output, None)`` tuple for scan carry.
        """
        return _jax_block_apply(layer_params, carry, cfg, cos, sin), None

    return jax.lax.scan(body, x, params)[0]


def _jax_remat_forward(params: dict[str, list[dict[str, jax.Array]]], x: jax.Array, cfg: LlamaBenchConfig) -> jax.Array:
    """Run the raw-JAX Llama body with checkpointed block calls."""
    cos, sin = _rope_freqs(x.shape[1], cfg.head_dim, cfg.rope_theta, x.dtype)
    remat_block = jax.checkpoint(lambda block, x: _jax_block_apply(block, x, cfg, cos, sin))
    for block in params["blocks"]:
        x = remat_block(block, x)
    return x


def _jax_cond_forward(
    true_block: dict[str, jax.Array],
    false_block: dict[str, jax.Array],
    x: jax.Array,
    cfg: LlamaBenchConfig,
) -> jax.Array:
    """Run one Llama block through ``jax.lax.cond``."""
    cos, sin = _rope_freqs(x.shape[1], cfg.head_dim, cfg.rope_theta, x.dtype)
    return jax.lax.cond(
        _cond_pred(x),
        lambda carry: _jax_block_apply(true_block, carry, cfg, cos, sin),
        lambda carry: _jax_block_apply(false_block, carry, cfg, cos, sin),
        x,
    )


def _jax_switch_forward(
    blocks: tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]],
    x: jax.Array,
    cfg: LlamaBenchConfig,
) -> jax.Array:
    """Run one Llama block through ``jax.lax.switch``."""
    cos, sin = _rope_freqs(x.shape[1], cfg.head_dim, cfg.rope_theta, x.dtype)
    branches = [lambda carry, blk=block: _jax_block_apply(blk, carry, cfg, cos, sin) for block in blocks]
    return jax.lax.switch(_switch_index(x), branches, x)


def _jax_fori_loop_forward(block: dict[str, jax.Array], x: jax.Array, cfg: LlamaBenchConfig) -> jax.Array:
    """Repeat a single Llama block with ``jax.lax.fori_loop``."""
    cos, sin = _rope_freqs(x.shape[1], cfg.head_dim, cfg.rope_theta, x.dtype)

    def body(_i: int, carry: jax.Array) -> jax.Array:
        """One fori_loop step: apply the block.

        Args:
            _i: Loop index (unused).
            carry: Input activation tensor.

        Returns:
            Output activation tensor.
        """
        return _jax_block_apply(block, carry, cfg, cos, sin)

    return jax.lax.fori_loop(0, cfg.n_layers, body, x)


def _jax_while_loop_forward(block: dict[str, jax.Array], x: jax.Array, cfg: LlamaBenchConfig) -> jax.Array:
    """Repeat a single Llama block with ``jax.lax.while_loop``."""
    cos, sin = _rope_freqs(x.shape[1], cfg.head_dim, cfg.rope_theta, x.dtype)

    def cond(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
        """Check whether the loop counter is still below ``cfg.n_layers``.

        Args:
            carry: ``(i, x)`` tuple.

        Returns:
            Boolean scalar.
        """
        i, _x = carry
        return i < cfg.n_layers

    def body(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        """One while_loop step: increment counter and apply the block.

        Args:
            carry: ``(i, x)`` tuple.

        Returns:
            ``(i + 1, block(x))`` tuple.
        """
        i, x_in = carry
        return i + 1, _jax_block_apply(block, x_in, cfg, cos, sin)

    return jax.lax.while_loop(cond, body, (jnp.int32(0), x))[1]


class NnxLlamaBlock(nnx.Module):
    """One Llama block in Flax NNX."""

    def __init__(self, cfg: LlamaBenchConfig, *, rngs: nnx.Rngs):
        """Initialize GQA attention + SwiGLU FFN sublayers.

        Args:
            cfg: Benchmark configuration.
            rngs: PRNG source for parameter init.
        """
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        kv_d = cfg.n_kv_heads * cfg.head_dim
        lin = dict(use_bias=False, dtype=cfg.dtype, param_dtype=cfg.dtype)
        self.norm1 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=cfg.dtype, rngs=rngs)
        self.q = nnx.Linear(cfg.d_model, cfg.d_model, rngs=rngs, **lin)
        self.k = nnx.Linear(cfg.d_model, kv_d, rngs=rngs, **lin)
        self.v = nnx.Linear(cfg.d_model, kv_d, rngs=rngs, **lin)
        self.o = nnx.Linear(cfg.d_model, cfg.d_model, rngs=rngs, **lin)
        self.norm2 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=cfg.dtype, rngs=rngs)
        self.gate = nnx.Linear(cfg.d_model, cfg.ffn, rngs=rngs, **lin)
        self.up = nnx.Linear(cfg.d_model, cfg.ffn, rngs=rngs, **lin)
        self.down = nnx.Linear(cfg.ffn, cfg.d_model, rngs=rngs, **lin)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
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
        h = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(self.head_dim),
            is_causal=True,
        ).reshape(b, t, self.d_model)
        x = x + self.o(h)
        h = self.norm2(x)
        h = self.down(jax.nn.silu(self.gate(h)) * self.up(h))
        return x + h


class NnxLlamaLoopModel(nnx.Module):
    """NNX Llama body using a Python loop across blocks."""

    def __init__(self, cfg: LlamaBenchConfig, *, seed: int = 0):
        """Create ``n_layers`` blocks in an :class:`nnx.List`.

        Args:
            cfg: Benchmark configuration.
            seed: Base PRNG seed; incremented per block.
        """
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_theta
        self.blocks = nnx.List([NnxLlamaBlock(cfg, rngs=nnx.Rngs(seed + i)) for i in range(cfg.n_layers)])

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run every block sequentially with shared RoPE tables.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        cos, sin = _rope_freqs(x.shape[1], self.head_dim, self.rope_theta, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return x


class NnxLlamaScanModel(nnx.Module):
    """NNX Llama body using ``nnx.scan`` over stacked blocks."""

    def __init__(self, cfg: LlamaBenchConfig, *, seed: int = 0):
        """Stack ``n_layers`` blocks via vmap and scan them over the input.

        Args:
            cfg: Benchmark configuration.
            seed: Base PRNG seed.
        """
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_theta

        @nnx.split_rngs(splits=cfg.n_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def build(rngs: nnx.Rngs) -> NnxLlamaBlock:
            """Create one NnxLlamaBlock under split RNGs."""
            return NnxLlamaBlock(cfg, rngs=rngs)

        self.blocks = build(nnx.Rngs(seed))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the scanned block stack on ``x``.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        cos, sin = _rope_freqs(x.shape[1], self.head_dim, self.rope_theta, x.dtype)

        @nnx.scan(in_axes=(nnx.Carry, None, None, 0), out_axes=nnx.Carry)
        def body(carry: jax.Array, cos: jax.Array, sin: jax.Array, block: NnxLlamaBlock) -> jax.Array:
            """Single scan step: apply one block."""
            return block(carry, cos, sin)

        return body(x, cos, sin, self.blocks)


_NNX_REMAT_BLOCK = nnx.remat(lambda block, x, cos, sin: block(x, cos, sin), graph_updates=False)


class NnxLlamaRematModel(nnx.Module):
    """NNX Llama body with rematerialized block calls."""

    def __init__(self, cfg: LlamaBenchConfig, *, seed: int = 0):
        """Create ``n_layers`` blocks in an :class:`nnx.List`.

        Args:
            cfg: Benchmark configuration.
            seed: Base PRNG seed; incremented per block.
        """
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_theta
        self.blocks = nnx.List([NnxLlamaBlock(cfg, rngs=nnx.Rngs(seed + i)) for i in range(cfg.n_layers)])

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run every block with rematerialized forward calls.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        cos, sin = _rope_freqs(x.shape[1], self.head_dim, self.rope_theta, x.dtype)
        for block in self.blocks:
            x = _NNX_REMAT_BLOCK(block, x, cos, sin)
        return x


class SpxLlamaBlock(spx.Module):
    """One Llama block in spectrax."""

    def __init__(self, cfg: LlamaBenchConfig, *, rngs: spx.Rngs):
        """Initialize GQA attention + SwiGLU FFN sublayers.

        Args:
            cfg: Benchmark configuration.
            rngs: PRNG source for parameter init.
        """
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        kv_d = cfg.n_kv_heads * cfg.head_dim
        lin = dict(use_bias=False, dtype=cfg.dtype, rngs=rngs)
        self.norm1 = nn.RMSNorm(cfg.d_model, dtype=cfg.dtype)
        self.q = nn.Linear(cfg.d_model, cfg.d_model, **lin)
        self.k = nn.Linear(cfg.d_model, kv_d, **lin)
        self.v = nn.Linear(cfg.d_model, kv_d, **lin)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, **lin)
        self.norm2 = nn.RMSNorm(cfg.d_model, dtype=cfg.dtype)
        self.gate = nn.Linear(cfg.d_model, cfg.ffn, **lin)
        self.up = nn.Linear(cfg.d_model, cfg.ffn, **lin)
        self.down = nn.Linear(cfg.ffn, cfg.d_model, **lin)

    def forward(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
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
        h = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(self.head_dim),
            is_causal=True,
        ).reshape(b, t, self.d_model)
        x = x + self.o(h)
        h = self.norm2(x)
        h = self.down(jax.nn.silu(self.gate(h)) * self.up(h))
        return x + h


class SpxLlamaLoopModel(spx.Module):
    """spectrax Llama body using a Python loop."""

    def __init__(self, cfg: LlamaBenchConfig, *, seed: int = 0):
        """Create ``n_layers`` blocks in a :class:`nn.ModuleList`.

        Args:
            cfg: Benchmark configuration.
            seed: Base PRNG seed; incremented per block.
        """
        super().__init__()
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_theta
        self.blocks = nn.ModuleList([SpxLlamaBlock(cfg, rngs=spx.Rngs(seed + i)) for i in range(cfg.n_layers)])

    def forward(self, x: jax.Array) -> jax.Array:
        """Run every block sequentially with shared RoPE tables.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        cos, sin = _rope_freqs(x.shape[1], self.head_dim, self.rope_theta, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return x


class SpxLlamaScanModel(spx.Module):
    """spectrax Llama body using pre-stacked ``ModuleList.scan``."""

    def __init__(self, cfg: LlamaBenchConfig, *, seed: int = 0):
        """Create ``n_layers`` blocks in a stacked :class:`nn.ModuleList`.

        Args:
            cfg: Benchmark configuration.
            seed: Base PRNG seed; incremented per block.
        """
        super().__init__()
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_theta
        self.blocks = nn.ModuleList([SpxLlamaBlock(cfg, rngs=spx.Rngs(seed + i)) for i in range(cfg.n_layers)]).stack()

    def forward(self, x: jax.Array) -> jax.Array:
        """Run blocks via :meth:`ModuleList.scan` with shared RoPE tables.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        cos, sin = _rope_freqs(x.shape[1], self.head_dim, self.rope_theta, x.dtype)
        return self.blocks.scan(lambda block, carry: block(carry, cos, sin), x)


_SPX_REMAT_BLOCK = spx.remat(lambda block, x, cos, sin: block(x, cos, sin))


class SpxLlamaRematModel(spx.Module):
    """spectrax Llama body with rematerialized block calls."""

    def __init__(self, cfg: LlamaBenchConfig, *, seed: int = 0):
        """Create ``n_layers`` blocks in a :class:`nn.ModuleList`.

        Args:
            cfg: Benchmark configuration.
            seed: Base PRNG seed; incremented per block.
        """
        super().__init__()
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_theta
        self.blocks = nn.ModuleList([SpxLlamaBlock(cfg, rngs=spx.Rngs(seed + i)) for i in range(cfg.n_layers)])

    def forward(self, x: jax.Array) -> jax.Array:
        """Run every block with rematerialized forward calls.

        Args:
            x: Input tensor ``(b, t, d)``.

        Returns:
            Output tensor ``(b, t, d)``.
        """
        cos, sin = _rope_freqs(x.shape[1], self.head_dim, self.rope_theta, x.dtype)
        for block in self.blocks:
            x = _SPX_REMAT_BLOCK(block, x, cos, sin)
        return x


def _build_jax_cases(cfg: LlamaBenchConfig) -> tuple[dict[str, Any], int]:
    """Build raw-JAX benchmark callables."""
    x, y = _dummy_batch(cfg, seed=0)
    x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    loop_model = _init_jax_loop_model(cfg, seed=0)
    scan_model = _init_jax_scan_model(cfg, seed=0)
    block0 = loop_model["blocks"][0]
    block1 = loop_model["blocks"][1]
    block2 = loop_model["blocks"][2]
    tangent_model = jax.tree.map(jnp.zeros_like, loop_model)
    tangent_x = jnp.ones_like(x)

    def loop_loss(params: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the raw-JAX loop model.

        Args:
            params: Raw-JAX parameter pytree.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(_jax_loop_forward(params, xb, cfg), yb)

    def scan_loss(params: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the raw-JAX scan model.

        Args:
            params: Raw-JAX parameter pytree.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(_jax_scan_forward(params, xb, cfg), yb)

    def remat_loss(params: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the raw-JAX remat model.

        Args:
            params: Raw-JAX parameter pytree.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(_jax_remat_forward(params, xb, cfg), yb)

    jit_forward = jax.jit(lambda params, xb: _jax_loop_forward(params, xb, cfg))
    grad_train = jax.jit(jax.grad(loop_loss))
    value_and_grad_train = jax.jit(jax.value_and_grad(loop_loss))
    jvp_loss = jax.jit(
        lambda params, xb, params_t, xb_t: jax.jvp(
            lambda p, x_: loop_loss(p, x_, y),
            (params, xb),
            (params_t, xb_t),
        )
    )

    def _vjp_only(params: Any, xb: jax.Array) -> Any:
        """VJP of ``loop_loss`` w.r.t. ``params``.

        Args:
            params: Raw-JAX parameter pytree.
            xb: Input batch.

        Returns:
            Gradient pytree for ``params``.
        """
        out, pullback = jax.vjp(lambda p, x_: loop_loss(p, x_, y), params, xb)
        return pullback(jnp.array(1.0, dtype=out.dtype))[0]

    vjp_loss = jax.jit(_vjp_only)
    vmap_forward = jax.jit(lambda params, xb: jax.vmap(lambda x1: _jax_loop_forward(params, x1[None, ...], cfg)[0])(xb))
    cond_forward = jax.jit(lambda true_blk, false_blk, xb: _jax_cond_forward(true_blk, false_blk, xb, cfg))
    switch_forward = jax.jit(lambda b0, b1, b2, xb: _jax_switch_forward((b0, b1, b2), xb, cfg))
    fori_loop_forward = jax.jit(lambda block, xb: _jax_fori_loop_forward(block, xb, cfg))
    while_loop_forward = jax.jit(lambda block, xb: _jax_while_loop_forward(block, xb, cfg))
    scan_train = jax.jit(jax.value_and_grad(scan_loss))
    remat_train = jax.jit(jax.value_and_grad(remat_loss))

    return {
        "eval_shape": lambda: jax.eval_shape(lambda params, xb: _jax_loop_forward(params, xb, cfg), loop_model, x_spec),
        "jit_forward": lambda: jit_forward(loop_model, x),
        "grad_train": lambda: grad_train(loop_model, x, y),
        "value_and_grad_train": lambda: value_and_grad_train(loop_model, x, y),
        "jvp_loss": lambda: jvp_loss(loop_model, x, tangent_model, tangent_x),
        "vjp_loss": lambda: vjp_loss(loop_model, x),
        "vmap_forward": lambda: vmap_forward(loop_model, x),
        "cond_forward": lambda: cond_forward(block0, block1, x),
        "switch_forward": lambda: switch_forward(block0, block1, block2, x),
        "fori_loop_forward": lambda: fori_loop_forward(block0, x),
        "while_loop_forward": lambda: while_loop_forward(block0, x),
        "scan_train": lambda: scan_train(scan_model, x, y),
        "remat_train": lambda: remat_train(loop_model, x, y),
    }, _tree_param_count(loop_model)


def _build_nnx_cases(cfg: LlamaBenchConfig) -> tuple[dict[str, Any], int]:
    """Build Flax NNX benchmark callables."""
    x, y = _dummy_batch(cfg, seed=0)
    x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    loop_model = NnxLlamaLoopModel(cfg, seed=0)
    scan_model = NnxLlamaScanModel(cfg, seed=0)
    remat_model = NnxLlamaRematModel(cfg, seed=0)
    block0 = loop_model.blocks[0]
    block1 = loop_model.blocks[1]
    block2 = loop_model.blocks[2]
    tangent_model = jax.tree.map(jnp.zeros_like, loop_model)
    tangent_x = jnp.ones_like(x)

    def loop_loss(model: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the NNX loop model.

        Args:
            model: NNX model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(model(xb), yb)

    def scan_loss(model: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the NNX scan model.

        Args:
            model: NNX model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(model(xb), yb)

    def remat_loss(model: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the NNX remat model.

        Args:
            model: NNX model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(model(xb), yb)

    @nnx.jit
    def jit_forward(model: Any, xb: jax.Array) -> jax.Array:
        """Jitted NNX forward pass.

        Args:
            model: NNX model.
            xb: Input batch.

        Returns:
            Model output tensor.
        """
        return model(xb)

    @nnx.jit
    def grad_train(model: Any, xb: jax.Array, yb: jax.Array) -> Any:
        """Jitted NNX grad training step.

        Args:
            model: NNX model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Gradient state.
        """
        return nnx.grad(loop_loss)(model, xb, yb)

    @nnx.jit
    def value_and_grad_train(model: Any, xb: jax.Array, yb: jax.Array) -> Any:
        """Jitted NNX value_and_grad training step.

        Args:
            model: NNX model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            ``(loss, grads)`` tuple.
        """
        return nnx.value_and_grad(loop_loss)(model, xb, yb)

    @nnx.jit
    def jvp_loss(model: Any, xb: jax.Array, model_t: Any, xb_t: jax.Array) -> Any:
        """Jitted NNX JVP of the training loss.

        Args:
            model: NNX model.
            xb: Input batch.
            model_t: Tangent for the model.
            xb_t: Tangent for the input.

        Returns:
            JVP result.
        """
        return nnx.jvp(lambda m, x_: loop_loss(m, x_, y), (model, xb), (model_t, xb_t), graph_updates=False)

    @nnx.jit
    def vjp_loss(model: Any, xb: jax.Array) -> Any:
        """Jitted NNX VJP of the training loss.

        Args:
            model: NNX model.
            xb: Input batch.

        Returns:
            VJP result.
        """
        out, pullback = nnx.vjp(lambda m, x_: loop_loss(m, x_, y), model, xb, graph_updates=False)
        return pullback(jnp.array(1.0, dtype=out.dtype))[0]

    @nnx.jit
    def vmap_forward(model: Any, xb: jax.Array) -> jax.Array:
        """Jitted NNX batched forward via vmap.

        Args:
            model: NNX model.
            xb: Batched input.

        Returns:
            Batched output.
        """
        return nnx.vmap(lambda m, x1: m(x1[None, ...])[0], in_axes=(None, 0))(model, xb)

    @nnx.jit
    def cond_forward(true_block: Any, false_block: Any, xb: jax.Array) -> jax.Array:
        """Jitted NNX cond forward.

        Args:
            true_block: Branch taken when predicate is true.
            false_block: Branch taken when predicate is false.
            xb: Input batch.

        Returns:
            Output tensor from the selected branch.
        """
        cos, sin = _rope_freqs(xb.shape[1], cfg.head_dim, cfg.rope_theta, xb.dtype)
        return nnx.cond(
            _cond_pred(xb),
            lambda tb, fb, carry: tb(carry, cos, sin),
            lambda tb, fb, carry: fb(carry, cos, sin),
            true_block,
            false_block,
            xb,
        )

    @nnx.jit
    def switch_forward(first: Any, second: Any, third: Any, xb: jax.Array) -> jax.Array:
        """Jitted NNX switch forward across three branches.

        Args:
            first: First branch block.
            second: Second branch block.
            third: Third branch block.
            xb: Input batch.

        Returns:
            Output tensor from the selected branch.
        """
        cos, sin = _rope_freqs(xb.shape[1], cfg.head_dim, cfg.rope_theta, xb.dtype)
        branches = [
            lambda a, b, c, carry: a(carry, cos, sin),
            lambda a, b, c, carry: b(carry, cos, sin),
            lambda a, b, c, carry: c(carry, cos, sin),
        ]
        return nnx.switch(_switch_index(xb), branches, first, second, third, xb)

    @nnx.jit
    def fori_loop_forward(block: Any, xb: jax.Array) -> jax.Array:
        """Jitted NNX fori_loop forward.

        Args:
            block: Block module to repeat.
            xb: Input batch.

        Returns:
            Output tensor after ``cfg.n_layers`` iterations.
        """
        cos, sin = _rope_freqs(xb.shape[1], cfg.head_dim, cfg.rope_theta, xb.dtype)

        def body(_i: int, carry: jax.Array) -> jax.Array:
            """One fori_loop step: apply ``block`` with RoPE."""
            return block(carry, cos, sin)

        return nnx.fori_loop(0, cfg.n_layers, body, xb)

    @nnx.jit
    def while_loop_forward(block: Any, xb: jax.Array) -> jax.Array:
        """Jitted NNX while_loop forward.

        Args:
            block: Block module to repeat.
            xb: Input batch.

        Returns:
            Output tensor after ``cfg.n_layers`` iterations.
        """
        cos, sin = _rope_freqs(xb.shape[1], cfg.head_dim, cfg.rope_theta, xb.dtype)

        def cond(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            """Check whether the loop counter is still below ``cfg.n_layers``."""
            i, _x = carry
            return i < cfg.n_layers

        def body(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            """One while_loop step: increment counter and apply ``block``."""
            i, x_in = carry
            return i + 1, block(x_in, cos, sin)

        return nnx.while_loop(cond, body, (jnp.int32(0), xb))[1]

    @nnx.jit
    def scan_train(model: Any, xb: jax.Array, yb: jax.Array) -> Any:
        """Jitted NNX scan training step.

        Args:
            model: NNX model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            ``(loss, grads)`` tuple.
        """
        return nnx.value_and_grad(scan_loss)(model, xb, yb)

    @nnx.jit
    def remat_train(model: Any, xb: jax.Array, yb: jax.Array) -> Any:
        """Jitted NNX remat training step.

        Args:
            model: NNX model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            ``(loss, grads)`` tuple.
        """
        return nnx.value_and_grad(remat_loss)(model, xb, yb)

    return {
        "eval_shape": lambda: nnx.eval_shape(lambda model, xb: model(xb), loop_model, x_spec, graph_updates=False),
        "jit_forward": lambda: jit_forward(loop_model, x),
        "grad_train": lambda: grad_train(loop_model, x, y),
        "value_and_grad_train": lambda: value_and_grad_train(loop_model, x, y),
        "jvp_loss": lambda: jvp_loss(loop_model, x, tangent_model, tangent_x),
        "vjp_loss": lambda: vjp_loss(loop_model, x),
        "vmap_forward": lambda: vmap_forward(loop_model, x),
        "cond_forward": lambda: cond_forward(block0, block1, x),
        "switch_forward": lambda: switch_forward(block0, block1, block2, x),
        "fori_loop_forward": lambda: fori_loop_forward(block0, x),
        "while_loop_forward": lambda: while_loop_forward(block0, x),
        "scan_train": lambda: scan_train(scan_model, x, y),
        "remat_train": lambda: remat_train(remat_model, x, y),
    }, _tree_param_count(loop_model)


def _build_spx_cases(cfg: LlamaBenchConfig) -> tuple[dict[str, Any], int]:
    """Build spectrax benchmark callables."""
    x, y = _dummy_batch(cfg, seed=0)
    x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    loop_model = SpxLlamaLoopModel(cfg, seed=0)
    scan_model = SpxLlamaScanModel(cfg, seed=0)
    remat_model = SpxLlamaRematModel(cfg, seed=0)
    block0 = loop_model.blocks[0]
    block1 = loop_model.blocks[1]
    block2 = loop_model.blocks[2]
    tangent_model = jax.tree.map(jnp.zeros_like, loop_model)
    tangent_x = jnp.ones_like(x)

    def loop_loss(model: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the spectrax loop model.

        Args:
            model: spectrax model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(model(xb), yb)

    def scan_loss(model: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the spectrax scan model.

        Args:
            model: spectrax model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(model(xb), yb)

    def remat_loss(model: Any, xb: jax.Array, yb: jax.Array) -> jax.Array:
        """MSE loss for the spectrax remat model.

        Args:
            model: spectrax model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Scalar MSE.
        """
        return _loss(model(xb), yb)

    @spx.jit
    def jit_forward(model: Any, xb: jax.Array) -> jax.Array:
        """Jitted spectrax forward pass.

        Args:
            model: spectrax model.
            xb: Input batch.

        Returns:
            Model output tensor.
        """
        return model(xb)

    @spx.jit
    def grad_train(model: Any, xb: jax.Array, yb: jax.Array) -> Any:
        """Jitted spectrax grad training step.

        Args:
            model: spectrax model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            Gradient state.
        """
        return spx.grad(loop_loss)(model, xb, yb)

    @spx.jit
    def value_and_grad_train(model: Any, xb: jax.Array, yb: jax.Array) -> Any:
        """Jitted spectrax value_and_grad training step.

        Args:
            model: spectrax model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            ``(loss, grads)`` tuple.
        """
        return spx.value_and_grad(loop_loss)(model, xb, yb)

    @spx.jit
    def jvp_loss(model: Any, xb: jax.Array, model_t: Any, xb_t: jax.Array) -> Any:
        """Jitted spectrax JVP of the training loss.

        Args:
            model: spectrax model.
            xb: Input batch.
            model_t: Tangent for the model.
            xb_t: Tangent for the input.

        Returns:
            JVP result.
        """
        return spx.jvp(lambda m, x_: loop_loss(m, x_, y), (model, xb), (model_t, xb_t))

    @spx.jit
    def vjp_loss(model: Any, xb: jax.Array) -> Any:
        """Jitted spectrax VJP of the training loss.

        Args:
            model: spectrax model.
            xb: Input batch.

        Returns:
            VJP result.
        """
        out, pullback = spx.vjp(lambda m, x_: loop_loss(m, x_, y), model, xb)
        return pullback(jnp.array(1.0, dtype=out.dtype))[0]

    @spx.jit
    def vmap_forward(model: Any, xb: jax.Array) -> jax.Array:
        """Jitted spectrax batched forward via vmap.

        Args:
            model: spectrax model.
            xb: Batched input.

        Returns:
            Batched output.
        """
        return spx.vmap(lambda m, x1: m(x1[None, ...])[0], in_axes=(None, 0))(model, xb)

    @spx.jit
    def cond_forward(true_block: Any, false_block: Any, xb: jax.Array) -> jax.Array:
        """Jitted spectrax cond forward.

        Args:
            true_block: Branch taken when predicate is true.
            false_block: Branch taken when predicate is false.
            xb: Input batch.

        Returns:
            Output tensor from the selected branch.
        """
        cos, sin = _rope_freqs(xb.shape[1], cfg.head_dim, cfg.rope_theta, xb.dtype)
        return spx.cond(
            _cond_pred(xb),
            lambda tb, fb, carry: tb(carry, cos, sin),
            lambda tb, fb, carry: fb(carry, cos, sin),
            true_block,
            false_block,
            xb,
        )

    @spx.jit
    def switch_forward(first: Any, second: Any, third: Any, xb: jax.Array) -> jax.Array:
        """Jitted spectrax switch forward across three branches.

        Args:
            first: First branch block.
            second: Second branch block.
            third: Third branch block.
            xb: Input batch.

        Returns:
            Output tensor from the selected branch.
        """
        cos, sin = _rope_freqs(xb.shape[1], cfg.head_dim, cfg.rope_theta, xb.dtype)
        branches = [
            lambda a, b, c, carry: a(carry, cos, sin),
            lambda a, b, c, carry: b(carry, cos, sin),
            lambda a, b, c, carry: c(carry, cos, sin),
        ]
        return spx.switch(_switch_index(xb), branches, first, second, third, xb)

    @spx.jit
    def fori_loop_forward(block: Any, xb: jax.Array) -> jax.Array:
        """Jitted spectrax fori_loop forward.

        Args:
            block: Block module to repeat.
            xb: Input batch.

        Returns:
            Output tensor after ``cfg.n_layers`` iterations.
        """
        cos, sin = _rope_freqs(xb.shape[1], cfg.head_dim, cfg.rope_theta, xb.dtype)

        def body(_i: int, blk: Any, carry: jax.Array) -> jax.Array:
            """One fori_loop step: apply ``blk`` with RoPE."""
            return blk(carry, cos, sin)

        return spx.fori_loop(0, cfg.n_layers, body, block, xb)

    @spx.jit
    def while_loop_forward(block: Any, xb: jax.Array) -> jax.Array:
        """Jitted spectrax while_loop forward.

        Args:
            block: Block module to repeat.
            xb: Input batch.

        Returns:
            Output tensor after ``cfg.n_layers`` iterations.
        """
        cos, sin = _rope_freqs(xb.shape[1], cfg.head_dim, cfg.rope_theta, xb.dtype)

        def cond(blk: Any, carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            """Check whether the loop counter is still below ``cfg.n_layers``."""
            i, _x = carry
            return i < cfg.n_layers

        def body(blk: Any, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            """One while_loop step: increment counter and apply ``blk``."""
            i, x_in = carry
            return i + 1, blk(x_in, cos, sin)

        return spx.while_loop(cond, body, block, (jnp.int32(0), xb))[1]

    @spx.jit
    def scan_train(model: Any, xb: jax.Array, yb: jax.Array) -> Any:
        """Jitted spectrax scan training step.

        Args:
            model: spectrax model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            ``(loss, grads)`` tuple.
        """
        return spx.value_and_grad(scan_loss)(model, xb, yb)

    @spx.jit
    def remat_train(model: Any, xb: jax.Array, yb: jax.Array) -> Any:
        """Jitted spectrax remat training step.

        Args:
            model: spectrax model.
            xb: Input batch.
            yb: Target batch.

        Returns:
            ``(loss, grads)`` tuple.
        """
        return spx.value_and_grad(remat_loss)(model, xb, yb)

    return {
        "eval_shape": lambda: spx.eval_shape(lambda model, xb: model(xb), loop_model, x_spec),
        "jit_forward": lambda: jit_forward(loop_model, x),
        "grad_train": lambda: grad_train(loop_model, x, y),
        "value_and_grad_train": lambda: value_and_grad_train(loop_model, x, y),
        "jvp_loss": lambda: jvp_loss(loop_model, x, tangent_model, tangent_x),
        "vjp_loss": lambda: vjp_loss(loop_model, x),
        "vmap_forward": lambda: vmap_forward(loop_model, x),
        "cond_forward": lambda: cond_forward(block0, block1, x),
        "switch_forward": lambda: switch_forward(block0, block1, block2, x),
        "fori_loop_forward": lambda: fori_loop_forward(block0, x),
        "while_loop_forward": lambda: while_loop_forward(block0, x),
        "scan_train": lambda: scan_train(scan_model, x, y),
        "remat_train": lambda: remat_train(remat_model, x, y),
    }, _tree_param_count(loop_model)


def _geomean(values: list[float]) -> float:
    """Geometric mean of positive values."""
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    prod = math.prod(vals)
    return prod ** (1.0 / len(vals))


def _generate_plots(payload: dict[str, Any], plots_dir: Path, tag: str) -> None:
    """Generate matplotlib plots from benchmark payload."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Skipping plots: matplotlib not available ({exc})")
        return

    rows = payload["rows"]
    ops: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        ops.setdefault(r["op"], {})[r["library"]] = r
    op_names = [op for op in OP_ORDER if op in ops]
    if not op_names:
        return

    libs = ["jax", "nnx", "spx"]
    lib_labels = {"jax": "Raw JAX", "nnx": "Flax NNX", "spx": "spectrax"}
    colors = {"jax": "#264653", "nnx": "#e76f51", "spx": "#2a9d8f"}
    ink = "#1f2522"
    muted = "#6f766f"
    grid = "#d8d0c2"
    panel = "#fffaf1"
    paper = "#f4efe4"
    out_prefix = str(plots_dir / f"{tag}")
    op_labels = [op.replace("_and_", "+").replace("_", " ") for op in op_names]
    n_ops = len(op_names)
    y = np.arange(n_ops)
    bar_h = 0.22

    def metric(lib: str, op: str, key: str) -> float:
        """Return a numeric metric for a given library and operation.

        Args:
            lib: Library key (``"jax"``, ``"nnx"``, or ``"spx"``).
            op: Operation name.
            key: Payload key to extract (e.g. ``"median_ms"``).

        Returns:
            The metric as a positive ``float``, or ``nan`` if missing or
            non-positive.
        """
        value = ops[op].get(lib, {}).get(key, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return value if value > 0 else float("nan")

    def values_for(lib: str, key: str) -> np.ndarray:
        """Collect a metric across all operations for one library.

        Args:
            lib: Library key.
            key: Payload key to extract.

        Returns:
            1-D array of metric values (``nan`` where missing).
        """
        return np.asarray([metric(lib, op, key) for op in op_names], dtype=float)

    def positive(values: np.ndarray) -> np.ndarray:
        """Filter to finite, strictly-positive values.

        Args:
            values: Input array.

        Returns:
            1-D array of positive finite values.
        """
        return values[np.isfinite(values) & (values > 0)]

    def fmt_ms(value: float, _pos: float | None = None) -> str:
        """Format a millisecond value for axis tick labels.

        Args:
            value: Duration in milliseconds.
            _pos: Unused matplotlib position argument.

        Returns:
            Human-readable string such as ``"1.2s"``, ``"12ms"``, or
            ``"0.5ms"``.
        """
        if value >= 1000:
            return f"{value / 1000:.1f}s"
        if value >= 10:
            return f"{value:.0f}ms"
        return f"{value:.2g}ms"

    def fmt_ratio(value: float, _pos: float | None = None) -> str:
        """Format a ratio value for axis tick labels.

        Args:
            value: Ratio value.
            _pos: Unused matplotlib position argument.

        Returns:
            String such as ``"1.5x"``.
        """
        return f"{value:g}x"

    def new_fig(width: float, height: float):
        """Create a styled matplotlib figure and axis.

        Args:
            width: Figure width in inches.
            height: Figure height in inches.

        Returns:
            ``(fig, ax)`` tuple with paper/panel background colours applied.
        """
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
        fig.patch.set_facecolor(paper)
        ax.set_facecolor(panel)
        return fig, ax

    def polish_axis(ax: Any, *, xlabel: str, title: str, subtitle: str | None = None) -> None:
        """Apply shared styling to a plot axis.

        Args:
            ax: Matplotlib axis object.
            xlabel: Label for the x-axis.
            title: Bold left-aligned title.
            subtitle: Optional smaller subtitle above the title.
        """
        ax.set_xlabel(xlabel, color=muted, labelpad=10)
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color=ink, pad=18)
        if subtitle:
            ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=muted, fontsize=9, va="bottom")
        ax.tick_params(axis="x", colors=muted)
        ax.tick_params(axis="y", colors=ink, length=0)
        ax.grid(axis="x", color=grid, linewidth=0.8, alpha=0.85)
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.set_axisbelow(True)

    def save(fig: Any, path: str) -> None:
        """Save a figure to disk and close it.

        Args:
            fig: Matplotlib figure object.
            path: Destination file path.
        """
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  plot: {path}")

    style = {
        "axes.edgecolor": grid,
        "axes.labelsize": 10,
        "axes.titlelocation": "left",
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }

    with plt.rc_context(style):
        fig_h = max(7.0, n_ops * 0.62)

        fig, ax = new_fig(11.5, fig_h)
        all_runtime = positive(np.concatenate([values_for(lib, "median_ms") for lib in libs]))
        for i, lib in enumerate(libs):
            vals = values_for(lib, "median_ms")
            ax.barh(
                y + (i - 1) * bar_h,
                vals,
                bar_h * 0.9,
                label=lib_labels[lib],
                color=colors[lib],
                edgecolor=panel,
                linewidth=0.7,
            )
        if all_runtime.size:
            ax.set_xscale("log")
            ax.set_xlim(all_runtime.min() * 0.7, all_runtime.max() * 1.55)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_ms))
        ax.set_yticks(y, labels=op_labels)
        ax.invert_yaxis()
        polish_axis(
            ax,
            xlabel="Median runtime per call",
            title="Runtime by transform",
            subtitle="Log scale. Lower bars are better; grouped by benchmark operation.",
        )
        ax.legend(loc="lower right", ncols=3, bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
        save(fig, f"{out_prefix}_median_runtime.png")

        nnx_ratios = np.asarray(
            [
                (
                    metric("nnx", op, "median_ms") / metric("jax", op, "median_ms")
                    if np.isfinite(metric("jax", op, "median_ms"))
                    else np.nan
                )
                for op in op_names
            ],
            dtype=float,
        )
        spx_ratios = np.asarray(
            [
                (
                    metric("spx", op, "median_ms") / metric("jax", op, "median_ms")
                    if np.isfinite(metric("jax", op, "median_ms"))
                    else np.nan
                )
                for op in op_names
            ],
            dtype=float,
        )
        nnx_gm = _geomean(nnx_ratios.tolist())
        spx_gm = _geomean(spx_ratios.tolist())
        ratio_values = positive(np.concatenate([nnx_ratios, spx_ratios, np.asarray([1.0])]))

        fig, ax = new_fig(11.5, fig_h)
        if ratio_values.size:
            low = max(ratio_values.min() * 0.75, 0.05)
            high = ratio_values.max() * 1.35
            ax.set_xscale("log")
            ax.set_xlim(low, high)
            ax.axvspan(low, 1.0, color="#dfeee4", alpha=0.8, zorder=0)
            ax.axvspan(1.0, high, color="#f7ded4", alpha=0.65, zorder=0)
        for offset, lib, vals in [(-bar_h / 1.8, "nnx", nnx_ratios), (bar_h / 1.8, "spx", spx_ratios)]:
            left = np.minimum(vals, 1.0)
            width = np.abs(vals - 1.0)
            ax.barh(
                y + offset,
                width,
                bar_h,
                left=left,
                label=f"{lib_labels[lib]} / JAX",
                color=colors[lib],
                edgecolor=panel,
                linewidth=0.7,
            )
            for yy, value in zip(y + offset, vals, strict=False):
                if np.isfinite(value):
                    text_x = value * (1.035 if value >= 1.0 else 0.965)
                    ha = "left" if value >= 1.0 else "right"
                    ax.text(text_x, yy, f"{value:.2f}x", va="center", ha=ha, fontsize=7.5, color=ink)
        ax.axvline(1.0, color=ink, linewidth=1.1, alpha=0.9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_ratio))
        ax.set_yticks(y, labels=op_labels)
        ax.invert_yaxis()
        polish_axis(
            ax,
            xlabel="Runtime ratio vs raw JAX",
            title="Relative runtime",
            subtitle=f"Left of 1.0 is faster; right of 1.0 is slower. Geomean: NNX {nnx_gm:.3f}x, spectrax {spx_gm:.3f}x.",
        )
        ax.legend(loc="lower right", ncols=2, bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
        save(fig, f"{out_prefix}_speedup_vs_jax.png")

        fig, ax = new_fig(11.5, fig_h)
        all_compile = positive(np.concatenate([values_for(lib, "compile_ms") for lib in libs]))
        for i, lib in enumerate(libs):
            vals = values_for(lib, "compile_ms")
            ax.barh(
                y + (i - 1) * bar_h,
                vals,
                bar_h * 0.9,
                label=lib_labels[lib],
                color=colors[lib],
                edgecolor=panel,
                linewidth=0.7,
            )
        if all_compile.size:
            ax.set_xscale("log")
            ax.set_xlim(all_compile.min() * 0.7, all_compile.max() * 1.55)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_ms))
        ax.set_yticks(y, labels=op_labels)
        ax.invert_yaxis()
        polish_axis(
            ax,
            xlabel="First-call compile + execution latency",
            title="Compile latency",
            subtitle="Log scale. Includes first execution so XLA compilation is visible.",
        )
        ax.legend(loc="lower right", ncols=3, bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
        save(fig, f"{out_prefix}_compile_time.png")

        fig, ax = new_fig(11.5, fig_h)
        all_spread = positive(np.concatenate([values_for(lib, "p95_ms") for lib in libs]))
        for i, lib in enumerate(libs):
            med = values_for(lib, "median_ms")
            p05 = values_for(lib, "p05_ms")
            p95 = values_for(lib, "p95_ms")
            xerr = np.vstack([np.maximum(0, med - p05), np.maximum(0, p95 - med)])
            ax.errorbar(
                med,
                y + (i - 1) * bar_h,
                xerr=xerr,
                fmt="o",
                ms=4.5,
                capsize=2.5,
                elinewidth=1.6,
                color=colors[lib],
                ecolor=colors[lib],
                alpha=0.95,
                label=lib_labels[lib],
            )
        if all_spread.size:
            ax.set_xscale("log")
            ax.set_xlim(all_spread.min() * 0.7, all_spread.max() * 1.6)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_ms))
        ax.set_yticks(y, labels=op_labels)
        ax.invert_yaxis()
        polish_axis(
            ax,
            xlabel="Runtime spread, p05 to p95",
            title="Timing stability",
            subtitle="Dot is median; whiskers show p05-p95. Lower and tighter is better.",
        )
        ax.legend(loc="lower right", ncols=3, bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
        save(fig, f"{out_prefix}_p05_p95.png")

        fig, ax = new_fig(7.5, 5.0)
        summary = np.asarray([nnx_gm, spx_gm], dtype=float)
        labels = ["Flax NNX", "spectrax"]
        summary_colors = [colors["nnx"], colors["spx"]]
        ymax = max(1.15, float(np.nanmax(summary)) * 1.25 if np.isfinite(summary).any() else 1.15)
        ax.axhspan(0.0, 1.0, color="#dfeee4", alpha=0.8, zorder=0)
        ax.axhspan(1.0, ymax, color="#f7ded4", alpha=0.65, zorder=0)
        bars = ax.bar(labels, summary, color=summary_colors, width=0.56, edgecolor=panel, linewidth=1.0)
        ax.axhline(1.0, color=ink, linewidth=1.1)
        ax.set_ylim(0, ymax)
        for bar, value in zip(bars, summary, strict=False):
            if not np.isfinite(value):
                continue
            status = "faster" if value < 1.0 else "slower"
            ax.annotate(
                f"{value:.3f}x\n{status}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=ink,
                fontweight="bold",
            )
        polish_axis(
            ax,
            xlabel="",
            title="Overall runtime ratio",
            subtitle=f"{payload.get('device', '?').upper()} / {payload.get('preset', '?')} preset. 1.0 is raw JAX parity.",
        )
        ax.set_ylabel("Geomean runtime ratio vs JAX", color=muted, labelpad=10)
        ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.85)
        ax.tick_params(axis="x", colors=ink, length=0)
        save(fig, f"{out_prefix}_summary_geomean.png")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Three-way Llama transform benchmark")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--device", default=os.environ.get("JAX_PLATFORMS", "cpu"), choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="Use fewer warmup/iters for local iteration.")
    parser.add_argument(
        "--ops",
        default="all",
        help=(
            "Comma-separated subset from "
            "eval_shape,jit_forward,grad_train,value_and_grad_train,jvp_loss,vjp_loss,"
            "vmap_forward,cond_forward,switch_forward,fori_loop_forward,while_loop_forward,"
            "scan_train,remat_train"
        ),
    )
    parser.add_argument("--out", default="results", help="Output directory under benchmarks/")
    parser.add_argument("--tag", default="llama_transforms_3way")
    parser.add_argument(
        "--plots", action="store_true", help="Generate matplotlib plots and save them alongside results."
    )
    parser.add_argument("--plots-dir", default="benchmarks-plots", help="Directory to write plots into.")
    args = parser.parse_args(argv)

    cfg = _cfg_from_args(args)
    warmup = 1 if args.quick else args.warmup
    iters = 3 if args.quick else args.iters
    selected_ops = set(OP_ORDER if args.ops == "all" else [op.strip() for op in args.ops.split(",") if op.strip()])

    versions = {
        "jax_version": jax.__version__,
        "flax_version": __import__("flax").__version__,
        "spectrax_version": getattr(spx, "__version__", "unknown"),
    }

    print("Three-way Llama transform benchmark")
    print(f"  device       : {args.device} ({len(jax.devices())} devices visible)")
    print(
        "  config       : "
        f"layers={cfg.n_layers} d_model={cfg.d_model} heads={cfg.n_heads} "
        f"kv_heads={cfg.n_kv_heads} ffn={cfg.ffn} batch={cfg.batch} seq={cfg.seq_len} dtype={cfg.dtype_name}"
    )
    print(f"  warmup/iters : {warmup}/{iters}")
    print()

    jax_cases, jax_params = _build_jax_cases(cfg)
    nnx_cases, nnx_params = _build_nnx_cases(cfg)
    spx_cases, spx_params = _build_spx_cases(cfg)
    print(f"  param counts : jax={jax_params:,} nnx={nnx_params:,} spx={spx_params:,}")
    print()

    libraries = {
        "jax": jax_cases,
        "nnx": nnx_cases,
        "spx": spx_cases,
    }
    rows: list[dict[str, Any]] = []

    header = f"{'op':<22} {'lib':<5} {'compile_ms':>11} {'median_ms':>10} {'p05':>8} {'p95':>8}"
    print(header)
    print("-" * len(header))

    for op in OP_ORDER:
        if op not in selected_ops:
            continue
        op_rows: dict[str, dict[str, Any]] = {}
        for lib in ("jax", "nnx", "spx"):
            timing = _time_case(libraries[lib][op], warmup, iters)
            row = {
                "op": op,
                "library": lib,
                "device": args.device,
                "preset": args.preset,
                "params": {"jax": jax_params, "nnx": nnx_params, "spx": spx_params}[lib],
                **asdict(cfg),
                **versions,
                **timing,
            }
            op_rows[lib] = row
            print(
                f"{op:<22} {lib:<5} "
                f"{timing['compile_ms']:>11.3f} {timing['median_ms']:>10.3f} "
                f"{timing['p05_ms']:>8.3f} {timing['p95_ms']:>8.3f}"
            )

        base_compile = max(op_rows["jax"]["compile_ms"], 1e-9)
        base_median = max(op_rows["jax"]["median_ms"], 1e-9)
        for _lib, row in op_rows.items():
            row["compile_vs_jax"] = row["compile_ms"] / base_compile
            row["median_vs_jax"] = row["median_ms"] / base_median
            rows.append(row)
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": args.device,
        "preset": args.preset,
        "config": asdict(cfg),
        **versions,
        "rows": rows,
    }
    json_path = out_dir / f"{args.tag}.json"
    md_path = out_dir / f"{args.tag}.md"
    latest_json = out_dir / f"{args.tag}_latest.json"
    latest_md = out_dir / f"{args.tag}_latest.md"

    text = json.dumps(payload, indent=2)
    json_path.write_text(text)
    latest_json.write_text(text)

    md = _make_markdown(cfg, args.device, rows)
    md_path.write_text(md)
    latest_md.write_text(md)

    geomean_nnx = (
        statistics.geometric_mean([row["median_vs_jax"] for row in rows if row["library"] == "nnx"])
        if any(row["library"] == "nnx" for row in rows)
        else 1.0
    )
    geomean_spx = (
        statistics.geometric_mean([row["median_vs_jax"] for row in rows if row["library"] == "spx"])
        if any(row["library"] == "spx" for row in rows)
        else 1.0
    )

    print("Summary ratios vs raw JAX (median runtime geomean)")
    print(f"  nnx/jax : {geomean_nnx:.3f}")
    print(f"  spx/jax : {geomean_spx:.3f}")
    print()
    print("What this means:")
    print("  - These are RUNTIME ratios, not speedups.")
    print("  - 1.0 means exactly as fast as raw JAX.")
    print("  - >1.0 means SLOWER than raw JAX (higher = worse).")
    print("  - <1.0 means FASTER than raw JAX (lower = better).")
    print()
    if geomean_nnx > 1.0:
        print(f"  Example: nnx/jax = {geomean_nnx:.3f} means NNX is ~{geomean_nnx:.2f}x slower than raw JAX on average.")
    else:
        print(
            f"  Example: nnx/jax = {geomean_nnx:.3f} means NNX is ~{1.0 / geomean_nnx:.2f}x faster than raw JAX on average."
        )
    if geomean_spx > 1.0:
        print(
            f"  Example: spx/jax = {geomean_spx:.3f} means spectrax is ~{geomean_spx:.2f}x slower than raw JAX on average."
        )
    else:
        print(
            f"  Example: spx/jax = {geomean_spx:.3f} means spectrax is ~{1.0 / geomean_spx:.2f}x faster than raw JAX on average."
        )
    print()
    print(f"JSON: {json_path}")
    print(f"MD  : {md_path}")

    if args.plots:
        plots_dir = ROOT / args.plots_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        _generate_plots(payload, plots_dir, args.tag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
