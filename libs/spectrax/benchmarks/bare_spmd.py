# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bare-bones SPMD pipeline-parallel reference.

Pure JAX — no spectrax abstractions. Serves as the floor for what
our :func:`spectrax.pipeline.spmd_run` should achieve: stack per-stage
params, shard them along the pipeline axis of a mesh, write a plain
``jax.jit`` over an unrolled microbatch loop, and let XLA schedule the
cross-device transfers automatically.

Usage::

    python -m benchmarks.bare_spmd --model-size 300m --n-stages 4 --iters 3
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def init_block(key, d, ffn, n_heads, dtype=jnp.bfloat16):
    """Return a dict of param arrays for one transformer block.

    Args:
        key: JAX PRNG key.
        d: Model dimension.
        ffn: FFN hidden dimension.
        n_heads: Number of attention heads (used for scale validation).
        dtype: Parameter dtype.

    Returns:
        Dictionary with ``qkv``, ``o``, ``fc1``, ``fc2`` weight arrays.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d // n_heads
    scale_qkv = jnp.asarray(1.0 / jnp.sqrt(d), dtype=dtype)
    scale_o = jnp.asarray(1.0 / jnp.sqrt(d), dtype=dtype)
    scale_fc = jnp.asarray(1.0 / jnp.sqrt(d), dtype=dtype)
    scale_proj = jnp.asarray(1.0 / jnp.sqrt(ffn), dtype=dtype)
    return {
        "qkv": jax.random.normal(k1, (d, 3 * d), dtype=dtype) * scale_qkv,
        "o": jax.random.normal(k2, (d, d), dtype=dtype) * scale_o,
        "fc1": jax.random.normal(k3, (d, ffn), dtype=dtype) * scale_fc,
        "fc2": jax.random.normal(k4, (ffn, d), dtype=dtype) * scale_proj,
    }


def block_apply(p, x, n_heads):
    """Forward: pre-norm attention + pre-norm FFN with residuals.

    Args:
        p: Parameter dict for this block (``qkv``, ``o``, ``fc1``, ``fc2``).
        x: Input tensor ``(b, t, d)``.
        n_heads: Number of attention heads.

    Returns:
        Output tensor ``(b, t, d)``.
    """
    b, t, d = x.shape
    head_dim = d // n_heads
    h = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-5)
    qkv = h @ p["qkv"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)
    attn = jax.nn.softmax((q @ k.swapaxes(-1, -2)) / jnp.sqrt(head_dim).astype(x.dtype), axis=-1)
    h = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
    h = h @ p["o"]
    x = x + h
    h = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-5)
    h = jax.nn.gelu(h @ p["fc1"]) @ p["fc2"]
    return x + h


def stage_apply(stage_params, x, n_heads, n_blocks_per_stage):
    """Apply one pipeline stage: ``n_blocks_per_stage`` sequential blocks.

    Args:
        stage_params: Stacked params pytree with leading block axis.
        x: Input tensor.
        n_heads: Number of attention heads.
        n_blocks_per_stage: How many blocks to run in this stage.

    Returns:
        Output tensor after all blocks.
    """
    for b in range(n_blocks_per_stage):
        bp = jax.tree.map(lambda t, i=b: t[i], stage_params)
        x = block_apply(bp, x, n_heads)
    return x


def build_step(n_stages, n_blocks_per_stage, n_heads, microbatches):
    """Return a jitted ``(stacked_params, x_mb, y_mb) -> (loss, grads)`` step.

    ``stacked_params``: pytree where every leaf has a leading axis of
    size ``n_stages`` — must be sharded along that axis on the
    pipeline mesh axis. Each microbatch threads through every stage
    by indexing ``stacked_params[s]`` at static offset ``s`` — XLA
    places each indexed slice on the device that holds it and inserts
    the cross-stage transfer automatically.

    Args:
        n_stages: Number of pipeline stages.
        n_blocks_per_stage: Blocks owned by each stage.
        n_heads: Number of attention heads.
        microbatches: Number of microbatches per step.

    Returns:
        A jitted ``step(stacked_params, xs, ys)`` function.
    """

    def loss_fn(out, y):
        """MSE loss between ``out`` and ``y`` in float32."""
        return ((out.astype(jnp.float32) - y.astype(jnp.float32)) ** 2).mean()

    def forward_through_stages(stacked_params, x):
        """Thread ``x`` through every stage via static-index slicing."""
        for s in range(n_stages):
            sp = jax.tree.map(lambda t, i=s: t[i], stacked_params)
            x = stage_apply(sp, x, n_heads, n_blocks_per_stage)
        return x

    def total_loss(stacked_params, xs, ys):
        """Average ``loss_fn`` across all microbatches."""
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for mb in range(microbatches):
            out = forward_through_stages(stacked_params, xs[mb])
            total = total + loss_fn(out, ys[mb])
        return total / jnp.asarray(microbatches, dtype=jnp.float32)

    @jax.jit
    def step(stacked_params, xs, ys):
        """Jitted training step: returns ``(loss, grads)``."""
        return jax.value_and_grad(total_loss)(stacked_params, xs, ys)

    return step


@dataclass
class Config:
    """Model + pipeline shape for the bare SPMD benchmark."""

    n_layers: int
    d_model: int
    ffn: int
    n_heads: int
    batch: int
    seq_len: int
    n_stages: int
    microbatches: int

    @property
    def blocks_per_stage(self) -> int:
        """Blocks owned by each pipeline stage; requires even division."""
        if self.n_layers % self.n_stages:
            raise ValueError("n_layers must divide n_stages")
        return self.n_layers // self.n_stages

    @property
    def total_params(self) -> int:
        """Approximate total parameter count across all blocks."""
        per_block = 4 * self.d_model * self.d_model + 2 * self.d_model * self.ffn
        return self.n_layers * per_block


def init_stacked_params(cfg, key):
    """Build a stacked-params pytree of shape ``(n_stages, n_blocks, ...)``.

    Inner ``jax.tree.map`` call stacks per-stage blocks along the
    leading axis to shape ``(blocks, ...)``; the outer call stacks
    stages to shape ``(n_stages, blocks, ...)``.

    Args:
        cfg: :class:`Config` object with model/pipeline shape.
        key: JAX PRNG key.

    Returns:
        Stacked parameter pytree.
    """
    keys = jax.random.split(key, cfg.n_stages * cfg.blocks_per_stage)
    keys = keys.reshape(cfg.n_stages, cfg.blocks_per_stage, 2)
    blocks = []
    for s in range(cfg.n_stages):
        stage_blocks = [init_block(keys[s, b], cfg.d_model, cfg.ffn, cfg.n_heads) for b in range(cfg.blocks_per_stage)]
        stage = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *stage_blocks)
        blocks.append(stage)
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *blocks)
    return stacked


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — parse args, run the benchmark, print the summary."""
    parser = argparse.ArgumentParser(description="Bare-bones SPMD pipeline benchmark")
    parser.add_argument("--model-size", choices=["300m", "1b", "5b", "7b"], default="300m")
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args(argv)

    presets = {
        "300m": dict(n_layers=12, d_model=1024, n_heads=16, ffn=4096, batch=4, seq_len=128),
        "1b": dict(n_layers=24, d_model=2048, n_heads=16, ffn=8192, batch=4, seq_len=128),
        "5b": dict(n_layers=40, d_model=3200, n_heads=20, ffn=12800, batch=4, seq_len=128),
        "7b": dict(n_layers=40, d_model=4096, n_heads=32, ffn=14336, batch=4, seq_len=128),
    }
    p = presets[args.model_size]
    cfg = Config(
        n_layers=p["n_layers"],
        d_model=p["d_model"],
        ffn=p["ffn"],
        n_heads=p["n_heads"],
        batch=p["batch"],
        seq_len=p["seq_len"],
        n_stages=args.n_stages,
        microbatches=args.microbatches,
    )

    devices = jax.devices()[: cfg.n_stages]
    if len(devices) < cfg.n_stages:
        raise SystemExit(f"need {cfg.n_stages} devices; have {len(devices)}")
    mesh = Mesh(devices, axis_names=("pp",))
    pp_sharding = NamedSharding(mesh, PartitionSpec("pp"))

    print(f"bare SPMD ({len(jax.devices())} {jax.devices()[0].platform})")
    print(f"  model: {args.model_size}  ~{cfg.total_params / 1e9:.2f}B params")
    print(f"  layers: {cfg.n_layers}  blocks/stage: {cfg.blocks_per_stage}")
    print(f"  d_model: {cfg.d_model}  heads: {cfg.n_heads}  ffn: {cfg.ffn}")
    print(f"  batch: {cfg.batch} x seq {cfg.seq_len}  M: {cfg.microbatches}")

    key = jax.random.PRNGKey(0)
    stacked = init_stacked_params(cfg, key)
    stacked = jax.device_put(stacked, pp_sharding)

    def dummy_batch(step):
        """Return an ``(xs, ys)`` microbatched bf16 batch for ``step``."""
        kx, ky = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), step))
        x = jax.random.normal(kx, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=jnp.bfloat16)
        y = jax.random.normal(ky, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=jnp.bfloat16)
        return x.reshape(cfg.microbatches, -1, cfg.seq_len, cfg.d_model), y.reshape(
            cfg.microbatches, -1, cfg.seq_len, cfg.d_model
        )

    step = build_step(cfg.n_stages, cfg.blocks_per_stage, cfg.n_heads, cfg.microbatches)

    xs, ys = dummy_batch(0)
    t0 = time.perf_counter_ns()
    loss, _ = step(stacked, xs, ys)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, args.iters + 1):
        xs, ys = dummy_batch(i)
        t0 = time.perf_counter_ns()
        loss, _ = step(stacked, xs, ys)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    median = sorted(step_times)[len(step_times) // 2]
    print()
    print(f"  compile_ms : {compile_ms:>10.1f}")
    print(f"  step_ms    : {median:>10.2f}  (median of {args.iters})")
    print(f"  loss       : {float(loss):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
