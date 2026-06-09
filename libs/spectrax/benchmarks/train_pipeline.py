# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline-parallel benchmark harness.

Builds an N-stage :class:`~spectrax.pipeline.PipelineSequential` where
each stage bundles ``n_blocks_per_stage`` identical transformer
blocks, then runs one training step under each available schedule
via **SPMD** (:func:`spectrax.pipeline.pipeline_step`) and **MPMD**
(:func:`spectrax.pipeline.sxcall`) for comparison.

The per-stage-identical structure lets both runtimes run the same
model: SPMD requires structural equivalence across stages; MPMD can
do it either way but here we use the same structure for an
apples-to-apples check.

Usage::

    python -m benchmarks.train_pipeline --model-size 5b --n-stages 4 --iters 3
    python -m benchmarks.train_pipeline --model-size 1b --n-stages 4
    python -m benchmarks.train_pipeline --model-size custom \
        --n-layers 40 --d-model 3200 --ffn 12800 --n-heads 20

``model-size`` presets produce transformers of the named parameter
count (approximately); specify ``custom`` plus the shape flags to
override.

The benchmark defers to whatever JAX backend is available. To
simulate multi-device CPU locally instead, export before running::

    export JAX_PLATFORMS=cpu
    export XLA_FLAGS='--xla_force_host_platform_device_count=4'
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import spectrax as spx
from spectrax import functional as F
from spectrax import nn
from spectrax.nn import PipelineSequential
from spectrax.runtime.mpmd import sxcall
from spectrax.runtime.schedules import GPipe, InterleavedH1, Std1F1B, ZeroBubbleH1
from spectrax.runtime.types import MpMdMesh


class Block(spx.Module):
    """One transformer block: pre-norm attention + FFN with residuals."""

    def __init__(self, d, ffn, n_heads, *, rngs):
        """Initialize attention + FFN sublayers."""
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, rngs=rngs, dtype=jnp.bfloat16)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn, rngs=rngs, dtype=jnp.bfloat16)
        self.fc2 = nn.Linear(ffn, d, rngs=rngs, dtype=jnp.bfloat16)

    def forward(self, x):
        """Apply pre-norm attention + pre-norm FFN with residuals."""
        h = self.ln1(x)
        h = self.attn(h, h, h)
        x = x + h
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class Stage(spx.Module):
    """One pipeline stage: ``n_blocks`` identical :class:`Block` s."""

    def __init__(self, n_blocks, d, ffn, n_heads, *, rngs):
        """Initialize ``n_blocks`` identical :class:`Block` instances in a Sequential."""
        super().__init__()
        self.blocks = nn.Sequential(*[Block(d, ffn, n_heads, rngs=rngs) for _ in range(n_blocks)])

    def forward(self, x):
        """Apply every block sequentially and return the final activation."""
        return self.blocks(x)


def _warn_if_device_count_mismatch(requested: int) -> None:
    """Emit a warning when the JAX-visible device count doesn't match.

    JAX's device count is fixed at initialization time by
    ``XLA_FLAGS=--xla_force_host_platform_device_count``, which this
    script sets to 4 by default. Users requesting a different stage
    count via ``--n-stages`` need to export the env var themselves.
    """
    available = len(jax.devices())
    if available < requested:
        print(
            f"WARNING: --n-stages={requested} exceeds JAX-visible device "
            f"count ({available}). Re-run with "
            f"XLA_FLAGS='--xla_force_host_platform_device_count={requested}'."
        )


@dataclass
class ModelConfig:
    """Transformer config used by :func:`build_model`."""

    n_layers: int
    d_model: int
    ffn: int
    n_heads: int
    batch: int
    seq_len: int
    n_stages: int

    @property
    def params_per_block(self) -> int:
        """Approximate parameter count per transformer block."""
        return 4 * self.d_model * self.d_model + 2 * self.d_model * self.ffn

    @property
    def total_params(self) -> int:
        """Approximate total parameter count across all blocks."""
        return self.n_layers * self.params_per_block

    @property
    def blocks_per_stage(self) -> int:
        """How many blocks each pipeline stage owns."""
        if self.n_layers % self.n_stages:
            raise ValueError(
                f"n_layers ({self.n_layers}) must divide n_stages "
                f"({self.n_stages}); got remainder "
                f"{self.n_layers % self.n_stages}."
            )
        return self.n_layers // self.n_stages


def build_model(cfg: ModelConfig, virtual_stages: int = 1) -> PipelineSequential:
    """Build a :class:`PipelineSequential` of identical-shape stages.

    For virtual-stage schedules (``virtual_stages > 1``) we split
    ``n_layers`` into ``virtual_stages * n_stages`` smaller logical
    stages rather than ``n_stages``. Total parameter count and compute
    is unchanged; only the partitioning differs.
    """
    rngs = spx.Rngs(0)
    n_logical = virtual_stages * cfg.n_stages
    if cfg.n_layers % n_logical:
        raise ValueError(
            f"n_layers={cfg.n_layers} must divide n_logical={n_logical} "
            f"(virtual_stages={virtual_stages} x n_stages={cfg.n_stages})"
        )
    blocks_per_logical = cfg.n_layers // n_logical
    return PipelineSequential(
        *[Stage(blocks_per_logical, cfg.d_model, cfg.ffn, cfg.n_heads, rngs=rngs) for _ in range(n_logical)]
    )


def _dummy_batch(cfg: ModelConfig, step: int):
    """Return a deterministic ``(x, y)`` bf16 batch for ``step``."""
    key = jax.random.fold_in(jax.random.PRNGKey(0), step)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=jnp.bfloat16)
    y = jax.random.normal(ky, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=jnp.bfloat16)
    return x, y


def _median(xs: list[float]) -> float:
    """Return the median of ``xs`` (in same units)."""
    s = sorted(xs)
    return s[len(s) // 2]


def run_spmd(cfg: ModelConfig, schedule_name: str, microbatches: int, iters: int):
    """Time ``iters`` SPMD pipeline steps under ``schedule_name``.

    Compiles once (first call) then measures steady-state median step
    latency. Returns ``(compile_ms, median_step_ms, final_loss)``.
    """
    from jax.sharding import Mesh

    from spectrax.runtime.schedules import DualPipeV, Eager1F1B, Interleaved1F1BPlusOne, InterleavedGPipe, KimiK2

    schedule_cls = {
        "gpipe": GPipe,
        "1f1b": Std1F1B,
        "zb_h1": ZeroBubbleH1,
        "interleaved": InterleavedH1,
        "eager1f1b": Eager1F1B,
        "interleaved_gpipe": InterleavedGPipe,
        "interleaved_plus_one": Interleaved1F1BPlusOne,
        "kimi_k2": KimiK2,
        "dualpipev": DualPipeV,
    }[schedule_name]
    if schedule_name in ("interleaved", "interleaved_gpipe", "interleaved_plus_one", "kimi_k2"):
        schedule = schedule_cls(microbatches=microbatches, virtual_stages=2)
    else:
        schedule = schedule_cls(microbatches=microbatches)

    devices = jax.devices()[: cfg.n_stages]
    mpmd_mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")

    model = build_model(cfg, virtual_stages=schedule.virtual_stages_per_rank())

    def loss_fn(out, y):
        """MSE loss for the final stage output."""
        return ((out.astype(jnp.float32) - y.astype(jnp.float32)) ** 2).mean()

    x, y = _dummy_batch(cfg, 0)

    t0 = time.perf_counter_ns()
    loss, _grads = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=loss_fn)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times: list[float] = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _grads = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=loss_fn)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def run_mpmd(cfg: ModelConfig, schedule_name: str, microbatches: int, iters: int):
    """Time ``iters`` MPMD pipeline steps under ``schedule_name``."""
    from spectrax.runtime.schedules import DualPipeV, Eager1F1B, Interleaved1F1BPlusOne, InterleavedGPipe, KimiK2

    schedule_cls = {
        "gpipe": GPipe,
        "1f1b": Std1F1B,
        "zb_h1": ZeroBubbleH1,
        "interleaved": InterleavedH1,
        "eager1f1b": Eager1F1B,
        "interleaved_gpipe": InterleavedGPipe,
        "interleaved_plus_one": Interleaved1F1BPlusOne,
        "kimi_k2": KimiK2,
        "dualpipev": DualPipeV,
    }[schedule_name]
    if schedule_name in ("interleaved", "interleaved_gpipe", "interleaved_plus_one", "kimi_k2"):
        schedule = schedule_cls(microbatches=microbatches, virtual_stages=2)
    else:
        schedule = schedule_cls(microbatches=microbatches)

    devices = jax.devices()[: cfg.n_stages]
    mpmd_mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")
    model = build_model(cfg, virtual_stages=schedule.virtual_stages_per_rank())

    def loss_fn(out, y):
        """MSE loss for the final stage output."""
        return ((out.astype(jnp.float32) - y.astype(jnp.float32)) ** 2).mean()

    x, y = _dummy_batch(cfg, 0)

    t0 = time.perf_counter_ns()
    loss, _grads = sxcall(
        model,
        (x, y),
        mesh=mpmd_mesh,
        schedule=schedule,
        loss_fn=loss_fn,
    )
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times: list[float] = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _grads = sxcall(
            model,
            (x, y),
            mesh=mpmd_mesh,
            schedule=schedule,
            loss_fn=loss_fn,
        )
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — parse args, dispatch, print a summary table."""
    parser = argparse.ArgumentParser(description="SPMD vs MPMD pipeline-parallel benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument(
        "--model-size",
        choices=["300m", "1b", "3b", "5b", "7b", "custom"],
        default="1b",
    )
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=None)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--schedules",
        default="gpipe,1f1b,zb_h1",
        help="comma-separated list from {gpipe, 1f1b, zb_h1, interleaved}",
    )
    parser.add_argument(
        "--modes",
        default="spmd,mpmd",
        help="comma-separated list from {spmd, mpmd}",
    )
    args = parser.parse_args(argv)

    presets = {
        "300m": {"n_layers": 12, "d_model": 1024, "n_heads": 16, "ffn": 4096, "batch": 4, "seq_len": 128},
        "1b": {"n_layers": 24, "d_model": 2048, "n_heads": 16, "ffn": 8192, "batch": 4, "seq_len": 128},
        "3b": {"n_layers": 32, "d_model": 2816, "n_heads": 22, "ffn": 11264, "batch": 4, "seq_len": 128},
        "5b": {"n_layers": 40, "d_model": 3200, "n_heads": 20, "ffn": 12800, "batch": 4, "seq_len": 128},
        "7b": {"n_layers": 40, "d_model": 4096, "n_heads": 32, "ffn": 14336, "batch": 4, "seq_len": 128},
        "custom": {},
    }
    preset = presets[args.model_size]
    args.n_layers = args.n_layers if args.n_layers is not None else preset.get("n_layers", 24)
    args.d_model = args.d_model if args.d_model is not None else preset.get("d_model", 2048)
    args.n_heads = args.n_heads if args.n_heads is not None else preset.get("n_heads", 16)
    args.ffn = args.ffn if args.ffn is not None else preset.get("ffn", 8192)
    args.batch = args.batch if args.batch is not None else preset.get("batch", 2)
    args.seq_len = args.seq_len if args.seq_len is not None else preset.get("seq_len", 128)
    args.microbatches = args.microbatches if args.microbatches is not None else args.n_stages

    _warn_if_device_count_mismatch(args.n_stages)

    cfg = ModelConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        ffn=args.ffn,
        n_heads=args.n_heads,
        batch=args.batch,
        seq_len=args.seq_len,
        n_stages=args.n_stages,
    )

    schedules = args.schedules.split(",")
    modes = args.modes.split(",")

    print("spectrax pipeline-parallel benchmark")
    print(f"  device         : {args.device} ({len(jax.devices())} devices)")
    print(f"  model size     : {args.model_size}  (~{cfg.total_params / 1e9:.2f}B params)")
    print(f"  n_layers       : {cfg.n_layers} ({cfg.blocks_per_stage} per stage x {cfg.n_stages} stages)")
    print(f"  d_model/heads  : {cfg.d_model} / {cfg.n_heads}")
    print(f"  ffn            : {cfg.ffn}")
    print(f"  batch / seq    : {cfg.batch} x {cfg.seq_len}")
    print(f"  microbatches   : {args.microbatches}")
    print(f"  schedules      : {schedules}")
    print(f"  modes          : {modes}")
    print(f"  iters          : {args.iters}")

    header = f"{'mode':<6} {'schedule':<12} {'compile_ms':>11} {'step_ms':>10} {'loss':>10}"
    print()
    print(header)
    print("-" * len(header))

    results: list[dict] = []
    for mode in modes:
        runner = run_spmd if mode == "spmd" else run_mpmd
        for sched in schedules:
            try:
                compile_ms, step_ms, loss = runner(cfg, sched, args.microbatches, args.iters)
                print(f"{mode:<6} {sched:<12} {compile_ms:>11.1f} {step_ms:>10.2f} {loss:>10.4f}")
                results.append(
                    {
                        "mode": mode,
                        "schedule": sched,
                        "compile_ms": compile_ms,
                        "step_ms": step_ms,
                        "loss": loss,
                    }
                )
            except Exception as e:
                print(f"{mode:<6} {sched:<12} ERROR: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
