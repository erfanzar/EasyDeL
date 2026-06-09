# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Llama 3 8B benchmark — spectrax MPMD vs SPMD vs plain ``jax.jit``.

Three runtimes against the **same** Llama 3 8B model architecture
(transformer blocks only — embeddings + lm_head excluded so step time
reflects the body):

* **MPMD** — :func:`~spectrax.pipeline.sxcall` with the user's
  schedule choice. Python-driven schedule loop, per-stage subjits.
* **SPMD** — :func:`~spectrax.pipeline.pipeline_step` (single-jit
  step over the whole pipeline; XLA picks the execution order).
* **JIT**  — plain ``jax.jit`` on the full model with FSDP-style
  parameter sharding across all 4 chips. No pipeline parallelism;
  data flows the natural single-program way.

Llama 3 8B specs:

  ============  ===========
  attribute     value
  ============  ===========
  n_layers      32
  d_model       4096
  n_heads       32
  n_kv_heads    8 (GQA)
  head_dim      128
  ffn (SwiGLU)  14336
  norm          RMSNorm
  position      RoPE (theta=500000)
  dtype         bfloat16
  ============  ===========

Usage::

    python -m benchmarks.llama3_8b --modes mpmd,spmd,jit --schedules gpipe
    python -m benchmarks.llama3_8b --modes mpmd --schedules gpipe,1f1b,kimi_k2 --microbatches 8

Note the ``jit`` mode only accepts ``--schedules gpipe`` (its placeholder).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import spectrax as spx
from spectrax import nn
from spectrax.nn import PipelineSequential
from spectrax.runtime.mpmd import sxcall
from spectrax.runtime.schedules import (
    DualPipeV,
    Eager1F1B,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    Std1F1B,
    ZeroBubbleH1,
)
from spectrax.runtime.types import MpMdMesh
from spectrax.sharding import (
    get_named_sharding,
    logical_axis_rules,
    with_sharding_constraint_by_name,
)


def _rope_freqs(seq_len: int, head_dim: int, theta: float = 500_000.0):
    """Precompute RoPE cos/sin tables of shape ``(seq_len, head_dim // 2)``."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv_freq)
    return jnp.cos(freqs).astype(jnp.bfloat16), jnp.sin(freqs).astype(jnp.bfloat16)


def _apply_rope(x, cos, sin):
    """Rotate the last-dim pairs of ``x`` ``(..., seq, n_heads, head_dim)``.

    Splits the last axis into ``(..., 2, head_dim/2)`` halves before rotating.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class Llama3Block(spx.Module):
    """One Llama 3 transformer block: RMSNorm + GQA attn + RMSNorm + SwiGLU FFN.

    Biases are disabled on every Linear to match the Llama 3 convention.
    """

    def __init__(self, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize GQA attention + SwiGLU FFN sublayers."""
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d // n_heads
        kv_d = n_kv_heads * self.head_dim
        self.norm1 = nn.RMSNorm(d)
        self.q = nn.Linear(d, d, use_bias=False, rngs=rngs, dtype=jnp.bfloat16)
        self.k = nn.Linear(d, kv_d, use_bias=False, rngs=rngs, dtype=jnp.bfloat16)
        self.v = nn.Linear(d, kv_d, use_bias=False, rngs=rngs, dtype=jnp.bfloat16)
        self.o = nn.Linear(d, d, use_bias=False, rngs=rngs, dtype=jnp.bfloat16)
        self.norm2 = nn.RMSNorm(d)
        self.gate = nn.Linear(d, ffn, use_bias=False, rngs=rngs, dtype=jnp.bfloat16)
        self.up = nn.Linear(d, ffn, use_bias=False, rngs=rngs, dtype=jnp.bfloat16)
        self.down = nn.Linear(ffn, d, use_bias=False, rngs=rngs, dtype=jnp.bfloat16)

    def forward(self, x, cos, sin):
        """Run pre-norm GQA+RoPE attention followed by pre-norm SwiGLU FFN.

        K and V heads are repeated to match Q's head count for GQA
        broadcasting; tensors are transposed to ``(b, h, t, d)`` before
        the attention matmul.
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


class Llama3Stage(spx.Module):
    """A pipeline stage = ``n_blocks`` Llama 3 blocks in sequence.

    Stages share the same RoPE cos/sin which we pass through ``forward`` —
    the table itself is non-trainable and lives outside the Module.

    TP-aware variants below use Megatron-style column/row parallelism.
    Logical axis rules used at runtime: ``"tp_head" -> "tp"`` (shards the
    num_heads dim of q/k/v out, o in) and ``"tp_ffn" -> "tp"`` (shards
    the ffn dim of gate/up out, down in). Two distinct logical names
    (instead of one ``"tp"``) allow the same mesh axis to play different
    roles per tensor and keep the option to map them to different
    physical axes later (e.g. ``tp_head -> "tp"``, ``tp_ffn -> "ep"``).
    """

    def __init__(self, n_blocks, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize ``n_blocks`` :class:`Llama3Block` instances in a Sequential."""
        super().__init__()
        self.blocks = nn.Sequential(*[Llama3Block(d, ffn, n_heads, n_kv_heads, rngs=rngs) for _ in range(n_blocks)])

    def forward(self, x):
        """Run every block with RoPE tables recomputed at stage entry.

        Pipeline stages accept a single positional input — RoPE tables
        are recomputed inside on the first call; a small cached cost.
        """
        _b, t, _d = x.shape
        head_dim = self.blocks[0].head_dim if hasattr(self.blocks[0], "head_dim") else 128
        cos, sin = _rope_freqs(t, head_dim)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x


class Llama3BlockTP(spx.Module):
    """Llama 3 block with role-specific logical sharding annotations.

    Each Linear's weight carries a logical ``("in", "out")`` partition
    spec; resolution to physical mesh axes happens at sharding-derivation
    time inside an active ``logical_axis_rules`` context. Inter-op
    activations are pinned with ``with_sharding_constraint_by_name`` so
    XLA emits the expected reduce-scatter / all-reduce collectives.
    """

    def __init__(self, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize TP-aware GQA attention + SwiGLU FFN sublayers."""
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d // n_heads
        kv_d = n_kv_heads * self.head_dim
        lin = dict(use_bias=False, dtype=jnp.bfloat16)
        self.norm1 = nn.RMSNorm(d)
        self.q = nn.Linear(d, d, sharding=(None, "tp_head"), rngs=rngs, **lin)
        self.k = nn.Linear(d, kv_d, sharding=(None, "tp_head"), rngs=rngs, **lin)
        self.v = nn.Linear(d, kv_d, sharding=(None, "tp_head"), rngs=rngs, **lin)
        self.o = nn.Linear(d, d, sharding=("tp_head", None), rngs=rngs, **lin)
        self.norm2 = nn.RMSNorm(d)
        self.gate = nn.Linear(d, ffn, sharding=(None, "tp_ffn"), rngs=rngs, **lin)
        self.up = nn.Linear(d, ffn, sharding=(None, "tp_ffn"), rngs=rngs, **lin)
        self.down = nn.Linear(ffn, d, sharding=("tp_ffn", None), rngs=rngs, **lin)

    def forward(self, x, cos, sin):
        """Run the TP-aware attention + SwiGLU FFN with activation sharding pins."""
        b, t, _ = x.shape
        x = with_sharding_constraint_by_name(x, (None, None, None))
        h = self.norm1(x)
        q = self.q(h).reshape(b, t, self.n_heads, self.head_dim)
        k = self.k(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        v = self.v(h).reshape(b, t, self.n_kv_heads, self.head_dim)
        q = with_sharding_constraint_by_name(q, (None, None, "tp_head", None))
        k = with_sharding_constraint_by_name(k, (None, None, "tp_head", None))
        v = with_sharding_constraint_by_name(v, (None, None, "tp_head", None))
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
        h = with_sharding_constraint_by_name(h, (None, None, "tp_head"))
        h = self.o(h)
        h = with_sharding_constraint_by_name(h, (None, None, None))
        x = x + h
        h = self.norm2(x)
        gate_proj = self.gate(h)
        up_proj = self.up(h)
        gate_proj = with_sharding_constraint_by_name(gate_proj, (None, None, "tp_ffn"))
        up_proj = with_sharding_constraint_by_name(up_proj, (None, None, "tp_ffn"))
        h = jax.nn.silu(gate_proj) * up_proj
        h = self.down(h)
        h = with_sharding_constraint_by_name(h, (None, None, None))
        return x + h


class Llama3StageTP(spx.Module):
    """A pipeline stage of TP-aware Llama 3 blocks (RoPE recomputed inside)."""

    def __init__(self, n_blocks, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize ``n_blocks`` :class:`Llama3BlockTP` instances in a Sequential."""
        super().__init__()
        self.blocks = nn.Sequential(*[Llama3BlockTP(d, ffn, n_heads, n_kv_heads, rngs=rngs) for _ in range(n_blocks)])

    def forward(self, x):
        """Run every TP-aware block with RoPE tables recomputed at stage entry."""
        _b, t, _d = x.shape
        head_dim = self.blocks[0].head_dim if hasattr(self.blocks[0], "head_dim") else 128
        cos, sin = _rope_freqs(t, head_dim)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x


def build_full_model_tp(cfg: Llama3Config) -> Llama3StageTP:
    """Build a single TP-aware ``Llama3StageTP`` with all blocks (for bare-TP JIT)."""
    rngs = spx.Rngs(0)
    return Llama3StageTP(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)


@dataclass
class Llama3Config:
    """Llama 3 8B (transformer body only)."""

    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn: int = 14336
    batch: int = 4
    seq_len: int = 128
    n_stages: int = 4
    microbatches: int = 4

    @property
    def total_params(self) -> int:
        """Approximate transformer-body parameter count.

        Attention: ``q (d**2) + k+v (2 * d * d * kv_heads/heads) + o (d**2)``.
        SwiGLU: ``gate + up + down = 3 * d * ffn``.
        """
        attn = 2 * self.d_model * self.d_model + 2 * self.d_model * (self.d_model * self.n_kv_heads // self.n_heads)
        ffn = 3 * self.d_model * self.ffn
        return self.n_layers * (attn + ffn)


def build_pipeline_model(cfg: Llama3Config, virtual_stages: int = 1) -> PipelineSequential:
    """Build a ``PipelineSequential`` of ``virtual_stages * n_stages`` Llama3Stages."""
    rngs = spx.Rngs(0)
    n_logical = virtual_stages * cfg.n_stages
    if cfg.n_layers % n_logical:
        raise ValueError(
            f"n_layers={cfg.n_layers} must divide n_logical={n_logical} "
            f"(virtual_stages={virtual_stages} x n_stages={cfg.n_stages})"
        )
    blocks_per_logical = cfg.n_layers // n_logical
    return PipelineSequential(
        *[
            Llama3Stage(
                blocks_per_logical,
                cfg.d_model,
                cfg.ffn,
                cfg.n_heads,
                cfg.n_kv_heads,
                rngs=rngs,
            )
            for _ in range(n_logical)
        ]
    )


def build_full_model(cfg: Llama3Config) -> Llama3Stage:
    """Build a single ``Llama3Stage`` with all 32 blocks (for the JIT runtime)."""
    rngs = spx.Rngs(0)
    return Llama3Stage(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)


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


_SCHEDULE_MAP = {
    "gpipe": GPipe,
    "1f1b": Std1F1B,
    "zb_h1": ZeroBubbleH1,
    "interleaved": InterleavedH1,
    "eager1f1b": Eager1F1B,
    "interleaved_gpipe": InterleavedGPipe,
    "interleaved_plus_one": Interleaved1F1BPlusOne,
    "kimi_k2": KimiK2,
    "dualpipev": DualPipeV,
}
_VIRTUAL_NAMES = {"interleaved", "interleaved_gpipe", "interleaved_plus_one", "kimi_k2"}


def _make_schedule(name: str, microbatches: int):
    """Instantiate the schedule class for ``name`` (virtual-stage schedules get ``virtual_stages=2``)."""
    cls = _SCHEDULE_MAP[name]
    if name in _VIRTUAL_NAMES:
        return cls(microbatches=microbatches, virtual_stages=2)
    return cls(microbatches=microbatches)


def run_spmd(cfg: Llama3Config, schedule_name: str, iters: int):
    """Time ``iters`` pipeline steps via :func:`sxcall`.

    SPMD-friendly path: uses the standard ``MpMdMesh`` with ``sxcall``
    which routes GPipe and other flat schedules through the vmap-based
    ``_gpipe_run`` fast-path (equivalent of the old SPMD single-jit).
    """
    schedule = _make_schedule(schedule_name, cfg.microbatches)
    devices = jax.devices()[: cfg.n_stages]
    mpmd_mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")
    model = build_pipeline_model(cfg, virtual_stages=schedule.virtual_stages_per_rank())

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)
    return compile_ms, _median(step_times), float(loss)


def run_spmd_scheduled(cfg: Llama3Config, schedule_name: str, iters: int):
    """Alias for :func:`run_spmd` — schedule-driven pipeline via :func:`sxcall`."""
    return run_spmd(cfg, schedule_name, iters)


def run_mpmd(cfg: Llama3Config, schedule_name: str, iters: int):
    """Time ``iters`` MPMD pipeline steps via :func:`sxcall`."""
    schedule = _make_schedule(schedule_name, cfg.microbatches)
    devices = jax.devices()[: cfg.n_stages]
    mpmd_mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")
    model = build_pipeline_model(cfg, virtual_stages=schedule.virtual_stages_per_rank())

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)
    return compile_ms, _median(step_times), float(loss)


def run_jit(cfg: Llama3Config, _schedule_name: str, iters: int):
    """Plain ``jax.jit`` over the full single-stage model.

    Replicates the model on every device (data-parallel-style: each
    device sees the same params, the activation flows through one big
    HLO). No pipeline parallelism; an intentionally vanilla baseline.
    """
    devices = jax.devices()[: cfg.n_stages]
    mesh = Mesh(devices, axis_names=("dp",))
    repl = NamedSharding(mesh, PartitionSpec())
    model = build_full_model(cfg)
    gdef, state = spx.export(model)
    state = jax.device_put(state, repl)

    def step_fn(state, x, y):
        """Bind the graphdef, compute loss, and return ``(loss, grads)``."""

        def loss_fn(state):
            """Forward + MSE loss under an ephemeral :func:`spx.bind`."""
            m = spx.bind(gdef, state)
            return _loss_fn(m(x), y)

        return jax.value_and_grad(loss_fn)(state)

    step = jax.jit(step_fn)
    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = step(state, x, y)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = step(state, x, y)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)
    return compile_ms, _median(step_times), float(loss)


def run_mpmd_tp(cfg: Llama3Config, schedule_name: str, iters: int):
    """MPMD pipeline with intra-stage TP.

    Builds a 2D mesh ``(pp, tp)`` of size ``cfg.n_stages``. The model
    is the TP-aware :class:`Llama3StageTP`; each pipeline stage runs
    over a ``tp``-sharded sub-mesh. Per-leaf shardings are resolved
    inside :func:`sxcall` from the model's logical axis names under
    an active :func:`logical_axis_rules` context.
    """
    schedule = _make_schedule(schedule_name, cfg.microbatches)
    n_stages = cfg.n_stages
    n_devices = len(jax.devices()[:n_stages])
    pp = 2
    tp = n_devices // pp
    assert pp * tp == n_devices, f"need pp({pp})*tp({tp})={n_devices} chips"
    devices = list(jax.devices()[:n_devices])
    import numpy as np

    devmat = np.array(devices).reshape(pp, tp)
    mesh = Mesh(devmat, axis_names=("pp", "tp"))
    mpmd_mesh = MpMdMesh(mesh, "pp")
    rules = [("tp_head", "tp"), ("tp_ffn", "tp")]

    cfg_pp = Llama3Config(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        ffn=cfg.ffn,
        batch=cfg.batch,
        seq_len=cfg.seq_len,
        n_stages=pp,
        microbatches=cfg.microbatches,
    )
    rngs = spx.Rngs(0)
    n_logical = schedule.virtual_stages_per_rank() * pp
    if cfg_pp.n_layers % n_logical:
        raise ValueError(f"n_layers={cfg_pp.n_layers} not divisible by n_logical={n_logical}")
    blocks_per_logical = cfg_pp.n_layers // n_logical
    with logical_axis_rules(rules), mesh:
        model = PipelineSequential(
            *[
                Llama3StageTP(
                    blocks_per_logical,
                    cfg_pp.d_model,
                    cfg_pp.ffn,
                    cfg_pp.n_heads,
                    cfg_pp.n_kv_heads,
                    rngs=rngs,
                )
                for _ in range(n_logical)
            ]
        )

        x, y = _dummy_batch(cfg, 0)
        t0 = time.perf_counter_ns()
        loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
        jax.block_until_ready(loss)
        compile_ms = (time.perf_counter_ns() - t0) / 1e6

        step_times = []
        for i in range(1, iters + 1):
            x, y = _dummy_batch(cfg, i)
            t0 = time.perf_counter_ns()
            loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
            jax.block_until_ready(loss)
            step_times.append((time.perf_counter_ns() - t0) / 1e6)
    return compile_ms, _median(step_times), float(loss)


def run_jit_tp(cfg: Llama3Config, _schedule_name: str, iters: int):
    """Bare TP single-jit baseline: tp = ``cfg.n_stages`` chips, no PP.

    Uses :class:`Llama3StageTP` whose Linears carry role-specific logical
    sharding. Resolved against a ``("tp",)`` mesh under
    ``logical_axis_rules([("tp_head","tp"),("tp_ffn","tp")])`` — q/k/v +
    gate/up become column-parallel, o + down become row-parallel.
    """
    devices = jax.devices()[: cfg.n_stages]
    mesh = Mesh(devices, axis_names=("tp",))
    rules = [("tp_head", "tp"), ("tp_ffn", "tp")]
    repl = NamedSharding(mesh, PartitionSpec())

    with logical_axis_rules(rules):
        model = build_full_model_tp(cfg)
        gdef, state = spx.export(model)
        shards = get_named_sharding(model, mesh)

        def _to_sharded(path, leaf):
            """Place ``leaf`` with its per-leaf sharding (falling back to replication)."""
            for col_shards in shards.values():
                if path in col_shards:
                    return jax.device_put(leaf, col_shards[path])
            return jax.device_put(leaf, repl)

        new_collections: dict[str, dict[str, Any]] = {}
        for col, path, leaf in state.items():
            new_collections.setdefault(col, {})[path] = _to_sharded(path, leaf)
        state = type(state)(new_collections)

        def step_fn(state, x, y):
            """Jitted step: bind, compute loss, and return ``(loss, grads)``."""
            with logical_axis_rules(rules):

                def loss_fn(state):
                    """Forward + MSE loss under an ephemeral :func:`spx.bind`."""
                    m = spx.bind(gdef, state)
                    return _loss_fn(m(x), y)

                return jax.value_and_grad(loss_fn)(state)

        step = jax.jit(step_fn)
        x, y = _dummy_batch(cfg, 0)
        x = jax.device_put(x, repl)
        y = jax.device_put(y, repl)
        with mesh:
            t0 = time.perf_counter_ns()
            loss, _ = step(state, x, y)
            jax.block_until_ready(loss)
            compile_ms = (time.perf_counter_ns() - t0) / 1e6

            step_times = []
            for i in range(1, iters + 1):
                x, y = _dummy_batch(cfg, i)
                x = jax.device_put(x, repl)
                y = jax.device_put(y, repl)
                t0 = time.perf_counter_ns()
                loss, _ = step(state, x, y)
                jax.block_until_ready(loss)
                step_times.append((time.perf_counter_ns() - t0) / 1e6)
    return compile_ms, _median(step_times), float(loss)


_RUNNERS = {
    "spmd": run_spmd,
    "spmd_scheduled": run_spmd_scheduled,
    "mpmd": run_mpmd,
    "mpmd_tp": run_mpmd_tp,
    "jit": run_jit,
    "jit_tp": run_jit_tp,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — parse args, dispatch, print a results table."""
    parser = argparse.ArgumentParser(description="Llama 3 8B pipeline benchmark")
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--schedules",
        default="gpipe,1f1b,zb_h1,kimi_k2",
        help="comma list from " + ",".join(_SCHEDULE_MAP),
    )
    parser.add_argument(
        "--modes",
        default="mpmd,spmd,jit",
        help="comma list from {mpmd, spmd, jit}",
    )
    parser.add_argument("--device", default="tpu")
    args = parser.parse_args(argv)

    cfg = Llama3Config(
        batch=args.batch,
        seq_len=args.seq_len,
        n_stages=args.n_stages,
        microbatches=args.microbatches,
    )
    schedules = args.schedules.split(",")
    modes = args.modes.split(",")

    print("spectrax Llama 3 8B benchmark")
    print(f"  device       : {args.device} ({len(jax.devices())} devices)")
    print(f"  total params : ~{cfg.total_params / 1e9:.2f}B (transformer body only)")
    print(f"  n_layers     : {cfg.n_layers} ({cfg.n_layers // cfg.n_stages} per stage x {cfg.n_stages})")
    print(f"  d / heads    : {cfg.d_model} / {cfg.n_heads} (kv {cfg.n_kv_heads})")
    print(f"  ffn (SwiGLU) : {cfg.ffn}")
    print(f"  batch / seq  : {cfg.batch} x {cfg.seq_len}")
    print(f"  microbatches : {cfg.microbatches}")
    print(f"  schedules    : {schedules}")
    print(f"  modes        : {modes}")
    print()

    header = f"{'mode':<13} {'schedule':<22} {'compile_ms':>11} {'step_ms':>10} {'loss':>10}"
    print(header)
    print("-" * len(header))

    for mode in modes:
        runner = _RUNNERS[mode]
        for sched in schedules:
            try:
                if mode in ("jit", "jit_tp") and sched != schedules[0]:
                    continue
                compile_ms, step_ms, loss = runner(cfg, sched, args.iters)
                label = "(n/a)" if mode in ("jit", "jit_tp") else sched
                print(f"{mode:<13} {label:<22} {compile_ms:>11.1f} {step_ms:>10.2f} {loss:>10.4f}")
            except Exception as e:
                print(f"{mode:<13} {sched:<22} ERROR: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
