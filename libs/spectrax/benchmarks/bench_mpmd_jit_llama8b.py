#!/usr/bin/env python3
"""Llama 3 8B benchmark: sxjit (transparent grad) vs SPMD vs JIT.

Benchmarks every scheduler under sxjit, plus SPMD (pipeline_call)
and plain jax.jit with TP=4. Uses proper sharding for each path.

sxjit receives a flat parameter pytree to work around custom_vjp
limitations with Module objects as direct positional args.

Usage::

    # TPU (uses real devices)
    uv run python -m benchmarks.bench_mpmd_jit_llama8b --iters 5
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import spectrax as spx
from spectrax import nn
from spectrax.nn import PipelineSequential
from spectrax.runtime.mpmd import sxcall, sxjit, sxstage_iter
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
from spectrax.runtime.spmd.api import pipeline_call
from spectrax.runtime.types import MpMdMesh
from spectrax.runtime.types.stage import PipelineStage
from spectrax.sharding import logical_axis_rules, with_sharding_constraint_by_name


def _null_ctx():
    """Return a no-op context manager."""
    return contextlib.nullcontext()


class Llama3Config:
    """Llama 3 8B transformer body."""

    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn: int = 14336
    batch: int = 4
    seq_len: int = 128
    n_stages: int = 4
    microbatches: int = 4

    def __init__(self, **kwargs):
        """Override any default attribute via keyword arguments."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def total_params(self) -> int:
        """Approximate total parameter count across all layers."""
        attn = 2 * self.d_model * self.d_model + 2 * self.d_model * (self.d_model * self.n_kv_heads // self.n_heads)
        ffn = 3 * self.d_model * self.ffn
        return self.n_layers * (attn + ffn)


def _rope_freqs(seq_len: int, head_dim: int, theta: float = 500_000.0):
    """Precompute RoPE cos/sin tables of shape ``(seq_len, head_dim // 2)``.

    Args:
        seq_len: Sequence length.
        head_dim: Per-head dimension.
        theta: RoPE base frequency.

    Returns:
        ``(cos, sin)`` tuple of bf16 arrays.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv_freq)
    return jnp.cos(freqs).astype(jnp.bfloat16), jnp.sin(freqs).astype(jnp.bfloat16)


def _apply_rope(x, cos, sin):
    """Apply rotary embeddings to ``x`` using ``cos`` and ``sin`` tables.

    Args:
        x: Tensor of shape ``(..., seq, heads, head_dim)``.
        cos: RoPE cosine table.
        sin: RoPE sine table.

    Returns:
        Rotated tensor of same shape as ``x``.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class Llama3Block(spx.Module):
    """One Llama 3 transformer block: RMSNorm + GQA attn + RMSNorm + SwiGLU FFN."""

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


class Llama3BlockTP(spx.Module):
    """Llama 3 block with role-specific logical sharding annotations for TP."""

    def __init__(self, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize TP-aware GQA attention + SwiGLU FFN sublayers.

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
        """Run the TP-aware attention + SwiGLU FFN with activation sharding pins.

        Args:
            x: Input tensor ``(b, t, d)``.
            cos: RoPE cosine table.
            sin: RoPE sine table.

        Returns:
            Output tensor ``(b, t, d)``.
        """
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


class Llama3Stage(spx.Module):
    """A pipeline stage = ``n_blocks`` Llama 3 blocks in sequence."""

    def __init__(self, n_blocks, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize ``n_blocks`` :class:`Llama3Block` instances in a Sequential.

        Args:
            n_blocks: Number of blocks in this stage.
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads.
            rngs: PRNG source for parameter init.
        """
        super().__init__()
        self.blocks = nn.Sequential(*[Llama3Block(d, ffn, n_heads, n_kv_heads, rngs=rngs) for _ in range(n_blocks)])

    def forward(self, x):
        """Run every block with RoPE tables recomputed at stage entry.

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


class Llama3StageTP(spx.Module):
    """A pipeline stage of TP-aware Llama 3 blocks (RoPE recomputed inside)."""

    def __init__(self, n_blocks, d, ffn, n_heads, n_kv_heads, *, rngs):
        """Initialize ``n_blocks`` :class:`Llama3BlockTP` instances in a Sequential.

        Args:
            n_blocks: Number of blocks in this stage.
            d: Model dimension.
            ffn: FFN hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads.
            rngs: PRNG source for parameter init.
        """
        super().__init__()
        self.blocks = nn.Sequential(*[Llama3BlockTP(d, ffn, n_heads, n_kv_heads, rngs=rngs) for _ in range(n_blocks)])

    def forward(self, x):
        """Run every TP-aware block with RoPE tables recomputed at stage entry.

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


def _dummy_batch(cfg: Llama3Config, step: int):
    """Return a deterministic ``(x, y)`` bf16 batch for ``step``.

    Args:
        cfg: Model configuration.
        step: Step index used to fold the PRNG key.

    Returns:
        ``(x, y)`` tuple of random bf16 tensors.
    """
    key = jax.random.fold_in(jax.random.PRNGKey(0), step)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=jnp.bfloat16)
    y = jax.random.normal(ky, (cfg.batch, cfg.seq_len, cfg.d_model), dtype=jnp.bfloat16)
    return x, y


def _loss_fn(out, y):
    """MSE loss between ``out`` and ``y`` computed in float32.

    Args:
        out: Model output tensor.
        y: Target tensor.

    Returns:
        Scalar float32 MSE.
    """
    return ((out.astype(jnp.float32) - y.astype(jnp.float32)) ** 2).mean()


def _median(xs: list[float]) -> float:
    """Return the median of ``xs``.

    Args:
        xs: List of float timings.

    Returns:
        Median value.
    """
    return sorted(xs)[len(xs) // 2]


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
    """Instantiate the schedule class for ``name``.

    Args:
        name: Schedule key (e.g. ``"gpipe"``, ``"1f1b"``).
        microbatches: Number of microbatches.

    Returns:
        Schedule instance.
    """
    cls = _SCHEDULE_MAP[name]
    if name in _VIRTUAL_NAMES:
        return cls(microbatches=microbatches, virtual_stages=2)
    return cls(microbatches=microbatches)


def build_stages(cfg: Llama3Config, pp: int, virtual_stages: int = 1, tp: bool = False):
    """Build pipeline stages and return (stages_list, gdefs, state_structs, flat_params).

    Args:
        cfg: Model configuration.
        pp: Pipeline parallelism degree.
        virtual_stages: Virtual stages per rank.
        tp: Whether to use tensor-parallel block variants.

    Returns:
        Tuple of ``(stages, gdefs, state_structs, flat_params)``.
    """
    rngs = spx.Rngs(0)
    n_logical = virtual_stages * pp
    if cfg.n_layers % n_logical:
        raise ValueError(f"n_layers={cfg.n_layers} not divisible by n_logical={n_logical}")
    blocks_per_logical = cfg.n_layers // n_logical
    StageCls = Llama3StageTP if tp else Llama3Stage
    stages = [
        StageCls(blocks_per_logical, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)
        for _ in range(n_logical)
    ]
    gdefs = []
    state_structs = []
    flat_params = []
    for s in stages:
        g, st = spx.export(s)
        gdefs.append(g)
        state_structs.append(st)
        flat_params.extend(jax.tree.leaves(st))
    return stages, gdefs, state_structs, flat_params


def build_full_model(cfg: Llama3Config, tp: bool = False):
    """Build a single stage with all layers (for the JIT runtime).

    Args:
        cfg: Model configuration.
        tp: Whether to use tensor-parallel block variants.

    Returns:
        A :class:`Llama3Stage` or :class:`Llama3StageTP` with all blocks.
    """
    rngs = spx.Rngs(0)
    StageCls = Llama3StageTP if tp else Llama3Stage
    return StageCls(cfg.n_layers, cfg.d_model, cfg.ffn, cfg.n_heads, cfg.n_kv_heads, rngs=rngs)


def bench_mpmd_jit(cfg: Llama3Config, schedule_name: str, pp: int, tp: int, iters: int):
    """sxjit with transparent jax.grad using flat params."""
    schedule = _make_schedule(schedule_name, cfg.microbatches)
    schedule.virtual_stages_per_rank() * pp
    devices = np.array(jax.devices()[: pp * tp]).reshape(pp, tp)
    mesh = Mesh(devices, axis_names=("pp", "tp"))
    mpmd_mesh = MpMdMesh(mesh, "pp")

    _stages, gdefs, state_structs, flat_params = build_stages(cfg, pp, schedule.virtual_stages_per_rank(), tp=tp > 1)
    flat_params = [jnp.asarray(p) for p in flat_params]
    jax.tree.structure(tuple(flat_params))

    @sxjit(mesh=mpmd_mesh, schedule=schedule)
    def forward(params, x, y):
        """MPMD forward with flat params: run stages, compute MSE loss.

        Args:
            params: Flat tuple of parameter arrays.
            x: Input tensor.
            y: Target tensor.

        Returns:
            Scalar loss.
        """
        with mesh if tp > 1 else _null_ctx():
            flat_list = list(params)
            idx = 0
            h = x
            for i, (g, st_struct) in enumerate(zip(gdefs, state_structs, strict=False)):
                n_leaves = len(jax.tree.leaves(st_struct))
                stage_state = jax.tree.unflatten(jax.tree.structure(st_struct), flat_list[idx : idx + n_leaves])
                idx += n_leaves
                stage = spx.bind(g, stage_state)
                h = stage(h)
                if i < len(gdefs) - 1:
                    h = sxstage_iter(h)
            return _loss_fn(h, y)

    x, y = _dummy_batch(cfg, 0)
    args = (tuple(flat_params), x, y)

    t0 = time.perf_counter_ns()
    loss = forward(*args)
    jax.block_until_ready(loss)
    fwd_compile_ms = (time.perf_counter_ns() - t0) / 1e6

    fwd_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        args = (tuple(flat_params), x, y)
        t0 = time.perf_counter_ns()
        loss = forward(*args)
        jax.block_until_ready(loss)
        fwd_times.append((time.perf_counter_ns() - t0) / 1e6)

    grad_fn = jax.jit(jax.grad(forward, argnums=0))
    t0 = time.perf_counter_ns()
    grads = grad_fn(*args)
    jax.block_until_ready(grads[0] if isinstance(grads, tuple) else grads)
    grad_compile_ms = (time.perf_counter_ns() - t0) / 1e6

    grad_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        args = (tuple(flat_params), x, y)
        t0 = time.perf_counter_ns()
        grads = grad_fn(*args)
        jax.block_until_ready(grads[0] if isinstance(grads, tuple) else grads)
        grad_times.append((time.perf_counter_ns() - t0) / 1e6)

    return fwd_compile_ms, _median(fwd_times), grad_compile_ms, _median(grad_times), float(loss)


def bench_spmd(cfg: Llama3Config, schedule_name: str, iters: int):
    """SPMD pipeline_call (single-HLO shard_map path)."""
    schedule = _make_schedule(schedule_name, cfg.microbatches)
    devices = jax.devices()[: cfg.n_stages]
    mpmd_mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")

    _stages, gdefs, state_structs, _ = build_stages(cfg, cfg.n_stages, schedule.virtual_stages_per_rank(), tp=False)

    stage_fns = []
    stage_params = []
    for g, st in zip(gdefs, state_structs, strict=False):
        stage_fns.append(lambda p, s, x, _g=g, _s=st: (spx.bind(_g, _s)(x), s))
        stage_params.append(st)

    stages_wrapped = tuple(
        PipelineStage(fn=sf, params=sp, init_state=()) for sf, sp in zip(stage_fns, stage_params, strict=False)
    )

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    loss, _ = pipeline_call(
        stages_wrapped,
        (x, y),
        mesh=mpmd_mesh,
        microbatches=cfg.microbatches,
        mode="train",
        loss_fn=_loss_fn,
        schedule=schedule,
    )
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        loss, _ = pipeline_call(
            stages_wrapped,
            (x, y),
            mesh=mpmd_mesh,
            microbatches=cfg.microbatches,
            mode="train",
            loss_fn=_loss_fn,
            schedule=schedule,
        )
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def bench_mpmd_call(cfg: Llama3Config, schedule_name: str, pp: int, tp: int, iters: int):
    """MPMD pipeline via sxcall (explicit loss_fn)."""
    schedule = _make_schedule(schedule_name, cfg.microbatches)
    schedule.virtual_stages_per_rank() * pp

    devices = np.array(jax.devices()[: pp * tp]).reshape(pp, tp)
    mesh = Mesh(devices, axis_names=("pp", "tp"))
    mpmd_mesh = MpMdMesh(mesh, "pp")
    rules = [("tp_head", "tp"), ("tp_ffn", "tp")]

    stages, _gdefs, _state_structs, _ = build_stages(cfg, pp, schedule.virtual_stages_per_rank(), tp=tp > 1)
    model = PipelineSequential(*stages)

    x, y = _dummy_batch(cfg, 0)
    t0 = time.perf_counter_ns()
    if tp > 1:
        with logical_axis_rules(rules):
            loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
    else:
        loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(cfg, i)
        t0 = time.perf_counter_ns()
        if tp > 1:
            with logical_axis_rules(rules):
                loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
        else:
            loss, _ = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=_loss_fn)
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def bench_jit_tp(cfg: Llama3Config, iters: int):
    """Plain jax.jit with TP=4 sharding."""
    devices = jax.devices()[: cfg.n_stages]
    mesh = Mesh(devices, axis_names=("tp",))
    rules = [("tp_head", "tp"), ("tp_ffn", "tp")]
    repl = NamedSharding(mesh, PartitionSpec())

    with logical_axis_rules(rules):
        model = build_full_model(cfg, tp=True)
        gdef, state = spx.export(model)

    def step_fn(state, x, y):
        """Jitted step: bind, compute loss, and return ``(loss, grads)``."""
        with mesh, logical_axis_rules(rules):

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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — parse args, dispatch benchmarks, print table."""
    parser = argparse.ArgumentParser(description="Llama 3 8B: sxjit vs SPMD vs JIT")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--schedules",
        default="gpipe,1f1b,zb_h1",
        help="comma list from " + ",".join(_SCHEDULE_MAP),
    )
    parser.add_argument(
        "--modes",
        default="mpmd_jit_pp4tp1,mpmd_jit_pp2tp2,spmd,mpmd_call_pp4tp1,mpmd_call_pp2tp2,jit_tp",
        help="comma list of benchmark modes",
    )
    args = parser.parse_args(argv)

    n_dev = len(jax.devices())
    cfg = Llama3Config(batch=args.batch, seq_len=args.seq_len, microbatches=args.microbatches)
    schedules = args.schedules.split(",")
    modes = args.modes.split(",")

    print("=" * 90)
    print("Llama 3 8B Benchmark: sxjit (transparent grad) vs SPMD vs JIT")
    print("=" * 90)
    print(f"  backend      : {jax.default_backend()} ({n_dev} devices)")
    print(f"  params       : ~{cfg.total_params / 1e9:.2f}B")
    print(f"  layers       : {cfg.n_layers}")
    print(f"  d_model      : {cfg.d_model}")
    print(f"  batch x seq  : {cfg.batch} x {cfg.seq_len}")
    print(f"  microbatches : {cfg.microbatches}")
    print(f"  schedules    : {schedules}")
    print(f"  modes        : {modes}")
    print(f"  iters        : {args.iters}")
    print()

    header = (
        f"{'mode':<22} {'schedule':<18} {'fwd_c_ms':>10} {'fwd_ms':>8} {'grad_c_ms':>10} {'grad_ms':>8} {'loss':>10}"
    )
    print(header)
    print("-" * len(header))

    for mode in modes:
        if mode == "mpmd_jit_pp4tp1":
            for sched in schedules:
                try:
                    fc, fs, gc, gs, loss = bench_mpmd_jit(cfg, sched, pp=4, tp=1, iters=args.iters)
                    print(
                        f"{'sxjit(pp=4,tp=1)':<22} {sched:<18} {fc:>10.1f} {fs:>8.2f} {gc:>10.1f} {gs:>8.2f} {loss:>10.4f}"
                    )
                except Exception as e:
                    print(f"{'sxjit(pp=4,tp=1)':<22} {sched:<18} ERROR: {type(e).__name__}: {e}")

        elif mode == "mpmd_jit_pp2tp2":
            for sched in schedules:
                try:
                    fc, fs, gc, gs, loss = bench_mpmd_jit(cfg, sched, pp=2, tp=2, iters=args.iters)
                    print(
                        f"{'sxjit(pp=2,tp=2)':<22} {sched:<18} {fc:>10.1f} {fs:>8.2f} {gc:>10.1f} {gs:>8.2f} {loss:>10.4f}"
                    )
                except Exception as e:
                    print(f"{'sxjit(pp=2,tp=2)':<22} {sched:<18} ERROR: {type(e).__name__}: {e}")

        elif mode == "mpmd_call_pp4tp1":
            for sched in schedules:
                try:
                    c, s, loss = bench_mpmd_call(cfg, sched, pp=4, tp=1, iters=args.iters)
                    print(f"{'sxcall(pp=4,tp=1)':<22} {sched:<18} {c:>10.1f} {s:>8.2f} {'-':>10} {'-':>8} {loss:>10.4f}")
                except Exception as e:
                    print(f"{'sxcall(pp=4,tp=1)':<22} {sched:<18} ERROR: {type(e).__name__}: {e}")

        elif mode == "mpmd_call_pp2tp2":
            for sched in schedules:
                try:
                    c, s, loss = bench_mpmd_call(cfg, sched, pp=2, tp=2, iters=args.iters)
                    print(f"{'sxcall(pp=2,tp=2)':<22} {sched:<18} {c:>10.1f} {s:>8.2f} {'-':>10} {'-':>8} {loss:>10.4f}")
                except Exception as e:
                    print(f"{'sxcall(pp=2,tp=2)':<22} {sched:<18} ERROR: {type(e).__name__}: {e}")

        elif mode == "spmd":
            for sched in schedules:
                try:
                    c, s, loss = bench_spmd(cfg, sched, args.iters)
                    print(
                        f"{'spmd(pipeline_call)':<22} {sched:<18} {c:>10.1f} {s:>8.2f} {'-':>10} {'-':>8} {loss:>10.4f}"
                    )
                except Exception as e:
                    print(f"{'spmd(pipeline_call)':<22} {sched:<18} ERROR: {type(e).__name__}: {e}")

        elif mode == "jit_tp":
            try:
                c, s, la = bench_jit_tp(cfg, args.iters)
                print(f"{'jax.jit(tp=4)':<22} {'-':<18} {c:>10.1f} {s:>8.2f} {'-':>10} {'-':>8} {la:>10.4f}")
            except Exception as e:
                print(f"{'jax.jit(tp=4)':<22} {'-':<18} ERROR: {type(e).__name__}: {e}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
