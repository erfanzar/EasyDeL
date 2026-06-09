#!/usr/bin/env python3
"""Benchmark sxjit (transparent jax.grad) against SPMD pipeline_call.

This uses a simple MLP so both paths run the same computation without
Module-pytree complications in the custom_vjp backward rule.

Usage::

    export JAX_PLATFORMS=cpu
    export XLA_FLAGS='--xla_force_host_platform_device_count=4'
    uv run python -m benchmarks.bench_mpmd_jit_vs_spmd
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from spectrax.runtime.mpmd import sxjit, sxstage_iter
from spectrax.runtime.schedules import GPipe, Std1F1B, ZeroBubbleH1
from spectrax.runtime.spmd.api import pipeline_call
from spectrax.runtime.types import MpMdMesh
from spectrax.runtime.types.stage import PipelineStage


def _median(xs: list[float]) -> float:
    """Return the median of ``xs``.

    Args:
        xs: List of float timings.

    Returns:
        Median value.
    """
    s = sorted(xs)
    return s[len(s) // 2]


def _dummy_batch(batch: int, d: int, step: int):
    """Return a deterministic ``(x, y)`` float32 batch for ``step``.

    Args:
        batch: Batch size.
        d: Feature dimension.
        step: Step index used to fold the PRNG key.

    Returns:
        ``(x, y)`` tuple of random float32 tensors.
    """
    key = jax.random.fold_in(jax.random.PRNGKey(0), step)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (batch, d), dtype=jnp.float32)
    y = jax.random.normal(ky, (batch, d), dtype=jnp.float32)
    return x, y


def build_params(n_stages: int, d: int, key: jax.Array):
    """Return a flat tuple of (W, b) pairs — one per stage.

    Args:
        n_stages: Number of pipeline stages.
        d: Feature dimension.
        key: JAX PRNG key.

    Returns:
        Flat tuple of weight and bias arrays.
    """
    params = []
    for _i in range(n_stages):
        k1, k2, key = jax.random.split(key, 3)
        params.append(jax.random.normal(k1, (d, d)) * 0.01)
        params.append(jax.random.normal(k2, (d,)) * 0.01)
    return tuple(params)


def mlp_forward(*params_and_data):
    """Reference forward: params = (W0,b0,W1,b1,...) then x, y.

    Args:
        *params_and_data: Flat parameter sequence followed by ``x`` and ``y``.

    Returns:
        Scalar MSE loss.
    """
    *params, x, y = params_and_data
    h = x
    for i in range(0, len(params), 2):
        w, b = params[i], params[i + 1]
        h = jnp.maximum(h @ w + b, 0)
    return jnp.mean((h - y) ** 2)


def bench_mpmd_jit(params, d: int, n_stages: int, batch: int, mb: int, schedule, iters: int):
    """sxjit with transparent jax.grad (custom_vjp schedule path)."""
    devices = jax.devices()[:n_stages]
    mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")

    @sxjit(mesh=mesh, schedule=schedule)
    def forward(*args):
        """MPMD forward: apply MLP stages, compute MSE loss.

        Args:
            *args: Flat parameter sequence followed by ``x`` and ``y``.

        Returns:
            Scalar MSE loss.
        """
        *stage_params, x, y = args
        h = x
        for i in range(0, len(stage_params), 2):
            w, b = stage_params[i], stage_params[i + 1]
            h = jnp.maximum(h @ w + b, 0)
            if i < len(stage_params) - 2:
                h = sxstage_iter(h)
        return jnp.mean((h - y) ** 2)

    x, y = _dummy_batch(batch, d, 0)
    args = (*params, x, y)

    t0 = time.perf_counter_ns()
    loss = forward(*args)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    grad_fn = jax.grad(forward)
    step_times: list[float] = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(batch, d, i)
        args = (*params, x, y)
        t0 = time.perf_counter_ns()
        grads = grad_fn(*args)
        jax.block_until_ready(grads[0])
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def bench_spmd(params, d: int, n_stages: int, batch: int, mb: int, schedule, iters: int):
    """SPMD pipeline_call (shard_map + switch inside single HLO)."""
    devices = jax.devices()[:n_stages]
    mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")

    def make_stage_fn(w, b):
        """Create a stage function for SPMD pipeline_call."""

        def stage_fn(params, state, x):
            """One pipeline stage: ReLU linear transform.

            Args:
                params: ``(w, b)`` tuple.
                state: Unused stage state.
                x: Input tensor.

            Returns:
                ``(output, state)`` tuple.
            """
            w, b = params
            return jnp.maximum(x @ w + b, 0), state

        return stage_fn

    stage_fns = tuple(make_stage_fn(params[i], params[i + 1]) for i in range(0, len(params), 2))
    stage_params = tuple((params[i], params[i + 1]) for i in range(0, len(params), 2))
    stages = tuple(
        PipelineStage(fn=sf, params=sp, init_state=()) for sf, sp in zip(stage_fns, stage_params, strict=False)
    )

    def loss_fn(out, target):
        """MSE loss for the final stage output.

        Args:
            out: Model output tensor.
            target: Target tensor.

        Returns:
            Scalar MSE.
        """
        return jnp.mean((out - target) ** 2)

    x, y = _dummy_batch(batch, d, 0)

    t0 = time.perf_counter_ns()
    loss, _grads = pipeline_call(
        stages,
        (x, y),
        mesh=mesh,
        microbatches=mb,
        mode="train",
        loss_fn=loss_fn,
        schedule=schedule,
    )
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    step_times: list[float] = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(batch, d, i)
        t0 = time.perf_counter_ns()
        loss, _grads = pipeline_call(
            stages,
            (x, y),
            mesh=mesh,
            microbatches=mb,
            mode="train",
            loss_fn=loss_fn,
            schedule=schedule,
        )
        jax.block_until_ready(loss)
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def bench_single_device(params, d: int, batch: int, iters: int):
    """Single-device reference (no pipeline parallelism)."""
    x, y = _dummy_batch(batch, d, 0)
    args = (*params, x, y)

    t0 = time.perf_counter_ns()
    loss = mlp_forward(*args)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    grad_fn = jax.grad(mlp_forward)
    step_times: list[float] = []
    for i in range(1, iters + 1):
        x, y = _dummy_batch(batch, d, i)
        args = (*params, x, y)
        t0 = time.perf_counter_ns()
        grads = grad_fn(*args)
        jax.block_until_ready(grads[0])
        step_times.append((time.perf_counter_ns() - t0) / 1e6)

    return compile_ms, _median(step_times), float(loss)


def main(argv: list[str] | None = None):
    """CLI entry point — parse args, run benchmarks, print table."""
    parser = argparse.ArgumentParser(description="sxjit vs SPMD benchmark")
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--schedules", default="gpipe,1f1b", help="schedules for sxjit")
    args = parser.parse_args(argv)

    if len(jax.devices()) < args.n_stages:
        print(
            f"WARNING: need {args.n_stages} devices, have {len(jax.devices())}. "
            f"Re-run with XLA_FLAGS='--xla_force_host_platform_device_count={args.n_stages}'"
        )
        return 1

    key = jax.random.PRNGKey(0)
    params = build_params(args.n_stages, args.d_model, key)
    n_params = sum(p.size for p in params)

    print("sxjit (transparent grad)  vs  SPMD (pipeline_call)  vs  Single-device")
    print(f"  devices      : {len(jax.devices())}")
    print(f"  stages       : {args.n_stages}")
    print(f"  d_model      : {args.d_model}")
    print(f"  batch        : {args.batch}")
    print(f"  microbatches : {args.microbatches}")
    print(f"  params       : {n_params / 1e6:.2f}M")
    print(f"  iters        : {args.iters}")
    print()

    header = f"{'runner':<22} {'schedule':<10} {'compile_ms':>11} {'step_ms':>10} {'loss':>10}"
    print(header)
    print("-" * len(header))

    c_ms, s_ms, loss = bench_single_device(params, args.d_model, args.batch, args.iters)
    print(f"{'single-device':<22} {'-':<10} {c_ms:>11.1f} {s_ms:>10.2f} {loss:>10.4f}")

    sched_map = {"gpipe": GPipe, "1f1b": Std1F1B, "zb_h1": ZeroBubbleH1}
    for name in args.schedules.split(","):
        sched_cls = sched_map.get(name.strip())
        if sched_cls is None:
            print(f"{'sxjit':<22} {name:<10} UNKNOWN SCHEDULE")
            continue
        sched = sched_cls(microbatches=args.microbatches)
        try:
            c_ms, s_ms, loss = bench_mpmd_jit(
                params, args.d_model, args.n_stages, args.batch, args.microbatches, sched, args.iters
            )
            print(f"{'sxjit':<22} {name:<10} {c_ms:>11.1f} {s_ms:>10.2f} {loss:>10.4f}")
        except Exception as e:
            print(f"{'sxjit':<22} {name:<10} ERROR: {type(e).__name__}: {e}")

    try:
        c_ms, s_ms, loss = bench_spmd(
            params,
            args.d_model,
            args.n_stages,
            args.batch,
            args.microbatches,
            GPipe(microbatches=args.microbatches),
            args.iters,
        )
        print(f"{'spmd (pipeline_call)':<22} {'gpipe':<10} {c_ms:>11.1f} {s_ms:>10.2f} {loss:>10.4f}")
    except Exception as e:
        print(f"{'spmd (pipeline_call)':<22} {'gpipe':<10} ERROR: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
