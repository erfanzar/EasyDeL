# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark Llama 3 8B (bs=64, seq=4k) under SPMD and MPMD via ``spx.run``.

One model class. Two meshes. Forward-only and forward+backward.

Opts into gradient checkpointing bench-side without touching the
main model file: ``spx.remat(Llama3Block)`` returns a Module subclass
whose ``forward`` is checkpointed once, and :class:`RematLlama3`
swaps that in as a drop-in replacement for ``Llama3Block``.

Usage::

    python -m benchmarks.spx_run_8b_4k --modes spmd,mpmd --bs 64 --seq 4096
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import spectrax as spx
from examples.models.llama import FSDP_TP_RULES, Llama3, Llama3Block, Llama3Config
from spectrax import nn
from spectrax.sharding import logical_axis_rules

RematBlock = spx.remat(Llama3Block)


class RematLlama3(Llama3):
    """Llama 3 with checkpointed blocks (instances of ``RematBlock``).

    Skips the parent ``Llama3.__init__`` (which would build vanilla
    ``Llama3Block``) and builds ``RematBlock`` instances directly.
    """

    def __init__(self, cfg: Llama3Config, *, rngs: spx.Rngs):
        """Build embed / remat-wrapped blocks / lm-head submodules."""
        spx.Module.__init__(self)
        from examples.models.llama import Llama3Embed, Llama3LMHead

        self.embed = Llama3Embed(cfg, rngs=rngs)
        self.blocks = nn.ModuleList([RematBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.head = Llama3LMHead(cfg, rngs=rngs)


def cross_entropy(logits, labels):
    """Mean cross-entropy loss between ``logits`` and integer ``labels``."""
    return -(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1])).sum(-1).mean()


def _median(xs):
    """Return the median of ``xs``."""
    return sorted(xs)[len(xs) // 2]


def bench(label, fn, iters):
    """Compile ``fn`` once, then time ``iters`` steady-state calls and print stats."""
    t0 = time.perf_counter_ns()
    out = fn()
    jax.block_until_ready(out if not isinstance(out, tuple) else out[0])
    compile_ms = (time.perf_counter_ns() - t0) / 1e6

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = fn()
        jax.block_until_ready(out if not isinstance(out, tuple) else out[0])
        times.append((time.perf_counter_ns() - t0) / 1e6)
    print(
        f"  {label:30s} compile={compile_ms:8.0f}ms  "
        f"step_ms min/med/max = {min(times):7.1f} / {_median(times):7.1f} / {max(times):7.1f}"
    )


def run_spmd(cfg, ids, labels, iters):
    """Run pure-SPMD forward and fwd+bwd benchmarks across all chips.

    Uses TP=4 (shards heads + ffn). For an 8B model at bs=64 seq=4k,
    plain FSDP doesn't fit per-chip activation memory; TP is required
    to make it run on 4 v5p chips.
    """
    mesh = spx.create_mesh(axis_dims=(1, 1, 1, 1, -1, 1))
    print(f"\n[SPMD] mesh: {dict(mesh.shape)}")
    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = RematLlama3(cfg, rngs=spx.Rngs(0))
        bench(
            "spx.run forward (SPMD)",
            lambda: spx.run(model, inputs=ids, mesh=mesh, mode="forward"),
            iters,
        )
        bench(
            "spx.run train fwd+bwd (SPMD)",
            lambda: spx.run(
                model,
                inputs=ids,
                targets=labels,
                mesh=mesh,
                mode="train",
                loss_fn=cross_entropy,
            ),
            iters,
        )


def run_mpmd(cfg, ids, labels, iters, mb):
    """Run MPMD forward and fwd+bwd benchmarks with pp=2 x tp=2 (4 chips total)."""
    mesh = spx.create_mesh(axis_dims=(2, 1, 1, 1, -1, 1), mpmd_axis="pp")
    print(f"\n[MPMD] mesh: {dict(mesh.shape)}  microbatches={mb}")
    with logical_axis_rules(FSDP_TP_RULES), mesh:
        model = RematLlama3(cfg, rngs=spx.Rngs(0))
        bench(
            "spx.run forward (MPMD)",
            lambda: spx.run(
                model,
                inputs=ids,
                mesh=mesh,
                mode="forward",
                microbatches=mb,
            ),
            iters,
        )
        bench(
            "spx.run train fwd+bwd (MPMD)",
            lambda: spx.run(
                model,
                inputs=ids,
                targets=labels,
                mesh=mesh,
                mode="train",
                loss_fn=cross_entropy,
                microbatches=mb,
            ),
            iters,
        )


def main(argv=None):
    """CLI entry point — parse args, construct inputs, dispatch to SPMD/MPMD runners."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--n-layers", type=int, default=32)
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--modes", default="spmd,mpmd")
    args = ap.parse_args(argv)

    cfg = Llama3Config(
        vocab=args.vocab,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        ffn=14336,
        n_layers=args.n_layers,
        dtype=jnp.bfloat16,
    )
    print(
        f"Llama 3 ~8B: layers={cfg.n_layers} d={cfg.d_model} ffn={cfg.ffn} "
        f"heads={cfg.n_heads}/{cfg.n_kv_heads} dtype={cfg.dtype.__name__ if hasattr(cfg.dtype, '__name__') else cfg.dtype}"
    )
    print(f"workload: bs={args.bs} seq={args.seq} vocab={args.vocab}")
    print(f"devices  : {len(jax.devices())} x {jax.devices()[0].platform}")

    rng = jax.random.PRNGKey(0)
    ids = jax.random.randint(jax.random.fold_in(rng, 1), (args.bs, args.seq), 0, args.vocab)
    labels = jax.random.randint(jax.random.fold_in(rng, 2), (args.bs, args.seq), 0, args.vocab)

    modes = set(args.modes.split(","))
    if "spmd" in modes:
        run_spmd(cfg, ids, labels, args.iters)
    if "mpmd" in modes:
        run_mpmd(cfg, ids, labels, args.iters, args.microbatches)


if __name__ == "__main__":
    raise SystemExit(main())
