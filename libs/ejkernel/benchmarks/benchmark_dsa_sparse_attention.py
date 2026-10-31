#!/usr/bin/env python3
# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-chip A/B: pool-bitmap block-sparse DSA attention vs the mask-fused ``sparse_selected_attention`` path.

Both attend over the same selection (pools of ``kpool`` keys + tail keys). The existing path consumes decompressed
per-head q / k / v and a dense ``[B, S, S]`` boolean mask (built once outside the timed region); the pool kernel
consumes absorbed-MLA queries ``[B, H, S, R]``, the shared latent ``[B, S, R]`` and the pool selection, and builds
its block tables inside the timed region. Reports steady-state forward and forward+backward milliseconds (median of
``--iters`` after ``--warmup``), plus the active fraction of causal block pairs.

    python libs/ejkernel/benchmarks/benchmark_dsa_sparse_attention.py --seq 8192 --heads 16 --kpool 16
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.dsa_sparse_attention._tables import build_tables, expand_open_mask


def _selection(rng, seq: int, kpool: int, topk: int, window: float):
    """Causal pool selection: each query picks ``topk // kpool`` pools uniformly from the last ``window`` fraction
    of its visible pools, plus the last (at most 8, the kernel's tail-row limit) keys of its trailing incomplete pool
    as tail keys."""
    n_sel = topk // kpool
    pool_mask = np.zeros((1, seq, seq // kpool), bool)
    n_tails = max(1, min(kpool - 1, 8))
    tails = -np.ones((1, seq, n_tails), np.int32)
    for s in range(seq):
        hi = (s + 1) // kpool
        if hi:
            lo = max(0, hi - max(n_sel, int(window * hi)))
            pool_mask[0, s, rng.choice(np.arange(lo, hi), size=min(n_sel, hi - lo), replace=False)] = True
        rem = min((s + 1) % kpool, n_tails)
        tails[0, s, :rem] = np.arange(s + 1 - rem, s + 1)
    return jnp.asarray(pool_mask), jnp.asarray(tails)


def _time(fn, *args, warmup: int, iters: int) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--latent", type=int, default=512, help="kv_lora_rank (absorbed head dim)")
    p.add_argument("--head-dim", type=int, default=256, help="decompressed qk / v head dim (existing path)")
    p.add_argument("--kpool", type=int, default=16)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--window", type=float, default=0.4, help="fraction of visible pools picks are drawn from")
    p.add_argument("--block-q", type=int, default=256)
    p.add_argument("--block-kv", type=int, default=512)
    p.add_argument("--backward", default="fused", choices=("fused", "two_pass"))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    seq, heads = args.seq, args.heads
    pool_mask, tails = _selection(rng, seq, args.kpool, args.topk, args.window)
    _, _, count, _ = build_tables(pool_mask, tails, kpool=args.kpool, block_q=args.block_q, block_kv=args.block_kv)
    n_qb, n_kb = seq // args.block_q, seq // args.block_kv
    causal = sum(min(n_kb, -(-((qb + 1) * args.block_q) // args.block_kv)) for qb in range(n_qb))
    density = float(np.asarray(count).sum()) / causal

    dt = jnp.bfloat16
    key = jax.random.split(jax.random.PRNGKey(0), 6)
    q_abs = jax.random.normal(key[0], (1, heads, seq, args.latent), dt)
    latent = jax.random.normal(key[1], (1, seq, args.latent), dt)
    g_abs = jax.random.normal(key[2], (1, heads, seq, args.latent), dt)
    q = jax.random.normal(key[3], (1, seq, heads, args.head_dim), dt)
    k = jax.random.normal(key[4], (1, seq, heads, args.head_dim), dt)
    v = jax.random.normal(key[5], (1, seq, heads, args.head_dim), dt)
    g = jnp.ones((1, seq, heads, args.head_dim), dt)
    mask = expand_open_mask(pool_mask, tails, kpool=args.kpool, seq_len=seq)

    pool = kernel_registry.get("dsa_sparse_attention", platform=Platform.PALLAS, backend=Backend.TPU)
    old = kernel_registry.get("sparse_selected_attention", platform=Platform.PALLAS, backend=Backend.TPU)
    kw = dict(kpool=args.kpool, block_q=args.block_q, block_kv=args.block_kv, backward=args.backward)
    ste = jnp.zeros(pool_mask.shape, jnp.float32)

    pool_fwd = jax.jit(lambda q, lat: pool(q, lat, pool_mask, tails, **kw))
    pool_fb = jax.jit(
        jax.grad(
            lambda q, lat, s: jnp.sum(pool(q, lat, pool_mask, tails, s, **kw).astype(jnp.float32) * g_abs),
            argnums=(0, 1, 2),
        )
    )
    old_fwd = jax.jit(lambda q, k, v: old(q, k, v, mask, causal=True))
    old_fb = jax.jit(
        jax.grad(lambda q, k, v: jnp.sum(old(q, k, v, mask, causal=True).astype(jnp.float32) * g), argnums=(0, 1, 2))
    )
    w = dict(warmup=args.warmup, iters=args.iters)
    res = {
        "pool fwd": _time(pool_fwd, q_abs, latent, **w),
        "pool fwd+bwd": _time(pool_fb, q_abs, latent, ste, **w),
        "sparse_selected fwd": _time(old_fwd, q, k, v, **w),
        "sparse_selected fwd+bwd": _time(old_fb, q, k, v, **w),
    }
    print(
        f"device={jax.devices()[0].device_kind} S={seq} H={heads} R={args.latent} Dqk=Dv={args.head_dim} "
        f"kpool={args.kpool} topk={args.topk} window={args.window} tiles={args.block_q}x{args.block_kv} "
        f"backward={args.backward} block_density={density:.3f}"
    )
    for name, ms in res.items():
        print(f"  {name:26s} {ms:9.2f} ms")


if __name__ == "__main__":
    main()
