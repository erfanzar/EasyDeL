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

"""One-chip forward / forward+backward timing of ``indexer_scores`` implementations.

    python libs/ejkernel/benchmarks/benchmark_indexer_scores.py --seq 8192 --heads 64 --dim 128 --entries 2048

Implementations: ``pallas`` (fused Pallas TPU forward + tiled backward kernels) and ``xla`` (the reference, which
materialises the ``[B, H, S, T]`` fp32 logits). Steady-state times exclude compilation, which is reported separately.
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.ops import FwdParams

IMPLS = {"pallas": (Platform.PALLAS, Backend.TPU), "xla": (Platform.XLA, Backend.ANY)}


def make_inputs(batch: int, heads: int, seq: int, entries: int, dim: int, dtype, seed: int = 0):
    """``(query [B,H,S,d], index_k [B,T,d], weights [B,H,S,1] fp32, cotangent [B,S,T] fp32)``."""
    ks = jax.random.split(jax.random.PRNGKey(seed), 4)
    q = jax.random.normal(ks[0], (batch, heads, seq, dim), dtype)
    k = jax.random.normal(ks[1], (batch, entries, dim), dtype)
    w = jax.random.normal(ks[2], (batch, heads, seq, 1), jnp.float32)
    cot = jax.random.normal(ks[3], (batch, seq, entries), jnp.float32)
    return q, k, w, cot


def timeit(fn, *args, iters: int) -> tuple[float, float]:
    """Return ``(first_call_s, steady_state_s_per_call)``."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return first, (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--entries", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--precision", default="default")
    ap.add_argument("--block-q", type=int, default=128)
    ap.add_argument("--block-t", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--impls", default="pallas,xla")
    args = ap.parse_args()

    print("devices:", jax.devices()[:1], "shape:", vars(args), flush=True)
    q, k, w, cot = make_inputs(args.batch, args.heads, args.seq, args.entries, args.dim, jnp.dtype(args.dtype))
    fwd_params = FwdParams(q_blocksize=args.block_q, kv_blocksize=args.block_t)
    for name in args.impls.split(","):
        platform, backend = IMPLS[name]
        impl = kernel_registry.get("indexer_scores", platform=platform, backend=backend)
        fn = functools.partial(impl, fwd_params=fwd_params, precision=args.precision)

        def loss(q, k, w, fn=fn):
            return jnp.sum(fn(q, k, w) * cot)

        try:
            c_f, t_f = timeit(jax.jit(fn), q, k, w, iters=args.iters)
            c_b, t_b = timeit(jax.jit(jax.grad(loss, argnums=(0, 1, 2))), q, k, w, iters=args.iters)
        except Exception as err:  # e.g. an HBM OOM of the materialising reference
            print(f"{name:>8}: failed: {type(err).__name__}: {str(err).splitlines()[0][:200]}", flush=True)
            continue
        print(
            f"{name:>8}: fwd {1e3 * t_f:8.2f} ms (first call {c_f:6.1f} s) | "
            f"fwd+bwd {1e3 * t_b:8.2f} ms (first call {c_b:6.1f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
