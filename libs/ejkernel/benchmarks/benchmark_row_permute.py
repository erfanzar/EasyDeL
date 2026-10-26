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

"""One-chip A/B: XLA ``x[index]`` gather vs the Pallas TPU streaming run permute.

    python libs/ejkernel/benchmarks/benchmark_row_permute.py --rows 262144 --width 4096 --sources 32 --experts 9

Builds a MoE-style receive buffer (``sources x experts`` source-major / expert-minor segments with skewed sizes, the
valid prefix ``--fill`` of the rows, plus one tail run covering the rest), the expert-major run table and the
equivalent gather index, then times the gather and the kernel (forward, inverse, forward+VJP) and checks bit parity.
One pass moves ``2 x rows x width x itemsize`` bytes. Steady-state timing after one warmup call; compile time is
reported separately.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from ejkernel.kernels._pallas.tpu.row_permute import invert_runs, permute_rows_runs, permute_rows_runs_pallas
from ejkernel.kernels._xla.row_permute import runs_to_index


def segment_runs(rng, rows: int, n_src: int, n_exp: int, skew: float, fill: float):
    """Skewed source-major / expert-minor segments -> expert-major run table (+ tail run) and the valid prefix."""
    w = rng.random((n_src, n_exp)) ** skew + 0.05
    total = int(rows * fill)
    sizes = np.floor(w / w.sum() * total).astype(np.int64)
    sizes[0, 0] += total - sizes.sum()
    recv_starts = np.cumsum(sizes.reshape(-1)) - sizes.reshape(-1)
    sorted_sizes = sizes.T.reshape(-1)
    sorted_starts = np.cumsum(sorted_sizes) - sorted_sizes
    runs = [
        (sorted_starts[e * n_src + s], recv_starts[s * n_exp + e], sorted_sizes[e * n_src + s])
        for e in range(n_exp)
        for s in range(n_src)
    ]
    if total < rows:
        runs.append((total, total, rows - total))  # tail: the rows past the valid prefix stay in place
    return np.asarray(runs, np.int32), total


def _timeit(fn, *args, iters: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return first, (time.perf_counter() - t0) / iters


def main() -> None:
    """Run the A/B and print one line per measurement."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=262144)
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--sources", type=int, default=32)
    ap.add_argument("--experts", type=int, default=9)
    ap.add_argument("--skew", type=float, default=3.0)
    ap.add_argument("--fill", type=float, default=0.8)
    ap.add_argument("--chunk-rows", type=int, nargs="+", default=[256])
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-vjp", action="store_true")
    args = ap.parse_args()

    print("devices:", jax.devices()[0].device_kind, "x", jax.device_count())
    rng = np.random.default_rng(0)
    dt = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    runs_np, valid = segment_runs(rng, args.rows, args.sources, args.experts, args.skew, args.fill)
    lens = runs_np[:, 2]
    print(
        f"shape [{args.rows}, {args.width}] {args.dtype}; runs {len(runs_np)}; valid {valid}; "
        f"run length min/median/max {lens.min()}/{int(np.median(lens))}/{lens.max()}"
    )
    x = jax.random.normal(jax.random.key(0), (args.rows, args.width), jnp.float32).astype(dt)
    runs = jnp.asarray(runs_np)
    inv = invert_runs(runs)
    idx, _ = runs_to_index(runs, args.rows)
    inv_idx, _ = runs_to_index(inv, args.rows)
    bytes_pass = 2 * args.rows * args.width * jnp.dtype(dt).itemsize

    def report(name, first, t):
        print(f"{name:34s}: {1e3 * t:8.3f} ms  ({bytes_pass / t / 1e9:7.1f} GB/s)  first call {first:6.2f} s")

    gather = jax.jit(lambda x, i: jnp.take(x, i, axis=0))
    report("xla gather x[idx]", *_timeit(gather, x, idx, iters=args.iters))
    report("xla gather inverse", *_timeit(gather, x, inv_idx, iters=args.iters))
    want = np.asarray(gather(x, idx))
    for c in args.chunk_rows:
        kern = jax.jit(lambda x, r, c=c: permute_rows_runs_pallas(x, r, chunk_rows=c))
        report(f"pallas permute chunk {c}", *_timeit(kern, x, runs, iters=args.iters))
        report(f"pallas inverse chunk {c}", *_timeit(kern, x, inv, iters=args.iters))
        got = np.asarray(kern(x, runs))
        same = np.array_equal(got.view(np.uint8), want.view(np.uint8))
        print(f"  parity chunk {c}: {'bit-identical' if same else 'MISMATCH'}")
        del got
    if not args.skip_vjp:
        cot = jax.random.normal(jax.random.key(1), (args.rows, args.width), jnp.float32).astype(dt)

        # Every array is an argument (closing over them would embed GiB-sized constants in the program).
        def loss_g(x, idx, cot):
            return jnp.sum((jnp.take(x, idx, axis=0) * cot).astype(jnp.float32))

        def loss_k(x, runs, inv, cot):
            out = permute_rows_runs(x, runs, inv, chunk_rows=args.chunk_rows[0])
            return jnp.sum((out * cot).astype(jnp.float32))

        # value_and_grad keeps the forward live (jax.grad alone lets XLA drop it: the gradient does not need it).
        report("xla gather fwd+VJP", *_timeit(jax.jit(jax.value_and_grad(loss_g)), x, idx, cot, iters=args.iters))
        report("pallas fwd+VJP", *_timeit(jax.jit(jax.value_and_grad(loss_k)), x, runs, inv, cot, iters=args.iters))
        report("xla gather VJP only", *_timeit(jax.jit(jax.grad(loss_g)), x, idx, cot, iters=args.iters))
        report("pallas VJP only", *_timeit(jax.jit(jax.grad(loss_k)), x, runs, inv, cot, iters=args.iters))


if __name__ == "__main__":
    main()
