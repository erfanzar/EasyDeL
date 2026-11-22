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

"""Benchmark sort-free top-k selection against the sort-based paths at indexer shapes.

Cases per ``[rows, width]`` / dtype, all producing ``jax.lax.top_k``'s pick set
(or its ``k``-th value):

* ``lax.top_k + mask``  -- ``jax.lax.top_k`` then the scatter-free mask rebuild
* ``topk values + mask`` -- ejkernel ``topk`` (heuristic: Pallas threshold-and-compact on TPU) then the same rebuild
* ``exact_mask xla``     -- ejkernel ``topk(mode="exact_mask")`` forced to the XLA reference
* ``exact_mask pallas``  -- ejkernel ``topk(mode="exact_mask")`` forced to the Pallas sort-free kernel
* ``kth xla`` / ``kth pallas`` -- ejkernel ``topk(mode="kth")``, the threshold only

Every timed result is checked for equality with the ``lax.top_k`` pick set.
Timing is the steady-state median after warmup; compile is excluded.
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from ejkernel.modules import topk
from ejkernel.modules.operations.configs import TopKConfig

# Explicit configs for every forced case: an explicit cfg is written to the executor's in-memory cache under the
# call key, so a later heuristic call with the same shapes and mode would silently reuse it.
_XLA = TopKConfig(platform="xla", backend="any")
_PALLAS = TopKConfig(platform="pallas", backend="tpu")


def _bench(fn, *args, warmup: int = 2, iters: int = 10) -> float:
    """Median milliseconds per call, blocking on the result."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _mask_from_top_k(scores, values, indices):
    """Scatter-free mask of a ``top_k`` result (keys above the k-th, plus ties up to the last picked index)."""

    def key(x):
        bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.int32)
        return jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits)

    keys, value_keys = key(scores), key(values)
    kth = value_keys[..., -1:]
    cut = jnp.max(jnp.where(value_keys == kth, indices, -1), axis=-1, keepdims=True)
    position = jax.lax.broadcasted_iota(jnp.int32, scores.shape, scores.ndim - 1)
    return (keys > kth) | ((keys == kth) & (position <= cut))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--widths", type=int, nargs="+", default=[8192, 131072])
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    print(f"backend={jax.default_backend()} device={jax.devices()[0].device_kind}")
    k = args.k
    cases = {
        "lax.top_k + mask": lambda a: _mask_from_top_k(a, *jax.lax.top_k(a, k)),
        "topk values + mask": lambda a: _mask_from_top_k(a, *topk(a, k)),
        "exact_mask xla": lambda a: topk(a, k, mode="exact_mask", cfg=_XLA),
        "exact_mask pallas": lambda a: topk(a, k, mode="exact_mask", cfg=_PALLAS),
        "kth xla": lambda a: topk(a, k, mode="kth", cfg=_XLA),
        "kth pallas": lambda a: topk(a, k, mode="kth", cfg=_PALLAS),
    }
    for width in args.widths:
        for dtype in args.dtypes:
            x = jax.random.normal(jax.random.key(width), (args.rows, width), jnp.float32).astype(dtype)
            want = np.asarray(jax.jit(cases["lax.top_k + mask"])(x))
            want_kth = np.asarray(jax.jit(lambda a: jax.lax.top_k(a, k)[0][..., -1])(x).astype(jnp.float32))
            print(f"\n[{args.rows}, {width}] {dtype} k={k}")
            for name, fn in cases.items():
                jfn = jax.jit(fn)
                is_kth = name.startswith("kth")
                got = np.asarray(jfn(x).astype(jnp.float32) if is_kth else jfn(x))
                exact = bool(np.array_equal(got, want_kth if is_kth else want))
                ms = _bench(jfn, x, iters=args.iters)
                print(f"  {name:22s} {ms:9.3f} ms  exact={exact}")
            del x


if __name__ == "__main__":
    main()
