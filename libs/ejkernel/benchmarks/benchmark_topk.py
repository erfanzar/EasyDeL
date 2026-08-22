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

"""Benchmark the fused top-k against the paths it would replace.

Shapes are the three real call sites, not synthetic sweeps:

* MoE router      -- ``[tokens, experts]``, tiny static k, run 43x per decode step
* DSA indexer     -- ``[rows, entries]``, large static k
* sampling filter -- ``[reqs, vocab]``, per-row dynamic k, mask not indices

Baselines are what those sites use today: ``jax.lax.top_k`` for the static-k
sites, and EasyDeL's threshold binary search for the sampler. Timing is
steady-state after warmup; compile is measured separately and labelled.
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from ejkernel.modules import topk


def _bench(fn, *args, warmup: int = 3, iters: int = 30) -> float:
    """Return median milliseconds per call, blocking on the result."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _compile_ms(fn, *args) -> float:
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", type=int, default=32, help="decode concurrency")
    ap.add_argument("--vocab", type=int, default=129280)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    print(f"backend={jax.default_backend()} device={jax.devices()[0].device_kind}\n")
    print(f"{'case':38s} {'baseline':>10s} {'topk':>10s} {'speedup':>8s}  {'exact?':>6s}")
    print("-" * 80)

    # ---- MoE router: narrow axis, tiny k, 43 of these per decode step --------
    x = jnp.asarray(rng.normal(size=(args.reqs, 256)), jnp.float32)
    base = _bench(jax.jit(lambda a: jax.lax.top_k(a, 6)), x)
    ours = _bench(jax.jit(lambda a: topk(a, k=6)), x)
    v, i = topk(x, k=6)
    rv, ri = jax.lax.top_k(x, 6)
    ok = bool(jnp.all(v == rv) and jnp.all(i == ri))
    label = f"MoE router [{args.reqs},256] k=6"
    print(f"{label:38s} {base:9.4f}m {ours:9.4f}m {base / ours:7.2f}x  {ok!s:>6s}")

    # ---- DSA indexer: large k ------------------------------------------------
    y = jnp.asarray(rng.normal(size=(args.reqs, 2048)), jnp.float32)
    base = _bench(jax.jit(lambda a: jax.lax.top_k(a, 512)), y)
    ours = _bench(jax.jit(lambda a: topk(a, k=512)), y)
    v, i = topk(y, k=512)
    rv, ri = jax.lax.top_k(y, 512)
    ok = bool(jnp.all(v == rv) and jnp.all(i == ri))
    label = f"DSA indexer [{args.reqs},2048] k=512"
    print(f"{label:38s} {base:9.4f}m {ours:9.4f}m {base / ours:7.2f}x  {ok!s:>6s}")

    # ---- wide axis, small static k: the Pallas regime -------------------------
    z = jnp.asarray(rng.normal(size=(args.reqs, args.vocab)), jnp.float32)
    for k in (8, 20, 50):
        base = _bench(jax.jit(lambda a, k=k: jax.lax.top_k(a, k)), z)
        ours = _bench(jax.jit(lambda a, k=k: topk(a, k=k)), z)
        v, i = topk(z, k=k)
        rv, ri = jax.lax.top_k(z, k)
        ok = bool(jnp.all(v == rv) and jnp.all(i == ri))
        label = f"vocab [{args.reqs},{args.vocab}] k={k}"
        print(f"{label:38s} {base:9.4f}m {ours:9.4f}m {base / ours:7.2f}x  {ok!s:>6s}")

    # ---- sampler: per-row dynamic k, mask ------------------------------------
    try:
        from easydel.inference.esurge.core.binary_search import apply_topk_mask
    except Exception as exc:  # pragma: no cover - easydel may not be importable
        print(f"\n(sampler baseline unavailable: {exc})")
        return

    ks = jnp.asarray(rng.integers(1, 100, size=(args.reqs,)), jnp.int32)
    min_val = float(jnp.finfo(jnp.float32).min)

    def baseline(a, kk):
        return jax.vmap(lambda row, k1: apply_topk_mask(row[None, :], k1, min_val)[0])(a, kk)

    def ours_fn(a, kk):
        return topk(a, kk, mode="filter", mask_fill=min_val)

    b = _bench(jax.jit(baseline), z, ks)
    o = _bench(jax.jit(ours_fn), z, ks)
    kept_b = (np.asarray(jax.jit(baseline)(z, ks)) > min_val).sum(-1)
    kept_o = (np.asarray(jax.jit(ours_fn)(z, ks)) > min_val).sum(-1)
    agree = bool(np.array_equal(kept_b, kept_o))
    label = f"sampler filter [{args.reqs},{args.vocab}] dyn-k"
    print(f"{label:38s} {b:9.4f}m {o:9.4f}m {b / o:7.2f}x  {agree!s:>6s}")
    print(
        f"\ncompile (first call): baseline {_compile_ms(jax.jit(baseline), z, ks):.0f} ms, "
        f"topk {_compile_ms(jax.jit(ours_fn), z, ks):.0f} ms"
    )


if __name__ == "__main__":
    main()
