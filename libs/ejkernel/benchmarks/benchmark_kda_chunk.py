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

"""One-chip forward / forward+backward timing of ``kda_chunk`` implementations (packed rows by default).

    python libs/ejkernel/benchmarks/benchmark_kda_chunk.py --seq 32768 --heads 16 --dim 128 --iters 3

Implementations: ``pallas`` (Pallas forward + Pallas backward), ``pallas-xlabwd`` (Pallas forward + reversed-scan
XLA backward), ``pallas-mxu`` (Pallas backward with MXU reductions), ``xla`` (rematerialised XLA scan). Steady-state
times exclude compilation, which is reported separately.
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from ejkernel.kernels import Backend, Platform, kernel_registry

IMPLS = {
    "pallas": (Platform.PALLAS, Backend.TPU, dict(backward="pallas", bwd_reduce="vpu")),
    "pallas-mxu": (Platform.PALLAS, Backend.TPU, dict(backward="pallas", bwd_reduce="mxu")),
    "pallas-xlabwd": (Platform.PALLAS, Backend.TPU, dict(backward="xla")),
    "xla": (Platform.XLA, Backend.ANY, {}),
}


def make_inputs(batch: int, heads: int, seq: int, dim: int, doc_len: int, seed: int = 0):
    """Random KDA inputs ``[B, H, L, D]`` and packed ids with ``doc_len``-token documents (0 = unsegmented)."""
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(ks[0], (batch, heads, seq, dim))
    k = jax.random.normal(ks[1], (batch, heads, seq, dim))
    v = jax.random.normal(ks[2], (batch, heads, seq, dim))
    beta = jax.nn.sigmoid(jax.random.normal(ks[3], (batch, heads, seq)))
    g = -jax.nn.softplus(jax.random.normal(ks[4], (batch, heads, seq, dim)))
    seg = None
    if doc_len > 0:
        seg = jnp.asarray(np.broadcast_to(np.arange(seq) // doc_len, (batch, seq)).astype(np.int32))
    return q, k, v, beta, g, seg


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
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--doc-len", type=int, default=1500)
    ap.add_argument("--block-h", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--impls", default="pallas,pallas-xlabwd,xla")
    args = ap.parse_args()

    print("devices:", jax.devices()[:1], "shape:", vars(args))
    q, k, v, beta, g, seg = make_inputs(args.batch, args.heads, args.seq, args.dim, args.doc_len)
    w = jax.random.normal(jax.random.PRNGKey(1), (*q.shape[:3], args.dim))
    for name in args.impls.split(","):
        platform, backend, kw = IMPLS[name]
        impl = kernel_registry.get("kda_chunk", platform=platform, backend=backend)
        if platform == Platform.PALLAS:
            kw = dict(kw, block_h=args.block_h)
        fn = functools.partial(impl, chunk_size=args.chunk, **kw)

        def fwd(q, k, v, beta, g, fn=fn):
            return fn(q, k, v, beta, g, None, seg)

        def loss(q, k, v, beta, g, fn=fn):
            out, fin = fn(q, k, v, beta, g, None, seg)
            return jnp.sum(out * w) + jnp.sum(fin)

        c_f, t_f = timeit(jax.jit(fwd), q, k, v, beta, g, iters=args.iters)
        c_b, t_b = timeit(jax.jit(jax.grad(loss, argnums=range(5))), q, k, v, beta, g, iters=args.iters)
        print(
            f"{name:>14}: fwd {1e3 * t_f:8.1f} ms (first call {c_f:6.1f} s) | "
            f"fwd+bwd {1e3 * t_b:8.1f} ms (first call {c_b:6.1f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
