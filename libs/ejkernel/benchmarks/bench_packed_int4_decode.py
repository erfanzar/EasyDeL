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

"""Decode speedup gate for the packed-int4 gemv (autoresearch verify).

Compares three paths at model-scale decode shapes:

* bf16 ``x @ w`` — the baseline every speedup is measured against;
* ``channelwise_quantized_matmul`` — the XLA fused-upcast path, whose int4
  arrays XLA stores unpacked (byte per element), capping it near 2x;
* ``packed_int4_gemv`` — the Pallas kernel streaming two weights per byte.

``MIN_SPEEDUP=`` is the minimum over the packed-kernel cells. Exits non-zero
if any packed cell's relerr vs the f32 dequantized reference exceeds
``MAX_RELERR`` — this path adds no quantization beyond the codes themselves,
so anything past bf16 rounding is a kernel bug.
"""

import statistics
import sys
import time

import jax
import jax.numpy as jnp
from ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv import (
    pack_int4_adjacent,
    pack_int4_split_k,
    packed_int4_gemv,
    w4a4_gemv,
)
from ejkernel.kernels._xla.quantized_matmul import channelwise_quantized_matmul

SHAPES = [
    ("13B gate_up", 5120, 27648),
    ("27B gate_up", 8192, 57344),
]
TOKENS = (8, 32)
MAX_RELERR = 0.03
WARMUP = 3
ITERS = 20

#: (tile_n, chunk) per shape name — the experiment-tuned table.
CONFIG = {
    "13B gate_up": (1024, 512),
    "27B gate_up": (1024, 512),
}


def timed(fn, *args):
    """Median step seconds of a compiled call."""
    jax.block_until_ready(fn(*args))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(ITERS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    """Run the grid; print per-cell rows and the MIN_SPEEDUP metric line."""
    print(f"device: {jax.devices()[0].device_kind} warmup={WARMUP} iters={ITERS}", flush=True)

    speedups = []
    failed = False
    for shape_name, k_dim, n_dim in SHAPES:
        weight = jax.random.normal(jax.random.PRNGKey(0), (k_dim, n_dim), jnp.float32)
        w_bf16 = weight.astype(jnp.bfloat16)

        absmax = jnp.abs(weight).max(axis=0, keepdims=True)
        scale = (absmax / 7.0).astype(jnp.float32)
        codes = jnp.clip(jnp.round(weight / jnp.where(scale == 0, 1, scale)), -7, 7).astype(jnp.int32)
        w_unpacked_int4 = codes.astype(jnp.int4)
        w_packed = pack_int4_split_k(codes)
        w_ref = codes.astype(jnp.float32) * scale  # dense f32 reference weight

        tile_n, chunk = CONFIG[shape_name]
        w_packed_adj = pack_int4_adjacent(codes)
        dense = jax.jit(lambda x, w: x @ w)

        for tokens in TOKENS:
            x = jax.random.normal(jax.random.PRNGKey(1), (tokens, k_dim), jnp.bfloat16)
            base_step = timed(dense, x, w_bf16)

            xla_step = timed(
                jax.jit(lambda x, w=w_unpacked_int4, s=scale: channelwise_quantized_matmul(x, w, s)), x
            )

            try:
                packed_fn = jax.jit(
                    lambda x, w=w_packed, s=scale, tn=tile_n, ck=chunk: packed_int4_gemv(
                        x, w, s, tile_n=tn, chunk=ck
                    )
                )
                packed_step = timed(packed_fn, x)
                got = packed_fn(x).astype(jnp.float32)
                ref = x.astype(jnp.float32) @ w_ref
                rel = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
                speedup = base_step / packed_step
                speedups.append(speedup)
                marker = "  RELERR-FAIL" if rel > MAX_RELERR else ""
                failed = failed or rel > MAX_RELERR
                # Informational W4A4 cell (opt-in path; excluded from the
                # MIN_SPEEDUP gate — its accuracy story needs calibration).
                x_abs = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=1, keepdims=True)
                x_scale4 = x_abs / 7.0
                x4 = jnp.clip(jnp.round(x.astype(jnp.float32) / x_scale4), -7, 7).astype(jnp.int4)
                w4a4_fn = jax.jit(lambda x4, w=w_packed_adj, s=scale: w4a4_gemv(x4, w, s, tile_n=1024))
                w4a4_step = timed(w4a4_fn, x4)
                print(
                    f"{shape_name:12s} t={tokens:<3d} bf16={base_step * 1e6:8.1f}us "
                    f"xla-int4={xla_step * 1e6:8.1f}us ({base_step / xla_step:4.2f}x) "
                    f"packed={packed_step * 1e6:8.1f}us  {speedup:5.2f}x  relerr={rel:.4f}{marker} "
                    f"| w4a4={w4a4_step * 1e6:8.1f}us ({base_step / w4a4_step:4.2f}x)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failed = True
                speedups.append(0.0)
                print(f"{shape_name:12s} t={tokens:<3d} packed FAILED: {type(exc).__name__}: {str(exc)[:130]}", flush=True)

    print(f"MIN_SPEEDUP={min(speedups):.3f}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
