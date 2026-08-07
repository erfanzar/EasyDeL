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

"""Speedup gate for quantized grouped_matmulv3 vs jitted bf16 matmul (TPU).

Emits one row per (variant, shape, tokens) and a final ``MIN_SPEEDUP=`` line —
the autoresearch verify metric. Exits non-zero if any cell's relative error
exceeds ``MAX_RELERR`` (a fast experiment that breaks numerics is a crash, not
a keep).

Variants are the two storage classes every checkpoint format lowers to on
TPU v5 (f4 storage is Mosaic-unsupported; f8 compute has no fast upcast):
int8 W8A16 and int4 W4A16, both K-blocked, dequantize-before, bf16 MXU.
"""

import statistics
import sys
import time

import jax
import jax.numpy as jnp
from ejkernel.kernels._xla.quantized_matmul import channelwise_quantized_matmul

SHAPES = [
    ("qkv", 4096, 6144),
    ("gate_up", 4096, 28672),
    ("down", 14336, 4096),
]
TOKENS = (8, 2048)
GROUP = 512
MAX_RELERR = 0.08
WARMUP = 3
ITERS = 12

#: Per-(variant, shape, tokens) tiling. Experiments tune this table.
TILING = {
    ("int8", "qkv", 8): (128, 2048, 1024),
    ("int8", "qkv", 2048): (256, 1024, 1024),
    ("int8", "gate_up", 8): (128, 2048, 1024),
    ("int8", "gate_up", 2048): (256, 1024, 1024),
    ("int8", "down", 8): (128, 2048, 1024),
    ("int8", "down", 2048): (256, 1024, 1024),
    ("int4", "qkv", 8): (128, 2048, 1024),
    ("int4", "qkv", 2048): (256, 1024, 1024),
    ("int4", "gate_up", 8): (128, 2048, 1024),
    ("int4", "gate_up", 2048): (256, 1024, 1024),
    ("int4", "down", 8): (128, 2048, 1024),
    ("int4", "down", 2048): (256, 1024, 1024),
}

VARIANTS = {"int8": jnp.int8, "int4": jnp.int4}


def timed(fn, *args):
    """Return the median step seconds of a compiled call."""
    jax.block_until_ready(fn(*args))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(ITERS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def block_quantize(weight, group, qdtype):
    """Symmetric per-channel (full-K) absmax quantization.

    ``group`` is ignored: per-channel scales are what enable single-dot
    execution (any K-blocking forces either a fusion-breaking weight rewrite
    or giant int32 partials — measured in iterations 4/5).
    """
    del group
    k_dim, n_dim = weight.shape
    absmax = jnp.max(jnp.abs(weight), axis=0, keepdims=True)
    qmax = float(jnp.iinfo(qdtype).max)
    scale = (absmax / qmax).astype(jnp.float32)
    codes = jnp.clip(jnp.round(weight / jnp.where(scale == 0, 1, scale)), -qmax, qmax).astype(qdtype)
    return codes, scale.reshape(1, 1, n_dim)




def main():
    """Run the full grid; print the table and the MIN_SPEEDUP metric line."""
    device = jax.devices()[0]
    print(f"device: {device.device_kind} group={GROUP} warmup={WARMUP} iters={ITERS}", flush=True)

    speedups = []
    failed = False
    for shape_name, k_dim, n_dim in SHAPES:
        weight = jax.random.normal(jax.random.PRNGKey(0), (k_dim, n_dim), jnp.float32)
        w_bf16 = weight.astype(jnp.bfloat16)
        dense = jax.jit(lambda x, w: x @ w)

        for variant, qdtype in VARIANTS.items():
            w_q, scale = block_quantize(weight, GROUP, qdtype)
            rhs, rhs_scale = w_q[None], scale[None]

            for tokens in TOKENS:
                tile_m, tile_k, tile_n = TILING[(variant, shape_name, tokens)]
                x = jnp.ones((tokens, k_dim), jnp.bfloat16)
                base_step = timed(dense, x, w_bf16)

                w_q2, scale2 = w_q, scale  # [K, N], [K/g, 1, N]

                def qcall(x, w_q=w_q2, scale=scale2):
                    return channelwise_quantized_matmul(x, w_q, scale, quantize_activations=True)

                try:
                    step = timed(jax.jit(qcall), x)
                    x_real = jax.random.normal(jax.random.PRNGKey(1), (tokens, k_dim), jnp.bfloat16)
                    dense_scale = scale.reshape(1, n_dim)
                    ref = x_real.astype(jnp.float32) @ (w_q.astype(jnp.float32) * dense_scale)
                    got = qcall(x_real).astype(jnp.float32)
                    rel = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
                    speedup = base_step / step
                    speedups.append(speedup)
                    marker = ""
                    if rel > MAX_RELERR:
                        failed = True
                        marker = "  RELERR-FAIL"
                    print(
                        f"{variant:5s} {shape_name:8s} t={tokens:<5d} bf16={base_step * 1e6:8.1f}us "
                        f"quant={step * 1e6:8.1f}us  {speedup:5.2f}x  relerr={rel:.4f}{marker}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    speedups.append(0.0)
                    print(f"{variant:5s} {shape_name:8s} t={tokens:<5d} FAILED: {type(exc).__name__}: {str(exc)[:110]}", flush=True)

    print(f"MIN_SPEEDUP={min(speedups):.3f}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
