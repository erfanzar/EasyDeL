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

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

# Mode budgets from ../quant-plans/jax_numerics_validation.md.
# We validate relative errors between fuse=True and fuse=False (dense reference)
# runs on GPU for strict-fuse QMM calls.
_BUDGETS = {
    "affine": (1.5e-2, 2.5e-2),
    "nf4": (3.5e-2, 5.0e-2),
    "mxfp4": (4.5e-2, 6.0e-2),
    "mxfp8": (2.0e-2, 3.0e-2),
    "nvfp4": (4.5e-2, 6.0e-2),
    "nvfp8": (2.5e-2, 3.5e-2),
}


def _relative_l2(a: jax.Array, b: jax.Array) -> jax.Array:
    diff = a - b
    num = jnp.linalg.norm(diff.reshape(-1))
    den = jnp.linalg.norm(b.reshape(-1)) + jnp.array(1e-12, dtype=num.dtype)
    return num / den


def _relative_mean_abs(a: jax.Array, b: jax.Array) -> jax.Array:
    diff = a - b
    num = jnp.mean(jnp.abs(diff))
    den = jnp.mean(jnp.abs(b)) + jnp.array(1e-12, dtype=num.dtype)
    return num / den


def test_qmm_strict_fuse_gpu_parity_budgets():
    backend = jax.default_backend()
    if backend not in {"gpu", "cuda"}:
        pytest.skip(f"GPU-only numerics parity check (backend={backend}).")

    # Keep this small enough for CI while still exercising both axes and
    # representative modes.
    dtype = jnp.bfloat16
    seed = 0
    n = 4096
    k = 4096
    m_values = (1, 128)
    axes = ("row", "col")

    # Cover affine bits and nf4 group sizes used by QMM perf gates.
    cases = [
        ("affine", 4, 128),
        ("affine", 8, 128),
        ("nf4", 4, 64),
        ("nf4", 4, 128),
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
        ("nvfp8", 8, 16),
    ]

    for mode, bits, group_size in cases:
        bud_rel_l2, bud_rel_abs = _BUDGETS[mode]

        for axis in axes:
            transpose = axis == "col"

            for m in m_values:
                key = jax.random.PRNGKey(seed)
                k1, k2 = jax.random.split(key, 2)
                x = jax.random.normal(k1, (m, k), dtype=dtype)
                w = jax.random.normal(k2, (n, k), dtype=dtype)

                packed = prepack_quantized_weights(
                    w,
                    mode=mode,
                    bits=bits,
                    group_size=group_size,
                    axis=axis,
                )
                if mode == "affine":
                    w_q, scales, zeros = packed
                else:
                    w_q, scales = packed
                    zeros = None

                # Metadata must be finite regardless of fused/reference execution.
                assert bool(jnp.all(jnp.isfinite(scales.astype(jnp.float32))))
                if zeros is not None:
                    assert bool(jnp.all(jnp.isfinite(zeros.astype(jnp.float32))))

                fn_ref = jax.jit(
                    lambda xi, wi, si, zi, mode=mode, bits=bits, group_size=group_size, axis=axis, transpose=transpose: (
                        quantized_matmul(
                            xi,
                            wi,
                            si,
                            zi,
                            mode=mode,
                            bits=bits,
                            group_size=group_size,
                            axis=axis,
                            transpose=transpose,
                            platform="xla",
                            fuse=False,
                        )
                    )
                )
                fn_fused = jax.jit(
                    lambda xi, wi, si, zi, mode=mode, bits=bits, group_size=group_size, axis=axis, transpose=transpose: (
                        quantized_matmul(
                            xi,
                            wi,
                            si,
                            zi,
                            mode=mode,
                            bits=bits,
                            group_size=group_size,
                            axis=axis,
                            transpose=transpose,
                            platform="xla",
                            fuse=True,
                            strict_fuse=True,
                        )
                    )
                )

                y_ref = fn_ref(x, w_q, scales, zeros)
                y_fused = fn_fused(x, w_q, scales, zeros)

                assert y_ref.shape == y_fused.shape
                assert bool(jnp.all(jnp.isfinite(y_fused)))

                rel_l2 = _relative_l2(y_fused, y_ref)
                rel_abs = _relative_mean_abs(y_fused, y_ref)
                rel_l2_f = float(rel_l2.block_until_ready())
                rel_abs_f = float(rel_abs.block_until_ready())

                assert rel_l2_f <= bud_rel_l2, (
                    f"mode={mode} bits={bits} group_size={group_size} axis={axis} m={m}: "
                    f"relative_l2={rel_l2_f:.4e} > budget={bud_rel_l2:.4e}"
                )
                assert rel_abs_f <= bud_rel_abs, (
                    f"mode={mode} bits={bits} group_size={group_size} axis={axis} m={m}: "
                    f"relative_mean_abs={rel_abs_f:.4e} > budget={bud_rel_abs:.4e}"
                )
