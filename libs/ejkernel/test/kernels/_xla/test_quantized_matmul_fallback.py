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
from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as xla_quantized_matmul
from ejkernel.quantization import prepack_quantized_weights


def _make_affine_row_group128_inputs():
    key_x, key_w = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.normal(key_x, (16, 64), dtype=jnp.float16)
    w = jax.random.normal(key_w, (128, 64), dtype=jnp.float16)  # (N, K)
    w_q, scales, zeros = prepack_quantized_weights(
        w,
        mode="affine",
        bits=4,
        group_size=128,
        axis="row",
    )
    return x, w_q, scales, zeros


def test_xla_quantized_matmul_allows_dense_fallback_by_default():
    x, w_q, scales, zeros = _make_affine_row_group128_inputs()

    # transpose=False uses block_n alignment; block_n=64 is illegal for group_size=128.
    fn = jax.jit(
        lambda xi, wi, si, zi: xla_quantized_matmul(
            xi,
            wi,
            si,
            zi,
            mode="affine",
            bits=4,
            group_size=128,
            axis="row",
            transpose=False,
            block_n=64,
            allow_dense_fallback=True,
        )
    )

    y = fn(x, w_q, scales, zeros)
    assert y.shape == (16, 128)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_affine_zero_scale_block_matches_dense_decode():
    """A zero-scale group decodes to zero even when its zero-point is nonzero."""
    x = jnp.ones((2, 32), dtype=jnp.float16)
    w_q = jnp.full((32, 8), 0x11111111, dtype=jnp.uint32)  # 64 codes of value one per K row.
    scales = jnp.broadcast_to(jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float16), (32, 4))
    zeros = jnp.full((32, 4), 3.0, dtype=jnp.float16)

    def run(block_n):
        return xla_quantized_matmul(
            x,
            w_q,
            scales,
            zeros,
            mode="affine",
            bits=4,
            group_size=16,
            transpose=False,
            block_m=2,
            block_n=block_n,
            block_k=32,
            use_bf16=False,
        )

    blocked = run(64)
    dense = run(8)  # Invalid tile alignment forces dense decode.
    # Groups run along N: columns 0-15 have scale 0 -> (1 - 3) * 0 = 0; the rest -> 32 * (1 - 3) * 1 = -64.
    # The pre-fix blocked path decoded the zero-scale group as -zero, giving 32 * -3 = -96 there.
    expected = jnp.concatenate(
        [jnp.zeros((2, 16), dtype=jnp.float32), jnp.full((2, 48), -64.0, dtype=jnp.float32)], axis=1
    )
    assert jnp.array_equal(dense, expected)
    assert jnp.array_equal(blocked, dense)


def test_xla_quantized_matmul_disallow_dense_fallback_raises():
    x, w_q, scales, zeros = _make_affine_row_group128_inputs()

    fn = jax.jit(
        lambda xi, wi, si, zi: xla_quantized_matmul(
            xi,
            wi,
            si,
            zi,
            mode="affine",
            bits=4,
            group_size=128,
            axis="row",
            transpose=False,
            block_n=64,
            allow_dense_fallback=False,
        )
    )

    with pytest.raises(ValueError, match="allow_dense_fallback"):
        _ = fn(x, w_q, scales, zeros)
