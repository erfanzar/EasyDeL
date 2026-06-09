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
