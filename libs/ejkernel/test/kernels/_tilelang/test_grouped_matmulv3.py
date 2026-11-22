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

"""TileLang parity tests for grouped_matmulv3."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ._helpers import (
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_grouped_matmulv3_scale_bias_parity(transpose_rhs):
    M, K, N = 64, 32, 16
    num_groups = 2
    key = jax.random.PRNGKey(_SEED + 22)
    ks = jax.random.split(key, 5)
    lhs = _randn(ks[0], (M, K), scale=0.2)
    rhs_k_n = _randn(ks[1], (num_groups, K, N), scale=0.2)
    rhs = jnp.swapaxes(rhs_k_n, 1, 2) if transpose_rhs else rhs_k_n
    group_sizes = jnp.array([20, 44], dtype=jnp.int32)
    rhs_scale = (jax.random.uniform(ks[2], (num_groups, 4, 1, N)) * 0.5 + 0.5).astype(jnp.float16)
    rhs_bias = _randn(ks[3], (num_groups, 1, N), scale=0.05)
    existing_out = _randn(ks[4], (M, N), scale=0.05)
    kwargs = {
        "existing_out": existing_out,
        "rhs_scale": rhs_scale,
        "rhs_bias": rhs_bias,
        "transpose_rhs": transpose_rhs,
    }
    out_tl = _tl("grouped_matmulv3")(lhs, rhs, group_sizes, **kwargs)
    out_xla = _xla("grouped_matmulv3")(lhs, rhs, group_sizes, **kwargs)
    assert _max_abs(out_tl, out_xla) < 5e-2

    def loss(fn, lhs_, rhs_, existing_, scale_, bias_):
        out = fn(
            lhs_,
            rhs_,
            group_sizes,
            existing_out=existing_,
            rhs_scale=scale_,
            rhs_bias=bias_,
            transpose_rhs=transpose_rhs,
        )
        return jnp.sum(out.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(
        _tl("grouped_matmulv3"),
        lhs,
        rhs,
        existing_out,
        rhs_scale,
        rhs_bias,
    )
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(
        _xla("grouped_matmulv3"),
        lhs,
        rhs,
        existing_out,
        rhs_scale,
        rhs_bias,
    )
    for a_tl, a_x, grad_name in zip(g_tl, g_x, "lhs rhs existing scale bias".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 8e-2, f"v3 grad d{grad_name} too large"


def test_grouped_matmulv3_group_offset_scale_bias():
    K, N = 16, 12
    group_sizes = jnp.array([5, 7, 9, 3], dtype=jnp.int32)
    group_offset = jnp.array([1], dtype=jnp.int32)
    M = 16
    num_groups = 2
    key = jax.random.PRNGKey(_SEED + 38)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    lhs = _randn(k1, (M, K), scale=0.2)
    rhs = _randn(k2, (num_groups, K, N), scale=0.2)
    rhs_scale = (jax.random.uniform(k3, (num_groups, 4, 1, N)) * 0.5 + 0.5).astype(jnp.float16)
    rhs_bias = _randn(k4, (num_groups, 1, N), scale=0.05)
    existing_out = _randn(k5, (M, N), scale=0.05)
    kwargs = {
        "group_offset": group_offset,
        "existing_out": existing_out,
        "rhs_scale": rhs_scale,
        "rhs_bias": rhs_bias,
    }
    out_tl = _tl("grouped_matmulv3")(lhs, rhs, group_sizes, **kwargs)
    out_xla = _xla("grouped_matmulv3")(lhs, rhs, group_sizes, **kwargs)
    assert _max_abs(out_tl, out_xla) < 5e-2

    def loss(fn, lhs_, rhs_, existing_, scale_, bias_):
        out = fn(
            lhs_,
            rhs_,
            group_sizes,
            group_offset=group_offset,
            existing_out=existing_,
            rhs_scale=scale_,
            rhs_bias=bias_,
        )
        return jnp.sum(out.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(
        _tl("grouped_matmulv3"),
        lhs,
        rhs,
        existing_out,
        rhs_scale,
        rhs_bias,
    )
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(
        _xla("grouped_matmulv3"),
        lhs,
        rhs,
        existing_out,
        rhs_scale,
        rhs_bias,
    )
    for a_tl, a_x, grad_name in zip(g_tl, g_x, "lhs rhs existing scale bias".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 8e-2, f"v3 group_offset grad d{grad_name} too large"
