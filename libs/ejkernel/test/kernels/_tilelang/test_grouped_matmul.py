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

"""TileLang parity tests for grouped_matmul."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ._helpers import _SEED, _max_abs, _randn, _tl, _xla

_ALGORITHM = "grouped_matmul"


@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_grouped_matmul_parity(transpose_rhs):
    M, K, N = 64, 32, 16
    num_groups = 4
    group_size = M // num_groups
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 3)
    lhs = _randn(ks[0], (M, K), scale=0.3)
    rhs_k_n = _randn(ks[1], (num_groups, K, N), scale=0.3)
    rhs = jnp.swapaxes(rhs_k_n, 1, 2) if transpose_rhs else rhs_k_n
    existing_out = _randn(ks[2], (M, N), scale=0.05)
    group_sizes = jnp.full((num_groups,), group_size, dtype=jnp.int32)
    kwargs = {"existing_out": existing_out, "transpose_rhs": transpose_rhs}
    out_tl = _tl(_ALGORITHM)(lhs, rhs, group_sizes, **kwargs)
    out_xla = _xla(_ALGORITHM)(lhs, rhs, group_sizes, **kwargs)
    assert _max_abs(out_tl, out_xla) < 5e-2

    def loss(fn, lhs_, rhs_, existing_):
        out = fn(lhs_, rhs_, group_sizes, existing_out=existing_, transpose_rhs=transpose_rhs)
        return jnp.sum(out.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3))(_tl(_ALGORITHM), lhs, rhs, existing_out)
    g_x = jax.grad(loss, argnums=(1, 2, 3))(_xla(_ALGORITHM), lhs, rhs, existing_out)
    for a_tl, a_x, grad_name in zip(g_tl, g_x, "lhs rhs existing".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 5e-2, f"grad d{grad_name} too large"


def test_grouped_matmul_group_offset():
    K, N = 16, 12
    group_sizes = jnp.array([5, 7, 9, 3], dtype=jnp.int32)
    group_offset = jnp.array([1], dtype=jnp.int32)
    M = 16
    num_groups = 2
    key = jax.random.PRNGKey(_SEED + 37)
    k1, k2, k3 = jax.random.split(key, 3)
    lhs = _randn(k1, (M, K), scale=0.25)
    rhs = _randn(k2, (num_groups, K, N), scale=0.25)
    existing_out = _randn(k3, (M, N), scale=0.05)
    kwargs = {"group_offset": group_offset, "existing_out": existing_out}
    out_tl = _tl(_ALGORITHM)(lhs, rhs, group_sizes, **kwargs)
    local_sizes = group_sizes[1:3]
    group_ids = jnp.repeat(jnp.arange(num_groups, dtype=jnp.int32), local_sizes, total_repeat_length=M)
    out_ref = jax.vmap(lambda row, group: row @ rhs[group])(lhs.astype(jnp.float32), group_ids).astype(jnp.float16)
    out_ref = out_ref + existing_out
    assert _max_abs(out_tl, out_ref) < 5e-2

    def loss_tl(lhs_, rhs_, existing_):
        out = _tl(_ALGORITHM)(lhs_, rhs_, group_sizes, group_offset=group_offset, existing_out=existing_)
        return jnp.sum(out.astype(jnp.float32))

    def loss_ref(lhs_, rhs_, existing_):
        out = jax.vmap(lambda row, group: row @ rhs_[group])(lhs_.astype(jnp.float32), group_ids).astype(jnp.float16)
        return jnp.sum((out + existing_).astype(jnp.float32))

    g_tl = jax.grad(loss_tl, argnums=(0, 1, 2))(lhs, rhs, existing_out)
    g_x = jax.grad(loss_ref, argnums=(0, 1, 2))(lhs, rhs, existing_out)
    for a_tl, a_x, grad_name in zip(g_tl, g_x, "lhs rhs existing".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 5e-2, f"group_offset grad d{grad_name} too large"
