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

"""Tests for XLA grouped matrix multiplication (ragged_dot)."""

import inspect

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels import Platform, kernel_registry
from ejkernel.kernels._xla.grouped_matmul import grouped_matmul
from ejkernel.kernels._xla.grouped_matmulv3 import grouped_matmulv3


def _naive_grouped_matmul(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, transpose_rhs: bool) -> jax.Array:
    outs = []
    start = 0
    for group_idx in range(int(group_sizes.shape[0])):
        rows = int(group_sizes[group_idx])
        end = start + rows
        if rows == 0:
            continue
        if transpose_rhs:
            # rhs[group_idx] is (n, k)
            w = rhs[group_idx].T
        else:
            # rhs[group_idx] is (k, n)
            w = rhs[group_idx]
        outs.append(lhs[start:end] @ w)
        start = end
    return jnp.concatenate(outs, axis=0) if outs else jnp.zeros((0, rhs.shape[-2 if transpose_rhs else -1]), lhs.dtype)


def _naive_grouped_matmul_v3(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    transpose_rhs: bool,
    existing_out: jax.Array | None = None,
) -> jax.Array:
    rhs_prepped = rhs.swapaxes(1, 2) if transpose_rhs else rhs
    if rhs_scale is not None:
        num_blocks = int(rhs_scale.shape[1])
        block_size = rhs_prepped.shape[1] // num_blocks
        scale = jnp.repeat(rhs_scale[:, :, 0, :], block_size, axis=1)
        rhs_prepped = rhs_prepped * scale.astype(rhs_prepped.dtype)
    group_ids = jnp.repeat(
        jnp.arange(group_sizes.shape[0], dtype=group_sizes.dtype),
        group_sizes,
        total_repeat_length=lhs.shape[0],
    )
    out = jax.vmap(lambda row, mat: jnp.matmul(row, mat, preferred_element_type=jnp.float32))(
        lhs, rhs_prepped[group_ids]
    )

    if rhs_bias is not None:
        out = out + rhs_bias[:, 0, :][group_ids].astype(out.dtype)
    if existing_out is not None:
        out = out + existing_out.astype(out.dtype)
    return out


@pytest.mark.parametrize("transpose_rhs", [False, True])
@pytest.mark.parametrize("use_jit", [False, True])
def test_matches_naive(transpose_rhs, use_jit):
    key = jax.random.PRNGKey(0)
    key, kl, kr = jax.random.split(key, 3)

    group_sizes = jnp.array([3, 5, 2], dtype=jnp.int32)
    m = int(group_sizes.sum())
    k = 7
    n = 4
    lhs = jax.random.normal(kl, (m, k), dtype=jnp.float32)

    if transpose_rhs:
        rhs = jax.random.normal(kr, (len(group_sizes), n, k), dtype=jnp.float32)
    else:
        rhs = jax.random.normal(kr, (len(group_sizes), k, n), dtype=jnp.float32)

    if use_jit:

        def _run(lhs, rhs, group_sizes):
            return grouped_matmul(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, tiling=None)

        fn = jax.jit(_run)
        out = fn(lhs, rhs, group_sizes)
    else:
        out = grouped_matmul(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, tiling=None)
    expected = _naive_grouped_matmul(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs)

    assert out.shape == expected.shape
    assert jnp.allclose(out, expected, rtol=0, atol=0.125)


def test_registry_alias_for_grouped_matmulv2():
    impl = kernel_registry.get("grouped_matmulv2", platform=Platform.XLA)
    assert inspect.unwrap(impl) is inspect.unwrap(grouped_matmul)


def test_registry_alias_for_grouped_matmulv3():
    impl = kernel_registry.get("grouped_matmulv3", platform=Platform.XLA)
    assert callable(impl)

    key = jax.random.PRNGKey(7)
    key, kl, kr = jax.random.split(key, 3)
    group_sizes = jnp.array([4, 4], dtype=jnp.int32)
    lhs = jax.random.normal(kl, (8, 6), dtype=jnp.float32)
    rhs = jax.random.normal(kr, (2, 6, 5), dtype=jnp.float32)

    out = grouped_matmulv3(lhs, rhs, group_sizes, tiling=None)
    expected = _naive_grouped_matmul(lhs, rhs, group_sizes, transpose_rhs=False)
    assert jnp.allclose(out, expected, rtol=0, atol=0.125)


@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_grouped_matmulv3_supports_rhs_scale_and_bias(transpose_rhs):
    key = jax.random.PRNGKey(17)
    key, kl, kr, ks, kb, ko = jax.random.split(key, 6)

    group_sizes = jnp.array([2, 3], dtype=jnp.int32)
    m = int(group_sizes.sum())
    k = 4
    n = 3
    lhs = jax.random.normal(kl, (m, k), dtype=jnp.float32)
    rhs_scale = jax.random.normal(ks, (2, 2, 1, n), dtype=jnp.float32)
    rhs_bias = jax.random.normal(kb, (2, 1, n), dtype=jnp.float32)
    existing_out = jax.random.normal(ko, (m, n), dtype=jnp.float32)

    if transpose_rhs:
        rhs = jax.random.normal(kr, (2, n, k), dtype=jnp.float32)
    else:
        rhs = jax.random.normal(kr, (2, k, n), dtype=jnp.float32)

    out = grouped_matmulv3(
        lhs,
        rhs,
        group_sizes,
        tiling=None,
        existing_out=existing_out,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        transpose_rhs=transpose_rhs,
    )
    expected = _naive_grouped_matmul_v3(
        lhs,
        rhs,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        transpose_rhs=transpose_rhs,
        existing_out=existing_out,
    )

    assert out.shape == expected.shape
    assert jnp.allclose(out, expected, rtol=0, atol=1e-5)


def test_grouped_matmulv3_scale_bias_gradients_match_reference():
    key = jax.random.PRNGKey(23)
    key, kl, kr, ks, kb, ko = jax.random.split(key, 6)

    group_sizes = jnp.array([2, 3], dtype=jnp.int32)
    lhs = jax.random.normal(kl, (5, 4), dtype=jnp.float32)
    rhs = jax.random.normal(kr, (2, 4, 3), dtype=jnp.float32)
    rhs_scale = jax.random.normal(ks, (2, 2, 1, 3), dtype=jnp.float32)
    rhs_bias = jax.random.normal(kb, (2, 1, 3), dtype=jnp.float32)
    existing_out = jax.random.normal(ko, (5, 3), dtype=jnp.float32)

    def kernel_loss(lhs, rhs, rhs_scale, rhs_bias):
        out = grouped_matmulv3(
            lhs,
            rhs,
            group_sizes,
            tiling=None,
            existing_out=existing_out,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
        )
        return jnp.sum(out**2)

    def ref_loss(lhs, rhs, rhs_scale, rhs_bias):
        out = _naive_grouped_matmul_v3(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            transpose_rhs=False,
            existing_out=existing_out,
        )
        return jnp.sum(out**2)

    grads = jax.grad(kernel_loss, argnums=(0, 1, 2, 3))(lhs, rhs, rhs_scale, rhs_bias)
    expected_grads = jax.grad(ref_loss, argnums=(0, 1, 2, 3))(lhs, rhs, rhs_scale, rhs_bias)

    for got, expected in zip(grads, expected_grads, strict=True):
        assert got.shape == expected.shape
        assert jnp.allclose(got, expected, rtol=0, atol=1e-4)
