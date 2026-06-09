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


import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.grouped_matmul import grouped_matmul
from ejkernel.kernels._xla.grouped_matmul import grouped_matmul as grouped_matmul_xla


def _has_tpu():
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def naive_grouped_matmul_reference(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    transpose_rhs: bool = False,
) -> jnp.ndarray:
    """Reference implementation of grouped matmul using standard operations."""
    m, _k = lhs.shape
    num_groups = group_sizes.shape[0]

    if transpose_rhs:
        n = rhs.shape[1]
    else:
        n = rhs.shape[2]

    output = jnp.zeros((m, n), dtype=lhs.dtype)

    row_offset = 0
    for group_idx in range(num_groups):
        group_size = int(group_sizes[group_idx])
        if group_size == 0:
            continue

        lhs_slice = lhs[row_offset : row_offset + group_size, :]

        if transpose_rhs:
            rhs_matrix = rhs[group_idx].T
        else:
            rhs_matrix = rhs[group_idx]

        result = jnp.dot(lhs_slice, rhs_matrix)
        output = output.at[row_offset : row_offset + group_size, :].set(result)

        row_offset += group_size

    return output


class TestGroupedMatMulTPU:
    """Test suite for TPU Pallas grouped matrix multiplication implementation."""

    def test_basic_forward_shape_and_finite(self):
        """Test basic forward pass with simple configuration."""
        m = 256
        k = 128
        n = 64
        num_groups = 4
        group_size = m // num_groups

        key = jax.random.PRNGKey(42)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([group_size] * num_groups, dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        assert output.shape == (m, n)
        assert jnp.isfinite(output).all()
        assert output.dtype == jnp.float32

    def test_multiple_groups(self):
        """Test with multiple groups."""
        m = 384
        k = 128
        n = 64
        num_groups = 3

        key = jax.random.PRNGKey(123)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)

        group_sizes = jnp.array([128, 128, 128], dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        assert output.shape == (m, n)
        assert jnp.isfinite(output).all()

    def test_transpose_rhs(self):
        """Test with transposed RHS."""
        m = 256
        k = 128
        n = 64
        num_groups = 4

        key = jax.random.PRNGKey(456)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)

        rhs = jax.random.normal(k2, (num_groups, n, k), dtype=jnp.float32)
        group_sizes = jnp.array([64] * num_groups, dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            transpose_rhs=True,
        )

        assert output.shape == (m, n)
        assert jnp.isfinite(output).all()

    def test_single_group(self):
        """Test with a single group (equivalent to regular matmul)."""
        m = 256
        k = 128
        n = 64
        num_groups = 1

        key = jax.random.PRNGKey(789)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([m], dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        expected = jnp.dot(lhs, rhs[0])

        assert output.shape == (m, n)
        assert jnp.allclose(output, expected, rtol=1e-4, atol=1e-4)

    def test_many_small_groups(self):
        """Test with many groups."""
        m = 512
        k = 128
        n = 64
        num_groups = 4
        group_size = m // num_groups

        key = jax.random.PRNGKey(1000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([group_size] * num_groups, dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        assert output.shape == (m, n)
        assert jnp.isfinite(output).all()

    def test_different_dimensions(self):
        """Test with various dimension combinations."""
        configs = [
            (256, 128, 128, 2),
            (512, 128, 128, 4),
            (512, 256, 256, 2),
        ]

        for m, k, n, num_groups in configs:
            key = jax.random.PRNGKey(2000 + m)
            key, k1, k2 = jax.random.split(key, 3)

            lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
            rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
            group_size = m // num_groups
            group_sizes = jnp.array([group_size] * num_groups, dtype=jnp.int32)

            output = grouped_matmul(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
            )

            assert output.shape == (m, n)
            assert jnp.isfinite(output).all()

    def test_dtypes(self):
        """Test with different dtypes."""
        m = 256
        k = 128
        n = 64
        num_groups = 4

        for dtype in [jnp.float32, jnp.bfloat16]:
            key = jax.random.PRNGKey(3000)
            key, k1, k2 = jax.random.split(key, 3)

            lhs = jax.random.normal(k1, (m, k), dtype=dtype)
            rhs = jax.random.normal(k2, (num_groups, k, n), dtype=dtype)
            group_sizes = jnp.array([64] * num_groups, dtype=jnp.int32)

            output = grouped_matmul(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                preferred_element_type=dtype,
            )

            assert output.shape == (m, n)
            assert output.dtype == dtype
            assert jnp.isfinite(output).all()

    def test_numerical_correctness_vs_xla(self):
        """Test numerical correctness against XLA reference implementation."""
        m = 256
        k = 128
        n = 64
        num_groups = 4

        key = jax.random.PRNGKey(4000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([64] * num_groups, dtype=jnp.int32)

        output_tpu = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        output_xla = grouped_matmul_xla(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        assert output_tpu.shape == output_xla.shape
        assert jnp.allclose(output_tpu, output_xla, rtol=1e-3, atol=1e-3)

    def test_numerical_correctness_with_naive_reference(self):
        """Test numerical correctness against naive reference for validation."""
        m = 256
        k = 128
        n = 64
        num_groups = 4

        key = jax.random.PRNGKey(4000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([64] * num_groups, dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        expected = naive_grouped_matmul_reference(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        assert jnp.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_numerical_correctness_transpose_rhs_with_naive_reference(self):
        """Test numerical correctness with transposed RHS against naive reference."""
        m = 256
        k = 128
        n = 64
        num_groups = 4

        key = jax.random.PRNGKey(5000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, n, k), dtype=jnp.float32)
        group_sizes = jnp.array([64] * num_groups, dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            transpose_rhs=True,
        )

        expected = naive_grouped_matmul_reference(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            transpose_rhs=True,
        )

        assert jnp.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_numerical_correctness_multiple_groups_vs_xla(self):
        """Test numerical correctness with multiple groups against XLA."""
        m = 384
        k = 128
        n = 64
        num_groups = 3

        key = jax.random.PRNGKey(6000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([128, 128, 128], dtype=jnp.int32)

        output_tpu = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        output_xla = grouped_matmul_xla(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        assert output_tpu.shape == output_xla.shape
        assert jnp.allclose(output_tpu, output_xla, rtol=1e-3, atol=1e-3)

    def test_numerical_correctness_multiple_groups_with_naive_reference(self):
        """Test numerical correctness with multiple groups against naive reference."""
        m = 384
        k = 128
        n = 64
        num_groups = 3

        key = jax.random.PRNGKey(6000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([128, 128, 128], dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        expected = naive_grouped_matmul_reference(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        assert jnp.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_backward_pass(self):
        """Test that backward pass works (gradient computation)."""
        m = 256
        k = 128
        n = 64
        num_groups = 2

        key = jax.random.PRNGKey(7000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([128, 128], dtype=jnp.int32)

        def loss_fn(l, r):
            out = grouped_matmul(
                lhs=l,
                rhs=r,
                group_sizes=group_sizes,
            )
            return jnp.sum(out**2)

        grads = jax.grad(loss_fn, argnums=(0, 1))(lhs, rhs)

        assert len(grads) == 2
        assert grads[0].shape == lhs.shape
        assert grads[1].shape == rhs.shape
        assert jnp.isfinite(grads[0]).all()
        assert jnp.isfinite(grads[1]).all()

    def test_different_tiling_sizes(self):
        """Test with different tiling configurations."""
        m = 256
        k = 128
        n = 64
        num_groups = 4

        key = jax.random.PRNGKey(10000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([64] * num_groups, dtype=jnp.int32)

        tiling_configs = [
            (128, 128, 128),
            (256, 256, 256),
        ]

        for tiling in tiling_configs:
            output = grouped_matmul(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                tiling=tiling,
            )

            assert output.shape == (m, n)
            assert jnp.isfinite(output).all()

    def test_with_group_offset(self):
        """Test with group offset for sharded execution."""
        m = 256
        k = 128
        n = 64
        num_groups = 4

        key = jax.random.PRNGKey(11000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)
        group_sizes = jnp.array([64] * num_groups, dtype=jnp.int32)

        group_offset = jnp.array(0, dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            group_offset=group_offset,
        )

        assert output.shape == (m, n)
        assert jnp.isfinite(output).all()

    def test_multiple_equal_groups(self):
        """Test with multiple equal-sized groups (4 groups of 64)."""
        m = 256
        k = 128
        n = 64
        num_groups = 4

        key = jax.random.PRNGKey(12000)
        key, k1, k2 = jax.random.split(key, 3)

        lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
        rhs = jax.random.normal(k2, (num_groups, k, n), dtype=jnp.float32)

        group_sizes = jnp.array([64, 64, 64, 64], dtype=jnp.int32)

        output = grouped_matmul(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
        )

        assert output.shape == (m, n)
        assert jnp.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
