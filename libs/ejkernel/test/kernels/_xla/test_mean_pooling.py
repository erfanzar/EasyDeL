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


"""Tests for XLA mean pooling implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import mean_pooling


class TestMeanPooling:
    """Test suite for mean pooling kernel."""

    def test_fixed_length_shape(self):
        """Test fixed-length pooling output shapes."""
        batch, seq_len, hidden_dim = 4, 20, 64
        x = jnp.ones((batch, seq_len, hidden_dim))

        output = mean_pooling(x)

        assert output.shape == (batch, hidden_dim)

    def test_fixed_length_values(self):
        """Test fixed-length pooling computes correct mean."""
        batch, seq_len, hidden_dim = 2, 10, 16

        x = jnp.arange(batch * seq_len * hidden_dim).reshape(batch, seq_len, hidden_dim).astype(jnp.float32)

        output = mean_pooling(x)

        expected = jnp.mean(x, axis=1)
        assert jnp.allclose(output, expected, atol=1e-5)

    def test_varlen_shape(self):
        """Test variable-length pooling output shapes."""
        total_tokens, hidden_dim = 150, 64
        num_seqs = 3

        x = jnp.ones((total_tokens, hidden_dim))
        cu_seqlens = jnp.array([0, 50, 100, 150])

        output = mean_pooling(x, cu_seqlens=cu_seqlens)

        assert output.shape == (num_seqs, hidden_dim)

    def test_varlen_values(self):
        """Test variable-length pooling computes correct means."""
        hidden_dim = 8

        x = jnp.array(
            [
                [1.0] * hidden_dim,
                [2.0] * hidden_dim,
                [3.0] * hidden_dim,
                [4.0] * hidden_dim,
                [5.0] * hidden_dim,
                [6.0] * hidden_dim,
                [7.0] * hidden_dim,
                [8.0] * hidden_dim,
                [9.0] * hidden_dim,
            ]
        )
        cu_seqlens = jnp.array([0, 3, 5, 9])

        output = mean_pooling(x, cu_seqlens=cu_seqlens)

        expected = jnp.array([[2.0] * hidden_dim, [4.5] * hidden_dim, [7.5] * hidden_dim])

        assert jnp.allclose(output, expected, atol=1e-5)

    def test_gradient_fixed_length(self):
        """Test gradient shapes for fixed-length pooling."""
        batch, seq_len, hidden_dim = 2, 10, 16
        x = jnp.ones((batch, seq_len, hidden_dim))

        def loss_fn(x):
            return jnp.sum(mean_pooling(x))

        dx = jax.grad(loss_fn)(x)

        assert dx.shape == x.shape

    def test_gradient_varlen(self):
        """Test gradient shapes for variable-length pooling."""
        total_tokens, hidden_dim = 30, 16
        x = jnp.ones((total_tokens, hidden_dim))
        cu_seqlens = jnp.array([0, 10, 20, 30])

        def loss_fn(x):
            return jnp.sum(mean_pooling(x, cu_seqlens=cu_seqlens))

        dx = jax.grad(loss_fn)(x)

        assert dx.shape == x.shape

    def test_gradient_distribution(self):
        """Test that gradients are correctly distributed in varlen case."""
        hidden_dim = 4
        x = jnp.ones((9, hidden_dim))
        cu_seqlens = jnp.array([0, 3, 5, 9])

        def loss_fn(x):
            return jnp.sum(mean_pooling(x, cu_seqlens=cu_seqlens))

        dx = jax.grad(loss_fn)(x)

        expected = jnp.array(
            [
                [1 / 3] * hidden_dim,
                [1 / 3] * hidden_dim,
                [1 / 3] * hidden_dim,
                [1 / 2] * hidden_dim,
                [1 / 2] * hidden_dim,
                [1 / 4] * hidden_dim,
                [1 / 4] * hidden_dim,
                [1 / 4] * hidden_dim,
                [1 / 4] * hidden_dim,
            ]
        )

        assert jnp.allclose(dx, expected, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
