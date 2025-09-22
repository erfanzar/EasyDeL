# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for normalization layers."""

import jax
import jax.numpy as jnp
import pytest

from easydel.layers.norms import RMSNorm, float8s


class TestRMSNorm:
    """Test RMSNorm layer."""

    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        return {
            "dim": 768,
            "eps": 1e-6,
            "batch_size": 4,
            "seq_len": 32,
        }

    def test_initialization(self, setup):
        """Test RMSNorm initialization."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        assert norm.dim == setup["dim"]
        assert norm.eps == setup["eps"]
        assert norm.dtype == jnp.float32
        assert norm.param_dtype == jnp.float32
        assert norm.kernel.value.shape == (setup["dim"],)
        # Kernel should be initialized to ones
        assert jnp.allclose(norm.kernel.value, 1.0)

    def test_forward_2d(self, setup):
        """Test RMSNorm forward pass with 2D input."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        inputs = jnp.ones((setup["batch_size"], setup["dim"]))
        output = norm(inputs)

        assert output.shape == inputs.shape
        assert output.dtype == inputs.dtype

        # With all ones input and weight, output should be ones
        assert jnp.allclose(output, jnp.ones_like(inputs))

    def test_forward_3d(self, setup):
        """Test RMSNorm forward pass with 3D input."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        inputs = jnp.ones((setup["batch_size"], setup["seq_len"], setup["dim"]))
        output = norm(inputs)

        assert output.shape == inputs.shape
        assert output.dtype == inputs.dtype

    def test_normalization_effect(self, setup):
        """Test that RMSNorm actually normalizes."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        # Create input with varying magnitudes
        key = jax.random.PRNGKey(42)
        inputs = jax.random.normal(key, (setup["batch_size"], setup["dim"])) * 10.0
        output = norm(inputs)

        # Compute RMS of output (should be close to 1)
        rms = jnp.sqrt(jnp.mean(output**2, axis=-1))
        assert jnp.allclose(rms, 1.0, rtol=1e-4)

    def test_zero_input(self, setup):
        """Test RMSNorm with zero input."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        inputs = jnp.zeros((setup["batch_size"], setup["dim"]))
        output = norm(inputs)

        assert output.shape == inputs.shape
        # Output should be zero (0 * weight = 0)
        assert jnp.allclose(output, 0.0)

    def test_epsilon_stability(self, setup):
        """Test that epsilon provides numerical stability."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        # Very small input that would cause instability without epsilon
        inputs = jnp.ones((setup["batch_size"], setup["dim"])) * 1e-10
        output = norm(inputs)

        assert output.shape == inputs.shape
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_bfloat16_dtype(self, setup):
        """Test RMSNorm with bfloat16 dtype."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )

        inputs = jnp.ones((setup["batch_size"], setup["dim"]), dtype=jnp.bfloat16)
        output = norm(inputs)

        assert output.shape == inputs.shape
        assert output.dtype == jnp.bfloat16

    def test_mixed_precision(self, setup):
        """Test RMSNorm with mixed precision."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.bfloat16,
        )

        # Input in float16
        inputs = jnp.ones((setup["batch_size"], setup["dim"]), dtype=jnp.float16)
        output = norm(inputs)

        assert output.shape == inputs.shape
        # Output should preserve input dtype
        assert output.dtype == jnp.float16

    def test_float8_handling(self, setup):
        """Test RMSNorm with float8 dtypes."""
        if not hasattr(jnp, "float8_e4m3fn"):
            pytest.skip("Float8 not available in this JAX version")

        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float8_e4m3fn,
            param_dtype=jnp.float32,
        )

        inputs = jnp.ones((setup["batch_size"], setup["dim"]), dtype=jnp.float32)
        output = norm(inputs)

        assert output.shape == inputs.shape
        # Internal computation should handle float8 properly

    def test_custom_kernel_values(self, setup):
        """Test RMSNorm with custom kernel values."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        # Set custom kernel values
        norm.kernel.value = jnp.ones(setup["dim"]) * 2.0

        inputs = jnp.ones((setup["batch_size"], setup["dim"]))
        output = norm(inputs)

        # Output should be scaled by 2
        assert jnp.allclose(output, 2.0)

    def test_gradient_flow(self, setup):
        """Test gradient flow through RMSNorm."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        def loss_fn(inputs):
            output = norm(inputs)
            return jnp.mean(output**2)

        inputs = jnp.ones((setup["batch_size"], setup["dim"]))

        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(inputs)

        assert grads.shape == inputs.shape
        assert not jnp.all(grads == 0)  # Gradients should be non-zero

    def test_different_input_shapes(self, setup):
        """Test RMSNorm with various input shapes."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        # 1D input (single sample)
        input_1d = jnp.ones(setup["dim"])
        output_1d = norm(input_1d)
        assert output_1d.shape == input_1d.shape

        # 2D input (batch)
        input_2d = jnp.ones((setup["batch_size"], setup["dim"]))
        output_2d = norm(input_2d)
        assert output_2d.shape == input_2d.shape

        # 3D input (batch, sequence, dim)
        input_3d = jnp.ones((setup["batch_size"], setup["seq_len"], setup["dim"]))
        output_3d = norm(input_3d)
        assert output_3d.shape == input_3d.shape

        # 4D input (batch, heads, sequence, dim)
        input_4d = jnp.ones((setup["batch_size"], 8, setup["seq_len"], setup["dim"]))
        output_4d = norm(input_4d)
        assert output_4d.shape == input_4d.shape

    def test_numerical_equivalence(self, setup):
        """Test numerical equivalence with manual RMS norm computation."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        key = jax.random.PRNGKey(42)
        inputs = jax.random.normal(key, (setup["batch_size"], setup["dim"]))

        # Compute using RMSNorm layer
        output = norm(inputs)

        # Manual computation
        rms = jnp.sqrt(jnp.mean(inputs**2, axis=-1, keepdims=True) + setup["eps"])
        expected = (inputs / rms) * norm.kernel.value

        assert jnp.allclose(output, expected, rtol=1e-5)

    def test_dtype_preservation(self, setup):
        """Test that output dtype matches input dtype."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        # Test with different input dtypes
        for input_dtype in [jnp.float16, jnp.float32, jnp.bfloat16]:
            inputs = jnp.ones((setup["batch_size"], setup["dim"]), dtype=input_dtype)
            output = norm(inputs)
            assert output.dtype == input_dtype

    def test_large_dimension(self):
        """Test RMSNorm with large dimension."""
        dim = 4096
        norm = RMSNorm(
            dim=dim,
            eps=1e-6,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        inputs = jnp.ones((2, dim))
        output = norm(inputs)

        assert output.shape == inputs.shape
        assert jnp.allclose(output, 1.0)

    def test_small_epsilon(self, setup):
        """Test RMSNorm with very small epsilon."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=1e-12,  # Very small epsilon
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        inputs = jnp.ones((setup["batch_size"], setup["dim"]))
        output = norm(inputs)

        assert output.shape == inputs.shape
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_jit_compilation(self, setup):
        """Test that RMSNorm works with JIT compilation."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        @jax.jit
        def forward(x):
            return norm(x)

        inputs = jnp.ones((setup["batch_size"], setup["dim"]))
        output = forward(inputs)

        assert output.shape == inputs.shape
        assert jnp.allclose(output, 1.0)

    def test_vmap_compatibility(self, setup):
        """Test that RMSNorm works with vmap."""
        norm = RMSNorm(
            dim=setup["dim"],
            eps=setup["eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        # Create a batch of batches
        inputs = jnp.ones((4, setup["batch_size"], setup["dim"]))

        # vmap over the first dimension
        vmapped_norm = jax.vmap(norm, in_axes=0, out_axes=0)
        output = vmapped_norm(inputs)

        assert output.shape == inputs.shape
        assert jnp.allclose(output, 1.0)


class TestFloat8Constants:
    """Test float8 dtype constants."""

    def test_float8_dtype_properties(self):
        """Test properties of float8 dtypes."""
        for dtype in float8s:
            # Check that these are valid JAX dtypes
            if hasattr(jnp, str(dtype).split(".")[-1]):
                arr = jnp.array([1.0], dtype=dtype)
                assert arr.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
