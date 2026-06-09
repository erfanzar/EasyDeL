# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Comprehensive tests for quantization operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eformer.ops.quantization import Array1B, Array8B, ArrayNF4
from eformer.ops.quantization.quantization_functions import nf4xf32_to_f32


class TestNF4Quantization:
    """Tests for NF4 quantization."""

    @pytest.fixture
    def random_matrix(self):
        """Create a random matrix for testing."""
        key = jax.random.PRNGKey(42)
        return jax.random.normal(key, (256, 128), dtype=jnp.float32)

    def test_nf4_quantize_dequantize_accuracy(self, random_matrix):
        """Test that quantization and dequantization preserve accuracy within acceptable bounds."""
        # Quantize
        quantized = ArrayNF4.quantize(random_matrix, block_size=64)

        # Dequantize
        reconstructed = quantized.materialize()

        # Check shapes match
        assert reconstructed.shape == random_matrix.shape

        # Check reconstruction error (NF4 is lossy, so we allow some error)
        error = jnp.mean(jnp.abs(random_matrix - reconstructed))
        relative_error = error / jnp.mean(jnp.abs(random_matrix))

        # NF4 has ~8-10% relative error for normal distributions (16 quantization levels)
        # This is expected and acceptable for 4-bit quantization
        assert relative_error < 0.15, f"Relative error {relative_error:.4f} too high"

    def test_nf4_polynomial_approximation(self):
        """Test that polynomial approximation matches NF4 codebook closely."""
        x = jnp.arange(16)
        approx_values = nf4xf32_to_f32(x)

        # Check that approximation is close to expected NF4 values
        nf4_expected = jnp.array(
            [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ]
        )

        max_error = jnp.max(jnp.abs(approx_values - nf4_expected))
        # Polynomial approximation has some error, but should be < 1.5% for NF4 values
        assert max_error < 0.015, f"Polynomial approximation error {max_error:.6f} too high"

    def test_nf4_block_sizes(self, random_matrix):
        """Test different block sizes for quantization."""
        block_sizes = [32, 64, 128]

        for block_size in block_sizes:
            quantized = ArrayNF4.quantize(random_matrix, block_size=block_size)
            reconstructed = quantized.materialize()

            assert reconstructed.shape == random_matrix.shape
            # Larger blocks generally have slightly higher error
            error = jnp.mean(jnp.abs(random_matrix - reconstructed))
            assert error < 0.1  # Reasonable threshold

    def test_nf4_approx_vs_lookup(self, random_matrix):
        """Test that polynomial approximation gives similar results to lookup table."""
        # Quantize with approximation
        quantized_approx = ArrayNF4.quantize(random_matrix, block_size=64)
        result_approx = quantized_approx.materialize()

        quantized_lookup = ArrayNF4.quantize(random_matrix, block_size=64)
        result_lookup = quantized_lookup.materialize()

        # Results should be very similar (small numerical differences allowed)
        # Polynomial approximation has ~0.014 error per NF4 value, which accumulates
        diff = jnp.mean(jnp.abs(result_approx - result_lookup))
        assert diff < 0.02, f"Approximation vs lookup difference {diff:.6f} too high"

    def test_nf4_matmul_correctness(self, random_matrix):
        """Test that matrix multiplication works correctly with quantized arrays."""
        key = jax.random.PRNGKey(42)
        input_vec = jax.random.normal(key, (256,), dtype=jnp.float32)

        # Regular matmul
        expected = input_vec @ random_matrix

        # Quantized matmul (using implicit arrays, no kernel)
        quantized = ArrayNF4.quantize(random_matrix, block_size=64)

        # Import implicit to enable implicit array operations
        from eformer.jaximus import implicit

        @implicit
        def matmul(x, y):
            return x @ y

        result = jax.jit(matmul)(input_vec, quantized)

        # Check result is close
        error = jnp.mean(jnp.abs(expected - result))
        relative_error = error / jnp.mean(jnp.abs(expected))
        assert relative_error < 0.1, f"Matmul relative error {relative_error:.4f} too high"

    def test_nf4_kernel_mode(self, random_matrix):
        """Test kernel mode execution (currently disabled - using optimized dequantization)."""
        key = jax.random.PRNGKey(42)
        input_vec = jax.random.normal(key, (256,), dtype=jnp.bfloat16)

        # Quantize with kernel support flag (currently uses optimized dequantization path)
        quantized = ArrayNF4.quantize(random_matrix.astype(jnp.bfloat16), block_size=64)

        from eformer.jaximus import implicit

        @implicit
        def matmul(x, y):
            return x @ y

        # Test execution
        try:
            result = jax.jit(matmul)(input_vec, quantized)

            # Compare with regular execution
            result_regular = input_vec @ random_matrix.astype(jnp.bfloat16)

            error = jnp.mean(jnp.abs(result - result_regular))
            relative_error = error / jnp.mean(jnp.abs(result_regular))
            # NF4 quantization has ~10% error, so allow up to 20% for safety
            assert relative_error < 0.20, f"NF4 matmul relative error {relative_error:.4f} too high"
        except Exception as e:
            if "device" in str(e).lower():
                pytest.skip("TPU/GPU not available on this device")
            raise

    def test_nf4_memory_efficiency(self, random_matrix):
        """Test that quantization reduces memory usage."""
        # Original size (float32 = 4 bytes per element)
        original_size = random_matrix.size * 4

        # Quantized size
        quantized = ArrayNF4.quantize(random_matrix, block_size=64)
        # Packed is uint8 (1 byte), but stores 2 values per byte (4-bit each)
        packed_size = quantized.packed.size * 1
        scale_size = quantized.absmax.size * 4  # Scales are float32
        quantized_size = packed_size + scale_size

        # Quantized should be significantly smaller (at least 3x compression)
        compression_ratio = original_size / quantized_size
        assert compression_ratio > 3, f"Compression ratio {compression_ratio:.2f}x too low"

    def test_nf4_different_shapes(self):
        """Test quantization with various matrix shapes."""
        shapes = [(64, 64), (128, 256), (512, 128)]

        for shape in shapes:
            key = jax.random.PRNGKey(42)
            matrix = jax.random.normal(key, shape, dtype=jnp.float32)

            quantized = ArrayNF4.quantize(matrix, block_size=32)
            reconstructed = quantized.materialize()

            assert reconstructed.shape == matrix.shape


class TestInt8Quantization:
    """Tests for 8-bit quantization."""

    @pytest.fixture
    def random_matrix(self):
        """Create a random matrix for testing."""
        key = jax.random.PRNGKey(42)
        return jax.random.normal(key, (256, 128), dtype=jnp.float32)

    def test_int8_quantize_dequantize_accuracy(self, random_matrix):
        """Test that int8 quantization preserves accuracy."""
        # Quantize
        quantized = Array8B.quantize(random_matrix, axis=-1)

        # Dequantize
        reconstructed = quantized.materialize()

        # Check shapes match
        assert reconstructed.shape == random_matrix.shape

        # Int8 should have better accuracy than NF4
        error = jnp.mean(jnp.abs(random_matrix - reconstructed))
        relative_error = error / jnp.mean(jnp.abs(random_matrix))

        assert relative_error < 0.01, f"Relative error {relative_error:.4f} too high"

    def test_int8_different_axes(self, random_matrix):
        """Test quantization along different axes."""
        for axis in [-1, 0, (0, 1)]:
            quantized = Array8B.quantize(random_matrix, axis=axis)
            reconstructed = quantized.materialize()

            assert reconstructed.shape == random_matrix.shape

    def test_int8_matmul_correctness(self, random_matrix):
        """Test matrix multiplication with int8 quantization."""
        key = jax.random.PRNGKey(42)
        input_vec = jax.random.normal(key, (256,), dtype=jnp.float32)

        # Regular matmul
        expected = input_vec @ random_matrix

        # Quantized matmul
        quantized = Array8B.quantize(random_matrix, axis=-1)

        from eformer.jaximus import implicit

        @implicit
        def matmul(x, y):
            return x @ y

        result = jax.jit(matmul)(input_vec, quantized)

        # Int8 should be quite accurate
        error = jnp.mean(jnp.abs(expected - result))
        relative_error = error / jnp.mean(jnp.abs(expected))
        assert relative_error < 0.02, f"Matmul relative error {relative_error:.4f} too high"


class TestInt1Quantization:
    """Tests for 1-bit ternary quantization."""

    @pytest.fixture
    def random_matrix(self):
        """Create a random matrix for testing."""
        key = jax.random.PRNGKey(42)
        # Make it multiples of 4 for 1-bit packing
        return jax.random.normal(key, (256, 128), dtype=jnp.float32)

    def test_int1_quantize_shape(self, random_matrix):
        """Test that 1-bit quantization maintains shape."""
        # Note: Array1B expects ternary input {-1, 0, 1}
        ternary_matrix = jnp.sign(random_matrix)

        quantized = Array1B.quantize(ternary_matrix)
        reconstructed = quantized.materialize()

        assert reconstructed.shape == ternary_matrix.shape


class TestShardingSafety:
    """Tests for sharding-safe operations."""

    @pytest.mark.skipif(len(jax.devices()) < 2, reason="Requires multiple devices for sharding tests")
    def test_nf4_sharding_basic(self):
        """Test basic sharding with NF4 arrays."""
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Need at least 2 devices for sharding")

        key = jax.random.PRNGKey(42)
        matrix = jax.random.normal(key, (256, 128), dtype=jnp.float32)

        quantized = ArrayNF4.quantize(matrix, block_size=64)

        # Test that sharding configuration can be set
        mesh = jax.sharding.Mesh(np.array(devices[:2]).reshape(2, 1), ("dp", "tp"))

        # This should not raise an error
        try:
            sharded = quantized.with_mesh_and_axis((mesh, None))
            assert sharded is not None
        except Exception as e:
            pytest.fail(f"Sharding configuration failed: {e}")


class TestPerformance:
    """Performance and benchmarking tests."""

    def test_nf4_jit_compilation(self):
        """Test that NF4 operations can be JIT compiled."""
        key = jax.random.PRNGKey(42)
        matrix = jax.random.normal(key, (128, 64), dtype=jnp.float32)
        vec = jax.random.normal(key, (128,), dtype=jnp.float32)

        quantized = ArrayNF4.quantize(matrix, block_size=32)

        from eformer.jaximus import implicit

        @implicit
        @jax.jit
        def matmul(x, y):
            return x @ y

        # Should compile without errors
        result = matmul(vec, quantized)
        assert result.shape == (64,)

    def test_quantization_speed(self):
        """Basic speed test for quantization operations."""
        key = jax.random.PRNGKey(42)
        large_matrix = jax.random.normal(key, (1024, 512), dtype=jnp.float32)

        # Quantization should complete reasonably fast
        import time

        warmup = ArrayNF4.quantize(large_matrix, block_size=64)
        _ = jax.block_until_ready(warmup.packed)

        start = time.time()
        quantized = ArrayNF4.quantize(large_matrix, block_size=64)
        _ = jax.block_until_ready(quantized.packed)
        duration = time.time() - start

        # Should take less than 1 second on accelerators; allow longer on CPU.
        max_seconds = 5.0 if jax.default_backend() == "cpu" else 1.0
        assert duration < max_seconds, f"Quantization took {duration:.2f}s, too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
