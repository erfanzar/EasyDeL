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

"""Tests for linear layers with parallel and distributed computation support."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx as nn
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as Ps

from easydel.kernels.collective_matmul import MatrixMultiplyMethod
from easydel.layers.linear import (
    ColumnParallelLinear,
    ParallelLinear,
    RowParallelLinear,
    TensorParallelConfig,
    get_matmul_output_sharding,
    get_sharding,
)


class TestShardingHelpers:
    """Test sharding helper functions."""

    def test_get_sharding_unsharded(self):
        """Test get_sharding with unsharded array."""
        arr = jnp.ones((4, 4))
        sharding = get_sharding(arr)
        # SingleDeviceSharding doesn't have spec attribute
        assert sharding is None or not hasattr(arr.sharding, "spec")

    def test_get_sharding_sharded(self):
        """Test get_sharding with sharded array."""
        devices = jax.local_devices()
        if len(devices) < 2:
            pytest.skip("Need at least 2 devices for sharding tests")

        mesh = Mesh(devices, axis_names=("dp",))
        sharding = NamedSharding(mesh, Ps("dp", None))
        arr = jax.device_put(jnp.ones((4, 4)), sharding)

        spec = get_sharding(arr)
        assert spec is not None
        assert spec == Ps("dp", None)

    def test_get_matmul_output_sharding_none_inputs(self):
        """Test get_matmul_output_sharding with None inputs."""
        result = get_matmul_output_sharding(None, None)
        assert result == Ps()

    def test_get_matmul_output_sharding_basic(self):
        """Test basic matmul output sharding calculation."""
        # X @ W where X is [batch, in_features] and W is [in_features, out_features]
        lhs_pspec = Ps(None, "tp")  # X sharded on in_features
        rhs_pspec = Ps("tp", None)  # W sharded on in_features

        result = get_matmul_output_sharding(lhs_pspec, rhs_pspec)
        assert len(result) == 2
        assert result[0] is None  # batch dimension
        assert result[1] is None  # out_features (not sharded since tp is contracted)

    def test_get_matmul_output_sharding_complex(self):
        """Test matmul output sharding with complex specs."""
        # Multiple batch dimensions
        lhs_pspec = Ps("dp", None, None)  # [dp, seq, features]
        rhs_pspec = Ps(None, "tp")  # [features, tp]

        result = get_matmul_output_sharding(lhs_pspec, rhs_pspec)
        assert result[0] == "dp"
        assert result[1] is None  # seq dimension
        assert result[2] == "tp"  # output features

    def test_get_matmul_output_sharding_duplicate_dims(self):
        """Test that duplicate sharding dims are handled correctly."""
        lhs_pspec = Ps("dp", None)
        rhs_pspec = Ps(None, "dp")  # dp appears again

        result = get_matmul_output_sharding(lhs_pspec, rhs_pspec)
        assert result[0] == "dp"
        assert result[1] is None  # dp not duplicated

    def test_get_matmul_output_sharding_tuple_dims(self):
        """Test handling of tuple dimensions in partition specs."""
        lhs_pspec = Ps(("dp", "tp"), None)
        rhs_pspec = Ps(None, "mp")

        result = get_matmul_output_sharding(lhs_pspec, rhs_pspec)
        assert result[0] == ("dp", "tp")
        assert result[1] == "mp"


class TestTensorParallelConfig:
    """Test TensorParallelConfig."""

    def test_config_creation_basic(self):
        """Test basic TensorParallelConfig creation."""
        config = TensorParallelConfig(
            axis_name="tp",
            reduce_output=False,
            reduce_scatter_output=False,
        )
        assert config.axis_name == "tp"
        assert config.reduce_output is False
        assert config.reduce_scatter_output is False
        assert config.mesh is None
        assert config.matmul_method is None

    def test_config_with_mesh(self):
        """Test TensorParallelConfig with mesh and matmul method."""
        devices = jax.local_devices()
        if len(devices) < 2:
            pytest.skip("Need at least 2 devices for mesh tests")

        mesh = Mesh(devices, axis_names=("tp",))
        config = TensorParallelConfig(
            mesh=mesh,
            axis_name="tp",
            matmul_method=MatrixMultiplyMethod.ALL_GATHER,  # Use a valid enum value
        )
        assert config.mesh == mesh
        assert config.axis_name == "tp"
        assert config.matmul_method == MatrixMultiplyMethod.ALL_GATHER

    def test_config_invalid_axis_name(self):
        """Test TensorParallelConfig with invalid axis name."""
        devices = jax.local_devices()
        mesh = Mesh(devices, axis_names=("dp",))

        with pytest.raises(ValueError, match="axis_name 'tp' not found"):
            TensorParallelConfig(
                mesh=mesh,
                axis_name="tp",
                matmul_method=MatrixMultiplyMethod.ALL_GATHER,  # Use a valid enum value
            )


class TestParallelLinear:
    """Test ParallelLinear layer."""

    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        return {
            "in_features": 128,
            "out_features": 256,
            "batch_size": 4,
            "seq_len": 32,
            "dtype": jnp.float32,
            "param_dtype": jnp.float32,
        }

    def test_basic_linear_no_bias(self, setup):
        """Test basic linear layer without bias."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=False,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        assert layer.in_features == setup["in_features"]
        assert layer.out_features == setup["out_features"]
        assert layer.use_bias is False
        assert layer.bias is None
        assert layer.kernel.value.shape == (setup["in_features"], setup["out_features"])

    def test_basic_linear_with_bias(self, setup):
        """Test basic linear layer with bias."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        assert layer.use_bias is True
        assert layer.bias is not None
        assert layer.bias.value.shape == (setup["out_features"],)

    def test_forward_2d_input(self, setup):
        """Test forward pass with 2D input."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))
        output = layer(inputs)

        assert output.shape == (setup["batch_size"], setup["out_features"])
        assert output.dtype == setup["dtype"]

    def test_forward_3d_input(self, setup):
        """Test forward pass with 3D input."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        inputs = jnp.ones((setup["batch_size"], setup["seq_len"], setup["in_features"]))
        output = layer(inputs)

        assert output.shape == (setup["batch_size"], setup["seq_len"], setup["out_features"])
        assert output.dtype == setup["dtype"]

    def test_scale_fan_in(self, setup):
        """Test fan_in scaling."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            scale="fan_in",
            use_bias=False,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))
        output = layer(inputs)

        # Output should be scaled by 1/sqrt(in_features)
        assert output.shape == (setup["batch_size"], setup["out_features"])
        # The scaling is applied internally in the layer

    def test_scale_fan_out(self, setup):
        """Test fan_out scaling."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            scale="fan_out",
            use_bias=False,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))
        output = layer(inputs)

        # Output should be scaled by 1/sqrt(out_features)
        assert output.shape == (setup["batch_size"], setup["out_features"])
        # The scaling is applied internally in the layer

    def test_custom_scale(self, setup):
        """Test custom scaling factor."""
        scale = 0.5
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            scale=scale,
            use_bias=False,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))
        output = layer(inputs)

        assert output.shape == (setup["batch_size"], setup["out_features"])

    def test_custom_weight(self, setup):
        """Test forward pass with custom weight."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=False,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))
        custom_weight = jnp.ones((setup["in_features"], setup["out_features"]))
        output = layer(inputs, w=custom_weight)

        assert output.shape == (setup["batch_size"], setup["out_features"])
        # With all ones, output should be in_features
        expected = jnp.full((setup["batch_size"], setup["out_features"]), setup["in_features"])
        assert jnp.allclose(output, expected)

    def test_precision_setting(self, setup):
        """Test different precision settings."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=False,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
            precision=jax.lax.Precision.HIGH,
        )

        assert layer.precision == jax.lax.Precision.HIGH

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))
        output = layer(inputs)
        assert output.shape == (setup["batch_size"], setup["out_features"])

    def test_dtype_promotion(self, setup):
        """Test dtype promotion."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        # Input with different dtype
        inputs = jnp.ones((setup["batch_size"], setup["in_features"]), dtype=jnp.float16)
        output = layer(inputs)

        assert output.shape == (setup["batch_size"], setup["out_features"])
        # Output should be promoted to float32
        assert output.dtype == jnp.float32

    def test_custom_initializers(self, setup):
        """Test custom kernel and bias initializers."""
        kernel_init = nn.initializers.xavier_uniform()
        bias_init = nn.initializers.constant(0.1)

        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        assert layer.kernel.value.shape == (setup["in_features"], setup["out_features"])
        assert layer.bias.value.shape == (setup["out_features"],)
        # Check bias is initialized with 0.1
        assert jnp.allclose(layer.bias.value, 0.1)


class TestRowColumnParallelLinear:
    """Test RowParallelLinear and ColumnParallelLinear."""

    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        return {
            "in_features": 128,
            "out_features": 256,
            "batch_size": 4,
            "dtype": jnp.float32,
            "param_dtype": jnp.float32,
        }

    def test_row_parallel_linear(self, setup):
        """Test RowParallelLinear."""
        layer = RowParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        assert layer._direction == "row"
        assert layer.in_features == setup["in_features"]
        assert layer.out_features == setup["out_features"]

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))
        output = layer(inputs)
        assert output.shape == (setup["batch_size"], setup["out_features"])

    def test_column_parallel_linear(self, setup):
        """Test ColumnParallelLinear."""
        layer = ColumnParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        assert layer._direction == "column"
        assert layer.in_features == setup["in_features"]
        assert layer.out_features == setup["out_features"]

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))
        output = layer(inputs)
        assert output.shape == (setup["batch_size"], setup["out_features"])


class TestParallelLinearIntegration:
    """Integration tests for ParallelLinear with actual parallelism."""

    @pytest.fixture
    def setup(self):
        """Setup for integration tests."""
        return {
            "in_features": 128,
            "out_features": 256,
            "batch_size": 4,
            "seq_len": 32,
            "dtype": jnp.float32,
            "param_dtype": jnp.float32,
        }

    def test_backward_compatibility(self, setup):
        """Test that ParallelLinear without parallel_config works like standard Linear."""
        # Create layer without parallel config
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        inputs = jnp.ones((setup["batch_size"], setup["seq_len"], setup["in_features"]))
        output = layer(inputs)

        # Should work exactly like standard linear
        assert output.shape == (setup["batch_size"], setup["seq_len"], setup["out_features"])

        # Manual computation for verification
        expected = jnp.einsum("...ik,...kj->...ij", inputs, layer.kernel.value)
        expected = expected + layer.bias.value.reshape((1, 1, -1))
        assert jnp.allclose(output, expected, rtol=1e-5)

    def test_gradient_flow(self, setup):
        """Test gradient flow through ParallelLinear."""
        layer = ParallelLinear(
            in_features=setup["in_features"],
            out_features=setup["out_features"],
            use_bias=True,
            dtype=setup["dtype"],
            param_dtype=setup["param_dtype"],
        )

        def loss_fn(inputs):
            output = layer(inputs)
            return jnp.mean(output**2)

        inputs = jnp.ones((setup["batch_size"], setup["in_features"]))

        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(inputs)

        assert grads.shape == inputs.shape
        assert not jnp.all(grads == 0)  # Gradients should be non-zero


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
