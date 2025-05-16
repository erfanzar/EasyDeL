import os
# Set JAX to use CPU to avoid conflicts with other processes using TPU
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import unittest
from easydel.infra.utils import ACT2FN


class TestActivations(unittest.TestCase):
    """Test activation functions in EasyDeL."""

    def test_relu_squared(self):
        """Test relu_squared activation function."""
        # Create a test input
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Apply relu_squared function
        result = ACT2FN["relu_squared"](x)
        
        # Expected output: the square of ReLU values
        expected = jnp.array([0.0, 0.0, 0.0, 1.0, 4.0])
        
        # Check if output matches expected values
        np.testing.assert_allclose(result, expected)

    def test_other_activations(self):
        """Test other activation functions to ensure they still work."""
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Test relu
        relu_result = ACT2FN["relu"](x)
        expected_relu = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(relu_result, expected_relu)
        
        # Test sigmoid
        sigmoid_result = ACT2FN["sigmoid"](x)
        expected_sigmoid = 1.0 / (1.0 + jnp.exp(-x))
        np.testing.assert_allclose(sigmoid_result, expected_sigmoid)


if __name__ == "__main__":
    unittest.main() 