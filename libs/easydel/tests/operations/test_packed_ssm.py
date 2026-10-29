"""Public packed scans preserve ragged state transitions (accelerator-only)."""

import math

import numpy as np
import pytest
from easydel.operations.kernels.ssm1 import packed_ssm1_fwd
from easydel.operations.kernels.ssm2 import packed_ssm2_fwd
from jax import numpy as jnp


@pytest.mark.parametrize("version", [1, 2])
def test_public_packed_scan_preserves_padding_and_final_state(version):
    inputs = np.array([[1.0, 9.0, 2.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    valid = np.array([[True, False, True], [False, False, False]])
    shape = (2, 3, 1) if version == 1 else (2, 3, 1, 1)
    state_shape = (2, 1, 1) if version == 1 else (2, 1, 1, 1)
    x = jnp.asarray(inputs.reshape(shape))
    coefficients = jnp.ones((2, 3, 1) if version == 1 else (2, 3, 1, 1))
    a = jnp.full((1, 1) if version == 1 else (1,), -1.0)
    dt = jnp.full((2, 3, 1), 0.5)
    initial = jnp.asarray(np.array([0.25, 0.75], dtype=np.float32).reshape(state_shape))
    scan = packed_ssm1_fwd if version == 1 else packed_ssm2_fwd
    output, state = scan(x, a, coefficients, coefficients, jnp.ones((1,)), dt, jnp.asarray(valid), initial)
    expected_state = np.array([0.25, 0.75], dtype=np.float32)
    expected_output = np.zeros_like(inputs)
    for slot in range(2):
        for col in range(3):
            if valid[slot, col]:
                expected_state[slot] = math.exp(-0.5) * expected_state[slot] + 0.5 * inputs[slot, col]
                expected_output[slot, col] = expected_state[slot] + inputs[slot, col]
    np.testing.assert_allclose(np.asarray(output).reshape(2, 3), expected_output, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(state).reshape(2), expected_state, rtol=1e-5)
    assert state.dtype == jnp.float32
