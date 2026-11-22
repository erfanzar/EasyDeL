"""Independent SSM1 scan, cache and packed-boundary references (accelerator only)."""

import jax
import numpy as np
import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.operations import OperationMetadata, OperationRegistry
from easydel.operations.kernels.ssm1 import packed_ssm1_fwd
from jax import numpy as jnp


def _inputs(length, dtype):
    rng = np.random.default_rng(614)
    shapes = ((3, length, 2), (2, 3), (3, length, 3), (3, length, 3), (2,), (3, length, 2))
    values = [rng.uniform(0.05, 0.5, shape).astype(np.float32) for shape in shapes]
    gate = rng.uniform(-1.0, 1.0, (3, length, 2)).astype(np.float32)
    initial = rng.uniform(-0.5, 0.5, (3, 2, 3)).astype(np.float32)
    # Quantize the independent reference inputs exactly as they enter the operation.
    arrays = [jnp.asarray(value, dtype=dtype) for value in (*values, gate, initial)]
    return arrays, [np.asarray(value, dtype=np.float64) for value in arrays]


def _reference(values, segments=None, valid=None, real_a=False):
    x, a, b, c, d, dt, gate, initial = values
    a = a if real_a else -np.exp(a)
    state = initial.copy()
    result = np.zeros_like(x)
    previous = [None] * len(x)
    seen = [False] * len(x)
    for col in range(x.shape[1]):
        for row in range(x.shape[0]):
            is_valid = (
                (segments[row, col] >= 0) if segments is not None else (True if valid is None else bool(valid[row, col]))
            )
            if not is_valid:
                if seen[row]:
                    previous[row] = None
                continue
            if segments is not None:
                segment = int(segments[row, col])
                if seen[row] and segment != previous[row]:
                    state[row] = 0.0
                previous[row] = segment
            seen[row] = True
            state[row] = np.exp(dt[row, col, :, None] * a) * state[row] + (
                dt[row, col, :, None] * b[row, col, None, :] * x[row, col, :, None]
            )
            y = np.sum(state[row] * c[row, col, None, :], axis=-1) + d * x[row, col]
            result[row, col] = y * gate[row, col] / (1.0 + np.exp(-gate[row, col]))
    return result, state


@pytest.mark.parametrize(
    "associative,mambapy", [(False, False), (True, False), (False, True), (True, True), (None, True)]
)
@pytest.mark.parametrize("length", [0, 1, 5])
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_registered_ssm1_scan_matches_independent_recurrence(associative, mambapy, length, segmented, dtype):
    arrays, values = _inputs(length, dtype)
    x, a, b, c, d, dt, gate, initial = arrays
    segments = np.array([[4, 4, 5, 5, -1], [-1, 7, 7, -1, 8], [-1, -1, -1, -1, -1]], dtype=np.int32)[:, :length]
    segment_ids = jnp.asarray(segments) if segmented else None
    operation = OperationRegistry.create(
        "ssm1",
        OperationMetadata(runtime_dtype=dtype, base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, -1, 1, 1, 1))),
    )
    conv_state = jnp.ones((3, 2, 2), dtype=dtype)
    output = jax.jit(
        lambda *args: operation(
            *args,
            gate=gate,
            ssm_state=initial,
            conv_state=conv_state,
            segment_ids=segment_ids,
            use_associative_scan=associative,
            use_mambapy=mambapy,
        )
    )(x, a, b, c, d, dt)
    expected_y, expected_state = _reference(values, segments if segmented else None)
    tolerance = 8e-3 if dtype == jnp.bfloat16 else 2e-5
    np.testing.assert_allclose(
        np.asarray(output.attention_outputs, dtype=np.float32), expected_y, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(output.ssm_state, dtype=np.float32), expected_state, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_array_equal(output.conv_state, conv_state)
    assert output.attention_outputs.dtype == dtype
    assert output.ssm_state.dtype == dtype


@pytest.mark.parametrize("associative", [False, True])
@pytest.mark.parametrize("length", [0, 1, 5])
def test_packed_ssm1_selected_scan_preserves_invalid_cells_and_idle_slots(associative, length):
    arrays, values = _inputs(length, jnp.float32)
    x, a, b, c, d, dt, _, initial = arrays
    valid = np.array([[True, False, True, True, False], [False, True, True, False, True], [False] * 5])[:, :length]
    # Poison padding: the packed public API must implement identity transitions,
    # rather than depending on callers having zero-filled the padded operands.
    x = jnp.where(jnp.asarray(valid[..., None]), x, jnp.nan)
    dt = jnp.where(jnp.asarray(valid[..., None]), dt, jnp.nan)
    output, state = jax.jit(
        lambda x, dt: packed_ssm1_fwd(
            x, -jnp.exp(a), b, c, d, dt, jnp.asarray(valid), initial, use_associative_scan=associative
        )
    )(x, dt)
    values[1] = -np.exp(values[1])
    # A constant gate chosen so SiLU(gate) == 1 is unnecessary: compare the
    # ungated packed recurrence directly using a separate scalar loop.
    expected_state = values[-1].copy()
    expected_output = np.zeros_like(values[0])
    for col in range(length):
        for row in range(3):
            if valid[row, col]:
                expected_state[row] = np.exp(values[5][row, col, :, None] * values[1]) * expected_state[row] + (
                    values[5][row, col, :, None] * values[2][row, col, None, :] * values[0][row, col, :, None]
                )
                expected_output[row, col] = np.sum(expected_state[row] * values[3][row, col, None, :], axis=-1) + (
                    values[4] * values[0][row, col]
                )
    np.testing.assert_allclose(output, expected_output, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(state, expected_state, rtol=2e-5, atol=2e-6)
    assert state.dtype == jnp.float32


@pytest.mark.parametrize("associative", [False, True])
def test_selected_scan_continues_cache_across_decode_chunks(associative):
    arrays, values = _inputs(5, jnp.float32)
    x, a, b, c, d, dt, gate, initial = arrays
    operation = OperationRegistry.create(
        "ssm1",
        OperationMetadata(
            runtime_dtype=jnp.float32, base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, -1, 1, 1, 1))
        ),
    )
    state = initial
    outputs = []
    for col in range(5):
        result = operation(
            x[:, col : col + 1],
            a,
            b[:, col : col + 1],
            c[:, col : col + 1],
            d,
            dt[:, col : col + 1],
            gate=gate[:, col : col + 1],
            ssm_state=state,
            segment_ids=jnp.full((3, 1), 7),
            use_associative_scan=associative,
        )
        outputs.append(result.attention_outputs)
        state = result.ssm_state
    expected_y, expected_state = _reference(values)
    np.testing.assert_allclose(jnp.concatenate(outputs, axis=1), expected_y, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(state, expected_state, rtol=2e-5, atol=2e-6)
