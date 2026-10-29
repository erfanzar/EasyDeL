"""Numerical contracts for packed-grid SSM scans used by Falcon-Mamba and Falcon-H1."""

import jax.numpy as jnp
import pytest
from easydel.operations.kernels.ssm1 import _single_step_ssm1_fwd, packed_ssm1_fwd
from easydel.operations.kernels.ssm2 import _single_step_ssm2_fwd, packed_ssm2_fwd


@pytest.mark.parametrize("lengths", [(3, 1, 2), (1, 0, 2)])
def test_falcon_mamba_packed_scan_matches_single_token_reference(lengths):
    """Ragged slots, including an empty row, preserve initial state through grid padding."""
    slots, width, channels, state_size = 3, 4, 2, 3
    starts = jnp.asarray([0, lengths[0], lengths[0] + lengths[1], sum(lengths)], dtype=jnp.int32)
    packed_count = sum(lengths)
    x_packed = jnp.arange(packed_count * channels, dtype=jnp.float32).reshape(packed_count, channels) / 7
    b_packed = jnp.ones((packed_count, state_size), dtype=jnp.float32) * 0.2
    c_packed = jnp.ones_like(b_packed) * 0.3
    dt_packed = jnp.ones((packed_count, channels), dtype=jnp.float32) * 0.1
    token = jnp.arange(packed_count, dtype=jnp.int32)
    token_slot = jnp.clip(jnp.searchsorted(starts, token, side="right") - 1, 0, slots - 1)
    token_col = token - starts[token_slot]
    valid_tokens = token < starts[-1]

    def grid(values):
        return jnp.zeros((slots, width, *values.shape[1:]), values.dtype).at[token_slot, token_col].set(values)

    valid = grid(valid_tokens).astype(jnp.bool_)
    x, b, c, dt = (grid(value) for value in (x_packed, b_packed, c_packed, dt_packed))
    a = -jnp.ones((channels, state_size), dtype=jnp.float32)
    d = jnp.ones((channels,), dtype=jnp.float32)
    init = jnp.arange(slots * channels * state_size, dtype=jnp.float32).reshape(slots, channels, state_size) / 20

    actual_y, actual_state = packed_ssm1_fwd(x, a, b, c, d, dt, valid, init)
    expected_y = jnp.zeros_like(actual_y)
    expected_state = init
    for slot, length in enumerate(lengths):
        state = init[slot : slot + 1]
        for col in range(length):
            idx = int(starts[slot]) + col
            out, state = _single_step_ssm1_fwd(
                x_packed[idx : idx + 1],
                a,
                b_packed[idx : idx + 1],
                c_packed[idx : idx + 1],
                d,
                dt_packed[idx : idx + 1],
                state,
            )
            expected_y = expected_y.at[slot, col].set(out[0])
        expected_state = expected_state.at[slot].set(state[0])

    assert jnp.allclose(actual_y, expected_y, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(actual_state, expected_state, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("lengths", [(3, 1, 2), (1, 0, 2)])
def test_falcon_h1_packed_scan_matches_single_token_reference(lengths):
    """SSM2's slot-major scan matches decode order from each slot's cached state."""
    slots, width, heads, head_dim, state_size, groups = 3, 4, 2, 2, 3, 1
    starts = jnp.asarray([0, lengths[0], lengths[0] + lengths[1], sum(lengths)], dtype=jnp.int32)
    packed_count = sum(lengths)
    x_packed = jnp.arange(packed_count * heads * head_dim, dtype=jnp.float32).reshape(packed_count, heads, head_dim) / 10
    b_packed = jnp.ones((packed_count, groups, state_size), dtype=jnp.float32) * 0.2
    c_packed = jnp.ones_like(b_packed) * 0.3
    dt_packed = jnp.ones((packed_count, heads), dtype=jnp.float32) * 0.1
    token = jnp.arange(packed_count, dtype=jnp.int32)
    token_slot = jnp.clip(jnp.searchsorted(starts, token, side="right") - 1, 0, slots - 1)
    token_col = token - starts[token_slot]

    def grid(values):
        return jnp.zeros((slots, width, *values.shape[1:]), values.dtype).at[token_slot, token_col].set(values)

    valid = grid(token < starts[-1]).astype(jnp.bool_)
    x, b, c, dt = (grid(value) for value in (x_packed, b_packed, c_packed, dt_packed))
    a = -jnp.ones((heads,), dtype=jnp.float32)
    d = jnp.ones((heads,), dtype=jnp.float32)
    init = (
        jnp.arange(slots * heads * head_dim * state_size, dtype=jnp.float32).reshape(slots, heads, head_dim, state_size)
        / 30
    )

    actual_y, actual_state = packed_ssm2_fwd(x, a, b, c, d, dt, valid, init, groups)
    expected_y = jnp.zeros_like(actual_y)
    expected_state = init
    for slot, length in enumerate(lengths):
        state = init[slot : slot + 1]
        for col in range(length):
            idx = int(starts[slot]) + col
            out, state = _single_step_ssm2_fwd(
                x_packed[idx : idx + 1],
                a,
                b_packed[idx : idx + 1],
                c_packed[idx : idx + 1],
                d,
                dt_packed[idx : idx + 1],
                state,
                groups,
            )
            expected_y = expected_y.at[slot, col].set(out[0])
        expected_state = expected_state.at[slot].set(state[0])

    assert jnp.allclose(actual_y, expected_y, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(actual_state, expected_state, rtol=1e-6, atol=1e-6)


def test_packed_conv_cache_tail_matches_tokenwise_roll_for_ragged_slots():
    """Packed convolution windows reproduce per-slot rolling cache updates and leave empty slots untouched."""
    lengths = (3, 1, 2)
    slots, width, channels, kernel_size = 3, 4, 2, 3
    starts = jnp.asarray([0, 3, 4, 6], dtype=jnp.int32)
    inputs = jnp.arange(12, dtype=jnp.float32).reshape(6, channels)
    token = jnp.arange(6, dtype=jnp.int32)
    slot_of_token = jnp.clip(jnp.searchsorted(starts, token, side="right") - 1, 0, slots - 1)
    col_of_token = token - starts[slot_of_token]
    grid = jnp.zeros((slots, width, channels), dtype=jnp.float32).at[slot_of_token, col_of_token].set(inputs)
    initial = jnp.arange(slots * channels * kernel_size, dtype=jnp.float32).reshape(slots, channels, kernel_size)
    window = jnp.concatenate((initial.transpose(0, 2, 1), grid), axis=1)
    kernel = jnp.arange(kernel_size * channels, dtype=jnp.float32).reshape(channels, kernel_size) / 10
    windows = jnp.stack([window[:, i + 1 : i + 1 + kernel_size] for i in range(width)], axis=1)
    conv = jnp.einsum("swki,ik->swi", windows, kernel)
    actual_tail = initial
    expected_tail = initial
    for slot, length in enumerate(lengths):
        state = initial[slot]
        for col in range(length):
            state = jnp.concatenate((state[:, 1:], grid[slot, col : col + 1].T), axis=-1)
            expected_tail = expected_tail.at[slot].set(state)
            expected = jnp.sum(state * kernel, axis=-1)
            assert jnp.allclose(conv[slot, col], expected, rtol=1e-6, atol=1e-6)
    counts = jnp.asarray(lengths, dtype=jnp.int32)
    positions = counts[:, None] + jnp.arange(kernel_size)[None, :]
    actual_tail = jnp.take_along_axis(window, positions[..., None], axis=1).transpose(0, 2, 1)
    actual_tail = jnp.where((counts > 0)[:, None, None], actual_tail, initial)
    assert jnp.allclose(actual_tail, expected_tail, rtol=1e-6, atol=1e-6)
