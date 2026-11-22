"""Cached packed-convolution regressions; requires accelerator execution."""

import pytest
from easydel.infra.sequence_packing import segmented_depthwise_causal_conv1d
from jax import numpy as jnp


@pytest.fixture
def kernel():
    return jnp.array([[1.0, 10.0, 100.0]], dtype=jnp.float32)


def test_initial_state_is_preserved_on_first_valid_token(kernel):
    x = jnp.array([[[4.0]]], dtype=jnp.float32)
    segment_ids = jnp.array([[0]], dtype=jnp.int32)
    initial_state = jnp.array([[[1.0, 2.0, 3.0]]], dtype=jnp.float32)

    output, final_state = segmented_depthwise_causal_conv1d(x, kernel, segment_ids, initial_state=initial_state)

    # Window rolls to [2, 3, 4]; 2*1 + 3*10 + 4*100.
    assert jnp.allclose(output, jnp.array([[[432.0]]]))
    assert jnp.allclose(final_state, jnp.array([[[2.0, 3.0, 4.0]]]))


def test_initial_segment_id_mismatch_resets_the_cached_window(kernel):
    x = jnp.array([[[4.0]]], dtype=jnp.float32)
    segment_ids = jnp.array([[0]], dtype=jnp.int32)
    initial_state = jnp.array([[[1.0, 2.0, 3.0]]], dtype=jnp.float32)

    output, _ = segmented_depthwise_causal_conv1d(
        x, kernel, segment_ids, initial_state=initial_state, initial_segment_id=-1
    )

    assert jnp.allclose(output, jnp.array([[[400.0]]]))


def test_initial_state_survives_leading_padding_and_resets_at_next_segment(kernel):
    x = jnp.array([[[9.0], [4.0], [5.0]]], dtype=jnp.float32)
    segment_ids = jnp.array([[-1, 0, 1]], dtype=jnp.int32)
    initial_state = jnp.array([[[1.0, 2.0, 3.0]]], dtype=jnp.float32)

    output, _ = segmented_depthwise_causal_conv1d(x, kernel, segment_ids, initial_state=initial_state)

    assert jnp.allclose(output, jnp.array([[[0.0], [432.0], [500.0]]]))


def test_no_initial_state_still_starts_fresh(kernel):
    x = jnp.array([[[4.0]]], dtype=jnp.float32)
    segment_ids = jnp.array([[0]], dtype=jnp.int32)

    output, _ = segmented_depthwise_causal_conv1d(x, kernel, segment_ids)

    assert jnp.allclose(output, jnp.array([[[400.0]]]))


def test_each_row_uses_its_own_initial_segment_id(kernel):
    x = jnp.array([[[4.0]], [[4.0]]], dtype=jnp.float32)
    initial_state = jnp.array([[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]], dtype=jnp.float32)

    output, final_state = segmented_depthwise_causal_conv1d(
        x,
        kernel,
        jnp.array([[2], [2]], dtype=jnp.int32),
        initial_state=initial_state,
        initial_segment_id=jnp.array([2, 1], dtype=jnp.int32),
    )

    assert jnp.allclose(output, jnp.array([[[432.0]], [[400.0]]]))
    assert jnp.allclose(final_state, jnp.array([[[2.0, 3.0, 4.0]], [[0.0, 0.0, 4.0]]]))


@pytest.mark.parametrize("seq_len", [0, 2])
def test_padding_or_empty_chunk_preserves_cached_state(kernel, seq_len):
    initial_state = jnp.array([[[1.0, 2.0, 3.0]]], dtype=jnp.float32)

    output, final_state = segmented_depthwise_causal_conv1d(
        jnp.zeros((1, seq_len, 1), dtype=jnp.float32),
        kernel,
        jnp.full((1, seq_len), -1, dtype=jnp.int32),
        initial_state=initial_state,
    )

    assert output.shape == (1, seq_len, 1)
    assert jnp.all(output == 0)
    assert jnp.array_equal(final_state, initial_state)
