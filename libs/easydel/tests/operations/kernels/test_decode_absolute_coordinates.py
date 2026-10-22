"""Decode preserves cache-owned absolute coordinates (numerics require an accelerator)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.operations import OperationMetadata
from easydel.operations.kernels.decode_attention import AutoRegressiveDecodeAttn
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import Mesh


def _metadata():
    return OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=Mesh(np.asarray(jax.devices()[:1]), ("tp",)),
    )


@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("mask_kind", ["dense", "baked", "segments"])
@pytest.mark.parametrize("window", [None, 2, (0, 1)])
@pytest.mark.parametrize("bias", [False, True])
def test_decode_absolute_positions_preserve_range_window_and_caller_mask(reordered, mask_kind, window, bias):
    positions = np.arange(5, 10)[None, :] + 10 * np.arange(4)[:, None]
    indexes = np.array([9, 19, 29, 39], np.int32)
    starts = np.array([0, 16, 27, 38], np.int32)
    # Explicit query positions need not equal indexes - 1 (e.g. an aligned
    # historical query). The window must be centered on the supplied position.
    query_positions = (indexes - np.array([1, 2, 1, 2], np.int32))[:, None]
    values = (2 * positions + np.arange(4)[:, None]).astype(np.float32)
    caller_visible = np.broadcast_to([True, False, True, True, True], (4, 5)).copy()
    additive_bias = np.broadcast_to(np.log([1.0, 2.0, 3.0, 4.0, 5.0]), (4, 5)).astype(np.float32)
    if reordered:
        order = [3, 0, 4, 1, 2]
        positions = positions[:, order]
        values = values[:, order]
        caller_visible = caller_visible[:, order]
        additive_bias = additive_bias[:, order]
    if mask_kind == "baked":
        caller_visible &= np.abs(positions - query_positions) <= 2
    kwargs = {
        "q_positions": jnp.asarray(query_positions),
        "kv_positions": jnp.asarray(positions),
        "sliding_window_baked_in": mask_kind == "baked",
    }
    if mask_kind == "segments":
        kwargs.update(
            _q_segment_ids=jnp.ones((4, 1), jnp.int32),
            _kv_segment_ids=jnp.asarray(np.where(caller_visible, 1, 2), jnp.int32),
        )
    else:
        kwargs["_attention_mask"] = jnp.asarray(caller_visible[:, None, None, :])
    output = AutoRegressiveDecodeAttn(_metadata()).forward_native(
        jnp.zeros((4, 1, 1, 2), jnp.float32),
        jnp.zeros((4, 5, 1, 2), jnp.float32),
        jnp.asarray(values[:, :, None, None]),
        SimpleNamespace(starts=jnp.asarray(starts), indexes=jnp.asarray(indexes)),
        mask_info=MaskInfo(**kwargs),
        bias=jnp.asarray(additive_bias[:, None, None, :]) if bias else None,
        sliding_window=window,
        precision=jax.lax.Precision.HIGHEST,
    )
    visible = caller_visible & (positions >= starts[:, None]) & (positions < indexes[:, None])
    if window is not None:
        left, right = (window - 1, window - 1) if isinstance(window, int) else window
        visible &= (positions >= query_positions - left) & (positions <= query_positions + right)
    weights = np.exp(additive_bias) if bias else np.ones_like(values)
    weights = np.where(visible, weights, 0.0)
    denominator = weights.sum(-1, keepdims=True)
    expected = (weights * values).sum(-1, keepdims=True) / np.where(denominator > 0, denominator, 1.0)
    np.testing.assert_allclose(output.attention_outputs, expected[:, :, None, None], rtol=2e-6, atol=2e-6)
    assert output.attention_outputs.shape == (4, 1, 1, 1)
    assert output.attention_outputs.dtype == jnp.float32


def test_decode_baked_mask_without_positions_remains_authoritative():
    # No absolute column map survives here. Reapplying starts to local columns
    # would incorrectly turn the already-aligned caller mask into an empty row.
    values = np.broadcast_to(np.array([6.0, 7.0, 8.0])[None, :, None, None], (4, 3, 1, 1))
    output = AutoRegressiveDecodeAttn(_metadata()).forward_native(
        jnp.zeros((4, 1, 1, 2), jnp.float32),
        jnp.zeros((4, 3, 1, 2), jnp.float32),
        jnp.asarray(values, jnp.float32),
        SimpleNamespace(starts=jnp.asarray([0, 1, 4, 6]), indexes=jnp.full((4,), 9)),
        mask_info=MaskInfo(
            _attention_mask=jnp.ones((4, 1, 1, 3), jnp.bool_),
            sliding_window_baked_in=True,
        ),
        precision=jax.lax.Precision.HIGHEST,
    )
    np.testing.assert_allclose(output.attention_outputs, np.full((4, 1, 1, 1), 7.0), rtol=2e-6, atol=2e-6)
