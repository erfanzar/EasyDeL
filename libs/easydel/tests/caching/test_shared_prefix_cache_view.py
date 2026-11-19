"""Prefix capture/fork cache views: gather layout, fork mask and gradient routing."""

import jax
import numpy as np
import pytest
from easydel.caching.shared_prefix import PrefixCaptureView, PrefixForkView
from ejkernel.types import MaskInfo
from jax import numpy as jnp

B, P, C, G, H, D = 2, 3, 2, 2, 1, 4


def _arrays():
    rng = np.random.default_rng(0)
    prefix_key = rng.standard_normal((B, P, H, D)).astype(np.float32)
    prefix_value = rng.standard_normal((B, P, H, D)).astype(np.float32)
    key = rng.standard_normal((B * G, C, H, D)).astype(np.float32)
    value = rng.standard_normal((B * G, C, H, D)).astype(np.float32)
    prefix_seg = np.array([[-1, 0, 0], [0, 0, 0]], np.int32)
    q_seg = np.array([[0, 0], [0, -1], [0, 0], [0, -1]], np.int32)
    return prefix_key, prefix_value, key, value, prefix_seg, q_seg


def _fork(prefix_key, prefix_value, prefix_seg):
    return PrefixForkView(
        prefix_key=jnp.asarray(prefix_key),
        prefix_value=jnp.asarray(prefix_value),
        prefix_segment_ids=jnp.asarray(prefix_seg),
        num_generations=G,
    )


def test_fork_view_gathers_group_prefix_and_builds_absolute_positions():
    prefix_key, prefix_value, key, value, prefix_seg, q_seg = _arrays()
    view = _fork(prefix_key, prefix_value, prefix_seg)
    k, v, mask, _ = view.concatenate_to_cache(
        key=jnp.asarray(key), value=jnp.asarray(value), mask_info=MaskInfo.from_segments(jnp.asarray(q_seg))
    )
    rows = np.arange(B * G) // G
    np.testing.assert_array_equal(np.asarray(k), np.concatenate([prefix_key[rows], key], axis=1))
    np.testing.assert_array_equal(np.asarray(v), np.concatenate([prefix_value[rows], value], axis=1))
    np.testing.assert_array_equal(np.asarray(mask.q_segment_ids), q_seg)
    np.testing.assert_array_equal(np.asarray(mask.kv_segment_ids), np.concatenate([prefix_seg[rows], q_seg], axis=1))
    np.testing.assert_array_equal(np.asarray(mask.q_positions), np.broadcast_to(P + np.arange(C), (B * G, C)))
    np.testing.assert_array_equal(np.asarray(mask.kv_positions), np.broadcast_to(np.arange(P + C), (B * G, P + C)))


def test_fork_view_gradient_sums_over_group_rows():
    prefix_key, prefix_value, key, value, prefix_seg, q_seg = _arrays()
    weights = np.random.default_rng(1).standard_normal((B * G, P + C, H, D)).astype(np.float32)

    def loss(pk):
        view = _fork(pk, prefix_value, prefix_seg)
        k, _, _, _ = view.concatenate_to_cache(
            key=jnp.asarray(key), value=jnp.asarray(value), mask_info=MaskInfo.from_segments(jnp.asarray(q_seg))
        )
        return jnp.sum(k * weights)

    grad = jax.grad(loss)(jnp.asarray(prefix_key))
    expected = weights[:, :P].reshape(B, G, P, H, D).sum(axis=1)
    np.testing.assert_allclose(np.asarray(grad), expected, rtol=1e-6)


def test_capture_view_passes_states_through_and_records_them():
    prefix_key, prefix_value, *_ = _arrays()
    k, _, mask, view = PrefixCaptureView().concatenate_to_cache(
        key=jnp.asarray(prefix_key), value=jnp.asarray(prefix_value), mask_info=None
    )
    np.testing.assert_array_equal(np.asarray(k), prefix_key)
    np.testing.assert_array_equal(np.asarray(view.value), prefix_value)
    assert mask is None


def test_fork_view_rejects_batch_not_equal_to_prompts_times_groups():
    prefix_key, prefix_value, key, value, prefix_seg, q_seg = _arrays()
    view = _fork(prefix_key, prefix_value, prefix_seg)
    with pytest.raises(ValueError, match="num_generations"):
        view.concatenate_to_cache(
            key=jnp.asarray(key[:3]),
            value=jnp.asarray(value[:3]),
            mask_info=MaskInfo.from_segments(jnp.asarray(q_seg[:3])),
        )
