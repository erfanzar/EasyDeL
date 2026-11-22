"""Independent distributed public ring visibility/VJP parity; accelerator only."""

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesMetadata
from easydel.operations import OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_compressed_cache_ring_safety import _distributed_metadata
from .test_paged_reference_contract import _cache


@pytest.mark.parametrize("name", ["ring", "ring_paged"])
@pytest.mark.parametrize("visibility", ["pairwise", "bias_with_segments", "bias_with_positions"])
def test_distributed_registered_ring_visibility_and_qkv_gradients(name, visibility):
    """Compare four-way ring output and all QKV gradients to independent dense attention."""
    operation = OperationRegistry.create(name, _distributed_metadata())
    q = jnp.asarray((np.arange(16 * 2 * 8).reshape(1, 16, 2, 8) % 7 - 3).astype(np.float32) / 8)
    k = jnp.asarray((np.arange(24 * 8).reshape(1, 24, 1, 8) % 5 - 2).astype(np.float32) / 8)
    v = jnp.asarray((np.arange(24 * 8).reshape(1, 24, 1, 8) % 13 - 6).astype(np.float32) / 8)
    rows = jnp.arange(16)[None, None, :, None]
    cols = jnp.arange(24)[None, None, None, :]
    bias = ((rows * cols + 3 * cols) % 11).astype(jnp.float32) / 8
    if visibility == "pairwise":
        visible = (rows + 2 * cols) % 5 != 0
        mask = MaskInfo(_attention_mask=visible)
        visible &= rows >= cols
    else:
        qs = jnp.arange(16)[None, :] // 8
        ks = jnp.arange(24)[None, :] // 12
        qp = jnp.arange(16)[None, :] % 8
        kp = jnp.arange(24)[None, :] % 12
        # MaskInfo derives segment-relative positions for segment-only calls.
        mask = MaskInfo(_q_segment_ids=qs, _kv_segment_ids=ks)
        if visibility == "bias_with_positions":
            qp = qp + 2
            mask = mask.replace(q_positions=qp, kv_positions=kp)
        visible = (qs[:, None, :, None] == ks[:, None, None, :]) & (qp[:, None, :, None] >= kp[:, None, None, :])

    def actual(q, k, v):
        return operation(
            q, k, v, bias=bias, mask_info=mask, softmax_scale=0.5, precision=jax.lax.Precision.HIGHEST
        ).attention_outputs

    def reference(q, k, v):
        dense_k = jnp.repeat(k, 2, axis=2)
        dense_v = jnp.repeat(v, 2, axis=2)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, dense_k, precision=jax.lax.Precision.HIGHEST) * 0.5 + bias
        logits = jnp.where(visible, logits, -jnp.inf)
        # Empty visibility rows define zero output and zero gradients, not
        # ordinary softmax(-inf, ...), which is undefined. Keep both branches
        # finite so autodiff never encounters a hidden NaN.
        has_keys = jnp.any(visible, axis=-1, keepdims=True)
        safe_logits = jnp.where(has_keys, logits, 0.0)
        weights = jnp.where(visible, jax.nn.softmax(safe_logits.astype(jnp.float32), axis=-1), 0.0)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, dense_v, precision=jax.lax.Precision.HIGHEST)

    cotangent = jnp.asarray((np.arange(q.size).reshape(q.shape) % 11 - 5).astype(np.float32) / 16)
    actual_output = actual(q, k, v)
    np.testing.assert_allclose(actual_output, reference(q, k, v), atol=2e-5, rtol=2e-5)
    assert actual_output.shape == q.shape
    assert actual_output.dtype == jnp.float32
    assert actual_output.sharding.spec[1] == "sp"
    actual_gradients = jax.grad(lambda q, k, v: (actual(q, k, v) * cotangent).sum(), argnums=(0, 1, 2))(q, k, v)
    expected_gradients = jax.grad(lambda q, k, v: (reference(q, k, v) * cotangent).sum(), argnums=(0, 1, 2))(q, k, v)
    for observed, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert np.isfinite(observed).all()
        assert np.isfinite(expected).all()
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=2e-5)
    if visibility == "pairwise":
        # The first row has pairwise-visible future keys but no causal key.
        np.testing.assert_array_equal(actual_output[:, 0], 0.0)
        np.testing.assert_array_equal(actual_gradients[0][:, 0], 0.0)


@pytest.mark.parametrize("name", ["ring", "ring_paged"])
@pytest.mark.parametrize("constraint", ["causal", "segments", "sliding"])
@pytest.mark.parametrize("sink_logit", [None, 0.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_distributed_ring_empty_visibility_intersections_have_zero_output_and_qkv_gradients(
    name, constraint, sink_logit, compiled
):
    """Hard exclusions compose with other masks without creating value or gradient mass."""
    operation = OperationRegistry.create(name, _distributed_metadata())
    q = jnp.asarray((np.arange(16 * 2 * 8).reshape(1, 16, 2, 8) % 7 - 3).astype(np.float32) / 8)
    k = jnp.asarray((np.arange(24 * 8).reshape(1, 24, 1, 8) % 5 - 2).astype(np.float32) / 8)
    v = jnp.ones((1, 24, 1, 8), dtype=jnp.float32)
    rows = jnp.arange(16)[None, None, :, None]
    cols = jnp.arange(24)[None, None, None, :]
    kwargs = {"causal": constraint == "causal"}
    if constraint == "segments":
        mask = MaskInfo(
            _attention_mask=jnp.ones((1, 1, 16, 24), dtype=jnp.bool_),
            _q_segment_ids=jnp.zeros((1, 16), dtype=jnp.int32),
            _kv_segment_ids=jnp.ones((1, 24), dtype=jnp.int32),
        )
    elif constraint == "sliding":
        mask = MaskInfo(_attention_mask=cols != rows)
        kwargs["sliding_window"] = (0, 0)
    else:
        mask = MaskInfo(_attention_mask=cols > rows)
    sinks = None if sink_logit is None else jnp.asarray([sink_logit], dtype=jnp.float32)

    def forward(q, k, v):
        return operation(q, k, v, mask_info=mask, softmax_aux=sinks, **kwargs).attention_outputs

    gradients = jax.grad(lambda q, k, v: forward(q, k, v).sum(), argnums=(0, 1, 2))
    if compiled:
        forward = jax.jit(forward)
        gradients = jax.jit(gradients)
    output = forward(q, k, v)
    np.testing.assert_array_equal(output, np.zeros(q.shape, dtype=np.float32))
    for observed, source in zip(gradients(q, k, v), (q, k, v), strict=True):
        np.testing.assert_array_equal(observed, np.zeros(source.shape, dtype=np.float32))


@pytest.mark.parametrize("name", ["ring", "ring_paged"])
def test_distributed_paged_ring_generated_request_visibility_and_gradients(name):
    """Continuous batches retain request isolation, historical values and cache writes across SP."""
    operation = OperationRegistry.create(name, _distributed_metadata())
    q = jnp.asarray((np.arange(4 * 2 * 8).reshape(4, 2, 8) % 7 - 3).astype(np.float32) / 8)
    k = jnp.asarray((np.arange(4 * 8).reshape(4, 1, 8) % 5 - 2).astype(np.float32) / 8)
    v = jnp.asarray((np.arange(4 * 8).reshape(4, 1, 8) % 13 - 6).astype(np.float32) / 8)
    old_keys = jnp.asarray((np.arange(8 * 8).reshape(4, 2, 1, 8) % 3 - 1).astype(np.float32) / 8)
    old_values = old_keys + 0.25
    runtime = RaggedPagesMetadata(
        jnp.array([[2, 0], [3, 1]], dtype=jnp.int32),
        jnp.array([3, 2], dtype=jnp.int32),
        jnp.array([0, 2, 4], dtype=jnp.int32),
        jnp.array([3, 2], dtype=jnp.int32),
        page_size=2,
    )

    def actual(q, k, v):
        result = operation(
            q,
            k,
            v,
            cache_view=_cache("unified", 8, (old_keys, old_values)),
            cache_metadata=runtime,
            softmax_scale=0.5,
            precision=jax.lax.Precision.HIGHEST,
        )
        return result.attention_outputs, result.cache_view.key_cache, result.cache_view.value_cache

    def reference(q, k, v):
        updated_k, updated_v = old_keys, old_values
        for token, (page, offset) in enumerate(((2, 1), (0, 0), (3, 0), (3, 1))):
            updated_k = updated_k.at[page, offset].set(k[token])
            updated_v = updated_v.at[page, offset].set(v[token])
        outputs = []
        for start, context, table in ((0, 3, [2, 0]), (2, 2, [3, 1])):
            request_k = jnp.repeat(updated_k[jnp.array(table)].reshape(4, 1, 8), 2, axis=1)
            request_v = jnp.repeat(updated_v[jnp.array(table)].reshape(4, 1, 8), 2, axis=1)
            logits = (
                jnp.einsum("qhd,khd->hqk", q[start : start + 2], request_k, precision=jax.lax.Precision.HIGHEST) * 0.5
            )
            position = jnp.arange(2) + context - 2
            visible = (jnp.arange(4)[None, :] < context) & (jnp.arange(4)[None, :] <= position[:, None])
            weights = jax.nn.softmax(jnp.where(visible[None], logits, -jnp.inf).astype(jnp.float32), axis=-1)
            outputs.append(jnp.einsum("hqk,khd->qhd", weights, request_v, precision=jax.lax.Precision.HIGHEST))
        return jnp.concatenate(outputs), updated_k, updated_v

    output, updated_k, updated_v = actual(q, k, v)
    expected, expected_k, expected_v = reference(q, k, v)
    np.testing.assert_allclose(output, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(updated_k, expected_k)
    np.testing.assert_array_equal(updated_v, expected_v)
    cotangent = jnp.asarray((np.arange(q.size).reshape(q.shape) % 11 - 5).astype(np.float32) / 16)
    observed_gradients = jax.grad(lambda q, k, v: (actual(q, k, v)[0] * cotangent).sum(), argnums=(0, 1, 2))(q, k, v)
    expected_gradients = jax.grad(lambda q, k, v: (reference(q, k, v)[0] * cotangent).sum(), argnums=(0, 1, 2))(q, k, v)
    for observed, expected in zip(observed_gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=2e-5)
