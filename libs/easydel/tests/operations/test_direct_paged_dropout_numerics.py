"""Direct registered BTHD paged dropout parity; accelerator execution only."""

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesMetadata
from easydel.operations import OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_paged_reference_contract import _cache, _metadata, _storage_shapes


@pytest.mark.parametrize("layout", ["v2", "v3", "unified"])
@pytest.mark.parametrize("feature", ["mask", "bias", "segments", "combined"])
@pytest.mark.parametrize("normalized", [False, True])
def test_direct_bthd_dropout_preserves_features_and_historical_cache(layout, feature, normalized):
    q = np.zeros((2, 2, 2, 4), dtype=np.float32)
    k = np.zeros((2, 2, 1, 4), dtype=np.float32)
    v = np.broadcast_to(np.array([2, 4, 8, 12], dtype=np.float32).reshape(2, 2, 1, 1), k.shape).copy()
    pages = [np.zeros(shape, dtype=np.float32) for shape in _storage_shapes(layout, 4)]
    history = [1, 3, 7, 11]
    if layout == "unified":
        pages[1][:, 0, 0, :] = np.array(history)[:, None]
    else:
        # v3 pads the physical feature dimension even for four-wide heads.
        # Flatten only the storage head/packing axes, never that feature axis.
        flat = pages[0].reshape(4, 2, -1, pages[0].shape[-1])
        flat[:, 0, 1, :4] = np.array(history)[:, None]
    feature_mask = np.ones((2, 1, 2, 4), dtype=bool)
    feature_mask[1, 0, 0, 0] = False
    feature_mask[0, 0, 1, 0] = False
    bias = np.zeros((2, 2, 2, 4), dtype=np.float32)
    bias[1, :, 1, 0] = -0.8
    q_segments = np.array([[1, 1], [2, 3]], dtype=np.int32)
    kv_segments = np.array([[1, 1, -1, -1], [2, 3, -1, -1]], dtype=np.int32)
    rng = jax.random.key(25)
    expected = np.zeros_like(q)
    for row in range(4):
        batch, local = divmod(row, 2)
        visible = np.array([True, True, False, False])
        if feature in ("mask", "combined"):
            visible &= feature_mask[batch, 0, local]
        if feature in ("segments", "combined"):
            visible &= q_segments[batch, local] == kv_segments[batch]
        logits = bias[batch, :, local] if feature == "bias" else np.zeros((2, 4), dtype=np.float32)
        logits = np.where(visible[None], logits, -np.inf)
        # The shared zero-logit sink participates in both row max and divisor.
        all_logits = np.concatenate((logits, np.zeros((2, 1))), axis=-1)
        weights = np.exp(all_logits - all_logits.max(axis=-1, keepdims=True))
        if normalized:
            weights /= weights.sum(axis=-1, keepdims=True)
        weights = weights[:, :4]
        keep = np.asarray(jax.random.bernoulli(jax.random.fold_in(rng, row), 0.75, (1, 2, 4, 4)))
        weights = np.where(keep[0, :, row], weights / 0.75, 0)
        vals = np.array([history[row], v.reshape(4, 1, 4)[row, 0, 0], 0, 0])
        expected[batch, local] = (weights * vals[None]).sum(axis=-1)[:, None]

    operation = OperationRegistry.create(
        {"v2": "ragged_page_attention_v2", "v3": "ragged_page_attention_v3", "unified": "unified_attention"}[layout],
        _metadata(),
    )
    cache = _cache(layout, 4, tuple(map(jnp.asarray, pages)))
    runtime = RaggedPagesMetadata(
        jnp.stack((jnp.arange(4), jnp.full((4,), -1)), axis=-1),
        jnp.full((4,), 2),
        jnp.arange(5),
        jnp.full((4,), 2),
        page_size=2,
    )
    kwargs = {}
    if feature == "combined":
        kwargs["mask_info"] = MaskInfo(
            _attention_mask=jnp.asarray(feature_mask),
            _q_segment_ids=jnp.asarray(q_segments),
            _kv_segment_ids=jnp.asarray(kv_segments),
        )
    elif feature == "mask":
        kwargs["mask_info"] = MaskInfo(_attention_mask=jnp.asarray(feature_mask))
    elif feature == "segments":
        kwargs["mask_info"] = MaskInfo.from_segments(jnp.asarray(q_segments), jnp.asarray(kv_segments))
    else:
        kwargs["bias"] = jnp.asarray(bias)
    result = operation(
        query=jnp.asarray(q),
        key=jnp.asarray(k),
        value=jnp.asarray(v),
        cache_view=cache,
        cache_metadata=runtime,
        compute_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        dropout_rng=rng,
        dropout_prob=0.25,
        deterministic=False,
        softmax_aux=jnp.zeros((1,)),
        normalize_output=normalized,
        **kwargs,
    )
    np.testing.assert_allclose(result.attention_outputs, expected, rtol=2e-5, atol=2e-6)
    if layout == "unified":
        updated = np.asarray(result.cache_view.value_cache)
    else:
        storage = np.asarray(result.cache_view.kv_pages)
        updated = storage.reshape(4, 2, -1, storage.shape[-1])[:, :, 1:2, :4]
    np.testing.assert_array_equal(updated[:, 0, 0, 0], history)
    np.testing.assert_array_equal(updated[:, 1, 0, :], v.reshape(4, 4))
