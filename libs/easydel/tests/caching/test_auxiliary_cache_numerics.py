"""Auxiliary/indexer and bidirectional cache numerics; accelerator execution only."""

import jax
import numpy as np
import pytest
from easydel.caching import TransformerCache, TransformerCacheConfig, TransformerCacheView, TransformerMetadata
from easydel.caching.transformer.cache import AttnMaskDetail
from easydel.infra.utils import AttnMaskType
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import Mesh
from spectrax import common_types


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("postpadded", [False, True])
def test_prefill_bidirectional_visibility_preserves_future_tokens_and_packed_padding(causal, sliding, postpadded):
    batch, block, capacity = 4, 6, 3 if sliding else 8
    labels = np.tile([0, 0, 0, 1, 1, -1 if postpadded else 1], (batch, 1)).astype(np.int32)
    config = TransformerCacheConfig.create(
        batch_size=batch, sequence_length=8, num_hidden_layers=1, num_heads=1, head_dim=1
    )
    view = TransformerCacheView(
        key=jnp.zeros((batch, capacity, 1, 1)),
        value=jnp.zeros((batch, capacity, 1, 1)),
        indexes=jnp.zeros((batch,), jnp.int32),
        starts=jnp.array([0, 1, 0, 1]),
        metadata=config,
        maximum_sequence_length=8,
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
    )
    tokens = jnp.broadcast_to(jnp.arange(1, block + 1, dtype=jnp.float32)[None, :, None, None], (batch, block, 1, 1))
    key, _, mask, updated, _ = view.concatenate_to_cache(
        query=tokens,
        key=tokens,
        value=tokens,
        mode=common_types.MODE_PREFILL,
        quantizer=lambda x: x,
        cache_metadata=TransformerMetadata(postpadded=postpadded),
        mask_info=MaskInfo.from_segments(jnp.asarray(labels)),
        causal=causal,
    )
    width = key.shape[1]
    expected = np.zeros((batch, 1, block, width), bool)
    for b in range(batch):
        for q in range(block):
            for k in range(min(width, block)):
                expected[b, 0, q, k] = (
                    labels[b, q] >= 0
                    and labels[b, q] == labels[b, k]
                    and k >= b % 2
                    and (not causal or k <= q)
                    and (not sliding or abs(k - q) < capacity)
                )
    np.testing.assert_array_equal(np.asarray(mask.attention_mask), expected)
    assert mask.causal_mask_baked_in == causal
    assert bool(np.asarray(mask.attention_mask)[0, 0, 0, 1]) == (not causal)
    np.testing.assert_array_equal(np.asarray(updated.indexes), np.full(batch, block - int(postpadded)))


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("postpadded", [False, True])
def test_auxiliary_and_kv_share_unequal_offsets_visibility_and_sparse_reference(sliding, postpadded):
    batch, block, capacity, dimension = 4, 4, 3 if sliding else 10, 9
    indexes = np.array([1, 3, 2, 0], np.int32)
    starts = np.array([0, 2, 1, 0], np.int32)
    lengths = np.array([2, 3, 4, 1], np.int32) if postpadded else np.full(batch, block, np.int32)
    old = np.zeros((batch, capacity), np.float32)
    old_labels = np.full((batch, capacity), -1, np.int32)
    for b in range(batch):
        first = max(indexes[b] - capacity, 0) if sliding else 0
        for column in range(min(indexes[b], capacity)):
            old[b, column] = 10 * b + first + column + 1
            old_labels[b, column] = 0
    new = np.arange(batch * block, dtype=np.float32).reshape(batch, block) + 100
    config = TransformerCacheConfig.create(
        batch_size=batch, sequence_length=10, num_hidden_layers=1, num_heads=1, head_dim=1, auxiliary_dim=dimension
    )
    view = TransformerCacheView(
        key=jnp.asarray(old[:, :, None, None]),
        value=jnp.asarray(old[:, :, None, None]),
        indexes=jnp.zeros((batch,), jnp.int32),
        starts=jnp.zeros((batch,), jnp.int32),
        metadata=config,
        maximum_sequence_length=10,
        segment_ids=jnp.asarray(old_labels),
        auxiliary_state=jnp.asarray(np.repeat(old[:, :, None], dimension, axis=-1)),
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
    )
    incoming = MaskInfo.from_segments(jnp.where(jnp.arange(block)[None, :] < jnp.asarray(lengths)[:, None], 0, -1))
    metadata = TransformerMetadata(indexes=jnp.asarray(indexes), starts=jnp.asarray(starts), postpadded=postpadded)

    def append(view):
        features, selection_mask, pending = view.update_auxiliary(
            jnp.asarray(np.repeat(new[:, :, None], dimension, axis=-1)), incoming, metadata
        )
        key, _, mask, updated, _ = pending.concatenate_to_cache(
            query=jnp.asarray(new[:, :, None, None]),
            key=jnp.asarray(new[:, :, None, None]),
            value=jnp.asarray(new[:, :, None, None]),
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=metadata,
            mask_info=incoming,
        )
        return features, selection_mask, key, mask, updated

    features, selection_mask, key, mask, updated = jax.jit(append)(view)
    np.testing.assert_array_equal(np.asarray(features[..., 0]), np.asarray(key[..., 0, 0]))
    np.testing.assert_array_equal(np.asarray(updated.read_auxiliary()[..., 0]), np.asarray(updated.key[..., 0, 0]))
    np.testing.assert_array_equal(np.asarray(selection_mask.attention_mask), np.asarray(mask.attention_mask))
    np.testing.assert_array_equal(np.asarray(updated.indexes), indexes + lengths)
    # Independent full-history top-1 sparse selection, with unused slots excluded
    # before ranking. This is over more candidates than topk=1, not a dense fallback.
    for b in range(batch):
        for q in range(lengths[b]):
            absolute_q = indexes[b] + q
            first = max(0, absolute_q - capacity + 1) if sliding else 0
            candidate_positions = range(max(first, starts[b]), absolute_q + 1)
            values = [10 * b + p + 1 if p < indexes[b] else new[b, p - indexes[b]] for p in candidate_positions]
            visible = np.asarray(selection_mask.attention_mask)[b, 0, q]
            selected = np.max(np.asarray(features)[b, visible, 0])
            assert selected == max(values)
    np.testing.assert_array_equal(np.asarray(updated.reset().read_auxiliary()), np.zeros_like(updated.read_auxiliary()))


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("postpadded", [False, True])
@pytest.mark.parametrize("block_size", [1, 2])
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("incoming_segment", [0, 2])
def test_lazy_segment_auxiliary_then_kv_continuation_has_stable_ownership(
    sliding, postpadded, block_size, materialized, incoming_segment
):
    batch, capacity, dimension = 4, 4, 3
    config = TransformerCacheConfig.create(
        batch_size=batch, sequence_length=8, num_hidden_layers=1, num_heads=1, head_dim=1, auxiliary_dim=dimension
    )
    history = np.tile([10.0, 20.0, 0.0, 0.0], (batch, 1)).astype(np.float32)
    view = TransformerCacheView(
        key=jnp.asarray(history[:, :, None, None]),
        value=jnp.asarray(history[:, :, None, None]),
        indexes=jnp.full((batch,), 2, jnp.int32),
        starts=jnp.zeros((batch,), jnp.int32),
        metadata=config,
        maximum_sequence_length=8,
        segment_ids=jnp.tile(jnp.array([[1, 1, -1, -1]], jnp.int32), (batch, 1)),
        auxiliary_state=jnp.asarray(np.repeat(history[:, :, None], dimension, axis=-1)),
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
    )
    original_mask = MaskInfo.from_segments(jnp.full((batch, block_size), incoming_segment, jnp.int32))
    if materialized:
        original_mask.get_or_compute_attention_mask()
    original_attention = original_mask._attention_mask
    # The default metadata path must not depend on postpadding having happened
    # to materialize the original mask before segment recovery.
    metadata = TransformerMetadata(postpadded=True) if postpadded else None
    features = jnp.broadcast_to(
        jnp.arange(30, 30 + block_size, dtype=jnp.float32)[None, :, None], (batch, block_size, dimension)
    )
    attention_features, selection_mask, pending = view.update_auxiliary(features, original_mask, metadata)
    tokens = features[..., :1, None]
    key, _, attention_mask, updated, _ = pending.concatenate_to_cache(
        query=tokens,
        key=tokens,
        value=tokens,
        mode=common_types.MODE_PREFILL,
        quantizer=lambda x: x,
        cache_metadata=metadata,
        mask_info=original_mask,
    )
    # Local zero labels continue the last historical segment; an explicit
    # nonzero label is preserved and excludes foreign historical ownership.
    expected_segment = 1 if incoming_segment == 0 else incoming_segment
    expected = np.zeros((batch, 1, block_size, key.shape[1]), bool)
    for q in range(block_size):
        expected[:, 0, q, :2] = incoming_segment == 0
        expected[:, 0, q, 2 : 3 + q] = True
    expected_kv_segments = np.full((batch, key.shape[1]), -1, np.int32)
    expected_kv_segments[:, :2] = 1
    expected_kv_segments[:, 2 : 2 + block_size] = expected_segment
    for mask in (selection_mask, attention_mask):
        np.testing.assert_array_equal(np.asarray(mask.attention_mask), expected)
        q_segments, kv_segments = mask.get_or_compute_segment_ids()
        np.testing.assert_array_equal(np.asarray(q_segments), np.full((batch, block_size), expected_segment))
        np.testing.assert_array_equal(np.asarray(kv_segments), expected_kv_segments)
    np.testing.assert_array_equal(np.asarray(attention_features[..., 0]), np.asarray(key[..., 0, 0]))
    np.testing.assert_array_equal(np.asarray(updated.read_auxiliary()[..., 0]), np.asarray(updated.key[..., 0, 0]))
    np.testing.assert_array_equal(np.asarray(updated.segment_ids), expected_kv_segments[:, :capacity])
    np.testing.assert_array_equal(np.asarray(updated.indexes), np.full(batch, 2 + block_size))
    # Both APIs are functional with respect to the shared input mask too.
    assert original_mask._attention_mask is original_attention
    np.testing.assert_array_equal(
        np.asarray(original_mask.q_segment_ids), np.full((batch, block_size), incoming_segment)
    )


@pytest.mark.parametrize("dimension", [4, 9])
def test_allocated_local_cache_auxiliary_survives_decode_and_reset(dimension):
    # A one-device local cache mirrors scalar-MTP allocation without an invalid
    # batch-1 sharding over the four-device test mesh.
    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    config = TransformerCacheConfig.create(
        batch_size=1, sequence_length=8, num_hidden_layers=1, num_heads=1, head_dim=1, auxiliary_dim=dimension
    )
    cache = TransformerCache.init_cache(mesh=mesh, config=config, dtype=jnp.float32)
    for offset, size in [(0, 3), (3, 2), (5, 1)]:
        original_mask = MaskInfo.from_attention_mask(jnp.ones((1, 1, size, size), jnp.bool_))
        features = jnp.broadcast_to(jnp.arange(offset + 1, offset + size + 1)[None, :, None], (1, size, dimension))
        _, _, pending = cache.views[0].update_auxiliary(features, original_mask)
        tokens = features[..., :1, None].astype(jnp.float32)
        _, _, _, view, _ = pending.concatenate_to_cache(
            query=tokens,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=None,
            mask_info=original_mask,
        )
        cache = cache.replace(views=[view])
        assert view.read_auxiliary().shape == (1, 8, dimension)
    np.testing.assert_array_equal(np.asarray(cache.views[0].read_auxiliary())[0, :, 0], [1, 2, 3, 4, 5, 6, 0, 0])
    leaves, tree = jax.tree_util.tree_flatten(cache)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_array_equal(
        np.asarray(restored.views[0].read_auxiliary()), np.asarray(cache.views[0].read_auxiliary())
    )
    np.testing.assert_array_equal(np.asarray(restored.views[0].reset().read_auxiliary()), np.zeros((1, 8, dimension)))
