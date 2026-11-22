"""Accelerator-only numerical segment-isolation regressions for the dense transformer cache."""

import jax
import numpy as np
import pytest
from easydel.caching import (
    HybridMetadata,
    OperationsMetadata,
    TransformerCacheConfig,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.caching.transformer.cache import AttnMaskDetail
from easydel.infra.utils import AttnMaskType
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("first_postpadded", [False, True])
@pytest.mark.parametrize("metadata_class", [TransformerMetadata, HybridMetadata])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("pairwise", [False, True])
def test_padded_append_only_sees_matching_cached_segments(sliding, first_postpadded, metadata_class, wrapped, pairwise):
    capacity = 4 if sliding else 8
    config = TransformerCacheConfig.create(
        batch_size=1, sequence_length=capacity, num_hidden_layers=1, num_heads=1, head_dim=1
    )
    view = TransformerCacheView(
        key=jnp.zeros((1, capacity, 1, 1)),
        value=jnp.zeros((1, capacity, 1, 1)),
        indexes=jnp.zeros((1,), jnp.int32),
        starts=jnp.zeros((1,), jnp.int32),
        metadata=config,
        maximum_sequence_length=capacity,
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
    )

    @jax.jit
    def append(view, segment_ids, postpadded):
        tokens = jnp.ones((1, 4, 1, 1))
        mask_info = MaskInfo.from_segments(segment_ids)
        if pairwise:
            # The pairwise factory retains padding labels only. Ownership must
            # be recovered even with a causal triangle already materialized.
            mask_info = MaskInfo.from_attention_mask(mask_info.apply_causal().attention_mask)
        return view.concatenate_to_cache(
            query=tokens,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=postpadded,
            mask_info=mask_info,
        )

    def metadata(postpadded):
        value = metadata_class(postpadded=postpadded)
        return OperationsMetadata.wrap(value) if wrapped else value

    _, _, _, first, _ = append(view, jnp.array([[0, 0, 1, 1]], jnp.int32), metadata(first_postpadded))
    _, _, mask, second, _ = append(first, jnp.array([[1, 1, 2, -1]], jnp.int32), metadata(True))
    all_segments = [0, 0, 1, 1, 1, 1, 2]
    attention_width = capacity + 4 if sliding else capacity
    expected = np.zeros((1, 1, 4, attention_width), dtype=bool)
    for row, segment in enumerate([1, 1, 2, -1]):
        for position in range(attention_width):
            expected[0, 0, row, position] = (
                segment >= 0
                and position < 7
                and position <= 4 + row
                and all_segments[position] == segment
                and (not sliding or position > 4 + row - capacity)
            )
    np.testing.assert_array_equal(np.asarray(mask.attention_mask), expected)
    first_position = 3 if sliding else 0
    expected_segments = all_segments[first_position : first_position + capacity]
    expected_segments += [-1] * (capacity - len(expected_segments))
    np.testing.assert_array_equal(np.asarray(second.segment_ids), [expected_segments])
    attention_segments = all_segments + [-1] * (attention_width - len(all_segments))
    np.testing.assert_array_equal(np.asarray(mask.get_or_compute_segment_ids()[1]), [attention_segments])
    # Cache reset must discard stale segment ownership as well as KV values.
    np.testing.assert_array_equal(np.asarray(second.reset().segment_ids), -np.ones((1, capacity), np.int32))


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("absolute_decode_segments", [False, True])
@pytest.mark.parametrize("metadata_class", [TransformerMetadata, HybridMetadata])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("pairwise_local_masks", [False, True])
def test_nonzero_ordinary_append_preserves_labels_for_next_padded_append(
    sliding, absolute_decode_segments, metadata_class, wrapped, pairwise_local_masks
):
    """Padded prefill -> ordinary decode at slot four -> padded packed append."""
    capacity = 4 if sliding else 8
    config = TransformerCacheConfig.create(
        batch_size=1, sequence_length=capacity, num_hidden_layers=1, num_heads=1, head_dim=1
    )
    view = TransformerCacheView(
        key=jnp.zeros((1, capacity, 1, 1)),
        value=jnp.zeros((1, capacity, 1, 1)),
        indexes=jnp.zeros((1,), jnp.int32),
        starts=jnp.zeros((1,), jnp.int32),
        metadata=config,
        maximum_sequence_length=capacity,
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
    )

    @jax.jit
    def append(view, tokens, q_segments, kv_segments, metadata):
        mask_info = MaskInfo.from_segments(q_segments, kv_segments)
        if pairwise_local_masks and q_segments.shape == kv_segments.shape:
            mask_info = MaskInfo.from_attention_mask(mask_info.apply_causal().attention_mask)
        return view.concatenate_to_cache(
            query=tokens,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=metadata,
            mask_info=mask_info,
        )

    def run(view, labels, postpadded, absolute_labels=None):
        segments = jnp.array([labels], jnp.int32)
        kv_segments = segments if absolute_labels is None else jnp.array([absolute_labels], jnp.int32)
        metadata = metadata_class(postpadded=postpadded)
        if wrapped:
            metadata = OperationsMetadata.wrap(metadata)
        tokens = jnp.ones((1, len(labels), 1, 1))
        return append(view, tokens, segments, kv_segments, metadata)

    _, _, _, view, _ = run(view, [0, 0, 1, 1], True)
    absolute_labels = [0, 0, 1, 1, 1, -1, -1, -1] if absolute_decode_segments else None
    _, _, decode_mask, decoded, _ = run(view, [1], False, absolute_labels)
    decode_expected = [False, True, True, True] if sliding else [False, False, True, True, True, False, False, False]
    np.testing.assert_array_equal(np.asarray(decode_mask.attention_mask[0, 0, 0]), decode_expected)
    np.testing.assert_array_equal(
        np.asarray(decode_mask.get_or_compute_segment_ids()[1]), np.asarray(decoded.segment_ids)
    )
    expected_decode_history = [0, 1, 1, 1] if sliding else [0, 0, 1, 1, 1, -1, -1, -1]
    np.testing.assert_array_equal(np.asarray(decoded.segment_ids), [expected_decode_history])
    np.testing.assert_array_equal(np.asarray(decoded.indexes), [5])

    _, _, mask, updated, _ = run(decoded, [1, 2, -1], True)
    all_segments = [0, 0, 1, 1, 1, 1, 2]
    attention_first = 1 if sliding else 0
    attention_width = capacity + 3 if sliding else capacity
    expected_mask = np.zeros((1, 1, 3, attention_width), dtype=bool)
    for row, segment in enumerate([1, 2, -1]):
        for column in range(attention_width):
            position = attention_first + column
            expected_mask[0, 0, row, column] = (
                segment >= 0
                and position < 7
                and position <= 5 + row
                and all_segments[position] == segment
                and (not sliding or position > 5 + row - capacity)
            )
    np.testing.assert_array_equal(np.asarray(mask.attention_mask), expected_mask)
    first_position = 3 if sliding else 0
    expected_history = all_segments[first_position:]
    expected_history += [-1] * (capacity - len(expected_history))
    np.testing.assert_array_equal(np.asarray(updated.segment_ids), [expected_history])
    np.testing.assert_array_equal(np.asarray(updated.indexes), [7])


def test_pairwise_new_segment_never_reuses_an_older_nonmonotone_label():
    """An inferred new segment must not inherit an unrelated older segment's KV."""
    config = TransformerCacheConfig.create(batch_size=1, sequence_length=8, num_hidden_layers=1, num_heads=1, head_dim=1)
    view = TransformerCacheView(
        key=jnp.zeros((1, 8, 1, 1)),
        value=jnp.zeros((1, 8, 1, 1)),
        indexes=jnp.array([4], jnp.int32),
        starts=jnp.zeros((1,), jnp.int32),
        metadata=config,
        maximum_sequence_length=8,
        segment_ids=jnp.array([[5, 5, 1, 1, -1, -1, -1, -1]], jnp.int32),
    )
    tokens = jnp.ones((1, 3, 1, 1))
    mask_info = MaskInfo.from_attention_mask(jnp.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 0]]]], jnp.bool_))
    _, _, mask, updated, _ = jax.jit(
        lambda view, tokens, mask: view.concatenate_to_cache(
            query=tokens,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=TransformerMetadata(postpadded=True),
            mask_info=mask,
        )
    )(view, tokens, mask_info)
    expected = np.zeros((1, 1, 3, 8), dtype=bool)
    expected[0, 0, 0, 2:5] = True
    expected[0, 0, 1, 5] = True
    np.testing.assert_array_equal(np.asarray(mask.attention_mask), expected)
    np.testing.assert_array_equal(np.asarray(updated.segment_ids), [[5, 5, 1, 1, 1, 6, -1, -1]])


@pytest.mark.parametrize("edges", range(8))
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("postpadded", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
def test_pairwise_recovery_preserves_every_allowed_incoming_edge(edges, heads, postpadded, sliding):
    """All three-token causal masks retain their exact pairwise restrictions.

    Every token has a self edge; independently choose the three lower-triangle
    edges. In particular, edges=2 is the long-range-only case:
    [[1, 0, 0], [0, 1, 0], [1, 0, 1]]. With two heads the long-range edge and
    self edge are in separate heads, but share the same ownership metadata.
    """
    block_size = 3
    capacity = 3 if sliding else 5
    config = TransformerCacheConfig.create(
        batch_size=1,
        sequence_length=capacity,
        num_hidden_layers=1,
        num_heads=heads,
        head_dim=1,
        key_heads=1,
        value_heads=1,
    )
    view = TransformerCacheView(
        key=jnp.zeros((1, capacity, 1, 1)),
        value=jnp.zeros((1, capacity, 1, 1)),
        indexes=jnp.zeros((1,), jnp.int32),
        starts=jnp.zeros((1,), jnp.int32),
        metadata=config,
        maximum_sequence_length=capacity,
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
    )
    incoming = np.broadcast_to(np.eye(block_size, dtype=bool), (1, heads, block_size, block_size)).copy()
    for bit, (query_row, key_column) in enumerate(((1, 0), (2, 0), (2, 1))):
        incoming[0, 0, query_row, key_column] = bool(edges & (1 << bit))
    if heads == 2:
        incoming[0, 0, 2, 2] = False
    tokens = jnp.ones((1, block_size, 1, 1))
    query = jnp.ones((1, block_size, heads, 1))
    _, _, mask, updated, _ = jax.jit(
        lambda view, mask_info: view.concatenate_to_cache(
            query=query,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=TransformerMetadata(postpadded=postpadded),
            mask_info=mask_info,
        )
    )(view, MaskInfo.from_attention_mask(jnp.asarray(incoming)))
    attention_width = capacity + block_size if sliding else capacity
    expected = np.pad(incoming, ((0, 0), (0, 0), (0, 0), (0, attention_width - block_size)))
    np.testing.assert_array_equal(np.asarray(mask.attention_mask), expected)
    q_segments, kv_segments = (np.asarray(x) for x in mask.get_or_compute_segment_ids())
    same_segment = q_segments[:, None, :, None] == kv_segments[:, None, None, :]
    # Kernels may consume ownership metadata in addition to the materialized
    # mask. It must not contradict any allowed edge in any attention head.
    np.testing.assert_array_equal(expected & same_segment, expected)
    np.testing.assert_array_equal(np.asarray(updated.indexes), [block_size])
