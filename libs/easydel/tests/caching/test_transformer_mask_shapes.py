"""Trace-only cache output-shape contracts; no JAX arrays are executed."""

import jax
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

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("metadata_class", [TransformerMetadata, HybridMetadata])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("capacity, block_size, absolute", [(8, 4, False), (3, 5, False), (3, 5, True)])
def test_postpadded_append_mask_shape_matches_returned_kv(
    metadata_class, wrapped, sliding, capacity, block_size, absolute
):
    """Abstract interpretation validates public shapes, not numerical correctness."""
    batch = 2
    config = TransformerCacheConfig.create(
        batch_size=batch, sequence_length=capacity, num_hidden_layers=1, num_heads=1, head_dim=1
    )
    width = capacity + block_size if absolute else block_size

    def append(old, tokens, indexes, starts, attention_mask):
        view = TransformerCacheView(
            key=old,
            value=old,
            indexes=indexes,
            starts=starts,
            metadata=config,
            maximum_sequence_length=capacity,
            masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
        )
        metadata = metadata_class(postpadded=True, indexes=indexes, starts=starts)
        if wrapped:
            metadata = OperationsMetadata.wrap(metadata)
        # Include optional positions and eagerly cached segment metadata. These
        # must remain shape-consistent after both column and query translation.
        mask_info = MaskInfo.from_attention_mask(
            attention_mask,
            q_positions=jnp.broadcast_to(jnp.arange(width), (batch, width)),
            kv_positions=jnp.broadcast_to(jnp.arange(width), (batch, width)),
        )
        key, value, mask, updated, _ = view.concatenate_to_cache(
            query=tokens,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=metadata,
            mask_info=mask_info,
        )
        q_segments, kv_segments = mask.get_or_compute_segment_ids()
        return (
            key,
            value,
            mask.attention_mask,
            updated.indexes,
            mask.q_positions,
            mask.kv_positions,
            q_segments,
            kv_segments,
        )

    result = jax.eval_shape(
        append,
        jax.ShapeDtypeStruct((batch, capacity, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((batch, block_size, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((batch,), jnp.int32),
        jax.ShapeDtypeStruct((batch,), jnp.int32),
        jax.ShapeDtypeStruct((batch, 1, width, width), jnp.bool_),
    )
    key, value, mask, indexes, q_positions, kv_positions, q_segments, kv_segments = result
    attention_width = capacity + block_size if sliding else capacity
    assert key.shape == value.shape == (batch, attention_width, 1, 1)
    assert mask.shape == (batch, 1, block_size, attention_width)
    assert indexes.shape == (batch,)
    assert q_positions.shape == q_segments.shape == (batch, block_size)
    assert kv_positions.shape == kv_segments.shape == (batch, attention_width)


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("first_postpadded", [False, True])
@pytest.mark.parametrize("pairwise", [False, True])
def test_two_appends_keep_segment_history_shape(sliding, first_postpadded, pairwise):
    """Trace the persisted-state path without executing any array operations."""
    capacity = 4 if sliding else 8
    config = TransformerCacheConfig.create(
        batch_size=1, sequence_length=capacity, num_hidden_layers=1, num_heads=1, head_dim=1
    )

    def append_twice(old, tokens, segments, indexes):
        view = TransformerCacheView(
            key=old,
            value=old,
            indexes=indexes,
            starts=indexes,
            metadata=config,
            maximum_sequence_length=capacity,
            masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
        )
        for postpadded in (first_postpadded, True):
            mask_info = MaskInfo.from_segments(segments)
            if pairwise:
                mask_info = MaskInfo.from_attention_mask(mask_info.apply_causal().attention_mask)
            _, _, mask, view, _ = view.concatenate_to_cache(
                query=tokens,
                key=tokens,
                value=tokens,
                mode=common_types.MODE_PREFILL,
                quantizer=lambda x: x,
                cache_metadata=TransformerMetadata(postpadded=postpadded),
                mask_info=mask_info,
            )
        return mask.attention_mask, view.segment_ids

    mask, stored_segments = jax.eval_shape(
        append_twice,
        jax.ShapeDtypeStruct((1, capacity, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((1, 4, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((1, 4), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    assert mask.shape == (1, 1, 4, capacity + 4 if sliding else capacity)
    assert stored_segments.shape == (1, capacity)
    assert stored_segments.dtype == jnp.int32


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("absolute_decode_segments", [False, True])
@pytest.mark.parametrize("pairwise_local_masks", [False, True])
def test_padded_prefill_ordinary_decode_padded_append_shapes(sliding, absolute_decode_segments, pairwise_local_masks):
    """Trace all three transitions, including decode at a nonzero cache index."""
    capacity = 4 if sliding else 8
    config = TransformerCacheConfig.create(
        batch_size=1, sequence_length=capacity, num_hidden_layers=1, num_heads=1, head_dim=1
    )

    def append_three_times(old, prefill, decode, padded, prefill_segments, decode_segments, padded_segments, indexes):
        view = TransformerCacheView(
            key=old,
            value=old,
            indexes=indexes,
            starts=indexes,
            metadata=config,
            maximum_sequence_length=capacity,
            masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
        )
        for tokens, labels, postpadded in (
            (prefill, prefill_segments, True),
            (decode, decode_segments, False),
            (padded, padded_segments, True),
        ):
            query_labels = labels[:, : tokens.shape[1]]
            mask_info = MaskInfo.from_segments(query_labels, labels)
            if pairwise_local_masks and query_labels.shape == labels.shape:
                mask_info = MaskInfo.from_attention_mask(mask_info.apply_causal().attention_mask)
            _, _, mask, view, _ = view.concatenate_to_cache(
                query=tokens,
                key=tokens,
                value=tokens,
                mode=common_types.MODE_PREFILL,
                quantizer=lambda x: x,
                cache_metadata=TransformerMetadata(postpadded=postpadded),
                mask_info=mask_info,
            )
        return mask.attention_mask, view.segment_ids, view.indexes

    mask, stored_segments, indexes = jax.eval_shape(
        append_three_times,
        jax.ShapeDtypeStruct((1, capacity, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((1, 4, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((1, 1, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((1, 3, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((1, 4), jnp.int32),
        jax.ShapeDtypeStruct((1, 8 if absolute_decode_segments else 1), jnp.int32),
        jax.ShapeDtypeStruct((1, 3), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    assert mask.shape == (1, 1, 3, capacity + 3 if sliding else capacity)
    assert stored_segments.shape == (1, capacity)
    assert stored_segments.dtype == jnp.int32
    assert indexes.shape == (1,)


@pytest.mark.parametrize("postpadded", [False, True])
@pytest.mark.parametrize("block_size", [1, 3])
@pytest.mark.parametrize("heads", [1, 2])
def test_pairwise_recovery_keeps_multihead_metadata_shapes(postpadded, block_size, heads):
    """Trace complete-key-interval recovery, including its empty split axis."""
    capacity = block_size + 2
    config = TransformerCacheConfig.create(
        batch_size=2,
        sequence_length=capacity,
        num_hidden_layers=1,
        num_heads=heads,
        head_dim=1,
        key_heads=1,
        value_heads=1,
    )

    def append(old, tokens, query, indexes, pairwise):
        view = TransformerCacheView(
            key=old,
            value=old,
            indexes=indexes,
            starts=indexes,
            metadata=config,
            maximum_sequence_length=capacity,
        )
        _, _, mask, updated, _ = view.concatenate_to_cache(
            query=query,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=TransformerMetadata(postpadded=postpadded),
            mask_info=MaskInfo.from_attention_mask(pairwise),
        )
        return mask.attention_mask, mask.get_or_compute_segment_ids(), updated.segment_ids

    mask, (q_segments, kv_segments), history = jax.eval_shape(
        append,
        jax.ShapeDtypeStruct((2, capacity, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((2, block_size, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((2, block_size, heads, 1), jnp.float32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
        jax.ShapeDtypeStruct((2, heads, block_size, block_size), jnp.bool_),
    )
    assert mask.shape == (2, heads, block_size, capacity)
    assert q_segments.shape == (2, block_size)
    assert kv_segments.shape == history.shape == (2, capacity)
    assert q_segments.dtype == kv_segments.dtype == history.dtype == jnp.int32
