"""Accelerator-only tests for dense-cache runtime metadata and padding."""

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
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from easydel.infra.utils import AttnMaskType
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis, common_types


def _view(batch=1, capacity=8, *, sliding=False):
    config = TransformerCacheConfig.create(
        batch_size=batch, sequence_length=capacity, num_hidden_layers=1, num_heads=1, head_dim=1
    )
    return TransformerCacheView(
        key=jnp.zeros((batch, capacity, 1, 1)),
        value=jnp.zeros((batch, capacity, 1, 1)),
        indexes=jnp.zeros((batch,), jnp.int32),
        starts=jnp.zeros((batch,), jnp.int32),
        metadata=config,
        maximum_sequence_length=capacity,
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
    )


def _concat(view, tokens, metadata, mask, *, resolver=None, mode=common_types.MODE_PREFILL):
    return view.concatenate_to_cache(
        query=tokens,
        key=tokens,
        value=tokens,
        mode=mode,
        quantizer=lambda x: x,
        cache_metadata=metadata,
        mask_info=MaskInfo.from_attention_mask(mask),
        runtime_sharding_resolver=resolver,
    )


def test_metadata_indexes_control_write_position_and_starts_mask_old_slots():
    view = _view()
    tokens = jnp.array([[[[7.0]]]])
    metadata = OperationsMetadata.wrap(
        TransformerMetadata(indexes=jnp.array([3], jnp.int32), starts=jnp.array([2], jnp.int32))
    )
    key, _, mask_info, updated, _ = _concat(view, tokens, metadata, jnp.ones((1, 1, 8, 8), dtype=jnp.bool_))
    np.testing.assert_array_equal(np.asarray(key[0, :, 0, 0]), [0, 0, 0, 7, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(updated.indexes), [4])
    np.testing.assert_array_equal(np.asarray(updated.starts), [2])
    np.testing.assert_array_equal(
        np.asarray(mask_info.attention_mask[0, 0, 0]), [False, False, True, True] + [False] * 4
    )
    np.testing.assert_array_equal(np.asarray(view.indexes), [0])


def test_postpadded_prefill_advances_each_row_by_valid_token_count_and_decode_overwrites_padding():
    view = _view(batch=2)
    tokens = jnp.arange(6, dtype=jnp.float32).reshape(2, 3, 1, 1) + 1
    mask = jnp.broadcast_to(jnp.array([[True, True, False], [True, True, True]])[:, None, None, :], (2, 1, 3, 3))
    _, _, _, updated, _ = _concat(view, tokens, TransformerMetadata(postpadded=True), mask)
    np.testing.assert_array_equal(np.asarray(updated.indexes), [2, 3])
    following = jnp.full((2, 1, 1, 1), 9.0)
    key, _, _, decoded, _ = _concat(
        updated, following, TransformerMetadata(), jnp.ones((2, 1, 8, 8), jnp.bool_), mode=common_types.MODE_DECODE
    )
    np.testing.assert_array_equal(np.asarray(decoded.indexes), [3, 4])
    assert float(key[0, 2, 0, 0]) == 9.0
    assert float(key[1, 3, 0, 0]) == 9.0


def test_postpadded_sliding_prefill_keeps_valid_tokens_not_trailing_padding():
    view = _view(capacity=3, sliding=True)
    tokens = jnp.array([1, 2, 3, 4, 90, 91], jnp.float32).reshape(1, 6, 1, 1)
    mask = jnp.broadcast_to(jnp.array([True, True, True, True, False, False]), (1, 1, 6, 6))
    key, _, returned_mask, updated, _ = _concat(view, tokens, TransformerMetadata(postpadded=True), mask)
    np.testing.assert_array_equal(np.asarray(updated.key[0, :, 0, 0]), [2, 3, 4])
    np.testing.assert_array_equal(np.asarray(key[0, :, 0, 0]), [1, 2, 3, 4, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(updated.indexes), [4])
    expected_mask = np.zeros((6, 9), dtype=bool)
    for row in range(6):
        for column in range(4):
            expected_mask[row, column] = row - 3 < column <= row
    np.testing.assert_array_equal(np.asarray(returned_mask.attention_mask[0, 0]), expected_mask)


@pytest.mark.parametrize("metadata_class", [TransformerMetadata, HybridMetadata])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("postpadded", [False, True])
def test_compiled_updates_accept_direct_and_wrapped_hybrid_metadata(metadata_class, wrapped, sliding, postpadded):
    view = _view(capacity=3, sliding=sliding)
    tokens = jnp.array([7, 8, 90], jnp.float32).reshape(1, 3, 1, 1)
    mask = jnp.broadcast_to(jnp.array([True, True, not postpadded]), (1, 1, 3, 3))
    metadata = metadata_class(postpadded=postpadded, starts=jnp.array([0], jnp.int32), indexes=jnp.array([0], jnp.int32))
    if wrapped:
        metadata = OperationsMetadata.wrap(metadata)
    compiled_update = jax.jit(lambda view, tokens, metadata, mask: _concat(view, tokens, metadata, mask))
    key, value, _, updated, _ = compiled_update(view, tokens, metadata, mask)
    expected = [7, 8, 0] if postpadded else [7, 8, 90]
    np.testing.assert_array_equal(np.asarray(updated.key[0, :, 0, 0]), expected)
    np.testing.assert_array_equal(np.asarray(updated.value[0, :, 0, 0]), expected)
    attention_expected = expected + [0] * 3 if sliding else expected
    np.testing.assert_array_equal(np.asarray(key[0, :, 0, 0]), attention_expected)
    np.testing.assert_array_equal(np.asarray(value[0, :, 0, 0]), attention_expected)
    np.testing.assert_array_equal(np.asarray(updated.indexes), [2 if postpadded else 3])


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("metadata_class", [TransformerMetadata, HybridMetadata])
@pytest.mark.parametrize(
    "capacity, index, block_size, valid_length",
    [(8, 6, 4, 2), (8, 3, 4, 2), (8, 6, 4, 0), (8, 6, 4, 3), (3, 1, 5, 2)],
)
def test_compiled_postpadded_append_preserves_unwritten_slots(
    metadata_class, wrapped, capacity, index, block_size, valid_length
):
    old = np.arange(1, capacity + 1, dtype=np.float32).reshape(1, capacity, 1, 1)
    view = _view(capacity=capacity).replace(key=jnp.asarray(old), value=jnp.asarray(-old))
    incoming = np.arange(101, 101 + block_size, dtype=np.float32).reshape(1, block_size, 1, 1)
    tokens = jnp.asarray(incoming)
    # A query-local incoming mask also covers a padded block wider than capacity.
    mask = jnp.broadcast_to(jnp.arange(block_size) < valid_length, (1, 1, block_size, block_size))
    metadata = metadata_class(postpadded=True, indexes=jnp.array([index], jnp.int32))
    if wrapped:
        metadata = OperationsMetadata.wrap(metadata)
    compiled_update = jax.jit(lambda view, tokens, metadata, mask: _concat(view, tokens, metadata, mask))
    key, value, returned_mask, updated, _ = compiled_update(view, tokens, metadata, mask)
    expected_key = old.copy()
    expected_value = -old.copy()
    # Independent bounded-write reference: neither padding nor out-of-range rows
    # may overwrite an existing slot or move the requested write position.
    for offset in range(valid_length):
        if index + offset < capacity:
            expected_key[0, index + offset] = incoming[0, offset]
            expected_value[0, index + offset] = incoming[0, offset]
    np.testing.assert_array_equal(np.asarray(key), expected_key)
    np.testing.assert_array_equal(np.asarray(value), expected_value)
    np.testing.assert_array_equal(np.asarray(updated.key), expected_key)
    np.testing.assert_array_equal(np.asarray(updated.value), expected_value)
    np.testing.assert_array_equal(np.asarray(updated.indexes), [index + valid_length])
    np.testing.assert_array_equal(np.asarray(view.key), old)
    # Validate the public attention mask, not just writes: a local padding
    # column must never hide history or land at column zero after an append.
    expected_mask = np.zeros((1, 1, block_size, capacity), dtype=bool)
    if valid_length:
        for row in range(block_size):
            for column in range(capacity):
                expected_mask[0, 0, row, column] = column < index + valid_length and column <= index + row
    np.testing.assert_array_equal(np.asarray(returned_mask.attention_mask), expected_mask)
    assert returned_mask.causal_mask_baked_in


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("metadata_class", [TransformerMetadata, HybridMetadata])
@pytest.mark.parametrize("baked_causal", [False, True])
@pytest.mark.parametrize("mask_kind", ["pairwise", "segments", "absolute"])
@pytest.mark.parametrize("sliding", [False, True])
def test_compiled_padded_append_mask_uses_storage_coordinates(wrapped, metadata_class, baked_causal, mask_kind, sliding):
    capacity, block_size = 8, 4
    # Include a partially filled window, an append that rolls it, and a
    # sequence whose logical index is already beyond physical window capacity.
    indexes = np.array([3, 7, 10] if sliding else [3, 6, 3], dtype=np.int32)
    lengths = np.array([2, 2, 1], dtype=np.int32)
    starts = np.array([1, 4, 5] if sliding else [1, 4, 0], dtype=np.int32)
    old = np.arange(1, 25, dtype=np.float32).reshape(3, capacity, 1, 1)
    view = _view(batch=3, capacity=capacity, sliding=sliding).replace(key=jnp.asarray(old), value=jnp.asarray(-old))
    incoming = np.arange(101, 113, dtype=np.float32).reshape(3, block_size, 1, 1)
    local_mask = np.zeros((3, 1, block_size, block_size), dtype=bool)
    for batch, length in enumerate(lengths):
        local_mask[batch, :, :, :length] = True
        if mask_kind == "segments":
            local_mask[batch, :, length:, :] = False
        if baked_causal:
            local_mask[batch] &= np.tril(np.ones((block_size, block_size), dtype=bool))
    if mask_kind == "segments":
        segments = np.where(np.arange(block_size)[None, :] < lengths[:, None], 7, -1).astype(np.int32)
        mask_info = MaskInfo.from_segments(jnp.asarray(segments))
        if baked_causal:
            mask_info = mask_info.apply_causal()
    elif mask_kind == "absolute":
        width = 16
        absolute_mask = np.zeros((3, 1, width, width), dtype=bool)
        for batch, (index, length) in enumerate(zip(indexes, lengths, strict=True)):
            absolute_mask[batch, :, :, : index + length] = True
            # A known historical mask restriction must survive normalization.
            absolute_mask[batch, :, :, 2] = False
            if baked_causal:
                absolute_mask[batch] &= np.tril(np.ones((width, width), dtype=bool))
        mask_info = MaskInfo.from_attention_mask(jnp.asarray(absolute_mask)).replace(causal_mask_baked_in=baked_causal)
    else:
        mask_info = MaskInfo.from_attention_mask(jnp.asarray(local_mask)).replace(causal_mask_baked_in=baked_causal)
    metadata = metadata_class(postpadded=True, indexes=jnp.asarray(indexes), starts=jnp.asarray(starts))
    if wrapped:
        metadata = OperationsMetadata.wrap(metadata)

    @jax.jit
    def append(view, tokens, metadata, mask_info):
        return view.concatenate_to_cache(
            query=tokens,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=metadata,
            mask_info=mask_info,
        )

    key, _, returned_mask, updated, _ = append(view, jnp.asarray(incoming), metadata, mask_info)
    attention_width = capacity + block_size if sliding else capacity
    expected_key = np.zeros((3, attention_width, 1, 1), np.float32) if sliding else old.copy()
    expected_storage = np.zeros_like(old) if sliding else old.copy()
    expected_mask = np.zeros((3, 1, block_size, attention_width), dtype=bool)
    for batch, (index, length) in enumerate(zip(indexes, lengths, strict=True)):
        old_first = max(index - capacity, 0) if sliding else 0
        first = old_first if sliding else 0
        storage_first = max(index + length - capacity, 0) if sliding else 0
        for array, origin in ((expected_key, first), (expected_storage, storage_first)):
            for column in range(array.shape[1]):
                position = origin + column
                if position < index:
                    array[batch, column] = old[batch, position - old_first]
                elif position < index + length:
                    array[batch, column] = incoming[batch, position - index]
        for column in range(attention_width):
            position = first + column
            for row in range(block_size):
                visible = starts[batch] <= position < index + length
                if sliding:
                    visible &= index + row - capacity < position <= index + row
                if mask_kind == "absolute":
                    visible &= position != 2
                    if baked_causal:
                        visible &= position <= index + row
                else:
                    visible &= position <= index + row
                    if mask_kind == "segments":
                        visible &= row < length
                expected_mask[batch, 0, row, column] = visible
    np.testing.assert_array_equal(np.asarray(key), expected_key)
    np.testing.assert_array_equal(np.asarray(updated.key), expected_storage)
    np.testing.assert_array_equal(np.asarray(updated.indexes), indexes + lengths)
    np.testing.assert_array_equal(np.asarray(returned_mask.attention_mask), expected_mask)
    # Materialized and lazily reconstructed metadata must have matching widths.
    q_segments, kv_segments = returned_mask.get_or_compute_segment_ids()
    assert q_segments.shape[-1] == block_size
    assert kv_segments.shape[-1] == key.shape[1]
    if mask_kind == "segments":
        np.testing.assert_array_equal(np.asarray(q_segments), segments)


@pytest.mark.parametrize("mode", [common_types.MODE_PREFILL, common_types.MODE_DECODE, common_types.MODE_TRAIN])
def test_runtime_mode_selects_result_cache_sharding(mode):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two accelerator devices for observable sharding")
    mesh = Mesh(np.array(devices[:2]), ("tp",))
    paxis = PartitionAxis(
        batch_axis="tp",
        decode_batch_axis=None,
        kv_head_axis=None,
        decode_kv_head_axis=None,
        key_sequence_axis=None,
        decode_key_sequence_axis=None,
    )
    resolver = coerce_runtime_sharding_resolver(paxis, mesh=mesh)
    view = _view(batch=2)
    tokens = jax.device_put(jnp.ones((2, 1, 1, 1)), NamedSharding(mesh, PartitionSpec()))
    key, _, _, updated, _ = _concat(view, tokens, None, jnp.ones((2, 1, 8, 8), jnp.bool_), resolver=resolver, mode=mode)
    expected = (
        PartitionSpec(None, None, None, None)
        if mode == common_types.MODE_DECODE
        else PartitionSpec("tp", None, None, None)
    )
    assert key.sharding.is_equivalent_to(NamedSharding(mesh, expected), ndim=4)
    expected_batch = PartitionSpec(None) if mode == common_types.MODE_DECODE else PartitionSpec("tp")
    assert updated.indexes.sharding.is_equivalent_to(NamedSharding(mesh, expected_batch), ndim=1)
