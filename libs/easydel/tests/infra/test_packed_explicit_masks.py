"""Packed boundaries intersect explicit visibility, never replace it."""

import jax
import numpy as np
import pytest
from easydel.infra.sequence_packing import (
    fold_sequence_packing_segments,
    packed_segment_ids_from_mask_info,
    pairwise_attention_mask_from_mask_info,
)
from ejkernel.types import MaskInfo
from jax import numpy as jnp


@pytest.mark.parametrize("mask_kind", ["causal", "sparse", "block"])
def test_explicit_pairwise_visibility_survives_padding_segments(mask_kind):
    if mask_kind == "causal":
        expected = np.tri(4, dtype=bool)
    elif mask_kind == "sparse":
        expected = np.eye(4, dtype=bool) | np.eye(4, k=-1, dtype=bool)
    else:
        expected = np.equal.outer([0, 0, 1, 1], [0, 0, 1, 1])
    info = MaskInfo.from_attention_mask(jnp.asarray(expected[None]))
    result = pairwise_attention_mask_from_mask_info(info, 4, 4)
    np.testing.assert_array_equal(result, expected[None])


def test_baked_causality_survives_segment_extraction():
    info = MaskInfo.from_segments(jnp.array([[0, 0, 1, 1]])).apply_causal()
    expected = np.tri(4, dtype=bool) & np.equal.outer([0, 0, 1, 1], [0, 0, 1, 1])
    np.testing.assert_array_equal(pairwise_attention_mask_from_mask_info(info, 4, 4), expected[None])


@pytest.mark.parametrize("mask_rank", [2, 3, 4])
@pytest.mark.parametrize("existing_kind", ["none", "empty", "padding", "explicit", "segments"])
def test_folding_preserves_boundaries_visibility_positions_and_rank(mask_rank, existing_kind):
    segments = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    visibility = np.broadcast_to(np.tri(4, dtype=bool), (4, 4, 4)).copy()
    visibility[:, 3, :] = False
    if mask_rank == 2:
        attention_mask = np.array([[1, 1, 1, 0]] * 4, dtype=bool)
        visibility = attention_mask[:, :, None] & attention_mask[:, None, :]
    else:
        attention_mask = visibility if mask_rank == 3 else visibility[:, None]
    positions = jnp.broadcast_to(jnp.arange(4), (4, 4))
    info = None
    if existing_kind == "empty":
        info = MaskInfo(q_positions=positions, batch_axis_name="dp")
    elif existing_kind == "padding":
        info = MaskInfo.from_attention_mask(jnp.ones((4, 4), dtype=bool), q_positions=positions)
    elif existing_kind == "explicit":
        info = MaskInfo.from_attention_mask(jnp.asarray(np.eye(4, dtype=bool)[None]), q_positions=positions)
        visibility &= np.eye(4, dtype=bool)[None]
    elif existing_kind == "segments":
        info = MaskInfo.from_segments(jnp.array([[0, 1, 1, 1]] * 4), q_positions=positions)
        visibility &= np.equal.outer([0, 1, 1, 1], [0, 1, 1, 1])[None]
    result = fold_sequence_packing_segments(
        {"segment_ids": jnp.asarray(segments), "attention_mask": jnp.asarray(attention_mask), "mask_info": info}
    )
    folded = result["mask_info"]
    assert "segment_ids" not in result
    assert folded._q_segment_ids.shape == (4, 4)
    assert folded._kv_segment_ids.shape == (4, 4)
    assert packed_segment_ids_from_mask_info(folded, 4).shape == (4, 4)
    expected = visibility & (segments[:, :, None] == segments[:, None, :])
    np.testing.assert_array_equal(pairwise_attention_mask_from_mask_info(folded, 4, 4), expected)
    np.testing.assert_array_equal(folded.attention_mask[:, 0], expected)
    if info is not None:
        assert folded.q_positions is positions
        assert folded.batch_axis_name == info.batch_axis_name
    if existing_kind == "segments":
        # Both sources' boundaries must reach recurrent consumers, not just attention.
        packed = np.asarray(packed_segment_ids_from_mask_info(folded, 4))
        assert np.all(packed[:, 0] != packed[:, 1])
        assert np.all(packed[:, 1] != packed[:, 2])


def test_folding_keeps_all_head_masks_and_pairwise_extraction_rejects_head_loss():
    per_head = np.stack([np.eye(4, dtype=bool), np.tri(4, dtype=bool)], axis=0)[None]
    info = fold_sequence_packing_segments(
        {"segment_ids": jnp.array([[0, 0, 1, 1]]), "attention_mask": jnp.asarray(per_head)}
    )["mask_info"]
    expected = per_head & np.equal.outer([0, 0, 1, 1], [0, 0, 1, 1])[None, None]
    np.testing.assert_array_equal(info.attention_mask, expected)
    with pytest.raises(ValueError, match="head-dependent"):
        pairwise_attention_mask_from_mask_info(info, 4, 4)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("old_batch", [1, 4])
@pytest.mark.parametrize("mask_shape", [(4, 4), (4, 4, 4), (4, 1, 4, 4), (4, 2, 4, 4)])
def test_abstract_folding_preserves_two_dimensional_segments(mask_shape, old_batch):
    def fold(segments, attention, old_segments):
        existing = MaskInfo(_q_segment_ids=old_segments, _kv_segment_ids=old_segments)
        return fold_sequence_packing_segments(
            {"segment_ids": segments, "attention_mask": attention, "mask_info": existing}
        )["mask_info"]

    result = jax.eval_shape(
        fold,
        jax.ShapeDtypeStruct((4, 4), jnp.int32),
        jax.ShapeDtypeStruct(mask_shape, jnp.bool_),
        jax.ShapeDtypeStruct((old_batch, 4), jnp.int32),
    )
    assert result._q_segment_ids.shape == (4, 4)
    assert result._kv_segment_ids.shape == (4, 4)
    if len(mask_shape) >= 3:
        assert result._attention_mask.shape == (4, mask_shape[1] if len(mask_shape) == 4 else 1, 4, 4)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("batches", [(4, 1, 1), (1, 4, 4), (1, 1, 4), (1, 4, 1)])
def test_abstract_folding_broadcasts_rectangular_query_and_key_segments(batches):
    def fold(supplied, old_q, old_kv, inputs):
        existing = MaskInfo(_q_segment_ids=old_q, _kv_segment_ids=old_kv)
        return fold_sequence_packing_segments({"segment_ids": supplied, "input_ids": inputs, "mask_info": existing})[
            "mask_info"
        ]

    supplied_batch, q_batch, kv_batch = batches
    result = jax.eval_shape(
        fold,
        jax.ShapeDtypeStruct((supplied_batch, 6), jnp.int32),
        jax.ShapeDtypeStruct((q_batch, 4), jnp.int32),
        jax.ShapeDtypeStruct((kv_batch, 6), jnp.int32),
        jax.ShapeDtypeStruct((4, 4), jnp.int32),
    )
    assert result._q_segment_ids.shape == (4, 4)
    assert result._kv_segment_ids.shape == (4, 6)
    assert result._attention_mask is None


@pytest.mark.parametrize("batches", [(4, 1, 1), (1, 4, 4), (1, 1, 4), (1, 4, 1)])
def test_broadcast_segment_partitions_preserve_rectangular_visibility(batches):
    supplied_batch, q_batch, kv_batch = batches
    supplied = np.tile(np.array([[7, 7, 9, 9, 7, 9]], dtype=np.int32), (supplied_batch, 1))
    old_q = np.array([[3, 3, 4, 4], [3, 4, 3, 4], [4, 4, 3, 3], [3, -1, 3, 4]], dtype=np.int32)[:q_batch]
    old_kv = np.array([[3, 3, 4, 4, 3, 4], [3, 4, 3, 4, 3, 4], [4, 4, 3, 3, 3, 4], [3, 3, 3, 4, -1, 3]], dtype=np.int32)[
        :kv_batch
    ]
    info = fold_sequence_packing_segments(
        {
            "segment_ids": jnp.asarray(supplied),
            "input_ids": jnp.zeros((4, 4), dtype=jnp.int32),
            "mask_info": MaskInfo.from_segments(jnp.asarray(old_q), jnp.asarray(old_kv)),
        }
    )["mask_info"]
    expected = (
        (supplied[:, :4, None] == supplied[:, None, :])
        & (old_q[:, :, None] == old_kv[:, None, :])
        & (old_q[:, :, None] >= 0)
        & (old_kv[:, None, :] >= 0)
    )
    np.testing.assert_array_equal(pairwise_attention_mask_from_mask_info(info, 4, 6), expected)
    np.testing.assert_array_equal(info.attention_mask[:, 0], expected)
    # Recurrent consumers need the same broadcast batch and query padding.
    packed = packed_segment_ids_from_mask_info(info, 4)
    assert packed.shape == (4, 4)
    np.testing.assert_array_equal(np.asarray(packed) < 0, np.broadcast_to(old_q < 0, (4, 4)))


def test_existing_and_supplied_segment_partitions_are_intersected_even_for_repeated_ids():
    supplied = np.array([[7, 9, 7, 7]], dtype=np.int32)
    existing = np.array([[3, 3, 3, 4]], dtype=np.int32)
    info = fold_sequence_packing_segments(
        {"segment_ids": jnp.asarray(supplied), "mask_info": MaskInfo.from_segments(jnp.asarray(existing))}
    )["mask_info"]
    expected = (supplied[:, :, None] == supplied[:, None, :]) & (existing[:, :, None] == existing[:, None, :])
    np.testing.assert_array_equal(pairwise_attention_mask_from_mask_info(info, 4, 4), expected)
