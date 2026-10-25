"""Cache numerics against NumPy references; accelerator-only (array computation)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.caching import AttentionMetadataBuilder, TransformerCacheConfig, TransformerCacheView, TransformerMetadata
from easydel.caching.hybrid.cache import HybridCacheView, ParallelHybridCacheView
from easydel.caching.mla_ragged_page.cache import MLARaggedPagesCacheView
from easydel.caching.recurrent.cache import RecurrentCacheView
from easydel.caching.transformer.cache import AttnMaskDetail
from easydel.caching.turboquant_ragged_page.cache import TurboQuantRaggedPagesCacheView
from easydel.infra.utils import AttnMaskType
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def test_local_ordinary_decode_attention_uses_physical_segment_history():
    config = TransformerCacheConfig.create(batch_size=1, sequence_length=8, num_hidden_layers=1, num_heads=1, head_dim=1)
    history = jnp.array([10, 20, 30, 40, 0, 0, 0, 0], jnp.float32).reshape(1, 8, 1, 1)
    view = TransformerCacheView(
        key=history,
        value=history,
        indexes=jnp.array([4], jnp.int32),
        starts=jnp.array([0], jnp.int32),
        metadata=config,
        maximum_sequence_length=8,
        segment_ids=jnp.array([[0, 0, 1, 1, -1, -1, -1, -1]], jnp.int32),
    )
    _, value, mask, updated, _ = view.concatenate_to_cache(
        query=jnp.zeros((1, 1, 1, 1)),
        key=jnp.full((1, 1, 1, 1), 50.0),
        value=jnp.full((1, 1, 1, 1), 50.0),
        mode=common_types.MODE_DECODE,
        quantizer=lambda x: x,
        cache_metadata=TransformerMetadata(),
        mask_info=MaskInfo.from_attention_mask(jnp.ones((1, 1, 1, 1), jnp.bool_)),
    )
    q_segments, kv_segments = mask.get_or_compute_segment_ids()
    visible = mask.attention_mask & (q_segments[:, None, :, None] == kv_segments[:, None, None, :])
    weights = jax.nn.softmax(jnp.where(visible, jnp.float32(0), -jnp.inf), axis=-1)
    output = jnp.einsum("bhqk,bkhd->bqhd", weights, value, precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_allclose(np.asarray(output), [[[[40.0]]]], atol=1e-5, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(kv_segments), [[0, 0, 1, 1, 1, -1, -1, -1]])
    np.testing.assert_array_equal(np.asarray(updated.segment_ids), np.asarray(kv_segments))


def test_precomputed_v3_factory_routes_nonempty_work_to_mixed_range():
    metadata = AttentionMetadataBuilder.build_ragged_page_metadata(
        pages_tables=[[0]], context_lens=[1], query_start_loc=[0, 1], num_seqs=1, version="v3"
    )
    np.testing.assert_array_equal(np.asarray(metadata.request_distribution), [0, 0, 1])


def test_raw_v2_metadata_builder_resolves_config_geometry():
    config = SimpleNamespace(page_size=4, num_slices_per_kv_cache_update_page=4, max_num_pages_per_req=2)
    metadata = AttentionMetadataBuilder.build_ragged_page_metadata(
        scheduled_full=[5, 0],
        active_mask_full=[True, False],
        num_computed_tokens=[0, 0],
        page_table=[[7, 8], [0, 0]],
        version="v2",
        ragged_config=config,
    )
    assert metadata.page_size == 4
    assert metadata.num_slices_per_kv_cache_update_page == 4
    np.testing.assert_array_equal(np.asarray(metadata.slot_mapping[:, :2]), [[28, 32], [0, 4], [4, 1]])


@pytest.mark.parametrize("chunks", [(6,), (2, 4), (1, 2, 3), (4, 1, 1)])
@pytest.mark.parametrize("postpadded", [False, True])
def test_sliding_attention_matches_independent_full_sequence_reference(chunks, postpadded):
    window = 3
    keys = np.array([0.2, -0.3, 0.7, 1.1, -0.6, 0.5], np.float32)
    values = np.array([1.0, 5.0, -2.0, 3.0, 4.0, -1.0], np.float32)
    queries = np.array([0.9, 0.3, -0.5, 0.6, 0.8, -0.2], np.float32)
    config = TransformerCacheConfig.create(
        batch_size=1, sequence_length=window, num_hidden_layers=1, num_heads=1, head_dim=1
    )
    view = TransformerCacheView(
        key=jnp.zeros((1, window, 1, 1)),
        value=jnp.zeros((1, window, 1, 1)),
        indexes=jnp.zeros((1,), jnp.int32),
        starts=jnp.zeros((1,), jnp.int32),
        metadata=config,
        maximum_sequence_length=window,
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=window),
    )
    outputs = []
    offset = 0
    for length in chunks:
        padded_length = length + 2 if postpadded else length
        q = np.pad(queries[offset : offset + length], (0, padded_length - length)).reshape(1, padded_length, 1, 1)
        k = np.pad(keys[offset : offset + length], (0, padded_length - length)).reshape(1, padded_length, 1, 1)
        v = np.pad(values[offset : offset + length], (0, padded_length - length)).reshape(1, padded_length, 1, 1)
        mask = np.broadcast_to(np.arange(padded_length) < length, (1, 1, padded_length, padded_length))
        key, value, mask_info, view, _ = view.concatenate_to_cache(
            query=jnp.asarray(q),
            key=jnp.asarray(k),
            value=jnp.asarray(v),
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=TransformerMetadata(postpadded=postpadded),
            mask_info=MaskInfo.from_attention_mask(jnp.asarray(mask)),
        )
        logits = jnp.einsum("bqhd,bkhd->bhqk", jnp.asarray(q), key, precision=jax.lax.Precision.HIGHEST)
        probabilities = jax.nn.softmax(
            jnp.where(mask_info.attention_mask, logits.astype(jnp.float32), -jnp.inf), axis=-1
        )
        output = jnp.einsum("bhqk,bkhd->bqhd", probabilities, value, precision=jax.lax.Precision.HIGHEST)
        outputs.extend(np.asarray(output)[0, :length, 0, 0])
        offset += length
    expected = []
    for row in range(len(queries)):
        first = max(0, row - window + 1)
        scores = queries[row] * keys[first : row + 1]
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        expected.append(np.sum(probabilities * values[first : row + 1]))
    np.testing.assert_allclose(outputs, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(np.asarray(view.key)[0, :, 0, 0], keys[-window:])
    np.testing.assert_array_equal(np.asarray(view.value)[0, :, 0, 0], values[-window:])


@pytest.mark.parametrize("layer_type", ["full_attention", "parallel_hybrid"])
def test_hybrid_writes_each_row_at_its_own_position(layer_type):
    old = jnp.zeros((2, 8, 1, 4))
    positions = jnp.array([1, 3], jnp.int32)
    view = HybridCacheView(
        key=old,
        value=old,
        positions=positions,
        metadata=None,
        conv_state=None,
        recurrent_state=None,
        layer_type=layer_type,
    )
    tokens = jnp.arange(8, dtype=jnp.float32).reshape(2, 1, 1, 4) + 1
    key, value, updated = jax.jit(
        lambda view, tokens: view.concatenate_to_cache(key_states=tokens, value_states=-tokens)
    )(view, tokens)
    expected = np.zeros((2, 8, 1, 4), np.float32)
    expected[0, 1] = np.asarray(tokens)[0, 0]
    expected[1, 3] = np.asarray(tokens)[1, 0]
    np.testing.assert_array_equal(np.asarray(key), expected)
    np.testing.assert_array_equal(np.asarray(value), -expected)
    np.testing.assert_array_equal(np.asarray(updated.positions), [2, 4])


def test_recurrent_conv_insertion_pairs_each_position_with_its_batch_row():
    old = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    state = np.array([[31, 32, 33], [41, 42, 43]], np.float32)
    view = RecurrentCacheView(
        conv_state=jnp.asarray(old), recurrent_state=None, positions=None, metadata=SimpleNamespace(conv_kernel_size=4)
    )
    actual, _, _ = view.concatenate_to_cache(conv_state=jnp.asarray(state), cache_position=jnp.array([0, 2]))
    expected = np.roll(old, -1, axis=-1)
    expected[0, :, 0] = state[0]
    expected[1, :, 2] = state[1]
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("specialization", ["mla", "turboquant"])
@pytest.mark.parametrize("composite", [False, True])
def test_specialized_reset_clears_all_owned_state_and_preserves_constants(specialization, composite):
    if specialization == "mla":
        names = ("kv_pages", "indexer_token_pages", "indexer_pool_pages")
        view = MLARaggedPagesCacheView(
            metadata=None,
            layer_index=0,
            kv_pages=jnp.ones((2, 4, 1, 8)),
            indexer_token_pages=jnp.ones((2, 4, 3)),
            indexer_pool_pages=jnp.ones((2, 2, 2)),
        )
    else:
        names = ("key_indices_pages", "key_signs_pages", "key_norms_pages", "value_indices_pages", "value_norms_pages")
        view = TurboQuantRaggedPagesCacheView(
            metadata=None,
            layer_index=0,
            kv_pages=jnp.ones(()),
            constants=("retained",),
            **{name: jnp.ones((2, 4, 1, 2)) for name in names},
        )
    reset = ParallelHybridCacheView(transformer=view, recurrent=None).reset().transformer if composite else view.reset()
    for name in names:
        np.testing.assert_array_equal(np.asarray(getattr(reset, name)), np.zeros(getattr(view, name).shape))
        np.testing.assert_array_equal(np.asarray(getattr(view, name)), np.ones(getattr(view, name).shape))
    assert reset.metadata is view.metadata
    if specialization == "turboquant":
        assert reset.constants is view.constants
    else:
        no_indexer = view.replace(indexer_token_pages=None, indexer_pool_pages=None).reset()
        assert no_indexer.indexer_token_pages is None
        assert no_indexer.indexer_pool_pages is None
