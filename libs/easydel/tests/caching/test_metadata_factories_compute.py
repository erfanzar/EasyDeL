"""Accelerator-only regressions for the documented metadata factory API."""

import numpy as np
from easydel.caching import AttentionMetadataBuilder, RaggedPagesMetadata, RecurrentMetadata, TransformerMetadata
from jax import numpy as jnp


def test_factory_infers_transformer_fields_and_allows_explicit_overrides():
    view = type("View", (), {"starts": jnp.array([2]), "indexes": jnp.array([5])})()
    metadata = AttentionMetadataBuilder.build(cache_view=view, postpadded=True)
    assert isinstance(metadata, TransformerMetadata)
    assert metadata.postpadded is True
    np.testing.assert_array_equal(np.asarray(metadata.starts), [2])
    np.testing.assert_array_equal(np.asarray(metadata.indexes), [5])
    explicit = AttentionMetadataBuilder.build_transformer_metadata(cache_view=view, starts=[1], indexes=[7])
    np.testing.assert_array_equal(np.asarray(explicit.starts), [1])
    np.testing.assert_array_equal(np.asarray(explicit.indexes), [7])


def test_factory_derives_paged_layout_from_raw_batch_and_honors_aliases():
    metadata = AttentionMetadataBuilder.build(
        attention_mechanism="ragged_page_attention_v2",
        version="v2",
        scheduled_full=[5, 0],
        active_mask_full=[True, False],
        num_computed_tokens=[0, 0],
        page_table=[[7, 8], [0, 0]],
        page_size=4,
        num_slices_per_kv_cache_update_page=4,
    )
    assert isinstance(metadata, RaggedPagesMetadata)
    np.testing.assert_array_equal(np.asarray(metadata.slot_mapping[:, :2]), [[28, 32], [0, 4], [4, 1]])
    np.testing.assert_array_equal(np.asarray(metadata.num_kv_update_slices), [2])
    alias = AttentionMetadataBuilder.build_ragged_page_metadata(
        block_tables=[[7, 8]], seq_lens=[5], query_start_loc=[0, 5], num_seqs=1, request_distribution=[0, 1, 1]
    )
    np.testing.assert_array_equal(np.asarray(alias.pages_tables), [[7, 8]])
    np.testing.assert_array_equal(np.asarray(alias.context_lens), [5])
    np.testing.assert_array_equal(np.asarray(alias.num_seqs), [1])
    assert isinstance(AttentionMetadataBuilder.build(expected_cache_type="recurrent"), RecurrentMetadata)
