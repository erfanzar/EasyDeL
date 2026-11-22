"""Rank-major SPMD DP step inputs of BatchMetadataPreparer, through its synchronous and its async buffer set."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer


def _dummy_metadata(*, max_num_seqs: int = 4, dp_size: int = 2):
    return SimpleNamespace(
        version="v3",
        get_max_num_seqs=lambda: max_num_seqs,
        max_num_pages_per_req=2,
        data_parallel_size=dp_size,
        page_size=16,
        num_pages=0,
    )


def _preparer() -> BatchMetadataPreparer:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    return BatchMetadataPreparer(
        metadata=_dummy_metadata(),
        empty_sharding=sharding,
        max_num_tokens=8,
        max_num_reqs=4,
        max_model_len=16,
        min_input_pad=1,
        input_sharding=sharding,
    )


def _prepare(buffer_set: str, **step):
    """One step's BatchMetadata from the synchronous entry point, or from the async (double-buffered) one."""
    preparer = _preparer()
    step.update(
        num_tokens_static=8,
        input_ids_buf=None,
        position_ids_buf=None,
        token_ids_cpu=np.arange(4 * 16, dtype=np.int32).reshape(4, 16),
        temperature_cpu=np.ones((4,), dtype=np.float32),
        top_p_cpu=np.ones((4,), dtype=np.float32),
        top_k_cpu=np.zeros((4,), dtype=np.int32),
        min_p_cpu=np.zeros((4,), dtype=np.float32),
        frequency_penalties_cpu=np.zeros((4,), dtype=np.float32),
        presence_penalties_cpu=np.zeros((4,), dtype=np.float32),
        repetition_penalties_cpu=np.ones((4,), dtype=np.float32),
        page_table_cpu=np.zeros((4, 2), dtype=np.int32),
        padded_num_reqs_in=4,
    )
    if buffer_set == "sync":
        metadata, *_ = preparer.prepare_batch_metadata(**step)
        return metadata
    preparer.start_async_prep(**step)
    (metadata, *_), _timings = preparer.get_async_prep_result()
    return metadata


@pytest.mark.parametrize("buffer_set", ["sync", "async"])
def test_spmd_packs_tokens_rank_major_and_builds_rank_local_metadata(buffer_set):
    metadata = _prepare(
        buffer_set,
        scheduled_full_cpu=np.array([2, 1, 3, 1], dtype=np.int32),
        active_mask_full_cpu=np.array([True, True, True, True], dtype=bool),
        num_computed_tokens_cpu=np.array([0, 5, 0, 7], dtype=np.int32),
        window_row_indices_cpu=np.arange(4, dtype=np.int32),
    )

    packed_misc = np.asarray(metadata.packed_misc_i32)
    assert packed_misc[0] == 4  # num_requests
    assert packed_misc[1] == 4  # padded_num_reqs
    np.testing.assert_array_equal(np.asarray(metadata.input_ids_buf), [0, 1, 21, 0, 32, 33, 34, 55])
    np.testing.assert_array_equal(np.asarray(metadata.position_ids_buf), [0, 1, 5, 0, 0, 1, 2, 7])
    np.testing.assert_array_equal(np.asarray(metadata.packed_i32_padded)[1, :4], [1, 2, 6, 7])
    # POSITIONAL v3 contract: rows are [prefill, decode, prefill, decode], so
    # the decode prefix is empty and the prefill run is a single row; the
    # interleaved tail goes to the mixed range (count-based fills desync the
    # kernel's per-class offsets and hang the device).
    np.testing.assert_array_equal(packed_misc[2:5], [0, 1, 4])
    np.testing.assert_array_equal(np.asarray(metadata.dp_query_start_loc), [[0, 2, 3], [0, 3, 4]])
    # Each rank's rows are [prefill, decode]: empty decode prefix, one prefill.
    np.testing.assert_array_equal(np.asarray(metadata.dp_request_distribution), [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_array_equal(np.asarray(metadata.dp_context_lens), [[2, 6], [3, 8]])
    np.testing.assert_array_equal(np.asarray(metadata.dp_recurrent_state_indices), [[0, 1], [0, 1]])


@pytest.mark.parametrize("buffer_set", ["sync", "async"])
def test_spmd_accepts_sparse_global_rows_and_state_indices(buffer_set):
    metadata = _prepare(
        buffer_set,
        scheduled_full_cpu=np.array([2, 1, 0, 0], dtype=np.int32),
        active_mask_full_cpu=np.array([True, True, False, False], dtype=bool),
        num_computed_tokens_cpu=np.array([0, 5, 0, 0], dtype=np.int32),
        window_row_indices_cpu=np.array([2, 3, 0, 0], dtype=np.int32),
        recurrent_slot_indices_cpu=np.array([3, 2, 0, 0], dtype=np.int32),
    )

    assert np.asarray(metadata.packed_misc_i32)[0] == 2  # num_requests
    np.testing.assert_array_equal(np.asarray(metadata.input_ids_buf), [0, 0, 0, 0, 0, 1, 21, 0])
    np.testing.assert_array_equal(np.asarray(metadata.position_ids_buf), [0, 0, 0, 0, 0, 1, 5, 0])
    np.testing.assert_array_equal(np.asarray(metadata.packed_i32_padded)[1, :2], [5, 6])
    np.testing.assert_array_equal(np.asarray(metadata.dp_query_start_loc), [[0, 0, 0], [0, 2, 3]])
    # rank1 rows are [prefill(2 tok), decode]: positional prefix is [0, 1, 2].
    np.testing.assert_array_equal(np.asarray(metadata.dp_request_distribution), [[0, 0, 0], [0, 1, 2]])
    np.testing.assert_array_equal(np.asarray(metadata.dp_context_lens), [[0, 0], [2, 6]])
    np.testing.assert_array_equal(np.asarray(metadata.dp_recurrent_state_indices), [[0, 0], [1, 0]])
