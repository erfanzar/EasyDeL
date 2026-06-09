from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np

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


def test_spmd_packs_tokens_rank_major_and_builds_rank_local_metadata():
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    preparer = BatchMetadataPreparer(
        metadata=_dummy_metadata(),
        empty_sharding=sharding,
        max_num_tokens=8,
        max_num_reqs=4,
        max_model_len=16,
        min_input_pad=1,
        input_sharding=sharding,
    )

    token_ids = np.arange(4 * 16, dtype=np.int32).reshape(4, 16)
    scheduled = np.array([2, 1, 3, 1], dtype=np.int32)
    num_computed = np.array([0, 5, 0, 7], dtype=np.int32)
    host_payload, padded_num_reqs, num_requests, _rows_to_copy = preparer._build_host_payload(
        num_tokens_static=8,
        scheduled_full_cpu=scheduled,
        active_mask_full_cpu=np.array([True, True, True, True], dtype=bool),
        token_ids_cpu=token_ids,
        num_computed_tokens_cpu=num_computed,
        temperature_cpu=np.ones((4,), dtype=np.float32),
        top_p_cpu=np.ones((4,), dtype=np.float32),
        top_k_cpu=np.zeros((4,), dtype=np.int32),
        min_p_cpu=np.zeros((4,), dtype=np.float32),
        frequency_penalties_cpu=np.zeros((4,), dtype=np.float32),
        presence_penalties_cpu=np.zeros((4,), dtype=np.float32),
        repetition_penalties_cpu=np.ones((4,), dtype=np.float32),
        page_table_cpu=np.zeros((4, 2), dtype=np.int32),
        page_table_version=None,
        padded_num_reqs_in=4,
        copy_slot_mapping=False,
        window_row_indices_cpu=np.arange(4, dtype=np.int32),
    )

    input_ids, positions, _packed_qsl_seqlens, _pages, packed_i32, _packed_f32, packed_misc = host_payload[:7]
    dp_query_start_loc, dp_request_distribution, dp_context_lens, dp_recurrent_state_indices = host_payload[7:11]

    assert padded_num_reqs == 4
    assert num_requests == 4
    np.testing.assert_array_equal(input_ids, np.array([0, 1, 21, 0, 32, 33, 34, 55], dtype=np.int32))
    np.testing.assert_array_equal(positions, np.array([0, 1, 5, 0, 0, 1, 2, 7], dtype=np.int32))
    np.testing.assert_array_equal(packed_i32[1, :4], np.array([1, 2, 6, 7], dtype=np.int32))
    np.testing.assert_array_equal(packed_misc[2:5], np.array([2, 4, 4], dtype=np.int32))
    np.testing.assert_array_equal(dp_query_start_loc, np.array([[0, 2, 3], [0, 3, 4]], dtype=np.int32))
    np.testing.assert_array_equal(dp_request_distribution, np.array([[1, 2, 2], [1, 2, 2]], dtype=np.int32))
    np.testing.assert_array_equal(dp_context_lens, np.array([[2, 6], [3, 8]], dtype=np.int32))
    np.testing.assert_array_equal(dp_recurrent_state_indices, np.array([[0, 1], [0, 1]], dtype=np.int32))


def test_spmd_accepts_sparse_global_rows_and_state_indices():
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    preparer = BatchMetadataPreparer(
        metadata=_dummy_metadata(),
        empty_sharding=sharding,
        max_num_tokens=8,
        max_num_reqs=4,
        max_model_len=16,
        min_input_pad=1,
        input_sharding=sharding,
    )

    host_payload, _padded_num_reqs, num_requests, _rows_to_copy = preparer._build_host_payload(
        num_tokens_static=8,
        scheduled_full_cpu=np.array([2, 1, 0, 0], dtype=np.int32),
        active_mask_full_cpu=np.array([True, True, False, False], dtype=bool),
        token_ids_cpu=np.arange(4 * 16, dtype=np.int32).reshape(4, 16),
        num_computed_tokens_cpu=np.array([0, 5, 0, 0], dtype=np.int32),
        temperature_cpu=np.ones((4,), dtype=np.float32),
        top_p_cpu=np.ones((4,), dtype=np.float32),
        top_k_cpu=np.zeros((4,), dtype=np.int32),
        min_p_cpu=np.zeros((4,), dtype=np.float32),
        frequency_penalties_cpu=np.zeros((4,), dtype=np.float32),
        presence_penalties_cpu=np.zeros((4,), dtype=np.float32),
        repetition_penalties_cpu=np.ones((4,), dtype=np.float32),
        page_table_cpu=np.zeros((4, 2), dtype=np.int32),
        page_table_version=None,
        padded_num_reqs_in=4,
        copy_slot_mapping=False,
        window_row_indices_cpu=np.array([2, 3, 0, 0], dtype=np.int32),
        recurrent_slot_indices_cpu=np.array([3, 2, 0, 0], dtype=np.int32),
    )

    input_ids, positions, _packed_qsl_seqlens, _pages, packed_i32, _packed_f32, _packed_misc = host_payload[:7]
    dp_query_start_loc, dp_request_distribution, dp_context_lens, dp_recurrent_state_indices = host_payload[7:11]

    assert num_requests == 2
    np.testing.assert_array_equal(input_ids, np.array([0, 0, 0, 0, 0, 1, 21, 0], dtype=np.int32))
    np.testing.assert_array_equal(positions, np.array([0, 0, 0, 0, 0, 1, 5, 0], dtype=np.int32))
    np.testing.assert_array_equal(packed_i32[1, :2], np.array([5, 6], dtype=np.int32))
    np.testing.assert_array_equal(dp_query_start_loc, np.array([[0, 0, 0], [0, 2, 3]], dtype=np.int32))
    np.testing.assert_array_equal(dp_request_distribution, np.array([[0, 0, 0], [1, 2, 2]], dtype=np.int32))
    np.testing.assert_array_equal(dp_context_lens, np.array([[0, 0], [2, 6]], dtype=np.int32))
    np.testing.assert_array_equal(dp_recurrent_state_indices, np.array([[0, 0], [1, 0]], dtype=np.int32))
