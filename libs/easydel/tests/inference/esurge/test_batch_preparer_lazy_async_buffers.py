# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Async double-buffer staging buffers are lazy; the public async API still works."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer

N_TOK, N_REQ, MODEL_LEN, N_PAGES = 8, 4, 16, 2


def _preparer(version: str = "v3") -> BatchMetadataPreparer:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    metadata = SimpleNamespace(
        version=version,
        get_max_num_seqs=lambda: N_REQ,
        max_num_pages_per_req=N_PAGES,
        data_parallel_size=1,
        page_size=16,
        num_pages=0,
        num_slices_per_kv_cache_update_page=1,
        get_padded_num_slices=lambda tokens, reqs: tokens,
    )
    return BatchMetadataPreparer(
        metadata=metadata,
        empty_sharding=sharding,
        max_num_tokens=N_TOK,
        max_num_reqs=N_REQ,
        max_model_len=MODEL_LEN,
        min_input_pad=1,
        input_sharding=sharding,
    )


def _kwargs():
    return dict(
        num_tokens_static=N_TOK,
        scheduled_full_cpu=np.ones((N_REQ,), dtype=np.int32),
        active_mask_full_cpu=np.ones((N_REQ,), dtype=bool),
        input_ids_buf=None,
        position_ids_buf=None,
        token_ids_cpu=np.zeros((N_REQ, MODEL_LEN), dtype=np.int32),
        num_computed_tokens_cpu=np.full((N_REQ,), 3, dtype=np.int32),
        temperature_cpu=np.zeros((N_REQ,), dtype=np.float32),
        top_p_cpu=np.ones((N_REQ,), dtype=np.float32),
        top_k_cpu=np.zeros((N_REQ,), dtype=np.int32),
        min_p_cpu=np.zeros((N_REQ,), dtype=np.float32),
        frequency_penalties_cpu=np.zeros((N_REQ,), dtype=np.float32),
        presence_penalties_cpu=np.zeros((N_REQ,), dtype=np.float32),
        repetition_penalties_cpu=np.ones((N_REQ,), dtype=np.float32),
        page_table_cpu=np.zeros((N_REQ, N_PAGES), dtype=np.int32),
        padded_num_reqs_in=N_REQ,
        page_table_version=1,
        window_row_indices_cpu=np.arange(N_REQ, dtype=np.int32),
    )


def test_async_staging_buffers_only_exist_after_first_async_prep():
    preparer = _preparer()
    assert not hasattr(preparer, "_async_input_ids_cpu")

    kwargs = _kwargs()
    kwargs["token_ids_cpu"][:] = 17
    preparer.start_async_prep(**kwargs)
    with pytest.raises(RuntimeError):
        preparer.start_async_prep(**kwargs)
    kwargs["scheduled_full_cpu"][:] = 0
    kwargs["num_computed_tokens_cpu"][:] = 9
    result = preparer.get_async_prep_result()
    assert result is not None
    (metadata, input_ids, positions, scheduled, active), timings = result
    np.testing.assert_array_equal(np.asarray(input_ids)[:N_REQ], [17] * N_REQ)
    np.testing.assert_array_equal(np.asarray(positions)[:N_REQ], [3] * N_REQ)
    np.testing.assert_array_equal(np.asarray(metadata.scheduled), [1] * N_REQ)
    # The trailing vectors are cached API-compatibility placeholders; the real
    # schedule is packed into metadata, not duplicated in these return values.
    np.testing.assert_array_equal(np.asarray(scheduled), [0] * N_REQ)
    np.testing.assert_array_equal(np.asarray(active), [False] * N_REQ)
    assert metadata.input_ids_buf is input_ids
    assert timings["padded_num_reqs"] == N_REQ
    assert preparer.get_async_prep_result() is None

    next_kwargs = _kwargs()
    next_kwargs["token_ids_cpu"][:] = 23
    next_kwargs["num_computed_tokens_cpu"][:] = 4
    preparer.start_async_prep(**next_kwargs)
    next_result = preparer.get_async_prep_result()
    assert next_result is not None
    (next_metadata, next_input_ids, next_positions, next_scheduled, next_active), _ = next_result
    np.testing.assert_array_equal(np.asarray(next_input_ids)[:N_REQ], [23] * N_REQ)
    np.testing.assert_array_equal(np.asarray(next_positions)[:N_REQ], [4] * N_REQ)
    np.testing.assert_array_equal(np.asarray(next_metadata.scheduled), [1] * N_REQ)
    assert next_scheduled is scheduled
    assert next_active is active
    assert preparer.get_async_prep_result() is None


def test_slot_mapping_and_request_distribution_properties_follow_metadata_version():
    v3 = _preparer("v3")
    assert v3.uses_request_distribution and not v3.uses_slot_mapping
    v2 = _preparer("v2")
    assert v2.uses_slot_mapping and not v2.uses_request_distribution
