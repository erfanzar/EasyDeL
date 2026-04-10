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

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from easydel.caching._metadatabuilder import AttentionMetadataBuilder
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer
from easydel.operations.kernels.multi_latent_ragged_page_attention import (
    _request_distribution_bounds as mla_request_distribution_bounds,
)
from easydel.operations.kernels.ragged_page_attention import (
    _request_distribution_bounds as rpa_request_distribution_bounds,
)


@pytest.mark.parametrize(
    ("scheduled", "context_lens", "expected"),
    (
        ([4], [4], [0, 1, 1]),
        ([4, 1, 0], [4, 9, 0], [1, 2, 2]),
    ),
)
def test_request_distribution_helpers_match_kernel_contract(scheduled, context_lens, expected):
    scheduled_arr = jnp.asarray(scheduled, dtype=jnp.int32)
    context_arr = jnp.asarray(context_lens, dtype=jnp.int32)
    expected_arr = np.asarray(expected, dtype=np.int32)

    np.testing.assert_array_equal(np.asarray(rpa_request_distribution_bounds(scheduled_arr, context_arr)), expected_arr)
    np.testing.assert_array_equal(np.asarray(mla_request_distribution_bounds(scheduled_arr, context_arr)), expected_arr)


def test_attention_metadata_builder_ragged_fields_use_prefill_end():
    fields = AttentionMetadataBuilder.compute_ragged_batch_fields_cpu(
        scheduled_full=np.asarray([4, 1, 0, 0], dtype=np.int32),
        active_mask_full=np.asarray([True, True, False, False]),
        num_computed_tokens=np.asarray([0, 8, 0, 0], dtype=np.int32),
        page_table=np.zeros((4, 4), dtype=np.int32),
        version="v3",
        max_num_reqs=4,
        max_num_tokens=8,
    )

    np.testing.assert_array_equal(fields["request_distribution"], np.asarray([1, 2, 2], dtype=np.int32))


def test_attention_metadata_builder_paged_fields_use_prefill_end():
    fields = AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu(
        num_tokens_static=5,
        scheduled_full=np.asarray([4, 1, 0, 0], dtype=np.int32),
        active_mask_full=np.asarray([True, True, False, False]),
        token_ids=np.arange(64, dtype=np.int32).reshape(4, 16),
        num_computed_tokens=np.asarray([0, 8, 0, 0], dtype=np.int32),
        page_table=np.zeros((4, 4), dtype=np.int32),
        padded_num_reqs_in=2,
        min_input_pad=1,
        version="v3",
        max_num_reqs=4,
        max_num_tokens=8,
        max_pages_per_req=4,
    )

    np.testing.assert_array_equal(fields["request_distribution"], np.asarray([1, 2, 2], dtype=np.int32))


def test_batch_preparer_packs_prefill_end_into_misc_buffer():
    metadata = SimpleNamespace(
        version="v3",
        get_max_num_seqs=lambda: 4,
        max_num_pages_per_req=4,
        data_parallel_size=1,
        page_size=128,
    )
    preparer = BatchMetadataPreparer(
        metadata=metadata,
        empty_sharding=None,
        max_num_tokens=8,
        max_num_reqs=4,
        max_model_len=16,
        min_input_pad=1,
    )

    host_payload, _, _, _ = preparer._build_host_payload(
        num_tokens_static=5,
        scheduled_full_cpu=np.asarray([4, 1, 0, 0], dtype=np.int32),
        active_mask_full_cpu=np.asarray([True, True, False, False]),
        token_ids_cpu=np.arange(64, dtype=np.int32).reshape(4, 16),
        num_computed_tokens_cpu=np.asarray([0, 8, 0, 0], dtype=np.int32),
        temperature_cpu=np.ones((4,), dtype=np.float32),
        top_p_cpu=np.ones((4,), dtype=np.float32),
        top_k_cpu=np.zeros((4,), dtype=np.int32),
        min_p_cpu=np.zeros((4,), dtype=np.float32),
        frequency_penalties_cpu=np.zeros((4,), dtype=np.float32),
        presence_penalties_cpu=np.zeros((4,), dtype=np.float32),
        repetition_penalties_cpu=np.ones((4,), dtype=np.float32),
        page_table_cpu=np.zeros((4, 4), dtype=np.int32),
        page_table_version=None,
        padded_num_reqs_in=2,
        copy_slot_mapping=True,
    )

    packed_misc_i32 = host_payload[6]
    np.testing.assert_array_equal(packed_misc_i32[2:5], np.asarray([1, 2, 2], dtype=np.int32))
