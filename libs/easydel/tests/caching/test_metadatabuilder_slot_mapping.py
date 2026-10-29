# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Regression tests: v2 slot-mapping layouts and v3 request distribution from the CPU metadata builder."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.caching._metadatabuilder import AttentionMetadataBuilder

pytestmark = pytest.mark.cpu_ok

_PAGE_TABLE = np.array([[7, 8], [0, 0]], dtype=np.int32)


def test_ragged_v2_slot_mapping_compact_layout():
    fields = AttentionMetadataBuilder.compute_ragged_batch_fields_cpu(
        scheduled_full=np.array([5, 0]),
        active_mask_full=np.array([True, False]),
        num_computed_tokens=np.array([0, 0]),
        page_table=_PAGE_TABLE,
        version="v2",
        page_size=4,
        num_slices_per_kv_cache_update_page=4,
    )
    slot_mapping = fields["slot_mapping"]
    assert slot_mapping.shape == (3, 4)
    # (cache_start, new_kv_start, length) for the two touched pages; remaining columns are zero padding.
    np.testing.assert_array_equal(slot_mapping[:, :2], [[28, 32], [0, 4], [4, 1]])
    assert not slot_mapping[:, 2:].any()
    np.testing.assert_array_equal(fields["num_kv_update_slices"], [2])


def _paged(*, scheduled, active, pad_value=99, max_padded=8):
    ragged_config = SimpleNamespace(
        max_num_pages_per_req=2,
        page_size=4,
        num_slices_per_kv_cache_update_page=4,
        get_max_num_seqs=lambda: 2,
        get_padded_num_slices=lambda *_a: max_padded,
    )
    return AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu(
        num_tokens_static=8,
        scheduled_full=np.array(scheduled),
        active_mask_full=np.array(active),
        token_ids=np.arange(32, dtype=np.int32).reshape(2, 16),
        num_computed_tokens=np.array([0, 0]),
        page_table=_PAGE_TABLE,
        padded_num_reqs_in=None,
        min_input_pad=2,
        version="v2",
        ragged_config=ragged_config,
        slot_mapping_padding_val=pad_value,
    )


def test_paged_v2_slot_mapping_fixed_shape_uses_pad_value():
    out = _paged(scheduled=[5, 0], active=[True, False])
    slot_mapping = out["slot_mapping"]
    assert slot_mapping.shape == (3, 8)
    np.testing.assert_array_equal(slot_mapping[:, :2], [[28, 32], [0, 4], [4, 1]])
    assert (slot_mapping[:, 2:] == 99).all()
    np.testing.assert_array_equal(out["num_kv_update_slices"], [2])


def test_paged_v2_empty_batch_keeps_fixed_shape():
    out = _paged(scheduled=[0, 0], active=[False, False])
    assert out["slot_mapping"].shape == (3, 8)
    assert (out["slot_mapping"] == 99).all()
    np.testing.assert_array_equal(out["num_kv_update_slices"], [0])


def test_v3_request_distribution_is_identical_for_both_entry_points():
    scheduled = np.array([1, 3, 1])
    computed = np.array([4, 0, 0])  # request 0 decodes; request 2 is a one-token prefill
    expected = [1, 3, 3]
    ragged = AttentionMetadataBuilder.compute_ragged_batch_fields_cpu(
        scheduled_full=scheduled,
        active_mask_full=np.array([True, True, True]),
        num_computed_tokens=computed,
        page_table=np.zeros((3, 2), dtype=np.int32),
        version="v3",
    )
    paged = AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu(
        num_tokens_static=8,
        scheduled_full=scheduled,
        active_mask_full=np.array([True, True, True]),
        token_ids=np.zeros((3, 16), dtype=np.int32),
        num_computed_tokens=computed,
        page_table=np.zeros((3, 2), dtype=np.int32),
        padded_num_reqs_in=None,
        min_input_pad=4,
        version="v3",
    )
    np.testing.assert_array_equal(ragged["request_distribution"], expected)
    np.testing.assert_array_equal(paged["request_distribution"], expected)
