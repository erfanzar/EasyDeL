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

from eformer import common_types
from eformer.escale import PartitionAxis

from easydel.axis import (
    ATTN_DP,
    register_attention_data_parallel_axis,
    reset_attention_data_parallel_axis,
)


def test_attn_dp_defaults_to_partition_axis_data_parallel_axis():
    reset_attention_data_parallel_axis()
    paxis = PartitionAxis(data_parallel_axis="ep")
    resolved = paxis.resolve_axis([ATTN_DP], mode=common_types.MODE_PREFILL)
    assert resolved == ["ep"]


def test_attn_dp_can_be_overridden_globally():
    reset_attention_data_parallel_axis()
    try:
        register_attention_data_parallel_axis("ep")
        paxis = PartitionAxis(data_parallel_axis="dp")
        resolved = paxis.resolve_axis([ATTN_DP], mode=common_types.MODE_PREFILL)
        assert resolved == ["ep"]
    finally:
        reset_attention_data_parallel_axis()
