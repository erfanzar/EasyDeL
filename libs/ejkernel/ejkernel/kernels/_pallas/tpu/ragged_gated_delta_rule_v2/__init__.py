# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""TPU Pallas ragged GDN v2 kernel."""

from ._gdn_policy import KERNEL_TILE_POLICIES, KernelTilePolicy, normalize_kernel_tile_policy
from ._interface import ragged_gated_delta_rule_v2, ragged_gdn_conv_decode
from ._pallas_gdn_chunk import (
    pallas_gdn_chunk_prefill,
    pallas_gdn_chunk_prefill_supported,
    pallas_gdn_conv_decode,
    pallas_gdn_conv_decode_supported,
    pallas_gdn_decode_inplace,
)
from ._pallas_impl_fwd import (
    ragged_gated_delta_rule_decode_only,
    ragged_gated_delta_rule_mixed_prefill,
    set_gdn_kernel_tile_policy,
)

__all__ = (
    "KERNEL_TILE_POLICIES",
    "KernelTilePolicy",
    "normalize_kernel_tile_policy",
    "pallas_gdn_chunk_prefill",
    "pallas_gdn_chunk_prefill_supported",
    "pallas_gdn_conv_decode",
    "pallas_gdn_conv_decode_supported",
    "pallas_gdn_decode_inplace",
    "ragged_gated_delta_rule_decode_only",
    "ragged_gated_delta_rule_mixed_prefill",
    "ragged_gated_delta_rule_v2",
    "ragged_gdn_conv_decode",
    "set_gdn_kernel_tile_policy",
)
