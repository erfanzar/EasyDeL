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

"""MoE dispatch/combine row gathers as tile-aligned per-row DMA (Pallas TPU)."""

from ._interface import (
    moe_row_gather,
    moe_row_gather_sum,
    row_gather_sum_unsupported_reason,
    row_gather_unsupported_reason,
)
from ._pallas_impl import gather_rows, gather_sum_rows, pack_rows

__all__ = (
    "gather_rows",
    "gather_sum_rows",
    "moe_row_gather",
    "moe_row_gather_sum",
    "pack_rows",
    "row_gather_sum_unsupported_reason",
    "row_gather_unsupported_reason",
)
