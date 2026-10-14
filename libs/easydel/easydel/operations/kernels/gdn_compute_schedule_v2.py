# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Copyright 2026 Google LLC
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
"""Grid schedule construction adapter for ragged Gated Delta Rule kernels v2.

This EasyDeL module is a compatibility surface for callers that import through
``easydel.operations.kernels``. The schedule implementation and backend
dispatch live in eJKernel; this file preserves the public EasyDeL API and the
schedule-table documentation while delegating to
``ejkernel.modules.operations.compute_schedule_table_v2``.

This module is not an EasyDeL ``OperationImpl`` adapter. Schedule construction
is a helper used by the ragged GDN backend, while the runtime kernel operations
live in the dedicated EasyDeL operation adapters.

The schedule table is consumed by the ragged GDN Pallas prefill/decode kernels
under continuous batching. Each grid iteration is classified as either a decode
batch, moving backwards from the prefill/decode boundary in ``BT``-token
strides, or a prefill chunk, moving forwards in ``chunk_size`` strides.
Transition rows handle sequences whose start or end is not sublane-aligned.
"""

from __future__ import annotations

import jax
from ejkernel.modules.operations import GDNComputeScheduleV2, GDNComputeScheduleV2Config
from ejkernel.modules.operations import compute_schedule_table_v2 as _ejkernel_compute_schedule_table_v2


def compute_schedule_table_v2(
    query_start_loc: jax.Array,
    decode_tokens: int | jax.Array,
    num_valid_seqs: int | jax.Array,
    max_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
    *,
    cfg: GDNComputeScheduleV2Config | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Build the per-iteration work schedule table for ragged GDN kernels.

    Each grid iteration represents either a *decode* batch or a *prefill*
    chunk. When a sequence boundary lands inside a sublane, the schedule
    inserts an explicit transition block at each end so the prefill kernel can
    fall back to token-by-token arithmetic for those rows while regular
    prefill work remains chunked on sublane-aligned offsets.

    Args:
        query_start_loc: CSR-style cumulative token offsets per request,
            shape ``[num_requests + 1]``.
        decode_tokens: Number of decode tokens at the start of the flat token
            stream, before any prefill work. Scalar ``int`` or scalar JAX
            array.
        num_valid_seqs: Number of valid non-padding sequences in the batch.
        max_tokens: Static upper bound on total tokens used to size the table.
        chunk_size: Number of tokens processed per prefill block.
        BT: Number of tokens processed per decode block. Defaults to
            ``chunk_size`` in eJKernel when omitted.
        alignment: Sublane alignment that block offsets must respect.
        cfg: Optional eJKernel operation config.

    Returns:
        ``(final_table, total_blocks)`` where ``final_table`` is an ``int32``
        array with shape ``[safe_max_blocks, 11 + 3 * alignment]`` and
        ``total_blocks`` is the count of valid grid iterations.

    Note:
        Column layout of ``final_table``:

        * 0: ``prefill_valid_ints``: 1 when this row carries prefill work.
        * 1: ``block_offset``: token offset of the prefill block start.
        * 2: ``r_for_block``: request index of the prefill block.
        * 3: ``block_count``: valid tokens in the prefill block.
        * 4: ``decode_valid``: 1 when this row carries decode work.
        * 5: ``decode_offsets``: start token of the decode batch.
        * 6: ``decode_req_ids``: starting request id in the decode batch.
        * 7: ``decode_counts``: number of decode requests in the batch.
        * 8: ``block_is_last``: 1 when this prefill block ends the request.
        * 9: ``block_is_first``: 1 when this prefill block starts it.
        * 10: ``is_trans_block``: 1 when the row is a sublane transition.
        * ``11 .. 10 + alignment``: per-sublane-token request ids.
        * next ``alignment``: per-sublane ``is_first_tok`` flags.
        * next ``alignment``: per-sublane ``is_last_tok`` flags.
    """
    return _ejkernel_compute_schedule_table_v2(
        query_start_loc,
        decode_tokens,
        num_valid_seqs,
        max_tokens,
        chunk_size,
        BT,
        alignment,
        cfg=cfg,
    )


__all__ = ("GDNComputeScheduleV2", "GDNComputeScheduleV2Config", "compute_schedule_table_v2")
