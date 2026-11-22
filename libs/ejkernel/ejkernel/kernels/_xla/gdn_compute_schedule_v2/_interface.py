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

"""GDN v2 schedule-table XLA kernel interface and registry entry.

This module exposes :func:`compute_schedule_table_v2` as the public, registry-
registered XLA entry point for building the ragged Gated Delta Rule v2 work
schedule. It wraps the raw XLA implementation in
:mod:`._xla_impl_fwd` with jaxtyping/beartype shape-and-dtype checking and
registers it under the ``gdn_compute_schedule_v2`` algorithm name for the
``Platform.XLA`` / ``Backend.ANY`` slot of the kernel registry, so the
operation dispatcher can resolve it generically across hardware backends.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import compute_schedule_table_v2 as _compute_schedule_table_v2_xla


@kernel_registry.register("gdn_compute_schedule_v2", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def compute_schedule_table_v2(
    query_start_loc: Int[Array, "num_requests_plus_one"],
    decode_tokens: int | Int[Array, ""],
    num_valid_seqs: int | Int[Array, ""],
    max_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
) -> tuple[Int[Array, "max_schedule_blocks table_cols"], Int[Array, ""]]:
    """Build the GDN v2 schedule table on the XLA backend.

    Registry-facing wrapper around the raw XLA implementation. It forwards every
    argument unchanged to :func:`._xla_impl_fwd.compute_schedule_table_v2` after
    jaxtyping/beartype validates the input shapes and dtypes, and returns the
    flattened per-iteration work schedule that the ragged GDN v2 Pallas kernels
    consume from SMEM.

    Args:
        query_start_loc: CSR-style request offsets into the packed token stream,
            integer array of shape ``[num_requests + 1]``. Entry ``i`` is the
            first token index of request ``i`` and entry ``i + 1`` its
            exclusive end.
        decode_tokens: Number of leading decode tokens in the packed stream,
            either a Python ``int`` or a scalar integer array.
        num_valid_seqs: Number of valid (non-padding) request rows, either a
            Python ``int`` or a scalar integer array. Used to mask padded tail
            entries of ``query_start_loc``.
        max_tokens: Static upper bound on the total packed token count; sizes
            the maximum number of prefill blocks.
        chunk_size: Prefill chunk size in tokens; the recurrent scan processes
            prefill in chunks of this length.
        BT: Decode token block size. When ``None`` it defaults to
            ``chunk_size``.
        alignment: Sublane/token-row alignment used when computing transition
            blocks and per-sublane metadata. Defaults to ``8``.

    Returns:
        A tuple ``(schedule_table, total_blocks)``:

        * ``schedule_table``: integer array of shape
          ``[safe_max_blocks, 11 + 3 * alignment]`` where each row is one grid
          step's work descriptor (see :func:`._xla_impl_fwd.compute_schedule_table_v2`
          for the full column layout).
        * ``total_blocks``: scalar integer array giving the number of active
          grid steps, ``max(total_prefill_blocks, num_decode_batches)``.
    """
    return _compute_schedule_table_v2_xla(
        query_start_loc=query_start_loc,
        decode_tokens=decode_tokens,
        num_valid_seqs=num_valid_seqs,
        max_tokens=max_tokens,
        chunk_size=chunk_size,
        BT=BT,
        alignment=alignment,
    )
