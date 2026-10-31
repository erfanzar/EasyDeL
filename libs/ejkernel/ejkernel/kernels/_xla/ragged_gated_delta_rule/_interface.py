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

"""Ragged Gated Delta Rule XLA kernel interface and registry entry.

Registers the ragged GDR implementation under ``"ragged_gated_delta_rule"``
for ``Platform.XLA, Backend.ANY``. The registered function accepts flat
token-stream tensors with CSR-style ``query_start_loc`` offsets and a
``state_indices`` mapping from requests to slots in a global state pool.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import ragged_gated_delta_rule_dispatch


@kernel_registry.register("ragged_gated_delta_rule", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_gated_delta_rule(
    query: Float[Array, "num_tokens num_heads qk_head_dim"],
    key: Float[Array, "num_tokens num_heads qk_head_dim"],
    value: Float[Array, "num_tokens num_heads v_head_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads"] | None,
    recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
    query_start_loc: Int[Array, "num_requests_plus_1"],
    state_indices: Int[Array, "num_requests"],
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "num_tokens num_heads v_head_dim"],
    Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
]:
    """Ragged Gated Delta Rule for packed continuous-batching inference.

    Processes variable-length sequences packed into a flat token stream.
    Automatically routes to a decode-only path (sequential scan) when all
    sequences have length 1, or a chunked prefill path (parallel intra-chunk
    via triangular solve + sequential inter-chunk scan) otherwise.

    The decode-only path uses ``lax.scan`` over tokens, gathering each
    request's state from the global pool and scattering updates back.

    The chunked prefill path pads each sequence to a multiple of
    ``chunk_size``, computes intra-chunk attention via
    ``solve_triangular((I + S), I)`` for the exact delta-rule inverse,
    then propagates state across chunks with a lightweight scan body
    (4 matmuls per chunk). A ``reset_mask`` resets state at sequence
    boundaries so multiple requests share one packed stream.

    Args:
        query: Flat token-stream queries of shape
            ``(num_tokens, num_heads, qk_head_dim)``.
        key: Flat token-stream keys of shape
            ``(num_tokens, num_heads, qk_head_dim)``.
        value: Flat token-stream values of shape
            ``(num_tokens, num_heads, v_head_dim)``.
        beta: Per-token gating coefficients of shape
            ``(num_tokens, num_heads)``. Controls the strength of the
            delta update at each position.
        decay: Per-token log-space decay of shape
            ``(num_tokens, num_heads)``, or ``None`` for no decay.
            The recurrent state is multiplied by ``exp(decay)`` before
            the delta update.
        recurrent_state: Global state pool of shape
            ``(num_slots, num_heads, qk_head_dim, v_head_dim)``.
            Each active request owns one slot in this pool.
        query_start_loc: Cumulative token offsets per request of shape
            ``(num_requests + 1,)``. ``query_start_loc[i]`` is the index
            of the first token belonging to request ``i``, and
            ``query_start_loc[-1]`` is the total number of valid tokens.
        state_indices: Maps each request to its slot in the state pool,
            shape ``(num_requests,)``. ``state_indices[i]`` is the row
            index into ``recurrent_state`` for request ``i``.
        chunk_size: Number of tokens per chunk for the chunked prefill
            path. Ignored for decode-only. Default 64.
        use_qk_l2norm: If ``True``, L2-normalize queries and keys before
            computing attention. Improves numerical stability.

    Returns:
        A tuple ``(output, updated_recurrent_state)`` where:

        - ``output`` has shape ``(num_tokens, num_heads, v_head_dim)`` —
          the attention output for every token in the packed stream.
        - ``updated_recurrent_state`` has shape
          ``(num_slots, num_heads, qk_head_dim, v_head_dim)`` — the
          state pool with updated entries for all active requests.
    """
    return ragged_gated_delta_rule_dispatch(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        query_start_loc,
        state_indices,
        chunk_size=chunk_size,
        use_qk_l2norm=use_qk_l2norm,
    )
