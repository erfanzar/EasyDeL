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

"""TileLang Ragged Gated Delta Rule — public interface.

Registers :func:`ragged_gated_delta_rule` with the ejkernel kernel registry
under ``("ragged_gated_delta_rule", Platform.TILELANG, Backend.GPU)`` and
delegates to :func:`~._impl.ragged_gdr_tilelang` which owns the JAX VJP logic.

Note: the ``chunk_size`` keyword argument is accepted for API compatibility
with other backends but is **silently ignored** by this TileLang path.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._impl import ragged_gdr_tilelang


@kernel_registry.register("ragged_gated_delta_rule", Platform.TILELANG, Backend.GPU)
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
    """Ragged Gated Delta Rule recurrence with native decode and prefill paths.

    Applies the GDR step to a batch of variable-length token sequences.  The
    recurrent state pool ``recurrent_state`` is updated in-place (via
    ``input_output_aliases``) and the updated pool is returned alongside the
    output activations.

    Routing: if ``num_tokens == num_requests`` (i.e. every request contributes
    exactly one token), the decode path is taken.  Otherwise the general ragged
    prefill path is used.

    Note: ``chunk_size`` is accepted for API compatibility with other backends
    but has **no effect** in this TileLang implementation.

    Args:
        query: ``[num_tokens, num_heads, qk_head_dim]``.
        key: ``[num_tokens, num_heads, qk_head_dim]``.
        value: ``[num_tokens, num_heads, v_head_dim]``.
        beta: Per-token, per-head scalar gate ``β ∈ (0, 1]``,
            shape ``[num_tokens, num_heads]``.
        decay: Optional per-token, per-head log-space decay ``g``,
            shape ``[num_tokens, num_heads]``.  ``None`` means ``exp(g) = 1``
            (no decay); a dummy buffer (equal to ``beta``) is passed to the kernel
            so the shapes remain fixed.
        recurrent_state: Float32 state pool of shape
            ``[num_slots, num_heads, qk_head_dim, v_head_dim]``.  Updated in-place.
        query_start_loc: CSR-style row-pointer array of length
            ``num_requests + 1``; ``query_start_loc[r]`` is the first token index
            for request ``r``.
        state_indices: Mapping from request index to slot index in
            ``recurrent_state``, shape ``[num_requests]``.
        chunk_size: Ignored.  Present for cross-backend API compatibility.
        use_qk_l2norm: If ``True`` (default), L2-normalise ``query`` and ``key``
            before the inner product.

    Returns:
        A tuple ``(output, updated_state)`` where:

        * ``output``: ``[num_tokens, num_heads, v_head_dim]`` in the input dtype.
        * ``updated_state``: ``[num_slots, num_heads, qk_head_dim, v_head_dim]``
          float32 — the state pool after applying all token updates.
    """
    _ = chunk_size
    return ragged_gdr_tilelang(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        query_start_loc,
        state_indices,
        use_qk_l2norm=use_qk_l2norm,
    )


__all__ = ["ragged_gated_delta_rule"]
