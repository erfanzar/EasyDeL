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

"""TPU Pallas interface for Ragged Gated Delta Rule.

Registers the Pallas TPU implementation under ``"ragged_gated_delta_rule"``
for ``Platform.PALLAS, Backend.TPU``. The decode path uses a Pallas kernel
with per-token parallelism. The prefill path falls back to XLA chunked.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.ragged_gated_delta_rule._xla_impl_fwd import (
    _ragged_gdr_chunked_prefill,
)
from ._pallas_impl_fwd import run_ragged_gdr_decode_pallas


def _decode_path(query, key, value, beta, decay, recurrent_state, state_indices, use_qk_l2norm):
    """Execute the Pallas decode path: L2 norm → gather → kernel → scatter."""
    if use_qk_l2norm:
        from ...._xla.gated_delta_rule._xla_impl_fwd import _l2norm

        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    gathered_state = recurrent_state[state_indices]

    output, updated_per_token_state = run_ragged_gdr_decode_pallas(
        query,
        key,
        value,
        beta,
        decay,
        gathered_state,
        use_l2norm=False,
    )

    updated_state = recurrent_state.at[state_indices].set(updated_per_token_state)
    return output, updated_state


def ragged_gated_delta_rule_decode(
    query: Float[Array, "num_tokens num_heads qk_head_dim"],
    key: Float[Array, "num_tokens num_heads qk_head_dim"],
    value: Float[Array, "num_tokens num_heads v_head_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads"],
    recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
    state_indices: Int[Array, "num_requests"],
    *,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "num_tokens num_heads v_head_dim"],
    Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
]:
    """Direct decode-only Pallas path without ``lax.cond`` dispatch.

    Executes the Pallas TPU kernel unconditionally for decode (single-token)
    inputs. Unlike :func:`ragged_gated_delta_rule`, this function does not
    perform mixed prefill/decode branching, so it is safe to call when all
    requests in the batch are decode-only (one query token per sequence).

    The function applies optional L2 normalisation to ``query`` and ``key``
    before invoking the kernel, gathers per-request recurrent states from the
    state pool, runs the in-place delta-rule update, and scatters the updated
    states back.

    Args:
        query: Packed query tokens [num_tokens, num_heads, qk_head_dim].
            One token per decode request; ``num_tokens == num_requests``.
        key: Packed key tokens [num_tokens, num_heads, qk_head_dim].
        value: Packed value tokens [num_tokens, num_heads, v_head_dim].
        beta: Per-token, per-head beta coefficients [num_tokens, num_heads].
            Controls the magnitude of the delta-rule state update.
        decay: Per-token, per-head decay factors [num_tokens, num_heads].
            Applied to the recurrent state before the update step.
        recurrent_state: Recurrent state pool [num_slots, num_heads,
            qk_head_dim, v_head_dim]. Each slot holds the running state for
            one active sequence; ``num_slots >= max(state_indices) + 1``.
        state_indices: Indices into ``recurrent_state`` for each request
            [num_requests]. ``state_indices[i]`` is the slot that stores the
            recurrent state for request ``i``.
        use_qk_l2norm: Whether to L2-normalise ``query`` and ``key`` along
            ``head_dim`` (eps=1e-6) before the kernel. Default: ``True``.

    Returns:
        A 2-tuple:

        - **output** – Attention output tokens
          [num_tokens, num_heads, v_head_dim].
        - **updated_recurrent_state** – Updated state pool with the same
          shape as ``recurrent_state`` [num_slots, num_heads,
          qk_head_dim, v_head_dim].
    """
    return _decode_path(query, key, value, beta, decay, recurrent_state, state_indices, use_qk_l2norm)


@kernel_registry.register("ragged_gated_delta_rule", Platform.PALLAS, Backend.TPU)
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
    """Ragged GDR with Pallas TPU decode kernel and XLA chunked prefill fallback.

    Dispatches to one of two implementations at runtime using ``lax.cond``:

    - **Decode path** (all sequences have exactly one query token):
      Runs :func:`ragged_gated_delta_rule_decode`, a Pallas TPU kernel with
      per-token parallelism and in-place recurrent-state updates.
    - **Prefill path** (any sequence has more than one query token):
      Falls back to ``_ragged_gdr_chunked_prefill`` (XLA chunked).

    The dispatch condition is ``all(seq_lengths <= 1)`` where
    ``seq_lengths = query_start_loc[1:] - query_start_loc[:-1]``.

    Both paths apply optional L2 normalisation to ``query`` and ``key``
    before the core computation.

    Args:
        query: Packed query tokens [num_tokens, num_heads, qk_head_dim].
            All sequences' query tokens are concatenated along dim 0.
        key: Packed key tokens [num_tokens, num_heads, qk_head_dim].
        value: Packed value tokens [num_tokens, num_heads, v_head_dim].
        beta: Per-token, per-head beta coefficients [num_tokens, num_heads].
        decay: Per-token, per-head decay factors [num_tokens, num_heads], or
            ``None`` to use all-zero decay (no state forgetting).
        recurrent_state: Recurrent state pool [num_slots, num_heads,
            qk_head_dim, v_head_dim]. Each slot stores the running state for
            one active sequence.
        query_start_loc: Cumulative query token counts [num_requests + 1].
            ``query_start_loc[i]`` is the index of the first query token for
            request ``i``; ``query_start_loc[-1] == num_tokens``.
        state_indices: Indices into ``recurrent_state`` for each request
            [num_requests]. ``state_indices[i]`` selects the slot for
            request ``i``.
        chunk_size: Chunk size used by the XLA chunked-prefill fallback path.
            Ignored when the decode path is taken. Default: ``64``.
        use_qk_l2norm: Whether to L2-normalise ``query`` and ``key`` along
            ``head_dim`` (eps=1e-6) before the kernel. Default: ``True``.

    Returns:
        A 2-tuple:

        - **output** – Attention output tokens
          [num_tokens, num_heads, v_head_dim].
        - **updated_recurrent_state** – Updated state pool with the same
          shape as ``recurrent_state`` [num_slots, num_heads,
          qk_head_dim, v_head_dim].

    Note:
        ``lax.cond`` traces both branches at compile time. To ensure shapes
        match, ``state_indices`` is padded or sliced to length ``num_tokens``
        before being passed to the decode path.
    """
    if decay is None:
        decay = jnp.zeros_like(beta)

    seq_lengths = query_start_loc[1:] - query_start_loc[:-1]
    is_all_decode = jnp.all(seq_lengths <= 1)

    num_tokens = query.shape[0]
    num_si = state_indices.shape[0]
    if num_tokens > num_si:
        decode_state_indices = jnp.pad(state_indices, (0, num_tokens - num_si))
    elif num_tokens < num_si:
        decode_state_indices = state_indices[:num_tokens]
    else:
        decode_state_indices = state_indices

    def decode_fn(_):
        return _decode_path(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            decode_state_indices,
            use_qk_l2norm,
        )

    def prefill_fn(_):
        new_state, out = _ragged_gdr_chunked_prefill(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            chunk_size,
            use_qk_l2norm,
        )
        return out, new_state

    return lax.cond(is_all_decode, decode_fn, prefill_fn, operand=None)
