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

"""XLA registry entry (priority 0 fallback and reference) for ragged per-channel-decay KDA."""

from __future__ import annotations

from jaxtyping import Array, Bool, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import ragged_kda_xla, validate_options, validate_shapes


@kernel_registry.register("ragged_kda", Platform.XLA, Backend.ANY, priority=0)
def ragged_kda(
    query: Float[Array, "num_tokens num_heads head_dim"],
    key: Float[Array, "num_tokens num_heads head_dim"],
    value: Float[Array, "num_tokens num_heads v_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads head_dim"],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    query_start_loc: Int[Array, "num_rows_plus_1"],
    state_indices: Int[Array, "num_rows"] | None,
    has_initial_state: Bool[Array, "num_rows"] | None = None,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    inverse: str = "auto",
    pairwise: str = "auto",
) -> tuple[Float[Array, "num_tokens num_heads v_dim"], Float[Array, "num_slots num_heads head_dim v_dim"]]:
    """Ragged (flat-token) KDA with per-channel decay over a recurrent-state pool.

    Request ``r`` owns tokens ``[query_start_loc[r], query_start_loc[r + 1])`` and pool slot ``state_indices[r]``.
    Single-token rows take one vectorised decode step; rows with two or more tokens run the chunked math on their
    own live chunks only. See :mod:`._xla_impl` for the algorithm.

    Args:
        query: Post-conv queries ``[T, H, K]``.
        key: Post-conv keys ``[T, H, K]``.
        value: Post-conv values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid already applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``), ideally fp32.
        recurrent_state: State pool ``[S, H, K, V]`` (fp32 recommended; the recurrence always runs in fp32).
        query_start_loc: Cumulative row offsets ``[R + 1]``; tokens at or past ``query_start_loc[R]`` are padding.
        state_indices: Pool slot of every row ``[R]``; out-of-range slots mark padding rows. ``None``: row ``r``
            owns slot ``r`` (identity layout, ``S >= R``; the decode phase updates those slots in place).
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state instead of their slot.
        chunk_size: Chunk length of the multi-token path.
        use_qk_l2norm: L2-normalise queries and keys first.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound`` (e.g. a safe-gate bound); with
            ``pairwise="auto"`` a tight enough bound selects the factored MXU pairwise decay. ``None`` keeps the
            dense ``[cs, cs, K]`` form.
        precision: ``"highest"`` or ``"default"`` for the non-inverse chunk dots.
        inverse: Chunk inverse: ``"auto"``, ``"masked"`` (block-recursive, matmul only) or ``"solve"`` (exact
            triangular solve).
        pairwise: ``"auto"`` or ``"dense"``.

    Returns:
        ``(out [T, H, V] fp32, new_recurrent_state [S, H, K, V])``: zero output at padding tokens; the pool keeps
        its dtype, and slots of rows without tokens are unchanged.
    """
    validate_options(chunk_size, precision, inverse, pairwise, decay_lower_bound)
    validate_shapes(query, key, value, beta, decay, recurrent_state, query_start_loc, state_indices, has_initial_state)
    return ragged_kda_xla(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        query_start_loc,
        state_indices,
        has_initial_state,
        chunk_size=int(chunk_size),
        use_qk_l2norm=use_qk_l2norm,
        decay_lower_bound=decay_lower_bound,
        precision=precision,
        inverse=inverse,
        pairwise=pairwise,
    )


__all__ = ("ragged_kda",)
