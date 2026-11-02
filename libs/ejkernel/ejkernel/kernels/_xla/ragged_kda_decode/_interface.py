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

"""XLA registry entry (priority 0 fallback and reference) for the per-row KDA decode step."""

from __future__ import annotations

from jaxtyping import Array, Bool, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import ragged_kda_decode_xla, validate_shapes


@kernel_registry.register("ragged_kda_decode", Platform.XLA, Backend.ANY, priority=0)
def ragged_kda_decode(
    query: Float[Array, "num_rows num_heads head_dim"],
    key: Float[Array, "num_rows num_heads head_dim"],
    value: Float[Array, "num_rows num_heads v_dim"],
    beta: Float[Array, "num_rows num_heads"],
    decay: Float[Array, "num_rows num_heads head_dim"],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    state_indices: Int[Array, "num_rows"] | None,
    valid: Bool[Array, "num_rows"],
    has_initial_state: Bool[Array, "num_rows"] | None = None,
    *,
    use_qk_l2norm: bool = True,
    block_rows: int = 0,
    block_h: int = 0,
) -> tuple[Float[Array, "num_rows num_heads v_dim"], Float[Array, "num_slots num_heads head_dim v_dim"]]:
    """One per-channel KDA decode token per row over a recurrent-state pool.

    Row ``r`` owns pool slot ``state_indices[r]`` (slot ``r`` without ``state_indices``). Rows with ``valid[r]`` and
    an in-range slot advance their state by one token and output it; every other row outputs zeros and every other
    slot keeps its state bit for bit. See :mod:`._xla_impl` for the arithmetic.

    Args:
        query: Post-conv queries ``[R, H, K]``.
        key: Post-conv keys ``[R, H, K]``.
        value: Post-conv values ``[R, H, V]``.
        beta: Write strengths ``[R, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[R, H, K]`` (``<= 0``).
        recurrent_state: State pool ``[S, H, K, V]`` (the step always runs in fp32).
        state_indices: Pool slot of every row ``[R]``, or ``None`` (row ``r`` owns slot ``r``, ``S >= R``).
        valid: ``[R]`` rows that advance.
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state.
        use_qk_l2norm: L2-normalise queries and keys first.
        block_rows: Rows per grid step of tiled backends; a tiling hint the XLA reference does not need.
        block_h: Heads per inner step of tiled backends; a tiling hint the XLA reference does not need.

    Returns:
        ``(out [R, H, V] fp32, new_recurrent_state [S, H, K, V])``; the pool keeps its dtype.
    """
    # block_rows / block_h are tiling hints of the accelerator backends; the reference ignores them.
    validate_shapes(query, key, value, beta, decay, recurrent_state, state_indices, valid, has_initial_state)
    return ragged_kda_decode_xla(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        state_indices,
        valid,
        has_initial_state,
        use_qk_l2norm=use_qk_l2norm,
    )


__all__ = ("ragged_kda_decode",)
