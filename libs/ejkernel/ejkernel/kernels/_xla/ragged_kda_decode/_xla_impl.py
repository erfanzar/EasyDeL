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

"""XLA reference for one per-channel KDA decode token per row over a recurrent-state pool.

Row ``r`` carries one token (``query[r]`` ... ``decay[r]``) and owns pool slot ``state_indices[r]`` (slot ``r`` when
``state_indices`` is ``None``). Rows with ``valid[r]`` set (and an in-range slot) take one step of::

    S = diag(exp(g)) S;   S += k (beta (v - k^T S))^T;   o = S^T q

with ``q`` scaled by ``K**-0.5`` and ``q`` / ``k`` optionally L2-normalised, with the arithmetic of the
``ragged_kda`` decode phase (Q/K normalised in their input dtype, then fp32 elementwise products and reductions).
Other rows output zeros and leave every slot bit for bit; rows whose ``has_initial_state`` flag is ``False`` start
from a zero state.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from ..ragged_kda._xla_impl import decode_step


def validate_shapes(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    recurrent_state: Array,
    state_indices: Array | None,
    valid: Array,
    has_initial_state: Array | None,
) -> None:
    """Check the ``ragged_kda_decode`` array contract.

    Args:
        query: ``[R, H, K]``.
        key: ``[R, H, K]``.
        value: ``[R, H, V]``.
        beta: ``[R, H]``.
        decay: ``[R, H, K]``.
        recurrent_state: ``[S, H, K, V]``.
        state_indices: ``[R]``, or ``None`` (row ``r`` owns slot ``r``; needs ``S >= R``).
        valid: ``[R]``.
        has_initial_state: ``[R]`` or ``None``.

    Raises:
        ValueError: If any shape disagrees with the contract above.
    """
    if query.ndim != 3:
        raise ValueError(f"query must be [num_rows, num_heads, head_dim], got shape {query.shape}")
    num_rows, num_heads, head_dim = query.shape
    if key.shape != query.shape:
        raise ValueError(f"key shape {key.shape} must equal query shape {query.shape}")
    if value.ndim != 3 or value.shape[:2] != (num_rows, num_heads):
        raise ValueError(f"value must be [{num_rows}, {num_heads}, v_dim], got shape {value.shape}")
    if beta.shape != (num_rows, num_heads):
        raise ValueError(f"beta must be [{num_rows}, {num_heads}], got shape {beta.shape}")
    if decay.shape != query.shape:
        raise ValueError(f"decay must be per-channel {query.shape}, got shape {decay.shape}")
    v_dim = value.shape[-1]
    if recurrent_state.ndim != 4 or recurrent_state.shape[1:] != (num_heads, head_dim, v_dim):
        raise ValueError(
            f"recurrent_state must be [num_slots, {num_heads}, {head_dim}, {v_dim}], got shape {recurrent_state.shape}"
        )
    if state_indices is None:
        if recurrent_state.shape[0] < num_rows:
            raise ValueError(
                f"state_indices=None maps row r to pool slot r, so the pool needs at least {num_rows} slots, "
                f"got {recurrent_state.shape[0]}"
            )
    elif state_indices.shape != (num_rows,):
        raise ValueError(f"state_indices must be [{num_rows}], got shape {state_indices.shape}")
    if valid.shape != (num_rows,):
        raise ValueError(f"valid must be [{num_rows}], got shape {valid.shape}")
    if has_initial_state is not None and has_initial_state.shape != (num_rows,):
        raise ValueError(f"has_initial_state must be [{num_rows}], got shape {has_initial_state.shape}")


def ragged_kda_decode_xla(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    recurrent_state: Array,
    state_indices: Array | None,
    valid: Array,
    has_initial_state: Array | None = None,
    *,
    use_qk_l2norm: bool = True,
) -> tuple[Array, Array]:
    """One KDA decode token per row over a recurrent-state pool (XLA).

    With ``state_indices=None`` the rows' states are the leading ``R`` slots, updated by one masked read-modify-write
    (no gather or scatter of pool-sized buffers); otherwise the states are gathered by slot and the advanced ones
    scattered back (rows that do not advance drop their write).

    Args:
        query: Post-conv queries ``[R, H, K]``.
        key: Post-conv keys ``[R, H, K]``.
        value: Post-conv values ``[R, H, V]``.
        beta: Write strengths ``[R, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[R, H, K]``.
        recurrent_state: State pool ``[S, H, K, V]``; computed in fp32, returned in its own dtype.
        state_indices: Pool slot of every row ``[R]`` (out-of-range: the row does not advance), or ``None``.
        valid: ``[R]`` rows that advance (rows with one token this step).
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state.
        use_qk_l2norm: L2-normalise queries and keys.

    Returns:
        ``(out [R, H, V] fp32, new_recurrent_state [S, H, K, V])``; rows that do not advance output zeros.
    """
    num_rows = query.shape[0]
    num_slots = recurrent_state.shape[0]
    live = jnp.asarray(valid).astype(jnp.bool_)
    if state_indices is None:
        slots = None
        held = recurrent_state if num_slots == num_rows else lax.slice_in_dim(recurrent_state, 0, num_rows, axis=0)
        state = held
    else:
        slots = state_indices.astype(jnp.int32)
        live = live & (slots >= 0) & (slots < num_slots)
        held = None
        state = recurrent_state[jnp.clip(slots, 0, num_slots - 1)]
    if has_initial_state is not None:
        keep = jnp.asarray(has_initial_state).astype(jnp.bool_)
        state = jnp.where(keep[:, None, None, None], state, jnp.zeros_like(state))
    out, new_state = decode_step(query, key, value, beta, decay, state, use_qk_l2norm)
    new_state = new_state.astype(recurrent_state.dtype)
    if held is not None:
        rows = jnp.where(live[:, None, None, None], new_state, held)
        pool = rows if num_slots == num_rows else lax.dynamic_update_slice_in_dim(recurrent_state, rows, 0, axis=0)
    else:
        pool = recurrent_state.at[jnp.where(live, slots, num_slots)].set(new_state, mode="drop")
    return jnp.where(live[:, None, None], out, 0.0), pool


__all__ = ("ragged_kda_decode_xla", "validate_shapes")
