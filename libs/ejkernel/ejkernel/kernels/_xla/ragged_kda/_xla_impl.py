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

"""XLA reference for ragged (flat-token) KDA with per-channel decay over a recurrent-state pool.

The packed stream ``[T, H, D]`` holds the tokens of ``R`` requests back to back: request ``r`` owns tokens
``[query_start_loc[r], query_start_loc[r + 1])`` and its recurrent state lives in pool slot ``state_indices[r]``.
Tokens at or past ``query_start_loc[R]`` are padding. Per request and head, per token ``t``::

    S_t = diag(exp(g_t)) S_{t-1};   S_t += k_t (beta_t (v_t - k_t^T S_t))^T;   o_t = S_t^T q_t

with ``q`` scaled by ``K**-0.5`` and ``q`` / ``k`` optionally L2-normalised (the eSurge KDA serving recurrence).

The work is split into two phases:

* **Decode** (rows with exactly one token): one vectorised recurrence step over the ``R`` states gathered by
  ``state_indices``, with the arithmetic of the per-token decode step (Q/K normalised in their input dtype, then
  fp32 elementwise products and reductions). The new states are scattered back for those rows only. With
  ``state_indices=None`` (row ``r`` owns slot ``r``) the step is a masked in-place read-modify-write of the leading
  ``R`` slots instead (no gather, no scatter). The phase always runs (its cost is fixed by ``R``): gating it with
  ``lax.cond`` would route the whole state pool through a conditional, which XLA may copy.
* **Chunked** (rows with two or more tokens): every such row is cut into ``chunk_size`` chunks aligned to its first
  token. A ``fori_loop`` whose trip count is the number of live chunks, ``sum_r ceil(len_r / chunk_size)`` over the
  multi-token rows, runs :func:`~ejkernel.kernels._xla.kda_chunk._xla_impl.chunk_core` (the shared ``kda_chunk``
  per-chunk math) on one chunk per step: it loads the row's state at the row's first chunk and stores it after the
  row's last one. Each chunk reads its own ``chunk_size``-token window of the flat stream, so no per-row padded
  copy of the inputs exists; the loop is empty on pure decode steps.

Work and temporary memory follow the real token count (plus the ``R`` gathered states of the decode phase), not
``R * T``. Rows with no tokens (and rows whose ``state_indices`` entry is out of range) keep their state
bit-for-bit; padding tokens produce zero output and touch no state.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from ..kda_chunk._xla_impl import (
    PrecisionLike,
    chunk_core,
    l2norm,
    pairwise_sub_block,
    resolve_precision,
    validate_decay_lower_bound,
)

INVERSE_METHODS = ("auto", "masked", "solve")
PAIRWISE_MODES = ("auto", "dense")


def resolve_inverse(inverse: str, chunk_size: int) -> str:
    """Resolve ``inverse="auto"`` for a chunk size.

    The serial chunk loop prefers the matmul-only block-recursive inverse (``"masked"``): XLA expands an exact
    ``triangular_solve`` into its own row loop, which would nest inside the chunk loop. ``"masked"`` needs a
    power-of-two chunk of at least 8 tokens; other sizes use the exact solve.

    Args:
        inverse: ``"auto"``, ``"masked"`` or ``"solve"``.
        chunk_size: Chunk length.

    Returns:
        ``"masked"`` or ``"solve"``.
    """
    if inverse != "auto":
        return inverse
    cs = int(chunk_size)
    return "masked" if cs >= 8 and not cs & (cs - 1) else "solve"


def validate_options(
    chunk_size: int,
    precision: PrecisionLike,
    inverse: str,
    pairwise: str,
    decay_lower_bound: float | None,
) -> None:
    """Validate the static options shared by every ``ragged_kda`` backend.

    Args:
        chunk_size: Chunk length of the multi-token path.
        precision: Precision of the non-inverse chunk dots (``"highest"`` / ``"default"`` / ``lax.Precision``).
        inverse: Chunk-inverse method, ``"auto"``, ``"masked"`` or ``"solve"``.
        pairwise: ``"auto"`` or ``"dense"``.
        decay_lower_bound: ``None`` or a finite log-decay bound ``<= 0``.

    Raises:
        ValueError: On any invalid option.
    """
    cs = int(chunk_size)
    if cs <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    resolve_precision(precision)
    if inverse not in INVERSE_METHODS:
        raise ValueError(f"inverse must be one of {INVERSE_METHODS}, got {inverse!r}")
    if inverse == "masked" and (cs < 8 or cs & (cs - 1)):
        raise ValueError(f"inverse='masked' needs a power-of-two chunk_size >= 8, got {chunk_size}")
    if pairwise not in PAIRWISE_MODES:
        raise ValueError(f"pairwise must be one of {PAIRWISE_MODES}, got {pairwise!r}")
    validate_decay_lower_bound(decay_lower_bound)


def validate_shapes(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    recurrent_state: Array,
    query_start_loc: Array,
    state_indices: Array | None,
    has_initial_state: Array | None,
) -> None:
    """Check the ``ragged_kda`` array contract.

    Args:
        query: ``[T, H, K]``.
        key: ``[T, H, K]``.
        value: ``[T, H, V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        recurrent_state: ``[S, H, K, V]``.
        query_start_loc: ``[R + 1]``.
        state_indices: ``[R]``, or ``None`` for the identity layout (row ``r`` owns slot ``r``; needs ``S >= R``).
        has_initial_state: ``[R]`` or ``None``.

    Raises:
        ValueError: If any shape disagrees with the contract above.
    """
    if query.ndim != 3:
        raise ValueError(f"query must be [num_tokens, num_heads, head_dim], got shape {query.shape}")
    num_tokens, num_heads, head_dim = query.shape
    if key.shape != query.shape:
        raise ValueError(f"key shape {key.shape} must equal query shape {query.shape}")
    if value.ndim != 3 or value.shape[:2] != (num_tokens, num_heads):
        raise ValueError(f"value must be [{num_tokens}, {num_heads}, v_dim], got shape {value.shape}")
    if beta.shape != (num_tokens, num_heads):
        raise ValueError(f"beta must be [{num_tokens}, {num_heads}], got shape {beta.shape}")
    if decay.shape != query.shape:
        raise ValueError(f"decay must be per-channel {query.shape}, got shape {decay.shape}")
    v_dim = value.shape[-1]
    if recurrent_state.ndim != 4 or recurrent_state.shape[1:] != (num_heads, head_dim, v_dim):
        raise ValueError(
            f"recurrent_state must be [num_slots, {num_heads}, {head_dim}, {v_dim}], got shape {recurrent_state.shape}"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError(f"query_start_loc must be [num_rows + 1] with num_rows >= 1, got {query_start_loc.shape}")
    num_rows = query_start_loc.shape[0] - 1
    if state_indices is None:
        if recurrent_state.shape[0] < num_rows:
            raise ValueError(
                f"state_indices=None maps row r to pool slot r, so the pool needs at least {num_rows} slots, "
                f"got {recurrent_state.shape[0]}"
            )
    elif state_indices.shape != (num_rows,):
        raise ValueError(f"state_indices must be [{num_rows}], got shape {state_indices.shape}")
    if has_initial_state is not None and has_initial_state.shape != (num_rows,):
        raise ValueError(f"has_initial_state must be [{num_rows}], got shape {has_initial_state.shape}")


def decode_step(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    state: Array,
    use_qk_l2norm: bool = True,
) -> tuple[Array, Array]:
    """One per-channel KDA token for every row, with the eSurge decode arithmetic.

    Q and K are L2-normalised in their input dtype, then everything runs in fp32 as elementwise products and
    reductions (no matmuls), exactly like the serving decode step.

    Args:
        query: ``[R, H, K]``.
        key: ``[R, H, K]``.
        value: ``[R, H, V]``.
        beta: ``[R, H]``.
        decay: ``[R, H, K]`` per-channel log-decay.
        state: ``[R, H, K, V]`` incoming state.
        use_qk_l2norm: L2-normalise Q and K first.

    Returns:
        ``(out [R, H, V], new_state [R, H, K, V])``, both fp32.
    """
    if use_qk_l2norm:
        query = l2norm(query)
        key = l2norm(key)
    head_dim = query.shape[-1]
    q = query.astype(jnp.float32) * (1.0 / (head_dim**0.5))
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)
    b = beta.astype(jnp.float32)
    s = state.astype(jnp.float32) * jnp.exp(decay.astype(jnp.float32))[..., None]
    kv_mem = jnp.sum(s * k[..., None], axis=-2)
    delta = (v - kv_mem) * b[..., None]
    s = s + k[..., None] * delta[..., None, :]
    return jnp.sum(s * q[..., None], axis=-2), s


def _decode_phase(
    out: Array,
    pool: Array,
    *,
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    starts: Array,
    lens: Array,
    state_indices: Array | None,
    has_init: Array | None,
    num_tokens: int,
    use_qk_l2norm: bool,
) -> tuple[Array, Array]:
    """Advance every single-token row by one decode step; other rows are left untouched.

    With ``state_indices=None`` (row ``r`` owns slot ``r``) the rows' states are the leading ``R`` slots of the pool,
    so the step is one masked read-modify-write over them: no gather and no scatter of pool-sized buffers. Otherwise
    the states are gathered by ``state_indices`` and scattered back.
    """
    num_slots = pool.shape[0]
    num_rows = lens.shape[0]
    tok = jnp.clip(starts, 0, num_tokens - 1)
    is_decode = lens == 1
    if state_indices is None:
        held = pool if num_slots == num_rows else lax.slice_in_dim(pool, 0, num_rows, axis=0)
        state = held
    else:
        held = None
        state = pool[jnp.clip(state_indices, 0, num_slots - 1)]
    if has_init is not None:
        state = jnp.where(has_init[:, None, None, None], state, jnp.zeros_like(state))
    o, new_state = decode_step(query[tok], key[tok], value[tok], beta[tok], decay[tok], state, use_qk_l2norm)
    if held is not None:
        # Rows without exactly one token keep their slot bit for bit.
        rows = jnp.where(is_decode[:, None, None, None], new_state.astype(pool.dtype), held)
        pool = rows if num_slots == num_rows else lax.dynamic_update_slice_in_dim(pool, rows, 0, axis=0)
    else:
        # Out-of-range indices drop the write, so rows without exactly one token never touch the pool.
        pool = pool.at[jnp.where(is_decode, state_indices, num_slots)].set(new_state.astype(pool.dtype), mode="drop")
    out = out.at[jnp.where(is_decode, tok, out.shape[0])].set(o, mode="drop")
    return out, pool


def _chunk_phase(
    out: Array,
    pool: Array,
    *,
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    query_start_loc: Array,
    lens: Array,
    state_indices: Array | None,
    has_init: Array | None,
    chunk_size: int,
    use_qk_l2norm: bool,
    precision: lax.Precision,
    inverse: str,
    sub: int,
) -> tuple[Array, Array]:
    """Run every multi-token row chunk by chunk (one live chunk per loop step).

    Each step reads the chunk's ``chunk_size``-token window straight from the flat stream. Near the end of the
    buffer the window start is clamped back, so the chunk's tokens may start after position 0 of the window: the
    tokens before them (and after the row's end) are zeroed, which makes them exactly inert in the chunk math (zero
    output, identity decay, no state write).

    The pool is updated in place. Every pool read of a step (the row's stored state, used at its first chunk) is
    materialised before the chunk math, and every step stores its running state in the row's slot, so no read of the
    old pool can be scheduled after the step's write (which would make XLA copy the whole pool every step).
    """
    padded_tokens, num_heads, head_dim = query.shape
    v_dim = value.shape[-1]
    num_rows = lens.shape[0]
    cs = chunk_size
    f32 = jnp.float32
    scale = 1.0 / (head_dim**0.5)

    n_chunks = jnp.where(lens >= 2, (lens + cs - 1) // cs, 0)
    chunk_end = jnp.cumsum(n_chunks)
    chunk_begin = chunk_end - n_chunks
    lane = jnp.arange(cs, dtype=jnp.int32)
    core = jax.vmap(functools.partial(chunk_core, seg=None, precision=precision, inverse=inverse, sub=sub))

    def window(x: Array, start: Array) -> Array:
        return lax.dynamic_slice_in_dim(x, start, cs, axis=0)

    def body(c, carry):
        pool, out, state = carry
        row = jnp.minimum(jnp.sum(chunk_end <= c), num_rows - 1).astype(jnp.int32)
        local = c - chunk_begin[row]
        chunk_start = query_start_loc[row] + local * cs
        win_start = jnp.clip(chunk_start, 0, padded_tokens - cs)
        pos = win_start + lane
        valid = (pos >= chunk_start) & (pos < jnp.minimum(chunk_start + cs, query_start_loc[row + 1]))
        vmask = valid[:, None, None]

        q = window(query, win_start).astype(f32)
        k = window(key, win_start).astype(f32)
        if use_qk_l2norm:
            q = l2norm(q)
            k = l2norm(k)
        q = jnp.where(vmask, q * scale, 0.0)
        k = jnp.where(vmask, k, 0.0)
        v = jnp.where(vmask, window(value, win_start).astype(f32), 0.0)
        g = jnp.where(vmask, window(decay, win_start).astype(f32), 0.0)
        b = jnp.where(valid[:, None], window(beta, win_start).astype(f32), 0.0)[..., None]
        q, k, v, g, kb, vb = (jnp.swapaxes(x, 0, 1) for x in (q, k, v, g, k * b, v * b))  # [H, cs, D]

        slot = row if state_indices is None else state_indices[row]
        stored = lax.dynamic_index_in_dim(pool, slot, axis=0, keepdims=False).astype(f32)
        if has_init is not None:
            stored = jnp.where(has_init[row], stored, jnp.zeros_like(stored))
        # One materialised incoming state. Left alone, XLA re-derives it (pool slot read included) inside each of its
        # consumers (the chunk dots, the new state, the output window); a consumer scheduled after this step's pool
        # write then keeps the old pool alive, and XLA copies the whole pool (twice per step, seen in serving).
        state = lax.optimization_barrier(jnp.where(local == 0, stored, state))

        g_cumsum = jnp.cumsum(g, axis=1)
        state, chunk_out = core(state, q, k, kb, vb, g_cumsum, g_cumsum[:, -1:, :])

        merged = jnp.where(vmask, jnp.swapaxes(chunk_out, 0, 1), window(out, win_start))
        out = lax.dynamic_update_slice_in_dim(out, merged, win_start, axis=0)
        # Every step stores the running state (a later chunk of the row overwrites it; the row's first chunk read the
        # slot above), so the write needs no read of the slot's old value. Rows without chunks are never written.
        pool = lax.dynamic_update_slice_in_dim(pool, state.astype(pool.dtype)[None], slot, axis=0)
        return pool, out, state

    state0 = jnp.zeros((num_heads, head_dim, v_dim), f32)
    pool, out, _ = lax.fori_loop(0, chunk_end[-1], body, (pool, out, state0))
    return out, pool


def ragged_kda_xla(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    recurrent_state: Array,
    query_start_loc: Array,
    state_indices: Array | None,
    has_initial_state: Array | None = None,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: PrecisionLike = "highest",
    inverse: str = "auto",
    pairwise: str = "auto",
) -> tuple[Array, Array]:
    """Ragged per-channel KDA over a packed token stream and a recurrent-state pool (XLA).

    Args:
        query: Post-conv queries ``[T, H, K]``.
        key: Post-conv keys ``[T, H, K]``.
        value: Post-conv values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid already applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``).
        recurrent_state: State pool ``[S, H, K, V]``; computed in fp32, returned in its own dtype.
        query_start_loc: Cumulative row offsets ``[R + 1]`` (nondecreasing).
        state_indices: Pool slot of every row ``[R]``; rows with an out-of-range slot are treated as empty. ``None``
            is the identity layout (row ``r`` owns slot ``r``, ``S >= R``): the decode phase then updates the leading
            ``R`` slots in place, without gathering or scattering pool-sized buffers.
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state.
        chunk_size: Chunk length of the multi-token path.
        use_qk_l2norm: L2-normalise queries and keys.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound``; with ``pairwise="auto"``
            it selects the factored MXU pairwise decay of the chunk math. ``None`` keeps the dense form.
        precision: Precision of the non-inverse chunk dots.
        inverse: ``"auto"`` (see :func:`resolve_inverse`), ``"masked"`` or ``"solve"``.
        pairwise: ``"auto"`` or ``"dense"``.

    Returns:
        ``(out [T, H, V] fp32, new_recurrent_state [S, H, K, V])``.
    """
    num_tokens, num_heads, _ = query.shape
    v_dim = value.shape[-1]
    cs = int(chunk_size)
    num_slots = recurrent_state.shape[0]
    qsl = query_start_loc.astype(jnp.int32)
    lens = jnp.maximum(qsl[1:] - qsl[:-1], 0)
    if state_indices is not None:
        state_indices = state_indices.astype(jnp.int32)
        slot_ok = (state_indices >= 0) & (state_indices < num_slots)
        lens = jnp.where(slot_ok, lens, 0)
    has_init = None if has_initial_state is None else jnp.asarray(has_initial_state).astype(jnp.bool_)
    sub = pairwise_sub_block(decay_lower_bound, cs) if pairwise == "auto" else 0
    inverse = resolve_inverse(inverse, cs)

    pad = max(cs - num_tokens, 0)  # a chunk window must fit in the buffer
    if pad:
        query, key, value, decay = (jnp.pad(x, ((0, pad), (0, 0), (0, 0))) for x in (query, key, value, decay))
        beta = jnp.pad(beta, ((0, pad), (0, 0)))
    arrays = dict(query=query, key=key, value=value, beta=beta, decay=decay)
    out = jnp.zeros((num_tokens + pad, num_heads, v_dim), jnp.float32)

    out, pool = _decode_phase(
        out,
        recurrent_state,
        **arrays,
        starts=qsl[:-1],
        lens=lens,
        state_indices=state_indices,
        has_init=has_init,
        num_tokens=num_tokens,
        use_qk_l2norm=use_qk_l2norm,
    )
    # Pure decode steps run zero iterations of the chunk loop.
    out, pool = _chunk_phase(
        out,
        pool,
        **arrays,
        query_start_loc=qsl,
        lens=lens,
        state_indices=state_indices,
        has_init=has_init,
        chunk_size=cs,
        use_qk_l2norm=use_qk_l2norm,
        precision=resolve_precision(precision),
        inverse=inverse,
        sub=sub,
    )
    return out[:num_tokens], pool


__all__ = ("decode_step", "ragged_kda_xla", "resolve_inverse", "validate_options", "validate_shapes")
