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

"""XLA reference for ragged (flat-token) KDA: fused causal conv, recurrence and per-token state capture.

The packed stream holds the tokens of ``R`` rows back to back: row ``r`` owns tokens
``[query_start_loc[r], query_start_loc[r + 1])`` (``lens[r]`` of them) and the group of pool slots that starts at its
base slot ``g = state_indices[r]``. With the six conv arrays (weights and window pools) the inputs are the raw
(pre-conv) q / k / v projections and the op first runs the depthwise causal conv (SiLU fused) with every row's carried
windows; without them the inputs are already convolved and no conv window is read or written. Then the per-channel
KDA recurrence::

    S_t = diag(exp(g_t)) S_{t-1};   S_t += k_t (beta_t (v_t - k_t^T S_t))^T;   o_t = S_t^T q_t

with ``q`` scaled by ``K**-0.5`` and ``q`` / ``k`` optionally L2-normalised.

Per row ``r`` with base slot ``g``, read offset ``a = read_offset[r]`` and commit length ``c = commit_lens[r]``:

1. the conv windows and the state are read from slot ``g + a`` (zeros when ``has_initial_state[r]`` is False);
2. the row's tokens are processed causally and every output is written;
3. rows of at most ``short_row_max`` (``W``) tokens write, for ``j = 0 .. lens - c``, the conv windows and the state
   as they stand after token ``c - 1 + j`` into slot ``g + j``: an ordinary row (``c = lens``) writes only its final
   state, into slot ``g``. Longer rows write their final windows and state into slot ``g`` (``commit_lens`` is ignored
   for them; the runner sends ``c = lens``).

A row reads before it writes, so a row may overwrite its own read slot. Rows own disjoint slot groups. Rows without
tokens, and rows whose slots (``g``, ``g + a``, ``g + lens - c``) fall out of range, produce zero output and write
nothing; tokens at or past ``query_start_loc[R]`` are padding (zero output).

Two phases:

* **Short rows** (``1 <= lens <= W``): a loop over the ``W`` token positions runs one conv token and one recurrence
  token for every short row with the arithmetic of separate decode steps: the carried windows and the state are
  rounded to their pool dtypes after every token (what storing them between decode steps does), and the captured
  slots are scattered after each position. A ``W``-token verify row therefore reproduces ``W`` one-token steps.
* **Chunk rows** (``lens > W``): the conv of the flat stream (``ragged_causal_conv1d``'s arithmetic) gives their
  post-conv tokens and final windows; every row is cut into ``chunk_size`` chunks aligned to its first token and a
  ``fori_loop`` over the live chunks runs :func:`~ejkernel.kernels._xla.kda_chunk._xla_impl.chunk_core` (the shared
  ``kda_chunk`` per-chunk math), loading the row's state from slot ``g + a`` at its first chunk and storing the
  running state into slot ``g``.

With ``W = 1``, ``a = 0`` and ``c = lens`` this is the ordinary serving step; on post-conv inputs it is the round-4
``ragged_kda`` arithmetic (single-token rows: :func:`decode_step`; longer rows: the chunk loop).
"""

from __future__ import annotations

import functools
import typing as tp

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
from ..ragged_causal_conv1d._xla_impl_fwd import _ragged_causal_conv1d_impl

INVERSE_METHODS = ("auto", "masked", "solve")
PAIRWISE_MODES = ("auto", "dense")
CONV_ARRAYS = ("q_conv_weight", "k_conv_weight", "v_conv_weight", "q_conv_state", "k_conv_state", "v_conv_state")


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
    short_row_max: int = 1,
) -> None:
    """Validate the static options shared by every ``ragged_kda`` backend.

    Args:
        chunk_size: Chunk length of the multi-token path.
        precision: Precision of the non-inverse chunk dots (``"highest"`` / ``"default"`` / ``lax.Precision``).
        inverse: Chunk-inverse method, ``"auto"``, ``"masked"`` or ``"solve"``.
        pairwise: ``"auto"`` or ``"dense"``.
        decay_lower_bound: ``None`` or a finite log-decay bound ``<= 0``.
        short_row_max: ``W``: rows of at most ``W`` tokens take the short-row path (``>= 1``).

    Raises:
        ValueError: On any invalid option.
    """
    cs = int(chunk_size)
    if cs <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if int(short_row_max) < 1:
        raise ValueError(f"short_row_max must be >= 1, got {short_row_max}")
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
    has_initial_state: Array | None = None,
    read_offset: Array | None = None,
    commit_lens: Array | None = None,
    q_conv_weight: Array | None = None,
    k_conv_weight: Array | None = None,
    v_conv_weight: Array | None = None,
    q_conv_state: Array | None = None,
    k_conv_state: Array | None = None,
    v_conv_state: Array | None = None,
) -> None:
    """Check the ``ragged_kda`` array contract.

    Args:
        query: Queries ``[T, H, K]`` (raw with the conv arrays, post-conv without).
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        recurrent_state: ``[S, H, K, V]``.
        query_start_loc: ``[R + 1]``.
        state_indices: ``[R]`` base slots, or ``None`` (row ``r``'s base slot is ``r``; needs ``S >= R``).
        has_initial_state: ``[R]`` or ``None``.
        read_offset: ``[R]`` or ``None``.
        commit_lens: ``[R]`` or ``None``.
        q_conv_weight: ``[H * K, d_conv]`` or ``None``.
        k_conv_weight: ``[H * K, d_conv]`` or ``None``.
        v_conv_weight: ``[H * V, d_conv]`` or ``None``.
        q_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        k_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        v_conv_state: ``[S, H * V, d_conv]`` or ``None``.

    Raises:
        ValueError: If any shape disagrees with the contract above, or only some of the six conv arrays are given.
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
    num_slots = recurrent_state.shape[0]
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError(f"query_start_loc must be [num_rows + 1] with num_rows >= 1, got {query_start_loc.shape}")
    num_rows = query_start_loc.shape[0] - 1
    if state_indices is None:
        if num_slots < num_rows:
            raise ValueError(
                f"state_indices=None maps row r to pool slot r, so the pool needs at least {num_rows} slots, "
                f"got {num_slots}"
            )
    elif state_indices.shape != (num_rows,):
        raise ValueError(f"state_indices must be [{num_rows}], got shape {state_indices.shape}")
    for name, arr in (
        ("has_initial_state", has_initial_state),
        ("read_offset", read_offset),
        ("commit_lens", commit_lens),
    ):
        if arr is not None and arr.shape != (num_rows,):
            raise ValueError(f"{name} must be [{num_rows}], got shape {arr.shape}")
    conv = (q_conv_weight, k_conv_weight, v_conv_weight, q_conv_state, k_conv_state, v_conv_state)
    given = sum(a is not None for a in conv)
    if given not in (0, len(conv)):
        raise ValueError(
            f"pass all six conv arrays ({', '.join(CONV_ARRAYS)}) with raw q / k / v, or none of them with post-conv "
            f"inputs; got {given} of them"
        )
    if not given:
        return
    if q_conv_weight.ndim != 2 or q_conv_weight.shape[-1] < 1:
        raise ValueError(f"q_conv_weight must be [{num_heads * head_dim}, d_conv], got shape {q_conv_weight.shape}")
    d_conv = q_conv_weight.shape[-1]
    for name, weight, state, dim in (
        ("q", q_conv_weight, q_conv_state, num_heads * head_dim),
        ("k", k_conv_weight, k_conv_state, num_heads * head_dim),
        ("v", v_conv_weight, v_conv_state, num_heads * v_dim),
    ):
        if weight.shape != (dim, d_conv):
            raise ValueError(f"{name}_conv_weight must be [{dim}, {d_conv}], got shape {weight.shape}")
        if state.shape != (num_slots, dim, d_conv):
            raise ValueError(f"{name}_conv_state must be [{num_slots}, {dim}, {d_conv}], got shape {state.shape}")


def is_lowfloat(dtype: jnp.dtype) -> bool:
    """Whether ``dtype`` is an 8- or 4-bit float (one byte per element), which JAX never promotes implicitly."""
    dtype = jnp.dtype(dtype)
    return bool(jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize <= 1)


def conv_compute_dtype(stream_dtype: jnp.dtype, pool_dtype: jnp.dtype) -> jnp.dtype:
    """Dtype in which the conv meets the token stream and the carried windows (float32 for 1-byte floats)."""
    if is_lowfloat(stream_dtype) or is_lowfloat(pool_dtype):
        return jnp.dtype(jnp.float32)
    return jnp.promote_types(stream_dtype, pool_dtype)


def conv_output_dtype(stream_dtype: jnp.dtype) -> jnp.dtype:
    """Dtype of the conv outputs fed to the recurrence: the stream's (bfloat16 for 1-byte float streams)."""
    return jnp.dtype(jnp.bfloat16) if is_lowfloat(stream_dtype) else jnp.dtype(stream_dtype)


class RowPlan(tp.NamedTuple):
    """Per-row step plan shared by every backend (all ``[R]``).

    Attributes:
        starts: First token of every row.
        lens: Tokens per row; zero for rows that are not live.
        short: Live rows of at most ``W`` tokens (short-row path).
        chunk: Live rows of more than ``W`` tokens (chunk path).
        read_slot: ``g + a``, the slot the row reads (meaningful for live rows).
        base_slot: ``g``.
        commit: ``c`` clipped to ``[1, lens]`` for short rows; ``lens`` for the others.
        init: ``has_initial_state`` (rows without an initial state start from zero windows and a zero state).
        identity: Static: no ``state_indices``, ``read_offset`` or ``commit_lens``, so row ``r`` reads and writes only
            slot ``r`` (once, after its last token): reads slice the leading ``R`` slots, writes update them in place.
    """

    starts: Array
    lens: Array
    short: Array
    chunk: Array
    read_slot: Array
    base_slot: Array
    commit: Array
    init: Array
    identity: bool = False


def plan_rows(
    query_start_loc: Array,
    state_indices: Array | None,
    has_initial_state: Array | None,
    read_offset: Array | None,
    commit_lens: Array | None,
    *,
    num_slots: int,
    short_row_max: int,
) -> RowPlan:
    """Resolve slots, offsets and the short / chunk split of every row (see the module docstring).

    Args:
        query_start_loc: ``[R + 1]`` row offsets.
        state_indices: ``[R]`` base slots, or ``None`` (``g = r``).
        has_initial_state: ``[R]`` or ``None`` (all True).
        read_offset: ``[R]`` or ``None`` (all zero).
        commit_lens: ``[R]`` or ``None`` (``c = lens``).
        num_slots: Pool slot count ``S``.
        short_row_max: ``W``.

    Returns:
        The :class:`RowPlan`.
    """
    i32 = jnp.int32
    qsl = query_start_loc.astype(i32)
    num_rows = qsl.shape[0] - 1
    lens = jnp.maximum(qsl[1:] - qsl[:-1], 0)
    base = jnp.arange(num_rows, dtype=i32) if state_indices is None else jnp.asarray(state_indices).astype(i32)
    offset = jnp.zeros((num_rows,), i32) if read_offset is None else jnp.asarray(read_offset).astype(i32)
    commit = lens if commit_lens is None else jnp.asarray(commit_lens).astype(i32)
    is_short = lens <= int(short_row_max)
    commit = jnp.where(is_short, jnp.clip(commit, 1, jnp.maximum(lens, 1)), lens)
    last_capture = base + jnp.where(is_short, lens - commit, 0)
    read = base + offset
    live = (lens > 0) & (base >= 0) & (offset >= 0) & (read < num_slots) & (last_capture < num_slots)
    init = jnp.ones((num_rows,), jnp.bool_) if has_initial_state is None else jnp.asarray(has_initial_state).astype(bool)
    return RowPlan(
        starts=qsl[:-1],
        lens=jnp.where(live, lens, 0),
        short=live & is_short,
        chunk=live & jnp.logical_not(is_short),
        read_slot=read,
        base_slot=base,
        commit=commit,
        init=init,
        identity=state_indices is None and read_offset is None and commit_lens is None,
    )


def read_rows(pool: Array, plan: RowPlan) -> Array:
    """Every row's read slot of ``pool`` ``[R, ...]``: a slice of the leading ``R`` slots in the identity layout."""
    num_rows = plan.lens.shape[0]
    if plan.identity:
        return pool if pool.shape[0] == num_rows else lax.slice_in_dim(pool, 0, num_rows, axis=0)
    return pool[jnp.clip(plan.read_slot, 0, pool.shape[0] - 1)]


def put_rows(pool: Array, rows: Array) -> Array:
    """Write ``rows`` ``[R, ...]`` into the leading ``R`` slots of ``pool`` (identity layout, in place)."""
    if pool.shape[0] == rows.shape[0]:
        return rows
    return lax.dynamic_update_slice_in_dim(pool, rows, 0, axis=0)


def conv_step(window: Array, x: Array, weight: Array, *, out_dtype: jnp.dtype) -> tuple[Array, Array]:
    """One causal-conv token for every row: shift ``x`` into the window, dot the taps in fp32, apply SiLU.

    The arithmetic of the round-4 eSurge decode driver's conv (sequential fp32 tap sum, SiLU, rounded through the
    window dtype).

    Args:
        window: ``[R, dim, d_conv]`` carried window in the conv compute dtype (oldest tap first).
        x: ``[R, dim]`` the rows' new raw inputs.
        weight: ``[dim, d_conv]`` depthwise taps (the last tap multiplies the current token).
        out_dtype: Dtype of the conv output.

    Returns:
        ``(out [R, dim] out_dtype, new_window [R, dim, d_conv] in window.dtype)``.
    """
    f32 = jnp.float32
    new_window = jnp.concatenate([window[:, :, 1:], x.astype(window.dtype)[:, :, None]], axis=-1)
    taps = new_window.astype(f32)
    weights = weight.astype(f32)
    mixed = taps[:, :, 0] * weights[None, :, 0]
    for j in range(1, int(weight.shape[-1])):
        mixed = mixed + taps[:, :, j] * weights[None, :, j]
    return jax.nn.silu(mixed).astype(window.dtype).astype(out_dtype), new_window


def flat_conv(stream: Array, weight: Array, windows: Array, query_start_loc: Array, init: Array) -> tuple[Array, Array]:
    """Causal conv of the whole flat stream with every row's carried window (``ragged_causal_conv1d`` arithmetic).

    Args:
        stream: ``[T, dim]`` raw inputs.
        weight: ``[dim, d_conv]``.
        windows: ``[R, dim, d_conv]`` the rows' carried windows (pool dtype).
        query_start_loc: ``[R + 1]``.
        init: ``[R]``; False rows start from a zero window.

    Returns:
        ``(out [T, dim] in conv_output_dtype(stream), new_windows [R, dim, d_conv] in the pool dtype)``: every row's
        window after its last token (rows without tokens keep theirs).
    """
    num_rows = windows.shape[0]
    compute = conv_compute_dtype(stream.dtype, windows.dtype)
    live = jnp.where(init[:, None, None], windows.astype(compute), jnp.zeros((), compute))
    out, new_windows = _ragged_causal_conv1d_impl(
        stream.astype(compute),
        live,
        weight,
        query_start_loc.astype(jnp.int32),
        jnp.arange(num_rows, dtype=jnp.int32),
        jnp.full((3,), num_rows, dtype=jnp.int32),
        d_conv=int(weight.shape[-1]),
        apply_silu=True,
    )
    return out.astype(conv_output_dtype(stream.dtype)), new_windows.astype(windows.dtype)


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


# step_fn(query, key, value, beta, decay, state, valid, init, use_qk_l2norm) -> (out fp32, new_state in state.dtype)
StepFn = tp.Callable[..., tuple[Array, Array]]


def xla_step(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    state: Array,
    valid: Array,
    init: Array | None,
    use_qk_l2norm: bool = True,
) -> tuple[Array, Array]:
    """One short-row token for every row: :func:`decode_step`, the new state rounded to ``state``'s dtype.

    Args:
        query: Post-conv queries ``[R, H, K]``.
        key: Post-conv keys ``[R, H, K]``.
        value: Post-conv values ``[R, H, V]``.
        beta: ``[R, H]``.
        decay: ``[R, H, K]``.
        state: ``[R, H, K, V]`` carried state (pool dtype; already zero for rows without an initial state).
        valid: ``[R]`` rows that take this token (unused: the caller masks).
        init: ``[R]`` initial-state flags at the row's first token, else ``None`` (unused: the state is zeroed).
        use_qk_l2norm: L2-normalise queries and keys.

    Returns:
        ``(out [R, H, V] fp32, new_state [R, H, K, V] in state.dtype)``.
    """
    del valid, init
    o, s = decode_step(query, key, value, beta, decay, state, use_qk_l2norm)
    return o, s.astype(state.dtype)


def short_phase(
    out: Array,
    pools: tuple[Array | None, Array | None, Array | None, Array],
    *,
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    weights: tuple[Array, Array, Array] | None,
    plan: RowPlan,
    short_row_max: int,
    use_qk_l2norm: bool,
    step_fn: StepFn | None = None,
) -> tuple[Array, tuple[Array | None, Array | None, Array | None, Array]]:
    """Advance every short row token by token, capturing windows and states into the row's slot group.

    Args:
        out: ``[T, H, V]`` fp32 output buffer.
        pools: ``(q_conv_state, k_conv_state, v_conv_state, recurrent_state)``; the conv pools are ``None`` (and
            untouched) for post-conv inputs.
        query: Queries ``[T, H, K]`` (raw with ``weights``, post-conv without).
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        weights: ``(q, k, v)`` conv taps ``[dim, d_conv]``, or ``None`` for post-conv inputs.
        plan: The step's :class:`RowPlan`.
        short_row_max: ``W`` (static): the loop runs ``W`` token positions.
        use_qk_l2norm: L2-normalise queries and keys.
        step_fn: Recurrence of one token for every row (:func:`xla_step`'s signature; default :func:`xla_step`).
            It must return the new state in the pool dtype; ``init`` is passed at the first position only.

    Returns:
        ``(out, pools)`` with the short rows' outputs and captured slots written.
    """
    q_pool, k_pool, v_pool, rec = pools
    num_tokens, num_heads, head_dim = query.shape
    v_dim = value.shape[-1]
    num_slots = rec.shape[0]
    num_rows = plan.lens.shape[0]
    step = step_fn or xla_step
    # Every read of the step happens here, before the first capture is written.
    held_rec = read_rows(rec, plan)
    state = jnp.where(plan.init[:, None, None, None], held_rec, jnp.zeros((), rec.dtype))
    fused = weights is not None
    if fused:
        conv_pools = [q_pool, k_pool, v_pool]
        streams = (
            query.reshape(num_tokens, num_heads * head_dim),
            key.reshape(num_tokens, num_heads * head_dim),
            value.reshape(num_tokens, num_heads * v_dim),
        )
        computes = [conv_compute_dtype(s.dtype, p.dtype) for s, p in zip(streams, conv_pools, strict=True)]
        held = [read_rows(p, plan) for p in conv_pools]
        windows = [jnp.where(plan.init[:, None, None], h, jnp.zeros((), h.dtype)) for h in held]
        shapes = ((num_rows, num_heads, head_dim), (num_rows, num_heads, head_dim), (num_rows, num_heads, v_dim))
    for j in range(int(short_row_max)):
        tok = jnp.clip(plan.starts + j, 0, num_tokens - 1)
        valid = plan.short & (j < plan.lens)
        if fused:
            conv = []
            for i in range(3):
                o_i, new_window = conv_step(
                    windows[i].astype(computes[i]),
                    streams[i][tok],
                    weights[i],
                    out_dtype=conv_output_dtype(streams[i].dtype),
                )
                conv.append(o_i.reshape(shapes[i]))
                # Stored between tokens exactly as separate decode steps store it: in the pool dtype.
                windows[i] = jnp.where(valid[:, None, None], new_window.astype(conv_pools[i].dtype), windows[i])
        else:
            conv = [query[tok], key[tok], value[tok]]
        o, new_state = step(
            conv[0],
            conv[1],
            conv[2],
            beta[tok],
            decay[tok],
            state,
            valid,
            plan.init if j == 0 else None,
            use_qk_l2norm,
        )
        state = jnp.where(valid[:, None, None, None], new_state, state)
        out = out.at[jnp.where(valid, tok, out.shape[0])].set(o, mode="drop")
        if plan.identity:
            continue
        capture = valid & (j >= plan.commit - 1)
        dest = jnp.where(capture, plan.base_slot + j - (plan.commit - 1), num_slots)
        rec = rec.at[dest].set(state, mode="drop")
        if fused:
            for i in range(3):
                conv_pools[i] = conv_pools[i].at[dest].set(windows[i], mode="drop")
    if plan.identity:
        # Row r reads and writes only slot r, after its last token: one masked update of the leading R slots.
        rec = put_rows(rec, jnp.where(plan.short[:, None, None, None], state, held_rec))
        if fused:
            for i in range(3):
                conv_pools[i] = put_rows(conv_pools[i], jnp.where(plan.short[:, None, None], windows[i], held[i]))
    if fused:
        q_pool, k_pool, v_pool = conv_pools
    return out, (q_pool, k_pool, v_pool, rec)


def chunk_phase(
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
    read_slot: Array,
    write_slot: Array,
    has_init: Array,
    chunk_size: int,
    use_qk_l2norm: bool,
    precision: lax.Precision,
    inverse: str,
    sub: int,
) -> tuple[Array, Array]:
    """Run every chunk row chunk by chunk (one live chunk per loop step) on post-conv inputs.

    Each step reads the chunk's ``chunk_size``-token window straight from the flat stream. Near the end of the
    buffer the window start is clamped back, so the chunk's tokens may start after position 0 of the window: the
    tokens before them (and after the row's end) are zeroed, which makes them exactly inert in the chunk math (zero
    output, identity decay, no state write).

    The pool is updated in place. Every pool read of a step (the row's stored state from ``read_slot``, used at its
    first chunk) is materialised before the chunk math, and every step stores its running state into ``write_slot``,
    so no read of the old pool can be scheduled after the step's write (which would make XLA copy the whole pool every
    step).

    Args:
        out: ``[T', H, V]`` fp32 output buffer.
        pool: ``[S, H, K, V]`` recurrent pool.
        query: Post-conv queries ``[T', H, K]`` (``T' >= chunk_size``).
        key: Post-conv keys.
        value: Post-conv values.
        beta: ``[T', H]``.
        decay: ``[T', H, K]``.
        query_start_loc: ``[R + 1]``.
        lens: ``[R]`` tokens of the chunk rows (zero for every other row).
        read_slot: ``[R]`` slot read at the row's first chunk.
        write_slot: ``[R]`` slot that receives the row's state.
        has_init: ``[R]``; False rows start from zero.
        chunk_size: Chunk length.
        use_qk_l2norm: L2-normalise queries and keys.
        precision: Precision of the non-inverse chunk dots.
        inverse: Resolved chunk inverse.
        sub: Factored pairwise sub-block (``0``: dense).

    Returns:
        ``(out, pool)``.
    """
    padded_tokens, num_heads, head_dim = query.shape
    v_dim = value.shape[-1]
    num_rows = lens.shape[0]
    num_slots = pool.shape[0]
    cs = chunk_size
    f32 = jnp.float32
    scale = 1.0 / (head_dim**0.5)

    n_chunks = jnp.where(lens >= 2, (lens + cs - 1) // cs, 0)
    chunk_end = jnp.cumsum(n_chunks)
    chunk_begin = chunk_end - n_chunks
    lane = jnp.arange(cs, dtype=jnp.int32)
    read_slot = jnp.clip(read_slot, 0, num_slots - 1)
    write_slot = jnp.clip(write_slot, 0, num_slots - 1)
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

        stored = lax.dynamic_index_in_dim(pool, read_slot[row], axis=0, keepdims=False).astype(f32)
        stored = jnp.where(has_init[row], stored, jnp.zeros_like(stored))
        # One materialised incoming state. Left alone, XLA re-derives it (pool slot read included) inside each of its
        # consumers (the chunk dots, the new state, the output window); a consumer scheduled after this step's pool
        # write then keeps the old pool alive, and XLA copies the whole pool (twice per step, seen in serving).
        state = lax.optimization_barrier(jnp.where(local == 0, stored, state))

        g_cumsum = jnp.cumsum(g, axis=1)
        state, chunk_out = core(state, q, k, kb, vb, g_cumsum, g_cumsum[:, -1:, :])

        merged = jnp.where(vmask, jnp.swapaxes(chunk_out, 0, 1), window(out, win_start))
        out = lax.dynamic_update_slice_in_dim(out, merged, win_start, axis=0)
        # Every step stores the running state (a later chunk of the row overwrites it; the row's first chunk read its
        # slot above), so the write needs no read of the slot's old value. Rows without chunks are never written.
        pool = lax.dynamic_update_slice_in_dim(pool, state.astype(pool.dtype)[None], write_slot[row], axis=0)
        return pool, out, state

    state0 = jnp.zeros((num_heads, head_dim, v_dim), f32)
    pool, out, _ = lax.fori_loop(0, chunk_end[-1], body, (pool, out, state0))
    return out, pool


def conv_and_short_rows(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    weights: tuple[Array, Array, Array] | None,
    pools: tuple[Array | None, Array | None, Array | None, Array],
    query_start_loc: Array,
    plan: RowPlan,
    *,
    short_row_max: int,
    use_qk_l2norm: bool,
    step_fn: StepFn | None = None,
) -> tuple[Array, tuple[Array | None, Array | None, Array | None, Array], tuple[Array, Array, Array]]:
    """Everything but the chunk recurrence: the flat conv, the short rows and the chunk rows' final conv windows.

    Args:
        query: Queries ``[T, H, K]`` (raw with ``weights``, post-conv without).
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        weights: ``(q, k, v)`` conv taps, or ``None`` for post-conv inputs.
        pools: ``(q_conv_state, k_conv_state, v_conv_state, recurrent_state)`` (conv pools ``None`` without weights).
        query_start_loc: ``[R + 1]``.
        plan: The step's :class:`RowPlan`.
        short_row_max: ``W``.
        use_qk_l2norm: L2-normalise queries and keys.
        step_fn: Per-token recurrence of the short rows (see :func:`short_phase`; default :func:`xla_step`).

    Returns:
        ``(out [T, H, V] fp32 with the short rows' outputs, pools with the short rows' captures and the chunk rows'
        conv windows written, post-conv (q, k, v) of the flat stream [T, H, D])``.
    """
    num_tokens, num_heads, head_dim = query.shape
    v_dim = value.shape[-1]
    out = jnp.zeros((num_tokens, num_heads, v_dim), jnp.float32)
    common = dict(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        weights=weights,
        plan=plan,
        short_row_max=short_row_max,
        use_qk_l2norm=use_qk_l2norm,
        step_fn=step_fn,
    )
    if weights is None:
        out, pools = short_phase(out, pools, **common)
        return out, pools, (query, key, value)
    q_pool, k_pool, v_pool, rec = pools
    num_slots = rec.shape[0]
    streams = (
        query.reshape(num_tokens, num_heads * head_dim),
        key.reshape(num_tokens, num_heads * head_dim),
        value.reshape(num_tokens, num_heads * v_dim),
    )
    # The flat conv reads every row's windows before any write of this step.
    convs = [
        flat_conv(s, w, read_rows(p, plan), query_start_loc, plan.init)
        for s, w, p in zip(streams, weights, (q_pool, k_pool, v_pool), strict=True)
    ]
    out, (q_pool, k_pool, v_pool, rec) = short_phase(out, (q_pool, k_pool, v_pool, rec), **common)
    if plan.identity:
        q_pool, k_pool, v_pool = (
            put_rows(p, jnp.where(plan.chunk[:, None, None], c[1], read_rows(p, plan)))
            for p, c in zip((q_pool, k_pool, v_pool), convs, strict=True)
        )
    else:
        dest = jnp.where(plan.chunk, plan.base_slot, num_slots)
        q_pool, k_pool, v_pool = (
            p.at[dest].set(c[1], mode="drop") for p, c in zip((q_pool, k_pool, v_pool), convs, strict=True)
        )
    post = (
        convs[0][0].reshape(num_tokens, num_heads, head_dim),
        convs[1][0].reshape(num_tokens, num_heads, head_dim),
        convs[2][0].reshape(num_tokens, num_heads, v_dim),
    )
    return out, (q_pool, k_pool, v_pool, rec), post


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
    read_offset: Array | None = None,
    commit_lens: Array | None = None,
    q_conv_weight: Array | None = None,
    k_conv_weight: Array | None = None,
    v_conv_weight: Array | None = None,
    q_conv_state: Array | None = None,
    k_conv_state: Array | None = None,
    v_conv_state: Array | None = None,
    *,
    short_row_max: int = 1,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: PrecisionLike = "highest",
    inverse: str = "auto",
    pairwise: str = "auto",
) -> tuple[Array, ...]:
    """Ragged per-channel KDA with the causal conv fused and per-token state capture (XLA reference).

    See the module docstring for the row contract.

    Args:
        query: Queries ``[T, H, K]``: raw (pre-conv) with the conv arrays, post-conv without them.
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid already applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``).
        recurrent_state: State pool ``[S, H, K, V]``; computed in fp32, stored in its own dtype.
        query_start_loc: Cumulative row offsets ``[R + 1]`` (nondecreasing).
        state_indices: Base slot ``g`` of every row ``[R]``, or ``None`` (``g = r``).
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from zero windows and a zero state.
        read_offset: Optional ``[R]`` read offsets ``a`` (zero when ``None``).
        commit_lens: Optional ``[R]`` commit lengths ``c`` (``lens`` when ``None``).
        q_conv_weight: Depthwise causal conv taps ``[H * K, d_conv]`` (the last tap is the current token), or ``None``.
        k_conv_weight: ``[H * K, d_conv]`` or ``None``.
        v_conv_weight: ``[H * V, d_conv]`` or ``None``.
        q_conv_state: Conv-window pool ``[S, H * K, d_conv]`` (oldest tap first), or ``None``.
        k_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        v_conv_state: ``[S, H * V, d_conv]`` or ``None``.
        short_row_max: ``W``: rows of at most ``W`` tokens take the short-row path and capture per token.
        chunk_size: Chunk length of the chunk path.
        use_qk_l2norm: L2-normalise queries and keys.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound``; with ``pairwise="auto"``
            it selects the factored MXU pairwise decay of the chunk math. ``None`` keeps the dense form.
        precision: Precision of the non-inverse chunk dots.
        inverse: ``"auto"`` (see :func:`resolve_inverse`), ``"masked"`` or ``"solve"``.
        pairwise: ``"auto"`` or ``"dense"``.

    Returns:
        With the conv arrays ``(out [T, H, V] fp32, q_conv_state, k_conv_state, v_conv_state, recurrent_state)``,
        without them ``(out, recurrent_state)``. The pools keep their dtypes; slots no live row writes are unchanged.
    """
    num_tokens = query.shape[0]
    cs = int(chunk_size)
    num_slots = recurrent_state.shape[0]
    fused = q_conv_weight is not None
    plan = plan_rows(
        query_start_loc,
        state_indices,
        has_initial_state,
        read_offset,
        commit_lens,
        num_slots=num_slots,
        short_row_max=int(short_row_max),
    )
    out, pools, (q, k, v) = conv_and_short_rows(
        query,
        key,
        value,
        beta,
        decay,
        (q_conv_weight, k_conv_weight, v_conv_weight) if fused else None,
        (q_conv_state, k_conv_state, v_conv_state, recurrent_state),
        query_start_loc,
        plan,
        short_row_max=int(short_row_max),
        use_qk_l2norm=use_qk_l2norm,
    )
    sub = pairwise_sub_block(decay_lower_bound, cs) if pairwise == "auto" else 0
    pad = max(cs - num_tokens, 0)  # a chunk window must fit in the buffer
    if pad:
        q, k, v, decay = (jnp.pad(x, ((0, pad), (0, 0), (0, 0))) for x in (q, k, v, decay))
        beta = jnp.pad(beta, ((0, pad), (0, 0)))
        out = jnp.pad(out, ((0, pad), (0, 0), (0, 0)))
    # Pure decode steps run zero iterations of the chunk loop.
    out, rec = chunk_phase(
        out,
        pools[3],
        query=q,
        key=k,
        value=v,
        beta=beta,
        decay=decay,
        query_start_loc=query_start_loc.astype(jnp.int32),
        lens=jnp.where(plan.chunk, plan.lens, 0),
        read_slot=plan.read_slot,
        write_slot=plan.base_slot,
        has_init=plan.init,
        chunk_size=cs,
        use_qk_l2norm=use_qk_l2norm,
        precision=resolve_precision(precision),
        inverse=resolve_inverse(inverse, cs),
        sub=sub,
    )
    out = out[:num_tokens]
    if fused:
        return out, pools[0], pools[1], pools[2], rec
    return out, rec


__all__ = (
    "CONV_ARRAYS",
    "RowPlan",
    "StepFn",
    "chunk_phase",
    "conv_and_short_rows",
    "conv_compute_dtype",
    "conv_output_dtype",
    "conv_step",
    "decode_step",
    "flat_conv",
    "is_lowfloat",
    "plan_rows",
    "put_rows",
    "ragged_kda_xla",
    "read_rows",
    "resolve_inverse",
    "short_phase",
    "validate_options",
    "validate_shapes",
    "xla_step",
)
