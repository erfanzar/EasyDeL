# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""Decode-only eSurge KDA driver and the EasyDeL adapter for eJKernel's ``ragged_kda_decode``.

:class:`RaggedKDADecodeOp` (registered as ``"ragged_kda_decode"``) advances every row that carries one token by one
per-channel KDA step over the recurrent-state pool. On TPU (``forward_tpu``) the eJKernel Pallas kernel reads and
writes each row's pool slot once, in place; ``forward_native`` (also what ``FORCE_NATIVE_RUNTIME=1`` selects) runs the
XLA reference.

:func:`kda_decode_rows_forward` is the eSurge driver for steps in which every row has at most one token: each short
causal convolution becomes a window shift plus a ``d_conv``-tap dot product per row (SiLU fused), and the recurrence
runs in :class:`RaggedKDADecodeOp`. :func:`kda_rows_forward` takes it at run time (``lax.cond``) when a step whose
token bucket holds at most one token per row is decode-only, and otherwise runs the flat-stream driver
:func:`~easydel.operations.kernels.kda_ragged_rows_forward` unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from ejkernel.modules.operations import ragged_kda_decode_head_sharded  # pyright: ignore[reportMissingTypeStubs]
from jax import lax
from jaxtyping import Array, Float
from spectrax import common_types

from easydel.caching import KDACacheView
from easydel.infra.sharding import mesh_axis_size

from .._operation_impl import OperationImpl, OperationRegistry
from .._operation_meta import OperationMetadata
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)
from .ragged_kda import _conv_compute_dtype, _is_lowfloat, kda_ragged_rows_forward

LEAF_NAMES = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")


def _head_axis(metadata: OperationMetadata, num_heads: int, num_tokens: int) -> str | tuple[str, ...] | None:
    """Mesh axis the heads are split over (same rule as :meth:`RaggedKDAOp.resolve_head_axis`), or ``None``."""
    mesh = metadata.mesh
    if mesh is None:
        return None
    mode = common_types.MODE_DECODE if int(num_tokens) == 1 else common_types.MODE_TRAIN
    query_spec = metadata.get_shardings(mode, layout="bthd").query
    if query_spec is None or len(query_spec) < 3 or query_spec[2] is None:
        return None
    head_axis = query_spec[2]
    shards = mesh_axis_size(mesh, head_axis)
    if shards <= 1 or int(num_heads) % shards:
        return None
    return head_axis


@OperationRegistry.register
class RaggedKDADecodeOp(OperationImpl):
    """One per-channel-decay KDA decode token per row over a recurrent-state pool.

    Inputs are post-convolution, one token per row: ``query`` / ``key`` ``[R, H, K]``, ``value`` ``[R, H, V]``,
    ``beta`` ``[R, H]`` (sigmoid applied), the per-channel log-decay ``decay`` ``[R, H, K]`` and the state pool
    ``[S, H, K, V]``. Row ``r`` owns pool slot ``state_indices[r]`` (slot ``r`` when ``state_indices`` is ``None``);
    rows flagged ``valid`` advance, every other row outputs zeros and every other slot keeps its state bit for bit.
    When the head axis of the metadata's mesh spans several devices and divides the head count, every device advances
    its own heads under ``shard_map`` (no collectives). ``metadata.get_operation_config("ragged_kda_decode")`` may
    supply an ``ejkernel.modules.operations.RaggedKDADecodeConfig`` (kernel tiling, platform).
    """

    @classmethod
    def get_impl_name(cls) -> str:
        """Return the registry key.

        Returns:
            ``"ragged_kda_decode"``.
        """
        return "ragged_kda_decode"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Declare the packed-serving metadata and the KDA cache view this operation reads.

        Args:
            mode: Execution mode; the requirements are the same for every mode.

        Returns:
            Requirements on ``SEQ_LENS``, ``POSITIONS``, ``HAS_INITIAL_STATE`` and ``STATE_INDICES`` metadata, with
            recurrent / hybrid caches whose layers are :class:`~easydel.caching.KDACacheView`.
        """
        return (
            RequirementsBuilder("ragged_kda_decode")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.POSITIONS
                | MetadataField.HAS_INITIAL_STATE
                | MetadataField.STATE_INDICES
            )
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .use_cache_view(KDACacheView)
            .build()
        )

    def resolve_head_axis(self, num_heads: int, num_tokens: int) -> str | tuple[str, ...] | None:
        """Return the mesh axis the heads are split over, or ``None`` for an unsharded call.

        Args:
            num_heads: Global head count.
            num_tokens: Length of the step's packed token stream (selects the decode or train sharding rules, as in
                :meth:`~easydel.operations.kernels.RaggedKDAOp.resolve_head_axis`).

        Returns:
            A mesh axis name (or tuple of names), or ``None``.
        """
        return _head_axis(self.get_impl_metadata(), num_heads, num_tokens)

    def _run(
        self,
        query: Array,
        key: Array,
        value: Array,
        beta: Array,
        decay: Array,
        recurrent_state: Array,
        state_indices: Array | None,
        valid: Array,
        has_initial_state: Array | None,
        *,
        num_tokens: int | None,
        use_qk_l2norm: bool,
        platform: str | None,
        head_axis: str | tuple[str, ...] | None = None,
    ) -> tuple[Array, Array]:
        metadata = self.get_impl_metadata()
        tokens = int(query.shape[0]) if num_tokens is None else int(num_tokens)
        return ragged_kda_decode_head_sharded(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            state_indices,
            valid,
            has_initial_state,
            mesh=metadata.mesh,
            head_axis=self.resolve_head_axis(int(query.shape[1]), tokens) if head_axis is None else head_axis,
            use_qk_l2norm=bool(use_qk_l2norm),
            platform=platform,
            cfg=metadata.get_operation_config("ragged_kda_decode"),
        )

    @jax.named_scope("easydel-ragged-kda-decode")
    def forward_native(
        self,
        query: Float[Array, "num_rows num_heads head_dim"],
        key: Float[Array, "num_rows num_heads head_dim"],
        value: Float[Array, "num_rows num_heads v_dim"],
        beta: Float[Array, "num_rows num_heads"],
        decay: Float[Array, "num_rows num_heads head_dim"],
        recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
        state_indices: Array | None,
        valid: Array,
        has_initial_state: Array | None = None,
        *,
        num_tokens: int | None = None,
        use_qk_l2norm: bool = True,
        head_axis: str | tuple[str, ...] | None = None,
    ) -> tuple[Float[Array, "num_rows num_heads v_dim"], Float[Array, "num_slots num_heads head_dim v_dim"]]:
        """Run the XLA reference of ``ejkernel.modules.operations.ragged_kda_decode`` (head-sharded).

        Args:
            query: Post-conv queries ``[R, H, K]``.
            key: Post-conv keys ``[R, H, K]``.
            value: Post-conv values ``[R, H, V]``.
            beta: Write strengths ``[R, H]`` (sigmoid applied).
            decay: Per-channel log-decay ``[R, H, K]`` (``<= 0``).
            recurrent_state: State pool ``[S, H, K, V]``; the step runs in fp32, the pool keeps its dtype.
            state_indices: Pool slot of every row ``[R]`` (out of range: the row does not advance), or ``None`` (row
                ``r`` owns slot ``r``).
            valid: ``[R]`` rows that advance.
            has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state.
            num_tokens: Token count of the step's packed stream (head-axis rule); defaults to ``R``.
            use_qk_l2norm: L2-normalise queries and keys first.
            head_axis: Mesh axis (or tuple of axes) to split the heads over, or ``None`` for
                :meth:`resolve_head_axis`.

        Returns:
            ``(out [R, H, V] float32, new_recurrent_state [S, H, K, V])``.
        """
        return self._run(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            state_indices,
            valid,
            has_initial_state,
            num_tokens=num_tokens,
            use_qk_l2norm=use_qk_l2norm,
            platform="xla",
            head_axis=head_axis,
        )

    @jax.named_scope("easydel-ragged-kda-decode")
    def forward_tpu(self, *args, **kwargs) -> tuple[Array, Array]:
        """TPU path: the operation config's platform (default ``"auto"``: the in-place Pallas TPU kernel).

        Args:
            *args: Same as :meth:`forward_native`.
            **kwargs: Same as :meth:`forward_native`.

        Returns:
            ``(out [R, H, V] float32, new_recurrent_state [S, H, K, V])``.
        """
        num_tokens = kwargs.pop("num_tokens", None)
        use_qk_l2norm = kwargs.pop("use_qk_l2norm", True)
        return self._run(*args, **kwargs, num_tokens=num_tokens, use_qk_l2norm=use_qk_l2norm, platform=None)


def kda_decode_rows_forward(
    query: Float[Array, "1 seq_len key_dim"],
    key: Float[Array, "1 seq_len key_dim"],
    value: Float[Array, "1 seq_len value_dim"],
    beta: Float[Array, "1 seq_len num_heads"],
    decay: Float[Array, "1 seq_len num_heads head_dim"],
    conv_kernels: tuple[Array, Array, Array],
    cache_view: KDACacheView,
    query_start_loc: Array,
    context_lens: Array | None = None,
    *,
    metadata: OperationMetadata,
    conv_dtype: jnp.dtype = jnp.bfloat16,
    state_indices: Array | None = None,
    head_axis: str | tuple[str, ...] | None = None,
) -> tuple[Float[Array, "1 seq_len num_heads d_state"], KDACacheView]:
    """Advance KDA state for an eSurge step in which every row has at most one token.

    Same contract as :func:`~easydel.operations.kernels.kda_ragged_rows_forward` restricted to such steps (callers
    route through :func:`kda_rows_forward`; a row with more tokens is left untouched and outputs zeros here). Row
    ``r``'s token sits at ``query_start_loc[r]``; it owns pool slot ``state_indices[r]`` (slot ``r`` without
    ``state_indices``: the conv windows and recurrent states of the leading rows are then updated in place). For each of
    q, k and v the row's carried window is shifted by the new raw token and dotted with the depthwise kernel (fp32,
    SiLU fused; the conv meets stream and windows in their common dtype, float32 when either is an 8- or 4-bit float,
    and windows keep the pool's dtype); the recurrence runs in :class:`RaggedKDADecodeOp`. A row that starts a request
    (``context_lens <= lens``) starts from a zero window and a zero state. Rows without a token (or with an out-of-range
    slot) write nothing; tokens at or past ``query_start_loc[-1]`` are bucket padding (zero output).

    Args:
        query: Packed Q projection (pre-conv) ``[1, T, key_dim]``.
        key: Packed K projection (pre-conv) ``[1, T, key_dim]``.
        value: Packed V projection (pre-conv) ``[1, T, value_dim]``.
        beta: Packed write strength ``[1, T, heads]`` (sigmoid applied).
        decay: Packed per-channel log-decay ``[1, T, heads, head_dim]``.
        conv_kernels: Depthwise causal conv kernels for q, k, v, each ``[dim, d_conv]`` (last tap = current token).
        cache_view: Per-layer KDA cache view; without ``state_indices`` its pool holds at least one slot per row.
        query_start_loc: Cumulative row offsets ``[rows + 1]``.
        context_lens: Tokens per row including this step, or ``None`` (single-token rows then continue).
        metadata: Operation metadata (mesh, sharding policy, optional ``"ragged_kda_decode"`` config).
        conv_dtype: Dtype of the conv outputs fed to the recurrence.
        state_indices: Pool slot of every row, ``[rows]`` or ``[1, rows]``, or ``None`` for slot ``r``.
        head_axis: Mesh axis (or tuple of axes) of the heads for the decode operation, or ``None`` for
            :meth:`RaggedKDADecodeOp.resolve_head_axis` (see :func:`~easydel.operations.kernels.kda_ragged_rows_forward`).

    Returns:
        tuple: ``(outputs, new_cache_view)``: outputs ``[1, T, heads, head_v_dim]`` float32 and the cache view with
        updated conv windows and recurrent states.

    Raises:
        ValueError: For 2-D ``query_start_loc``, a stream without rows, ``state_indices`` with fewer entries than rows,
            or (without ``state_indices``) a pool with fewer slots than rows.
        TypeError: If the registry resolves ``"ragged_kda_decode"`` to something other than a
            :class:`RaggedKDADecodeOp`.
    """
    if query_start_loc.ndim != 1:
        raise ValueError(f"kda_decode_rows_forward requires 1-D query_start_loc, got {tuple(query_start_loc.shape)}")
    rows = int(query_start_loc.shape[0]) - 1
    if rows < 1:
        raise ValueError(f"kda_decode_rows_forward needs at least one row, got query_start_loc {query_start_loc.shape}")
    num_slots = int(cache_view.recurrent_state.shape[0])
    if state_indices is None:
        if num_slots < rows:
            raise ValueError(
                f"kda_decode_rows_forward: without state_indices row r owns pool slot r, so the pool needs at least "
                f"{rows} slots, got {num_slots}"
            )
    else:
        state_indices = jnp.asarray(state_indices, dtype=jnp.int32).reshape(-1)
        if state_indices.shape[0] < rows:
            raise ValueError(
                f"kda_decode_rows_forward: state_indices needs one pool slot per row ({rows}), got shape "
                f"{tuple(state_indices.shape)}"
            )
        state_indices = state_indices[:rows]
    op = OperationRegistry.create("ragged_kda_decode", metadata)
    if not isinstance(op, RaggedKDADecodeOp):
        raise TypeError(f"'ragged_kda_decode' resolved to {type(op).__name__}, expected a RaggedKDADecodeOp")

    with jax.named_scope("easydel-kda-decode-rows"):
        f32 = jnp.float32
        qsl = jnp.asarray(query_start_loc, dtype=jnp.int32)
        seq_len = int(query.shape[1])
        num_heads = int(beta.shape[-1])
        head_k = int(query.shape[-1]) // num_heads
        head_v = int(value.shape[-1]) // num_heads
        lens = qsl[1:] - qsl[:-1]
        if context_lens is None:
            has_prefix = lens <= 1
        else:
            has_prefix = jnp.asarray(context_lens, dtype=jnp.int32)[:rows] > lens
        fresh = (lens > 0) & jnp.logical_not(has_prefix)
        advance = lens == 1
        if state_indices is not None:
            advance = advance & (state_indices >= 0) & (state_indices < num_slots)
            read_slots = jnp.clip(state_indices, 0, num_slots - 1)
            # Rows that do not advance (no token, out-of-range slot; eSurge points padding rows at slot 0) drop the write.
            write_slots = jnp.where(advance, state_indices, num_slots)
        tok = jnp.clip(qsl[:-1], 0, seq_len - 1)  # row r's token (rows without one read a clamped, unused token)

        def causal_conv(stream: Array, kernel: Array, pool_leaf: Array) -> tuple[Array, Array]:
            compute = _conv_compute_dtype(stream.dtype, pool_leaf.dtype)
            held = pool_leaf[:rows] if state_indices is None else pool_leaf[read_slots]
            window = jnp.where(fresh[:, None, None], jnp.zeros((), compute), held.astype(compute))
            x = stream[0][tok].astype(compute)
            new_window = jnp.concatenate([window[:, :, 1:], x[:, :, None]], axis=-1)  # [rows, dim, d_conv]
            taps = new_window.astype(f32)
            weights = kernel.astype(f32)
            mixed = taps[:, :, 0] * weights[None, :, 0]
            for j in range(1, int(kernel.shape[-1])):
                mixed = mixed + taps[:, :, j] * weights[None, :, j]
            out = jax.nn.silu(mixed).astype(compute).astype(conv_dtype)
            new_window = new_window.astype(pool_leaf.dtype)
            if state_indices is not None:
                return out, pool_leaf.at[write_slots].set(new_window, mode="drop")
            kept = jnp.where(advance[:, None, None], new_window, held)  # rows that do not advance keep their window
            if pool_leaf.shape[0] == rows:
                return out, kept
            return out, lax.dynamic_update_slice_in_dim(pool_leaf, kept, 0, axis=0)

        q_conv, q_pool = causal_conv(query, conv_kernels[0], cache_view.q_conv_state)
        k_conv, k_pool = causal_conv(key, conv_kernels[1], cache_view.k_conv_state)
        v_conv, v_pool = causal_conv(value, conv_kernels[2], cache_view.v_conv_state)
        row_beta = beta[0][tok]
        row_out, recurrent_pool = op(
            q_conv.reshape(rows, num_heads, head_k),
            k_conv.reshape(rows, num_heads, head_k),
            v_conv.reshape(rows, num_heads, head_v),
            row_beta.astype(f32) if _is_lowfloat(row_beta.dtype) else row_beta,
            decay[0][tok],
            cache_view.recurrent_state,
            state_indices,  # None: row r owns slot r (in-place kernel update)
            advance,
            jnp.logical_not(fresh),
            num_tokens=seq_len,
            head_axis=head_axis,
        )
        out = jnp.zeros((seq_len, num_heads, head_v), f32)
        out = out.at[jnp.where(advance, tok, seq_len)].set(row_out.astype(f32), mode="drop")
    new_cache_view = cache_view.update_kda_states(
        new_q_conv_state=q_pool,
        new_k_conv_state=k_pool,
        new_v_conv_state=v_pool,
        new_recurrent_state=recurrent_pool,
    )
    return out[None], new_cache_view


def kda_rows_forward(
    query: Float[Array, "1 seq_len key_dim"],
    key: Float[Array, "1 seq_len key_dim"],
    value: Float[Array, "1 seq_len value_dim"],
    beta: Float[Array, "1 seq_len num_heads"],
    decay: Float[Array, "1 seq_len num_heads head_dim"],
    conv_kernels: tuple[Array, Array, Array],
    cache_view: KDACacheView,
    query_start_loc: Array,
    context_lens: Array | None = None,
    *,
    metadata: OperationMetadata,
    conv_dtype: jnp.dtype = jnp.bfloat16,
    chunk_size: int = 64,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    state_indices: Array | None = None,
    flat_forward: Callable[..., tuple[Array, KDACacheView]] = kda_ragged_rows_forward,
    head_axis: str | tuple[str, ...] | None = None,
) -> tuple[Float[Array, "1 seq_len num_heads d_state"], KDACacheView]:
    """eSurge KDA serving driver: :func:`kda_decode_rows_forward` on decode-only steps, else ``flat_forward``.

    For token buckets of at most ``rows`` tokens the choice is made at run time with ``lax.cond`` on "every row has at
    most one token", so one compiled step serves decode, mixed and prefill batches; the pool leaves are the
    conditional's operands (updated in place in both branches). Mixed and prefill steps run ``flat_forward`` (default
    :func:`~easydel.operations.kernels.kda_ragged_rows_forward`) unchanged. Longer buckets compile ``flat_forward``
    alone, without the conditional (a decode-only step fits a bucket of ``rows`` tokens). Arguments and results follow
    :func:`~easydel.operations.kernels.kda_ragged_rows_forward`.

    Args:
        query: Packed Q projection (pre-conv) ``[1, T, key_dim]``.
        key: Packed K projection (pre-conv) ``[1, T, key_dim]``.
        value: Packed V projection (pre-conv) ``[1, T, value_dim]``.
        beta: Packed write strength ``[1, T, heads]`` (sigmoid applied).
        decay: Packed per-channel log-decay ``[1, T, heads, head_dim]``.
        conv_kernels: Depthwise causal conv kernels for q, k, v, each ``[dim, d_conv]``.
        cache_view: Per-layer KDA cache view.
        query_start_loc: Cumulative row offsets ``[rows + 1]``.
        context_lens: Tokens per row including this step, or ``None``.
        metadata: Operation metadata.
        conv_dtype: Dtype of the conv outputs fed to the recurrence.
        chunk_size: Chunk length of the multi-token recurrence (flat driver).
        decay_lower_bound: Promised lower bound of every log-decay, or ``None`` (flat driver).
        precision: ``"highest"`` or ``"default"`` for the flat driver's chunk dots.
        state_indices: Pool slot of every row, ``[rows]`` or ``[1, rows]``, or ``None`` for slot ``r``.
        flat_forward: Driver for steps with multi-token rows (same signature as ``kda_ragged_rows_forward``).
        head_axis: Mesh axis (or tuple of axes) of the heads, passed to both drivers when given (``None``: each
            operation resolves its own).

    Returns:
        tuple: ``(outputs [1, T, heads, head_v_dim] float32, new_cache_view)``.
    """
    common = dict(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        conv_kernels=conv_kernels,
        query_start_loc=query_start_loc,
        context_lens=context_lens,
        metadata=metadata,
        conv_dtype=conv_dtype,
        state_indices=state_indices,
    )
    if head_axis is not None:
        common["head_axis"] = head_axis
    flat_kwargs = dict(chunk_size=chunk_size, decay_lower_bound=decay_lower_bound, precision=precision)
    leaves = tuple(getattr(cache_view, name) for name in LEAF_NAMES)
    rows = int(query_start_loc.shape[-1]) - 1
    if (
        query_start_loc.ndim != 1
        or int(query.shape[1]) > rows
        or not all(isinstance(leaf, jax.Array) for leaf in leaves)
    ):
        # Rank-major DP metadata (the flat driver raises its own error), implicit (quantized) pool leaves, or a token
        # bucket longer than the row count. A decode-only step (at most one token per row) fits a bucket of ``rows``
        # tokens, so longer buckets compile the flat driver alone, as before (it is correct for every step).
        return flat_forward(cache_view=cache_view, **common, **flat_kwargs)
    qsl = jnp.asarray(query_start_loc, dtype=jnp.int32)
    decode_only = jnp.all(qsl[1:] - qsl[:-1] <= 1)

    def branch(forward: Callable[..., tuple[Array, KDACacheView]], **extra):
        def run(pool_leaves):
            out, new_view = forward(cache_view=cache_view.update_kda_states(*pool_leaves), **common, **extra)
            return out, tuple(getattr(new_view, name) for name in LEAF_NAMES)

        return run

    out, new_leaves = lax.cond(
        decode_only,
        branch(kda_decode_rows_forward),
        branch(flat_forward, **flat_kwargs),
        leaves,
    )
    return out, cache_view.update_kda_states(*new_leaves)


__all__ = ("RaggedKDADecodeOp", "kda_decode_rows_forward", "kda_rows_forward")
