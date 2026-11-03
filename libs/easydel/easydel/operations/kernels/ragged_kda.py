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
"""EasyDeL adapter for eJKernel ragged per-channel KDA, and the flat-stream eSurge KDA serving driver.

:class:`RaggedKDAOp` (registered as ``"ragged_kda"``) is the serving form of the per-channel-decay KDA recurrence used
by GLM-5-Next and Kimi Linear. The tokens of every scheduled request share one flat buffer ``[T, H, D]``,
``query_start_loc`` delimits the requests, and each request's ``[H, K, V]`` state lives in a pool slot. The recurrence
lives in eJKernel (``ejkernel.modules.operations.ragged_kda``); this adapter declares the EasyDeL requirements, takes the
mesh, head sharding and operation config from :class:`~easydel.operations.OperationMetadata`, and delegates.

:func:`kda_ragged_rows_forward` is the eSurge driver built on it. The three causal depthwise convolutions run through
eJKernel's ragged conv1d on the flat stream (as in Qwen3-Next serving) and the recurrence runs through
:class:`RaggedKDAOp`, so the work follows the scheduled token count. The row-padded driver
:func:`~easydel.operations.kernels.kda.kda_packed_rows_forward` copies every request into a bucket-wide row
(``rows * T`` token slots); it remains the path for pools that carry speculative candidate rows.
"""

import jax
import jax.numpy as jnp
from ejkernel.modules.operations import ragged_kda_head_sharded  # pyright: ignore[reportMissingTypeStubs]
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
from .inference_conv1d import ragged_causal_conv1d_head_sharded


def _is_lowfloat(dtype: jnp.dtype) -> bool:
    """Whether ``dtype`` is an 8- or 4-bit float (one byte per element), which JAX never promotes implicitly."""
    dtype = jnp.dtype(dtype)
    return bool(jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize <= 1)


def _conv_compute_dtype(stream_dtype: jnp.dtype, pool_dtype: jnp.dtype) -> jnp.dtype:
    """Dtype in which the ragged conv meets the token stream and the carried windows.

    The conv concatenates carried windows with tokens, and JAX does not promote 8-bit or 4-bit floats implicitly; an
    fp8 stream would also round the SiLU output to fp8. Any such low-precision float therefore computes in float32;
    other pairs promote as usual (bf16 with bf16 stays bf16).

    Args:
        stream_dtype: Dtype of the packed projections.
        pool_dtype: Dtype of the conv-window pool.

    Returns:
        The common compute dtype.
    """
    if _is_lowfloat(stream_dtype) or _is_lowfloat(pool_dtype):
        return jnp.dtype(jnp.float32)
    return jnp.promote_types(stream_dtype, pool_dtype)


@OperationRegistry.register
class RaggedKDAOp(OperationImpl):
    """Ragged per-channel-decay KDA over a flat packed token stream and a recurrent-state pool.

    Inputs are post-convolution: ``query`` / ``key`` ``[T, H, K]``, ``value`` ``[T, H, V]``, ``beta`` ``[T, H]``
    (sigmoid applied), the per-channel log-decay ``decay`` ``[T, H, K]`` and the state pool ``[S, H, K, V]``. Row
    ``r`` owns tokens ``[query_start_loc[r], query_start_loc[r + 1])`` and pool slot ``state_indices[r]``; tokens at or
    past ``query_start_loc[R]`` are padding (zero output, no state change) and slots owned by no row with tokens keep
    their state. When the head axis of the metadata's mesh spans several devices and divides the head count, every
    device advances its own heads under ``shard_map`` (no collectives); otherwise the call is unsharded.

    ``metadata.get_operation_config("ragged_kda")`` may supply an ``ejkernel.modules.operations.RaggedKDAConfig``
    (chunk inverse, pairwise form, platform).
    """

    @classmethod
    def get_impl_name(cls) -> str:
        """Return the registry key.

        Returns:
            ``"ragged_kda"``.
        """
        return "ragged_kda"

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
            RequirementsBuilder("ragged_kda")
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

        The axis is the ``HEAD`` entry of the metadata's query sharding (decode rules for a single token). It is used
        only when it spans more than one device and divides ``num_heads``.

        Args:
            num_heads: Global head count.
            num_tokens: Length of the packed token stream (one token selects the decode sharding rules).

        Returns:
            A mesh axis name (or tuple of names), or ``None``.
        """
        metadata = self.get_impl_metadata()
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

    @jax.named_scope("easydel-ragged-kda")
    def forward_native(
        self,
        query: Float[Array, "num_tokens num_heads head_dim"],
        key: Float[Array, "num_tokens num_heads head_dim"],
        value: Float[Array, "num_tokens num_heads v_dim"],
        beta: Float[Array, "num_tokens num_heads"],
        decay: Float[Array, "num_tokens num_heads head_dim"],
        recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
        query_start_loc: Array,
        state_indices: Array | None,
        has_initial_state: Array | None = None,
        *,
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        decay_lower_bound: float | None = None,
        precision: str = "highest",
        head_axis: str | tuple[str, ...] | None = None,
    ) -> tuple[Float[Array, "num_tokens num_heads v_dim"], Float[Array, "num_slots num_heads head_dim v_dim"]]:
        """Run ragged KDA through ``ejkernel.modules.operations.ragged_kda_head_sharded``.

        Args:
            query: Post-conv queries ``[T, H, K]``.
            key: Post-conv keys ``[T, H, K]``.
            value: Post-conv values ``[T, H, V]``.
            beta: Write strengths ``[T, H]`` (sigmoid applied).
            decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``).
            recurrent_state: State pool ``[S, H, K, V]``; the recurrence runs in fp32.
            query_start_loc: Cumulative row offsets ``[R + 1]`` (nondecreasing).
            state_indices: Pool slot of every row ``[R]``; an out-of-range slot marks a padding row. ``None``
                is the identity layout (row ``r`` owns slot ``r``): decode rows update their slots in place.
            has_initial_state: Optional ``[R]`` flags; ``False`` rows start from a zero state instead of their slot.
            chunk_size: Chunk length of the multi-token path.
            use_qk_l2norm: L2-normalise queries and keys first.
            decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound`` (a safe-gate bound selects
                the factored pairwise decay), or ``None``.
            precision: ``"highest"`` or ``"default"`` for the non-inverse chunk dots.
            head_axis: Mesh axis (or tuple of axes) to split the heads over, or ``None`` for
                :meth:`resolve_head_axis` (eSurge KDA steps with ``kda_serving_ep_split`` pass the cache's head axes).

        Returns:
            ``(out [T, H, V] float32, new_recurrent_state [S, H, K, V])``; the pool keeps its dtype.
        """
        return self._forward(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            has_initial_state,
            chunk_size=chunk_size,
            use_qk_l2norm=use_qk_l2norm,
            decay_lower_bound=decay_lower_bound,
            precision=precision,
            head_axis=head_axis,
            platform="xla",
        )

    @jax.named_scope("easydel-ragged-kda")
    def forward_tpu(self, *args, **kwargs) -> tuple[Array, Array]:
        """TPU path: the operation config's platform (default ``"auto"``: the Pallas TPU ``ragged_kda`` kernel).

        The Pallas entry runs the multi-token chunk path in one kernel and falls back to the XLA reference itself for
        options it does not lower (no usable ``decay_lower_bound``, head sizes that are not multiples of 128).

        Args:
            *args: Same as :meth:`forward_native`.
            **kwargs: Same as :meth:`forward_native`.

        Returns:
            ``(out [T, H, V] float32, new_recurrent_state [S, H, K, V])``.
        """
        return self._forward(*args, **kwargs, platform=None)

    def _forward(
        self,
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
        precision: str = "highest",
        head_axis: str | tuple[str, ...] | None = None,
        platform: str | None = None,
    ) -> tuple[Array, Array]:
        """Call ``ejkernel.modules.operations.ragged_kda_head_sharded`` on one platform.

        Args:
            query: Post-conv queries ``[T, H, K]``.
            key: Post-conv keys ``[T, H, K]``.
            value: Post-conv values ``[T, H, V]``.
            beta: Write strengths ``[T, H]``.
            decay: Per-channel log-decay ``[T, H, K]``.
            recurrent_state: State pool ``[S, H, K, V]``.
            query_start_loc: Cumulative row offsets ``[R + 1]``.
            state_indices: Pool slot of every row ``[R]``, or ``None`` (row ``r`` owns slot ``r``).
            has_initial_state: Optional ``[R]`` flags.
            chunk_size: Chunk length of the multi-token path.
            use_qk_l2norm: L2-normalise queries and keys first.
            decay_lower_bound: Promised lower bound of every log-decay, or ``None``.
            precision: Precision of the non-inverse chunk dots.
            head_axis: Mesh axis (or tuple of axes) of the heads, or ``None`` for :meth:`resolve_head_axis`.
            platform: ``"xla"`` (the reference), or ``None`` for the operation config's platform.

        Returns:
            ``(out [T, H, V] float32, new_recurrent_state [S, H, K, V])``.
        """
        metadata = self.get_impl_metadata()
        return ragged_kda_head_sharded(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            has_initial_state,
            mesh=metadata.mesh,
            head_axis=(
                self.resolve_head_axis(int(query.shape[1]), int(query.shape[0])) if head_axis is None else head_axis
            ),
            chunk_size=int(chunk_size),
            use_qk_l2norm=bool(use_qk_l2norm),
            decay_lower_bound=decay_lower_bound,
            precision=precision,
            platform=platform,
            cfg=metadata.get_operation_config("ragged_kda"),
        )


def kda_ragged_rows_forward(
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
    head_axis: str | tuple[str, ...] | None = None,
) -> tuple[Float[Array, "1 seq_len num_heads d_state"], KDACacheView]:
    """Advance per-request per-channel KDA state over an eSurge packed token stream, on the flat stream.

    Same contract as :func:`~easydel.operations.kernels.kda.kda_packed_rows_forward` for pools without speculative
    candidate rows (callers route those pools to the row-padded driver). Row ``r`` owns tokens
    ``[query_start_loc[r], query_start_loc[r + 1])`` and pool slot ``state_indices[r]`` (slot ``r`` without
    ``state_indices``: the in-place fast path); tokens at or past ``query_start_loc[-1]`` are bucket padding. eSurge
    must name the slots when a step's row window does not start at slot 0 (multi-window steps, sparse row packing).
    For each of q, k and v, eJKernel's ragged causal conv1d (SiLU fused) continues every row from its
    carried window and stores the raw-input window that ends at the row's last token (the conv runs in the common
    dtype of the stream and the window pool, float32 when either is an 8- or 4-bit float; windows keep the pool's
    dtype). The recurrence then runs in
    :class:`RaggedKDAOp`: one decode step for single-token rows and the chunked math over the live chunks of longer
    rows. A row that starts a request (``context_lens <= lens``; without ``context_lens``, any row of two or more
    tokens) starts from a zero conv window and a zero recurrent state, whatever its slots hold (the row-padded driver
    zeroes only the conv window and relies on eSurge clearing freed slots). Rows without tokens never write (whatever
    slot they name), and every slot that no row with tokens owns keeps its state bit for bit.

    Args:
        query: Packed Q projection (pre-conv) ``[1, T, key_dim]``.
        key: Packed K projection (pre-conv) ``[1, T, key_dim]``.
        value: Packed V projection (pre-conv) ``[1, T, value_dim]``.
        beta: Packed write strength ``[1, T, heads]`` (sigmoid applied).
        decay: Packed per-channel log-decay ``[1, T, heads, head_dim]``.
        conv_kernels: Depthwise causal conv kernels for q, k, v, each ``[dim, d_conv]`` (the last tap is the current
            token; ``conv1d.weight.value.squeeze(1).T`` for a grouped ``nn.Conv1d``).
        cache_view: Per-layer KDA cache view; without ``state_indices`` its pool holds at least one slot per row.
        query_start_loc: Cumulative row offsets ``[rows + 1]``.
        context_lens: Tokens per row including this step, or ``None`` (then only single-token rows continue from
            their carried conv window and recurrent state).
        metadata: Operation metadata (mesh, sharding policy, optional ``"ragged_kda"`` config) for the
            ``"ragged_kda"`` operation.
        conv_dtype: Dtype of the conv outputs fed to the recurrence.
        chunk_size: Chunk length of the multi-token recurrence.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound`` (the safe-gate bound), or ``None``.
        precision: ``"highest"`` or ``"default"`` for the non-inverse chunk dots.
        state_indices: Pool slot of every row, ``[rows]`` or ``[1, rows]`` (eSurge's
            ``cache_metadata.recurrent_state_indices``), or ``None`` for slot ``r``. Windows and states are then
            gathered and scattered by slot; rows without tokens or with an out-of-range slot write nothing.
        head_axis: Mesh axis (or tuple of axes) the conv features and heads split over, or ``None`` for the
            operation's :meth:`RaggedKDAOp.resolve_head_axis`. eSurge KDA cache views created with
            ``kda_serving_ep_split`` pass their head axes (head axis, expert-parallel axis); the pools must be split
            the same way.

    Returns:
        tuple: ``(outputs, new_cache_view)``: outputs ``[1, T, heads, head_v_dim]`` float32 (zero at bucket padding)
        and the cache view with updated conv windows and recurrent states.

    Raises:
        ValueError: For rank-major DP metadata (2-D ``query_start_loc``), a stream without rows, ``state_indices``
            with fewer entries than rows, or (without ``state_indices``) a pool with fewer slots than rows.
        TypeError: If the registry resolves ``"ragged_kda"`` to something other than a :class:`RaggedKDAOp`.
    """
    if query_start_loc.ndim != 1:
        raise ValueError(
            "kda_ragged_rows_forward requires 1-D query_start_loc [rows + 1]; rank-major DP metadata "
            f"(got shape {tuple(query_start_loc.shape)}) is not supported for per-channel KDA layers."
        )
    rows = int(query_start_loc.shape[0]) - 1
    if rows < 1:
        raise ValueError(f"kda_ragged_rows_forward needs at least one row, got query_start_loc {query_start_loc.shape}")
    num_slots = int(cache_view.recurrent_state.shape[0])
    if state_indices is None:
        if num_slots < rows:
            raise ValueError(
                f"kda_ragged_rows_forward: without state_indices row r owns pool slot r, so the pool needs at least "
                f"{rows} slots, got {num_slots}"
            )
    else:
        state_indices = jnp.asarray(state_indices, dtype=jnp.int32).reshape(-1)
        if state_indices.shape[0] < rows:
            raise ValueError(
                f"kda_ragged_rows_forward: state_indices needs one pool slot per row ({rows}), got shape "
                f"{tuple(state_indices.shape)}"
            )
        state_indices = state_indices[:rows]
    op = OperationRegistry.create("ragged_kda", metadata)
    if not isinstance(op, RaggedKDAOp):
        raise TypeError(f"'ragged_kda' resolved to {type(op).__name__}, expected a RaggedKDAOp")

    with jax.named_scope("easydel-kda-ragged-rows"):
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
        # The conv kernel addresses the step's windows by row position (``live`` below holds row r's window at r).
        window_rows = jnp.arange(rows, dtype=jnp.int32)
        # ``distribution[2]`` is the count of valid rows for the conv kernel: every row is positional (empty rows have
        # zero length and keep their windows).
        distribution = jnp.full((3,), rows, dtype=jnp.int32)
        mesh = metadata.mesh
        if head_axis is None:
            head_axis = op.resolve_head_axis(num_heads, seq_len)
        if state_indices is None:
            read_slots = write_slots = None
        else:
            in_range = (state_indices >= 0) & (state_indices < num_slots)
            read_slots = jnp.clip(state_indices, 0, num_slots - 1)
            # Only rows with tokens and an in-range slot write their windows back. eSurge points bucket-padding rows at
            # slot 0, which another request may own: their writes go to an out-of-range index and are dropped.
            write_slots = jnp.where((lens > 0) & in_range, state_indices, num_slots)

        def causal_conv(stream: Array, kernel: Array, pool_leaf: Array) -> tuple[Array, Array]:
            # Stream and carried windows meet in one dtype (a no-op for bf16 / bf16); new windows go back to the pool's.
            compute = _conv_compute_dtype(stream.dtype, pool_leaf.dtype)
            live = pool_leaf[:rows] if read_slots is None else pool_leaf[read_slots]
            live = jnp.where(fresh[:, None, None], jnp.zeros((), compute), live.astype(compute))
            out, new_live = ragged_causal_conv1d_head_sharded(
                stream[0].astype(compute),
                live,
                kernel.astype(jnp.float32) if _is_lowfloat(kernel.dtype) else kernel,  # the conv widens it to f32
                qsl,
                window_rows,
                distribution,
                split_sizes=(int(stream.shape[-1]),),
                mesh=mesh,
                head_axis=head_axis,
                d_conv=int(kernel.shape[-1]),
                apply_silu=True,
                cfg=metadata.get_operation_config("ragged_causal_conv1d"),
            )
            new_live = new_live.astype(pool_leaf.dtype)
            if write_slots is not None:
                return out.astype(conv_dtype), pool_leaf.at[write_slots].set(new_live, mode="drop")
            if pool_leaf.shape[0] != rows:
                new_live = jnp.concatenate([new_live, pool_leaf[rows:]], axis=0)
            return out.astype(conv_dtype), new_live

        q_conv, q_pool = causal_conv(query, conv_kernels[0], cache_view.q_conv_state)
        k_conv, k_pool = causal_conv(key, conv_kernels[1], cache_view.k_conv_state)
        v_conv, v_pool = causal_conv(value, conv_kernels[2], cache_view.v_conv_state)
        out, recurrent_pool = op(
            q_conv.reshape(seq_len, num_heads, head_k),
            k_conv.reshape(seq_len, num_heads, head_k),
            v_conv.reshape(seq_len, num_heads, head_v),
            beta[0].astype(jnp.float32) if _is_lowfloat(beta.dtype) else beta[0],  # the op widens to f32 anyway
            decay[0],
            cache_view.recurrent_state,
            qsl,
            state_indices,  # None: row r owns slot r (in-place decode update, no pool gather / scatter)
            jnp.logical_not(fresh),
            chunk_size=int(chunk_size),
            use_qk_l2norm=True,
            decay_lower_bound=decay_lower_bound,
            precision=precision,
            head_axis=head_axis,
        )
    new_cache_view = cache_view.update_kda_states(
        new_q_conv_state=q_pool,
        new_k_conv_state=k_pool,
        new_v_conv_state=v_pool,
        new_recurrent_state=recurrent_pool,
    )
    return out.astype(jnp.float32)[None], new_cache_view


__all__ = ("RaggedKDAOp", "kda_ragged_rows_forward")
