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
"""Rank-major (eSurge SPMD data-parallel) form of the flat-stream KDA serving driver.

eSurge serves with ``data_parallel_size > 1`` by giving every DP rank its own token block, request rows and contiguous
block of recurrent pool slots (rank-major packing). :func:`kda_ragged_rows_rank_major_forward` runs the single-rank
serving driver :func:`~easydel.operations.kernels.kda_rows_forward` (the decode fast path when the rank's step is
decode-only, else the flat-stream driver :func:`~easydel.operations.kernels.kda_ragged_rows_forward`) for every rank
under one ``shard_map`` over the attention data-parallel axis (``ATTN_DP``) and the head axis: each device advances its
rank's rows on its heads with its rank's slot block, without collectives. Only eSurge rank-major steps (a 2-D
``query_start_loc``) reach it; training and single-rank serving never do.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float

from easydel.axis import resolve_attention_data_parallel_axis
from easydel.caching import KDACacheView
from easydel.infra.sharding import mesh_axis_size

from .._operation_impl import OperationRegistry
from .._operation_meta import OperationMetadata
from .ragged_kda import RaggedKDAOp, kda_ragged_rows_forward
from .ragged_kda_decode import kda_rows_forward


def _meshless_metadata(metadata: OperationMetadata) -> OperationMetadata:
    """``metadata`` without a mesh, so the operations it creates run unsharded on a ``shard_map``'s per-device blocks.

    Args:
        metadata: Operation metadata of the layer.

    Returns:
        A copy with the same dtypes, backend, sharding policy and operation configs, and no mesh.
    """
    return OperationMetadata(
        runtime_dtype=metadata.runtime_dtype,
        runtime_softmax_dtype=metadata.runtime_softmax_dtype,
        sequence_axis_name=metadata.sequence_axis_name,
        platform=metadata.platform,
        backend=metadata.backend,
        axis_policy=metadata.axis_policy,
        operation_configs=metadata.operation_configs,
        _stored_mesh=None,
    )


def kda_ragged_rows_rank_major_forward(
    query: Float[Array, "dp tokens_per_rank key_dim"],
    key: Float[Array, "dp tokens_per_rank key_dim"],
    value: Float[Array, "dp tokens_per_rank value_dim"],
    beta: Float[Array, "dp tokens_per_rank num_heads"],
    decay: Float[Array, "dp tokens_per_rank num_heads head_dim"],
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
) -> tuple[Float[Array, "dp tokens_per_rank num_heads d_state"], KDACacheView]:
    """:func:`~easydel.operations.kernels.kda_rows_forward` for eSurge rank-major SPMD data parallelism.

    eSurge's rank-major layout (``data_parallel_size > 1``) gives every DP rank its own token block, its own rows and a
    contiguous block of pool slots: the token batch is ``[dp, T / dp, ...]``, ``query_start_loc`` is ``[dp, rows + 1]``
    with rank-local offsets, ``context_lens`` is ``[dp * rows]`` (or ``[dp, rows]``) and ``state_indices`` holds
    rank-local slots ``[dp, rows]`` (slot ``s`` of rank ``d`` is pool slot ``d * slots_per_rank + s``). The pools'
    slot axis is split over the attention data-parallel axis (``ATTN_DP``) and their heads / conv features over the
    head axis, so this runs the single-rank serving driver under one ``shard_map`` over (``ATTN_DP``, head axis): every
    device advances its rank's rows on its heads with its rank's slot block, without collectives. The per-device body is
    :func:`~easydel.operations.kernels.kda_rows_forward` unchanged (its operations unsharded): when a rank's token block
    holds at most one token per row, the rank picks the decode fast path at run time if its step is decode-only (each
    rank decides for itself), else ``flat_forward``. Every rank keeps the single-rank contract: rows without tokens
    write nothing (whatever slot they name), fresh rows start from zero, and slots no row with tokens owns keep their
    state bit for bit.

    Args:
        query: Packed Q projection (pre-conv) ``[dp, T / dp, key_dim]``.
        key: Packed K projection (pre-conv) ``[dp, T / dp, key_dim]``.
        value: Packed V projection (pre-conv) ``[dp, T / dp, value_dim]``.
        beta: Packed write strength ``[dp, T / dp, heads]`` (sigmoid applied).
        decay: Packed per-channel log-decay ``[dp, T / dp, heads, head_dim]``.
        conv_kernels: Depthwise causal conv kernels for q, k, v, each ``[dim, d_conv]``.
        cache_view: Per-layer KDA cache view whose pools hold ``dp * slots_per_rank`` slots.
        query_start_loc: Rank-local cumulative row offsets ``[dp, rows + 1]``.
        context_lens: Tokens per row including this step, ``[dp * rows]`` or ``[dp, rows]``, or ``None``.
        metadata: Operation metadata (mesh, sharding policy, optional ``"ragged_kda"`` config).
        conv_dtype: Dtype of the conv outputs fed to the recurrence.
        chunk_size: Chunk length of the multi-token recurrence.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound``, or ``None``.
        precision: ``"highest"`` or ``"default"`` for the non-inverse chunk dots.
        state_indices: Rank-local pool slot of every row, ``[dp, rows]`` or ``[dp * rows]``, or ``None`` (row ``r``
            of rank ``d`` owns its rank's slot ``r``).
        flat_forward: Single-rank driver for rank steps with multi-token rows (same signature as
            :func:`~easydel.operations.kernels.kda_ragged_rows_forward`), run per rank.
        head_axis: Mesh axis (or tuple of axes) the heads and conv features split over (must not share an axis with
            the attention data-parallel axis), or ``None`` for :meth:`RaggedKDAOp.resolve_head_axis`.

    Returns:
        tuple: ``(outputs, new_cache_view)``: outputs ``[dp, T / dp, heads, head_v_dim]`` float32 (zero at bucket
        padding) and the cache view with updated conv windows and recurrent states.

    Raises:
        ValueError: If ``query_start_loc`` is not 2-D, the token batch and ``query_start_loc`` disagree on the rank
            count, the pool does not split into equal rank blocks, the row metadata is short, or the metadata's mesh
            does not carry the rank count on its attention data-parallel axis.
        TypeError: If the registry resolves ``"ragged_kda"`` to something other than a :class:`RaggedKDAOp`.
    """
    if query_start_loc.ndim != 2:
        raise ValueError(
            "kda_ragged_rows_rank_major_forward requires rank-major query_start_loc [dp, rows + 1], got shape "
            f"{tuple(query_start_loc.shape)}"
        )
    dp_size = int(query_start_loc.shape[0])
    rows = int(query_start_loc.shape[1]) - 1
    if rows < 1:
        raise ValueError(
            f"kda_ragged_rows_rank_major_forward needs at least one row per rank, got {query_start_loc.shape}"
        )
    if int(query.shape[0]) != dp_size:
        raise ValueError(
            f"rank-major KDA rows need a [dp, T / dp, D] token batch: the batch {tuple(query.shape)} has "
            f"{query.shape[0]} ranks, query_start_loc {tuple(query_start_loc.shape)} has {dp_size}"
        )
    num_slots = int(cache_view.recurrent_state.shape[0])
    if num_slots % dp_size:
        raise ValueError(
            f"kda_ragged_rows_rank_major_forward: the pool's {num_slots} slots do not split into {dp_size} rank blocks"
        )
    row_arrays: list[Array] = []
    if context_lens is not None:
        context_lens = jnp.asarray(context_lens, dtype=jnp.int32).reshape(dp_size, -1)
        if int(context_lens.shape[1]) < rows:
            raise ValueError(f"context_lens {tuple(context_lens.shape)} has fewer than {rows} rows per rank")
        row_arrays.append(context_lens[:, :rows])
    if state_indices is not None:
        state_indices = jnp.asarray(state_indices, dtype=jnp.int32).reshape(dp_size, -1)
        if int(state_indices.shape[1]) < rows:
            raise ValueError(f"state_indices {tuple(state_indices.shape)} has fewer than {rows} rows per rank")
        row_arrays.append(state_indices[:, :rows])
    has_context, has_slots = context_lens is not None, state_indices is not None

    mesh = metadata.mesh
    dp_axis = resolve_attention_data_parallel_axis(metadata.runtime_sharding_resolver)
    if mesh is None or mesh_axis_size(mesh, dp_axis) != dp_size:
        raise ValueError(
            f"kda_ragged_rows_rank_major_forward: {dp_size} ranks need a mesh whose attention data-parallel axis "
            f"{dp_axis!r} has {dp_size} devices (mesh {getattr(mesh, 'shape', None)})"
        )
    op = OperationRegistry.create("ragged_kda", metadata)
    if not isinstance(op, RaggedKDAOp):
        raise TypeError(f"'ragged_kda' resolved to {type(op).__name__}, expected a RaggedKDAOp")
    if head_axis is None:
        head_axis = op.resolve_head_axis(int(beta.shape[-1]), int(query.shape[1]))
    local_metadata = _meshless_metadata(metadata)
    view_metadata, layer_index = cache_view.metadata, cache_view.layer_index

    def rank_body(q, k, v, b, g, kernels, rank_pools, qsl, *rank_rows):
        # One device: its rank's token block [1, T / dp, ...] on its heads, and its rank's slot block of every pool.
        rows_iter = iter(rank_rows)
        rank_context = next(rows_iter)[0] if has_context else None
        rank_slots = next(rows_iter)[0] if has_slots else None
        view = KDACacheView(
            q_conv_state=rank_pools[0],
            k_conv_state=rank_pools[1],
            v_conv_state=rank_pools[2],
            recurrent_state=rank_pools[3],
            positions=jnp.zeros((int(rank_pools[3].shape[0]),), dtype=jnp.int32),
            metadata=view_metadata,
            layer_index=layer_index,
        )
        out, new_view = kda_rows_forward(
            flat_forward=flat_forward,
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=g,
            conv_kernels=kernels,
            cache_view=view,
            query_start_loc=qsl[0],
            context_lens=rank_context,
            metadata=local_metadata,
            conv_dtype=conv_dtype,
            chunk_size=chunk_size,
            decay_lower_bound=decay_lower_bound,
            precision=precision,
            state_indices=rank_slots,
        )
        return out, (new_view.q_conv_state, new_view.k_conv_state, new_view.v_conv_state, new_view.recurrent_state)

    stream_spec = PartitionSpec(dp_axis, None, head_axis)  # q / k / v [dp, T / dp, features], beta [dp, T / dp, heads]
    head_dim_spec = PartitionSpec(dp_axis, None, head_axis, None)  # decay [dp, T / dp, heads, K] and the output
    kernel_spec = PartitionSpec(head_axis, None)  # conv kernels [features, d_conv]
    conv_pool_spec = PartitionSpec(dp_axis, head_axis, None)  # conv windows [slots, features, d_conv]
    recurrent_pool_spec = PartitionSpec(dp_axis, head_axis, None, None)  # recurrent states [slots, heads, K, V]
    pool_specs = (conv_pool_spec, conv_pool_spec, conv_pool_spec, recurrent_pool_spec)
    row_spec = PartitionSpec(dp_axis, None)  # query_start_loc [dp, rows + 1]; context lens and slots [dp, rows]
    pools = (cache_view.q_conv_state, cache_view.k_conv_state, cache_view.v_conv_state, cache_view.recurrent_state)
    in_specs = (
        stream_spec,
        stream_spec,
        stream_spec,
        stream_spec,
        head_dim_spec,
        (kernel_spec, kernel_spec, kernel_spec),
        pool_specs,
        row_spec,
        *((row_spec,) * len(row_arrays)),
    )
    with jax.named_scope("easydel-kda-ragged-rows-rank-major"):
        out, new_pools = jax.shard_map(
            rank_body,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(head_dim_spec, pool_specs),
            check_vma=False,
        )(
            query,
            key,
            value,
            beta,
            decay,
            tuple(conv_kernels),
            pools,
            jnp.asarray(query_start_loc, dtype=jnp.int32),
            *row_arrays,
        )
    new_cache_view = cache_view.update_kda_states(
        new_q_conv_state=new_pools[0],
        new_k_conv_state=new_pools[1],
        new_v_conv_state=new_pools[2],
        new_recurrent_state=new_pools[3],
    )
    return out, new_cache_view


__all__ = ("kda_ragged_rows_rank_major_forward",)
