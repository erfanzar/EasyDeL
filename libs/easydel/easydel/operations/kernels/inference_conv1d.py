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
"""Ragged causal depthwise 1-D convolution over packed batches.

This module provides a single-kernel implementation of the short causal
depthwise convolution that precedes the gated-delta-rule recurrence in models
like Qwen3-Next, Qwen3.5, and other linear-attention / state-space hybrids.

It targets the *packed / continuous-batching* inference setting where many
requests with heterogeneous sequence lengths share one contiguous token buffer:
single-token decode requests and multi-token prefill requests coexist in the
same call. The kernel handles both uniformly, fuses in SiLU, and refreshes the
per-request rolling state in one pass.

Exports:
    - :func:`ragged_causal_conv1d`: the jit-compiled functional entry point.
    - :class:`RaggedCausalConv1D`: a registered :class:`OperationImpl` wrapper
      that plugs the kernel into EasyDeL's operation-registry machinery so it
      can be resolved by name alongside other ragged inference kernels.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.sharding import mesh_axis_size

from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)


def _reorder_concatenated_tensor_for_sharding(
    concatenated_tensor: jax.Array,
    split_sizes: tuple[int, ...],
    n_shards: int,
    dim: int,
) -> jax.Array:
    """Arrange fused ``[A | B | C]`` features as ``[A0 | B0 | C0 | A1 | B1 | C1 | ...]``.

    Reorders the channel axis of a tensor that concatenates several fused
    feature groups so that, after the same axis is split into ``n_shards``
    along the mesh, each shard holds an equal slice of *every* original
    group. This matches the layout that the TPU serving path uses for sharded conv /
    QKV projections.

    Args:
        concatenated_tensor: Input tensor with the fused axis to reorder.
        split_sizes: Sizes of each contiguous group along ``dim`` before
            reordering. ``sum(split_sizes)`` must equal
            ``concatenated_tensor.shape[dim]``.
        n_shards: Number of TP shards the axis will be split into.
        dim: Axis to reorder. Negative values index from the end.

    Returns:
        jax.Array: A view with the same shape as ``concatenated_tensor``
        whose axis ``dim`` is reordered so per-shard slices interleave the
        original feature groups.
    """
    if dim < 0:
        dim += concatenated_tensor.ndim
    old_shape = concatenated_tensor.shape
    new_shape = (*old_shape[:dim], int(n_shards), -1, *old_shape[dim + 1 :])
    split_tensors = []
    start_offset = 0
    for split_size in split_sizes:
        split_tensor = jax.lax.slice_in_dim(
            concatenated_tensor,
            start_offset,
            start_offset + int(split_size),
            axis=dim,
        )
        split_tensors.append(split_tensor.reshape(new_shape))
        start_offset += int(split_size)
    reordered_tensor = jnp.concatenate(split_tensors, axis=dim + 1)
    return reordered_tensor.reshape(old_shape)


def _fix_query_start_loc(query_start_loc: jnp.ndarray, num_valid_seqs: jnp.ndarray) -> jnp.ndarray:
    """Clamp trailing padding entries of ``query_start_loc`` to a sentinel value.

    The schedule sometimes carries inactive trailing slots whose
    ``query_start_loc`` entries are not monotone. This helper rewrites all
    entries past ``num_valid_seqs`` to ``query_start_loc[num_valid_seqs]``
    so downstream length / boundary arithmetic produces zero-length
    sequences for those slots.

    Args:
        query_start_loc: Cumulative per-request token offsets, shape
            ``(num_slots + 1,)``.
        num_valid_seqs: Scalar number of valid (non-padding) sequences.

    Returns:
        jnp.ndarray: A copy of ``query_start_loc`` with padding entries
        clamped to the last valid offset.
    """
    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    return jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)


def _depthwise_conv1d_flat(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Run a depthwise causal conv over the flat packed token stream.

    Performs ``out[t, :] = sum_k padded_x[t + k, :] * kernel[:, k]`` with
    a ``(d_conv - 1)``-wide left pad of zeros, in float32. The output
    boundary tokens at the seam between requests are *not* yet correct —
    they are rewritten by the boundary fix-up in
    :func:`_ragged_causal_conv1d_impl`.

    Args:
        x: Packed input tokens, shape ``(num_tokens, conv_dim)``.
        kernel: Depthwise kernel, shape ``(conv_dim, d_conv)``.

    Returns:
        jnp.ndarray: Conv output in float32, shape
        ``(num_tokens, conv_dim)``.
    """
    num_tokens = x.shape[0]
    d_conv = kernel.shape[-1]
    padded_x = jnp.pad(x.astype(jnp.float32), ((d_conv - 1, 0), (0, 0)))
    kernel = kernel.astype(jnp.float32)
    out = jnp.zeros((num_tokens, x.shape[-1]), dtype=jnp.float32)
    for k in range(d_conv):
        out = out + padded_x[k : k + num_tokens, :] * kernel[None, :, k]
    return out


def _get_boundary_indices(
    starts: jnp.ndarray,
    lengths: jnp.ndarray,
    d_conv: int,
    num_valid_seqs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the gather/scatter indices used to fix per-request boundary tokens.

    For each request, the first ``d_conv - 1`` output tokens depend on
    historical state from the previous step rather than the zeros that the
    flat conv produced. This helper materialises:

    * ``gather_indices`` — token positions in ``x`` to read the head of
      each request from (clamped to the request length so we never index
      past its end).
    * ``scatter_indices`` — destination output positions to rewrite with
      the corrected boundary values, with ``-1`` for invalid (padding)
      entries.

    Args:
        starts: Per-request start offsets, shape ``(num_slots,)``.
        lengths: Per-request lengths, shape ``(num_slots,)``.
        d_conv: Convolution window size.
        num_valid_seqs: Scalar number of valid sequences.

    Returns:
        tuple: ``(gather_indices, scatter_indices)`` with shape
        ``(num_slots, d_conv - 1)`` each, both ``int32``.
    """
    valid_mask = jnp.arange(starts.shape[0]) < num_valid_seqs
    starts = jnp.where(valid_mask, starts, 1)[:, None]
    lengths = lengths[:, None]
    k_range = jnp.arange(d_conv - 1)[None, :]
    gather_indices = starts + jnp.minimum(k_range, lengths - 1)
    scatter_indices = jnp.where(
        (k_range < lengths) & valid_mask[:, None],
        starts + k_range,
        -1,
    )
    return gather_indices, scatter_indices


def _ragged_causal_conv1d_impl(
    x: jnp.ndarray,
    conv_state: jnp.ndarray,
    kernel: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    *,
    d_conv: int,
    apply_silu: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies causal depthwise conv1d over ragged sequences with rolling state.

    Single-pass kernel that computes one depthwise-conv output per token and
    refreshes each slot's rolling state to the last ``d_conv`` tokens of its
    request. Handles decode (single-token) and prefill (multi-token) slots
    uniformly in the same packed batch. SiLU is fused in by default.

    Algorithm
    ---------
    For each packed token at global index ``t`` belonging to request ``r`` at
    local offset ``local = t - q_start[r]``, the output is

        output[t, :] = silu( sum_{k = 0..d_conv - 1}
                             kernel[:, d_conv - 1 - k] * tok(local - k) )

    where ``tok(local - k)`` is resolved as:

        * ``x[t - k, :]``            if ``local - k >= 0``  (same request)
        * ``conv_state[r, :, j]``    if ``local - k <  0``  (historical), with
                                     ``j = d_conv + local - k``

    The kernel orientation follows :func:`apply_manual_depthwise_conv`: summing
    over the last axis of ``state * kernel`` is equivalent to a PyTorch-style
    causal depthwise conv where ``kernel[:, d_conv - 1]`` is the "current" tap
    and ``kernel[:, 0]`` is the ``d_conv - 1``-step-back tap.

    The state update produces, for each slot ``r``,

        new_state[r, :, j] = tok at seq pos ``(L_r - d_conv + j)``

    where ``L_r`` is the slot's request length. For positions that fall before
    the start of the current request (``L_r + j < d_conv``), the value is
    carried over from the incoming ``conv_state``; otherwise it is sourced from
    ``x``. For sequences with ``L_r >= d_conv`` the new state is exactly the
    trailing ``d_conv`` tokens of ``x`` for that slot.

    Conventions
    -----------
    EasyDeL's state convention stores the *full* ``d_conv``-wide window (channels
    first): after processing token ``N``, ``conv_state[:, :, d_conv - 1]`` holds
    token ``N`` and ``conv_state[:, :, 0]`` holds token ``N - d_conv + 1``. The
    next conv step operates on a shifted version of this window. Note that
    position ``0`` is never read while computing outputs for a subsequent
    request (the oldest entry falls outside the ``d_conv``-wide window) but it
    is carried through so sub-``d_conv`` request lengths keep continuity.

    Low-precision inputs (fp8 / fp4) are promoted to float32 for the
    accumulation; the final cast returns to ``x.dtype``.

    Args:
        x: Packed input stream, shape ``(num_tokens, conv_dim)``.
        conv_state: Per-slot rolling state,
            shape ``(num_slots, conv_dim, d_conv)``. Position ``d_conv - 1`` is
            the most recent historical token, position ``0`` is ``d_conv``
            tokens back.
        kernel: Depthwise kernel, shape ``(conv_dim, d_conv)``. Same layout as
            :func:`easydel.layers.linear_attention.apply_manual_depthwise_conv`.
        query_start_loc: Cumulative token offsets per request,
            shape ``(num_slots + 1,)``. ``query_start_loc[-1]`` must equal the
            number of valid tokens; any trailing "inactive" slots can be
            encoded by setting their length to 0 and/or using ``distribution``.
        state_indices: Request-to-state-slot mapping, shape ``(num_slots,)``.
            Used to gather the incoming state for each request and to scatter
            the updated state back into the pool.
        distribution: ``(decode_end, prefill_end, mixed_end)`` tensor of
            shape ``(3,)`` int32. Only ``distribution[2]`` is consumed here
            (number of valid sequences); trailing slots beyond that index keep
            their existing state unchanged and contribute no output updates.
        d_conv: Convolution kernel / state window size. Must match
            ``kernel.shape[-1]`` and ``conv_state.shape[-1]``.
        apply_silu: If True (default), applies ``jax.nn.silu`` after the
            accumulation, matching Qwen3-Next / GatedDeltaNet conventions.
            Pass False to get the raw linear convolution output.

    Returns:
        A tuple ``(output, updated_conv_state)``:

        - ``output``: Per-token conv output, shape ``(num_tokens, conv_dim)``,
          dtype matches ``x``.
        - ``updated_conv_state``: Conv-state pool with the slots indexed by
          ``state_indices`` refreshed, shape and dtype match ``conv_state``.
          The input buffer is donated (``donate_argnames=("conv_state",)``)
          to avoid an XLA copy.

    Notes:
        * The function is JIT-compiled with ``d_conv`` and ``apply_silu``
          marked static, so passing different values for these triggers
          recompilation.
        * The ``for k in range(d_conv)`` loop is Python-level and unrolled at
          trace time; pick ``d_conv`` such that the unroll is reasonable
          (typically 4, the Qwen3-Next / GDR default).
    """
    num_tokens, dim = x.shape
    max_reqs = state_indices.shape[0]

    num_valid_seqs = distribution[2]
    effective_query_start_loc = _fix_query_start_loc(query_start_loc, num_valid_seqs)
    lengths = effective_query_start_loc[1:] - effective_query_start_loc[:-1]

    gathered_state = conv_state[state_indices]

    out = _depthwise_conv1d_flat(x, kernel)

    starts = effective_query_start_loc[:-1]
    gather_indices, scatter_indices = _get_boundary_indices(starts, lengths, d_conv, num_valid_seqs)
    x_first = x[gather_indices]
    history = gathered_state[:, :, 1:].transpose(0, 2, 1)
    combined_tokens = jnp.concatenate([history, x_first], axis=1)
    boundary_out = jax.lax.conv_general_dilated(
        combined_tokens.astype(jnp.float32),
        kernel[:, None, :].astype(jnp.float32),
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "OIW", "NWC"),
        feature_group_count=dim,
        precision=jax.lax.Precision.HIGHEST,
    ).reshape(-1, dim)
    out = out.at[scatter_indices.flatten()].set(
        boundary_out.astype(out.dtype),
        mode="drop",
        wrap_negative_indices=False,
    )
    total_valid_tokens = effective_query_start_loc[num_valid_seqs]
    valid_token_mask = jnp.arange(num_tokens) < total_valid_tokens
    out = jnp.where(valid_token_mask[:, None], out, 0.0)

    if apply_silu:
        out = jax.nn.silu(out)

    padded_lengths = jnp.zeros(max_reqs, dtype=jnp.int32).at[: lengths.shape[0]].set(lengths)
    padded_q_end = jnp.zeros(max_reqs, dtype=jnp.int32).at[: lengths.shape[0]].set(effective_query_start_loc[1:])

    r_grid = jnp.arange(max_reqs)[:, None]
    j_grid = jnp.arange(d_conv)[None, :]
    is_from_old_state = (padded_lengths[:, None] + j_grid) < d_conv

    idx_state_new = jnp.where(is_from_old_state, padded_lengths[:, None] + j_grid, 0)
    idx_x_new = jnp.clip(padded_q_end[:, None] - d_conv + j_grid, 0, num_tokens - 1)

    new_state_hist = gathered_state[r_grid, :, idx_state_new]
    new_state_from_x = x[idx_x_new.reshape(-1)].reshape(max_reqs, d_conv, dim)

    new_state_jchw = jnp.where(
        is_from_old_state[..., None],
        new_state_hist,
        new_state_from_x.astype(new_state_hist.dtype),
    )
    new_state = new_state_jchw.transpose(0, 2, 1).astype(conv_state.dtype)

    true_valid_seq_mask = jnp.arange(max_reqs) < num_valid_seqs
    updated_conv_state = conv_state.at[state_indices].set(
        jnp.where(
            true_valid_seq_mask[:, None, None],
            new_state,
            conv_state[state_indices],
        )
    )

    return out.astype(x.dtype), updated_conv_state


@jax.jit(
    donate_argnames=("conv_state",),
    static_argnames=("d_conv", "apply_silu"),
)
@jax.named_scope("ragged_causal_conv1d_jax")
def ragged_causal_conv1d(
    x: jnp.ndarray,
    conv_state: jnp.ndarray,
    kernel: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    *,
    d_conv: int,
    apply_silu: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled entry point for the ragged causal depthwise conv1d.

    Thin wrapper that forwards to :func:`_ragged_causal_conv1d_impl` after
    being jit-compiled with ``d_conv`` and ``apply_silu`` marked static and
    ``conv_state`` donated.

    Args:
        x: Packed input stream, shape ``(num_tokens, conv_dim)``.
        conv_state: Per-slot rolling state, shape
            ``(num_slots, conv_dim, d_conv)``. Donated.
        kernel: Depthwise kernel, shape ``(conv_dim, d_conv)``.
        query_start_loc: Cumulative per-request token offsets, shape
            ``(num_slots + 1,)``.
        state_indices: Request-to-slot mapping, shape ``(num_slots,)``.
        distribution: ``(decode_end, prefill_end, mixed_end)`` int32
            triple of shape ``(3,)``.
        d_conv: Convolution window size (static).
        apply_silu: Whether to fuse SiLU into the output (static).

    Returns:
        tuple: ``(output, updated_conv_state)`` matching the return
        contract of :func:`_ragged_causal_conv1d_impl`.
    """
    return _ragged_causal_conv1d_impl(
        x=x,
        conv_state=conv_state,
        kernel=kernel,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        d_conv=d_conv,
        apply_silu=apply_silu,
    )


def ragged_causal_conv1d_head_sharded(
    x: jnp.ndarray,
    conv_state: jnp.ndarray,
    kernel: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    *,
    split_sizes: tuple[int, ...],
    mesh: object | None,
    head_axis: object | None,
    d_conv: int,
    apply_silu: bool = True,
    pre_sharded: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run ragged depthwise conv with the feature axis sharded across the TP mesh.

    When the mesh exposes a non-trivial TP axis and every entry of
    ``split_sizes`` is divisible by the TP size, the feature axis of ``x``
    and ``kernel`` is reordered with
    :func:`_reorder_concatenated_tensor_for_sharding` (unless ``pre_sharded``
    is ``True``) and the kernel is launched under ``jax.shard_map`` so each
    rank operates on its head shard. Otherwise the call falls back to the
    single-device :func:`ragged_causal_conv1d`.

    Args:
        x: Packed input stream, shape ``(num_tokens, conv_dim)``.
        conv_state: Per-slot rolling state, shape
            ``(num_slots, conv_dim, d_conv)``.
        kernel: Depthwise kernel, shape ``(conv_dim, d_conv)``.
        query_start_loc: Cumulative per-request token offsets, shape
            ``(num_slots + 1,)``.
        state_indices: Request-to-slot mapping, shape ``(num_slots,)``.
        distribution: ``(decode_end, prefill_end, mixed_end)`` triple of
            shape ``(3,)``.
        split_sizes: Sizes of the fused feature groups making up
            ``conv_dim`` (e.g. one per Q/K/V projection).
        mesh: Optional spectrax/jax mesh carrying ``head_axis``.
        head_axis: Mesh axis name used for TP-sharding the feature axis.
        d_conv: Convolution window size (static for the inner JIT).
        apply_silu: Whether to fuse SiLU into the output.
        pre_sharded: When ``True``, ``x`` and ``kernel`` are already in the
            interleaved-per-shard layout; skip the reordering step.

    Returns:
        tuple: ``(output, updated_conv_state)`` matching the return
        contract of :func:`ragged_causal_conv1d`.
    """
    tp_size = mesh_axis_size(mesh, head_axis)
    can_head_shard = (
        mesh is not None
        and head_axis is not None
        and int(tp_size) > 1
        and all(int(size) % int(tp_size) == 0 for size in split_sizes)
    )
    if not can_head_shard:
        return ragged_causal_conv1d(
            x,
            conv_state,
            kernel,
            query_start_loc,
            state_indices,
            distribution,
            d_conv=d_conv,
            apply_silu=apply_silu,
        )

    if not pre_sharded:
        x = _reorder_concatenated_tensor_for_sharding(x, split_sizes, int(tp_size), -1)
        kernel = _reorder_concatenated_tensor_for_sharding(kernel, split_sizes, int(tp_size), 0)

    feature_spec = PartitionSpec(None, head_axis)
    state_spec = PartitionSpec(None, head_axis, None)
    kernel_spec = PartitionSpec(head_axis, None)
    replicated = PartitionSpec()

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(feature_spec, state_spec, kernel_spec, replicated, replicated, replicated),
        out_specs=(feature_spec, state_spec),
        check_vma=False,
    )
    def _mapped(local_x, local_state, local_kernel, local_qsl, local_si, local_dist):
        return _ragged_causal_conv1d_impl(
            x=local_x,
            conv_state=local_state,
            kernel=local_kernel,
            query_start_loc=local_qsl,
            state_indices=local_si,
            distribution=local_dist,
            d_conv=d_conv,
            apply_silu=apply_silu,
        )

    return _mapped(x, conv_state, kernel, query_start_loc, state_indices, distribution)


@OperationRegistry.register
class RaggedCausalConv1D(OperationImpl):
    """Ragged causal depthwise conv1d operation for packed inference batches.

    First-class :class:`OperationImpl` wrapping :func:`ragged_causal_conv1d`.
    Provides a registry-resolvable handle (``"ragged_causal_conv1d"``) so the
    kernel can be instantiated by name alongside the ragged gated-delta-rule
    op when assembling a packed-inference model.

    All backend-specific forward methods delegate to :meth:`forward_native`,
    which in turn calls the JIT-compiled :func:`ragged_causal_conv1d`. XLA
    handles backend selection (TPU / GPU / CPU) automatically via the
    ``jax.jit`` decorator on the free function, so the implementation is
    identical across devices.

    Example:
        >>> from easydel.operations import OperationMetadata
        >>> from easydel.operations.kernels import RaggedCausalConv1D
        >>> metadata = OperationMetadata(runtime_dtype=jnp.bfloat16)
        >>> op = RaggedCausalConv1D(metadata)
        >>> out, new_state = op(
        ...     x=packed_tokens,
        ...     conv_state=conv_state_pool,
        ...     kernel=conv_weight,
        ...     query_start_loc=q_start,
        ...     state_indices=slot_map,
        ...     distribution=dist,
        ...     d_conv=4,
        ... )
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registry name for this operation.

        Returns:
            The string ``"ragged_causal_conv1d"`` — the key used by
            :class:`OperationRegistry` to look this class up.
        """
        return "ragged_causal_conv1d"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Describe the metadata and cache requirements of this op.

        The ragged conv1d op needs sequence-length metadata and state-slot
        mappings to build the packed layout, and it supports RECURRENT and
        HYBRID cache types (conv state lives alongside the recurrent GDN
        state in the KV-cache).

        Args:
            mode: Execution mode. Kept for signature compatibility with the
                :class:`OperationImpl` base — requirements are identical across
                modes for this op.

        Returns:
            An :class:`OperationRequirements` describing the op's metadata
            fields (``SEQ_LENS``, ``POSITIONS``, ``STATE_INDICES``) and
            supported cache types (``RECURRENT | HYBRID``).
        """
        return (
            RequirementsBuilder("ragged_causal_conv1d")
            .require_metadata(MetadataField.SEQ_LENS | MetadataField.POSITIONS | MetadataField.STATE_INDICES)
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .build()
        )

    @jax.named_scope("easydel-ragged-causal-conv1d-native")
    def forward_native(
        self,
        x: jnp.ndarray,
        conv_state: jnp.ndarray,
        kernel: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        *,
        d_conv: int,
        apply_silu: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Primary forward implementation; thin wrapper over the free function.

        See :func:`ragged_causal_conv1d` for the algorithmic details and
        argument / return semantics — this method forwards its inputs
        unchanged and exists only to plug the kernel into the
        :class:`OperationImpl` dispatch protocol.

        Args:
            x: Packed input stream, shape ``(num_tokens, conv_dim)``.
            conv_state: Per-slot rolling state,
                shape ``(num_slots, conv_dim, d_conv)``.
            kernel: Depthwise kernel, shape ``(conv_dim, d_conv)``.
            query_start_loc: Cumulative per-request token offsets,
                shape ``(num_slots + 1,)``.
            state_indices: Request-to-slot mapping, shape ``(num_slots,)``.
            distribution: ``(decode_end, prefill_end, mixed_end)``
                int32 triple of shape ``(3,)``.
            d_conv: Convolution window size (static).
            apply_silu: If True, fuse SiLU into the conv output (static).

        Returns:
            ``(output, updated_conv_state)`` — see
            :func:`ragged_causal_conv1d`.
        """
        return ragged_causal_conv1d(
            x,
            conv_state,
            kernel,
            query_start_loc,
            state_indices,
            distribution,
            d_conv=d_conv,
            apply_silu=apply_silu,
        )

    def forward_tpu(self, *args, **kwargs):
        """TPU forward — delegates to :meth:`forward_native`.

        The underlying function is already JIT-compiled and XLA lowers it
        efficiently to TPU ops, so there is no TPU-specific Pallas variant
        needed here.
        """
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs):
        """GPU forward — delegates to :meth:`forward_native`.

        XLA handles GPU lowering; the per-token gather + elementwise-multiply
        pattern fuses well into a small number of CUDA kernels.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        """CPU forward — delegates to :meth:`forward_native`.

        Primarily useful for unit tests and reference checks; performance is
        not a design goal on this backend.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        """CUDA forward — alias of :meth:`forward_gpu` for NVIDIA devices."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs):
        """ROCm forward — alias of :meth:`forward_gpu` for AMD devices."""
        return self.forward_native(*args, **kwargs)
