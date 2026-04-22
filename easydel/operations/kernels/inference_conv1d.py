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

import jax
import jax.numpy as jnp

from easydel.layers.norms import lowfloats

from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)


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
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)

    num_valid_seqs = distribution[2]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)

    req_indices = jnp.sum(token_idx[:, None] >= effective_query_start_loc[None, :], axis=1) - 1
    req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
    local_indices = token_idx - effective_query_start_loc[req_indices]
    lengths = effective_query_start_loc[1:] - effective_query_start_loc[:-1]

    if conv_state.dtype in lowfloats or kernel.dtype in lowfloats:
        compute_dtype = jnp.float32
    else:
        compute_dtype = jnp.promote_types(conv_state.dtype, kernel.dtype)

    gathered_state = conv_state[state_indices]
    gathered_state_ct = gathered_state.astype(compute_dtype)
    x_ct = x.astype(compute_dtype)
    kernel_ct = kernel.astype(compute_dtype)

    out = jnp.zeros((num_tokens, dim), dtype=compute_dtype)
    for k in range(d_conv):
        mask_from_x = local_indices >= k
        idx_x = jnp.clip(token_idx - k, 0, num_tokens - 1)
        idx_state_t = jnp.clip(d_conv + local_indices - k, 0, d_conv - 1)
        x_tokens = x_ct[idx_x]
        state_tokens = gathered_state_ct[req_indices, :, idx_state_t]
        token_k = jnp.where(mask_from_x[:, None], x_tokens, state_tokens)
        out = out + token_k * kernel_ct[None, :, d_conv - 1 - k]

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
