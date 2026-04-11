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

"""
Gated Delta Rule (GDR) linear attention implementation for EasyDeL.

This module provides the GatedDeltaRule operation, a linear attention mechanism
used in hybrid transformer architectures like Qwen3Next. The gated delta rule
combines:

1. Causal convolution for local context
2. Gated linear attention with delta rule updates
3. Learnable decay for forgetting previous state

Key characteristics:
- Linear complexity O(N) in sequence length (vs O(N²) for standard attention)
- Maintains recurrent state for efficient inference
- Supports chunked computation for efficient training

The algorithm:
    Training (chunked):
        - Process sequence in chunks for parallelism
        - Intra-chunk: parallel computation within each chunk
        - Inter-chunk: sequential state propagation via scan

    Inference (recurrent):
        - Single-step state update
        - h_t = decay * h_{t-1} + beta_t * (v_t ⊗ k_t)
        - o_t = h_t @ q_t

References:
    - Qwen3Next: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/
"""

import jax
import jax.numpy as jnp
from eformer.escale import with_sharding_constraint
from eformer.pytree import auto_pytree
from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule._interface import _decode_path
from ejkernel.kernels._xla.ragged_gated_delta_rule._xla_impl_fwd import _ragged_gdr_chunked_prefill
from ejkernel.modules import gated_delta_rule, ragged_gated_delta_rule
from ejkernel.modules.operations.configs import GatedDeltaRuleConfig
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float

from easydel.caching import RecurrentCacheView
from easydel.layers.linear_attention._conv_state import apply_manual_depthwise_conv, shift_conv_state_left
from easydel.utils import is_inference_mode
from easydel.utils.helpers import check_bool_flag

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)


def _gdr_grouped_decode_kernel(
    query_ref,
    key_ref,
    value_ref,
    beta_ref,
    decay_ref,
    state_ref,
    output_ref,
    state_out_ref,
):
    """Pallas kernel body for a single grouped GDR (Gated Delta Rule) decode step.

    This kernel runs on TPU inside a ``pallas_call`` with grid ``(batch,)``.
    Each program instance processes one batch element entirely in VMEM,
    iterating over all ``num_k_heads`` key heads and, for each key head,
    over its ``expand_ratio`` value-head groups. For every (key-head,
    value-group) pair the kernel:

    1. Applies exponential decay to the recurrent state:
       ``s = s * exp(decay)``.
    2. Retrieves the key-value memory via contraction with the key:
       ``kv_mem = sum(k[:, None] * s, axis=0)``.
    3. Computes the gated delta update:
       ``delta = (v - kv_mem) * beta``; ``s = s + k[:, None] * delta[None, :]``.
    4. Produces the output by contracting the updated state with the query:
       ``out = sum(q[:, None] * s, axis=0)``.

    The query is pre-scaled by ``head_dim ** -0.5`` inside the kernel.

    All arithmetic is performed in the recurrent-state/runtime dtype so the
    grouped decode path matches the dtype policy of the standard GDR kernels.

    Args:
        query_ref: Query tensor ref, shape ``[1, num_k_heads, head_dim]``.
        key_ref: Key tensor ref, shape ``[1, num_k_heads, head_dim]``.
        value_ref: Value tensor ref, shape ``[1, num_k_heads, expand_ratio, value_dim]``.
        beta_ref: Gating coefficient ref, shape ``[1, num_k_heads, expand_ratio]``.
        decay_ref: Log-space decay ref, shape ``[1, num_k_heads, expand_ratio]``.
        state_ref: Input recurrent state ref,
            shape ``[1, num_v_heads, head_dim, value_dim]``.
        output_ref: Output tensor ref, shape ``[1, num_v_heads, value_dim]``.
        state_out_ref: Updated recurrent state ref,
            shape ``[1, num_v_heads, head_dim, value_dim]``.
    """
    num_k_heads = query_ref.shape[1]
    head_dim = query_ref.shape[2]
    expand_ratio = value_ref.shape[2]
    compute_dtype = state_ref.dtype
    scale = jnp.asarray(head_dim**-0.5, dtype=compute_dtype)

    for kh in range(num_k_heads):
        q = query_ref[0, kh, :].astype(compute_dtype) * scale  # [head_dim]
        k = key_ref[0, kh, :].astype(compute_dtype)  # [head_dim]

        vh_start = kh * expand_ratio
        # Load all expand_ratio value heads for this key head at once.
        s_all = state_ref[0, vh_start : vh_start + expand_ratio, :, :].astype(compute_dtype)
        d_all = decay_ref[0, kh, :].astype(jnp.float32)
        beta_all = beta_ref[0, kh, :].astype(jnp.float32)
        v_all = value_ref[0, kh, :, :].astype(compute_dtype)

        # Decay, kv_mem contraction, delta update, output contraction.
        s_all = s_all * jnp.exp(d_all)[:, None, None].astype(compute_dtype)
        kv_mem_all = jnp.sum(k[None, :, None] * s_all, axis=1)
        delta_all = (v_all - kv_mem_all) * beta_all[:, None].astype(compute_dtype)
        s_all = s_all + k[None, :, None] * delta_all[:, None, :]
        out_all = jnp.sum(q[None, :, None] * s_all, axis=1)

        state_out_ref[0, vh_start : vh_start + expand_ratio, :, :] = s_all.astype(state_out_ref.dtype)
        output_ref[0, vh_start : vh_start + expand_ratio, :] = out_all.astype(output_ref.dtype)


def _fused_conv_decode_kernel(
    conv_state_ref,
    new_tokens_ref,
    kernel_ref,
    updated_state_ref,
    conv_output_ref,
):
    """Pallas kernel that fuses conv-state shift, depthwise convolution, and SiLU activation.

    This kernel is designed for TPU execution inside a ``pallas_call`` with
    grid ``(conv_dim_tiles,)``. It tiles over the ``conv_dim`` axis (in chunks
    determined by the caller's ``CONV_TILE`` parameter), processing all slots
    within each tile. This tiling strategy avoids the TPU block-alignment
    constraint that arises when tiling over ``num_slots`` (which may not be
    divisible by 8).

    For each tile the kernel performs three fused operations:

    1. **Shift**: Drops the oldest token from the conv state and appends the
       new token: ``new_state = concat(state[:, :, 1:], token[:, :, None], axis=2)``.
    2. **Depthwise conv**: Computes the dot product of the updated state with
       the per-channel kernel: ``conv_out = sum(new_state * kernel, axis=-1)``.
    3. **SiLU activation**: Applies the SiLU (Swish) non-linearity:
       ``conv_out = conv_out * sigmoid(conv_out)``.

    All computation is done in float32; results are cast to the output ref dtypes.

    Args:
        conv_state_ref: Current conv state ref, shape ``[num_slots, tile, d_conv]``.
        new_tokens_ref: New token embeddings ref, shape ``[num_slots, tile]``.
        kernel_ref: Depthwise conv kernel ref, shape ``[tile, d_conv]``.
        updated_state_ref: Output updated conv state ref,
            shape ``[num_slots, tile, d_conv]``.
        conv_output_ref: Output conv result ref, shape ``[num_slots, tile]``.
    """
    _tile_idx = pl.program_id(0)
    state = conv_state_ref[:, :, :].astype(jnp.float32)
    token = new_tokens_ref[:, :].astype(jnp.float32)
    kern = kernel_ref[:, :].astype(jnp.float32)

    new_state = jnp.concatenate([state[:, :, 1:], token[:, :, None]], axis=2)
    conv_out = jnp.sum(new_state * kern[None, :, :], axis=-1)
    conv_out = conv_out * jax.nn.sigmoid(conv_out)

    updated_state_ref[:, :, :] = new_state.astype(updated_state_ref.dtype)
    conv_output_ref[:, :] = conv_out.astype(conv_output_ref.dtype)


@auto_pytree
class GatedDeltaRuleOutput(AttentionOutput):
    """Output container for GatedDeltaRule operation.

    Extends AttentionOutput with recurrent state fields needed for
    hybrid attention models.

    Attributes:
        attention_outputs: Output tensor [batch, seq_len, num_heads, head_dim]
        attention_weights: Always None for linear attention (no explicit weights)
        conv_state: Updated convolution state [batch, d_inner, d_conv]
        recurrent_state: Updated recurrent state [batch, num_heads, head_dim, d_state]
    """

    conv_state: Float[Array, "batch d_inner d_conv"] | None = None
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None


@OperationRegistry.register
class GatedDeltaRuleOp(OperationImpl):
    """Gated Delta Rule linear attention operation.

    Implements the gated delta rule mechanism for efficient linear attention:
    - Training mode: Uses chunked algorithm for O(N) complexity
    - Inference mode: Uses recurrent update for single-token generation

    The gated delta rule updates state as:
        h_t = decay * h_{t-1} + beta_t * (v_t ⊗ k_t)
        o_t = h_t @ q_t

    Where:
    - beta_t is a learned gating signal
    - decay is an optional forgetting factor
    - v_t ⊗ k_t is the outer product

    Registered under the name "gated_delta_rule".

    Example:
        >>> from easydel.operations import OperationMetadata, OperationRegistry
        >>> metadata = OperationMetadata(runtime_dtype=jnp.float16)
        >>> gdr_op = OperationRegistry.create("gated_delta_rule", metadata)
        >>> output = gdr_op(
        ...     query=query,
        ...     key=key,
        ...     value=value,
        ...     beta=beta,
        ...     decay=decay,
        ...     chunk_size=64,
        ... )
    """

    def grouped_gdr_decode(
        self,
        query: Float[Array, "batch num_k_heads head_dim"],
        key: Float[Array, "batch num_k_heads head_dim"],
        value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
        beta: Float[Array, "batch num_k_heads expand_ratio"],
        decay: Float[Array, "batch num_k_heads expand_ratio"] | None,
        recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
    ) -> tuple[
        Float[Array, "batch num_v_heads value_dim"],
        Float[Array, "batch num_v_heads head_dim value_dim"],
    ]:
        """Perform a single grouped GDR decode step with pre-reshaped inputs.

        This is the production entry point for grouped decode. It accepts
        tensors that have already been reshaped from the standard ``[batch, 1,
        num_heads, dim]`` layout into the grouped layout where key/query heads
        are separated from the value-head expansion factor.

        The method currently dispatches to the pure-JAX implementation
        (``grouped_gdr_decode_jax``), which XLA fuses effectively on all
        backends. A separate ``grouped_gdr_decode_shard_map_pallas`` path is
        available for TPU benchmarking via ``shard_map`` + Pallas but is not
        used in production due to equivalent or worse latency.

        Args:
            query: Query tensor, shape ``[batch, num_k_heads, head_dim]``.
                Already squeezed from the seq_len=1 dimension.
            key: Key tensor, shape ``[batch, num_k_heads, head_dim]``.
            value: Value tensor reshaped to group layout,
                shape ``[batch, num_k_heads, expand_ratio, value_dim]``.
            beta: Gating coefficients per value-head group,
                shape ``[batch, num_k_heads, expand_ratio]``.
            decay: Optional log-space decay per group,
                shape ``[batch, num_k_heads, expand_ratio]``. Pass ``None``
                to skip decay (equivalent to decay=0).
            recurrent_state: Current recurrent state,
                shape ``[batch, num_v_heads, head_dim, value_dim]``.

        Returns:
            A tuple of:
            - output: Attention output, shape ``[batch, num_v_heads, value_dim]``.
            - new_state: Updated recurrent state,
              shape ``[batch, num_v_heads, head_dim, value_dim]``.
        """
        runtime_dtype = self.metadata.runtime_dtype
        return GatedDeltaRuleOp.grouped_gdr_decode_jax(
            query.astype(runtime_dtype),
            key.astype(runtime_dtype),
            value.astype(runtime_dtype),
            beta.astype(runtime_dtype),
            decay.astype(runtime_dtype) if decay is not None else None,
            recurrent_state.astype(runtime_dtype),
        )

    def grouped_gdr_decode_shard_map_pallas(
        self,
        query: Float[Array, "batch num_k_heads head_dim"],
        key: Float[Array, "batch num_k_heads head_dim"],
        value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
        beta: Float[Array, "batch num_k_heads expand_ratio"],
        decay: Float[Array, "batch num_k_heads expand_ratio"] | None,
        recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
    ) -> tuple[
        Float[Array, "batch num_v_heads value_dim"],
        Float[Array, "batch num_v_heads head_dim value_dim"],
    ]:
        """Reference TPU path that wraps the Pallas grouped decode kernel with ``shard_map``.

        This method is intended for benchmarking and reference rather than
        production use. It inspects the operation metadata to determine the
        tensor-parallel (TP) axis from the mesh and sharding specifications.
        If TP sharding is available (``tp_size > 1``), it invokes
        ``grouped_gdr_decode_pallas`` under ``jax.shard_map`` so each device
        shard runs the Pallas kernel independently on its local slice.

        Falls back to ``grouped_gdr_decode_jax`` when any of the following
        conditions hold:
        - The backend is not TPU.
        - ``decay`` is ``None`` (the Pallas kernel requires explicit decay).
        - No mesh is configured in the operation metadata.
        - TP size is 1 (no benefit from shard_map wrapping).
        - The grouped decode tensors are not float32 (the reference Pallas
          kernel currently only lowers reliably in float32).

        Args:
            query: Query tensor, shape ``[batch, num_k_heads, head_dim]``.
            key: Key tensor, shape ``[batch, num_k_heads, head_dim]``.
            value: Value tensor in group layout,
                shape ``[batch, num_k_heads, expand_ratio, value_dim]``.
            beta: Gating coefficients,
                shape ``[batch, num_k_heads, expand_ratio]``.
            decay: Log-space decay factors,
                shape ``[batch, num_k_heads, expand_ratio]``. Must not be
                ``None`` for the Pallas path to activate.
            recurrent_state: Current recurrent state,
                shape ``[batch, num_v_heads, head_dim, value_dim]``.

        Returns:
            A tuple of (output, new_state) with the same shapes as
            ``grouped_gdr_decode``.
        """
        mesh = self.metadata.mesh
        if (
            jax.default_backend() != "tpu"
            or decay is None
            or mesh is None
            or query.dtype != jnp.float32
            or key.dtype != jnp.float32
            or value.dtype != jnp.float32
            or beta.dtype != jnp.float32
            or recurrent_state.dtype != jnp.float32
        ):
            runtime_dtype = self.metadata.runtime_dtype
            return GatedDeltaRuleOp.grouped_gdr_decode_jax(
                query.astype(runtime_dtype),
                key.astype(runtime_dtype),
                value.astype(runtime_dtype),
                beta.astype(runtime_dtype),
                decay.astype(runtime_dtype) if decay is not None else None,
                recurrent_state.astype(runtime_dtype),
            )

        mode = self.get_mode(query=jnp.expand_dims(query, 1), BTHD=True)
        shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")
        tp_axis = shardings_bthd.query[2] if shardings_bthd.query is not None else None

        tp_size = 1
        if tp_axis is not None:
            axes = (tp_axis,) if isinstance(tp_axis, str) else tp_axis
            for ax in axes:
                tp_size *= mesh.shape.get(ax, 1) if isinstance(mesh.shape, dict) else 1

        if tp_size <= 1 or tp_axis is None:
            runtime_dtype = self.metadata.runtime_dtype
            return GatedDeltaRuleOp.grouped_gdr_decode_jax(
                query.astype(runtime_dtype),
                key.astype(runtime_dtype),
                value.astype(runtime_dtype),
                beta.astype(runtime_dtype),
                decay.astype(runtime_dtype) if decay is not None else None,
                recurrent_state.astype(runtime_dtype),
            )

        qk_spec = PartitionSpec(None, tp_axis, None)
        v_spec = PartitionSpec(None, tp_axis, None, None)
        bd_spec = PartitionSpec(None, tp_axis, None)
        state_spec = PartitionSpec(None, tp_axis, None, None)
        out_spec = PartitionSpec(None, tp_axis, None)

        @jax.named_scope("grouped_gdr_decode_pallas")
        def _run(q, k, v, b, d, s):
            """Execute the Pallas-backed grouped GDR decode on a single shard.

            This thin wrapper is used as the per-shard function for
            ``jax.shard_map``, delegating to the static Pallas kernel
            while keeping the named scope for profiling.

            Args:
                q: Query tensor shard.
                k: Key tensor shard.
                v: Value tensor shard.
                b: Beta tensor shard.
                d: Decay tensor shard.
                s: Recurrent state tensor shard.

            Returns:
                A tuple of (output, new_state) produced by the Pallas
                grouped GDR decode kernel.
            """
            return GatedDeltaRuleOp.grouped_gdr_decode_pallas(q, k, v, b, d, s)

        output, new_state = jax.shard_map(
            _run,
            mesh=mesh,
            in_specs=(qk_spec, qk_spec, v_spec, bd_spec, bd_spec, state_spec),
            out_specs=(out_spec, state_spec),
            check_vma=False,
        )(query, key, value, beta, decay, recurrent_state)

        return output, new_state

    @staticmethod
    def grouped_gdr_decode_jax(
        query: Float[Array, "batch num_k_heads head_dim"],
        key: Float[Array, "batch num_k_heads head_dim"],
        value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
        beta: Float[Array, "batch num_k_heads expand_ratio"],
        decay: Float[Array, "batch num_k_heads expand_ratio"] | None,
        recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
    ) -> tuple[
        Float[Array, "batch num_v_heads value_dim"],
        Float[Array, "batch num_v_heads head_dim value_dim"],
    ]:
        """Pure JAX implementation of the grouped GDR decode step.

        This is the default backend for ``grouped_gdr_decode`` and works on all
        JAX platforms (CPU, GPU, TPU). It operates on pre-reshaped inputs where
        the key/query head dimension is separated from the value-head expansion
        factor.

        Algorithm:
            1. Scale the query by ``head_dim ** -0.5``.
            2. Reshape the flat recurrent state into grouped layout:
               ``[batch, num_k_heads, expand_ratio, head_dim, value_dim]``.
            3. If decay is provided, apply exponential decay:
               ``state *= exp(decay)``.
            4. Compute the key-value memory by contracting the state with the
               key along the ``head_dim`` axis.
            5. Compute the gated delta: ``delta = (value - kv_mem) * beta``.
            6. Update the state: ``state += key * delta`` (outer product).
            7. Produce the output by contracting the updated state with the
               query along the ``head_dim`` axis.
            8. Reshape outputs back to the flat ``num_v_heads`` layout.

        All internal computation is performed in the dtype of
        ``recurrent_state`` so grouped decode tracks the runtime/cache dtype
        used by the rest of the GDR implementation.

        Args:
            query: Query tensor, shape ``[batch, num_k_heads, head_dim]``.
            key: Key tensor, shape ``[batch, num_k_heads, head_dim]``.
            value: Value tensor in group layout,
                shape ``[batch, num_k_heads, expand_ratio, value_dim]``.
            beta: Gating coefficients,
                shape ``[batch, num_k_heads, expand_ratio]``.
            decay: Optional log-space decay,
                shape ``[batch, num_k_heads, expand_ratio]``.
            recurrent_state: Current recurrent state,
                shape ``[batch, num_v_heads, head_dim, value_dim]``.

        Returns:
            A tuple of:
            - output: shape ``[batch, num_v_heads, value_dim]``.
            - new_state: shape ``[batch, num_v_heads, head_dim, value_dim]``.
        """
        batch, num_k_heads, head_dim = query.shape
        expand_ratio = value.shape[2]

        compute_dtype = recurrent_state.dtype
        query = query.astype(compute_dtype)
        key = key.astype(compute_dtype)

        scale = jnp.asarray(head_dim**-0.5, dtype=compute_dtype)
        query = query * scale

        value_dim = value.shape[-1]
        num_v_heads = num_k_heads * expand_ratio

        # Reshape state to grouped 5D layout (free view, no copy).
        # key/query stay at [batch, num_k_heads, head_dim] — no repeat needed.
        gs = recurrent_state.astype(compute_dtype).reshape(batch, num_k_heads, expand_ratio, head_dim, value_dim)
        value_c = value.astype(compute_dtype)
        beta_c = beta.astype(compute_dtype)

        # Decay: broadcast [batch, num_k_heads, expand_ratio] over [head_dim, value_dim].
        if decay is not None:
            gs = gs * jnp.exp(decay.astype(compute_dtype))[:, :, :, None, None]

        # kv_mem: contract grouped_state with key over head_dim.
        # einsum avoids materialising the broadcast product.
        kv_mem = jnp.einsum("bkehv,bkh->bkev", gs, key)

        # Gated delta update.
        delta = (value_c - kv_mem) * beta_c[:, :, :, None]

        # State update: rank-1 outer-product per (key-head, expand-group).
        gs = gs + jnp.einsum("bkh,bkev->bkehv", key, delta)

        # Output: contract updated state with query over head_dim.
        output = jnp.einsum("bkehv,bkh->bkev", gs, query)

        return (
            output.reshape(batch, num_v_heads, value_dim).astype(recurrent_state.dtype),
            gs.reshape(batch, num_v_heads, head_dim, value_dim).astype(recurrent_state.dtype),
        )

    @staticmethod
    def grouped_gdr_decode_pallas(
        query: Float[Array, "batch num_k_heads head_dim"],
        key: Float[Array, "batch num_k_heads head_dim"],
        value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
        beta: Float[Array, "batch num_k_heads expand_ratio"],
        decay: Float[Array, "batch num_k_heads expand_ratio"],
        recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
    ) -> tuple[
        Float[Array, "batch num_v_heads value_dim"],
        Float[Array, "batch num_v_heads head_dim value_dim"],
    ]:
        """Pallas-accelerated grouped GDR decode step targeting TPU.

        This static method wraps ``_gdr_grouped_decode_kernel`` in a
        ``pallas_call`` with a ``(batch,)`` grid, where each program instance
        handles one batch element. The entire grouped state update runs in
        TPU VMEM, avoiding materialization of the 5D intermediate tensor
        ``[batch, num_k_heads, expand_ratio, head_dim, value_dim]`` that the
        JAX path requires.

        For multi-device (tensor-parallel) execution, this method must be
        wrapped in ``jax.shard_map`` so that each device processes its local
        head shard. See ``grouped_gdr_decode_shard_map_pallas`` for an
        example of such wrapping.

        Note:
            Unlike ``grouped_gdr_decode_jax``, this method requires ``decay``
            to be non-None (the Pallas kernel always applies decay) and is
            intended for float32 benchmarking inputs.

        Args:
            query: Query tensor, shape ``[batch, num_k_heads, head_dim]``.
            key: Key tensor, shape ``[batch, num_k_heads, head_dim]``.
            value: Value tensor in group layout,
                shape ``[batch, num_k_heads, expand_ratio, value_dim]``.
            beta: Gating coefficients,
                shape ``[batch, num_k_heads, expand_ratio]``.
            decay: Log-space decay factors (required),
                shape ``[batch, num_k_heads, expand_ratio]``.
            recurrent_state: Current recurrent state,
                shape ``[batch, num_v_heads, head_dim, value_dim]``.

        Returns:
            A tuple of:
            - output: shape ``[batch, num_v_heads, value_dim]``.
            - new_state: shape ``[batch, num_v_heads, head_dim, value_dim]``.
        """
        batch, num_k_heads, head_dim = query.shape
        expand_ratio = value.shape[2]
        value_dim = value.shape[3]
        num_v_heads = num_k_heads * expand_ratio

        out_output = jax.ShapeDtypeStruct((batch, num_v_heads, value_dim), dtype=recurrent_state.dtype)
        out_state = jax.ShapeDtypeStruct((batch, num_v_heads, head_dim, value_dim), dtype=recurrent_state.dtype)

        output, new_state = pl.pallas_call(
            _gdr_grouped_decode_kernel,
            grid=(batch,),
            in_specs=[
                pl.BlockSpec((1, num_k_heads, head_dim), lambda b: (b, 0, 0)),
                pl.BlockSpec((1, num_k_heads, head_dim), lambda b: (b, 0, 0)),
                pl.BlockSpec((1, num_k_heads, expand_ratio, value_dim), lambda b: (b, 0, 0, 0)),
                pl.BlockSpec((1, num_k_heads, expand_ratio), lambda b: (b, 0, 0)),
                pl.BlockSpec((1, num_k_heads, expand_ratio), lambda b: (b, 0, 0)),
                pl.BlockSpec((1, num_v_heads, head_dim, value_dim), lambda b: (b, 0, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, num_v_heads, value_dim), lambda b: (b, 0, 0)),
                pl.BlockSpec((1, num_v_heads, head_dim, value_dim), lambda b: (b, 0, 0, 0)),
            ],
            out_shape=[out_output, out_state],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        )(query, key, value, beta, decay, recurrent_state)

        return output, new_state

    @staticmethod
    def fused_conv_decode(
        conv_state: Float[Array, "num_slots conv_dim d_conv"],
        new_tokens: Float[Array, "num_slots conv_dim"],
        kernel: Float[Array, "conv_dim d_conv"],
        *,
        output_dtype: jnp.dtype,
    ) -> tuple[
        Float[Array, "num_slots conv_dim d_conv"],
        Float[Array, "num_slots conv_dim"],
    ]:
        """Fused conv-state shift, depthwise convolution, and SiLU activation.

        This is the production entry point for the fused convolution decode
        step used during single-token generation. It performs three operations
        in sequence:

        1. **Shift**: Slides the conv state window left by one position,
           discarding the oldest token and appending ``new_tokens`` at the end.
        2. **Depthwise conv**: Applies a per-channel 1-D convolution of the
           updated state against ``kernel`` to produce the conv output.
        3. **SiLU**: Applies the SiLU (Swish) activation element-wise.

        Dispatches to the pure-JAX implementation (``fused_conv_decode_jax``)
        because XLA fuses the shift + conv + SiLU chain effectively, making
        the Pallas variant (``fused_conv_decode_pallas``) slower in practice.
        The Pallas variant is retained for reference and TPU microbenchmarking.

        Args:
            conv_state: Current conv state for each slot,
                shape ``[num_slots, conv_dim, d_conv]``.
            new_tokens: New token embeddings to append,
                shape ``[num_slots, conv_dim]``.
            kernel: Depthwise convolution kernel weights,
                shape ``[conv_dim, d_conv]``.
            output_dtype: Desired dtype for the convolution output tensor.

        Returns:
            A tuple of:
            - updated_state: The shifted conv state,
              shape ``[num_slots, conv_dim, d_conv]``.
            - conv_output: The activated convolution result,
              shape ``[num_slots, conv_dim]``.
        """
        return GatedDeltaRuleOp.fused_conv_decode_jax(
            conv_state,
            new_tokens,
            kernel,
            output_dtype=output_dtype,
        )

    @staticmethod
    def fused_conv_decode_jax(
        conv_state: Float[Array, "num_slots conv_dim d_conv"],
        new_tokens: Float[Array, "num_slots conv_dim"],
        kernel: Float[Array, "conv_dim d_conv"],
        *,
        output_dtype: jnp.dtype,
        activation: callable = jax.nn.silu,
    ) -> tuple[
        Float[Array, "num_slots conv_dim d_conv"],
        Float[Array, "num_slots conv_dim"],
    ]:
        """Pure JAX implementation of fused conv-state shift, depthwise conv, and activation.

        Performs the same three-step operation as ``fused_conv_decode`` using
        standard JAX primitives, allowing XLA to fuse and optimize the
        computation graph on any backend.

        The shift is performed by ``shift_conv_state_left`` which concatenates
        the state's trailing ``d_conv - 1`` columns with the new token column.
        The depthwise conv and activation are handled by
        ``apply_manual_depthwise_conv``.

        Args:
            conv_state: Current conv state, shape ``[num_slots, conv_dim, d_conv]``.
            new_tokens: New token embeddings, shape ``[num_slots, conv_dim]``.
            kernel: Depthwise conv kernel, shape ``[conv_dim, d_conv]``.
            output_dtype: Desired dtype for the convolution output.
            activation: Activation function to apply after convolution.
                Defaults to ``jax.nn.silu``.

        Returns:
            A tuple of:
            - updated_state: Shifted conv state,
              shape ``[num_slots, conv_dim, d_conv]``.
            - conv_output: Activated convolution output,
              shape ``[num_slots, conv_dim]``.
        """
        updated_state = shift_conv_state_left(conv_state, new_tokens)
        conv_output = apply_manual_depthwise_conv(
            updated_state,
            kernel,
            output_dtype=output_dtype,
            activation=activation,
        )
        return updated_state, conv_output

    @staticmethod
    def fused_conv_decode_pallas(
        conv_state: Float[Array, "num_slots conv_dim d_conv"],
        new_tokens: Float[Array, "num_slots conv_dim"],
        kernel: Float[Array, "conv_dim d_conv"],
        *,
        output_dtype: jnp.dtype,
    ) -> tuple[
        Float[Array, "num_slots conv_dim d_conv"],
        Float[Array, "num_slots conv_dim"],
    ]:
        """Pallas-accelerated fused conv-state update targeting TPU.

        This static method wraps ``_fused_conv_decode_kernel`` in a
        ``pallas_call`` that tiles over the ``conv_dim`` axis in chunks of
        ``CONV_TILE`` (default 128, must divide ``conv_dim``). All slots are
        processed within each tile, which satisfies TPU block-alignment
        constraints (the slot axis is not tiled, avoiding issues when
        ``num_slots`` is not divisible by 8).

        In practice, XLA's fusion of the JAX-based implementation
        (``fused_conv_decode_jax``) matches or exceeds this kernel's
        performance, so this variant is kept for reference and
        microbenchmarking rather than production use.

        Args:
            conv_state: Current conv state, shape ``[num_slots, conv_dim, d_conv]``.
            new_tokens: New token embeddings, shape ``[num_slots, conv_dim]``.
            kernel: Depthwise conv kernel, shape ``[conv_dim, d_conv]``.
            output_dtype: Desired dtype for the convolution output.

        Returns:
            A tuple of:
            - updated_state: Shifted conv state,
              shape ``[num_slots, conv_dim, d_conv]``.
            - conv_output: Activated convolution output,
              shape ``[num_slots, conv_dim]``.
        """
        num_slots, conv_dim, d_conv = conv_state.shape
        CONV_TILE = min(conv_dim, 128)
        assert conv_dim % CONV_TILE == 0, f"conv_dim={conv_dim} not divisible by {CONV_TILE}"
        num_tiles = conv_dim // CONV_TILE

        out_state_shape = jax.ShapeDtypeStruct((num_slots, conv_dim, d_conv), dtype=conv_state.dtype)
        out_conv_shape = jax.ShapeDtypeStruct((num_slots, conv_dim), dtype=output_dtype)

        updated_state, conv_output = pl.pallas_call(
            _fused_conv_decode_kernel,
            grid=(num_tiles,),
            in_specs=[
                pl.BlockSpec((num_slots, CONV_TILE, d_conv), lambda t: (0, t, 0)),
                pl.BlockSpec((num_slots, CONV_TILE), lambda t: (0, t)),
                pl.BlockSpec((CONV_TILE, d_conv), lambda t: (t, 0)),
            ],
            out_specs=[
                pl.BlockSpec((num_slots, CONV_TILE, d_conv), lambda t: (0, t, 0)),
                pl.BlockSpec((num_slots, CONV_TILE), lambda t: (0, t)),
            ],
            out_shape=[out_state_shape, out_conv_shape],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        )(conv_state, new_tokens, kernel)

        return updated_state, conv_output

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Returns the registered name of this operation.

        Returns:
            Tuple of names: ("gated_delta_rule", "gdr")
        """
        return ("gated_delta_rule", "gdr")

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for GatedDeltaRuleOp.

        GDR is a recurrent/linear attention mechanism that requires:
        - Basic metadata plus state management fields
        - Recurrent or Hybrid cache types for state persistence
        - Uses RecurrentCacheView for state management
        """
        return (
            RequirementsBuilder("gated_delta_rule")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.POSITIONS
                | MetadataField.HAS_INITIAL_STATE
                | MetadataField.STATE_INDICES
            )
            .optional_metadata(MetadataField.LOGITS_INDICES)
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .use_cache_view(RecurrentCacheView)
            .build()
        )

    @jax.named_scope("easydel-gated-delta-rule-native")
    def forward_native(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_heads head_dim"],
        value: Float[Array, "batch seq_len num_heads d_state"],
        beta: Float[Array, "batch seq_len num_heads head_dim"],
        decay: Float[Array, "num_heads head_dim"] | None = None,
        conv_state: Float[Array, "batch d_inner d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        use_qk_l2norm: bool = True,
        **kwargs,
    ) -> GatedDeltaRuleOutput:
        """Forward pass for gated delta rule attention via ejkernel.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, d_state]
            beta: Gating tensor [batch, seq_len, num_heads, head_dim]
            decay: Optional decay factors [num_heads, head_dim]
            conv_state: Optional convolution state (passed through, not used here)
            recurrent_state: Optional recurrent state for inference
            chunk_size: Chunk size for training mode (default: 64)
            **kwargs:
                - autotune_chunk_size: Optional bool kept for API compatibility.
                  ejkernel module integration handles autotune policy internally.
                - autotune_chunk_candidates: Optional list/tuple kept for API
                  compatibility; currently ignored in the ejkernel module path.

        Returns:
            GatedDeltaRuleOutput containing attention outputs and updated states
        """
        seq_len = query.shape[1]
        is_inference = seq_len == 1
        kernel_cfg = self.metadata.get_operation_config("gated_delta_rule")
        if kernel_cfg is None and not is_inference:
            adaptive_chunk = min(max(16, seq_len), 64)
            adaptive_chunk = 1 << (adaptive_chunk.bit_length() - 1) if isinstance(adaptive_chunk, int) else 64
            kernel_cfg = GatedDeltaRuleConfig(chunk_size=adaptive_chunk)

        mode = self.get_mode(query=query, BTHD=True)
        shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")

        runtime_dtype = self.metadata.runtime_dtype
        query = query.astype(runtime_dtype)
        key = key.astype(runtime_dtype)
        value = value.astype(runtime_dtype)

        beta = beta.astype(runtime_dtype)
        if beta.ndim == 4 and beta.shape[-1] == 1:
            beta = beta[..., 0]

        if decay is not None:
            decay = decay.astype(runtime_dtype)
            if decay.ndim == 4 and decay.shape[-1] == 1:
                decay = decay[..., 0]

        if recurrent_state is not None:
            recurrent_state = recurrent_state.astype(runtime_dtype)

        query_sharding = self.create_stable_sharding(
            shardings_bthd.query,
            tensor=query,
            preserved_indices=[0, 2],
        )
        key_sharding = self.create_stable_sharding(
            shardings_bthd.key,
            tensor=key,
            preserved_indices=[0, 2],
        )
        value_sharding = self.create_stable_sharding(
            shardings_bthd.value,
            tensor=value,
            preserved_indices=[0, 2],
        )
        beta_source = PartitionSpec(
            shardings_bthd.query[0],
            shardings_bthd.query[1],
            shardings_bthd.query[2],
        )
        beta_sharding = self.create_stable_sharding(
            beta_source,
            tensor=beta,
            preserved_indices=[0, 2],
        )
        decay_sharding = self.create_stable_sharding(
            beta_source,
            dep=decay,
            tensor=decay,
            preserved_indices=[0, 2],
        )
        state_source = None
        if query_sharding is not None:
            state_source = PartitionSpec(
                query_sharding[0],
                query_sharding[2],
                None,
                None,
            )
        state_in_sharding = self.create_stable_sharding(
            state_source,
            dep=recurrent_state,
            tensor=recurrent_state,
        )
        state_out_sharding = self.create_stable_sharding(
            state_source,
            tensor=recurrent_state,
        )
        output_sharding = self.create_stable_sharding(
            shardings_bthd.output,
            tensor=query,
            preserved_indices=[0, 2],
        )

        in_specs = None
        out_specs = None
        if self.metadata.mesh is not None:
            in_specs = (
                query_sharding,
                key_sharding,
                value_sharding,
                beta_sharding,
                decay_sharding,
                state_in_sharding,
            )
            out_specs = (output_sharding, state_out_sharding)

        platform = None
        if jax.default_backend() == "tpu" and kernel_cfg is None:
            if check_bool_flag("EASYDEL_GDR_XLA", False):
                platform = "xla"
            else:
                platform = "pallas"

        use_chunked_gdr = check_bool_flag("EASYDEL_GDR_CHUNKED", False) and not is_inference_mode()
        outputs, new_recurrent_state = gated_delta_rule(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            use_qk_l2norm=use_qk_l2norm,
            use_chunked=use_chunked_gdr,
            return_state=True,
            cfg=kernel_cfg,
            mesh=self.metadata.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            platform=platform,
        )

        if self.metadata.mesh is not None:
            with self.metadata.mesh:
                outputs = with_sharding_constraint(arr=outputs, sharding=shardings_bthd.output)

        return GatedDeltaRuleOutput(
            attention_outputs=outputs,
            attention_weights=None,
            conv_state=conv_state,
            recurrent_state=new_recurrent_state,
        )

    def forward_ragged(
        self,
        query: Float[Array, "total_tokens num_heads qk_head_dim"],
        key: Float[Array, "total_tokens num_heads qk_head_dim"],
        value: Float[Array, "total_tokens num_heads v_head_dim"],
        beta: Float[Array, "total_tokens num_heads"],
        decay: Float[Array, "total_tokens num_heads"] | None,
        recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
        query_start_loc: jax.Array,
        state_indices: jax.Array,
        use_qk_l2norm: bool = True,
        chunk_size: int = 64,
    ) -> GatedDeltaRuleOutput:
        """Ragged GDR forward for packed continuous-batching inference.

        Processes variable-length sequences in a flat token stream using
        ejkernel's ragged_gated_delta_rule. Handles both decode (seq_len=1)
        and prefill (seq_len>1) requests in a single fused call.

        This method is intended for eSurge inference mode where multiple
        requests with different sequence lengths are packed together.

        Args:
            query: Flat queries, shape (total_tokens, num_heads, qk_head_dim).
                For grouped-head models, Q/K heads must already be expanded
                to match num_v_heads before calling.
            key: Flat keys, shape (total_tokens, num_heads, qk_head_dim).
            value: Flat values, shape (total_tokens, num_heads, v_head_dim).
            beta: Per-token gating coefficients, shape (total_tokens, num_heads).
            decay: Per-token log-space decay, shape (total_tokens, num_heads),
                or None to skip decay.
            recurrent_state: Global state pool, shape
                (num_slots, num_heads, qk_head_dim, v_head_dim).
            query_start_loc: CSR-style cumulative token offsets per request,
                shape (num_requests + 1,).
            state_indices: Request-to-slot mapping, shape (num_requests,).
            use_qk_l2norm: Whether to L2-normalize queries and keys.
            chunk_size: Chunk size for the prefill path.

        Returns:
            GatedDeltaRuleOutput with attention_outputs (total_tokens, num_heads, v_head_dim)
            and recurrent_state (num_slots, num_heads, qk_head_dim, v_head_dim).
        """
        runtime_dtype = self.metadata.runtime_dtype
        query = query.astype(runtime_dtype)
        key = key.astype(runtime_dtype)
        value = value.astype(runtime_dtype)
        beta = beta.astype(runtime_dtype)
        if decay is not None:
            decay = decay.astype(runtime_dtype)
        else:
            decay = jnp.zeros_like(beta)
        recurrent_state = recurrent_state.astype(runtime_dtype)

        mesh = self.metadata.mesh
        if mesh is not None:

            mode = self.get_mode(query=jnp.expand_dims(query, 0), BTHD=False)
            shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")
            head_axis = shardings_bthd.query[2] if shardings_bthd.query is not None else None

            token_head_spec = PartitionSpec(None, head_axis, None)
            beta_spec = PartitionSpec(None, head_axis)
            state_spec = PartitionSpec(None, head_axis, None, None)
            PartitionSpec()
            Ps = PartitionSpec

            @jax.named_scope("ragged_gdr_decode_shard_map")
            def _decode_shard(q, k, v, b, d, s, si):
                return _decode_path(q, k, v, b, d, s, si, use_qk_l2norm)

            decode_shard_fn = jax.shard_map(
                _decode_shard,
                mesh=mesh,
                in_specs=(token_head_spec, token_head_spec, token_head_spec, beta_spec, beta_spec, state_spec, Ps()),
                out_specs=(token_head_spec, state_spec),
                check_vma=False,
            )

            _chunk_size = chunk_size
            _use_l2norm = use_qk_l2norm

            @jax.named_scope("ragged_gdr_prefill_shard_map")
            def _prefill_shard(q, k, v, b, d, s, qsl, si):
                new_s, out = _ragged_gdr_chunked_prefill(
                    q,
                    k,
                    v,
                    b,
                    d,
                    s,
                    qsl,
                    si,
                    _chunk_size,
                    _use_l2norm,
                )
                return out, new_s

            prefill_shard_fn = jax.shard_map(
                _prefill_shard,
                mesh=mesh,
                in_specs=(
                    token_head_spec,
                    token_head_spec,
                    token_head_spec,
                    beta_spec,
                    beta_spec,
                    state_spec,
                    Ps(),
                    Ps(),
                ),
                out_specs=(token_head_spec, state_spec),
                check_vma=False,
            )

            seq_lengths = query_start_loc[1:] - query_start_loc[:-1]
            is_all_decode = jnp.all(seq_lengths <= 1)

            num_tokens = query.shape[0]
            num_si = state_indices.shape[0]
            if num_tokens > num_si:
                decode_state_indices = jnp.pad(state_indices, (0, num_tokens - num_si))
            elif num_tokens < num_si:
                decode_state_indices = state_indices[:num_tokens]
            else:
                decode_state_indices = state_indices

            def _run_decode(_):
                return decode_shard_fn(
                    query,
                    key,
                    value,
                    beta,
                    decay,
                    recurrent_state,
                    decode_state_indices,
                )

            def _run_prefill(_):
                return prefill_shard_fn(
                    query,
                    key,
                    value,
                    beta,
                    decay,
                    recurrent_state,
                    query_start_loc,
                    state_indices,
                )

            output, new_state = jax.lax.cond(
                is_all_decode,
                _run_decode,
                _run_prefill,
                operand=None,
            )
        else:
            output, new_state = ragged_gated_delta_rule(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=recurrent_state,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                chunk_size=chunk_size,
                use_qk_l2norm=use_qk_l2norm,
            )

        return GatedDeltaRuleOutput(
            attention_outputs=output,
            attention_weights=None,
            conv_state=None,
            recurrent_state=new_state,
        )

    def forward_tpu(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """TPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """GPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """CPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """CUDA forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """ROCm forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_heads head_dim"],
        value: Float[Array, "batch seq_len num_heads d_state"],
        beta: Float[Array, "batch seq_len num_heads head_dim"],
        decay: Float[Array, "num_heads head_dim"] | None = None,
        conv_state: Float[Array, "batch d_inner d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        use_qk_l2norm: bool = True,
        **kwargs,
    ) -> GatedDeltaRuleOutput:
        """Execute the gated delta rule operation.

        Dispatches to appropriate backend via parent __call__.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, d_state]
            beta: Gating tensor [batch, seq_len, num_heads, head_dim]
            decay: Optional decay factors [num_heads, head_dim]
            conv_state: Optional convolution state
            recurrent_state: Optional recurrent state
            chunk_size: Chunk size for training mode
            **kwargs:
                - autotune_chunk_size: API-compatible flag (ejkernel handles
                  autotune policy internally in this integration).
                - autotune_chunk_candidates: API-compatible argument,
                  currently ignored in this integration path.

        Returns:
            GatedDeltaRuleOutput with attention outputs and updated states
        """
        return super().__call__(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            use_qk_l2norm=use_qk_l2norm,
            **kwargs,
        )
