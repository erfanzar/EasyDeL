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

from collections.abc import Callable

import jax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from ejkernel.modules import gated_delta_rule, ragged_gated_delta_rule
from ejkernel.modules.operations import (
    fused_conv_decode as _ejkernel_fused_conv_decode,
)
from ejkernel.modules.operations import (
    gated_delta_rule_grouped_decode as _ejkernel_gated_delta_rule_grouped_decode,
)
from ejkernel.modules.operations.configs import GatedDeltaRuleConfig
from jaxtyping import Array, Float
from spectrax import with_sharding_constraint

from easydel.caching import RecurrentCacheView
from easydel.utils import is_inference_mode

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)


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

        This is the production entry point for grouped decode. It delegates to
        the eJKernel ``gated_delta_rule_grouped_decode`` public operation.

        Args:
            query: Query tensor, shape ``[batch, num_k_heads, head_dim]``.
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
        return _ejkernel_gated_delta_rule_grouped_decode(
            query.astype(runtime_dtype),
            key.astype(runtime_dtype),
            value.astype(runtime_dtype),
            beta.astype(runtime_dtype),
            decay.astype(runtime_dtype) if decay is not None else None,
            recurrent_state.astype(runtime_dtype),
            platform="xla",
        )

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

        This static helper is retained for backward compatibility and
        direct call sites that expect the grouped-head layout. It delegates
        to the eJKernel XLA reference implementation, which matches the
        original pure-JAX semantics exactly.

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
        return _ejkernel_gated_delta_rule_grouped_decode(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            platform="xla",
        )

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
        step used during single-token generation. It delegates to the eJKernel
        ``fused_conv_decode`` public operation.

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
        return _ejkernel_fused_conv_decode(
            conv_state,
            new_tokens,
            kernel,
            output_dtype=output_dtype,
            platform="xla",
        )

    @staticmethod
    def fused_conv_decode_jax(
        conv_state: Float[Array, "num_slots conv_dim d_conv"],
        new_tokens: Float[Array, "num_slots conv_dim"],
        kernel: Float[Array, "conv_dim d_conv"],
        *,
        output_dtype: jnp.dtype,
        activation: Callable[[jax.Array], jax.Array] | None = jax.nn.silu,
    ) -> tuple[
        Float[Array, "num_slots conv_dim d_conv"],
        Float[Array, "num_slots conv_dim"],
    ]:
        """Pure JAX implementation of fused conv-state shift, depthwise conv, and activation.

        This static helper is retained for backward compatibility. It delegates
        to the eJKernel XLA reference implementation.

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
        return _ejkernel_fused_conv_decode(
            conv_state,
            new_tokens,
            kernel,
            output_dtype=output_dtype,
            activation=activation,
            platform="xla",
        )

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
            conv_state: Optional convolution state (passed through unchanged into the
                returned output, not consumed by this method)
            recurrent_state: Optional recurrent state for inference
            use_qk_l2norm: Whether to L2-normalize queries and keys before the
                recurrence. Defaults to True.
            **kwargs: Additional keyword arguments. Recognized keys:

                - seg_ids: Optional segment ids used to pack multiple sequences
                  per row; disables the Pallas chunked path when present.
                - autotune_chunk_size: Optional bool kept for API compatibility.
                  ejkernel module integration handles autotune policy internally.
                - autotune_chunk_candidates: Optional list/tuple kept for API
                  compatibility; currently ignored in the ejkernel module path.

                Backend and chunk-size selection come from the
                ``gated_delta_rule`` operation config when one is provided.
                Without an explicit config, dense TPU training prefill uses
                the measured Pallas chunked path; non-TPU and segmented inputs
                use the exact XLA chunked path with an adaptive chunk size.

        Returns:
            GatedDeltaRuleOutput containing attention outputs and updated states.
            The ``conv_state`` field echoes the ``conv_state`` argument; the
            ``recurrent_state`` field holds the newly computed recurrent state.
        """
        seq_len = query.shape[1]
        is_inference = seq_len == 1
        kernel_cfg = self.metadata.get_operation_config("gated_delta_rule")
        seg_ids = kwargs.get("seg_ids", None)
        if kernel_cfg is None and not is_inference:
            if jax.default_backend() == "tpu" and seg_ids is None:
                kernel_cfg = GatedDeltaRuleConfig(
                    platform="pallas",
                    backend="tpu",
                    chunk_size=256,
                    use_chunked=True,
                    use_input_dtype_phase1_outputs=True,
                    use_input_dtype_state=recurrent_state is None,
                )
            else:
                adaptive_chunk = min(max(16, seq_len), 64)
                adaptive_chunk = 1 << (adaptive_chunk.bit_length() - 1) if isinstance(adaptive_chunk, int) else 64
                kernel_cfg = GatedDeltaRuleConfig(
                    platform="xla",
                    backend="any",
                    chunk_size=adaptive_chunk,
                    use_chunked=True,
                )
        use_chunked_gdr = bool(getattr(kernel_cfg, "use_chunked", True)) and not is_inference_mode()

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
        beta_source = jax.sharding.PartitionSpec(
            shardings_bthd.value[0],
            shardings_bthd.value[1],
            shardings_bthd.value[2],
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
        if value_sharding is not None:
            state_source = jax.sharding.PartitionSpec(
                value_sharding[0],
                value_sharding[2],
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
            tensor=value,
            preserved_indices=[0, 2],
        )

        seg_source = None
        if query_sharding is not None:
            seg_source = jax.sharding.PartitionSpec(query_sharding[0], None)
        seg_sharding = self.create_stable_sharding(
            seg_source,
            dep=seg_ids,
            tensor=seg_ids,
        )

        in_specs = None
        out_specs = None
        mesh = self.metadata.mesh
        output_constraint_mesh = mesh
        if mesh is not None:
            in_specs = (
                query_sharding,
                key_sharding,
                value_sharding,
                beta_sharding,
                decay_sharding,
                state_in_sharding,
            )
            # When packing, the shard_map wrapper appends seg_ids to its operands; supply the
            # matching trailing in_spec here so the wrapper never has to invent a PartitionSpec.
            if seg_ids is not None:
                in_specs = (*in_specs, seg_sharding)
            out_specs = (output_sharding, state_out_sharding)

        # ``seg_ids`` is positional-only (between ``decay`` and ``initial_state``) on the op.
        outputs, new_recurrent_state = gated_delta_rule(
            query,
            key,
            value,
            beta,
            decay,
            seg_ids,
            recurrent_state,
            use_qk_l2norm=use_qk_l2norm,
            use_chunked=use_chunked_gdr,
            return_state=True,
            cfg=kernel_cfg,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            platform=None,
        )

        if output_constraint_mesh is not None:
            with output_constraint_mesh:
                outputs = with_sharding_constraint(
                    arr=outputs,
                    sharding=shardings_bthd.output,
                    mesh=output_constraint_mesh,
                )

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
        platform = None
        in_specs = None
        out_specs = None
        if mesh is not None and jax.default_backend() == "tpu":
            mode = self.get_mode(query=jnp.expand_dims(query, 0), BTHD=False)
            shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")
            head_axis = shardings_bthd.query[2] if shardings_bthd.query is not None else None

            token_head_spec = jax.sharding.PartitionSpec(None, head_axis, None)
            beta_spec = jax.sharding.PartitionSpec(None, head_axis)
            state_spec = jax.sharding.PartitionSpec(None, head_axis, None, None)
            Ps = jax.sharding.PartitionSpec
            in_specs = (
                token_head_spec,
                token_head_spec,
                token_head_spec,
                beta_spec,
                beta_spec,
                state_spec,
                Ps(),
                Ps(),
            )
            out_specs = (token_head_spec, state_spec)
            platform = "pallas"

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
            platform=platform,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
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
            use_qk_l2norm: Whether to L2-normalize queries and keys. Defaults to True.
            **kwargs: Additional keyword arguments forwarded to the backend
                ``forward_*`` method. Recognized keys:

                - seg_ids: Optional segment ids for packed multi-sequence rows.
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
