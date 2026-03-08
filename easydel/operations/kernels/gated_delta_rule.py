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
from eformer.escale import with_sharding_constraint
from eformer.pytree import auto_pytree
from ejkernel.modules import gated_delta_rule
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float

from easydel.caching import RecurrentCacheView

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
        if jax.default_backend() == "tpu":
            platform = "pallas"

        outputs, new_recurrent_state = gated_delta_rule(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            use_qk_l2norm=use_qk_l2norm,
            use_chunked=not bool(is_inference),
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
