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

"""Base utilities for transformer decoder layers.

This module provides common patterns and utility functions for implementing
transformer decoder layers. Rather than requiring strict inheritance, it offers
composable static methods that can be used by any decoder layer implementation.

The utilities follow the pre-norm transformer architecture pattern where
normalization is applied before each sub-layer (attention, MLP) with
residual connections around the sub-layers.

Key Patterns:
    Pre-Norm Residual:
        The standard pattern used by most modern transformers::

            hidden = hidden + sublayer(norm(hidden))

        This is implemented via pre_norm_residual_attn() and pre_norm_residual_mlp().

    Block-wise FFN:
        For memory efficiency during training, the MLP can be processed in
        chunks along the sequence dimension via block_wise_ffn().

    Standard Decoder Layer:
        The complete pattern combining attention and MLP::

            hidden = hidden + attention(norm(hidden))
            hidden = hidden + mlp(norm(hidden))

Classes:
    BaseDecoderLayer:
        Collection of static methods for decoder layer operations.

Functions:
    block_wise_ffn:
        Apply MLP in chunks for memory efficiency.

Example:
    Using pre-norm residual patterns in a custom decoder layer::

        >>> from easydel.layers.attention import BaseDecoderLayer
        >>> import flax.nnx as nn
        >>>
        >>> class MyDecoderLayer(nn.Module):
        ...     def __init__(self, config, rngs):
        ...         self.attention = MyAttention(config, rngs=rngs)
        ...         self.mlp = MyMLP(config, rngs=rngs)
        ...         self.input_layernorm = RMSNorm(config.hidden_size, rngs=rngs)
        ...         self.post_attention_layernorm = RMSNorm(config.hidden_size, rngs=rngs)
        ...
        ...     def __call__(self, hidden_states, mask_info, position_ids, mode, ...):
        ...         # Attention with pre-norm residual
        ...         hidden_states, attn_out = BaseDecoderLayer.pre_norm_residual_attn(
        ...             hidden_states,
        ...             self.attention,
        ...             self.input_layernorm,
        ...             mask_info,
        ...             position_ids,
        ...             mode
        ...         )
        ...         # MLP with pre-norm residual
        ...         hidden_states = BaseDecoderLayer.pre_norm_residual_mlp(
        ...             hidden_states,
        ...             self.mlp,
        ...             self.post_attention_layernorm
        ...         )
        ...         return hidden_states, attn_out

    Using the complete standard decoder layer::

        >>> output = BaseDecoderLayer.standard_decoder_layer_call(
        ...     hidden_states,
        ...     attention_module=self.attention,
        ...     mlp_module=self.mlp,
        ...     input_norm=self.input_layernorm,
        ...     post_attn_norm=self.post_attention_layernorm,
        ...     mask_info=mask_info,
        ...     position_ids=position_ids,
        ...     mode=mode,
        ...     partition_manager=config.partition_manager
        ... )

Note:
    These utilities are designed to work with JAX's checkpoint system via
    checkpoint_name() for memory-efficient gradient computation during training.

See Also:
    - easydel.layers.attention_unified: UnifiedAttention for attention implementation
    - easydel.layers.caching: Cache view implementations for KV caching
    - easydel.infra.modeling_outputs: Output container types
"""

from collections.abc import Callable

import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int

from easydel.caching import (
    OperationsMetadata,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCacheView,
    TransformerMetadata,
    UnifiedAttentionCacheView,
)
from easydel.infra.modeling_outputs import AttentionLayerOutput, DecoderLayerOutput


class BaseDecoderLayer:
    """Utility class providing common decoder layer patterns.

    This class provides static methods for common operations in transformer
    decoder layers. It is not meant to be instantiated directly, but rather
    used as a collection of composable utilities.

    The methods implement the pre-norm transformer architecture pattern
    where layer normalization is applied before each sub-layer, with
    residual connections around the sub-layers.

    Static Methods:
        pre_norm_residual_attn:
            Apply attention with pre-norm residual connection.
            Pattern: ``hidden = hidden + attention(norm(hidden))``

        pre_norm_residual_mlp:
            Apply MLP with pre-norm residual connection.
            Pattern: ``hidden = hidden + mlp(norm(hidden))``

        apply_output_sharding:
            Apply sharding constraints to decoder layer output.

        standard_decoder_layer_call:
            Complete standard decoder layer forward pass combining
            both attention and MLP with pre-norm residuals.

    Example:
        >>> # Use individual methods for custom control flow
        >>> hidden, attn_out = BaseDecoderLayer.pre_norm_residual_attn(
        ...     hidden_states, attention_fn, norm_fn, mask_info, ...
        ... )
        >>> hidden = BaseDecoderLayer.pre_norm_residual_mlp(
        ...     hidden, mlp_fn, post_norm_fn
        ... )
        >>>
        >>> # Or use the combined standard call
        >>> output = BaseDecoderLayer.standard_decoder_layer_call(
        ...     hidden_states, attention_fn, mlp_fn, input_norm, post_norm, ...
        ... )

    Note:
        These methods are designed to work with JAX's gradient checkpointing
        via checkpoint_name() for memory-efficient training.
    """

    @staticmethod
    def pre_norm_residual_attn(
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        attention_module: Callable,
        norm_module: Callable,
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        checkpoint_names: tuple[str, str] = ("norm", "residual"),
    ) -> tuple[Float[Array, "batch seq_len hidden_dim"], AttentionLayerOutput]:
        """Apply attention with pre-norm residual connection.

        Pattern: `hidden = hidden + attention(norm(hidden))`

        Args:
            hidden_states: Input tensor
            attention_module: Attention module to apply
            norm_module: Normalization module (e.g., RMSNorm, LayerNorm)
            mask_info: Mask information for attention
            position_ids: Position indices
            mode: Runtime mode
            cache_view: Optional cache view
            cache_metadata: Optional cache metadata
            output_attentions: Whether to return attention weights
            frequencies: Optional RoPE frequencies
            checkpoint_names: Names for checkpointing (norm_name, residual_name)

        Returns:
            Tuple of (updated_hidden_states, attention_outputs)
        """
        norm_name: str
        residual_name: str
        norm_name, residual_name = checkpoint_names

        # Apply normalization
        normed: Float[Array, "batch_size seq_len hidden_dim"] = checkpoint_name(norm_module(hidden_states), norm_name)

        # Apply attention
        attn_outputs: AttentionLayerOutput = attention_module(
            normed,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )

        # Residual connection
        residual_added: Float[Array, "batch_size seq_len hidden_dim"] = hidden_states + attn_outputs.attention_output
        hidden_states_updated: Float[Array, "batch_size seq_len hidden_dim"] = checkpoint_name(
            residual_added, residual_name
        )

        return hidden_states_updated, attn_outputs

    @staticmethod
    def pre_norm_residual_mlp(
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mlp_module: Callable,
        norm_module: Callable,
        use_scan: bool = False,
        scan_chunk_size: int = 1024,
        checkpoint_names: tuple[str, str] = ("norm", "residual"),
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply MLP with pre-norm residual connection.

        Pattern: `hidden = hidden + mlp(norm(hidden))`

        Args:
            hidden_states: Input tensor
            mlp_module: MLP module to apply
            norm_module: Normalization module (e.g., RMSNorm, LayerNorm)
            use_scan: Whether to use block-wise scan for memory efficiency
            scan_chunk_size: Chunk size for scan (if use_scan=True)
            checkpoint_names: Names for checkpointing (norm_name, residual_name)

        Returns:
            Updated hidden states
        """
        norm_name: str
        residual_name: str
        norm_name, residual_name = checkpoint_names

        # Apply normalization
        normed: Float[Array, "batch_size seq_len hidden_dim"] = checkpoint_name(norm_module(hidden_states), norm_name)

        # Apply MLP (with optional scan)
        mlp_output: Float[Array, "batch_size seq_len hidden_dim"]
        if use_scan:
            mlp_output = block_wise_ffn(mlp_module, normed, scan_chunk_size)
        else:
            mlp_output = mlp_module(normed)

        # Residual connection
        residual_added: Float[Array, "batch_size seq_len hidden_dim"] = hidden_states + mlp_output
        hidden_states_updated: Float[Array, "batch_size seq_len hidden_dim"] = checkpoint_name(
            residual_added, residual_name
        )

        return hidden_states_updated

    @staticmethod
    def apply_output_sharding(
        hidden_states: Float[Array, "batch seq_len hidden_dim"], partition_manager
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply sharding to decoder layer output.

        Args:
            hidden_states: Output tensor
            partition_manager: Partition manager for sharding

        Returns:
            Sharded hidden states
        """
        return apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=partition_manager,
        )

    @staticmethod
    def standard_decoder_layer_call(
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        attention_module: Callable,
        mlp_module: Callable,
        input_norm: Callable,
        post_attn_norm: Callable,
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        partition_manager,
        cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
    ) -> DecoderLayerOutput:
        """Complete standard decoder layer forward pass.

        Combines attention and MLP with pre-norm residuals:
        1. `hidden = hidden + attention(norm(hidden))`
        2. `hidden = hidden + mlp(norm(hidden))`

        Args:
            hidden_states: Input tensor
            attention_module: Attention module
            mlp_module: MLP module
            input_norm: Pre-attention normalization
            post_attn_norm: Pre-MLP normalization
            mask_info: Mask information
            position_ids: Position indices
            mode: Runtime mode
            partition_manager: Partition manager for sharding
            cache_view: Optional cache view
            cache_metadata: Optional cache metadata
            output_attentions: Whether to return attention weights
            frequencies: Optional RoPE frequencies
            use_scan_mlp: Whether to use block-wise scan for MLP
            scan_mlp_chunk_size: Chunk size for scan

        Returns:
            DecoderLayerOutput with hidden states and optional attention weights
        """
        # Attention with pre-norm residual
        hidden_states_after_attn: Float[Array, "batch_size seq_len hidden_dim"]
        attn_outputs: AttentionLayerOutput
        hidden_states_after_attn, attn_outputs = BaseDecoderLayer.pre_norm_residual_attn(
            hidden_states,
            attention_module,
            input_norm,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
            checkpoint_names=("attn_norm", "attn_residual"),
        )

        # MLP with pre-norm residual
        hidden_states_after_mlp: Float[Array, "batch_size seq_len hidden_dim"] = BaseDecoderLayer.pre_norm_residual_mlp(
            hidden_states_after_attn,
            mlp_module,
            post_attn_norm,
            use_scan_mlp,
            scan_mlp_chunk_size,
            checkpoint_names=("mlp_norm", "mlp_residual"),
        )

        # Apply output sharding
        hidden_states_sharded: Float[Array, "batch_size seq_len hidden_dim"] = BaseDecoderLayer.apply_output_sharding(
            hidden_states_after_mlp, partition_manager
        )

        output: DecoderLayerOutput = DecoderLayerOutput(
            hidden_states=hidden_states_sharded,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )
        return output


def block_wise_ffn(
    mlp_module: Callable,
    inputs: Float[Array, "batch_size seq_len hidden_dim"],
    chunk_size: int = 1024,
) -> Float[Array, "batch_size seq_len hidden_dim"]:
    """Apply MLP block-wise for memory efficiency.

    Processes the input in chunks along the sequence dimension to reduce
    peak memory usage during training. This is particularly useful for
    long sequences where the MLP intermediate states can be very large.

    The function splits the sequence into chunks, applies the MLP to each
    chunk independently, and concatenates the results. This trades off
    some computational efficiency for reduced memory footprint.

    Args:
        mlp_module: MLP module (callable) to apply. Should accept input
            tensors of shape [batch, seq_chunk, hidden_dim] and return
            tensors of the same shape.
        inputs: Input tensor with shape [batch_size, seq_len, hidden_dim].
        chunk_size: Maximum size of chunks along the sequence dimension.
            Defaults to 1024. Smaller values use less memory but may be
            slower due to increased loop overhead.

    Returns:
        MLP output tensor with shape [batch_size, seq_len, hidden_dim],
        equivalent to mlp_module(inputs) but computed with lower peak memory.

    Example:
        >>> # Standard MLP application
        >>> mlp_out = mlp(hidden_states)  # May OOM on long sequences
        >>>
        >>> # Block-wise for memory efficiency
        >>> mlp_out = block_wise_ffn(mlp, hidden_states, chunk_size=512)

    Note:
        If the sequence length is less than or equal to chunk_size, the
        MLP is applied directly without chunking for efficiency.

    Warning:
        The mlp_module should be stateless or handle batched computation
        correctly, as different chunks are processed independently.
    """
    seq_len: int = inputs.shape[1]

    # If sequence is shorter than chunk size, process directly
    if seq_len <= chunk_size:
        mlp_out: Float[Array, "batch_size seq_len hidden_dim"] = mlp_module(inputs)
        return mlp_out

    # Process in chunks
    num_chunks: int = (seq_len + chunk_size - 1) // chunk_size
    outputs: list[Float[Array, "batch_size chunk_seq_len hidden_dim"]] = []

    for i in range(num_chunks):
        start_idx: int = i * chunk_size
        end_idx: int = min((i + 1) * chunk_size, seq_len)
        chunk: Float[Array, "batch_size chunk_seq_len hidden_dim"] = inputs[:, start_idx:end_idx, :]
        chunk_output: Float[Array, "batch_size chunk_seq_len hidden_dim"] = mlp_module(chunk)
        outputs.append(chunk_output)

    concatenated: Float[Array, "batch_size seq_len hidden_dim"] = jnp.concatenate(outputs, axis=1)
    return concatenated
