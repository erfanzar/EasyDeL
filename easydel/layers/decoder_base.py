# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Base utilities for decoder layers.

Provides common patterns and utilities for transformer decoder layers:
- Pre-norm residual connections
- KV cache updates
- Standard decoder layer signature
- Block-wise FFN for memory efficiency

These are provided as utility functions/mixins that decoder layers can use
rather than requiring strict inheritance.

Example:
    >>> class LlamaDecoderLayer(nn.Module):
    ...     def __call__(self, hidden_states, ...):
    ...         # Attention with pre-norm residual
    ...         hidden_states = BaseDecoderLayer.pre_norm_residual_attn(
    ...             hidden_states, self.self_attn, self.input_layernorm, ...
    ...         )
    ...         # MLP with pre-norm residual
    ...         hidden_states = BaseDecoderLayer.pre_norm_residual_mlp(
    ...             hidden_states, self.mlp, self.post_attention_layernorm
    ...         )
    ...         return hidden_states
"""

from collections.abc import Callable

import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int

from easydel.infra.modeling_outputs import AttentionLayerOutput, DecoderLayerOutput

from .caching import (
    OperationsMetadata,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCacheView,
    TransformerMetadata,
)


class BaseDecoderLayer:
    """Utility class providing common decoder layer patterns.

    This class provides static/class methods for common operations in
    decoder layers. It is not meant to be instantiated directly, but
    rather used as a collection of utilities.

    Methods:
        pre_norm_residual_attn: Apply attention with pre-norm residual
        pre_norm_residual_mlp: Apply MLP with pre-norm residual
        update_cache_view: Update KV cache view after attention
    """

    @staticmethod
    def pre_norm_residual_attn(
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        attention_module: Callable,
        norm_module: Callable,
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
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
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
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

    Processes the input in chunks along the sequence dimension to
    reduce memory usage during training.

    Args:
        mlp_module: MLP module to apply
        inputs: Input tensor
        chunk_size: Size of chunks to process

    Returns:
        MLP output
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
