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

"""SmolLM3 model implementation in EasyDeL.

SmolLM3 is a transformer decoder model with:
- Pre-norm architecture
- Conditional RoPE (NoPE): Some layers don't use positional embeddings
- Optional sliding window attention
- Grouped Query Attention (GQA)
"""

from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput, DecoderLayerOutput
from easydel.infra.utils import block_wise_ffn
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule, BaseSequenceClassificationModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm

from .smollm3_configuration import SmolLM3Config


class SmolLM3Attention(UnifiedAttention):
    """SmolLM3 Attention module with conditional RoPE (NoPE).

    This attention module supports:
    - Conditional RoPE: Layers can use RoPE or skip it (NoPE)
    - Optional sliding window attention
    - Grouped Query Attention (GQA)
    - Optional bias in projections
    """

    def __init__(
        self,
        config: SmolLM3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = jax.lax.Precision("fastest"),
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SmolLM3Attention.

        Args:
            config: Model configuration.
            layer_idx: Layer index for this attention module.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matmul operations.
            rngs: RNG keys for initialization.
        """
        self.layer_idx = layer_idx

        self.use_rope = bool(config.no_rope_layers[layer_idx])

        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"

        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",  # SmolLM3 uses standard RoPE-based attention
            causal=True,  # Causal language modeling
            sliding_window=config.sliding_window if self.is_sliding else None,
            use_qk_norm=False,  # SmolLM3 doesn't use Q/K normalization
            use_fused_qkv=False,  # Separate Q/K/V projections
            use_gqa=True,  # SmolLM3 uses Grouped Query Attention
        )

    def _create_rotary(self, config: SmolLM3Config, dtype: jnp.dtype):
        """Create rotary embedding - returns dummy function for NoPE layers.

        This implements the NoPE (No Position Embedding) feature: layers with
        no_rope_layers[i] = 0 return a dummy function that skips RoPE entirely.

        Returns:
            Rotary function that either applies RoPE or returns inputs unchanged.
        """

        def _dummy(query, key, positions=None, frequencies=None):
            """Dummy RoPE function that returns query/key unchanged (NoPE)."""
            return query, key

        if not self.use_rope:
            return _dummy
        return super()._create_rotary(config, dtype)


class SmolLM3DecoderLayer(nn.Module):
    """SmolLM3 Decoder Layer with pre-norm architecture.

    Architecture:
        hidden = residual + attention(norm(hidden))
        hidden = residual + mlp(norm(hidden))

    This is pre-norm (different from SmolLM3's post-norm).
    """

    def __init__(
        self,
        config: SmolLM3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = jax.lax.Precision("fastest"),
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SmolLM3DecoderLayer.

        Args:
            config: Model configuration.
            layer_idx: Layer index.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matmul operations.
            rngs: RNG keys for initialization.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        # Self-attention
        self.self_attn = SmolLM3Attention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        # MLP
        self.mlp = self._create_mlp(config, dtype, param_dtype, precision, rngs)

        # Layer norms (pre-norm architecture)
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=param_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=param_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_mlp(
        self,
        config: SmolLM3Config,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.Precision,
        rngs: nn.Rngs,
    ):
        """Create MLP module."""
        return SmolLM3MLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through decoder layer.

        Args:
            hidden_states: Input hidden states.
            mask_info: Mask information for attention.
            position_ids: Position indices for RoPE.
            mode: Runtime mode (train/eval/decode).
            cache_view: KV cache view.
            cache_metadata: Cache metadata.
            output_attentions: Whether to return attention weights.
            frequencies: Precomputed RoPE frequencies.

        Returns:
            DecoderLayerOutput with hidden states and optional attention weights.
        """
        # Pre-norm architecture: norm -> attention -> residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attention_output = self.self_attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = residual + attention_output.attention_output

        # Pre-norm architecture: norm -> mlp -> residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        if self.config.use_scan_mlp:
            mlp_output = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            mlp_output = self.mlp(hidden_states)

        hidden_states = residual + mlp_output

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = checkpoint_name(hidden_states, "layer_output")

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attention_output.attention_weight,
            cache_view=attention_output.cache_view,
        )


class SmolLM3MLP(nn.Module):
    """SmolLM3 MLP module with SwiGLU activation."""

    def __init__(
        self,
        config: SmolLM3Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = jax.lax.Precision("fastest"),
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SmolLM3MLP.

        Args:
            config: Model configuration.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matmul operations.
            rngs: RNG keys for initialization.
        """
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = column_parallel_linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
        )
        self.up_proj = column_parallel_linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
        )
        self.down_proj = row_parallel_linear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
        )

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass through MLP.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Output hidden states.
        """
        # SwiGLU activation: silu(gate) * up
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = nn.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


@register_module(
    TaskType.BASE_MODULE,
    config=SmolLM3Config,
    model_type="smollm3",
    embedding_layer_names=["embed_tokens"],
)
class SmolLM3Model(EasyDeLBaseModule):
    """SmolLM3 base model (decoder-only transformer)."""

    def __init__(
        self,
        config: SmolLM3Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = jax.lax.Precision("fastest"),
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SmolLM3Model.

        Args:
            config: Model configuration.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matmul operations.
            rngs: RNG keys for initialization.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            SmolLM3DecoderLayer(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=param_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through SmolLM3 base model.

        Processes input tokens through embedding, multiple decoder layers with conditional RoPE,
        and final normalization. Supports caching for efficient generation.

        Args:
            input_ids (Int[Array, "batch seq_len"] | None): Input token IDs.
                Shape: (batch_size, sequence_length). Must be None if inputs_embeds is provided.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"] | None): Pre-computed embeddings.
                Shape: (batch_size, sequence_length, hidden_size). Used instead of input_ids if provided.
            attention_mask (Bool[Array, "batch seq_len"] | None): Attention mask for padding.
                Shape: (batch_size, sequence_length). True/1 for valid tokens, False/0 for padding.
            mask_info (MaskInfo | None): Structured mask information for efficient attention computation.
                If None, computed from attention_mask.
            position_ids (Int[Array, "batch seq_len"] | None): Position indices for RoPE.
                Shape: (batch_size, sequence_length). If None, uses sequential positions from mask_info.
            mode (common_types.RUNTIME_MODE_TYPES | None): Runtime mode controlling behavior:
                - MODE_TRAIN: Training mode
                - MODE_EVAL: Evaluation mode
                - MODE_DECODE: Generation mode (auto-detected from seq_len=1 and cache presence)
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None): KV cache
                from previous generation steps. None for first step or non-generation use.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None):
                Metadata for cache management and paged attention.
            output_attentions (bool | None): Whether to return attention weights from all layers.
            output_hidden_states (bool | None): Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final hidden states (batch_size, sequence_length, hidden_size)
                - hidden_states: Tuple of hidden states from all layers if output_hidden_states=True
                - attentions: Tuple of attention weights from all layers if output_attentions=True
                - past_key_values: Updated KV cache if caching is enabled
        """
        # Validate inputs
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        sequence_length = inputs_embeds.shape[1]

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            position_ids = mask_info.q_position_ids

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )

        # Initialize KV cache if needed
        if past_key_values is None and cache_metadata is not None:
            if isinstance(cache_metadata, RaggedPagesMetadata):
                past_key_values = RaggedPagesCache.init_empty(self.config)
            else:
                past_key_values = TransformerCache.init_empty(self.config)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Forward pass through layers
        hidden_states = inputs_embeds

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx] if past_key_values is not None else None,
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
            )

            hidden_states = layer_outputs.hidden_states

            if past_key_values is not None:
                past_key_values[idx] = layer_outputs.cache_view

            if output_attentions and layer_outputs.attention_weight is not None:
                all_attentions = (
                    *all_attentions,
                    layer_outputs.attention_weight,
                )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_embedding(self):
        """Returns the embedding layer of the module."""
        return self.embed_tokens

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self


@register_module(TaskType.CAUSAL_LM, config=SmolLM3Config, model_type="smollm3")
class SmolLM3ForCausalLM(BaseCausalLMModule[SmolLM3Model, SmolLM3Config]):
    """SmolLM3 model with a Causal Language Modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "smollm3"
    _config_class = SmolLM3Config

    def __init__(
        self,
        config: SmolLM3Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SmolLM3 for causal language modeling.

        Args:
            config: Model configuration object.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matrix multiplications.
            rngs: Random number generators.
        """
        super().__init__(
            config=config,
            base_model_class=SmolLM3Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=SmolLM3Config, model_type="smollm3")
class SmolLM3ForSequenceClassification(BaseSequenceClassificationModule[SmolLM3Model, SmolLM3Config]):
    """SmolLM3 model with a Sequence Classification head."""

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "smollm3"
    _config_class = SmolLM3Config

    def __init__(
        self,
        config: SmolLM3Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SmolLM3 for sequence classification.

        Args:
            config: Model configuration object.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matrix multiplications.
            rngs: Random number generators.
        """
        super().__init__(
            config=config,
            base_model_class=SmolLM3Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            score_bias=False,
        )
