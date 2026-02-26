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


import functools

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
from easydel.layers import ColumnParallelLinear, Embed, RMSNorm, RowParallelLinear
from easydel.layers.attention import UnifiedAttention
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .olmo2_configuration import Olmo2Config


class Olmo2MLP(nn.Module):
    """OLMo-2 MLP module.

    This module implements the feed-forward network (MLP) used in the OLMo-2 model.
    It consists of gate, up, and down projections with a SiLU activation.

    Attributes:
        config (Olmo2Config): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        gate_proj (ParallelLinear): Linear layer for the gate projection.
        down_proj (ParallelLinear): Linear layer for the down projection.
        up_proj (ParallelLinear): Linear layer for the up projection.
        act_fn (callable): Activation function (SiLU).
    """

    def __init__(
        self,
        config: Olmo2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Olmo2MLP module.

        Args:
            config (Olmo2Config): The configuration object for the OLMo-2 model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the Olmo2MLP module implementing a Gated Linear Unit structure.

        This method applies the standard GLU-based feedforward transformation used in
        modern transformer models:
        1. Gate stream: hidden_states -> gate_proj -> SiLU activation
        2. Value stream: hidden_states -> up_proj
        3. Combine: element-wise multiply gate and value
        4. Project down: down_proj to original hidden dimension

        Args:
            hidden_states (Float[Array, "batch seq_len hidden_dim"]): Input hidden states tensor.
                Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            Float[Array, "batch seq_len hidden_dim"]: Output hidden states after MLP transformation.
                Shape: (batch_size, sequence_length, hidden_size)

        Note:
            The method applies logical sharding for distributed training and uses
            checkpoint_name for gradient checkpointing at intermediate computation steps.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Olmo2Attention(UnifiedAttention):
    """Multi-head attention layer with Q/K normalization for OLMo-2 models.

    Extends UnifiedAttention with RMSNorm-based query and key normalization
    for improved training stability. Uses standard RoPE-based causal attention
    without sliding window.

    Key differences from OLMo v1:
        - Applies RMSNorm to query and key projections before attention
        - Uses post-normalization architecture in the decoder layer
        - No QKV clipping (relies on normalization instead)
    """

    def __init__(
        self,
        config: Olmo2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize OLMo-2 attention layer with Q/K normalization.

        Args:
            config (Olmo2Config): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            use_qk_norm=True,  # Enable Q/K normalization
        )

    def _create_k_norm(self, config: Olmo2Config, dtype: jnp.dtype, param_dtype: jnp.dtype, rngs: nn.Rngs):
        """Create key normalization layer using RMSNorm.

        Args:
            config (Olmo2Config): Model configuration with normalization parameters.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            rngs (nn.Rngs): Random number generator state.

        Returns:
            RMSNorm: Normalization layer for key projections with dimension
                matching num_attention_heads * head_dim.
        """
        return RMSNorm(
            dim=self.config.num_attention_heads * self.head_dim,
            eps=config.rms_norm_eps,
            dtype=param_dtype,
            rngs=rngs,
        )

    def _create_q_norm(self, config: Olmo2Config, dtype: jnp.dtype, param_dtype: jnp.dtype, rngs: nn.Rngs):
        """Create query normalization layer using RMSNorm.

        Args:
            config (Olmo2Config): Model configuration with normalization parameters.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            rngs (nn.Rngs): Random number generator state.

        Returns:
            RMSNorm: Normalization layer for query projections with dimension
                matching num_key_value_heads * head_dim.
        """
        return RMSNorm(
            dim=self.config.num_key_value_heads * self.head_dim,
            eps=config.rms_norm_eps,
            dtype=param_dtype,
            rngs=rngs,
        )

    def _preprocess_qkv(self, query_states, key_states, value_states):
        """Apply Q/K normalization before attention computation.

        Normalizes query and key projections using RMSNorm to improve
        training stability in OLMo-2 models. Value states are passed through unchanged.

        Args:
            query_states: Query projections before normalization.
            key_states: Key projections before normalization.
            value_states: Value projections (unchanged).

        Returns:
            Tuple of (normalized_query, normalized_key, value_states).
        """
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Olmo2DecoderLayer(nn.Module):
    """OLMo-2 Transformer Decoder Layer.

    This module represents a single decoder layer in the OLMo-2 model,
    combining self-attention and MLP sub-layers with residual connections
    and layer normalization applied before each sub-layer.

    Attributes:
        config (Olmo2Config): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        self_attn (Olmo2Attention): The self-attention module.
        mlp (Olmo2MLP): The feed-forward (MLP) module.
        input_layernorm (RMSNorm): Layer normalization before the attention layer.
        post_attention_layernorm (RMSNorm): Layer normalization before the MLP layer.
    """

    def __init__(
        self,
        config: Olmo2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initializes the Olmo2DecoderLayer.

        Args:
            config (Olmo2Config): The configuration object for the OLMo-2 model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        attn_block = auto_remat(
            Olmo2Attention,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        mlp_block = auto_remat(
            Olmo2MLP,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
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
    ):
        """Forward pass of the Olmo2DecoderLayer with post-normalization architecture.

        This method implements the OLMo2 decoder layer with the following structure:
        1. Self-attention on input hidden states
        2. Post-attention normalization
        3. Residual connection (input + normalized attention output)
        4. Feed-forward network on normalized hidden states
        5. Post-feedforward normalization
        6. Residual connection (previous + normalized FFN output)

        Note that OLMo2 uses post-normalization (normalization after the sub-layer)
        rather than the more common pre-normalization.

        Args:
            hidden_states (Float[Array, "batch seq_len hidden_dim"]): Input hidden states.
                Shape: (batch_size, sequence_length, hidden_size)
            mask_info (MaskInfo, optional): Mask information object containing attention masks
                and position information for causal masking.
            position_ids (Int[Array, "batch seq_len"]): Position indices for the tokens.
                Shape: (batch_size, sequence_length)
            mode (common_types.RUNTIME_MODE_TYPES): Runtime mode (training, generation, etc.)
                controlling attention behavior and caching.
            cache_view (TransformerCacheView | RaggedPagesCacheView, optional): Cache view
                for storing and retrieving key-value pairs during generation.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for paged attention or other cache optimizations.
            output_attentions (bool, optional): Whether to return attention weights.
                Defaults to False.
            frequencies (Float[Array, "seq_len head_dim"], optional): Precomputed RoPE
                frequency embeddings. Shape: (max_positions, head_dim)

        Returns:
            DecoderLayerOutput: Object containing:
                - hidden_states (Array): Output hidden states after the decoder layer.
                    Shape: (batch_size, sequence_length, hidden_size)
                - attention_weight (Array, optional): Attention weights if output_attentions=True.
                    Shape: (batch_size, num_heads, sequence_length, sequence_length)
                - cache_view: Updated cache view with new key-value pairs

        Note:
            The post-normalization architecture (norm after sub-layer + residual) is different
            from pre-normalization (norm before sub-layer) and may affect training dynamics.
        """
        residual = hidden_states
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

        hidden_states = attention_output[0]
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        residual = hidden_states
        if self.config.use_scan_mlp:
            hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "layer_output")
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attention_output.attention_weight,
            cache_view=attention_output.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Olmo2Config, model_type="olmo2")
class Olmo2Model(EasyDeLBaseModule):
    """The base OLMo-2 model transformer.

    This class represents the core transformer architecture of the OLMo-2 model,
    consisting of an embedding layer, multiple Olmo2DecoderLayer layers,
    and a final RMS normalization layer.

    Attributes:
        config (Olmo2Config): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        embed_tokens (Embed): Embedding layer for input tokens.
        layers (tp.List[Olmo2DecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: Olmo2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Olmo2Model.

        Args:
            config (Olmo2Config): The configuration object for the OLMo-2 model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = nn.List(
            [
                Olmo2DecoderLayer(
                    config=config,
                    layer_idx=idx,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for idx in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
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
        """Forward pass of the Olmo2Model base transformer.

        This method processes input tokens through the entire OLMo2 transformer stack:
        1. Embed input tokens (or accept pre-computed embeddings)
        2. Process through all decoder layers with causal attention
        3. Apply final layer normalization
        4. Optionally collect hidden states and attention weights from all layers

        Args:
            input_ids (Int[Array, "batch seq_len"], optional): Input token IDs.
                Shape: (batch_size, sequence_length). Either this or inputs_embeds must be provided.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"], optional): Pre-computed
                input embeddings. Shape: (batch_size, sequence_length, hidden_size).
                Either this or input_ids must be provided.
            attention_mask (Bool[Array, "batch seq_len"], optional): Boolean mask to avoid
                attention on padding tokens. Shape: (batch_size, sequence_length).
                True indicates positions to mask.
            mask_info (MaskInfo, optional): Structured mask information object. If None,
                will be constructed from attention_mask.
            position_ids (Int[Array, "batch seq_len"], optional): Position indices for tokens.
                Shape: (batch_size, sequence_length). If None, uses sequential positions.
            mode (common_types.RUNTIME_MODE_TYPES, optional): Runtime mode controlling
                attention behavior. If None, inferred from sequence_length and past_key_values.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Pre-computed key-value states for efficient generation. If None, initializes empty.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for paged attention or other cache optimizations.
            output_attentions (bool, optional): Whether to return attention weights from all layers.
                Defaults to config.output_attentions.
            output_hidden_states (bool, optional): Whether to return hidden states from all layers.
                Defaults to config.output_hidden_states.

        Returns:
            BaseModelOutput: Object containing:
                - last_hidden_state (Array): Final layer output after normalization.
                    Shape: (batch_size, sequence_length, hidden_size)
                - hidden_states (tuple, optional): Hidden states from all layers if requested.
                    Each element shape: (batch_size, sequence_length, hidden_size)
                - attentions (tuple, optional): Attention weights from all layers if requested.
                    Each element shape: (batch_size, num_heads, sequence_length, sequence_length)
                - past_key_values: Updated cache with new key-value pairs

        Raises:
            ValueError: If neither input_ids nor inputs_embeds is provided, or if both are provided.
            AssertionError: If sequence_length exceeds max_position_embeddings.

        Note:
            The model uses post-normalization architecture and supports various caching
            strategies for efficient generation (TransformerCache, RaggedPagesCache, HybridCache).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        sequence_length = inputs_embeds.shape[1]

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Olmo2Config, model_type="olmo2")
class Olmo2ForCausalLM(BaseCausalLMModule[Olmo2Model, Olmo2Config]):
    """OLMo-2 model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with causal attention masks
    applied to perform autoregressive language generation. It uses Q/K normalization
    and post-normalization architecture for improved training stability.

    Attributes:
        config (Olmo2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "olmo2"
    _config_class = Olmo2Config

    def __init__(
        self,
        config: Olmo2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize OLMo-2 model for causal language modeling.

        Args:
            config (Olmo2Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Olmo2Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
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
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass of the Olmo2ForCausalLM model for next-token prediction.

        This method processes input tokens through the base OLMo2 transformer and applies
        a language modeling head to produce logits for next-token prediction.

        Args:
            input_ids (Int[Array, "batch seq_len"], optional): Input token IDs.
                Shape: (batch_size, sequence_length). Either this or inputs_embeds must be provided.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"], optional): Pre-computed
                input embeddings. Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Bool[Array, "batch seq_len"], optional): Boolean mask for padding.
                Shape: (batch_size, sequence_length). True indicates positions to mask.
            mask_info (MaskInfo, optional): Structured mask information object.
            position_ids (Int[Array, "batch seq_len"], optional): Position indices for tokens.
                Shape: (batch_size, sequence_length).
            mode (common_types.RUNTIME_MODE_TYPES, optional): Runtime mode for attention behavior.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key-value states for generation.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for cache optimizations.
            apply_lm_head (bool, optional): Whether to apply the language modeling head.
                Defaults to True. Set to False to get only hidden states.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return all hidden states.

        Returns:
            CausalLMOutput: Object containing:
                - logits (Array, optional): Next-token prediction logits if apply_lm_head=True.
                    Shape: (batch_size, sequence_length, vocab_size)
                - last_hidden_state (Array): Final hidden states from the base model.
                    Shape: (batch_size, sequence_length, hidden_size)
                - hidden_states (tuple, optional): All layer hidden states if requested.
                - attentions (tuple, optional): All layer attention weights if requested.
                - past_key_values: Updated cache for generation

        Note:
            For generation, typically only the last token's logits are needed:
            `next_token_logits = outputs.logits[:, -1, :]`
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Olmo2Config, model_type="olmo2")
class Olmo2ForSequenceClassification(BaseSequenceClassificationModule[Olmo2Model, Olmo2Config]):
    """OLMo-2 model for sequence classification tasks.

    This class extends the base OLMo-2 model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.
    Uses Q/K normalization and post-normalization architecture for improved training stability.

    Attributes:
        config (Olmo2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "olmo2"
    _config_class = Olmo2Config

    def __init__(
        self,
        config: Olmo2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize OLMo-2 model for sequence classification.

        Args:
            config (Olmo2Config): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Olmo2Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            classifier_name="score",
            classifier_bias=False,
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
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass of the Olmo2ForSequenceClassification model for sequence classification.

        This method processes input sequences through the base OLMo2 transformer and applies
        a classification head to the last token's hidden state to produce class logits.

        Args:
            input_ids (Int[Array, "batch seq_len"], optional): Input token IDs.
                Shape: (batch_size, sequence_length). Either this or inputs_embeds must be provided.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"], optional): Pre-computed
                input embeddings. Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Bool[Array, "batch seq_len"], optional): Boolean mask for padding.
                Shape: (batch_size, sequence_length). True indicates positions to mask.
            mask_info (MaskInfo, optional): Structured mask information object.
            position_ids (Int[Array, "batch seq_len"], optional): Position indices for tokens.
                Shape: (batch_size, sequence_length).
            mode (common_types.RUNTIME_MODE_TYPES, optional): Runtime mode for attention behavior.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key-value states (typically not used for classification).
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for cache optimizations.
            apply_lm_head (bool, optional): Unused parameter for API compatibility.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return all hidden states.

        Returns:
            SequenceClassifierOutput: Object containing:
                - logits (Array): Classification logits pooled from the last token position.
                    Shape: (batch_size, num_labels)
                - hidden_states (tuple, optional): All layer hidden states if requested.
                - attentions (tuple, optional): All layer attention weights if requested.
                - past_key_values: Updated cache (typically None for classification)

        Raises:
            ValueError: If config.pad_token_id is None and batch_size > 1, as the model
                cannot determine the last non-padded token position.

        Note:
            The classification is performed on the last token's hidden state. For sequences
            with padding, the last non-padded token is automatically selected using the
            pad_token_id from the configuration.
        """
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = jnp.argmax(jnp.equal(input_ids, self.config.pad_token_id).astype("i4"), -1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model has a sequence classification head, not an LM Head.
        """
        raise NotImplementedError("This model has a sequence classification head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()
