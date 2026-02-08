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


import functools

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
from easydel.infra.utils import (
    ACT2FN,
    auto_remat,
    block_wise_ffn,
)
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
from easydel.layers.components import ColumnParallelLinear, Embed, RMSNorm, RowParallelLinear

from .exaone4_configuration import Exaone4Config


class Exaone4MLP(nn.Module):
    """Gated Multi-Layer Perceptron module for Exaone4 models.

    Implements a SwiGLU-style gated feedforward network with configurable activation
    function for enhanced representation learning in Exaone4 architecture. Uses
    column-parallel gate and up projections with row-parallel down projection for
    efficient tensor parallelism.
    """

    def __init__(
        self,
        config: Exaone4Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Exaone4 gated MLP block.

        Args:
            config (Exaone4Config): Model configuration with MLP parameters including
                hidden_size, intermediate_size, and hidden_act.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision | None, optional): Numerical precision for matrix
                operations. Defaults to None.
            rngs (nn.Rngs): Random number generator state for initialization.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,  # Exaone4 uses no bias
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,  # Exaone4 uses no bias
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply gated feedforward transformation.

        Computes: down_proj(act_fn(gate_proj(x)) * up_proj(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Array: Transformed hidden states of shape (batch_size, seq_len, hidden_size).
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


class Exaone4Attention(UnifiedAttention):
    """Multi-head attention layer with NoPE (No Position Embedding) for Exaone4 models.

    This attention implementation supports the NoPE architecture where full attention layers
    skip rotary position embeddings entirely, while sliding window attention layers use
    standard RoPE. Includes Q/K normalization via RMSNorm for training stability.
    """

    def __init__(
        self,
        config: Exaone4Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Exaone4 attention layer with conditional RoPE (NoPE).

        Args:
            config (Exaone4Config): Model configuration with attention parameters including
                layer_types, sliding_window, and head_dim.
            layer_idx (int): Index of this layer in the model, used to determine attention type.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window if self.is_sliding else None,
            use_qk_norm=True,  # Exaone4 uses Q/K normalization
        )

    def _create_rotary(self, config: Exaone4Config, dtype: jnp.dtype):
        """Create rotary embedding based on layer type (NoPE for full attention).

        This implements the key NoPE (No Position Embedding) feature: full attention
        layers return a dummy function that passes query/key unchanged, effectively
        skipping RoPE entirely in those layers. Sliding attention layers use standard
        RoPE for local position awareness.

        Args:
            config (Exaone4Config): Model configuration containing RoPE parameters.
            dtype (jnp.dtype): Data type for the rotary embedding computation.

        Returns:
            Callable: Either a dummy pass-through function (for full attention) or
                standard RoPE function (for sliding attention).
        """

        def _dummy(query, key, positions=None, frequencies=None):
            """Dummy RoPE function that returns query/key unchanged (NoPE)."""
            return query, key

        if not self.is_sliding:
            # Full attention layer: Return dummy function (NoPE - No Position Embedding)
            return _dummy
        # Sliding attention layer: Use standard RoPE
        return super()._create_rotary(config, dtype)

    def _create_q_norm(self, config: Exaone4Config, dtype: jnp.dtype, param_dtype: jnp.dtype, rngs: nn.Rngs):
        """Create query normalization layer using RMSNorm.

        Normalization is applied per-head over the head_dim dimension to stabilize
        training dynamics, matching HuggingFace's Exaone4 implementation.

        Args:
            config (Exaone4Config): Model configuration with head_dim and rms_norm_eps.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for normalization parameters.
            rngs (nn.Rngs): Random number generator state.

        Returns:
            RMSNorm: Query normalization layer.
        """
        return RMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            dtype=param_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_k_norm(self, config: Exaone4Config, dtype: jnp.dtype, param_dtype: jnp.dtype, rngs: nn.Rngs):
        """Create key normalization layer using RMSNorm.

        Normalization is applied per-head over the head_dim dimension to stabilize
        training dynamics, matching HuggingFace's Exaone4 implementation.

        Args:
            config (Exaone4Config): Model configuration with head_dim and rms_norm_eps.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for normalization parameters.
            rngs (nn.Rngs): Random number generator state.

        Returns:
            RMSNorm: Key normalization layer.
        """
        return RMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            dtype=param_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _postprocess_qkv(
        self,
        query_states: jnp.ndarray,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply Q/K normalization per-head for training stability.

        This method is called by UnifiedAttention after QKV projection and reshape.
        Applies RMSNorm to query and key states per-head over the head_dim dimension,
        matching HuggingFace's Exaone4 implementation.

        Args:
            query_states (jnp.ndarray): Query tensor of shape
                (batch, seq_len, num_heads, head_dim).
            key_states (jnp.ndarray): Key tensor of shape
                (batch, seq_len, num_key_value_heads, head_dim).
            value_states (jnp.ndarray): Value tensor of shape
                (batch, seq_len, num_key_value_heads, head_dim).

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Tuple of (normalized_query,
                normalized_key, value) states with same shapes as inputs.
        """
        query_states = self.query_normalization(query_states)
        key_states = self.key_normalization(key_states)

        return query_states, key_states, value_states


class Exaone4DecoderLayer(nn.Module):
    """Single decoder layer for Exaone4 models.

    Implements a post-normalization architecture combining multi-head attention with
    NoPE (conditional RoPE) and gated feedforward networks with RMSNorm and residual
    connections. Uses post-norm pattern: residual + norm(sublayer(x)).
    """

    def __init__(
        self,
        config: Exaone4Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Exaone4 decoder layer.

        Args:
            config (Exaone4Config): Model configuration with layer parameters.
            layer_idx (int): Index of this layer in the model, determines attention type.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        attn_block = auto_remat(
            Exaone4Attention,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        mlp_block = auto_remat(
            Exaone4MLP,
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

        # Post-norm architecture: normalization after residual
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
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies post-normalization architecture: residual + norm(attn(x)) followed by
        residual + norm(mlp(x)).

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view
                for autoregressive generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata for KV cache management. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, optional attention weights, and cache view.
        """
        # Post-norm pattern: residual + norm(sublayer(x))

        # Self-attention block
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
        hidden_states = self.post_attention_layernorm(attention_output.attention_output)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        if self.config.use_scan_mlp:
            mlp_output = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            mlp_output = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + hidden_states

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


@register_module(TaskType.BASE_MODULE, config=Exaone4Config, model_type="exaone4")
class Exaone4Model(EasyDeLBaseModule):
    """Exaone4 model implementation.

    This implements the Exaone4 language model architecture with NoPE (No Position Embedding)
    for full attention layers, utilizing transformer blocks with post-normalization RMSNorm,
    conditional rotary position embeddings, and Q/K normalization for training stability.

    Attributes:
        config (Exaone4Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Exaone4Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Exaone4 base model.

        Args:
            config (Exaone4Config): Model configuration with architecture parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = nn.List(
            [
                Exaone4DecoderLayer(
                    config=config,
                    layer_idx=i,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for i in range(config.num_hidden_layers)
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
        """Forward pass through the Exaone4 base model.

        Processes input tokens through embedding, all decoder layers with conditional
        RoPE (NoPE for full attention, RoPE for sliding), Q/K normalization, and final
        RMSNorm normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
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

        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_embedding(self):
        """Returns the embedding layer of the module.

        Returns:
            Embed: The token embedding layer.
        """
        return self.embed_tokens

    def get_decoder(self):
        """Returns the decoder part of the model's graph definition.

        Returns:
            Exaone4Model: Self, as this is a decoder-only model.
        """
        return self


@register_module(TaskType.CAUSAL_LM, config=Exaone4Config, model_type="exaone4")
class Exaone4ForCausalLM(BaseCausalLMModule[Exaone4Model, Exaone4Config]):
    """Exaone4 model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with causal attention masks
    applied to perform autoregressive language generation. Uses NoPE (No Position
    Embedding) for full attention layers and RoPE for sliding window attention.

    Attributes:
        config (Exaone4Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "exaone4"
    _config_class = Exaone4Config

    def __init__(
        self,
        config: Exaone4Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Exaone4 model for causal language modeling.

        Args:
            config (Exaone4Config): Model configuration with language modeling parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Exaone4Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Exaone4Config, model_type="exaone4")
class Exaone4ForSequenceClassification(BaseSequenceClassificationModule[Exaone4Model, Exaone4Config]):
    """Exaone4 model for sequence classification tasks.

    This class extends the base Exaone4 model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
        config (Exaone4Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "exaone4"
    _config_class = Exaone4Config

    def __init__(
        self,
        config: Exaone4Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Exaone4 model for sequence classification.

        Args:
            config (Exaone4Config): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Exaone4Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            score_bias=False,
        )
