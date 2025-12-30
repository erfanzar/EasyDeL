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
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm

from .exaone4_configuration import Exaone4Config


class Exaone4MLP(nn.Module):
    """Feed-forward network used inside Exaone4 decoder layers."""

    def __init__(
        self,
        config: Exaone4Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the Exaone4MLP module.

        Args:
            config: Model configuration object containing hyperparameters.
            dtype: Data type for computations (default: float32).
            param_dtype: Data type for parameters (default: float32).
            precision: JAX precision for matrix multiplications (optional).
            rngs: Random number generators for initialization.
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

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the MLP.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
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
    """Multi-head attention block configured for Exaone4."""

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
        """Initialize the Exaone4Attention module with NoPE (conditional RoPE).

        Args:
            config: Model configuration object.
            layer_idx: Index of this layer in the model.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matrix multiplications.
            rngs: Random number generators.
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
        """Create rotary embedding - NoPE for full attention layers.

        This implements the key NoPE feature: full attention layers return None for rotary,
        which causes RoPE to be skipped entirely in those layers.
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
        """Create Q normalization layer (RMSNorm).

        Note: Normalization is applied per-head over head_dim dimension,
        matching HuggingFace's implementation.
        """
        return RMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            dtype=param_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_k_norm(self, config: Exaone4Config, dtype: jnp.dtype, param_dtype: jnp.dtype, rngs: nn.Rngs):
        """Create K normalization layer (RMSNorm).

        Note: Normalization is applied per-head over head_dim dimension,
        matching HuggingFace's implementation.
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
        """Apply Q/K normalization per-head (matching HuggingFace).

        This method is called by UnifiedAttention before reshaping to head dimensions.
        We temporarily reshape to apply per-head normalization, then reshape back.

        Args:
            query_states: Query tensor (batch, seq_len, num_heads, head_dim).
            key_states: Key tensor (batch, seq_len, num_key_value_heads, head_dim).
            value_states: Value tensor (batch, seq_len, num_key_value_heads, head_dim).

        Returns:
            Normalized query, key, and value states.
        """
        query_states = self.query_normalization(query_states)
        key_states = self.key_normalization(key_states)

        return query_states, key_states, value_states


class Exaone4DecoderLayer(nn.Module):
    """Single Exaone4 decoder layer with attention and MLP."""

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
        """Initialize a single Exaone4 decoder layer.

        Args:
            config: Model configuration object.
            layer_idx: Index of this layer in the model.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matrix multiplications.
            rngs: Random number generators.
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

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size).
            mask_info: Mask information for attention.
            position_ids: Position indices for RoPE.
            mode: Runtime mode (train/eval/decode).
            cache_view: Cache view for autoregressive generation.
            cache_metadata: Cache metadata.
            output_attentions: Whether to return attention weights.
            frequencies: Precomputed RoPE frequencies.

        Returns:
            DecoderLayerOutput with hidden states and optional attention weights.
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
    """Exaone4 decoder stack with embeddings, transformer layers, and final norm."""

    def __init__(
        self,
        config: Exaone4Config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the base Exaone4 model.

        Args:
            config: Model configuration object.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matrix multiplications.
            rngs: Random number generators.
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
        """Forward pass through the Exaone4 model.

        Args:
            input_ids: Input token IDs (batch, seq_len).
            inputs_embeds: Pre-computed embeddings (alternative to input_ids).
            attention_mask: Attention mask for padding.
            mask_info: Mask information for attention.
            position_ids: Position indices for RoPE.
            mode: Runtime mode (train/eval/decode).
            past_key_values: Cache from previous forward passes.
            cache_metadata: Cache metadata.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.

        Returns:
            Model outputs with last hidden state and optional attention weights/hidden states.
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
        """Returns the embedding layer of the module."""
        return self.embed_tokens

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self


@register_module(TaskType.CAUSAL_LM, config=Exaone4Config, model_type="exaone4")
class Exaone4ForCausalLM(BaseCausalLMModule[Exaone4Model, Exaone4Config]):
    """Exaone4 model with a Causal Language Modeling head."""

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
        """Initialize Exaone4 for causal language modeling.

        Args:
            config: Model configuration object.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matrix multiplications.
            rngs: Random number generators.
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
    """Exaone4 model with a Sequence Classification head."""

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
        """Initialize Exaone4 for sequence classification.

        Args:
            config: Model configuration object.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision for matrix multiplications.
            rngs: Random number generators.
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
