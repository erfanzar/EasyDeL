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
from typing import ClassVar

import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
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

from .gpt_neox_configuration import GPTNeoXConfig as GPTNeoXConfig


class GPTNeoXAttention(UnifiedAttention):
    """GPT-NeoX Attention with partial RoPE.

    Inherits from UnifiedAttention.
    Uses combined QKV projection (query_key_value) and partial rotary embeddings.
    Overrides forward_standard to efficiently handle fused QKV projection.
    """

    projection_mapping: ClassVar[dict[str, str]] = {
        "query_projection": "q_proj",
        "key_projection": "k_proj",
        "value_projection": "v_proj",
        "output_projection": "dense",
        "query_key_value_projection": "query_key_value",
        # MLA-specific projections (DeepSeek V2/V3)
        "mla_q_proj": "q_proj",
        "mla_q_a_proj": "q_a_proj",
        "mla_q_a_layernorm": "q_a_layernorm",
        "mla_q_b_proj": "q_b_proj",
        "mla_kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
        "mla_kv_a_layernorm": "kv_a_layernorm",
        "mla_kv_b_proj": "kv_b_proj",
    }

    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize GPT-NeoX attention module.

        Args:
            config: GPTNeoXConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
            layer_idx: Index of this layer in the model (0-indexed).
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
            use_fused_qkv=True,
        )

    def _create_rotary(self, config: GPTNeoXConfig, dtype: jnp.dtype):
        """Create GPTNeoX-specific rotary embedding with partial RoPE."""
        return config.get_basic_rope(
            dtype=dtype,
            head_size=self.head_dim,
            rotary_dim=int(self.head_dim * config.rotary_pct),  # Partial RoPE
            base=config.rotary_emb_base,
        )

    def _create_attention_performer(self, config: GPTNeoXConfig, rngs: nn.Rngs):
        """Create attention performer with config dropout."""
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )


class GPTNeoXMlp(nn.Module):
    """GPT-NeoX MLP module.

    This module implements the feed-forward network used in the GPT-NeoX model.

    Attributes:
            config (GPTNeoXConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ) -> None:
        """Initialize GPT-NeoX MLP module.

        Args:
            config: GPTNeoXConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
            layer_idx: Index of this layer in the model (0-indexed).
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense_h_to_4h = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.dense_4h_to_h = RowParallelLinear(
            self.config.intermediate_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the GPTNeoXMlp module.

        Args:
            hidden_states: Input hidden states with shape (batch_size, sequence_length, hidden_size).

        Returns:
            Output hidden states with shape (batch_size, sequence_length, hidden_size) after
            processing through dense_h_to_4h -> activation -> dense_4h_to_h.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = checkpoint_name(
            self.dense_4h_to_h(self.act(checkpoint_name(self.dense_h_to_4h(hidden_states), name="mlp_up"))),
            name="mlp_down",
        )
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class GPTNeoXBlock(nn.Module):
    """GPT-NeoX Transformer block.

    This module represents a single transformer block in the GPT-NeoX model,
    containing self-attention and MLP sub-layers with residual connections
    and layer normalization. It supports both standard and parallel residual connections.

    Attributes:
            config (GPTNeoXConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ) -> None:
        """Initialize GPT-NeoX transformer block.

        Args:
            config: GPTNeoXConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
            layer_idx: Index of this layer in the model (0-indexed).
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.use_parallel_residual = config.use_parallel_residual

        attn_block = GPTNeoXAttention
        mlp_block = GPTNeoXMlp

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = GPTNeoXMlp(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
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
        """Forward pass of the GPTNeoXBlock module.

        Supports both parallel and sequential residual connections based on config.use_parallel_residual.
        - Parallel: attn and mlp both operate on input_layernorm(x), outputs summed with residual
        - Sequential: attn output added to residual, then mlp operates on result

        Args:
            hidden_states: Input hidden states with shape (batch_size, sequence_length, hidden_size).
            mask_info: Masking information containing attention masks and positions.
            position_ids: Position indices for tokens with shape (batch_size, sequence_length).
            mode: Runtime mode (train/decode/prefill) for cache handling.
            cache_view: Optional cache view for key/value states in decoder inference.
            cache_metadata: Optional metadata for cache handling.
            output_attentions: Whether to return attention weights (default: False).
            frequencies: Optional precomputed rotary embedding frequencies with shape
                (sequence_length, head_dim).

        Returns:
            DecoderLayerOutput containing:
                - hidden_states: Output hidden states with shape (batch_size, sequence_length, hidden_size).
                - attention_weight: Optional attention weights if output_attentions=True.
                - cache_view: Updated cache view if cache is used.
        """

        attn_outputs = self.attention(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        if self.use_parallel_residual:
            mlp = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = mlp + hidden_states + attn_outputs.attention_output
        else:
            hidden_states = attn_outputs.attention_output + hidden_states
            hidden_states = self.mlp(self.post_attention_layernorm(hidden_states)) + hidden_states

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=GPTNeoXConfig, model_type="gpt_neox")
class GPTNeoXModel(EasyDeLBaseModule):
    """GPT-NeoX model implementation.

    This class implements the main GPT-NeoX transformer model architecture, consisting of
    an embedding layer, multiple GPTNeoXBlock layers, and a final layer normalization.

    Attributes:
            config (GPTNeoXConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GPT-NeoX base model.

        Args:
            config: GPTNeoXConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embed_in = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.emb_dropout = nn.Dropout(config.hidden_dropout, rngs=rngs)
        self.layers = [
            GPTNeoXBlock(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @functools.cached_property
    def frequencies(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        return self.config.get_basic_frequencies(
            head_size=head_dim,
            rotary_dim=int(head_dim * self.config.rotary_pct),
            base=self.config.rotary_emb_base,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        extra_embedding: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the GPTNeoXModel.

        Args:
            input_ids: Input token IDs with shape (batch_size, sequence_length). Mutually exclusive
                with inputs_embeds.
            attention_mask: Boolean mask with shape (batch_size, sequence_length) to avoid attention
                on padding tokens. Values are True for tokens to attend to, False for padding.
            mask_info: Pre-computed masking information. If None, computed from attention_mask.
            position_ids: Position indices with shape (batch_size, sequence_length). If None,
                automatically generated as sequential positions.
            mode: Runtime mode for cache handling. One of MODE_TRAIN, MODE_DECODE, or MODE_PREFILL.
                Auto-detected if None.
            past_key_values: Cache containing precomputed key/value states from previous decode steps.
            cache_metadata: Metadata for cache operations (positions, sequence lengths, etc.).
            inputs_embeds: Pre-computed input embeddings with shape (batch_size, sequence_length, hidden_size).
                Mutually exclusive with input_ids.
            extra_embedding: Additional embeddings to add to inputs_embeds, with shape
                (batch_size, sequence_length, hidden_size).
            output_attentions: Whether to return attention weights from all layers (default: False).
            output_hidden_states: Whether to return hidden states from all layers (default: False).

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final hidden states with shape (batch_size, sequence_length, hidden_size).
                - hidden_states: Tuple of hidden states from all layers if output_hidden_states=True.
                - attentions: Tuple of attention weights from all layers if output_attentions=True.
                - past_key_values: Updated cache for next decode step.
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        hidden_states = self.emb_dropout(
            inputs_embeds + extra_embedding if extra_embedding is not None else inputs_embeds
        )

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
                frequencies=self.frequencies,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.final_layer_norm(hidden_states)
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
        return self.embed_in


@register_module(TaskType.CAUSAL_LM, config=GPTNeoXConfig, model_type="gpt_neox")
class GPTNeoXForCausalLM(BaseCausalLMModule[GPTNeoXModel, GPTNeoXConfig]):
    """GPT-NeoX model with a language modeling head for autoregressive text generation.

    This model extends GPTNeoXModel with a linear language modeling head that projects
    hidden states to vocabulary logits for next-token prediction.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gpt_neox"
    _config_class = GPTNeoXConfig

    def __init__(
        self,
        config: GPTNeoXConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GPT-NeoX for causal language modeling.

        Args:
            config: GPTNeoXConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
        """
        super().__init__(
            config=config,
            base_model_class=GPTNeoXModel,
            base_model_name="gpt_neox",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            lm_head_name="embed_out",
        )
