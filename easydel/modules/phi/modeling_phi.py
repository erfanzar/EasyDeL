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

import jax.lax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
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

from .phi_configuration import PhiConfig


class PhiMLP(nn.Module):
    """Phi MLP module.

    This module implements the feed-forward network (MLP) used in the Phi model.
    It consists of two linear projections with a GELU activation in between.

    Attributes:
        config (PhiConfig): Configuration object for the model.
        layer_idx (int, optional): Index of the current layer.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        fc1 (ParallelLinear): First linear projection layer (up-projection).
        fc2 (ParallelLinear): Second linear projection layer (down-projection).
        act (callable): Activation function.
    """

    def __init__(
        self,
        config: PhiConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the PhiMLP module.

        Args:
            config (PhiConfig): The configuration object for the Phi model.
            layer_idx (int, optional): Index of the current layer. Defaults to None.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike, optional): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            kernel_init=nn.initializers.normal(config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            kernel_init=nn.initializers.normal(config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the PhiMLP module.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Output hidden states after MLP transformation.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act(self.fc1(hidden_states)), "mlp_gate")
        hidden_states = checkpoint_name(self.fc2(gate), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class PhiAttention(UnifiedAttention):
    """Phi Attention with Q/K normalization.

    Inherits Q/K normalization from QKNormAttention.
    Features:
    - Uses LayerNorm instead of RMSNorm
    - Standard LayerNorm on full hidden_size (not per-head)
    - Partial RoPE (partial_rotary_factor)
    - Custom bias configuration
    """

    norms_mapping: ClassVar[dict[str, str]] = {
        "query_normalization": "q_layernorm",
        "key_normalization": "k_layernorm",
    }
    projection_mapping: ClassVar[dict[str, str]] = {
        "query_projection": "q_proj",
        "key_projection": "k_proj",
        "value_projection": "v_proj",
        "output_projection": "dense",
    }

    def __init__(
        self,
        config: PhiConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.qk_layernorm = config.qk_layernorm
        config.attention_bias = True
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx if layer_idx is not None else -1,
            attention_type="standard",
            causal=True,
            use_qk_norm=config.qk_layernorm,
        )

        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.partial_rotary_factor = config.partial_rotary_factor
        self.rotary_emb_dim = int(config.partial_rotary_factor * self.head_dim)
        self.is_causal = True

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Override to use standard LayerNorm on hidden_size if qk_layernorm is enabled."""

        return nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=True,
            rngs=rngs,
        )

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Override to use standard LayerNorm on hidden_size if qk_layernorm is enabled."""

        return nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=True,
            rngs=rngs,
        )

    def _create_rotary(self, config, dtype):
        """Override for partial RoPE."""
        return config.get_basic_rope(
            dtype,
            head_size=int(config.partial_rotary_factor * (config.hidden_size // config.num_attention_heads)),
            rotary_dim=int(config.partial_rotary_factor * (config.hidden_size // config.num_attention_heads)),
        )

    def _preprocess_qkv(self, query_states, key_states, value_states):
        if self.use_qk_norm:
            return self.query_normalization(query_states), self.key_normalization(key_states), value_states
        return query_states, key_states, value_states


class PhiDecoderLayer(nn.Module):
    """Phi Transformer Decoder Layer.

    This module represents a single decoder layer in the Phi model,
    combining self-attention and MLP sub-layers with residual connections
    and layer normalization.

    Attributes:
        config (PhiConfig): Configuration object for the model.
        layer_idx (int, optional): Index of the current layer.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        input_layernorm (nn.LayerNorm): Layer normalization applied before the attention and MLP blocks.
        resid_dropout (nn.Dropout): Dropout applied to the residual connection after the MLP block.
        self_attn (PhiAttention): The self-attention module.
        mlp (PhiMLP): The feed-forward (MLP) module.
    """

    def __init__(
        self,
        config: PhiConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the PhiDecoderLayer.

        Args:
            config (PhiConfig): The configuration object for the Phi model.
            layer_idx (int, optional): Index of the current layer. Defaults to None.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike, optional): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        attn_block = PhiAttention
        mlp_block = PhiMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
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
            layer_idx=layer_idx,
        )
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.resid_dropout = nn.Dropout(self.config.resid_pdrop)

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
        """Forward pass of the PhiDecoderLayer module.

        Args:
            hidden_states (Array): Input hidden states.
            attention_mask (Array): Mask to apply on the attention scores.
            position_ids (Array): Position indices for the tokens. Shape: (batch_size, sequence_length).
            causal_mask (tp.Optional[Array | bool]): Causal mask for ensuring autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | RaggedPagesCacheView]): Cache view for attention KVs.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]): Metadata for paged attention.
            segment_ids (tp.Optional[Array]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to return attention weights. Default is False.
            fcm_mask (tp.Optional[Array]): Flash Chunking Mask (FCM) for attention.
            frequencies (tp.Optional[Array]): Precomputed rotary frequency embeddings.

        Returns:
            tp.Tuple[Array, tp.Optional[Array]]:
                A tuple containing the output hidden states and optionally the attention weights.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        attn_outputs = self.self_attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        feed_forward_hidden_states = self.resid_dropout(feed_forward_hidden_states)
        hidden_states = checkpoint_name(
            self.resid_dropout(attn_outputs.attention_output) + feed_forward_hidden_states + residual, "residual"
        )
        hidden_states = checkpoint_name(hidden_states, "layer_output")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=PhiConfig, model_type="phi")
class PhiModel(EasyDeLBaseModule):
    """The base Phi model transformer.

    This class represents the core transformer architecture of the Phi model,
    consisting of an embedding layer, multiple PhiDecoderLayer layers,
    and a final layer normalization.

    Attributes:
        config (PhiConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        embed_tokens (nn.Embed): Embedding layer for input tokens.
        layers (tp.List[PhiDecoderLayer]): List of decoder layers.
        final_layernorm (nn.LayerNorm): Final layer normalization.
        embed_dropout (nn.Dropout): Dropout layer applied after embeddings.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: PhiConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the PhiModel.

        Args:
            config (PhiConfig): The configuration object for the Phi model.
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
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        embed_block = auto_remat(
            nn.Embed,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.embed_dropout = nn.Dropout(config.embd_pdrop, rngs=rngs)
        self.layers = [
            PhiDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_idx=idx,
                rngs=rngs,
            )
            for idx in range(self.config.num_hidden_layers)
        ]
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @functools.cached_property
    def frequencies(self):
        return self.config.get_basic_frequencies(
            head_size=int(
                self.config.partial_rotary_factor * (self.config.hidden_size // self.config.num_attention_heads)
            ),
            rotary_dim=int(
                self.config.partial_rotary_factor * (self.config.hidden_size // self.config.num_attention_heads)
            ),
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
        """Performs a forward pass through the Phi transformer model.

        Processes input tokens through embedding, multiple decoder layers with partial RoPE
        and optional Q/K normalization, and final layer normalization to produce contextualized
        hidden states.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Either this
                or `inputs_embeds` must be provided but not both.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length,
                hidden_size). Use instead of `input_ids` for custom embeddings.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) where True
                indicates tokens to attend to and False indicates padding to ignore.
            mask_info: Pre-computed mask information object. If provided, `attention_mask`
                is ignored.
            position_ids: Explicit position indices of shape (batch_size, sequence_length).
                Auto-generated from mask_info if not provided.
            mode: Runtime mode controlling behavior (MODE_TRAIN, MODE_DECODE, MODE_INFER).
                Auto-detected based on sequence length and cache presence if None.
            past_key_values: Cached key/value states from previous forward passes for
                efficient autoregressive generation. Supports TransformerCache,
                RaggedPagesCache, or HybridCache formats.
            cache_metadata: Metadata for paged attention mechanisms, required when using
                RaggedPagesCache.
            output_attentions: Whether to return attention weights from all layers.
                Defaults to config.output_attentions.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to config.output_hidden_states.

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final layer output of shape (batch, seq_len, hidden_size)
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True
                - attentions: Tuple of all attention weights if output_attentions=True
                - past_key_values: Updated cache for next generation step

        Raises:
            ValueError: If neither `input_ids` nor `inputs_embeds` is provided, or if both
                are provided simultaneously.
            AssertionError: If sequence length exceeds max_position_embeddings.
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

        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

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


@register_module(TaskType.CAUSAL_LM, config=PhiConfig, model_type="phi")
class PhiForCausalLM(BaseCausalLMModule[PhiModel, PhiConfig]):
    """Phi model with a Causal Language Modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "phi"
    _config_class = PhiConfig

    def __init__(
        self,
        config: PhiConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=PhiModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=True,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:
        """Performs forward pass for causal language modeling with Phi.

        Processes input through the base Phi transformer and applies a language modeling
        head to produce next-token prediction logits.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Either this
                or `inputs_embeds` must be provided but not both.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length,
                hidden_size). Use instead of `input_ids` for custom embeddings.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) indicating
                which tokens to attend to (True) and which to ignore (False).
            mask_info: Pre-computed mask information. If provided, overrides `attention_mask`.
            position_ids: Explicit position indices of shape (batch_size, sequence_length).
                Auto-generated if not provided.
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER). Auto-detected if None.
            past_key_values: Cached key/value states for efficient generation.
            cache_metadata: Metadata for paged attention caching.
            apply_lm_head: Whether to apply the language modeling head. Set False to get
                only hidden states without computing logits.

        Returns:
            CausalLMOutput containing:
                - logits: Next-token prediction logits of shape (batch, seq_len, vocab_size)
                    if apply_lm_head=True, else None
                - hidden_states: Tuple of layer outputs if output_hidden_states=True
                - last_hidden_state: Final layer output of shape (batch, seq_len, hidden_size)
                - attentions: Tuple of attention weights if output_attentions=True
                - past_key_values: Updated cache for next generation step
        """
        outputs = self.model(
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
        hidden_states = outputs.last_hidden_state
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
