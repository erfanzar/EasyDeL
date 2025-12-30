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
import math
import typing

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
from easydel.infra.utils import auto_remat, block_wise_ffn
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

from .falcon_configuration import FalconConfig


def built_bloom_alibi(attention_mask, num_attention_heads):
    """The built_bloom_alibi function is used to create a bloom alibi for the attention mask.
    The bloom alibi is used in the Bloom Attention layer to ensure that each token has a unique
    attention vector, even if it's masked out. This ensures that all tokens have an equal chance of being selected as
    the most important token in the sequence, which helps with training stability and performance.

    Args:
        attention_mask: Mask out the padding tokens in the input
            sequence
        num_attention_heads: Determine the number of attention heads in
            the model

    Returns:
        A tensor of shape (batch_size, num_attention_heads, 1,
        sequence_length)
    """
    batch_size, sequence_length = attention_mask.shape
    cp2 = 2 ** math.floor(math.log2(num_attention_heads))
    base = jnp.asarray(2 ** (-(2 ** -(math.log2(cp2) - 3))), dtype=jnp.float32)
    powers = jnp.arange(1, 1 + cp2, dtype=jnp.float32)
    slops = jnp.power(base, powers)
    if cp2 != num_attention_heads:
        extra_base = jnp.asarray(2 ** (-(2 ** -(math.log2(2 * cp2) - 3))), dtype=jnp.float32)
        num_rem_heads = min(cp2, num_attention_heads - cp2)
        extra_power = jnp.arange(1, 1 + 2 * num_rem_heads, 2, dtype=jnp.dtype)
        slops = jnp.concatenate([slops, jnp.power(extra_base, extra_power)], axis=0)
    arange_tensor = (((jnp.cumsum(attention_mask, axis=-1)) - 1) * attention_mask)[:, jnp.newaxis, :]
    alibi = slops[..., jnp.newaxis].astype(jnp.bfloat16) * arange_tensor
    return alibi.reshape(batch_size, num_attention_heads, 1, sequence_length)


def dropout_add(
    nn_drop: nn.Dropout,
    x: Array,
    residual: Array,
) -> Array:
    """The dropout_add function is a helper function that adds the residual to the output of
    the dropout layer. This is necessary because we want to use deterministic=True when
    we are evaluating our model, but we still need to add in the residual. The reason for this
    is that during training, we have two paths through our network: one with dropout and one without.
    The path without dropout (residual) allows us to backpropagate gradients through both paths at once.

    Args:
        nn_drop: nn.Dropout: Specify the dropout layer
        x: Array: Pass in the input to the dropout layer
        residual: Array: Add the residual to the output of
            dropout_add
        deterministic: bool: Determine whether the dropout layer is
            active or not

    Returns:
        A tensor that is the sum of the residual and a dropout layer
    """
    out = nn_drop(inputs=x)
    out = residual + out
    return out


class FalconAttention(UnifiedAttention):
    """Falcon attention built on top of the unified attention backend."""

    projection_mapping: typing.ClassVar = dict(UnifiedAttention.projection_mapping)
    projection_mapping.update(
        {
            "query_key_value_projection": "query_key_value",
            "output_projection": "dense",
        }
    )

    def __init__(
        self,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        use_gqa = config.multi_query or (
            config.num_attention_heads != getattr(config, "num_kv_heads", config.num_attention_heads)
        )
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="alibi" if config.alibi else "standard",
            causal=True,
            use_fused_qkv=True,
            use_gqa=use_gqa,
        )

    def _create_fused_qkv_proj(
        self,
        config: FalconConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ) -> ColumnParallelLinear:
        return ColumnParallelLinear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            rngs=rngs,
            use_bias=config.bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )

    def _create_o_proj(
        self,
        config: FalconConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ) -> RowParallelLinear:
        return RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=config.bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )


class FalconMlp(nn.Module):
    """Gated feed-forward network for Falcon decoder blocks."""

    def __init__(
        self,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_bias=self.config.bias,
        )
        self.dense_h_to_4h = linear(
            self.config.hidden_size,
            self.config.ff_factor * self.config.hidden_size,
            rngs=rngs,
        )
        self.dense_4h_to_h = linear(
            self.config.ff_factor * self.config.hidden_size,
            self.config.hidden_size,
            rngs=rngs,
        )

    def __call__(self, x: Array, deterministic: bool = True):
        x = apply_logical_sharding(
            x,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        x = checkpoint_name(
            self.dense_4h_to_h(nn.gelu(checkpoint_name(self.dense_h_to_4h(x), name="mlp_up"), approximate=False)),
            name="mlp_down",
        )
        x = apply_logical_sharding(
            x,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return x


class FalconBlock(nn.Module):
    """Single Falcon transformer block with attention and MLP."""

    def __init__(
        self,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        # Match HuggingFace: set default num_ln_in_parallel_attn for new_decoder_architecture
        if config.num_ln_in_parallel_attn is None and config.new_decoder_architecture:
            config.num_ln_in_parallel_attn = 2

        if not config.parallel_attn:
            self.post_attention_layernorm = nn.LayerNorm(
                self.config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
            self.input_layernorm = nn.LayerNorm(
                self.config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            if config.num_ln_in_parallel_attn == 2:
                self.ln_attn = nn.LayerNorm(
                    self.config.hidden_size,
                    epsilon=config.layer_norm_epsilon,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    rngs=rngs,
                )
                self.ln_mlp = nn.LayerNorm(
                    self.config.hidden_size,
                    epsilon=config.layer_norm_epsilon,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    rngs=rngs,
                )
            else:
                self.input_layernorm = nn.LayerNorm(
                    self.config.hidden_size,
                    epsilon=config.layer_norm_epsilon,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    rngs=rngs,
                )
        attn_block, mlp_block = auto_remat(
            FalconAttention,
            FalconMlp,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.self_attention = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.dropout = nn.Dropout(self.config.attention_dropout)
        self.dropout_mlp = nn.Dropout(self.config.hidden_dropout)

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
        alibi: Array | None = None,
    ) -> DecoderLayerOutput:
        """
        Forward pass of the FalconBlock module.

        Args:
            hidden_states (Array): Input hidden states.
            attention_mask (Array): Mask to apply on the attention scores.
            position_ids (Array): Position indices for the tokens.
            causal_mask (Array, optional): Causal mask for ensuring autoregressive behavior.
            alibi (tp.Optional[Array], optional): Alibi tensor for adding positional bias.
            init_cache (bool, optional): If True, initializes cache for caching keys and values.
            output_attentions (bool, optional): If True, outputs attention weights alongside the hidden states.
            deterministic (bool, optional): If True, disables dropout for deterministic behavior.

        Returns:
            tp.Union[Array, tp.Tuple[Array, Array]]: The output tensor and optionally
                the attention weights.
        """
        residual = hidden_states

        # Match HuggingFace logic for layer normalization
        if self.config.new_decoder_architecture and self.config.num_ln_in_parallel_attn == 2:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attention(
            attention_layernorm_out,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
            alibi,
        )

        # Match HuggingFace logic for mlp_layernorm_out assignment
        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(self.dropout, attn_outputs.attention_output, residual)
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        if (
            self.config.new_decoder_architecture
            and self.config.parallel_attn
            and self.config.num_ln_in_parallel_attn == 1
        ):
            mlp_layernorm_out = attention_layernorm_out

        if self.config.use_scan_mlp:
            mlp_output = block_wise_ffn(
                self.mlp,
                mlp_layernorm_out,
                self.config.scan_mlp_chunk_size,
            )
        else:
            mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attn_outputs.attention_output

        output = dropout_add(self.dropout_mlp, mlp_output, residual)
        return DecoderLayerOutput(
            hidden_states=output,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=FalconConfig, model_type="falcon")
class FalconModel(EasyDeLBaseModule):
    """Falcon decoder-only transformer with embeddings, blocks, and final norm."""

    def __init__(
        self,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.word_embeddings = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.h = [
            FalconBlock(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.ln_f = nn.LayerNorm(
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=config.layer_norm_epsilon,
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
        """Performs forward pass through the Falcon transformer model.

        Processes input tokens through embeddings, stacked Falcon decoder blocks with optional
        ALiBi positional biases or RoPE, and final layer normalization. Supports both parallel
        and sequential attention/MLP computation modes.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Either this
                or `inputs_embeds` must be provided but not both.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length,
                hidden_size). Use instead of `input_ids` for custom embeddings.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) indicating
                which tokens to attend to (True) and which to ignore (False).
            mask_info: Pre-computed mask information. If provided, overrides `attention_mask`.
            position_ids: Explicit position indices of shape (batch_size, sequence_length).
                Required when using RoPE (alibi=False), auto-generated if not provided.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER). Auto-detected if None.
            past_key_values: Cached key/value states for efficient autoregressive generation.
            cache_metadata: Metadata for paged attention mechanisms.
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final layer output of shape (batch, seq_len, hidden_size)
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True
                - attentions: Tuple of all attention weights if output_attentions=True
                - past_key_values: Updated cache for next generation step

        Raises:
            ValueError: If neither `input_ids` nor `inputs_embeds` is provided, or if both
                are provided simultaneously.
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            position_ids = mask_info.q_position_ids

        alibi = None
        if self.config.alibi:
            alibi = built_bloom_alibi(
                mask_info,
                self.config.num_attention_heads,
            ).astype(inputs_embeds.dtype)
        elif position_ids is None:
            position_ids = mask_info.q_position_ids
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.h))
        hidden_states = inputs_embeds
        for idx, layer in enumerate(self.h):
            layer_outputs = layer(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
                alibi=alibi,
            )
            hidden_states = layer_outputs.hidden_states
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.ln_f(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states += hidden_states

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
        return self.word_embeddings


@register_module(TaskType.CAUSAL_LM, config=FalconConfig, model_type="falcon")
class FalconForCausalLM(BaseCausalLMModule[FalconModel, FalconConfig]):
    """Falcon model with a language modeling head for causal language modeling tasks."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "falcon"
    _config_class = FalconConfig

    def __init__(
        self,
        config: FalconConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize a FalconForCausalLM model.

        Args:
            config (FalconConfig): Configuration object for the model.
            dtype (jnp.dtype, optional): Data type for activations and weights. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for computations. Defaults to None.
            rngs (nn.Rngs): Random number generator keys for initialization.
        """
        super().__init__(
            config=config,
            base_model_class=FalconModel,
            base_model_name="transformer",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )
