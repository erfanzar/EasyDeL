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


import math
from functools import cached_property, partial

import chex
import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from einops import rearrange
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import auto_remat, get_dot_general_by_bits
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    PagesCache,
    PagesCacheView,
    PagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear

from .mosaic_configuration import MptConfig as MptConfig


class MptMLP(nn.Module):
    """MPT MLP module.

    This module implements the feed-forward network (MLP) used in the MPT model.
    It consists of an up-projection, GELU activation, and a down-projection, followed by dropout.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        up_proj (ParallelLinear): Linear layer for up-projection.
        down_proj (ParallelLinear): Linear layer for down-projection.
        hidden_dropout (nn.Dropout): Dropout layer applied to the output.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the MptMLP module.

        Args:
            config (MptConfig): The configuration object for the MPT model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        linear_class = partial(
            ParallelLinear,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.up_proj = linear_class(
            self.config.hidden_size,
            self.config.expansion_ratio * self.config.hidden_size,
            rngs=rngs,
        )
        self.down_proj = linear_class(
            self.config.expansion_ratio * self.config.hidden_size,
            self.config.hidden_size,
            rngs=rngs,
        )
        self.hidden_dropout = nn.Dropout(
            self.config.attn_config.attn_pdrop,
            rngs=rngs,
        )

    def __call__(self, hidden_states: chex.Array, residual: chex.Array):
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        up = jax.nn.gelu(self.up_proj(hidden_states), approximate=False)
        hidden_states = self.down_proj(up)

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return self.hidden_dropout(hidden_states) + residual


class MptAttention(AttentionModule):
    """MPT Attention module.

    This module implements the multi-head attention mechanism used in the MPT model.
    It supports ALiBi positional bias and allows for different attention implementations.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        hidden_size (int): Dimensionality of the hidden states.
        Wqkv (ParallelLinear): Combined linear layer for query, key, and value projections.
        out_proj (ParallelLinear): Linear layer for the output projection.
        dropout (nn.Dropout): Dropout layer applied after the output projection.
        n_heads (int): Number of attention heads.
        max_seq_length (int): Maximum sequence length supported.
        head_dim (int): Dimensionality of each attention head.
        softmax_scale (float): Scale factor for the softmax function.
        attention_performer (FlexibleAttentionModule): Module to perform the core attention computation.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the MptAttention module.

        Args:
            config (MptConfig): The configuration object for the MPT model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size
        self.Wqkv = ParallelLinear(
            config.hidden_size,
            config.hidden_size * 3,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.out_proj = ParallelLinear(
            config.hidden_size,
            config.hidden_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.dropout = nn.Dropout(
            self.config.attn_config.attn_pdrop,
            rngs=rngs,
        )

        self.hidden_size = self.config.hidden_size
        self.n_heads = self.config.n_heads
        self.max_seq_length = self.config.max_seq_len
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = self.config.attn_config.softmax_scale

        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=self.config.attn_config.attn_pdrop,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        position_bias: chex.Array | tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
    ):
        inp_shape = hidden_states.shape
        mixed_qkv = self.Wqkv(hidden_states)
        query_states, key_states, value_states = jnp.split(mixed_qkv, 3, -1)

        query_states = rearrange(
            query_states,
            "b s (h d) -> b s h d",
            h=self.config.n_heads,
        )
        key_states = rearrange(
            key_states,
            "b s (h d) -> b s h d",
            h=self.config.n_heads,
        )
        value_states = rearrange(
            value_states,
            "b s (h d) -> b s h d",
            h=self.config.n_heads,
        )
        (
            key_states,
            value_states,
            attention_mask,
            _,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            fcm_mask=fcm_mask,
        )
        if position_bias is not None:
            position_bias_query_index = max(0, position_bias.shape[2] - query_states.shape[1])
            position_bias_key_index = max(0, position_bias.shape[3] - key_states.shape[1])

            position_bias = position_bias[
                :,
                :,
                position_bias_query_index:,
                position_bias_key_index:,
            ]
        attention_mask = attention_mask.repeat(position_bias.shape[1], 1)
        attention_bias = lax.select(
            attention_mask.astype("bool"),
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype) + position_bias.astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        attention = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=attention_bias,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=lambda: attention_bias,
            attention_mask=None,
            segment_ids=segment_ids,
            causal=False,
        )

        attn_output = self.out_proj(
            self.shard_attention_prod(
                attention.attention_outputs.reshape(inp_shape),
            )
        )

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attention.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class MptBlock(nn.Module):
    """MPT Transformer block.

    This module represents a single transformer block in the MPT model,
    containing self-attention and MLP sub-layers with residual connections
    and layer normalization. It utilizes ALiBi for positional information.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        norm_1 (nn.LayerNorm): Layer normalization before the attention layer.
        attn (MptAttention): The self-attention module.
        norm_2 (nn.LayerNorm): Layer normalization before the MLP layer.
        ffn (MptMLP): The feed-forward (MLP) module.
        resid_attn_dropout (nn.Dropout): Dropout applied after the attention layer's residual connection.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the MptBlock module.

        Args:
            config (MptConfig): The configuration object for the MPT model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        attn_block = MptAttention
        mlp_block = MptMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )

        self.norm_1 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self.attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.norm_2 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self.ffn = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.dropout_rate = self.config.attn_config.attn_pdrop
        self.resid_attn_dropout = nn.Dropout(self.dropout_rate, rngs=rngs)

    def __call__(
        self,
        hidden_states: chex.Array,
        position_bias: chex.Array | tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
    ):
        attn_outputs = self.attn(
            self.norm_1(hidden_states),
            position_bias,
            attention_mask,
            causal_mask,
            mode,
            segment_ids,
            cache_view,
            cache_metadata,
            output_attentions,
            fcm_mask,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs.attention_output) + hidden_states
        output = self.ffn(self.norm_2(hidden_states), hidden_states)

        return DecoderLayerOutput(
            hidden_states=output,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8):
    """Builds the ALiBi tensor for MPT models.

    ALiBi (Attention with Linear Biases) is a method to incorporate positional information
    into transformer models without explicit position embeddings. It adds a bias to the
    attention scores based on the distance between query and key positions.

    Args:
        num_heads (int): The number of attention heads.
        sequence_length (int): The length of the sequence.
        alibi_bias_max (int, optional): The maximum bias value allowed by ALiBi. Defaults to 8.

    Returns:
        chex.Array: The ALiBi tensor of shape (1, num_heads, sequence_length, sequence_length).
    """
    alibi = jnp.arange(
        1 - sequence_length,
        1,
        dtype="i4",
    ).reshape(
        1,
        1,
        1,
        sequence_length,
    )
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    base = jnp.arange(1, num_heads_power_of_2 + 1, dtype=jnp.int32).astype("float32")
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / jnp.pow(2, base)
    slopes = slopes.reshape(
        1,
        num_heads_power_of_2,
        1,
        1,
    )

    if num_heads_power_of_2 != num_heads:
        slopes = jnp.concat(
            [slopes[:, 1::2, ...], slopes[:, ::2, ...]],
            axis=1,
        )[:, :num_heads, ...]

    alibi = alibi * slopes
    return alibi


@register_module(TaskType.BASE_MODULE, config=MptConfig, model_type="mpt")
class MptModel(EasyDeLBaseModule):
    """MPT model implementation.

    This class implements the main MPT transformer model architecture, consisting of
    an embedding layer (token and optional positional), multiple MptBlock layers,
    and a final layer normalization.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        wte (nn.Embed): Token embedding layer.
        emb_drop (nn.Dropout): Dropout layer applied after embeddings.
        blocks (tp.List[MptBlock]): List of transformer blocks.
        norm_f (nn.LayerNorm): Final layer normalization.
        alibi (chex.Array, optional): Precomputed ALiBi tensor if using ALiBi.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the MptModel.

        Args:
            config (MptConfig): The configuration object for the MPT model.
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
        self.wte = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.d_model,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.blocks = [
            MptBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.n_layers)
        ]

        self.norm_f = nn.LayerNorm(
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=config.layer_norm_epsilon,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )

    @cached_property
    def alibi(self):
        return build_mpt_alibi_tensor(
            sequence_length=self.config.max_seq_len,
            num_heads=self.config.n_heads,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        output_attentions: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")

        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, (1, 2))

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.blocks))

        for idx, block in enumerate(self.blocks):
            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                position_bias=self.alibi,
                segment_ids=segment_ids,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            past_key_values[idx] = layer_outputs.cache_view
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        hidden_states = self.norm_f(hidden_states)
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
        return self.wte


@register_module(TaskType.CAUSAL_LM, config=MptConfig, model_type="mpt")
class MptForCausalLM(EasyDeLBaseModule):
    """MPT model with a language modeling head.

    This model extends the base MptModel by adding a linear layer (lm_head)
    on top to predict the next token in a sequence, making it suitable for causal
    language modeling tasks.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        transformer (MptModel): The core MPT transformer model.
        lm_head (ParallelLinear, optional): The language modeling head. If `use_lm_head`
            in the config is True (tying embeddings), this will be None.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the MptForCausalLM model.

        Args:
            config (MptConfig): The configuration object for the MPT model.
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
        self.transformer = MptModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        output_attentions: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        outputs: BaseModelOutput = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            inputs_embeds=inputs_embeds,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(outputs.last_hidden_state)

        return CausalLMOutput(
            logits=logits,
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
        return self.transformer.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.transformer.get_embedding()
