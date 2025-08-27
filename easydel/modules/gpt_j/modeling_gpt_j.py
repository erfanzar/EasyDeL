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


from functools import cached_property, partial

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn, get_dot_general_by_bits
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

from .gpt_j_configuration import GPTJConfig as GPTJConfig

logger = get_logger(__name__)


class GPTJAttention(AttentionModule):
    """GPT-J Attention module.

    This module implements the attention mechanism used in the GPT-J model,
    including rotary position embeddings.

    Attributes:
            config (GPTJConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            causal (bool): Whether the attention is causal.
            is_cross_attention (bool): Whether the attention is cross-attention.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)

        self.precision = precision
        self.dtype = dtype
        self.rngs = rngs
        self.is_cross_attention = is_cross_attention
        self.causal = causal
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.rotary_dim = config.rotary_dim

        linear = partial(
            ParallelLinear,
            self.embed_dim,
            self.embed_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            param_dtype=param_dtype,
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.q_proj, self.k_proj, self.v_proj = (
            linear(rngs=rngs),
            linear(rngs=rngs),
            linear(rngs=rngs),
        )
        self.out_proj = linear(rngs=rngs)

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)

        self.rotary = self.config.get_basic_rope(
            self.dtype,
            head_size=self.embed_dim,
            rotary_dim=self.rotary_dim,
            base=10000,
            is_neox_style=False,
        )

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=config.attn_pdrop,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        causal_mask: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        frequencies: chex.Array | None = None,
    ):
        """Forward pass of the GPTJAttention module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array, optional): Causal mask for ensuring autoregressive behavior.
            segment_ids (tp.Optional[chex.Array], optional): Segment IDs for segment-based attention.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView], optional): Cache
                view for key_states/value_states states.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata], optional):
                Metadata for cache handling.
            output_attentions (bool, optional): Whether to return attention weights.
            frequencies (tp.Optional[chex.Array], optional): Precomputed rotary frequencies.

        Returns:
            tp.Tuple[chex.Array, tp.Optional[chex.Array]]: A tuple containing the attention output
                and optionally the attention weights.
        """
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        query_states, key_states = self.rotary(
            positions=position_ids,
            query=query_states,
            key=key_states,
            frequencies=frequencies,
        )

        (
            key_states,
            value_states,
            attention_mask,
            init_attention_bias,
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
            fcm_mask=None,
        )
        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            causal=True,
        )
        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class GPTJMLP(nn.Module):
    """GPT-J MLP module.

    This module implements the feed-forward network used in the GPT-J model.

    Attributes:
            config (GPTJConfig): Configuration object for the model.
            intermediate_size (int): Dimensionality of the intermediate layer.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTJConfig,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config: GPTJConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.intermediate_size = intermediate_size
        embed_dim = config.hidden_size
        kernel_init = nn.initializers.normal(config.initializer_range)

        self.fc_in = ParallelLinear(
            embed_dim,
            intermediate_size,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.fc_out = ParallelLinear(
            intermediate_size,
            embed_dim,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(rate=config.resid_pdrop)

    def __call__(self, hidden_states):
        """Forward pass of the GPTJMLP module.

        Args:
            hidden_states (chex.Array): Input hidden states.

        Returns:
            chex.Array: Output hidden states after processing through the MLP.
        """
        hidden_states = self.dropout(self.fc_out(self.act(self.fc_in(hidden_states))))
        return hidden_states


class GPTJBlock(nn.Module):
    """GPT-J Transformer block.

    This module represents a single transformer block in the GPT-J model,
    containing self-attention and MLP sub-layers with residual connections
    and layer normalization.

    Attributes:
            config (GPTJConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config: GPTJConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        hidden_size = self.config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        attn_block = GPTJAttention
        mlp_block = GPTJMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )
        self.ln_1 = nn.LayerNorm(
            self.config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.attn = attn_block(
            config,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

        self.mlp = mlp_block(
            config,
            inner_dim,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        causal_mask: chex.Array | None = None,
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        frequencies: chex.Array | None = None,
    ):
        """Forward pass of the GPTJBlock module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array, optional): Causal mask for ensuring autoregressive behavior.
            segment_ids (tp.Optional[chex.Array], optional): Segment IDs for segment-based attention.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView], optional): Cache view for
                key_states/value_states states.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata], optional): Metadata for
                cache handling.
            output_attentions (bool, optional): Whether to return attention weights.
            frequencies (tp.Optional[chex.Array], optional): Precomputed rotary frequencies.

        Returns:
            tp.Tuple[chex.Array, tp.Optional[chex.Array]]: A tuple containing the output hidden states and
                optionally the attention weights.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            attention_mask,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            causal_mask,
            segment_ids,
            output_attentions,
            frequencies,
        )
        attn_output = attn_outputs.attention_output
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = attn_output + feed_forward_hidden_states + residual
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


@register_module(TaskType.BASE_MODULE, config=GPTJConfig, model_type="gptj")
class GPTJModel(EasyDeLBaseModule):
    """GPT-J model implementation.

    This class implements the main GPT-J transformer model architecture, consisting of
    an embedding layer, multiple GPTJBlock layers, and a final layer normalization.

    Attributes:
            config (GPTJConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
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
        self.embed_dim = config.hidden_size
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(
            rate=self.config.embd_pdrop,
            rngs=rngs,
        )
        self.h = [
            GPTJBlock(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.ln_f = nn.LayerNorm(
            self.config.hidden_size,
            epsilon=self.config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @cached_property
    def frequencies(self):
        embed_dim = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = embed_dim // num_heads

        rotary_dim = self.config.rotary_dim
        return self.config.get_basic_frequencies(
            rotary_dim=rotary_dim,
            head_size=head_dim,
            base=10000,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        inputs_embeds: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        extra_embedding: chex.Array | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the GPTJModel.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing precomputed
                key_states/value_states states.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
            extra_embedding (chex.Array, optional): Additional embedding to add to input embeddings.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
            Union[BaseModelOutput, Tuple]: Model outputs (last hidden state, optional hidden states, optional attentions)
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        hidden_states = inputs_embeds + extra_embedding if extra_embedding is not None else inputs_embeds

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.h))

        hidden_states = self.dropout(inputs_embeds)
        for idx, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                position_ids=position_ids,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
                frequencies=self.frequencies,
                causal_mask=self.causal_mask,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            past_key_values[idx] = layer_outputs.cache_view
        hidden_states = self.ln_f(hidden_states)

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


@register_module(TaskType.CAUSAL_LM, config=GPTJConfig, model_type="gptj")
class GPTJForCausalLM(EasyDeLBaseModule):
    """GPT-J model with a language modeling head.

    This model extends the base GPTJModel by adding a linear layer on top to
    predict the next token in a sequence, making it suitable for causal language
    modeling tasks.

    Attributes:
            config (GPTJConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
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
        self.transformer = GPTJModel(
            self.config,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=self.precision,
            rngs=rngs,
        )
        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            rngs=rngs,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            param_dtype=self.dtype,
            precision=self.precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        extra_embedding: chex.Array | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the GPTJForCausalLM model.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing precomputed
                key_states/value_states states.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
            extra_embedding (chex.Array, optional): Additional embedding to add to input embeddings.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
            Union[CausalLMOutput, Tuple]: Model outputs (logits, optional hidden states, optional attentions)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            extra_embedding=extra_embedding,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            position_ids=position_ids,
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
            lm_logits = self.apply_lm_head(hidden_states)

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
