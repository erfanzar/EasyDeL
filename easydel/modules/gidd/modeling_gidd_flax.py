# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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


import typing as tp
from functools import partial
import warnings

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
)
from easydel.infra.utils import (
    ACT2FN,
    auto_remat,
    block_wise_ffn,
    get_dot_general_by_bits,
)
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

from .gidd_configuration import GiddConfig




class GiddMLP(nn.Module):
    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        linear_class = partial(
            ParallelLinear,
            scale="fan_in",
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self.config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.init_scale),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size)

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        h = apply_logical_sharding(
            h,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        h = self.up_proj(h)
        h = nn.relu(h) ** 2
        h = self.down_proj(h)
        h = apply_logical_sharding(
            h,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return h


class GiddAttention(AttentionModule):
    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", head_dim)

        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_eps = config.qk_norm_eps
        if self.use_qk_norm:
            self.qk_scale = nn.Param(
                jnp.full(
                    (1, 1, self.config.num_attention_heads, 1),
                    2 * jnp.log(config.max_position_embeddings),
                    dtype=self.param_dtype,
                ),
            )
        else:
            self.qk_scale = 1.0

        linear_class = partial(
            ParallelLinear,
            scale="fan_in",
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.init_scale),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.q_proj = linear_class(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.k_proj = linear_class(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.v_proj = linear_class(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.o_proj = linear_class(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
        )

        self.rotary = self.config.get_basic_rope(
            self.dtype,
            self.head_dim,
            self.head_dim,
            True,
        )

        self.attention_performer = FlexibleAttentionModule(
            base_config=self.config,
            softmax_scale=1.0 if self.use_qk_norm else 1.0 / self.head_dim**0.5,
            dropout_prob=0.0,
        )

        
    @jax.named_scope("gidd-flax-attention-concatenate")
    def concatenate(
        self,
        *,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        attention_mask: chex.Array,
        noise_mask: chex.Array,
        # mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
    ) -> tp.Tuple[chex.Array, chex.Array, chex.Array, tp.Callable[[], chex.Array]]:
        """
        Adapted from parent class
        """

        assert query.shape[1] == key.shape[1], "Query and Key lengths must match for GIDD attention."
        if attention_mask is not None:
            if attention_mask.dtype != jnp.bool:
                warnings.warn("attention_mask should be a boolean array", stacklevel=1)
                attention_mask = (attention_mask == 1).astype("b1")

        attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_mask = jnp.repeat(attention_mask, query.shape[1], -2)
        # shape: [Batch, 1, q_len, kv_len]

        if noise_mask is not None:
            if noise_mask.dtype != jnp.bool:
                warnings.warn("noise_mask should be a boolean array", stacklevel=1)
                noise_mask = (noise_mask == 1).astype("b1")
            noise_mask_q = jnp.expand_dims(noise_mask, axis=-1)
            noise_mask_kv = jnp.expand_dims(noise_mask, axis=-2)
            noise_attn_mask = jnp.expand_dims(noise_mask_q >= noise_mask_kv, axis=-3)
            attention_mask = jnp.logical_and(attention_mask, noise_attn_mask)

        def init_attention_bias():
            return jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

        return key, value, attention_mask, init_attention_bias, cache_view
    
    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).sum(-1, keepdims=True) + self.qk_norm_eps)

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        noise_mask: chex.Array,
        position_ids: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: tp.Optional[TransformerCacheView | PagesCacheView] = None,
        cache_metadata: tp.Optional[TransformerMetadata | PagesMetadata] = None,
        segment_ids: tp.Optional[chex.Array] = None,
        output_attentions: bool = False,
        frequencies: tp.Optional[chex.Array] = None,
    ) -> tp.Tuple[chex.Array, chex.Array]:
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        if self.use_qk_norm:
            query_states = self._norm(query_states)
            key_states = self._norm(key_states)

        qshape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        kv_shape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        query_states = query_states.reshape(qshape)
        key_states = key_states.reshape(kv_shape)
        value_states = value_states.reshape(kv_shape)
        (
            query_states,
            key_states,
            value_states,
        ) = self.apply_qkv_shardings(query_states, key_states, value_states)

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
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            value=value_states,
            attention_mask=attention_mask,
            noise_mask=noise_mask,
        )

        attentions = self.attention_performer.forward(
            query_states=query_states * self.qk_scale,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            causal=False,
        )
        attn_output = self.o_proj(
            self.shard_attention_prod(
                attn_output=self._merge_heads(attentions.attention_outputs)
            )
        )
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class GiddRMSNorm(nn.Module):
    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.config = config
        self.epsilon = self.config.rms_norm_eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel = nn.Param(jnp.zeros(self.config.hidden_size, dtype=param_dtype))

    def __call__(self, hidden_states):
        variance = hidden_states.astype(jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return (1 + self.kernel.value.astype(self.dtype)) * jnp.asarray(
            hidden_states, dtype=self.dtype
        )


class GiddLayer(nn.Module):
    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        resid_scale: float = 1.0,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.resid_scale = resid_scale
        attn_block = GiddAttention
        mlp_block = GiddMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )

        self.self_attn: GiddAttention = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.mlp: GiddMLP = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = GiddRMSNorm(
            config=config,
            dtype=dtype,
            param_dtype=jnp.float32,
        )
        self.post_attention_layernorm = GiddRMSNorm(
            config=config,
            dtype=dtype,
            param_dtype=jnp.float32,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        noise_mask: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: tp.Optional[TransformerCacheView | PagesCacheView] = None,
        cache_metadata: tp.Optional[TransformerMetadata | PagesMetadata] = None,
        segment_ids: tp.Optional[chex.Array] = None,
        output_attentions: bool = False,
        frequencies: tp.Optional[chex.Array] = None,
    ):
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            noise_mask,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            segment_ids,
            output_attentions,
            frequencies,
        )
        hidden_states = hidden_states + self.resid_scale * attn_outputs.attention_output

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        hidden_states = hidden_states + self.resid_scale * feed_forward_hidden_states
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


@register_module(
    TaskType.BASE_MODULE,
    config=GiddConfig,
    model_type="Gidd",
)
class GiddModel(EasyDeLBaseModule):
    """Gidd model implementation.

    This implements the Gidd language model architecture, utilizing transformer blocks
    with RMSNorm, rotary position embeddings, and a specific attention mechanism.

    Attributes:
        config (GiddConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
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

        self.resid_scale = config.resid_scale / config.num_hidden_layers

        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.emb_init_scale),
            rngs=rngs,
        )
        self.layers = [
            GiddLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                resid_scale=self.resid_scale,
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = GiddRMSNorm(
            config=config,
            dtype=dtype,
            param_dtype=jnp.float32,
        )

    def __call__(
        self,
        input_ids: tp.Optional[chex.Array] = None,
        inputs_embeds: tp.Optional[chex.Array] = None,
        attention_mask: tp.Optional[chex.Array] = None,
        position_ids: tp.Optional[chex.Array] = None,
        log_snr: tp.Optional[chex.Array] = None,
        noise_mask: tp.Optional[chex.Array] = None,
        segment_ids: tp.Optional[chex.Array] = None,
        mode: tp.Optional[common_types.RUNTIME_MODE_TYPES] = None,  # type:ignore
        past_key_values: tp.Optional[TransformerCache | PagesCache] = None,
        cache_metadata: tp.Optional[TransformerMetadata | PagesMetadata] = None,
        output_attentions: tp.Optional[bool] = None,
        output_hidden_states: tp.Optional[bool] = None,
    ) -> BaseModelOutput:
        """Forward pass through the Gidd model.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing precomputed key/value states.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
            Union[BaseModelOutput, Tuple]: Model outputs (last hidden state, optional hidden states, optional attentions)
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )
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
                attention_mask=attention_mask,
                position_ids=position_ids,
                noise_mask=noise_mask,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                segment_ids=segment_ids,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )


@register_module(
    TaskType.DIFFUSION_LM,
    config=GiddConfig,
    model_type="Gidd",
)
class GiddForDiffusionLM(EasyDeLBaseModule):
    """Gidd model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with causal attention masks
    applied to perform autoregressive language generation.

    Attributes:
        config (GiddConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.float32).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.float32).
        precision (tp.Optional[tp.Union[str, jax.lax.Precision]]): Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
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
        self.model = GiddModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.head_init_scale),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: tp.Optional[chex.Array] = None,
        inputs_embeds: tp.Optional[chex.Array] = None,
        attention_mask: tp.Optional[chex.Array] = None,
        position_ids: tp.Optional[chex.Array] = None,
        segment_ids: tp.Optional[chex.Array] = None,
        log_snr: tp.Optional[chex.Array] = None,
        noise_mask: tp.Optional[chex.Array] = None,
        mode: tp.Optional[common_types.RUNTIME_MODE_TYPES] = None,  # type:ignore
        past_key_values: tp.Optional[TransformerCache | PagesCache] = None,
        cache_metadata: tp.Optional[TransformerMetadata | PagesMetadata] = None,
        output_attentions: tp.Optional[bool] = None,
        output_hidden_states: tp.Optional[bool] = None,
    ) -> CausalLMOutput:
        """Forward pass through the Gidd model for causal language modeling.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing precomputed key/value states.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
            Union[CausalLMOutput, Tuple]: Model outputs (logits, optional hidden states, optional attentions)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            log_snr=log_snr,
            noise_mask=noise_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        if self.config.tie_word_embeddings:
            lm_logits = jax.lax.dot_general(
                hidden_states,
                self.model.embed_tokens.embedding.value.T,
                (((hidden_states.ndim - 1), (0,)), ((), ())),
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )
