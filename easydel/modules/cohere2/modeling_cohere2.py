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


from functools import partial

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
    SequenceClassifierOutput,
)
from easydel.infra.utils import auto_remat, block_wise_ffn, get_dot_general_by_bits
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

from .cohere2_configuration import Cohere2Config


class Cohere2LayerNorm(nn.Module):
    """Cohere Layer Normalization.

    Attributes:
      dim (Union[int, tuple]): The dimension(s) to normalize over.
      eps (float): A small epsilon value to prevent division by zero.
      dtype (jnp.dtype): The data type for computation.
      param_dtype (jnp.dtype): The data type for the parameters.
      rngs (Optional[nn.Rngs]): Random number generators.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        dim: int | tuple,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nn.Rngs = None,
    ):
        super().__init__()

        if rngs is None:
            rngs = nn.Rngs(0)
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel = nn.Param(
            nn.initializers.ones(
                key=rngs.params(),
                shape=(self.dim,) if isinstance(self.dim, int) else self.dim,
                dtype=self.param_dtype,
            ),
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the Layer Normalization for a given input tensor."""
        mean = jnp.mean(x, -1, keepdims=True)
        variance = jnp.mean(jnp.pow((x - mean), 2), -1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(variance + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies Layer Normalization to the input tensor.

        Args:
          x (jnp.ndarray): The input tensor.

        Returns:
          jnp.ndarray: The normalized output tensor.
        """
        if self.dtype in [
            jnp.float8_e4m3b11fnuz,
            jnp.float8_e4m3fn,
            jnp.float8_e4m3fnuz,
            jnp.float8_e5m2,
            jnp.float8_e5m2fnuz,
        ]:
            x = x.astype(jnp.float32)
        else:
            x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = self.kernel.value.astype(self.dtype)
        return output * weight


class Cohere2Attention(AttentionModule):
    """
    Cohere2 Attention module, incorporating features like RoPE and sliding window attention.

    Attributes:
      config (Cohere2Config): Configuration object.
      layer_idx (int): The index of the current layer.
      dtype (jnp.dtype): Data type for computation.
      param_dtype (jnp.dtype): Data type for parameters.
      precision (jax.lax.PrecisionLike): JAX precision level.
      rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Cohere2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads

        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.q_proj = linear_class(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = linear_class(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = linear_class(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = linear_class(config.num_attention_heads * self.head_dim, config.hidden_size)
        self.layer_idx = layer_idx
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.rotary = self.config.get_basic_rope(self.dtype, self.head_dim, self.head_dim, False)
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=config.attention_dropout,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """Forward pass for the Cohere2 attention module.

        Args:
          hidden_states (chex.Array): Input hidden states.
          attention_mask (chex.Array): Attention mask.
          position_ids (chex.Array): Position IDs for RoPE.
          causal_mask (Optional[chex.Array | bool]): Causal mask.
          cache_view (Optional[TransformerCacheView | PagesCacheView]): Cache view for kv-caching.
          cache_metadata (Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention.
          segment_ids (Optional[chex.Array]): Segment IDs (if applicable).
          output_attentions (bool): Whether to output attention weights.
          fcm_mask (Optional[chex.Array]): FCM mask (if applicable).
          frequencies (Optional[chex.Array]): Precomputed RoPE frequencies.

        Returns:
          Tuple[chex.Array, Optional[chex.Array]]: Attention output and optionally attention weights.
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        (query_states, key_states, value_states) = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )

        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        if self.sliding_window is not None:
            query_states, key_states = self.rotary(
                query=query_states,
                key=key_states,
                positions=position_ids,
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
            fcm_mask=fcm_mask,
            sliding_window=self.sliding_window,
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
            sliding_window=self.sliding_window,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = self.o_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Cohere2MLP(nn.Module):
    def __init__(
        self,
        config: Cohere2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = linear_class(config.hidden_size, config.intermediate_size)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size)
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size)

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = jax.nn.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class Cohere2Block(nn.Module):
    def __init__(
        self,
        config: Cohere2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        attn_block = Cohere2Attention
        mlp_block = Cohere2MLP

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )
        self.self_attn = attn_block(
            config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = mlp_block(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = Cohere2LayerNorm(
            self.config.hidden_size,
            eps=self.config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.is_sliding = (layer_idx + 1) % self.config.sliding_window_pattern != 0
        self.sliding_window = config.sliding_window

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """
        Forward pass of the module block.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights alongside the hidden states.
            fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
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
            attention_mask,
            position_ids,
            causal_mask,
            mode,
            cache_view,
            cache_metadata,
            segment_ids,
            output_attentions,
            fcm_mask,
            frequencies,
        )

        feed_forward_input = hidden_states

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        hidden_states = attn_outputs.attention_output + feed_forward_hidden_states + residual
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=None,
            gate_loss=None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Cohere2Config, model_type="cohere2")
class Cohere2Model(EasyDeLBaseModule):
    def __init__(
        self,
        config: Cohere2Config,
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
        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Cohere2Block(
                config=config,
                layer_idx=idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = Cohere2LayerNorm(
            self.config.hidden_size,
            eps=self.config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
    ) -> BaseModelOutput:
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
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
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
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For Cohere2Model (decoder-only), this is not applicable.
        """
        raise NotImplementedError("Cohere2Model is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For Cohere2Model, this is the model itself.
        """
        return self

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        Cohere2Model does not include the lm_head.
        """
        raise NotImplementedError("Cohere2Model does not include the language model head. See Cohere2ForCausalLM.")

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Cohere2Config, model_type="cohere2")
class Cohere2ForCausalLM(EasyDeLBaseModule):
    def __init__(
        self,
        config: Cohere2Config,
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
        self.model = Cohere2Model(
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
            kernel_init=nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.logit_scale = self.config.logit_scale

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:
        """
        Forward pass through the Cohere module.

        Args:
            input_ids (chex.Array): Input tensor containing token IDs.
            attention_mask (chex.Array): Mask for attention.
            position_ids (chex.Array): Positional indices.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for different input parts.
            inputs_embeds (tp.Optional[chex.Array]): Embedded input tensor.
            output_attentions (tp.Optional[bool]): If True, output attention weights.
            output_hidden_states (tp.Optional[bool]): If True, output hidden states.
            init_cache (bool): If True, initialize cache for decoding.
            deterministic (bool): If True, disable dropout.

        Returns:
            CausalLMOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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

    def apply_lm_head(self, hidden_states: chex.Array) -> chex.Array:
        """
        Applies the language model head to the hidden states.

        Args:
            hidden_states (chex.Array): The last hidden states from the model.

        Returns:
            chex.Array: The logits after applying the language model head.
        """
        lm_logits = self.lm_head(hidden_states)
        if self.logit_scale is not None:
            lm_logits *= self.logit_scale
        return lm_logits

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For Cohere2ForCausalLM (decoder-only), this is not applicable.
        """
        raise NotImplementedError("Cohere2ForCausalLM is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For Cohere2ForCausalLM, this is the underlying Cohere2Model.
        """
        return self.model.get_decoder()  # self.model is the Cohere2Model instance

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        # Access the embedding layer through the decoder (Cohere2Model)
        return self.model.get_embedding()  # Leverages Cohere2Model's get_embedding


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Cohere2Config, model_type="cohere2")
class Cohere2ForSequenceClassification(EasyDeLBaseModule):
    def __init__(
        self,
        config: Cohere2Config,
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
        self.model = Cohere2Model(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        assert hasattr(config, "num_labels"), (
            "in order to use `SequenceClassification` Models in `EasyDeL` "
            "you first need to attach `num_labels` to model `config`"
        )
        self.score = ParallelLinear(
            config.hidden_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
    ) -> SequenceClassifierOutput:
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
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

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For Cohere2ForSequenceClassification (decoder-only), this is not applicable.
        """
        raise NotImplementedError(
            "Cohere2ForSequenceClassification is a decoder-only model and does not have a separate encoder."
        )

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For Cohere2ForSequenceClassification, this is the underlying Cohere2Model.
        """
        return self.model  # self.model is the Cohere2Model instance

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        Cohere2ForSequenceClassification uses a classification head instead.
        """
        raise NotImplementedError(
            "Cohere2ForSequenceClassification uses a classification head (self.score), not an lm_head."
        )

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        # Access the embedding layer through the decoder (Cohere2Model)
        return self.model.get_embedding()  # Leverages Cohere2Model's get_embedding
