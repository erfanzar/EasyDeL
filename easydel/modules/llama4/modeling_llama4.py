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
import typing as tp
from functools import partial

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.pytree import auto_pytree
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
    EncoderLayerOutput,
    ModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, get_dot_general_by_bits
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
from easydel.layers.norms import RMSNorm as Llama4TextRMSNorm
from easydel.utils.compiling_utils import ejit

from .llama4_configuration import Llama4Config, Llama4TextConfig, Llama4VisionConfig


@auto_pytree
class Llama4CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llama4Vision causal language model (or autoregressive) outputs.

    Args:
        loss (`chex.Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`chex.Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(chex.Array))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`):
            Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(chex.Array)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `chex.Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(chex.Array)`, *optional*, returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`):
            Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`chex.Array`, *optional*):
            A `chex.Array` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: chex.Array | None = None
    logits: chex.Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    image_hidden_states: chex.Array | None = None


def bmm(inputs, kernel, precision):
    subscript = "...ik,...kj->...ij" if inputs.ndim > 1 else "...k,...kj->...j"

    return jnp.einsum(
        subscript,
        inputs,
        kernel,
        precision=precision,
        optimize=True,
    )


@ejit(static_argnums=(0, 1, 2, 3))
def _vision_freqs(idx, hidden_size, num_attention_heads, rope_theta):
    img_idx = jnp.arange(idx**2, dtype="i4").reshape(idx**2, 1)
    img_idx = jnp.concatenate([img_idx, img_idx[:1]], axis=0)
    img_idx = img_idx.at[-1, -1].set(-2)
    frequencies_x = img_idx % idx
    frequencies_y = img_idx // idx
    freq_dim = hidden_size // num_attention_heads // 2
    rope_arange = jnp.arange(0, freq_dim, 2)
    rope_arange_sliced = rope_arange[: (freq_dim // 2)]
    rope_freq = 1.0 / (rope_theta ** (rope_arange_sliced.astype("f4") / freq_dim))
    rope_freq_broadcast = rope_freq[None, None, :]
    freqs_x = jnp.repeat((frequencies_x + 1).astype("f4")[..., None] * rope_freq_broadcast, 2, axis=-1)
    freqs_y = jnp.repeat((frequencies_y + 1).astype("f4")[..., None] * rope_freq_broadcast, 2, axis=-1)
    freqs = jnp.concatenate([freqs_x, freqs_y], axis=-1)[..., ::2]
    freqs = jnp.where(img_idx.reshape(-1, 1, 1) < 0, 0.0, freqs)
    return jnp.exp(1j * freqs)


def _create_chunked_attention_mask(
    attention_chunk_size: int,
    start: int,
    end: int,
):
    blcok_position = jnp.abs(
        (jnp.arange(start, end)[None, :] // attention_chunk_size)
        - jnp.arange(start, end)[:, None] // attention_chunk_size
    )
    token_position = jnp.arange(start, end)[None, :] - jnp.arange(start, end)[:, None]
    return ((blcok_position == 0) & (token_position <= 0)).astype("b1")


class Llama4TextExperts(nn.Module):
    """Mixture of Experts module for Llama4 text models.

    Implements a sparse mixture of experts with top-k routing,
    enabling efficient scaling and specialization of model capacity.
    """

    def __init__(
        self,
        config: Llama4Config,
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
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        kernel_init = jax.nn.initializers.normal(config.initializer_range)

        self.gate_up_proj = nn.Param(
            kernel_init(
                rngs.params(),
                (self.num_experts, self.hidden_size, 2 * self.expert_dim),
                self.param_dtype,
            )
        )
        self.down_proj = nn.Param(
            kernel_init(
                rngs.params(),
                (self.num_experts, self.expert_dim, self.hidden_size),
                self.param_dtype,
            )
        )

        self.activation_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = hidden_states.reshape(self.num_experts, -1, self.hidden_size)
        gate_up = bmm(hidden_states, self.gate_up_proj, self.precision)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        next_states = bmm((up * self.activation_fn(gate)), self.down_proj, self.precision)
        return next_states.reshape(-1, self.hidden_size)


class Llama4TextL2Norm(nn.Module):
    """L2 normalization layer for Llama4 text models.

    Normalizes inputs using L2 norm with learned scaling parameters,
    providing stable gradients during training.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    @jax.named_scope("easydel-L2norm")
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._norm(x.astype(jnp.float32)).astype(x.dtype)


class Llama4TextMLP(nn.Module):
    """Multi-Layer Perceptron for Llama4 text models.

    Implements feedforward network with SwiGLU activation function
    for improved representation learning.
    """

    def __init__(
        self,
        config: Llama4Config,
        intermediate_size=None,
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
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = linear_class(config.hidden_size, intermediate_size)
        self.down_proj = linear_class(intermediate_size, config.hidden_size)
        self.up_proj = linear_class(config.hidden_size, intermediate_size)
        self.activation_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        gate = self.activation_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        return hidden_states


class Llama4TextMoe(nn.Module):
    """Mixture of Experts layer for Llama4 text models.

    Routes inputs to specialized expert networks based on learned routing,
    allowing for conditional computation and increased model capacity.
    """

    def __init__(
        self,
        config: Llama4Config,
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
        self.rngs = rngs
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.router = ParallelLinear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=False,
            precision=precision,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.shared_expert = Llama4TextMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        assert hidden_dim == self.hidden_dim, "Input hidden_dim mismatch"

        # Reshape to [batch*seq_len, hidden_dim]
        flattened_hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        tokens_per_expert = flattened_hidden_states.shape[0]

        router_logits = self.router(flattened_hidden_states)
        router_top_value, router_indices_topk = jax.lax.top_k(router_logits, self.top_k)

        scores_base = jnp.full_like(router_logits, -jnp.inf)
        token_idx = jnp.arange(tokens_per_expert)[:, None]
        expert_idx = router_indices_topk
        scores_scattered = scores_base.at[token_idx, expert_idx].set(router_top_value)

        router_scores = jax.nn.sigmoid(scores_scattered.astype(jnp.float32)).astype(hidden_states.dtype)
        out = self.shared_expert(flattened_hidden_states)
        expert_outputs = jnp.zeros_like(out)
        for expert_idx in range(self.num_experts):
            expert_mask = router_scores[:, expert_idx : expert_idx + 1]
            expert_inputs = flattened_hidden_states * expert_mask
            expert_output = self.experts(expert_inputs)
            expert_outputs = expert_outputs + expert_output
        final_output = out + expert_outputs
        final_output = final_output.reshape(batch, seq_len, hidden_dim)
        router_scores_transposed = router_scores.T
        return final_output, router_scores_transposed


class Llama4TextAttention(AttentionModule):
    def __init__(
        self,
        config: Llama4TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads

        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.q_proj = linear_class(config.hidden_size, config.num_attention_heads * self.head_dim, rngs=rngs)
        self.k_proj = linear_class(config.hidden_size, config.num_key_value_heads * self.head_dim, rngs=rngs)
        self.v_proj = linear_class(config.hidden_size, config.num_key_value_heads * self.head_dim, rngs=rngs)
        self.o_proj = linear_class(config.num_attention_heads * self.head_dim, config.hidden_size, rngs=rngs)
        self.use_rope = int((layer_idx + 1) % 4 != 0)
        if self.use_rope:
            self.rotary = self.config.get_basic_rope(
                self.dtype,
                self.head_dim,
                self.head_dim,
                True,
            )
        self.scaling = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=self.config,
            softmax_scale=self.scaling,
            dropout_prob=0.0,
        )
        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm()

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
    ) -> AttentionLayerOutput:
        batch_size, sequence_length = hidden_states.shape[:2]
        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        qshape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        kv_shape = (
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        query_states = query_states.reshape(qshape)
        key_states = key_states.reshape(kv_shape)
        value_states = value_states.reshape(kv_shape)
        if self.use_rope:
            query_states, key_states = self.apply_complex_rotary(
                query_states,
                key_states,
                frequencies,
            )
        if hasattr(self, "qk_norm"):
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                jnp.log(jnp.floor((position_ids.astype("f4") + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            attn_scales = attn_scales.reshape((*input_shape, 1, 1))
            query_states = (query_states * attn_scales).astype(query_states.dtype)
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
            causal=False,
        )
        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = self.o_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Llama4TextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        layer_idx: int,
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
        attn_block = Llama4TextAttention
        mlp_block = Llama4TextMLP
        moe_block = Llama4TextMoe
        attn_block, mlp_block, moe_block = auto_remat(
            attn_block,
            mlp_block,
            moe_block,
            policy=config.gradient_checkpointing,
        )

        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)  # <=> use rope
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = moe_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.feed_forward = mlp_block(
                config=config,
                intermediate_size=config.intermediate_size_mlp,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

        self.input_layernorm = Llama4TextRMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = Llama4TextRMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layer_idx = layer_idx

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
        output_router_logits: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
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

        hidden_states = hidden_states + attn_outputs.attention_output

        feed_forward_input = self.post_attention_layernorm(hidden_states)
        # TODO: Support Chunked MLP for LLaMA4
        # if self.config.use_scan_mlp:
        # 	feed_forward_hidden_states = block_wise_ffn(
        # 		self.feed_forward,
        # 		feed_forward_input,
        # 		self.config.scan_mlp_chunk_size,
        # 	)
        # else:
        feed_forward_hidden_states = self.feed_forward(feed_forward_input)
        if self.is_moe_layer:
            feed_forward_hidden_states, router_logits = feed_forward_hidden_states
        else:
            router_logits = None

        hidden_states = hidden_states + feed_forward_hidden_states.reshape(feed_forward_input.shape)
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Llama4TextConfig, model_type="llama4_text")
class Llama4TextModel(EasyDeLBaseModule):
    def __init__(
        self,
        config: Llama4TextConfig,
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
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            rngs=rngs,
        )
        self.layers = [
            Llama4TextDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(self.config.num_hidden_layers)
        ]
        self.norm = Llama4TextRMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Llama model.

        Args:
          input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
          inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
          attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
          position_ids (chex.Array, optional): Indices of positions of each input sequence token.
          segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
          past_key_values (TransformerCache | PagesCache, optional):
            Cache containing precomputed key/value states.
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
        causal_mask = jnp.expand_dims(
            _create_chunked_attention_mask(
                self.config.attention_chunk_size,
                0,
                sequence_length,
            ),
            (0, 1),
        )
        frequencies = self.compute_complex_rotary(position_ids)

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
                causal_mask=causal_mask,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
                frequencies=frequencies,
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


@register_module(TaskType.CAUSAL_LM, config=Llama4TextConfig, model_type="llama4_text")
class Llama4ForCausalLM(EasyDeLBaseModule):
    def __init__(
        self,
        config: Llama4TextConfig,
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
        self.model = Llama4TextModel(
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
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass through the Llama model for causal language modeling.

        Args:
          input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
          inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
          attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
          position_ids (chex.Array, optional): Indices of positions of each input sequence token.
          segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
          past_key_values (TransformerCache | PagesCache, optional):
            Cache containing precomputed key/value states.
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


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Llama4TextConfig, model_type="llama4_text")
class Llama4ForSequenceClassification(EasyDeLBaseModule):
    """Llama model for sequence classification tasks.

    This class extends the base Llama model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
      config (LlamaConfig): Configuration for the model.
      dtype (jnp.dtype): Data type for computations.
      param_dtype (jnp.dtype): Data type for parameters.
      precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Llama4TextConfig,
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
        self.model = Llama4TextModel(
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
            self.config.hidden_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the Llama model for sequence classification.

        This method processes input sequences through the Llama model and applies
        a classification head to the output.

        Args:
          input_ids (chex.Array, optional): Input token IDs,
            shape (batch_size, sequence_length).
          inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
          attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
          position_ids (chex.Array, optional): Indices of positions of each input sequence token.
          segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
          past_key_values (TransformerCache | PagesCache, optional):
            Cache containing precomputed key/value states.
          cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
          output_attentions (bool, optional): Whether to return attention weights.
          output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
          Union[SequenceClassifierOutput, Tuple]: Classification outputs including logits and optional model outputs
        """
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


class Llama4VisionMLP2(nn.Module):
    """Two-layer MLP module for Llama4 vision models.

    Implements a simple two-layer feedforward network with GELU activation
    for vision feature transformation.
    """

    def __init__(
        self,
        config,
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
        self.rngs = rngs
        self.activation_fn = ACT2FN["gelu"]
        linear_class = partial(
            ParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc1 = linear_class(self.intermediate_size, config.projector_input_dim)
        self.fc2 = linear_class(config.projector_output_dim, config.projector_output_dim)

    def __call__(self, hidden_states: chex.Array) -> chex.Array:
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return self.activation_fn(hidden_states)


class Llama4MultiModalProjector(nn.Module):
    """Multi-modal projector for Llama4 vision-language models.

    Projects vision features into the text embedding space using MLP layers,
    enabling cross-modal understanding and generation.
    """

    def __init__(
        self,
        config,
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
        self.rngs = rngs
        self.linear_1 = ParallelLinear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.01),
        )

    def __call__(self, hidden_states: chex.Array) -> chex.Array:
        return self.linear_1(hidden_states)


def pixel_shuffle(input_tensor, shuffle_ratio):
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.shape

    reshaped_tensor = input_tensor.reshape(
        batch_size,
        height,
        int(width * shuffle_ratio),
        int(channels / shuffle_ratio),
    )
    reshaped_tensor = jnp.transpose(reshaped_tensor, (0, 2, 1, 3))
    reshaped_tensor = reshaped_tensor.reshape(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped_tensor = jnp.transpose(reshaped_tensor, (0, 2, 1, 3))

    output_tensor = reshaped_tensor.reshape(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionPixelShuffleMLP(nn.Module):
    """Pixel shuffle MLP for Llama4 vision models.

    Performs spatial downsampling of vision features through pixel shuffling
    and MLP transformations for efficient processing.
    """

    def __init__(
        self,
        config,
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
        self.rngs = rngs

        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(config.projector_input_dim // (self.pixel_shuffle_ratio**2))
        self.output_dim = config.projector_output_dim

        self.mlp = Llama4VisionMLP2(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, encoded_patches: chex.Array) -> chex.Array:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


def reshape_for_broadcast(frequencies: jax.Array, query: jax.Array) -> jax.Array:
    ndim = query.ndim
    return jnp.reshape(
        frequencies,
        [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query.shape)],
    )


def vision_apply_rotary_emb(
    query: jax.Array,
    key: jax.Array,
    frequencies: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    query_dtype = query.dtype
    key_dtype = key.dtype
    query_reshaped = query.astype(jnp.float32).reshape((*query.shape[:-1], -1, 2))
    key_reshaped = key.astype(jnp.float32).reshape((*key.shape[:-1], -1, 2))
    query_complex = jax.lax.complex(query_reshaped[..., 0], query_reshaped[..., 1])
    key_complex = jax.lax.complex(key_reshaped[..., 0], key_reshaped[..., 1])
    frequencies_broadcast = reshape_for_broadcast(frequencies, query_complex)
    query_rotated = query_complex * frequencies_broadcast
    key_rotated = key_complex * frequencies_broadcast
    query_out_real_imag = jnp.stack(
        [jnp.real(query_rotated), jnp.imag(query_rotated)],
        axis=-1,
    )
    key_out_real_imag = jnp.stack([jnp.real(key_rotated), jnp.imag(key_rotated)], axis=-1)
    query_out = query_out_real_imag.reshape(query.shape)
    key_out = key_out_real_imag.reshape(key.shape)
    return query_out.astype(query_dtype), key_out.astype(key_dtype)


class Llama4VisionAttention(AttentionModule):
    def __init__(
        self,
        config: Llama4VisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout

        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.q_proj = linear_class(self.embed_dim, self.num_heads * self.head_dim)
        self.k_proj = linear_class(self.embed_dim, self.num_heads * self.head_dim)
        self.v_proj = linear_class(self.embed_dim, self.num_heads * self.head_dim)
        self.o_proj = linear_class(self.num_heads * self.head_dim, self.embed_dim)

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=self.config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        frequencies: chex.Array | None = None,
        output_attentions: bool = False,
    ) -> AttentionLayerOutput:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        query_states = query_states.reshape(*hidden_shape)
        key_states = key_states.reshape(*hidden_shape)
        value_states = value_states.reshape(*hidden_shape)
        query_states, key_states = vision_apply_rotary_emb(
            query_states,
            key_states,
            frequencies=frequencies,
        )
        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=common_types.MODE_TRAIN,
            bias=None,
            cache_metadata=None,
            cache_view=None,
            init_bias=None,
            attention_mask=None,
            segment_ids=None,
            causal=False,
        )
        attn_output = attentions.attention_outputs.reshape(*input_shape, -1)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = self.o_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
        )


class Llama4VisionMLP(nn.Module):
    """MLP module for Llama4 vision transformer.

    Standard feedforward network with GELU activation for vision
    feature transformation within transformer blocks.
    """

    def __init__(
        self,
        config,
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
        self.rngs = rngs

        linear_class = partial(
            ParallelLinear,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.01),
        )

        self.fc1 = linear_class(config.hidden_size, config.intermediate_size)
        self.fc2 = linear_class(config.intermediate_size, config.hidden_size)
        self.activation_fn = ACT2FN["gelu"]

    def __call__(self, hidden_states: chex.Array) -> chex.Array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Llama4VisionEncoderLayer(nn.Module):
    """Single encoder layer for Llama4 vision models.

    Combines self-attention and feedforward networks with layer normalization
    and residual connections for vision feature encoding.
    """

    def __init__(
        self,
        config: Llama4VisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        attn_block = Llama4VisionAttention
        mlp_block = Llama4VisionMLP

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )

        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.input_layernorm = nn.LayerNorm(
            num_features=config.hidden_size,
            epsilon=0.00001,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            num_features=config.hidden_size,
            epsilon=0.00001,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layer_idx = layer_idx

    def __call__(
        self,
        hidden_states: chex.Array,
        output_attentions: bool = False,
        frequencies: chex.Array | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        attn_outputs = self.self_attn(
            hidden_states,
            frequencies,
            output_attentions,
        )
        hidden_states = residual + attn_outputs.attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return EncoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
        )


class Llama4VisionEncoder(nn.Module):
    """Vision encoder stack for Llama4 models.

    Stacks multiple vision encoder layers to progressively encode
    visual features for downstream processing.
    """

    def __init__(
        self,
        config: Llama4VisionConfig,
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

        self.layers = [
            Llama4VisionEncoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: jax.Array,
        frequencies: jax.Array,
        attention_mask: jax.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = (*encoder_states, hidden_states)
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                output_attentions=output_attentions,
                frequencies=frequencies,
            )

            if output_attentions:
                all_attentions = (*all_attentions, layer_outputs.attention_weight)

            hidden_states = layer_outputs.hidden_states

        if output_hidden_states:
            encoder_states = (*encoder_states, hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class Llama4UnfoldConvolution(nn.Module):
    """Unfold convolution module for Llama4 vision models.

    Implements patch extraction with optional convolution,
    converting images into sequences of patch embeddings.
    """

    def __init__(
        self,
        config: Llama4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        patch_size_val = config.patch_size
        if isinstance(patch_size_val, int):
            self.kernel_size: tuple[int, int] = (patch_size_val, patch_size_val)
        else:
            self.kernel_size: tuple[int, int] = patch_size_val

        self.stride = config.patch_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        self.num_channels: int = config.num_channels
        self.hidden_size: int = config.hidden_size

        # Linear layer similar to PyTorch's version
        in_features = self.num_channels * self.kernel_size[0] * self.kernel_size[1]
        self.linear = ParallelLinear(
            in_features=in_features,
            out_features=self.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        batch_size = hidden_states.shape[0]

        hidden_states_nhwc = jnp.transpose(hidden_states, (0, 2, 3, 1))
        patches = jax.lax.conv_general_dilated_patches(
            lhs=hidden_states_nhwc,
            filter_shape=self.kernel_size,
            window_strides=self.stride,
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        num_patches = patches.shape[1] * patches.shape[2]
        patches_reshaped = jnp.reshape(patches, (batch_size, num_patches, -1))
        hidden_states = self.linear(patches_reshaped)

        return hidden_states


@register_module(TaskType.BASE_VISION, config=Llama4VisionConfig, model_type="llama4_vision")
@register_module(TaskType.BASE_MODULE, config=Llama4VisionConfig, model_type="llama4_vision")
class Llama4VisionModel(EasyDeLBaseModule):
    def __init__(
        self,
        config: Llama4VisionConfig,
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
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = Llama4UnfoldConvolution(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.class_embedding = nn.Param(
            self.scale
            * jax.random.normal(
                rngs.params(),
                (self.hidden_size,),
                param_dtype,
            )
        )
        self.positional_embedding_vlm = nn.Param(
            self.scale
            * jax.random.normal(
                rngs.params(),
                (self.num_patches, self.hidden_size),
                param_dtype,
            )
        )
        self.layernorm_pre = nn.LayerNorm(
            num_features=self.hidden_size,
            epsilon=0.00001,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layernorm_post = nn.LayerNorm(
            num_features=self.hidden_size,
            epsilon=0.00001,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # encoders
        self.model = Llama4VisionEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_adapter = Llama4VisionPixelShuffleMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_idx = self.config.image_size // self.config.patch_size

    def __call__(
        self,
        pixel_values: jax.Array,
        attention_mask: jax.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        batch_size_times_num_tiles = pixel_values.shape[0]
        num_concurrent_media = 1
        num_chunks = 1
        hidden_states = self.patch_embedding(pixel_values)
        _, num_patches, hidden_dim = hidden_states.shape

        # Add cls token
        hidden_states = hidden_states.reshape(
            batch_size_times_num_tiles * num_concurrent_media * num_chunks,
            num_patches,
            hidden_dim,
        )
        class_embedding = jnp.broadcast_to(
            self.class_embedding.value,
            (hidden_states.shape[0], 1, hidden_states.shape[-1]),
        )
        hidden_states = jnp.concatenate([hidden_states, class_embedding], axis=1)
        num_patches += 1

        # Position embeddings
        hidden_states = hidden_states.reshape(
            batch_size_times_num_tiles * num_concurrent_media,
            num_chunks,
            num_patches,
            hidden_dim,
        )
        hidden_states = hidden_states + self.positional_embedding_vlm
        hidden_states = self.layernorm_pre(hidden_states)
        hidden_states = hidden_states.reshape(batch_size_times_num_tiles, -1, hidden_dim)
        output = self.model(
            hidden_states,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            frequencies=_vision_freqs(
                self.vision_idx,
                self.config.hidden_size,
                self.config.num_attention_heads,
                self.config.rope_theta,
            ),
        )
        hidden_states = output.last_hidden_state
        hidden_states = self.layernorm_post(hidden_states)
        hidden_states = hidden_states[:, :-1, :]
        hidden_states = self.vision_adapter(hidden_states)
        all_hidden_states = output.hidden_states if output_hidden_states else None

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=output.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        This vision model acts as the encoder.
        """
        return self

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model and does not have a decoder.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This vision model does not have a language model head.
        """
        raise NotImplementedError("This vision model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.patch_embedding


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Llama4Config, model_type="llama4")
class Llama4ForConditionalGeneration(EasyDeLBaseModule):
    """
    Llama4Vision model for conditional text generation based on image inputs.
    Combines a vision tower and a language model with a multi-modal projector.

    Attributes:
        config (Llama4VisionConfig): Configuration object.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): JAX precision level.
        rngs (nn.Rngs): Random number generators.
    """

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Llama4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Llama4VisionForConditionalGeneration model."""
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_model = Llama4VisionModel(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.multi_modal_projector = Llama4MultiModalProjector(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = Llama4ForCausalLM(
            config=config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_image_features(self, pixel_values: chex.Array, **kwargs) -> chex.Array:
        """Extracts and projects image features from the vision tower.

        Args:
            pixel_values (chex.Array): Input pixel values for the images.

        Returns:
            chex.Array: Processed image features ready for the language model.
        """
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_model(
            pixel_values,
            output_hidden_states=False,
            **kwargs,
        )
        hidden_states = image_outputs.last_hidden_state
        return hidden_states

    def __call__(
        self,
        input_ids: chex.Array = None,
        pixel_values: chex.Array = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        inputs_embeds: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids
            llm_input_ids = jnp.where(special_image_mask, 0, llm_input_ids)
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.language_model.model.embed_tokens(llm_input_ids)

        if pixel_values is not None:
            orgshape = inputs_embeds.shape

            image_features = self.get_image_features(pixel_values)
            vision_flat = image_features.reshape(-1, image_features.shape[-1])
            projected_vision_flat = self.multi_modal_projector(vision_flat)
            final_mask = jnp.expand_dims(input_ids == self.config.image_token_id, axis=-1)
            inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            final_mask_1d = final_mask[..., 0].reshape(-1)
            num_projected_tokens = projected_vision_flat.shape[0]
            image_token_indices = jnp.where(
                final_mask_1d,
                size=num_projected_tokens,
                fill_value=-1,
            )[0]
            inputs_embeds_updated_flat = inputs_embeds_flat.at[image_token_indices].set(projected_vision_flat)
            inputs_embeds = inputs_embeds_updated_flat.reshape(orgshape)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
            **lm_kwargs,
        )

        return Llama4CausalLMOutputWithPast(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size,
        max_length,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ):
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def _get_compile_model_kwargs(
        self,
        batch_size: int,
        input_tokens_length: int,
        input_sharding: jax.sharding.PartitionSpec,
        rngs: jax.random.PRNGKey,
        vision_included: bool = False,
        vision_batch_size: int = 1,
        vision_channels: int = 3,
        vision_height: int | None = None,
        vision_width: int | None = None,
        required_props: tp.Mapping[str, dict[str, tp.Any]] | None = None,
        **kwargs,
    ):
        """Helper function to get keyword arguments for model compilation, potentially including vision inputs.

        Args:
            batch_size (int): Batch size for text inputs.
            input_tokens_length (int): Sequence length for text inputs.
            input_sharding (jax.sharding.PartitionSpec): Sharding specification for text inputs.
            rngs (jax.random.PRNGKey): Random number generator key.
            vision_included (bool): Whether to include dummy vision inputs. Defaults to False.
            vision_batch_size (int): Batch size for vision inputs. Defaults to 1.
            vision_channels (int): Number of channels for vision inputs. Defaults to 3.
            vision_height (Optional[int]): Height for vision inputs (defaults to config).
            vision_width (Optional[int]): Width for vision inputs (defaults to config).
            required_props (Optional[Mapping[str, Dict[str, Any]]]): Required properties.
            **kwargs: Additional arguments passed to the language model's compile kwargs method.

        Returns:
            dict: Keyword arguments for model compilation.
        """
        basics = self.language_model._get_compile_model_kwargs(
            batch_size=batch_size,
            input_tokens_length=input_tokens_length,
            input_sharding=input_sharding,
            rngs=rngs,
            vision_included=vision_included,
            vision_batch_size=vision_batch_size,
            vision_channels=vision_channels,
            vision_height=vision_height,
            vision_width=vision_width,
            required_props=required_props,
            **kwargs,
        )

        if vision_included:
            pixel_values = jnp.ones(
                (
                    vision_batch_size or 1,
                    vision_channels or 3,
                    self.config.vision_config.image_size,
                    self.config.vision_config.image_size,
                ),
                dtype="f4",
            )
            basics.update({"pixel_values": pixel_values})
        return basics

    def prepare_inputs_for_generation(
        self,
        input_ids: chex.Array,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
    ):
        """Prepares inputs for text generation, including pixel values if provided.

        Args:
            input_ids (chex.Array): Initial input token IDs.
            max_length (int): Maximum generation length.
            pixel_values (Optional[chex.Array]): Pixel values for image input.
            attention_mask (Optional[chex.Array]): Attention mask.

        Returns:
            dict: Model inputs ready for generation.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        model_inputs["pixel_values"] = pixel_values
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Updates model inputs for the next step of generation, removing pixel values after the first step.

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs: Current keyword arguments for the model.

        Returns:
            dict: Updated model keyword arguments.
        """
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)  # only effect first iter
        return model_kwargs

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        The vision tower acts as the encoder in this multi-modal setup.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.language_model.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.language_model.get_lm_head()

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.language_model.get_embedding()
