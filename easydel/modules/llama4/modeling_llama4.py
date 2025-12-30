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

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    DecoderLayerOutput,
    EncoderLayerOutput,
    ModelOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule, BaseSequenceClassificationModule, BaseVisionLanguageModule
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
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.norms import RMSNorm as Llama4TextRMSNorm
from easydel.utils.compiling_utils import ejit

from .llama4_configuration import Llama4Config, Llama4TextConfig, Llama4VisionConfig


@auto_pytree
class Llama4CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llama4Vision causal language model (or autoregressive) outputs.

    Args:
        loss (`Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(Array))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`):
            Tuple of `tuple(Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(Array)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(Array)`, *optional*, returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`):
            Tuple of `Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`Array`, *optional*):
            An `Array` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Array | None = None
    logits: Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


@ejit(static_argnums=(0, 1, 2, 3))
def _vision_freqs(idx, hidden_size, num_attention_heads, rope_theta):
    """Compute rotary frequencies for the vision transformer grid."""
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
    """Create a chunked causal attention mask for sliding window attention."""
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

    reform_param: tp.ClassVar = {
        # HuggingFace has fused gate_up_proj, we split into separate gate_proj and up_proj
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.kernel", "spliter": lambda x: x[:, :, : x.shape[-1] // 2]},
                {"name": "up_proj.kernel", "spliter": lambda x: x[:, :, x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda g, u: jnp.concatenate([g, u], axis=-1),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.kernel", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

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

        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=False,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=False,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(config.initializer_range),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Forward pass through MoE experts."""
        gate = self.gate_proj(hidden_states, group_sizes, sorted_experts)
        up = self.up_proj(hidden_states, group_sizes, sorted_experts)
        return self.down_proj(self.act_fn(gate) * up, group_sizes, sorted_experts)


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
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(config.hidden_size, intermediate_size)
        self.down_proj = row_parallel_linear(intermediate_size, config.hidden_size)
        self.up_proj = column_parallel_linear(config.hidden_size, intermediate_size)
        self.activation_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
        gate = checkpoint_name(self.activation_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        return checkpoint_name(hidden_states, "mlp_output")


class Llama4TextMoe(BaseMoeModule):
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
        super().__init__(
            config=config,
            n_routed_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )

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
        self.router = ColumnParallelLinear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=False,
            precision=precision,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.shared_expert = Llama4TextMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Configure hooks for Llama4 sigmoid-based INPUT SCALING routing
        # HF Llama4 uses INPUT scaling: expert(input * weight)
        # EasyDeL sparse MoE uses OUTPUT scaling by default: weight * expert(input)
        # To match HF, we use scale_replicated_inputs to scale inputs BEFORE expert processing,
        # then set output weights to 1.0 to avoid double-scaling.

        def _sigmoid_topk_weights(logits: jax.Array) -> jax.Array:
            """Apply sigmoid to logits, zero out non-top-k."""
            _, top_idx = jax.lax.top_k(logits, k=self.num_experts_per_tok)
            sigmoid_weights = jax.nn.sigmoid(logits.astype(jnp.float32))
            mask = jnp.zeros_like(logits, dtype=jnp.bool_)
            row_idx = jnp.arange(logits.shape[0])[:, None]
            mask = mask.at[row_idx, top_idx].set(True)
            return jnp.where(mask, sigmoid_weights, 0.0).astype(logits.dtype)

        def _scale_inputs(inputs: jax.Array, weights: jax.Array) -> jax.Array:
            """Scale replicated inputs by their corresponding weights (input scaling).

            Args:
                inputs: Replicated token representations, shape (tokens*k, hidden)
                weights: Flattened weights, shape (tokens*k,)

            Returns:
                Scaled inputs, shape (tokens*k, hidden)
            """
            return inputs * weights[:, None]

        def _passthrough_weights(weights: jax.Array) -> jax.Array:
            """Pass through weights unchanged (avoid default sum normalization).

            Called by refine_weights_hook after top-k selection. For Llama4,
            we use sigmoid weights directly without sum normalization.
            """
            return weights

        def _unity_output_weights(weights: jax.Array) -> jax.Array:
            """Replace output weights with 1.0 since scaling is done on inputs.

            This is called during unpermute to avoid double-scaling.
            The weights have shape (tokens, k) where k = num_experts_per_tok.
            """
            return jnp.ones_like(weights)

        # Configure for Llama4's input scaling: expert(input * weight)
        # - normalize_gate_logits: Apply sigmoid with top-k masking to get weights
        # - refine_weights_hook: Pass through weights (avoid default sum normalization)
        # - scale_replicated_inputs: Scale inputs by weights before expert processing
        # - output_weights_hook: Set output combination weights to 1.0 to avoid double-scaling
        self.moe_hooks = self.moe_hooks.replace(
            normalize_gate_logits=_sigmoid_topk_weights,
            refine_weights_hook=_passthrough_weights,
            scale_replicated_inputs=_scale_inputs,
            output_weights_hook=_unity_output_weights,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        training: bool = False,
        layer_idx: int | None = None,
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        del training

        # Shared expert output
        shared_out = self.shared_expert(hidden_states)

        # MoE expert output
        def ffn_activation(gate, up):
            return self.experts.act_fn(gate) * up

        expert_out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.router,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
            ffn_activation=ffn_activation,
            layer_idx=layer_idx,
        )

        final_output = checkpoint_name(shared_out + expert_out, "moe_expert_output")
        return final_output, checkpoint_name(router_logits, "moe_router_logits")


class Llama4TextAttention(UnifiedAttention):
    """Attention module for the Llama4 text decoder with optional sliding windows."""

    def __init__(
        self,
        config: Llama4TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.use_rope = (layer_idx + 1) % 4 != 0
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=False,
        )
        self.qk_norm = Llama4TextL2Norm() if config.use_qk_norm and self.use_rope else None
        self._cached_position_ids: Int[Array, "batch seq_len"] | None = None

    def _create_attention_performer(self, config: Llama4TextConfig, rngs: nn.Rngs):
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
        )

    def _create_rotary(self, config: Llama4TextConfig, dtype: jnp.dtype):
        # RoPE is handled via custom complex rotary frequencies when enabled.
        return None if not self.use_rope else super()._create_rotary(config, dtype)

    def _apply_rotary(
        self,
        query_states: Float[Array, "batch seq_len num_heads head_dim"],
        key_states: Float[Array, "batch seq_len num_kv_heads head_dim"],
        position_ids: Int[Array, "batch seq_len"],
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch seq_len num_kv_heads head_dim"]]:
        if not self.use_rope:
            return query_states, key_states
        if frequencies is not None:
            return self.apply_complex_rotary(query_states, key_states, frequencies)
        return super()._apply_rotary(query_states, key_states, position_ids, frequencies)

    def _postprocess_qkv(
        self,
        query_states: Float[Array, "batch seq_len num_heads head_dim"],
        key_states: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value_states: Float[Array, "batch seq_len num_kv_heads head_dim"],
    ) -> tuple[
        Float[Array, "batch seq_len num_heads head_dim"],
        Float[Array, "batch seq_len num_kv_heads head_dim"],
        Float[Array, "batch seq_len num_kv_heads head_dim"],
    ]:
        if self.qk_norm is not None:
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)
        if self.attn_temperature_tuning and not self.use_rope and self._cached_position_ids is not None:
            attn_scales = (
                jnp.log(jnp.floor((self._cached_position_ids.astype("f4") + 1.0) / self.floor_scale) + 1.0)
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales.reshape((*attn_scales.shape, 1, 1))
            query_states = (query_states * attn_scales).astype(query_states.dtype)
        return query_states, key_states, value_states


class Llama4TextDecoderLayer(nn.Module):
    """Single Llama4 text decoder block combining attention and MLP."""

    def __init__(
        self,
        config: Llama4TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
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
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )

        hidden_states = hidden_states + attn_outputs.attention_output

        feed_forward_input = self.post_attention_layernorm(hidden_states)
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
    """Decoder-only Llama4 text model built from embeddings and decoder blocks."""

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

        embed_block = auto_remat(
            nn.Embed,
            policy=self.config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
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
        """Forward pass through the Llama model.

        Args:
          input_ids (Array, optional): Input token IDs, shape (batch_size, sequence_length).
          inputs_embeds (Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
          attention_mask (Array, optional): Mask to avoid attention on padding tokens.
          position_ids (Array, optional): Indices of positions of each input sequence token.
          segment_ids (Array, optional): Segment token indices for segment embeddings.
          past_key_values (TransformerCache | RaggedPagesCache, optional):
            Cache containing precomputed key/value states.
          cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Metadata for cache handling.
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
        mask_info = mask_info.apply_chunked(self.config.attention_chunk_size)
        frequencies = self.compute_complex_rotary(position_ids)

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
class Llama4ForCausalLM(BaseCausalLMModule[Llama4TextModel, Llama4TextConfig]):
    """Llama4 model with a Causal Language Modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "llama4_text"
    _config_class = Llama4TextConfig

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
            base_model_class=Llama4TextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Llama4TextConfig, model_type="llama4_text")
class Llama4ForSequenceClassification(BaseSequenceClassificationModule[Llama4TextModel, Llama4TextConfig]):
    """Llama4 model for sequence classification tasks."""

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "llama4_text"
    _config_class = Llama4TextConfig

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
            base_model_class=Llama4TextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            classifier_name="score",
            classifier_bias=False,
        )


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
        self.linear_1 = RowParallelLinear(
            config.vision_config.vision_output_dim,
            config.get_text_config().hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.01),
        )

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        return self.linear_1(hidden_states)


def pixel_shuffle(input_tensor, shuffle_ratio):
    """Rearrange flattened vision tokens to a denser spatial grid."""
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

    def __call__(self, encoded_patches: Array) -> Array:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


def reshape_for_broadcast(frequencies: jax.Array, query: jax.Array) -> jax.Array:
    """Reshape rotary frequencies so they broadcast over the complex query tensor."""
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
    """Apply rotary position embeddings to complex-valued vision queries and keys."""
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
    """Attention module for the Llama4 vision transformer."""

    def __init__(
        self,
        config: Llama4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
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
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
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
            requires_cache=False,  # Vision encoder doesn't need KV cache
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
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
            mask_info=None,
            causal=False,
        )
        attn_output = attentions.attention_outputs.reshape(*input_shape, -1)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = self.o_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
        )


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
            ColumnParallelLinear,
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

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return self.activation_fn(hidden_states)


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
            ColumnParallelLinear,
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

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
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
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
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
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
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
        self.linear = ColumnParallelLinear(
            in_features=in_features,
            out_features=self.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jax.Array:
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
    """Vision transformer for Llama4 including patchify stem, transformer blocks, and final norm."""

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
        self.class_embedding = ArrayParam.bound(
            shape=(self.hidden_size,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": self.scale},
            key=rngs.params(),
        )
        self.positional_embedding_vlm = ArrayParam.bound(
            shape=(self.num_patches, self.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": self.scale},
            key=rngs.params(),
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
class Llama4ForConditionalGeneration(BaseVisionLanguageModule[Llama4ForCausalLM, Llama4Config]):
    """Llama4 Vision model for conditional text generation based on image inputs.

    Combines a vision tower and a language model with a multi-modal projector.

    Note: Llama4 has a unique architecture where the language_model is already
    a complete Llama4ForCausalLM (with its own lm_head), unlike other VLMs where
    the base model doesn't include the lm_head.

    Attributes:
        config (Llama4Config): Configuration object.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): JAX precision level.
        rngs (nn.Rngs): Random number generators.

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "llama4" model identifier
        _supports_video: True (Llama4 supports video input)
        _uses_mrope: False (uses standard RoPE)
    """

    # Class attributes for VLM capabilities
    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "llama4"
    _config_class = Llama4Config
    _auto_register = False  # Already registered via decorator
    _supports_video = True
    _uses_mrope = False

    # Component name mapping
    _vision_tower_name = "vision_model"
    _projector_name = "multi_modal_projector"
    _language_model_name = "language_model"

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
        """Initializes the Llama4ForConditionalGeneration model."""
        language_model = Llama4ForCausalLM(
            config=config.get_text_config(),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        super().__init__(
            config=config,
            base_model=language_model,
            base_model_name="language_model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            vision_feature_layer=getattr(config, "vision_feature_layer", -1),
            vision_feature_select_strategy=getattr(config, "vision_feature_select_strategy", "default"),
            image_token_index=getattr(config, "image_token_id", None),
            video_token_index=getattr(config, "video_token_id", None),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            create_lm_head=False,
            lm_head_bias=False,
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
        self.vocab_size = config.get_text_config().vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_image_features(self, pixel_values: Array, **kwargs) -> Array:
        """Extracts and projects image features from the vision tower.

        Args:
            pixel_values (Array): Input pixel values for the images.

        Returns:
            Array: Processed image features ready for the language model.
        """
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_model(
            pixel_values,
            output_hidden_states=False,
            **kwargs,
        )
        hidden_states = image_outputs.last_hidden_state
        return hidden_states

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        image_features: Array | None = None,
        pixel_values: Array | None = None,
        **kwargs,
    ) -> Array:
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        image_token_id = self.config.image_token_id
        if image_token_id >= self.vocab_size:
            llm_input_ids = jnp.where(input_ids == image_token_id, 0, input_ids)
        else:
            llm_input_ids = input_ids

        inputs_embeds = super().compute_embedding(llm_input_ids)

        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values, **kwargs)

        if image_features is not None:
            orgshape = inputs_embeds.shape
            vision_flat = image_features.reshape(-1, image_features.shape[-1])
            projected_vision_flat = self.multi_modal_projector(vision_flat).astype(inputs_embeds.dtype)

            image_token_mask_1d = (input_ids == image_token_id).reshape(-1)
            inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])

            num_projected_tokens = projected_vision_flat.shape[0]
            image_token_indices = jnp.where(
                image_token_mask_1d,
                size=num_projected_tokens,
                fill_value=-1,
            )[0]
            inputs_embeds = inputs_embeds_flat.at[image_token_indices].set(projected_vision_flat).reshape(orgshape)

        return inputs_embeds

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        pixel_values: Array = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
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
        if pixel_values is not None and input_ids is None:
            raise ValueError("`input_ids` must be provided when `pixel_values` is not None.")

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
            )
        outputs = self.language_model(
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            **lm_kwargs,
        )

        return VLMCausalLMOutput(
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

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
    ):
        """Prepares inputs for text generation, including pixel values if provided.

        Args:
            input_ids (Array): Initial input token IDs.
            max_length (int): Maximum generation length.
            pixel_values (Optional[Array]): Pixel values for image input.
            attention_mask (Optional[Array]): Attention mask.

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
        """Returns the encoder part of the model (vision tower)."""
        return self.vision_model

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self.language_model.get_decoder()

    def get_lm_head(self):
        """Returns the language model head."""
        return self.language_model.get_lm_head()

    def get_embedding(self):
        """Returns the embedding layer."""
        return self.language_model.get_embedding()

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision tower component."""
        return self.vision_model

    def get_projector(self) -> nn.Module:
        """Returns the multimodal projector component."""
        return self.multi_modal_projector

    def get_language_model(self) -> nn.Module:
        """Returns the language model component."""
        return self.language_model
