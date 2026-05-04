# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Spectrax implementation of OpenAI's GPT-OSS sparse Mixture-of-Experts decoder.

GPT-OSS is a large-scale sparse MoE language model that scales parameter
count via sparsely routed experts while keeping per-token compute fixed,
combined with alternating sliding-window and full attention layers and
YARN-scaled RoPE for very long context lengths.

Architectural traits:
    - Sparse MoE FFN with ``num_local_experts`` experts and top-k routing
      (``num_experts_per_tok``); strong auxiliary load-balancing coefficient.
    - Grouped-query attention with sliding-window attention on alternating
      layers and full attention on the rest.
    - YARN-scaled rotary positional embeddings for extended context.
    - RMSNorm pre-normalization throughout.

Exports:
    - :class:`GptOssModel`: Backbone returning hidden states.
    - :class:`GptOssForCausalLM`: Decoder LM with optional tied LM head.
    - :class:`GptOssForSequenceClassification`: Pooled classifier head.
"""

import typing

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RMSNorm,
    RowParallelMoELinear,
)
from easydel.layers.attention import UnifiedAttention
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .gpt_oss_configuration import GptOssConfig


class GptOssRMSNorm(RMSNorm):
    """GPT-OSS RMS Normalization layer.

    This is a simple extension of RMSNorm for GPT-OSS models. RMS (Root Mean Square)
    Layer Normalization normalizes the input tensor by its RMS value, providing
    stable training dynamics without centering the activations.

    Attributes:
        dim (int): Dimensionality of the input features.
        eps (float): Small epsilon value for numerical stability.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
    """

    ...


class GptOssExperts(spx.Module):
    """Stacked expert FFNs for GPT-OSS using a clipped Swish-GLU with bias-shifted up.

    Holds the parameters for all ``num_local_experts`` experts as 3-D
    tensors and lets the fused MoE dispatch invoke them with
    ``grouped_matmul`` over expert groups. Each individual expert
    implements a *non-standard* gated MLP, deliberately tuned for
    stability under aggressive bf16 / int8 quantisation:

    1. ``w0 = gate_proj(x)`` → clipped to ``(-inf, 7.0]``.
    2. ``w1 = up_proj(x)`` → clipped to ``[-7.0, 7.0]``.
    3. ``glu = w0 * sigmoid(alpha * w0)`` with ``alpha=1.702`` (the
       Swish-style approximation to GeLU used by xAI / OpenAI).
    4. ``intermediate = (w1 + 1.0) * glu`` — note the **+1 shift on the up
       branch**: this keeps the multiplicative path well-defined even when
       ``up_proj`` is exactly zero, and is what lets the experts initialise
       to "approximately identity" before training.
    5. ``out = down_proj(intermediate)``.

    All three projections (``gate_proj``, ``up_proj``, ``down_proj``)
    carry **bias parameters** (unlike the typical biasless gated-MLP) and
    are stored as MoE-parallel linears: column-sharded for gate/up,
    row-sharded for down. The ``reform_param`` class variable splits the
    HF-fused ``gate_up_proj`` weight (interleaved gate/up channels via
    ``[..., 0::2]`` / ``[..., 1::2]``) into the separate ``gate_proj`` /
    ``up_proj`` tensors expected by EasyDeL.

    Attributes:
        config: Source ``GptOssConfig``.
        dtype: Activation dtype.
        param_dtype: Parameter dtype.
        precision: ``jax.lax.PrecisionLike`` for the three matmuls.
        intermediate_size: Per-expert intermediate dimension.
        num_experts: Number of routed experts (``config.num_local_experts``).
        hidden_size: Decoder hidden dimension.
        expert_dim: Alias for ``intermediate_size``.
        gate_proj: ``ColumnParallelMoELinear`` storing all experts'
            ``hidden -> intermediate`` gate weights.
        up_proj: ``ColumnParallelMoELinear`` storing the up weights.
        down_proj: ``RowParallelMoELinear`` storing
            ``intermediate -> hidden`` down weights.
        alpha: Sigmoid scale fixed at ``1.702`` (the Swish-GLU /
            ``gelu_approx`` constant).
        act_fn: Configured activation (kept for API parity; the actual
            GLU math is hand-rolled in :meth:`forward`).
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.weight", "spliter": lambda x: x[..., 0::2]},
                {"name": "up_proj.weight", "spliter": lambda x: x[..., 1::2]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "gate_up_proj_bias$": {
            "splits": [
                {"name": "gate_proj.bias", "spliter": lambda x: x[..., 0::2]},
                {"name": "up_proj.bias", "spliter": lambda x: x[..., 1::2]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.weight", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
        "down_proj_bias$": {
            "splits": [
                {"name": "down_proj.bias", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GptOssExperts module.

        Args:
            config (GptOssConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=True,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.alpha = 1.702
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Forward pass through the expert MLP network.

        Applies the expert feed-forward transformation with a modified GLU activation.
        Values are clipped for numerical stability before the activation function.

        Args:
            hidden_states (Array): Input hidden states of shape (batch, seq_len, hidden_dim).
            group_sizes (Array): Sizes of each expert group for routing.
            sorted_experts (Array, optional): Sorted expert indices for efficient routing.
                Defaults to None.

        Returns:
            Array: Output hidden states after expert processing with shape
                (batch, seq_len, hidden_dim).
        """
        w0 = self.gate_proj(hidden_states, group_sizes, sorted_experts)
        w1 = self.up_proj(hidden_states, group_sizes, sorted_experts)

        w0 = jnp.clip(w0, min=None, max=7.0)
        w1 = jnp.clip(w1, min=-7.0, max=7.0)

        glu = w0 * jax.nn.sigmoid(w0 * self.alpha)
        intermediate = (w1 + 1.0) * glu

        return self.down_proj(intermediate, group_sizes, sorted_experts)


class GptOssMLP(BaseMoeModule):
    """Sparse MoE block: 128 experts, top-4 routing, scattered-softmax weights.

    Implements GPT-OSS's flagship sparse FFN. Routing protocol per token:

    1. **Router projection** (``self.router``, a ColumnParallelLinear with
       bias) emits per-expert logits of shape ``(num_local_experts,)``.
    2. **Top-k selection** picks the ``num_experts_per_tok`` (=4)
       highest-scoring experts via :func:`jax.lax.top_k`.
    3. **Softmax-on-selected** (the ``after_gate=_scatter_topk_probs``
       hook) computes a softmax *over the four selected scores only* and
       scatters the resulting probabilities back into a zero-initialised
       full-width vector, so the unrouted experts contribute exactly zero
       — equivalent to the standard top-k softmax but written so the
       fused-MoE dispatch never has to look at unselected experts.
    4. **Refinement** (``refine_weights_hook=_softmax_topk_weights``) is a
       no-op renormalisation safety net used by the parent
       :class:`BaseMoeModule` to handle hooks uniformly.
    5. **Expert dispatch** is handled by :meth:`moe_call` in
       :class:`BaseMoeModule`: tokens are sorted by their assigned expert,
       each contiguous run is processed by :class:`GptOssExperts` via
       ``grouped_matmul``, and the outputs are scattered back into token
       order and convex-combined by the top-k weights.

    The activation inside each expert is a *clipped GLU with bias-shifted
    up branch* — see :class:`GptOssExperts` for the math. There is no
    shared expert; every token's contribution is purely the weighted sum
    of its four routed experts.
    """

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GptOssMLP module.

        Sets up the router network and expert modules, along with custom routing
        hooks for top-k probability computation and weight normalization.

        Args:
            config (GptOssConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generators.
        """
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

        self.router = ColumnParallelLinear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.experts = GptOssExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        def _scatter_topk_probs(logits: jax.Array) -> jax.Array:
            top_vals, top_idx = jax.lax.top_k(logits, k=self.num_experts_per_tok)
            top_probs = jax.nn.softmax(top_vals, axis=-1)
            out = jnp.zeros_like(logits)
            row_idx = jnp.arange(logits.shape[0])[:, None]
            return out.at[row_idx, top_idx].set(top_probs)

        def _softmax_topk_weights(weights: jax.Array) -> jax.Array:
            return jax.nn.softmax(weights, axis=-1)

        self.moe_hooks = self.moe_hooks.replace(
            after_gate=_scatter_topk_probs,
            refine_weights_hook=_softmax_topk_weights,
        )

    def forward(self, hidden_states, training=False, layer_idx=None):
        """Forward pass through the MoE MLP.

        Routes input hidden states through the expert network based on router
        decisions. Each token is processed by the top-k experts with weighted
        contributions.

        Args:
            hidden_states (Array): Input hidden states of shape
                (batch, seq_len, hidden_dim).
            training (bool): Whether in training mode. Defaults to False.
                Currently unused.
            layer_idx: Layer index for debugging/logging. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - output (Array): Expert-processed hidden states of shape
                    (batch, seq_len, hidden_dim).
                - router_logits (Array): Router logits for auxiliary loss
                    computation.
        """
        del training

        def ffn_activation(w0, w1):
            w0 = jnp.clip(w0, min=None, max=7.0)
            w1 = jnp.clip(w1, min=-7.0, max=7.0)
            glu = w0 * jax.nn.sigmoid(w0 * self.experts.alpha)
            return (w1 + 1.0) * glu

        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.router,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.weight.value,
            wu_kernel=self.experts.up_proj.weight.value,
            wd_kernel=self.experts.down_proj.weight.value,
            wi_bias=self.experts.gate_proj.bias.value,
            wu_bias=self.experts.up_proj.bias.value,
            wd_bias=self.experts.down_proj.bias.value,
            act_fn=self.experts.act_fn,
            ffn_activation=ffn_activation,
            layer_idx=layer_idx,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class GptOssAttention(UnifiedAttention):
    """GQA attention with per-layer sliding window and learnable softmax-sink logits.

    Builds on :class:`UnifiedAttention` with two GPT-OSS-specific extensions:

    **Per-layer sliding/full alternation.** Each layer reads its mode from
    ``config.layer_types[layer_idx]``. When set to ``"sliding_attention"``,
    the attention kernel is given ``sliding_window=config.sliding_window``
    and tokens beyond the window in the past are masked out in addition to
    the causal mask. Other layers run full causal attention. The published
    GPT-OSS checkpoints alternate the two so most layers benefit from the
    sliding-window's long-context speedup while a minority retain global
    receptive field for cross-document mixing.

    **Per-head attention sinks** (``self.sinks``). One *learnable scalar
    logit per attention head* is appended into the softmax denominator at
    every attention step:

        ``softmax([Q·K^T_full ; sink_h])`` along the key axis,

    where ``sink_h`` is the head-specific sink logit broadcast across all
    queries. The softmax probability mass that lands on the sink slot is
    *discarded* (no value is gathered for it), which gives the model an
    explicit "attend to nothing" outlet. This is the same mechanism used in
    the DeepSeek family and in StreamingLLM-style papers, and it
    significantly improves stability for long-context inference where one
    or two outlier-dominated tokens would otherwise hijack the softmax
    distribution. The sink logits are initialised from
    ``Normal(0, initializer_range)``.

    Attributes:
        config: Source ``GptOssConfig``.
        dtype: Activation dtype.
        param_dtype: Parameter dtype.
        precision: ``jax.lax.PrecisionLike`` for matmuls.
        layer_idx: Layer index used to look up the per-layer attention type.
        sinks: Bound :class:`ArrayParam` of shape ``(num_attention_heads,)``
            holding one learnable softmax-sink logit per head.
    """

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize GPT-OSS attention with sink tokens.

        Args:
            config (GptOssConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generators.
            layer_idx (int): Index of this layer in the model. Used to determine
                whether this layer uses sliding window attention.
        """
        is_sliding = config.layer_types is not None and config.layer_types[layer_idx] == "sliding_attention"
        sliding_window = None

        if is_sliding:
            sliding_window = config.sliding_window

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=sliding_window,
        )

        # Sink tokens for improved attention
        self.sinks = ArrayParam.bound(
            shape=(config.num_attention_heads,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )


class GptOssDecoderLayer(spx.Module):
    """Pre-norm decoder block: sliding-or-full attention with sinks → 128-expert MoE.

    Single transformer block of GPT-OSS, in standard pre-norm shape:
    ``x + attn(input_norm(x))`` followed by ``x + mlp(post_attn_norm(x))``.
    Two pieces are model-specific:

    - The attention sub-block is :class:`GptOssAttention`, which (1) reads
      ``config.layer_types[layer_idx]`` to pick between sliding-window and
      full causal attention, and (2) maintains per-head learnable
      *attention sinks* — extra logits that absorb softmax mass without
      contributing values, stabilising long-context decoding.
    - The MLP is :class:`GptOssMLP`, a sparse Mixture-of-Experts block
      with 128 experts and top-4 routing per token in the released
      checkpoint. The router emits ``router_logits`` for trainer-side
      auxiliary losses.

    The ``attention_type`` attribute caches the layer's regime
    (``"sliding_attention"`` or ``"standard"``) for downstream consumers
    that need to distinguish the two without re-reading
    ``config.layer_types``.
    """

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the GPT-OSS decoder layer.

        Args:
            config (GptOssConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generators.
            layer_idx (int): Index of this layer in the model.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.self_attn = GptOssAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.mlp = GptOssMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = GptOssRMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = GptOssRMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention_type = config.layer_types[layer_idx] if config.layer_types is not None else "standard"

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """Forward pass of the GPT-OSS decoder layer.

        Applies self-attention with pre-normalization, followed by MoE MLP
        with pre-normalization. Both sub-layers use residual connections.

        Args:
            hidden_states (Array): Input hidden states of shape
                (batch, seq_len, hidden_dim).
            mask_info (MaskInfo): Mask information for attention computation.
            position_ids (Array): Position indices of shape (batch, seq_len).
            mode: Runtime mode (train, decode, infer).
            cache_view (TransformerCacheView | RaggedPagesCacheView, optional):
                Cache view for key/value states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata,
                optional): Metadata for cache handling. Defaults to None.
            output_attentions (bool): Whether to return attention weights.
                Defaults to False.
            output_router_logits (bool): Whether to return router logits.
                Defaults to False.
            frequencies (Array, optional): Rotary embedding frequencies.
                Defaults to None.

        Returns:
            DecoderLayerOutput: Output containing:
                - hidden_states: Processed hidden states
                - attention_weight: Attention weights if output_attentions=True
                - cache_view: Updated cache view
                - router_logits: Router logits if output_router_logits=True
        """
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

        feed_forward_hidden_states, router_logits = self.mlp(feed_forward_input)
        hidden_states = hidden_states + feed_forward_hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
            router_logits=router_logits,
        )


@register_module(TaskType.BASE_MODULE, config=GptOssConfig, model_type="gpt_oss")
class GptOssModel(EasyDeLBaseModule):
    """The base GptOss model transformer.

    This class represents the core transformer architecture of the GptOss model,
    consisting of an embedding layer, multiple GptOssDecoderLayer layers (with sparse MoE),
    and a final layer normalization.

    Attributes:
        config (GptOssConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (spx.Rngs): Random number generators.
        embed_tokens (Embed): Embedding layer for input tokens.
        layers (tp.List[GptOssDecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initializes the GptOssModel.

        Args:
            config (GptOssConfig): The configuration object for the GptOss model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (spx.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        remat_layer_block = auto_remat(
            GptOssDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = GptOssRMSNorm(
                dim=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        """Performs forward pass through the GPT-OSS transformer model.

        Processes input tokens through token embeddings and multiple decoder layers
        with Mixture-of-Experts MLPs. Supports both global and sliding window
        attention patterns based on layer configuration. Uses RMSNorm for
        pre-normalization and rotary position embeddings.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
                Either this or `inputs_embeds` must be provided but not both.
            inputs_embeds: Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Use instead of
                `input_ids` for custom embeddings.
            attention_mask: Boolean mask of shape (batch_size, sequence_length)
                indicating which tokens to attend to (True) and which to ignore
                (False).
            mask_info: Pre-computed mask information. If provided, overrides
                `attention_mask`.
            position_ids: Explicit position indices of shape
                (batch_size, sequence_length). Auto-generated if not provided.
            output_attentions: Whether to return attention weights from all layers.
                Defaults to config.output_attentions.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to config.output_hidden_states.
            output_router_logits: Whether to return router logits from MoE layers.
                Defaults to config.output_router_logits.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER).
                Auto-detected if None.
            past_key_values: Cached key/value states for efficient autoregressive
                generation.
            cache_metadata: Metadata for paged attention mechanisms.

        Returns:
            MoeModelOutput containing:
                - last_hidden_state: Final layer output of shape
                    (batch, seq_len, hidden_size)
                - past_key_values: Updated cache for next generation step
                - hidden_states: Tuple of all layer outputs if
                    output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
                - router_logits: Tuple of router logits if output_router_logits=True

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are provided,
                or if neither is provided.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]

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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        def _layer_loop(block, carry):
            hidden_states, all_hidden_states, all_self_attns, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=self.frequencies,
                )

            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_self_attns += (layer_outputs.attention_weight,)

            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)

            return hidden_states, all_hidden_states, all_self_attns, all_router_logits, idx + 1

        hidden_states, all_hidden_states, all_self_attns, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_self_attns, all_router_logits, 0),
            trace=True,
        )
        hidden_states = self.norm(hidden_states)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
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


@register_module(TaskType.CAUSAL_LM, config=GptOssConfig, model_type="gpt_oss")
class GptOssForCausalLM(BaseCausalLMModule[GptOssModel, GptOssConfig]):  # type: ignore
    """GPT-OSS model with a Causal Language Modeling head.

    This model extends the base GPT-OSS transformer by adding a linear layer on top
    to predict the next token in a sequence, making it suitable for causal language
    modeling tasks. It leverages Mixture-of-Experts (MoE) routing for increased model
    capacity with sparse computation.

    The model supports auxiliary load balancing loss to encourage balanced routing
    across experts, which is crucial for efficient MoE training.

    Attributes:
        config (GptOssConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (spx.Rngs): Random number generators.
        base_model (GptOssModel): The underlying GPT-OSS transformer model.
        lm_head: Linear layer projecting hidden states to vocabulary logits.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gpt_oss"
    _config_class = GptOssConfig

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GPT-OSS Causal LM module.

        Args:
            config: Model configuration
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
        """
        super().__init__(
            config=config,
            base_model_class=GptOssModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=config.router_aux_loss_coef,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass for GPT-OSS Causal Language Model.

        Processes input tokens through the GPT-OSS transformer with MoE layers
        and projects the output to vocabulary logits for next-token prediction.
        Computes auxiliary load balancing loss for MoE training when router
        logits are available.

        Args:
            input_ids (Array, optional): Input token IDs of shape
                (batch_size, sequence_length). Either this or `inputs_embeds`
                must be provided.
            inputs_embeds (Array, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Use instead of
                `input_ids` for custom embeddings.
            attention_mask (Array, optional): Boolean mask of shape
                (batch_size, sequence_length) indicating which tokens to attend
                to (True) and which to ignore (False).
            mask_info (MaskInfo, optional): Pre-computed mask information.
                If provided, overrides `attention_mask`.
            position_ids (Array, optional): Position indices of shape
                (batch_size, sequence_length). Auto-generated if not provided.
            output_attentions (bool, optional): Whether to return attention weights.
                Defaults to config.output_attentions.
            output_hidden_states (bool, optional): Whether to return hidden states
                from all layers. Defaults to config.output_hidden_states.
            output_router_logits (bool, optional): Whether to return router logits
                from MoE layers. Defaults to config.output_router_logits.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER).
                Auto-detected if None.
            past_key_values: Cached key/value states for efficient autoregressive
                generation.
            cache_metadata: Metadata for paged attention mechanisms.
            apply_lm_head (bool): Whether to apply the language modeling head.
                Set to False to get hidden states only. Defaults to True.

        Returns:
            MoeCausalLMOutput containing:
                - logits: Vocabulary logits of shape (batch, seq_len, vocab_size)
                - aux_loss: Auxiliary load balancing loss for MoE training
                - past_key_values: Updated cache for next generation step
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
                - router_logits: Tuple of router logits if output_router_logits=True
        """

        def _aux_loss_fn(outputs, attention_mask):
            """Custom auxiliary loss for GPT-OSS."""
            if outputs.router_logits is None or len(outputs.router_logits) == 0:
                return None
            return auxiliary_load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )

        return self.forward_moe(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            aux_loss_fn=_aux_loss_fn,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=GptOssConfig, model_type="gpt_oss")
class GptOssForSequenceClassification(BaseSequenceClassificationModule[GptOssModel, GptOssConfig]):  # type: ignore
    """GptOss model with a Sequence Classification head.

    This model consists of the base GptOss transformer (`GptOssModel`) followed by a
    linear layer (`score`) that projects the transformer's output hidden states
    (typically the hidden state of the first token) to the number of classes for classification.
    It also handles the calculation of the auxiliary loss from the MoE layers.

    Attributes:
        config (GptOssConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (spx.Rngs): Random number generators.
        model (GptOssModel): The core GptOss transformer model.
        score (ParallelLinear): The linear layer for classification.
        num_experts (int): Total number of experts.
        num_experts_per_tok (int): Number of experts to route per token.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "gpt_oss"
    _config_class = GptOssConfig

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initializes the GptOssForSequenceClassification model.

        Args:
            config (GptOssConfig): The configuration object for the GptOss model.
                Must include `num_labels`.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (spx.Rngs): Random number generators.

        Raises:
            AssertionError: If `config.num_labels` is not defined.
        """
        super().__init__(
            config=config,
            base_model_class=GptOssModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy="last",
            score_head_bias=False,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass of the GptOssForSequenceClassification model.

        Args:
            input_ids (tp.Optional[Array]): Input token IDs. Shape: (batch_size, sequence_length).
            inputs_embeds (tp.Optional[Array]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (tp.Optional[Array]): Mask to avoid performing attention on padding token indices.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            segment_ids (tp.Optional[Array]): Segment IDs (unused).
            output_attentions (tp.Optional[bool]): Whether to return attention weights. Defaults to
                `config.output_attentions`.
            output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            output_router_logits (tp.Optional[bool]): Whether to return router logits from the MoE layers.
                Defaults to `config.output_router_logits`.
            past_key_values (tp.Optional[TransformerCache | RaggedPagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]): Metadata for paged attention.


        Returns:
           SequenceClassifierOutput: The model's output.
                returns a `SequenceClassifierOutput` object containing `logits`, `aux_loss` (optional),
                `hidden_states` (optional), `attentions` (optional), and `router_logits` (optional).


        Raises:
            ValueError: If `config.pad_token_id` is None and `batch_size > 1`.
        """
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            inputs_embeds=inputs_embeds,
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
        aux_loss = None
        if output_router_logits and transformer_outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=transformer_outputs.router_logits,
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            aux_loss += aux_loss * self.config.router_aux_loss_coef

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            aux_loss=aux_loss,
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

    def get_task_head(self):
        """Returns the sequence classification head."""
        return self.score
