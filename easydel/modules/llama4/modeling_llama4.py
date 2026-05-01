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

"""Llama4 multimodal model implementation.

This module implements Meta's Llama4 (a.k.a. Llama 3.3) family of models, which
extends the LLaMA decoder architecture with:

- A sparse Mixture-of-Experts (MoE) feedforward path interleaved with dense MLP
  blocks (controlled by ``moe_layers`` / ``interleave_moe_layer_step``).
- Per-layer RoPE control (some layers skip RoPE) and chunked attention with
  attention-temperature tuning for very long contexts.
- Optional QK normalization in attention.
- A ViT-style vision encoder with pixel-shuffle downsampling and a multi-modal
  projector that maps vision features into the text embedding space.

Exports:
    - ``Llama4TextExperts``, ``Llama4TextMLP``, ``Llama4TextMoe``: MoE / FFN blocks.
    - ``Llama4TextAttention``, ``Llama4TextDecoderLayer``: text decoder primitives.
    - ``Llama4TextModel``, ``Llama4ForCausalLM``, ``Llama4ForSequenceClassification``:
      text-only model variants.
    - ``Llama4VisionAttention``, ``Llama4VisionEncoder``, ``Llama4VisionEncoderLayer``,
      ``Llama4VisionMLP``, ``Llama4VisionMLP2``, ``Llama4VisionPixelShuffleMLP``,
      ``Llama4UnfoldConvolution``, ``Llama4VisionModel``: vision tower components.
    - ``Llama4MultiModalProjector``, ``Llama4ForConditionalGeneration``: multimodal
      projection and full vision-language model.
    - ``Llama4CausalLMOutputWithPast``: structured output for autoregressive runs.
"""

import math
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.pytree import auto_pytree
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
from easydel.infra.base_module import EasyDeLBaseModule, EasyDeLLayerStackMixin
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
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelLinear,
    RowParallelMoELinear,
)
from easydel.layers import RMSNorm as Llama4TextRMSNorm
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule, UnifiedAttention
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule, BaseVisionLanguageModule
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


@ejit(static_argnums=(0, 1, 2, 3))  # pyright: ignore[reportUntypedFunctionDecorator]
def _vision_freqs(idx, hidden_size, num_attention_heads, rope_theta):
    """Compute complex rotary frequencies for the vision transformer grid.

    Builds 2D rotary embeddings over an ``idx`` x ``idx`` patch grid (plus a
    sentinel position used by the class token), interleaving x- and y-axis
    frequencies and returning them in complex form ready to be multiplied with
    queries/keys.

    Args:
        idx (int): Patch grid side length (i.e. ``image_size // patch_size``).
        hidden_size (int): Vision encoder hidden dimension.
        num_attention_heads (int): Number of vision attention heads.
        rope_theta (float): RoPE base frequency.

    Returns:
        jnp.ndarray: Complex frequencies of shape ``(idx**2 + 1, 1, head_dim)``,
        with the trailing position zeroed out to act as a class-token sentinel.
    """
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


def _create_chunked_attention_mask(  # pyright: ignore[reportUnusedFunction]
    attention_chunk_size: int,
    start: int,
    end: int,
):
    """Create a chunked causal attention mask for sliding-window attention.

    Builds a boolean mask over positions ``[start, end)`` such that token ``i``
    can attend to token ``j`` only if they fall in the same chunk
    (``i // chunk == j // chunk``) and ``j <= i``.

    Args:
        attention_chunk_size (int): Length of one attention chunk.
        start (int): Inclusive start position.
        end (int): Exclusive end position.

    Returns:
        jnp.ndarray: Boolean mask of shape ``(end - start, end - start)``.
    """
    blcok_position = jnp.abs(
        (jnp.arange(start, end)[None, :] // attention_chunk_size)
        - jnp.arange(start, end)[:, None] // attention_chunk_size
    )
    token_position = jnp.arange(start, end)[None, :] - jnp.arange(start, end)[:, None]
    return ((blcok_position == 0) & (token_position <= 0)).astype("b1")


class Llama4TextExperts(spx.Module):
    """Batched SwiGLU expert bank for Llama-4's MoE FFN.

    Llama-4 FFN experts are stacked along an extra leading axis of size
    ``num_local_experts`` and executed in a single grouped GEMM via
    :class:`ColumnParallelMoELinear` / :class:`RowParallelMoELinear` rather
    than looped in Python. Each expert is a SwiGLU MLP::

        y = down( silu(gate(x)) * up(x) )

    The HuggingFace checkpoint stores the gate and up projections fused
    along the last dim — the ``reform_param`` map below splits and rejoins
    them so EasyDeL state-dicts round-trip cleanly. ``ColumnParallelMoELinear``
    handles both the ``num_experts`` axis and the tensor-parallel hidden
    axis, so this single module can be partitioned across both expert
    parallelism (EP) and tensor parallelism (TP).

    Attributes:
        gate_proj, up_proj (ColumnParallelMoELinear): Per-expert gate and
            value projections, ``hidden_size -> intermediate_size``.
        down_proj (RowParallelMoELinear): Per-expert output projection,
            ``intermediate_size -> hidden_size``.
        act_fn (Callable): Per-expert non-linearity (typically SiLU).
        num_experts (int): Number of routed experts (``num_local_experts``).
        intermediate_size, hidden_size, expert_dim (int): Mirrors
            corresponding fields on ``Llama4Config``.
    """

    reform_param: tp.ClassVar = {
        # HuggingFace has fused gate_up_proj, we split into separate gate_proj and up_proj
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.weight", "spliter": lambda x: x[:, :, : x.shape[-1] // 2]},
                {"name": "up_proj.weight", "spliter": lambda x: x[:, :, x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda g, u: jnp.concatenate([g, u], axis=-1),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.weight", "spliter": lambda x: x},
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
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 text experts module.

        Args:
            config (Llama4Config): Model configuration with expert parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            use_bias=False,
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
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            use_bias=False,
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
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Forward pass through MoE experts.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].
            group_sizes: Array specifying the sizes of expert groups.
            sorted_experts: Optional array of sorted expert indices.

        Returns:
            Transformed hidden states after expert processing [batch, seq_len, hidden_dim].
        """
        gate = self.gate_proj(hidden_states, group_sizes, sorted_experts)
        up = self.up_proj(hidden_states, group_sizes, sorted_experts)
        return self.down_proj(self.act_fn(gate) * up, group_sizes, sorted_experts)


class Llama4TextL2Norm(spx.Module):
    """Parameter-free L2 normalization (RMSNorm without affine scale).

    Used by :class:`Llama4TextAttention` as the optional Q/K normalization.
    Computes ``x / sqrt(mean(x ** 2) + eps)`` along the last axis, with the
    rescale done in fp32 for numerical stability and the result cast back
    to the input dtype. Unlike :class:`RMSNorm` there is no learnable
    weight — that is intentional: Q/K-norm is purely about taming pre-softmax
    activation scale, and Llama-4 found a learnable scale was redundant
    with the per-head temperature tuning.
    """

    kernel_init = staticmethod(jax.nn.initializers.ones)

    def __init__(self, eps: float = 1e-6) -> None:
        """Initialize L2 normalization layer.

        Args:
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        """
        self.eps = eps

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute L2 normalization along the last axis.

        Args:
            x (jnp.ndarray): Input array of shape ``(..., hidden_dim)``.

        Returns:
            jnp.ndarray: Normalized array of the same shape as ``x``.
        """
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    @jax.named_scope("easydel-L2norm")
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply L2 normalization.

        Args:
            x: Input tensor to normalize.

        Returns:
            L2-normalized tensor with the same shape as input.
        """
        return self._norm(x.astype(jnp.float32)).astype(x.dtype)


class Llama4TextMLP(spx.Module):
    """Dense SwiGLU FFN used as the *shared* expert path inside the Llama-4 MoE.

    Identical in structure to LLaMA's standard MLP (gate / up / down with
    SiLU), but instantiated separately because Llama-4's MoE block routes
    every token through this dense expert in addition to the top-k routed
    experts (see :class:`Llama4TextMoe`). The shared expert captures
    "always-useful" features and stabilizes early-training routing.

    Attributes:
        gate_proj (ColumnParallelLinear): ``hidden_size -> intermediate_size``.
        up_proj (ColumnParallelLinear): ``hidden_size -> intermediate_size``.
        down_proj (RowParallelLinear): ``intermediate_size -> hidden_size``.
        activation_fn (Callable): SiLU (default) or anything in ``ACT2FN``.
    """

    def __init__(
        self,
        config: Llama4Config,
        intermediate_size=None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 text MLP block.

        Args:
            config (Llama4Config): Model configuration with MLP parameters.
            intermediate_size (int, optional): Size of intermediate layer. If None, uses
                config.intermediate_size. Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim].
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = checkpoint_name(self.activation_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Llama4TextMoe(BaseMoeModule):
    """Sparse Mixture-of-Experts FFN for Llama-4 text decoder layers.

    Llama-4 differs from Mixtral / DeepSeek-V3 in two important ways that
    are wired up by the ``moe_hooks`` overrides at the bottom of ``__init__``:

    1. **Sigmoid + top-k routing (no softmax)**. Router logits are turned
       into per-expert weights by ``sigmoid(logit)``, then everything outside
       the top-``num_experts_per_tok`` is masked to zero. There is *no*
       sum-to-one renormalization — sigmoid routing intentionally lets a
       token's total expert mass vary.
    2. **Input scaling, not output scaling**. The router weights are folded
       into the *input* of each expert (``expert(input * weight)``) instead
       of multiplied with the expert's output. The output-combination
       weights are therefore set to all-ones so that
       :func:`BaseMoeModule.unpermute` does not double-scale.

    A single dense :class:`Llama4TextMLP` (the "shared expert") is also run
    on every token and added on top of the routed-expert sum. Compute cost
    per token is ``shared_expert + num_experts_per_tok routed experts``.

    Attributes:
        router (ColumnParallelLinear): Single-layer linear projecting
            ``hidden_size -> num_local_experts`` for routing logits.
        experts (Llama4TextExperts): Batched SwiGLU expert bank.
        shared_expert (Llama4TextMLP): Always-on dense expert.
        top_k (int): Equal to ``num_experts_per_tok``.
        num_experts (int): Equal to ``num_local_experts`` (routed only).
    """

    def __init__(
        self,
        config: Llama4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 Mixture of Experts layer.

        Args:
            config (Llama4Config): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        training: bool = False,
        layer_idx: int | None = None,
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass through the MoE layer.

        Routes inputs through both shared and specialized experts,
        combining their outputs for the final result.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].
            training (bool, optional): Whether in training mode. Defaults to False.
            layer_idx (int | None, optional): Index of the current layer. Defaults to None.

        Returns:
            Tuple of (combined expert output, router logits) where output has
            shape [batch, seq_len, hidden_dim].
        """
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
            wi_kernel=self.experts.gate_proj.weight.value,
            wu_kernel=self.experts.up_proj.weight.value,
            wd_kernel=self.experts.down_proj.weight.value,
            act_fn=self.experts.act_fn,
            ffn_activation=ffn_activation,
            layer_idx=layer_idx,
        )

        final_output = checkpoint_name(shared_out + expert_out, "moe_expert_output")
        return final_output, checkpoint_name(router_logits, "moe_router_logits")  # pyright: ignore[reportReturnType]


class Llama4TextAttention(UnifiedAttention):
    """Llama-4 text attention with per-layer NoPE and runtime temperature tuning.

    Two architectural twists over standard LLaMA attention:

    * **Per-layer RoPE schedule (NoPE)**. Layers where
      ``(layer_idx + 1) % 4 == 0`` skip RoPE entirely (``use_rope = False``)
      and rely solely on positional information leaked through earlier
      layers — Llama-4's "NoPE" trick that improves length extrapolation.
      QK-norm (``Llama4TextL2Norm``) is also disabled on those NoPE layers
      since there is no rotary subspace to renormalize.
    * **Attention-temperature tuning**. When ``attn_temperature_tuning`` is
      enabled, an extra per-token *log-position*-dependent scaling factor
      is multiplied into the queries before the softmax, sharpening
      attention to local context for short sequences and broadening it for
      long ones. The schedule depends on ``attn_scale`` and ``floor_scale``.

    Inherits the standard MHA / GQA implementation from
    :class:`UnifiedAttention`; ``causal=False`` here because the causality
    is enforced through the supplied :class:`MaskInfo`, allowing chunked
    bidirectional attention on Llama-4's vision-text fusion path to share
    the same module.

    Attributes:
        use_rope (bool): Per-layer flag controlling the NoPE schedule.
        attn_scale, floor_scale (float): Coefficients of the
            log-position-based temperature schedule.
        attn_temperature_tuning (bool): Master switch for the schedule.
        qk_norm (Llama4TextL2Norm | None): Q/K L2-norm on RoPE layers; ``None``
            on NoPE layers and when ``use_qk_norm`` is off.
    """

    def __init__(
        self,
        config: Llama4TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Llama4 text attention layer.

        Args:
            config (Llama4TextConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model (determines RoPE usage).
        """
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

    def _create_attention_performer(self, config: Llama4TextConfig, rngs: spx.Rngs):
        """Build the attention computation backend.

        Args:
            config (Llama4TextConfig): Text decoder configuration.
            rngs (spx.Rngs): Random number generator state.

        Returns:
            FlexibleAttentionModule: Configured attention performer.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
        )

    def _create_rotary(self, config: Llama4TextConfig, dtype: jnp.dtype):
        """Build the rotary embedding for this layer, or skip RoPE entirely.

        Some Llama4 layers intentionally use no rotary embedding; this method returns
        ``None`` for those layers and falls back to the parent rotary otherwise.

        Args:
            config (Llama4TextConfig): Text decoder configuration.
            dtype (jnp.dtype): Computation dtype for the rotary tables.

        Returns:
            Rotary module or ``None`` when this layer skips RoPE.
        """
        # RoPE is handled via custom complex rotary frequencies when enabled.
        return None if not self.use_rope else super()._create_rotary(config, dtype)

    def _apply_rotary(
        self,
        query_states: Float[Array, "batch seq_len num_heads head_dim"],
        key_states: Float[Array, "batch seq_len num_kv_heads head_dim"],
        position_ids: Int[Array, "batch seq_len"],
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch seq_len num_kv_heads head_dim"]]:
        """Apply rotary position embeddings to queries and keys.

        Skips rotation entirely on no-RoPE layers and uses the complex-valued path
        when precomputed ``frequencies`` are supplied.

        Args:
            query_states (Array): Query tensor of shape
                ``(batch, seq_len, num_heads, head_dim)``.
            key_states (Array): Key tensor of shape
                ``(batch, seq_len, num_kv_heads, head_dim)``.
            position_ids (Array): Position indices of shape ``(batch, seq_len)``.
            frequencies (Array | None, optional): Precomputed complex rotary
                frequencies of shape ``(seq_len, head_dim)``. Defaults to None.

        Returns:
            tuple[Array, Array]: Rotated query and key tensors with shapes matching
            their inputs.
        """
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
        """Apply QK normalization and attention temperature tuning.

        On layers configured with ``use_qk_norm``, applies L2 normalization to
        queries and keys. On no-RoPE layers, optionally rescales queries by a
        position-dependent temperature controlled by ``attn_scale`` / ``floor_scale``.

        Args:
            query_states (Array): Query tensor.
            key_states (Array): Key tensor.
            value_states (Array): Value tensor.

        Returns:
            tuple[Array, Array, Array]: Possibly normalized/rescaled
            ``(query, key, value)`` tensors.
        """
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


class Llama4TextDecoderLayer(spx.Module):
    """One Llama-4 text decoder block (pre-norm attention + dense-or-MoE FFN).

    Two layer-index-driven branches are decided in ``__init__``:

    * The attention block alternates between RoPE+chunked-attention layers
      (``(layer_idx + 1) % 4 != 0``) and NoPE+full-attention layers
      (``(layer_idx + 1) % 4 == 0``); the chunked behaviour is what allows
      the long-context Llama-4 variants to run efficiently with limited
      memory by chunking the attention into local windows on RoPE layers.
    * The FFN is a dense :class:`Llama4TextMLP` on layers *not* listed in
      ``config.moe_layers`` and a sparse :class:`Llama4TextMoe` otherwise.
      In the 128E (Llama-4 Maverick) checkpoint, dense and sparse FFN
      layers interleave; in the 16E (Llama-4 Scout) variant every block is
      MoE except for early dense ones.

    Pre-norm residual layout::

        x = x + self_attn(input_layernorm(x))
        x = x + feed_forward(post_attention_layernorm(x))   # router_logits returned for MoE

    Attributes:
        self_attn (Llama4TextAttention): Per-layer attention with RoPE/NoPE
            and chunked windowing decided by ``layer_idx``.
        feed_forward (Llama4TextMLP | Llama4TextMoe): Dense MLP or sparse MoE.
        input_layernorm, post_attention_layernorm (Llama4TextRMSNorm): Pre-
            attention and pre-FFN RMSNorms.
        is_moe_layer (bool): Whether this block uses the MoE FFN.
        use_chunked_attention (int): 1 if attention runs chunked (RoPE
            layers), 0 otherwise; consumed downstream by mask construction.
        layer_idx (int): Layer index used for cache slot selection.
    """

    def __init__(
        self,
        config: Llama4TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Llama4 text decoder layer.

        Args:
            config (Llama4TextConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.self_attn = Llama4TextAttention(
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
            self.feed_forward = Llama4TextMoe(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.feed_forward = Llama4TextMLP(
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

    def forward(
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
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + ffn(norm(x)).

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return MoE router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, router logits, and cache view.
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
    """Llama-4 text trunk: token embeddings + heterogenous decoder stack + final RMSNorm.

    Each block in :class:`Llama4TextDecoderLayer` is heterogenous along
    two axes: attention is RoPE-with-chunked-mask vs. NoPE-with-full-mask
    based on layer index, and the FFN is dense (:class:`Llama4TextMLP`)
    vs. sparse (:class:`Llama4TextMoe`) based on whether the layer index
    appears in ``config.moe_layers``. The trunk itself is therefore the
    *only* level that does not need to know about either dispatch — it
    just runs the layers via :meth:`nn.ModuleList.scan` with the cache
    threaded through, and lets each block decide what to do.

    Attributes:
        embed_tokens (Embed): Token embedding ``(vocab_size, hidden_size)``.
        layers (nn.ModuleList[Llama4TextDecoderLayer]): Heterogenous block
            stack assigned to pipeline stages via :func:`spx.assign_stage`
            and rematerialized per ``gradient_checkpointing``.
        norm (Llama4TextRMSNorm): Final RMSNorm at ``rms_norm_eps``.
    """

    def __init__(
        self,
        config: Llama4TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 text base model.

        Args:
            config (Llama4TextConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            rngs=rngs,
        )
        remat_layer_block = auto_remat(
            Llama4TextDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(self.config.num_hidden_layers):
            with spx.assign_stage(total=self.config.num_hidden_layers, current=layer_idx):
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
        self.norm = Llama4TextRMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Llama4 text base model.

        Processes input tokens through embedding, all decoder layers with RoPE and RMSNorm,
        and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
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
            partition_manager=self.config.runtime_sharding_resolver,
        )
        mask_info = mask_info.apply_chunked(self.config.attention_chunk_size)
        frequencies = self.compute_complex_rotary(position_ids)

        def _layer_loop(block, carry):
            hidden_states, all_hidden_states, all_attentions, idx = carry
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
                    frequencies=frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)

            return hidden_states, all_hidden_states, all_attentions, idx + 1

        hidden_states, all_hidden_states, all_attentions, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, 0),
            trace=True,
        )
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
    """Llama4 model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with causal attention masks
    applied to perform autoregressive language generation.

    Attributes:
        config (Llama4TextConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

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
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 model for causal language modeling.

        Args:
            config (Llama4TextConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
    """Llama4 model for sequence classification tasks.

    This class extends the base Llama4 model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
        config (Llama4TextConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

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
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 model for sequence classification.

        Args:
            config (Llama4TextConfig): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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


class Llama4MultiModalProjector(spx.Module):
    """Linear projection from vision-encoder output space to the text decoder's hidden space.

    Llama-4 vision tokens (after :class:`Llama4VisionPixelShuffleMLP` has
    pixel-shuffled the patch grid down) live in
    ``vision_config.vision_output_dim``-dimensional space; the text decoder
    expects ``text_config.hidden_size``-dimensional embeddings. This module
    is the *only* learned tensor on that bridge — a single bias-free linear,
    initialized small (``stddev = 0.01``) so that vision tokens enter the
    decoder near zero magnitude and do not destabilize the language model
    on the first training step. The text decoder treats the projected
    tokens identically to text tokens; positional information for the
    vision tokens comes from the surrounding text positions, not from the
    vision encoder's own RoPE.

    Attributes:
        linear_1 (RowParallelLinear): The vision-to-text projection.
    """

    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 multi-modal projector.

        Args:
            config: Model configuration with vision and text parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Project vision features to text embedding space.

        Args:
            hidden_states: Vision features [batch, seq_len, vision_hidden_dim].

        Returns:
            Projected features [batch, seq_len, text_hidden_dim].
        """
        return self.linear_1(hidden_states)


def pixel_shuffle(input_tensor, shuffle_ratio):
    """Rearrange flattened vision tokens to a denser spatial grid.

    Splits each spatial position's channels across a smaller spatial grid scaled
    by ``shuffle_ratio``, trading channel count for fewer tokens. The number of
    output tokens is ``num_patches * shuffle_ratio**2``.

    Args:
        input_tensor (jnp.ndarray): Flattened patch features of shape
            ``(batch, num_patches, channels)`` where ``num_patches`` is a perfect
            square.
        shuffle_ratio (float): Spatial downsampling ratio (typically 0.5).

    Returns:
        jnp.ndarray: Shuffled features of shape
        ``(batch, new_num_patches, channels / shuffle_ratio**2)``.
    """
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


class Llama4VisionPixelShuffleMLP(spx.Module):
    """Sub-pixel-shuffle downsampler followed by a SwiGLU MLP.

    Llama-4's vision encoder emits a square grid of
    ``(patch_size, patch_size)`` tokens at the encoder's hidden dim. To
    drop the token count fed to the text decoder by ``1 / pixel_shuffle_ratio**2``
    *without throwing information away*, this module reshuffles the
    spatial dimensions of each feature into the channel dimension via
    :func:`pixel_shuffle` (the inverse of PyTorch's ``pixel_shuffle``):
    the output is a ``shuffle_ratio``× smaller spatial grid with channels
    inflated by the inverse-square. A subsequent :class:`Llama4VisionMLP2`
    rescales those inflated channels back to ``projector_output_dim`` so
    they fit the next stage's expectations.

    This is the standard "sub-pixel/space-to-depth" trade-off used in
    vision transformers to balance resolution vs. token count.

    Attributes:
        pixel_shuffle_ratio (float): Spatial scaling factor (e.g. 0.5 keeps
            1 token per 4 input tokens).
        inner_dim (int): Channel count after shuffle, ``projector_input_dim
            / pixel_shuffle_ratio**2``.
        output_dim (int): Channel count after the trailing MLP.
        mlp (Llama4VisionMLP2): SwiGLU MLP that maps ``inner_dim ->
            output_dim``.
    """

    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 vision pixel shuffle MLP.

        Args:
            config: Vision configuration with pixel shuffle parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, encoded_patches: Array) -> Array:
        """Apply pixel shuffle and MLP transformation to vision features.

        Args:
            encoded_patches: Vision patch embeddings [batch, num_patches, hidden_dim].

        Returns:
            Downsampled and transformed features [batch, reduced_patches, output_dim].
        """
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


def reshape_for_broadcast(frequencies: jax.Array, query: jax.Array) -> jax.Array:
    """Reshape rotary frequencies so they broadcast over the complex query tensor.

    Inserts singleton dimensions everywhere except along the sequence (axis 1)
    and feature (last axis) dimensions, so that ``frequencies`` lines up with a
    ``(batch, seq, ..., head_dim)`` query.

    Args:
        frequencies (jax.Array): Complex rotary frequencies of shape
            ``(seq, head_dim)``.
        query (jax.Array): Reference complex query tensor whose shape determines
            the broadcast layout.

    Returns:
        jax.Array: Frequencies broadcastable to ``query.shape``.
    """
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
    """Apply rotary position embeddings to vision queries and keys.

    Treats consecutive feature pairs as the real and imaginary parts of a complex
    number, multiplies by precomputed complex ``frequencies``, and returns the
    rotated query and key tensors in their original real layout.

    Args:
        query (jax.Array): Query tensor with even-sized last dimension.
        key (jax.Array): Key tensor with the same last dimension as ``query``.
        frequencies (jax.Array): Complex rotary frequencies, broadcastable to the
            query/key complex layout.

    Returns:
        tuple[jax.Array, jax.Array]: Rotated query and key with shapes/dtypes
        matching their inputs.
    """
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
    """Bidirectional MHA with 2-D RoPE for the Llama-4 vision tower.

    Vision-encoder analog of :class:`Llama4TextAttention` but adapted for
    a ViT-style image tower:

    * **Bidirectional** (``causal=False``): vision tokens see each other
      symmetrically — there is no causal mask in image space.
    * **2-D RoPE**: the rotary frequencies are computed from a
      ``(grid_h, grid_w)`` patch grid by ``_vision_freqs`` and applied
      to Q/K via :func:`vision_apply_rotary_emb`. Even feature pairs
      encode the row coordinate, odd pairs the column coordinate, so the
      attention is intrinsically aware of 2-D spatial structure.
    * **MHA, not GQA** (``num_key_value_groups = 1``): vision encoders
      are typically narrow enough that GQA is unnecessary.
    * Biased linears (vision tower convention) and ``attention_dropout``
      drawn from the vision config.

    Attributes:
        embed_dim, num_heads, head_dim (int): Standard MHA shape parameters.
        attention_dropout (float): Dropout on attention probabilities.
    """

    def __init__(
        self,
        config: Llama4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Llama4 vision attention layer.

        Args:
            config (Llama4VisionConfig): Vision model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the encoder.
        """
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

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        output_attentions: bool = False,
    ) -> AttentionLayerOutput:
        """Forward pass through vision attention layer.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, num_patches, hidden_dim).
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.

        Returns:
            AttentionLayerOutput: Contains attention output and optional attention weights.
        """
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
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)
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


class Llama4VisionMLP2(spx.Module):
    """Post-shuffle MLP that maps inflated channels back into ``projector_output_dim``.

    Used inside :class:`Llama4VisionPixelShuffleMLP` after the
    spatial-to-channel fold has multiplied the channel count by
    ``1 / pixel_shuffle_ratio**2``. This module compresses those inflated
    channels back to ``projector_output_dim`` through ``fc1 -> GELU ->
    fc2 -> GELU`` (note the activation on *both* sides — this is a "block"
    activation, not a single MLP). All linears are bias-free and
    initialized small (``stddev = 0.01``) to keep the bridge near
    identity at initialization.

    Attributes:
        fc1, fc2 (ColumnParallelLinear): The two linear projections; their
            input/output widths come from ``config.intermediate_size``,
            ``config.projector_input_dim`` and ``config.projector_output_dim``.
        activation_fn (Callable): GELU.
        hidden_size, intermediate_size: Mirror config.
    """

    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 vision two-layer MLP.

        Args:
            config: Vision configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Apply two-layer feedforward transformation with GELU activation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Transformed features [batch, seq_len, output_dim].
        """
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return self.activation_fn(hidden_states)


class Llama4VisionMLP(spx.Module):
    """Vision-encoder MLP block — the FFN inside :class:`Llama4VisionEncoderLayer`.

    A standard ViT-style feed-forward block: ``fc1 -> GELU -> fc2`` with
    bias on both linears (vision encoders typically keep biases unlike
    the text decoder). Width expansion is ``hidden_size -> intermediate_size
    -> hidden_size``. Initialized small (``stddev = 0.01``) to keep the
    pre-trained CLIP-style weights well-conditioned at fine-tuning time.

    Attributes:
        fc1 (ColumnParallelLinear): Expansion projection.
        fc2 (ColumnParallelLinear): Contraction projection.
        activation_fn (Callable): GELU.
    """

    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 vision MLP block.

        Args:
            config: Vision configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Apply feedforward transformation with GELU activation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim].
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Llama4VisionEncoderLayer(spx.Module):
    """One ViT block: pre-norm bidirectional attention + GELU MLP.

    Vision-tower analog of :class:`Llama4TextDecoderLayer` but
    bidirectional (``causal=False`` on the attention) and with biased
    LayerNorm at ``epsilon = 1e-5``, biased linears, and 2-D RoPE
    frequencies precomputed by ``_vision_freqs`` and threaded in from the
    parent encoder. Pre-norm residual layout::

        x = x + self_attn(input_layernorm(x))
        x = x + mlp(post_attention_layernorm(x))

    Attributes:
        self_attn (Llama4VisionAttention): Bidirectional MHA with 2-D RoPE.
        mlp (Llama4VisionMLP): GELU FFN.
        input_layernorm, post_attention_layernorm (LayerNorm): Biased
            LayerNorms (epsilon = 1e-5).
        layer_idx (int): Index of this block within the vision encoder.
    """

    def __init__(
        self,
        config: Llama4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Llama4 vision encoder layer.

        Args:
            config (Llama4VisionConfig): Vision model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the encoder.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.self_attn = Llama4VisionAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = Llama4VisionMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.input_layernorm = LayerNorm(
            num_features=config.hidden_size,
            epsilon=0.00001,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = LayerNorm(
            num_features=config.hidden_size,
            epsilon=0.00001,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """Forward pass through the vision encoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x)).

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, num_patches, hidden_dim).
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            EncoderLayerOutput: Contains hidden states and optional attention weights.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        return EncoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
        )


class Llama4VisionEncoder(EasyDeLLayerStackMixin, spx.Module):
    """Stack of :class:`Llama4VisionEncoderLayer` blocks executed via scan.

    Holds ``config.num_hidden_layers`` ViT blocks in an ``nn.ModuleList``
    that is consumed by :meth:`EasyDeLLayerStackMixin._layer_stage_context`
    so the layers can be assigned to pipeline stages and rematerialized
    according to ``config.gradient_checkpointing``. The 2-D RoPE
    frequencies (``frequencies`` argument on forward) are computed *once*
    in :class:`Llama4VisionModel` from the input image's patch grid and
    threaded through every block — the encoder itself does not own the
    rotary table.

    Attributes:
        layers (nn.ModuleList[Llama4VisionEncoderLayer]): The stacked ViT
            blocks (each rematerialized via :func:`auto_remat`).
    """

    def __init__(
        self,
        config: Llama4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 vision encoder.

        Args:
            config (Llama4VisionConfig): Vision model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        remat_layer_block = auto_remat(
            Llama4VisionEncoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(self.config.num_hidden_layers):
            with spx.assign_stage(total=self.config.num_hidden_layers, current=layer_idx):
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

    def forward(
        self,
        hidden_states: jax.Array,
        frequencies: jax.Array,
        attention_mask: jax.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through all vision encoder layers.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, num_patches, hidden_dim).
            frequencies (Array): Precomputed RoPE frequencies for positional encoding.
            attention_mask (Array | None, optional): Attention mask. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states. Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional hidden_states, and optional attentions.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        def _layer_loop(encoder_layer, carry):
            hidden_states, encoder_states, all_attentions, idx = carry
            if output_hidden_states:
                assert encoder_states is not None
                encoder_states = (*encoder_states, hidden_states)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = encoder_layer(
                    hidden_states=hidden_states,
                    output_attentions=output_attentions,
                    frequencies=frequencies,
                )

            if output_attentions:
                assert all_attentions is not None
                all_attentions = (*all_attentions, layer_outputs.attention_weight)

            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            return hidden_states, encoder_states, all_attentions, idx + 1

        hidden_states, encoder_states, all_attentions, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, encoder_states, all_attentions, 0),
            trace=output_hidden_states
            or output_attentions
            or not self.config.scan_layers
            or self._pipeline_stage_count() > 1,
        )
        if output_hidden_states:
            assert encoder_states is not None
            encoder_states = (*encoder_states, hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class Llama4UnfoldConvolution(spx.Module):
    """Patchify stem implemented as ``unfold`` + linear (im2col-style ViT embedding).

    Llama-4 ViTs do not use a strided convolution for the patch
    embedding; instead they unfold non-overlapping ``(patch_size,
    patch_size)`` windows from the input image into per-patch vectors of
    length ``num_channels * patch_h * patch_w`` (via
    :func:`jax.lax.conv_general_dilated_patches` with ``"VALID"`` padding
    and stride ``= patch_size``) and then apply a single bias-free linear
    map to ``hidden_size``. This is mathematically equivalent to a strided
    conv but factors cleanly into ``(reshape, linear)`` which is friendlier
    for tensor-parallel sharding of the embedding dimension.

    Attributes:
        kernel_size (tuple[int, int]): Patch height and width.
        stride (tuple[int, int]): Equal to ``kernel_size`` for non-overlapping
            patches.
        num_channels (int): Input image channel count.
        hidden_size (int): Output embedding dimension.
        linear (ColumnParallelLinear): Per-patch projection
            ``num_channels * patch_h * patch_w -> hidden_size``.
    """

    def __init__(
        self,
        config: Llama4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 unfold convolution layer.

        Args:
            config (Llama4VisionConfig): Vision model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jax.Array:
        """Extract and embed image patches.

        Args:
            hidden_states: Input image tensor [batch, channels, height, width].

        Returns:
            Patch embeddings [batch, num_patches, hidden_dim].
        """
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
    """Llama-4 ViT vision tower: patchify + class token + ViT encoder + projector.

    Pipeline at call time:

    1. :class:`Llama4UnfoldConvolution` patchifies the image into a
       sequence of ``(image_size / patch_size)**2`` token embeddings.
    2. A learnable ``class_embedding`` is appended to the sequence (so
       there are ``num_patches + 1`` tokens), and a learnable
       ``positional_embedding_vlm`` is added to all tokens.
    3. ``layernorm_pre`` (LayerNorm) applied; the encoder
       (:class:`Llama4VisionEncoder`) processes the tokens with 2-D
       RoPE-augmented bidirectional attention.
    4. ``layernorm_post`` applied, the trailing class token is dropped
       (Llama-4 uses CLS only as a pooling target during pre-training,
       not for downstream language modelling), and
       :class:`Llama4VisionPixelShuffleMLP` reduces the spatial token
       count via sub-pixel shuffle so the language model gets a manageable
       sequence length.

    Attributes:
        patch_embedding (Llama4UnfoldConvolution): Im2col-style patchifier.
        class_embedding (ArrayParam): Learnable CLS token.
        positional_embedding_vlm (ArrayParam): Learnable absolute positional
            table covering ``num_patches + 1`` tokens.
        layernorm_pre, layernorm_post (LayerNorm): Pre- and post-encoder
            LayerNorms.
        model (Llama4VisionEncoder): The ViT encoder stack.
        vision_adapter (Llama4VisionPixelShuffleMLP): Sub-pixel-shuffle
            spatial downsampler + MLP into ``projector_output_dim``.
        vision_idx (int): Patch grid edge length, ``image_size // patch_size``.
    """

    def __init__(
        self,
        config: Llama4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 vision model.

        Args:
            config (Llama4VisionConfig): Vision model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
            key=rngs.parameters,
        )
        self.positional_embedding_vlm = ArrayParam.bound(
            shape=(self.num_patches, self.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": self.scale},
            key=rngs.parameters,
        )
        self.layernorm_pre = LayerNorm(
            num_features=self.hidden_size,
            epsilon=0.00001,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layernorm_post = LayerNorm(
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

    def forward(
        self,
        pixel_values: jax.Array,
        attention_mask: jax.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Llama4 vision model.

        Processes input images through patch embedding, transformer encoder layers,
        and a vision adapter for output projection.

        Args:
            pixel_values (Array): Input images of shape (batch_size, num_channels, height, width).
            attention_mask (Array | None, optional): Attention mask for the encoder. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states. Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state (processed vision features),
                optional hidden_states, and optional attentions.
        """
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
class Llama4ForConditionalGeneration(BaseVisionLanguageModule[Llama4ForCausalLM, Llama4Config]):  # type: ignore
    """End-to-end Llama-4 Vision-Language model: ViT + projector + Llama-4 text decoder.

    Pipeline at call time:

    1. ``vision_model`` (a :class:`Llama4VisionModel`, an aspect-ratio-aware
       ViT with pixel-shuffle downsampling) ingests pixel tiles and produces
       a sequence of vision tokens at ``vision_config.vision_output_dim``.
    2. ``multi_modal_projector`` (:class:`Llama4MultiModalProjector`) maps
       those tokens into the text decoder's ``hidden_size`` space.
    3. The projected vision tokens are spliced into the text token sequence
       at the positions marked by the special image token id, producing a
       merged ``inputs_embeds`` for the language model.
    4. ``language_model`` is a *complete* :class:`Llama4ForCausalLM` —
       contrary to most VLMs in EasyDeL where the base trunk lacks an LM
       head — so the LM head and embedding tying live inside the language
       model and not at this level.

    Class attributes flag IMAGE_TEXT_TO_TEXT capability and indicate that
    Llama-4 supports interleaved video frames (``_supports_video = True``)
    while using standard RoPE rather than M-RoPE (``_uses_mrope = False``).

    Attributes:
        vision_model (Llama4VisionModel): The visual tower; named
            ``vision_model`` to match the upstream HF checkpoint layout
            (declared via ``_vision_tower_name``).
        multi_modal_projector (Llama4MultiModalProjector): Vision-to-text
            projection (declared via ``_projector_name``).
        language_model (Llama4ForCausalLM): Full causal LM with its own LM
            head (declared via ``_language_model_name``).
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
        rngs: spx.Rngs,
    ):
        """Initialize Llama4 vision-language model for conditional generation.

        Args:
            config (Llama4Config): Full model configuration including vision and text configs.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        image_features: Array | None = None,
        pixel_values: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute input embeddings with merged image and text features.

        Processes input token IDs through the text embedding layer, extracts
        image features if pixel_values are provided, projects them through the
        multi-modal projector, and replaces image token positions with the
        projected vision features.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            image_features (Array | None, optional): Pre-extracted image features.
                If None and pixel_values provided, features are extracted. Defaults to None.
            pixel_values (Array | None, optional): Raw pixel values for image extraction.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to get_image_features.

        Returns:
            Array: Combined embeddings of shape (batch_size, sequence_length, hidden_size)
                with projected vision features merged at image token positions.

        Raises:
            ValueError: If input_ids is None.
        """
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

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
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
        """Forward pass through the vision-language model.

        Processes images through the vision encoder if pixel_values are provided,
        projects visual features, and generates text conditioned on both modalities.

        Args:
            input_ids (Array, optional): Input token IDs of shape (batch_size, sequence_length).
            pixel_values (Array, optional): Input images of shape (batch_size, num_channels, height, width).
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens.
                Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information. Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode for optimization. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states. Defaults to None.
            **lm_kwargs: Additional arguments passed to the language model.

        Returns:
            VLMCausalLMOutput: Contains logits, past_key_values, hidden_states, attentions,
                and image_hidden_states.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            ValueError: If pixel_values is provided without input_ids.
        """
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
        """Initialize the key-value cache for autoregressive generation.

        Delegates to the underlying language model's cache initialization.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions for cache initialization.
                Defaults to None.
            shardings (Any | None, optional): Sharding specifications for the cache.
                Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache for the language model.
        """
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

    def get_vision_tower(self) -> spx.Module:
        """Returns the vision tower component."""
        return self.vision_model

    def get_projector(self) -> spx.Module:
        """Returns the multimodal projector component."""
        return self.multi_modal_projector

    def get_language_model(self) -> spx.Module:
        """Returns the language model component."""
        return self.language_model
