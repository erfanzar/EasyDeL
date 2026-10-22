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

"""Qwen3-VL-MoE multimodal model implementation for EasyDeL.

This module implements the Mixture-of-Experts variant of Qwen3-VL: the
same vision tower and 3-D mRoPE language schedule as Qwen3-VL, but with
the dense MLP in the language backbone replaced by a routed sparse MoE
block (with optional shared expert).

Exposes the vision tower components, the LM trunk
:class:`Qwen3VLMoeModel`, and the multimodal wrapper
:class:`Qwen3VLMoeForConditionalGeneration`. Token-to-expert routing
contributes a load-balancing auxiliary loss when training.
"""

from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
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
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    DecoderLayerOutput,
    EmbeddingInfo,
    ModelOutput,
    MoeCausalLMOutput,
    VLMCausalLMOutput,
)
from easydel.infra.sequence_packing import token_attention_mask_from_mask_info
from easydel.infra.utils import ACT2FN, auto_remat, blockwise_ffn
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RMSNorm,
    RowParallelMoELinear,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
)
from easydel.layers.mlp import moe_gated_expert_forward
from easydel.modules._base import BaseVisionLanguageModule, DeepStackVisualMixin, QwenVLMultimodalMixin
from easydel.modules.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VisionTransformerPretrainedModel,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
)
from easydel.modules.qwen3_vl.modeling_qwen3_vl import (
    get_rope_index as qwen3_vl_get_rope_index,
)
from easydel.modules.qwen3_vl.modeling_qwen3_vl import (
    merge_multimodal_embeddings as merge_multimodal_embeddings,
)

from .qwen3_vl_moe_configuration import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig, Qwen3VLMoeVisionConfig


@auto_pytree
class Qwen3VLMoeCausalLMOutputWithPast(VLMCausalLMOutput):
    """Output class for Qwen3-VL-MoE causal language model.

    Attributes:
        loss: Language modeling loss if labels provided.
        logits: Prediction scores of the language model head.
        past_key_values: Tuple of past key values for efficient generation.
        hidden_states: Tuple of hidden states at each layer.
        attentions: Tuple of attention weights at each layer.
        rope_deltas: RoPE position deltas for mRoPE.
        image_hidden_states: Hidden states from image processing.
        router_logits: Tuple of router logits from MoE layers.
    """

    loss: Array | None = None
    logits: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None
    router_logits: tuple[Array] | None = None

    def __init__(
        self,
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        rope_deltas=None,
        image_hidden_states=None,
        router_logits=None,
        **kwargs,
    ):
        """Preserve the original positional API while accepting unified VLM fields."""
        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            rope_deltas=rope_deltas,
            image_hidden_states=image_hidden_states,
            router_logits=router_logits,
            **kwargs,
        )

    def __reduce__(self):
        """Restore fields and ordered mapping through pickle state, not positional arguments.

        ModelOutput's positional reconstruction follows inherited dataclass order,
        whereas this class deliberately keeps its legacy constructor order.
        OrderedDict's state and item iterator already preserve every field.
        """
        constructor, _args, *state = super().__reduce__()
        return constructor, (), *state

    def __post_init__(self):
        """Keep legacy mapping, tuple and integer-index ordering before added VLM fields."""
        super().__post_init__()
        self._restore_legacy_order()

    def _sync_output_field(self, name, value):
        """Preserve legacy ordering on mutation and pickle/deepcopy reconstruction."""
        super()._sync_output_field(name, value)
        self._restore_legacy_order()

    def _restore_legacy_order(self):
        """Place historical fields ahead of newly inherited VLM fields."""
        for name in reversed(
            (
                "loss",
                "logits",
                "past_key_values",
                "hidden_states",
                "attentions",
                "rope_deltas",
                "image_hidden_states",
                "router_logits",
            )
        ):
            if name in self:
                self.move_to_end(name, last=False)


@auto_pytree
class Qwen3VLMoeModelOutputWithPast(ModelOutput):
    """Base-model output for Qwen3-VL-MoE with optional mRoPE deltas.

    Attributes:
        last_hidden_state: Final hidden state from the model.
        past_key_values: Tuple of past key values for efficient generation.
        hidden_states: Tuple of hidden states at each layer.
        attentions: Tuple of attention weights at each layer.
        rope_deltas: RoPE position deltas for mRoPE.
        router_logits: Tuple of router logits from MoE layers.
    """

    last_hidden_state: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    router_logits: tuple[Array] | None = None


def get_rope_index(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None = None,
    video_grid_thw: np.ndarray | None = None,
    attention_mask: np.ndarray | None = None,
    spatial_merge_size: int = 1,
    image_token_id: int = -1,
    video_token_id: int = -1,
    vision_start_token_id: int = -1,
    tokens_per_second: float = 1.0,
    second_per_grid_ts: list[float] | None = None,
    context_len: int = 0,
    seq_len: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Calculate 3D RoPE indices for multimodal inputs.

    Computes position IDs with temporal, height, and width dimensions for
    proper 3D rotary position embeddings (mRoPE) in Qwen3-VL-MoE.

    Args:
        input_ids: Token IDs of shape (batch_size, sequence_length).
        image_grid_thw: Temporal/height/width grid for images.
        video_grid_thw: Temporal/height/width grid for videos.
        attention_mask: Attention mask for padding.
        spatial_merge_size: Spatial merge factor for vision.
        image_token_id: Token ID representing images.
        video_token_id: Token ID representing videos.
        vision_start_token_id: Token ID for vision sequence start.
        tokens_per_second: Temporal scaling for video tokens.
        second_per_grid_ts: Per-video seconds per temporal grid step.
        context_len: Existing KV context length offset.
        seq_len: Target sequence length.

    Returns:
        Tuple of (position_ids, mrope_position_deltas).
        position_ids has shape (3, batch_size, sequence_length).
    """
    return qwen3_vl_get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        tokens_per_second=tokens_per_second,
        second_per_grid_ts=second_per_grid_ts,
        context_len=context_len,
        seq_len=seq_len,
    )


@register_module(TaskType.BASE_VISION, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeVisionTransformerPretrainedModel(Qwen3VisionTransformerPretrainedModel):
    """Qwen3-VL-MoE vision tower.

    Structurally identical to the Qwen3-VL tower: the patch embed, patch
    merger, MLP, attention, block and the tower itself compared
    code-identical (docstrings aside) between the two families, so this
    subclasses rather than restating them. The MoE variant differs only
    in its text stack.

    The name is load-bearing -- ``easydel/__init__.py``,
    ``modules/qwen3_vl_moe/__init__.py`` and
    ``modules/qwen3_5_moe/modeling_qwen3_5_moe.py`` all refer to it.
    """

    config_class = Qwen3VLMoeVisionConfig


# The MoE text attention was code-identical to the dense one; alias rather
# than restate it. Kept as a module-level name because the decoder layer
# below constructs it.
Qwen3VLMoeTextAttention = Qwen3VLTextAttention


class Qwen3VLMoeTextMLP(Qwen3VLTextMLP):
    """Dense SwiGLU block, sharing Qwen3-VL's parameter and checkpoint layout."""


class Qwen3VLMoeMLPStack(spx.Module):
    """Qwen3-VL-MoE MoE MLP using the new ParallelMoELinear layers.

    Implements the expert MLP stack for Mixture of Experts, where each expert
    is a SwiGLU MLP that processes tokens routed to it.
    """

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL-MoE MLP stack for MoE.

        Args:
            config (Qwen3VLMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate_up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=2 * config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        return {
            **moe_fused_gate_up_reform_param(config=self.config),
            **moe_down_projection_reform_param(),
        }

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply the per-expert SwiGLU MLP on grouped tokens.

        ``hidden_states`` is expected to be already partitioned into
        contiguous groups by destination expert (see the parent
        :class:`BaseMoeModule` for the routing dispatcher). Each grouped
        tile is then projected through the per-expert ``gate``/``up``
        ``ColumnParallelMoELinear``s, mixed via SwiGLU
        (``act_fn(gate) * up``), and reduced back to ``hidden_size`` by
        the ``RowParallelMoELinear`` ``down_proj``.

        Args:
            hidden_states: Token activations after dispatch, shape
                ``(num_dispatched_tokens, hidden_size)``.
            group_sizes: Per-expert group sizes used to slice
                ``hidden_states`` into the right per-expert blocks.
            sorted_experts: Optional permutation of expert ids matching
                the dispatch order; needed when expert tensor parallelism
                is enabled.

        Returns:
            Per-expert MLP outputs of shape
            ``(num_dispatched_tokens, hidden_size)`` ready to be
            scattered back to their source token positions.
        """
        return moe_gated_expert_forward(self, hidden_states, group_sizes, sorted_experts)


class Qwen3VLMoeTextSparseBlock(BaseMoeModule):
    """Sparse Mixture of Experts (MoE) block for Qwen3-VL-MoE text decoder.

    This block routes input hidden states to a selected subset of experts
    and combines their outputs.

    Attributes:
        config (Qwen3VLMoeTextConfig): Configuration object for the model.
        gate (ParallelLinear): Linear layer for the gating network.
        experts (Qwen3VLMoeMLPStack): Stack of expert MLP modules.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (spx.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initializes the Qwen3VLMoeTextSparseBlock module.

        Args:
            config (Qwen3VLMoeTextConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.bfloat16).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.bfloat16).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (spx.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K if config.norm_topk_prob else MoeRoutingStrategy.TOP_K_NDIV,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        # Qwen normalizes even a single selected expert, unlike the shared
        # Mixtral-style TOP_K default. Install the family policy once, not in
        # forward, so mutable-aware transforms see no structural mutation.
        if config.norm_topk_prob and self.moe_hooks.refine_weights_hook is None and self.moe_hooks.select_hook is None:
            self.moe_hooks = self.moe_hooks.replace(refine_weights_hook=self.normalize_selected_weights)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

        self.experts = Qwen3VLMoeMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_idx: int | None = None

    @staticmethod
    def normalize_selected_weights(weights: Array) -> Array:
        """Normalize Qwen's selected probabilities, including top-1 routing.

        Args:
            weights: Selected expert probabilities with experts on the last axis.

        Returns:
            Probabilities normalized over the selected experts for each token.
        """
        return weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Forward pass of the Sparse MoE block.

        Args:
            hidden_states (Array): Input hidden states (batch_size * sequence_length, hidden_dim).

        Returns:
            tuple[Array, Array]: A tuple containing:
                - final_hidden_states (Array): The output hidden states after MoE processing.
                - router_logits (Array): The logits output by the gating network.
        """
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            gate_up_kernel=self.experts.gate_up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Qwen3VLMoeTextDecoderLayer(spx.Module):
    """Transformer decoder layer for Qwen3-VL-MoE text model with conditional MoE/dense MLP.

    Combines multi-head attention with Q/K normalization and feedforward networks
    (either dense MLP or MoE) with RMS normalization and residual connections.
    """

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3-VL-MoE text decoder layer.

        Args:
            config (Qwen3VLMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.self_attn = Qwen3VLMoeTextAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.is_moe = (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and ((layer_idx + 1) % config.decoder_sparse_step == 0)
        )

        if self.is_moe:
            self.mlp = Qwen3VLMoeTextSparseBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            self.mlp.layer_idx = layer_idx
        else:
            self.mlp = Qwen3VLMoeTextMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x)).
        Uses either dense MLP or MoE depending on layer configuration.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices, shape (3, batch, seq_len) for mRoPE.
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
        attn_input = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            attn_input,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = checkpoint_name(hidden_states + attn_outputs.attention_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp and not self.is_moe:
            feed_forward_hidden_states = blockwise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        router_logits = None
        if self.is_moe:
            feed_forward_hidden_states, router_logits = feed_forward_hidden_states

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3VLMoeTextConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeTextModel(DeepStackVisualMixin, EasyDeLBaseModule):
    """Text decoder model for Qwen3-VL-MoE with MoE support.

    Implements the text decoder component of Qwen3-VL-MoE, utilizing transformer blocks
    with RMSNorm, Q/K normalization, multi-dimensional rotary position embeddings (mRoPE),
    and Mixture of Experts (MoE) for handling multimodal inputs.

    Attributes:
        config: Configuration for the text decoder.
        embed_tokens: Token embedding layer.
        layers: List of transformer decoder layers (dense and MoE).
        norm: Final layer normalization.
    """

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL-MoE text decoder model.

        Args:
            config (Qwen3VLMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        assert isinstance(config, Qwen3VLMoeTextConfig), (
            f"expected config to be of type Qwen3VLMoeTextConfig but got {type(config)}"
        )
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.embed_tokens = Embed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            Qwen3VLMoeTextDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            with self.assign_layer_stage(i, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=i,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    @cached_property
    def frequencies(self):
        """Get cached RoPE frequency tensor from config.

        Computes and caches the rotary position embedding frequencies
        based on the model's head dimension and rope_theta configuration.

        Returns:
            Array: Precomputed RoPE frequencies for position encoding.
        """
        head_dim = getattr(self.config, "qk_rope_head_dim", None) or self.config.head_dim
        return self.config.get_basic_frequencies(
            head_size=head_dim,
            rotary_dim=head_dim,
            base=self.config.rope_theta,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        visual_pos_masks: Bool[Array, "batch seq_len"] | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
    ) -> BaseModelOutput:
        """Forward pass through text decoder with MoE and optional deepstack visual integration.

        Processes input embeddings through transformer layers with mixture-of-experts feedforward
        networks. Supports deepstack visual feature injection at specified layer positions.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Mutually exclusive
                with inputs_embeds. Converted to embeddings via self.embed_tokens.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length, hidden_size).
                Mutually exclusive with input_ids. Use when embeddings are already prepared.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) where True indicates
                valid tokens, False indicates padding.
            mask_info: Pre-computed mask information for efficient attention operations. Auto-computed
                if None using MaskInfo.dynamic_init.
            position_ids: Position IDs of shape (3, batch_size, sequence_length) for 3D mRoPE.
                Dimensions: [temporal, height, width]. Defaults to standard 1D positions if None.
            mode: Runtime mode (MODE_TRAIN/DECODE/EVAL). Auto-detected: MODE_DECODE if seq_len=1
                and cache exists, else MODE_TRAIN.
            past_key_values: Cached key-value states from previous steps. Initialized as empty
                TransformerCache if None.
            cache_metadata: Metadata for paged attention (sequence lengths, block tables, etc.).
            output_attentions: Whether to return attention weights from all layers. Defaults to
                config.output_attentions.
            output_hidden_states: Whether to return hidden states from all layers. Defaults to
                config.output_hidden_states.
            output_router_logits: Whether to return MoE router logits for auxiliary loss. Defaults
                to config.output_router_logits.
            visual_pos_masks: Boolean mask of shape (batch_size, sequence_length) indicating positions
                where deepstack visual embeddings should be injected. Required if deepstack_visual_embeds provided.
            deepstack_visual_embeds: List of visual embedding arrays (one per layer) for deepstack injection.
                Each array shape: (num_visual_tokens, hidden_size). Injected additively at visual_pos_masks positions.

        Returns:
            MoeCausalLMOutput containing:
                - last_hidden_state: Final layer hidden states of shape (batch_size, sequence_length, hidden_size).
                - hidden_states: Tuple of all layer hidden states if output_hidden_states=True, else None.
                - attentions: Tuple of attention weights from all layers if output_attentions=True, else None.
                - past_key_values: Updated cache for next generation step.
                - router_logits: Tuple of router logits from MoE layers if output_router_logits=True, else None.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify either input_ids or inputs_embeds, but not both.")

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        sequence_length = inputs_embeds.shape[1]

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached! "
            f"(Expected <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        attention_mask = token_attention_mask_from_mask_info(mask_info, sequence_length)

        if position_ids is None:
            batch_size = inputs_embeds.shape[0]
            pos_2d = mask_info.q_position_ids
            position_ids = jnp.broadcast_to(pos_2d[None, :, :], (3, batch_size, sequence_length))

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

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        # Router-logit aggregation grows a Python tuple per layer, which
        # lax.scan cannot carry — collecting it forces the trace path too.
        # Deepstack injection indexes a per-layer Python list, so it also
        # requires the trace path.
        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or bool(output_router_logits) or bool(deepstack_visual_embeds),
        )
        cache_views = views if trace_layers else None
        # Hoisted out of the loop body: the caching property must not be
        # first materialized inside the lax.scan trace (tracer leak).
        frequencies = self.frequencies

        def _layer_loop(block, carry):
            """Apply a single language-model decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan``; runs ``block`` on the current
            hidden states, optionally accumulates per-layer hidden
            states, attention weights and MoE router logits, and returns
            the updated carry tuple ``(hidden_states, cv,
            all_hidden_states, all_attentions, all_router_logits, idx)``.
            """
            hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            # Truthiness so an empty list never compares against a traced scan index (r11 S10-2).
            if deepstack_visual_embeds and idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[idx],
                )

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )

            return hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, _, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, cache_views, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=trace_layers,
        )
        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeCausalLMOutput(  # pyright: ignore[reportReturnType]
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits if output_router_logits else None,
        )

    def get_encoder(self):
        """Get the encoder component.

        Raises:
            NotImplementedError: Text model does not have an encoder.
        """
        raise NotImplementedError("Text model does not have an encoder.")

    def get_decoder(self):
        """Get the decoder component.

        Returns:
            Qwen3VLMoeTextModel: The text decoder itself.
        """
        return self

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: Base model does not have a language model head.
        """
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        """Get the embedding layer.

        Returns:
            Embed: The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.VISION_LM, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeModel(QwenVLMultimodalMixin, EasyDeLBaseModule):
    """Qwen3-VL-MoE multimodal model integrating vision encoder and text decoder with MoE.

    This model handles the fusion of visual and textual information by:
    1. Processing images/videos through the vision encoder
    2. Merging visual embeddings into text embeddings at placeholder positions
    3. Computing 3D position IDs for proper mRoPE handling
    4. Running the fused embeddings through the text decoder with MoE layers

    Architecture matches HuggingFace pattern:
    - visual: Qwen3VLMoeVisionTransformerPretrainedModel
    - language_model: Qwen3VLMoeTextModel
    """

    def __init__(
        self,
        config: Qwen3VLMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL-MoE multimodal model.

        Args:
            config (Qwen3VLMoeConfig): Configuration for the multimodal model.
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

        self.visual = Qwen3VLMoeVisionTransformerPretrainedModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.language_model = Qwen3VLMoeTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_input_embeddings(self):
        """Get the input embeddings layer.

        Returns:
            Embed: Token embedding layer from the language model.
        """
        return self.language_model.get_embedding()

    def set_input_embeddings(self, value):
        """Set the input embeddings layer.

        Args:
            value: New embedding layer to use.
        """
        self.language_model.embed_tokens = value

    def set_decoder(self, decoder):
        """Set the decoder component.

        Args:
            decoder: New decoder module to use.
        """
        self.language_model = decoder

    def get_decoder(self):
        """Get the decoder component.

        Returns:
            Qwen3VLMoeTextModel: The language model decoder.
        """
        return self.language_model

    def get_encoder(self):
        """Get the encoder component.

        Returns:
            Qwen3VLMoeVisionTransformerPretrainedModel: The vision encoder.
        """
        return self.visual

    def get_rope_index(
        self,
        input_ids: Array,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        """Calculate 3D RoPE indices for multimodal inputs.

        Different from Qwen2-VL, Qwen3-VL-MoE uses timestamps rather than absolute
        time position IDs for videos.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length).
            image_grid_thw: Temporal/height/width grid for images.
            video_grid_thw: Temporal/height/width grid for videos.
            attention_mask: Attention mask for padding.

        Returns:
            Tuple of (position_ids, mrope_position_deltas).
            position_ids has shape (3, batch_size, sequence_length).
        """
        return Qwen3VLModel.get_rope_index(
            self, input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, attention_mask=attention_mask
        )

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute embeddings with optional multimodal fusion.

        Processes text embeddings and optionally merges visual embeddings
        at placeholder token positions.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length).
            inputs_embeds: Pre-computed text embeddings.
            pixel_values: Image pixel values.
            pixel_values_videos: Video pixel values.
            image_grid_thw: Grid dimensions for images.
            video_grid_thw: Grid dimensions for videos.
            image_max_grid_size: Maximum grid size for images.
            video_max_grid_size: Maximum grid size for videos.
            image_embeds: Pre-computed image embeddings.
            video_embeds: Pre-computed video embeddings.
            **kwargs: Additional arguments (unused).

        Returns:
            Array: Fused embeddings with visual features merged in.
        """
        return super().compute_embedding(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            image_embeds=image_embeds,
            video_embeds=video_embeds,
            **kwargs,
        )

    def compute_embedding_with_info(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        attention_mask: Array | None = None,
        deepstack_image_embeds: list[Array] | None = None,
        deepstack_video_embeds: list[Array] | None = None,
        **kwargs,
    ) -> tuple[Array, EmbeddingInfo]:
        """Compute embeddings with full information for generation.

        Similar to compute_embedding but returns additional metadata needed
        for generation, including position IDs, rope deltas, and deepstack
        visual embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length).
            inputs_embeds: Pre-computed text embeddings.
            pixel_values: Image pixel values.
            pixel_values_videos: Video pixel values.
            image_grid_thw: Grid dimensions for images.
            video_grid_thw: Grid dimensions for videos.
            image_max_grid_size: Maximum grid size for images.
            video_max_grid_size: Maximum grid size for videos.
            image_embeds: Pre-computed image embeddings.
            video_embeds: Pre-computed video embeddings.
            attention_mask: Attention mask for padding.
            deepstack_image_embeds: Optional cached per-layer image residuals, paired with image_embeds.
            deepstack_video_embeds: Optional cached per-layer video residuals, paired with video_embeds.
            **kwargs: Additional compatibility arguments.

        Returns:
            Tuple of (embeddings, EmbeddingInfo) containing fused embeddings
            and metadata for generation.

        Raises:
            ValueError: If input_ids is None or cached residual shapes/layer counts disagree.
        """
        for name, features, residuals in (
            ("image", image_embeds, deepstack_image_embeds),
            ("video", video_embeds, deepstack_video_embeds),
        ):
            if residuals is not None and (
                features is None or any(residual.shape != features.shape for residual in residuals)
            ):
                raise ValueError(f"Cached {name} DeepStack residuals require matching `{name}_embeds` shapes.")
        embeddings, info = super().compute_embedding_with_info(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            image_embeds=image_embeds,
            video_embeds=video_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        if deepstack_image_embeds is not None or deepstack_video_embeds is not None:
            images = info.deepstack_image_embeds if deepstack_image_embeds is None else deepstack_image_embeds
            videos = info.deepstack_video_embeds if deepstack_video_embeds is None else deepstack_video_embeds
            if images and videos and len(images) != len(videos):
                raise ValueError("Image and video DeepStack residuals must have the same number of layers.")
            image_mask, video_mask = self.get_placeholder_mask(input_ids, embeddings)
            info.visual_pos_masks, info.deepstack_visual_embeds = self.merge_deepstack_features(
                image_mask if images else None, video_mask if videos else None, images, videos
            )
            info.deepstack_image_embeds, info.deepstack_video_embeds = images, videos
        return embeddings, info

    def get_video_features(
        self,
        pixel_values_videos: Array,
        video_grid_thw: Array | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encodes videos into continuous embeddings.

        The deepstack visual features are also returned.

        Args:
            pixel_values_videos: The tensors corresponding to the input videos.
            video_grid_thw: The temporal, height and width of feature shape
                of each video in LLM.
            video_max_grid_size: Maximum grid size for videos.

        Returns:
            Tuple of (video_embeds_tuple, deepstack_video_embeds)
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(
        self,
        pixel_values: Array,
        image_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encodes images into continuous embeddings.

        The deepstack visual features are also returned.

        Args:
            pixel_values: The tensors corresponding to the input images.
            image_grid_thw: The temporal, height and width of feature shape
                of each image in LLM.
            image_max_grid_size: Maximum grid size for images.

        Returns:
            Tuple of (image_embeds_tuple, deepstack_image_embeds)
        """
        return super().get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

    def get_placeholder_mask(
        self,
        input_ids: Array | None,
        inputs_embeds: Array,
        image_features: Array | None = None,
        video_features: Array | None = None,
    ) -> tuple[Array, Array]:
        """Obtains multimodal placeholder mask from input_ids or inputs_embeds.

        Checks that the placeholder token count equals the length of multimodal features.
        If the lengths are different, an error is raised.

        Args:
            input_ids: Input token IDs.
            inputs_embeds: Input embeddings.
            image_features: Image features to validate against mask.
            video_features: Video features to validate against mask.

        Returns:
            Tuple of (special_image_mask, special_video_mask).
        """
        return super().get_placeholder_mask(input_ids, inputs_embeds, image_features, video_features)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        visual_pos_masks: Bool[Array, "batch seq_len"] | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        cache_position: Array | None = None,
    ) -> BaseModelOutput:
        """Forward pass for multimodal base model without LM head.

        Orchestrates vision-language fusion by:
        1. Processing images/videos through vision encoder
        2. Merging visual embeddings into text token sequence
        3. Computing 3D mRoPE position IDs
        4. Running fused embeddings through text decoder with MoE

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Must contain
                special tokens (image_token_id, video_token_id) at positions where visual
                features should be inserted.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length, hidden_size).
                If provided, input_ids must still be provided for multimodal merging.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) where True=valid, False=padding.
            mask_info: Pre-computed mask information for attention. Auto-computed if None.
            position_ids: 3D position IDs of shape (3, batch_size, sequence_length) for mRoPE.
                Dimensions: [temporal, height, width]. Auto-computed from grid_thw if None.
            mode: Runtime mode (MODE_TRAIN/DECODE/EVAL). Auto-detected from context if None.
            past_key_values: Cached key-value states for autoregressive generation.
            cache_metadata: Metadata for paged attention (sequence lengths, block tables).
            output_attentions: Whether to return attention weights. Defaults to config.output_attentions.
            output_hidden_states: Whether to return all layer hidden states. Defaults to config.output_hidden_states.
            output_router_logits: Whether to return MoE router logits. Defaults to config.output_router_logits.
            visual_pos_masks: Boolean mask of shape (batch_size, sequence_length) indicating visual token
                positions. Auto-computed from pixel_values if not provided.
            deepstack_visual_embeds: List of visual embeddings for injection at each decoder layer.
                Auto-computed from pixel_values if not provided.
            pixel_values: Image pixel values of shape (num_images, channels, height, width).
            pixel_values_videos: Video pixel values of shape (num_videos, channels, frames, height, width).
            image_grid_thw: List of (temporal, height, width) tuples for each image grid.
            video_grid_thw: List of (temporal, height, width) tuples for each video grid.
            image_max_grid_size: Maximum grid size for adaptive image resolution.
            video_max_grid_size: Maximum grid size for adaptive video resolution.
            cache_position: Cache position indices for generation (internal use).

        Returns:
            Qwen3VLMoeModelOutputWithPast containing:
                - last_hidden_state: Final hidden states of shape (batch_size, sequence_length, hidden_size).
                - past_key_values: Updated cache for next generation step.
                - hidden_states: Tuple of all layer hidden states if output_hidden_states=True.
                - attentions: Tuple of attention weights if output_attentions=True.
                - rope_deltas: mRoPE position deltas of shape (batch_size, 1).
                - router_logits: Tuple of MoE router logits if output_router_logits=True.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        inputs_embeds, embedding_info = self.prepare_multimodal_embeddings(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        visual_pos_masks = embedding_info.visual_pos_masks
        deepstack_visual_embeds = embedding_info.deepstack_visual_embeds

        rope_deltas = None
        # Embeddings-only text input has no placeholder IDs to parse; let the
        # text decoder infer its three position axes from the effective mask.
        if position_ids is None and input_ids is not None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw if pixel_values is not None else None,
                video_grid_thw if pixel_values_videos is not None else None,
                attention_mask=attention_mask,
            )

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        router_logits = getattr(outputs, "router_logits", None) if output_router_logits else None
        return Qwen3VLMoeModelOutputWithPast(  # pyright: ignore[reportReturnType]
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
            router_logits=router_logits,
        )

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: Qwen3VLMoeModel does not have a language model head.
        """
        raise NotImplementedError("Qwen3VLMoeModel does not have a language model head.")

    def get_embedding(self):
        """Get the embedding layer.

        Returns:
            Embed: The token embedding layer from the language model.
        """
        return self.language_model.embed_tokens


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeForConditionalGeneration(BaseVisionLanguageModule[Qwen3VLMoeModel, Qwen3VLMoeConfig]):
    """Qwen3-VL-MoE model for conditional generation from images/videos and text.

    Inherits from BaseVisionLanguageModule to leverage common VLM infrastructure.

    Architecture matches HuggingFace:
    - model: Qwen3VLMoeModel (contains visual + language_model with MoE)
    - lm_head: Linear projection to vocab

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "qwen3_vl_moe" model identifier
        _supports_video: True (supports video input)
        _uses_mrope: True (uses multi-dimensional RoPE)
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen3_vl_moe"
    _config_class = Qwen3VLMoeConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen3VLMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL-MoE for conditional generation.

        Args:
            config (Qwen3VLMoeConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3VLMoeModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            image_token_index=config.image_token_id,
            video_token_index=config.video_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            router_aux_loss_coef=getattr(config.text_config, "router_aux_loss_coef", 0.001),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

    def get_input_embeddings(self):
        """Get the input embeddings layer.

        Returns:
            Embed: Token embedding layer.
        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the input embeddings layer.

        Args:
            value: New embedding layer to use.
        """
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        """Set the decoder component.

        Args:
            decoder: New decoder module to use.
        """
        self.model.set_decoder(decoder)

    def get_decoder(self):
        """Get the decoder component.

        Returns:
            Qwen3VLMoeTextModel: The language model decoder.
        """
        return self.model.get_decoder()

    @property
    def visual(self):
        """Property to access the vision transformer for backward compatibility.

        Returns:
            Qwen3VLMoeVisionTransformerPretrainedModel: The vision encoder.
        """
        return self.model.visual

    @property
    def language_model(self):
        """Property to access the language model for backward compatibility.

        Returns:
            Qwen3VLMoeTextModel: The text decoder.
        """
        return self.model.language_model

    def get_video_features(
        self,
        pixel_values_videos: Array,
        video_grid_thw: Array | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encode videos into continuous embeddings.

        Delegates to self.model.get_video_features.

        Args:
            pixel_values_videos: The tensors corresponding to the input videos.
            video_grid_thw: The temporal, height and width grid for each video.
            video_max_grid_size: Maximum grid size for videos.

        Returns:
            Tuple of (video_embeds_tuple, deepstack_video_embeds).
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(
        self,
        pixel_values: Array,
        image_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encode images into continuous embeddings.

        Delegates to self.model.get_image_features.

        Args:
            pixel_values: The tensors corresponding to the input images.
            image_grid_thw: The temporal, height and width grid for each image.
            image_max_grid_size: Maximum grid size for images.

        Returns:
            Tuple of (image_embeds_tuple, deepstack_image_embeds).
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

    def compute_embedding_with_info(
        self, input_ids, *args, deepstack_image_embeds=None, deepstack_video_embeds=None, **kwargs
    ):
        """Prepare embeddings and cached/native DeepStack metadata in one pass.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments forwarded to the multimodal model.
            deepstack_image_embeds: Optional cached per-layer image residuals.
            deepstack_video_embeds: Optional cached per-layer video residuals.
            **kwargs: Embedding, pixel, grid and attention inputs for the model.

        Returns:
            Fused embeddings and EmbeddingInfo, without a second vision encode.
        """
        return self.model.compute_embedding_with_info(
            input_ids,
            *args,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
            **kwargs,
        )

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute embeddings with optional multimodal fusion.

        Delegates to self.model.compute_embedding.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Array: Computed embeddings.
        """
        return self.model.compute_embedding(input_ids, *args, **kwargs)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        visual_pos_masks: Bool[Array, "batch seq_len"] | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        cache_position: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for vision-language generation with MoE.

        Processes multimodal inputs (text, images, videos) through the full model pipeline
        to generate next-token predictions. Handles vision encoding, multimodal fusion,
        3D RoPE positioning, MoE routing, and language modeling.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
                Special tokens: config.image_token_id for images, config.video_token_id for videos.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) where True indicates
                valid tokens and False indicates padding. Defaults to all-ones if not provided.
            mask_info: Pre-computed mask information for attention operations. If None, computed
                automatically from input_ids and attention_mask.
            position_ids: 3D position IDs for mRoPE of shape (3, batch_size, sequence_length).
                Dimensions are [temporal, height, width]. Auto-computed from grid_thw if None.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_EVAL). Auto-detected if None based on
                sequence_length and past_key_values presence.
            past_key_values: Cached key-value states from previous forward passes for autoregressive
                generation. Can be TransformerCache, RaggedPagesCache, or HybridCache.
            cache_metadata: Metadata for paged attention operations (sequence lengths, block tables).
            apply_lm_head: Whether to apply the language modeling head to get logits. Set to False
                if you only need hidden states. Defaults to True.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length, hidden_size).
                If provided, input_ids is ignored. Useful for custom embedding manipulation.
            output_attentions: Whether to return attention weights from all layers. Defaults to
                config.output_attentions.
            output_hidden_states: Whether to return hidden states from all layers. Defaults to
                config.output_hidden_states.
            output_router_logits: Whether to return MoE router logits for auxiliary loss computation.
                Defaults to config.text_config.output_router_logits.
            visual_pos_masks: Pre-computed boolean mask of shape (batch_size, sequence_length) indicating
                positions where visual tokens appear. Auto-computed if pixel_values provided.
            deepstack_visual_embeds: Pre-computed list of visual embeddings for deepstack injection at
                each decoder layer. Auto-computed if pixel_values provided.
            pixel_values: Image pixel values of shape (num_images, channels, height, width).
                Images are processed through vision encoder and merged at image_token_id positions.
            pixel_values_videos: Video pixel values of shape (num_videos, channels, frames, height, width).
                Videos are processed similarly to images but with temporal modeling.
            image_grid_thw: List of (temporal, height, width) tuples for each image specifying the
                grid structure after patch embedding. Required if pixel_values provided.
            video_grid_thw: List of (temporal, height, width) tuples for each video. Required if
                pixel_values_videos provided.
            cache_position: Position indices for cached states during generation. Used internally.
            image_max_grid_size: Maximum grid size for adaptive image resolution. If None, uses
                original resolution.
            video_max_grid_size: Maximum grid size for adaptive video resolution. If None, uses
                original resolution.
            **kwargs: Additional arguments (unused, for API compatibility).

        Returns:
            VLMCausalLMOutput containing:
                - logits: Language modeling logits of shape (batch_size, sequence_length, vocab_size)
                    if apply_lm_head=True, else None.
                - past_key_values: Updated cache for next generation step.
                - hidden_states: Tuple of hidden states from all layers if output_hidden_states=True.
                - last_hidden_state: Final hidden state of shape (batch_size, sequence_length, hidden_size).
                - attentions: Tuple of attention weights if output_attentions=True.
                - rope_deltas: mRoPE position deltas of shape (batch_size, 1) for tracking position offsets.
                - router_logits: Tuple of router logits from MoE layers if output_router_logits=True.
                - image_hidden_states: None (reserved for future use).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.get_text_config().output_router_logits
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(hidden_states)
            lm_logits = self.apply_logit_cap(lm_logits)

        aux_loss = self.compute_router_aux_loss(outputs)

        return Qwen3VLMoeCausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", None),
            image_hidden_states=None,
            router_logits=getattr(outputs, "router_logits", None) if output_router_logits else None,
            aux_loss=aux_loss,
        )

    def get_vision_tower(self) -> spx.Module:
        """Get the vision tower component.

        Returns:
            Qwen3VLMoeVisionTransformerPretrainedModel: The vision encoder.
        """
        return self.model.visual

    def get_language_model(self) -> spx.Module:
        """Get the language model component.

        Returns:
            Qwen3VLMoeTextModel: The text decoder.
        """
        return self.model.language_model
