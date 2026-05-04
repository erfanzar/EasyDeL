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

"""Spectrax implementation of Google's Gemma2 decoder-only language model.

Gemma2 builds on Gemma with a hybrid attention pattern, additional normalization,
and logit soft-capping for improved training stability. This module exposes the
core building blocks (RMSNorm, hybrid sliding/full attention, gated MLP, decoder
layer with pre/post-norm) and the full models for inference, causal language
modeling, and sequence classification.

Architectural traits:
    - Hybrid attention: alternates between sliding-window attention on odd layers
      (default window 4096) and full attention on even layers.
    - Pre- and post-normalization (RMSNorm with ``1 + weight`` scaling) around
      both the attention and feedforward sub-blocks.
    - Custom query pre-attention scalar (default 224) replaces the standard
      ``head_dim ** -0.5`` softmax scale; optional attention-logit soft-capping.
    - Final logit soft-capping ``cap * tanh(logits / cap)`` (default cap 30.0).
    - Embedding scaling by ``sqrt(hidden_size)`` and RoPE positional encoding.

Exports:
    - :class:`Gemma2Model`: Backbone returning hidden states.
    - :class:`Gemma2ForCausalLM`: Decoder LM with soft-capped output logits.
    - :class:`Gemma2ForSequenceClassification`: Pooled classifier over the last token.
"""

from functools import partial

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
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
    CausalLMOutput,
    DecoderLayerOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat, block_wise_ffn
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .gemma2_configuration import Gemma2Config

logger = get_logger(__name__)


class Gemma2RMSNorm(spx.Module):
    """Gemma2's RMSNorm with the ``(1 + weight)`` scaling and fp32 variance reduction.

    Identical formulation to Gemma's RMSNorm (zero-init weight applied as
    ``1 + weight`` so the layer is the identity at step 0; variance is computed
    in float32 even when activations are bfloat16 to avoid precision loss in
    the inverse-square-root). Gemma2 uses two of these per block — one before
    attention and one before the MLP — plus optional post-attention and
    post-FFN copies depending on the layer-norm placement schema.

    Attributes:
        config: Source ``Gemma2Config``; reads ``hidden_size`` and ``rms_norm_eps``.
        epsilon: Floor under the inverse-RMS to keep the gradient well-defined.
        dtype: Cast-back dtype for the normalised activations.
        weight: Bound :class:`ArrayParam` of shape ``(hidden_size,)`` initialised
            to ones (consumed as ``1 + weight``).
    """

    kernel_init = staticmethod(jax.nn.initializers.ones)

    def __init__(self, config: Gemma2Config, dtype: jnp.dtype = jnp.float32):
        """Allocate the per-feature scale and capture epsilon from ``config``.

        ``dtype`` controls the cast-back of the normalised tensor only — the
        variance reduction itself is forced to float32 inside :meth:`forward`.
        """
        self.config = config
        self.epsilon = self.config.rms_norm_eps
        self.dtype = dtype
        self.weight = ArrayParam.bound(
            shape=(self.config.hidden_size,),
            dtype=dtype,
            init_method="ones",
            key=None,
        )

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Compute ``(1 + weight) * x / sqrt(mean(x**2, axis=-1) + eps)``.

        The mean of squared activations is taken in float32 over the trailing
        hidden dimension, the inverse RMS scale is then applied to the
        original tensor, and finally the learnable scale ``(1 + weight)``
        modulates the output. The cast back to ``self.dtype`` happens at the
        very end so the multiplication runs at full precision.

        Args:
            hidden_states: Tensor of shape ``[batch, seq_len, hidden_dim]``.

        Returns:
            Tensor of the same shape, with each ``hidden_dim`` slice having
            unit RMS prior to the learned ``(1 + weight)`` rescaling.
        """
        variance = hidden_states.astype(jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return (1 + self.weight.value.astype(self.dtype)) * jnp.asarray(hidden_states, dtype=self.dtype)


class Gemma2Attention(UnifiedAttention):
    """Gemma2 attention with alternating sliding-window / full layers and custom softmax scale.

    Gemma2 alternates two attention regimes layer-by-layer, controlled by
    ``config.layer_types[layer_idx]``: a sliding-window mode that bounds each
    query's receptive field to ``config.sliding_window`` past tokens (typically
    set on odd layers in the published checkpoints) and a full-causal mode
    elsewhere. The window is forwarded into :class:`UnifiedAttention` so the
    underlying ejkernel kernel masks tokens beyond the window in addition to
    the causal mask.

    Two further departures from vanilla MHA are introduced here:

    1. The softmax scale is **not** the usual ``1 / sqrt(head_dim)``. Gemma2
       uses ``1 / sqrt(query_pre_attn_scalar)`` (see
       :meth:`_create_attention_performer`), which decouples the temperature
       from the head dimension and stabilises training under the wider hidden
       dimensions used in the larger Gemma2 sizes.
    2. Softmax is forced to float32 via ``attention_softmax_in_fp32`` whenever
       the activation dtype is not already float32 — Gemma2's logit
       distribution can be sharp enough that bfloat16 softmax loses precision.

    Attributes:
        config: Source ``Gemma2Config``.
        head_dim: Per-head dimensionality.
        attention_softmax_in_fp32: ``True`` whenever the activation dtype is
            not float32.
        is_cross_attention: Reserved flag for cross-attention layers (always
            ``False`` in the public Gemma2 checkpoints).
    """

    def __init__(
        self,
        config: Gemma2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: spx.Rngs,
    ):
        """Resolve the per-layer attention regime and forward to :class:`UnifiedAttention`.

        Reads ``config.layer_types[layer_idx]``: if it equals
        ``"sliding_attention"`` the constructor passes
        ``sliding_window=config.sliding_window`` down to the unified attention
        backend; otherwise ``sliding_window=None`` (full causal). Sets
        ``attention_softmax_in_fp32`` based on ``dtype`` and stores
        ``is_cross_attention`` for later cross-attention specialisation.
        """
        # Set layer-specific attributes before super().__init__
        self.is_cross_attention = is_cross_attention

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=causal,
            sliding_window=(
                config.sliding_window
                if config.layer_types is not None and config.layer_types[layer_idx] == "sliding_attention"
                else None
            ),
        )

        # Gemma2-specific attributes
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

    def _create_rotary(self, config: Gemma2Config, dtype: jnp.dtype):
        """Build Gemma2's full-head-dim rotary embedding helper.

        Gemma2 applies RoPE across the full head dimension
        (``rotary_dim == head_dim``). The fourth positional argument
        (``True``) toggles the legacy half-rotation interleaving that matches
        Google's reference implementation. The returned helper precomputes
        cos/sin tables for use inside the attention kernel.
        """
        return config.get_basic_rope(dtype, self.head_dim, self.head_dim, True)

    def _create_attention_performer(self, config: Gemma2Config, rngs: spx.Rngs):
        """Build the flexible attention performer with Gemma2's custom temperature.

        Replaces the default ``1/sqrt(head_dim)`` softmax scale with
        ``1/sqrt(config.query_pre_attn_scalar)``. This is the key knob that
        lets Gemma2's larger sizes share head dimensions with smaller variants
        without rebalancing the attention temperature. ``dropout_prob`` is
        wired through from ``config.attention_dropout``.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=config.query_pre_attn_scalar**-0.5,
            dropout_prob=config.attention_dropout,
        )

    def _merge_heads(self, hidden_states):
        """Collapse the per-head axis back into the contiguous hidden dimension.

        Folds the trailing ``(num_heads, head_dim)`` axes of an attention
        output back into a single ``hidden_dim = num_heads * head_dim`` axis,
        ready for the output projection.

        Args:
            hidden_states: Array of shape ``[batch, seq_len, num_heads, head_dim]``.

        Returns:
            Array of shape ``[batch, seq_len, num_heads * head_dim]``.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads * self.head_dim))

    def _split_heads(self, hidden_states, num_heads):
        """Reshape a flat hidden tensor into ``[..., num_heads, head_dim]``.

        Inverse of :meth:`_merge_heads`. Used after the Q/K/V linear
        projections to expose the per-head axis to the attention kernel.

        Args:
            hidden_states: Array of shape ``[batch, seq_len, num_heads * head_dim]``.
            num_heads: Number of heads to split into. The per-head dimension
                ``head_dim`` is read from ``self.head_dim``.

        Returns:
            Array of shape ``[batch, seq_len, num_heads, head_dim]``.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], num_heads, self.head_dim))


class Gemma2MLP(spx.Module):
    """Gated MLP (GeGLU) feedforward network for Gemma2 models.

    Implements the gated linear unit feedforward network:
    ``down_proj(act(gate_proj(x)) * up_proj(x))``. Uses approximate GeLU
    (gelu_pytorch_tanh) by default, matching Google's Gemma2 implementation.

    Attributes:
        config (Gemma2Config): Model configuration.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Numerical precision for matrix operations.
        act: Activation function applied to the gate projection.
    """

    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Gemma2 MLP block.

        Args:
            config (Gemma2Config): Model configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision for
                matrix operations. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim
        kernel_init = jax.nn.initializers.normal(config.initializer_range)

        self.act = ACT2FN[self.config.hidden_activation]

        column_parallel_linear = partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(
            embed_dim,
            inner_dim,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            inner_dim,
            embed_dim,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            embed_dim,
            inner_dim,
            rngs=rngs,
        )

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass through the MLP block.

        Applies gated linear units with activation function: down_proj(act(gate_proj(x)) * up_proj(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            Array: Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = checkpoint_name(self.act(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Gemma2DecoderLayer(spx.Module):
    """Single decoder layer for Gemma2 models with post-norm architecture.

    Implements a transformer decoder layer with both pre- and post-normalization:
    ``x + post_attn_norm(attn(pre_attn_norm(x)))`` followed by
    ``x + post_ff_norm(mlp(pre_ff_norm(x)))``. The additional post-normalization
    layers improve training stability compared to standard pre-norm.

    Attributes:
        config (Gemma2Config): Model configuration.
        layer_idx (int): Index of this layer; determines sliding vs full attention.
        is_sliding (bool): Whether this layer uses sliding window attention.
        input_layernorm (Gemma2RMSNorm): Pre-attention normalization.
        post_attention_layernorm (Gemma2RMSNorm): Post-attention normalization.
        pre_feedforward_layernorm (Gemma2RMSNorm): Pre-MLP normalization.
        post_feedforward_layernorm (Gemma2RMSNorm): Post-MLP normalization.
        self_attn (Gemma2Attention): Multi-head attention module.
        mlp (Gemma2MLP): Feedforward network module.
    """

    def __init__(
        self,
        config: Gemma2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Gemma2 decoder layer.

        Args:
            config (Gemma2Config): Model configuration.
            layer_idx (int): Index of this layer in the model, determines attention type
                (sliding window on odd layers, full attention on even layers).
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision for
                matrix operations. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.is_sliding = bool(self.layer_idx % 2)
        self.self_attn = Gemma2Attention(
            self.config,
            layer_idx=self.layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = Gemma2MLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.input_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
        self.post_attention_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
        self.post_feedforward_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture with additional post-attention and post-feedforward
        normalization: x + post_attn_norm(attn(norm(x))) followed by x + post_ff_norm(mlp(pre_ff_norm(x)))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for each token, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): View into the
                key-value cache for this layer. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache operations. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, optional attention weights, and updated cache view.
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
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )

        hidden_states = self.post_attention_layernorm(attn_outputs.attention_output)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")
        hidden_states = checkpoint_name(hidden_states, "layer_output")
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Gemma2Config, model_type="gemma2")
class Gemma2Model(EasyDeLBaseModule):
    """Gemma2 model implementation.

    This implements the Gemma2 language model architecture, utilizing transformer blocks
    with RMSNorm, rotary position embeddings, and a hybrid attention mechanism that
    alternates between sliding window attention and full attention across layers.
    Gemma2 uses embedding scaling by sqrt(hidden_size) for training stability.

    Attributes:
        config (Gemma2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Gemma2 base model.

        Args:
            config (Gemma2Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.hidden_size = self.config.hidden_size

        self.embed_tokens = Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        remat_layer_block = auto_remat(
            Gemma2DecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for i in range(self.config.num_hidden_layers):
            with self.assign_layer_stage(i, total_layers=self.config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        self.config,
                        layer_idx=i,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        if self.config.scan_layers and self._pipeline_stage_count() == 1:
            self.layers = self.layers.stack()
        final_layer_idx = max(0, self.config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=self.config.num_hidden_layers):
            self.norm = Gemma2RMSNorm(self.config, dtype=self.dtype)

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
        trace: bool = False,
    ) -> BaseModelOutput:
        """Forward pass through the Gemma2 base model.

        Processes input tokens through token embedding, applies scaling, and passes them through
        the decoder layers with alternating sliding window and full attention patterns.

        Args:
            input_ids (Int[Array, "batch seq_len"], optional): Input token IDs.
                Shape: (batch_size, sequence_length). Must be provided if inputs_embeds is None.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"], optional): Pre-computed
                input embeddings. Shape: (batch_size, sequence_length, hidden_size).
                Must be provided if input_ids is None.
            attention_mask (Bool[Array, "batch seq_len"], optional): Attention mask indicating
                which tokens should be attended to (True) and which should be ignored (False).
                Shape: (batch_size, sequence_length). Default: None (all tokens attended).
            mask_info (MaskInfo, optional): Pre-computed mask information encoding attention
                patterns and positions. If None, computed from attention_mask or input_ids.
            position_ids (Int[Array, "batch seq_len"], optional): Position indices for each token.
                Shape: (batch_size, sequence_length). If None, uses sequential positions.
            mode (RUNTIME_MODE_TYPES, optional): Execution mode controlling attention computation.
                Options: MODE_TRAIN (full attention), MODE_DECODE (single-token), MODE_PREFILL.
                Auto-detected if None based on sequence length and cache presence.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key-value states from previous forward passes for efficient autoregressive
                generation. Hybrid cache recommended for Gemma2's mixed attention pattern.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for managing paged attention caches in serving scenarios.
            output_attentions (bool, optional): Whether to return attention weights from all layers.
                Default: None (uses config value).
            output_hidden_states (bool, optional): Whether to return hidden states from all layers.
                Default: None (uses config value).

        Returns:
            BaseModelOutput: Model outputs containing:
                - last_hidden_state (jnp.ndarray): Final layer hidden states after RMSNorm.
                  Shape: (batch_size, sequence_length, hidden_size).
                - hidden_states (tuple[jnp.ndarray], optional): Hidden states from each layer
                  if output_hidden_states=True. Each tensor has shape
                  (batch_size, sequence_length, hidden_size).
                - attentions (tuple[jnp.ndarray], optional): Attention weights from each layer
                  if output_attentions=True. Each tensor has shape
                  (batch_size, num_heads, sequence_length, key_value_length).
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache): Updated
                  cache with new key-value states for all layers.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None or both are provided.
            AssertionError: If sequence_length exceeds max_position_embeddings.

        Note:
            Gemma2 applies sqrt(hidden_size) scaling to input embeddings before processing.
            The model alternates between sliding window attention (4096 tokens) on odd layers
            and full attention on even layers for efficiency.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        sequence_length = inputs_embeds.shape[1]

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids
        inputs_embeds = inputs_embeds * (self.config.hidden_size**0.5)
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

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
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        frequencies = self.config.get_basic_frequencies()

        trace_layers = self._layer_scan_trace(
            trace,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache,
        )
        cache_views = views if trace_layers else None

        def _run_layer(block, carry):
            hs, cv, ah, aa, idx = carry
            if output_hidden_states:
                ah = (*ah, hs)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hs,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=frequencies,
                )
            hs = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)
            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_attentions:
                aa = (*aa, layer_outputs.attention_weight)
            return hs, cv, ah, aa, idx + 1

        init_carry = (hidden_states, cache_views, all_hidden_states, all_attentions, 0)
        hidden_states, _, all_hidden_states, all_attentions, _ = self.layers.scan(
            _run_layer,
            init_carry,
            trace=trace_layers,
        )
        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

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


@register_module(TaskType.CAUSAL_LM, config=Gemma2Config, model_type="gemma2")
class Gemma2ForCausalLM(BaseCausalLMModule[Gemma2Model, Gemma2Config]):
    """Gemma2 model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with causal attention masks
    applied to perform autoregressive language generation. Gemma2 includes final
    logit softcapping for improved training stability.

    Attributes:
        config (Gemma2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gemma2"
    _config_class = Gemma2Config

    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Gemma2 model for causal language modeling.

        Args:
            config (Gemma2Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Gemma2Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
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
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass of the Gemma2 causal language model.

        Processes input tokens through the base Gemma2 model and applies the language modeling
        head to produce next-token prediction logits with optional soft-capping.

        Args:
            input_ids (Int[Array, "batch seq_len"], optional): Input token IDs.
                Shape: (batch_size, sequence_length). Must be provided if inputs_embeds is None.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"], optional): Pre-computed
                input embeddings. Shape: (batch_size, sequence_length, hidden_size).
                Must be provided if input_ids is None.
            attention_mask (Bool[Array, "batch seq_len"], optional): Attention mask indicating
                which tokens should be attended to. Shape: (batch_size, sequence_length).
                Default: None (all tokens attended).
            mask_info (MaskInfo, optional): Pre-computed mask information. If None, computed
                from attention_mask or input_ids.
            position_ids (Int[Array, "batch seq_len"], optional): Position indices for each token.
                Shape: (batch_size, sequence_length). If None, uses sequential positions.
            mode (RUNTIME_MODE_TYPES, optional): Execution mode (MODE_TRAIN, MODE_DECODE, MODE_PREFILL).
                Auto-detected if None based on sequence length and cache.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key-value states for efficient autoregressive generation. HybridCache
                is recommended for Gemma2's mixed attention patterns.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for paged attention cache management.
            apply_lm_head (bool): Whether to apply the language modeling head to produce logits.
                Set to False to only get hidden states. Default: True.
            output_attentions (bool, optional): Whether to return attention weights from all layers.
                Default: None (uses config value).
            output_hidden_states (bool, optional): Whether to return hidden states from all layers.
                Default: None (uses config value).

        Returns:
            CausalLMOutput: Model outputs containing:
                - logits (jnp.ndarray, optional): Next-token prediction logits if apply_lm_head=True.
                  Shape: (batch_size, sequence_length, vocab_size). Soft-capping is applied if
                  final_logit_softcapping is configured (logits = tanh(logits / cap) * cap).
                - hidden_states (tuple[jnp.ndarray], optional): Hidden states from each layer
                  if output_hidden_states=True.
                - attentions (tuple[jnp.ndarray], optional): Attention weights from each layer
                  if output_attentions=True.
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache): Updated
                  cache with new key-value states.

        Note:
            Gemma2 applies final_logit_softcapping (default: 30.0) via tanh to prevent extreme
            logit values and improve training stability. The formula is:
            logits = tanh(logits / final_logit_softcapping) * final_logit_softcapping
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(self.prepare_lm_head_inputs(outputs.last_hidden_state))

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

    def compute_lm_logits(self, hidden_states: Array) -> Array:
        """Project hidden states to vocabulary logits with optional soft-capping.

        Calls the base LM-head projection, then applies Gemma-2's logit
        soft-capping when ``config.final_logit_softcapping`` is set:
        ``cap * tanh(logits / cap)``, which smoothly bounds logit
        magnitudes to ``[-cap, cap]``.

        Args:
            hidden_states: Hidden representations, shape ``[B, T, H]``.

        Returns:
            Logits with shape ``[B, T, V]``, optionally soft-capped.
        """
        lm_logits = super().compute_lm_logits(hidden_states)
        if self.config.final_logit_softcapping is not None:
            cap = jnp.array(self.config.final_logit_softcapping, dtype=lm_logits.dtype)
            lm_logits = cap * jax.nn.tanh(lm_logits / cap)
        return lm_logits

    def make_lm_head_fn(self):
        """Trace-safe projection with Gemma-2 soft-capping."""
        base_fn = super().make_lm_head_fn()
        cap_value = self.config.final_logit_softcapping
        if cap_value is None:
            return base_fn

        def _project(hidden_states):
            logits = base_fn(hidden_states)
            cap = jnp.array(cap_value, dtype=logits.dtype)
            return cap * jax.nn.tanh(logits / cap)

        return _project

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


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Gemma2Config, model_type="gemma2")
class Gemma2ForSequenceClassification(BaseSequenceClassificationModule[Gemma2Model, Gemma2Config]):
    """Gemma2 model for sequence classification tasks.

    This class extends the base Gemma2 model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
        config (Gemma2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "gemma2"
    _config_class = Gemma2Config

    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Gemma2 model for sequence classification.

        Args:
            config (Gemma2Config): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Gemma2Model,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the Gemma2 sequence classification model.

        Runs the base model and applies a classification head to the last token's hidden state
        to produce class logits.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings. Defaults to None.
            attention_mask (Array | None, optional): Mask to avoid attention on padding tokens.
                Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information. Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states. Defaults to None.

        Returns:
            SequenceClassifierOutput: Contains classification logits, hidden states, and attentions.

        Raises:
            ValueError: If batch size > 1 and no padding token is defined.
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
        return self.model

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
