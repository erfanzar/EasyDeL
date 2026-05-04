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

"""Spectrax implementation of Google's Gemma decoder-only language model.

Gemma is a family of lightweight open language models developed by Google DeepMind.
This module provides the building blocks (RMSNorm, attention, gated MLP, decoder
layer) and full models for base inference, causal language modeling, and sequence
classification.

Architectural traits:
    - Pre-normalization decoder transformer with RMSNorm (uses ``1 + weight`` scaling).
    - Grouped-query attention (GQA) with rotary position embeddings (RoPE).
    - Gated MLP with approximate GeLU activation (``gelu_pytorch_tanh``).
    - Embedding scaling by ``sqrt(hidden_size)`` applied to token embeddings.
    - Optional gradient checkpointing and pipeline-parallel layer staging.

Exports:
    - :class:`GemmaModel`: Backbone model returning hidden states.
    - :class:`GemmaForCausalLM`: Decoder LM with tied / weight-shared output head.
    - :class:`GemmaForSequenceClassification`: Pooled classifier head over the last token.
"""

import functools
import warnings

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
from easydel.layers.attention import UnifiedAttention
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .gemma_configuration import GemmaConfig as GemmaConfig

logger = get_logger(__name__)


class GemmaRMSNorm(spx.Module):
    """Gemma's Root Mean Square LayerNorm with the ``(1 + weight)`` scaling convention.

    Gemma's RMSNorm differs from the textbook formulation in two ways. First, the
    learnable scale parameter is initialised to **zero** and applied as
    ``(1 + weight) * hat_x`` instead of ``weight * hat_x``; this means a freshly
    initialised norm acts as the identity, which Google's reference implementation
    relies on for stable warm-up. Second, the variance is computed in float32
    regardless of the activation dtype to avoid catastrophic cancellation when
    running under bfloat16.

    Attributes:
        config: Source ``GemmaConfig``; only ``hidden_size`` and ``rms_norm_eps``
            are read at runtime.
        epsilon: Numerical stabiliser added inside the inverse-RMS square-root.
        dtype: Output / compute dtype the normalised activations are cast back to.
        weight: Learnable scale of shape ``(hidden_size,)`` initialised to zeros
            and consumed as ``1 + weight`` so the layer starts as the identity.
    """

    kernel_init = staticmethod(jax.nn.initializers.ones)

    def __init__(self, config: GemmaConfig, dtype: jnp.dtype = jnp.float32):
        """Build the per-feature scale and capture the epsilon from ``config``.

        The scale is registered as a bound :class:`ArrayParam` of shape
        ``(config.hidden_size,)`` initialised to ones-then-shifted-by-minus-one
        (i.e. ``init_method="ones"`` storage; the ``+1`` happens at apply time).
        ``dtype`` is only the cast-back dtype — the variance reduction itself
        always promotes to float32.
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
        """Normalise along the last axis and apply the ``(1 + weight)`` scale.

        Computes ``hat_x = x / sqrt(mean(x**2, axis=-1, keepdims=True) + eps)``
        in float32, then returns ``(1 + weight) * hat_x`` cast back to
        ``self.dtype``. The mean is taken over the hidden dimension only;
        batch and sequence dims are preserved.

        Args:
            hidden_states: Tensor of shape ``[batch, seq_len, hidden_dim]`` to
                be normalised. May be any dtype — the variance reduction is
                lifted to float32 internally.

        Returns:
            Tensor of the same shape as the input, with the last axis
            re-scaled to unit RMS and modulated by ``(1 + weight)``.
        """
        variance = hidden_states.astype(jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return (1 + self.weight.value.astype(self.dtype)) * jnp.asarray(hidden_states, dtype=self.dtype)


class GemmaAttention(UnifiedAttention):
    """Causal multi-head / grouped-query attention block for Gemma.

    Thin specialisation of :class:`UnifiedAttention` that fixes
    ``attention_type="standard"`` and ``causal=True``, so the underlying ejkernel
    backend dispatches to a dense causal kernel (or its GQA variant when
    ``num_key_value_heads < num_attention_heads``) with rotary position
    embeddings applied to Q and K before the softmax. Q/K/V projections are
    laid out via ColumnParallel/RowParallel linears in the parent class so the
    head axis is sharded along the model-parallel mesh axis.

    The :meth:`_create_rotary` override is what carries Gemma's full-head-dim
    rotary embedding (``rotary_dim == head_dim``) and the configurable
    ``rope_theta`` base; everything else is inherited.
    """

    def __init__(
        self,
        config: GemmaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Forward all parameters to :class:`UnifiedAttention` with Gemma defaults.

        Pins ``attention_type="standard"`` (dense softmax, no sliding window
        like Gemma2) and ``causal=True`` so prefill and decode share the same
        masked kernel. ``layer_idx`` is propagated so the parent can route the
        appropriate KV-cache view per layer.
        """
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
        )

    def _create_rotary(self, config: GemmaConfig, dtype: jnp.dtype):
        """Build the rotary-embedding helper for this layer's head geometry.

        Gemma applies RoPE across the full head dimension (``rotary_dim ==
        head_dim``) and uses ``config.rope_theta`` as the frequency base. The
        helper is the standard precomputed cos/sin table generator returned by
        :meth:`EasyDeLBaseConfig.get_basic_rope`.

        Args:
            config: Source config exposing ``rope_theta``.
            dtype: Compute dtype the cos/sin table is cast to (typically the
                attention dtype, not the parameter dtype).

        Returns:
            A rotary-embedding callable that, given ``positions``, yields the
            ``(cos, sin)`` tables consumed by the attention kernel.
        """
        return config.get_basic_rope(
            dtype=dtype,
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            base=config.rope_theta,
        )


class GemmaMLP(spx.Module):
    """Gated MLP (GeGLU) feedforward network for Gemma models.

    Implements a gated linear unit feedforward network:
    ``down_proj(act(gate_proj(x)) * up_proj(x))``. Uses approximate GeLU
    (gelu_pytorch_tanh) by default, matching Google's Gemma implementation.

    Attributes:
        config (GemmaConfig): Model configuration with MLP parameters.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Numerical precision for matrix operations.
        act: Activation function applied to the gate projection.
    """

    def __init__(
        self,
        config: GemmaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Construct the three projections (gate, up, down) and pin the activation.

        The gate and up projections are ColumnParallel (output-sharded across
        the model-parallel mesh) while the down projection is RowParallel
        (input-sharded), matching the canonical Megatron MLP layout. The
        activation is read from ``config.hidden_activation`` and falls back to
        ``"gelu_pytorch_tanh"`` with a warning if it is left unset, because
        Gemma's reference uses approximate GeLU rather than exact GeLU.
        ``layer_idx`` is accepted for API parity with Gemma2's per-layer MLP
        but is not used here.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim
        kernel_init = jax.nn.initializers.normal(config.initializer_range)

        if self.config.hidden_activation is None:
            warnings.warn(
                "Gemma's activation function should be approximate GeLU and not exact GeLU. "
                "Changing the activation function to `gelu_pytorch_tanh`."
                f"if you want to use the legacy `{self.config.hidden_act}`, "
                f"edit the `model.config` to set `hidden_activation={self.config.hidden_act}` ",
                stacklevel=1,
            )
            hidden_activation = "gelu_pytorch_tanh"
        else:
            hidden_activation = self.config.hidden_activation
        self.act = ACT2FN[hidden_activation]

        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        row_parallel_linear = functools.partial(
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


class GemmaDecoderLayer(spx.Module):
    """Single decoder layer for Gemma models.

    Implements a pre-normalization transformer decoder layer with residual connections:
    ``x + attn(norm(x))`` followed by ``x + mlp(norm(x))``. Uses RMSNorm for
    normalization and supports gradient checkpointing.

    Attributes:
        config (GemmaConfig): Model configuration.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Numerical precision for matrix operations.
        input_layernorm (GemmaRMSNorm): Pre-attention normalization.
        post_attention_layernorm (GemmaRMSNorm): Pre-MLP normalization.
        self_attn (GemmaAttention): Multi-head attention module.
        mlp (GemmaMLP): Feedforward network module.
    """

    def __init__(
        self,
        config: GemmaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Wire up the four sub-modules of one decoder block.

        Builds, in order: pre-attention RMSNorm, causal GQA attention,
        pre-MLP RMSNorm, gated GeGLU MLP. ``layer_idx`` is propagated to the
        attention sub-module so it can pick its KV-cache view; it has no
        effect on the MLP. The same ``rngs`` stream is forked to all
        sub-modules; init keys are derived deterministically from the parent
        seed.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        # Define layers
        self.input_layernorm = GemmaRMSNorm(self.config, dtype=self.dtype)
        self.post_attention_layernorm = GemmaRMSNorm(self.config, dtype=self.dtype)
        self.self_attn = GemmaAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = GemmaMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

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

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x))

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

        hidden_states = checkpoint_name(residual + attn_outputs.attention_output, "residual")

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = checkpoint_name(residual + feed_forward_hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        hidden_states = checkpoint_name(hidden_states, "layer_output")
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=GemmaConfig, model_type="gemma")
class GemmaModel(EasyDeLBaseModule):
    """Backbone Gemma transformer returning hidden states (no LM head).

    Implements the standard pre-norm decoder stack: ``embed -> [decoder_layer
    x num_hidden_layers] -> RMSNorm``. Token embeddings are scaled by
    ``sqrt(hidden_size)`` immediately before entering layer 0, matching
    Google's reference; this scale is what makes the otherwise-identity
    ``(1 + 0)`` RMSNorm at init produce bounded activations.

    The model supports two cache flavours via ``past_key_values`` —
    :class:`TransformerCache` for dense KV layouts and
    :class:`RaggedPagesCache` for paged (vLLM-style) attention used by the
    eSurge inference engine. Layers are stage-assigned for pipeline
    parallelism (see :meth:`assign_layer_stage`) and optionally
    ``ModuleList.stack`` ed when ``config.scan_layers`` is set and there is
    only a single pipeline stage.

    Attributes:
        config: Source ``GemmaConfig``.
        dtype: Activation/compute dtype (typically ``bfloat16``).
        param_dtype: Storage dtype of the parameter tree.
        precision: ``jax.lax.PrecisionLike`` forwarded into all matmuls.
        embed_tokens: Token embedding lookup of shape ``[vocab, hidden]``.
        layers: ``ModuleList`` (optionally stacked) of
            :class:`GemmaDecoderLayer`.
        norm: Final ``GemmaRMSNorm`` applied to the last hidden state.
    """

    def __init__(
        self,
        config: GemmaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build embeddings, ``num_hidden_layers`` decoder blocks, and final norm.

        Each decoder block is wrapped through :func:`auto_remat` with the
        gradient-checkpointing policy from
        ``config.gradient_checkpointing`` and the named save/exclude points
        from ``config.gradient_checkpointing_targets`` so backward pass can
        recompute the per-block activations selectively. When
        ``config.scan_layers`` is true *and* there is only one pipeline stage,
        the ``ModuleList`` is collapsed via ``stack()`` so the forward becomes
        a single ``lax.scan`` over a parameter-stacked block.
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
            GemmaDecoderLayer,
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
            self.norm = GemmaRMSNorm(self.config, dtype=self.dtype)

    # Ignore copy
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
        """Forward pass through the Gemma base model.

        Processes input tokens through embedding, all decoder layers, and final normalization.
        Embeddings are scaled by sqrt(hidden_size) for training stability.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask of shape (batch_size, sequence_length)
                to avoid attention on padding tokens. Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token in the sequence,
                shape (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Defaults to None (auto-detected).
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache containing precomputed key-value states for fast generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for managing the cache. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional all hidden_states,
                optional attention weights, and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        _batch_size, sequence_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
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


@register_module(TaskType.CAUSAL_LM, config=GemmaConfig, model_type="gemma")
class GemmaForCausalLM(BaseCausalLMModule[GemmaModel, GemmaConfig]):
    """Causal LM wrapper: ``GemmaModel`` + biasless tied LM head.

    Subclass of :class:`BaseCausalLMModule` that supplies the backbone class
    (:class:`GemmaModel`), the attribute name under which the backbone is
    stored (``"model"``), and ``lm_head_bias=False``. Because Gemma defaults
    ``tie_word_embeddings=True`` in :class:`GemmaConfig`, the LM head shares
    weights with ``embed_tokens`` and the unembedding logits are computed via
    ``compute_lm_logits`` in :class:`BaseCausalLMModule`.

    Attributes:
        config: Source ``GemmaConfig`` (read by the parent for vocab size and
            tie-embedding flag).
        dtype: Activation dtype (defaults to ``bfloat16``).
        param_dtype: Parameter dtype (defaults to ``bfloat16``).
        precision: ``jax.lax.PrecisionLike`` forwarded to all matmuls.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gemma"
    _config_class = GemmaConfig

    def __init__(
        self,
        config: GemmaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Delegate to :class:`BaseCausalLMModule` with Gemma-specific defaults.

        Pins ``base_model_class=GemmaModel``, ``base_model_name="model"`` (so
        the backbone is reachable as ``self.model``) and ``lm_head_bias=False``
        — Gemma never trains a bias on the unembedding projection.
        """
        super().__init__(
            config=config,
            base_model_class=GemmaModel,
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
        """Forward pass through the Gemma causal language model.

        Runs the base model and optionally applies the language modeling head to produce
        token logits for next-token prediction.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens.
                Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention. Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode for optimizations. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache management metadata. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language modeling head. Defaults to True.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states. Defaults to None.

        Returns:
            CausalLMOutput: Contains logits (if apply_lm_head=True), hidden states, attentions,
                and updated cache.
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

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(hidden_states)

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


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=GemmaConfig, model_type="gemma")
class GemmaForSequenceClassification(BaseSequenceClassificationModule[GemmaModel, GemmaConfig]):
    """Sequence-classification wrapper: ``GemmaModel`` + linear score head.

    Adds a single ``Linear(hidden_size -> num_labels)`` classification head on
    top of :class:`GemmaModel`. The head is fed the *last non-pad* token's
    hidden state (``pooling_strategy="last"``) — i.e., the wrapper finds the
    final position before ``config.pad_token_id`` and gathers from
    ``last_hidden_state`` at that index. With no pad token configured, the
    forward asserts ``batch == 1`` and uses position ``-1`` directly.

    Attributes:
        config: Source ``GemmaConfig``; ``num_labels`` and ``pad_token_id``
            drive the head shape and pooling logic respectively.
        dtype: Activation dtype.
        param_dtype: Parameter dtype.
        precision: ``jax.lax.PrecisionLike`` for all matmuls.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "gemma"
    _config_class = GemmaConfig

    def __init__(
        self,
        config: GemmaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Delegate to :class:`BaseSequenceClassificationModule` with last-token pooling.

        Pins ``base_model_class=GemmaModel``, ``base_model_name="model"``,
        ``pooling_strategy="last"`` (last non-pad token) and
        ``score_head_bias=False``. The classification head is created as a
        plain ``Linear`` of shape ``(hidden_size, config.num_labels)`` inside
        the parent constructor.
        """
        super().__init__(
            config=config,
            base_model_class=GemmaModel,
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
        """Forward pass through the Gemma sequence classification model.

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
