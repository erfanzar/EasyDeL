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

"""OLMo (Open Language Model) implementation.

Implements AI2's OLMo decoder family — a LLaMA-like architecture with
non-parametric LayerNorm (no learnable affine), SwiGLU MLPs, RoPE, optional
grouped-query attention, and optional QKV clipping (``clip_qkv``).

Exports:
    - ``OlmoMLP``: SwiGLU feed-forward block.
    - ``OlmoAttention``: standard attention with optional QKV clipping.
    - ``OlmoDecoderLayer``: a single transformer block.
    - ``OlmoModel``: base transformer trunk.
    - ``OlmoForCausalLM``: causal LM head wrapper.
    - ``OlmoForSequenceClassification``: classification head wrapper.
"""

import functools

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
from easydel.infra.modeling_outputs import BaseModelOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers.attention import UnifiedAttention
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .olmo_configuration import OlmoConfig


class OlmoMLP(spx.Module):
    """SwiGLU FFN for OLMo decoder layers.

    Same gate / up / down SwiGLU shape as LLaMA's MLP, but with biases
    forced off on every linear (consistent with OLMo-1's "no learnable
    affine" stance — see :class:`OlmoDecoderLayer`'s ``LayerNorm``
    instantiation that also drops both bias and scale).

    Attributes:
        gate_proj, up_proj (ColumnParallelLinear): Bias-free expansion
            projections, ``hidden_size -> intermediate_size``.
        down_proj (RowParallelLinear): Bias-free contraction
            projection, ``intermediate_size -> hidden_size``.
        act_fn (Callable): SiLU (or whatever ``hidden_act`` selects).
    """

    def __init__(
        self,
        config: OlmoConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize OLMo MLP block.

        Args:
            config (OlmoConfig): Model configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim]
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class OlmoAttention(UnifiedAttention):
    """OLMo attention: standard MHA + RoPE + optional QKV clipping.

    OLMo-1's distinguishing trick is **QKV clipping** — symmetric clamp of
    the Q, K, V projection outputs to ``[-clip_qkv, clip_qkv]`` to bound
    the dynamic range that flows into the softmax. This is OLMo-1's
    answer to the activation outliers that LLaMA-style training without
    QK-norm sometimes accumulates. Clipping is implemented in
    :meth:`_preprocess_qkv` *before* the head reshape so the clamp is
    applied to the raw projections; it is a no-op when ``config.clip_qkv``
    is ``None``.

    Inherits the standard ``"standard"`` attention path from
    :class:`UnifiedAttention` for everything else (causal mask, RoPE,
    GQA support).

    Attributes:
        clip_qkv (float | None): Clip threshold mirrored from
            ``config.clip_qkv``; absent on most checkpoints.
    """

    def __init__(
        self,
        config: OlmoConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize OLMo attention layer with grouped-query attention support.

        Args:
            config (OlmoConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        self.clip_qkv = getattr(config, "clip_qkv", None)
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
        )

    def _preprocess_qkv(
        self,
        query_states: jnp.ndarray,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply optional clipping before reshaping QKV tensors.

        Clips query, key, and value states to [-clip_qkv, clip_qkv] if configured.
        This helps with training stability in OLMo models.

        Args:
            query_states: Query projections before reshaping.
            key_states: Key projections before reshaping.
            value_states: Value projections before reshaping.

        Returns:
            Tuple of (query_states, key_states, value_states) with optional clipping applied.
        """
        query_states, key_states, value_states = super()._preprocess_qkv(query_states, key_states, value_states)
        if self.clip_qkv is not None:
            clip_val = self.clip_qkv
            query_states = jnp.clip(query_states, -clip_val, clip_val)
            key_states = jnp.clip(key_states, -clip_val, clip_val)
            value_states = jnp.clip(value_states, -clip_val, clip_val)
        return query_states, key_states, value_states


class OlmoDecoderLayer(spx.Module):
    """One OLMo-1 block: pre-norm attention + SwiGLU FFN with parameter-free LayerNorm.

    OLMo-1's defining choice at the block level is the **non-parametric
    LayerNorm** — both ``input_layernorm`` and ``post_attention_layernorm``
    are constructed with ``use_bias=False, use_scale=False``, so they
    perform plain ``(x - mean(x)) / sqrt(var(x) + eps)`` with neither a
    learnable affine nor a bias. The intuition (per the OLMo-1 report) is
    that the redundancy with the surrounding linear layers' biases hurt
    training stability more than the affine helped representational
    capacity. Pre-norm residual layout otherwise follows LLaMA::

        x = x + self_attn(input_layernorm(x))
        x = x + mlp(post_attention_layernorm(x))

    Attributes:
        self_attn (OlmoAttention): Optionally-clipped MHA / GQA attention.
        mlp (OlmoMLP): Bias-free SwiGLU FFN.
        input_layernorm, post_attention_layernorm (LayerNorm): Parameter-free
            LayerNorms (no scale, no bias).
    """

    def __init__(
        self,
        config: OlmoConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize OLMo decoder layer.

        Args:
            config (OlmoConfig): Model configuration.
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

        self.self_attn = OlmoAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = OlmoMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.input_layernorm = LayerNorm(
            config.hidden_size,
            epsilon=1e-5,
            use_bias=False,
            use_scale=False,
            rngs=rngs,
        )
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size,
            epsilon=1e-5,
            use_bias=False,
            use_scale=False,
            rngs=rngs,
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
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, and cache view.
        """
        residual = hidden_states
        attention_output = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )

        hidden_states = checkpoint_name(attention_output.attention_output + residual, "residual")
        ffd_inp = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(self.mlp, ffd_inp, self.config.scan_mlp_chunk_size)
        else:
            feed_forward_hidden_states = self.mlp(ffd_inp)

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "layer_output")
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attention_output.attention_weight,
            cache_view=attention_output.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=OlmoConfig, model_type="olmo")
class OlmoModel(EasyDeLBaseModule):
    """OLMo-1 base trunk: token embeddings + decoder stack + parameter-free final LayerNorm.

    Implements AI2's OLMo-1 architecture — a LLaMA-style decoder where
    every LayerNorm is non-parametric (no scale, no bias), every linear is
    bias-free, and attention optionally clips Q/K/V to bound activation
    outliers. Unlike OLMo-2 there is no Q/K-norm and the residual layout
    is the standard pre-norm.

    Attributes:
        embed_tokens (Embed): Token embedding ``(vocab_size, hidden_size)``.
        layers (nn.ModuleList[OlmoDecoderLayer]): Decoder blocks assigned
            to pipeline stages via :func:`spx.assign_stage`; stacked into a
            single scanned op when ``config.scan_layers`` is set.
        norm (LayerNorm): Parameter-free final LayerNorm.
    """

    def __init__(
        self,
        config: OlmoConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize OLMo base model.

        Args:
            config (OlmoConfig): Model configuration.
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
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        remat_layer_block = auto_remat(
            OlmoDecoderLayer,
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
        if self.config.scan_layers and self._pipeline_stage_count() == 1:
            self.layers = self.layers.stack()
        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = LayerNorm(
                config.hidden_size,
                epsilon=1e-5,
                use_bias=False,
                use_scale=False,
                rngs=rngs,
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
        trace: bool = False,
    ) -> BaseModelOutput:
        """Forward pass through the OLMo base model.

        Processes input tokens through embedding, all decoder layers with RoPE and LayerNorm,
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
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
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
        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

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


@register_module(TaskType.CAUSAL_LM, config=OlmoConfig, model_type="olmo")
class OlmoForCausalLM(BaseCausalLMModule[OlmoModel, OlmoConfig]):
    """Causal LM head wrapper around :class:`OlmoModel` for next-token prediction.

    Adds a bias-free LM projection and the standard ``compute_lm_logits``
    helper from :class:`BaseCausalLMModule`. Tied embeddings are governed
    by ``config.tie_word_embeddings`` (default false on OLMo-1, matching
    upstream).
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "olmo"
    _config_class = OlmoConfig

    def __init__(
        self,
        config: OlmoConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize OLMo model for causal language modeling.

        Args:
            config (OlmoConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=OlmoModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=OlmoConfig, model_type="olmo")
class OlmoForSequenceClassification(BaseSequenceClassificationModule[OlmoModel, OlmoConfig]):
    """Sequence classification head on top of :class:`OlmoModel`.

    Pools the trunk's last hidden state at the position of the last
    non-padded token (``pooling_strategy="last"``) and feeds it into a
    bias-free linear ``score`` projecting to ``config.num_labels``. The
    pooling step requires ``config.pad_token_id`` to be set when
    ``batch_size > 1`` so the pooled token can be located unambiguously.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "olmo"
    _config_class = OlmoConfig

    def __init__(
        self,
        config: OlmoConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize OLMo model for sequence classification.

        Args:
            config (OlmoConfig): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=OlmoModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy="last",
            score_head_bias=False,
        )
