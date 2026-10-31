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

"""Qwen3.5 text and multimodal model wrappers.

Qwen3.5 is the Qwen3-Next + Qwen3-VL bundle: text-only Qwen3.5 reuses
the Qwen3-Next decoder (hybrid full-attention/linear-attention layers,
MoE FFN, Pallas-fused decode kernel), while the multimodal variant
combines Qwen3-Next's language trunk with the Qwen3-VL vision tower
and 3-D rotary position embeddings (mRoPE) over interleaved text and
visual tokens.

This module exposes the task-specific wrappers for that stack:

- :class:`Qwen3_5ForCausalLM` — text-only causal LM, registered under
  the ``qwen3_5`` model type. Reuses :class:`Qwen3NextModel` /
  :class:`Qwen3NextForCausalLM` and supports the fused decode path.
- :class:`Qwen3_5VLForConditionalGeneration` — vision-language wrapper
  combining the Qwen3-VL vision tower with the Qwen3-Next language
  backbone and merging visual embeddings at placeholder token
  positions before invoking the LM head.

The shared helpers :func:`easydel.modules._base.mrope_position_ids_from_mm_token_types` and
:func:`easydel.modules._base.flatten_mrope_position_ids_for_text` compute the multimodal
mRoPE indices and bridge between 1-D text-only position IDs and the
3-D layout expected by the decoder.
"""

import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.types import MaskInfo
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int
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
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import MoeCausalLMOutput
from easydel.infra.utils import auto_remat, blockwise_ffn
from easydel.layers import ColumnParallelLinear
from easydel.modules._base import (
    BaseCausalLMModule,
    BaseVisionLanguageModule,
)
from easydel.modules.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextFullAttention,
    Qwen3NextMLP,
    Qwen3NextModel,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
)
from easydel.modules.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VisionTransformerPretrainedModel,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLConfig, Qwen3VLTextConfig

from .mtp import Qwen3_5MTPMixin, Qwen3_5VisionMTPMixin
from .multimodal import qwen3_5_multimodal_forward
from .qwen3_5_configuration import Qwen3_5Config, Qwen3_5TextConfig


@dataclass(frozen=True)
class Qwen3_5MTPOutput:
    """Output of the Qwen3.5 MTP head.

    Attributes:
        last_hidden_state: ``(batch, seq_len, hidden_size)`` MTP hidden
            states ready for projection through the shared LM head.
        past_key_values: Updated MTP-local KV cache (one ``cache_view``
            per MTP layer). ``None`` during training.
    """

    last_hidden_state: Float[Array, "batch seq_len hidden"]
    past_key_values: tuple[TransformerCacheView | RaggedPagesCacheView | None, ...] | None = None


class Qwen3_5MTPLayer(spx.Module):
    """Single MTP transformer block.

    Combines a Qwen3-Next full-attention layer (with attention output
    gating + per-head qk-norm) and a feed-forward network identical in
    shape to one Qwen3.5 decoder layer. The FFN is a dense SwiGLU MLP
    for the dense Qwen3.5 family and a :class:`Qwen3NextSparseMoeBlock`
    (routed experts + shared expert + gate) for the Qwen3.5-MoE family;
    the choice is driven by ``use_moe_mlp`` so the HF ``mtp.*`` tensors
    bind by name in both cases. Allocated outside the main model so MTP
    carries its own KV cache namespace.
    """

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        use_moe_mlp: bool = False,
        rngs: spx.Rngs,
    ):
        """Build the MTP block.

        Args:
            config: Qwen3.5 text config.
            layer_idx: Position within the MTP stack (0-indexed); also
                used as the cache index into MTP-local KV caches.
            dtype: Computation dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX matmul precision.
            use_moe_mlp: When ``True`` build a
                :class:`Qwen3NextSparseMoeBlock` (routed experts +
                shared expert + gate) as the FFN, matching the
                Qwen3.5-MoE checkpoint's ``mtp.*.mlp.experts`` /
                ``shared_expert`` / ``gate`` tensors. When ``False``
                (the dense Qwen3.5 default) build a dense
                :class:`Qwen3NextMLP`, keeping the 27B behavior
                byte-identical.
            rngs: PRNG container.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.is_moe = use_moe_mlp

        mtp_attn_config = copy.copy(config)
        mtp_attn_config.attn_mechanism = "vanilla"
        mtp_attn_config.decode_attn_mechanism = None
        self.self_attn = Qwen3NextFullAttention(
            config=mtp_attn_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        if use_moe_mlp:
            self.mlp = Qwen3NextSparseMoeBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = Qwen3NextMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> tuple[Float[Array, "batch seq_len hidden"], TransformerCacheView | RaggedPagesCacheView | None]:
        """Run one MTP block.

        Args:
            hidden_states: Fused (concat+fc) input hidden states.
            mask_info: Causal mask info (typically reused from the main
                model's forward pass).
            position_ids: Position indices, ``(batch, seq_len)``.
            mode: Runtime mode (train, decode, prefill).
            cache_view: MTP-local cache view at this layer's index.
            cache_metadata: Cache metadata.
            frequencies: Precomputed RoPE frequencies (reused from main
                model).

        Returns:
            Tuple of ``(updated_hidden_states, updated_cache_view)``.
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            h,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            False,
            frequencies,
        )
        hidden_states = checkpoint_name(residual + attn_out.attention_output, "mtp_attn_residual")

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            # Qwen3NextSparseMoeBlock returns (output, router_logits); the MTP
            # head is auxiliary so the router logits are discarded here (the
            # MoE aux load-balancing loss rides the main decoder's layers).
            h, _router_logits = self.mlp(h)
        elif self.config.use_scan_mlp:
            h = blockwise_ffn(
                self.mlp,
                h,
                self.config.scan_mlp_chunk_size,
            )
        else:
            h = self.mlp(h)
        hidden_states = checkpoint_name(residual + h, "mtp_mlp_residual")
        return hidden_states, attn_out.cache_view


class Qwen3_5MTPHead(spx.Module):
    """Qwen3.5 Multi-Token Prediction (MTP) head.

    Implements the DeepSeek-V3-style MTP: fuse previous-position hidden
    state with next-token embedding via concat → pre-fc RMSNorms → fc,
    process through ``N`` MTP decoder layers, final norm. The shared LM
    head is applied by the caller.

    For ``N > 1`` (not used by Qwen3.5 today; reserved for future
    multi-depth variants) the layers are stacked sequentially; each
    additional layer further conditions on the same fused hidden state
    rather than re-fusing.

    Training-time use::

        last_hidden = main_model(input_ids).last_hidden_state
        next_embeds = embed_tokens(jnp.roll(input_ids, shift=-1, axis=-1))
        mtp_hidden = mtp_head(last_hidden, next_embeds, ...).last_hidden_state
        mtp_logits = lm_head(mtp_hidden)
    """

    #: The head keeps an MTP-local KV cache; keep it out of the decoder's operation-cache discovery.
    owns_operation_cache = True

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        use_moe_mlp: bool = False,
        rngs: spx.Rngs,
    ):
        """Build the MTP head.

        Args:
            config: Qwen3.5 text config. Reads ``mtp_num_hidden_layers``,
                ``hidden_size``, ``rms_norm_eps``, and ``initializer_range``.
            dtype: Computation dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX matmul precision.
            use_moe_mlp: When ``True`` every MTP block builds a
                :class:`Qwen3NextSparseMoeBlock` FFN (Qwen3.5-MoE); when
                ``False`` (default) a dense :class:`Qwen3NextMLP` FFN
                (dense Qwen3.5). Threaded to each :class:`Qwen3_5MTPLayer`.
            rngs: PRNG container.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.use_moe_mlp = use_moe_mlp

        num_mtp_layers = int(getattr(config, "mtp_num_hidden_layers", 0))
        if num_mtp_layers < 1:
            raise ValueError(
                f"Qwen3_5MTPHead requires mtp_num_hidden_layers >= 1, got {num_mtp_layers}. "
                "Set config.mtp_num_hidden_layers > 0 to enable the head."
            )
        self.num_mtp_layers = num_mtp_layers

        self.fc = ColumnParallelLinear(
            2 * config.hidden_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        remat_layer = auto_remat(
            Qwen3_5MTPLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList(
            [
                remat_layer(
                    config=config,
                    layer_idx=i,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    use_moe_mlp=use_moe_mlp,
                    rngs=rngs,
                )
                for i in range(num_mtp_layers)
            ]
        )
        self.norm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        prev_hidden_states: Float[Array, "batch seq_len hidden"],
        next_token_embeds: Float[Array, "batch seq_len hidden"],
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        attention_mask: Float[Array, "batch seq_len"] | None = None,
    ) -> Qwen3_5MTPOutput:
        """Forward through the MTP head.

        Args:
            prev_hidden_states: ``(B, S, H)`` last_hidden_state from the
                main model (already post-final-norm).
            next_token_embeds: ``(B, S, H)`` token embeddings for the
                ground-truth next position. Caller is responsible for
                the shift (``embed_tokens(jnp.roll(input_ids, -1))``).
                For boundary positions where there is no next token,
                pass any embedding; the MTP loss should mask those
                positions out.
            mask_info: Causal mask info, reused from main forward.
            position_ids: Position indices; reused from main forward.
            mode: Runtime mode.
            past_key_values: MTP-local KV cache. Must be allocated by
                the caller with ``num_mtp_layers`` entries; do NOT pass
                the main model's cache (different layer count).
            cache_metadata: Cache metadata.
            frequencies: RoPE frequencies (reuse main model's).
            attention_mask: Optional padding mask used to build
                ``mask_info`` / ``position_ids`` when those are not
                supplied by the caller.

        Returns:
            Qwen3_5MTPOutput with ``last_hidden_state`` shaped
            ``(B, S, H)`` and the updated MTP cache.
        """
        normed_h = self.pre_fc_norm_hidden(prev_hidden_states)
        normed_e = self.pre_fc_norm_embedding(next_token_embeds)
        # Fusion order is [embedding, hidden] — the layout the released Qwen3.5/3.6
        # ``mtp.fc`` weights were trained with (vLLM's Qwen3NextMTP does
        # ``cat([inputs_embeds, hidden_states])``; the eSurge drafter fusion in
        # ``inference/speculative.py`` matches it). The reversed order silently
        # produces near-uniform draft logits from a trained head: HF ignores
        # ``mtp.*`` on load, so only the drafter/serving path ever validated this.
        fused = jnp.concatenate([normed_e, normed_h], axis=-1)
        h = self.fc(fused)
        h = apply_logical_sharding(
            h,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        batch_size, seq_len = h.shape[:2]
        if mask_info is None:
            mask_info = MaskInfo.dynamic_init(
                mask_info=None,
                input_ids=None,
                inputs_embeds=h,
                attention_mask=attention_mask,
            )
        if position_ids is None:
            position_ids = (
                mask_info.q_position_ids
                if hasattr(mask_info, "q_position_ids")
                else (jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0))
            )

        views = past_key_values.views if past_key_values is not None else None
        new_views: list[TransformerCacheView | RaggedPagesCacheView | None] = []
        for i, layer in enumerate(self.layers):
            cv_in = views[i] if views is not None and i < len(views) else None
            h, cv_out = layer(
                h,
                mask_info,
                position_ids,
                mode,
                cv_in,
                cache_metadata,
                frequencies,
            )
            new_views.append(cv_out)

        h = self.norm(h)
        h = checkpoint_name(h, "mtp_output")
        return Qwen3_5MTPOutput(last_hidden_state=h, past_key_values=tuple(new_views) or None)


@register_module(TaskType.BASE_MODULE, config=Qwen3_5TextConfig, model_type="qwen3_5_text")
class Qwen3_5TextModel(Qwen3NextModel):
    """Qwen3.5 text-only base model (no LM head).

    Thin wrapper around :class:`Qwen3NextModel` registered with the
    ``qwen3_5_text`` model type.
    """


@register_module(TaskType.CAUSAL_LM, config=Qwen3_5TextConfig, model_type="qwen3_5")
@register_module(TaskType.CAUSAL_LM, config=Qwen3_5TextConfig, model_type="qwen3_5_text")
class Qwen3_5ForCausalLM(Qwen3_5MTPMixin, Qwen3NextForCausalLM):
    """Qwen3.5 text causal language model.

    Wraps :class:`Qwen3_5TextModel` with a linear LM head for next-token
    prediction, plus an optional DeepSeek-V3-style Multi-Token
    Prediction (MTP) head when ``config.mtp_num_hidden_layers > 0``.

    Args:
        config: Qwen3.5 text configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    _model_type = "qwen3_5"
    _config_class = Qwen3_5TextConfig

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3.5 text causal LM (optionally with MTP head).

        When ``config.mtp_num_hidden_layers > 0``, an
        :class:`~easydel.modules.qwen3_5.mtp.Qwen3_5MTPHead` is
        instantiated as ``self.mtp`` so HF ``mtp.*`` checkpoint tensors
        auto-bind by name during ``from_pretrained``.

        Args:
            config: Qwen3.5 text configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX matmul precision.
            rngs: PRNG key container.
        """
        BaseCausalLMModule.__init__(
            self,
            config=config,
            base_model_class=Qwen3_5TextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )
        if int(getattr(config, "mtp_num_hidden_layers", 0)) > 0:
            self.mtp = Qwen3_5MTPHead(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mtp = None

    def forward(
        self,
        input_ids: jax.Array | None = None,
        inputs_embeds: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        mask_info=None,
        position_ids: jax.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Causal-LM forward with the MTP auxiliary loss folded in.

        Runs the standard Qwen3-Next causal-LM forward, then — during
        training/eval and when an MTP head is present — adds the MTP
        cross-entropy loss to ``outputs.aux_loss`` via
        :meth:`_maybe_add_mtp_aux_loss`, so the trainer trains the
        MTP head automatically. See that method for the gating rules.
        """
        outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
        )
        mtp_mode = mode
        if mtp_mode is None and (past_key_values is not None or cache_metadata is not None):
            seq_len = None
            if input_ids is not None:
                seq_len = input_ids.shape[1]
            elif inputs_embeds is not None:
                seq_len = inputs_embeds.shape[1]
            mtp_mode = common_types.MODE_DECODE if seq_len == 1 else common_types.MODE_PREFILL
        return self._maybe_add_mtp_aux_loss(
            outputs, input_ids, attention_mask, mtp_mode, mask_info=mask_info, position_ids=position_ids
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3_5Config, model_type="qwen3_5")
@register_module(TaskType.VISION_LM, config=Qwen3_5Config, model_type="qwen3_5")
class Qwen3_5Model(Qwen3VLModel):
    """Qwen3.5 multimodal (vision-language) base model.

    Combines a :class:`Qwen3VisionTransformerPretrainedModel` vision encoder
    with a :class:`Qwen3_5TextModel` language backbone. Image and video pixels
    are encoded into continuous embeddings, fused into the token embedding
    stream, and processed by the language model with 3D mRoPE position ids.

    Args:
        config: Qwen3.5 multimodal configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: Qwen3_5Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3.5 multimodal model with vision encoder and text decoder.

        Args:
            config: Qwen3.5 multimodal configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX matmul precision.
            rngs: PRNG key container.
        """
        bootstrap_text_config = Qwen3VLTextConfig(
            vocab_size=config.text_config.vocab_size,
            hidden_size=config.text_config.hidden_size,
            intermediate_size=config.text_config.intermediate_size,
            num_hidden_layers=config.text_config.num_hidden_layers,
            num_attention_heads=config.text_config.num_attention_heads,
            num_key_value_heads=config.text_config.num_key_value_heads,
            head_dim=config.text_config.head_dim,
            hidden_act=config.text_config.hidden_act,
            max_position_embeddings=config.text_config.max_position_embeddings,
            initializer_range=config.text_config.initializer_range,
            rms_norm_eps=config.text_config.rms_norm_eps,
            use_cache=config.text_config.use_cache,
            tie_word_embeddings=getattr(config.text_config, "tie_word_embeddings", False),
            rope_theta=config.text_config.rope_theta,
            attention_bias=config.text_config.attention_bias,
            attention_dropout=config.text_config.attention_dropout,
            rope_scaling=getattr(config.text_config, "rope_scaling", None),
            layer_types=getattr(config.text_config, "layer_types", None),
        )
        bootstrap_config = Qwen3VLConfig(
            vision_config=(
                config.vision_config.to_dict()
                if hasattr(config.vision_config, "to_dict")
                else vars(config.vision_config)
            ),
            text_config=bootstrap_text_config.to_dict(),
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
        )
        super().__init__(
            config=bootstrap_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.config = config
        # Rebuild vision tower from the final Qwen3.5 vision config so
        # deepstack settings and parameter names match HF checkpoints.
        self.visual = Qwen3VisionTransformerPretrainedModel(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = Qwen3_5TextModel(
            config=config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        input_ids: jax.Array | None = None,
        inputs_embeds: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        mask_info: object | None = None,
        position_ids: jax.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        visual_pos_masks: jax.Array | None = None,  # compatibility no-op
        deepstack_visual_embeds: list[jax.Array] | None = None,  # compatibility no-op
        pixel_values: jax.Array | None = None,
        image_embeds: jax.Array | None = None,
        pixel_values_videos: jax.Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        cache_position: jax.Array | None = None,  # compatibility no-op
        rope_deltas: jax.Array | None = None,  # compatibility no-op
        mm_token_type_ids: jax.Array | None = None,
        **kwargs,
    ) -> Qwen3VLModelOutputWithPast:
        """Forward pass through the Qwen3.5 multimodal model.

        Encodes image/video inputs via the vision tower, merges them into
        the text embedding stream at placeholder token positions, derives
        3-D mRoPE indices (from explicit ``mm_token_type_ids`` or by
        inferring spans from token IDs), then runs the Qwen3-Next-based
        language decoder.

        Args:
            input_ids: Text token ids of shape ``(batch, seq_len)``.
                Mutually exclusive with ``inputs_embeds``.
            inputs_embeds: Pre-computed input embeddings of shape
                ``(batch, seq_len, hidden_size)``. Mutually exclusive
                with ``input_ids``.
            attention_mask: Boolean mask ``(batch, seq_len)`` marking
                non-padding positions.
            mask_info: Optional pre-built ``MaskInfo`` carrying causal
                / sliding masks; constructed automatically when omitted.
            position_ids: Position indices. May be ``(batch, seq_len)``
                for 1-D RoPE or ``(3, batch, seq_len)`` for 3-D mRoPE
                (text/height/width); flattened by
                :func:`flatten_mrope_position_ids_for_text` when the
                text config disables mRoPE.
            mode: Runtime mode (train/decode). Auto-detected if ``None``.
            past_key_values: KV cache used for autoregressive decoding.
                One of ``TransformerCache``, ``RaggedPagesCache``, or
                ``HybridCache``.
            cache_metadata: Cache metadata accompanying ``past_key_values``.
            output_attentions: Whether to return per-layer attention weights.
            output_hidden_states: Whether to return per-layer hidden states.
            visual_pos_masks: Compatibility no-op accepted for parity
                with sibling Qwen3-VL forwards.
            deepstack_visual_embeds: Compatibility no-op (Qwen3.5 does
                not use deepstack mergers).
            pixel_values: Packed image pixel values for the vision tower.
            image_embeds: Pre-computed post-vision-tower image embeds of
                shape ``(num_image_tokens, hidden_size)`` scattered at
                image placeholder positions, letting training skip the
                vision tower. Used only when ``pixel_values`` is ``None``;
                still requires ``image_grid_thw`` for correct 3-D mRoPE.
            pixel_values_videos: Packed video pixel values for the vision tower.
            image_grid_thw: Per-image ``(T, H, W)`` grid dimensions.
            video_grid_thw: Per-video ``(T, H, W)`` grid dimensions.
            image_max_grid_size: Optional shared maximum grid size for
                static-shape compilation of image batches.
            video_max_grid_size: Same as above for videos.
            cache_position: Compatibility no-op.
            rope_deltas: Compatibility no-op (overwritten internally).
            mm_token_type_ids: Per-token modality ids
                (0=text, 1=image, 2=video) used to build mRoPE indices
                when ``position_ids`` is omitted.
            **kwargs: Forward-compatibility sink; ignored.

        Returns:
            Qwen3VLModelOutputWithPast: ``last_hidden_state``,
            ``past_key_values``, optional ``hidden_states`` /
            ``attentions``, and computed ``rope_deltas``.

        Raises:
            ValueError: If both ``input_ids`` and ``inputs_embeds`` are
                provided, or both are ``None``.
        """
        return qwen3_5_multimodal_forward(
            self,
            output_class=Qwen3VLModelOutputWithPast,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            image_embeds=image_embeds,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            mm_token_type_ids=mm_token_type_ids,
        )


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen3_5Config, model_type="qwen3_5")
class Qwen3_5ForConditionalGeneration(Qwen3_5VisionMTPMixin, BaseVisionLanguageModule[Qwen3_5Model, Qwen3_5Config]):
    """Qwen3.5 multimodal conditional generation model.

    End-to-end vision-language model that wraps :class:`Qwen3_5Model` and
    adds a causal LM head for image/video-conditioned text generation.
    Supports both image and video inputs via the underlying vision encoder.

    Args:
        config: Qwen3.5 multimodal configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen3_5"
    _config_class = Qwen3_5Config
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen3_5Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3.5 for conditional generation with vision-language support.

        Args:
            config: Qwen3.5 multimodal configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX matmul precision.
            rngs: PRNG key container.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3_5Model,
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
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size
        if int(getattr(config.text_config, "mtp_num_hidden_layers", 0)) > 0:
            self.mtp = Qwen3_5MTPHead(
                config=config.text_config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mtp = None

    def forward(self, *args, **kwargs):
        """Vision-language forward with the MTP auxiliary loss folded in.

        Delegates to :meth:`BaseVisionLanguageModule.forward`, then —
        during training/eval with an MTP head present — adds the MTP
        cross-entropy loss to ``outputs.aux_loss`` so the trainer
        trains the MTP head. See :meth:`_maybe_add_mtp_aux_loss`.
        """
        return self._vision_forward_with_mtp(super().forward, *args, **kwargs)

    def get_input_embeddings(self):
        """Return the shared text token embedding layer.

        Delegates to the wrapped :class:`Qwen3VLModel` which owns the
        text token ``Embed`` module used to project ``input_ids`` to the
        decoder's ``hidden_size`` before visual fusion.

        Returns:
            spx.Module: The token embedding module backing
            ``self.model.language_model``.
        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Replace the shared text token embedding layer.

        Mutates the wrapped :class:`Qwen3VLModel` in-place; useful when
        loading partial checkpoints where the embedding table is patched
        independently of the rest of the language model.

        Args:
            value: Replacement embedding module. Must accept ``input_ids``
                of shape ``(batch, seq_len)`` and produce hidden states of
                shape ``(batch, seq_len, hidden_size)``.
        """
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        """Replace the underlying Qwen3-Next decoder stack.

        Args:
            decoder: Replacement language-model module (typically a
                :class:`Qwen3NextModel` or compatible). Used by checkpoint
                surgery and unit tests to swap in custom decoders without
                rebuilding the VLM wrapper.
        """
        self.model.set_decoder(decoder)

    def get_decoder(self):
        """Return the wrapped Qwen3-Next decoder stack.

        Returns:
            spx.Module: The text decoder used to consume the merged
            text + visual hidden states; identical to
            ``self.model.language_model``.
        """
        return self.model.get_decoder()

    @property
    def visual(self):
        """Backward-compatible alias for the Qwen3-VL vision tower.

        Older HuggingFace-style call sites expect the vision encoder to
        live under ``model.visual``. This property forwards to the same
        attribute on the wrapped :class:`Qwen3VLModel` so existing
        utilities continue to work.

        Returns:
            spx.Module: The Qwen3-VL vision transformer (a
            :class:`Qwen3VisionTransformerPretrainedModel`).
        """
        return self.model.visual

    @property
    def language_model(self):
        """Backward-compatible alias for the Qwen3-Next language trunk.

        Mirrors the HuggingFace ``model.language_model`` attribute so
        downstream code can reach the text decoder directly. The returned
        module owns the embedding table, transformer layers, and final
        norm; it does NOT include the LM head, which lives on the
        outer :class:`Qwen3_5ForConditionalGeneration` wrapper.

        Returns:
            spx.Module: The wrapped Qwen3-Next decoder.
        """
        return self.model.language_model

    def get_video_features(
        self,
        pixel_values_videos: jax.Array,
        video_grid_thw: jax.Array | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple[tuple[jax.Array, ...], list[jax.Array]]:
        """Encode video tensors into per-grid visual embeddings.

        Routes through the vision tower and projector that are shared
        with image processing; the temporal grid axis distinguishes
        videos from images.

        Args:
            pixel_values_videos: Packed pixel values for the videos in
                the batch. Concrete shape is ``(num_video_patches, ...)``
                following the Qwen3-VL processor convention.
            video_grid_thw: Per-video ``(T, H, W)`` grid dimensions used
                to locate placeholder tokens and reconstruct the spatial
                layout. ``None`` when no video grids are supplied.
            video_max_grid_size: Optional padding size for the largest
                grid; needed when batched videos must share a static
                shape for compilation.

        Returns:
            A tuple ``(per_video_embeddings, split_sizes)`` where the
            first element is a tuple of arrays of shape
            ``(num_tokens_i, hidden_size)`` (one per video) and the
            second is a list of per-video token counts useful for
            scatter-merging into the text sequence.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(
        self,
        pixel_values: jax.Array,
        image_grid_thw: jax.Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[jax.Array, ...], list[jax.Array]]:
        """Encode image tensors into per-image visual embeddings.

        Args:
            pixel_values: Packed pixel values for all images in the batch
                with shape ``(num_image_patches, ...)``.
            image_grid_thw: Per-image ``(T, H, W)`` grid dimensions
                (``T == 1`` for still images). Used to locate placeholder
                positions in the text sequence.
            image_max_grid_size: Optional shared upper bound on grid size
                for static-shape compilation paths.

        Returns:
            A tuple ``(per_image_embeddings, split_sizes)`` where the
            first element is a tuple of arrays of shape
            ``(num_tokens_i, hidden_size)`` (one per image) ready for
            placeholder substitution, and the second is the matching
            list of per-image token counts.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute the merged text + multimodal input embeddings.

        Delegates to the wrapped :class:`Qwen3VLModel`, which: (1)
        embeds ``input_ids`` via the text token embedding, (2) optionally
        encodes images/videos with the vision tower, and (3) splices the
        visual embeddings into the placeholder positions identified by
        ``image_token_id`` / ``video_token_id`` from the config.

        Args:
            input_ids: Input token ids of shape ``(batch, seq_len)``.
            *args: Forwarded positional arguments (e.g. ``pixel_values``,
                ``pixel_values_videos``, ``image_grid_thw``,
                ``video_grid_thw``).
            **kwargs: Forwarded keyword arguments accepted by
                ``Qwen3VLModel.compute_embedding``.

        Returns:
            Merged input embeddings of shape ``(batch, seq_len, hidden_size)``
            with visual hidden states substituted at placeholder token
            positions.
        """
        return self.model.compute_embedding(input_ids, *args, **kwargs)


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MTPHead",
    "Qwen3_5MTPLayer",
    "Qwen3_5MTPOutput",
    "Qwen3_5Model",
    "Qwen3_5TextConfig",
    "Qwen3_5TextModel",
]
