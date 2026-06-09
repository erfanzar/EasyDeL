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

Helper functions :func:`_get_rope_index_from_mm_token_types` and
:func:`_maybe_flatten_position_ids_for_text` compute the multimodal
mRoPE indices and bridge between 1-D text-only position IDs and the
3-D layout expected by the decoder.
"""

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
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
from easydel.modules._base import BaseCausalLMModule, BaseVisionLanguageModule
from easydel.modules.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextFullAttention,
    Qwen3NextMLP,
    Qwen3NextModel,
    Qwen3NextRMSNorm,
)
from easydel.modules.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VisionTransformerPretrainedModel,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLConfig, Qwen3VLTextConfig

from .qwen3_5_configuration import Qwen3_5Config, Qwen3_5TextConfig


def _get_rope_index_from_mm_token_types(
    input_ids: jax.Array,
    mm_token_type_ids: jax.Array,
    image_grid_thw: jax.Array | None = None,
    video_grid_thw: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    spatial_merge_size: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Compute 3D mRoPE position ids from modality token-type ids.

    Groups consecutive tokens by modality (0 = text, 1 = image, 2 = video)
    and assigns separate temporal/height/width position ids for visual tokens
    according to their spatial grid layout.

    Args:
        input_ids: Input token ids of shape ``(batch, seq_len)``.
        mm_token_type_ids: Per-token modality type ids (0=text, 1=image, 2=video).
        image_grid_thw: Grid dimensions ``(T, H, W)`` for each image.
        video_grid_thw: Grid dimensions ``(T, H, W)`` for each video.
        attention_mask: Boolean attention mask of shape ``(batch, seq_len)``.
        spatial_merge_size: Spatial merge factor from the vision config.

    Returns:
        Tuple of ``(position_ids, mrope_position_deltas)`` where ``position_ids``
        has shape ``(3, batch, seq_len)`` and deltas has shape ``(batch, 1)``.
    """
    input_ids_np = np.asarray(input_ids)
    token_types_np = np.asarray(mm_token_type_ids)
    attention_mask_np = np.asarray(attention_mask).astype(bool) if attention_mask is not None else None

    image_iter = iter(np.asarray(image_grid_thw)) if image_grid_thw is not None else None
    video_iter = iter(np.asarray(video_grid_thw)) if video_grid_thw is not None else None

    batch_size, seq_len = input_ids_np.shape
    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int32)
    mrope_position_deltas: list[int] = []

    for batch_idx in range(batch_size):
        current_input_ids = input_ids_np[batch_idx]
        current_types = token_types_np[batch_idx]
        if attention_mask_np is not None:
            valid_mask = attention_mask_np[batch_idx]
            current_input_ids = current_input_ids[valid_mask]
            current_types = current_types[valid_mask]

        groups = []
        for key, group in itertools.groupby(enumerate(current_types.tolist()), lambda x: x[1]):
            group = list(group)
            groups.append((key, group[0][0], group[-1][0] + 1))

        current_pos = 0
        llm_pos_ids_list: list[np.ndarray] = []
        for modality_type, start_idx, end_idx in groups:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    np.arange(text_len, dtype=np.int32).reshape(1, -1).repeat(3, axis=0) + current_pos,
                )
                current_pos += text_len
            else:
                grid_iter = image_iter if modality_type == 1 else video_iter
                if grid_iter is None:
                    continue
                grid_thw = next(grid_iter)
                llm_grid_t = int(grid_thw[0])
                llm_grid_h = int(grid_thw[1]) // spatial_merge_size
                llm_grid_w = int(grid_thw[2]) // spatial_merge_size

                image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
                position_width = np.arange(current_pos, current_pos + llm_grid_w, dtype=np.int32).repeat(
                    llm_grid_h * llm_grid_t
                )
                position_height = np.arange(current_pos, current_pos + llm_grid_h, dtype=np.int32).repeat(
                    llm_grid_w * llm_grid_t
                )
                position_temporal = np.full((image_seq_length,), current_pos, dtype=np.int32)
                llm_pos_ids_list.append(np.stack([position_temporal, position_height, position_width], axis=0))

                current_pos += max(int(grid_thw[1]), int(grid_thw[2])) // spatial_merge_size

        if len(llm_pos_ids_list) == 0:
            llm_positions = np.zeros((3, 0), dtype=np.int32)
        else:
            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)

        if attention_mask_np is not None:
            position_ids[:, batch_idx, attention_mask_np[batch_idx]] = llm_positions
        else:
            position_ids[:, batch_idx, : llm_positions.shape[1]] = llm_positions
        delta = int(llm_positions.max() + 1 - len(current_input_ids)) if llm_positions.shape[1] > 0 else 0
        mrope_position_deltas.append(delta)

    return jnp.asarray(position_ids, dtype=jnp.int32), jnp.asarray(mrope_position_deltas, dtype=jnp.int32).reshape(-1, 1)


def _maybe_flatten_position_ids_for_text(config: Qwen3_5TextConfig, position_ids: jax.Array) -> jax.Array:
    """Flatten 3D mRoPE position ids to 1D when the text config does not enable mRoPE.

    Args:
        config: Text model configuration.
        position_ids: Position ids, either ``(batch, seq)`` or ``(3, batch, seq)``.

    Returns:
        Position ids with shape ``(batch, seq)`` (1D) or unchanged if mRoPE is enabled.
    """
    rope_scaling = getattr(config, "rope_scaling", None)
    uses_mrope = isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling
    if position_ids.ndim == 3 and not uses_mrope:
        return position_ids[0]
    return position_ids


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
    gating + per-head qk-norm) and a dense SwiGLU MLP, identical in
    shape to one Qwen3.5 full-attention decoder layer. Allocated
    outside the main model so MTP carries its own KV cache namespace.
    """

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
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
            rngs: PRNG container.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.self_attn = Qwen3NextFullAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
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
        if self.config.use_scan_mlp:
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

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the MTP head.

        Args:
            config: Qwen3.5 text config. Reads ``mtp_num_hidden_layers``,
                ``hidden_size``, ``rms_norm_eps``, and ``initializer_range``.
            dtype: Computation dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX matmul precision.
            rngs: PRNG container.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

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
        fused = jnp.concatenate([normed_h, normed_e], axis=-1)
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
class Qwen3_5ForCausalLM(Qwen3NextForCausalLM):
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

    def has_mtp(self) -> bool:
        """Report whether this model carries an MTP head.

        Returns:
            ``True`` iff ``config.mtp_num_hidden_layers > 0`` and the
            head was instantiated.
        """
        return self.mtp is not None

    def compute_mtp_outputs(
        self,
        outputs: MoeCausalLMOutput,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        mask_info=None,
        position_ids: jax.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        mtp_past_key_values=None,
        cache_metadata=None,
    ) -> Qwen3_5MTPOutput | None:
        """Run the MTP head and produce per-depth-1 hidden states + logits.

        Given the main model's ``outputs.last_hidden_state``, fuses each
        position's hidden state with the embedding of the NEXT
        ground-truth token (``input_ids`` rolled left by 1) through the
        MTP head, then projects through the shared LM head to obtain
        ``mtp_logits`` predicting position ``t + 2``.

        Args:
            outputs: Result of the main causal-LM forward pass.
                ``outputs.last_hidden_state`` is required.
            input_ids: ``(batch, seq_len)`` token IDs of the main forward
                pass. Used to look up next-token embeddings via the
                shared ``embed_tokens`` table.
            attention_mask: Optional padding mask.
            mask_info: Causal mask info from the main forward (reused
                so the MTP attention sees the same mask).
            position_ids: Position indices, reused from main forward.
            mode: Runtime mode. ``None`` falls back to train.
            mtp_past_key_values: MTP-local KV cache. Pass ``None`` for
                training; allocate with ``num_mtp_layers`` entries for
                generation.
            cache_metadata: Cache metadata.

        Returns:
            :class:`Qwen3_5MTPOutput` with ``last_hidden_state`` shaped
            ``(B, S, H)`` and the updated MTP cache, or ``None`` when
            the model has no MTP head.

            The caller projects ``output.last_hidden_state`` through
            ``self.compute_lm_logits(self.prepare_lm_head_inputs(...))``
            to obtain MTP logits.
        """
        if self.mtp is None:
            return None

        next_input_ids = jnp.concatenate(
            [input_ids[:, 1:], jnp.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)],
            axis=-1,
        )
        next_token_embeds = self.model.get_embedding()(next_input_ids.astype("i4"))

        frequencies = getattr(self.model, "frequencies", None)
        return self.mtp(
            prev_hidden_states=outputs.last_hidden_state,
            next_token_embeds=next_token_embeds,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=mtp_past_key_values,
            cache_metadata=cache_metadata,
            frequencies=frequencies,
            attention_mask=attention_mask,
        )

    def compute_mtp_logits(self, mtp_output: Qwen3_5MTPOutput) -> jax.Array:
        """Project MTP hidden states through the shared LM head.

        Args:
            mtp_output: Output of :meth:`compute_mtp_outputs`.

        Returns:
            ``(batch, seq_len, vocab_size)`` MTP logits over the
            "skip-one" prediction targets (``input_ids[t + 2]``).
        """
        h = self.prepare_lm_head_inputs(mtp_output.last_hidden_state)
        return self.compute_lm_logits(h)

    def compute_mtp_chain(
        self,
        outputs: MoeCausalLMOutput,
        input_ids: jax.Array,
        n_steps: int,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array | None:
        """Recursively apply the (depth-1) MTP head ``n_steps`` times, teacher-forced.

        This mirrors how the inference drafter chains the inline MTP block to draft
        more than one token ahead (feed the block's output hidden + the next token
        back in). Step ``k`` (1-indexed) fuses the *previous* MTP hidden state with
        the embedding of the ground-truth token at offset ``k`` and predicts the
        token at position ``t + k + 1``:

            h^0 = outputs.last_hidden_state
            h^k = MTP(prev=h^{k-1}, next_embed=Emb(input_ids shifted left by k))
            logits^k = LMHead(h^k)            # predicts x_{t+k+1}

        Teacher-forcing (feeding ground-truth tokens, like DeepSeek-V3's MTP
        training) keeps it fully differentiable. ``n_steps == 1`` reproduces
        :meth:`compute_mtp_outputs` + :meth:`compute_mtp_logits` exactly.

        Args:
            outputs: Main causal-LM forward output (``last_hidden_state`` required).
            input_ids: ``(B, S)`` token IDs of the main forward pass.
            n_steps: Number of recursive draft steps (>= 1).
            attention_mask: Optional padding mask.

        Returns:
            ``(n_steps, B, S, vocab)`` MTP logits — entry ``k-1`` predicts
            ``input_ids[t + k + 1]`` — or ``None`` when the model has no MTP head.
        """
        if self.mtp is None or n_steps < 1:
            return None
        b = input_ids.shape[0]
        embed = self.model.get_embedding()
        frequencies = getattr(self.model, "frequencies", None)
        prev_hidden = outputs.last_hidden_state
        step_logits: list[jax.Array] = []
        for k in range(1, n_steps + 1):
            shifted = jnp.concatenate(
                [input_ids[:, k:], jnp.zeros((b, k), dtype=input_ids.dtype)],
                axis=-1,
            )
            mtp_out = self.mtp(
                prev_hidden_states=prev_hidden,
                next_token_embeds=embed(shifted.astype("i4")),
                frequencies=frequencies,
                attention_mask=attention_mask,
            )
            prev_hidden = mtp_out.last_hidden_state
            step_logits.append(self.compute_mtp_logits(mtp_out))
        return jnp.stack(step_logits, axis=0)

    def compute_mtp_loss(
        self,
        mtp_logits: jax.Array,
        labels: jax.Array,
        attention_mask: jax.Array | None = None,
        ignore_index: int = -100,
    ) -> jax.Array:
        """Compute the DeepSeek-V3-style MTP cross-entropy loss.

        With one MTP depth (the Qwen3.5 default), the head predicts
        token at position ``t + 2`` from a fusion of the hidden state
        at ``t`` and the embedding at ``t + 1``. The CE loss therefore
        compares ``mtp_logits[..., t, :]`` against
        ``labels[..., t + 2]``.

        Args:
            mtp_logits: ``(batch, seq_len, vocab_size)`` from
                :meth:`compute_mtp_logits`.
            labels: ``(batch, seq_len)`` ground-truth token IDs
                (typically the same as ``input_ids``).
            attention_mask: Optional ``(batch, seq_len)`` mask;
                positions where ``attention_mask == 0`` AT THE TARGET
                INDEX (``t + 2``) are excluded.
            ignore_index: Label sentinel for "skip this position";
                applied to the trailing 2 positions where there is no
                ``t + 2`` target.

        Returns:
            Scalar mean CE loss over non-ignored positions, BEFORE
            multiplication by ``config.mtp_loss_coef``. Callers should
            scale by ``self.config.mtp_loss_coef`` (or
            ``self.config.text_config.mtp_loss_coef`` for the
            multimodal wrapper) and add to the main CE loss.
        """
        batch_size = labels.shape[0]
        labels.shape[1]
        pad = jnp.full((batch_size, 2), ignore_index, dtype=labels.dtype)
        shifted_labels = jnp.concatenate([labels[:, 2:], pad], axis=-1)

        if attention_mask is not None:
            mask_shifted = jnp.concatenate(
                [attention_mask[:, 2:], jnp.zeros((batch_size, 2), dtype=attention_mask.dtype)],
                axis=-1,
            )
            shifted_labels = jnp.where(mask_shifted.astype(jnp.bool_), shifted_labels, ignore_index)

        logits = mtp_logits.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, jnp.maximum(shifted_labels, 0)[..., None], axis=-1).squeeze(-1)
        valid = (shifted_labels != ignore_index).astype(jnp.float32)
        loss_sum = jnp.sum(nll * valid)
        loss_count = jnp.maximum(jnp.sum(valid), 1.0)
        return loss_sum / loss_count

    def _maybe_add_mtp_aux_loss(
        self,
        outputs: MoeCausalLMOutput,
        input_ids: jax.Array | None,
        attention_mask: jax.Array | None,
        mode: common_types.RUNTIME_MODE_TYPES | None,  # type: ignore
    ) -> MoeCausalLMOutput:
        """Fold the MTP CE loss into ``outputs.aux_loss`` during training.

        EasyDeL's trainer (:meth:`EasyDeLBaseModule.compute_loss`) adds
        ``outputs.aux_loss`` to the main causal-LM loss automatically —
        the same channel MoE router losses use. Routing the MTP loss
        through ``aux_loss`` is therefore what makes the MTP head
        actually receive gradients during ``fit`` / fine-tuning; no
        trainer changes are needed.

        The MTP loss is computed only when:

        - an MTP head exists (``config.mtp_num_hidden_layers > 0``),
        - ``config.mtp_loss_coef > 0`` (set it to ``0`` to freeze /
          ignore the MTP head while fine-tuning the base model),
        - ``input_ids`` are available (the MTP targets are
          ``input_ids`` shifted by 2 — self-supervised, so no extra
          ``labels`` are needed),
        - the run is training/eval, not autoregressive decode/prefill
          (skipping MTP during generation avoids wasted compute).

        Args:
            outputs: The main causal-LM forward output.
            input_ids: Token IDs of the current batch.
            attention_mask: Optional padding mask.
            mode: Runtime mode; decode/prefill/insert skip the MTP loss.

        Returns:
            ``outputs`` with ``aux_loss`` updated to include
            ``mtp_loss_coef * mtp_ce_loss`` (added to any existing
            MoE router aux loss), or unchanged when MTP is inactive.
        """
        if self.mtp is None or input_ids is None:
            return outputs
        coef = float(getattr(self.config, "mtp_loss_coef", 0.0))
        if coef <= 0.0:
            return outputs
        if mode in (common_types.MODE_DECODE, common_types.MODE_PREFILL, common_types.MODE_INSERT):
            return outputs
        if getattr(outputs, "last_hidden_state", None) is None:
            return outputs

        mtp_output = self.compute_mtp_outputs(
            outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode=mode,
        )
        if mtp_output is None:
            return outputs
        mtp_logits = self.compute_mtp_logits(mtp_output)
        mtp_loss = self.compute_mtp_loss(mtp_logits, input_ids, attention_mask) * coef
        existing = getattr(outputs, "aux_loss", None)
        new_aux = mtp_loss if existing is None else existing + mtp_loss
        # Expose the MTP logits (already computed here) so a distillation/training step
        # can supervise the MTP head with a teacher distribution without recomputing the
        # large-vocab projection. Reused by the offline DistillationTrainer.
        return outputs.replace(aux_loss=new_aux, mtp_logits=mtp_logits)

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
        return self._maybe_add_mtp_aux_loss(outputs, input_ids, attention_mask, mtp_mode)


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
                :func:`_maybe_flatten_position_ids_for_text` when the
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
        del rope_deltas, kwargs
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(input_ids)

        video_embeds = None
        if pixel_values is not None:
            image_embeds_tuple, _deepstack_image_embeds = self.get_image_features(
                pixel_values,
                image_grid_thw,
                image_max_grid_size,
            )
            image_embeds = jnp.concatenate(image_embeds_tuple, axis=0).astype(inputs_embeds.dtype)
        if pixel_values_videos is not None:
            video_embeds_tuple, _deepstack_video_embeds = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                video_max_grid_size,
            )
            video_embeds = jnp.concatenate(video_embeds_tuple, axis=0).astype(inputs_embeds.dtype)
        if image_embeds is not None or video_embeds is not None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds,
                video_embeds=video_embeds,
            )

        rope_deltas = None
        if position_ids is None:
            if mm_token_type_ids is not None:
                position_ids, rope_deltas = _get_rope_index_from_mm_token_types(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                )
            else:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw if (pixel_values is not None or image_embeds is not None) else None,
                    video_grid_thw=video_grid_thw if pixel_values_videos is not None else None,
                    attention_mask=attention_mask,
                )

        position_ids = _maybe_flatten_position_ids_for_text(self.config.text_config, position_ids)

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
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen3_5Config, model_type="qwen3_5")
class Qwen3_5ForConditionalGeneration(BaseVisionLanguageModule[Qwen3_5Model, Qwen3_5Config]):
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

    def has_mtp(self) -> bool:
        """Whether this multimodal wrapper carries an MTP head."""
        return self.mtp is not None

    def compute_mtp_outputs(
        self,
        outputs,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        mask_info=None,
        position_ids: jax.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        mtp_past_key_values=None,
        cache_metadata=None,
    ) -> Qwen3_5MTPOutput | None:
        """Run the MTP head over the multimodal model's hidden states.

        See :meth:`Qwen3_5ForCausalLM.compute_mtp_outputs` for the
        per-position semantics. For multimodal inputs the MTP head sees
        the merged text+visual hidden states the language trunk
        produced; only the text positions contribute meaningfully to
        the MTP CE loss, so the caller should mask non-text positions
        in the label.

        Args:
            outputs: Result of the multimodal forward pass; must expose
                ``last_hidden_state``.
            input_ids: Token IDs (with placeholder tokens at visual
                positions).
            attention_mask: Optional padding mask.
            mask_info: Causal mask info, reused from main forward.
            position_ids: Position indices, reused from main forward.
            mode: Runtime mode.
            mtp_past_key_values: MTP-local KV cache (do NOT share with
                the main model's cache).
            cache_metadata: Cache metadata.

        Returns:
            :class:`Qwen3_5MTPOutput` or ``None`` if no MTP head.
        """
        if self.mtp is None:
            return None

        next_input_ids = jnp.concatenate(
            [input_ids[:, 1:], jnp.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)],
            axis=-1,
        )
        next_token_embeds = self.model.get_input_embeddings()(next_input_ids.astype("i4"))

        language_model = self.model.language_model
        frequencies = getattr(language_model, "frequencies", None)
        return self.mtp(
            prev_hidden_states=outputs.last_hidden_state,
            next_token_embeds=next_token_embeds,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=mtp_past_key_values,
            cache_metadata=cache_metadata,
            frequencies=frequencies,
            attention_mask=attention_mask,
        )

    def compute_mtp_logits(self, mtp_output: Qwen3_5MTPOutput) -> jax.Array:
        """Project MTP hidden states through the shared LM head."""
        # prepare_lm_head_inputs is a BaseCausalLMModule method; the vision-language
        # MRO lacks it, so fall back to the raw hidden state on that path.
        h = mtp_output.last_hidden_state
        if hasattr(self, "prepare_lm_head_inputs"):
            h = self.prepare_lm_head_inputs(h)
        return self.compute_lm_logits(h)

    def compute_mtp_chain(
        self,
        outputs: MoeCausalLMOutput,
        input_ids: jax.Array,
        n_steps: int,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array | None:
        """Recursively apply the (depth-1) MTP head ``n_steps`` times, teacher-forced.

        This mirrors how the inference drafter chains the inline MTP block to draft
        more than one token ahead (feed the block's output hidden + the next token
        back in). Step ``k`` (1-indexed) fuses the *previous* MTP hidden state with
        the embedding of the ground-truth token at offset ``k`` and predicts the
        token at position ``t + k + 1``:

            h^0 = outputs.last_hidden_state
            h^k = MTP(prev=h^{k-1}, next_embed=Emb(input_ids shifted left by k))
            logits^k = LMHead(h^k)            # predicts x_{t+k+1}

        Teacher-forcing (feeding ground-truth tokens, like DeepSeek-V3's MTP
        training) keeps it fully differentiable. ``n_steps == 1`` reproduces
        :meth:`compute_mtp_outputs` + :meth:`compute_mtp_logits` exactly.

        Args:
            outputs: Main causal-LM forward output (``last_hidden_state`` required).
            input_ids: ``(B, S)`` token IDs of the main forward pass.
            n_steps: Number of recursive draft steps (>= 1).
            attention_mask: Optional padding mask.

        Returns:
            ``(n_steps, B, S, vocab)`` MTP logits — entry ``k-1`` predicts
            ``input_ids[t + k + 1]`` — or ``None`` when the model has no MTP head.
        """
        if self.mtp is None or n_steps < 1:
            return None
        b = input_ids.shape[0]
        embed = self.model.get_embedding()
        frequencies = getattr(self.model, "frequencies", None)
        prev_hidden = outputs.last_hidden_state
        step_logits: list[jax.Array] = []
        for k in range(1, n_steps + 1):
            shifted = jnp.concatenate(
                [input_ids[:, k:], jnp.zeros((b, k), dtype=input_ids.dtype)],
                axis=-1,
            )
            mtp_out = self.mtp(
                prev_hidden_states=prev_hidden,
                next_token_embeds=embed(shifted.astype("i4")),
                frequencies=frequencies,
                attention_mask=attention_mask,
            )
            prev_hidden = mtp_out.last_hidden_state
            step_logits.append(self.compute_mtp_logits(mtp_out))
        return jnp.stack(step_logits, axis=0)

    def compute_mtp_loss(
        self,
        mtp_logits: jax.Array,
        labels: jax.Array,
        attention_mask: jax.Array | None = None,
        ignore_index: int = -100,
    ) -> jax.Array:
        """Compute MTP cross-entropy loss for the multimodal model.

        See :meth:`Qwen3_5ForCausalLM.compute_mtp_loss` for semantics.
        For multimodal inputs the caller should additionally mask out
        non-text positions (image/video placeholders) by setting their
        labels to ``ignore_index`` before invoking this method.
        """
        batch_size = labels.shape[0]
        pad = jnp.full((batch_size, 2), ignore_index, dtype=labels.dtype)
        shifted_labels = jnp.concatenate([labels[:, 2:], pad], axis=-1)
        if attention_mask is not None:
            mask_shifted = jnp.concatenate(
                [attention_mask[:, 2:], jnp.zeros((batch_size, 2), dtype=attention_mask.dtype)],
                axis=-1,
            )
            shifted_labels = jnp.where(mask_shifted.astype(jnp.bool_), shifted_labels, ignore_index)
        logits = mtp_logits.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, jnp.maximum(shifted_labels, 0)[..., None], axis=-1).squeeze(-1)
        valid = (shifted_labels != ignore_index).astype(jnp.float32)
        loss_sum = jnp.sum(nll * valid)
        loss_count = jnp.maximum(jnp.sum(valid), 1.0)
        return loss_sum / loss_count

    def _maybe_add_mtp_aux_loss(self, outputs, input_ids, attention_mask, mode):
        """Fold the MTP CE loss into ``outputs.aux_loss`` during training.

        Multimodal counterpart of
        :meth:`Qwen3_5ForCausalLM._maybe_add_mtp_aux_loss`. ``VLMCausalLMOutput``
        carries an ``aux_loss`` field that the trainer's ``compute_loss``
        adds to the main loss, so routing the MTP loss here makes the
        MTP head train during vision-language fine-tuning. Gating rules
        are identical (MTP head present, ``mtp_loss_coef > 0``,
        ``input_ids`` available, training/eval mode).
        """
        if self.mtp is None or input_ids is None:
            return outputs
        coef = float(getattr(self.config.text_config, "mtp_loss_coef", 0.0))
        if coef <= 0.0:
            return outputs
        if mode in (common_types.MODE_DECODE, common_types.MODE_PREFILL, common_types.MODE_INSERT):
            return outputs
        if getattr(outputs, "last_hidden_state", None) is None:
            return outputs
        mtp_output = self.compute_mtp_outputs(outputs, input_ids=input_ids, attention_mask=attention_mask, mode=mode)
        if mtp_output is None:
            return outputs
        mtp_logits = self.compute_mtp_logits(mtp_output)
        mtp_loss = self.compute_mtp_loss(mtp_logits, input_ids, attention_mask) * coef
        existing = getattr(outputs, "aux_loss", None)
        new_aux = mtp_loss if existing is None else existing + mtp_loss
        # Expose the MTP logits (already computed here) so a distillation/training step
        # can supervise the MTP head with a teacher distribution without recomputing the
        # large-vocab projection. Reused by the offline DistillationTrainer.
        return outputs.replace(aux_loss=new_aux, mtp_logits=mtp_logits)

    def forward(self, *args, **kwargs):
        """Vision-language forward with the MTP auxiliary loss folded in.

        Delegates to :meth:`BaseVisionLanguageModule.forward`, then —
        during training/eval with an MTP head present — adds the MTP
        cross-entropy loss to ``outputs.aux_loss`` so the trainer
        trains the MTP head. See :meth:`_maybe_add_mtp_aux_loss`.
        """
        outputs = super().forward(*args, **kwargs)
        if self.mtp is None:
            return outputs
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        mode = kwargs.get("mode")
        past_key_values = kwargs.get("past_key_values")
        cache_metadata = kwargs.get("cache_metadata")
        if input_ids is None and args:
            input_ids = args[0]
        if mode is None and (past_key_values is not None or cache_metadata is not None):
            inputs_embeds = kwargs.get("inputs_embeds")
            seq_len = None
            if input_ids is not None:
                seq_len = input_ids.shape[1]
            elif inputs_embeds is not None:
                seq_len = inputs_embeds.shape[1]
            mode = common_types.MODE_DECODE if seq_len == 1 else common_types.MODE_PREFILL
        return self._maybe_add_mtp_aux_loss(outputs, input_ids, attention_mask, mode)

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
