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

"""DSpark drafter modules.

DSpark predicts multiple future-token blocks in parallel from cached target
hidden states.  Each sampled anchor creates a draft block whose queries attend
to target context before the anchor and to draft positions in the same block.
The model can add low-rank Markov logit bias and confidence logits for
acceptance calibration.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from jaxtyping import Array, Bool, Float, Int
from spectrax import nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.spec_decode import SpecDecodeBase
from easydel.layers import ColumnParallelLinear, Embed, RMSNorm
from easydel.modules._drafting import (
    AcceptRatePredictor,
    DraftAttention,
    DraftMLP,
    DraftModelOutput,
    build_eval_mask,
    build_markov_head,
    causal_attention_mask,
    create_dspark_attention_mask,
    create_noise_ids,
    expand_attention_mask,
    extract_context_feature,
    gather_sequence,
    make_position_ids,
    sample_anchor_positions,
)

from .dspark_configuration import DSparkConfig

DSparkForwardOutput = DraftModelOutput


class DSparkAttention(DraftAttention):
    """Attention used by DSpark and DFlash block drafters.

    Queries come from draft/noise embeddings.  Keys and values are the
    concatenation of projected target context and the same draft sequence,
    masked so a block sees context before its anchor and draft positions from
    only that block.
    """

    def __init__(
        self,
        config: DSparkConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize DSpark attention with drafter hidden size as Q/KV input size.

        Args:
            config: DSpark model configuration.
            dtype: Activation dtype. Defaults to ``jnp.bfloat16``.
            param_dtype: Parameter dtype. Defaults to ``jnp.bfloat16``.
            precision: JAX matmul precision. Defaults to ``None``.
            rngs: SpectraX random number generators.

        Returns:
            None.
        """
        super().__init__(
            config,
            q_input_size=config.hidden_size,
            kv_input_size=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )


class DSparkDecoderLayer(spx.Module):
    """DSpark block-refinement layer.

    This is the transformer layer applied to the flattened draft blocks.  It
    performs DSpark's target-context cross/block self-attention followed by a
    residual SwiGLU MLP.
    """

    def __init__(
        self,
        config: DSparkConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the block-refinement layer (RMSNorm, attention, MLP).

        Args:
            config: DSpark model configuration.
            dtype: Activation dtype. Defaults to ``jnp.bfloat16``.
            param_dtype: Parameter dtype. Defaults to ``jnp.bfloat16``.
            precision: JAX matmul precision. Defaults to ``None``.
            rngs: SpectraX random number generators.

        Returns:
            None.
        """
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.self_attn = DSparkAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = DraftMLP(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        target_hidden_states: Float[Array, "batch ctx hidden"],
        attention_mask: Bool[Array, "batch seq ctx_seq"] | None,
        q_position_ids: Int[Array, "batch seq"],
        kv_position_ids: Int[Array, "batch ctx_seq"],
        frequencies: Float[Array, "seq head_dim"] | None = None,
    ) -> Float[Array, "batch seq hidden"]:
        """Refine draft-block states against projected target context.

        Applies pre-attention RMSNorm, DSpark attention over concatenated target
        and draft states, post-attention RMSNorm, and a residual SwiGLU MLP.

        Args:
            hidden_states: Draft hidden states of shape ``[batch, seq, hidden]``.
            target_hidden_states: Projected target context of shape ``[batch, ctx, hidden]``.
            attention_mask: Boolean mask of shape ``[batch, seq, ctx_seq]`` or ``None``.
            q_position_ids: Query position IDs of shape ``[batch, seq]``.
            kv_position_ids: Key/value position IDs of shape ``[batch, ctx_seq]``.
            frequencies: Optional RoPE frequencies of shape ``[seq, head_dim]``.

        Returns:
            Float[Array, "batch seq hidden"]: Refined hidden states after attention and MLP.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        key_value_states = jnp.concatenate((target_hidden_states, hidden_states), axis=1)
        hidden_states = residual + self.self_attn(
            hidden_states,
            key_value_states,
            attention_mask,
            q_position_ids=q_position_ids,
            kv_position_ids=kv_position_ids,
            frequencies=frequencies,
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


@register_module(TaskType.BASE_MODULE, config=DSparkConfig, model_type="dspark")
class DSparkModel(EasyDeLBaseModule, SpecDecodeBase):
    """DeepSpec-compatible DSpark block drafter.

    The public forward supports two modes.  Without `loss_mask` it returns
    sequence logits shaped `[batch, seq, vocab]`, which is useful for generic
    speculative trainers.  With `loss_mask` or `return_block_outputs=True`, it
    follows DeepSpec's DSpark training path and returns block logits, target
    IDs, evaluation masks, anchor positions, optional Markov-adjusted logits,
    and optional confidence logits.

    As a :class:`SpecDecodeBase`, DSpark is a runner-native drafter: the shared
    ``draft(...)`` tail calls :meth:`_draft_logits`, which runs the sequence-mode
    forward. ``DFlashModel`` subclasses this class and inherits the drafter
    surface unchanged.
    """

    _task_type = TaskType.BASE_MODULE
    _model_type = "dspark"
    _config_class = DSparkConfig

    def _draft_logits(
        self,
        input_ids: Int[Array, "batch seq"],
        *,
        target_hidden_states: tp.Any | None,
        position_ids: Int[Array, "batch seq"] | None = None,
        target_kv_cache: tp.Any | None = None,
        attention_mask: tp.Any | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Produce DSpark draft logits + hidden rows for :meth:`SpecDecodeBase.draft`.

        Runs the sequence-mode :meth:`forward` (no ``loss_mask`` /
        ``return_block_outputs``) so it returns ``[batch, seq, vocab]`` logits
        instead of the block-mode training output.

        Args:
            input_ids: ``(batch, seq)`` seed/verified tokens.
            target_hidden_states: Target hidden state(s) to project (single
                layer or pre-concatenated target features).
            position_ids: Ignored (sequence-mode forward builds its own).
            target_kv_cache: Ignored (DSpark does not cross-attend to target K/V).
            attention_mask: Optional attention mask.

        Returns:
            Tuple ``(logits, hidden_states)``.
        """
        del position_ids, target_kv_cache
        out = self.forward(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
        )
        return out.logits, out.hidden_states

    def __init__(
        self,
        config: DSparkConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build target projection, draft-block transformer, Markov head, and confidence head.

        Args:
            config: DSpark model configuration.
            dtype: Activation dtype. Defaults to ``jnp.bfloat16``.
            param_dtype: Parameter dtype. Defaults to ``jnp.bfloat16``.
            precision: JAX matmul precision. Defaults to ``None``.
            rngs: SpectraX random number generators.

        Returns:
            None.
        """
        super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        with self.assign_layer_stage(0, total_layers=self.config.num_hidden_layers):
            self.embed_tokens = Embed(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=jax.nn.initializers.normal(config.initializer_range),
                rngs=rngs,
            )
            self.fc = ColumnParallelLinear(
                len(config.target_layer_ids) * config.target_hidden_size,
                config.hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                rngs=rngs,
            )
            self.hidden_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=self.config.num_hidden_layers):
                self.layers.append(
                    DSparkDecoderLayer(
                        config=config,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        with self.assign_layer_stage(self.config.num_hidden_layers - 1, total_layers=self.config.num_hidden_layers):
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
            if not bool(config.tie_word_embeddings):
                self.lm_head = ColumnParallelLinear(
                    config.hidden_size,
                    config.vocab_size,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    rngs=rngs,
                )
        self.markov_head = build_markov_head(config, dtype, param_dtype, precision, rngs)
        self.confidence_head = None
        if bool(config.enable_confidence_head):
            input_dim = config.hidden_size + (config.markov_rank if config.confidence_head_with_markov else 0)
            self.confidence_head = AcceptRatePredictor(
                input_dim,
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    @property
    def trainable_selector(self) -> spx.SelectorSugar:
        """Return the DSpark trainable selector, optionally freezing copied IO weights."""
        selector = spx.select().variables("parameters")
        if bool(getattr(self.config, "freeze_embeddings_and_lm_head", False)):
            frozen = spx.select().variables("parameters").at_path("embed_tokens.**", "lm_head.**")
            selector = selector - frozen
        return selector

    @staticmethod
    def _copy_parameter_value(dst: tp.Any, src: tp.Any, *, name: str) -> None:
        """Copy a SpectraX parameter value with an early shape check."""
        if not hasattr(dst, "value") or not hasattr(src, "value"):
            raise TypeError(f"{name} must be a SpectraX-style parameter with a `.value` attribute.")
        dst_shape = tuple(dst.value.shape)
        src_shape = tuple(src.value.shape)
        if dst_shape != src_shape:
            raise ValueError(f"{name} shape mismatch: DSpark has {dst_shape}, target has {src_shape}.")
        dst.value = src.value.astype(dst.value.dtype)

    def initialize_embeddings_and_head_from_target(self, target_model: tp.Any) -> None:
        """Initialize DSpark token embeddings and LM head from a target model.

        DeepSpec initializes these two modules from the frozen target and then
        leaves them out of the optimized drafter state.  This helper performs the
        same copy for EasyDeL target modules that expose the conventional
        ``get_input_embeddings`` and ``get_lm_head`` accessors.
        """
        get_embeddings = getattr(target_model, "get_input_embeddings", None)
        if not callable(get_embeddings):
            get_embeddings = getattr(target_model, "get_embedding", None)
        get_lm_head = getattr(target_model, "get_lm_head", None)
        if not callable(get_embeddings) or not callable(get_lm_head):
            raise TypeError("Target model must expose `get_input_embeddings`/`get_embedding` and `get_lm_head`.")

        target_embeddings = get_embeddings()
        target_lm_head = get_lm_head()
        self._copy_parameter_value(self.embed_tokens.weight, target_embeddings.weight, name="embed_tokens.weight")
        if bool(self.config.tie_word_embeddings):
            return
        self._copy_parameter_value(self.lm_head.weight, target_lm_head.weight, name="lm_head.weight")

    def project_target_hidden_states(self, target_hidden_states: tp.Any) -> jax.Array:
        """Extract configured target layers and project them into DSpark hidden space.

        Args:
            target_hidden_states: Target model hidden states (arbitrary pytree format
                accepted by ``extract_context_feature``).

        Returns:
            jax.Array: Projected and normalized context features of shape
            ``[batch, seq, hidden_size]``.
        """
        context = extract_context_feature(
            target_hidden_states,
            self.config.target_layer_ids,
            target_hidden_size=self.config.target_hidden_size,
            allow_embedding=True,
        )
        return self.hidden_norm(self.fc(context))

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Project drafter hidden states to vocabulary logits.

        Args:
            hidden_states: Hidden states of shape ``[..., hidden_size]``.

        Returns:
            jax.Array: Vocabulary logits of shape ``[..., vocab_size]``.
        """
        if bool(self.config.tie_word_embeddings):
            return self.embed_tokens.attend(hidden_states)
        return self.lm_head(hidden_states)

    def _target_logits(self, target_last_hidden_states: jax.Array) -> jax.Array:
        """Project aligned target final hidden states through the drafter LM head.

        Args:
            target_last_hidden_states: Target final hidden states of shape
                ``[batch, seq, hidden_size]``.

        Returns:
            jax.Array: Target vocabulary logits of shape ``[batch, seq, vocab_size]``.

        Raises:
            ValueError: If the last dimension of ``target_last_hidden_states`` does not
                match ``config.hidden_size``.
        """
        if int(target_last_hidden_states.shape[-1]) != int(self.config.hidden_size):
            raise ValueError("DSpark target logits require target hidden width to match `hidden_size`.")
        if bool(self.config.tie_word_embeddings):
            return self.embed_tokens.attend(target_last_hidden_states)
        return self.lm_head(target_last_hidden_states)

    def predict_confidence_step(
        self, hidden_states: jax.Array, prev_token_ids: jax.Array | None = None
    ) -> jax.Array | None:
        """Predict per-position acceptance confidence logits when enabled.

        Args:
            hidden_states: Hidden states of shape ``[batch, seq, hidden_size]`` or
                ``[batch, blocks, block_size, hidden_size]``.
            prev_token_ids: Previous token IDs for Markov feature lookup, of shape
                matching ``hidden_states`` prefix dimensions. ``None`` when Markov is
                not used.

        Returns:
            jax.Array | None: Confidence logits if the confidence head is enabled,
            otherwise ``None``.

        Raises:
            ValueError: If ``config.confidence_head_with_markov`` is ``True`` but
                ``markov_head`` is ``None`` or ``prev_token_ids`` is ``None``.
        """
        if self.confidence_head is None:
            return None
        features = hidden_states
        if self.config.confidence_head_with_markov:
            if self.markov_head is None or prev_token_ids is None:
                raise ValueError("confidence_head_with_markov requires markov_head and prev_token_ids.")
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).astype(hidden_states.dtype)
            features = jnp.concatenate((hidden_states, prev_embeddings), axis=-1)
        return tp.cast(jax.Array, self.confidence_head(features))

    def _forward_backbone(
        self,
        *,
        noise_embedding: jax.Array,
        target_hidden_states: jax.Array,
        attention_mask: jax.Array | None,
        q_position_ids: jax.Array,
        kv_position_ids: jax.Array,
    ) -> jax.Array:
        """Run the shared DSpark/DFlash transformer over draft states.

        Args:
            noise_embedding: Draft token embeddings of shape ``[batch, seq, hidden_size]``.
            target_hidden_states: Projected target context of shape ``[batch, ctx, hidden_size]``.
            attention_mask: Attention mask of shape ``[batch, seq, ctx_seq]`` or ``None``.
            q_position_ids: Query position IDs of shape ``[batch, seq]``.
            kv_position_ids: Key/value position IDs of shape ``[batch, ctx_seq]``.

        Returns:
            jax.Array: Normalized hidden states of shape ``[batch, seq, hidden_size]``.
        """
        hidden_states = noise_embedding
        frequencies = self.config.get_basic_frequencies()
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                q_position_ids=q_position_ids,
                kv_position_ids=kv_position_ids,
                frequencies=frequencies,
            )
        return self.norm(hidden_states)

    def _full_sequence_forward(
        self,
        input_ids: jax.Array,
        target_hidden_states: jax.Array,
        attention_mask: jax.Array | None,
    ) -> DSparkForwardOutput:
        """Run a sequence-shaped drafter forward for generic trainer use.

        Constructs a causal attention mask over concatenated target context and
        draft tokens, runs the transformer backbone, and computes logits.

        Args:
            input_ids: Input token IDs of shape ``[batch, seq]``.
            target_hidden_states: Projected target context of shape ``[batch, ctx, hidden_size]``.
            attention_mask: Optional boolean attention mask of shape ``[batch, seq]``.

        Returns:
            DSparkForwardOutput: Output containing hidden states, draft logits,
            sequence logits, and optional confidence logits.
        """
        hidden_states = self.embed_tokens(input_ids.astype(jnp.int32))
        batch_size, seq_len = input_ids.shape
        context_position_ids = make_position_ids(inputs_embeds=target_hidden_states)
        draft_position_ids = make_position_ids(input_ids=input_ids)
        kv_position_ids = jnp.concatenate((context_position_ids, draft_position_ids), axis=1)
        mask = causal_attention_mask(batch_size, seq_len, target_hidden_states.shape[1] + seq_len)
        if attention_mask is not None:
            context_mask = expand_attention_mask(attention_mask, q_len=seq_len, kv_len=target_hidden_states.shape[1])
            draft_mask = expand_attention_mask(attention_mask, q_len=seq_len, kv_len=seq_len)
            mask = mask & jnp.concatenate((context_mask, draft_mask), axis=-1)
        hidden_states = self._forward_backbone(
            noise_embedding=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=mask,
            q_position_ids=draft_position_ids,
            kv_position_ids=kv_position_ids,
        )
        draft_logits = self.compute_logits(hidden_states)
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_logits(draft_logits, token_ids=input_ids, hidden_states=hidden_states)
        confidence_logits = self.predict_confidence_step(hidden_states, input_ids)
        return DSparkForwardOutput(
            hidden_states=hidden_states,
            draft_logits=draft_logits,
            logits=draft_logits,
            confidence_logits=confidence_logits,
        )

    def _block_forward(
        self,
        input_ids: jax.Array,
        target_hidden_states: jax.Array,
        loss_mask: jax.Array,
        target_last_hidden_states: jax.Array | None,
        target_logits: jax.Array | None,
        anchor_positions: jax.Array | None,
        block_keep_mask: jax.Array | None,
    ) -> DSparkForwardOutput:
        """Run the DeepSpec block-drafter training forward.

        Samples or consumes anchor positions, creates masked teacher-forced draft
        blocks, applies DSpark block attention, aligns targets for each block
        position, and returns the block supervision tensors used by DSpark/DFlash
        losses.

        Args:
            input_ids: Input token IDs of shape ``[batch, seq]``.
            target_hidden_states: Projected target context of shape ``[batch, ctx, hidden_size]``.
            loss_mask: Float mask of shape ``[batch, seq]`` indicating positions
                eligible for anchor sampling.
            target_last_hidden_states: Optional target final hidden states of shape
                ``[batch, seq, hidden_size]`` for computing aligned target logits.
            target_logits: Optional pre-computed target logits of shape
                ``[batch, seq, vocab_size]``.
            anchor_positions: Optional pre-sampled anchor positions of shape
                ``[batch, anchors]``. ``None`` triggers sampling.
            block_keep_mask: Optional boolean mask of shape ``[batch, anchors]``
                selecting which blocks to retain. ``None`` triggers creation.

        Returns:
            DSparkForwardOutput: Block supervision tensors including block logits,
            target IDs, evaluation masks, anchor positions, confidence logits, and
            aligned target logits.
        """
        if anchor_positions is None:
            anchor_positions, sampled_keep = sample_anchor_positions(loss_mask, num_anchors=self.config.num_anchors)
            block_keep_mask = sampled_keep if block_keep_mask is None else block_keep_mask
        if block_keep_mask is None:
            block_keep_mask = anchor_positions < (input_ids.shape[1] - 1)
        noise_ids = create_noise_ids(
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.config.mask_token_id,
            block_size=self.config.block_size,
        )
        noise_embedding = self.embed_tokens(noise_ids)
        batch_size, seq_len = input_ids.shape
        context_position_ids = make_position_ids(inputs_embeds=target_hidden_states)
        offsets = jnp.arange(self.config.block_size, dtype=jnp.int32).reshape(1, 1, -1)
        draft_position_ids = (anchor_positions[:, :, None] + offsets).reshape(batch_size, -1)
        kv_position_ids = jnp.concatenate((context_position_ids, draft_position_ids), axis=1)
        mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.config.block_size,
        )
        output_hidden = self._forward_backbone(
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=mask,
            q_position_ids=draft_position_ids,
            kv_position_ids=kv_position_ids,
        )

        num_blocks = anchor_positions.shape[1]
        output_hidden_4d = output_hidden.reshape(batch_size, num_blocks, self.config.block_size, -1)
        label_offsets = jnp.arange(1, self.config.block_size + 1, dtype=jnp.int32).reshape(1, 1, -1)
        label_indices = anchor_positions[:, :, None] + label_offsets
        safe_label_indices = jnp.clip(label_indices, 0, seq_len - 1)
        target_ids = jnp.take_along_axis(
            jnp.broadcast_to(input_ids[:, None, :], (batch_size, num_blocks, seq_len)),
            safe_label_indices,
            axis=2,
        )
        eval_mask = build_eval_mask(
            input_ids=input_ids,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = jnp.take_along_axis(input_ids, anchor_positions, axis=1)
        prev_token_ids = jnp.concatenate((anchor_token_ids[:, :, None], target_ids[:, :, :-1]), axis=-1)
        draft_logits = self.compute_logits(output_hidden).reshape(batch_size, num_blocks, self.config.block_size, -1)
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_logits(
                draft_logits, token_ids=prev_token_ids, hidden_states=output_hidden_4d
            )
        confidence_logits = self.predict_confidence_step(output_hidden_4d, prev_token_ids)

        aligned_target_logits = None
        target_pred_indices = jnp.clip(safe_label_indices - 1, 0, seq_len - 1)
        if target_logits is not None:
            aligned_target_logits = gather_sequence(target_logits, target_pred_indices)
        elif target_last_hidden_states is not None and int(target_last_hidden_states.shape[-1]) == int(
            self.config.hidden_size
        ):
            aligned_target_hidden = gather_sequence(target_last_hidden_states, target_pred_indices)
            aligned_target_logits = self._target_logits(aligned_target_hidden)
        return DSparkForwardOutput(
            hidden_states=output_hidden,
            draft_logits=draft_logits,
            block_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            anchor_positions=anchor_positions,
            confidence_logits=confidence_logits,
            aligned_target_logits=aligned_target_logits,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: tp.Any,
        loss_mask: Float[Array, "batch seq"] | None = None,
        target_last_hidden_states: Float[Array, "batch seq hidden"] | None = None,
        target_last_hidden_state: Float[Array, "batch seq hidden"] | None = None,
        target_logits: Float[Array, "batch seq vocab"] | None = None,
        attention_mask: Bool[Array, "batch seq"] | None = None,
        anchor_positions: Int[Array, "batch anchors"] | None = None,
        block_keep_mask: Bool[Array, "batch anchors"] | None = None,
        return_block_outputs: bool = False,
        **kwargs,
    ) -> DSparkForwardOutput:
        """Run DSpark in sequence mode or DeepSpec block mode.

        Passing ``loss_mask``, ``anchor_positions``, or ``return_block_outputs=True``
        selects block mode. Otherwise the method returns sequence logits for the
        same input IDs, using projected target hidden states as context.

        Args:
            input_ids: Input token IDs of shape ``[batch, seq]``.
            target_hidden_states: Target model hidden states (pytree) to be projected.
            loss_mask: Float mask for anchor sampling in block mode. ``None`` for
                sequence mode.
            target_last_hidden_states: Target final hidden states for aligned target
                logits. ``None`` to skip.
            target_last_hidden_state: Backward-compatible alias for
                ``target_last_hidden_states``.
            target_logits: Pre-computed target logits. ``None`` to compute from
                hidden states.
            attention_mask: Boolean attention mask for sequence mode. ``None`` for
                full context.
            anchor_positions: Pre-sampled anchor positions for block mode. ``None`` to
                sample.
            block_keep_mask: Block selection mask for block mode. ``None`` to infer.
            return_block_outputs: Force block-mode output even without ``loss_mask``.
            **kwargs: Absorbed and discarded (reserved for API compatibility).

        Returns:
            DSparkForwardOutput: Either sequence-mode or block-mode output depending
            on the active path.
        """
        del kwargs
        target_context = self.project_target_hidden_states(target_hidden_states)
        target_last = target_last_hidden_states if target_last_hidden_states is not None else target_last_hidden_state
        if return_block_outputs or loss_mask is not None or anchor_positions is not None:
            if loss_mask is None:
                loss_mask = jnp.ones_like(input_ids, dtype=jnp.float32)
            return self._block_forward(
                input_ids, target_context, loss_mask, target_last, target_logits, anchor_positions, block_keep_mask
            )
        output = self._full_sequence_forward(input_ids, target_context, attention_mask)
        return DSparkForwardOutput(
            hidden_states=output.hidden_states,
            draft_logits=output.draft_logits,
            logits=output.logits,
            target_logits=target_logits
            if target_logits is not None
            else self._target_logits(target_last)
            if target_last is not None and int(target_last.shape[-1]) == int(self.config.hidden_size)
            else None,
            confidence_logits=output.confidence_logits,
        )


__all__ = ("DSparkAttention", "DSparkDecoderLayer", "DSparkForwardOutput", "DSparkModel")
