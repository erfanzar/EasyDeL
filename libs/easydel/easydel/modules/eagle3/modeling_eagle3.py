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

"""EAGLE3 drafter modules.

The model consumes target-model hidden states, extracts the configured five
decoder layers, projects their concatenation into drafter space, and refines it
with a shallow Qwen-style stack conditioned on the current token embeddings.
It exposes both hidden states and vocabulary logits so it can be used by the
speculative-decoding trainer or by EAGLE-style draft sampling.
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
    DraftAttention,
    DraftMLP,
    DraftModelOutput,
    causal_attention_mask,
    expand_attention_mask,
    extract_context_feature,
    make_position_ids,
)

from .eagle3_configuration import Eagle3Config

Eagle3ForwardOutput = DraftModelOutput


class Eagle3Attention(DraftAttention):
    """Attention for EAGLE3's mixed token/target representation.

    EAGLE3 forms each attention input by concatenating the normalized token
    embedding with the normalized projected target feature.  That doubles the
    Q/K/V input width while the output remains the drafter hidden size.
    """

    def __init__(
        self,
        config: Eagle3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize EAGLE3 attention with doubled input width.

        Concatenated token and target features feed Q/K/V, so the query and key/value
        input sizes are each ``2 * config.hidden_size``. Output width remains the
        drafter ``hidden_size``.

        Args:
            config: EAGLE3 configuration object.
            dtype: Activation dtype for the module. Defaults to ``jnp.bfloat16``.
            param_dtype: Parameter dtype for the module. Defaults to ``jnp.bfloat16``.
            precision: JAX precision descriptor for matmuls. Defaults to ``None``.
            rngs: SpectraX random number generators.

        Returns:
            None.
        """
        super().__init__(
            config,
            q_input_size=2 * config.hidden_size,
            kv_input_size=2 * config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )


class Eagle3DecoderLayer(spx.Module):
    """EAGLE3 refinement block.

    The block normalizes token embeddings and projected target features
    separately, concatenates them for self-attention, then applies a residual
    SwiGLU MLP.  This follows the DeepSpec/SpecForge EAGLE3 layer shape rather
    than a normal target-model decoder layer.
    """

    def __init__(
        self,
        config: Eagle3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the EAGLE3 refinement block.

        Creates separate RMSNorms for token embeddings and target features, an
        ``Eagle3Attention`` that consumes their concatenation, and a ``DraftMLP``
        SwiGLU residual block.

        Args:
            config: EAGLE3 configuration object.
            dtype: Activation dtype for the module. Defaults to ``jnp.bfloat16``.
            param_dtype: Parameter dtype for the module. Defaults to ``jnp.bfloat16``.
            precision: JAX precision descriptor for matmuls. Defaults to ``None``.
            rngs: SpectraX random number generators.

        Returns:
            None.
        """
        self.hidden_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.self_attn = Eagle3Attention(
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
        input_embeds: Float[Array, "batch seq hidden"],
        hidden_states: Float[Array, "batch seq hidden"],
        attention_mask: Bool[Array, "batch seq seq"] | None,
        position_ids: Int[Array, "batch seq"],
        frequencies: Float[Array, "seq head_dim"] | None = None,
    ) -> Float[Array, "batch seq hidden"]:
        """Refine projected target features under a causal EAGLE3 mask.

        Separately normalizes ``input_embeds`` and ``hidden_states``, concatenates
        them along the feature axis for self-attention, then applies a residual
        SwiGLU MLP.

        Args:
            input_embeds: Token embeddings of shape ``[batch, seq, hidden]``.
            hidden_states: Projected target features of shape ``[batch, seq, hidden]``.
            attention_mask: Optional boolean mask of shape ``[batch, seq, seq]``.
                ``True`` positions are allowed to attend.
            position_ids: Integer position indices of shape ``[batch, seq]``.
            frequencies: Optional precomputed RoPE frequencies of shape
                ``[seq, head_dim]``.

        Returns:
            Refined hidden states of shape ``[batch, seq, hidden]``.
        """
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_embeds = self.input_layernorm(input_embeds)
        mixed_states = jnp.concatenate((input_embeds, hidden_states), axis=-1)
        hidden_states = residual + self.self_attn(
            mixed_states,
            mixed_states,
            attention_mask,
            q_position_ids=position_ids,
            kv_position_ids=position_ids,
            frequencies=frequencies,
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


@register_module(TaskType.BASE_MODULE, config=Eagle3Config, model_type="eagle3")
class Eagle3Model(EasyDeLBaseModule, SpecDecodeBase):
    """DeepSpec-compatible EAGLE3 draft model.

    Inputs are either pre-projected drafter hidden states or target hidden
    states from the base model.  When target states are provided, EAGLE3 selects
    exactly the configured five decoder layers, concatenates them, projects to
    `hidden_size`, and runs the EAGLE3 refinement blocks.  `return_logits=True`
    returns `Eagle3ForwardOutput` with drafter logits and optional target logits.

    As a :class:`SpecDecodeBase`, EAGLE3 is a runner-native drafter: the shared
    ``draft(...)`` tail calls :meth:`_draft_logits`, which delegates to
    ``forward(..., return_logits=True)``. EAGLE3 accepts a multi-token prefix,
    so ``supports_prefix_draft`` is ``True``.
    """

    _task_type = TaskType.BASE_MODULE
    _model_type = "eagle3"
    _config_class = Eagle3Config

    supports_prefix_draft = True

    def _draft_logits(
        self,
        input_ids: Int[Array, "batch seq"],
        *,
        target_hidden_states: tp.Any | None,
        position_ids: Int[Array, "batch seq"] | None = None,
        target_kv_cache: tp.Any | None = None,
        attention_mask: tp.Any | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Produce EAGLE3 draft logits + hidden rows for :meth:`SpecDecodeBase.draft`.

        Delegates to :meth:`forward` with ``return_logits=True`` and returns the
        drafter logits alongside the refined hidden states.

        Args:
            input_ids: ``(batch, seq)`` seed/verified tokens.
            target_hidden_states: Target hidden state(s) to project (single
                layer or pre-concatenated target features).
            position_ids: Optional absolute positions for the input rows.
            target_kv_cache: Ignored (EAGLE3 does not cross-attend to target K/V).
            attention_mask: Optional attention mask.

        Returns:
            Tuple ``(logits, hidden_states)``.
        """
        del target_kv_cache
        out = self.forward(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_logits=True,
        )
        return out.logits, out.hidden_states

    def __init__(
        self,
        config: Eagle3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build EAGLE3 embeddings, target projection, refinement layers, and LM head.

        Layers are registered with the EasyDeL stage-assignment system so that
        sharding and gradient-checkpointing policies are applied correctly.

        Args:
            config: EAGLE3 configuration object.
            dtype: Activation dtype for the module. Defaults to ``jnp.bfloat16``.
            param_dtype: Parameter dtype for the module. Defaults to ``jnp.bfloat16``.
            precision: JAX precision descriptor for matmuls. Defaults to ``None``.
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
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=self.config.num_hidden_layers):
                self.layers.append(
                    Eagle3DecoderLayer(
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

    def project_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
        """Project pre-concatenated target features when they are not already in drafter space.

        If the last dimension of ``hidden_states`` already matches the expected
        concatenated target size, the states are passed through ``self.fc``.

        Args:
            hidden_states: Array of shape
                ``[batch, seq, len(target_layer_ids) * target_hidden_size]`` or
                already-projected drafter states.

        Returns:
            Projected drafter hidden states of shape ``[batch, seq, hidden_size]``.
        """
        if int(hidden_states.shape[-1]) == len(self.config.target_layer_ids) * int(self.config.target_hidden_size):
            return self.fc(hidden_states)
        return hidden_states

    def project_target_hidden_states(self, target_hidden_states: tp.Any) -> jax.Array:
        """Extract the five configured target layers and project them to drafter width.

        Uses ``extract_context_feature`` to select the layers specified by
        ``config.target_layer_ids`` and concatenates them before projection.

        Args:
            target_hidden_states: Target model hidden states, either a tuple/list of
                per-layer arrays or a stacked array.

        Returns:
            Projected drafter hidden states of shape ``[batch, seq, hidden_size]``.
        """
        context = extract_context_feature(
            target_hidden_states,
            self.config.target_layer_ids,
            target_hidden_size=self.config.target_hidden_size,
            allow_embedding=False,
        )
        return self.fc(context)

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Normalize drafter states and project them to vocabulary logits.

        Applies the final RMSNorm and then the LM head (either tied embeddings or
        a separate ``ColumnParallelLinear``).

        Args:
            hidden_states: Final drafter hidden states of shape ``[batch, seq, hidden_size]``.

        Returns:
            Vocabulary logits of shape ``[batch, seq, vocab_size]``.
        """
        hidden_states = self.norm(hidden_states)
        if bool(self.config.tie_word_embeddings):
            return self.embed_tokens.attend(hidden_states)
        return self.lm_head(hidden_states)

    def target_logits_only(self, target_last_hidden_states: jax.Array) -> jax.Array:
        """Project target final hidden states through the drafter vocabulary head.

        Args:
            target_last_hidden_states: Target model final hidden states of shape
                ``[batch, seq, hidden_size]``.

        Returns:
            Vocabulary logits of shape ``[batch, seq, vocab_size]``.

        Raises:
            ValueError: If the last dimension of ``target_last_hidden_states`` does not
                match ``config.hidden_size``.
        """
        if int(target_last_hidden_states.shape[-1]) != int(self.config.hidden_size):
            raise ValueError("EAGLE3 target logits require target hidden width to match `hidden_size`.")
        if bool(self.config.tie_word_embeddings):
            return self.embed_tokens.attend(target_last_hidden_states)
        return self.lm_head(target_last_hidden_states)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"] | None = None,
        input_ids: Int[Array, "batch seq"] | None = None,
        input_embeds: Float[Array, "batch seq hidden"] | None = None,
        target_hidden_states: tp.Any | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        attention_mask: Bool[Array, "batch seq"] | Bool[Array, "batch seq seq"] | None = None,
        target_last_hidden_states: Float[Array, "batch seq hidden"] | None = None,
        target_last_hidden_state: Float[Array, "batch seq hidden"] | None = None,
        target_logits_only: bool = False,
        return_logits: bool = False,
        **kwargs,
    ) -> Float[Array, "batch seq hidden"] | Eagle3ForwardOutput | jax.Array:
        """Run an EAGLE3 forward pass.

        Inputs are either pre-projected drafter hidden states or raw target hidden
        states. When ``target_logits_only=True`` the model short-circuits to
        ``target_logits_only`` without running the drafter stack.

        Args:
            hidden_states: Either drafter-width states or concatenated target
                features shaped ``[batch, seq, len(target_layer_ids) * target_hidden_size]``.
            input_ids: Token IDs used to form EAGLE3 token embeddings. Required if
                ``input_embeds`` is not provided.
            input_embeds: Optional precomputed token embeddings of shape
                ``[batch, seq, hidden_size]``.
            target_hidden_states: Target model hidden-state tuple/list or layer-stacked
                array used when ``hidden_states`` is not provided.
            position_ids: Optional integer position indices of shape ``[batch, seq]``.
                If ``None``, auto-generated from ``input_ids`` or ``input_embeds``.
            attention_mask: Optional boolean mask. Can be ``[batch, seq]`` (padding mask)
                or ``[batch, seq, seq]`` (full 2-D mask). ``True`` means attend.
            target_last_hidden_states: Target model final hidden states for computing
                target-side logits.
            target_last_hidden_state: Deprecated alias for ``target_last_hidden_states``.
            target_logits_only: If ``True``, return only logits for target final hidden
                states, matching the DeepSpec helper path.
            return_logits: If ``True``, return ``Eagle3ForwardOutput`` with draft and
                target logits; otherwise return final drafter hidden states.
            **kwargs: Ignored. Reserved for interface compatibility.

        Returns:
            If ``return_logits=False``: final drafter hidden states of shape
            ``[batch, seq, hidden_size]``.
            If ``return_logits=True``: ``Eagle3ForwardOutput`` containing
            ``hidden_states``, ``draft_logits``, ``logits``, and optionally
            ``target_logits``.

        Raises:
            ValueError: If ``target_logits_only=True`` but ``target_last_hidden_states``
                (or ``target_last_hidden_state``) is ``None``.
            ValueError: If neither ``hidden_states`` nor ``target_hidden_states`` is provided.
            ValueError: If neither ``input_ids`` nor ``input_embeds`` is provided.
        """
        del kwargs
        target_last = target_last_hidden_states if target_last_hidden_states is not None else target_last_hidden_state
        if target_logits_only:
            if target_last is None:
                raise ValueError("`target_last_hidden_states` is required when `target_logits_only=True`.")
            return self.target_logits_only(target_last)
        if hidden_states is None:
            if target_hidden_states is None:
                raise ValueError("Eagle3Model requires `hidden_states` or `target_hidden_states`.")
            hidden_states = self.project_target_hidden_states(target_hidden_states)
        else:
            hidden_states = self.project_hidden_states(hidden_states)
        if input_embeds is None:
            if input_ids is None:
                raise ValueError("Eagle3Model requires `input_ids` or `input_embeds`.")
            input_embeds = self.embed_tokens(input_ids.astype(jnp.int32))
        if position_ids is None:
            position_ids = make_position_ids(input_ids=input_ids, inputs_embeds=input_embeds)

        batch_size, seq_len = hidden_states.shape[:2]
        mask = causal_attention_mask(batch_size, seq_len, seq_len)
        pad_mask = expand_attention_mask(attention_mask, q_len=seq_len, kv_len=seq_len)
        if pad_mask is not None:
            mask = mask & pad_mask
        frequencies = self.config.get_basic_frequencies()
        for layer in self.layers:
            hidden_states = tp.cast(jax.Array, layer(input_embeds, hidden_states, mask, position_ids, frequencies))
        if not return_logits:
            return hidden_states
        draft_logits = self.compute_logits(hidden_states)
        target_logits = self.target_logits_only(target_last) if target_last is not None else None
        return Eagle3ForwardOutput(
            hidden_states=hidden_states, draft_logits=draft_logits, logits=draft_logits, target_logits=target_logits
        )

    def compute_mtp_chain(
        self,
        outputs: Eagle3ForwardOutput,
        input_ids: jax.Array,
        n_steps: int,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Build a teacher-forced multi-token draft-logit chain.

        The speculative-decoding trainer expects draft logits shaped
        ``[draft_steps, batch, seq, vocab]``. EAGLE3's training objective is
        offset-based, so this helper shifts the base logits by each draft step.

        Args:
            outputs: ``Eagle3ForwardOutput`` containing draft logits.
            input_ids: Token IDs (currently ignored; reserved for future use).
            n_steps: Number of draft steps to generate.
            attention_mask: Attention mask (currently ignored; reserved for future use).

        Returns:
            Stacked logits of shape ``[n_steps, batch, seq, vocab_size]``.

        Raises:
            ValueError: If ``outputs`` does not contain ``draft_logits`` or ``logits``.
        """
        del input_ids, attention_mask
        logits = outputs.draft_logits if outputs.draft_logits is not None else outputs.logits
        if logits is None:
            raise ValueError("`outputs` must contain draft logits.")
        steps = []
        for offset in range(int(n_steps)):
            if offset == 0:
                steps.append(logits)
            else:
                tail = jnp.broadcast_to(logits[:, -1:, :], (logits.shape[0], offset, logits.shape[-1]))
                steps.append(jnp.concatenate((logits[:, offset:, :], tail), axis=1))
        return jnp.stack(steps, axis=0)


__all__ = ("Eagle3Attention", "Eagle3DecoderLayer", "Eagle3ForwardOutput", "Eagle3Model")
