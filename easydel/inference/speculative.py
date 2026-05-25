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
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: F821

"""Generic drafter interface + draft-then-verify primitives.

A drafter is any module that, given the target model's last hidden
state and the recent input tokens, produces ``N`` candidate next-token
IDs (with optional acceptance probabilities) for the target to verify
in a single forward pass.

Two drafter shapes are supported:

1. **Inline MTP drafter** — the drafter shares params with the
   target (DeepSeek-V3-style MTP head). Example: Qwen3.5's
   ``Qwen3_5ForCausalLM.compute_mtp_outputs(...)``. Each call returns
   a one-step look-ahead; multi-step drafts come from repeated calls
   with the drafter's own KV cache.

2. **Standalone drafter** — the drafter is a separate small model
   (Gemma4 Assistant). Speculative decoding requires the target's
   per-layer K/V tensors to be fed into the drafter's Q-only
   attention (cross-model KV sharing). The drafter outputs sparse
   centroid logits per step.

This module defines the :class:`DrafterProtocol` interface that
:func:`speculative_step` consumes, plus :func:`accept_or_reject`
which implements the standard speculative-decoding rejection sampling
correction. The full draft-then-verify loop with cache management
lives in the esurge runner; this file is the JAX-pure primitives.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from ejkernel.types import MaskInfo
from jaxtyping import Array, Float, Int
from spectrax import common_types

from easydel.caching import TransformerCache, TransformerCacheConfig, TransformerMetadata


@dataclass(frozen=True)
class DraftStep:
    """One step of drafter output.

    Attributes:
        token_ids: ``(batch,)`` proposed next token IDs.
        log_probs: ``(batch,)`` log-prob assigned by the drafter to
            ``token_ids``. Used for the rejection-sampling correction in
            :func:`accept_or_reject`. ``None`` for greedy drafters.
        full_log_probs: Optional ``(batch, vocab)`` full distribution over the
            vocabulary. Required if the verifier wants to do
            distribution-correct rejection sampling rather than the simple
            ratio test. ``None`` to skip the correction (uses greedy
            verification: accept if ``argmax(target) == draft``).
        hidden_states: Optional drafter hidden rows for this step. Inline MTP
            drafters reuse this as the next step's ``hidden_states``, matching
            vLLM's repeated-MTP proposal loop.
    """

    token_ids: Int[Array, "batch"]
    log_probs: Float[Array, "batch"] | None = None
    full_log_probs: Float[Array, "batch vocab"] | None = None
    hidden_states: Float[Array, "batch seq hidden"] | None = None


class _HiddenStateCarrier:
    """Minimal ``outputs``-shaped object exposing ``last_hidden_state``.

    The Qwen3.5 MTP path (``compute_mtp_outputs``) expects an object
    with a ``.last_hidden_state`` attribute. This carrier lets the
    drafter reuse that path while passing a bare hidden-state array.
    It is created transiently during JIT tracing (not a pytree leaf
    / not a jit input), so it adds no overhead.
    """

    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state: Array):
        """Store ``last_hidden_state`` for later ``.last_hidden_state`` access.

        Args:
            last_hidden_state: Hidden-state tensor to expose as the
                ``.last_hidden_state`` attribute on this carrier.
        """
        self.last_hidden_state = last_hidden_state


class DrafterProtocol(typing.Protocol):
    """Interface for any speculative-decode drafter.

    Concrete drafters wrap their model-specific forward path behind
    this protocol so the speculative-decode controller can be generic.
    """

    def reset(self, batch_size: int) -> None:
        """Reset the drafter's internal state for a new generation.

        Implementations should clear any KV cache, prefix state, or per-batch
        buffers. Called once per generation, not per step.

        Args:
            batch_size: Number of sequences in the upcoming generation.
        """
        ...

    def draft(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: Float[Array, "batch seq hidden"] | None = None,
        target_kv_cache: typing.Any | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        return_full_log_probs: bool = False,
    ) -> DraftStep:
        """Propose one next-token draft per batch element.

        Args:
            input_ids: The verified tokens so far (target context).
            target_hidden_states: Last hidden state of the target
                model, supplied for drafters that condition on it
                (DeepSeek-V3 MTP, Gemma4 Assistant). Inline drafters
                that share the target model receive this from the
                same forward pass.
            target_kv_cache: Target model's KV cache; required for
                drafters that share K/V with the target (Gemma4
                Assistant). ``None`` for drafters that maintain their
                own KV cache.
            position_ids: Absolute positions corresponding to the
                target hidden rows. Inline MTP drafters fuse each
                hidden row with the next-token embedding while keeping
                the hidden row's RoPE position.
            return_full_log_probs: Request a dense drafter
                distribution for sampled speculative decoding.

        Returns:
            :class:`DraftStep` with the proposed token IDs and (if
            supported) the distribution for rejection sampling.
        """
        ...




def accept_or_reject(
    draft_log_probs: Float[Array, "batch"],
    target_log_probs: Float[Array, "batch"],
    rng_key: jax.Array,
) -> Int[Array, "batch"]:
    """Standard speculative-decoding acceptance test.

    Accepts the draft token at probability ``min(1, p_target / p_draft)``.
    Returns a boolean accept mask as int32 ``(0, 1)``.

    Args:
        draft_log_probs: ``(batch,)`` log-prob the drafter assigned
            to its proposed token.
        target_log_probs: ``(batch,)`` log-prob the target model
            assigned to the same token under its own distribution.
        rng_key: PRNG key for the acceptance draw.

    Returns:
        ``(batch,)`` int32 mask, ``1`` for accepted tokens, ``0``
        for rejected. Per Leviathan et al., a rejected token's
        position should be re-sampled from
        ``norm(max(0, p_target - p_draft))``; that step is handled by
        the caller using ``full_log_probs`` from the :class:`DraftStep`.
    """
    log_ratio = target_log_probs - draft_log_probs
    accept_prob = jnp.minimum(1.0, jnp.exp(log_ratio))
    u = jax.random.uniform(rng_key, accept_prob.shape, dtype=jnp.float32)
    return (u < accept_prob).astype(jnp.int32)


def resample_rejected(
    target_log_probs: Float[Array, "batch vocab"],
    draft_log_probs: Float[Array, "batch vocab"],
    rng_key: jax.Array,
) -> Int[Array, "batch"]:
    """Sample a replacement token after a rejection.

    Following Leviathan et al. (2023), the corrected distribution for
    a rejected position is ``norm(relu(p_target - p_draft))``.

    Args:
        target_log_probs: ``(batch, vocab)`` log-probs from the
            target model at the rejected position.
        draft_log_probs: ``(batch, vocab)`` log-probs from the
            drafter at the same position.
        rng_key: PRNG key.

    Returns:
        ``(batch,)`` replacement token IDs.
    """
    p_target = jnp.exp(target_log_probs)
    p_draft = jnp.exp(draft_log_probs)
    residual = jnp.maximum(p_target - p_draft, 0.0)
    residual = residual / jnp.maximum(jnp.sum(residual, axis=-1, keepdims=True), 1e-9)
    return jax.random.categorical(rng_key, jnp.log(residual + 1e-9))




class Qwen3_5MTPDrafter:
    """Wrap a Qwen3.5 MTP head as a :class:`DrafterProtocol`.

    This drafter is *inline* — it shares parameters with the target
    Qwen3.5 model. Calling ``draft()`` runs the model's
    ``compute_mtp_outputs`` and projects through the shared LM head,
    then argmaxes to produce a draft token. The drafter is greedy
    (no full distribution returned) — rejection sampling falls back
    to ``argmax(target) == draft`` verification.

    Multi-step drafts come from chaining: feed the verified token
    back and call ``draft()`` again.

    For a sampling drafter (returning ``full_log_probs``), call
    :meth:`draft_sampled` instead.
    """

    def __init__(
        self,
        model: typing.Any,
        *,
        num_draft_tokens: int = 1,
        use_cache: bool = True,
        cache_length: int | None = None,
    ):
        """Wrap a Qwen3.5 causal-LM (or multimodal generation) model.

        Args:
            model: An instance of ``Qwen3_5ForCausalLM`` or
                ``Qwen3_5ForConditionalGeneration``. Must expose
                ``has_mtp()``, ``compute_mtp_outputs``, and
                ``compute_mtp_logits``.
            num_draft_tokens: Number of speculative tokens to propose per
                verify window. Values greater than one repeatedly apply the
                same inline MTP block, matching vLLM's MTP behavior.
            use_cache: Whether to maintain the MTP-local attention cache.
            cache_length: Optional MTP-local cache length. The eSurge runner
                sets this to the runner max length; standalone use falls back
                to the model config length.

        Raises:
            ValueError: If the model lacks an MTP head.
        """
        if not getattr(model, "has_mtp", lambda: False)():
            raise ValueError("Qwen3_5MTPDrafter requires a Qwen3.5 model with config.mtp_num_hidden_layers > 0.")
        self.model = model
        self.supports_return_full_log_probs = True
        self.supports_prefix_draft = True
        self.uses_mtp_cache = bool(use_cache)
        self._mtp_cache: TransformerCache | None = None
        self._mtp_cache_max_length: int | None = max(1, int(cache_length)) if cache_length is not None else None
        self.num_draft_tokens = max(1, int(num_draft_tokens))
        self._jit_mtp: dict[tuple, typing.Callable] = {}

    def set_max_length(self, max_length: int) -> None:
        """Configure the MTP-local cache length used by runner-native decode.

        Args:
            max_length: Maximum number of tokens the MTP cache should hold;
                values less than one are clamped up to one.
        """
        self._mtp_cache_max_length = max(1, int(max_length))

    def reset(self, batch_size: int) -> None:
        """Reset the MTP-local KV cache for a new request.

        Args:
            batch_size: Number of sequences in the upcoming generation.
        """
        if not self.uses_mtp_cache:
            self._mtp_cache = None
            return
        self._mtp_cache = self._init_mtp_cache(int(batch_size))

    def _init_mtp_cache(self, batch_size: int) -> TransformerCache | None:
        """Allocate a full-attention cache for the inline MTP block.

        Args:
            batch_size: Number of sequences the cache must accommodate.

        Returns:
            A freshly allocated :class:`TransformerCache`, or ``None`` if the
            drafter is configured to run without an MTP cache or the model is
            missing the required config / mesh.
        """
        if not self.uses_mtp_cache:
            return None
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg)
        if text_cfg is None:
            return None
        mesh = getattr(self.model, "mesh", None) or getattr(text_cfg, "mesh", None)
        if mesh is None:
            return None
        head_dim = int(getattr(text_cfg, "head_dim", 0) or 0)
        if head_dim <= 0:
            num_heads = int(getattr(text_cfg, "num_attention_heads", 1) or 1)
            head_dim = int(getattr(text_cfg, "hidden_size", 0)) // max(1, num_heads)
        num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", getattr(text_cfg, "num_attention_heads", 1)))
        getattr(self.model, "mtp", None)
        max_length = self._mtp_cache_max_length
        if max_length is None:
            max_length = int(
                getattr(
                    text_cfg,
                    "max_position_embeddings",
                    getattr(text_cfg, "max_sequence_length", 4096),
                )
                or 4096
            )
        cache_config = TransformerCacheConfig.create(
            batch_size=int(batch_size),
            sequence_length=max(1, int(max_length)),
            num_hidden_layers=int(getattr(text_cfg, "mtp_num_hidden_layers", 1) or 1),
            pad_token_id=0,
            num_heads=int(getattr(text_cfg, "num_attention_heads", num_kv_heads) or num_kv_heads),
            head_dim=head_dim,
            key_heads=num_kv_heads,
            value_heads=num_kv_heads,
            key_dim=head_dim,
            value_dim=head_dim,
        )
        return TransformerCache.init_cache(
            mesh=mesh,
            config=cache_config,
            runtime_sharding_resolver=getattr(text_cfg, "runtime_sharding_resolver", None),
            dtype=getattr(self.model, "dtype", jnp.bfloat16),
        )

    def _ensure_mtp_cache(self, batch_size: int) -> TransformerCache | None:
        """Return the MTP cache, lazily allocating it on first use.

        Args:
            batch_size: Batch size used to size the cache if it has not been
                initialized yet.

        Returns:
            The current :class:`TransformerCache`, or ``None`` when the
            drafter is configured to run without one.
        """
        if not self.uses_mtp_cache:
            return None
        if self._mtp_cache is None:
            self._mtp_cache = self._init_mtp_cache(batch_size)
        return self._mtp_cache

    def _mtp_hidden_and_logits(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: Float[Array, "batch seq hidden"],
        position_ids: Int[Array, "batch seq"] | None = None,
        mtp_cache: TransformerCache | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Float[Array, "batch seq vocab"], TransformerCache | None]:
        """JIT-cached MTP-head forward → vocab logits ``(B, S, V)``.

        Compiles the MTP-head + lm-head path once per
        ``(input_ids, target_hidden_states)`` shape so repeated draft
        calls run as one fused device kernel instead of eager op-by-op
        dispatch. Here ``input_ids`` are the already-sampled next-token
        embeddings supplied to the DeepSeek-style MTP fusion, not the
        unshifted training labels used by ``compute_mtp_outputs``.
        """
        key = (
            tuple(input_ids.shape),
            str(input_ids.dtype),
            tuple(target_hidden_states.shape),
            str(target_hidden_states.dtype),
            None if position_ids is None else tuple(position_ids.shape),
            None if position_ids is None else str(position_ids.dtype),
            None
            if mtp_cache is None
            else tuple(
                None
                if view is None
                else (
                    tuple(view.key.shape),
                    str(view.key.dtype),
                    tuple(view.value.shape),
                    str(view.value.dtype),
                    tuple(view.indexes.shape),
                    str(view.indexes.dtype),
                )
                for view in mtp_cache.views
            ),
        )
        fn = self._jit_mtp.get(key)
        if fn is None:

            def _embed_next(ids):
                if hasattr(self.model, "get_input_embeddings"):
                    return self.model.get_input_embeddings()(ids.astype("i4"))
                return self.model.model.get_embedding()(ids.astype("i4"))

            def _compute(ids, hidden, pos, cache):
                mtp = self.model.mtp
                next_embeds = _embed_next(ids)
                normed_e = mtp.pre_fc_norm_embedding(next_embeds)
                normed_h = mtp.pre_fc_norm_hidden(hidden)
                h = mtp.fc(jnp.concatenate([normed_e, normed_h], axis=-1))
                mask_info = MaskInfo.dynamic_init(
                    mask_info=None,
                    input_ids=None,
                    inputs_embeds=h,
                    attention_mask=None,
                )
                if pos is None:
                    batch_size, seq_len = h.shape[:2]
                    pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(
                        batch_size,
                        axis=0,
                    )
                base_model = getattr(self.model, "model", None)
                language_model = getattr(base_model, "language_model", None)
                frequencies_owner = language_model if language_model is not None else base_model
                frequencies = getattr(frequencies_owner, "frequencies", None)
                views = cache.views if cache is not None else None
                new_views = []
                mtp_mode = common_types.MODE_PREFILL if ids.shape[1] > 1 else common_types.MODE_DECODE
                for layer in mtp.layers:
                    cv_in = views[layer.layer_idx] if views is not None and layer.layer_idx < len(views) else None
                    cache_metadata = (
                        TransformerMetadata(
                            postpadded=False,
                            starts=cv_in.starts,
                            indexes=cv_in.indexes,
                        )
                        if cv_in is not None
                        else None
                    )
                    h, _cache_view = layer(
                        h,
                        mask_info,
                        pos,
                        mtp_mode,
                        cv_in,
                        cache_metadata,
                        frequencies,
                    )
                    new_views.append(_cache_view)
                h = mtp.norm(h)
                new_cache = TransformerCache(views=new_views) if cache is not None else None
                return h, self.model.apply_lm_head(h), new_cache

            _ = _compute(input_ids, target_hidden_states, position_ids, mtp_cache)

            @jax.jit
            def _fn(ids, hidden, pos, cache):
                return _compute(ids, hidden, pos, cache)

            fn = _fn
            self._jit_mtp[key] = fn
        return fn(input_ids, target_hidden_states, position_ids, mtp_cache)

    def draft(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: Float[Array, "batch seq hidden"] | None = None,
        target_kv_cache: typing.Any = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        return_full_log_probs: bool = False,
        sample: bool = False,
        rng_key: jax.Array | None = None,
    ) -> DraftStep:
        """Produce one MTP draft token per batch element.

        Args:
            input_ids: ``(batch, seq)`` verified tokens.
            target_hidden_states: ``(batch, seq, hidden)`` target
                last hidden state. Required.
            target_kv_cache: Ignored (MTP is inline).
            position_ids: Absolute positions for ``target_hidden_states``.
                Decode call sites should pass the hidden row positions,
                matching the shifted-input contract used by MTP
                proposers.
            return_full_log_probs: Materialize full drafter
                log-probs. Required for distribution-correct sampled
                speculative decoding, skipped on the greedy fast path.
            sample: Whether to sample from the MTP distribution
                rather than argmax. Requires ``rng_key``.
            rng_key: PRNG key for sampling.

        Returns:
            :class:`DraftStep` with the predicted token at the LAST
            input position (i.e. one-step lookahead from
            ``input_ids[:, -1]``).
        """
        if target_hidden_states is None:
            raise ValueError("Qwen3_5MTPDrafter.draft requires target_hidden_states")

        mtp_cache = self._ensure_mtp_cache(int(input_ids.shape[0]))
        hidden_states, logits, new_cache = self._mtp_hidden_and_logits(
            input_ids,
            target_hidden_states,
            position_ids,
            mtp_cache,
        )
        if new_cache is not None:
            self._mtp_cache = new_cache
        last = logits[:, -1, :].astype(jnp.float32)
        if sample:
            if rng_key is None:
                raise ValueError("rng_key required when sample=True")
            token_ids = jax.random.categorical(rng_key, last)
        else:
            token_ids = jnp.argmax(last, axis=-1)
        log_probs = None
        token_log_probs = None
        if sample or return_full_log_probs:
            log_probs = jax.nn.log_softmax(last, axis=-1)
            token_log_probs = jnp.take_along_axis(log_probs, token_ids[:, None], axis=-1).squeeze(-1)
        return DraftStep(
            token_ids=token_ids.astype(jnp.int32),
            log_probs=token_log_probs,
            full_log_probs=log_probs,
            hidden_states=hidden_states,
        )




class Gemma4AssistantDrafter:
    """Wrap a Gemma4 Assistant model as a :class:`DrafterProtocol`.

    This drafter is *standalone* — it owns its own params separate
    from the target model. ``draft()`` requires the target's last
    hidden state and the target's per-drafter-layer K/V tensors.

    **Important**: extracting per-layer K/V from the target requires
    a controller in the inference runtime; this class only knows how
    to consume those tensors once provided. See ``esurge`` runner
    follow-up for the controller. The class is provided here so that
    standalone-drafter call sites can be written ahead of time.
    """

    def __init__(
        self,
        assistant_model: typing.Any,
        target_embed_module: typing.Any,
        *,
        layer_mapping: list[int] | None = None,
        target_config: typing.Any | None = None,
    ):
        """Wrap an assistant + reference to the target's embedding.

        Args:
            assistant_model: ``Gemma4AssistantForCausalLM`` instance.
            target_embed_module: ``embed_tokens`` of the target
                Gemma4 model. Needed to compute
                ``target_token_embeds`` at the target's hidden size.
            layer_mapping: Optional assistant-layer -> target-layer
                mapping. ``None`` uses the eSurge default heuristic.
            target_config: Optional target config used to resolve the
                default mapping before a cache is available.
        """
        self.assistant = assistant_model
        self.supports_return_full_log_probs = True
        self.target_embed = target_embed_module
        self.layer_mapping = list(layer_mapping) if layer_mapping is not None else None
        self.target_config = target_config
        self.requires_target_kv_cache = True
        backbone_h = int(assistant_model.config.backbone_hidden_size)
        self._embed_scale = jnp.sqrt(jnp.array(backbone_h, dtype=jnp.float32))
        self._jit_assistant: dict[tuple, typing.Callable] = {}

    def resolve_layer_mapping(self, target_cache: typing.Any | None = None) -> list[int]:
        """Return the assistant-layer to target-layer K/V mapping.

        Uses the explicit ``layer_mapping`` if provided, otherwise derives a
        default mapping from the assistant's number of layers and the target's
        number of layers (inferred from ``target_config`` or ``target_cache``).

        Args:
            target_cache: Optional target KV cache used to infer the target's
                layer count when no ``target_config`` was supplied.

        Returns:
            A list whose i-th entry is the target layer index from which the
            i-th assistant layer should source its K/V tensors.
        """
        if self.layer_mapping is not None:
            return list(self.layer_mapping)

        assistant_layers = int(self.assistant.config.text_config.num_hidden_layers)
        target_layers = None
        if self.target_config is not None:
            text_config = getattr(self.target_config, "text_config", self.target_config)
            target_layers = getattr(text_config, "num_hidden_layers", None)
        if target_layers is None and target_cache is not None:
            views = getattr(target_cache, "views", None)
            if views is not None:
                target_layers = len(views)
        if target_layers is None:
            target_layers = assistant_layers

        from easydel.inference.esurge.speculative_decoding import default_assistant_layer_mapping

        return default_assistant_layer_mapping(assistant_layers, int(target_layers))

    def reset(self, batch_size: int) -> None:
        """Drafter is stateless within JAX-functional forward.

        State (KV cache) is owned by the caller / runtime.
        """
        del batch_size

    def _assistant_forward(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: Float[Array, "batch seq backbone_hidden"],
        target_kv_cache: list[tuple[Array, Array] | None] | None,
        position_ids: Int[Array, "batch seq"] | None,
        attention_mask: Float[Array, "batch 1 q_len kv_len"] | None,
        return_dense_logits: bool,
    ) -> typing.Any:
        """JIT-cached Gemma4 Assistant forward (compiled once per shape).

        ``target_kv_cache`` is part of the traced pytree (a list of
        ``(K, V)`` arrays, or ``None``); a structural change in it
        triggers one extra compile, which is fine for the fixed
        decode geometry.
        """
        has_kv = target_kv_cache is not None
        kv_signature = None
        if target_kv_cache is not None:
            kv_signature = tuple(
                None
                if pair is None
                else (
                    (tuple(pair[0].shape), str(pair[0].dtype)),
                    (tuple(pair[1].shape), str(pair[1].dtype)),
                )
                for pair in target_kv_cache
            )
        key = (
            tuple(input_ids.shape),
            str(input_ids.dtype),
            tuple(target_hidden_states.shape),
            str(target_hidden_states.dtype),
            has_kv,
            kv_signature,
            None if position_ids is None else tuple(position_ids.shape),
            None if position_ids is None else str(position_ids.dtype),
            None if attention_mask is None else tuple(attention_mask.shape),
            None if attention_mask is None else str(attention_mask.dtype),
            bool(return_dense_logits),
        )
        fn = self._jit_assistant.get(key)
        if fn is None:
            target_embeds = self.target_embed(input_ids.astype("i4")) * self._embed_scale.astype(
                target_hidden_states.dtype
            )
            _ = self.assistant(
                backbone_hidden_states=target_hidden_states,
                target_token_embeds=target_embeds,
                target_key_value_pairs=target_kv_cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dense_logits=return_dense_logits,
            )

            @jax.jit
            def _fn(ids, hidden, kv, pos, mask):
                embeds = self.target_embed(ids.astype("i4")) * self._embed_scale.astype(hidden.dtype)
                return self.assistant(
                    backbone_hidden_states=hidden,
                    target_token_embeds=embeds,
                    target_key_value_pairs=kv,
                    position_ids=pos,
                    attention_mask=mask,
                    return_dense_logits=return_dense_logits,
                )

            fn = _fn
            self._jit_assistant[key] = fn
        return fn(input_ids, target_hidden_states, target_kv_cache, position_ids, attention_mask)

    def draft(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: Float[Array, "batch seq backbone_hidden"] | None = None,
        target_kv_cache: list[tuple[Array, Array] | None] | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        attention_mask: Float[Array, "batch 1 q_len kv_len"] | None = None,
        return_full_log_probs: bool = False,
        sample: bool = False,
        rng_key: jax.Array | None = None,
    ) -> DraftStep:
        """Produce one drafted token per batch element.

        Args:
            input_ids: ``(B, S)`` verified tokens. The drafter uses
                their target-side embedding as part of
                ``pre_projection``.
            target_hidden_states: ``(B, S, backbone_hidden)`` final
                hidden states from the target model. Required.
            target_kv_cache: Per-drafter-layer ``(K, V)`` tuples
                from the target's KV cache, aligned to the drafter
                layers' attention types (sliding ↔ sliding,
                full ↔ full). Required for semantically correct
                drafts.
            position_ids: Position IDs for Q-side RoPE.
            attention_mask: Optional mask used when K/V is padded to a
                static length by the runner.
            return_full_log_probs: Materialize dense candidate
                log-probs for distribution-correct sampled spec decode.
            sample: Whether to sample from the centroid distribution.
            rng_key: PRNG for sampling.

        Returns:
            :class:`DraftStep` with the proposed token at the last
            input position.
        """
        if target_hidden_states is None:
            raise ValueError("Gemma4AssistantDrafter.draft requires target_hidden_states")
        out = self._assistant_forward(
            input_ids,
            target_hidden_states,
            target_kv_cache,
            position_ids,
            attention_mask,
            bool(return_full_log_probs or sample),
        )
        if out.top_logits is not None and out.top_token_ids is not None:
            last_logits = out.top_logits[:, -1, :].astype(jnp.float32)
            last_ids = out.top_token_ids[:, -1, :].astype(jnp.int32)
            local_lp = jax.nn.log_softmax(last_logits, axis=-1)
            full_log_probs = None
            if out.logits is not None:
                full_log_probs = jax.nn.log_softmax(out.logits[:, -1, :].astype(jnp.float32), axis=-1)
            if sample:
                if rng_key is None:
                    raise ValueError("rng_key required when sample=True")
                pick = jax.random.categorical(rng_key, last_logits)
            else:
                pick = jnp.argmax(last_logits, axis=-1)
            token_ids = jnp.take_along_axis(last_ids, pick[:, None], axis=-1).squeeze(-1)
            token_log_probs = jnp.take_along_axis(local_lp, pick[:, None], axis=-1).squeeze(-1)
            return DraftStep(
                token_ids=token_ids,
                log_probs=token_log_probs,
                full_log_probs=full_log_probs,
                hidden_states=out.backbone_hidden_state,
            )

        last = out.logits[:, -1, :].astype(jnp.float32)
        log_probs = jax.nn.log_softmax(last, axis=-1)
        if sample:
            if rng_key is None:
                raise ValueError("rng_key required when sample=True")
            token_ids = jax.random.categorical(rng_key, last)
        else:
            token_ids = jnp.argmax(last, axis=-1)
        token_log_probs = jnp.take_along_axis(log_probs, token_ids[:, None], axis=-1).squeeze(-1)
        return DraftStep(
            token_ids=token_ids.astype(jnp.int32),
            log_probs=token_log_probs,
            full_log_probs=log_probs,
            hidden_states=out.backbone_hidden_state,
        )


__all__ = [
    "DraftStep",
    "DrafterProtocol",
    "Gemma4AssistantDrafter",
    "Qwen3_5MTPDrafter",
    "accept_or_reject",
    "resample_rejected",
]
