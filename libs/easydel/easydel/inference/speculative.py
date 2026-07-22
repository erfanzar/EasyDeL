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
import spectrax as spx
from ejkernel.types import MaskInfo
from jaxtyping import Array, Float, Int
from spectrax import common_types

from easydel.caching import TransformerCache, TransformerCacheConfig, TransformerMetadata
from easydel.infra.spec_decode import SpecDecodeBase, register_drafter


@dataclass(frozen=True)
class DraftStep:
    """One step of drafter output.

    Attributes:
        token_ids: ``(batch,)`` proposed next token IDs.
        log_probs: ``(batch,)`` log-prob assigned by the drafter to
            ``token_ids``. Used for the rejection-sampling correction in
            :func:`accept_or_reject`. ``None`` for greedy drafters.
        full_log_probs: Optional ``(batch, vocab)`` full drafter distribution
            over the vocabulary. Required if the verifier wants to do
            distribution-correct rejection sampling rather than the simple
            ratio test. ``None`` to skip the correction (uses greedy
            verification: accept if ``argmax(target) == draft``). These are
            normalized log-probs on the drafter-internal ``sample=True`` path;
            on the external-resample path (``return_full_log_probs=True`` with
            ``sample=False``) a drafter may instead hand back the RAW
            (unnormalized) last-position logits, which the caller filters and
            ``log_softmax``-normalizes itself — numerically equivalent and
            cheaper on the hot draft loop.
        hidden_states: Optional drafter hidden rows for this step. Inline MTP
            drafters reuse this as the next step's ``hidden_states``, matching
            repeated-MTP proposal loops.
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

    Capability flags (the eSurge runner reads them via ``getattr`` with the
    documented defaults, so a drafter only sets the ones it changes):

    Attributes:
        supports_return_full_log_probs: The drafter can materialize a dense
            ``(batch, vocab)`` distribution for distribution-correct sampled
            speculative decoding (default ``False`` when unset by the drafter).
        supports_prefix_draft: The drafter accepts a multi-token prefix instead
            of a single seed token for its first draft step (default ``False``).
        requires_target_kv_cache: The drafter cross-attends to the target's
            per-layer K/V and must be fed a target K/V payload (default
            ``False``).
        uses_mtp_cache: The drafter keeps an internal MTP KV cache that the
            runner should account for (default ``False``).
        num_draft_tokens: Speculative tokens proposed per verify window (the
            runner also accepts ``num_speculative_tokens`` as a fallback name).
    """

    supports_return_full_log_probs: bool
    supports_prefix_draft: bool
    requires_target_kv_cache: bool
    num_draft_tokens: int

    def reset(self, batch_size: int) -> None:
        """Reset the drafter's internal state for a new generation.

        Implementations should clear any KV cache, prefix state, or per-batch
        buffers. Called once per generation, not per step.

        Args:
            batch_size: Number of sequences in the upcoming generation.
        """
        ...

    def reset_request(self, request_id: str | None = None) -> None:
        """Reset per-request / per-verify-window drafter state.

        Drafters that carry an internal KV cache across ``draft`` calls clear
        it here so state never leaks across requests. The eSurge runner calls
        this at the start of each verify window. Stateless drafters make it a
        no-op (the runner probes it via ``getattr`` and skips when absent).

        Args:
            request_id: Identifier of the request being drafted for, or ``None``.
        """
        ...

    def draft(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: Float[Array, "batch seq hidden"] | None = None,
        target_kv_cache: typing.Any | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        return_full_log_probs: bool = False,
        sample: bool = False,
        rng_key: jax.Array | None = None,
        attention_mask: typing.Any | None = None,
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
            sample: Sample from the drafter distribution instead of
                taking the argmax.
            rng_key: PRNG key required when ``sample=True``.
            attention_mask: Optional mask used by drafters that consume a
                statically-padded target K/V cache.

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
        persist_kv: bool = True,
    ):
        """Wrap a Qwen3.5 causal-LM (or multimodal generation) model.

        Args:
            model: An instance of ``Qwen3_5ForCausalLM`` or
                ``Qwen3_5ForConditionalGeneration``. Must expose
                ``has_mtp()``, ``compute_mtp_outputs``, and
                ``compute_mtp_logits``.
            num_draft_tokens: Number of speculative tokens to propose per
                verify window. Values greater than one repeatedly apply the
                same inline MTP block, matching repeated-MTP behavior.
            use_cache: Whether to maintain the MTP-local attention cache.
            cache_length: Optional MTP-local cache length. The eSurge runner
                sets this to the runner max length; standalone use falls back
                to the model config length.
            persist_kv: Persist-with-rollback of the MTP KV cache across
                verify windows (EAGLE / vLLM style) instead of a per-window
                full clear. When ``True`` (the default) the eSurge strategy
                keeps the accepted context K/V across windows and only rolls
                it back to the committed sequence position via
                :meth:`rollback_to`; without it, acceptance collapses with
                draft depth (measured 58% -> 25% at K=4). ``False`` restores
                the historical per-window clear (:meth:`reset_request`). The
                ``EASYDEL_MTP_PERSIST_KV`` env var (``"0"``/``"1"``) overrides
                this flag either way at the eSurge strategy level.

        Raises:
            ValueError: If the model lacks an MTP head.
        """
        if not getattr(model, "has_mtp", lambda: False)():
            raise ValueError("Qwen3_5MTPDrafter requires a Qwen3.5 model with config.mtp_num_hidden_layers > 0.")
        self.model = model
        self.supports_return_full_log_probs = True
        self.supports_prefix_draft = True
        self.uses_mtp_cache = bool(use_cache)
        self.persist_kv = bool(persist_kv)
        self._mtp_cache: TransformerCache | None = None
        self._mtp_cache_batch_size: int | None = None
        self._mtp_cache_max_length: int | None = max(1, int(cache_length)) if cache_length is not None else None
        self.num_draft_tokens = max(1, int(num_draft_tokens))
        self._jit_mtp: dict[tuple, typing.Callable] = {}
        self._jit_adaptive: dict[tuple, typing.Callable] = {}
        self._jit_batched: dict[tuple, typing.Callable] = {}
        self.supports_batched_draft = True
        # Cached static GraphDefs for the batched-draft submodule bundle
        # (mtp block, input embeddings, LM head) — see
        # ``_batched_bundle_graph_and_states``.
        self._bundle_graphdefs: tuple | None = None
        # Cached static GraphDef for ``self.model`` (structurally invariant for a
        # fixed model). The model State (the actual weight arrays) is threaded
        # through the drafter jits as a runtime INPUT, never closed over — see
        # ``_model_graph_and_state``.
        self._graphdef: spx.GraphDef | None = None

    def _batched_bundle_graph_and_states(self):
        """Split ONLY the modules the MTP forward touches into (GraphDefs, States).

        The batched drafter runs once per decode step, and threading the FULL
        27B model State through its ``jax.jit`` costs ~14 ms of host dispatch
        (1100+ leaves flattened/matched per call) plus ~9 ms re-exporting the
        whole module tree — 4-6x the actual device time of the k MTP forwards.
        The forward only reads the inline MTP block, the input embeddings, the
        LM head, and the rotary frequency table, so export just those: a few
        dozen leaves, microsecond-scale export and dispatch.

        GraphDefs (static structure) are cached on first use; the sub-States
        are re-exported every call so live/updated weights are picked up, same
        contract as ``_model_graph_and_state``.

        Returns:
            ``((mtp_gd, embed_gd, head_gd), (mtp_st, embed_st, head_st),
            frequencies, tie_embeddings)`` where ``frequencies`` is the rotary
            table array (or ``None``) and ``tie_embeddings`` the static
            weight-tying flag consumed by the LM-head application.
        """
        model = self.model
        mtp_gd, mtp_st = spx.export(model.mtp)
        if hasattr(model, "get_input_embeddings"):
            embed_module = model.get_input_embeddings()
        else:
            embed_module = model.model.get_embedding()
        embed_gd, embed_st = spx.export(embed_module)
        head_gd, head_st = spx.export(model.get_lm_head())
        if self._bundle_graphdefs is None:
            self._bundle_graphdefs = (mtp_gd, embed_gd, head_gd)
        base_model = getattr(model, "model", None)
        language_model = getattr(base_model, "language_model", None)
        frequencies_owner = language_model if language_model is not None else base_model
        frequencies = getattr(frequencies_owner, "frequencies", None)
        cfg = getattr(model, "config", None)
        tie_embeddings = next(
            (
                getattr(cfg, key)
                for key in ("tie_word_embeddings", "use_lm_head", "share_input_output_layers")
                if hasattr(cfg, key)
            ),
            False,
        )
        return self._bundle_graphdefs, (mtp_st, embed_st, head_st), frequencies, bool(tie_embeddings)

    @staticmethod
    def _mtp_forward_compute_parts(
        mtp: typing.Any,
        embed_module: typing.Any,
        lm_head_module: typing.Any,
        frequencies: Array | None,
        tie_embeddings: bool,
        ids: Int[Array, "batch seq"],
        hidden: Float[Array, "batch seq hidden"],
        pos: Int[Array, "batch seq"] | None,
        cache: TransformerCache | None,
        row: jax.Array | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Float[Array, "batch seq vocab"], TransformerCache | None]:
        """One inline-MTP-head forward from its constituent modules.

        Same numerics as ``_mtp_forward_compute``, but the target model's
        pieces are passed in explicitly so the drafter jits can thread ONLY
        their (small) States as inputs instead of the full model State — see
        ``_batched_bundle_graph_and_states``.

        Args:
            mtp: The inline MTP block (``model.mtp``).
            embed_module: The input-embedding module (also supplies the tied
                LM-head weight when ``tie_embeddings``).
            lm_head_module: The LM-head projection module.
            frequencies: Rotary frequency table array, or ``None``.
            tie_embeddings: Static weight-tying flag (matches
                ``apply_lm_head``'s config resolution).
            ids: ``(batch, seq)`` already-sampled next-token ids.
            hidden: ``(batch, seq, hidden)`` target hidden rows to fuse.
            pos: Absolute positions for ``hidden``; ``None`` builds ``arange``.
            cache: Inline-MTP ``TransformerCache`` to read/advance, or ``None``.
            row: Optional traced batch row for per-request row targeting: the
                pool cache is sliced down to this row for a batch-1 forward and
                the advanced view scattered back (same semantics as
                ``_mtp_forward_compute``'s ``row``).

        Returns:
            ``(hidden_out, logits, new_cache)``.
        """

        def _slice_view_row(view, row_):
            """Slice a batch-N cache view down to a batch-1 view at ``row``."""
            return view.replace(
                key=jax.lax.dynamic_slice_in_dim(view.key, row_, 1, axis=0),
                value=jax.lax.dynamic_slice_in_dim(view.value, row_, 1, axis=0),
                indexes=jax.lax.dynamic_slice_in_dim(view.indexes, row_, 1, axis=0),
                starts=jax.lax.dynamic_slice_in_dim(view.starts, row_, 1, axis=0),
            )

        def _scatter_view_row(full, updated, row_):
            """Write a batch-1 updated view back into ``row`` of the pool view."""
            return full.replace(
                key=jax.lax.dynamic_update_slice_in_dim(full.key, updated.key, row_, axis=0),
                value=jax.lax.dynamic_update_slice_in_dim(full.value, updated.value, row_, axis=0),
                indexes=jax.lax.dynamic_update_slice_in_dim(full.indexes, updated.indexes, row_, axis=0),
                starts=jax.lax.dynamic_update_slice_in_dim(full.starts, updated.starts, row_, axis=0),
            )

        next_embeds = embed_module(ids.astype("i4"))
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
            pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)
        views = cache.views if cache is not None else None
        new_views = []
        mtp_mode = common_types.MODE_PREFILL if ids.shape[1] > 1 else common_types.MODE_DECODE
        for layer in mtp.layers:
            cv_full = views[layer.layer_idx] if views is not None and layer.layer_idx < len(views) else None
            cv_in = _slice_view_row(cv_full, row) if (cv_full is not None and row is not None) else cv_full
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
            if cv_full is not None and row is not None:
                _cache_view = _scatter_view_row(cv_full, _cache_view, row)
            new_views.append(_cache_view)
        h = mtp.norm(h)
        new_cache = TransformerCache(views=new_views) if cache is not None else None
        w = embed_module.weight.value.T if tie_embeddings else None
        return h, lm_head_module(h, w=w), new_cache

    def _model_graph_and_state(self) -> tuple[spx.GraphDef, spx.State]:
        """Split ``self.model`` into (cached static GraphDef, live State).

        The MTP-head forward reaches the target's weights (``model.mtp``, the
        input embeddings, and ``model.apply_lm_head``). If those weights are
        reached through the ``self.model`` *closure* of a plain ``jax.jit`` they
        are baked into the compiled program as captured CONSTANTS — for the 27B
        that is ~6 GB copied into every drafter executable, which makes the
        program too large to serialize (never caches, re-lowers each process) and
        stacks a second full weight copy on top of the KV pool at warmup (batched
        spec-decode then OOMs on TPU).

        Passing the ``State`` as a jit input instead makes the compiled program
        reference the already-resident weight buffers. ``spx.export`` only
        traverses the module pytree; it references the existing device arrays and
        does not copy them, so this adds no HBM (the drafter shares ``self.model``
        with the main runner — same device buffers). The GraphDef is cached once;
        the State is re-exported each call so live/updated weights are picked up
        and the arrays are never part of the compile key.
        """
        graphdef, state = spx.export(self.model)
        if self._graphdef is None:
            self._graphdef = graphdef
        return self._graphdef, state

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
            self._mtp_cache_batch_size = None
            return
        batch_size = int(batch_size)
        self._mtp_cache = self._init_mtp_cache(batch_size)
        self._mtp_cache_batch_size = batch_size if self._mtp_cache is not None else None

    def reset_request(self, request_id: str | None = None) -> None:
        """Clear the MTP-local KV cache at a request / verify-window boundary.

        The MTP cache is a single rolling KV buffer keyed only on batch size, so
        without an explicit boundary it is reused across requests and windows —
        the previous request's/window's K/V then pollutes the next drafter call
        and acceptance collapses. Clearing it here forces ``_ensure_mtp_cache``
        to reallocate a fresh cache on the next ``draft`` so each verify window
        starts from an empty MTP state. The eSurge runner calls this once per
        window (see ``DrafterSpeculation.draft_next``). Output is unchanged; this
        only prevents cross-request cache leakage.

        Args:
            request_id: Identifier of the request being drafted for (unused;
                accepted for protocol compatibility).
        """
        del request_id
        self._mtp_cache = None
        self._mtp_cache_batch_size = None

    def rollback_to(self, committed_len: int, row_pos: int | None = None) -> int:
        """Roll the MTP KV cache back to a committed length (persist-with-rollback).

        Instead of clearing the whole MTP cache at each verify-window boundary
        (:meth:`reset_request`), keep the confirmed-context K/V and discard only
        the K/V of the previous window's rejected / tentative draft tokens
        written past ``committed_len``. The next window's writes overwrite the
        slots at and beyond ``committed_len``, so its MTP self-attention sees the
        full accepted-sequence context (the EAGLE / vLLM persistent-drafter
        approach). The eSurge strategy calls this once per window (in place of
        the full clear) when persistence is enabled.

        Rollback is a truncation of the per-layer write index (``indexes``): the
        attention reads only ``[0, indexes)`` and later writes overwrite the
        dropped slots before they are read, so lowering ``indexes`` drops the
        rejected K/V without touching the stored tensors (the same
        truncate-and-overwrite invariant paged attention relies on). Each view's
        index is additionally clamped down to its current value (the written
        extent), so a window that accepted more tokens than the drafter laid down
        fused states for can never point ``indexes`` at an unwritten slot.

        With ``row_pos`` the rollback is **per row**: only that batch row's write
        index is truncated, leaving every other concurrent request's row
        untouched. This is what batched serving needs — the MTP cache is sized to
        the request pool (:meth:`ensure_batch_size`) and each request keeps its
        own row across verify windows, so a window boundary for one request must
        not disturb another's accumulated context. Without ``row_pos`` (the
        legacy single-request path) every row is truncated, which is what a
        batch-1 cache wants.

        Args:
            committed_len: Number of leading cache slots to retain — the accepted
                MTP context length, in the cache's compacted slot space. Clamped
                up to zero.
            row_pos: Optional batch row (the request's stable KV-pool slot) to
                roll back. ``None`` truncates every row (legacy single-request /
                batch-1 behavior).

        Returns:
            The requested ``committed_len`` clamped to ``>= 0`` (the value the
            per-layer write index is set to, subject to the written-extent clamp
            applied per view), or ``0`` when the drafter holds no MTP cache.
        """
        if not self.uses_mtp_cache or self._mtp_cache is None:
            return 0
        target = max(0, int(committed_len))
        target_arr = jnp.asarray(target, dtype=jnp.int32)
        row = None if row_pos is None else int(row_pos)
        new_views: list[typing.Any] = []
        for view in self._mtp_cache.views:
            if view is None:
                new_views.append(view)
                continue
            idx = view.indexes.astype(jnp.int32)
            if row is None:
                clamped = jnp.minimum(idx, target_arr)
            else:
                # Truncate only this request's row; other rows keep their K/V.
                clamped = idx.at[row].set(jnp.minimum(idx[row], target_arr))
            new_views.append(view.replace(indexes=clamped.astype(view.indexes.dtype)))
        self._mtp_cache = TransformerCache(views=new_views)
        return target

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
        batch_size = int(batch_size)
        if self._mtp_cache is None or self._mtp_cache_batch_size != batch_size:
            self._mtp_cache = self._init_mtp_cache(batch_size)
            self._mtp_cache_batch_size = batch_size if self._mtp_cache is not None else None
        return self._mtp_cache

    def ensure_batch_size(self, batch_size: int) -> None:
        """Grow the MTP cache so it can hold at least ``batch_size`` rows.

        Unlike :meth:`_ensure_mtp_cache` / :meth:`reset` (which reallocate on any
        size *mismatch* and therefore clear the cache), this only reallocates when
        the current cache is too small. Batched persist-with-rollback calls it
        once per draft with the request-pool size (``max_num_reqs``): the first
        call allocates the batch-N cache and every later call is a no-op, so a
        concurrent request's accumulated K/V is never wiped mid-generation.

        Args:
            batch_size: Minimum number of rows (concurrent request slots) the MTP
                cache must accommodate; clamped up to one.
        """
        if not self.uses_mtp_cache:
            return
        batch_size = max(1, int(batch_size))
        if self._mtp_cache is None or int(self._mtp_cache_batch_size or 0) < batch_size:
            self._mtp_cache = self._init_mtp_cache(batch_size)
            self._mtp_cache_batch_size = batch_size if self._mtp_cache is not None else None

    def _mtp_forward_compute(
        self,
        model: typing.Any,
        ids: Int[Array, "batch seq"],
        hidden: Float[Array, "batch seq hidden"],
        pos: Int[Array, "batch seq"] | None,
        cache: TransformerCache | None,
        row: jax.Array | None,
    ) -> tuple[Float[Array, "batch seq hidden"], Float[Array, "batch seq vocab"], TransformerCache | None]:
        """One inline-MTP-head forward: ``(ids, hidden) -> (hidden_out, logits, new_cache)``.

        This is the traceable core shared by the single-step decode path
        (:meth:`_mtp_hidden_and_logits`, one call per drafter step) and the
        adaptive on-device draft loop (:meth:`draft_adaptive`, which threads the
        returned cache through a ``jax.lax.while_loop`` carry so the K_max-step
        recursion runs as one compiled program with early exit). Keeping it a
        method (rather than a closure) is what lets the while_loop reuse the exact
        row-targeted forward the per-request persistence path relies on, so a
        fixed-K adaptive run (``conf_threshold <= 0``) is byte-identical to K
        chained :meth:`draft` calls.

        Args:
            model: The target module whose MTP head, input embeddings, and LM
                head this forward reads. Passed in explicitly (rather than read
                off ``self.model``) so callers can hand in a model reconstructed
                from a State that was threaded through their ``jax.jit`` as an
                input — keeping the weights out of the compiled program's
                captured constants (see ``_model_graph_and_state``).
            ids: ``(batch, seq)`` already-sampled next-token ids fed to the MTP
                fusion (embeddings + shifted hidden), not training labels.
            hidden: ``(batch, seq, hidden)`` target hidden rows to fuse.
            pos: Absolute positions for ``hidden``; ``None`` builds ``arange``.
            cache: Inline-MTP ``TransformerCache`` to read/advance, or ``None`` to
                run cache-free (each step re-fuses only its own hidden + token).
            row: Traced batch row for per-request row targeting (the shared
                request-pool cache is sliced down to this row for a batch-1
                forward, then the result is scattered back). ``None`` runs the
                legacy path where the cache batch equals the input batch.

        Returns:
            ``(hidden_out, logits, new_cache)`` — the MTP hidden rows, the vocab
            logits ``(batch, seq, vocab)``, and the advanced cache (or ``None``
            when ``cache`` was ``None``).
        """

        def _embed_next(ids_):
            if hasattr(model, "get_input_embeddings"):
                return model.get_input_embeddings()(ids_.astype("i4"))
            return model.model.get_embedding()(ids_.astype("i4"))

        def _slice_view_row(view, row_):
            """Slice a batch-N cache view down to a batch-1 view at ``row``."""
            return view.replace(
                key=jax.lax.dynamic_slice_in_dim(view.key, row_, 1, axis=0),
                value=jax.lax.dynamic_slice_in_dim(view.value, row_, 1, axis=0),
                indexes=jax.lax.dynamic_slice_in_dim(view.indexes, row_, 1, axis=0),
                starts=jax.lax.dynamic_slice_in_dim(view.starts, row_, 1, axis=0),
            )

        def _scatter_view_row(full, updated, row_):
            """Write a batch-1 updated view back into ``row`` of the pool view."""
            return full.replace(
                key=jax.lax.dynamic_update_slice_in_dim(full.key, updated.key, row_, axis=0),
                value=jax.lax.dynamic_update_slice_in_dim(full.value, updated.value, row_, axis=0),
                indexes=jax.lax.dynamic_update_slice_in_dim(full.indexes, updated.indexes, row_, axis=0),
                starts=jax.lax.dynamic_update_slice_in_dim(full.starts, updated.starts, row_, axis=0),
            )

        mtp = model.mtp
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
        base_model = getattr(model, "model", None)
        language_model = getattr(base_model, "language_model", None)
        frequencies_owner = language_model if language_model is not None else base_model
        frequencies = getattr(frequencies_owner, "frequencies", None)
        views = cache.views if cache is not None else None
        new_views = []
        mtp_mode = common_types.MODE_PREFILL if ids.shape[1] > 1 else common_types.MODE_DECODE
        for layer in mtp.layers:
            cv_full = views[layer.layer_idx] if views is not None and layer.layer_idx < len(views) else None
            # Per-request row targeting: slice this request's row out of the
            # shared pool cache so the batch-1 attention only touches it.
            cv_in = _slice_view_row(cv_full, row) if (cv_full is not None and row is not None) else cv_full
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
            if cv_full is not None and row is not None:
                _cache_view = _scatter_view_row(cv_full, _cache_view, row)
            new_views.append(_cache_view)
        h = mtp.norm(h)
        new_cache = TransformerCache(views=new_views) if cache is not None else None
        return h, model.apply_lm_head(h), new_cache

    def _mtp_hidden_and_logits(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: Float[Array, "batch seq hidden"],
        position_ids: Int[Array, "batch seq"] | None = None,
        mtp_cache: TransformerCache | None = None,
        row_pos: int | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Float[Array, "batch seq vocab"], TransformerCache | None]:
        """JIT-cached MTP-head forward → vocab logits ``(B, S, V)``.

        Compiles the MTP-head + lm-head path once per
        ``(input_ids, target_hidden_states)`` shape so repeated draft
        calls run as one fused device kernel instead of eager op-by-op
        dispatch. Here ``input_ids`` are the already-sampled next-token
        embeddings supplied to the DeepSeek-style MTP fusion, not the
        unshifted training labels used by ``compute_mtp_outputs``.

        When ``row_pos`` is given the MTP cache is the shared request-pool cache
        (batch ``N`` == ``max_num_reqs``) but the forward still runs batch-1 for a
        single request: each layer's cache view is sliced down to that request's
        row, advanced by the batch-1 attention, and the batch-1 result scattered
        back into the pool row. So one concurrent request never reads or writes
        another's row, the per-token MTP compute stays batch-1, and — because the
        row is a *traced* index rather than a Python constant — every row shares a
        single compiled program (no per-row recompile).
        """
        use_row = row_pos is not None and mtp_cache is not None
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
            bool(use_row),
        )
        fn = self._jit_mtp.get(key)
        if fn is None:
            # Eager warmup on ``self.model`` (concrete, NOT under a trace)
            # materializes any lazily-created state (rotary ``frequencies``,
            # deferred variables) into the module before it is split/traced, so
            # the exported State is complete and no trace-time tracer is cached
            # onto the shared model.
            if use_row:
                row_arr = jnp.asarray(int(row_pos), dtype=jnp.int32)
                _ = self._mtp_forward_compute(
                    self.model, input_ids, target_hidden_states, position_ids, mtp_cache, row_arr
                )
            else:
                _ = self._mtp_forward_compute(self.model, input_ids, target_hidden_states, position_ids, mtp_cache, None)
            # Thread ONLY the MTP-touched submodules' States through the jit as
            # inputs (see ``_batched_bundle_graph_and_states``): a few dozen
            # leaves instead of the full model State (1100+ leaves whose
            # per-call re-export + dispatch dominated the per-request draft).
            graphdefs, _states, _freqs, tie_embeddings = self._batched_bundle_graph_and_states()
            mtp_gd, embed_gd, head_gd = graphdefs
            if use_row:

                @jax.jit
                def _fn(mtp_st, embed_st, head_st, freqs, ids, hidden, pos, cache, row):
                    return self._mtp_forward_compute_parts(
                        spx.bind(mtp_gd, mtp_st),
                        spx.bind(embed_gd, embed_st),
                        spx.bind(head_gd, head_st),
                        freqs,
                        tie_embeddings,
                        ids,
                        hidden,
                        pos,
                        cache,
                        row,
                    )
            else:

                @jax.jit
                def _fn(mtp_st, embed_st, head_st, freqs, ids, hidden, pos, cache):
                    return self._mtp_forward_compute_parts(
                        spx.bind(mtp_gd, mtp_st),
                        spx.bind(embed_gd, embed_st),
                        spx.bind(head_gd, head_st),
                        freqs,
                        tie_embeddings,
                        ids,
                        hidden,
                        pos,
                        cache,
                        None,
                    )

            fn = _fn
            self._jit_mtp[key] = fn
        _gds, (mtp_st, embed_st, head_st), freqs, _tie = self._batched_bundle_graph_and_states()
        if use_row:
            return fn(
                mtp_st,
                embed_st,
                head_st,
                freqs,
                input_ids,
                target_hidden_states,
                position_ids,
                mtp_cache,
                jnp.asarray(int(row_pos), dtype=jnp.int32),
            )
        return fn(mtp_st, embed_st, head_st, freqs, input_ids, target_hidden_states, position_ids, mtp_cache)

    def draft(
        self,
        input_ids: Int[Array, "batch seq"],
        target_hidden_states: Float[Array, "batch seq hidden"] | None = None,
        target_kv_cache: typing.Any = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        return_full_log_probs: bool = False,
        sample: bool = False,
        rng_key: jax.Array | None = None,
        row_pos: int | None = None,
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
            row_pos: Optional per-request row for batched persist-with-rollback.
                ``None`` (default) is the legacy single-request path: the MTP
                cache batch equals the input batch and writes land at row 0.
                When given, the MTP cache is the shared request-pool cache and
                this request reads/writes only its own stable row ``row_pos``
                (the forward stays batch-1); other rows are untouched.

        Returns:
            :class:`DraftStep` with the predicted token at the LAST
            input position (i.e. one-step lookahead from
            ``input_ids[:, -1]``).
        """
        if target_hidden_states is None:
            raise ValueError("Qwen3_5MTPDrafter.draft requires target_hidden_states")

        if row_pos is None:
            mtp_cache = self._ensure_mtp_cache(int(input_ids.shape[0]))
        else:
            # Batched path: the cache is sized to the request pool (see
            # ``ensure_batch_size``); never shrink it to the batch-1 input here.
            mtp_cache = self._ensure_mtp_cache(max(int(self._mtp_cache_batch_size or 0), int(row_pos) + 1))
        hidden_states, logits, new_cache = self._mtp_hidden_and_logits(
            input_ids,
            target_hidden_states,
            position_ids,
            mtp_cache,
            None if row_pos is None else int(row_pos),
        )
        if new_cache is not None:
            self._mtp_cache = new_cache
            if row_pos is None:
                self._mtp_cache_batch_size = int(input_ids.shape[0])
            # else: keep the pool batch size — the batch-1 input is not the cache batch.
        last = logits[:, -1, :].astype(jnp.float32)
        if sample:
            # Drafter-internal sampling path (used by callers that consume the
            # drafter's own token + normalized log-probs directly). Keep the
            # honest log_softmax and the gathered per-token log-prob here.
            if rng_key is None:
                raise ValueError("rng_key required when sample=True")
            token_ids = jax.random.categorical(rng_key, last)
            log_probs = jax.nn.log_softmax(last, axis=-1)
            token_log_probs = jnp.take_along_axis(log_probs, token_ids[:, None], axis=-1).squeeze(-1)
            return DraftStep(
                token_ids=token_ids.astype(jnp.int32),
                log_probs=token_log_probs,
                full_log_probs=log_probs,
                hidden_states=hidden_states,
            )
        token_ids = jnp.argmax(last, axis=-1)
        if return_full_log_probs:
            # External-resample path (eSurge non-greedy speculative strategy):
            # the caller re-applies the request's top-k/top-p/min-p/temperature
            # filter and a final ``log_softmax`` before sampling + on-device
            # rejection (``DrafterSpeculation.filter_and_sample_log_probs`` ->
            # ``verify_sampled_window``). Return the RAW last-position logits and
            # skip the drafter-side ``log_softmax`` + the unused per-token gather:
            # the downstream filter is rank/softmax based and its final
            # ``log_softmax`` is shift-invariant, so a raw-logit input yields a
            # bit-identical filtered distribution, sampled token, and residual as
            # a pre-normalized one (masked positions collapse to exactly zero
            # probability either way). This removes redundant full-vocab passes
            # from the hot per-draft loop; the on-device rejection math is
            # unchanged. ``token_ids`` is the greedy pick, kept only for the
            # DraftStep contract — the strategy re-samples and ignores it.
            return DraftStep(
                token_ids=token_ids.astype(jnp.int32),
                log_probs=None,
                full_log_probs=last,
                hidden_states=hidden_states,
            )
        return DraftStep(
            token_ids=token_ids.astype(jnp.int32),
            log_probs=None,
            full_log_probs=None,
            hidden_states=hidden_states,
        )

    def _get_adaptive_fn(
        self,
        k_max: int,
        cur_token: jax.Array,
        cur_hidden: jax.Array,
        position: jax.Array,
        mtp_cache: TransformerCache | None,
        use_row: bool,
    ) -> typing.Callable:
        """Build (or fetch) the jitted early-exit draft ``while_loop`` for a shape.

        The returned callable runs the whole ``k_max``-step MTP recursion as a
        single ``jax.lax.while_loop`` whose ``cond`` early-exits once the top-1
        confidence drops below the (traced) threshold, so it executes FEWER MTP
        forwards on device when the drafter is unsure — without any per-step host
        sync. The compiled program is fixed (its trip count is data-dependent),
        so it compiles once per shape signature: ``conf_threshold`` and the target
        ``row`` are traced arguments, never part of the key, so varying the
        threshold or the early-exit point reuses one compiled program.

        Args:
            k_max: Maximum number of draft tokens (static: sizes the pre-allocated
                ``drafts`` buffer and bounds the loop).
            cur_token: ``(batch, 1)`` seed token, used only for its shape/dtype.
            cur_hidden: ``(batch, 1, hidden)`` seed hidden, used for shape/dtype.
            position: ``(batch, 1)`` seed position, used for shape/dtype.
            mtp_cache: The MTP cache pytree threaded through the carry (shape/dtype
                signature keys the compile), or ``None`` for the cache-free path.
            use_row: Whether the forward targets a single pool-cache row (the row
                is a traced argument, so all rows share one compiled program).

        Returns:
            A jitted ``fn(state, cur_token, cur_hidden, position, cache,
            conf_threshold, row) -> (drafts[k_max], valid_count, new_cache)``.
            ``state`` is the target model's weight State (threaded as an input so
            the weights are not captured constants).
        """
        has_cache = mtp_cache is not None
        key = (
            int(k_max),
            tuple(cur_token.shape),
            str(cur_token.dtype),
            tuple(cur_hidden.shape),
            str(cur_hidden.dtype),
            tuple(position.shape),
            str(position.dtype),
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
            bool(use_row),
        )
        fn = self._jit_adaptive.get(key)
        if fn is not None:
            return fn

        batch = int(cur_token.shape[0])
        # Cache the static GraphDef so the compiled while_loop closes over
        # structure only; the weight State is a runtime jit input (see
        # ``_model_graph_and_state``) and thus never a captured constant.
        graphdef, _ = self._model_graph_and_state()

        def _step(model, ct, ch, pos, cache_in, row):
            """One MTP forward -> (next_token[B], next_hidden[B,1,H], confidence, new_cache)."""
            h, logits, new_cache = self._mtp_forward_compute(model, ct, ch, pos, cache_in, row)
            last = logits[:, -1, :].astype(jnp.float32)
            probs = jax.nn.softmax(last, axis=-1)
            # Top-1 confidence = the argmax token's softmax prob == max softmax
            # prob; take the batch-min so the loop stops when the least-confident
            # row drops (batch is 1 on the eSurge single-request draft path).
            confidence = jnp.min(jnp.max(probs, axis=-1))
            next_tok = jnp.argmax(last, axis=-1).astype(jnp.int32)
            return next_tok, h[:, -1:, :], confidence, new_cache

        def _run(state, ct0, ch0, pos0, cache0, conf_threshold, row):
            model = spx.bind(graphdef, state)
            drafts0 = jnp.zeros((k_max,), dtype=jnp.int32)
            row_arg = row if use_row else None

            def _cond(carry):
                i = carry[0]
                keep = carry[-1]
                return jnp.logical_and(i < k_max, keep)

            def _body(carry):
                if has_cache:
                    i, ct, ch, pos, drafts, cache_in, _keep = carry
                else:
                    i, ct, ch, pos, drafts, _keep = carry
                    cache_in = None
                next_tok, next_hidden, confidence, new_cache = _step(model, ct, ch, pos, cache_in, row_arg)
                # Keep the carried hidden's dtype invariant across iterations (the
                # ``while_loop`` requires it). The seed hidden is model-sourced, so
                # this is an identity cast in the expected case (seed dtype == MTP
                # output dtype) and only guards a hypothetical mismatch.
                next_hidden = next_hidden.astype(ch0.dtype)
                drafts = drafts.at[i].set(next_tok[0])
                # Write token i, then decide whether to draft i+1: stop AFTER a
                # low-confidence token (it is still proposed and target-verified).
                # ``conf_threshold <= 0`` keeps this True (confidence in (0, 1]), so
                # the loop always runs the full k_max (fixed-K behavior).
                keep_next = confidence >= conf_threshold
                ct_next = next_tok.reshape(batch, 1)
                pos_next = pos + 1
                if has_cache:
                    return (i + 1, ct_next, next_hidden, pos_next, drafts, new_cache, keep_next)
                return (i + 1, ct_next, next_hidden, pos_next, drafts, keep_next)

            keep0 = jnp.asarray(True)
            if has_cache:
                init = (jnp.asarray(0, dtype=jnp.int32), ct0, ch0, pos0, drafts0, cache0, keep0)
            else:
                init = (jnp.asarray(0, dtype=jnp.int32), ct0, ch0, pos0, drafts0, keep0)
            final = jax.lax.while_loop(_cond, _body, init)
            valid_count = final[0]
            drafts_final = final[4]
            new_cache = final[5] if has_cache else None
            return drafts_final, valid_count, new_cache

        fn = jax.jit(_run)
        self._jit_adaptive[key] = fn
        return fn

    def draft_adaptive(
        self,
        seed_token: int,
        seed_hidden: Float[Array, "batch seq hidden"] | Float[Array, "hidden"],
        seed_position: int,
        k_max: int,
        conf_threshold: float,
        *,
        row_pos: int | None = None,
    ) -> tuple[Int[Array, "k_max"], Int[Array, ""]]:
        """Greedy adaptive-length draft: up to ``k_max`` tokens, early-exit on low confidence.

        Runs the ``k_max``-step MTP recursion as a single on-device
        ``jax.lax.while_loop`` (via :meth:`_get_adaptive_fn`) that stops once the
        MTP's top-1 confidence for the token it just produced falls below
        ``conf_threshold``. Because the exit test lives in the loop ``cond``, low
        confidence draws FEWER MTP forwards on device — the point of adaptive
        drafting is to spend deep drafts only while the MTP is confident. All draft
        tokens are still target-verified downstream, so this only changes HOW MANY
        tokens are proposed, never correctness; greedy spec output is unchanged.

        The MTP KV cache is carried through the loop and (when ``row_pos`` is given)
        each iteration reads/advances only that request's pool-cache row, so
        per-request persist-with-rollback keeps working inside the loop. With
        ``conf_threshold <= 0`` the loop always runs the full ``k_max`` and the
        result is byte-identical to ``k_max`` chained :meth:`draft` calls.

        Args:
            seed_token: Seed token id (position ``seed_position``).
            seed_hidden: Target hidden at the seed; ``(H,)`` or ``(batch, 1, H)``.
            seed_position: Absolute position of ``seed_token``.
            k_max: Maximum draft tokens to propose (the compiled loop bound).
            conf_threshold: Top-1 softmax-probability threshold below which the
                loop stops. ``<= 0`` disables early exit (drafts exactly ``k_max``).
            row_pos: Optional per-request pool-cache row for batched persist-with-
                rollback; ``None`` uses the legacy single-request cache.

        Returns:
            ``(drafts, valid_count)`` device arrays: ``drafts`` is the ``k_max``
            pre-allocated buffer (only ``drafts[:valid_count]`` are valid) and
            ``valid_count`` is the number of tokens actually drafted. The caller
            does a single host readback of both.
        """
        k_max = max(1, int(k_max))
        cur_token = jnp.asarray([[int(seed_token)]], dtype=jnp.int32)
        cur_hidden = jnp.asarray(seed_hidden)
        if cur_hidden.ndim == 1:
            cur_hidden = cur_hidden[None, None, :]
        position = jnp.asarray([[int(seed_position)]], dtype=jnp.int32)
        conf_threshold_arr = jnp.asarray(float(conf_threshold), dtype=jnp.float32)

        if row_pos is None:
            mtp_cache = self._ensure_mtp_cache(int(cur_token.shape[0]))
        else:
            # Batched path: keep the pool cache sized to the request pool (see
            # ``ensure_batch_size``); never shrink it to the batch-1 input here.
            mtp_cache = self._ensure_mtp_cache(max(int(self._mtp_cache_batch_size or 0), int(row_pos) + 1))

        use_row = row_pos is not None and mtp_cache is not None
        fn = self._get_adaptive_fn(k_max, cur_token, cur_hidden, position, mtp_cache, use_row)
        # Pass the live model State as the first jit input (never closed over),
        # so the weights stay out of the compiled program's captured constants.
        _, state = self._model_graph_and_state()
        row_arg = jnp.asarray(int(row_pos) if use_row else 0, dtype=jnp.int32)
        drafts, valid_count, new_cache = fn(
            state, cur_token, cur_hidden, position, mtp_cache, conf_threshold_arr, row_arg
        )
        if new_cache is not None:
            self._mtp_cache = new_cache
            if row_pos is None:
                self._mtp_cache_batch_size = int(cur_token.shape[0])
            # else: keep the pool batch size — the batch-1 input is not the cache batch.
        return drafts, valid_count

    def _get_batched_draft_fn(
        self,
        k: int,
        ids0: jax.Array,
        hidden0: jax.Array,
        pos0: jax.Array,
        mtp_cache: TransformerCache | None,
    ) -> typing.Callable:
        """Build (or fetch) the jitted fixed-``k`` batched-draft program for a shape.

        Unlike :meth:`_mtp_hidden_and_logits` (which slices ONE pool row for a
        batch-1 forward and scatters it back), this program runs the whole
        ``batch`` dimension through the MTP recurrence at once: the input is the
        FULL request pool ``[N, 1]`` and the MTP cache is the pool cache ``[N, …]``,
        so the ``k`` chained MTP forwards are ``k`` natural batched decodes rather
        than ``N * k`` row-sliced batch-1 forwards. Every pool row is advanced by
        the forward, so persistence is preserved with a pair of ``[N]`` index
        fixups computed inside the program:

        * ``committed_lens[r]`` rolls each *collected* row's write index back to
          its committed length before the forwards (per-row EAGLE rollback,
          vectorized as ``minimum(index, committed_lens)``); non-collected rows
          keep their index.
        * After the ``k`` forwards, each collected row keeps the advanced write
          index (``committed + k``) while every non-collected row's write index
          (and ``starts``) is RESTORED to its pre-forward value. A non-collected
          row's forward only writes at ``[index, index + k)`` (never inside its
          read window ``[0, index)``), so restoring the index leaves its trusted
          K/V untouched — the batched pass cannot corrupt a row a concurrent
          per-request draft (or an idle slot) owns.

        Because ``N`` is fixed for a runner, this compiles ONCE regardless of how
        many rows are live in a given step (idle rows ride along masked); the
        per-row rollback targets and the collected mask are traced arguments and
        never part of the compile key.

        Args:
            k: Number of draft tokens per row (static; unrolls the recurrence).
            ids0: ``(N, 1)`` seed tokens (shape/dtype only).
            hidden0: ``(N, 1, H)`` seed hidden (shape/dtype only).
            pos0: ``(N, 1)`` seed positions (shape/dtype only).
            mtp_cache: The pool ``TransformerCache`` (shape/dtype signature keys the
                compile) or ``None`` for the cache-free path.

        Returns:
            ``fn(state, ids0, hidden0, pos0, cache, committed_lens, collected_mask)
            -> (drafts[N, k], new_cache)``.
        """
        has_cache = mtp_cache is not None
        key = (
            int(k),
            tuple(ids0.shape),
            str(ids0.dtype),
            tuple(hidden0.shape),
            str(hidden0.dtype),
            tuple(pos0.shape),
            str(pos0.dtype),
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
        fn = self._jit_batched.get(key)
        if fn is not None:
            return fn

        # Eager warmup on the concrete model (NOT under a trace) materializes any
        # lazily-created state before it is split/traced (see
        # ``_model_graph_and_state``).
        _ = self._mtp_forward_compute(self.model, ids0, hidden0, pos0, mtp_cache, None)
        graphdefs, _states, _freqs, tie_embeddings = self._batched_bundle_graph_and_states()
        mtp_gd, embed_gd, head_gd = graphdefs

        def _run(mtp_st, embed_st, head_st, freqs, ids0, hidden0, pos0, cache0, committed_lens, collected_mask):
            mtp = spx.bind(mtp_gd, mtp_st)
            embed_module = spx.bind(embed_gd, embed_st)
            head_module = spx.bind(head_gd, head_st)
            orig_indexes: list = []
            orig_starts: list = []
            if has_cache:
                committed = committed_lens.astype(jnp.int32)
                rolled_views = []
                for v in cache0.views:
                    if v is None:
                        rolled_views.append(None)
                        orig_indexes.append(None)
                        orig_starts.append(None)
                        continue
                    orig_indexes.append(v.indexes)
                    orig_starts.append(v.starts)
                    idx = v.indexes.astype(jnp.int32)
                    rolled = jnp.where(collected_mask, jnp.minimum(idx, committed), idx)
                    rolled_views.append(v.replace(indexes=rolled.astype(v.indexes.dtype)))
                cache: TransformerCache | None = TransformerCache(views=rolled_views)
            else:
                cache = None

            ct, ch, pos = ids0, hidden0, pos0
            drafts_cols = []
            for _ in range(int(k)):
                h, logits, cache = self._mtp_forward_compute_parts(
                    mtp, embed_module, head_module, freqs, tie_embeddings, ct, ch, pos, cache
                )
                last = logits[:, -1, :].astype(jnp.float32)
                next_tok = jnp.argmax(last, axis=-1).astype(jnp.int32)
                drafts_cols.append(next_tok)
                ct = next_tok.reshape(-1, 1)
                ch = h[:, -1:, :].astype(hidden0.dtype)
                pos = pos + 1
            drafts = jnp.stack(drafts_cols, axis=1)

            if has_cache and cache is not None:
                fixed_views = []
                for v, orig_idx, orig_st in zip(cache.views, orig_indexes, orig_starts, strict=False):
                    if v is None:
                        fixed_views.append(None)
                        continue
                    adv_idx = v.indexes.astype(jnp.int32)
                    fixed_idx = jnp.where(collected_mask, adv_idx, orig_idx.astype(jnp.int32))
                    v2 = v.replace(indexes=fixed_idx.astype(v.indexes.dtype))
                    if orig_st is not None:
                        adv_st = v.starts
                        mask_st = collected_mask.reshape((-1,) + (1,) * (adv_st.ndim - 1))
                        v2 = v2.replace(starts=jnp.where(mask_st, adv_st, orig_st))
                    fixed_views.append(v2)
                cache = TransformerCache(views=fixed_views)
            return drafts, cache

        fn = jax.jit(_run)
        self._jit_batched[key] = fn
        return fn

    def draft_batched(
        self,
        *,
        seed_tokens: jax.Array,
        seed_hidden: jax.Array,
        seed_positions: jax.Array,
        committed_lens: jax.Array,
        collected_mask: jax.Array,
        num_speculative_tokens: int,
    ) -> Int[Array, "pool k"]:
        """Draft ``k`` greedy tokens for every pool row in ONE batched pass.

        This is the batched-serving counterpart of chaining :meth:`draft`
        per-request: all spec-active requests in a decode step are drafted
        together over the pooled MTP cache, so a running batch of ``N`` requests
        costs ``k`` batched MTP forwards instead of ``N * k`` batch-1 forwards.

        Args:
            seed_tokens: ``(N,)`` seed token id per pool row (dummy for idle /
                non-collected rows; those are masked out by ``collected_mask``).
            seed_hidden: ``(N, 1, H)`` (or ``(N, H)``) target hidden per pool row.
            seed_positions: ``(N,)`` absolute seed position per pool row.
            committed_lens: ``(N,)`` per-row EAGLE rollback target (the committed
                MTP context length); ignored for non-collected rows. Use ``0`` for
                the non-persistent per-window-clear behavior.
            collected_mask: ``(N,)`` boolean; ``True`` for rows actually drafted
                this step. Non-collected rows ride the batched forward but have
                their cache write index restored (never corrupted).
            num_speculative_tokens: Number of draft tokens ``k`` per row.

        Returns:
            ``(N, k)`` device array of drafted token ids. Row ``r`` is valid only
            when ``collected_mask[r]`` is ``True``.
        """
        ids0 = jnp.asarray(seed_tokens, dtype=jnp.int32).reshape(-1, 1)
        pos0 = jnp.asarray(seed_positions, dtype=jnp.int32).reshape(-1, 1)
        hidden0 = jnp.asarray(seed_hidden)
        if hidden0.ndim == 2:
            hidden0 = hidden0[:, None, :]
        n = int(ids0.shape[0])
        self.ensure_batch_size(n)
        mtp_cache = self._mtp_cache if self.uses_mtp_cache else None
        committed = jnp.asarray(committed_lens, dtype=jnp.int32).reshape(-1)
        mask = jnp.asarray(collected_mask).astype(bool).reshape(-1)
        fn = self._get_batched_draft_fn(int(num_speculative_tokens), ids0, hidden0, pos0, mtp_cache)
        _gds, (mtp_st, embed_st, head_st), freqs, _tie = self._batched_bundle_graph_and_states()
        drafts, new_cache = fn(mtp_st, embed_st, head_st, freqs, ids0, hidden0, pos0, mtp_cache, committed, mask)
        if new_cache is not None:
            self._mtp_cache = new_cache
        return drafts


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

        from easydel.inference.esurge.runners.spec import default_assistant_layer_mapping

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


def _drafter_target_layer_count(drafter: typing.Any) -> int:
    """Return how many target layers a standalone drafter concatenates.

    Reads ``drafter.config.target_layer_ids``; defaults to ``1`` when absent or
    not sized.

    Args:
        drafter: The standalone drafter model instance.

    Returns:
        The number of configured target layers.
    """
    cfg = getattr(drafter, "config", None)
    ids = getattr(cfg, "target_layer_ids", None)
    if ids is None:
        return 1
    try:
        return len(tuple(ids))
    except TypeError:
        return 1


@register_drafter("mtp", "qwen3_5_mtp", "qwen35_mtp", "inline_mtp")
def _build_mtp_drafter(
    target_model: typing.Any,
    *,
    num_draft_tokens: int = 1,
    assistant_model: typing.Any | None = None,
    target_embed_module: typing.Any | None = None,
    layer_mapping: list[int] | None = None,
    target_config: typing.Any | None = None,
    drafter_model: typing.Any | None = None,
    **kwargs: typing.Any,
) -> "Qwen3_5MTPDrafter":
    """Build the inline Qwen3.5 MTP drafter for ``target_model``.

    Args:
        target_model: Target causal-LM exposing ``has_mtp()`` + MTP head.
        num_draft_tokens: Speculative tokens proposed per verify window.
        assistant_model: Ignored (MTP is inline).
        target_embed_module: Ignored (MTP is inline).
        layer_mapping: Ignored (MTP is inline).
        target_config: Ignored (MTP reads ``target_model.config``).
        drafter_model: Ignored (MTP is inline).
        **kwargs: Forwarded to :class:`Qwen3_5MTPDrafter` (``use_cache``,
            ``cache_length``).

    Returns:
        A configured :class:`Qwen3_5MTPDrafter`.

    Raises:
        ValueError: If ``target_model`` has no inline MTP head.
    """
    del assistant_model, target_embed_module, layer_mapping, target_config, drafter_model
    if not bool(getattr(target_model, "has_mtp", lambda: False)()):
        raise ValueError(f"{type(target_model).__name__} does not expose an inline MTP head.")
    return Qwen3_5MTPDrafter(target_model, num_draft_tokens=max(1, int(num_draft_tokens)), **kwargs)


@register_drafter("gemma4_assistant", "assistant", "gemma4", "gemma4_assistant_drafter")
def _build_gemma4_assistant_drafter(
    target_model: typing.Any,
    *,
    num_draft_tokens: int = 1,
    assistant_model: typing.Any | None = None,
    target_embed_module: typing.Any | None = None,
    layer_mapping: list[int] | None = None,
    target_config: typing.Any | None = None,
    drafter_model: typing.Any | None = None,
    **kwargs: typing.Any,
) -> "Gemma4AssistantDrafter":
    """Build the standalone Gemma4 assistant drafter.

    Args:
        target_model: Target model whose embedding + config seed the assistant.
        num_draft_tokens: Speculative tokens proposed per verify window.
        assistant_model: The standalone ``Gemma4AssistantForCausalLM`` (also
            accepted via ``drafter_model``).
        target_embed_module: Optional target token embedding module; resolved
            from ``target_model`` when omitted.
        layer_mapping: Optional assistant-layer to target-layer mapping.
        target_config: Optional target config; defaults to ``target_model.config``.
        drafter_model: Alternative slot for ``assistant_model``.
        **kwargs: Forwarded to :class:`Gemma4AssistantDrafter`.

    Returns:
        A configured :class:`Gemma4AssistantDrafter`.

    Raises:
        ValueError: If no assistant model is supplied.
    """
    assistant = assistant_model if assistant_model is not None else drafter_model
    if assistant is None:
        raise ValueError("method='gemma4_assistant' requires assistant_model=...")
    drafter = Gemma4AssistantDrafter(
        assistant_model=assistant,
        target_embed_module=target_embed_module or target_model._resolve_drafter_target_embedding(),
        layer_mapping=layer_mapping,
        target_config=target_config if target_config is not None else getattr(target_model, "config", None),
        **kwargs,
    )
    drafter.num_draft_tokens = max(1, int(num_draft_tokens))
    return drafter


@register_drafter("eagle3")
@register_drafter("dspark")
@register_drafter("dflash")
def _build_model_native_drafter(
    target_model: typing.Any,
    *,
    num_draft_tokens: int = 1,
    assistant_model: typing.Any | None = None,
    target_embed_module: typing.Any | None = None,
    layer_mapping: list[int] | None = None,
    target_config: typing.Any | None = None,
    drafter_model: typing.Any | None = None,
    **kwargs: typing.Any,
) -> typing.Any:
    """Configure a standalone model-native drafter (EAGLE3 / DSpark / DFlash).

    The drafter is a separate ``EasyDeLBaseModule`` + :class:`SpecDecodeBase`
    instance supplied via ``drafter_model`` (or, for config-driven builds, the
    ``assistant_model`` slot). This builder validates it, records the target
    config + draft-token count, and returns it unchanged.

    Args:
        target_model: Target model whose config seeds the drafter.
        num_draft_tokens: Speculative tokens proposed per verify window.
        assistant_model: Fallback slot for ``drafter_model``.
        target_embed_module: Ignored (model-native drafters own their embedding).
        layer_mapping: Ignored (the runner seeds a single hidden per window).
        target_config: Optional target config; defaults to ``target_model.config``.
        drafter_model: The standalone drafter instance.
        **kwargs: Ignored.

    Returns:
        The configured drafter instance.

    Raises:
        ValueError: If no drafter instance is supplied, or it is a multi-layer
            standalone drafter (which the single-hidden runner seed cannot feed).
        TypeError: If the supplied drafter is not a :class:`SpecDecodeBase`.
    """
    del target_embed_module, layer_mapping, kwargs
    drafter = drafter_model if drafter_model is not None else assistant_model
    if drafter is None:
        raise ValueError(
            "Model-native drafters (eagle3/dspark/dflash) require a standalone "
            "drafter_model=... (an EasyDeLBaseModule + SpecDecodeBase instance)."
        )
    if not isinstance(drafter, SpecDecodeBase):
        raise TypeError(f"drafter_model must be an EasyDeLBaseModule + SpecDecodeBase; got {type(drafter).__name__}.")
    # Seed-hidden shape gate (5b): the eSurge runner seeds the drafter with ONE
    # target hidden state per verify window, but a multi-layer standalone
    # drafter's `extract_context_feature` expects `len(target_layer_ids)`
    # concatenated layers. Fail early with a clear message instead of a
    # far-away shape error deep inside the drafter forward.
    # TODO(5a): wire a multi-layer runner-seed gather so multi-layer standalone
    #   drafters can be seeded directly; that change lives in the recurrent /
    #   hidden-gather path of model_runner.py and is intentionally out of scope
    #   here.
    n_layers = _drafter_target_layer_count(drafter)
    if n_layers > 1:
        raise ValueError(
            f"{type(drafter).__name__} is configured with {n_layers} target layers "
            "(target_layer_ids), but eSurge runner-native speculative decoding seeds the "
            "drafter with a single target hidden state per window. Configure the drafter "
            "with a single target layer, or pre-concatenate the target hidden states. "
            "Multi-layer runner seeding is not yet wired."
        )
    drafter.target_config = target_config if target_config is not None else getattr(target_model, "config", None)
    if hasattr(drafter, "set_num_draft_tokens"):
        drafter.set_num_draft_tokens(int(num_draft_tokens))
    else:
        drafter.num_draft_tokens = max(1, int(num_draft_tokens))
    return drafter


__all__ = [
    "DraftStep",
    "DrafterProtocol",
    "Gemma4AssistantDrafter",
    "Qwen3_5MTPDrafter",
    "accept_or_reject",
    "resample_rejected",
]
