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

"""Runner-native speculative decoding as a pluggable strategy.

:class:`DrafterSpeculation` owns every piece of the eSurge runner's
draft/verify/commit machinery that used to live inline on
:class:`~easydel.inference.esurge.runners.model_runner.eSurgeRunner` as
``_spec_*`` methods: drafter invocation, verify/resample JIT caches,
rejection-backoff state, recurrent-candidate commits, KV replay, and the
acceptance counters. The code was moved verbatim; only attribute paths
changed (runner-owned state is reached through the documented ``_runner``
back-reference).
"""

from __future__ import annotations

import os
import time
import typing
from bisect import bisect_left
from dataclasses import dataclass

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from ...core.binary_search import apply_topk_mask, apply_topp_mask

if typing.TYPE_CHECKING:
    from easydel.inference.speculative import DrafterProtocol

    from ...scheduler import SchedulerOutput
    from ..model_runner import eSurgeRunner
    from ..states import CachedRequestState

logger = get_logger("eSurge")


@dataclass(frozen=True)
class _SpecVerifyMetadata:
    """Host-side token/index contract for one speculative verification window."""

    request_id: str
    row_pos: int
    req_idx: int
    start_pos: int
    real_count: int
    target_local_indices: list[int]
    bonus_local_index: int
    draft_token_positions: list[int]
    scheduled_draft_tokens: list[int]
    buffer_draft_tokens: list[int]

    @property
    def num_drafts(self) -> int:
        """Number of draft tokens described by this verification metadata."""
        return len(self.buffer_draft_tokens)


def _window_hidden_row(
    hidden_states: jax.Array,
    *,
    row_pos: int,
    token_offset: int,
    token_index: int,
    total_window_tokens: int,
    padded_reqs: int,
) -> jax.Array:
    """Return the hidden row for a window token, tolerating old gathered outputs."""
    hidden_len = int(hidden_states.shape[0])
    full_index = int(token_offset) + int(token_index)
    if hidden_len >= int(total_window_tokens) and 0 <= full_index < hidden_len:
        return hidden_states[full_index]
    if hidden_len >= int(padded_reqs) and 0 <= int(row_pos) < hidden_len:
        return hidden_states[int(row_pos)]
    return hidden_states[min(max(full_index, 0), hidden_len - 1)]


def _copy_kv_tree(kv_pages: typing.Any) -> typing.Any:
    """Return a deep copy of every device array leaf in a KV-pages pytree."""
    return jax.tree_util.tree_map(
        lambda leaf: jnp.array(leaf, copy=True) if isinstance(leaf, jax.Array) else leaf,
        kv_pages,
    )


class DrafterSpeculation:
    """Runner-native speculative decoding driven by an attached drafter.

    Owns the drafter object, the verify/resample JIT caches, the
    rejection-backoff map, debug traces, and the acceptance counters.

    The strategy collaborates with the runner through a documented
    back-reference (``self._runner``). The moved methods touch well over six
    live runner handles (sequence buffer, request map, executor manager,
    cache metadata, KV-cache groups, token buckets, host scratch buffers,
    device input buffers, pending recurrent-commit queue), several of which
    are re-created by ``eSurgeRunner._setup_variables`` on weight updates —
    so holding the runner handle, rather than snapshotting individual
    collaborators, keeps this code byte-identical in behavior to its
    pre-extraction form.
    """

    def __init__(
        self,
        *,
        runner: eSurgeRunner,
        drafter: DrafterProtocol,
        num_speculative_tokens: int,
    ) -> None:
        """Initialize the drafter-backed strategy.

        Args:
            runner: Owning :class:`eSurgeRunner`; documented back-reference
                used to reach live runner state.
            drafter: Speculative drafter implementing
                :class:`~easydel.inference.speculative.DrafterProtocol`.
            num_speculative_tokens: Draft tokens proposed per verify window.
        """
        self._runner = runner
        self.drafter = drafter
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.num_drafts_generated = 0
        self.num_drafts_accepted = 0
        self.num_verify_steps = 0
        self.debug_traces: list[dict[str, typing.Any]] = []
        self.debug_max_traces = int(os.environ.get("EASYDEL_SPEC_DECODE_DEBUG_STEPS", "0") or 0)
        self.draft_full_log_probs_by_req: dict[str, jax.Array] = {}
        self._filter_sample_fns: dict[tuple, typing.Callable] = {}
        self._verify_sample_fns: dict[tuple, typing.Callable] = {}
        self._recurrent_commit_fn: typing.Callable | None = None
        self._recurrent_commit_candidate_count: int = 0
        self.reject_backoff_steps = 0
        self._backoff_by_req: dict[str, int] = {}

    @property
    def uses_recurrent_candidates(self) -> bool:
        """Whether hybrid recurrent caches keep speculative candidate rows."""
        return bool(getattr(self._runner, "spec_decode_recurrent_candidates", False))

    def on_reset(self) -> None:
        """Clear per-request speculative state and reset the drafter.

        Mirrors the speculative portion of ``eSurgeRunner.reset_state``.
        """
        self.draft_full_log_probs_by_req.clear()
        self._backoff_by_req.clear()
        if self.drafter is not None:
            try:
                self.drafter.reset(batch_size=max(1, int(self._runner.max_num_seqs)))
            except Exception:
                logger.debug("Speculative drafter reset failed.", exc_info=True)

    def is_greedy_request(self, req_state: CachedRequestState | None) -> bool:
        """Return whether speculative decoding should use the greedy path for ``req_state``.

        Args:
            req_state: The cached request state to inspect; ``None`` short-circuits to False.

        Returns:
            ``True`` when a drafter is attached and the request's sampling
            temperature is non-positive (greedy).
        """
        if self.drafter is None or req_state is None:
            return False
        sampling_params = req_state.sampling_params
        temperature = float(getattr(sampling_params, "temperature", 0.0) or 0.0)
        return temperature <= 0.0

    def is_spec_request(self, req_state: CachedRequestState | None) -> bool:
        """Return whether ``req_state`` is eligible for speculative decoding.

        Requests using sampling penalties, ``n > 1``, or no attached drafter
        fall back to the standard single-token decode path because the runner's
        spec-verify kernels don't support those features yet.

        Args:
            req_state: Cached request state to inspect.

        Returns:
            ``True`` if the drafter is set, ``num_speculative_tokens > 0``, and
            the request uses default-shaped sampling parameters; ``False``
            otherwise.
        """
        if self.drafter is None or req_state is None or self.num_speculative_tokens <= 0:
            return False
        sampling_params = req_state.sampling_params
        if getattr(sampling_params, "n", 1) != 1:
            return False
        if getattr(sampling_params, "presence_penalty", 0.0) != 0.0:
            return False
        if getattr(sampling_params, "frequency_penalty", 0.0) != 0.0:
            return False
        if getattr(sampling_params, "repetition_penalty", 1.0) != 1.0:
            return False
        return True

    def filtered_log_probs(
        self,
        logits_or_log_probs: jax.Array,
        req_state: CachedRequestState,
    ) -> jax.Array:
        """Apply the request's filtering+temperature and return log probabilities.

        Used by the eager (non-jit) speculative paths to score draft tokens
        against the target distribution under the same top-k / top-p / min-p
        filters that the runner sampler would have applied for this request.

        Args:
            logits_or_log_probs: Per-row logits (or already-log probabilities)
                with shape ``[..., vocab_size]``.
            req_state: Request state providing the sampling parameters.

        Returns:
            Log-softmax values of the filtered/scaled logits along the last
            axis, ``float32``.
        """
        sampling_params = req_state.sampling_params
        logits = logits_or_log_probs.astype(jnp.float32)
        min_val = -1e10

        top_k = int(getattr(sampling_params, "top_k", 0) or 0)
        if top_k > 0:
            logits = apply_topk_mask(logits, jnp.asarray(top_k, dtype=jnp.int32), min_val)

        top_p = float(getattr(sampling_params, "top_p", 1.0) or 1.0)
        if top_p < 1.0:
            logits = apply_topp_mask(logits, jnp.asarray(top_p, dtype=jnp.float32), min_val)

        temperature = float(getattr(sampling_params, "temperature", 1.0) or 1.0)
        if temperature > 0.0 and temperature != 1.0:
            logits = logits / jnp.asarray(temperature, dtype=jnp.float32)

        min_p = float(getattr(sampling_params, "min_p", 0.0) or 0.0)
        if min_p > 0.0:
            max_logits = jnp.max(logits, axis=-1, keepdims=True)
            logits = jnp.where(
                logits >= (jnp.asarray(min_p, dtype=jnp.float32) * max_logits),
                logits,
                jnp.full_like(logits, min_val),
            )

        return jax.nn.log_softmax(logits, axis=-1)

    def rng_split(self, n: int = 1) -> list[jax.Array]:
        """Advance the manager-owned RNG by ``n`` and return ``n`` fresh keys.

        Splits ``executor_manager.rng_key`` into ``n + 1`` subkeys, keeps
        the first as the new state, and hands out the remaining ``n``.

        Args:
            n: Number of fresh keys to return.

        Returns:
            List of ``n`` fresh ``jax.Array`` random keys.
        """
        keys = jax.random.split(self._runner.executor_manager.rng_key, int(n) + 1)
        self._runner.executor_manager.rng_key = keys[0]
        return list(keys[1:])

    def sample_distribution(self, log_probs: jax.Array, req_state: CachedRequestState) -> int:
        """Sample one token id from ``log_probs`` using the request's mode.

        Returns the argmax for greedy requests; otherwise draws a categorical
        sample with a freshly split RNG key.

        Args:
            log_probs: Per-position log probabilities with vocabulary as the
                last axis.
            req_state: Cached request state used to detect the greedy path.

        Returns:
            Sampled token id as a Python int.
        """
        if self.is_greedy_request(req_state):
            return int(np.asarray(jnp.argmax(log_probs, axis=-1)).reshape(-1)[0])
        key = self.rng_split(1)[0]
        return int(np.asarray(jax.random.categorical(key, log_probs)).reshape(-1)[0])

    @staticmethod
    def filter_logits_for_sampling(
        logits_or_log_probs: jax.Array,
        *,
        temperature: jax.Array,
        top_p: jax.Array,
        top_k: jax.Array,
        min_p: jax.Array,
    ) -> jax.Array:
        """Apply the runner's supported sampling filters before log-softmax.

        Conditionally applies top-k masking, top-p (nucleus) masking, temperature
        scaling, and min-p masking using ``jax.lax.cond`` so the function stays
        jit-traceable with scalar gate values. Filters that are at their no-op
        value (``top_k <= 0``, ``top_p >= 1``, ``temperature <= 0``,
        ``min_p <= 0``) are skipped.

        Args:
            logits_or_log_probs: Logits (or log probabilities) with shape
                ``[..., vocab_size]``; cast to float32 internally.
            temperature: Scalar device array temperature divisor.
            top_p: Scalar device array nucleus probability threshold.
            top_k: Scalar device array top-k count.
            min_p: Scalar device array min-p relative threshold.

        Returns:
            Filtered logits (float32) with masked entries set to a large
            negative value, ready for ``log_softmax``.
        """
        logits = logits_or_log_probs.astype(jnp.float32)
        min_val = -1e10

        logits = jax.lax.cond(
            top_k > 0,
            lambda legi: apply_topk_mask(legi, top_k.astype(jnp.int32), min_val),
            lambda legi: legi,
            logits,
        )
        logits = jax.lax.cond(
            top_p < 1.0,
            lambda legi: apply_topp_mask(legi, top_p.astype(jnp.float32), min_val),
            lambda legi: legi,
            logits,
        )
        logits = jax.lax.cond(
            temperature > 0.0,
            lambda legi: legi / temperature.astype(jnp.float32),
            lambda legi: legi,
            logits,
        )

        def _apply_min_p(legi):
            """Mask logits below ``min_p * max(logits)`` to ``min_val``."""
            max_logits = jnp.max(legi, axis=-1, keepdims=True)
            return jnp.where(
                legi >= (min_p.astype(jnp.float32) * max_logits),
                legi,
                jnp.full_like(legi, min_val),
            )

        return jax.lax.cond(min_p > 0.0, _apply_min_p, lambda legi: legi, logits)

    def sampling_args(
        self,
        req_state: CachedRequestState,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Pack a request's sampling parameters into device scalars for spec kernels.

        Args:
            req_state: Cached request state to read sampling parameters from.

        Returns:
            Tuple of ``(temperature, top_p, top_k, min_p)`` device scalars in
            the dtypes expected by :meth:`filter_logits_for_sampling`.
        """
        sampling_params = req_state.sampling_params
        temperature = float(getattr(sampling_params, "temperature", 1.0) or 1.0)
        top_p = float(getattr(sampling_params, "top_p", 1.0) or 1.0)
        top_k = int(getattr(sampling_params, "top_k", 0) or 0)
        min_p = float(getattr(sampling_params, "min_p", 0.0) or 0.0)
        return (
            jnp.asarray(temperature, dtype=jnp.float32),
            jnp.asarray(top_p, dtype=jnp.float32),
            jnp.asarray(top_k, dtype=jnp.int32),
            jnp.asarray(min_p, dtype=jnp.float32),
        )

    def filter_and_sample_log_probs(
        self,
        logits_or_log_probs: jax.Array,
        req_state: CachedRequestState,
    ) -> tuple[jax.Array, jax.Array]:
        """Return sampled token IDs and filtered log-probs using one cached JIT.

        Looks up (or compiles and caches, keyed by shape+dtype) a jitted filter +
        categorical sampler, then runs it with the request's sampling scalars and
        a freshly split RNG key.

        Args:
            logits_or_log_probs: Per-row logits (or log probabilities) with shape
                ``[..., vocab_size]``.
            req_state: Request state providing the sampling parameters.

        Returns:
            Tuple ``(token_ids, log_probs)`` where ``token_ids`` are the sampled
            int32 ids and ``log_probs`` are the filtered log-softmax values.
        """
        key = (tuple(logits_or_log_probs.shape), str(logits_or_log_probs.dtype))
        fn = self._filter_sample_fns.get(key)
        if fn is None:

            @jax.jit
            def _fn(logits, rng_key, temperature, top_p, top_k, min_p):
                """JIT-compiled filter + categorical sampler for spec drafts."""
                filtered = DrafterSpeculation.filter_logits_for_sampling(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                log_probs = jax.nn.log_softmax(filtered, axis=-1)
                token_ids = jax.random.categorical(rng_key, log_probs).astype(jnp.int32)
                return token_ids, log_probs

            fn = _fn
            self._filter_sample_fns[key] = fn

        temperature, top_p, top_k, min_p = self.sampling_args(req_state)
        return fn(
            logits_or_log_probs,
            self.rng_split(1)[0],
            temperature,
            top_p,
            top_k,
            min_p,
        )

    def verify_sampled_window(
        self,
        *,
        target_logits: jax.Array,
        draft_log_probs: jax.Array,
        draft_tokens: list[int],
        req_state: CachedRequestState,
    ) -> tuple[int, int]:
        """Compiled rejection sampler for one non-greedy spec window.

        Looks up (or compiles and caches by shape) a jitted speculative
        rejection sampler that verifies each draft token against the target
        distribution, drawing a residual-distribution replacement on the first
        rejection and a bonus token if all drafts are accepted.

        Args:
            target_logits: Target-model logits for the draft + bonus positions,
                shape ``[num_drafts + 1, vocab_size]``.
            draft_log_probs: Draft-model log probabilities for the draft
                positions, shape ``[num_drafts, vocab_size]``.
            draft_tokens: The drafted token ids being verified.
            req_state: Request state providing the sampling parameters.

        Returns:
            Tuple ``(accepted, corrected)`` where ``accepted`` is the number of
            leading accepted draft tokens and ``corrected`` is the replacement
            or bonus token id (both Python ints).
        """
        draft_token_arr = jnp.asarray(draft_tokens, dtype=jnp.int32)
        key = (
            tuple(target_logits.shape),
            str(target_logits.dtype),
            tuple(draft_log_probs.shape),
            str(draft_log_probs.dtype),
            tuple(draft_token_arr.shape),
        )
        fn = self._verify_sample_fns.get(key)
        if fn is None:

            @jax.jit
            def _fn(target_logits_i, draft_log_probs_i, draft_token_ids, rng_key, temperature, top_p, top_k, min_p):
                """JIT-compiled rejection sampler for one non-greedy spec window."""
                target_filtered = DrafterSpeculation.filter_logits_for_sampling(
                    target_logits_i,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                target_log_probs = jax.nn.log_softmax(target_filtered, axis=-1)
                num_drafts = draft_token_ids.shape[0]
                keys = jax.random.split(rng_key, num_drafts * 2 + 1)

                def _resample_residual(t_lp, d_lp, key_i):
                    """Draw a single replacement from the ``(p_target - p_draft)+`` residual."""
                    p_target = jnp.exp(t_lp)
                    p_draft = jnp.exp(d_lp)
                    residual = jnp.maximum(p_target - p_draft, 0.0)
                    residual = residual / jnp.maximum(jnp.sum(residual, axis=-1, keepdims=True), 1e-9)
                    return jax.random.categorical(key_i, jnp.log(residual + 1e-9)).astype(jnp.int32)[0]

                def _body(i, carry):
                    """One rejection-sampling step over draft index ``i`` carrying accept state."""
                    accepted, corrected, rejected = carry
                    tok = draft_token_ids[i]
                    target_full = jax.lax.dynamic_slice_in_dim(target_log_probs, i, 1, axis=0)
                    draft_full = jax.lax.dynamic_slice_in_dim(draft_log_probs_i, i, 1, axis=0)
                    tok_2d = tok.reshape(1, 1)
                    d_lp = jnp.take_along_axis(draft_full, tok_2d, axis=-1).squeeze()
                    t_lp = jnp.take_along_axis(target_full, tok_2d, axis=-1).squeeze()
                    accept_prob = jnp.minimum(1.0, jnp.exp(t_lp - d_lp))
                    accept = jax.random.uniform(keys[i], (), dtype=jnp.float32) < accept_prob
                    replacement = _resample_residual(target_full, draft_full, keys[num_drafts + i])
                    still_verifying = ~rejected
                    take_accept = still_verifying & accept
                    take_reject = still_verifying & (~accept)
                    accepted = accepted + take_accept.astype(jnp.int32)
                    corrected = jnp.where(take_reject, replacement, corrected)
                    rejected = rejected | take_reject
                    return accepted, corrected, rejected

                accepted, corrected, rejected = jax.lax.fori_loop(
                    0,
                    num_drafts,
                    _body,
                    (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)),
                )
                bonus = jax.random.categorical(keys[-1], target_log_probs[num_drafts : num_drafts + 1]).astype(
                    jnp.int32
                )[0]
                corrected = jnp.where(rejected, corrected, bonus)
                return accepted, corrected

            fn = _fn
            self._verify_sample_fns[key] = fn

        temperature, top_p, top_k, min_p = self.sampling_args(req_state)
        accepted, corrected = fn(
            target_logits,
            draft_log_probs,
            draft_token_arr,
            self.rng_split(1)[0],
            temperature,
            top_p,
            top_k,
            min_p,
        )
        return (
            int(np.asarray(accepted).reshape(-1)[0]),
            int(np.asarray(corrected).reshape(-1)[0]),
        )

    def build_verify_metadata(
        self,
        *,
        request_id: str,
        row_pos: int,
        req_idx: int,
        start_pos: int,
        real_count: int,
        scheduled_draft_tokens: list[int],
        token_ids_window_cpu: np.ndarray,
        token_offset: int,
    ) -> _SpecVerifyMetadata:
        """Build target-logit/draft-token verification indices.

        A hidden/logit row for token position ``p`` predicts the token at
        ``p + 1``. For a spec window containing ``real_count`` real tokens
        followed by ``N`` drafts, target rows are the local rows for
        ``start + real_count - 1 + i`` and draft tokens are read from the
        materialized input buffer at ``start + real_count + i``.

        Args:
            request_id: Identifier of the request being verified.
            row_pos: Window-local row index of the request.
            req_idx: Sequence-buffer row index of the request.
            start_pos: Absolute sequence position of the first scheduled token.
            real_count: Number of confirmed real tokens before the drafts.
            scheduled_draft_tokens: Draft token ids the scheduler provided.
            token_ids_window_cpu: CPU token buffer for the active window, indexed
                by ``row_pos`` to recover the materialized draft tokens.
            token_offset: Flattened token offset of this row within the window
                (start of its hidden/logit rows).

        Returns:
            A :class:`_SpecVerifyMetadata` capturing the target/bonus local
            indices, draft token positions, and both the scheduled and the
            buffer-materialized draft tokens.
        """
        num_drafts = len(scheduled_draft_tokens)
        target_start = int(token_offset) + int(real_count) - 1
        target_local_indices = [target_start + i for i in range(num_drafts)]
        bonus_local_index = target_start + num_drafts
        draft_token_positions = [int(start_pos) + int(real_count) + i for i in range(num_drafts)]

        row_tokens = np.asarray(token_ids_window_cpu[int(row_pos)])
        buffer_draft_tokens: list[int] = []
        for i, pos in enumerate(draft_token_positions):
            if 0 <= pos < int(row_tokens.shape[0]):
                buffer_draft_tokens.append(int(row_tokens[pos]))
            else:
                buffer_draft_tokens.append(int(scheduled_draft_tokens[i]))

        return _SpecVerifyMetadata(
            request_id=str(request_id),
            row_pos=int(row_pos),
            req_idx=int(req_idx),
            start_pos=int(start_pos),
            real_count=int(real_count),
            target_local_indices=target_local_indices,
            bonus_local_index=int(bonus_local_index),
            draft_token_positions=draft_token_positions,
            scheduled_draft_tokens=[int(t) for t in scheduled_draft_tokens],
            buffer_draft_tokens=buffer_draft_tokens,
        )

    def record_verify_trace(
        self,
        *,
        meta: _SpecVerifyMetadata,
        logits_rows: jax.Array | np.ndarray | None,
        accepted: int,
        corrected_token: int,
        greedy: bool,
        source: str,
    ) -> None:
        """Record a small host trace for speculative acceptance diagnosis.

        No-op unless ``debug_max_traces`` is positive and the trace
        budget is not yet exhausted. Builds a per-draft diagnostic record
        (scheduler-vs-buffer token match, target argmax, and the draft token's
        rank under the target distribution) and appends it to
        ``debug_traces`` while logging it at WARNING.

        Args:
            meta: Verification metadata describing the window's draft tokens.
            logits_rows: Per-draft target logits (jax/numpy) used to compute
                argmax/rank diagnostics, or ``None`` to skip those fields.
            accepted: Number of accepted draft tokens for this window.
            corrected_token: Replacement or bonus token id chosen by verify.
            greedy: Whether the request used the greedy spec path.
            source: Tag identifying the verify code path (e.g. ``"full-hidden"``
                or ``"replay"``).
        """
        if self.debug_max_traces <= 0:
            return
        if len(self.debug_traces) >= self.debug_max_traces:
            return

        rows: list[dict[str, typing.Any]] = []
        logits_np = np.asarray(logits_rows) if logits_rows is not None else None
        for i, draft_token in enumerate(meta.buffer_draft_tokens):
            row: dict[str, typing.Any] = {
                "draft_index": int(i),
                "target_local_index": int(meta.target_local_indices[i]),
                "draft_token_position": int(meta.draft_token_positions[i]),
                "scheduled_draft_token": int(meta.scheduled_draft_tokens[i]),
                "buffer_draft_token": int(draft_token),
                "scheduler_buffer_match": bool(int(meta.scheduled_draft_tokens[i]) == int(draft_token)),
            }
            if logits_np is not None and i < int(logits_np.shape[0]):
                row_logits = logits_np[i]
                draft_id = int(draft_token)
                target_argmax = int(np.argmax(row_logits))
                row["target_argmax"] = target_argmax
                if 0 <= draft_id < int(row_logits.shape[-1]):
                    draft_logit = float(row_logits[draft_id])
                    row["draft_rank_under_target"] = int(np.count_nonzero(row_logits > draft_logit) + 1)
                else:
                    row["draft_rank_under_target"] = None
            rows.append(row)

        trace = {
            "request_id": meta.request_id,
            "row_pos": int(meta.row_pos),
            "req_idx": int(meta.req_idx),
            "start_pos": int(meta.start_pos),
            "real_count": int(meta.real_count),
            "bonus_local_index": int(meta.bonus_local_index),
            "accepted": int(accepted),
            "corrected_token": int(corrected_token),
            "greedy": bool(greedy),
            "source": source,
            "rows": rows,
        }
        self.debug_traces.append(trace)
        logger.warning("Spec decode verify trace: %s", trace)

    def on_states_applied(self, scheduler_output: SchedulerOutput) -> None:
        """Write scheduler-provided speculative tokens into the sequence buffer.

        For each request that has scheduled draft tokens this step, writes the
        prefix of confirmed real tokens followed by the draft tokens into the
        request's row in the sequence buffer. Updates ``num_tokens_no_spec`` so
        the runner remembers where verified tokens end and drafts begin.

        Args:
            scheduler_output: Scheduler decision carrying the per-request
                ``scheduled_spec_decode_tokens`` mapping.
        """
        if self.drafter is None or not scheduler_output.scheduled_spec_decode_tokens:
            return
        for req_id, spec_tokens in scheduler_output.scheduled_spec_decode_tokens.items():
            if not spec_tokens:
                continue
            req_state = self._runner.requests.get(req_id)
            req_idx = self._runner.sequence_buffer.req_id_to_index.get(req_id)
            if req_state is None or req_idx is None:
                continue
            scheduled = int(scheduler_output.num_scheduled_tokens.get(req_id, 0))
            start = int(self._runner.sequence_buffer.num_computed_tokens[req_idx])
            real_count = scheduled - len(spec_tokens)
            if scheduled <= 0 or real_count <= 0:
                continue
            all_known_tokens = list(req_state.prompt_token_ids) + list(req_state.output_token_ids)
            real_tokens = all_known_tokens[start : start + real_count]
            if len(real_tokens) != real_count:
                logger.debug(
                    "Skipping spec-token materialization for %s: need %d real tokens at %d, have %d.",
                    req_id,
                    real_count,
                    start,
                    len(real_tokens),
                )
                continue
            full_tokens = real_tokens + [int(t) for t in spec_tokens]
            end = start + len(full_tokens)
            if end > self._runner.max_model_len:
                continue
            self._runner.sequence_buffer.token_ids[req_idx, start:end] = np.asarray(full_tokens, dtype=np.int32)
            self._runner.sequence_buffer.num_tokens_no_spec[req_idx] = start + real_count
            self._runner.sequence_buffer.num_tokens[req_idx] = end

    def project_hidden_rows(self, hidden_rows: jax.Array) -> jax.Array:
        """Project arbitrary-row hidden states through the LM head for spec verify.

        The standard runner LM head is compiled for the active ``padded_num_reqs``
        bucket; speculative verification often inspects a different number of
        rows. This method looks up (or compiles on demand) an LM-head executable
        sized for the actual ``hidden_rows`` and returns the resulting logits.

        Args:
            hidden_rows: Hidden-state rows shaped ``[num_rows, hidden_dim]``.

        Returns:
            Vocabulary logits of shape ``[num_rows, vocab_size]``.
        """
        num_rows = int(hidden_rows.shape[0])
        model_executor = self._runner.executor_manager._model_executor
        try:
            lm_head_fn = model_executor.get_lm_head(padded_num_reqs=num_rows)
        except KeyError:
            graphdef, graphstate, graphother, inputs = self._runner.executor_manager.get_compile_configurations(
                self._runner.executor_manager.kv_pages,
                self._runner.executor_manager.rng_key,
                self._runner.max_num_reqs,
                self._runner.max_num_reqs,
                self._runner.max_num_reqs,
                self._runner.metadata,
            )
            model_executor.compile_lm_head(
                padded_num_reqs=num_rows,
                graphdef=graphdef,
                graphstate=graphstate,
                graphother=graphother,
                inputs=inputs,
                dtype=hidden_rows.dtype,
            )
            lm_head_fn = model_executor.get_lm_head(padded_num_reqs=num_rows)
        graphstate_arg, graphother_arg = model_executor.runtime_graph_args()
        return lm_head_fn(graphstate_arg, graphother_arg, hidden_rows)

    def cache_group_index_for_layer(self, layer_idx: int) -> int | None:
        """Resolve the eSurge page-table group that owns a target layer's K/V.

        Args:
            layer_idx: Zero-based transformer layer index to look up.

        Returns:
            The cache-group index whose ``layer_names`` contains
            ``"layer.{layer_idx}"`` (or ``0`` when there are no groups / a single
            unnamed group), or ``None`` when no group claims the layer.
        """
        if not self._runner.kv_cache_groups:
            return 0
        target = f"layer.{int(layer_idx)}"
        for group_idx, group in enumerate(self._runner.kv_cache_groups):
            layer_names = getattr(group, "layer_names", None)
            if layer_names is None and len(self._runner.kv_cache_groups) == 1:
                return group_idx
            if layer_names is not None and target in set(layer_names):
                return group_idx
        return None

    def gather_paged_kv_for_request(
        self,
        view: typing.Any,
        *,
        group_idx: int,
        req_idx: int,
        kv_len: int,
    ) -> tuple[jax.Array, jax.Array] | None:
        """Gather contiguous K/V slices from paged storage for one request.

        Reads the request's page-table row for ``group_idx``, then assembles
        a dense ``[1, kv_len, ...]`` key/value pair by indexing
        ``view.key_pages`` / ``view.value_pages`` with the request's logical
        token positions. Used by speculative drafters that need a contiguous
        target-side K/V cache view for cross-attention.

        Args:
            view: Cache view exposing ``key_pages`` and ``value_pages``.
            group_idx: KV cache page-table group index that owns the layer.
            req_idx: Request row index in the sequence buffer.
            kv_len: Number of K/V positions to gather (clamped to actual size).

        Returns:
            Tuple ``(keys, values)`` of arrays with a leading batch axis of 1,
            or ``None`` when the view lacks paged K/V or no pages exist.
        """
        page_table = self._runner.sequence_buffer.page_table[group_idx]
        page_table_cpu = page_table.get_cpu_tensor()
        page_size = int(
            getattr(getattr(view, "metadata", None), "page_size", self._runner.page_size) or self._runner.page_size
        )
        page_size = max(1, page_size)
        pages_needed = min(int(page_table_cpu.shape[1]), max(1, (int(kv_len) + page_size - 1) // page_size))
        if pages_needed <= 0:
            return None
        kv_len = min(int(kv_len), pages_needed * page_size)

        page_ids_cpu = np.zeros((pages_needed,), dtype=np.int32)
        row = np.asarray(page_table_cpu[int(req_idx), :pages_needed], dtype=np.int32)
        page_ids_cpu[: row.shape[0]] = row
        page_ids = jnp.asarray(page_ids_cpu, dtype=jnp.int32)
        token_offsets = jnp.arange(int(kv_len), dtype=jnp.int32)
        page_indices = page_ids[token_offsets // page_size]
        offsets = token_offsets % page_size

        key_pages = getattr(view, "key_pages", None)
        value_pages = getattr(view, "value_pages", None)
        if key_pages is None or value_pages is None:
            return None
        keys = key_pages[page_indices, offsets][None, ...]
        values = value_pages[page_indices, offsets][None, ...]
        return keys, values

    def target_kv_payload_for_drafter(
        self,
        *,
        req_id: str,
        seed_position: int,
    ) -> tuple[list[tuple[jax.Array, jax.Array] | None], jax.Array] | None:
        """Build target K/V pairs + static attention mask for an assistant drafter.

        Used by drafters that cross-attend to the target model's cache (e.g. the
        Gemma4 assistant). Maps each drafter layer to a target layer, gathers
        that layer's K/V (either directly from the view or by paging in from
        ragged storage), and builds a causal-style mask covering positions up to
        ``seed_position``.

        Args:
            req_id: Request identifier whose target K/V cache should be read.
            seed_position: Absolute position of the spec window's seed token,
                used to bound the visible context length.

        Returns:
            Tuple ``(pairs, attention_mask)`` where ``pairs`` is one
            ``(keys, values)`` tuple (or ``None``) per drafter layer and
            ``attention_mask`` is a ``[1, 1, 1, kv_len]`` additive float mask, or
            ``None`` when the drafter does not require a target K/V cache, the
            request is unknown, the cache has no views, or no layer yielded K/V.
        """
        if self.drafter is None or not bool(getattr(self.drafter, "requires_target_kv_cache", False)):
            return None

        req_idx = self._runner.sequence_buffer.req_id_to_index.get(req_id)
        if req_idx is None:
            return None
        target_cache = self._runner.executor_manager.kv_pages
        views = getattr(target_cache, "views", None)
        if views is None:
            return None

        resolver = getattr(self.drafter, "resolve_layer_mapping", None)
        if callable(resolver):
            layer_mapping = resolver(target_cache)
        else:
            layer_mapping = list(range(len(views)))
        kv_len = int(self._runner.max_model_len)
        target_context_len = max(1, min(kv_len, int(seed_position) + 1))

        pairs: list[tuple[jax.Array, jax.Array] | None] = []
        for target_layer_idx in layer_mapping:
            if target_layer_idx < 0 or target_layer_idx >= len(views):
                pairs.append(None)
                continue
            view = views[int(target_layer_idx)]
            if view is None:
                pairs.append(None)
                continue
            raw_view = getattr(view, "transformer", view)
            k = getattr(raw_view, "key", None)
            v = getattr(raw_view, "value", None)
            if k is not None and v is not None:
                pairs.append(
                    (
                        k[int(req_idx) : int(req_idx) + 1, :kv_len],
                        v[int(req_idx) : int(req_idx) + 1, :kv_len],
                    )
                )
                continue

            group_idx = self.cache_group_index_for_layer(int(target_layer_idx))
            if group_idx is None:
                pairs.append(None)
                continue
            pairs.append(
                self.gather_paged_kv_for_request(
                    raw_view,
                    group_idx=group_idx,
                    req_idx=int(req_idx),
                    kv_len=kv_len,
                )
            )

        if not any(pair is not None for pair in pairs):
            return None
        mask_positions = jnp.arange(kv_len, dtype=jnp.int32)[None, None, None, :]
        attention_mask = jnp.where(
            mask_positions < jnp.asarray(target_context_len, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(-1.0e10, dtype=jnp.float32),
        )
        return pairs, attention_mask

    def draft_next(
        self,
        *,
        req_id: str,
        seed_token: int,
        seed_position: int,
        seed_hidden: jax.Array,
        req_state: CachedRequestState | None,
        prefix_input_ids: jax.Array | None = None,
        prefix_hidden_states: jax.Array | None = None,
        prefix_position_ids: jax.Array | None = None,
    ) -> list[int]:
        """Generate up to ``num_speculative_tokens`` draft tokens for one request.

        Runs the attached drafter from ``seed_token`` / ``seed_hidden`` at
        ``seed_position``. When the drafter supports prefix drafting and a
        ``prefix_*`` set is provided, the prefix is fed in instead of the
        single seed token to stabilize generation.

        Args:
            req_id: Identifier of the request being drafted for.
            seed_token: Token id at ``seed_position`` (the spec window's seed).
            seed_position: Absolute position of ``seed_token`` in the sequence.
            seed_hidden: Hidden state at ``seed_position``.
            req_state: Cached request state used to look up sampling parameters
                and gate spec eligibility.
            prefix_input_ids: Optional prefix token ids for prefix-drafting.
            prefix_hidden_states: Optional prefix hidden states matching
                ``prefix_input_ids``.
            prefix_position_ids: Optional prefix position ids.

        Returns:
            List of drafted token ids (length may be less than the maximum on
            drafter failure or backoff).
        """
        if self.drafter is None or self.num_speculative_tokens <= 0 or not self.is_spec_request(req_state):
            return []
        backoff = int(self._backoff_by_req.get(str(req_id), 0))
        if backoff > 0:
            self._backoff_by_req[str(req_id)] = backoff - 1
            return []
        use_prefix = (
            prefix_input_ids is not None
            and prefix_hidden_states is not None
            and prefix_position_ids is not None
            and bool(getattr(self.drafter, "supports_prefix_draft", False))
        )
        cur_token = (
            jnp.asarray(prefix_input_ids, dtype=jnp.int32)
            if use_prefix
            else jnp.asarray([[int(seed_token)]], dtype=jnp.int32)
        )
        cur_position = int(seed_position)
        seed_hidden_bsh = jnp.asarray(prefix_hidden_states) if use_prefix else jnp.asarray(seed_hidden)[None, None, :]
        drafted: list[int] = []
        draft_fulls: list[jax.Array] = []
        greedy = self.is_greedy_request(req_state)
        target_kv_payload = self.target_kv_payload_for_drafter(
            req_id=req_id,
            seed_position=seed_position,
        )
        if bool(getattr(self.drafter, "requires_target_kv_cache", False)) and target_kv_payload is None:
            logger.debug("Speculative assistant drafter has no target K/V payload; skipping drafts.")
            return []
        for draft_idx in range(self.num_speculative_tokens):
            position_ids = (
                jnp.asarray(prefix_position_ids, dtype=jnp.int32)
                if use_prefix and draft_idx == 0
                else jnp.asarray([[cur_position]], dtype=jnp.int32)
            )
            try:
                draft_kwargs = {
                    "input_ids": cur_token,
                    "target_hidden_states": seed_hidden_bsh,
                    "position_ids": position_ids,
                    "sample": False,
                    "rng_key": None,
                }
                if not greedy and bool(getattr(self.drafter, "supports_return_full_log_probs", False)):
                    draft_kwargs["return_full_log_probs"] = True
                if target_kv_payload is not None:
                    target_kv_cache, attention_mask = target_kv_payload
                    draft_kwargs.update(
                        {
                            "target_kv_cache": target_kv_cache,
                            "attention_mask": attention_mask,
                        }
                    )
                step = self.drafter.draft(**draft_kwargs)
            except Exception:
                logger.warning("Speculative drafter failed; disabling drafts for this request.", exc_info=True)
                return []
            if greedy:
                token = int(np.asarray(step.token_ids).reshape(-1)[0])
            else:
                if step.full_log_probs is None or req_state is None:
                    return []
                sampled_ids, filtered = self.filter_and_sample_log_probs(step.full_log_probs, req_state)
                draft_fulls.append(filtered[0])
                token = int(np.asarray(sampled_ids).reshape(-1)[0])
            drafted.append(token)
            cur_token = jnp.asarray([[token]], dtype=jnp.int32)
            cur_position += 1
            step_hidden = getattr(step, "hidden_states", None)
            if step_hidden is not None:
                seed_hidden_bsh = jnp.asarray(step_hidden)[:, -1:, :]
        self.num_drafts_generated += len(drafted)
        if not greedy and draft_fulls:
            self.draft_full_log_probs_by_req[str(req_id)] = jnp.stack(draft_fulls, axis=0)
        return drafted

    def record_acceptance(self, req_id: str, accepted: int, num_drafts: int) -> None:
        """Update the adaptive drafter stop-loss after a verify window.

        Hybrid recurrent decode can keep alternate states for speculative
        tokens. Without that cache layout, low-acceptance
        prompts can spend more wallclock on MTP + verification than they save.
        This small gate is intentionally drafter-only: after a full rejection,
        skip a configurable number of future drafter calls and let normal
        eSurge decode advance the sequence.

        Args:
            req_id: Request identifier the verify window belonged to.
            accepted: Number of draft tokens accepted this window.
            num_drafts: Number of draft tokens that were verified.
        """
        if self.reject_backoff_steps <= 0 or num_drafts <= 0:
            return
        key = str(req_id)
        if int(accepted) <= 0:
            self._backoff_by_req[key] = int(self.reject_backoff_steps)
        else:
            self._backoff_by_req.pop(key, None)

    def recurrent_candidate_count(self) -> int:
        """Return per-request speculative recurrent candidate rows, if present."""
        if not getattr(self._runner, "spec_decode_recurrent_candidates", False):
            return 0
        try:
            from easydel.caching import HybridCache, ParallelHybridCacheView, RecurrentCacheView
        except Exception:
            return 0
        cache = self._runner.executor_manager.kv_pages
        if not isinstance(cache, HybridCache):
            return 0
        base_slots = max(1, int(self._runner.max_num_reqs))
        for view in cache.views:
            rec = None
            if isinstance(view, RecurrentCacheView):
                rec = view
            elif isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                rec = view.recurrent
            if rec is None:
                continue
            state = rec.conv_state if rec.conv_state is not None else rec.recurrent_state
            if state is None:
                continue
            n_slots = int(getattr(state, "shape", (0,))[0])
            if n_slots > base_slots:
                return max(0, (n_slots - base_slots) // base_slots)
        return 0

    def cache_has_recurrent_state(self) -> bool:
        """Whether the current eSurge cache contains recurrent/SSM rows."""
        try:
            from easydel.caching import HybridCache, ParallelHybridCacheView, RecurrentCacheView
        except Exception:
            return False
        cache = self._runner.executor_manager.kv_pages
        if not isinstance(cache, HybridCache):
            return False
        for view in cache.views:
            if isinstance(view, RecurrentCacheView):
                return view.conv_state is not None or view.recurrent_state is not None
            if isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                rec = view.recurrent
                return rec.conv_state is not None or rec.recurrent_state is not None
        return False

    def cache_replay_required(self) -> bool:
        """Pure paged-attention caches can ignore stale rejected draft KV rows.

        A future step overwrites rejected draft positions before those positions
        are admitted by ``context_lens``. Recurrent layers are different: their
        state is a single rolling row, so they need either candidate
        rows or an explicit replay from the pre-verify snapshot.
        """
        if not self.cache_has_recurrent_state():
            return False
        return self.recurrent_candidate_count() <= 0

    def commit_recurrent_candidate_state(self, row_pos: int, prefix_len: int) -> bool:
        """Copy a speculative recurrent candidate row into the live recurrent row.

        Hybrid recurrent caches keep extra candidate rows holding the per-step
        prefix states produced during a fused speculative window. After
        verification settles on an accepted ``prefix_len``, this copies the
        matching candidate row (conv and recurrent state) into the request's
        live row via a cached JIT so the next decode step continues from the
        same state as non-speculative generation.

        Args:
            row_pos: Live recurrent row index of the request.
            prefix_len: Number of accepted prefix tokens (1-based selector into
                the candidate rows); must be in ``[1, candidate_count]``.

        Returns:
            ``True`` if the commit was performed; ``False`` when there are no
            candidate rows, ``prefix_len`` is out of range, the cache is not a
            ``HybridCache``, or ``row_pos`` is out of bounds.
        """
        candidate_count = self.recurrent_candidate_count()
        if candidate_count <= 0 or prefix_len <= 0 or prefix_len > candidate_count:
            return False
        try:
            from easydel.caching import HybridCache, ParallelHybridCacheView, RecurrentCacheView
        except Exception:
            return False
        cache = self._runner.executor_manager.kv_pages
        if not isinstance(cache, HybridCache):
            return False

        base_slots = max(1, int(self._runner.max_num_reqs))
        row_pos = int(row_pos)
        if row_pos < 0 or row_pos >= base_slots:
            return False

        commit_fn = self._recurrent_commit_fn
        if commit_fn is None or int(self._recurrent_commit_candidate_count) != int(candidate_count):

            def _copy_candidate_row(arr: jax.Array | None, row: jax.Array, prefix: jax.Array) -> jax.Array | None:
                """Copy one speculative candidate row into its live destination row."""
                if arr is None:
                    return None
                n_slots = int(arr.shape[0])
                candidate_row = base_slots + row * int(candidate_count) + prefix - 1
                safe_dst = jnp.clip(row, 0, n_slots - 1)
                safe_src = jnp.clip(candidate_row, 0, n_slots - 1)
                valid = (row >= 0) & (row < n_slots) & (prefix > 0) & (prefix <= int(candidate_count))
                valid = valid & (candidate_row >= 0) & (candidate_row < n_slots)
                return jax.lax.cond(
                    valid,
                    lambda x: x.at[safe_dst].set(x[safe_src]),
                    lambda x: x,
                    arr,
                )

            @jax.jit
            def _commit(cache_in: HybridCache, row: jax.Array, prefix: jax.Array) -> HybridCache:
                """JIT-compiled commit copying candidate states across cache views."""
                new_views = list(cache_in.views)
                for idx, view in enumerate(new_views):
                    rec = None
                    parallel_view = None
                    if isinstance(view, RecurrentCacheView):
                        rec = view
                    elif isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                        rec = view.recurrent
                        parallel_view = view
                    if rec is None:
                        continue
                    new_conv = _copy_candidate_row(rec.conv_state, row, prefix)
                    new_rec = _copy_candidate_row(rec.recurrent_state, row, prefix)
                    updated_rec = rec.replace(conv_state=new_conv, recurrent_state=new_rec)
                    if parallel_view is not None:
                        new_views[idx] = parallel_view.replace(
                            recurrent=updated_rec,
                            conv_state=new_conv,
                            recurrent_state=new_rec,
                        )
                    else:
                        new_views[idx] = updated_rec
                return HybridCache(views=new_views)

            commit_fn = _commit
            self._recurrent_commit_fn = commit_fn
            self._recurrent_commit_candidate_count = int(candidate_count)

        row_arr = jnp.asarray(row_pos, dtype=jnp.int32)
        prefix_arr = jnp.asarray(int(prefix_len), dtype=jnp.int32)
        self._runner.executor_manager.kv_pages = commit_fn(cache, row_arr, prefix_arr)
        return True

    def queue_or_commit_recurrent_candidate_state(
        self,
        *,
        req_id: str,
        row_pos: int,
        prefix_len: int,
        defer: bool,
    ) -> bool:
        """Make the live recurrent row match the accepted speculative prefix.

        Qwen3.5/Next recurrent layers may execute a verify window through a
        fast multi-token ragged GDN path, while normal greedy decode advances
        the state one token at a time. The speculative candidate rows are the
        single-step prefix states; commit one after every verify window, not
        only on rejection, so the next decode step starts from the same state
        as non-speculative generation.

        Args:
            req_id: Request identifier the commit applies to.
            row_pos: Live recurrent row index of the request.
            prefix_len: Number of accepted prefix tokens (1-based candidate-row
                selector); must be in ``[1, candidate_count]``.
            defer: When True, queue the commit (applied inside the next compiled
                step via ``spec_recurrent_commit_cpu``); when False, commit
                immediately on the host.

        Returns:
            ``True`` if the commit was queued or performed; ``False`` when there
            are no candidate rows or ``prefix_len`` is out of range.
        """
        candidate_count = self.recurrent_candidate_count()
        prefix_len = int(prefix_len)
        if candidate_count <= 0 or prefix_len <= 0 or prefix_len > candidate_count:
            return False
        if defer:
            self._runner._pending_spec_recurrent_commit_by_req[str(req_id)] = prefix_len
            return True
        return self.commit_recurrent_candidate_state(
            row_pos=int(row_pos),
            prefix_len=prefix_len,
        )

    def commit_cache_replay(
        self,
        *,
        pre_step_kv_pages: typing.Any,
        commit_scheduled_list: list[int],
        window_row_indices: np.ndarray,
        recurrent_slot_indices_cpu: np.ndarray | None,
        num_tokens_static_original: int,
        padded_num_reqs: int,
        token_ids_window_cpu: np.ndarray,
        num_computed_tokens_window_cpu: np.ndarray,
        temperature_window_cpu: np.ndarray,
        top_p_window_cpu: np.ndarray,
        top_k_window_cpu: np.ndarray,
        min_p_window_cpu: np.ndarray,
        frequency_penalties_window_cpu: np.ndarray,
        presence_penalties_window_cpu: np.ndarray,
        repetition_penalties_window_cpu: np.ndarray,
        page_table_window_cpu: np.ndarray,
        page_table_window_version: int | None,
        mrope_position_ids_cpu: np.ndarray | None,
        prefill_embeds_cpu: np.ndarray | None,
        prefill_embeds_mask_cpu: np.ndarray | None,
        visual_pos_masks_cpu: np.ndarray | None,
        deepstack_visual_embeds_cpu: list[np.ndarray] | None,
    ) -> None:
        """Replay the executor step over the committed speculative prefix.

        Restores the pre-step KV pages snapshot and re-runs the model step with
        ``commit_scheduled_list`` token counts per row, which results in the
        cache being advanced exactly through the accepted speculative prefix.
        Used by the recurrent speculative path when the cache does not have
        candidate rows and must be replayed.

        Args:
            pre_step_kv_pages: KV pages snapshot taken before the speculative
                forward pass was launched.
            commit_scheduled_list: Per-row token counts to feed back into the
                replay step (length matches the active window).
            window_row_indices: Window-local row indices for the active rows.
            num_tokens_static_original: Token bucket size used on the original
                forward pass; bounds the replay bucket.
            padded_num_reqs: Request bucket size used on the original step.
            token_ids_window_cpu, num_computed_tokens_window_cpu: Token and
                compute-progress CPU buffers shaped for the window.
            temperature_window_cpu, top_p_window_cpu, top_k_window_cpu,
            min_p_window_cpu, frequency_penalties_window_cpu,
            presence_penalties_window_cpu, repetition_penalties_window_cpu:
                Per-row sampling parameter CPU buffers.
            page_table_window_cpu: KV-cache page-table block ids for the
                active window.
            page_table_window_version: Optional page-table cache version, or
                ``None`` for packed/sparse layouts that cannot be cached.
            mrope_position_ids_cpu, prefill_embeds_cpu, prefill_embeds_mask_cpu,
            visual_pos_masks_cpu, deepstack_visual_embeds_cpu: VLM helper
                buffers; passed through to the replay step.
        """
        total_commit = int(sum(commit_scheduled_list))
        self._runner.executor_manager.kv_pages = pre_step_kv_pages
        if total_commit <= 0:
            return
        idx = bisect_left(self._runner.num_tokens_paddings, total_commit)
        if idx >= len(self._runner.num_tokens_paddings):
            idx = len(self._runner.num_tokens_paddings) - 1
        num_tokens_static = int(self._runner.num_tokens_paddings[idx])
        if num_tokens_static > int(num_tokens_static_original):
            num_tokens_static = int(num_tokens_static_original)

        commit_scheduled_full_cpu = self._runner._scheduled_full_cpu.copy()
        commit_scheduled_full_cpu.fill(0)
        commit_scheduled_full_cpu[: len(commit_scheduled_list)] = np.asarray(commit_scheduled_list, dtype=np.int32)

        commit_req_num_tokens = np.asarray(num_computed_tokens_window_cpu).copy()
        for i, n in enumerate(commit_scheduled_list):
            commit_req_num_tokens[i] = int(num_computed_tokens_window_cpu[i]) + int(n) + 1

        self._runner.executor_manager.execute(
            num_tokens=num_tokens_static,
            scheduled_full_cpu=commit_scheduled_full_cpu,
            req_num_tokens_full_cpu=commit_req_num_tokens,
            active_mask_full_cpu=self._runner._active_mask_full_cpu,
            window_row_indices_cpu=window_row_indices,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            input_ids_buf=self._runner.input_ids_buf,
            position_ids_buf=self._runner.position_ids_buf,
            padded_num_reqs=padded_num_reqs,
            token_ids_cpu=token_ids_window_cpu,
            num_computed_tokens_cpu=num_computed_tokens_window_cpu,
            temperature_cpu=temperature_window_cpu,
            top_p_cpu=top_p_window_cpu,
            top_k_cpu=top_k_window_cpu,
            min_p_cpu=min_p_window_cpu,
            frequency_penalties_cpu=frequency_penalties_window_cpu,
            presence_penalties_cpu=presence_penalties_window_cpu,
            repetition_penalties_cpu=repetition_penalties_window_cpu,
            page_table_cpu=page_table_window_cpu,
            page_table_version=page_table_window_version,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            wait_for_outputs=True,
        )

    def run_suffix_sample(
        self,
        row_pos: int,
        start_len: int,
        suffix_len: int = 1,
        *,
        num_computed_tokens_window_cpu,
        window_row_indices_cpu,
        recurrent_slot_indices_cpu,
        padded_num_reqs,
        token_ids_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        frequency_penalties_window_cpu,
        presence_penalties_window_cpu,
        repetition_penalties_window_cpu,
        page_table_window_cpu,
        page_table_window_version,
        mrope_position_ids_cpu,
        prefill_embeds_cpu,
        prefill_embeds_mask_cpu,
        visual_pos_masks_cpu,
        deepstack_visual_embeds_cpu,
    ) -> tuple[int, jax.Array]:
        """Execute a one-row suffix forward pass and return the sampled token + hidden row.

        Used by speculative decoding after a rejected prefix to re-sample
        the corrective token at the rejection boundary without touching
        any other window row.
        """
        suffix_timer_start = time.time()
        suffix_scheduled = self._runner._scheduled_full_cpu.copy()
        suffix_scheduled.fill(0)
        suffix_scheduled[int(row_pos)] = int(suffix_len)

        suffix_active = self._runner._active_mask_full_cpu.copy()
        suffix_active.fill(False)
        suffix_active[int(row_pos)] = True

        suffix_num_computed = np.asarray(num_computed_tokens_window_cpu).copy()
        suffix_num_computed[int(row_pos)] = int(num_computed_tokens_window_cpu[int(row_pos)]) + int(start_len)

        suffix_req_num_tokens = np.asarray(num_computed_tokens_window_cpu).copy()
        suffix_req_num_tokens[int(row_pos)] = (
            int(num_computed_tokens_window_cpu[int(row_pos)]) + int(start_len) + int(suffix_len)
        )

        total_suffix = int(suffix_len)
        suffix_idx = bisect_left(self._runner.num_tokens_paddings, total_suffix)
        if suffix_idx >= len(self._runner.num_tokens_paddings):
            suffix_idx = len(self._runner.num_tokens_paddings) - 1
        suffix_num_tokens_static = int(self._runner.num_tokens_paddings[suffix_idx])

        out_tokens_suffix, _, _, _, hidden_suffix, _, _ = self._runner.executor_manager.execute(
            num_tokens=suffix_num_tokens_static,
            scheduled_full_cpu=suffix_scheduled,
            req_num_tokens_full_cpu=suffix_req_num_tokens,
            active_mask_full_cpu=suffix_active,
            window_row_indices_cpu=window_row_indices_cpu,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            input_ids_buf=self._runner.input_ids_buf,
            position_ids_buf=self._runner.position_ids_buf,
            padded_num_reqs=padded_num_reqs,
            token_ids_cpu=token_ids_window_cpu,
            num_computed_tokens_cpu=suffix_num_computed,
            temperature_cpu=temperature_window_cpu,
            top_p_cpu=top_p_window_cpu,
            top_k_cpu=top_k_window_cpu,
            min_p_cpu=min_p_window_cpu,
            frequency_penalties_cpu=frequency_penalties_window_cpu,
            presence_penalties_cpu=presence_penalties_window_cpu,
            repetition_penalties_cpu=repetition_penalties_window_cpu,
            page_table_cpu=page_table_window_cpu,
            page_table_version=page_table_window_version,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            wait_for_outputs=True,
        )
        suffix_token = int(np.asarray(out_tokens_suffix)[int(row_pos)])
        suffix_hidden = _window_hidden_row(
            hidden_suffix,
            row_pos=int(row_pos),
            token_offset=0,
            token_index=max(0, int(suffix_len) - 1),
            total_window_tokens=int(suffix_len),
            padded_reqs=int(padded_num_reqs),
        )
        self._runner._spec_suffix_time_acc += time.time() - suffix_timer_start
        return suffix_token, suffix_hidden

    def replay_prefix_sample(
        self,
        row_pos: int,
        prefix_len: int,
        *,
        pre_step_kv_pages,
        num_computed_tokens_window_cpu,
        num_tokens_static,
        window_row_indices_cpu,
        recurrent_slot_indices_cpu,
        padded_num_reqs,
        token_ids_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        frequency_penalties_window_cpu,
        presence_penalties_window_cpu,
        repetition_penalties_window_cpu,
        page_table_window_cpu,
        page_table_window_version,
        mrope_position_ids_cpu,
        prefill_embeds_cpu,
        prefill_embeds_mask_cpu,
        visual_pos_masks_cpu,
        deepstack_visual_embeds_cpu,
        window_token_offsets,
        total_scheduled,
        rng_after_verify,
    ) -> tuple[int, jax.Array]:
        """Replay the accepted speculative prefix in one fused forward pass.

        Restores the pre-spec KV snapshot, then re-runs the row with
        ``prefix_len`` scheduled tokens so the cache is advanced exactly
        through the accepted positions before the next decode step.
        """
        replay_timer_start = time.time()
        if pre_step_kv_pages is None:
            raise RuntimeError("Speculative replay requires a pre-step KV snapshot.")
        replay_scheduled = self._runner._scheduled_full_cpu.copy()
        replay_scheduled.fill(0)
        replay_scheduled[int(row_pos)] = int(prefix_len)

        replay_active = self._runner._active_mask_full_cpu.copy()
        replay_active.fill(False)
        replay_active[int(row_pos)] = True

        replay_req_num_tokens = np.asarray(num_computed_tokens_window_cpu).copy()
        replay_req_num_tokens[int(row_pos)] = int(num_computed_tokens_window_cpu[int(row_pos)]) + int(prefix_len)

        self._runner.executor_manager.kv_pages = _copy_kv_tree(pre_step_kv_pages)
        out_tokens_replay, _, _, _, hidden_replay, _, _ = self._runner.executor_manager.execute(
            num_tokens=num_tokens_static,
            scheduled_full_cpu=replay_scheduled,
            req_num_tokens_full_cpu=replay_req_num_tokens,
            active_mask_full_cpu=replay_active,
            window_row_indices_cpu=window_row_indices_cpu,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            input_ids_buf=self._runner.input_ids_buf,
            position_ids_buf=self._runner.position_ids_buf,
            padded_num_reqs=padded_num_reqs,
            token_ids_cpu=token_ids_window_cpu,
            num_computed_tokens_cpu=num_computed_tokens_window_cpu,
            temperature_cpu=temperature_window_cpu,
            top_p_cpu=top_p_window_cpu,
            top_k_cpu=top_k_window_cpu,
            min_p_cpu=min_p_window_cpu,
            frequency_penalties_cpu=frequency_penalties_window_cpu,
            presence_penalties_cpu=presence_penalties_window_cpu,
            repetition_penalties_cpu=repetition_penalties_window_cpu,
            page_table_cpu=page_table_window_cpu,
            page_table_version=page_table_window_version,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            wait_for_outputs=True,
        )
        replay_token = int(np.asarray(out_tokens_replay)[int(row_pos)])
        replay_hidden = _window_hidden_row(
            hidden_replay,
            row_pos=int(row_pos),
            token_offset=int(window_token_offsets[int(row_pos)]),
            token_index=int(prefix_len) - 1,
            total_window_tokens=int(total_scheduled),
            padded_reqs=int(padded_num_reqs),
        )
        if rng_after_verify is not None:
            self._runner.executor_manager.rng_key = rng_after_verify
        self._runner._spec_replay_time_acc += time.time() - replay_timer_start
        return replay_token, replay_hidden

    def replay_prefix_sequential(
        self,
        row_pos: int,
        prefix_len: int,
        *,
        pre_step_kv_pages,
        num_computed_tokens_window_cpu,
        window_row_indices_cpu,
        recurrent_slot_indices_cpu,
        padded_num_reqs,
        token_ids_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        frequency_penalties_window_cpu,
        presence_penalties_window_cpu,
        repetition_penalties_window_cpu,
        page_table_window_cpu,
        page_table_window_version,
        mrope_position_ids_cpu,
        prefill_embeds_cpu,
        prefill_embeds_mask_cpu,
        visual_pos_masks_cpu,
        deepstack_visual_embeds_cpu,
        rng_after_verify,
    ) -> tuple[int, jax.Array]:
        """Replay an accepted speculative prefix one token at a time.

        Used when the model has recurrent layers that cannot be replayed
        in a single fused window (no candidate-row cache available). Runs
        ``prefix_len`` single-token decode steps on top of the pre-spec
        KV snapshot to advance recurrent state exactly to the boundary.
        """
        replay_timer_start = time.time()
        if pre_step_kv_pages is None:
            raise RuntimeError("Speculative sequential replay requires a pre-step KV snapshot.")
        prefix_len = int(prefix_len)
        if prefix_len <= 0:
            raise ValueError("prefix_len must be positive for speculative sequential replay.")

        self._runner.executor_manager.kv_pages = _copy_kv_tree(pre_step_kv_pages)
        replay_scheduled = self._runner._scheduled_full_cpu.copy()
        replay_active = self._runner._active_mask_full_cpu.copy()
        replay_num_computed = np.asarray(num_computed_tokens_window_cpu).copy()
        replay_req_num_tokens = np.asarray(num_computed_tokens_window_cpu).copy()
        one_token_idx = bisect_left(self._runner.num_tokens_paddings, 1)
        if one_token_idx >= len(self._runner.num_tokens_paddings):
            one_token_idx = len(self._runner.num_tokens_paddings) - 1
        one_token_static = int(self._runner.num_tokens_paddings[one_token_idx])

        replay_token = 0
        replay_hidden = None
        for step_idx in range(prefix_len):
            replay_scheduled.fill(0)
            replay_scheduled[int(row_pos)] = 1
            replay_active.fill(False)
            replay_active[int(row_pos)] = True
            replay_num_computed[int(row_pos)] = int(num_computed_tokens_window_cpu[int(row_pos)]) + int(step_idx)
            replay_req_num_tokens[int(row_pos)] = (
                int(num_computed_tokens_window_cpu[int(row_pos)]) + int(step_idx) + 1
            )

            out_tokens_replay, _, _, _, hidden_replay, _, _ = self._runner.executor_manager.execute(
                num_tokens=one_token_static,
                scheduled_full_cpu=replay_scheduled,
                req_num_tokens_full_cpu=replay_req_num_tokens,
                active_mask_full_cpu=replay_active,
                window_row_indices_cpu=window_row_indices_cpu,
                recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
                input_ids_buf=self._runner.input_ids_buf,
                position_ids_buf=self._runner.position_ids_buf,
                padded_num_reqs=padded_num_reqs,
                token_ids_cpu=token_ids_window_cpu,
                num_computed_tokens_cpu=replay_num_computed,
                temperature_cpu=temperature_window_cpu,
                top_p_cpu=top_p_window_cpu,
                top_k_cpu=top_k_window_cpu,
                min_p_cpu=min_p_window_cpu,
                frequency_penalties_cpu=frequency_penalties_window_cpu,
                presence_penalties_cpu=presence_penalties_window_cpu,
                repetition_penalties_cpu=repetition_penalties_window_cpu,
                page_table_cpu=page_table_window_cpu,
                page_table_version=page_table_window_version,
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
                wait_for_outputs=True,
            )
            replay_token = int(np.asarray(out_tokens_replay)[int(row_pos)])
            replay_hidden = _window_hidden_row(
                hidden_replay,
                row_pos=int(row_pos),
                token_offset=0,
                token_index=0,
                total_window_tokens=1,
                padded_reqs=int(padded_num_reqs),
            )

        if rng_after_verify is not None:
            self._runner.executor_manager.rng_key = rng_after_verify
        self._runner._spec_replay_time_acc += time.time() - replay_timer_start
        if replay_hidden is None:
            raise RuntimeError("Speculative sequential replay did not produce hidden state.")
        return replay_token, replay_hidden
