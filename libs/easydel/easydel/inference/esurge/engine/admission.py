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

"""Request admission for the eSurge engine.

:class:`RequestAdmission` owns everything between a caller-supplied prompt
and scheduler enqueue: request-id generation, per-request sampling-params
preparation (callback, extra stops, generation-config EOS ids), context
budget enforcement with prompt truncation, request-registry bookkeeping,
per-request parser construction, and ``n>1`` sample fan-out.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import time
import typing
import uuid
from collections.abc import Sequence
from typing import Any

import jax

from easydel.inference.parsing import DelegatingParser
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.tools.tool_calling_mixin import build_tool_parser_or_none

from ..logger import logger
from ..metrics import get_metrics_collector
from ..request import EngineRequest
from ..utils import truncate_tokens
from .chat_templating import normalize_stop_sequences
from .registry import RequestRecord, RequestRegistry


def compute_admission_key(
    prompt_token_ids: Sequence[int],
    sampling_params: Any,
    n: int,
    *,
    salt: str = "",
) -> str:
    """Content hash identifying a logical admission across ranks.

    The unified request plane keys cross-rank coalescing on
    ``(origin_counter, content_hash)``: two ranks admitting the same tokens
    with the same sampling parameters in the same admission slot are the
    same logical request. The hash canonicalizes the sampling params via
    their public attribute dict (attribute insertion order is identical
    across ranks for equal construction paths, and ``repr`` of equal values
    is deterministic), so equal-valued params hash equally on every rank.

    Args:
        prompt_token_ids: Post-truncation prompt token ids.
        sampling_params: The request's ``SamplingParams`` (or ``None``).
        n: Parallel-sampling fan-out.
        salt: Extra discriminator; callers pass the explicit request id when
            one was supplied so only auto-id (trainer) traffic coalesces.

    Returns:
        Hex SHA-256 digest of the canonicalized admission content.
    """
    parts = [repr(tuple(int(tok) for tok in prompt_token_ids)), repr(int(n or 1)), str(salt)]
    if sampling_params is not None:
        state = {key: value for key, value in vars(sampling_params).items() if not key.startswith("_")}
        parts.append(repr(sorted((str(key), repr(value)) for key, value in state.items())))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def clone_sampling_params(sampling_params: SamplingParams) -> SamplingParams:
    """Create a deep copy of sampling parameters.

    Args:
        sampling_params: Parameters to clone.

    Returns:
        Deep copy of the parameters, or original if cloning fails.
    """
    try:
        return copy.deepcopy(sampling_params)
    except Exception:
        logger.exception("Failed to clone sampling params; using original instance")
        return sampling_params


def _set_requested_new(sp, n: int):
    """Write ``n`` to whichever max-tokens field the SamplingParams flavour exposes.

    Different OpenAI-compatible client libraries spell the cap as
    ``max_tokens`` (Chat Completions) or ``max_new_tokens`` (HF-style). The
    engine accepts either, so when the context-length manager needs to clamp
    the requested generation budget it writes the new value to *both* fields
    (when present) so subsequent reads agree.

    Args:
        sp: SamplingParams-like instance (or any object exposing one or both
            of the two attribute names).
        n: New cap, written as ``int(n)``.
    """
    if hasattr(sp, "max_tokens"):
        sp.max_tokens = int(n)
    if hasattr(sp, "max_new_tokens"):
        sp.max_new_tokens = int(n)


class RequestAdmission:
    """Admission component turning prompts into scheduler-ready requests.

    All engine coupling is injected explicitly; there is no back-reference
    to the engine except the opaque ``callback_engine`` object surfaced in
    the sampling-params callback metadata (part of that callback's public
    contract).

    Args:
        registry: Shared :class:`RequestRegistry` holding request records,
            outputs, events, and their locks.
        scheduler_submit: Engine callable that acquires the scheduler lock
            and enqueues a batch of :class:`EngineRequest` objects.
        tokenizer_client: Worker-pipeline tokenizer client
            (``tokenize(request_id, prompt) -> list[int]``).
        tokenizer: In-process tokenizer used for segment tokenization,
            truncated-prompt decoding, and EOS fallbacks.
        context_config: The engine's :class:`eSurgeContextConfig`.
        reserve_tokens: Tokens reserved for generation headroom.
        max_model_len: Maximum sequence length (prompt + generation).
        tool_parser_class: Tool-parser class instantiated per request, or
            ``None``.
        reasoning_parser_class: Reasoning-parser class instantiated per
            request, or ``None``.
        sampling_params_callback: Zero-arg getter returning the engine's
            current sampling-params callback (or ``None``).
        generation_config_dict: Model ``generation_config`` payload used to
            merge extra EOS ids into per-request stop tokens.
        primary_eos_token_id: The model's primary EOS id, if known.
        eos_token_ids: Engine-normalized EOS id list.
        extra_stops: Engine-level extra stop strings merged into every
            request.
        ignore_stop_strings_in_reasoning: Engine default for the
            per-request ``ignore_stop_strings_in_reasoning`` flag.
        on_activity: Callback marking engine activity for the idle watchdog.
        info: Engine-level info logger callable.
        callback_engine: Opaque object surfaced as ``metadata["engine"]``
            to the sampling-params callback.
    """

    def __init__(
        self,
        *,
        registry: RequestRegistry,
        scheduler_submit: typing.Callable[[list[EngineRequest]], None],
        tokenizer_client,
        tokenizer,
        context_config,
        reserve_tokens: int,
        max_model_len: int,
        tool_parser_class,
        reasoning_parser_class,
        sampling_params_callback: typing.Callable[[], typing.Callable | None],
        generation_config_dict: dict | None,
        primary_eos_token_id: int | None,
        eos_token_ids: Sequence[int],
        extra_stops: Sequence[str],
        ignore_stop_strings_in_reasoning: bool,
        on_activity: typing.Callable[[], None],
        info: typing.Callable[..., None],
        callback_engine: Any = None,
    ) -> None:
        self._registry = registry
        self._scheduler_submit = scheduler_submit
        self._tokenizer_client = tokenizer_client
        self.tokenizer = tokenizer
        self.context_config = context_config
        self.reserve_tokens = int(reserve_tokens)
        self._max_model_len = int(max_model_len)
        self._tool_parser_class = tool_parser_class
        self._reasoning_parser_class = reasoning_parser_class
        self._sampling_params_callback = sampling_params_callback
        self._generation_config_dict = generation_config_dict
        self._primary_eos_token_id = primary_eos_token_id
        self._eos_ids = list(eos_token_ids)
        self.extra_stops = list(extra_stops)
        self._ignore_stop_strings_in_reasoning = bool(ignore_stop_strings_in_reasoning)
        self._on_activity = on_activity
        self._info = info
        self._callback_engine = callback_engine

        # Registry aliases keep the moved method bodies verbatim.
        self._request_lock = registry.request_lock
        self._output_lock = registry.output_lock
        self._active_requests = registry.records
        self._request_events = registry.events
        self._request_outputs = registry.outputs

        self._request_counter = 0
        self._counter_lock = threading.Lock()

    def generate_request_id(self) -> str:
        """Allocate a fresh request id under the counter lock.

        Two formats are produced depending on the JAX process count:

        * Multi-host (``jax.process_count() > 1``) — the id is purely
          counter-based (``req-{counter:010d}``) so every host generates
          the *same* sequence of ids when handed the same input order.
          This is essential because the SPMD/MPMD runner and distributed
          control plane key on request id when verifying step digests
          across ranks.
        * Single-host — combines a UUID4 hex with the counter
          (``req-{uuid}-{counter}``) so ids are unique even across engine
          restarts while keeping a useful arrival-order suffix for logs.

        The internal counter wraps at 2**32 to bound id length.

        Returns:
            Newly-allocated request id string.
        """
        with self._counter_lock:
            self._request_counter = (self._request_counter + 1) % (1 << 32)
            if jax.process_count() > 1:
                return f"req-{self._request_counter:010d}"
            return f"req-{uuid.uuid4().hex}-{self._request_counter}"

    def next_arrival_stamp(self) -> float:
        """Advance the admission counter and return it as an arrival stamp.

        The request plane's owner side stamps remotely-admitted requests
        with this so they interleave FCFS with owner-local admissions:
        both share one monotonic counter scale (local multi-host admissions
        stamp ``arrival_time`` from the same counter).

        Returns:
            The advanced counter value as a float arrival time.
        """
        with self._counter_lock:
            self._request_counter = (self._request_counter + 1) % (1 << 32)
            return float(self._request_counter)

    def _apply_extra_stops_to_sampling_params(self, sampling_params: SamplingParams) -> SamplingParams:
        """Splice engine-level ``extra_stops`` into a request's ``stop`` list.

        Merges the injected ``extra_stops`` with whatever stop sequences
        the request already carries. The normalized union (deduplicated,
        order-preserving via :func:`normalize_stop_sequences`) is written
        back to ``sampling_params.stop`` *in place* and the same params
        object is returned for chaining. No-op when the engine has no
        extra stops.

        Args:
            sampling_params: Per-request sampling parameters; mutated when
                a merge is required.

        Returns:
            ``sampling_params`` (same instance) for chaining.
        """

        extra_stops = normalize_stop_sequences(self.extra_stops)
        if not extra_stops:
            return sampling_params

        merged = normalize_stop_sequences(getattr(sampling_params, "stop", None))
        seen = set(merged)
        for stop in extra_stops:
            if stop in seen:
                continue
            seen.add(stop)
            merged.append(stop)
        sampling_params.stop = merged
        return sampling_params

    def _apply_generation_config_to_sampling_params(self, sampling_params: SamplingParams) -> SamplingParams:
        """Augment request stop-token policy with the model's generation-config EOS ids.

        Some HF models distribute multiple EOS tokens through their
        ``generation_config.json`` (Llama 3 ``<|eot_id|>``, Qwen
        ``<|im_end|>``, …). The tokenizer alone often only knows about
        the *primary* EOS, so requests that don't explicitly enumerate
        ``stop_token_ids`` would otherwise miss those alternates. This
        helper folds the generation-config EOS ids into
        ``sampling_params.stop_token_ids`` (deduplicated, in-place) and
        returns the same object for chaining. No-op when the model has
        no extra EOS ids.

        Args:
            sampling_params: Per-request sampling parameters; mutated when
                additional EOS ids exist.

        Returns:
            ``sampling_params`` (same instance) for chaining.
        """

        generation_config = self._generation_config_dict
        primary_eos_token_id = self._primary_eos_token_id

        if not generation_config and primary_eos_token_id is None:
            return sampling_params

        try:
            sampling_params.update_with_generation_config(
                generation_config or {},
                model_eos_token_id=primary_eos_token_id,
            )
        except Exception:
            logger.debug("Failed to merge generation_config EOS token IDs into sampling params", exc_info=True)
        return sampling_params

    def _clone_sampling_params(self, sampling_params: SamplingParams) -> SamplingParams:
        """Create a deep copy of sampling parameters.

        Args:
            sampling_params: Parameters to clone.

        Returns:
            Deep copy of the parameters, or original if cloning fails.
        """
        return clone_sampling_params(sampling_params)

    def prepare_sampling_params_for_request(
        self,
        template: SamplingParams,
        *,
        request_id: str,
        prompt: str,
    ) -> SamplingParams:
        """Prepare sampling parameters for a specific request.

        Clones the template and applies the sampling_params_callback if configured.

        Args:
            template: Base sampling parameters to clone.
            request_id: Request ID for callback context.
            prompt: Prompt text for callback context.

        Returns:
            Prepared SamplingParams instance for this request.
        """
        params = self._clone_sampling_params(template)
        callback = self._sampling_params_callback()

        def _finalize(prepared: SamplingParams) -> SamplingParams:
            """Apply engine-level stop / generation overrides to a per-request params object.

            Inherits ``ignore_stop_strings_in_reasoning`` from the engine when
            the per-request value is unset, then layers on extra stop strings
            and generation-config defaults (top-p, temperature, etc.) via the
            corresponding ``_apply_*`` helpers.
            """
            if getattr(prepared, "ignore_stop_strings_in_reasoning", None) is None:
                prepared.ignore_stop_strings_in_reasoning = self._ignore_stop_strings_in_reasoning
            prepared = self._apply_extra_stops_to_sampling_params(prepared)
            prepared = self._apply_generation_config_to_sampling_params(prepared)
            return prepared

        if callback is None:
            return _finalize(params)

        metadata = {"request_id": request_id, "prompt": prompt, "engine": self._callback_engine}
        try:
            result = callback(params, metadata)
            if result is None:
                return _finalize(params)
            return _finalize(result)
        except Exception:
            logger.exception("Sampling params callback failed; falling back to unmodified parameters")
            return _finalize(params)

    def _prepare_prompt_segments(self, prompt: typing.Any) -> list[str]:
        """Convert a prompt to a list of string segments.

        Args:
            prompt: Input prompt, can be a string or list of strings/objects.

        Returns:
            List of string segments.
        """
        if isinstance(prompt, list):
            return [segment if isinstance(segment, str) else str(segment) for segment in prompt]
        return [prompt if isinstance(prompt, str) else str(prompt)]

    def _tokenize_prompt_segments(self, prompt: typing.Any) -> list[list[int]]:
        """Tokenize prompt segments individually.

        Args:
            prompt: Input prompt, can be a string or list of strings.

        Returns:
            List of token ID lists, one per segment.
        """
        segments = self._prepare_prompt_segments(prompt)
        token_segments: list[list[int]] = []
        for segment in segments:
            try:
                encoded = self.tokenizer(
                    segment,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                ids = encoded.get("input_ids", [])
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
            except Exception:
                ids = []
            token_segments.append([int(tok) for tok in ids])
        return token_segments

    def _configure_reasoning_parser_for_prompt(
        self,
        reasoning_parser: Any | None,
        prompt_text: str,
        prompt_token_ids: Sequence[int],
    ) -> None:
        """Hand the prompt context to a fresh reasoning parser, swallowing failures.

        Each request gets its own reasoning-parser instance (so per-request
        parser state — open reasoning blocks, partial deltas, etc. — does
        not leak across requests). Some parsers want the prompt up front so
        they can pre-anchor their state machine; this helper invokes their
        ``configure_prompt_context`` hook and demotes any exception to a
        DEBUG log so a misbehaving parser cannot block request enqueue.

        Args:
            reasoning_parser: Per-request parser instance, or ``None`` when
                the engine has no reasoning parser configured.
            prompt_text: Detokenized prompt string passed verbatim to the
                hook.
            prompt_token_ids: Token-id sequence corresponding to
                ``prompt_text``; some parsers prefer ids over text for
                position alignment.
        """
        if reasoning_parser is None:
            return
        try:
            reasoning_parser.configure_prompt_context(prompt_text=prompt_text, prompt_token_ids=prompt_token_ids)
        except Exception:
            logger.debug("Failed to configure reasoning parser prompt context", exc_info=True)

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None = None,
        tool_parser_request: Any | None = None,
        defer_scheduler_enqueue: bool = False,
        # Vision-language model data (optional)
        pixel_values: Any | None = None,
        image_grid_thw: Any | None = None,
        pixel_values_videos: Any | None = None,
        video_grid_thw: Any | None = None,
        mm_features: list | None = None,
    ) -> list[EngineRequest] | None:
        """Add a new request to the scheduler queue with intelligent context management.

        Tokenizes the prompt, applies context length management policies,
        creates request tracking structures, and adds the request to the
        scheduler for processing. Handles prompt truncation and token
        reservation to ensure generation fits within model constraints.

        Args:
            request_id: Unique identifier for the request.
            prompt: Text prompt to generate from. May be truncated based on
                context management settings.
            sampling_params: Generation parameters including max_tokens/max_new_tokens.
            prompt_token_ids: Pre-tokenized prompt token ids; when ``None``
                the engine tokenizes ``prompt`` on the fly.
            tool_parser_request: Optional :class:`ChatCompletionRequest`
                handed to the request's tool parser so it can disambiguate
                forced-function calls.
            defer_scheduler_enqueue: When ``True``, return the constructed
                :class:`EngineRequest` objects to the caller instead of
                enqueueing them on the scheduler. Used by :meth:`eSurge.generate`
                to batch-add all sample children under a single
                scheduler-lock window.
            pixel_values: Optional vision tensor (image pixels) attached
                to the request for vision-language models.
            image_grid_thw: Vision grid dimensions (temporal, height, width)
                aligned with ``pixel_values``.
            pixel_values_videos: Optional vision tensor for video inputs.
            video_grid_thw: Grid dimensions aligned with ``pixel_values_videos``.
            mm_features: Optional list of :class:`MultiModalFeature` items
                used by the prefix cache to share multimodal KV blocks
                across ``n>1`` samples.

        Returns:
            List of :class:`EngineRequest` to enqueue when
            ``defer_scheduler_enqueue=True``; otherwise ``None`` (the
            requests are enqueued internally before returning).

        Raises:
            ValueError: If the prompt exceeds the context budget under
                ``strict_context=True``, or if the requested
                ``max_new_tokens`` cannot be satisfied even after
                truncation.

        Context Management:
            The method implements a context management strategy:
                1. Calculates available token budget (max_model_len - reserve_tokens)
                2. If prompt exceeds budget:
                    - Truncates based on truncate_mode (left/right/middle)
                    - Or raises error if strict_context=True
                3. Adjusts max_new_tokens to fit within remaining context
                4. Prioritizes based on prefer_preserve_prompt setting

        Truncation Strategies:
            - "left": Removes tokens from beginning (keeps recent context)
            - "right": Removes tokens from end (keeps initial context)
            - "middle": Removes tokens from middle (keeps both ends)

        Note:
            This method ensures that prompt_len + max_new_tokens + reserve_tokens
            never exceeds max_model_len, preventing OOM errors during generation.
        """
        from ..esurge_engine import CompletionOutput, RequestOutput

        self._on_activity()

        max_model_len = int(self._max_model_len)

        def _get_requested_new(sp):
            """Read the request's max-new-tokens budget from a :class:`SamplingParams`-like object.

            Honours both legacy (``max_tokens``) and current
            (``max_new_tokens``) attribute names, returning the int budget or
            ``None`` when neither is set so the caller can engage the
            auto-infer path that derives a budget from
            ``max_model_len`` minus the prompt length.
            """
            if hasattr(sp, "max_tokens") and sp.max_tokens is not None:
                return int(sp.max_tokens)
            if hasattr(sp, "max_new_tokens") and sp.max_new_tokens is not None:
                return int(sp.max_new_tokens)
            return None

        requested_new_raw = _get_requested_new(sampling_params)
        auto_infer_new_tokens = requested_new_raw is None
        requested_new = int(requested_new_raw) if requested_new_raw is not None else 0
        original_requested_new = requested_new if not auto_infer_new_tokens else -1

        token_ids_source = (
            prompt_token_ids if prompt_token_ids is not None else self._tokenizer_client.tokenize(request_id, prompt)
        )
        token_ids = list(token_ids_source)
        prompt_len = len(token_ids)

        max_prompt_budget = max(0, max_model_len - self.reserve_tokens)
        truncated = False
        tokens_dropped = 0

        if prompt_len > max_prompt_budget:
            if not self.context_config.auto_truncate_prompt and self.context_config.strict_context:
                raise ValueError(
                    f"Prompt too long: length={prompt_len} > budget={max_prompt_budget} "
                    f"(model_max={max_model_len}, reserve={self.reserve_tokens})."
                )
            new_tokens, dropped = truncate_tokens(token_ids, max_prompt_budget, self.context_config.truncate_mode)
            token_ids = new_tokens
            prompt_len = len(token_ids)
            truncated = dropped > 0
            tokens_dropped += dropped
            logger.warn(
                f"Truncated prompt by {dropped} tokens to fit model budget "
                f"(mode={self.context_config.truncate_mode}, new_len={prompt_len}, budget={max_prompt_budget})."
            )

        if auto_infer_new_tokens:
            requested_new = max(0, max_model_len - prompt_len - self.reserve_tokens)
            _set_requested_new(sampling_params, requested_new)
            logger.debug(
                "Auto-inferred max_tokens=%s for request %s (prompt_len=%s, reserve=%s, model_max=%s).",
                requested_new,
                request_id,
                prompt_len,
                self.reserve_tokens,
                max_model_len,
            )

        # Keep prompt *and* reserve safety margin when capping new tokens.
        allowed_new_if_keep_prompt = max(0, max_model_len - prompt_len - self.reserve_tokens)

        if requested_new > allowed_new_if_keep_prompt:
            do_cap_first = self.context_config.prefer_preserve_prompt or not self.context_config.auto_truncate_prompt

            if do_cap_first:
                if self.context_config.auto_cap_new_tokens:
                    logger.warn(
                        f"Capping max_new_tokens from {requested_new} to {allowed_new_if_keep_prompt} "
                        f"to preserve prompt (prompt_len={prompt_len}, reserve={self.reserve_tokens}, "
                        f"model_max={max_model_len})."
                    )
                    requested_new = allowed_new_if_keep_prompt
                    _set_requested_new(sampling_params, requested_new)
                else:
                    if self.context_config.strict_context:
                        raise ValueError(
                            f"Requested max_new_tokens={requested_new} exceeds allowed={allowed_new_if_keep_prompt} "
                            f"for prompt_len={prompt_len}."
                        )
                    logger.warn(
                        f"auto_cap_new_tokens disabled but strict_context=False; "
                        f"capping new tokens to {allowed_new_if_keep_prompt}."
                    )
                    requested_new = allowed_new_if_keep_prompt
                    _set_requested_new(sampling_params, requested_new)
            else:
                target_prompt_budget = max(0, max_model_len - requested_new - self.reserve_tokens)
                if target_prompt_budget == 0 and requested_new > 0:
                    if self.context_config.auto_cap_new_tokens:
                        logger.warn(
                            f"Requested max_new_tokens={requested_new} leaves no room for prompt; "
                            f"capping to {allowed_new_if_keep_prompt} to preserve prompt."
                        )
                        requested_new = allowed_new_if_keep_prompt
                        _set_requested_new(sampling_params, requested_new)
                    else:
                        if self.context_config.strict_context:
                            raise ValueError("Requested output too large; would require dropping entire prompt.")
                        requested_new = allowed_new_if_keep_prompt
                        _set_requested_new(sampling_params, requested_new)
                else:
                    if prompt_len > target_prompt_budget:
                        new_tokens, dropped = truncate_tokens(
                            token_ids, target_prompt_budget, self.context_config.truncate_mode
                        )
                        token_ids = new_tokens
                        prompt_len = len(token_ids)
                        truncated = truncated or dropped > 0
                        tokens_dropped += dropped
                        self._info(
                            f"Truncated prompt by {dropped} tokens (mode={self.context_config.truncate_mode}) "
                            f"to honor requested max_new_tokens={requested_new}. "
                            f"New prompt_len={prompt_len}, target_prompt_budget={target_prompt_budget}."
                        )

        allowed_new_final = max(0, max_model_len - prompt_len - self.reserve_tokens)
        if requested_new > allowed_new_final:
            if self.context_config.strict_context and not self.context_config.auto_cap_new_tokens:
                raise ValueError(
                    f"After adjustments, requested_new={requested_new} still exceeds allowed={allowed_new_final}."
                )
            logger.warn(
                f"Final cap: max_new_tokens {requested_new} -> {allowed_new_final} "
                f"(prompt_len={prompt_len}, reserve={self.reserve_tokens}, model_max={max_model_len})."
            )
            requested_new = allowed_new_final
            _set_requested_new(sampling_params, requested_new)

        prompt_for_engine = prompt
        if truncated and self.context_config.decode_truncated_prompt:
            try:
                prompt_for_engine = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception:
                prompt_for_engine = prompt
                logger.warn("Failed to decode truncated prompt; keeping original prompt text.")

        start_ts = time.perf_counter()
        ev = threading.Event()

        with self._request_lock:
            self._request_events[request_id] = ev
            self._active_requests[request_id] = RequestRecord(
                prompt=prompt_for_engine,
                prompt_token_ids=token_ids,
                sampling_params=sampling_params,
                start_time=start_ts,
                last_decode_time=start_ts,
                truncated=truncated,
                tokens_dropped=tokens_dropped,
                requested_new_tokens_original=original_requested_new,
                requested_new_tokens_final=requested_new,
                reserve_tokens=self.reserve_tokens,
                max_model_len=max_model_len,
            )
            _rp = self._reasoning_parser_class(self.tokenizer) if self._reasoning_parser_class else None
            self._configure_reasoning_parser_for_prompt(
                reasoning_parser=_rp,
                prompt_text=prompt_for_engine,
                prompt_token_ids=token_ids,
            )
            _tp = build_tool_parser_or_none(self._tool_parser_class, self.tokenizer, context=f"request {request_id}")
            self._active_requests[request_id].delegating_parser = DelegatingParser(
                reasoning_parser=_rp,
                tool_parser=_tp,
                tool_request=tool_parser_request,
            )

        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.start_request(request_id, len(token_ids))

        prompt_token_segments = self._tokenize_prompt_segments(prompt_for_engine)

        # Handle n > 1 sampling: create multiple EngineRequest objects
        n_samples = getattr(sampling_params, "n", 1) or 1

        # Create n CompletionOutput objects for the RequestOutput
        completion_outputs = [CompletionOutput(index=i, text="", token_ids=[]) for i in range(n_samples)]

        with self._output_lock:
            self._request_outputs[request_id] = RequestOutput(
                request_id=request_id,
                prompt=prompt_for_engine,
                prompt_token_ids=prompt_token_segments,
                outputs=completion_outputs,
                finished=False,
                accumulated_text="",
                delta_text="",
                raw_accumulated_text="",
                raw_delta_text="",
                tokens_per_second=0.0,
                num_generated_tokens=0,
                time_spent_generating=0.0,
                first_token_time=None,
                processing_time=0.0,
                update_seq=0,
                delta_seq=0,
            )

        # Prepare EOS token IDs from engine-normalized EOS set
        eos_token_ids = [int(tid) for tid in (self._eos_ids or []) if tid is not None]

        if not eos_token_ids:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            fallback_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            eos_token_ids = [int(tid) for tid in fallback_ids if tid is not None]

        # Use the first EOS token as the primary one for backwards compatibility
        primary_eos_token_id = eos_token_ids[0] if eos_token_ids else self._primary_eos_token_id

        # Add all EOS tokens to sampling_params.stop_token_ids if not already present
        if eos_token_ids:
            current_stop_ids = set(sampling_params.stop_token_ids or [])
            all_stop_ids = getattr(sampling_params, "_all_stop_token_ids", None)
            if all_stop_ids is None:
                all_stop_ids = set(current_stop_ids)
                sampling_params._all_stop_token_ids = all_stop_ids
            for eos_id in eos_token_ids:
                if eos_id not in all_stop_ids:
                    sampling_params.stop_token_ids.append(eos_id)
                    all_stop_ids.add(eos_id)

        # Create n EngineRequest objects for parallel sampling
        mm_features_cache_key_only = None
        if mm_features and n_samples > 1:
            mm_features_cache_key_only = []
            for feat in mm_features:
                try:
                    mm_features_cache_key_only.append(
                        type(feat)(
                            mm_hash=getattr(feat, "mm_hash", ""),
                            modality=getattr(feat, "modality", "image"),
                            pixel_values=None,
                            grid_thw=None,
                        )
                    )
                except Exception:
                    # Worst-case fallback: keep the original feature object.
                    # This preserves correctness at the cost of extra memory.
                    mm_features_cache_key_only.append(feat)

        scheduler_requests: list[EngineRequest] = []
        for sample_idx in range(n_samples):
            if n_samples == 1:
                # For n=1, use the original request_id
                child_request_id = request_id
                parent_id = None
            else:
                # For n>1, create child request IDs
                child_request_id = f"{request_id}-{sample_idx}"
                parent_id = request_id

                # Create tracking entries for child requests
                # IMPORTANT: Create a fresh dict for each sample to avoid sharing mutable objects
                with self._request_lock:
                    self._request_events[child_request_id] = self._request_events[request_id]
                    self._active_requests[child_request_id] = RequestRecord(
                        prompt=prompt_for_engine,
                        prompt_token_ids=token_ids,
                        sampling_params=sampling_params,
                        start_time=start_ts,
                        last_decode_time=start_ts,
                        truncated=truncated,
                        tokens_dropped=tokens_dropped,
                        requested_new_tokens_original=original_requested_new,
                        requested_new_tokens_final=requested_new,
                        reserve_tokens=self.reserve_tokens,
                        max_model_len=max_model_len,
                        sample_index=sample_idx,
                        parent_request_id=request_id,
                    )
                    _rp2 = self._reasoning_parser_class(self.tokenizer) if self._reasoning_parser_class else None
                    self._configure_reasoning_parser_for_prompt(
                        reasoning_parser=_rp2,
                        prompt_text=prompt_for_engine,
                        prompt_token_ids=token_ids,
                    )
                    _tp2 = build_tool_parser_or_none(
                        self._tool_parser_class, self.tokenizer, context=f"request {child_request_id}"
                    )
                    self._active_requests[child_request_id].delegating_parser = DelegatingParser(
                        reasoning_parser=_rp2,
                        tool_parser=_tp2,
                        tool_request=tool_parser_request,
                    )

            # In multi-host mode, use a deterministic arrival_time so all
            # hosts agree on preemption/priority ordering.
            _arrival = float(self._request_counter) if jax.process_count() > 1 else None
            scheduler_requests.append(
                EngineRequest(
                    request_id=child_request_id,
                    prompt_token_ids=token_ids,
                    sampling_params=sampling_params,
                    eos_token_id=primary_eos_token_id,
                    parent_request_id=parent_id,
                    sample_index=sample_idx,
                    arrival_time=_arrival,
                    # Vision-language model data (only for first sample to save memory)
                    pixel_values=pixel_values if sample_idx == 0 else None,
                    image_grid_thw=image_grid_thw if sample_idx == 0 else None,
                    pixel_values_videos=pixel_values_videos if sample_idx == 0 else None,
                    video_grid_thw=video_grid_thw if sample_idx == 0 else None,
                    # Keep multimodal cache keys for all samples so n>1 can share
                    # KV-prefix pages via prefix caching without duplicating pixel buffers.
                    mm_features=mm_features if sample_idx == 0 else mm_features_cache_key_only,
                )
            )

        if defer_scheduler_enqueue:
            return scheduler_requests

        self._scheduler_submit(scheduler_requests)

        self._info(
            f"Queued request {request_id}: prompt_len={prompt_len}, "
            f"max_tokens={requested_new}, n={n_samples}, reserve={self.reserve_tokens}, "
            f"model_max={max_model_len}, dropped={tokens_dropped}"
        )
        return None
