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

from __future__ import annotations

import threading
import time
import typing
import uuid
from typing import Any

from easydel.inference.sampling_params import SamplingParams

from ..logger import logger
from ..metrics import get_metrics_collector, log_metrics_summary
from ..request import EngineRequest, EngineRequestStatus
from ..utils import truncate_tokens


def _set_requested_new(sp, n: int):
    """Set the max_tokens or max_new_tokens attribute on a SamplingParams object."""
    if hasattr(sp, "max_tokens"):
        sp.max_tokens = int(n)
    if hasattr(sp, "max_new_tokens"):
        sp.max_new_tokens = int(n)


class EngineRequestsMixin:
    def _configure_reasoning_parser_for_prompt(
        self,
        reasoning_parser: Any | None,
        prompt_text: str,
        prompt_token_ids: typing.Sequence[int],
    ) -> None:
        """Configure prompt context on a request-scoped reasoning parser."""
        if reasoning_parser is None:
            return
        try:
            reasoning_parser.configure_prompt_context(prompt_text=prompt_text, prompt_token_ids=prompt_token_ids)
        except Exception:
            logger.debug("Failed to configure reasoning parser prompt context", exc_info=True)

    def _add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None = None,
        # Vision-language model data (optional)
        pixel_values: Any | None = None,
        image_grid_thw: Any | None = None,
        pixel_values_videos: Any | None = None,
        video_grid_thw: Any | None = None,
        mm_features: list | None = None,
    ) -> None:
        """Add a new request to the scheduler queue with intelligent context management.

        Internal method that tokenizes the prompt, applies context length management
        policies, creates request tracking structures, and adds the request to the
        scheduler for processing. Handles prompt truncation and token reservation
        to ensure generation fits within model constraints.

        Args:
            request_id: Unique identifier for the request.
            prompt: Text prompt to generate from. May be truncated based on
                context management settings.
            sampling_params: Generation parameters including max_tokens/max_new_tokens.

        Context Management:
            The method implements a sophisticated context management strategy:
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

        self._touch_activity()

        # ---- Config knobs ----
        max_model_len = int(self.runner.max_model_len)

        def _get_requested_new(sp):
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
            prompt_token_ids if prompt_token_ids is not None else self._tokenize_prompt(request_id, prompt)
        )
        token_ids = list(token_ids_source)
        prompt_len = len(token_ids)

        max_prompt_budget = max(0, max_model_len - self.reserve_tokens)
        truncated = False
        tokens_dropped = 0

        if prompt_len > max_prompt_budget:
            if not self.auto_truncate_prompt and self.strict_context:
                raise ValueError(
                    f"Prompt too long: length={prompt_len} > budget={max_prompt_budget} "
                    f"(model_max={max_model_len}, reserve={self.reserve_tokens})."
                )
            new_tokens, dropped = truncate_tokens(token_ids, max_prompt_budget, self.truncate_mode)
            token_ids = new_tokens
            prompt_len = len(token_ids)
            truncated = dropped > 0
            tokens_dropped += dropped
            logger.warn(
                f"Truncated prompt by {dropped} tokens to fit model budget "
                f"(mode={self.truncate_mode}, new_len={prompt_len}, budget={max_prompt_budget})."
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
            do_cap_first = self.prefer_preserve_prompt or not self.auto_truncate_prompt

            if do_cap_first:
                if self.auto_cap_new_tokens:
                    logger.warn(
                        f"Capping max_new_tokens from {requested_new} to {allowed_new_if_keep_prompt} "
                        f"to preserve prompt (prompt_len={prompt_len}, reserve={self.reserve_tokens}, "
                        f"model_max={max_model_len})."
                    )
                    requested_new = allowed_new_if_keep_prompt
                    _set_requested_new(sampling_params, requested_new)
                else:
                    if self.strict_context:
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
                    if self.auto_cap_new_tokens:
                        logger.warn(
                            f"Requested max_new_tokens={requested_new} leaves no room for prompt; "
                            f"capping to {allowed_new_if_keep_prompt} to preserve prompt."
                        )
                        requested_new = allowed_new_if_keep_prompt
                        _set_requested_new(sampling_params, requested_new)
                    else:
                        if self.strict_context:
                            raise ValueError("Requested output too large; would require dropping entire prompt.")
                        requested_new = allowed_new_if_keep_prompt
                        _set_requested_new(sampling_params, requested_new)
                else:
                    if prompt_len > target_prompt_budget:
                        new_tokens, dropped = truncate_tokens(token_ids, target_prompt_budget, self.truncate_mode)
                        token_ids = new_tokens
                        prompt_len = len(token_ids)
                        truncated = truncated or dropped > 0
                        tokens_dropped += dropped
                        self._info(
                            f"Truncated prompt by {dropped} tokens (mode={self.truncate_mode}) "
                            f"to honor requested max_new_tokens={requested_new}. "
                            f"New prompt_len={prompt_len}, target_prompt_budget={target_prompt_budget}."
                        )

        allowed_new_final = max(0, max_model_len - prompt_len - self.reserve_tokens)
        if requested_new > allowed_new_final:
            if self.strict_context and not self.auto_cap_new_tokens:
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
        if truncated and self.decode_truncated_prompt:
            try:
                prompt_for_engine = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception:
                prompt_for_engine = prompt
                logger.warn("Failed to decode truncated prompt; keeping original prompt text.")

        start_ts = time.perf_counter()
        ev = threading.Event()

        with self._request_lock:
            self._request_events[request_id] = ev
            self._active_requests[request_id] = {
                "prompt": prompt_for_engine,
                "prompt_token_ids": token_ids,
                "sampling_params": sampling_params,
                "generated_tokens": [],
                "reported_generated_count": 0,
                "last_decoded_index": 0,
                "start_time": start_ts,
                "first_token_time": None,
                "last_decode_time": start_ts,
                "decoder_visible_text": "",
                "truncated": truncated,
                "tokens_dropped": tokens_dropped,
                "requested_new_tokens_original": original_requested_new,
                "requested_new_tokens_final": requested_new,
                "reserve_tokens": self.reserve_tokens,
                "max_model_len": max_model_len,
                # Per-request parser instances (fresh per request for streaming state isolation)
                "tool_parser_instance": self._tool_parser_class(self.tokenizer) if self._tool_parser_class else None,
                "reasoning_parser_instance": (
                    self._reasoning_parser_class(self.tokenizer) if self._reasoning_parser_class else None
                ),
                "parser_previous_text": "",
                "parser_previous_token_ids": [],
                "accumulated_reasoning": "",
                "accumulated_content": "",
            }
            _rp = self._active_requests[request_id].get("reasoning_parser_instance")
            self._configure_reasoning_parser_for_prompt(
                reasoning_parser=_rp,
                prompt_text=prompt_for_engine,
                prompt_token_ids=token_ids,
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
                tokens_per_second=0.0,
                num_generated_tokens=0,
                time_spent_generating=0.0,
                first_token_time=None,
                processing_time=0.0,
                update_seq=0,
                delta_seq=0,
            )

        # Prepare EOS token IDs from engine-normalized EOS set
        eos_token_ids = [int(tid) for tid in (getattr(self, "_eos_ids", None) or []) if tid is not None]

        if not eos_token_ids:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            fallback_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            eos_token_ids = [int(tid) for tid in fallback_ids if tid is not None]

        # Use the first EOS token as the primary one for backwards compatibility
        primary_eos_token_id = eos_token_ids[0] if eos_token_ids else getattr(self, "_primary_eos_token_id", None)

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
                    self._active_requests[child_request_id] = {
                        "prompt": prompt_for_engine,
                        "prompt_token_ids": token_ids,
                        "sampling_params": sampling_params,
                        "generated_tokens": [],  # Fresh list for each sample
                        "reported_generated_count": 0,
                        "last_decoded_index": 0,
                        "start_time": start_ts,
                        "first_token_time": None,
                        "last_decode_time": start_ts,
                        "decoder_visible_text": "",
                        "truncated": truncated,
                        "tokens_dropped": tokens_dropped,
                        "requested_new_tokens_original": original_requested_new,
                        "requested_new_tokens_final": requested_new,
                        "reserve_tokens": self.reserve_tokens,
                        "max_model_len": max_model_len,
                        "sample_index": sample_idx,
                        "parent_request_id": request_id,
                        # Per-request parser instances (fresh per sample)
                        "tool_parser_instance": (
                            self._tool_parser_class(self.tokenizer) if self._tool_parser_class else None
                        ),
                        "reasoning_parser_instance": (
                            self._reasoning_parser_class(self.tokenizer) if self._reasoning_parser_class else None
                        ),
                        "parser_previous_text": "",
                        "parser_previous_token_ids": [],
                        "accumulated_reasoning": "",
                        "accumulated_content": "",
                    }
                    _rp2 = self._active_requests[child_request_id].get("reasoning_parser_instance")
                    self._configure_reasoning_parser_for_prompt(
                        reasoning_parser=_rp2,
                        prompt_text=prompt_for_engine,
                        prompt_token_ids=token_ids,
                    )

            with self._scheduler_lock:
                self.scheduler.add_request(
                    EngineRequest(
                        request_id=child_request_id,
                        prompt_token_ids=token_ids,
                        sampling_params=sampling_params,
                        eos_token_id=primary_eos_token_id,
                        parent_request_id=parent_id,
                        sample_index=sample_idx,
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

        self._info(
            f"Queued request {request_id}: prompt_len={prompt_len}, "
            f"max_tokens={requested_new}, n={n_samples}, reserve={self.reserve_tokens}, "
            f"model_max={max_model_len}, dropped={tokens_dropped}"
        )

    def _generate_request_id(self) -> str:
        """Generate a unique request ID with overflow protection.

        Uses UUID for uniqueness and a counter for ordering. The counter
        uses modulo arithmetic to prevent unbounded growth in long-running
        services.

        Returns:
            Unique request ID with format 'req-{uuid}-{counter}'.
        """
        with self._counter_lock:
            self._request_counter = (self._request_counter + 1) % (1 << 32)  # Reset after ~4 billion requests
            return f"req-{uuid.uuid4().hex}-{self._request_counter}"

    def abort_request(self, request_id: str) -> None:
        """Abort an in-progress request.

        Marks the request as aborted and signals any waiting threads.
        The request will be removed from the scheduler queue if still waiting.

        Args:
            request_id: ID of the request to abort.
        """
        detokenizer_reset_ids: set[str] = set()
        parent_request_id = request_id
        sample_index = 0
        metrics_collector = get_metrics_collector()

        # Acquire all locks atomically to prevent race conditions
        with self._scheduler_lock, self._request_lock, self._output_lock:
            rd = self._active_requests.get(request_id)
            if rd is not None:
                parent_request_id = rd.get("parent_request_id", request_id)
                sample_index = int(rd.get("sample_index", 0) or 0)

            # Resolve scheduler-side IDs to abort (n=1: request_id; n>1: children of parent)
            abort_ids: set[str] = set()
            if request_id in self.scheduler.requests:
                abort_ids.add(request_id)
                parent_request_id = self.scheduler.requests[request_id].parent_request_id or parent_request_id
            abort_ids.update(
                rid
                for rid, req in self.scheduler.requests.items()
                if getattr(req, "parent_request_id", None) == request_id
            )

            if abort_ids:
                self.scheduler.finish_requests(abort_ids, EngineRequestStatus.FINISHED_ABORTED)
                detokenizer_reset_ids |= abort_ids

            # Clean up active request tracking (output retention honors max_request_outputs).
            for rid in abort_ids:
                self._active_requests.pop(rid, None)
            self._active_requests.pop(parent_request_id, None)
            for rid in abort_ids:
                if rid != parent_request_id:
                    self._request_events.pop(rid, None)

            # Update output state
            ro = self._request_outputs.get(parent_request_id)
            n_samples = len(ro.outputs) if ro is not None else 0
            if ro is not None:
                if request_id == parent_request_id:
                    ro.finished = True
                    for output in ro.outputs:
                        output.finish_reason = "abort"
                    if metrics_collector:
                        metrics_collector.complete_request(parent_request_id, finish_reason="abort")
                else:
                    if 0 <= sample_index < len(ro.outputs):
                        ro.outputs[sample_index].finish_reason = "abort"
                    ro.finished = all(output.finish_reason is not None for output in ro.outputs)
                    if ro.finished and metrics_collector:
                        metrics_collector.complete_request(parent_request_id, finish_reason="abort")
                ro.update_seq += 1

            # Get event while still holding lock (streaming uses parent event)
            ev = self._request_events.get(parent_request_id)

            if not detokenizer_reset_ids:
                detokenizer_reset_ids.add(request_id)

        # Reset detokenizer state (outside locks to avoid blocking)
        for rid in detokenizer_reset_ids:
            try:
                self._detokenizer_client.reset(rid)
                # Remove from failed set if it was there
                self._failed_detokenizer_resets.discard(rid)
            except Exception:
                logger.debug("Failed to reset detokenizer state for %s", rid, exc_info=True)
                # Track failed reset
                self._failed_detokenizer_resets.add(rid)

        # Trigger cleanup if threshold reached
        if len(self._failed_detokenizer_resets) >= self._detokenizer_cleanup_threshold:
            self._cleanup_detokenizer_state()

        # Notify waiters
        if ev:
            ev.set()
        self._output_event.set()
        log_metrics_summary()
        if ro is not None and ro.finished:
            with self._request_lock:
                self._request_events.pop(parent_request_id, None)
                if n_samples > 1:
                    for sample_idx in range(n_samples):
                        self._request_events.pop(f"{parent_request_id}-{sample_idx}", None)
            if self._max_request_outputs is not None:
                with self._output_lock:
                    self._track_finished_output(parent_request_id)

    def _cleanup_detokenizer_state(self) -> None:
        """Attempt to clean up failed detokenizer states.

        Retries resetting detokenizer state for all tracked failed requests.
        Clears successfully reset requests from the tracking set.
        """
        if not self._failed_detokenizer_resets:
            return

        self._info(
            "Attempting to clean up %d failed detokenizer states",
            len(self._failed_detokenizer_resets),
        )

        successfully_reset = set()
        for request_id in list(self._failed_detokenizer_resets):
            try:
                self._detokenizer_client.reset(request_id)
                successfully_reset.add(request_id)
            except Exception:
                # Still failing, keep in set
                pass

        # Remove successfully reset requests
        self._failed_detokenizer_resets -= successfully_reset

        if successfully_reset:
            self._info("Successfully cleaned up %d detokenizer states", len(successfully_reset))
        if self._failed_detokenizer_resets:
            logger.warning(
                "%d detokenizer states still failed to reset",
                len(self._failed_detokenizer_resets),
            )

    @property
    def num_pending_requests(self) -> int:
        """Get the number of requests waiting in queue.

        Returns:
            Count of requests in the waiting queue.
        """
        with self._scheduler_lock:
            return len(self.scheduler.waiting)

    @property
    def num_running_requests(self) -> int:
        """Get the number of actively running requests.

        Returns:
            Count of requests currently being processed.
        """
        with self._scheduler_lock:
            return len(self.scheduler.running)
