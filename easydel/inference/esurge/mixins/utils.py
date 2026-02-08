# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import copy
import threading
import time
import typing
from typing import Any

from eformer.loggings import get_logger

from easydel.inference.sampling_params import SamplingParams
from easydel.workers.esurge.pipeline import DetokenizerResult

from ..metrics import get_metrics_collector

logger = get_logger("eSurgeEngine")
WORKER_DRAIN_MAX_RETRIES = 3
WORKER_DRAIN_INITIAL_DELAY = 0.1


class EngineUtilsMixin:
    def _format_chat_prompt(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Format chat messages into a prompt string using the tokenizer's chat template.

        Converts a list of chat messages into a formatted prompt string that can be
        passed to the model for generation. Uses the tokenizer's built-in chat template
        or a custom template if provided.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                Roles can be 'system', 'user', 'assistant', etc.
            add_generation_prompt: Whether to add the generation prompt token/string
                at the end to indicate the model should generate a response.
            chat_template: Optional custom chat template to override the tokenizer's
                default template. Should be a Jinja2 template string.
            tools: Optional list of tool/function definitions that the model can use.
                Format depends on the specific model's tool calling conventions.

        Returns:
            Formatted prompt string ready for tokenization and generation.

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is 2+2?"}
            ... ]
            >>> prompt = engine._format_chat_prompt(messages)
            >>> # Returns formatted string like: "<|system|>You are a helpful assistant.<|user|>What is 2+2?<|assistant|>"

        Note:
            The exact format depends on the tokenizer's chat template. Different models
            use different special tokens and formatting conventions.
        """

        if chat_template_kwargs is None:
            chat_template_kwargs = {}
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            **chat_template_kwargs,
        )

    def _tokenize_prompt(self, request_id: str, prompt: str) -> list[int]:
        """Tokenize a prompt string using the worker pipeline.

        Args:
            request_id: Request ID for tracking in the tokenizer worker.
            prompt: Text prompt to tokenize.

        Returns:
            List of token IDs.
        """
        return self._tokenizer_client.tokenize(request_id, prompt)

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

    def _filter_eos_tokens(self, tokens: list[int]) -> list[int]:
        """Remove EOS tokens from a token list before decoding.

        Args:
            tokens: List of token IDs.

        Returns:
            Token list with EOS tokens removed.
        """
        eos_set = getattr(self, "_eos_set", None)
        if eos_set is None:
            # Backward-compat fallback for pre-refactor state layouts.
            eos_set = getattr(self, "_eSurge__eos_set", None)
        if not eos_set:
            return tokens
        return [tok for tok in tokens if tok not in eos_set]

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

    def _decode_with_pipeline(
        self,
        request_id: str,
        generated_tokens: list[int],
        *,
        finished: bool,
        skip_special_tokens: bool = False,
    ) -> DetokenizerResult:
        """Decode tokens using the detokenizer worker pipeline.

        Performs incremental detokenization with streaming text delta support.

        Args:
            request_id: Request ID for tracking state in the detokenizer.
            generated_tokens: Full list of generated tokens so far.
            finished: Whether generation is complete (triggers final flush).
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            DetokenizerResult with accumulated_text, delta_text, and indices.
        """
        tokens_for_decode = self._filter_eos_tokens(generated_tokens)
        return self._detokenizer_client.decode(
            request_id,
            tokens_for_decode,
            finished=finished,
            skip_special_tokens=skip_special_tokens,
        )

    @staticmethod
    def _compute_snapshot_delta_text(current_text: str, previous_text: str, fallback_delta: str) -> str:
        """Compute a safe streaming delta from accumulated text snapshots.

        Prefer exact prefix-diff semantics and avoid replaying stale fallback
        chunks when text has not advanced. If prefix alignment is lost, attempt
        suffix-prefix overlap recovery before falling back.
        """

        current_text = current_text or ""
        previous_text = previous_text or ""

        if current_text.startswith(previous_text):
            return current_text[len(previous_text) :]

        max_overlap = min(len(previous_text), len(current_text))
        for overlap in range(max_overlap, 0, -1):
            if previous_text.endswith(current_text[:overlap]):
                return current_text[overlap:]

        if previous_text:
            logger.warning(
                "Stream delta alignment mismatch. prev_len=%s, curr_len=%s; using guarded fallback.",
                len(previous_text),
                len(current_text),
            )

        if fallback_delta and (not previous_text or not previous_text.endswith(fallback_delta)):
            return fallback_delta
        return current_text if not previous_text else ""

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        """Convert a value to a Python scalar if possible.

        Args:
            value: Value to convert (may be a JAX/numpy array).

        Returns:
            Python scalar if conversion possible, original value otherwise.
        """
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _sanitize_metrics_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Sanitize a metrics payload by converting arrays to scalars.

        Args:
            payload: Dictionary of metric values.

        Returns:
            Dictionary with all values converted to Python scalars.
        """
        return {k: self._to_python_scalar(v) for k, v in payload.items()}

    def _kv_cache_metadata(self) -> dict[str, Any]:
        """Get current KV cache configuration metadata.

        Returns:
            Dictionary with cache configuration including max_model_len,
            max_num_seqs, page_size, and executor-specific attributes.
        """
        metadata = getattr(getattr(self.runner, "executor_manager", None), "metadata", None)
        details: dict[str, Any] = {
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "page_size": self.page_size,
        }
        if metadata is not None:
            for attr in ("num_pages", "page_size", "max_model_length", "hbm_utilization"):
                value = getattr(metadata, attr, None)
                if value is not None:
                    details[attr] = self._to_python_scalar(value)
        return details

    def _record_cache_event(self, event: str, payload: dict[str, Any]) -> None:
        """Record a cache event to the metrics collector.

        Args:
            event: Event name (e.g., "kv_cache_destroyed", "kv_cache_reinitialized").
            payload: Event details dictionary.
        """
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_cache_event(event, payload)

    def _log_cache_event(self, event: str, extra: dict[str, Any] | None = None) -> None:
        """Log a KV cache event with metadata.

        Args:
            event: Event name for logging.
            extra: Additional event details to include.
        """
        payload = self._kv_cache_metadata()
        if extra:
            payload.update(extra)
        sanitized = self._sanitize_metrics_payload(payload)
        self._info("KV cache %s: %s", event, sanitized)
        self._record_cache_event(event, sanitized)

    def _drain_pipeline_workers(self, reason: str) -> None:
        """Drain tokenizer/detokenizer workers with retry logic.

        Args:
            reason: Reason for draining (for logging).
        """
        manager = getattr(self, "_worker_manager", None)
        if not manager:
            return

        max_retries = WORKER_DRAIN_MAX_RETRIES
        retry_delay = WORKER_DRAIN_INITIAL_DELAY

        for attempt in range(max_retries):
            try:
                manager.drain_workers()
                self._info("Drained tokenizer/detokenizer workers (%s)", reason)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Failed to drain workers (attempt %d/%d): %s. Retrying in %.2fs...",
                        attempt + 1,
                        max_retries,
                        e,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        "Failed to drain tokenizer/detokenizer workers after %d attempts during %s",
                        max_retries,
                        reason,
                        exc_info=True,
                    )

    def _touch_activity(self) -> None:
        """Update the last-activity timestamp for idle reset tracking."""
        if self._idle_reset_seconds is None:
            return
        self._idle_reset_last_activity = time.time()

    def _start_idle_monitor(self) -> None:
        """Start the idle-reset monitor thread if enabled."""
        if self._idle_reset_seconds is None:
            return
        if self._idle_monitor_thread and self._idle_monitor_thread.is_alive():
            return
        self._idle_monitor_event.clear()

        check_interval = min(1.0, max(self._idle_reset_seconds / 4.0, 0.1))

        def _idle_loop() -> None:
            while not self._idle_monitor_event.wait(check_interval):
                if not self._scheduler_running:
                    continue
                now = time.time()
                if self._idle_reset_seconds is None:
                    continue
                idle_for = now - self._idle_reset_last_activity
                if idle_for < self._idle_reset_seconds:
                    continue
                if now - self._idle_reset_last_reset < self._idle_reset_min_interval:
                    continue
                if self.num_running_requests != 0 or self.num_pending_requests != 0 or self._active_requests:
                    # Activity resumed while waiting.
                    self._idle_reset_last_activity = now
                    continue
                self._idle_reset_last_reset = now
                self._idle_reset_last_activity = now
                self._info("Idle reset triggered after %.1fs of inactivity", idle_for)
                try:
                    self.pause()
                    self.resume()
                except Exception:
                    logger.exception("Idle reset failed")

        self._idle_monitor_thread = threading.Thread(target=_idle_loop, daemon=True)
        self._idle_monitor_thread.start()

    def _stop_idle_monitor(self) -> None:
        """Stop the idle-reset monitor thread if running."""
        if not self._idle_monitor_thread:
            return
        self._idle_monitor_event.set()
        if threading.current_thread() is self._idle_monitor_thread:
            # Avoid joining the current thread; it will exit on the next wake-up.
            self._idle_monitor_thread = None
            return
        self._idle_monitor_thread.join(timeout=2.0)
        if self._idle_monitor_thread.is_alive():
            logger.debug("Idle monitor thread did not stop gracefully")
        self._idle_monitor_thread = None

    def _clone_sampling_params(self, sampling_params: SamplingParams) -> SamplingParams:
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

    def _prepare_sampling_params_for_request(
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
        callback = self._sampling_params_callback
        if callback is None:
            return params

        metadata = {"request_id": request_id, "prompt": prompt, "engine": self}
        try:
            result = callback(params, metadata)
            if result is None:
                return params
            return result
        except Exception:
            logger.exception("Sampling params callback failed; falling back to unmodified parameters")
            return params
