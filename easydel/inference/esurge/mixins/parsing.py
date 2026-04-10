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

import time

from ...stream_protocol import compute_stream_delta_text
from ..engine_types import EngineCoreOutputs
from ..logger import logger
from ..metrics import get_metrics_collector
from ..request import EngineRequestStatus


class EngineParsingMixin:
    """Mixin for output parsing, stop-string detection, and token processing.

    Handles the pipeline from raw engine token outputs to structured text
    with reasoning extraction, tool call parsing, and stop-string trimming.
    Coordinates incremental decoding with interval-based batching to reduce
    tokenizer overhead during streaming, and signals per-request events
    for streaming consumers.

    Methods:
        _process_engine_outputs: Core method that processes scheduler outputs
            into decoded text, applies parsers, and updates request state.
        _run_output_parsers: Runs reasoning and tool parsers on decoded text.
        _apply_stop_string_policy: Trims text at stop-string boundaries.
    """

    @staticmethod
    def _find_first_stop_string(text: str, stop_sequences: list[str]) -> tuple[int, str] | None:
        """Return the earliest stop-string match in text, if any."""
        best_index: int | None = None
        best_stop: str | None = None
        for stop_seq in stop_sequences:
            if not stop_seq:
                continue
            idx = text.find(stop_seq)
            if idx < 0:
                continue
            if best_index is None or idx < best_index or (idx == best_index and len(stop_seq) > len(best_stop or "")):
                best_index = idx
                best_stop = stop_seq
        if best_index is None or best_stop is None:
            return None
        return best_index, best_stop

    def _apply_stop_string_policy(
        self,
        rd: dict,
        *,
        accumulated_text: str,
        fallback_delta: str,
    ) -> tuple[str, str, bool, str | None]:
        """Apply stop-string trimming policy to decoded text.

        Returns:
            visible_text: Text allowed to be exposed to downstream parsers/clients.
            visible_delta: Delta computed against prior visible text.
            stop_triggered: Whether a stop string was matched in accumulated_text.
            stop_reason: The matched stop string when stop_triggered is True.
        """
        sampling_params = rd.get("sampling_params")
        previous_visible_text = rd.get("decoder_visible_text", "") or ""
        visible_text = accumulated_text
        stop_triggered = False
        stop_reason = None

        if sampling_params is not None:
            stop_sequences = [str(s) for s in (getattr(sampling_params, "stop", None) or []) if s]
            include_stop = bool(getattr(sampling_params, "include_stop_str_in_output", False))
            if stop_sequences:
                matched = self._find_first_stop_string(accumulated_text, stop_sequences)
                if matched is not None:
                    stop_index, stop_reason = matched
                    cutoff = stop_index + len(stop_reason) if include_stop else stop_index
                    visible_text = accumulated_text[:cutoff]
                    stop_triggered = True

        # Avoid replaying buffered/retracted text via stale fallback deltas.
        delta_fallback = fallback_delta if visible_text == accumulated_text else ""
        visible_delta = self._compute_snapshot_delta_text(visible_text, previous_visible_text, delta_fallback)
        return visible_text, visible_delta, stop_triggered, stop_reason

    @staticmethod
    def _stop_strings_ignore_reasoning(rd: dict) -> bool:
        """Return whether stop strings should only match parsed visible content."""
        sampling_params = rd.get("sampling_params")
        if sampling_params is None:
            return False
        if not bool(getattr(sampling_params, "ignore_stop_strings_in_reasoning", False)):
            return False
        dp = rd.get("delegating_parser")
        return dp is not None and dp.reasoning_parser is not None

    def _parse_with_stop_string_policy(
        self,
        rd: dict,
        *,
        accumulated_text: str,
        delta_text: str,
        token_ids: list[int],
        finished: bool,
    ) -> tuple[dict, str, str, bool, str | None]:
        """Parse decoded text and apply stop-string policy in the correct domain."""
        if self._stop_strings_ignore_reasoning(rd):
            parsed = self._run_output_parsers(
                rd=rd,
                accumulated_text=accumulated_text,
                delta_text=delta_text,
                token_ids=token_ids,
                finished=finished,
            )
            visible_text, visible_delta, stop_hit, stop_reason = self._apply_stop_string_policy(
                rd,
                accumulated_text=parsed["accumulated_content"],
                fallback_delta=parsed["delta_content"] or delta_text,
            )
            rd["decoder_visible_text"] = visible_text
            parsed["accumulated_content"] = visible_text
            parsed["delta_content"] = visible_delta
            return parsed, visible_text, visible_delta, stop_hit, stop_reason

        visible_text, visible_delta, stop_hit, stop_reason = self._apply_stop_string_policy(
            rd,
            accumulated_text=accumulated_text,
            fallback_delta=delta_text,
        )
        rd["decoder_visible_text"] = visible_text
        parsed = self._run_output_parsers(
            rd=rd,
            accumulated_text=visible_text,
            delta_text=visible_delta,
            token_ids=token_ids,
            finished=finished or stop_hit,
        )
        return parsed, visible_text, visible_delta, stop_hit, stop_reason

    def _run_output_parsers(
        self,
        rd: dict,
        accumulated_text: str,
        delta_text: str,
        token_ids: list[int],
        finished: bool,
    ) -> dict:
        """Run reasoning and tool parsers on decoded text via DelegatingParser.

        Returns:
            Dict with keys: delta_reasoning, delta_content, accumulated_reasoning,
            accumulated_content, tool_calls, delta_tool_calls.
        """
        dp = rd.get("delegating_parser")
        if dp is None:
            return {
                "delta_reasoning": None,
                "delta_content": delta_text,
                "accumulated_reasoning": "",
                "accumulated_content": accumulated_text,
                "tool_calls": None,
                "delta_tool_calls": None,
            }

        prev_text = rd.get("parser_previous_text", "")
        prev_token_ids = rd.get("parser_previous_token_ids", [])

        if finished:
            result = dp.process_final(accumulated_text, token_ids)
        else:
            result = dp.process_delta(accumulated_text, delta_text, token_ids, prev_text, prev_token_ids)

        rd["parser_previous_text"] = accumulated_text
        rd["parser_previous_token_ids"] = list(token_ids)

        return result.to_dict()

    @staticmethod
    def _resolve_public_finish_reason(outputs) -> str | None:
        """Collapse per-sample completion reasons into a single public reason.

        Priority: abort > length > tool_calls > stop.
        tool_calls beats stop because the client needs to know to execute tools.
        """
        finish_reason = outputs[0].finish_reason if outputs else None
        if any(output.finish_reason == "abort" for output in outputs):
            return "abort"
        if any(output.finish_reason == "length" for output in outputs):
            return "length"
        if any(output.finish_reason == "tool_calls" for output in outputs):
            return "tool_calls"
        if any(output.finish_reason == "stop" for output in outputs):
            return "stop"
        return finish_reason

    def _finish_request_from_scheduler_signal(
        self,
        request_id: str,
        *,
        metrics_collector,
    ) -> None:
        """Finalize a request when the scheduler reports completion without token output."""

        rd = self._active_requests.get(request_id)
        if rd is None:
            return

        parent_request_id = rd.get("parent_request_id", request_id)
        sample_index = int(rd.get("sample_index", 0) or 0)
        ro = self._request_outputs.get(parent_request_id)
        if ro is None:
            self._active_requests.pop(request_id, None)
            return
        if sample_index < 0 or sample_index >= len(ro.outputs):
            self._active_requests.pop(request_id, None)
            return

        comp = ro.outputs[sample_index]
        if comp.finish_reason is None:
            logger.warning(
                "Finishing request %s from scheduler finished_requests without a terminal EngineCoreOutput; "
                "defaulting finish_reason=abort.",
                request_id,
            )
            comp.finish_reason = "abort"

        start_time = rd.get("start_time")
        if start_time is not None:
            elapsed = max(0.0, time.perf_counter() - float(start_time))
            ro.processing_time = max(ro.processing_time, elapsed)
            ro.time_spent_generating = max(ro.time_spent_generating, elapsed)

        if sample_index == 0:
            ro.delta_text = ""
            ro.raw_delta_text = ""
            ro.delta_reasoning_content = None
            ro.delta_tool_calls = None

        if len(ro.outputs) == 1:
            ro.finished = True
        else:
            ro.finished = all(output.finish_reason is not None for output in ro.outputs)

        if ro.finished and metrics_collector:
            metrics_collector.complete_request(
                parent_request_id,
                finish_reason=self._resolve_public_finish_reason(ro.outputs),
            )

        try:
            self._detokenizer_client.reset(request_id)
        except Exception:
            logger.debug("Failed to reset detokenizer state for %s", request_id, exc_info=True)

        self._active_requests.pop(request_id, None)
        if ro.finished and parent_request_id != request_id:
            self._active_requests.pop(parent_request_id, None)

        ro.update_seq += 1
        ev = self._request_events.get(parent_request_id)
        if ev:
            ev.set()

    def _decode_and_parse(
        self,
        request_id: str,
        rd: dict,
        decodable_tokens: list[int],
        now: float,
        finished: bool,
    ) -> tuple[dict | None, str, str, bool, str | None]:
        """Decode tokens and run parser pipeline.

        Returns:
            (parsed, raw_accumulated_text, raw_delta_text, stop_hit, stop_reason)
            or (None, "", "", False, None) if decode was skipped.
        """
        last_idx = rd["last_decoded_index"]
        num_decodable = len(decodable_tokens)
        sampling_params = rd.get("sampling_params")
        skip_special_tokens = bool(getattr(sampling_params, "skip_special_tokens", False))
        spaces_between_special_tokens = bool(getattr(sampling_params, "spaces_between_special_tokens", True))

        if not finished:
            has_stop_strings = bool(getattr(sampling_params, "stop", None))
            if has_stop_strings:
                interval_tokens = min(self.decode_interval_tokens, 4)
                interval_secs = min(self.decode_interval_secs, 0.02)
            else:
                interval_tokens = self.decode_interval_tokens
                interval_secs = self.decode_interval_secs

            should_decode = (
                num_decodable - last_idx >= interval_tokens or (now - rd.get("last_decode_time", now)) >= interval_secs
            )
            if not should_decode or num_decodable <= last_idx:
                return None, "", "", False, None
        else:
            if num_decodable == 0 and last_idx == 0:
                return None, "", "", False, None

        prompt_ctx = rd.get("prompt_token_ids") if last_idx == 0 else None
        pipeline_result = self._decode_with_pipeline(
            request_id,
            decodable_tokens,
            finished=finished,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            prompt_context=prompt_ctx[-8:] if prompt_ctx else None,
        )
        rd["last_decoded_index"] = pipeline_result.last_decoded_index
        if not finished:
            rd["last_decode_time"] = now

        raw_accumulated = pipeline_result.accumulated_text
        raw_delta = pipeline_result.delta_text or ""

        parsed, _visible, _vis_delta, stop_hit, stop_reason = self._parse_with_stop_string_policy(
            rd,
            accumulated_text=pipeline_result.accumulated_text,
            delta_text=pipeline_result.delta_text,
            token_ids=decodable_tokens,
            finished=finished,
        )
        return parsed, raw_accumulated, raw_delta, stop_hit, stop_reason

    @staticmethod
    def _update_outputs(
        comp,
        ro,
        sample_index: int,
        parsed: dict,
        raw_accumulated: str,
        raw_delta: str,
        visible_delta: str = "",
    ) -> tuple[bool, bool]:
        """Apply parsed results to CompletionOutput and RequestOutput.

        Returns:
            (text_changed, structured_changed) for event signaling decisions.
        """
        text_changed = False
        structured_changed = False

        comp.text = parsed["accumulated_content"]
        comp.raw_text = raw_accumulated
        if parsed["accumulated_reasoning"]:
            comp.reasoning_content = parsed["accumulated_reasoning"]
        if parsed["tool_calls"]:
            comp.tool_calls = parsed["tool_calls"]

        if sample_index == 0:
            previous_accumulated_text = ro.accumulated_text
            ro.raw_accumulated_text = raw_accumulated
            ro.raw_delta_text = raw_delta or ""
            ro.accumulated_text = parsed["accumulated_content"]
            effective_delta = compute_stream_delta_text(
                parsed["accumulated_content"],
                previous_accumulated_text,
                parsed["delta_content"] or visible_delta,
            )
            ro.delta_text = effective_delta or ""
            ro.delta_reasoning_content = parsed["delta_reasoning"]
            ro.reasoning_content = parsed["accumulated_reasoning"] or None
            ro.delta_tool_calls = parsed["delta_tool_calls"]
            if parsed["tool_calls"]:
                ro.tool_calls = parsed["tool_calls"]
            if effective_delta:
                ro.delta_seq += 1
                text_changed = True
            if parsed["delta_reasoning"] is not None or parsed["delta_tool_calls"]:
                structured_changed = True

        return text_changed, structured_changed

    @staticmethod
    def _update_metrics(rd: dict, ro, now: float, num_generated: int) -> None:
        """Update TTFT, tokens/sec, and timing metrics."""
        elapsed = now - rd["start_time"]
        ro.processing_time = elapsed
        ro.time_spent_generating = elapsed

        prev_reported = rd.get("reported_generated_count", 0)
        if num_generated >= prev_reported:
            ro.num_generated_tokens += num_generated - prev_reported
            rd["reported_generated_count"] = num_generated
        else:
            rd["reported_generated_count"] = num_generated

        if ro.first_token_time is not None and ro.num_generated_tokens > 0:
            generation_time = elapsed - ro.first_token_time
            ro.tokens_per_second = ro.num_generated_tokens / generation_time if generation_time > 0 else 0.0
        else:
            ro.tokens_per_second = 0.0

    def _finalize_request(
        self,
        request_id: str,
        rd: dict,
        ro,
        comp,
        sample_index: int,
        parent_request_id: str,
        engine_output,
        force_finished: bool,
        stop_string_finishes: dict[str, str],
        metrics_collector,
        now: float,
    ) -> tuple[bool, bool]:
        """Handle request completion: final decode, finish_reason, cleanup.

        Returns:
            (text_changed, structured_changed) for event signaling.
        """
        text_changed = False
        structured_changed = False

        if force_finished and not engine_output.finished:
            comp.finish_reason = "stop"
        else:
            comp.finish_reason = str(engine_output.finish_reason) if engine_output.finish_reason else "finished"

        n_samples = len(ro.outputs)
        if n_samples == 1:
            ro.finished = True
        else:
            ro.finished = all(output.finish_reason is not None for output in ro.outputs)

        decodable_tokens = self._filter_eos_tokens(rd["generated_tokens"])
        parsed, raw_accumulated, raw_delta, stop_hit, stop_reason = self._decode_and_parse(
            request_id,
            rd,
            decodable_tokens,
            now,
            finished=True,
        )

        if stop_hit and stop_reason:
            stop_string_finishes[request_id] = stop_reason

        if parsed is not None:
            comp.token_ids = list(rd["generated_tokens"])
            if parsed["tool_calls"]:
                comp.finish_reason = "tool_calls"
            text_changed, structured_changed = self._update_outputs(
                comp,
                ro,
                sample_index,
                parsed,
                raw_accumulated,
                raw_delta,
            )

        elapsed = now - rd["start_time"]
        num_prompt_tokens = (
            len(rd["prompt_token_ids"]) if "prompt_token_ids" in rd else sum(len(seg) for seg in ro.prompt_token_ids)
        )
        ro.processing_time = elapsed
        ro.time_spent_generating = elapsed
        if ro.first_token_time is not None and ro.num_generated_tokens > 0:
            generation_time = elapsed - ro.first_token_time
            ro.tokens_per_second = ro.num_generated_tokens / generation_time if generation_time > 0 else 0.0
        else:
            ro.tokens_per_second = 0.0
        ro.metrics = {
            "prompt_tokens": num_prompt_tokens,
            "generated_tokens": ro.num_generated_tokens,
            "total_tokens": num_prompt_tokens + ro.num_generated_tokens,
            "processing_time": elapsed,
            "first_token_time": ro.first_token_time,
            "tokens_per_second": ro.tokens_per_second,
        }

        if metrics_collector and ro.finished:
            metrics_collector.complete_request(
                parent_request_id,
                finish_reason=self._resolve_public_finish_reason(ro.outputs),
            )
        try:
            self._detokenizer_client.reset(request_id)
        except Exception:
            logger.debug("Failed to reset detokenizer state for %s", request_id, exc_info=True)
        self._active_requests.pop(request_id, None)
        if ro.finished and parent_request_id != request_id:
            self._active_requests.pop(parent_request_id, None)

        return text_changed, structured_changed

    def _process_engine_outputs(self, engine_outputs: dict[int, EngineCoreOutputs]) -> None:
        """Process engine outputs and update request outputs (thread-safe).

        Decodes tokens, runs reasoning + tool parsers, updates metrics, and
        signals streaming consumers. Uses interval-based decoding to reduce
        tokenizer overhead.
        """
        metrics_collector = get_metrics_collector()
        if engine_outputs:
            self._touch_activity()

        stop_string_finishes: dict[str, str] = {}

        with self._request_lock, self._output_lock:
            for client_outputs in engine_outputs.values():
                for engine_output in client_outputs.outputs:
                    request_id = engine_output.request_id
                    rd = self._active_requests.get(request_id)
                    if rd is None:
                        continue

                    parent_request_id = rd.get("parent_request_id", request_id)
                    sample_index = rd.get("sample_index", 0)
                    ro = self._request_outputs.get(parent_request_id)
                    if ro is None:
                        continue

                    text_changed = False
                    structured_changed = False
                    force_finished = False
                    new_tokens = engine_output.new_token_ids
                    now = time.perf_counter()

                    if new_tokens:
                        rd["generated_tokens"].extend(new_tokens)
                        num_generated = len(rd["generated_tokens"])

                        if rd["first_token_time"] is None and num_generated > 0:
                            rd["first_token_time"] = now - rd["start_time"]
                        if rd["first_token_time"] is not None:
                            if ro.first_token_time is None:
                                ro.first_token_time = rd["first_token_time"]
                                if metrics_collector:
                                    metrics_collector.record_first_token(parent_request_id)
                            else:
                                ro.first_token_time = min(ro.first_token_time, rd["first_token_time"])
                        if metrics_collector:
                            metrics_collector.add_generated_tokens(parent_request_id, len(new_tokens))

                        decodable_tokens = self._filter_eos_tokens(rd["generated_tokens"])
                        parsed, raw_accumulated, raw_delta, stop_hit, stop_reason = self._decode_and_parse(
                            request_id,
                            rd,
                            decodable_tokens,
                            now,
                            finished=False,
                        )
                        if stop_hit:
                            force_finished = True
                            if stop_reason:
                                stop_string_finishes[request_id] = stop_reason

                        if parsed is not None:
                            comp = ro.outputs[sample_index]
                            comp.token_ids = list(rd["generated_tokens"])
                            text_changed, structured_changed = self._update_outputs(
                                comp,
                                ro,
                                sample_index,
                                parsed,
                                raw_accumulated,
                                raw_delta,
                            )

                        self._update_metrics(rd, ro, now, num_generated)

                    if engine_output.finished or force_finished:
                        comp = ro.outputs[sample_index]
                        tc, sc = self._finalize_request(
                            request_id,
                            rd,
                            ro,
                            comp,
                            sample_index,
                            parent_request_id,
                            engine_output,
                            force_finished,
                            stop_string_finishes,
                            metrics_collector,
                            now,
                        )
                        text_changed = text_changed or tc
                        structured_changed = structured_changed or sc

                    ro.update_seq += 1
                    if text_changed or structured_changed or engine_output.finished or force_finished:
                        ev = self._request_events.get(parent_request_id)
                        if ev:
                            ev.set()

                for finished_request_id in client_outputs.finished_requests or ():
                    self._finish_request_from_scheduler_signal(
                        finished_request_id,
                        metrics_collector=metrics_collector,
                    )

        if stop_string_finishes:
            with self._scheduler_lock:
                for rid, stop_reason in stop_string_finishes.items():
                    request = self.scheduler.requests.get(rid)
                    if request is None:
                        continue
                    request.stop_reason = stop_reason
                self.scheduler.finish_requests(stop_string_finishes.keys(), EngineRequestStatus.FINISHED_STOPPED)

        self._output_event.set()
