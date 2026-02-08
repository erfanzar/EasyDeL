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

import time

from eformer.loggings import get_logger

from ..engine_types import EngineCoreOutputs
from ..metrics import get_metrics_collector
from ..request import EngineRequestStatus

logger = get_logger("eSurgeEngine")


class EngineParsingMixin:
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

    def _run_output_parsers(
        self,
        rd: dict,
        accumulated_text: str,
        delta_text: str,
        token_ids: list[int],
        finished: bool,
    ) -> dict:
        """Run reasoning and tool parsers on decoded text.

        Reasoning runs first, then tool parsing on content portion only.

        Returns:
            Dict with keys: delta_reasoning, delta_content, accumulated_reasoning,
            accumulated_content, tool_calls, delta_tool_calls.
        """
        result = {
            "delta_reasoning": None,
            "delta_content": delta_text,
            "accumulated_reasoning": rd.get("accumulated_reasoning", ""),
            "accumulated_content": rd.get("accumulated_content", ""),
            "tool_calls": None,
            "delta_tool_calls": None,
        }

        reasoning_parser = rd.get("reasoning_parser_instance")
        tool_parser = rd.get("tool_parser_instance")
        prev_text = rd.get("parser_previous_text", "")
        prev_token_ids = rd.get("parser_previous_token_ids", [])

        content_for_tools = accumulated_text

        # Step 1: Reasoning extraction
        if reasoning_parser is not None:
            try:
                if finished:
                    reasoning, content = reasoning_parser.extract_reasoning(accumulated_text)
                    result["accumulated_reasoning"] = reasoning or ""
                    if content is None:
                        content_for_tools = "" if reasoning is not None else accumulated_text
                    else:
                        content_for_tools = content
                    result["accumulated_content"] = content_for_tools
                    # Calculate delta from previous
                    old_reasoning = rd.get("accumulated_reasoning", "")
                    if reasoning and len(reasoning) > len(old_reasoning):
                        result["delta_reasoning"] = reasoning[len(old_reasoning) :]
                    # Content delta: use content-only portion
                    if content is not None:
                        _, prev_content = reasoning_parser.extract_reasoning(prev_text)
                        prev_content = prev_content or ""
                        if len(content) > len(prev_content):
                            result["delta_content"] = content[len(prev_content) :]
                        else:
                            result["delta_content"] = ""
                    elif reasoning is not None:
                        result["delta_content"] = ""
                else:
                    delta_ids = token_ids[len(prev_token_ids) :] if prev_token_ids else token_ids
                    delta_msg = reasoning_parser.extract_reasoning_streaming(
                        previous_text=prev_text,
                        current_text=accumulated_text,
                        delta_text=delta_text,
                        previous_token_ids=prev_token_ids,
                        current_token_ids=token_ids,
                        delta_token_ids=delta_ids,
                    )
                    if delta_msg is not None:
                        result["delta_reasoning"] = delta_msg.reasoning_content
                        if delta_msg.content is not None:
                            result["delta_content"] = delta_msg.content
                        elif delta_msg.reasoning_content is not None:
                            # Explicitly hide reasoning-only deltas from content.
                            result["delta_content"] = ""
                        # Accumulate reasoning
                        if delta_msg.reasoning_content:
                            result["accumulated_reasoning"] = (
                                rd.get("accumulated_reasoning", "") + delta_msg.reasoning_content
                            )

                    # Extract content portion for tool parser
                    reasoning, content = reasoning_parser.extract_reasoning(accumulated_text)
                    if content is None:
                        content_for_tools = "" if reasoning is not None else accumulated_text
                    else:
                        content_for_tools = content
                    result["accumulated_content"] = content_for_tools
            except Exception:
                result["accumulated_content"] = accumulated_text
                logger.debug("Reasoning extraction failed for request", exc_info=True)
        else:
            result["accumulated_content"] = accumulated_text

        # Step 2: Tool call extraction on content portion
        if tool_parser is not None and content_for_tools:
            try:
                if finished:
                    from easydel.inference.openai_api_modules import ChatCompletionRequest, ChatMessage

                    dummy_request = ChatCompletionRequest(
                        model="dummy",
                        messages=[ChatMessage(role="user", content="")],
                    )
                    extracted = tool_parser.extract_tool_calls(content_for_tools, dummy_request)
                    if extracted.tools_called and extracted.tool_calls:
                        result["tool_calls"] = extracted.tool_calls
                        # Update delta_content to exclude tool call markup
                        if extracted.content is not None:
                            result["accumulated_content"] = extracted.content
                            result["delta_content"] = ""  # Content was already streamed
                else:
                    # For streaming, compute content deltas
                    if reasoning_parser is not None:
                        _, prev_content = reasoning_parser.extract_reasoning(prev_text)
                    else:
                        prev_content = prev_text

                    if prev_content is None:
                        prev_content = prev_text
                    content_delta = result["delta_content"]
                    if content_delta is None:
                        content_delta = delta_text

                    from easydel.inference.openai_api_modules import ChatCompletionRequest, ChatMessage

                    dummy_request = ChatCompletionRequest(
                        model="dummy",
                        messages=[ChatMessage(role="user", content="")],
                    )
                    delta_ids = token_ids[len(prev_token_ids) :] if prev_token_ids else token_ids
                    delta_msg = tool_parser.extract_tool_calls_streaming(
                        previous_text=prev_content,
                        current_text=content_for_tools,
                        delta_text=content_delta or "",
                        previous_token_ids=prev_token_ids,
                        current_token_ids=token_ids,
                        delta_token_ids=delta_ids,
                        request=dummy_request,
                    )
                    if delta_msg is not None:
                        if hasattr(delta_msg, "tool_calls") and delta_msg.tool_calls:
                            result["delta_tool_calls"] = delta_msg.tool_calls
                        # If tool parser consumed the content, clear delta_content
                        if delta_msg.tool_calls and not delta_msg.content:
                            result["delta_content"] = ""
                        elif delta_msg.content is not None:
                            result["delta_content"] = delta_msg.content
            except Exception:
                logger.debug("Tool call extraction failed for request", exc_info=True)

        # Update tracking state
        rd["parser_previous_text"] = accumulated_text
        rd["parser_previous_token_ids"] = list(token_ids)
        rd["accumulated_reasoning"] = result["accumulated_reasoning"]
        rd["accumulated_content"] = result["accumulated_content"]

        return result

    def _process_engine_outputs(self, engine_outputs: dict[int, EngineCoreOutputs]) -> None:
        """Process engine outputs and update request outputs (thread-safe).

        Core method that processes tokens from the model, performs incremental
        decoding, updates metrics, and signals waiting threads. Uses interval-based
        decoding to reduce tokenizer overhead during streaming.

        Args:
            engine_outputs: Dictionary mapping client IDs to engine outputs containing
                new tokens, completion status, and metadata.

        Processing Flow:
            1. Extracts new tokens from engine outputs
            2. Performs interval-based decoding (every 4 tokens or 20ms)
            3. Updates accumulated and delta text fields
            4. Tracks performance metrics (TTFT, tokens/sec)
            5. Handles request completion with final token flush
            6. Signals per-request events for streaming consumers

        Thread Safety:
            Uses request_lock and output_lock to ensure atomic updates across
            multiple concurrent requests and streaming consumers.
        """
        metrics_collector = get_metrics_collector()
        if engine_outputs:
            self._touch_activity()

        stop_string_finishes: dict[str, str] = {}

        # Update both request_data and public outputs atomically
        with self._request_lock, self._output_lock:
            for client_outputs in engine_outputs.values():
                for engine_output in client_outputs.outputs:
                    request_id = engine_output.request_id
                    rd = self._active_requests.get(request_id)
                    if rd is None:
                        continue

                    # Handle n>1 sampling: get parent request and sample index
                    parent_request_id = rd.get("parent_request_id", request_id)
                    sample_index = rd.get("sample_index", 0)
                    ro = self._request_outputs.get(parent_request_id)
                    if ro is None:
                        continue

                    text_changed = False
                    force_finished = False
                    new_tokens = engine_output.new_token_ids
                    now = time.perf_counter()
                    elapsed = now - rd["start_time"]
                    if new_tokens:
                        rd["generated_tokens"].extend(new_tokens)
                        num_generated = len(rd["generated_tokens"])
                        decodable_tokens = self._filter_eos_tokens(rd["generated_tokens"])
                        num_decodable = len(decodable_tokens)

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

                        last_idx = rd["last_decoded_index"]
                        sampling_params = rd.get("sampling_params")
                        has_stop_strings = bool(getattr(sampling_params, "stop", None))
                        if has_stop_strings:
                            stop_decode_interval_tokens = min(self.decode_interval_tokens, 4)
                            stop_decode_interval_secs = min(self.decode_interval_secs, 0.02)
                            should_decode = (
                                num_decodable - last_idx >= stop_decode_interval_tokens
                                or (now - rd.get("last_decode_time", now)) >= stop_decode_interval_secs
                            )
                        else:
                            should_decode = (
                                num_decodable - last_idx >= self.decode_interval_tokens
                                or (now - rd.get("last_decode_time", now)) >= self.decode_interval_secs
                            )
                        if should_decode and num_decodable > last_idx:
                            pipeline_result = self._decode_with_pipeline(
                                request_id,
                                decodable_tokens,
                                finished=False,
                            )
                            rd["last_decoded_index"] = pipeline_result.last_decoded_index
                            rd["last_decode_time"] = now

                            visible_text, visible_delta, stop_hit, stop_reason = self._apply_stop_string_policy(
                                rd,
                                accumulated_text=pipeline_result.accumulated_text,
                                fallback_delta=pipeline_result.delta_text,
                            )
                            rd["decoder_visible_text"] = visible_text
                            if stop_hit:
                                force_finished = True
                                if stop_reason:
                                    stop_string_finishes[request_id] = stop_reason

                            # Run reasoning and tool parsers on decoded text
                            parsed = self._run_output_parsers(
                                rd=rd,
                                accumulated_text=visible_text,
                                delta_text=visible_delta,
                                token_ids=decodable_tokens,
                                finished=force_finished,
                            )

                            # Update the specific sample's completion output
                            comp = ro.outputs[sample_index]
                            comp.text = parsed["accumulated_content"]
                            comp.token_ids = list(rd["generated_tokens"])
                            if parsed["accumulated_reasoning"]:
                                comp.reasoning_content = parsed["accumulated_reasoning"]
                            if parsed["tool_calls"]:
                                comp.tool_calls = parsed["tool_calls"]

                            # For backwards compatibility, set ro fields to first sample
                            if sample_index == 0:
                                ro.accumulated_text = parsed["accumulated_content"]
                                effective_delta = parsed["delta_content"]
                                if effective_delta is None:
                                    effective_delta = visible_delta
                                ro.delta_text = effective_delta or ""
                                ro.delta_reasoning_content = parsed["delta_reasoning"]
                                ro.reasoning_content = parsed["accumulated_reasoning"] or None
                                ro.delta_tool_calls = parsed["delta_tool_calls"]
                                if parsed["tool_calls"]:
                                    ro.tool_calls = parsed["tool_calls"]

                            if visible_delta:
                                ro.delta_seq += 1
                                text_changed = True

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
                            ro.tokens_per_second = (
                                ro.num_generated_tokens / generation_time if generation_time > 0 else 0.0
                            )
                        else:
                            ro.tokens_per_second = 0.0

                    if engine_output.finished or force_finished:
                        comp = ro.outputs[sample_index]
                        if force_finished and not engine_output.finished:
                            comp.finish_reason = "stop"
                        else:
                            comp.finish_reason = (
                                str(engine_output.finish_reason) if engine_output.finish_reason else "finished"
                            )

                        # For n>1, mark RequestOutput as finished only when ALL samples are done
                        n_samples = len(ro.outputs)
                        if n_samples == 1:
                            ro.finished = True
                        else:
                            # Check if all samples have finish_reason set
                            all_finished = all(output.finish_reason is not None for output in ro.outputs)
                            ro.finished = all_finished

                        num_generated = len(rd["generated_tokens"])
                        decodable_tokens = self._filter_eos_tokens(rd["generated_tokens"])
                        num_decodable = len(decodable_tokens)
                        last_idx = rd["last_decoded_index"]
                        if num_decodable > last_idx:
                            pipeline_result = self._decode_with_pipeline(
                                request_id,
                                decodable_tokens,
                                finished=True,
                            )
                            rd["last_decoded_index"] = pipeline_result.last_decoded_index
                            visible_text, visible_delta, stop_hit, stop_reason = self._apply_stop_string_policy(
                                rd,
                                accumulated_text=pipeline_result.accumulated_text,
                                fallback_delta=pipeline_result.delta_text,
                            )
                            rd["decoder_visible_text"] = visible_text
                            if stop_hit and stop_reason:
                                stop_string_finishes[request_id] = stop_reason

                            # Run reasoning and tool parsers (final decode)
                            parsed = self._run_output_parsers(
                                rd=rd,
                                accumulated_text=visible_text,
                                delta_text=visible_delta,
                                token_ids=decodable_tokens,
                                finished=True,
                            )

                            # Update the specific sample's completion output
                            comp.text = parsed["accumulated_content"]
                            comp.token_ids = list(rd["generated_tokens"])
                            if parsed["accumulated_reasoning"]:
                                comp.reasoning_content = parsed["accumulated_reasoning"]
                            if parsed["tool_calls"]:
                                comp.tool_calls = parsed["tool_calls"]
                                comp.finish_reason = "tool_calls"

                            # For backwards compatibility, set ro fields to first sample
                            if sample_index == 0:
                                ro.accumulated_text = parsed["accumulated_content"]
                                effective_delta = parsed["delta_content"]
                                if effective_delta is None:
                                    effective_delta = visible_delta
                                ro.delta_text = effective_delta or ""
                                ro.delta_reasoning_content = parsed["delta_reasoning"]
                                ro.reasoning_content = parsed["accumulated_reasoning"] or None
                                ro.delta_tool_calls = parsed["delta_tool_calls"]
                                if parsed["tool_calls"]:
                                    ro.tool_calls = parsed["tool_calls"]

                            if visible_delta:
                                ro.delta_seq += 1
                                text_changed = True

                        num_prompt_tokens = (
                            len(rd["prompt_token_ids"])
                            if "prompt_token_ids" in rd
                            else sum(len(seg) for seg in ro.prompt_token_ids)
                        )

                        ro.processing_time = elapsed
                        ro.time_spent_generating = elapsed

                        if ro.first_token_time is not None and ro.num_generated_tokens > 0:
                            generation_time = elapsed - ro.first_token_time
                            ro.tokens_per_second = (
                                ro.num_generated_tokens / generation_time if generation_time > 0 else 0.0
                            )
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
                            finish_reason = comp.finish_reason
                            if any(out.finish_reason == "abort" for out in ro.outputs):
                                finish_reason = "abort"
                            elif any(out.finish_reason == "length" for out in ro.outputs):
                                finish_reason = "length"
                            elif any(out.finish_reason == "stop" for out in ro.outputs):
                                finish_reason = "stop"
                            metrics_collector.complete_request(
                                parent_request_id,
                                finish_reason=finish_reason,
                            )
                        try:
                            self._detokenizer_client.reset(request_id)
                        except Exception:
                            logger.debug("Failed to reset detokenizer state for %s", request_id, exc_info=True)
                        self._active_requests.pop(request_id, None)
                        if ro.finished and parent_request_id != request_id:
                            self._active_requests.pop(parent_request_id, None)
                    ro.update_seq += 1
                    if text_changed or engine_output.finished or force_finished:
                        # Signal the parent request event
                        ev = self._request_events.get(parent_request_id)
                        if ev:
                            ev.set()

        if stop_string_finishes:
            with self._scheduler_lock:
                for rid, stop_reason in stop_string_finishes.items():
                    request = self.scheduler.requests.get(rid)
                    if request is None:
                        continue
                    request.stop_reason = stop_reason
                self.scheduler.finish_requests(stop_string_finishes.keys(), EngineRequestStatus.FINISHED_STOPPED)

        self._output_event.set()
