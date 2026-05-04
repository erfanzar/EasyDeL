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

"""Parsing mixin for the eSurge engine.

Drives the per-request decoder pipeline that turns raw token outputs into
final user-visible text:

- Tool-call extraction and merging via the configured tool parser.
- Reasoning-block separation (e.g. ``<think>...</think>``) via the
  configured reasoning parser.
- Streaming delta computation that respects reasoning/tool boundaries.
- Stop-string handling and EOS bookkeeping.

Exposes :class:`EngineParsingMixin`, mixed into :class:`eSurge`.
"""

from __future__ import annotations

from collections.abc import Sequence
import time

from ...stream_protocol import compute_stream_delta_text
from ..engine_types import EngineCoreOutputs
from ..logger import logger
from ..metrics import get_metrics_collector
from ..request import EngineRequestStatus


class _TokenPrefixView(Sequence[int]):
    """Fixed-length view over an append-only token list.

    Streaming parsers need the previous token count to compute the current
    delta. Keeping a prefix view avoids copying the full token history every
    token while still behaving like the previous list for ``len()``,
    iteration, indexing, and membership checks.
    """

    __slots__ = ("_tokens", "_length")

    def __init__(self, tokens: list[int], length: int):
        self._tokens = tokens
        self._length = int(length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            return [self._tokens[idx] for idx in range(start, stop, step)]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return self._tokens[index]

    def __contains__(self, value: object) -> bool:
        for idx in range(self._length):
            if self._tokens[idx] == value:
                return True
        return False


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
        """Locate the earliest stop sequence in ``text`` (longest wins on tie).

        Iterates over ``stop_sequences`` and finds each one's first
        occurrence in ``text`` via ``str.find``. Returns the match with the
        smallest index; ties are broken in favour of the *longest* stop
        sequence so that a more specific marker like ``"</tool_call>"``
        wins over a generic ``">"``.

        Args:
            text: Currently-accumulated detokenized output to search.
            stop_sequences: List of strings; empty entries are skipped.

        Returns:
            ``(index, matched_string)`` for the earliest/longest match, or
            ``None`` when no stop sequence appears in ``text``.
        """
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
        """Trim ``accumulated_text`` at the first stop string per sampling-params policy.

        Each request's :class:`SamplingParams` may carry a ``stop`` list and
        an ``include_stop_str_in_output`` flag. This helper consults both
        and either:

        * leaves ``accumulated_text`` untouched when no stop sequence
          matches, or
        * cuts the text at the start of the first match (default), or
        * cuts after the match when ``include_stop_str_in_output=True``.

        It then computes a "visible delta" against the previous visible
        text recorded in ``rd["decoder_visible_text"]`` so the streaming
        client never sees retracted (post-stop) tokens. ``fallback_delta``
        is used only when the visible text equals the accumulated text
        (no trimming happened) — otherwise it would replay text that was
        just cut off.

        Args:
            rd: Per-request bookkeeping dict; ``decoder_visible_text`` is
                read here to compute the delta but updated by the caller.
            accumulated_text: Full detokenized text seen so far for this
                sample.
            fallback_delta: Delta to surface when no trimming occurred (i.e.
                the standard streaming delta from the detokenizer).

        Returns:
            ``(visible_text, visible_delta, stop_triggered, stop_reason)``
            where ``visible_text`` is what downstream parsers/clients may
            see, ``visible_delta`` is its delta against the prior visible
            text, ``stop_triggered`` is ``True`` when a stop sequence
            matched, and ``stop_reason`` is the matched string (or
            ``None``).
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
        """Whether stop-string matching is suppressed inside reasoning blocks.

        When both (a) the request opted into
        ``ignore_stop_strings_in_reasoning`` on its sampling params and
        (b) a reasoning parser is wired up via the request's
        ``DelegatingParser``, stop strings should match only against the
        *visible content* the reasoning parser has emitted, not the raw
        accumulated text — so a stop sequence that happens to occur inside
        an open chain-of-thought block does not prematurely terminate the
        request.

        Args:
            rd: Per-request bookkeeping dict (carries
                ``sampling_params`` and ``delegating_parser``).

        Returns:
            ``True`` iff the request is configured to defer stop-string
            matching until after the reasoning parser has classified text,
            and a reasoning parser is actually present.
        """
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
        """Run reasoning/tool parsers and stop-string trimming in the right order.

        Two operating regimes:

        * **Stop in raw domain** (default) — apply :meth:`_apply_stop_string_policy`
          to the accumulated text *before* feeding it to
          :meth:`_run_output_parsers`, so the reasoning/tool parsers only see
          text up to the stop boundary. Stop strings can match anywhere,
          including inside reasoning blocks.
        * **Stop in parsed domain** (when
          :meth:`_stop_strings_ignore_reasoning` is True) — run the
          parsers first, then apply the stop policy to the parser's
          ``accumulated_content`` so reasoning blocks shield their text from
          stop-string matching.

        In either branch, ``rd["decoder_visible_text"]`` is updated to the
        new visible text so subsequent calls compute correct deltas.

        Args:
            rd: Per-request bookkeeping dict.
            accumulated_text: Full detokenized text so far for this sample.
            delta_text: Newly-decoded text since the previous call.
            token_ids: Token-id sequence aligned with ``accumulated_text``;
                forwarded into the parser for token-level decisions.
            finished: ``True`` iff the request has finished (engine-side
                EOS / length / abort). Forces the parser into final mode.

        Returns:
            ``(parsed, visible_text, visible_delta, stop_hit, stop_reason)``
            — the parser-output dict (with content/reasoning/tool fields),
            the post-policy visible text and delta, a stop-hit flag, and
            the matched stop string (or ``None``).
        """
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
        """Drive the per-request :class:`DelegatingParser` for one decode step.

        The :class:`DelegatingParser` instance held in ``rd`` chains a
        reasoning parser and a tool-call parser; this method picks the
        right entry point — ``process_delta`` for streaming updates,
        ``process_final`` for terminal ones — and threads the previous
        ``parser_previous_text`` / ``parser_previous_token_ids`` so the
        parser can compute cumulative state correctly. After invocation
        those previous-* slots are advanced in ``rd``.

        When no parser is configured for the request (``rd["delegating_parser"]
        is None``), returns a pass-through dict where ``content`` carries
        the verbatim text and ``reasoning`` / ``tool_calls`` are empty.

        Args:
            rd: Per-request bookkeeping dict; reads ``delegating_parser``,
                ``parser_previous_text``, ``parser_previous_token_ids`` and
                writes the latter two on exit. Token ids are stored as a
                prefix view so the hot path keeps the old length without
                copying the full generated-token history.
            accumulated_text: Full detokenized text so far.
            delta_text: New text since the previous call.
            token_ids: Token-id sequence aligned with ``accumulated_text``.
            finished: When ``True``, switches the parser into final mode
                (it may flush partial reasoning blocks or close open tool
                calls).

        Returns:
            Plain ``dict`` with keys ``delta_reasoning``,
            ``delta_content``, ``accumulated_reasoning``,
            ``accumulated_content``, ``tool_calls``, and
            ``delta_tool_calls`` (any may be ``None`` when not applicable
            to the current step).
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
        rd["parser_previous_token_ids"] = _TokenPrefixView(token_ids, len(token_ids))

        return result.to_dict()

    @staticmethod
    def _resolve_public_finish_reason(outputs) -> str | None:
        """Reduce per-sample finish reasons into the parent's single OpenAI reason.

        For ``n>1`` parallel sampling, each sample carries its own
        ``finish_reason``. The parent ``RequestOutput`` exposes one string
        to the client; this helper picks the most informative one using the
        priority order ``"abort" > "length" > "tool_calls" > "stop"``.

        ``"tool_calls"`` outranks ``"stop"`` because the client needs to
        know to execute tools (and re-prompt the model with their results)
        even if other samples in the same request happened to finish on a
        natural stop. ``"abort"`` and ``"length"`` are explicit failure
        modes and override anything else.

        Args:
            outputs: Iterable of ``CompletionOutput`` objects (one per
                sample) carrying a ``finish_reason`` field.

        Returns:
            Highest-priority finish reason found, the first sample's
            reason if none of the priority labels matched, or ``None``
            when ``outputs`` is empty.
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
        now: float,
    ) -> None:
        """Close out a request that the scheduler ended without a final token.

        The normal finish path runs through :meth:`_finalize_request` when
        an :class:`EngineCoreOutput` carries ``finished=True``. Some
        scheduler-side terminations (preemption that didn't recover, hard
        aborts, fatal cache errors) skip that final output and instead
        list the request id in
        ``SchedulerOutput.finished_req_ids``/``finished_requests``. For
        those, the parser mixin still needs to update the per-sample
        ``CompletionOutput`` (defaulting ``finish_reason`` to
        ``"abort"`` with a WARNING when no upstream reason was set),
        bump processing time, mark the parent ``RequestOutput`` finished,
        notify the metrics collector, and drop the entry from
        ``_active_requests``. Streaming clients waiting on the
        per-request event are woken so they can observe the terminal
        state.

        Args:
            request_id: Sample-level id reported by the scheduler.
            metrics_collector: Optional metrics collector;
                ``complete_request`` is invoked when the parent finishes.
            now: Generation-loop timestamp for the scheduler output that
                reported the finish. Metrics use this instead of output-worker
                wall time so parser latency cannot depress generation TPS.
        """

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
            elapsed = max(0.0, now - float(start_time))
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
            tokens_are_eos_filtered=True,
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

    def _append_decodable_tokens(self, rd: dict, new_tokens: list[int]) -> list[int]:
        """Append newly generated tokens to the per-request decode stream.

        ``rd["generated_tokens"]`` intentionally records the exact engine
        output, including EOS ids for accounting and final ``token_ids``.
        The detokenizer wants the same stream with EOS ids removed. Keeping
        ``rd["decodable_tokens"]`` incremental avoids re-filtering the full
        generated list on every streaming update.
        """
        decodable_tokens = rd.setdefault("decodable_tokens", [])
        eos_set = getattr(self, "_eos_set", None)
        if eos_set is None:
            eos_set = getattr(self, "_eSurge__eos_set", None)
        if eos_set:
            decodable_tokens.extend(token_id for token_id in new_tokens if token_id not in eos_set)
        else:
            decodable_tokens.extend(new_tokens)
        return decodable_tokens

    def _get_decodable_tokens_for_final(self, rd: dict) -> list[int]:
        """Return the EOS-filtered decode stream for a terminal parse pass.

        Normal generation updates keep this list incrementally. The fallback
        preserves correctness for older request dictionaries or tests that
        construct ``rd`` directly without ``decodable_tokens``.
        """
        decodable_tokens = rd.get("decodable_tokens")
        if decodable_tokens is not None:
            return decodable_tokens
        return self._filter_eos_tokens(rd["generated_tokens"])

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
        """Write parser output back into ``CompletionOutput`` / ``RequestOutput``.

        Updates the per-sample :class:`CompletionOutput` (``comp``) for any
        ``sample_index``; only the *first* sample (index 0) is mirrored
        onto the parent :class:`RequestOutput` (``ro``) because that's the
        view streaming clients see. The first-sample branch also computes
        an ``effective_delta`` against ``ro.accumulated_text`` to guarantee
        clients always observe a strictly-growing visible text — useful
        when stop-string trimming or reasoning-block retraction would
        otherwise produce a non-monotonic delta.

        Args:
            comp: Per-sample completion output to update.
            ro: Parent request output (only mutated when ``sample_index == 0``).
            sample_index: Position of this sample within ``ro.outputs``.
            parsed: Output dict from :meth:`_run_output_parsers` /
                :meth:`_parse_with_stop_string_policy`.
            raw_accumulated: Pre-parser detokenized text so far (also
                surfaced on ``comp.raw_text`` / ``ro.raw_accumulated_text``).
            raw_delta: Newly-decoded raw text for this step.
            visible_delta: Stop-policy-trimmed delta from
                :meth:`_apply_stop_string_policy`; used as a fallback when
                the parser emits no delta but the visible text changed.

        Returns:
            ``(text_changed, structured_changed)``: the first flag is
            ``True`` when the visible text moved (so streamers should be
            woken), the second when reasoning or tool-call structure
            changed (used to bump update sequencing without forcing a
            text refresh).
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
        """Recompute the per-request streaming metrics shown on ``RequestOutput``.

        Recomputed every time the parser advances (so streaming clients can
        poll ``tokens_per_second`` mid-flight). The token count is treated
        idempotently: ``rd["reported_generated_count"]`` records the last
        observed total so out-of-order or duplicate updates don't double-
        count, and a regression (lower count than reported) silently resets
        the watermark.

        Args:
            rd: Per-request bookkeeping dict; reads ``start_time`` and
                ``reported_generated_count``, mutates the latter.
            ro: Parent :class:`RequestOutput` whose ``processing_time`` /
                ``time_spent_generating`` / ``num_generated_tokens`` /
                ``tokens_per_second`` fields are refreshed.
            now: Scheduler-output timestamp used for elapsed computation.
                This is intentionally produced before asynchronous output
                parsing so host-side parsing latency does not change the
                reported generation TPS.
            num_generated: Cumulative count of generated tokens reported
                for this step.
        """
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
        """Run the terminal-step pipeline for a finishing request/sample.

        Called when the engine output indicates this sample has finished
        (either naturally or because ``force_finished`` was raised by a
        stop-string match upstream). Sets ``comp.finish_reason``, marks
        the parent ``ro.finished`` if all samples are done, runs a *final*
        decode pass through :meth:`_decode_and_parse` (which flushes any
        held-back partial reasoning blocks / tool calls), updates the
        completion outputs, recomputes final metrics, and reports
        completion to the metrics collector.

        Args:
            request_id: Sample-level request id (parent or child for n>1).
            rd: Per-request bookkeeping dict.
            ro: Parent :class:`RequestOutput`.
            comp: Per-sample completion output being finalized.
            sample_index: Position within ``ro.outputs``.
            parent_request_id: Id used for metrics-collector accounting.
            engine_output: Engine-side ``EngineCoreOutput`` carrying the
                raw ``finish_reason``.
            force_finished: ``True`` when the parser detected a stop
                string that the engine has not yet acted on; promotes the
                finish reason to ``"stop"``.
            stop_string_finishes: Out-parameter dict; populated with
                ``request_id -> matched_stop`` so the engine can record
                which stop string ended each request.
            metrics_collector: Optional metrics collector;
                ``complete_request`` is called when ``ro.finished``.
            now: Scheduler-output timestamp used for elapsed-time computations.

        Returns:
            ``(text_changed, structured_changed)`` flags for the streaming
            event-signal logic, mirroring :meth:`_update_outputs`.
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

        decodable_tokens = self._get_decodable_tokens_for_final(rd)
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
                now = float(client_outputs.timestamp or time.perf_counter())
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

                        decodable_tokens = self._append_decodable_tokens(rd, new_tokens)
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
                        now=now,
                    )

        if stop_string_finishes:
            enqueue_stops = getattr(self, "_enqueue_parser_stop_requests", None)
            if enqueue_stops is not None:
                enqueue_stops(stop_string_finishes)
            else:
                with self._scheduler_lock:
                    for rid, stop_reason in stop_string_finishes.items():
                        request = self.scheduler.requests.get(rid)
                        if request is not None:
                            request.stop_reason = stop_reason
                    self.scheduler.finish_requests(stop_string_finishes.keys(), EngineRequestStatus.FINISHED_STOPPED)

        self._output_event.set()
