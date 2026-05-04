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

"""Unified delegating parser that orchestrates reasoning and tool extraction.

Wraps a ReasoningParser and a ToolParser behind a single interface with an
explicit phase state machine, eliminating the ad-hoc orchestration and
expensive retokenization previously done in EngineParsingMixin._run_output_parsers().

Design inspired by vLLM's DelegatingParser pattern.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

from ..openai_api_modules import ChatCompletionRequest, ChatMessage, DeltaFunctionCall, DeltaToolCall, ToolCall
from ..reasoning.abstract_reasoning import ReasoningParser
from ..stream_protocol import compute_stream_delta_text
from ..tools.abstract_tool import ToolParser

logger = logging.getLogger("easydel.inference.parsing")


class ParsePhase(enum.IntEnum):
    """Explicit phase of the combined reasoning + tool parsing pipeline.

    The ``DelegatingParser`` maintains a single phase variable that
    governs how incoming token deltas are routed:

    - ``REASONING``: the model is inside a reasoning section (e.g.
      ``<think>…</think>``).  All deltas are forwarded to the reasoning
      parser; visible content is suppressed.
    - ``CONTENT``: normal text generation.  Deltas are forwarded to the
      tool parser for protocol detection and, if no tool protocol is
      active, surfaced as visible content.
    - ``TOOL_CALL``: a tool-call protocol has been detected and is
      actively being parsed.  Visible content is frozen to prevent
      markup leaking to the client.
    - ``BUFFERING``: the parser suspects a tool-call start marker may
      be forming but cannot confirm yet.  Text is withheld from the
      client until the ambiguity resolves (either entering
      ``TOOL_CALL`` or falling back to ``CONTENT``).
    """

    REASONING = 0
    CONTENT = 1
    TOOL_CALL = 2
    BUFFERING = 3


@dataclass
class ParseResult:
    """Structured output returned by ``DelegatingParser.process_delta`` / ``process_final``.

    Each field captures one dimension of the parse state:

    - ``delta_reasoning`` / ``delta_content``: the incremental text chunks
      the client should append to its reasoning and content buffers
      respectively.  ``None`` means "no change this step".
    - ``accumulated_reasoning`` / ``accumulated_content``: the full text
      seen so far after reasoning extraction and tool-markup stripping.
    - ``tool_calls``: populated only on final processing when batch
      extraction finds completed tool calls.
    - ``delta_tool_calls``: incremental tool-call deltas for streaming.
    - ``phase``: the parser phase *after* this step (useful for callers
      that need to know whether buffering is active).

    ``to_dict()`` flattens the result into the dict format consumed by
    ``EngineParsingMixin._process_engine_outputs``.
    """

    delta_reasoning: str | None = None
    delta_content: str | None = None
    accumulated_reasoning: str = ""
    accumulated_content: str = ""
    tool_calls: list | None = None
    delta_tool_calls: list | None = None
    phase: ParsePhase = ParsePhase.CONTENT

    def to_dict(self) -> dict:
        """Flatten into the dict format consumed by ``EngineParsingMixin._run_output_parsers``."""
        return {
            "delta_reasoning": self.delta_reasoning,
            "delta_content": self.delta_content,
            "accumulated_reasoning": self.accumulated_reasoning,
            "accumulated_content": self.accumulated_content,
            "tool_calls": self.tool_calls,
            "delta_tool_calls": self.delta_tool_calls,
        }


@dataclass
class TrackedToolCallState:
    """Per-tool-call accumulator that tracks what has already been streamed.

    During streaming, tool-call deltas arrive incrementally (first the
    function name, then argument chunks).  This state records what the
    client has already received so that ``_build_missing_final_tool_deltas``
    can synthesize a catch-up delta at finalization time for any data
    that the streaming parser missed or that arrived only in the batch
    extraction.
    """

    index: int
    tool_call_id: str | None = None
    tool_type: str | None = None
    function_name: str = ""
    function_arguments: str = ""


class DelegatingParser:
    """Orchestrates reasoning and tool parsers with an explicit phase state machine.

    Instead of tangling reasoning extraction, tool parsing, buffering detection,
    and retokenization in one method, this class maintains an explicit ``phase``
    and delegates to the appropriate parser at each step.

    Key design: when reasoning ends, the tool parser's previous-text view is
    **reset** (not retokenized). The tool parser only ever sees content-portion
    text, starting from a clean slate after the reasoning boundary.
    """

    def __init__(
        self,
        reasoning_parser: ReasoningParser | None = None,
        tool_parser: ToolParser | None = None,
        tool_request: ChatCompletionRequest | None = None,
    ):
        """Initialize the orchestrator with optional reasoning/tool parsers.

        Args:
            reasoning_parser: Parser used to detect and extract a reasoning
                section from the model output. ``None`` disables reasoning
                extraction and starts the state machine in
                :attr:`ParsePhase.CONTENT`.
            tool_parser: Parser used to detect tool calls in the visible
                content. ``None`` disables tool-call extraction.
            tool_request: Original chat-completion request used to look up
                ``tools``/``tool_choice`` and decide whether tool calls are
                actually permitted.
        """
        self.reasoning_parser = reasoning_parser
        self.tool_parser = tool_parser
        self.tool_request = tool_request

        self.phase: ParsePhase = ParsePhase.REASONING if reasoning_parser is not None else ParsePhase.CONTENT

        self._accumulated_reasoning: str = ""
        self._accumulated_content: str = ""
        self._raw_content_text: str = ""

        self._tool_previous_text: str = ""
        self._tool_previous_token_ids: list[int] = []
        self._streamed_tool_call_state: dict[int, TrackedToolCallState] = {}

        self._content_committed: bool = False

    @staticmethod
    def _compute_visible_content_delta(
        current_text: str,
        previous_text: str,
        fallback_delta: str = "",
    ) -> str:
        """Compute a canonical visible-text delta from accumulated snapshots.

        Thin wrapper around :func:`compute_stream_delta_text` exposed as a
        method so subclasses can override the strategy.

        Args:
            current_text: Current cumulative visible content.
            previous_text: Visible content prior to this delta.
            fallback_delta: Engine-supplied delta used when prefix matching
                fails.

        Returns:
            The newly visible text since ``previous_text``.
        """
        return compute_stream_delta_text(current_text, previous_text, fallback_delta)

    def _merge_streamed_tool_call_state(self, delta_tool_calls: list | None) -> None:
        """Accumulate streamed tool-call metadata for final delta synthesis.

        Args:
            delta_tool_calls: Iterable of :class:`DeltaToolCall` instances or
                raw dicts emitted during streaming. Invalid entries are
                silently skipped.
        """

        if not delta_tool_calls:
            return

        for raw_call in delta_tool_calls:
            tool_call: DeltaToolCall
            if isinstance(raw_call, DeltaToolCall):
                tool_call = raw_call
            else:
                try:
                    tool_call = DeltaToolCall.model_validate(raw_call)
                except Exception:
                    continue

            state = self._streamed_tool_call_state.setdefault(
                int(tool_call.index),
                TrackedToolCallState(index=int(tool_call.index)),
            )
            if tool_call.id:
                state.tool_call_id = tool_call.id
            if tool_call.type:
                state.tool_type = tool_call.type
            if tool_call.function is not None:
                if tool_call.function.name:
                    state.function_name = tool_call.function.name
                if tool_call.function.arguments:
                    state.function_arguments += tool_call.function.arguments

    def _build_missing_final_tool_deltas(self, tool_calls: list | None) -> list[DeltaToolCall] | None:
        """Build final delta_tool_calls for any tool data not streamed earlier.

        Compares the tracked streaming state against the batch-extracted
        tool calls and synthesizes a delta covering any missing function
        names or arguments. This guarantees clients receive the full call
        even when the streaming parser missed a portion.

        Args:
            tool_calls: Final batch-extracted tool calls.

        Returns:
            A list of synthesized :class:`DeltaToolCall` objects, or ``None``
            when no catch-up deltas are required.
        """

        if not tool_calls:
            return None

        synthesized: list[DeltaToolCall] = []
        for index, raw_tool_call in enumerate(tool_calls):
            tool_call: ToolCall
            if isinstance(raw_tool_call, ToolCall):
                tool_call = raw_tool_call
            else:
                try:
                    tool_call = ToolCall.model_validate(raw_tool_call)
                except Exception:
                    continue

            state = self._streamed_tool_call_state.get(index)
            function_name = tool_call.function.name
            function_arguments = tool_call.function.arguments or ""

            missing_name = function_name if state is None or not state.function_name else None
            if state is None:
                missing_arguments = function_arguments
            elif function_arguments.startswith(state.function_arguments):
                missing_arguments = function_arguments[len(state.function_arguments) :]
            elif function_arguments != state.function_arguments:
                missing_arguments = function_arguments
            else:
                missing_arguments = ""

            if not missing_name and not missing_arguments:
                continue

            synthesized.append(
                DeltaToolCall(
                    index=index,
                    id=tool_call.id if state is None else None,
                    type=tool_call.type if state is None else None,
                    function=DeltaFunctionCall(
                        name=missing_name,
                        arguments=missing_arguments or None,
                    ),
                )
            )

        return synthesized or None

    def _canonicalize_reasoning_content(
        self,
        *,
        accumulated_text: str,
        fallback_content: str,
    ) -> tuple[str, str | None]:
        """Return the canonical visible content after reasoning extraction.

        Args:
            accumulated_text: Full accumulated text from the engine.
            fallback_content: Visible content to fall back to when the
                reasoning parser cannot infer the visible portion.

        Returns:
            Tuple ``(visible_content, reasoning_text)`` where ``reasoning_text``
            may be ``None`` when no reasoning section was detected.
        """

        if self.reasoning_parser is None:
            return fallback_content, None

        reasoning, content = self.reasoning_parser.extract_reasoning(accumulated_text)
        if content is None:
            return fallback_content, reasoning
        return content, reasoning

    def process_delta(
        self,
        accumulated_text: str,
        delta_text: str,
        token_ids: list[int],
        prev_text: str,
        prev_token_ids: list[int],
    ) -> ParseResult:
        """Process a streaming delta through reasoning then tool parsing.

        Args:
            accumulated_text: Full decoded text so far.
            delta_text: New text in this chunk.
            token_ids: All token IDs decoded so far.
            prev_text: Accumulated text as of previous call.
            prev_token_ids: Token IDs as of previous call.

        Returns:
            ParseResult with deltas and accumulated state.
        """
        result = ParseResult(
            accumulated_reasoning=self._accumulated_reasoning,
            accumulated_content=self._accumulated_content,
            phase=self.phase,
        )

        content_text = accumulated_text
        content_delta = delta_text

        if self.reasoning_parser is not None:
            content_text, content_delta = self._process_reasoning_delta(
                result=result,
                accumulated_text=accumulated_text,
                delta_text=delta_text,
                token_ids=token_ids,
                prev_text=prev_text,
                prev_token_ids=prev_token_ids,
            )

        if self.tool_parser is not None and self.phase != ParsePhase.REASONING:
            self._process_tool_delta(
                result=result,
                content_text=content_text,
                content_delta=content_delta,
                token_ids=token_ids,
            )
        else:
            result.accumulated_content = (
                content_text if self.phase != ParsePhase.REASONING else self._accumulated_content
            )
            if self.phase != ParsePhase.REASONING and content_delta:
                result.delta_content = content_delta

        self._accumulated_reasoning = result.accumulated_reasoning
        self._raw_content_text = content_text
        self._accumulated_content = result.accumulated_content
        result.phase = self.phase
        return result

    def process_final(
        self,
        accumulated_text: str,
        token_ids: list[int],
    ) -> ParseResult:
        """Process the final (finished) output through both parsers in batch mode.

        Args:
            accumulated_text: Complete decoded text.
            token_ids: All token IDs.

        Returns:
            ParseResult with final accumulated state.
        """
        result = ParseResult(
            accumulated_reasoning=self._accumulated_reasoning,
            accumulated_content=self._accumulated_content,
            phase=self.phase,
        )

        content_for_tools = accumulated_text

        if self.reasoning_parser is not None:
            try:
                reasoning, content = self.reasoning_parser.extract_reasoning(accumulated_text)
                result.accumulated_reasoning = reasoning or ""
                if content is None:
                    content_for_tools = "" if reasoning is not None else accumulated_text
                else:
                    content_for_tools = content
                result.accumulated_content = content_for_tools

                old_reasoning = self._accumulated_reasoning
                if reasoning and len(reasoning) > len(old_reasoning):
                    result.delta_reasoning = reasoning[len(old_reasoning) :]

                old_content = self._raw_content_text
                if content_for_tools:
                    result.delta_content = self._compute_visible_content_delta(
                        content_for_tools,
                        old_content,
                    )
                elif reasoning is not None:
                    result.delta_content = ""
            except Exception:
                logger.warning("Reasoning extraction failed in final processing", exc_info=True)
                result.accumulated_content = accumulated_text
                content_for_tools = accumulated_text
        else:
            result.accumulated_content = accumulated_text

        if self.tool_parser is not None and content_for_tools:
            self._process_tool_final(result, content_for_tools)

        self._accumulated_reasoning = result.accumulated_reasoning
        self._raw_content_text = content_for_tools
        self._accumulated_content = result.accumulated_content
        self.phase = ParsePhase.CONTENT
        result.phase = self.phase
        return result

    def _process_reasoning_delta(
        self,
        result: ParseResult,
        accumulated_text: str,
        delta_text: str,
        token_ids: list[int],
        prev_text: str,
        prev_token_ids: list[int],
    ) -> tuple[str, str]:
        """Run reasoning parser on streaming delta. Returns (content_text, content_delta).

        Handles both REASONING phase and non-REASONING phase (where the model
        may start reasoning mid-stream). Transitions phase to CONTENT when the
        end token is detected.

        Args:
            result: Mutable :class:`ParseResult` updated in place with the
                reasoning/content deltas and accumulated state.
            accumulated_text: Full decoded text seen so far.
            delta_text: Newly added text in this engine snapshot.
            token_ids: All token IDs decoded so far.
            prev_text: Decoded text up to the previous snapshot.
            prev_token_ids: Token IDs up to the previous snapshot.

        Returns:
            Tuple ``(content_text, content_delta)`` describing the visible
            content after reasoning has been peeled off.
        """
        content_text = self._raw_content_text
        content_delta = ""

        try:
            delta_ids = token_ids[len(prev_token_ids) :] if prev_token_ids else token_ids

            if self.phase == ParsePhase.REASONING:
                delta_msg = self.reasoning_parser.extract_reasoning_streaming(
                    previous_text=prev_text,
                    current_text=accumulated_text,
                    delta_text=delta_text,
                    previous_token_ids=prev_token_ids,
                    current_token_ids=token_ids,
                    delta_token_ids=delta_ids,
                )

                if delta_msg is not None:
                    if delta_msg.reasoning_content is not None:
                        result.delta_reasoning = delta_msg.reasoning_content
                        result.accumulated_reasoning = self._accumulated_reasoning + delta_msg.reasoning_content

                    if delta_msg.content is not None:
                        self.phase = ParsePhase.CONTENT
                        canonical_content, canonical_reasoning = self._canonicalize_reasoning_content(
                            accumulated_text=accumulated_text,
                            fallback_content=self._raw_content_text + delta_msg.content,
                        )
                        content_text = canonical_content
                        content_delta = self._compute_visible_content_delta(
                            content_text,
                            self._raw_content_text,
                            delta_msg.content,
                        )
                        result.delta_content = content_delta
                        result.accumulated_content = content_text
                        if canonical_reasoning is not None:
                            result.accumulated_reasoning = canonical_reasoning

                        self._tool_previous_text = ""
                        self._tool_previous_token_ids = []
                    elif delta_msg.reasoning_content is not None:
                        result.delta_content = ""
                        result.accumulated_content = self._raw_content_text

                if self.phase == ParsePhase.REASONING and delta_msg is None:
                    reasoning, content = self.reasoning_parser.extract_reasoning(accumulated_text)
                    if content is not None and content != accumulated_text:
                        self.phase = ParsePhase.CONTENT
                        old_content = self._raw_content_text
                        candidate_delta = self._compute_visible_content_delta(content, old_content, delta_text)
                        if content.startswith(old_content) or candidate_delta:
                            content_text = content
                            content_delta = candidate_delta
                        else:
                            content_text = old_content
                            content_delta = ""
                        result.accumulated_content = content_text
                        result.delta_content = content_delta
                        result.accumulated_reasoning = reasoning or self._accumulated_reasoning
                        self._tool_previous_text = ""
                        self._tool_previous_token_ids = []
                    elif delta_msg is None:
                        result.delta_content = ""
                        result.accumulated_content = self._raw_content_text
            else:
                reasoning, content = self.reasoning_parser.extract_reasoning(accumulated_text)

                if reasoning is not None and reasoning != self._accumulated_reasoning:
                    old_reasoning = self._accumulated_reasoning
                    result.accumulated_reasoning = reasoning
                    if len(reasoning) > len(old_reasoning):
                        result.delta_reasoning = reasoning[len(old_reasoning) :]

                if content is not None:
                    old_content = self._raw_content_text
                    candidate_delta = self._compute_visible_content_delta(content, old_content, delta_text)
                    if content.startswith(old_content) or candidate_delta:
                        content_text = content
                        content_delta = candidate_delta
                    else:
                        content_text = old_content
                        content_delta = ""
                else:
                    content_text = self._raw_content_text
                    content_delta = ""

        except Exception:
            logger.warning(
                "Reasoning streaming extraction failed; treating delta as content",
                exc_info=True,
            )
            self.phase = ParsePhase.CONTENT
            content_text = accumulated_text
            content_delta = delta_text
            result.accumulated_content = content_text

        return content_text, content_delta

    def _get_tool_request(self) -> ChatCompletionRequest:
        """Return the tool request, creating a dummy one if needed.

        Returns:
            The original :class:`ChatCompletionRequest` if available, else
            a minimal dummy request that satisfies tool-parser interfaces.
        """
        if self.tool_request is not None:
            return self.tool_request
        return ChatCompletionRequest(
            model="dummy",
            messages=[ChatMessage(role="user", content="")],
        )

    def _is_tools_enabled(self) -> bool:
        """Check whether tool calling is actually enabled for this request.

        Returns:
            ``True`` when the request declares tools and ``tool_choice`` is
            not ``"none"``; ``False`` otherwise.
        """
        if self.tool_request is None:
            return False
        tools = getattr(self.tool_request, "tools", None)
        if not tools:
            return False
        tool_choice = getattr(self.tool_request, "tool_choice", None)
        if isinstance(tool_choice, str) and tool_choice.strip().lower() == "none":
            return False
        return True

    def _process_tool_delta(
        self,
        result: ParseResult,
        content_text: str,
        content_delta: str,
        token_ids: list[int],
    ) -> None:
        """Run tool parser on a streaming content delta. Mutates *result* in place.

        Args:
            result: Mutable :class:`ParseResult` updated in place with content
                and tool-call deltas.
            content_text: Cumulative visible content (post reasoning).
            content_delta: Newly produced visible text since the last call.
            token_ids: All token IDs decoded so far.
        """
        tool_request = self._get_tool_request()
        tools_enabled = self._is_tools_enabled()
        previous_visible_content = self._accumulated_content

        try:
            tool_current_token_ids = self._tokenize_for_tool_view(content_text)
            tool_delta_token_ids = self._tokenize_for_tool_view(content_delta) if content_delta else []

            delta_msg = self.tool_parser.extract_tool_calls_streaming(
                previous_text=self._tool_previous_text,
                current_text=content_text,
                delta_text=content_delta or "",
                previous_token_ids=self._tool_previous_token_ids,
                current_token_ids=tool_current_token_ids,
                delta_token_ids=tool_delta_token_ids,
                request=tool_request,
            )

            buffering = bool(
                hasattr(self.tool_parser, "is_buffering_protocol")
                and self.tool_parser.is_buffering_protocol(
                    current_text=content_text,
                    delta_text=content_delta or "",
                )
            )

            if delta_msg is not None:
                if hasattr(delta_msg, "tool_calls") and delta_msg.tool_calls:
                    result.delta_tool_calls = delta_msg.tool_calls
                    self._merge_streamed_tool_call_state(result.delta_tool_calls)
                    if tools_enabled:
                        self.phase = ParsePhase.TOOL_CALL
                    result.accumulated_content = previous_visible_content
                    result.delta_content = ""
                elif buffering or self.phase == ParsePhase.BUFFERING:
                    if tools_enabled and not self._content_committed:
                        self.phase = ParsePhase.BUFFERING
                    result.accumulated_content = previous_visible_content
                    result.delta_content = ""
                elif self.phase == ParsePhase.TOOL_CALL:
                    result.accumulated_content = previous_visible_content
                    result.delta_content = ""
                elif delta_msg.content is not None:
                    visible_delta = delta_msg.content if isinstance(delta_msg.content, str) else str(delta_msg.content)
                    result.accumulated_content = f"{previous_visible_content}{visible_delta}"
                    result.delta_content = visible_delta
                    if tools_enabled:
                        self._content_committed = True
                else:
                    result.accumulated_content = content_text
                    result.delta_content = content_delta
            elif buffering or self.phase == ParsePhase.BUFFERING:
                if tools_enabled and not self._content_committed:
                    self.phase = ParsePhase.BUFFERING
                result.accumulated_content = previous_visible_content
                result.delta_content = ""
            elif self.phase == ParsePhase.TOOL_CALL:
                result.accumulated_content = previous_visible_content
                result.delta_content = ""
            else:
                result.accumulated_content = content_text
                result.delta_content = content_delta

            self._tool_previous_text = content_text
            self._tool_previous_token_ids = tool_current_token_ids

        except Exception:
            logger.warning(
                "Tool streaming extraction failed; passing content through",
                exc_info=True,
            )
            result.accumulated_content = content_text
            result.delta_content = content_delta

    def _process_tool_final(self, result: ParseResult, content_for_tools: str) -> None:
        """Run tool parser batch extraction on the final content. Mutates *result*.

        Args:
            result: Mutable :class:`ParseResult` updated with batch-extracted
                tool calls and any catch-up deltas.
            content_for_tools: Final visible content fed to the tool parser.
        """
        tool_request = self._get_tool_request()
        tools_enabled = self._is_tools_enabled()
        previous_visible_content = self._accumulated_content

        try:
            extracted = self.tool_parser.extract_tool_calls(content_for_tools, tool_request)
            if extracted.tools_called and extracted.tool_calls:
                result.tool_calls = extracted.tool_calls
                result.delta_tool_calls = self._build_missing_final_tool_deltas(extracted.tool_calls)
                self._merge_streamed_tool_call_state(result.delta_tool_calls)
                self.phase = ParsePhase.TOOL_CALL
                result.accumulated_content = previous_visible_content
                result.delta_content = ""
            elif self.phase == ParsePhase.TOOL_CALL:
                result.accumulated_content = previous_visible_content
                result.delta_content = ""
            elif tools_enabled:
                visible = extracted.content if extracted.content is not None else content_for_tools
                result.accumulated_content = visible
                result.delta_content = self._compute_visible_content_delta(
                    visible,
                    previous_visible_content,
                )
        except Exception:
            logger.warning(
                "Tool batch extraction failed; passing content through",
                exc_info=True,
            )

    def _tokenize_for_tool_view(self, text: str) -> list[int]:
        """Tokenize text for the tool parser's independent view.

        Tool parsers maintain their own token-id view starting after the
        reasoning boundary. This helper encodes ``text`` with the parser's
        tokenizer (when available) and gracefully handles tokenizers whose
        ``encode`` method does not accept ``add_special_tokens``.

        Args:
            text: Visible content to tokenize.

        Returns:
            List of integer token IDs, or an empty list when no tokenizer
            is available or encoding fails.
        """
        if not text:
            return []

        tokenizer = getattr(self.tool_parser, "model_tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return []

        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            try:
                ids = tokenizer.encode(text)
            except Exception:
                return []
        except Exception:
            return []

        try:
            return [int(tid) for tid in ids]
        except Exception:
            return []
