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

"""Tool parser implementation for Kimi K2 models.

This module provides a tool parser specifically designed for Kimi K2 model outputs.
Kimi K2 uses a hierarchical token-based structure with section and individual call
delimiters, and encodes function identity in a namespace.function:id format.

The parser handles nested token structures with distinct section-level and
call-level markers, making it suitable for complex multi-tool scenarios.

Example format:
    <|tool_calls_section_begin|>
    <|tool_call_begin|>namespace.function_name:123<|tool_call_argument_begin|>{"arg": "value"}<|tool_call_end|>
    <|tool_calls_section_end|>
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from eformer.loggings import get_logger
from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ..utils import stable_json_stream_prefix

logger = get_logger(__name__)


_KIMI_CALL_HEADER_RE = re.compile(r"\s*(?P<tool_call_id>[^<>\s]+:\d+)\s*<\|tool_call_argument_begin\|>")


def _strip_trailing_partial_marker(text: str, marker: str) -> str:
    """Withhold any trailing non-empty proper prefix of *marker* from *text*."""
    max_len = min(len(marker) - 1, len(text))
    for k in range(max_len, 0, -1):
        if text.endswith(marker[:k]):
            return text[:-k]
    return text


def _has_open_marker_or_trailing_prefix(text: str, start_token: str, end_token: str) -> bool:
    """Return True if *text* has an unclosed outer section or ends with a partial start marker."""
    pos = 0
    n = len(text)
    start_len = len(start_token)
    end_len = len(end_token)
    while pos < n:
        start_idx = text.find(start_token, pos)
        if start_idx == -1:
            return ToolParser._has_trailing_marker_prefix(text[pos:], (start_token,))
        end_idx = text.find(end_token, start_idx + start_len)
        if end_idx == -1:
            return True
        pos = end_idx + end_len
    return False


@ToolParserManager.register_module(["kimi_k2"])  # pyright: ignore[reportUntypedClassDecorator]
class KimiK2ToolParser(ToolParser):
    """Tool parser for Kimi K2 model outputs.

    This parser handles the extraction of function/tool calls from Kimi K2 model
    outputs. Kimi K2 uses a hierarchical token structure with section-level and
    call-level delimiters, and encodes tool identity in namespace.function:id format.

    The parser handles two levels of nesting:
    - Section level: <|tool_calls_section_begin|> ... <|tool_calls_section_end|>
    - Call level: <|tool_call_begin|> ... <|tool_call_end|>

    Tool IDs follow the format: namespace.function_name:numeric_id

    Attributes:
        current_tool_name_sent: Flag indicating if the current tool name has been sent.
        prev_tool_call_arr: List of previously parsed tool call dictionaries.
        current_tool_id: Index of the current tool being processed (-1 means none).
        streamed_args_for_tool: List of argument strings streamed for each tool.
        tool_calls_start_token: Section start marker.
        tool_calls_end_token: Section end marker.
        tool_call_start_token: Individual call start marker.
        tool_call_end_token: Individual call end marker.
        tool_call_regex: Regex for extracting complete tool calls.
        stream_tool_call_portion_regex: Regex for partial tool call with arguments.
        stream_tool_call_name_regex: Regex for tool name/id extraction.
        tool_calls_start_token_id: Token ID for section start.
        tool_calls_end_token_id: Token ID for section end.
        tool_call_start_token_id: Token ID for call start.
        tool_call_end_token_id: Token ID for call end.

    Raises:
        ValueError: If the tokenizer is not provided.
        RuntimeError: If required tool call tokens are not in the tokenizer.

    Example:
        >>> parser = KimiK2ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
        >>> if result.tools_called:
        ...     for tool_call in result.tool_calls:
        ...         print(f"Function: {tool_call.function.name}")
        ...         print(f"ID: {tool_call.id}")
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Kimi K2 tool parser.

        Sets up token mappings and regex patterns for parsing Kimi K2 format
        tool calls. Validates that required section and call delimiter tokens
        exist in the tokenizer vocabulary.

        Args:
            tokenizer: A HuggingFace tokenizer instance used for token processing.
                This tokenizer should be compatible with the Kimi K2 model and
                must contain the section and call delimiter tokens.

        Raises:
            ValueError: If the tokenizer is not provided (is None or falsy).
            RuntimeError: If the tokenizer doesn't contain required tool call
                section start/end tokens.
        """
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] | None = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self._tool_name_sent: list[bool] = []
        self._tool_call_ids: list[str] = []
        self._emitted_content_pos: int | None = None
        self._pending_tail_start: int | None = None
        self._last_current_text: str = ""
        self._streaming_initialized: bool = False
        self._in_open_section: bool = False
        self._scan_pos: int = 0
        self._closed_span_count: int = 0

        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<>\s]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"^\s*(?P<tool_call_id>[^<>\s]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)",
            re.DOTALL,
        )

        self.stream_tool_call_name_regex = re.compile(r"^\s*(?P<tool_call_id>[^<>\s]+:\d+)\s+")

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None:
            raise RuntimeError("Kimi-K2 Tool parser could not locate tool call start/end tokens in the tokenizer!")
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            logger.debug(
                "Kimi-K2 Tool parser could not locate per-call start/end token IDs; using string markers."
            )

    @staticmethod
    def _parse_function_name(function_id: str) -> str:
        """Extract the function name from a Kimi K2 ``[namespace.]name[:digits]`` identifier."""
        base = re.sub(r":\d+$", "", function_id)
        return base.split(".", 1)[1] if "." in base else base

    def get_streaming_buffer_marker_pairs(self) -> Sequence[tuple[str, str | None]]:
        """Scope protocol buffering strictly to outer ``<|tool_calls_section_begin|>`` sections."""
        return ((self.tool_calls_start_token, self.tool_calls_end_token),)

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether an outer tool-calls section or partial start marker is buffered."""
        del delta_text
        if self._streaming_initialized:
            return self._in_open_section or self._pending_tail_start is not None
        return _has_open_marker_or_trailing_prefix(
            current_text,
            self.tool_calls_start_token,
            self.tool_calls_end_token,
        )

    def _reset_streaming_state(self, emitted_pos: int = 0) -> None:
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self._tool_name_sent = []
        self._tool_call_ids = []
        self._emitted_content_pos = emitted_pos
        self._pending_tail_start = None
        self._last_current_text = ""
        self._streaming_initialized = False
        self._in_open_section = False
        self._scan_pos = 0
        self._closed_span_count = 0

    def flush_streaming_content(self) -> str | None:
        """Flush any trailing partial marker prefix withheld at end-of-stream."""
        flushed: str | None = None
        if self._pending_tail_start is not None and self._pending_tail_start < len(self._last_current_text):
            flushed = self._last_current_text[self._pending_tail_start :] or None
            self._emitted_content_pos = len(self._last_current_text)
            self._pending_tail_start = None
        self._last_current_text = ""
        return flushed

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Parses the model output to find and extract function calls in Kimi K2 format.
        Extracts tool ID (in namespace.function:id format) and JSON arguments from
        each tool call block.

        Args:
            model_output: The complete text output from the model to parse.
            request: The chat completion request that triggered this response.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called: True if valid tool calls were found, False otherwise.
                - tool_calls: List of ToolCall objects with id, function name, and arguments.
                - content: Any text content before the tool calls section, or the full
                    output if no valid tool calls were found.

        Note:
            The function name is extracted from the tool_call_id by parsing the
            namespace.function:id format and extracting the function portion.
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        else:
            try:
                tool_calls: list[ToolCall] = []
                start_len = len(self.tool_call_start_token)
                end_len = len(self.tool_call_end_token)
                pos = 0
                n = len(model_output)
                while pos < n:
                    call_start = model_output.find(self.tool_call_start_token, pos)
                    if call_start == -1:
                        break
                    body_start = call_start + start_len
                    header_match = _KIMI_CALL_HEADER_RE.match(model_output, body_start)
                    if header_match is None:
                        pos = body_start
                        continue
                    args_start = header_match.end()
                    call_end = model_output.find(self.tool_call_end_token, args_start)
                    if call_end == -1:
                        break
                    function_id = header_match.group("tool_call_id")
                    function_args = model_output[args_start:call_end].strip()
                    function_name = self._parse_function_name(function_id)
                    tool_calls.append(
                        ToolCall(
                            id=function_id,
                            type="function",
                            function=FunctionCall(name=function_name, arguments=function_args),
                        )
                    )
                    pos = call_end + end_len

                if not tool_calls:
                    return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls incrementally during streaming generation.

        Processes tokens as they are generated to progressively extract and emit
        tool call information. Handles the hierarchical Kimi K2 format with
        section and call level delimiters.

        The method tracks the number of tool call start/end tokens to determine
        when new tools begin and when arguments should be streamed.

        Args:
            previous_text: The accumulated text from all previous tokens.
            current_text: The complete text including the current delta.
            delta_text: The new text added in this streaming step.
            previous_token_ids: Token IDs from all previous generation steps.
            current_token_ids: All token IDs including the current step.
            delta_token_ids: The new token IDs added in this step.
            request: The chat completion request for context.

        Returns:
            DeltaMessage or None: A delta message containing either:
                - Content text if outside tool call section.
                - Tool call name/id when a new tool starts.
                - Argument deltas as JSON content is streamed.
                - None if waiting for more tokens or parsing state is ambiguous.

        Note:
            The streaming logic handles several state transitions:
            - Detecting entry into tool calls section
            - Recognizing new tool calls by counting start/end tokens
            - Incrementally streaming arguments as they appear
            - Properly closing tool calls when end tokens are seen
        """
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        if not previous_text:
            self._reset_streaming_state(0)
        elif self._emitted_content_pos is None:
            self._reset_streaming_state(len(previous_text))
        elif self._scan_pos > len(current_text):
            self._reset_streaming_state(0)

        self._streaming_initialized = True
        self._last_current_text = current_text
        self._pending_tail_start = None
        self._in_open_section = False

        try:
            pos = self._scan_pos
            next_scan_pos = pos
            next_closed_span_count = self._closed_span_count
            spans: list[tuple[str, bool]] = []
            visible_parts: list[tuple[int, int]] = []
            while pos < len(current_text):
                sec_start = current_text.find(self.tool_calls_start_token, pos)
                if sec_start == -1:
                    tail = current_text[pos:]
                    safe_tail = _strip_trailing_partial_marker(tail, self.tool_calls_start_token)
                    safe_end = pos + len(safe_tail)
                    if safe_tail:
                        visible_parts.append((pos, safe_end))
                    if len(safe_tail) < len(tail):
                        self._pending_tail_start = safe_end
                    next_scan_pos = safe_end
                    break
                if sec_start > pos:
                    visible_parts.append((pos, sec_start))
                sec_body_start = sec_start + len(self.tool_calls_start_token)
                sec_end = current_text.find(self.tool_calls_end_token, sec_body_start)
                scan_limit = sec_end if sec_end != -1 else len(current_text)

                call_pos = sec_body_start
                sec_span_count = 0
                while call_pos < scan_limit:
                    call_start = current_text.find(self.tool_call_start_token, call_pos, scan_limit)
                    if call_start == -1:
                        break
                    body_start = call_start + len(self.tool_call_start_token)
                    call_end = current_text.find(self.tool_call_end_token, body_start, scan_limit)
                    if call_end == -1:
                        spans.append((current_text[body_start:scan_limit], False))
                        sec_span_count += 1
                        break
                    spans.append((current_text[body_start:call_end], True))
                    sec_span_count += 1
                    call_pos = call_end + len(self.tool_call_end_token)

                if sec_end == -1:
                    self._in_open_section = True
                    next_scan_pos = sec_start
                    break
                pos = sec_end + len(self.tool_calls_end_token)
                next_scan_pos = pos
                next_closed_span_count += sec_span_count

            content_chunks: list[str] = []
            emitted_pos = self._emitted_content_pos or 0
            for seg_start, seg_end in visible_parts:
                if seg_end > emitted_pos:
                    content_chunks.append(current_text[max(seg_start, emitted_pos) : seg_end])
                    emitted_pos = seg_end
            self._emitted_content_pos = emitted_pos
            content_out = "".join(content_chunks)

            if self.prev_tool_call_arr is None:
                self.prev_tool_call_arr = []

            tool_deltas: list[DeltaToolCall] = []
            base_idx = self._closed_span_count
            for offset, (body, is_closed) in enumerate(spans):
                idx = base_idx + offset
                while len(self.prev_tool_call_arr) <= idx:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= idx:
                    self.streamed_args_for_tool.append("")
                while len(self._tool_name_sent) <= idx:
                    self._tool_name_sent.append(False)
                while len(self._tool_call_ids) <= idx:
                    self._tool_call_ids.append("")

                if idx > self.current_tool_id:
                    self.current_tool_id = idx
                    self.current_tool_name_sent = self._tool_name_sent[idx]

                tool_id: str | None = None
                func_name: str | None = None
                target_args: str | None = None

                portion_match = self.stream_tool_call_portion_regex.match(body)
                if portion_match:
                    tool_id = portion_match.group("tool_call_id")
                    func_name = self._parse_function_name(tool_id)
                    raw_args = portion_match.group("function_arguments")
                    if is_closed:
                        target_args = raw_args.rstrip()
                    else:
                        cleaned_args = _strip_trailing_partial_marker(raw_args, self.tool_call_end_token).rstrip()
                        target_args = stable_json_stream_prefix(cleaned_args)
                elif not is_closed:
                    name_match = self.stream_tool_call_name_regex.match(body)
                    if not name_match:
                        continue
                    tool_id = name_match.group("tool_call_id")
                    func_name = self._parse_function_name(tool_id)
                    target_args = ""
                else:
                    continue

                if not func_name or not tool_id:
                    continue

                self.prev_tool_call_arr[idx] = {"id": tool_id, "name": func_name, "arguments": target_args}

                first = not self._tool_name_sent[idx]
                if first:
                    self._tool_call_ids[idx] = tool_id
                    self._tool_name_sent[idx] = True
                    if idx == self.current_tool_id:
                        self.current_tool_name_sent = True

                streamed = self.streamed_args_for_tool[idx]
                diff = ""
                if target_args is not None and target_args.startswith(streamed):
                    diff = target_args[len(streamed) :]
                if diff:
                    self.streamed_args_for_tool[idx] = streamed + diff

                if first or diff:
                    tool_deltas.append(
                        DeltaToolCall(
                            index=idx,
                            id=self._tool_call_ids[idx] if first else None,
                            type="function" if first else None,
                            function=DeltaFunctionCall(
                                name=func_name if first else None,
                                arguments=diff if diff else None,
                            ).model_dump(exclude_none=True),
                        )
                    )

            self._scan_pos = next_scan_pos
            self._closed_span_count = next_closed_span_count

            if content_out and tool_deltas:
                return DeltaMessage(content=content_out, tool_calls=tool_deltas)
            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            if content_out:
                return DeltaMessage(content=content_out)
            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
