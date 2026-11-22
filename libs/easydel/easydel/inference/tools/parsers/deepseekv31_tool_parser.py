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

"""DeepSeek V3.1 tool parser module for parsing tool calls from DeepSeek V3.1 model outputs.

This module provides the DeepSeekV31ToolParser class which handles the updated
tool call format used by DeepSeek V3.1 models. The format is similar to V3 but
uses a simplified structure without the tool type field.

Example tool call format:
    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Beijing"}<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import uuid4

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


@ToolParserManager.register_module("deepseek_v31")  # pyright: ignore[reportUntypedClassDecorator]
class DeepSeekV31ToolParser(ToolParser):
    """Tool parser for DeepSeek V3.1 models.

    This parser handles the updated tool call format used by DeepSeek V3.1 models.
    Unlike V3, this format directly uses function name and arguments without
    an explicit tool type field.

    The format structure:
    - Tool calls wrapped in <｜tool▁calls▁begin｜> and <｜tool▁calls▁end｜>
    - Individual calls wrapped in <｜tool▁call▁begin｜> and <｜tool▁call▁end｜>
    - Function name and arguments separated by <｜tool▁sep｜>

    Attributes:
        current_tool_name_sent (bool): Tracks if tool name has been sent in streaming.
        prev_tool_call_arr (list[dict]): Previous tool calls for comparison in streaming.
        current_tool_id (int): Index of current tool being processed.
        streamed_args_for_tool (list[str]): Arguments streamed so far for each tool.
        tool_calls_start_token (str): Token marking start of tool calls section.
        tool_calls_end_token (str): Token marking end of tool calls section.
        tool_call_start_token (str): Token marking start of individual tool call.
        tool_call_end_token (str): Token marking end of individual tool call.
        tool_call_regex (re.Pattern): Regex pattern for parsing complete tool calls.
        stream_tool_call_portion_regex (re.Pattern): Regex for parsing streaming portions.
        stream_tool_call_name_regex (re.Pattern): Regex for extracting tool names.
        tool_calls_start_token_id (int | None): Token ID for tool calls start marker.
        tool_calls_end_token_id (int | None): Token ID for tool calls end marker.
        tool_call_start_token_id (int | None): Token ID for individual call start.
        tool_call_end_token_id (int | None): Token ID for individual call end.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v3.1")
        >>> parser = DeepSeekV31ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the DeepSeek V3.1 tool parser.

        Sets up the token markers, regex patterns, and token IDs required for
        parsing DeepSeek V3.1 tool call format.

        Args:
            tokenizer: The tokenizer associated with the DeepSeek V3.1 model.
                Must contain the special tool call tokens in its vocabulary.

        Raises:
            ValueError: If the tokenizer is not provided.
            RuntimeError: If the tool call start/end tokens cannot be found
                in the tokenizer vocabulary.
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

        self.tool_calls_start_token: str = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end_token: str = "<｜tool▁calls▁end｜>"

        self.tool_call_start_token: str = "<｜tool▁call▁begin｜>"
        self.tool_call_end_token: str = "<｜tool▁call▁end｜>"

        self.tool_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*)",
            re.DOTALL,
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<function_name>.*?)<｜tool▁sep｜>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None:
            raise RuntimeError("DeepSeek-V3.1 Tool parser could not locate tool call start/end tokens in the tokenizer!")
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            logger.debug(
                "DeepSeek-V3.1 Tool parser could not locate per-call start/end token IDs; using string markers."
            )

    def get_streaming_buffer_marker_pairs(self) -> Sequence[tuple[str, str | None]]:
        """Scope protocol buffering strictly to outer ``<｜tool▁calls▁begin｜>`` sections."""
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
        """Extract tool calls from complete model output.

        Parses the DeepSeek V3.1 format to extract function names and arguments
        from the model's response. Uses regex pattern matching to identify
        tool call boundaries and extract the function details.

        Args:
            model_output: Complete text output from the model containing
                potential tool calls in DeepSeek V3.1 format.
            request: Original chat completion request with tool definitions.
                Used for context but not directly accessed in parsing.

        Returns:
            ExtractedToolCallInformation: Contains the following fields:
                - tools_called (bool): True if any tools were invoked.
                - tool_calls (list[ToolCall]): List of parsed tool calls with
                  function names and argument strings.
                - content (str | None): Text content before tool calls, or
                  None if no content precedes the tool calls.

        Example:
            >>> output = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"NYC\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
            >>> result = parser.extract_tool_calls(output, request)
            >>> result.tools_called
            True
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            sep_token = "<｜tool▁sep｜>"
            start_len = len(self.tool_call_start_token)
            sep_len = len(sep_token)
            end_len = len(self.tool_call_end_token)
            tool_calls: list[ToolCall] = []
            pos = 0
            n = len(model_output)
            while pos < n:
                call_start = model_output.find(self.tool_call_start_token, pos)
                if call_start == -1:
                    break
                body_start = call_start + start_len
                sep_idx = model_output.find(sep_token, body_start)
                if sep_idx == -1:
                    break
                args_start = sep_idx + sep_len
                call_end = model_output.find(self.tool_call_end_token, args_start)
                if call_end == -1:
                    break
                function_name = model_output[body_start:sep_idx]
                function_args = model_output[args_start:call_end]
                tool_calls.append(
                    ToolCall(type="function", function=FunctionCall(name=function_name, arguments=function_args))
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
        """Extract tool calls from streaming model output.

        Handles incremental parsing of DeepSeek V3.1 tool call format during
        streaming generation. Tracks state across chunks to identify tool
        boundaries and progressively emit argument content.

        The method handles several streaming scenarios:
        - Starting a new tool call (detecting start token)
        - Updating an existing tool call (accumulating arguments)
        - Closing a tool call (detecting end token)
        - Passing through regular content tokens

        Args:
            previous_text: Text generated up to the previous chunk.
            current_text: All text generated so far including current chunk.
            delta_text: New text added in this chunk only.
            previous_token_ids: Sequence of token IDs up to previous chunk.
            current_token_ids: Sequence of all token IDs generated so far.
            delta_token_ids: Sequence of new token IDs in this chunk.
            request: Original chat completion request with tool definitions.

        Returns:
            DeltaMessage | None: A delta message containing:
                - Tool call information (name, arguments) for streaming updates
                - Content text if not in a tool call
                - None if more data is needed before emitting

        Note:
            This method maintains internal state across calls including
            current_tool_id, current_tool_name_sent, prev_tool_call_arr,
            and streamed_args_for_tool.
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
            sep_token = "<｜tool▁sep｜>"
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

                sep_idx = body.find(sep_token)
                if sep_idx == -1:
                    continue
                func_name = body[:sep_idx]
                raw_args = body[sep_idx + len(sep_token) :]
                if not func_name:
                    continue

                if is_closed:
                    target_args = raw_args
                else:
                    cleaned_args = _strip_trailing_partial_marker(raw_args, self.tool_call_end_token)
                    target_args = stable_json_stream_prefix(cleaned_args)

                self.prev_tool_call_arr[idx] = {"name": func_name, "arguments": target_args}

                first = not self._tool_name_sent[idx]
                if first:
                    self._tool_call_ids[idx] = f"chatcmpl-tool-{uuid4()}"
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
