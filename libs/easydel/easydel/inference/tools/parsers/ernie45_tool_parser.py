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

"""Tool parser implementation for ERNIE 4.5 models.

This module provides a tool parser specifically designed for ERNIE 4.5 (Baidu) model outputs.
ERNIE 4.5 is a thinking model that uses a specific format with </think> tags followed by
<tool_call> tags containing JSON function call data.

The parser handles the complex token structure including thinking sections, response tags,
and tool call delimiters, with special handling for newline tokens and streaming state.

Example format:
    reasoning content</think>

    <tool_call>
    {"name": "function_name", "arguments": {"param": "value"}}
    </tool_call>
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import partial_json_parser  # pyright: ignore[reportMissingTypeStubs]
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow  # pyright: ignore[reportMissingTypeStubs]
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
from ..utils import consume_space, partial_json_loads

logger = get_logger(__name__)

# Maximum lookahead past an encountered ``</tool_call>`` when an unclosed JSON
# string may contain a literal ``</tool_call>`` before the true closing tag.
_MAX_CROSS_TAG_LOOKAHEAD = 512


def _has_open_marker_or_trailing_prefix(text: str, start_token: str, end_token: str) -> bool:
    """Return True if *text* has an unclosed block or ends with a partial start marker."""
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


@ToolParserManager.register_module("ernie45")  # pyright: ignore[reportUntypedClassDecorator]
class Ernie45ToolParser(ToolParser):
    """Tool parser for ERNIE 4.5 (Baidu) model outputs.

    This parser handles the extraction of function/tool calls from ERNIE 4.5 model
    outputs. ERNIE 4.5 is a thinking model that outputs reasoning in a thinking
    section followed by tool calls in XML-style tags.

    The parser handles several special tokens and structures:
    - </think> tag marking the end of the thinking section
    - <response>...</response> tags for regular responses
    - <tool_call>...</tool_call> tags for function calls
    - Special newline token handling (<0x0A>)

    Attributes:
        current_tool_name_sent: Flag indicating if the current tool name has been sent.
        prev_tool_call_arr: List of previously parsed tool call dictionaries.
        current_tool_id: Index of the current tool being processed (-1 means none).
        streamed_args_for_tool: List of argument strings streamed for each tool.
        think_end_token: Token marking end of thinking section.
        response_start_token: Token marking start of response.
        response_end_token: Token marking end of response.
        tool_call_start_token: Token marking start of tool call.
        tool_call_end_token: Token marking end of tool call.
        tool_calls_start_token: Alias for tool_call_start_token.
        newline_token: Special newline token representation.
        tool_call_regex: Compiled regex for extracting tool call JSON.
        think_end_token_id: Token ID for think_end_token.
        response_start_token_id: Token ID for response_start_token.
        response_end_token_id: Token ID for response_end_token.
        tool_call_start_token_id: Token ID for tool_call_start_token.
        tool_call_end_token_id: Token ID for tool_call_end_token.
        newline_token_id: Token ID for newline_token.
        parser_token_ids: List of special parser token IDs.

    Raises:
        ValueError: If the tokenizer is not provided during initialization.

    Example:
        >>> parser = Ernie45ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
        >>> if result.tools_called:
        ...     for tool_call in result.tool_calls:
        ...         print(f"Function: {tool_call.function.name}")
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the ERNIE 4.5 tool parser.

        Sets up token mappings and regex patterns for parsing ERNIE 4.5 format
        tool calls. Validates that a tokenizer is provided and maps special
        tokens to their IDs.

        Args:
            tokenizer: A HuggingFace tokenizer instance used for token processing.
                This tokenizer should be compatible with the ERNIE 4.5 model and
                must contain the special tokens used by this parser.

        Raises:
            ValueError: If the tokenizer is not provided (is None or falsy).

        Note:
            The parser expects the tokenizer vocabulary to contain special tokens
            like </think>, <response>, </response>, <tool_call>, </tool_call>,
            and <0x0A> (newline). Missing tokens will result in None values for
            the corresponding token IDs.
        """
        super().__init__(tokenizer)

        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []

        self.think_end_token = "</think>"
        self.response_start_token: str = "<response>"
        self.response_end_token: str = "</response>"
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_calls_start_token = self.tool_call_start_token
        self.tool_calls_end_token = self.tool_call_end_token
        self.newline_token: str = "<0x0A>"

        self.tool_call_regex = re.compile(
            r"<tool_call>\s*(?P<json>\{(?:(?!</tool_call>).)*?\})\s*</tool_call>",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.response_start_token_id = self.vocab.get(self.response_start_token)
        self.response_end_token_id = self.vocab.get(self.response_end_token)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.newline_token_id = self.vocab.get(self.newline_token)
        self.parser_token_ids = [
            self.think_end_token_id,
            self.response_start_token_id,
            self.response_end_token_id,
        ]

        self._buffer = ""
        self._streaming_initialized: bool = False

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether an open ``<tool_call>`` block or partial start marker is buffered."""
        del delta_text
        if self._streaming_initialized:
            return bool(self._buffer)
        return _has_open_marker_or_trailing_prefix(
            current_text,
            self.tool_call_start_token,
            self.tool_call_end_token,
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Parses the model output to find and extract function calls in ERNIE 4.5 format.
        Searches for <tool_call>...</tool_call> tags containing JSON function data.

        Args:
            model_output: The complete text output from the model to parse.
            request: The chat completion request that triggered this response.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called: True if valid tool calls were found, False otherwise.
                - tool_calls: List of ToolCall objects representing parsed function calls.
                - content: Any text content before the tool call section, or the full
                    output if no valid tool calls were found.

        Note:
            If no <tool_call> start token is found, returns immediately with
            tools_called=False and the full output as content.
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            dec = json.JSONDecoder()
            tool_calls: list[ToolCall] = []
            pos = 0
            while pos < len(model_output):
                start = model_output.find(self.tool_call_start_token, pos)
                if start == -1:
                    break
                search_from = start + len(self.tool_call_start_token)
                end_tag = model_output.find(self.tool_call_end_token, search_from)
                if end_tag == -1:
                    break

                payload_start = consume_space(search_from, model_output)
                next_pos = end_tag + len(self.tool_call_end_token)
                tool_call_dict: object = None
                if payload_start < len(model_output) and model_output[payload_start] == "{":
                    try:
                        decoded, json_end = dec.raw_decode(model_output, payload_start)
                        after_json = consume_space(json_end, model_output)
                        if model_output.startswith(self.tool_call_end_token, after_json):
                            tool_call_dict = decoded
                            next_pos = after_json + len(self.tool_call_end_token)
                    except (json.JSONDecodeError, ValueError, RecursionError):
                        tool_call_dict = None

                pos = next_pos
                if not isinstance(tool_call_dict, dict):
                    continue
                name = tool_call_dict.get("name", "")
                if not isinstance(name, str) or not name:
                    continue
                raw_args = tool_call_dict.get("arguments", {})
                args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(name=name, arguments=args_str),
                    )
                )

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            content = model_output[: model_output.find(self.tool_calls_start_token)].rstrip("\n")
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def _clean_plain_content(
        self,
        text: str,
        previous_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> str:
        """Apply ERNIE 4.5 response-tag and leading-newline stripping to non-tool content."""
        if self.current_tool_id > 0:
            return ""
        cur_text = text
        if (
            self.current_tool_id == -1
            and self.newline_token_id is not None
            and all(token_id == self.newline_token_id for token_id in previous_token_ids)
        ):
            cur_text = cur_text.strip("\n")

        content = cur_text
        if self.response_start_token_id is not None and self.response_start_token_id in delta_token_ids:
            content = content.lstrip("\n")
            response_start_idx = content.find(self.response_start_token)
            if response_start_idx != -1:
                content = content[response_start_idx + len(self.response_start_token) :]
            response_end_idx = content.rfind(self.response_end_token)
            if response_end_idx != -1:
                content = content[:response_end_idx]
        elif self.response_end_token_id is not None and self.response_end_token_id in delta_token_ids:
            response_end_idx = content.rfind(self.response_end_token)
            if response_end_idx != -1:
                content = content[:response_end_idx]

        if (
            len(previous_token_ids) > 0
            and previous_token_ids[-1] in self.parser_token_ids
            and self.newline_token_id is not None
        ) and (len(delta_token_ids) > 0 and delta_token_ids[0] == self.newline_token_id):
            content = content.lstrip("\n")

        return content

    def flush_streaming_content(self) -> str:
        """Release text withheld as a possible split ``<tool_call>`` prefix at EOS."""
        buffered, self._buffer = self._buffer, ""
        if not buffered or self.current_tool_id > 0:
            return ""
        start_idx = buffered.find(self.tool_call_start_token)
        if start_idx == -1:
            return buffered
        # If we were speculatively waiting past a ``</tool_call>`` for a valid JSON close
        # that never arrived before EOS, discard the malformed block and release trailing prose.
        end_idx = buffered.find(self.tool_call_end_token, start_idx + len(self.tool_call_start_token))
        if end_idx != -1:
            trailing = buffered[end_idx + len(self.tool_call_end_token) :]
            if trailing and self.tool_call_start_token not in trailing:
                return trailing
        return ""

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
        tool call information. Handles the complex ERNIE 4.5 format including
        thinking sections, response tags, and tool call delimiters.

        This method buffers content between tool call start and end tokens,
        then extracts and emits complete tool calls when end tokens are found.

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
                - Tool call information when complete tool call(s) are parsed.
                - None if buffering content within a tool call.
        """
        if not previous_text:
            self._buffer = ""
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = []

        self._streaming_initialized = True
        self._buffer += delta_text
        cur_text = self._buffer
        content_parts: list[str] = []
        emitted_tool_calls: list[DeltaToolCall] = []

        while cur_text:
            start_idx = cur_text.find(self.tool_call_start_token)
            if start_idx == -1:
                withheld = ""
                max_check = min(len(self.tool_call_start_token) - 1, len(cur_text))
                for size in range(max_check, 0, -1):
                    if cur_text.endswith(self.tool_call_start_token[:size]):
                        withheld = cur_text[-size:]
                        break
                self._buffer = withheld
                plain = cur_text[: len(cur_text) - len(withheld)]
                if plain:
                    cleaned = self._clean_plain_content(plain, previous_token_ids, delta_token_ids)
                    if cleaned:
                        content_parts.append(cleaned)
                break

            logger.debug("cur_text = %s", cur_text)
            search_from = start_idx + len(self.tool_call_start_token)
            payload_start = consume_space(search_from, cur_text)
            block_end: int | None = None
            waiting_for_close = False

            if payload_start < len(cur_text) and cur_text[payload_start] == "{":
                try:
                    _decoded, json_end = json.JSONDecoder().raw_decode(cur_text, payload_start)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    end_idx = cur_text.find(self.tool_call_end_token, search_from)
                    if end_idx == -1:
                        waiting_for_close = True
                    else:
                        candidate_end = end_idx + len(self.tool_call_end_token)
                        after_end_text = cur_text[candidate_end:]
                        if (
                            len(after_end_text) <= _MAX_CROSS_TAG_LOOKAHEAD
                            and self.tool_call_end_token not in after_end_text
                        ):
                            try:
                                partial_json_loads(cur_text[payload_start:], Allow.ALL)
                                waiting_for_close = True
                            except (
                                partial_json_parser.core.exceptions.MalformedJSON,
                                ValueError,
                                AssertionError,
                                RecursionError,
                            ):
                                block_end = candidate_end
                        else:
                            block_end = candidate_end
                else:
                    after_json = consume_space(json_end, cur_text)
                    if cur_text.startswith(self.tool_call_end_token, after_json):
                        block_end = after_json + len(self.tool_call_end_token)
                    elif after_json == len(cur_text) or self.tool_call_end_token.startswith(cur_text[after_json:]):
                        waiting_for_close = True
                    else:
                        end_idx = cur_text.find(self.tool_call_end_token, search_from)
                        if end_idx == -1:
                            waiting_for_close = True
                        else:
                            block_end = end_idx + len(self.tool_call_end_token)
            else:
                end_idx = cur_text.find(self.tool_call_end_token, search_from)
                if end_idx == -1:
                    waiting_for_close = True
                else:
                    block_end = end_idx + len(self.tool_call_end_token)

            if waiting_for_close or block_end is None:
                if start_idx > 0 and self.current_tool_id <= 0:
                    pre = cur_text[:start_idx].rstrip("\n")
                    if pre:
                        content_parts.append(pre)
                self._buffer = cur_text[start_idx:]
                break

            block_text = cur_text[start_idx:block_end]
            extracted_tool_calls = self.extract_tool_calls(block_text, request)

            if len(extracted_tool_calls.tool_calls) == 0:
                logger.warning("Failed to extract any tool calls from block; advancing buffer.")
                if start_idx > 0 and self.current_tool_id <= 0:
                    content_parts.append(cur_text[:start_idx])
                cur_text = cur_text[block_end:]
                self._buffer = cur_text
                continue

            if start_idx > 0 and self.current_tool_id <= 0:
                pre = cur_text[:start_idx].rstrip("\n")
                if pre:
                    content_parts.append(pre)

            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []

            for tool_call in extracted_tool_calls.tool_calls:
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                try:
                    parsed_args = json.loads(tool_call.function.arguments)
                except Exception:
                    parsed_args = tool_call.function.arguments

                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": tool_call.function.name,
                    "arguments": parsed_args,
                }
                self.streamed_args_for_tool[self.current_tool_id] = tool_call.function.arguments

                emitted_tool_calls.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=DeltaFunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                )
                self.current_tool_id += 1

            cur_text = cur_text[block_end:]
            self._buffer = cur_text

        content_out = "".join(content_parts)
        if emitted_tool_calls:
            return DeltaMessage(content=content_out if content_out else None, tool_calls=emitted_tool_calls)
        return DeltaMessage(content=content_out if content_out else None)
