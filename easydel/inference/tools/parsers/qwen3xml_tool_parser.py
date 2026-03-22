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

"""Qwen3 XML tool parser module.

This module provides the Qwen3XMLToolParser class as an alias for the
Qwen3CoderToolParser. It handles the same XML-style tool call format
used by Qwen3 models in non-coder variants.

The parser is registered under the 'qwen3_xml' module name and inherits
all functionality from Qwen3CoderToolParser.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParserManager
from .qwen3coder_tool_parser import Qwen3CoderToolParser


@ToolParserManager.register_module(["qwen3_xml"])  # pyright: ignore[reportUntypedClassDecorator]
class Qwen3XMLToolParser(Qwen3CoderToolParser):
    """Dedicated parser for Qwen3 XML outputs.

    Unlike `qwen3_coder`, this parser accepts both wrapped and bare XML:

    `<tool_call><function=lookup>...</function></tool_call>`
    `<function=lookup>...</function>`
    `<function name="lookup">...</function>`
    """

    def __init__(self, tokenizer):
        """Initialize the Qwen3 XML tool parser.

        Extends the parent ``Qwen3CoderToolParser`` with additional regex
        patterns that support bare ``<function=...>`` and
        ``<function name="...">`` syntax without requiring ``<tool_call>``
        wrappers.

        Args:
            tokenizer: The tokenizer associated with the Qwen3 model.
        """
        super().__init__(tokenizer)
        self.function_start_regex = re.compile(
            r"<function(?:=([^>\s]+)|\s+name=\"([^\"]+)\"|\s+name='([^']+)')\s*>",
            re.DOTALL,
        )
        self.parameter_regex = re.compile(
            r"<parameter(?:=([^>\s]+)|\s+name=\"([^\"]+)\"|\s+name='([^']+)')\s*>(.*?)(?:</parameter>|(?=<parameter(?:=|\s+name=))|(?=</function>)|$)",
            re.DOTALL,
        )

    @staticmethod
    def _first_nonempty(*values: str | None) -> str | None:
        """Return the first non-empty string from the arguments, or None.

        Args:
            *values: String values to check (may be None or empty).

        Returns:
            The first truthy string, or None if all are empty/None.
        """
        for value in values:
            if value:
                return value
        return None

    def _find_first_tool_start(self, text: str) -> int:
        """Find the index of the first tool call or function tag in text.

        Checks for both ``<tool_call>`` wrappers and bare ``<function`` tags,
        returning whichever appears first.

        Args:
            text: The text to search.

        Returns:
            Index of the earliest match, or -1 if neither is found.
        """
        indices = [idx for idx in (text.find(self.tool_call_start_token), text.find("<function")) if idx >= 0]
        return min(indices) if indices else -1

    def _extract_function_name_from_match(self, match: re.Match[str]) -> str | None:
        """Extract the function name from a ``function_start_regex`` match.

        Args:
            match: A regex match from ``function_start_regex`` with three
                capture groups for different name syntaxes.

        Returns:
            The function name string, or None if no group matched.
        """
        return self._first_nonempty(match.group(1), match.group(2), match.group(3))

    def _extract_parameter_name_from_match(self, match: re.Match[str]) -> str | None:
        """Extract the parameter name from a ``parameter_regex`` match.

        Args:
            match: A regex match from ``parameter_regex`` with three
                capture groups for different name syntaxes.

        Returns:
            The parameter name string, or None if no group matched.
        """
        return self._first_nonempty(match.group(1), match.group(2), match.group(3))

    def _iter_function_blocks_from_chunk(
        self,
        chunk: str,
        *,
        complete_only: bool,
    ) -> list[str]:
        """Extract function block substrings from a single text chunk.

        Finds all ``<function ...>`` opening tags and pairs each with its
        closing ``</function>`` tag (or the next opening tag as a boundary).

        Args:
            chunk: A text segment to scan for function blocks.
            complete_only: If True, only return blocks that have a matching
                ``</function>`` closing tag. If False, also include
                incomplete trailing blocks.

        Returns:
            List of raw function block strings including their tags.
        """
        starts = list(self.function_start_regex.finditer(chunk))
        if not starts:
            return []

        function_blocks: list[str] = []
        for index, start_match in enumerate(starts):
            next_start = starts[index + 1].start() if index + 1 < len(starts) else -1
            close_idx = chunk.find(self.function_end_token, start_match.end())

            if close_idx != -1 and (next_start == -1 or close_idx < next_start):
                end_idx = close_idx + len(self.function_end_token)
            elif complete_only:
                continue
            elif next_start != -1:
                end_idx = next_start
            else:
                end_idx = len(chunk)

            function_blocks.append(chunk[start_match.start() : end_idx])

        return function_blocks

    def _iter_function_blocks(self, text: str, *, complete_only: bool) -> list[str]:
        """Extract all function blocks from text, handling optional wrappers.

        First splits the text into ``<tool_call>`` regions (if any), then
        delegates to ``_iter_function_blocks_from_chunk`` for each region.

        Args:
            text: The full text to search for function blocks.
            complete_only: If True, only return blocks with closing tags.

        Returns:
            List of raw function block strings across all chunks.
        """
        chunks: list[str] = []
        matched_ranges = self.tool_call_regex.findall(text)
        if matched_ranges:
            chunks = [match[0] if match[0] else match[1] for match in matched_ranges]
        else:
            chunks = [text]

        function_blocks: list[str] = []
        for chunk in chunks:
            function_blocks.extend(self._iter_function_blocks_from_chunk(chunk, complete_only=complete_only))
        return function_blocks

    def _parse_xml_function_call(self, function_block: str, tools) -> ToolCall | None:
        """Parse an XML function block into a ToolCall object.

        Supports both ``<function=name>`` and ``<function name="name">``
        syntaxes. Extracts all parameters, applies schema-based type
        conversion, and returns a structured ``ToolCall``.

        Args:
            function_block: The full function block string including tags.
            tools: Tool definitions for schema-based type conversion.

        Returns:
            A ``ToolCall`` with the function name and JSON-serialized
            arguments, or None if parsing fails.
        """
        start_match = self.function_start_regex.search(function_block)
        if start_match is None:
            return None

        function_name = self._extract_function_name_from_match(start_match)
        if not function_name:
            return None

        parameters_text = function_block[start_match.end() :]
        if parameters_text.endswith(self.function_end_token):
            parameters_text = parameters_text[: -len(self.function_end_token)]

        param_config = self._get_arguments_config(function_name, tools)
        param_dict = {}
        for match in self.parameter_regex.finditer(parameters_text):
            param_name = self._extract_parameter_name_from_match(match)
            if not param_name:
                continue

            param_value = match.group(4)
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(param_value, param_name, param_config, function_name)

        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete (non-streaming) model output.

        Searches for ``<function ...>`` blocks (with or without
        ``<tool_call>`` wrappers), parses each into a ``ToolCall``, and
        separates any leading content text.

        Args:
            model_output: The full model output text.
            request: The chat completion request containing tool definitions.

        Returns:
            An ``ExtractedToolCallInformation`` with parsed tool calls and
            any content preceding the first tool call.
        """
        if "<function" not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            function_blocks = self._iter_function_blocks(model_output, complete_only=False)
            if not function_blocks:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls = [
                self._parse_xml_function_call(function_block, request.tools) for function_block in function_blocks
            ]
            valid_tool_calls = [tool_call for tool_call in tool_calls if tool_call is not None]

            self.prev_tool_call_arr.clear()
            for tool_call in valid_tool_calls:
                self.prev_tool_call_arr.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )

            content_index = self._find_first_tool_start(model_output)
            content = model_output[:content_index] if content_index > 0 else None
            return ExtractedToolCallInformation(
                tools_called=bool(valid_tool_calls),
                tool_calls=valid_tool_calls,
                content=content,
            )
        except Exception:
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

        Uses a buffer-until-complete strategy: function blocks are
        accumulated until a closing ``</function>`` tag arrives, then
        parsed and emitted as ``DeltaToolCall`` objects in one shot.
        Content before the first tool tag is forwarded as text.

        Args:
            previous_text: All text generated before this step.
            current_text: All text generated up to and including this step.
            delta_text: The new text added in this step.
            previous_token_ids: Token IDs for ``previous_text`` (unused).
            current_token_ids: Token IDs for ``current_text`` (unused).
            delta_token_ids: Token IDs for ``delta_text``.
            request: The chat completion request with tool definitions.

        Returns:
            A ``DeltaMessage`` with content text, tool call deltas, or both.
            Returns None when there is nothing new to emit.
        """
        del previous_token_ids, current_token_ids

        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        tool_start_idx = self._find_first_tool_start(current_text)
        if tool_start_idx == -1:
            return DeltaMessage(content=delta_text) if delta_text else None

        content_before = None
        if not self.is_tool_call_started:
            self.is_tool_call_started = True
            if tool_start_idx > len(previous_text):
                content_before = current_text[len(previous_text) : tool_start_idx] or None

        complete_blocks = self._iter_function_blocks(current_text[tool_start_idx:], complete_only=True)
        delta_tool_calls: list[DeltaToolCall] = []

        if len(complete_blocks) > self.current_tool_index:
            pending_blocks = complete_blocks[self.current_tool_index :]
            start_index = self.current_tool_index
            self.current_tool_index += len(pending_blocks)

            for offset, function_block in enumerate(pending_blocks):
                parsed_tool = self._parse_xml_function_call(function_block, request.tools if request else None)
                if parsed_tool is None:
                    continue

                tool_id = self._generate_tool_call_id()
                delta_tool_calls.append(
                    DeltaToolCall(
                        index=start_index + offset,
                        id=tool_id,
                        function=DeltaFunctionCall(
                            name=parsed_tool.function.name,
                            arguments=parsed_tool.function.arguments,
                        ),
                        type="function",
                    )
                )
                self.prev_tool_call_arr.append(
                    {
                        "name": parsed_tool.function.name,
                        "arguments": parsed_tool.function.arguments,
                    }
                )
                self.streamed_args_for_tool.append(parsed_tool.function.arguments)

        if delta_tool_calls or content_before:
            return DeltaMessage(content=content_before, tool_calls=delta_tool_calls or None)
        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")
        return None
