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

"""OLMo-3 Pythonic tool parser module.

This module provides the Olmo3PythonicToolParser class for parsing tool calls
in a Python function call syntax used by OLMo-3 models. The parser uses AST
parsing to extract function names and keyword arguments.

Example format parsed:
    <function_calls>
    func_name(arg1="value1", arg2=123)
    another_func(param="test")
    </function_calls>

Or without tags:
    func_name(arg1="value1")
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

from eformer.loggings import get_logger
from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ._pythonic_ast import UnexpectedAstError, compute_tool_delta, handle_single_tool, make_valid_python

logger = get_logger(__name__)


@ToolParserManager.register_module("olmo3")  # pyright: ignore[reportUntypedClassDecorator]
class Olmo3PythonicToolParser(ToolParser):
    """Parser for OLMo-3 models that emit newline-separated pythonic tool calls.

    Uses Python's AST module to parse function call syntax, supporting
    both complete and streaming extraction with argument diffing.

    Features:
        - AST-based parsing for reliable extraction
        - Support for various literal types (str, int, float, bool, dict, list)
        - Streaming with partial syntax completion
        - Optional function_calls XML wrapper support
        - Bracket balancing for streaming validation

    Format:
        <function_calls>
        func_name(arg1="value", arg2=123)
        </function_calls>

        Or simply:
        func_name(arg1="value")

    Attributes:
        TOOL_CALL_REGEX: Compiled regex for validating tool call format.
    """

    TOOL_CALL_REGEX = re.compile(
        r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
        re.DOTALL,
    )

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the OLMo-3 Pythonic tool parser.

        Args:
            tokenizer: The tokenizer instance used for encoding/decoding tokens.
        """
        super().__init__(tokenizer)
        self.tool_call_start_token = "<function_calls>"
        self.tool_call_end_token = "</function_calls>"

    @property
    def current_tool_index(self) -> int:
        """Get the current tool index being processed.

        Returns:
            The index of the current tool (same as current_tool_id).
        """
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        """Set the current tool index.

        Args:
            value: The new tool index value.
        """
        self.current_tool_id = value

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Parses the model output using Python's AST module to extract
        function calls with their keyword arguments.

        Args:
            model_output: The complete model output string to parse.
            request: The chat completion request (unused but required by interface).

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: Whether any tools were called
                - tool_calls: List of parsed ToolCall objects
                - content: Original output if no tools found, else None
        """
        original_model_output = model_output

        match = re.search(r"<function_calls>(.*?)</function_calls>", model_output, re.DOTALL)
        if match:
            model_output = match.group(1).strip()

        model_output = ", ".join([line.strip() for line in model_output.splitlines() if line.strip()])
        model_output = f"[{model_output}]"

        if self.TOOL_CALL_REGEX.match(model_output) is None:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_model_output)

        try:
            module = ast.parse(model_output)
            parsed = getattr(module.body[0], "value", None)
            if isinstance(parsed, ast.List) and all(isinstance(e, ast.Call) for e in parsed.elts):
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=[handle_single_tool(e) for e in parsed.elts],
                    content=None,
                )
            raise UnexpectedAstError("Tool output must be a list of function calls")
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_model_output)

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

        Processes incremental output by completing partial Python syntax
        and parsing with AST. Computes argument diffs for streaming.

        Args:
            previous_text: The accumulated text from previous chunks.
            current_text: The current accumulated text including the new delta.
            delta_text: The new text in this chunk.
            previous_token_ids: Token IDs from previous chunks.
            current_token_ids: All token IDs including the new chunk.
            delta_token_ids: Token IDs for just the new chunk.
            request: The chat completion request.

        Returns:
            A DeltaMessage containing either:
                - content: Regular text if not a tool call
                - tool_calls: Delta with function name or argument fragments
                - None if more data needed or parsing incomplete
        """
        # All tool calls start with <function_calls>, but streaming may see partial tags.
        if not current_text.startswith("<"):
            return DeltaMessage(content=delta_text)

        try:
            if current_text.startswith("<function_calls>"):
                current_text = current_text[len("<function_calls>") :]
            if current_text.endswith("</function_calls>"):
                current_text = current_text[: -len("</function_calls>")]

            valid_and_added_text = make_valid_python(current_text)
            if valid_and_added_text is None:
                return None
            valid_text, added_text = valid_and_added_text

            valid_text = ", ".join([line.strip() for line in valid_text.splitlines() if line.strip()])
            valid_text = f"[{valid_text}]"

            module = ast.parse(valid_text)
            parsed = getattr(module.body[0], "value", None)
            if not isinstance(parsed, ast.List) or not all(isinstance(e, ast.Call) for e in parsed.elts):
                raise UnexpectedAstError("Tool output must be a sequence of newline-separated calls")

            tool_calls = [handle_single_tool(e) for e in parsed.elts]

            tool_deltas: list[DeltaToolCall] = []
            for index, new_call in enumerate(tool_calls):
                if index < self.current_tool_index:
                    continue

                self.current_tool_index = index
                if len(self.streamed_args_for_tool) == index:
                    self.streamed_args_for_tool.append("")

                new_call_complete = index < len(tool_calls) - 1 or ")" not in added_text
                if new_call_complete:
                    self.current_tool_index += 1

                withheld_suffix = added_text[:-1] if not new_call_complete else ""
                if not new_call_complete and added_text[-1] == ")":
                    withheld_suffix = withheld_suffix + "}"

                withheld_suffix = withheld_suffix.replace("'", '"')

                delta = compute_tool_delta(self.streamed_args_for_tool[index], new_call, index, withheld_suffix)
                if delta is not None:
                    tool_deltas.append(delta)
                    if delta.function is not None and delta.function.arguments is not None:
                        self.streamed_args_for_tool[index] += delta.function.arguments

            # Ensure finish_reason is set to tool_calls once at least one tool is called.
            if tool_deltas and not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            if not added_text and self.current_tool_id > 0:
                return DeltaMessage(content="")
            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
