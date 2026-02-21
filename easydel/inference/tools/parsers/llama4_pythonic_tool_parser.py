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

"""Llama 4 Pythonic tool parser with optional delimiter token support.

This module provides a specialized tool call parser for Llama 4 models that
generate tool calls as Python function invocations. It extends the base
pythonic parser with support for optional <|python_start|> and <|python_end|>
delimiter tokens.

Example formats:
    <|python_start|>[func1(arg1="val1"), func2(arg2="val2")]<|python_end|>
    [func1(arg1="val1"), func2(arg2="val2")]

Features:
    - Python AST-based parsing for safe evaluation
    - Optional delimiter token handling
    - Streaming with bracket completion
    - Regex timeout handling for complex patterns
    - Support for nested data structures
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

from eformer.loggings import get_logger
from transformers import PreTrainedTokenizerBase

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

logger = get_logger(__name__)


class _UnexpectedAstError(Exception):
    """Exception raised when AST parsing encounters unexpected structure.

    This exception is used internally to signal that the parsed AST does not
    match the expected format for tool calls (e.g., not a list of function
    calls, invalid argument types).
    """

    pass


@ToolParserManager.register_module("llama4_pythonic")
class Llama4PythonicToolParser(ToolParser):
    """Tool call parser for Llama 4 models with Pythonic syntax.

    Specialized version of the pythonic parser for Llama 4 models that
    generate tool calls as Python function invocations. Supports optional
    <|python_start|> and <|python_end|> delimiters.

    Features:
        - Python AST-based parsing
        - Optional delimiter token support
        - Streaming with bracket completion
        - Regex timeout handling for complex patterns
        - Support for nested data structures

    Format:
        <|python_start|>[func1(arg1="val1"), func2(arg2="val2")]<|python_end|>
        or simply: [func1(arg1="val1"), func2(arg2="val2")]

    Attributes:
        TOOL_CALL_REGEX: Compiled regex pattern for detecting tool call syntax.

    Use --enable-auto-tool-choice --tool-call-parser llama4_pythonic
    """

    TOOL_CALL_REGEX = re.compile(
        r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
        re.DOTALL,
    )

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """Initialize the Llama4PythonicToolParser.

        Args:
            tokenizer: The tokenizer associated with the Llama 4 model.
                Used for token-level operations during streaming.
        """
        super().__init__(tokenizer)

    @property
    def current_tool_index(self) -> int:
        """Get the current tool index being processed.

        Returns:
            The index of the current tool being processed during streaming.
        """
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        """Set the current tool index being processed.

        Args:
            value: The new tool index value.
        """
        self.current_tool_id = value

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response.

        Parses the model output to identify and extract Python-style function
        calls. Handles optional <|python_start|> and <|python_end|> delimiters.
        Uses regex for initial detection and AST parsing for extraction.

        Args:
            model_output: The complete text output from the model.
            request: The chat completion request containing context information.

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: True if valid tool calls were found
                - tool_calls: List of ToolCall objects extracted
                - content: Original content if no tools called, None otherwise
        """

        if model_output.startswith("<|python_start|>"):
            model_output = model_output[len("<|python_start|>") :]
            model_output = model_output.replace("<|python_end|>", "")

        is_tool_call_pattern = False
        try:
            is_tool_call_pattern = self.TOOL_CALL_REGEX.match(model_output, timeout=1) is not None
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")
            logger.debug("Regex timeout occurred when matching user input: %s", model_output)

        if not is_tool_call_pattern:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            module = ast.parse(model_output)
            parsed = getattr(module.body[0], "value", None)
            if isinstance(parsed, ast.List) and all(isinstance(e, ast.Call) for e in parsed.elts):
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=[
                        _handle_single_tool(e)  # type: ignore
                        for e in parsed.elts
                    ],
                    content=None,
                )
            else:
                raise _UnexpectedAstError("Tool output must be a list of function calls")
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
        """Extract tool calls incrementally during streaming.

        Processes streaming output token by token, handling optional
        <|python_start|> and <|python_end|> delimiters. Completes partial
        Python code to extract tool call information as it becomes available.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Complete text including the current delta.
            delta_text: The new text added in this streaming chunk.
            previous_token_ids: Token IDs before this delta.
            current_token_ids: All token IDs including current delta.
            delta_token_ids: Token IDs in the current delta.
            request: The chat completion request for context.

        Returns:
            DeltaMessage containing either content (if not a tool call) or
            tool_calls (if tool call deltas detected), or None if more tokens
            are needed to determine the output type.
        """
        if not current_text.startswith("[") and not current_text.startswith("<|python_start|>"):
            return DeltaMessage(content=delta_text)

        try:
            if current_text.startswith("<|python_start|>"):
                current_text = current_text[len("<|python_start|>") :]
            if current_text.endswith("<|python_end|>"):
                current_text = current_text[: current_text.rfind("<|python_end|>")]
            valid_and_added_text = _make_valid_python(current_text)
            if valid_and_added_text is None:
                return None
            valid_text, added_text = valid_and_added_text

            module = ast.parse(valid_text)
            parsed = getattr(module.body[0], "value", None)
            if not isinstance(parsed, ast.List) or not all(isinstance(e, ast.Call) for e in parsed.elts):
                raise _UnexpectedAstError("Tool output must be a list of function calls")
            tool_calls = [
                _handle_single_tool(e)  # type: ignore
                for e in parsed.elts
            ]

            tool_deltas = []
            for index, new_call in enumerate(tool_calls):
                if index < self.current_tool_index:
                    continue

                self.current_tool_index = index
                if len(self.streamed_args_for_tool) == index:
                    self.streamed_args_for_tool.append("")

                new_call_complete = index < len(tool_calls) - 1 or ")]" not in added_text
                if new_call_complete:
                    self.current_tool_index += 1

                withheld_suffix = added_text[:-2] if not new_call_complete else ""
                if not new_call_complete and added_text[-2] == ")":
                    withheld_suffix = withheld_suffix + "}"
                withheld_suffix = withheld_suffix.replace("'", '"')
                delta = _compute_tool_delta(self.streamed_args_for_tool[index], new_call, index, withheld_suffix)

                if delta is not None:
                    tool_deltas.append(delta)
                    if delta.function is not None and delta.function.arguments is not None:
                        self.streamed_args_for_tool[index] += delta.function.arguments

            # HACK: serving_chat.py inspects the internal state of tool parsers
            if tool_deltas and not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            elif not added_text and self.current_tool_id > 0:
                return DeltaMessage(content="")
            else:
                return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return None


def _get_parameter_value(val: ast.expr) -> Any:
    """Extract parameter value from AST expression.

    Recursively processes AST nodes to extract Python literals
    including constants, dictionaries, and lists.

    Args:
        val: AST expression node to evaluate.

    Returns:
        Python value extracted from the AST node. Can be:
            - str, int, float, bool, None for ast.Constant
            - dict for ast.Dict (with recursive value extraction)
            - list for ast.List (with recursive value extraction)

    Raises:
        _UnexpectedAstError: If expression is not a supported literal type
            or if dict keys are not constants.
    """
    if isinstance(val, ast.Constant):
        return val.value
    elif isinstance(val, ast.Dict):
        if not all(isinstance(k, ast.Constant) for k in val.keys):
            raise _UnexpectedAstError("Dict tool call arguments must have literal keys")
        return {
            k.value: _get_parameter_value(v)  # type: ignore
            for k, v in zip(val.keys, val.values, strict=False)
        }
    elif isinstance(val, ast.List):
        return [_get_parameter_value(v) for v in val.elts]
    else:
        raise _UnexpectedAstError("Tool call arguments must be literals")


def _handle_single_tool(call: ast.Call) -> ToolCall:
    """Convert AST Call node to ToolCall object.

    Extracts function name and keyword arguments from Python
    function call AST representation.

    Args:
        call: AST Call node representing a function invocation.

    Returns:
        ToolCall object with:
            - type: "function"
            - function: FunctionCall with name and JSON-encoded arguments

    Raises:
        _UnexpectedAstError: If the function name is not a simple identifier
            (ast.Name) or if arguments cannot be extracted.
    """
    if not isinstance(call.func, ast.Name):
        raise _UnexpectedAstError("Invalid tool call name")
    function_name = call.func.id
    arguments = {}
    for keyword in call.keywords:
        arguments[keyword.arg] = _get_parameter_value(keyword.value)
    return ToolCall(type="function", function=FunctionCall(name=function_name, arguments=json.dumps(arguments)))


def _make_valid_python(text: str) -> tuple[str, str] | None:
    """Complete partial Python code by adding closing brackets.

    Tracks bracket stack to determine what closing characters
    are needed to make partial Python code syntactically valid.
    Handles brackets, parentheses, braces, and quotes.

    Args:
        text: Partial Python code string that may be incomplete.

    Returns:
        Tuple of (completed_text, added_text) where:
            - completed_text: The input text with closing brackets appended
            - added_text: The characters that were added to complete the code
        Returns None if the text ends with incomplete parameter name or value
        (e.g., ends with '=' or ':'), indicating more tokens are needed.

    Raises:
        _UnexpectedAstError: If mismatched brackets are detected.
    """
    bracket_stack = []
    for index, char in enumerate(text):
        if char in {"[", "(", "{"}:
            bracket_stack.append(char)
        elif char == "]":
            if not bracket_stack or bracket_stack.pop() != "[":
                raise _UnexpectedAstError("Mismatched square brackets")
        elif char == ")":
            if not bracket_stack or bracket_stack.pop() != "(":
                raise _UnexpectedAstError("Mismatched parentheses")
        elif char == "}":
            if not bracket_stack or bracket_stack.pop() != "{":
                raise _UnexpectedAstError("Mismatched curly braces")
        elif char in {"'", '"'}:
            if bracket_stack and bracket_stack[-1] == char:
                if index > 0 and text[index - 1] == "\\":
                    pass
                else:
                    bracket_stack.pop()
            elif bracket_stack and bracket_stack[-1] in {"'", '"'}:
                pass
            else:
                bracket_stack.append(char)

    text = text.rstrip()
    if text.endswith("=") or text.endswith(":"):
        return None
    if bracket_stack and bracket_stack[-1] == "{":
        trailing_dict_text = text[: text.rfind("{")]
        num_keys = trailing_dict_text.count(":")
        num_values = trailing_dict_text.count(",")
        if num_keys <= num_values:
            return None
    if bracket_stack and bracket_stack[-1] == "(":
        trailing_params_text = text[: text.rfind("(")]
        num_full_param_names = trailing_params_text.count("=")
        num_full_param_values = trailing_params_text.count(",")
        if num_full_param_names <= num_full_param_values:
            return None
    if text.endswith(","):
        text = text[:-1]
    if bracket_stack and bracket_stack[-1] == "[" and not text.endswith("[") and not text.endswith(")"):
        return None

    added_text = ""
    for char in reversed(bracket_stack):
        if char == "[":
            added_text += "]"
        elif char == "(":
            added_text += ")"
        elif char == "{":
            added_text += "}"
        elif char == "'":
            added_text += "'"
        elif char == '"':
            added_text += '"'

    return text + added_text, added_text


def _compute_tool_delta(
    previously_sent_args: str, new_call: ToolCall, index: int, withheld_suffix: str
) -> DeltaToolCall | None:
    """Compute the delta between previously sent and current tool call arguments.

    Calculates what new argument data should be sent in a streaming response
    by comparing previously sent arguments with the current state.

    Args:
        previously_sent_args: JSON string of arguments already sent to client.
        new_call: The current ToolCall with updated arguments.
        index: The index of this tool call in the response.
        withheld_suffix: Characters to withhold from the end of arguments
            (used for incomplete streaming data).

    Returns:
        DeltaToolCall containing the incremental update, or None if no new
        data to send. For the first delta of a tool call, includes the
        function name; subsequent deltas only include argument differences.
    """
    new_call_args = new_call.function.arguments
    if withheld_suffix:
        assert new_call_args.endswith(withheld_suffix)
        new_call_args = new_call_args[: -len(withheld_suffix)]
    if not previously_sent_args:
        return DeltaToolCall(
            id=new_call.id,
            type="function",
            index=index,
            function=DeltaFunctionCall(
                name=new_call.function.name,
                arguments=new_call_args,
            ),
        )

    arg_diff = new_call_args[len(previously_sent_args) :]
    return DeltaToolCall(id=None, index=index, function=DeltaFunctionCall(arguments=arg_diff)) if arg_diff else None
