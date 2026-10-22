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
    - Regex timeout handling with a linear-time structural pre-check
    - Support for nested data structures
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

import regex as re
from eformer.loggings import get_logger
from transformers import PreTrainedTokenizerBase

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ._pythonic_ast import UnexpectedAstError, compute_tool_delta, handle_single_tool, make_valid_python

logger = get_logger(__name__)

# Backward-compatible private alias for the shared exception type.
_UnexpectedAstError = UnexpectedAstError


@ToolParserManager.register_module("llama4_pythonic")  # pyright: ignore[reportUntypedClassDecorator]
class Llama4PythonicToolParser(ToolParser):
    """Tool call parser for Llama 4 models with Pythonic syntax.

    Specialized version of the pythonic parser for Llama 4 models that
    generate tool calls as Python function invocations. Supports optional
    <|python_start|> and <|python_end|> delimiters.

    Features:
        - Python AST-based parsing
        - Optional delimiter token support
        - Streaming with bracket completion
        - Regex timeout handling with a linear-time structural pre-check
        - Support for nested data structures

    Format:
        <|python_start|>[func1(arg1="val1"), func2(arg2="val2")]<|python_end|>
        or simply: [func1(arg1="val1"), func2(arg2="val2")]

    Attributes:
        TOOL_CALL_REGEX: Compiled structural pre-check. AST parsing validates the syntax.
        regex_timeout: Maximum seconds allowed for the regex pre-check.

    Use --enable-auto-tool-choice --tool-call-parser llama4_pythonic
    """

    TOOL_CALL_REGEX = re.compile(r"\s*\[.*\]\s*", re.DOTALL)

    def __init__(self, tokenizer: PreTrainedTokenizerBase, regex_timeout: float = 0.05):
        """Initialize the Llama4PythonicToolParser.

        Args:
            tokenizer: The tokenizer associated with the Llama 4 model.
                Used for token-level operations during streaming.
            regex_timeout: Positive time budget in seconds for the structural regex.

        Raises:
            ValueError: If the regex timeout is not positive and finite.
        """
        super().__init__(tokenizer)
        if not 0 < regex_timeout < float("inf"):
            raise ValueError("regex_timeout must be positive and finite")
        self.regex_timeout = regex_timeout
        self.tool_call_start_token = "["
        self.tool_call_end_token = "]"
        self.bot_token = "<|python_start|>"

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

    @staticmethod
    def _validate_tool_call_shape(module: ast.Module) -> list[ast.Call]:
        """Validate structure before converting the supported keyword-only calls.

        Args:
            module: Parsed complete or bracket-completed tool output.

        Returns:
            Calls whose arguments can be converted without silently dropping data.

        Raises:
            UnexpectedAstError: If output is not one list of simple calls, or
                calls contain positional/starred arguments, keyword unpacking,
                or repeated keyword names. Literal values are validated by the
                shared AST converter.
        """
        if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr):
            raise UnexpectedAstError("Tool output must be one list of function calls")
        parsed = module.body[0].value
        if not isinstance(parsed, ast.List) or not all(isinstance(call, ast.Call) for call in parsed.elts):
            raise UnexpectedAstError("Tool output must be a list of function calls")
        for call in parsed.elts:
            if not isinstance(call.func, ast.Name) or call.args:
                raise UnexpectedAstError("Tool calls require a simple name and keyword-only arguments")
            names = [keyword.arg for keyword in call.keywords]
            if None in names or len(names) != len(set(names)):
                raise UnexpectedAstError("Tool calls cannot unpack or repeat keyword arguments")
        return parsed.elts

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response.

        Parses the model output to identify and extract Python-style function
        calls. Handles optional <|python_start|> and <|python_end|> delimiters.
        Uses a linear-time bracket check for initial detection and AST parsing for extraction.

        Args:
            model_output: The complete text output from the model.
            request: The chat completion request containing context information.

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: True if valid tool calls were found
                - tool_calls: List of ToolCall objects extracted
                - content: Original content if no tools called, None otherwise
        """

        original_output = model_output
        if model_output.startswith("<|python_start|>"):
            model_output = model_output[len("<|python_start|>") :].removesuffix("<|python_end|>")

        # Keep the gate structural (no nested repetition over argument values) and
        # enforce the documented regex time budget with the timeout-capable engine.
        stripped = model_output.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)

        try:
            if self.TOOL_CALL_REGEX.fullmatch(stripped, timeout=self.regex_timeout) is None:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)
            calls = self._validate_tool_call_shape(ast.parse(stripped))
            if not calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[handle_single_tool(call) for call in calls],
                content=None,
            )
        except TimeoutError:
            logger.warning("Regex timed out while detecting Llama 4 tool calls")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)
        except SyntaxError:
            # Bracketed prose such as "[Note] ..." is ordinary content, not a malformed call.
            logger.debug("Model output is not valid Python; treating it as plain content.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)

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
            valid_and_added_text = make_valid_python(current_text)
            if valid_and_added_text is None:
                return None
            valid_text, added_text = valid_and_added_text

            module = ast.parse(valid_text)
            if not module.body:
                # Nothing parseable yet (e.g. only the <|python_start|> prefix arrived).
                return None
            calls = self._validate_tool_call_shape(module)
            tool_calls = [handle_single_tool(call) for call in calls]

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
                delta = compute_tool_delta(self.streamed_args_for_tool[index], new_call, index, withheld_suffix)

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
