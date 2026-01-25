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
"""Mistral model tool call parser.

This module provides a specialized tool call parser for Mistral AI models
(7B Instruct v0.3 and later). The parser handles Mistral's specific tool call
format which uses the [TOOL_CALLS] token followed by a JSON array of function
calls.

The parser is designed to work with the mistral_common library and the
tool_chat_template_mistral.jinja template. It supports automatic tool ID
generation and is compatible with different Mistral tokenizer versions.

Example:
    >>> from easydel.inference.tools.parsers.mistral_tool_parser import MistralToolParser
    >>> parser = MistralToolParser(tokenizer)
    >>> result = parser.extract_tool_calls(
    ...     '[TOOL_CALLS][{"name": "search", "arguments": {"q": "test"}}]',
    ...     request
    ... )
    >>> result.tools_called
    True

See Also:
    - https://github.com/mistralai/mistral-common/ - Mistral Common library
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits

import partial_json_parser
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow
from pydantic import Field
from transformers import AutoTokenizer as AnyTokenizer

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except ImportError:
    MistralTokenizer = None

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
from ..utils import extract_intermediate_diff

logger = get_logger(__name__)

ALPHANUMERIC = ascii_letters + digits


class MistralToolCall(ToolCall):
    """A specialized ToolCall class for Mistral models with auto-generated IDs.

    Extends the base ToolCall class to provide automatic 9-character alphanumeric
    ID generation, which is the format expected by Mistral's tool calling system.

    Attributes:
        id (str): A 9-character alphanumeric identifier, automatically generated
            if not provided. This ID format is specific to Mistral's requirements.

    Example:
        >>> call = MistralToolCall(
        ...     type="function",
        ...     function=FunctionCall(name="search", arguments='{"q": "test"}')
        ... )
        >>> len(call.id)
        9
        >>> call.id.isalnum()
        True
    """

    id: str = Field(default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id() -> str:
        """Generate a random 9-character alphanumeric ID.

        Creates a unique identifier suitable for Mistral tool calls by randomly
        selecting 9 characters from the alphanumeric character set.

        Returns:
            str: A 9-character string containing only letters (a-z, A-Z) and
                digits (0-9).

        Example:
            >>> id = MistralToolCall.generate_random_id()
            >>> len(id)
            9
            >>> id.isalnum()
            True
        """
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:  # noqa
        """Validate whether a string is a valid Mistral tool call ID.

        Checks if the provided ID matches Mistral's expected format: exactly
        9 alphanumeric characters.

        Args:
            id (str): The ID string to validate.

        Returns:
            bool: True if the ID is exactly 9 alphanumeric characters,
                False otherwise.

        Example:
            >>> MistralToolCall.is_valid_id("abc123XYZ")
            True
            >>> MistralToolCall.is_valid_id("short")
            False
            >>> MistralToolCall.is_valid_id("has-dash!")
            False
        """
        return id.isalnum() and len(id) == 9


def _is_fn_name_regex_support(model_tokenizer: AnyTokenizer) -> bool:
    """Check if the tokenizer supports regex-based function name extraction.

    Determines whether the provided tokenizer is a Mistral tokenizer with
    version 11 or higher, which enables enhanced regex-based function name
    parsing capabilities.

    Args:
        model_tokenizer (AnyTokenizer): The tokenizer instance to check.
            Can be any tokenizer type, but only MistralTokenizer v11+
            will return True.

    Returns:
        bool: True if the tokenizer is a MistralTokenizer with version >= 11,
            False otherwise (including when mistral_common is not installed).
    """
    return MistralTokenizer and isinstance(model_tokenizer, MistralTokenizer) and model_tokenizer.version >= 11


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolParser):
    """Tool call parser for Mistral models (7B Instruct v0.3+).

    This parser is designed specifically for Mistral AI models and handles their
    unique tool call format which uses the [TOOL_CALLS] token followed by a JSON
    array of function calls. It is compatible with both the mistral_common library
    tokenizers and standard HuggingFace tokenizers.

    Designed for use with:
        - `mistral_common <https://github.com/mistralai/mistral-common/>`_
        - The examples/tool_chat_template_mistral.jinja template

    Supported Formats:
        - Standard JSON array: `[TOOL_CALLS][{"name": "func", "arguments": {...}}]`
        - Multiple calls: `[TOOL_CALLS][{"name": "func1", ...}, {"name": "func2", ...}]`
        - Regex pattern (v11+): `function_name{...}` without outer array

    Features:
        - Automatic 9-character alphanumeric tool ID generation
        - Support for multiple tool calls in a single response
        - Streaming with incremental argument parsing
        - Compatibility with different Mistral tokenizer versions
        - Regex-based function name extraction for v11+ tokenizers

    Attributes:
        bot_token (str): The special token "[TOOL_CALLS]" marking tool call sections.
        bot_token_id (int | None): The token ID for bot_token in the vocabulary.
        tool_call_regex (re.Pattern): Pattern for extracting JSON array tool calls.
        fn_name_regex (re.Pattern | None): Pattern for function name extraction
            (only available with MistralTokenizer v11+).
        prev_tool_call_arr (list[dict]): Previous tool calls for streaming diff.
        current_tool_id (int): Index of current tool being processed.
        current_tool_name_sent (bool): Whether function name was sent in stream.
        streamed_args_for_tool (list[str]): Accumulated arguments per tool.

    Example:
        >>> parser = MistralToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(
        ...     '[TOOL_CALLS][{"name": "search", "arguments": {"q": "test"}}]',
        ...     request
        ... )
        >>> result.tools_called
        True
        >>> result.tool_calls[0].function.name
        'search'

    Raises:
        RuntimeError: If the [TOOL_CALLS] token is not found in the tokenizer
            vocabulary during initialization.

    Note:
        Use with --enable-auto-tool-choice --tool-call-parser mistral.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Mistral tool parser.

        Args:
            tokenizer (AnyTokenizer): The tokenizer instance for the Mistral model.
                Can be either a MistralTokenizer from mistral_common or a standard
                HuggingFace tokenizer.

        Raises:
            RuntimeError: If the [TOOL_CALLS] token cannot be found in the
                tokenizer's vocabulary.
        """
        super().__init__(tokenizer)

        if MistralTokenizer and not isinstance(self.model_tokenizer, MistralTokenizer):
            pass

        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        if _is_fn_name_regex_support(self.model_tokenizer):
            self.fn_name_regex = re.compile(r"([a-zA-Z0-9_-]+)(\{[\s\S]*?\})(?=\s*$|,|\s)", re.DOTALL)
        else:
            self.fn_name_regex = None

        if self.bot_token_id is None:
            raise RuntimeError("Mistral Tool Parser could not locate the tool call token in the tokenizer!")

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust the chat completion request for Mistral tool parsing compatibility.

        Modifies the request to ensure special tokens are preserved during decoding
        when using a non-Mistral tokenizer with tool calling enabled. This is
        necessary because the [TOOL_CALLS] token must be visible in the output
        for proper parsing.

        Args:
            request (ChatCompletionRequest): The original chat completion request
                to potentially modify.

        Returns:
            ChatCompletionRequest: The adjusted request with skip_special_tokens
                set to False if tools are enabled and a non-Mistral tokenizer
                is being used.

        Note:
            This adjustment is only applied when:
            - mistral_common is available
            - The tokenizer is NOT a MistralTokenizer
            - Tools are defined in the request
            - tool_choice is not set to "none"
        """
        if (
            MistralTokenizer
            and not isinstance(self.model_tokenizer, MistralTokenizer)
            and request.tools
            and request.tool_choice != "none"
        ):
            # Note: we don't want skip_special_tokens=False
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Mistral model response.

        Parses the model output to find the [TOOL_CALLS] token and extract
        the following JSON array or function call patterns. The method supports
        both standard JSON format parsing and regex-based extraction for newer
        tokenizer versions (v11+).

        Args:
            model_output (str): The complete text output from the Mistral model,
                expected to contain the [TOOL_CALLS] token followed by JSON.
            request (ChatCompletionRequest): The original chat completion request.
                Currently unused but included for API consistency.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called (bool): True if valid tool calls were extracted.
                - tool_calls (list[MistralToolCall]): List of MistralToolCall
                  objects with auto-generated IDs.
                - content (str | None): Text content before the [TOOL_CALLS]
                  token, or None if there was no preceding content.

        Example:
            >>> result = parser.extract_tool_calls(
            ...     'Let me search. [TOOL_CALLS][{"name": "search", "arguments": {}}]',
            ...     request
            ... )
            >>> result.tools_called
            True
            >>> result.content
            'Let me search. '
            >>> result.tool_calls[0].function.name
            'search'

        Note:
            For regex-based parsing (v11+ tokenizers), the format is:
            `function_name{arguments}` rather than a JSON array.
        """

        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        tool_content = model_output.replace(self.bot_token, "").strip()

        try:
            try:
                if self.fn_name_regex:
                    matches = self.fn_name_regex.findall(tool_content)

                    function_call_arr = []
                    for match in matches:
                        fn_name = match[0]
                        args = match[1]

                        function_call_arr.append({"name": fn_name, "arguments": json.loads(args)})
                else:
                    function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # NOTE: This use case should not happen if the model is trained
                raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
                function_call_arr = json.loads(raw_tool_call)

            tool_calls: list[MistralToolCall] = [
                MistralToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        arguments=json.dumps(raw_function_call["arguments"], ensure_ascii=False),
                    ),
                )
                for raw_function_call in function_call_arr
            ]

            content = model_output.split(self.bot_token)[0]
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content if len(content) > 0 else None
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=tool_content)

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

        Processes streaming Mistral model output to extract tool calls as they
        are being generated. Handles the [TOOL_CALLS] token detection and
        incrementally parses the following JSON array to emit function names
        and argument updates.

        Args:
            previous_text (str): The accumulated text before this delta.
            current_text (str): The accumulated text including this delta.
            delta_text (str): The new text generated in this streaming chunk.
            previous_token_ids (Sequence[int]): Token IDs before this delta.
            current_token_ids (Sequence[int]): Token IDs including this delta.
            delta_token_ids (Sequence[int]): New token IDs in this chunk.
            request (ChatCompletionRequest): The original chat completion request.

        Returns:
            DeltaMessage | None: A delta message containing either:
                - Content text if no tool call token has been seen yet.
                - None if only the [TOOL_CALLS] token was received (waiting for JSON).
                - Tool call name update when a new function name is parsed.
                - Tool call arguments update with incremental argument text.
                - None if parsing failed or more tokens are needed.

        Note:
            This method maintains state across calls via instance attributes:
            - prev_tool_call_arr: Tracks previous tool call state for diff.
            - current_tool_id: Index of current tool being streamed.
            - current_tool_name_sent: Whether function name has been emitted.
            - streamed_args_for_tool: Accumulated arguments per tool.

            The method performs special handling for quotes in delta_text,
            replacing single quotes with double quotes for JSON compatibility.

        Warning:
            Due to quote handling, tool call arguments should ideally avoid
            single quotes to prevent parsing ambiguities.
        """
        if self.bot_token not in current_text:
            return DeltaMessage(content=delta_text)

        if self.bot_token_id in delta_token_ids and len(delta_token_ids) == 1:
            return None

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            parsable_arr = current_text.split(self.bot_token)[-1]

            try:
                tool_call_arr: list[dict] = partial_json_parser.loads(parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}

            if len(tool_call_arr) == 0:
                return None

            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    diff: str | None = current_tool_call.get("arguments")

                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(
                            self.streamed_args_for_tool[self.current_tool_id], ""
                        )
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                    else:
                        delta = None
                else:
                    delta = None
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=MistralToolCall.generate_random_id(),
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = None

            else:
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                cur_arguments = current_tool_call.get("arguments")

                new_text = delta_text.replace("'", '"')
                if '"}' in new_text:
                    new_text = new_text[: new_text.rindex('"}')]

                if not cur_arguments and not prev_arguments:
                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error("INVARIANT - impossible to have arguments reset mid-arguments")
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)[:-2]
                    logger.debug("finding %s in %s", new_text, cur_arguments_json)

                    if new_text not in cur_arguments_json:
                        return None
                    arguments_delta = cur_arguments_json[: cur_arguments_json.rindex(new_text) + len(new_text)]
                    logger.debug("First tokens in arguments received: %s", arguments_delta)
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=arguments_delta).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta

                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    logger.debug("Searching for diff between \n%s\n%s", cur_args_json, prev_args_json)

                    argument_diff = extract_intermediate_diff(cur_args_json, prev_args_json)
                    logger.debug("got arguments diff: %s", argument_diff)
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                else:
                    delta = None

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return None
