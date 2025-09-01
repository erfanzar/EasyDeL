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
    id: str = Field(default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:  # noqa
        return id.isalnum() and len(id) == 9


def _is_fn_name_regex_support(model_tokenizer: AnyTokenizer) -> bool:
    return MistralTokenizer and isinstance(model_tokenizer, MistralTokenizer) and model_tokenizer.version >= 11


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral models (7B Instruct v0.3+).

    Designed for use with:
    - [`mistral_common`](https://github.com/mistralai/mistral-common/)
    - the examples/tool_chat_template_mistral.jinja template

    Handles Mistral's specific tool call format with [TOOL_CALLS] token
    and JSON array of function calls. Supports both standard JSON parsing
    and regex-based parsing for function names with v11+ tokenizers.

    Features:
    - Automatic tool ID generation (9-character alphanumeric)
    - Support for multiple tool calls in single response
    - Streaming with incremental argument parsing
    - Compatibility with different Mistral tokenizer versions

    Used when --enable-auto-tool-choice --tool-call-parser mistral are set.
    """

    def __init__(self, tokenizer: AnyTokenizer):
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
        """
        Extract tool calls from complete Mistral model response.

        Parses the [TOOL_CALLS] token followed by JSON array or
        function call patterns. Handles both standard JSON format
        and regex-based extraction for newer tokenizer versions.

        Args:
            model_output: Complete model output with tool calls
            request: Original request (unused)

        Returns:
            Extracted tool information with MistralToolCall objects

        Note:
            Tool call arguments should avoid quotes as parser may
            need to replace single quotes with double quotes.
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
