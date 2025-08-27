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
import json
import re
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from eformer.loggings import get_logger
from transformers import PreTrainedTokenizerBase

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager

logger = get_logger(__name__)


@ToolParserManager.register_module("phi4_mini_json")
class Phi4MiniJsonToolParser(ToolParser):
    """
    Tool call parser for Phi-4-mini models.

    Handles the functools format used by Phi-4-mini models. Extracts
    function calls from functools[...] wrapper with JSON array content.
    Currently supports non-streaming extraction only.

    Features:
    - Regex-based extraction of functools wrapper
    - JSON array parsing of function calls
    - Support for both 'arguments' and 'parameters' fields
    - Automatic tool ID generation

    Format:
    functools[{"name": "func", "arguments": {...}}, ...]

    Used when --enable-auto-tool-choice --tool-call-parser phi4_mini_json
    are all set.

    Note: Streaming extraction is not yet implemented (returns None).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.bot_token: str = "functools"

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        logger.debug("Model output: %s", model_output)

        pattern = r"functools\[(.*?)\]"
        matches = re.search(pattern, model_output, re.DOTALL)

        if not matches:
            logger.debug("No function calls found")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            function_call_arr: list[dict[str, Any]] = []
            try:
                json_content = "[" + matches.group(1) + "]"

                function_call_arr = json.loads(json_content)
                logger.debug("Successfully extracted %d function calls", len(function_call_arr))
            except json.JSONDecodeError as e:
                logger.error("Failed to parse function calls from model output. Error: %s", str(e))

            tool_calls: list[ToolCall] = [
                ToolCall(
                    id=f"chatcmpl-tool-{uuid4()}",
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        arguments=json.dumps(
                            raw_function_call["arguments"]
                            if "arguments" in raw_function_call
                            else raw_function_call["parameters"],
                            ensure_ascii=False,
                        ),
                    ),
                )
                for raw_function_call in function_call_arr
            ]

            ret = ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=None)
            return ret

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
        return None
