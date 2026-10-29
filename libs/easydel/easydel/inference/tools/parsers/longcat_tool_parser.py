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

"""Longcat Flash tool parser extending Hermes-style parsing.

This module provides a tool call parser for Longcat Flash models that
extends the HermesToolParser with custom tool call tokens. It uses
<longcat_tool_call> and </longcat_tool_call> delimiters instead of
the standard Hermes tokens.

Example format:
    <longcat_tool_call>{"name": "func", "arguments": {...}}</longcat_tool_call>

Features:
    - Extends HermesToolParser functionality
    - Custom Longcat-specific token delimiters
    - Pre-encoded token arrays for efficient streaming
    - Full inheritance of Hermes parsing logic
"""

from __future__ import annotations

import re

from transformers import AutoTokenizer as AnyTokenizer

from ..abstract_tool import ToolParserManager
from .hermes_tool_parser import HermesToolParser


@ToolParserManager.register_module("longcat")  # pyright: ignore[reportUntypedClassDecorator]
class LongcatFlashToolParser(HermesToolParser):
    """Tool parser for Longcat Flash models.

    Extends HermesToolParser with Longcat-specific token delimiters.
    Uses <longcat_tool_call> and </longcat_tool_call> to wrap tool
    call JSON content.

    Features:
        - Inherits all Hermes parsing functionality
        - Custom start/end tokens for Longcat format
        - Pre-computed token ID arrays for efficient detection
        - Regex pattern for both complete and partial matches

    Format:
        <longcat_tool_call>{"name": "func", "arguments": {...}}</longcat_tool_call>

    Attributes:
        tool_call_start_token: The opening delimiter '<longcat_tool_call>'.
        tool_call_end_token: The closing delimiter '</longcat_tool_call>'.
        tool_call_regex: Compiled regex for extracting tool call content,
            handles both complete and partial (streaming) matches.
        tool_call_start_token_ids: List of token IDs for the start token.
        tool_call_end_token_ids: List of token IDs for the end token.
        tool_call_start_token_array: Decoded individual tokens for start.
        tool_call_end_token_array: Decoded individual tokens for end.

    Note:
        All other methods (extract_tool_calls, extract_tool_calls_streaming,
        adjust_request, etc.) are inherited from HermesToolParser.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the LongcatFlashToolParser.

        Sets up the parser with Longcat-specific tool call tokens and
        pre-computes token ID arrays for efficient streaming detection.

        Args:
            tokenizer: The tokenizer associated with the Longcat Flash model.
                Used for encoding tokens and token-level operations.
        """
        super().__init__(tokenizer)

        self.tool_call_start_token = "<longcat_tool_call>"
        self.tool_call_end_token = "</longcat_tool_call>"

        self.tool_call_regex = re.compile(
            r"<longcat_tool_call>(.*?)</longcat_tool_call>|<longcat_tool_call>(.*)",
            re.DOTALL,
        )

        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token,
            add_special_tokens=False,
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(self.tool_call_end_token, add_special_tokens=False)

        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id]) for token_id in self.tool_call_start_token_ids
        ]
        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id]) for token_id in self.tool_call_end_token_ids
        ]
