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

"""GLM-4.7 MoE tool parser module for parsing tool calls from GLM-4.7 MoE model outputs.

This module provides the Glm47MoeModelToolParser class which extends the
GLM-4 MoE parser with updated regex patterns for the GLM-4.7 format.
The format is similar to GLM-4 but with modified function detail extraction.

Example tool call format:
    <tool_call>get_weather
    <arg_key>location</arg_key>
    <arg_value>Beijing</arg_value>
    </tool_call>
"""

from __future__ import annotations

import re

from transformers import AutoTokenizer as AnyTokenizer

from ..abstract_tool import ToolParserManager
from .glm4_moe_tool_parser import Glm4MoeModelToolParser


@ToolParserManager.register_module(["glm47", "glm-4.7"])
class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    """Tool parser for GLM-4.7 MoE (Mixture of Experts) models.

    Extends the GLM-4 MoE parser with updated regex patterns to handle
    the slightly modified tool call format in GLM-4.7 models. The main
    difference is in how function details and arguments are structured.

    Key differences from GLM-4 MoE:
    - Function detail regex handles cases where arg_key may be optional
    - Argument regex allows newlines/whitespace between key and value tags

    Inherits all functionality from Glm4MoeModelToolParser including:
    - extract_tool_calls() for complete output parsing
    - extract_tool_calls_streaming() for incremental parsing
    - Buffer-based streaming accumulation

    Attributes:
        func_detail_regex (re.Pattern): Updated regex for function details
            that makes the argument section optional.
        func_arg_regex (re.Pattern): Updated regex that allows newlines
            and whitespace between arg_key and arg_value tags.

    Note:
        All other attributes are inherited from Glm4MoeModelToolParser.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("glm-4.7-moe")
        >>> parser = Glm47MoeModelToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the GLM-4.7 MoE tool parser.

        Calls the parent constructor and then overrides the regex patterns
        with GLM-4.7 specific versions that handle the updated format.

        Args:
            tokenizer: The tokenizer associated with the GLM-4.7 MoE model.
                Must contain the special tool call tokens in its vocabulary.

        Raises:
            ValueError: If the tokenizer is not provided (from parent).
        """
        super().__init__(tokenizer)
        self.func_detail_regex = re.compile(r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL)
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
