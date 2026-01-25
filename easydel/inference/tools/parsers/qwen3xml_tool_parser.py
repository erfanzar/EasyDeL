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

"""Qwen3 XML tool parser module.

This module provides the Qwen3XMLToolParser class as an alias for the
Qwen3CoderToolParser. It handles the same XML-style tool call format
used by Qwen3 models in non-coder variants.

The parser is registered under the 'qwen3_xml' module name and inherits
all functionality from Qwen3CoderToolParser.
"""

from __future__ import annotations

from ..abstract_tool import ToolParserManager
from .qwen3coder_tool_parser import Qwen3CoderToolParser


@ToolParserManager.register_module(["qwen3_xml"])
class Qwen3XMLToolParser(Qwen3CoderToolParser):
    """Alias parser for Qwen3 XML tool calling format (non-coder variant).

    This class provides the same XML-based tool parsing functionality as
    Qwen3CoderToolParser but is registered under a different module name
    for use with non-coder Qwen3 model variants.

    The XML format parsed is identical to Qwen3CoderToolParser:
        <tool_call>
        <function=name>
        <parameter=param1>value1</parameter>
        <parameter=param2>value2</parameter>
        </function>
        </tool_call>

    See Also:
        Qwen3CoderToolParser: The parent class providing all implementation.
    """

    pass
