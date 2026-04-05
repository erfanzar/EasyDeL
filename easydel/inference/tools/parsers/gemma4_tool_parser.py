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

"""Gemma4 tool parser.

Gemma4 instruction checkpoints use the same ``call:name{...}`` payload shape as
FunctionGemma, but wrap it in Gemma4-specific control tokens:

    <|tool_call>call:lookup{query:<|"|>weather<|"|>}<tool_call|>
"""

from __future__ import annotations

from ..abstract_tool import ToolParserManager
from .functiongemma_tool_parser import FunctionGemmaToolParser


@ToolParserManager.register_module("gemma4")  # pyright: ignore[reportUntypedClassDecorator]
class Gemma4ToolParser(FunctionGemmaToolParser):
    """Parse Gemma4 ``<|tool_call>call:...<tool_call|>`` tool invocations."""

    tool_call_start_token = "<|tool_call>"
    tool_call_end_token = "<tool_call|>"
    escape_tokens = ('<|"|>', "<escape>")
