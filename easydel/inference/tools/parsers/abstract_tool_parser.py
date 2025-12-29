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

"""Compatibility shim for vLLM-style imports.

vLLM keeps the ToolParser base class under `tool_parsers/abstract_tool_parser.py`.
EasyDeL keeps it under `easydel.inference.tools.abstract_tool`.

This module exists so code ported from vLLM (or referencing that layout)
can import from `easydel.inference.tools.parsers.abstract_tool_parser` without
needing to know EasyDeL's internal path.
"""

from __future__ import annotations

from ..abstract_tool import ToolParser, ToolParserManager

__all__ = ["ToolParser", "ToolParserManager"]
