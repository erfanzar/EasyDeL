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

from ..abstract_tool import ToolParserManager
from .qwen3coder_tool_parser import Qwen3CoderToolParser


@ToolParserManager.register_module(["qwen3_xml"])
class Qwen3XMLToolParser(Qwen3CoderToolParser):
    """Alias for Qwen3 XML tool calling format (non-coder)."""

    pass
