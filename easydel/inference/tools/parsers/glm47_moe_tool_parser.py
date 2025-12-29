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

import re

from transformers import AutoTokenizer as AnyTokenizer

from ..abstract_tool import ToolParserManager
from .glm4_moe_tool_parser import Glm4MoeModelToolParser


@ToolParserManager.register_module(["glm47", "glm-4.7"])
class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.func_detail_regex = re.compile(r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL)
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
