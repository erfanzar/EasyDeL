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
from .hermes_tool_parser import HermesToolParser


@ToolParserManager.register_module("longcat")
class LongcatFlashToolParser(HermesToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
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
