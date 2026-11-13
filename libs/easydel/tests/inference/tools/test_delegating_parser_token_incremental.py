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

"""Incremental tool-view tokenization must stay exact and bounded per delta."""

import os

from easydel.inference.openai_api_modules import DeltaMessage
from easydel.inference.parsing import DelegatingParser, ParseResult
from transformers import AutoTokenizer


class _CountingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.encoded_chars = 0

    def encode(self, text, add_special_tokens=False):
        self.encoded_chars += len(text)
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids, skip_special_tokens=False):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class _ToolParser:
    def __init__(self, tokenizer):
        self.model_tokenizer = tokenizer

    def extract_tool_calls_streaming(self, **kwargs):
        return DeltaMessage(content=kwargs["delta_text"])

    def is_buffering_protocol(self, *, current_text="", delta_text=""):
        return False


def test_incremental_tool_view_matches_full_tokenization_with_bounded_work():
    tokenizer = _CountingTokenizer(
        AutoTokenizer.from_pretrained(os.environ.get("GLM_TEST_TOKENIZER", "zai-org/GLM-5.3-Flash"))
    )
    parser = DelegatingParser(tool_parser=_ToolParser(tokenizer))
    chunks = [
        "Reasoning output ",
        "with spaces, punctuation, ",
        "and unicode 世界 ",
        "continues across chunk boundaries ",
        "without tool calls. ",
    ] * 16
    cumulative = ""
    for chunk in chunks:
        cumulative += chunk
        parser._process_tool_delta(ParseResult(), cumulative, chunk, [])
        expected = tokenizer.tokenizer.encode(cumulative, add_special_tokens=False)
        assert parser._tool_previous_token_ids == expected
    naive_chars = sum(len("".join(chunks[: i + 1])) + len(chunks[i]) for i in range(len(chunks)))
    assert tokenizer.encoded_chars < naive_chars // 3
