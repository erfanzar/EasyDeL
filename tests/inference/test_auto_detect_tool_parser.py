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

"""Tests for tool parser auto-detection logic."""

import pytest

from easydel.inference.tools.auto_detect import (
    _DEFAULT_PARSER,
    _TEMPLATE_HINTS,
    _VOCAB_HINTS,
    MODEL_TYPE_TO_TOOL_PARSER,
    detect_tool_parser,
)


class _FakeTokenizer:
    def __init__(self, chat_template=None, vocab=None):
        self.chat_template = chat_template
        self._vocab = vocab or {}

    def get_vocab(self):
        return self._vocab


class TestDetectToolParserExplicitName:
    def test_explicit_valid_name_returned(self):
        result = detect_tool_parser(parser_name="hermes")
        assert result == "hermes"

    def test_explicit_invalid_name_falls_through(self):
        result = detect_tool_parser(parser_name="nonexistent_parser_xyz")
        assert result == _DEFAULT_PARSER

    def test_explicit_name_takes_priority_over_model_type(self):
        result = detect_tool_parser(parser_name="hermes", model_type="mistral")
        assert result == "hermes"


class TestDetectToolParserModelType:
    @pytest.mark.parametrize(
        "model_type,expected",
        [
            ("qwen3", "hermes"),
            ("qwen3_5", "qwen3_coder"),
            ("qwen3_5_moe", "qwen3_coder"),
            ("qwen3_next", "qwen3_xml"),
            ("deepseek_v3", "deepseek_v3"),
            ("llama4", "llama4_json"),
            ("llama", "llama3_json"),
            ("mistral", "mistral"),
            ("mistral3", "mistral"),
            ("gemma4", "gemma4"),
            ("glm4_moe", "glm47"),
            ("olmo3", "olmo3"),
            ("phi4", "phi4_mini_json"),
            ("kimi_k2", "kimi_k2"),
        ],
    )
    def test_known_model_types(self, model_type, expected):
        assert detect_tool_parser(model_type=model_type) == expected

    def test_case_insensitive(self):
        assert detect_tool_parser(model_type="QWEN3") == "hermes"
        assert detect_tool_parser(model_type="Mistral") == "mistral"

    def test_unknown_model_type_returns_default(self):
        assert detect_tool_parser(model_type="completely_unknown_model") == _DEFAULT_PARSER

    def test_prefix_matching_longest_first(self):
        assert detect_tool_parser(model_type="qwen3_5_moe") == "qwen3_coder"

    def test_model_type_with_suffix(self):
        result = detect_tool_parser(model_type="qwen3_custom_variant")
        assert result == "hermes"


class TestDetectToolParserChatTemplate:
    @pytest.mark.parametrize(
        "hint,expected",
        _TEMPLATE_HINTS,
    )
    def test_template_hints(self, hint, expected):
        tokenizer = _FakeTokenizer(chat_template=f"some template {hint} more text")
        assert detect_tool_parser(tokenizer=tokenizer) == expected

    def test_empty_template_falls_to_vocab(self):
        tokenizer = _FakeTokenizer(chat_template="", vocab={"<tool_call>": 100})
        assert detect_tool_parser(tokenizer=tokenizer) == "hermes"

    def test_none_template_falls_to_vocab(self):
        tokenizer = _FakeTokenizer(chat_template=None, vocab={"<tool_call>": 100})
        assert detect_tool_parser(tokenizer=tokenizer) == "hermes"


class TestDetectToolParserVocab:
    @pytest.mark.parametrize(
        "token,expected",
        _VOCAB_HINTS,
    )
    def test_vocab_hints(self, token, expected):
        tokenizer = _FakeTokenizer(vocab={token: 100})
        assert detect_tool_parser(tokenizer=tokenizer) == expected


class TestDetectToolParserFallback:
    def test_no_args_returns_default(self):
        assert detect_tool_parser() == _DEFAULT_PARSER

    def test_empty_tokenizer_returns_default(self):
        tokenizer = _FakeTokenizer()
        assert detect_tool_parser(tokenizer=tokenizer) == _DEFAULT_PARSER

    def test_tokenizer_without_get_vocab_returns_default(self):
        class BareTokenizer:
            chat_template = ""

        assert detect_tool_parser(tokenizer=BareTokenizer()) == _DEFAULT_PARSER


class TestModelTypeMapping:
    def test_all_values_are_strings(self):
        for key, value in MODEL_TYPE_TO_TOOL_PARSER.items():
            assert isinstance(key, str), f"Key {key} is not a string"
            assert isinstance(value, str), f"Value {value} for key {key} is not a string"

    def test_no_empty_keys(self):
        for key in MODEL_TYPE_TO_TOOL_PARSER:
            assert key.strip(), "Empty key found in MODEL_TYPE_TO_TOOL_PARSER"
