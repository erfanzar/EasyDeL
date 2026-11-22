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

"""Regression tests for GigaChat3 streaming, DeepSeek-V3.2 typing and tool-parser auto-detection."""

from __future__ import annotations

import json

import pytest
from easydel.inference.tools.abstract_tool import ToolParserManager
from easydel.inference.tools.auto_detect import detect_tool_parser
from easydel.inference.tools.parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from easydel.inference.tools.parsers.gigachat3_tool_parser import GigaChat3ToolParser

from ._helpers import DummyTokenizer, function_tool, make_request, stream

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "header_chunks",
    [
        ["function call", "<|role_sep|>", "\n"],
        ["function call", "\n"],
        ["function call", "<|role_sep|>"],
        ["function call"],
    ],
)
def test_gigachat3_streaming_does_not_leak_protocol_prefix(header_chunks):
    parser = GigaChat3ToolParser(DummyTokenizer())
    chunks = [*header_chunks, '{"name": "f", "arguments": {"a": 1}}']
    deltas = stream(parser, chunks, make_request())
    leaked = "".join(d.content or "" for d in deltas)
    assert leaked == ""
    names = [tc.function.name for d in deltas for tc in (d.tool_calls or []) if tc.function and tc.function.name]
    assert names == ["f"]


def test_gigachat3_streaming_still_flushes_ordinary_text():
    parser = GigaChat3ToolParser(DummyTokenizer())
    deltas = stream(parser, ["function call", " is a phrase"], make_request())
    assert "".join(d.content or "" for d in deltas) == "function call is a phrase"


DSML_CALL = (
    "<｜DSML｜function_calls>"
    '<｜DSML｜invoke name="f">'
    '<｜DSML｜parameter name="n" string="false">3</｜DSML｜parameter>'
    '<｜DSML｜parameter name="flag" string="false">true</｜DSML｜parameter>'
    '<｜DSML｜parameter name="s" string="true">7</｜DSML｜parameter>'
    "</｜DSML｜invoke>"
    "</｜DSML｜function_calls>"
)


def test_deepseek_v32_batch_extraction_applies_schema_types():
    request = make_request(
        [function_tool("f", {"n": {"type": "integer"}, "flag": {"type": "boolean"}, "s": {"type": "string"}})]
    )
    parser = DeepSeekV32ToolParser(DummyTokenizer())
    result = parser.extract_tool_calls(DSML_CALL, request)
    assert result.tools_called is True
    assert json.loads(result.tool_calls[0].function.arguments) == {"n": 3, "flag": True, "s": "7"}


def test_deepseek_v32_explicit_dsml_types_work_without_schema():
    request = make_request()
    batch = DeepSeekV32ToolParser(DummyTokenizer()).extract_tool_calls(DSML_CALL, request)
    deltas = stream(DeepSeekV32ToolParser(DummyTokenizer()), [DSML_CALL], request)
    expected = {"n": 3, "flag": True, "s": "7"}
    assert json.loads(batch.tool_calls[0].function.arguments) == expected
    assert json.loads(deltas[0].tool_calls[0].function.arguments) == expected


def test_deepseek_v32_string_null_is_not_coerced_to_json_null():
    output = DSML_CALL.replace(">7</｜DSML｜parameter>", ">null</｜DSML｜parameter>")
    request = make_request([function_tool("f", {"s": {"type": "string"}})])
    batch = DeepSeekV32ToolParser(DummyTokenizer()).extract_tool_calls(output, request)
    assert json.loads(batch.tool_calls[0].function.arguments)["s"] == "null"


def test_deepseek_v32_batch_and_streaming_agree():
    request = make_request([function_tool("f", {"n": {"type": "integer"}, "flag": {"type": "boolean"}})])
    batch = DeepSeekV32ToolParser(DummyTokenizer()).extract_tool_calls(DSML_CALL, request)
    deltas = stream(DeepSeekV32ToolParser(DummyTokenizer()), [DSML_CALL], request)
    streamed = [tc for d in deltas for tc in (d.tool_calls or [])]
    assert len(streamed) == 1
    assert json.loads(streamed[0].function.arguments) == json.loads(batch.tool_calls[0].function.arguments)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("deepseek_v32", "deepseek_v32"),
        ("deepseek_v31", "deepseek_v31"),
        ("deepseek_v3", "deepseek_v3"),
        ("deepseek_v2", "deepseek_v3"),
    ],
)
def test_detect_tool_parser_resolves_deepseek_versions(model_type, expected):
    name = detect_tool_parser(model_type=model_type)
    assert name == expected
    ToolParserManager.get_tool_parser(name)


class _TemplateTokenizer:
    def __init__(self, chat_template: str = "", vocab: dict[str, int] | None = None):
        self.chat_template = chat_template
        self._vocab = vocab or {}

    def get_vocab(self):
        return dict(self._vocab)


def test_detect_tool_parser_uses_fullwidth_deepseek_markers():
    marker = "<｜tool▁calls▁begin｜>"
    assert detect_tool_parser(tokenizer=_TemplateTokenizer(chat_template=f"x {marker} y")) == "deepseek_v3"
    assert detect_tool_parser(tokenizer=_TemplateTokenizer(vocab={marker: 7})) == "deepseek_v3"
    dsml = "<｜DSML｜function_calls>"
    assert detect_tool_parser(tokenizer=_TemplateTokenizer(chat_template=dsml)) == "deepseek_v32"
