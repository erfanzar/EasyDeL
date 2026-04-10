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

import json

import pytest

from easydel.inference.openai_api_modules import ChatCompletionRequest, ChatMessage
from easydel.inference.tools.abstract_tool import ToolParserManager
from easydel.inference.tools.auto_detect import detect_tool_parser
from easydel.inference.tools.parsers import (
    DeepSeekV31ToolParser,
    DeepSeekV32ToolParser,
    Ernie45ToolParser,
    FunctionGemmaToolParser,
    GigaChat3ToolParser,
    Glm47MoeModelToolParser,
    LongcatFlashToolParser,
    MinimaxM2ToolParser,
    Olmo3PythonicToolParser,
    OpenAIToolParser,
    Qwen3CoderToolParser,
    Qwen3XMLToolParser,
    xLAMToolParser,
)


class _DummyTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._vocab:
            return [self._vocab[text]]
        token_id = len(self._vocab) + 1
        self._vocab[text] = token_id
        self._id_to_token[token_id] = text
        return [token_id]

    def decode(self, token_ids):
        return "".join(self._id_to_token.get(i, "") for i in token_ids)


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="dummy", messages=[ChatMessage(role="user", content="hi")])


def _make_request_with_tools(tools: list[dict]) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
        }
    )


@pytest.fixture()
def dummy_tokenizer():
    vocab = {
        "<tool_call>": 1,
        "</tool_call>": 2,
        "<longcat_tool_call>": 3,
        "</longcat_tool_call>": 4,
        "<minimax:tool_call>": 5,
        "</minimax:tool_call>": 6,
        "<｜tool▁calls▁begin｜>": 7,
        "<｜tool▁calls▁end｜>": 8,
        "<｜tool▁call▁begin｜>": 9,
        "<｜tool▁call▁end｜>": 10,
        "<0x0A>": 11,
        "</think>": 12,
        "<response>": 13,
        "</response>": 14,
    }
    return _DummyTokenizer(vocab)


def test_tool_parser_manager_includes_new_parsers():
    for name in [
        "deepseek_v31",
        "deepseek_v32",
        "ernie45",
        "functiongemma",
        "gemma4",
        "gigachat3",
        "glm47",
        "glm-4.7",
        "longcat",
        "minimax_m2",
        "olmo3",
        "openai",
        "qwen3_xml",
        "xlam",
    ]:
        ToolParserManager.get_tool_parser(name)


def test_detect_tool_parser_prefers_qwen3_xml_for_qwen3_family():
    assert detect_tool_parser(model_type="qwen3") == "hermes"
    assert detect_tool_parser(model_type="qwen3_moe") == "hermes"
    assert detect_tool_parser(model_type="qwen3_next") == "qwen3_xml"
    assert detect_tool_parser(model_type="qwen3_5_text") == "qwen3_coder"
    assert detect_tool_parser(model_type="qwen3_5_moe_text") == "qwen3_coder"


def test_deepseek_v31_extract_tool_calls(dummy_tokenizer):
    parser = DeepSeekV31ToolParser(dummy_tokenizer)
    request = _make_request()
    output = (
        "hello "
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert extracted.content == "hello "
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"x": 1}
    assert extracted.tool_calls[0].id


def test_deepseek_v32_extract_tool_calls(dummy_tokenizer):
    parser = DeepSeekV32ToolParser(dummy_tokenizer)
    request = _make_request()
    output = (
        "hi "
        "<｜DSML｜function_calls>"
        '<｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>'
        '<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜function_calls>"
    )
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert extracted.content == "hi "
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"location": "杭州", "date": "2024-01-16"}
    assert extracted.tool_calls[0].id


def test_ernie45_extract_tool_calls(dummy_tokenizer):
    parser = Ernie45ToolParser(dummy_tokenizer)
    request = _make_request()
    output = '</think>\n<tool_call>{"name":"foo","arguments":{"a":1}}</tool_call>\n'
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert extracted.content == "</think>"
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"a": 1}
    assert extracted.tool_calls[0].id


def test_functiongemma_extract_tool_calls(dummy_tokenizer):
    parser = FunctionGemmaToolParser(dummy_tokenizer)
    request = _make_request()
    output = "<start_function_call>call:foo{a:<escape>1<escape>}<end_function_call>"
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"a": 1}
    assert extracted.tool_calls[0].id


def test_gigachat3_extract_tool_calls(dummy_tokenizer):
    parser = GigaChat3ToolParser(dummy_tokenizer)
    request = _make_request()
    output = 'hi\nfunction call{"name":"foo","arguments":{"x":1}}'
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert extracted.content == "hi"
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"x": 1}
    assert extracted.tool_calls[0].id


def test_glm47_extract_tool_calls(dummy_tokenizer):
    parser = Glm47MoeModelToolParser(dummy_tokenizer)
    request = _make_request()
    output = "<tool_call>foo<arg_key>a</arg_key>\n<arg_value>1</arg_value></tool_call>"
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"a": 1}
    assert extracted.tool_calls[0].id


def test_longcat_extract_tool_calls(dummy_tokenizer):
    parser = LongcatFlashToolParser(dummy_tokenizer)
    request = _make_request()
    output = '<longcat_tool_call>{"name":"foo","arguments":{"x":1}}</longcat_tool_call>'
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"x": 1}
    assert extracted.tool_calls[0].id


def test_minimax_m2_extract_tool_calls(dummy_tokenizer):
    parser = MinimaxM2ToolParser(dummy_tokenizer)
    request = _make_request()
    output = (
        'prefix <minimax:tool_call><invoke name="foo"><parameter name="x">1</parameter></invoke></minimax:tool_call>'
    )
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert extracted.content == "prefix "
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"x": "1"}
    assert extracted.tool_calls[0].id


def test_olmo3_extract_tool_calls(dummy_tokenizer):
    parser = Olmo3PythonicToolParser(dummy_tokenizer)
    request = _make_request()
    output = "<function_calls>\nfoo(a=1)\n</function_calls>"
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert extracted.content is None
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"a": 1}
    assert extracted.tool_calls[0].id


def test_openai_extract_tool_calls(dummy_tokenizer):
    parser = OpenAIToolParser(dummy_tokenizer)
    request = _make_request()
    output = 'preamble\n```json\n[{"name":"foo","arguments":{"x":1}}]\n```'
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert extracted.content == "preamble"
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"x": 1}
    assert extracted.tool_calls[0].id


def test_qwen3xml_extract_tool_calls(dummy_tokenizer):
    parser = Qwen3XMLToolParser(dummy_tokenizer)
    request = _make_request()
    output = "<tool_call><function=foo><parameter=a>1</parameter></function></tool_call>"
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"a": "1"}
    assert extracted.tool_calls[0].id


def test_hermes_does_not_parse_qwen_xml_but_qwen3xml_does(dummy_tokenizer):
    request = _make_request()
    output = (
        "<tool_call><function=read>"
        "<parameter=filePath>/Users/erfan/Documents/Projects/Calute/README.md</parameter>"
        "<parameter=offset>1</parameter>"
        "</function></tool_call>"
    )

    hermes = ToolParserManager.get_tool_parser("hermes")(dummy_tokenizer)
    qwen = Qwen3XMLToolParser(dummy_tokenizer)

    hermes_extracted = hermes.extract_tool_calls(output, request)
    qwen_extracted = qwen.extract_tool_calls(output, request)

    assert hermes_extracted.tools_called is False
    assert qwen_extracted.tools_called is True
    assert qwen_extracted.tool_calls[0].function.name == "read"
    assert json.loads(qwen_extracted.tool_calls[0].function.arguments) == {
        "filePath": "/Users/erfan/Documents/Projects/Calute/README.md",
        "offset": "1",
    }


def test_qwen3xml_extract_tool_calls_without_wrapper_and_with_named_attrs(dummy_tokenizer):
    parser = Qwen3XMLToolParser(dummy_tokenizer)
    request = _make_request_with_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                },
            }
        ]
    )
    output = (
        '<function name="lookup"><parameter=limit>5</parameter><parameter name="enabled">true</parameter></function>'
    )

    extracted = parser.extract_tool_calls(output, request)

    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "lookup"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"limit": 5, "enabled": True}


def test_qwen3xml_streaming_emits_completed_bare_function(dummy_tokenizer):
    parser = Qwen3XMLToolParser(dummy_tokenizer)
    request = _make_request()

    first = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="preface <function=lookup>",
        delta_text="preface <function=lookup>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )
    second = parser.extract_tool_calls_streaming(
        previous_text="preface <function=lookup>",
        current_text="preface <function=lookup><parameter=query>AI</parameter></function>",
        delta_text="<parameter=query>AI</parameter></function>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert first is not None
    assert first.content == "preface "
    assert second is not None
    assert second.tool_calls is not None
    assert second.tool_calls[0].function.name == "lookup"
    assert json.loads(second.tool_calls[0].function.arguments) == {"query": "AI"}


def test_qwen3coder_extract_tool_calls_converts_types_like_vllm(dummy_tokenizer):
    parser = Qwen3CoderToolParser(dummy_tokenizer)
    request = _make_request_with_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                            "return_metadata": {"type": "boolean"},
                            "filters": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                            "tags": {"type": "array"},
                        },
                    },
                },
            }
        ]
    )
    output = (
        "<tool_call><function=lookup>"
        "<parameter=query>\nAI\n</parameter>"
        "<parameter=limit>\n5\n</parameter>"
        "<parameter=return_metadata>\ntrue\n</parameter>"
        '<parameter=filters>\n{"region":"us-en"}\n</parameter>'
        '<parameter=tags>\n["news","ml"]\n</parameter>'
        "</function></tool_call>"
    )

    extracted = parser.extract_tool_calls(output, request)

    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 1
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "query": "AI",
        "limit": 5,
        "return_metadata": True,
        "filters": {"region": "us-en"},
        "tags": ["news", "ml"],
    }


def test_qwen3coder_extract_tool_calls_filters_invalid_partial_functions(dummy_tokenizer):
    parser = Qwen3CoderToolParser(dummy_tokenizer)
    request = _make_request()
    output = "<tool_call><function=lookup</function></tool_call>"

    extracted = parser.extract_tool_calls(output, request)

    assert extracted.tools_called is False
    assert extracted.tool_calls == []
    assert extracted.content is None


def test_qwen3coder_streaming_initializes_stream_buffers(dummy_tokenizer):
    parser = Qwen3CoderToolParser(dummy_tokenizer)
    request = _make_request()
    first_delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<tool_call>",
        delta_text="<tool_call>",
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
        request=request,
    )
    delta = parser.extract_tool_calls_streaming(
        previous_text="<tool_call>",
        current_text="<tool_call><function=lookup>",
        delta_text="<function=lookup>",
        previous_token_ids=[1],
        current_token_ids=[1],
        delta_token_ids=[],
        request=request,
    )

    assert first_delta is None
    assert delta is not None
    assert delta.tool_calls is not None
    assert delta.tool_calls[0].function.name == "lookup"
    assert parser.prev_tool_call_arr == [{"name": "lookup", "arguments": "{}"}]
    assert parser.streamed_args_for_tool == [""]


def test_deepseek_v32_streaming_buffers_until_complete_invoke(dummy_tokenizer):
    parser = DeepSeekV32ToolParser(dummy_tokenizer)
    request = _make_request_with_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer"},
                        },
                    },
                },
            }
        ]
    )

    first = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="hi <｜DSML｜function_calls>",
        delta_text="hi <｜DSML｜function_calls>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )
    second = parser.extract_tool_calls_streaming(
        previous_text="hi <｜DSML｜function_calls>",
        current_text=(
            "hi <｜DSML｜function_calls>"
            '<｜DSML｜invoke name="lookup">'
            '<｜DSML｜parameter name="limit" string="true">5</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
        ),
        delta_text=(
            '<｜DSML｜invoke name="lookup">'
            '<｜DSML｜parameter name="limit" string="true">5</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
        ),
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert first is not None
    assert first.content == "hi "
    assert first.tool_calls in (None, [])
    assert second is not None
    assert second.tool_calls is not None
    assert second.tool_calls[0].function.name == "lookup"
    assert json.loads(second.tool_calls[0].function.arguments) == {"limit": 5}


def test_xlam_extract_tool_calls(dummy_tokenizer):
    parser = xLAMToolParser(dummy_tokenizer)
    request = _make_request()
    output = '[{"name":"foo","arguments":{"x":1}}]'
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called is True
    assert extracted.content is None
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "foo"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"x": 1}
    assert extracted.tool_calls[0].id
