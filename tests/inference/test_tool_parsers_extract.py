import json

import pytest

from easydel.inference.openai_api_modules import ChatCompletionRequest, ChatMessage
from easydel.inference.tools.abstract_tool import ToolParserManager
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
