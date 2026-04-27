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

"""Tests for tool parsers not yet covered by ``test_tool_parsers_extract.py``.

Targets the three highest-leverage uncovered parsers from the audit:

* ``MistralToolParser`` (538 LoC) -- ``[TOOL_CALLS][...]`` JSON-array format
* ``Llama4PythonicToolParser`` (440 LoC) -- ``<|python_start|>[func(arg=v)]`` AST-based
* ``HunyuanA13BToolParser`` (517 LoC) -- ``<tool_calls>[{...}]</tool_calls>`` XML+JSON

Each parser's ``extract_tool_calls`` is tested for:

* Happy path: valid output yields ``tools_called=True``
* No tool token: returns ``tools_called=False`` and original content
* Malformed payload: returns ``tools_called=False`` (exception is caught and logged)
* Content-before-token: preserved in the result's ``content`` field
"""

from __future__ import annotations

import json

import pytest

from easydel.inference.openai_api_modules import ChatCompletionRequest, ChatMessage
from easydel.inference.tools.parsers import (
    HunyuanA13BToolParser,
    Llama4PythonicToolParser,
    MistralToolParser,
)


class _DummyTokenizer:
    """Minimal tokenizer that satisfies the ``ToolParser`` base-class interface."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab = dict(vocab or {})
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
    return ChatCompletionRequest(
        model="dummy",
        messages=[ChatMessage(role="user", content="hi")],
    )


@pytest.fixture()
def mistral_parser() -> MistralToolParser:
    """Build a parser whose tokenizer carries the ``[TOOL_CALLS]`` token in vocab."""
    return MistralToolParser(_DummyTokenizer({"[TOOL_CALLS]": 1}))


def test_mistral_parser_init_requires_tool_call_token():
    """Without ``[TOOL_CALLS]`` in vocab the constructor raises RuntimeError."""
    with pytest.raises(RuntimeError, match="tool call token"):
        MistralToolParser(_DummyTokenizer({}))


def test_mistral_extracts_single_tool_call(mistral_parser: MistralToolParser):
    output = '[TOOL_CALLS][{"name": "search", "arguments": {"q": "test"}}]'
    result = mistral_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "search"
    assert json.loads(call.function.arguments) == {"q": "test"}


def test_mistral_extracts_multiple_tool_calls(mistral_parser: MistralToolParser):
    output = (
        '[TOOL_CALLS]['
        '{"name": "search", "arguments": {"q": "x"}},'
        '{"name": "translate", "arguments": {"text": "y"}}'
        ']'
    )
    result = mistral_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert {c.function.name for c in result.tool_calls} == {"search", "translate"}


def test_mistral_returns_no_tool_when_token_absent(mistral_parser: MistralToolParser):
    """Plain text response without [TOOL_CALLS] -> tools_called=False, content preserved."""
    output = "Just a regular response with no tool call."
    result = mistral_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is False
    assert result.content == output
    assert result.tool_calls == []


def test_mistral_preserves_content_before_token(mistral_parser: MistralToolParser):
    """Text before [TOOL_CALLS] becomes the result's content field."""
    output = 'Let me search. [TOOL_CALLS][{"name": "search", "arguments": {}}]'
    result = mistral_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert result.content == "Let me search. "


def test_mistral_returns_no_tool_on_malformed_json(mistral_parser: MistralToolParser):
    """Malformed JSON inside [TOOL_CALLS] is caught by the broad except path."""
    output = '[TOOL_CALLS]{this is broken json'
    result = mistral_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is False
    assert result.tool_calls == []


def test_mistral_adjust_request_no_op_when_no_tools(mistral_parser: MistralToolParser):
    """Without tools defined, ``adjust_request`` is identity."""
    req = _make_request()
    out = mistral_parser.adjust_request(req)

    assert out is req


@pytest.fixture()
def llama4_pythonic_parser() -> Llama4PythonicToolParser:
    return Llama4PythonicToolParser(_DummyTokenizer({"<|python_start|>": 1, "<|python_end|>": 2}))


def test_llama4_pythonic_extracts_single_function_call(
    llama4_pythonic_parser: Llama4PythonicToolParser,
):
    output = '[search(q="hello")]'
    result = llama4_pythonic_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "search"
    assert json.loads(call.function.arguments) == {"q": "hello"}


def test_llama4_pythonic_extracts_multiple_function_calls(
    llama4_pythonic_parser: Llama4PythonicToolParser,
):
    output = '[search(q="hello"), translate(text="world", lang="es")]'
    result = llama4_pythonic_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    names = [c.function.name for c in result.tool_calls]
    assert names == ["search", "translate"]


def test_llama4_pythonic_strips_python_delimiter_tokens(
    llama4_pythonic_parser: Llama4PythonicToolParser,
):
    """The optional <|python_start|> and <|python_end|> wrappers are stripped before parsing."""
    output = '<|python_start|>[search(q="hi")]<|python_end|>'
    result = llama4_pythonic_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "search"


def test_llama4_pythonic_no_tool_call_when_format_invalid(
    llama4_pythonic_parser: Llama4PythonicToolParser,
):
    """Plain text that doesn't match the regex returns tools_called=False with content."""
    output = "This is a plain answer, no tools."
    result = llama4_pythonic_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is False
    assert result.content == output


def test_llama4_pythonic_handles_unparseable_ast_gracefully(
    llama4_pythonic_parser: Llama4PythonicToolParser,
):
    """Output that matches the regex but isn't a valid Python list of calls returns no tools."""

    output = "[func(a=)]"
    result = llama4_pythonic_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is False


@pytest.fixture()
def hunyuan_parser() -> HunyuanA13BToolParser:
    return HunyuanA13BToolParser(_DummyTokenizer({"<tool_calls>": 1, "</tool_calls>": 2}))


def test_hunyuan_extracts_single_tool_call(hunyuan_parser: HunyuanA13BToolParser):
    output = '<tool_calls>[{"name": "search", "arguments": {"q": "x"}}]</tool_calls>'
    result = hunyuan_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "search"


def test_hunyuan_extracts_multiple_tool_calls(hunyuan_parser: HunyuanA13BToolParser):
    output = (
        '<tool_calls>['
        '{"name": "search", "arguments": {"q": "x"}},'
        '{"name": "translate", "arguments": {"text": "y"}}'
        ']</tool_calls>'
    )
    result = hunyuan_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert len(result.tool_calls) == 2


def test_hunyuan_no_tool_when_tag_absent(hunyuan_parser: HunyuanA13BToolParser):
    output = "Plain text, no tool call tags."
    result = hunyuan_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is False
    assert result.content == output


def test_hunyuan_ignores_tool_calls_inside_think_block(hunyuan_parser: HunyuanA13BToolParser):
    """Per the parser's docstring, ``<tool_calls>`` inside ``<think>`` are filtered out."""
    output = (
        "<think>I might call <tool_calls>[{\"name\": \"search\", \"arguments\": {}}]</tool_calls></think>"
        " But I won't."
    )
    result = hunyuan_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is False


def test_hunyuan_preserves_content_before_tool_calls(hunyuan_parser: HunyuanA13BToolParser):
    """Text before <tool_calls> is preserved in the content field."""
    output = 'Looking up. <tool_calls>[{"name": "search", "arguments": {}}]</tool_calls>'
    result = hunyuan_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is True
    assert result.content is None or "Looking up." in result.content


def test_hunyuan_returns_no_tool_on_malformed_json(hunyuan_parser: HunyuanA13BToolParser):
    """Malformed JSON inside <tool_calls> is rejected by the parser."""
    output = "<tool_calls>not actually json</tool_calls>"
    result = hunyuan_parser.extract_tool_calls(output, _make_request())
    assert result.tools_called is False


def test_hunyuan_preprocess_returns_content_and_tool_payload(hunyuan_parser: HunyuanA13BToolParser):
    """``preprocess_model_output`` returns (content_before_tag, tool_calls_json)."""
    output = 'Hi there <tool_calls>[{"name": "search", "arguments": {}}]</tool_calls>'
    content, payload = hunyuan_parser.preprocess_model_output(output)
    assert content == "Hi there "
    assert payload is not None
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    assert parsed[0]["name"] == "search"


def test_hunyuan_preprocess_returns_none_payload_when_no_tag(hunyuan_parser: HunyuanA13BToolParser):
    output = "no tag here"
    content, payload = hunyuan_parser.preprocess_model_output(output)
    assert content == output
    assert payload is None
