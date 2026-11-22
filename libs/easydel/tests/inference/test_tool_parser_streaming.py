"""Public parser regressions for coalesced chunks, nested data and schema membership."""

from __future__ import annotations

import inspect
import json
import re
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin
from easydel.inference.vwhisper import generation
from easydel.inference.vwhisper.config import vWhisperInferenceConfig
from easydel.inference.vwhisper.core import vWhisperInference

from .test_implicit_union_schema import _XMLTokenizer
from .test_tool_parsers import _parser, _request

pytestmark = pytest.mark.cpu_ok


def _stream(parser, chunks, request=None, finalize=False):
    request = request or _request()
    if finalize:
        host = ToolCallingMixin()
        host.tool_parsers = {"m": parser}
    names, arguments, ids = [], {}, {}
    content = ""
    previous = ""
    for index, chunk in enumerate(chunks):
        kwargs = dict(
            previous_text=previous,
            current_text=previous + chunk,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if finalize:
            delta = host.extract_tool_calls_streaming(model_name="m", finished=index == len(chunks) - 1, **kwargs)
        else:
            delta = parser.extract_tool_calls_streaming(**kwargs)
        previous += chunk
        content += (delta.content or "") if delta else ""
        for call in (delta.tool_calls if delta else None) or []:
            if call.id:
                assert call.index not in ids
                ids[call.index] = call.id
            if call.function:
                if call.function.name:
                    names.append((call.index, call.function.name))
                arguments[call.index] = arguments.get(call.index, "") + (call.function.arguments or "")
    return names, {index: json.loads(value) for index, value in arguments.items()}, content


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
@pytest.mark.parametrize("chunking", ["coalesced-end", "whole", "characters"])
def test_seed_coalesced_parameters_and_optional_thinking(thinking, finalize, chunking):
    prefix = ["<seed:think>", "x", "</seed:think>"] if thinking else []
    call = (
        "<seed:tool_call><function=f>\n<parameter=x>v</parameter><parameter=y>w</parameter></function></seed:tool_call>"
    )
    if chunking == "whole":
        chunks = [*prefix, call]
    elif chunking == "characters":
        chunks = prefix + list(call)
    else:
        chunks = [
            *prefix,
            "<seed:tool_call>",
            "<function=f>",
            "\n",
            "<parameter=x>v</parameter><parameter=y>w</parameter></function>",
            "</seed:tool_call>",
        ]
    names, args, content = _stream(_parser("seed_oss"), chunks, finalize=finalize)
    assert names == [(0, "f")]
    assert args == {0: {"x": "v", "y": "w"}}
    assert "<seed:tool_call>" not in content
    assert "<parameter=" not in content


def test_minimax_m2_multiple_envelopes_do_not_replay_calls():
    text = "".join(f'<minimax:tool_call><invoke name="{name}">\n</invoke>\n</minimax:tool_call>' for name in ("f", "g"))
    chunks = []
    for part in re.split(r"(</?minimax:tool_call>)", text):
        chunks.extend([part] if re.fullmatch(r"</?minimax:tool_call>", part) else list(part))
    names, args, _ = _stream(_parser("minimax_m2"), chunks)
    assert names == [(0, "f"), (1, "g")]
    assert args == {0: {}, 1: {}}


@pytest.mark.parametrize("parser_name", ["openai", "xlam"])
@pytest.mark.parametrize("chunking", ["whole", "characters"])
def test_json_stream_announces_every_top_level_call_only(parser_name, chunking):
    text = json.dumps(
        [{"name": "f", "arguments": {"name": "Alice", "deep": {"name": "Bob"}}}, {"name": "g", "arguments": {}}]
    )
    chunks = [text] if chunking == "whole" else list(text)
    names, args, _ = _stream(_parser(parser_name), chunks + [" "] * 12)
    assert names == [(0, "f"), (1, "g")]
    assert args == {0: {"name": "Alice", "deep": {"name": "Bob"}}, 1: {}}


@pytest.mark.parametrize("parser_name", ["openai", "xlam"])
def test_json_stream_partial_strings_and_numeric_exponents(parser_name):
    chunks = ['[{"name":"f","arguments":{"text":"hello', ' world", "n":1', 'e-2, "more":"x"}}]']
    names, args, _ = _stream(_parser(parser_name), chunks)
    assert names == [(0, "f")]
    assert args == {0: {"text": "hello world", "n": 0.01, "more": "x"}}


@pytest.mark.parametrize("text", ['{"answer":42}', 'Before ```json\n{"answer":42}\n``` after'])
def test_openai_non_tool_json_is_preserved(text):
    result = _parser("openai").extract_tool_calls(text, _request())
    assert not result.tools_called
    assert result.content == text


@pytest.mark.parametrize("value", [{"config": {"x": 1}}, {"code": 'a}; b{; "quoted"'}])
def test_llama_nested_json_and_quoted_braces(value):
    text = json.dumps({"name": "f", "arguments": value}) + ";" + json.dumps({"name": "g", "arguments": {}})
    result = _parser("llama3_json").extract_tool_calls(text, _request())
    assert result.tools_called
    assert [call.function.name for call in result.tool_calls] == ["f", "g"]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [value, {}]


@pytest.mark.parametrize("parser_name", ["pythonic", "olmo3", "llama4_pythonic"])
@pytest.mark.parametrize("value", ["]", "[({})]", "a\\", 'say "yes" ]', "a\\\\"])
@pytest.mark.parametrize("chunking", ["whole", "characters"])
def test_pythonic_string_brackets_and_escapes(parser_name, value, chunking):
    expression = "f(x=" + repr(value) + ")"
    text = "<function_calls>" + expression + "</function_calls>" if parser_name == "olmo3" else "[" + expression + "]"
    batch = _parser(parser_name).extract_tool_calls(text, _request())
    assert batch.tools_called
    assert json.loads(batch.tool_calls[0].function.arguments) == {"x": value}
    chunks = [text] if chunking == "whole" else list(text)
    names, args, _ = _stream(_parser(parser_name), chunks)
    assert names == [(0, "f")]
    assert args == {0: {"x": value}}


def test_minimax_batch_decodes_pretty_printed_and_multiple_values():
    text = (
        "<tool_calls>"
        + "\n".join(json.dumps({"name": n, "arguments": {"x": 1}}, indent=2) for n in ("f", "g"))
        + "</tool_calls>"
    )
    result = _parser("minimax").extract_tool_calls(text, _request())
    assert result.tools_called
    assert [call.function.name for call in result.tool_calls] == ["f", "g"]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [{"x": 1}, {"x": 1}]


def test_minimax_non_tool_region_is_preserved():
    text = '<tool_calls>{"answer":42}</tool_calls>'
    result = _parser("minimax").extract_tool_calls(text, _request())
    assert not result.tools_called
    assert result.content == text


@pytest.mark.parametrize("limit,expected", [(7, 7), (None, 448)])
def test_vwhisper_forwards_resolved_generation_limit(monkeypatch, limit, expected):
    """Exercise the public engine configuration at its generation boundary without JAX arrays/JIT."""

    def force_generate(**kwargs):
        config = kwargs["generation_config"]
        return kwargs.get("max_length", config.max_length)

    model_config = SimpleNamespace(max_length=448)
    model = SimpleNamespace(generation_config=model_config, mesh=nullcontext(), _force_generate=force_generate)
    monkeypatch.setattr(generation.spx, "export", lambda model: ("graph", "state"))
    monkeypatch.setattr(generation.spx, "bind", lambda graph, state: model)
    engine = vWhisperInference(
        model=model,
        tokenizer=_XMLTokenizer(),
        processor=SimpleNamespace(feature_extractor=object()),
        inference_config=vWhisperInferenceConfig(max_length=limit),
    )
    result = inspect.unwrap(generation._compiled_generate)(
        engine.graphdef, engine.graphstate, engine.inference_config, "features", [], False
    )
    assert result == expected
    assert engine.max_length == expected
    assert model_config.max_length == 448
