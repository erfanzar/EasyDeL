"""Parser and wire-stream regressions (no array computation)."""

import json
from types import SimpleNamespace

import pytest
from easydel.inference.openai_api_modules import DeltaMessage
from easydel.inference.parsing.delegating_parser import DelegatingParser
from easydel.inference.stream_protocol import (
    coerce_stream_delta_message,
    iter_chat_completion_stream_responses,
    iter_responses_stream_frames,
)
from easydel.inference.tools.parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser
from easydel.inference.tools.parsers.gemma4_tool_parser import Gemma4ToolParser
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser
from easydel.inference.tools.parsers.hunyuan_a13b_tool_parser import HunyuanA13BToolParser
from easydel.inference.tools.parsers.kimi_k2_tool_parser import KimiK2ToolParser
from easydel.inference.tools.parsers.llama4_pythonic_tool_parser import Llama4PythonicToolParser

from ._helpers import DummyTokenizer, function_tool, make_request, stream

pytestmark = pytest.mark.cpu_ok


def _calls(deltas):
    calls = {}
    for delta in deltas:
        for call in delta.tool_calls or []:
            state = calls.setdefault(call.index, {"name": None, "arguments": "", "id": None})
            if call.function.name:
                state["name"] = call.function.name
            if call.id:
                state["id"] = call.id
            state["arguments"] += call.function.arguments or ""
    return list(calls.values())


def test_kimi_adjacent_calls_have_separate_headers_and_multiline_arguments():
    tokenizer = DummyTokenizer({"<|tool_calls_section_begin|>": 1, "<|tool_calls_section_end|>": 2})
    parser = KimiK2ToolParser(tokenizer)
    output = (
        '<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{"n":1}'
        '<|tool_call_end|><|tool_call_begin|>functions.g:1<|tool_call_argument_begin|>{\n"n":2\n}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )
    result = parser.extract_tool_calls(output, make_request())
    assert result.tools_called
    assert [(call.id, call.function.name, json.loads(call.function.arguments)) for call in result.tool_calls] == [
        ("functions.f:0", "f", {"n": 1}),
        ("functions.g:1", "g", {"n": 2}),
    ]


@pytest.mark.parametrize("delegated", [False, True])
def test_hunyuan_only_top_level_names_are_executable_and_all_arguments_arrive(delegated):
    output = (
        '<tool_calls>[{"name":"f","arguments":{"name":"nested","deep":{"name":"third"},'
        '"items":[{"name":"fourth"}],"text":"braces }{ and \\"quotes\\""}},'
        '{"arguments":{},"name":"g"}]</tool_calls>'
    )
    expected = [
        {"name": "nested", "deep": {"name": "third"}, "items": [{"name": "fourth"}], "text": 'braces }{ and "quotes"'},
        {},
    ]
    request = make_request([function_tool(name, {}) for name in ["f", "g", "nested", "third", "fourth"]])
    for chunks in ([output], list(output), *[[output[:i], output[i:]] for i in range(1, len(output))]):
        parser = HunyuanA13BToolParser(DummyTokenizer())
        if delegated:
            delegating = DelegatingParser(tool_parser=parser, tool_request=request)
            previous = ""
            deltas = []
            for chunk in chunks:
                current = previous + chunk
                result = delegating.process_delta(current, chunk, [], previous, [])
                deltas.append(DeltaMessage(tool_calls=result.delta_tool_calls))
                previous = current
            final = delegating.process_final(previous, [])
            deltas.append(DeltaMessage(tool_calls=final.delta_tool_calls))
        else:
            deltas = stream(parser, chunks, request)
        calls = _calls(deltas)
        assert [call["name"] for call in calls] == ["f", "g"], chunks
        assert [json.loads(call["arguments"]) for call in calls] == expected, chunks


def _deepseek_tokenizer():
    return DummyTokenizer(
        {
            token: i + 1
            for i, token in enumerate(
                ["<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>", "<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>"]
            )
        }
    )


def test_deepseek_multiline_calls_and_unrecognized_regions_preserve_content():
    parser = DeepSeekV3ToolParser(_deepseek_tokenizer())
    output = (
        "<｜tool▁calls▁begin｜>"
        + "".join(
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>" + name + "\n```json\n" + args + "\n```<｜tool▁call▁end｜>"
            for name, args in [("f", '{\n  "items": [1, {"n": 2}]\n}'), ("g", '{\n "n": 3\n}')]
        )
        + "<｜tool▁calls▁end｜>"
    )
    result = parser.extract_tool_calls(output, make_request())
    assert [call.function.name for call in result.tool_calls] == ["f", "g"]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [{"items": [1, {"n": 2}]}, {"n": 3}]
    malformed = "<｜tool▁calls▁begin｜>not a call<｜tool▁calls▁end｜>"
    fallback = parser.extract_tool_calls(malformed, make_request())
    assert not fallback.tools_called
    assert fallback.content == malformed


def test_deepseek_multiline_stream_keeps_nested_json():
    tokenizer = _deepseek_tokenizer()
    parser = DeepSeekV3ToolParser(tokenizer)
    chunks = [
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "function<｜tool▁sep｜>f\n",
        "```json\n{\n",
        '  "items": [1,\n',
        ' {"n": 2}]\n}',
        "\n```",
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    ]
    previous, previous_ids, deltas = "", [], []
    for chunk in chunks:
        current_ids = previous_ids + tokenizer.encode(chunk)
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, current, chunk, previous_ids, current_ids, tokenizer.encode(chunk), make_request()
        )
        if delta:
            deltas.append(delta)
        previous, previous_ids = current, current_ids
    calls = _calls(deltas)
    assert [call["name"] for call in calls] == ["f"]
    assert json.loads(calls[0]["arguments"]) == {"items": [1, {"n": 2}]}


@pytest.mark.parametrize("literal", ["[", "]", "][", "{[]}"])
def test_gemma_nested_array_strings_are_not_nesting_tokens(literal):
    output = '<|tool_call>call:f{x:[[<|"|>' + literal + '<|"|>]]}<tool_call|>'
    batch = Gemma4ToolParser(DummyTokenizer()).extract_tool_calls(output, make_request())
    assert json.loads(batch.tool_calls[0].function.arguments) == {"x": [[literal]]}
    for chunks in ([output], list(output)):
        calls = _calls(stream(Gemma4ToolParser(DummyTokenizer()), chunks, make_request()))
        assert [call["name"] for call in calls] == ["f"]
        assert json.loads(calls[0]["arguments"]) == {"x": [[literal]]}


def test_llama_literal_protocol_delimiter_is_argument_data_and_fallback_is_verbatim():
    output = '<|python_start|>[f(x="literal <|python_end|> token")]<|python_end|>'
    result = Llama4PythonicToolParser(DummyTokenizer()).extract_tool_calls(output, make_request())
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": "literal <|python_end|> token"}
    calls = _calls(stream(Llama4PythonicToolParser(DummyTokenizer()), [output], make_request()))
    assert json.loads(calls[0]["arguments"]) == {"x": "literal <|python_end|> token"}
    invalid = "<|python_start|>[not valid python]<|python_end|>"
    fallback = Llama4PythonicToolParser(DummyTokenizer()).extract_tool_calls(invalid, make_request())
    assert not fallback.tools_called
    assert fallback.content == invalid


def _mixed_snapshot(text="intro ", delta_text="intro ", tools=True):
    tool_calls = [{"id": "call_f", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    return SimpleNamespace(
        delta_text=delta_text,
        accumulated_text=text,
        delta_reasoning_content=None,
        reasoning_content=None,
        delta_tool_calls=[dict(tool_calls[0], index=0)] if tools else None,
        tool_calls=tool_calls if tools else None,
        outputs=[],
        prompt_token_ids=[1, 2],
        num_generated_tokens=3,
        tokens_per_second=1.0,
        processing_time=1.0,
        first_token_time=0.1,
    )


def test_chat_sse_keeps_real_parser_certified_content_alongside_tool_delta():
    parser = HermesToolParser(DummyTokenizer())
    result = stream(parser, ['intro <tool_call>{"name":"f","arguments":{}}</tool_call>'], make_request())[0]
    assert result.content == "intro " and result.tool_calls
    assert coerce_stream_delta_message(result).content == "intro "
    snapshot = _mixed_snapshot()
    snapshot.delta_text, snapshot.delta_tool_calls = result.content, result.tool_calls
    chunks = list(iter_chat_completion_stream_responses(iter([snapshot]), model="dummy"))
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "intro "
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.parametrize("sequence", ["mixed", "before", "after", "missing_final_text"])
def test_responses_mixed_stream_completes_every_item_and_retains_text(sequence):
    if sequence == "mixed":
        snapshots = [_mixed_snapshot()]
    elif sequence in ("before", "missing_final_text"):
        snapshots = [_mixed_snapshot(tools=False), _mixed_snapshot(delta_text="")]
        if sequence == "missing_final_text":
            snapshots[-1].accumulated_text = ""
    else:
        snapshots = [_mixed_snapshot(text="", delta_text=""), _mixed_snapshot(delta_text="intro ")]
        snapshots[-1].delta_tool_calls = None
    frames = list(iter_responses_stream_frames(iter(snapshots), response_id="resp_test", model="dummy"))
    payloads = [frame.payload for frame in frames]
    completed = payloads[-1].response
    assert all(item.status == "completed" for item in completed.output)
    messages = [item for item in completed.output if item.type == "message"]
    assert len(messages) == 1
    assert messages[0].content[0].text == "intro "
    assert "".join(payload.delta for payload in payloads if payload.type == "response.output_text.delta") == "intro "
    assert [payload.text for payload in payloads if payload.type == "response.output_text.done"] == ["intro "]
    done_ids = {payload.item.id for payload in payloads if payload.type == "response.output_item.done"}
    assert done_ids == {item.id for item in completed.output}
