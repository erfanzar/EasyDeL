"""Responses wire formats and lossless streaming contracts (no array computation)."""

import pytest
from easydel.inference.inference_engine_interface import BaseInferenceApiServer
from easydel.inference.logits_process import LogitsProcessorList
from easydel.inference.oai_proxies import InferenceApiRouter
from easydel.inference.openai_api_modules import ChatCompletionRequest, ResponsesRequest
from easydel.inference.parsing import DelegatingParser
from easydel.inference.reasoning import ReasoningParserManager
from easydel.inference.reasoning.auto_detect import make_reasoning_stripper
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

pytestmark = pytest.mark.cpu_ok


class TextTokenizer:
    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, token_ids):
        return "".join(map(chr, token_ids))


def _reasoning_parser(name):
    return ReasoningParserManager.get_reasoning_parser(name)(TextTokenizer())


def _stream_reasoning(name, chunks, delegating):
    parser = _reasoning_parser(name)
    if delegating:
        parser = DelegatingParser(reasoning_parser=parser)
    previous, reasoning, content = "", "", ""
    for chunk in chunks:
        current = previous + chunk
        if delegating:
            delta = parser.process_delta(current, chunk, [], previous, [])
            reasoning += delta.delta_reasoning or ""
            content += delta.delta_content or ""
        else:
            delta = parser.extract_reasoning_streaming(previous, current, chunk, [], [], [])
            if delta:
                reasoning += delta.reasoning_content or ""
                content += delta.content or ""
        previous = current
    if delegating:
        delta = parser.process_final(previous, [])
        reasoning += delta.delta_reasoning or ""
        content += delta.delta_content or ""
        assert (delta.accumulated_reasoning, delta.accumulated_content) == (reasoning, content)
        again = parser.process_final(previous, [])
        assert not again.delta_reasoning and not again.delta_content
    else:
        delta = parser.finalize_streaming(previous)
        if delta:
            reasoning += delta.reasoning_content or ""
            content += delta.content or ""
        assert parser.finalize_streaming(previous) is None
    return reasoning, content


def test_responses_flat_tools_roundtrip_and_chat_boundary():
    wire = {"type": "function", "name": "f", "parameters": {"type": "object"}, "strict": True}
    request = ResponsesRequest(model="m", tools=[wire], tool_choice={"type": "function", "name": "f"})
    params = object.__new__(InferenceApiRouter).build_oai_params_from_responses_request(request)
    assert params["tools"] == [wire]
    assert params["tool_choice"] == {"type": "function", "name": "f"}
    raw, chat = BaseInferenceApiServer._extract_responses_tools(request)
    assert raw[0].function.name == "f"
    assert chat == [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}, "strict": True}}]
    nested = ResponsesRequest(model="m", tools=chat)
    assert nested.tools[0].function.name == "f"


def test_responses_mixed_conversation_preserves_roles_ids_and_arguments():
    request = ResponsesRequest(
        model="m",
        input=[
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "call1", "name": "f", "arguments": '{"x":1}'},
            {"type": "function_call", "call_id": "call2", "name": "g", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call1", "output": "ok"},
            {"type": "function_call_output", "call_id": "call2", "output": 0},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ],
    )
    messages = BaseInferenceApiServer._responses_payload_to_messages(request)
    assert [message.role for message in messages] == ["user", "assistant", "tool", "tool", "assistant"]
    assert messages[1].tool_calls == [
        {"id": "call1", "type": "function", "function": {"name": "f", "arguments": '{"x":1}'}},
        {"id": "call2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
    ]
    assert messages[2].tool_call_id == "call1" and messages[2].content == "ok"
    assert messages[3].tool_call_id == "call2" and messages[3].content == "0"
    assert messages[4].content == [{"type": "text", "text": "done"}]


@pytest.mark.parametrize("name", ["minimax_m2", "minimax_m2_append_think", "step3", "step3p5"])
@pytest.mark.parametrize("delegating", [False, True])
def test_asymmetric_truncated_reasoning_never_becomes_visible(name, delegating):
    # Stateless batch callers cannot infer reasoning from unmarked text.
    expected = ("secret", None) if name == "minimax_m2_append_think" else (None, "secret")
    assert _reasoning_parser(name).extract_reasoning("secret") == expected
    if name != "minimax_m2_append_think":
        final = DelegatingParser(reasoning_parser=_reasoning_parser(name)).process_final("secret", [])
        assert (final.accumulated_reasoning, final.accumulated_content) == ("", "secret")
    assert _stream_reasoning(name, ["sec", "ret"], delegating) == ("secret", "")
    assert _stream_reasoning(name, ["secret</thi", "nk>answer"], delegating) == ("secret", "answer")
    text = "<think>secret</think>answer"
    assert _stream_reasoning(name, list(text), delegating) == ("secret", "answer")
    assert _stream_reasoning(name, ["secret</thi"], delegating) == ("secret</thi", "")


@pytest.mark.parametrize("name", ["minimax_m2", "minimax_m2_append_think", "step3", "step3p5"])
@pytest.mark.parametrize("delegating", [False, True])
@pytest.mark.parametrize(
    "prefix", sorted({marker[:end] for marker in ("<think>", "</think>") for end in range(1, len(marker))})
)
def test_asymmetric_fully_buffered_prefix_finalizes_as_reasoning(name, delegating, prefix):
    parser = _reasoning_parser(name)
    assert parser.extract_reasoning_streaming("", prefix, prefix, [], [], []) is None
    assert _stream_reasoning(name, [prefix], delegating) == (prefix, "")
    assert _stream_reasoning(name, list(prefix), delegating) == (prefix, "")
    if name != "minimax_m2_append_think":
        final = DelegatingParser(reasoning_parser=_reasoning_parser(name)).process_final(prefix, [])
        assert (final.accumulated_reasoning, final.accumulated_content) == ("", prefix)


@pytest.mark.parametrize("name", ["minimax_m2", "step3", "step3p5"])
def test_asymmetric_prompt_context_reset_restores_terminal_only_content(name):
    parser = _reasoning_parser(name)
    assert parser.extract_reasoning_streaming("", "</thi", "</thi", [], [], []) is None
    assert parser.extract_reasoning_final("</thi") == ("</thi", None)
    parser.configure_prompt_context("ordinary prompt", [])
    assert parser.extract_reasoning_final("</thi") == (None, "</thi")
    parser.configure_prompt_context("prompt<think>", [])
    assert parser.extract_reasoning_final("</thi") == ("</thi", None)


@pytest.mark.parametrize("delegating", [False, True])
@pytest.mark.parametrize("step", [1, 2, 4, 1000])
@pytest.mark.parametrize(
    "name, text",
    [
        ("deepseek_r1", "<think>secret</think>answer"),
        ("gptoss", "<|channel|>secret<|message|>answer"),
        ("gemma4", "<|channel>thought<channel|>secret<|channel>final<channel|>answer"),
    ],
)
def test_arbitrary_split_and_coalesced_reasoning_markers(name, text, step, delegating):
    chunks = [text[index : index + step] for index in range(0, len(text), step)]
    assert _stream_reasoning(name, chunks, delegating) == ("secret", "answer")


@pytest.mark.parametrize("body", ["t", "th", "thought"])
@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("delegating", [False, True])
def test_gemma_ambiguous_label_flushes_on_exit_and_finalization(body, close, delegating):
    chunks = ["<|channel>thought<channel|>" + body]
    if close:
        chunks += ["<|channel>", "final", "<channel|>answer"]
    assert _stream_reasoning("gemma4", chunks, delegating) == (body, "answer" if close else "")


@pytest.mark.parametrize(
    "name,text",
    [
        ("granite", "Here's my thought process:secretHere's my response:answer"),
        ("gptoss", "<|channel|>secret<|message|>answer"),
        ("step3", "secret</think>answer"),
    ],
)
def test_stripper_uses_registered_grammar(name, text):
    assert make_reasoning_stripper(parser_name=name)(text) == "answer"


@pytest.mark.parametrize(
    "name,text", [("deepseek_r1", "<think>"), ("gptoss", "<|channel|>"), ("gemma4", "<|channel>thought<channel|>")]
)
def test_empty_reasoning_boundaries_are_not_visible_at_finalization(name, text):
    assert _stream_reasoning(name, [text], True) == ("", "")


def test_stripper_preserves_multiple_closed_and_unclosed_blocks():
    strip = make_reasoning_stripper(parser_name="deepseek_r1")
    assert strip("<think>one</think>A<think>two</think>B<think>unfinished") == "AB"


def _tool_parser(choice="auto"):
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice=choice,
    )
    return DelegatingParser(tool_parser=HermesToolParser(TextTokenizer()), tool_request=request)


@pytest.mark.parametrize("streamed", [False, True, "partial"])
@pytest.mark.parametrize(
    "text,expected,called",
    [
        ('Before. <tool_call>{"na', "Before. ", False),
        ('Before. <tool_call>{"name":"f","arguments":{}}</tool_call>After.', "Before. After.", True),
        ('Before. <tool_call>{"name":"f","arguments":{}}</tool_call>After. <tool_', "Before. After. <tool_", True),
    ],
)
def test_terminal_tool_update_preserves_ordinary_text_without_protocol_leaks(text, expected, called, streamed):
    parser = _tool_parser()
    visible = ""
    if streamed:
        first = text[:-3] if streamed == "partial" else text
        delta = parser.process_delta(first, first, [], "", [])
        visible += delta.delta_content or ""
    final = parser.process_final(text, [])
    visible += final.delta_content or ""
    assert visible == final.accumulated_content == expected
    assert bool(final.tool_calls) is called
    assert not parser.process_final(text, []).delta_content


@pytest.mark.parametrize("streamed", [False, True])
def test_explicit_tool_opt_out_preserves_literal_content(streamed):
    text = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
    parser = _tool_parser("none")
    visible = ""
    if streamed:
        delta = parser.process_delta(text, text, [], "", [])
        visible += delta.delta_content or ""
        assert not delta.delta_tool_calls
    final = parser.process_final(text, [])
    assert visible + (final.delta_content or "") == final.accumulated_content == text
    assert not final.tool_calls and not final.delta_tool_calls


def test_custom_logits_processor_required_optional_and_filtered_kwargs():
    class Required:
        def __call__(self, input_ids, scores, cur_len, *, extra):
            return scores + extra + cur_len

    class Optional:
        def __call__(self, input_ids, scores, cur_len, extra=2):
            return scores * extra

    class CatchAll:
        def __call__(self, input_ids, scores, cur_len, **kwargs):
            return scores + kwargs["other"]

    assert LogitsProcessorList([Required(), Optional(), CatchAll()])(None, 1, 5, extra=3, other=4) == 31
    assert LogitsProcessorList([Optional()])(None, 2, 5, unused=7) == 4
    with pytest.raises(ValueError, match="extra"):
        LogitsProcessorList([Required()])(None, 1, 5)


@pytest.mark.parametrize("aliases", [["ok", 1], ("ok",), 1])
def test_public_reasoning_registry_rejects_invalid_aliases(aliases):
    with pytest.raises(TypeError):
        ReasoningParserManager.register_module(aliases, module=type(_reasoning_parser("deepseek_r1")))
