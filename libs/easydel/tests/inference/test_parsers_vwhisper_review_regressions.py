# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive regression tests for Subsection 3 inference review findings (`S3-C1`..`S3-L2`)."""

from __future__ import annotations

import json
import random
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from easydel.inference.parsing import DelegatingParser, ParsePhase, ParseResult
from easydel.inference.reasoning import ReasoningMixin
from easydel.inference.reasoning.auto_detect import detect_reasoning_parser, make_reasoning_stripper
from easydel.inference.reasoning.parsers import (
    DeepSeekR1ReasoningParser,
    DeepSeekV3ReasoningParser,
    Gemma4ReasoningParser,
    GptOssReasoningParser,
    GraniteReasoningParser,
    MiniMaxM2ReasoningParser,
    MistralReasoningParser,
    Qwen3ReasoningParser,
    SeedOSSReasoningParser,
    Step3ReasoningParser,
)
from easydel.inference.tools import (
    Gemma4ToolParser,
    Step3p5ToolParser,
    ToolCallingMixin,
    ToolParser,
    ToolParserManager,
    detect_tool_parser,
)
from easydel.inference.tools.parsers._pythonic_ast import (
    UnexpectedAstError,
    compute_tool_delta,
    handle_single_tool,
    make_valid_python,
)
from easydel.inference.tools.parsers._schema_params import convert_param_value, get_arguments_config
from easydel.inference.tools.parsers.utils import stream_json_calls

pytestmark = pytest.mark.cpu_ok


class _UniversalTokenizer:
    """Deterministic tokenizer supporting all tool & reasoning special tokens."""

    _SPECIAL_TOKENS: tuple[str, ...] = (
        "<think>",
        "</think>",
        "[THINK]",
        "[/THINK]",
        "<seed:think>",
        "</seed:think>",
        "<|channel>",
        "<channel|>",
        "<|channel|>",
        "<|message|>",
        "<|turn>",
        "<|tool_response>",
        "<tool_call>",
        "</tool_call>",
        "<longcat_tool_call>",
        "</longcat_tool_call>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁calls▁end｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁call▁end｜>",
        "<｜tool▁sep｜>",
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_call_argument_begin|>",
        "<｜DSML｜function_calls>",
        "</｜DSML｜function_calls>",
        "<｜DSML｜invoke",
        "</｜DSML｜invoke>",
        "<｜DSML｜parameter",
        "</｜DSML｜parameter>",
        "[TOOL_CALLS]",
        "<|python_tag|>",
        "<|action_start|>",
        "<|action_end|>",
        "<|plugin|>",
        "<|tool_call|>",
        "<|tool_call>",
        "<tool_call|>",
        "<start_function_call>",
        "<end_function_call>",
        "<escape>",
        '<|"|>',
        "<minimax:tool_call>",
        "</minimax:tool_call>",
        "<seed:tool_call>",
        "</seed:tool_call>",
        "<function_calls>",
        "</function_calls>",
        "<|python_start|>",
        "<|python_end|>",
        "<｜tool_calls_begin｜>",
        "<｜tool_calls_end｜>",
        "<｜tool_call_begin｜>",
        "<｜tool_call_end｜>",
        "<｜tool_sep｜>",
        "<tool_calls>",
        "</tool_calls>",
        "<function_call>",
        "<response>",
        "</response>",
        "<|im_end|>",
    )

    def __init__(self, *, chat_template: str = "", special_tokens: tuple[str, ...] | None = None):
        self.chat_template = chat_template
        self._vocab: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_id = 1000
        for idx, tok in enumerate(self._SPECIAL_TOKENS, start=1):
            self._vocab[tok] = idx
            self._id_to_token[idx] = tok
        self.all_special_tokens = list(special_tokens) if special_tokens is not None else []
        self._sorted_specials = sorted(self._SPECIAL_TOKENS, key=len, reverse=True)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tok: str) -> int | None:
        return self._vocab.get(tok)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched = None
            for tok in self._sorted_specials:
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                ids.append(self._vocab[matched])
                i += len(matched)
            else:
                ch = text[i]
                tid = self._vocab.get(ch)
                if tid is None:
                    tid = self._next_id
                    self._next_id += 1
                    self._vocab[ch] = tid
                    self._id_to_token[tid] = ch
                ids.append(tid)
                i += 1
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        specials = set(self.all_special_tokens) if skip_special_tokens else set()
        return "".join(
            self._id_to_token.get(tid, "")
            for tid in token_ids
            if self._id_to_token.get(tid, "") not in specials
        )


def _make_request(tools: list[dict] | None = None) -> ChatCompletionRequest:
    default_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "count": {"type": "integer"},
                        "rate": {"type": "number"},
                        "flag": {"type": "boolean"},
                        "meta": {"type": "object"},
                        "items": {"type": "array"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    return ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": default_tools if tools is None else tools,
        }
    )


def _chunk_text(text: str, mode: str) -> list[str]:
    if mode == "single":
        return [text]
    if mode == "char":
        return list(text)
    if mode == "random":
        rng = random.Random(42)
        chunks: list[str] = []
        i = 0
        while i < len(text):
            step = rng.randint(1, 5)
            chunks.append(text[i : i + step])
            i += step
        return chunks
    raise ValueError(mode)


def _run_delegating_stream(
    parser_name: str,
    text: str,
    chunks: list[str],
    request: ChatCompletionRequest | None = None,
) -> tuple[str, list[tuple[str, dict]]]:
    tok = _UniversalTokenizer()
    cls = ToolParserManager.get_tool_parser(parser_name)
    dp = DelegatingParser(tool_parser=cls(tok), tool_request=request if request is not None else _make_request())
    prev_text = ""
    prev_ids: list[int] = []
    visible = ""
    streamed_names: dict[int, str] = {}
    streamed_args: dict[int, str] = {}

    for chunk in chunks:
        cur_text = prev_text + chunk
        cur_ids = tok.encode(cur_text, add_special_tokens=False)
        res = dp.process_delta(cur_text, chunk, cur_ids, prev_text, prev_ids)
        if res.delta_content:
            visible += res.delta_content
        for tc in res.delta_tool_calls or []:
            idx = int(tc.index)
            if tc.function:
                if tc.function.name:
                    streamed_names[idx] = tc.function.name
                if tc.function.arguments:
                    streamed_args[idx] = streamed_args.get(idx, "") + tc.function.arguments
        prev_text = cur_text
        prev_ids = cur_ids

    final = dp.process_final(prev_text, prev_ids)
    if final.delta_content:
        visible += final.delta_content
    for tc in final.delta_tool_calls or []:
        idx = int(tc.index)
        if tc.function:
            if tc.function.name:
                streamed_names[idx] = tc.function.name
            if tc.function.arguments:
                streamed_args[idx] = streamed_args.get(idx, "") + tc.function.arguments

    assert visible == final.accumulated_content
    calls = []
    for idx in sorted(set(streamed_names) | set(streamed_args)):
        calls.append((streamed_names.get(idx, ""), json.loads(streamed_args.get(idx, "{}"))))
    return visible, calls


# =============================================================================
# S3-C1: ToolParser.is_buffering_protocol & DelegatingParser ordinary prose
# =============================================================================


@pytest.mark.parametrize(
    "parser_name",
    ["openai", "xlam", "pythonic", "llama4_pythonic", "phi4_mini_json", "granite", "llama3_json", "hermes"],
)
@pytest.mark.parametrize(
    "prose",
    [
        "See citation [1] and list [1, 2, 3] for details.",
        "Use {key: value} notation and check x < y in code.",
        "In Python, import functools and use functools.partial(fn, 1).",
        "Markdown [link](https://example.com) followed by more prose.",
    ],
)
@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_s3_c1_ordinary_prose_with_brackets_braces_functools_preserved(parser_name, prose, mode):
    visible, calls = _run_delegating_stream(parser_name, prose, _chunk_text(prose, mode))
    assert visible == prose
    assert calls == []


def test_s3_c1_is_buffering_protocol_split_prefix_and_closed_block():
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser("hermes")(tok)
    assert parser.is_buffering_protocol(current_text="Hello world <tool", delta_text="<tool") is True
    assert parser.is_buffering_protocol(current_text="Hello world <", delta_text="<") is True
    assert parser.is_buffering_protocol(current_text="2 < ", delta_text=" ") is False
    closed = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>After.'
    assert parser.is_buffering_protocol(current_text=closed, delta_text="After.") is False


# =============================================================================
# S3-C2: Qwen3ReasoningParser withheld prefix & mid-response literal <think>
# =============================================================================


def test_s3_c2_qwen3_does_not_drop_withheld_prefix_or_replay_literal_think():
    tok = _UniversalTokenizer()
    parser = Qwen3ReasoningParser(tok)
    d1 = parser.extract_reasoning_streaming("", "<th", "<th", [], [1], [1])
    assert d1 is None
    d2 = parser.extract_reasoning_streaming("<th", "<the answer", "e answer", [1], [1, 2], [2])
    assert d2 is not None
    assert d2.content == "<the answer"

    # Mid-response literal <think> must NOT replay from character 0 in DelegatingParser.
    dp = DelegatingParser(reasoning_parser=Qwen3ReasoningParser(tok))
    chunks = ["hello world ", "<think>", " literal ", "</think>", " end"]
    prev = ""
    visible = ""
    for chunk in chunks:
        cur = prev + chunk
        res = dp.process_delta(cur, chunk, [], prev, [])
        visible += res.delta_content or ""
        prev = cur
    final = dp.process_final(prev, [])
    visible += final.delta_content or ""
    assert visible == "hello world <think> literal </think> end"
    assert final.accumulated_content == "hello world <think> literal </think> end"
    assert final.accumulated_reasoning == ""


def test_s3_c2_qwen3_leading_spaces_before_think_no_duplication():
    tok = _UniversalTokenizer()
    dp = DelegatingParser(reasoning_parser=Qwen3ReasoningParser(tok))
    chunks = ["        ", "<think>", "deep", "</think>", "ok"]
    prev = ""
    visible = ""
    reasoning = ""
    for chunk in chunks:
        cur = prev + chunk
        res = dp.process_delta(cur, chunk, [], prev, [])
        visible += res.delta_content or ""
        reasoning += res.delta_reasoning or ""
        prev = cur
    final = dp.process_final(prev, [])
    visible += final.delta_content or ""
    reasoning += final.delta_reasoning or ""
    assert reasoning == "deep"
    assert visible == "        ok"


# =============================================================================
# S3-C3: Gemma4ReasoningParser dual grammar, missing vocab, unfinished reasoning
# =============================================================================


def test_s3_c3_gemma4_native_and_header_tag_formats_batch_streaming_and_is_reasoning_end():
    tok = _UniversalTokenizer()
    native = "<|channel>thought\nStep 1\nStep 2<channel|>Final answer"
    header = "<|channel>thought<channel|>Step 1\nStep 2<|channel>final<channel|>Final answer"

    for sample in (native, header):
        batch_parser = Gemma4ReasoningParser(tok)
        assert batch_parser.extract_reasoning(sample) == ("Step 1\nStep 2", "Final answer")

        stream_parser = Gemma4ReasoningParser(tok)
        prev = ""
        r_acc, c_acc = "", ""
        for ch in sample:
            cur = prev + ch
            msg = stream_parser.extract_reasoning_streaming(prev, cur, ch, [], [], [])
            if msg:
                r_acc += msg.reasoning_content or ""
                c_acc += msg.content or ""
            prev = cur
        tail = stream_parser.finalize_streaming(prev)
        if tail:
            r_acc += tail.reasoning_content or ""
            c_acc += tail.content or ""
        assert (r_acc, c_acc) == ("Step 1\nStep 2", "Final answer")

    # is_reasoning_end must agree with BOTH formats.
    p = Gemma4ReasoningParser(tok)
    assert p.is_reasoning_end(tok.encode("<|channel>thought\nstill thinking")) is False
    assert p.is_reasoning_end(tok.encode("<|channel>thought\ndone<channel|>answer")) is True
    assert p.is_reasoning_end(tok.encode("<|channel>thought<channel|>still thinking")) is False
    assert p.is_reasoning_end(tok.encode("<|channel>thought<channel|>done<|channel>final<channel|>answer")) is True

    # Missing vocab tokens must NOT raise KeyError.
    bare = Gemma4ReasoningParser(object())
    assert bare.start_token_id is None
    assert bare.end_token_id is None
    assert bare.is_reasoning_end([1, 2, 3]) is False

    # Unfinished native reasoning and literal <|channel> in content must not be wiped on final=True.
    p_unfin = Gemma4ReasoningParser(tok)
    assert p_unfin.extract_reasoning("<|channel>thought\nunfinished thought") == ("unfinished thought", None)
    assert p_unfin.extract_reasoning_final("<|channel>thought\nunfinished thought") == ("unfinished thought", None)
    assert p_unfin.extract_reasoning_final("Answer mentions <|channel> in prose") == (
        None,
        "Answer mentions <|channel> in prose",
    )


# =============================================================================
# S3-H1: ReasoningMixin per-stream multi-request isolation
# =============================================================================


def test_s3_h1_reasoning_mixin_multi_request_and_concurrent_isolation():
    tok = _UniversalTokenizer()
    mixin = ReasoningMixin()
    mixin.reasoning_parsers = {
        "r1": DeepSeekR1ReasoningParser(tok),
        "granite": GraniteReasoningParser(tok),
    }

    # Sequential requests on BaseThinkingReasoningParser without leaking _emitted_*
    d1 = mixin.extract_reasoning_streaming("r1", "", "<think>first</think>ans1", "<think>first</think>ans1", final=True)
    assert d1 is not None and d1.reasoning_content == "first" and d1.content == "ans1"
    d2 = mixin.extract_reasoning_streaming(
        "r1", "", "<think>second</think>ans2", "<think>second</think>ans2", final=True
    )
    assert d2 is not None and d2.reasoning_content == "second" and d2.content == "ans2"

    # Sequential requests on GraniteReasoningParser without leaking _reasoning_done
    g1 = mixin.extract_reasoning_streaming(
        "granite",
        "",
        "Here's my thought process:t1Here's my response:c1",
        "Here's my thought process:t1Here's my response:c1",
        final=True,
    )
    assert g1 is not None and g1.reasoning_content == "t1" and g1.content == "c1"
    g2 = mixin.extract_reasoning_streaming(
        "granite",
        "",
        "Here's my thought process:t2Here's my response:c2",
        "Here's my thought process:t2Here's my response:c2",
        final=True,
    )
    assert g2 is not None and g2.reasoning_content == "t2" and g2.content == "c2"

    # Interleaved concurrent streams keyed by stream_id
    a1 = mixin.extract_reasoning_streaming("r1", "", "<think>A", "<think>A", stream_id="req-a")
    b1 = mixin.extract_reasoning_streaming("r1", "", "<think>B", "<think>B", stream_id="req-b")
    a2 = mixin.extract_reasoning_streaming("r1", "<think>A", "<think>A</think>doneA", "</think>doneA", stream_id="req-a", final=True)
    b2 = mixin.extract_reasoning_streaming("r1", "<think>B", "<think>B</think>doneB", "</think>doneB", stream_id="req-b", final=True)
    assert a1.reasoning_content == "A" and a2.content == "doneA"
    assert b1.reasoning_content == "B" and b2.content == "doneB"


# =============================================================================
# S3-H2: DeepSeekV3ReasoningParser delegation & make_reasoning_stripper
# =============================================================================


def test_s3_h2_deepseek_v3_forwards_hooks_and_handles_empty_think():
    tok = _UniversalTokenizer(chat_template="template with enable_thinking")
    p = DeepSeekV3ReasoningParser(tok)
    assert p.streaming_all_phases is True

    dp = DelegatingParser(reasoning_parser=p)
    res = dp.process_final("<think></think>", [])
    assert res.accumulated_content == ""
    assert res.accumulated_reasoning == ""

    # tokenizer=None (used by make_reasoning_stripper) must strip <think> blocks.
    for mt in ("deepseek_v3", "glm4_moe", "kimi_k2"):
        stripper = make_reasoning_stripper(model_type=mt)
        assert stripper("<think>secret</think>visible") == "visible"


# =============================================================================
# S3-H3: BaseThinkingReasoningParser leading whitespace + repeated <think>
# =============================================================================


def test_s3_h3_prompt_started_leading_whitespace_before_repeated_think():
    tok = _UniversalTokenizer()
    p = DeepSeekR1ReasoningParser(tok)
    p.configure_prompt_context("user <think>", [1])
    dp = DelegatingParser(reasoning_parser=p)

    chunks = ["\n", "<think>", "hello", " world", "</think>", "done"]
    prev = ""
    r_acc, c_acc = "", ""
    for chunk in chunks:
        cur = prev + chunk
        res = dp.process_delta(cur, chunk, [], prev, [])
        r_acc += res.delta_reasoning or ""
        c_acc += res.delta_content or ""
        prev = cur
    final = dp.process_final(prev, [])
    r_acc += final.delta_reasoning or ""
    c_acc += final.delta_content or ""
    assert r_acc == "hello world"
    assert c_acc == "done"


# =============================================================================
# S3-H4 & S3-M7: DelegatingParser retokenization, final flush, catch-up, mixin
# =============================================================================


def test_s3_h4_m7_delegating_final_flush_and_catchup_no_double_json():
    tok = _UniversalTokenizer()
    # tools_enabled=False (tool_request=None) must still flush trailing partial marker text
    dp = DelegatingParser(tool_parser=ToolParserManager.get_tool_parser("hermes")(tok), tool_request=None)
    r1 = dp.process_delta("Hello <tool_", "Hello <tool_", [], "", [])
    assert r1.delta_content == "Hello "
    fin = dp.process_final("Hello <tool_", [])
    assert fin.accumulated_content == "Hello <tool_"
    assert fin.delta_content == "<tool_"

    # Never concatenate full JSON onto non-matching partial JSON prefix; complete compact/spaced JSON cleanly
    dp2 = DelegatingParser()
    dp2._merge_streamed_tool_call_state(
        [
            DeltaToolCall(
                index=0,
                id="call_0",
                type="function",
                function=DeltaFunctionCall(name="f", arguments='{"x":1'),
            ),
            DeltaToolCall(
                index=1,
                id="call_1",
                type="function",
                function=DeltaFunctionCall(name="g", arguments='{"other": '),
            ),
        ]
    )
    missing = dp2._build_missing_final_tool_deltas(
        [
            ToolCall(id="call_0", function=FunctionCall(name="f", arguments='{"x": 1}')),
            ToolCall(id="call_1", function=FunctionCall(name="g", arguments='{"x": 2}')),
        ]
    )
    assert missing is not None
    assert len(missing) == 1
    assert missing[0].index == 0
    assert json.loads('{"x":1' + (missing[0].function.arguments or "")) == {"x": 1}

    # ToolCallingMixin preserves intentional empty content=""
    class _EmptyContentParser(ToolParser):
        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content="")

    mixin = ToolCallingMixin()
    mixin.tool_parsers = {"m": _EmptyContentParser(tok)}
    msg, reason = mixin.extract_tool_calls_batch("<stripped/>", _make_request(), "m")
    assert reason == "stop"
    assert msg.content == ""


def test_s3_h4_split_special_token_delta_ids_and_utf8_boundary():
    class _Utf8ByteTokenizer:
        """Simulates byte-level BPE that produces \\ufffd if a multi-byte char is split mid-token."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            if "<tool_call>" in text:
                out: list[int] = []
                parts = text.split("<tool_call>")
                for idx, part in enumerate(parts):
                    if idx > 0:
                        out.append(999)
                    out.extend(list(part.encode("utf-8")))
                return out
            return list(text.encode("utf-8"))

        def decode(self, ids, skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            buf = bytearray()
            pieces: list[str] = []
            for tid in ids:
                if tid == 999:
                    if buf:
                        pieces.append(buf.decode("utf-8", errors="replace"))
                        buf.clear()
                    pieces.append("<tool_call>")
                else:
                    buf.append(tid)
            if buf:
                pieces.append(buf.decode("utf-8", errors="replace"))
            return "".join(pieces)

    class _RecorderParser(ToolParser):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self.seen_delta_ids: list[list[int]] = []
            self.seen_current_ids: list[list[int]] = []

        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        def extract_tool_calls_streaming(
            self, previous_text, current_text, delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
        ):
            self.seen_delta_ids.append(list(delta_token_ids))
            self.seen_current_ids.append(list(current_token_ids))
            return DeltaMessage(content=delta_text)

    tok = _Utf8ByteTokenizer()
    recorder = _RecorderParser(tok)
    dp = DelegatingParser(tool_parser=recorder, tool_request=_make_request())

    # Build >32 bytes containing multi-byte UTF-8 emojis so index -32 splits inside a 4-byte UTF-8 char
    text = "prefix " + ("😀" * 12) + " <tool"
    dp.process_delta(text, text, [], "", [])
    next_text = text + "_call>"
    dp.process_delta(next_text, "_call>", [], text, [])
    assert recorder.seen_current_ids[-1] == tok.encode(next_text)
    assert 999 in recorder.seen_delta_ids[-1]


# =============================================================================
# S3-H5 & S3-H10 & S3-M2 & S3-M3: SeedOSS, adjust_request, auto_detect, parity
# =============================================================================


def test_s3_h5_seedoss_supports_seed_think_and_think():
    tok = _UniversalTokenizer()
    p = SeedOSSReasoningParser(tok)
    assert p.extract_reasoning("<seed:think>plan</seed:think>answer") == ("plan", "answer")
    assert p.extract_reasoning("<think>plan</think>answer") == ("plan", "answer")
    assert p.is_reasoning_end(tok.encode("<seed:think>plan</seed:think>")) is True
    assert p.is_reasoning_end(tok.encode("<think>plan</think>")) is True
    assert detect_reasoning_parser(tokenizer=SimpleNamespace(chat_template="", get_vocab=lambda: {"<seed:think>": 1})) == "seed_oss"


def test_s3_h10_reasoning_parser_adjust_request_preserves_special_delimiters():
    tok_special = _UniversalTokenizer(special_tokens=("<think>", "</think>", "[THINK]", "[/THINK]", "<seed:think>"))
    for cls in (DeepSeekR1ReasoningParser, Qwen3ReasoningParser, MistralReasoningParser, SeedOSSReasoningParser):
        req = SimpleNamespace(skip_special_tokens=True)
        cls(tok_special).adjust_request(req)
        assert req.skip_special_tokens is False

    # DeepSeekV3 forwards adjust_request to its R1 delegate
    tok_v3 = _UniversalTokenizer(chat_template="enable_thinking", special_tokens=("<think>", "</think>"))
    req_v3 = SimpleNamespace(skip_special_tokens=True)
    DeepSeekV3ReasoningParser(tok_v3).adjust_request(req_v3)
    assert req_v3.skip_special_tokens is False

    # Non-special delimiters leave skip_special_tokens untouched
    tok_plain = _UniversalTokenizer(special_tokens=())
    req_plain = SimpleNamespace(skip_special_tokens=True)
    DeepSeekR1ReasoningParser(tok_plain).adjust_request(req_plain)
    assert req_plain.skip_special_tokens is True


def test_s3_m2_auto_detect_boundary_matching_and_gpt_oss():
    assert detect_reasoning_parser(model_type="dolphin") == "deepseek_r1"
    assert detect_reasoning_parser(model_type="nextstep") == "deepseek_r1"
    assert detect_reasoning_parser(model_type="reseed") == "deepseek_r1"
    assert detect_tool_parser(model_type="dolphin") == "hermes"
    assert detect_tool_parser(model_type="nextstep") == "hermes"
    assert detect_tool_parser(model_type="reseed") == "hermes"
    assert detect_tool_parser(model_type="gpt_oss") == "openai"


def test_s3_m3_reasoning_batch_streaming_parity():
    tok = _UniversalTokenizer()
    granite = GraniteReasoningParser(tok)
    assert granite.extract_reasoning("Pre. Here's my thought process:rHere's my response:a") == ("r", "Pre. a")
    assert granite.extract_reasoning("Pre. Here's my response:a") == (None, "Pre. a")

    gptoss = GptOssReasoningParser(tok)
    assert gptoss.extract_reasoning("Literal <|channel|> in prose") == (None, "Literal <|channel|> in prose")

    for cls in (MiniMaxM2ReasoningParser, Step3ReasoningParser):
        p = cls(tok)
        assert p.extract_reasoning("thinking</think>") == ("thinking", None)
        assert p.extract_reasoning("mention <think> inside</think>ans") == ("mention <think> inside", "ans")


# =============================================================================
# S3-C6, S3-C7, S3-H8, S3-L1: Shared tool parser infrastructure & ReDoS guard
# =============================================================================


def test_s3_c6_pythonic_ast_completion_and_numeric_transition():
    completed = make_valid_python('[get_weather(city="NYC"')
    assert completed is not None
    assert completed[0] == '[get_weather(city="NYC")]'

    completed_dict = make_valid_python('[get_weather(meta={"a": 1')
    assert completed_dict is not None
    assert completed_dict[0] == '[get_weather(meta={"a": 1})]'

    # Positional args & **kwargs rejected
    import ast

    with pytest.raises(UnexpectedAstError):
        handle_single_tool(ast.parse("f(1)").body[0].value)
    with pytest.raises(UnexpectedAstError):
        handle_single_tool(ast.parse("f(**d)").body[0].value)

    # Numeric transition 1 -> 1e-2 must never corrupt into 1.01
    c1 = ToolCall(id="call_1", function=FunctionCall(name="f", arguments='{"rate": 1}'))
    c2 = ToolCall(id="call_1", function=FunctionCall(name="f", arguments='{"rate": 0.01}'))
    d1 = compute_tool_delta("", c1, 0, "}")
    assert d1 is not None
    sent = d1.function.arguments or ""
    d2 = compute_tool_delta(sent, c2, 0, "")
    assert d2 is not None
    sent += d2.function.arguments or ""
    assert json.loads(sent) == {"rate": 0.01}


@pytest.mark.parametrize("parser_name", ["olmo3", "pythonic"])
def test_s3_c7_pythonic_redos_non_exponential(parser_name):
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser(parser_name)(tok)
    adversarial = "[f(" + "a=1," * 30 + "x"
    t0 = time.perf_counter()
    res = parser.extract_tool_calls(adversarial, _make_request())
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.25, f"{parser_name} took {elapsed:.3f}s on adversarial input"
    assert res.tools_called is False
    assert res.content == adversarial


def test_s3_h8_schema_params_and_stream_json_calls_surrogate():
    # Explicit type: "string" must NOT coerce "null" to None
    assert convert_param_value("null", "city", {"city": {"type": "string"}}, "get_weather") == "null"
    assert convert_param_value("null", "count", {"count": {"type": "integer"}}, "get_weather") is None
    assert convert_param_value("null", "city", {"city": {"type": ["string", "null"]}}, "get_weather") is None

    # Huge exponent in union must NOT hang or exhaust RAM
    t0 = time.perf_counter()
    huge = convert_param_value(
        "1e100000000",
        "x",
        {"x": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
        "f",
    )
    assert time.perf_counter() - t0 < 0.2
    assert huge == "1e100000000"

    # properties: None must return {} safely
    req = _make_request(
        [{"type": "function", "function": {"name": "empty_props", "parameters": {"type": "object", "properties": None}}}]
    )
    assert get_arguments_config("empty_props", req.tools) == {}

    # stream_json_calls must withhold trailing lone high surrogate
    prev: list[dict] = [{"name": "f", "arguments": {"s": "\ud83d"}}]
    sent = [""]
    ids = [""]
    d1 = stream_json_calls([{"name": "f", "arguments": {"s": "\ud83d"}}], prev, sent, ids, complete=False)
    assert "\ud83d" not in (d1[0].function.arguments or "")
    d2 = stream_json_calls([{"name": "f", "arguments": {"s": "\U0001f600"}}], prev, sent, ids, complete=True)
    full_args = (d1[0].function.arguments or "") + (d2[0].function.arguments or "")
    assert json.loads(full_args) == {"s": "\U0001f600"}


def test_s3_l1_public_tool_exports():
    import easydel.inference.tools as tools_pkg

    assert Gemma4ToolParser is tools_pkg.Gemma4ToolParser
    assert Step3p5ToolParser is tools_pkg.Step3p5ToolParser
    assert "Gemma4ToolParser" in tools_pkg.__all__
    assert "Step3p5ToolParser" in tools_pkg.__all__


# =============================================================================
# S3-C8, S3-C9, S3-H9, S3-M1, S3-M4, S3-M5: Parametrized Tool Parser Matrix
# =============================================================================


_STREAMING_MATRIX: list[tuple[str, str, list[tuple[str, dict]]]] = [
    (
        "deepseek_v3",
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "NYC", "count": 3}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>ping\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        [("get_weather", {"city": "NYC", "count": 3}), ("ping", {})],
    ),
    (
        "deepseek_v31",
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "NYC", "count": 3}<｜tool▁call▁end｜><｜tool▁call▁begin｜>ping<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        [("get_weather", {"city": "NYC", "count": 3}), ("ping", {})],
    ),
    (
        "kimi_k2",
        '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "NYC", "count": 3}<|tool_call_end|><|tool_call_begin|>functions.ping:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>',
        [("get_weather", {"city": "NYC", "count": 3}), ("ping", {})],
    ),
    (
        "hermes",
        '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC", "count": 3}}</tool_call><tool_call>{"name": "ping", "arguments": {}}</tool_call>',
        [("get_weather", {"city": "NYC", "count": 3}), ("ping", {})],
    ),
    (
        "longcat",
        '<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</longcat_tool_call><longcat_tool_call>{"name": "ping", "arguments": {}}</longcat_tool_call>',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "minimax_m2",
        '<minimax:tool_call><invoke name="get_weather"><parameter name="city">NYC</parameter><parameter name="count">3</parameter></invoke><invoke name="ping"></invoke></minimax:tool_call>',
        [("get_weather", {"city": "NYC", "count": 3}), ("ping", {})],
    ),
    (
        "minimax",
        '<tool_calls>{"name": "get_weather", "arguments": {"city": "a < b"}}\n{"name": "ping", "arguments": {}}</tool_calls>',
        [("get_weather", {"city": "a < b"}), ("ping", {})],
    ),
    (
        "openai",
        '[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "ping", "arguments": {}}]',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "xlam",
        '[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "ping", "arguments": {}}]',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "phi4_mini_json",
        'functools[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "ping", "arguments": {}}]',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "granite",
        '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "ping", "arguments": {}}]',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "granite-20b-fc",
        '<function_call>{"name": "get_weather", "arguments": {"city": "NYC"}}<function_call>{"name": "ping", "arguments": {}}',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "jamba",
        '<tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "ping", "arguments": {}}]</tool_calls>',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "hunyuan_a13b",
        'Sure! <tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "ping", "arguments": {}}]</tool_calls>',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "ernie45",
        '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call><tool_call>{"name": "ping", "arguments": {}}</tool_call>',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "glm45",
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>ping</tool_call>",
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "gemma4",
        '<|tool_call>call:get_weather{city:<|"|>NYC<|"|>,rate:1e-2}<tool_call|><|tool_call>call:ping{}<tool_call|>',
        [("get_weather", {"city": "NYC", "rate": 0.01}), ("ping", {})],
    ),
    (
        "functiongemma",
        "<start_function_call>call:get_weather{city:<escape>42<escape>}<end_function_call><start_function_call>call:ping{}<end_function_call>",
        [("get_weather", {"city": "42"}), ("ping", {})],
    ),
    (
        "internlm",
        '<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "NYC"}}<|action_end|><|action_start|><|plugin|>{"name": "ping", "parameters": {}}<|action_end|>',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "llama3_json",
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "NYC"}};{"name": "ping", "parameters": {}}',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "mistral",
        '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "ping", "arguments": {}}]',
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "qwen3_coder",
        "<tool_call><function=get_weather><parameter=city>NYC</parameter></function></tool_call><tool_call><function=ping></function></tool_call>",
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "qwen3_xml",
        "<tool_call><function=get_weather><parameter=city>NYC</parameter></function></tool_call><tool_call><function=ping></function></tool_call>",
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "step3p5",
        "<tool_call><function=get_weather><parameter=city>NYC</parameter></function></tool_call><tool_call><function=ping></function></tool_call>",
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "seed_oss",
        "<seed:tool_call><function=get_weather><parameter=city>NYC</parameter></function></seed:tool_call><seed:tool_call><function=ping></function></seed:tool_call>",
        [("get_weather", {"city": "NYC"}), ("ping", {})],
    ),
    (
        "step3",
        '<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="get_weather"><steptml:parameter name="city">a < b</steptml:parameter></steptml:invoke><｜tool_call_end｜><｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="ping"></steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>',
        [("get_weather", {"city": "a < b"}), ("ping", {})],
    ),
    (
        "pythonic",
        '[get_weather(city="NYC", count=3), ping()]',
        [("get_weather", {"city": "NYC", "count": 3}), ("ping", {})],
    ),
    (
        "llama4_pythonic",
        '<|python_start|>[get_weather(city="NYC", count=3), ping()]<|python_end|>',
        [("get_weather", {"city": "NYC", "count": 3}), ("ping", {})],
    ),
    (
        "olmo3",
        '<function_calls>\nget_weather(city="NYC", count=3)\nping()\n</function_calls>',
        [("get_weather", {"city": "NYC", "count": 3}), ("ping", {})],
    ),
]


@pytest.mark.parametrize("parser_name,payload,expected", _STREAMING_MATRIX, ids=[row[0] for row in _STREAMING_MATRIX])
@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_s3_tool_parser_streaming_matrix_and_zero_arg_calls(parser_name, payload, expected, mode):
    _visible, calls = _run_delegating_stream(parser_name, payload, _chunk_text(payload, mode))
    assert calls == expected


@pytest.mark.parametrize(
    "parser_name,chunks",
    [
        ("granite", ['<|tool_call|>[{"name": "get_weather", "arguments": {"rate": 1', "0", "e-2}}]"]),
        ("granite-20b-fc", ['<function_call>{"name": "get_weather", "arguments": {"rate": 1', "0", "e-2}}"]),
        ("jamba", ['<tool_calls>[{"name": "get_weather", "arguments": {"rate": 1', "0", "e-2}}]</tool_calls>"]),
        ("llama3_json", ['<|python_tag|>{"name": "get_weather", "parameters": {"rate": 1', "2", "e-3}}"]),
        ("openai", ['[{"name": "get_weather", "arguments": {"rate": 1', "e-2}}]"]),
        ("pythonic", ['[get_weather(rate=1', "e-2)]"]),
        ("llama4_pythonic", ['<|python_start|>[get_weather(rate=1', "e-2)]<|python_end|>"]),
        ("olmo3", ['<function_calls>\nget_weather(rate=1', "e-2)\n</function_calls>"]),
    ],
)
def test_s3_numeric_exponent_streaming_no_corruption(parser_name, chunks):
    payload = "".join(chunks)
    _visible, calls = _run_delegating_stream(parser_name, payload, chunks)
    assert len(calls) == 1
    expected_val = 0.012 if parser_name == "llama3_json" else (0.1 if "0" in chunks[1] else 0.01)
    assert calls[0][0] == "get_weather"
    assert calls[0][1]["rate"] == pytest.approx(expected_val)


def test_s3_c9_minimax_m2_no_deadlock_when_param_and_invoke_close_together():
    chunks = [
        "<minimax:tool_call>",
        '<invoke name="get_weather">',
        '<parameter name="city">hello',
        "</parameter></invoke>",
        '<invoke name="ping"></invoke>',
        "</minimax:tool_call>",
    ]
    _visible, calls = _run_delegating_stream("minimax_m2", "".join(chunks), chunks)
    assert calls == [("get_weather", {"city": "hello"}), ("ping", {})]


def test_s3_c9_openai_mid_stream_preamble_no_attribute_error():
    chunks = ["Sure! ", '```json\n[{"name": "get_weather", "arguments": {"city": "Paris"}}]\n```']
    visible, calls = _run_delegating_stream("openai", "".join(chunks), chunks)
    assert visible == "Sure! "
    assert calls == [("get_weather", {"city": "Paris"})]


def test_s3_h9_openai_plain_object_list_not_misclassified_as_tool_call():
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser("openai")(tok)
    plain = '[{"name": "Alice", "age": 30}]'
    res = parser.extract_tool_calls(plain, _make_request())
    assert res.tools_called is False
    assert res.content == plain


def test_s3_h9_gigachat3_arguments_before_name_streaming():
    chunks = ['function call{"arguments": {"city": "NYC"}, "name": "get_weather"}<|im_end|>']
    _visible, calls = _run_delegating_stream("gigachat3", "".join(chunks), chunks)
    assert calls == [("get_weather", {"city": "NYC"})]


def test_s3_h9_kimi_k2_dotted_and_namespaceless_ids():
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser("kimi_k2")(tok)
    out = (
        '<|tool_calls_section_begin|>'
        '<|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{"city": "LA"}<|tool_call_end|>'
        '<|tool_call_begin|>functions.mcp.get_weather:1<|tool_call_argument_begin|>{"city": "SF"}<|tool_call_end|>'
        '<|tool_calls_section_end|>'
    )
    res = parser.extract_tool_calls(out, _make_request())
    assert res.tools_called is True
    assert [c.function.name for c in res.tool_calls] == ["get_weather", "mcp.get_weather"]


def test_s3_m1_mistral_adjust_request_without_mistral_common():
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser("mistral")(tok)
    req = SimpleNamespace(skip_special_tokens=True, tools=[{"type": "function"}], tool_choice="auto")
    parser.adjust_request(req)
    assert req.skip_special_tokens is False


@pytest.mark.parametrize(
    "parser_name,empty_payload",
    [
        ("phi4_mini_json", "functools[]"),
        ("granite", "<|tool_call|>[]"),
        ("jamba", "<tool_calls>[]</tool_calls>"),
        ("ernie45", "<tool_call>not json</tool_call>"),
        ("deepseek_v31", "<｜tool▁calls▁begin｜><｜tool▁calls▁end｜>"),
        ("kimi_k2", "<|tool_calls_section_begin|><|tool_calls_section_end|>"),
    ],
)
def test_s3_m4_empty_batch_tool_calls_returns_tools_called_false(parser_name, empty_payload):
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser(parser_name)(tok)
    res = parser.extract_tool_calls(empty_payload, _make_request())
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content == empty_payload


@pytest.mark.parametrize(
    "parser_name,payload",
    [
        ("minimax_m2", '<minimax:tool_call><invoke name="ping"></invoke></minimax:tool_call>'),
        ("qwen3_coder", "<tool_call><function=ping></function></tool_call>"),
        ("qwen3_xml", "<tool_call><function=ping></function></tool_call>"),
        ("seed_oss", "<seed:tool_call><function=ping></function></seed:tool_call>"),
    ],
)
def test_s3_l1_batch_extract_does_not_mutate_prev_tool_call_arr(parser_name, payload):
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser(parser_name)(tok)
    assert parser.prev_tool_call_arr == []
    res = parser.extract_tool_calls(payload, _make_request())
    assert res.tools_called is True
    assert parser.prev_tool_call_arr == []


# =============================================================================
# S3-C4, S3-C5, S3-H6, S3-H7, S3-L1/L2: vWhisper Audio Inference Regressions
# =============================================================================


def test_s3_c4_vwhisper_non_multilingual_get_decoder_input_ids():
    from easydel.inference.vwhisper.generation import get_decoder_input_ids

    cfg = SimpleNamespace(
        is_multilingual=False,
        no_timestamps_token_id=50362,
        task_to_id={},
        lang_to_id={},
    )
    ids_no_ts = get_decoder_input_ids(model_config=cfg, generation_config=cfg, return_timestamps=False)
    assert ids_no_ts == [(1, 50362)]
    ids_with_ts = get_decoder_input_ids(model_config=cfg, generation_config=cfg, return_timestamps=True)
    assert ids_with_ts == []


def test_s3_c5_vwhisper_chunk_iter_empty_and_no_invalid_tail_chunk():
    from easydel.inference.vwhisper.utils import chunk_iter_with_batch

    class _FakeFE:
        sampling_rate = 16_000

        def __call__(self, audio, sampling_rate=16000, return_tensors="np"):
            del audio, sampling_rate, return_tensors
            return {"input_features": np.zeros((1, 80, 10), dtype=np.float32)}

    # Empty audio must yield 0 batches without ValueError
    assert list(chunk_iter_with_batch(np.zeros((0,), dtype=np.float32), 480000, 80000, 80000, 1, _FakeFE())) == []

    # 22.5s @ 16kHz = 360_000 samples with 30s chunk / 5s stride must produce 1 valid chunk (stride_left <= chunk_len)
    batches = list(
        chunk_iter_with_batch(np.zeros((360_000,), dtype=np.float32), 480_000, 80_000, 80_000, 2, _FakeFE())
    )
    strides = [s for batch in batches for s in batch["stride"]]
    assert len(strides) == 1
    chunk_len, s_left, s_right = strides[0]
    assert chunk_len == 360_000 and s_left == 0 and s_right == 0


def test_s3_h7_vwhisper_resampling_stride_scaled_before_validation(monkeypatch):
    import sys

    from easydel.inference.vwhisper.utils import process_audio_input

    def _fake_resample(arr, orig_sr, target_sr):
        new_len = round(len(arr) * target_sr / orig_sr)
        return np.zeros((new_len,), dtype=arr.dtype)

    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(resample=_fake_resample))

    class _FakeFE:
        sampling_rate = 16_000

    # 10s @ 48kHz with 2s/2s stride: before scaling 96000+96000=192000 > 160000 would wrongly fail!
    audio = np.zeros((480_000,), dtype=np.float32)
    resampled, stride = process_audio_input(
        {"array": audio, "sampling_rate": 48_000, "stride": (96_000, 96_000)},
        _FakeFE(),
    )
    assert resampled.shape[0] == 160_000
    assert stride == (160_000, 32_000, 32_000)


def test_s3_h6_vwhisper_server_formats_filename_none_and_singleton_recovery(monkeypatch):
    import asyncio
    import io

    from easydel.inference.vwhisper import server as vw_server
    from fastapi import UploadFile
    from fastapi.dependencies import utils as fastapi_dep_utils
    from fastapi.responses import PlainTextResponse
    from fastapi.routing import APIRoute

    monkeypatch.setattr(fastapi_dep_utils, "ensure_multipart_is_installed", lambda: None)
    vw_server.WhisperModel.reset()
    attempts = {"n": 0}

    def _flaky_init(self):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient download error")
        self.inference = lambda audio_input, **kwargs: {
            "text": "hello world",
            "chunks": [{"text": "hello world", "timestamp": (0.0, 1.5)}],
        }

    monkeypatch.setattr(vw_server.WhisperModel, "_initialize", _flaky_init)
    try:
        with pytest.raises(RuntimeError, match="transient"):
            vw_server.WhisperModel(model_name="tiny.en")
        assert vw_server.WhisperModel._instance is None

        app = vw_server.create_whisper_app(model_name="tiny.en")
        transcribe_ep = next(
            route.endpoint
            for route in app.routes
            if isinstance(route, APIRoute) and route.path == "/v1/audio/transcriptions"
        )

        async def _exercise():
            r_text = await transcribe_ep(
                file=UploadFile(filename=None, file=io.BytesIO(b"RIFF0000")),
                model="tiny.en",
                prompt=None,
                response_format=vw_server.ResponseFormat.text,
                temperature=0.0,
                language=None,
                timestamp_granularities=None,
                timestamp_granularities_bracket=None,
            )
            assert isinstance(r_text, PlainTextResponse)
            assert r_text.body.decode() == "hello world"

            r_verbose = await transcribe_ep(
                file=UploadFile(filename="clip.wav", file=io.BytesIO(b"RIFF0000")),
                model="tiny.en",
                prompt=None,
                response_format=vw_server.ResponseFormat.verbose_json,
                temperature=0.0,
                language="en",
                timestamp_granularities=["segment"],
                timestamp_granularities_bracket=None,
            )
            assert r_verbose["text"] == "hello world"
            assert len(r_verbose["segments"]) == 1

        asyncio.run(_exercise())
    finally:
        vw_server.WhisperModel.reset()


def test_s3_l1_l2_vwhisper_init_docstring():
    import easydel.inference.vwhisper as vw

    doc = vw.__doc__ or ""
    assert "--model-id" not in doc
    assert "engine.transcribe(" not in doc


# =============================================================================
# Audit Follow-Up Regressions (#144): Items 1, 3, 4 (Granite, Llama, OpenAI, xLAM)
# =============================================================================


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item1_granite_bare_array_prose_and_preamble_marker(mode):
    bare_call = '[{"name": "get_weather", "arguments": {"city": "NYC"}}]'
    visible, calls = _run_delegating_stream("granite", bare_call, _chunk_text(bare_call, mode))
    assert visible == ""
    assert calls == [("get_weather", {"city": "NYC"})]

    prose = "[Note] hello world"
    visible, calls = _run_delegating_stream("granite", prose, _chunk_text(prose, mode))
    assert visible == prose
    assert calls == []

    non_tool_arr = '[{"name": "Alice", "age": 30}]'
    visible, calls = _run_delegating_stream("granite", non_tool_arr, _chunk_text(non_tool_arr, mode))
    assert visible == non_tool_arr
    assert calls == []

    preamble_call = 'Sure! <|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    visible, calls = _run_delegating_stream("granite", preamble_call, _chunk_text(preamble_call, mode))
    assert visible == "Sure! "
    assert calls == [("get_weather", {"city": "Paris"})]
    batch = ToolParserManager.get_tool_parser("granite")(_UniversalTokenizer()).extract_tool_calls(
        preamble_call, _make_request()
    )
    assert batch.tools_called is True
    assert batch.content == "Sure! "
    assert len(batch.tool_calls) == 1
    assert batch.tool_calls[0].function.name == "get_weather"


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item3_llama_prose_json_no_phantom_call_or_tail_erasure(mode):
    text = 'Here is {"name": "Alice", "age": 30} done!'
    visible, calls = _run_delegating_stream("llama3_json", text, _chunk_text(text, mode))
    assert visible == text
    assert calls == []

    mixed = 'Here is {"name": "Alice", "age": 30} and {"name": "get_weather", "arguments": {"city": "LA"}}'
    visible, calls = _run_delegating_stream("llama3_json", mixed, _chunk_text(mixed, mode))
    assert visible == 'Here is {"name": "Alice", "age": 30} and '
    assert calls == [("get_weather", {"city": "LA"})]


@pytest.mark.parametrize("parser_name", ["openai", "xlam"])
@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item4a_non_json_code_fence_parity_preserved(parser_name, mode):
    py_then_tool = (
        "```python\nx = 1\n```\n"
        '```json\n{"name": "get_weather", "arguments": {"city": "NYC"}}\n```'
    )
    visible, calls = _run_delegating_stream(parser_name, py_then_tool, _chunk_text(py_then_tool, mode))
    assert "```python\nx = 1\n```" in visible
    assert '"get_weather"' not in visible
    assert calls == [("get_weather", {"city": "NYC"})]

    tool_then_py = (
        '```json\n{"name": "get_weather", "arguments": {"city": "NYC"}}\n```\n'
        "```python\nx = 1\n```\nImportant after!"
    )
    visible, calls = _run_delegating_stream(parser_name, tool_then_py, _chunk_text(tool_then_py, mode))
    assert "```python\nx = 1\n```\nImportant after!" in visible
    assert calls == [("get_weather", {"city": "NYC"})]


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item4b_openai_multi_fence_non_tool_after_call_and_missing_name(mode):
    two_fences = (
        'Intro\n```json\n{"name": "get_weather", "arguments": {"city": "NYC"}}\n```\n'
        'Mid\n```json\n{"name": "ping", "arguments": {}}\n```\nEnd'
    )
    visible, calls = _run_delegating_stream("openai", two_fences, _chunk_text(two_fences, mode))
    assert calls == [("get_weather", {"city": "NYC"}), ("ping", {})]
    assert "Intro" in visible and "Mid" in visible and "End" in visible

    tool_then_nontool = '```json\n{"name": "ping", "arguments": {}}\n```\n```json\n{"example": 123}\n```'
    visible, calls = _run_delegating_stream("openai", tool_then_nontool, _chunk_text(tool_then_nontool, mode))
    assert calls == [("ping", {})]
    assert '{"example": 123}' in visible

    missing_name = '{"type": "function", "function": {"other": 1}}'
    visible, calls = _run_delegating_stream("openai", missing_name, _chunk_text(missing_name, mode))
    assert visible == missing_name
    assert calls == []


@pytest.mark.parametrize("parser_name", ["openai", "xlam"])
def test_audit_item4b_malformed_json_does_not_stall_until_eof(parser_name):
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser(parser_name)(tok)
    req = _make_request()
    emitted = ""
    acc = ""
    for ch in "{foo} bar":
        delta = parser.extract_tool_calls_streaming(acc, acc + ch, ch, [], [], [], req)
        acc += ch
        if delta and delta.content:
            emitted += delta.content
    assert emitted == "{foo} bar"


@pytest.mark.parametrize(
    "chunks",
    [
        ["<think>reason</think>", '[{"name": "get_weather", "arguments": {"city": "LA"}}]'],
        list('<think>reason</think>[{"name": "get_weather", "arguments": {"city": "LA"}}]'),
        ["<think>reason</think>\n", '[{"name": "get_weather", "arguments": {"city": "LA"}}]'],
        ['<think>reason</think>[{"name": "get_weather", "arguments": {"city": "LA"}}]'],
    ],
)
def test_audit_item4c_xlam_think_json_split_across_chunks(chunks):
    text = "".join(chunks)
    visible, calls = _run_delegating_stream("xlam", text, chunks)
    assert visible.strip() == "<think>reason</think>"
    assert '"get_weather"' not in visible
    assert calls == [("get_weather", {"city": "LA"})]


# =============================================================================
# Audit Follow-Up Regressions (#144): Items 2, 5, 6, 10, 13 (Misc Tool Parsers)
# =============================================================================


def test_audit_item2_mistral_multi_tool_partial_name_not_truncated():
    chunks = [
        '[TOOL_CALLS][{"name": "ping", "arguments": {}}',
        ', {"name": "get_',
        'weather", "arguments": {"city": "NYC"}}]',
    ]
    text = "".join(chunks)
    visible, calls = _run_delegating_stream("mistral", text, chunks)
    assert visible == ""
    assert calls == [("ping", {}), ("get_weather", {"city": "NYC"})]


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item5_phi4mini_malformed_functools_does_not_wedge_or_erase_tail(mode):
    text = "Hi functools[not json] bye"
    visible, calls = _run_delegating_stream("phi4_mini_json", text, _chunk_text(text, mode))
    assert visible == text
    assert calls == []


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item6_gigachat3_prose_function_call_does_not_wedge_or_erase_tail(mode):
    text = 'Hi: function call{"not_tool": 1} bye!'
    visible, calls = _run_delegating_stream("gigachat3", text, _chunk_text(text, mode))
    assert visible == text
    assert calls == []


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item10_functiongemma_and_gemma4_dense_indices_unclosed_and_quotes(mode):
    fg_malformed = (
        "<start_function_call>bad<end_function_call>"
        "<start_function_call>call:get_weather{city:<escape>NYC<escape>}<end_function_call>"
    )
    _vis, fg_calls = _run_delegating_stream("functiongemma", fg_malformed, _chunk_text(fg_malformed, mode))
    assert fg_calls == [("get_weather", {"city": "NYC"})]

    g4_malformed = (
        "<|tool_call>bad<tool_call|>"
        '<|tool_call>call:get_weather{city:<|"|>NYC<|"|>}<tool_call|>'
    )
    _vis, g4_calls = _run_delegating_stream("gemma4", g4_malformed, _chunk_text(g4_malformed, mode))
    assert g4_calls == [("get_weather", {"city": "NYC"})]

    fg_parser = ToolParserManager.get_tool_parser("functiongemma")(_UniversalTokenizer())
    unclosed = fg_parser.extract_tool_calls("<start_function_call>call:get_weather{count:1}", _make_request())
    assert unclosed.tools_called is True
    assert len(unclosed.tool_calls) == 1
    assert json.loads(unclosed.tool_calls[0].function.arguments) == {"count": 1}

    quoted = fg_parser.extract_tool_calls(
        '<start_function_call>call:get_weather{city:<escape>"42"<escape>}<end_function_call>',
        _make_request(),
    )
    assert quoted.tools_called is True
    assert json.loads(quoted.tool_calls[0].function.arguments) == {"city": '"42"'}


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item13_ernie45_granite20b_hunyuan_jamba_internlm2_cleanups(mode):
    ernie_text = (
        '<tool_call>{"name": "broken"</tool_call>'
        '<tool_call>{"name": "get_weather", "arguments": {"city": "LA"}}</tool_call>'
    )
    _v, ernie_calls = _run_delegating_stream("ernie45", ernie_text, _chunk_text(ernie_text, mode))
    assert ernie_calls == [("get_weather", {"city": "LA"})]

    g20_text = (
        '<function_call>{"name": "ping", "arguments": {}}'
        " middle text "
        '<function_call>{"name": "get_weather", "arguments": {"city": "LA"}}'
    )
    _v, g20_calls = _run_delegating_stream("granite-20b-fc", g20_text, _chunk_text(g20_text, mode))
    assert g20_calls == [("ping", {}), ("get_weather", {"city": "LA"})]

    hy_parser = ToolParserManager.get_tool_parser("hunyuan_a13b")(_UniversalTokenizer())
    hy_empty = hy_parser.extract_tool_calls("Hello <tool_calls>{}</tool_calls>", _make_request())
    assert hy_empty.tools_called is False
    assert hy_empty.content == "Hello <tool_calls>{}</tool_calls>"

    hy_multi = (
        '<tool_calls>{"bad": 1}</tool_calls>'
        '<tool_calls>[{"bad": 1}, {"name": "get_weather", "arguments": {"city": "LA"}}]</tool_calls>'
    )
    _v, hy_calls = _run_delegating_stream("hunyuan_a13b", hy_multi, _chunk_text(hy_multi, mode))
    assert hy_calls == [("get_weather", {"city": "LA"})]

    jamba_multi = (
        "<tool_calls>{bad}</tool_calls>"
        '<tool_calls>[{"bad": 1}, {"name": "get_weather", "arguments": {"city": "LA"}}]</tool_calls>'
    )
    _v, jamba_calls = _run_delegating_stream("jamba", jamba_multi, _chunk_text(jamba_multi, mode))
    assert jamba_calls == [("get_weather", {"city": "LA"})]

    intern_multi = (
        "<|action_start|><|plugin|>{bad}<|action_end|>"
        '<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "LA"}}<|action_end|>'
    )
    _v, intern_calls = _run_delegating_stream("internlm", intern_multi, _chunk_text(intern_multi, mode))
    assert intern_calls == [("get_weather", {"city": "LA"})]


# =============================================================================
# Audit Follow-Up Regressions (#144): Items 7, 8, 9, 11, 12, 13 (XML / AST / SeedOSS)
# =============================================================================


@pytest.mark.parametrize("mode", ["char", "random", "single"])
@pytest.mark.parametrize(
    "parser_name,prose",
    [
        ("minimax_m2", "Use <parameter name=x> here. Rest!"),
        ("qwen3_coder", "Use <parameter=x> here. Rest!"),
        ("step3p5", "Use <parameter=x> here. Rest!"),
        ("qwen3_xml", "Use <function invalid> here. More prose!"),
    ],
)
def test_audit_item7_inner_marker_prose_not_erased_under_delegating_parser(parser_name, prose, mode):
    visible, calls = _run_delegating_stream(parser_name, prose, _chunk_text(prose, mode))
    assert visible == prose
    assert calls == []


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item8_seed_oss_unclosed_last_parameter_does_not_swallow_following_content(mode):
    text = (
        "<seed:tool_call><function=get_weather><parameter=city>NYC<parameter=count>2</function></seed:tool_call>"
        " mid "
        "<seed:tool_call><function=get_weather><parameter=city>LA</parameter></function></seed:tool_call>"
    )
    visible, calls = _run_delegating_stream("seed_oss", text, _chunk_text(text, mode))
    assert visible == " mid "
    assert calls == [("get_weather", {"city": "NYC", "count": 2}), ("get_weather", {"city": "LA"})]


@pytest.mark.parametrize(
    "parser_name,payload",
    [
        ("pythonic", '[get_weather(city=b"x")]'),
        ("pythonic", "[get_weather(rate=1j)]"),
        ("pythonic", "[get_weather(city=...)]"),
        ("pythonic", '[get_weather(meta={b"a": 1})]'),
        ("llama4_pythonic", '<|python_start|>[get_weather(city=b"x")]<|python_end|>'),
        ("olmo3", '<function_calls>\nget_weather(rate=1j)\n</function_calls>'),
    ],
)
def test_audit_item9_pythonic_non_json_constants_fall_back_gracefully(parser_name, payload):
    tok = _UniversalTokenizer()
    parser = ToolParserManager.get_tool_parser(parser_name)(tok)
    req = _make_request()
    batch = parser.extract_tool_calls(payload, req)
    assert batch.tools_called is False
    assert batch.tool_calls == []
    # Also verify streaming never raises TypeError.
    acc = ""
    for ch in payload:
        parser.extract_tool_calls_streaming(acc, acc + ch, ch, [], [], [], req)
        acc += ch


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item11_qwen3coder_and_qwen3xml_truncated_and_unclosed_params(mode):
    tok = _UniversalTokenizer()
    q3c = ToolParserManager.get_tool_parser("qwen3_coder")(tok)
    trunc = q3c.extract_tool_calls("Hello <function=truncated_no_gt", _make_request())
    assert trunc.tools_called is False
    assert trunc.content == "Hello <function=truncated_no_gt"

    wrapped_trunc = q3c.extract_tool_calls("<tool_call><function=truncated_no_gt</tool_call>", _make_request())
    assert wrapped_trunc.tools_called is False
    assert wrapped_trunc.tool_calls == []

    bare_unclosed = "<function=get_weather><parameter=city>bare_unclosed_param</function>"
    for pname in ("qwen3_coder", "qwen3_xml"):
        _v, calls = _run_delegating_stream(pname, bare_unclosed, _chunk_text(bare_unclosed, mode))
        assert calls == [("get_weather", {"city": "bare_unclosed_param"})]

    no_fn_end = "<tool_call><function=get_weather><parameter=city>no_fn_end</parameter></tool_call>"
    _v, xml_calls = _run_delegating_stream("qwen3_xml", no_fn_end, _chunk_text(no_fn_end, mode))
    assert xml_calls == [("get_weather", {"city": "no_fn_end"})]


@pytest.mark.parametrize("mode", ["char", "random", "single"])
def test_audit_item12_olmo3_unclosed_bare_prose_and_indented_multicall_block(mode):
    unclosed_prose = "Hello(world"
    visible, calls = _run_delegating_stream("olmo3", unclosed_prose, _chunk_text(unclosed_prose, mode))
    assert visible == unclosed_prose
    assert calls == []

    indented = '<function_calls>\n  ping()\n  get_weather(city="LA")\n</function_calls>'
    visible, calls = _run_delegating_stream("olmo3", indented, _chunk_text(indented, mode))
    assert visible == ""
    assert calls == [("ping", {}), ("get_weather", {"city": "LA"})]


def test_audit_item13_deepseekv32_union_schema_null_and_seedoss_earliest_end_tag():
    tok = _UniversalTokenizer()
    req = _make_request(
        [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "val": {"type": ["string", "integer"]},
                        },
                    },
                },
            }
        ]
    )
    ds32_text = (
        "<｜DSML｜function_calls>"
        '<｜DSML｜invoke name="f">'
        '<｜DSML｜parameter name="val" string="false">null</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜function_calls>"
    )
    ds32 = ToolParserManager.get_tool_parser("deepseek_v32")(tok)
    batch = ds32.extract_tool_calls(ds32_text, req)
    assert batch.tools_called is True
    assert json.loads(batch.tool_calls[0].function.arguments) == {"val": "null"}
    _v, streamed_calls = _run_delegating_stream("deepseek_v32", ds32_text, list(ds32_text), request=req)
    assert streamed_calls == [("f", {"val": "null"})]

    # SeedOSSReasoningParser under assume_reasoning=True must pick earliest end tag.
    rp = SeedOSSReasoningParser(tok)
    rp.assume_reasoning = True
    text = "step 1</think>Answer mentions </seed:think> literally."
    reasoning, content = rp.extract_reasoning(text)
    assert reasoning == "step 1"
    assert content == "Answer mentions </seed:think> literally."
    # Streaming parity across 1-char deltas.
    rp_s = SeedOSSReasoningParser(tok)
    rp_s.assume_reasoning = True
    acc_r, acc_c = "", ""
    prev = ""
    for ch in text:
        cur = prev + ch
        delta = rp_s.extract_reasoning_streaming(prev, cur, ch, [], [], [])
        if delta is not None:
            if delta.reasoning_content:
                acc_r += delta.reasoning_content
            if delta.content:
                acc_c += delta.content
        prev = cur
    fin = rp_s.finalize_streaming(prev)
    if fin is not None:
        if fin.reasoning_content:
            acc_r += fin.reasoning_content
        if fin.content:
            acc_c += fin.content
    assert acc_r == "step 1"
    assert acc_c == "Answer mentions </seed:think> literally."


# =============================================================================
# Leak / ReDoS / O(N^2)-O(N^3) / Regression Audit (#191 items #10..#14, #24, #27, #28)
# =============================================================================


@pytest.mark.parametrize(
    "parser_name,prefix",
    [
        ("hermes", "<scratch_pad>Use <tool_call> format</scratch_pad>"),
        ("longcat", "<scratch_pad>Use <longcat_tool_call> format</scratch_pad>"),
        ("hunyuan_a13b", "<think>Use <tool_calls> format</think>"),
        ("deepseek_v3", "Mention <｜tool▁call▁begin｜> in prose."),
        ("deepseek_v31", "Mention <｜tool▁call▁begin｜> in prose."),
        ("kimi_k2", "Mention <|tool_call_begin|> in prose."),
        ("step3", "Mention <｜tool_call_begin｜> in prose."),
    ],
)
def test_leak_10_valid_input_no_suppressed_buffer_growth_or_prose_loss(parser_name, prefix):
    tok = _UniversalTokenizer()
    cls = ToolParserManager.get_tool_parser(parser_name)
    dp = DelegatingParser(tool_parser=cls(tok), tool_request=_make_request())
    chunks = [prefix] + [" hello"] * 120
    prev = ""
    visible = ""
    max_supp = 0
    for ch in chunks:
        cur = prev + ch
        res = dp.process_delta(cur, ch, [], prev, [])
        max_supp = max(max_supp, len(dp._suppressed_buffer_text))
        visible += res.delta_content or ""
        prev = cur
    fin = dp.process_final(prev, [])
    visible += fin.delta_content or ""
    assert max_supp == 0
    assert dp.phase != ParsePhase.BUFFERING
    assert visible == fin.accumulated_content == "".join(chunks)


@pytest.mark.parametrize(
    "parser_name,malformed",
    [
        ("deepseek_v32", "<｜DSML｜function_calls><｜DSML｜function_calls>bad</｜DSML｜function_calls>"),
        ("ernie45", "<tool_call>bad <tool_call>bad</tool_call>"),
        ("glm-4.5", "<tool_call>bad <tool_call>bad</tool_call>"),
        ("glm-4.7", "<tool_call>bad <tool_call>bad</tool_call>"),
        ("functiongemma", "<start_function_call>bad <start_function_call>bad<end_function_call>"),
        ("gemma4", "<|tool_call>bad <|tool_call>bad<tool_call|>"),
        ("internlm", "<|action_start|><|plugin|>bad <|action_start|><|plugin|>bad<|action_end|>"),
        ("jamba", "<tool_calls>bad <tool_calls>bad</tool_calls>"),
        ("olmo3", '<function_calls>x = "<function_calls>"</function_calls>'),
        ("minimax_m2", "<minimax:tool_call><minimax:tool_call></minimax:tool_call>"),
    ],
)
def test_leak_10_duplicate_open_block_no_suppressed_buffer_growth_or_prose_loss(parser_name, malformed):
    tok = _UniversalTokenizer()
    cls = ToolParserManager.get_tool_parser(parser_name)
    parser = cls(tok)
    dp = DelegatingParser(tool_parser=parser, tool_request=_make_request())
    tail_chunks = [" prose"] * 120
    expected_tail = "".join(tail_chunks)
    prev = ""
    visible = ""
    max_supp = 0
    for ch in [malformed, *tail_chunks]:
        cur = prev + ch
        res = dp.process_delta(cur, ch, [], prev, [])
        max_supp = max(max_supp, len(dp._suppressed_buffer_text))
        visible += res.delta_content or ""
        prev = cur
    assert parser.is_buffering_protocol(current_text=prev, delta_text=" prose") is False
    fin = dp.process_final(prev, [])
    visible += fin.delta_content or ""
    assert max_supp == 0
    assert visible == fin.accumulated_content
    assert visible.endswith(expected_tail)


@pytest.mark.parametrize("parser_name,char", [("gigachat3", "f"), ("granite", "<")])
def test_leak_10_repeated_1char_prefix_not_dropped_during_streaming(parser_name, char):
    tok = _UniversalTokenizer()
    cls = ToolParserManager.get_tool_parser(parser_name)
    dp = DelegatingParser(tool_parser=cls(tok), tool_request=_make_request())
    chunks = [char] * 200 + ["!"]
    prev = ""
    streamed = ""
    max_supp = 0
    for ch in chunks:
        cur = prev + ch
        res = dp.process_delta(cur, ch, [], prev, [])
        max_supp = max(max_supp, len(dp._suppressed_buffer_text))
        streamed += res.delta_content or ""
        prev = cur
    assert max_supp == 0
    assert streamed == char * 200 + "!"
    fin = dp.process_final(prev, [])
    assert fin.accumulated_content == char * 200 + "!"


def test_leak_10_legacy_parser_suppressed_buffer_has_safety_cap():
    class _StuckLegacyParser(ToolParser):
        tool_call_start_token = "<legacy_call>"
        tool_call_end_token = "</legacy_call>"

        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        def extract_tool_calls_streaming(self, **kwargs):
            return DeltaMessage(content=kwargs["delta_text"])

        def is_buffering_protocol(self, *, current_text="", delta_text=""):
            return True

    dp = DelegatingParser(tool_parser=_StuckLegacyParser(_UniversalTokenizer()), tool_request=_make_request())
    prev = ""
    streamed = ""
    max_supp = 0
    for _ in range(400):
        ch = "abcdefghij"
        cur = prev + ch
        res = dp.process_delta(cur, ch, [], prev, [])
        max_supp = max(max_supp, len(dp._suppressed_buffer_text))
        streamed += res.delta_content or ""
        prev = cur
    assert max_supp <= dp._max_suppressed_buffer_chars
    assert len(streamed) >= 3000


def test_leak_11_gigachat3_whitespace_run_dead_set_and_post_complete_reset():
    tok = _UniversalTokenizer()
    cls = ToolParserManager.get_tool_parser("gigachat3")
    req = _make_request()

    # Whitespace run must not stall _cursor, grow content_buffer, or rescan O(N^2).
    p_ws = cls(tok)
    assert not hasattr(p_ws, "_skipped_header_starts")
    t0 = time.perf_counter()
    prev = "Hello."
    out = ""
    d0 = p_ws.extract_tool_calls_streaming("", prev, prev, [], [], [], req)
    out += (d0.content or "") if d0 else ""
    for _ in range(500):
        cur = prev + " "
        d = p_ws.extract_tool_calls_streaming(prev, cur, " ", [], [], [], req)
        out += (d.content or "") if d else ""
        prev = cur
    elapsed_ws = time.perf_counter() - t0
    assert p_ws.content_buffer == ""
    assert p_ws.is_buffering_protocol(current_text=prev, delta_text=" ") is False
    assert out == "Hello." + " " * 500
    assert elapsed_ws < 0.5

    # Post-completion must clear tool_started and not re-parse trailing tokens.
    p_done = cls(tok)
    call_text = 'function call{"name":"ping","arguments":{}}'
    d = p_done.extract_tool_calls_streaming("", call_text, call_text, [], [], [], req)
    assert d is not None and d.tool_calls
    assert p_done.tool_started is False
    assert p_done.is_buffering_protocol(current_text=call_text, delta_text=call_text) is False
    prev = call_text
    t0 = time.perf_counter()
    for _ in range(300):
        cur = prev + " ok"
        p_done.extract_tool_calls_streaming(prev, cur, " ok", [], [], [], req)
        prev = cur
    assert time.perf_counter() - t0 < 0.5
    assert p_done.tool_started is False


def test_leak_12_ernie45_unclosed_string_spanning_end_tag_bounded_and_embedded_preserved():
    tok = _UniversalTokenizer()
    cls = ToolParserManager.get_tool_parser("ernie45")
    req = _make_request()

    # Unclosed JSON string spanning </tool_call> must not pin 100% of stream in _buffer.
    p = cls(tok)
    dp = DelegatingParser(tool_parser=p, tool_request=req)
    bad_prefix = '<tool_call>{"name": "get_weather", "arguments": {"x": "unclosed</tool_call>'
    tail_chunks = [" prose"] * 200
    prev = ""
    visible = ""
    t0 = time.perf_counter()
    for ch in [bad_prefix, *tail_chunks]:
        cur = prev + ch
        res = dp.process_delta(cur, ch, [], prev, [])
        visible += res.delta_content or ""
        prev = cur
    elapsed = time.perf_counter() - t0
    fin = dp.process_final(prev, [])
    visible += fin.delta_content or ""
    assert len(p._buffer) == 0
    assert elapsed < 0.5
    assert " prose" * 50 in visible

    # Valid JSON string containing literal </tool_call> must still parse.
    valid = '<tool_call>{"name": "get_weather", "arguments": {"city": "a</tool_call>b"}}</tool_call>'
    _vis, calls = _run_delegating_stream("ernie45", valid, _chunk_text(valid, "random"), request=req)
    assert calls == [("get_weather", {"city": "a</tool_call>b"})]


def test_leak_13_no_cubic_regex_or_quadratic_streaming_blowups():
    from easydel.inference.tools.utils import find_common_suffix

    tok = _UniversalTokenizer()
    req = _make_request()

    # 1. KimiK2 batch regex: unclosed call + 2000 spaces (previously 4.4s).
    kimi = ToolParserManager.get_tool_parser("kimi_k2")(tok)
    t0 = time.perf_counter()
    kimi.extract_tool_calls("<|tool_call_begin|>a:0<|tool_call_argument_begin|>" + " " * 2000, req)
    assert time.perf_counter() - t0 < 0.2

    # 2. DeepSeekV31 batch regex: 500x repeated unclosed calls (previously 8.5s).
    dsv31 = ToolParserManager.get_tool_parser("deepseek_v31")(tok)
    t0 = time.perf_counter()
    dsv31.extract_tool_calls("<｜tool▁call▁begin｜>a<｜tool▁sep｜>" * 500, req)
    assert time.perf_counter() - t0 < 0.2

    # 3. GLM4 / GLM4.7 func_arg_regex: 200x repeated unclosed arg_value (previously 0.57s).
    glm_payload = "<tool_call>get_weather" + "<arg_key>k</arg_key><arg_value>v" * 200 + "</tool_call>"
    for name in ("glm-4.5", "glm-4.7"):
        gp = ToolParserManager.get_tool_parser(name)(tok)
        t0 = time.perf_counter()
        gp.extract_tool_calls(glm_payload, req)
        assert time.perf_counter() - t0 < 0.2
        t0 = time.perf_counter()
        gp.extract_tool_calls_streaming("", glm_payload, glm_payload, [], [], [], req)
        assert time.perf_counter() - t0 < 0.2

    # 4. Gemma4 streaming large array (no O(N^2) suffix copy per char).
    gemma = ToolParserManager.get_tool_parser("gemma4")(tok)
    arr_chunks = ["<|tool_call>call:get_weather{items:[1"] + [",1"] * 400 + ["]}<tool_call|>"]
    prev = ""
    t0 = time.perf_counter()
    for ch in arr_chunks:
        cur = prev + ch
        gemma.extract_tool_calls_streaming(prev, cur, ch, [], [], [], req)
        prev = cur
    assert time.perf_counter() - t0 < 0.5

    # 5. Hermes & Longcat prose streaming (previously >2.1s on 2k chunks).
    for name in ("hermes", "longcat"):
        hp = ToolParserManager.get_tool_parser(name)(tok)
        prev = ""
        t0 = time.perf_counter()
        for _ in range(1000):
            ch = "hello "
            cur = prev + ch
            hp.extract_tool_calls_streaming(prev, cur, ch, [], [], [], req)
            prev = cur
        assert time.perf_counter() - t0 < 0.5

    # 6. find_common_suffix O(N) slice.
    s = "!" * 20000
    t0 = time.perf_counter()
    assert len(find_common_suffix("a" + s, "b" + s)) == 20000
    assert time.perf_counter() - t0 < 0.1


def test_leak_14_mismatched_brackets_ast_overflow_recursion_and_schema_bounds():
    tok = _UniversalTokenizer()
    req = _make_request()

    # 1. Mismatched brackets ({], [}, {}}) must not crash or wedge streaming parsers.
    for pname, bad_prefix, recovery in [
        ("llama3_json", "{] ", '<|python_tag|>{"name": "ping", "arguments": {}}'),
        ("minimax", "<tool_calls>{]</tool_calls> ", '<tool_calls>{"name": "ping", "arguments": {}}</tool_calls>'),
        ("openai", "{] ", '```json\n{"name": "ping", "arguments": {}}\n```'),
        ("xlam", "{] ", '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'),
        ("granite", "<|tool_call|>[} ", '<|tool_call|>[{"name": "ping", "arguments": {}}]'),
    ]:
        dp = DelegatingParser(tool_parser=ToolParserManager.get_tool_parser(pname)(tok), tool_request=req)
        full = bad_prefix + recovery
        prev = ""
        for ch in list(full):
            cur = prev + ch
            dp.process_delta(cur, ch, [], prev, [])
            prev = cur
        fin = dp.process_final(prev, [])
        assert fin.tool_calls is not None and fin.tool_calls[0].function.name == "ping", pname

    # 2. CPython 3.13 ast.parse MemoryError & deep JSON RecursionError must not escape.
    ast_bomb = "[a" + "{" * 1000 + "]"
    for pname in ("pythonic", "olmo3", "llama4_pythonic"):
        p = ToolParserManager.get_tool_parser(pname)(tok)
        p.extract_tool_calls(ast_bomb, req)
        ps = ToolParserManager.get_tool_parser(pname)(tok)
        ps.extract_tool_calls_streaming("", ast_bomb, ast_bomb, [], [], [], req)

    json_bomb = "[" * 15000 + "]" * 15000
    for pname in ("openai", "step3", "llama3_json", "minimax", "xlam", "phi4_mini_json"):
        p = ToolParserManager.get_tool_parser(pname)(tok)
        p.extract_tool_calls(json_bomb, req)
        ps = ToolParserManager.get_tool_parser(pname)(tok)
        ps.extract_tool_calls_streaming("", json_bomb, json_bomb, [], [], [], req)

    # 3. _schema_params union conversion guards >4300-digit ints & deep nesting.
    union_cfg = {"v": {"anyOf": [{"type": "array"}, {"type": "string"}]}}
    huge_int_arr = "[" + "1" + "0" * 5000 + "]"
    deep_arr = "[" * 1500 + "]" * 1500
    assert convert_param_value(huge_int_arr, "v", union_cfg, "fn") == huge_int_arr
    assert convert_param_value(deep_arr, "v", union_cfg, "fn") == deep_arr


def test_leak_24_stuck_state_machines_and_streaming_linear_complexity():
    tok = _UniversalTokenizer()
    req = _make_request()

    # 1. minimax_m2 missing </invoke> before </minimax:tool_call> must recover.
    mm2_text = (
        '<minimax:tool_call><invoke name="bad">oops</minimax:tool_call>'
        " middle "
        '<minimax:tool_call><invoke name="ping"></invoke></minimax:tool_call>'
    )
    vis, calls = _run_delegating_stream("minimax_m2", mm2_text, _chunk_text(mm2_text, "random"), request=req)
    assert "middle" in vis
    assert any(name == "ping" for name, _ in calls)

    # 2. step3 missing TOOL_CALL_END before TOOL_CALLS_END must recover.
    step3_text = (
        "<｜tool_calls_begin｜><｜tool_call_begin｜>bad<｜tool_calls_end｜>"
        " middle "
        "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
        "<steptml:invoke name=\"ping\"></steptml:invoke>"
        "<｜tool_call_end｜><｜tool_calls_end｜>"
    )
    vis, calls = _run_delegating_stream("step3", step3_text, _chunk_text(step3_text, "random"), request=req)
    assert "middle" in vis
    assert any(name == "ping" for name, _ in calls)

    # 3. llama4_pythonic resets _in_delimited_block & preserves trailing prose (stream + batch).
    l4_bad = "<|python_start|>not a list<|python_end|> trailing prose"
    vis, calls = _run_delegating_stream("llama4_pythonic", l4_bad, _chunk_text(l4_bad, "char"), request=req)
    assert vis.endswith(" trailing prose")
    assert calls == []

    l4_good = "<|python_start|>[ping()]<|python_end|> trailing prose"
    vis, calls = _run_delegating_stream("llama4_pythonic", l4_good, _chunk_text(l4_good, "char"), request=req)
    assert calls == [("ping", {})]
    assert vis.endswith(" trailing prose")
    l4_batch = ToolParserManager.get_tool_parser("llama4_pythonic")(tok).extract_tool_calls(l4_good, req)
    assert l4_batch.tools_called is True
    assert l4_batch.tool_calls[0].function.name == "ping"
    assert l4_batch.content is not None and "trailing prose" in l4_batch.content

    # 4. seed_oss stray <function=> must not swallow trailing prose.
    seed_text = "Hello <function=> trailing prose ok"
    vis, calls = _run_delegating_stream("seed_oss", seed_text, _chunk_text(seed_text, "char"), request=req)
    assert vis == seed_text
    assert calls == []

    # 5. xlam non-JSON [TOOL_CALLS] & unclosed generic fence must not stall streaming.
    xlam_dp = DelegatingParser(tool_parser=ToolParserManager.get_tool_parser("xlam")(tok), tool_request=req)
    prev = ""
    streamed = ""
    for ch in ["See [TOOL_CALLS] in docs", *([" more"] * 40)]:
        cur = prev + ch
        res = xlam_dp.process_delta(cur, ch, [], prev, [])
        streamed += res.delta_content or ""
        prev = cur
    assert " more" * 30 in streamed

    for pname in ("xlam", "openai"):
        dp = DelegatingParser(tool_parser=ToolParserManager.get_tool_parser(pname)(tok), tool_request=req)
        prev = ""
        streamed = ""
        for ch in ["```\nnot json plain text", *([" line"] * 20)]:
            cur = prev + ch
            res = dp.process_delta(cur, ch, [], prev, [])
            streamed += res.delta_content or ""
            prev = cur
        assert " line" * 10 in streamed

    # 6. granite-20b-fc & mistral malformed markers do not stick in BUFFERING.
    for pname, text in [
        ("granite-20b-fc", "Note: <function_call> is not json. " + "ok " * 40),
        ("mistral", "Note: [TOOL_CALLS] is not json. " + "ok " * 40),
    ]:
        vis, calls = _run_delegating_stream(pname, text, _chunk_text(text, "random"), request=req)
        assert vis == text
        assert calls == []

    # 7. O(N) trailing prose timing for pythonic, llama4_pythonic, olmo3, mistral, minimax.
    for pname, call_prefix in [
        ("pythonic", "[ping()]"),
        ("llama4_pythonic", "[ping()]"),
        ("olmo3", "ping()"),
        ("mistral", '[TOOL_CALLS][{"name":"ping","arguments":{}}]'),
        ("minimax", '<tool_calls>{"name":"get_weather","arguments":{"city":"' + "x" * 4000 + '"}}</tool_calls>'),
    ]:
        p = ToolParserManager.get_tool_parser(pname)(tok)
        p.extract_tool_calls_streaming("", call_prefix, call_prefix, [], [], [], req)
        prev = call_prefix
        t0 = time.perf_counter()
        for _ in range(400):
            ch = " prose"
            cur = prev + ch
            p.extract_tool_calls_streaming(prev, cur, ch, [], [], [], req)
            prev = cur
        assert time.perf_counter() - t0 < 0.5, pname

    # 8. DelegatingParser._tokenize_for_tool_view_append stays O(1) on whitespace-free text.
    class _CountingCharTok:
        def __init__(self):
            self.encoded_chars = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            self.encoded_chars += len(text)
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            return "".join(chr(i) for i in ids)

    class _PassThroughTP(ToolParser):
        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        def extract_tool_calls_streaming(self, **kwargs):
            return DeltaMessage(content=kwargs["delta_text"])

    ctok = _CountingCharTok()
    dp_inc = DelegatingParser(tool_parser=_PassThroughTP(ctok))
    chunks = ['{"k":"v",'] * 40
    cum = ""
    for ch in chunks:
        cum += ch
        dp_inc._process_tool_delta(ParseResult(), cum, ch, [])
        assert dp_inc._tool_previous_token_ids == [ord(c) for c in cum]
    naive_chars = sum(len("".join(chunks[: i + 1])) for i in range(len(chunks)))
    assert ctok.encoded_chars < naive_chars // 2


def test_leak_27_post_flush_text_release_and_batch_singleton_statelessness():
    tok = _UniversalTokenizer()
    req = _make_request()

    seen = set()
    for _name, cls in sorted(ToolParserManager.tool_parsers.items()):
        if cls in seen or not cls.__module__.startswith("easydel.inference.tools.parsers"):
            continue
        seen.add(cls)
        p = cls(tok)
        prose = "Ordinary streaming output sentence. " * 25
        prev = ""
        for i in range(0, len(prose), 18):
            ch = prose[i : i + 18]
            cur = prev + ch
            p.extract_tool_calls_streaming(prev, cur, ch, [], [], [], req)
            prev = cur
        p.flush_streaming_content()
        assert (p.flush_streaming_content() or "") == "", cls.__name__
        for attr, val in p.__dict__.items():
            if isinstance(val, str):
                assert len(val) <= 200, f"{cls.__name__}.{attr} retained {len(val)} chars post-flush"
            if attr == "streaming_request":
                assert val is None, f"{cls.__name__}.streaming_request not cleared on flush"

    # DeepSeekV32 batch extract_tool_calls must not mutate shared singleton failed_count.
    ds32 = ToolParserManager.get_tool_parser("deepseek_v32")(tok)
    assert ds32.failed_count == 0
    ds32.extract_tool_calls("<｜DSML｜function_calls><｜DSML｜invoke name=\"broken\">", req)
    assert ds32.failed_count == 0

    # Qwen3XML must not retain write-only streaming_request.
    q3x = ToolParserManager.get_tool_parser("qwen3_xml")(tok)
    q3x.extract_tool_calls_streaming("", "hi", "hi", [], [], [], req)
    assert getattr(q3x, "streaming_request", None) is None


def test_leak_28_mixin_thread_safety_and_no_shared_singleton_fallback():
    tok = _UniversalTokenizer()
    req = _make_request()

    # 1. Multithreaded concurrent stream & abort stress on ReasoningMixin and ToolCallingMixin.
    rmixin = ReasoningMixin()
    rmixin.reasoning_parsers = {"m": DeepSeekR1ReasoningParser(tok)}
    tmixin = ToolCallingMixin()
    tmixin.tool_parsers = {"m": ToolParserManager.get_tool_parser("hermes")(tok)}

    errors: list[BaseException] = []

    def _worker(tid: int) -> None:
        try:
            for step in range(25):
                sid = f"t{tid}-s{step}"
                r1 = rmixin.extract_reasoning_streaming("m", "", "<think>r", "<think>r", stream_id=sid)
                r2 = rmixin.extract_reasoning_streaming(
                    "m", "<think>r", "<think>r</think>ok", "</think>ok", stream_id=sid, final=True
                )
                assert r1 is not None and r2 is not None and r2.content == "ok"

                msg = '<tool_call>{"name":"ping","arguments":{}}</tool_call>'
                mid = len(msg) // 2
                t1 = tmixin.extract_tool_calls_streaming("m", "", msg[:mid], msg[:mid], request=req, stream_id=sid)
                t2 = tmixin.extract_tool_calls_streaming(
                    "m", msg[:mid], msg, msg[mid:], request=req, stream_id=sid, finished=True
                )
                seen_names = [
                    tc.function.name
                    for delta in (t1, t2)
                    if delta is not None and delta.tool_calls
                    for tc in delta.tool_calls
                    if tc.function and tc.function.name
                ]
                assert seen_names == ["ping"]
                if step % 5 == 0:
                    rmixin.abort_reasoning_stream("m", stream_id=f"abort-{sid}")
                    tmixin.abort_tool_stream("m", stream_id=f"abort-{sid}")
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(getattr(rmixin, "_mixin_reasoning_streams", {})) == 0
    assert len(getattr(tmixin, "_mixin_tool_streams", {})) == 0

    # 2. ToolCallingMixin must NOT silently fall back to a shared mutable ToolParser singleton.
    class _UnclonableToolParser(ToolParser):
        def __init__(self, tokenizer):
            if tokenizer is None:
                raise RuntimeError("tokenizer required")
            super().__init__(tokenizer)
            self.mutated = False

        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        def extract_tool_calls_streaming(self, **kwargs):
            self.mutated = True
            return DeltaMessage(content=kwargs["delta_text"])

    shared_singleton = _UnclonableToolParser(tok)
    shared_singleton.model_tokenizer = None
    assert ToolCallingMixin._new_stream_parser(shared_singleton) is None

    tmixin_iso = ToolCallingMixin()
    tmixin_iso.tool_parsers = {"m": shared_singleton}
    delta = tmixin_iso.extract_tool_calls_streaming("m", "", "hello", "hello", request=req, finished=True)
    assert delta is not None and delta.content == "hello"
    assert shared_singleton.mutated is False
    assert len(getattr(tmixin_iso, "_mixin_tool_streams", {})) == 0


def test_leak_verifier_234_huge_int_gemma4_depth_delegating_prefix_and_seedoss_cursor():
    tok = _UniversalTokenizer()
    req = _make_request()
    huge = "1" + "0" * 5000

    # 1. >4300-digit integer literals must never wedge DelegatingParser or crash is_buffering_protocol.
    for pname, payload in [
        ("llama3_json", f'{{"name":"fn","arguments":{{"x":{huge}}}}}'),
        ("minimax", f'<tool_calls>{{"name":"fn","arguments":{{"x":{huge}}}}}</tool_calls>'),
        ("phi4_mini_json", f'functools[{{"name":"fn","arguments":{{"x":{huge}}}}}]'),
        ("xlam", f'[TOOL_CALLS]{{"name":"fn","arguments":{{"x":{huge}}}}}'),
        ("mistral", f'[TOOL_CALLS][{{"name":"fn","arguments":{{"x":{huge}}}}}]'),
        ("mistral", f'[TOOL_CALLS]fn{{"x":{huge}}}'),
    ]:
        cls = ToolParserManager.get_tool_parser(pname)
        assert isinstance(cls(tok).is_buffering_protocol(current_text=payload, delta_text=""), bool)
        dp = DelegatingParser(tool_parser=cls(tok), tool_request=req)
        mid = len(payload) // 2
        dp.process_delta(payload[:mid], payload[:mid], [], "", [])
        dp.process_delta(payload, payload[mid:], [], payload[:mid], [])
        assert dp._tool_previous_text == payload, pname
        dp.process_delta(payload + " ok", " ok", [], payload, [])
        assert dp._tool_previous_text == payload + " ok", pname
        dp.process_final(payload + " ok", [])

    # hunyuan_a13b & functiongemma huge-int / malformed JSON helpers.
    hy = ToolParserManager.get_tool_parser("hunyuan_a13b")(tok)
    hy._try_parse_json_tools(f'[{{"x":{huge}}}]')
    hy._ensure_state_arrays(1)
    hy.current_tools_sent[0] = True
    assert hy._handle_tool_args_streaming('{"name":"fn","arguments":{]}}', 0, 1) is None
    assert hy._handle_tool_args_streaming("[" * 15000 + "]" * 15000, 0, 1) is None

    fg = ToolParserManager.get_tool_parser("functiongemma")(tok)
    fg_res = fg.extract_tool_calls(f"<start_function_call>call:ping{{x:{huge},y:1}}<end_function_call>", req)
    assert fg_res.tools_called is True
    assert json.loads(fg_res.tool_calls[0].function.arguments) == {"x": huge, "y": 1}

    # 2. Gemma4 deeply nested containers must parse linearly in single pass with depth cap.
    gm = ToolParserManager.get_tool_parser("gemma4")(tok)
    for text in (
        "<|tool_call>call:fn{x:" + "[" * 5000 + "]" * 5000 + "}<tool_call|>",
        "<|tool_call>call:fn{x:" + "{a:" * 5000 + "}" * 5000 + "}<tool_call|>",
    ):
        t0 = time.perf_counter()
        gm.extract_tool_calls(text, req)
        assert time.perf_counter() - t0 < 0.2

    # 3. DelegatingParser._process_tool_delta prefix diff stays O(N) on long whitespace-free streams.
    class _RevBpe4Tok:
        def __init__(self):
            self.s2i: dict[str, int] = {}
            self.i2s: dict[int, str] = {}

        def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            out: list[int] = []
            for i in range(0, len(s), 4):
                piece = s[i : i + 4]
                tid = self.s2i.get(piece)
                if tid is None:
                    tid = len(self.s2i) + 1
                    self.s2i[piece] = tid
                    self.i2s[tid] = piece
                out.append(tid)
            return out

        def decode(self, ids, skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            return "".join(self.i2s[i] for i in ids)

    class _PassThroughTP(ToolParser):
        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        def extract_tool_calls_streaming(self, **kwargs):
            return DeltaMessage(content=kwargs["delta_text"])

    rt = _RevBpe4Tok()
    dp_lin = DelegatingParser(tool_parser=_PassThroughTP(rt))
    cum = ""
    t0 = time.perf_counter()
    for _ in range(1500):
        ch = "abcd"
        cum += ch
        dp_lin._process_tool_delta(ParseResult(), cum, ch, [])
    assert time.perf_counter() - t0 < 0.5
    assert dp_lin._tool_previous_token_ids == rt.encode(cum)

    # 4. SeedOSS trailing prose stays O(N) after closed call, and <seed:think> before DP engagement is honored.
    sp = ToolParserManager.get_tool_parser("seed_oss")(tok)
    prefix = "<seed:tool_call><function=ping></function></seed:tool_call>"
    sp.extract_tool_calls_streaming("", prefix, prefix, [], [], [], req)
    prev = prefix
    t0 = time.perf_counter()
    for _ in range(1500):
        ch = " prose"
        cur = prev + ch
        sp.extract_tool_calls_streaming(prev, cur, ch, [], [], [], req)
        assert sp.is_buffering_protocol(current_text=cur, delta_text=ch) is False
        prev = cur
    assert time.perf_counter() - t0 < 0.2

    seed_think_chunks = [
        "<seed:think>Let me show an example: ",
        "<seed:tool_call><function=fn></function></seed:tool_call></seed:think> Done.",
    ]
    vis_st, calls_st = _run_delegating_stream("seed_oss", "".join(seed_think_chunks), seed_think_chunks, request=req)
    assert calls_st == []
    assert vis_st == "".join(seed_think_chunks)

    # 5. Gemma4 stray closing brackets/braces at depth 0 must never loop infinitely.
    from easydel.inference.tools.parsers.gemma4_tool_parser import _parse_gemma4_args, _parse_gemma4_array

    assert _parse_gemma4_array("]") == []
    assert _parse_gemma4_array("1]2") == [1]
    assert _parse_gemma4_args("a:]b:1") == {"b": 1}
    assert _parse_gemma4_args("}") == {}
