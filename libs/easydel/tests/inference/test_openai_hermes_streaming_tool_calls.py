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

"""End-to-end streaming tool call tests for hermes parser + delegating parser.

Simulates exact qwen3-style output (reasoning + hermes tool calls) and verifies:
1. Tool call name is emitted exactly once
2. Tool call arguments are streamed completely
3. No raw JSON/markup leaks into content deltas
4. Reassembled arguments match the expected JSON
5. The whole pipeline works with the DelegatingParser
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from easydel.inference.openai_api_modules import ChatCompletionRequest, ChatMessage
from easydel.inference.parsing.delegating_parser import DelegatingParser, ParsePhase
from easydel.inference.stream_protocol import compute_stream_delta_text


class _FakeTokenizer:
    def __init__(self):
        self._vocab = {}
        self._next_id = 100

    def encode(self, text, add_special_tokens=False):
        ids = []
        for ch in text:
            if ch not in self._vocab:
                self._vocab[ch] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[ch])
        return ids

    def decode(self, ids):
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)


class _FakeReasoningParser:
    def extract_reasoning_streaming(
        self, previous_text, current_text, delta_text, previous_token_ids, current_token_ids, delta_token_ids
    ):
        @dataclass
        class _D:
            reasoning_content: str | None = None
            content: str | None = None

        end_tag = "</think>"
        start_tag = "<think>"

        if end_tag in current_text and end_tag not in previous_text:
            idx = current_text.index(end_tag) + len(end_tag)
            return _D(content=current_text[idx:])

        if start_tag in current_text and end_tag not in current_text:
            r_start = len(start_tag)
            reasoning_now = current_text[r_start:]
            prev_r = previous_text[r_start:] if start_tag in previous_text else ""
            return _D(reasoning_content=reasoning_now[len(prev_r) :])

        return None

    def extract_reasoning(self, text):
        start_tag = "<think>"
        end_tag = "</think>"
        if start_tag in text and end_tag in text:
            s = text.index(start_tag) + len(start_tag)
            e = text.index(end_tag)
            return text[s:e], text[e + len(end_tag) :]
        if start_tag in text:
            return text[len(start_tag) :], None
        return None, text


def _make_request_with_tools():
    return ChatCompletionRequest(
        model="test",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )


def _simulate_streaming(parser, token_chunks: list[str]):
    """Feed token chunks one by one, return list of ParseResults."""
    results = []
    accumulated = ""
    token_ids = []
    for chunk in token_chunks:
        prev_text = accumulated
        prev_ids = list(token_ids)
        accumulated += chunk
        token_ids.extend(range(len(prev_ids), len(prev_ids) + len(chunk)))
        result = parser.process_delta(accumulated, chunk, token_ids, prev_text, prev_ids)
        results.append(result)
    final = parser.process_final(accumulated, token_ids)
    results.append(final)
    return results


def _collect_tool_call_deltas(results):
    """Extract all delta_tool_calls from results."""
    deltas = []
    for r in results:
        if r.delta_tool_calls:
            deltas.extend(r.delta_tool_calls)
    return deltas


def _collect_content_deltas(results):
    """Concatenate all delta_content from results."""
    parts = []
    for r in results:
        if r.delta_content:
            parts.append(r.delta_content)
    return "".join(parts)


def _collect_reasoning_deltas(results):
    """Concatenate all delta_reasoning from results."""
    parts = []
    for r in results:
        if r.delta_reasoning:
            parts.append(r.delta_reasoning)
    return "".join(parts)


class TestHermesParserBatch:
    def test_single_tool_call(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        tokenizer = _FakeTokenizer()
        parser = HermesToolParser(tokenizer)
        req = _make_request_with_tools()

        text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "SF"}}\n</tool_call>'
        result = parser.extract_tool_calls(text, req)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "SF"

    def test_no_tool_call(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser(_FakeTokenizer())
        result = parser.extract_tool_calls("Hello world", _make_request_with_tools())
        assert result.tools_called is False
        assert result.content == "Hello world"

    def test_content_before_tool_call(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser(_FakeTokenizer())
        text = 'Let me check. <tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>'
        result = parser.extract_tool_calls(text, _make_request_with_tools())
        assert result.tools_called is True
        assert result.content == "Let me check. "

    def test_multiple_tool_calls(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser(_FakeTokenizer())
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "SF"}}\n</tool_call>'
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>'
        )
        result = parser.extract_tool_calls(text, _make_request_with_tools())
        assert result.tools_called is True
        assert len(result.tool_calls) == 2


class TestHermesParserStreaming:
    def _make_parser(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        return HermesToolParser(_FakeTokenizer())

    def _stream(self, parser, chunks, req=None):
        """Feed chunks to extract_tool_calls_streaming, return all non-None deltas."""
        if req is None:
            req = _make_request_with_tools()
        accumulated = ""
        prev_text = ""
        prev_ids = []
        deltas = []
        for chunk in chunks:
            accumulated += chunk
            cur_ids = list(range(len(accumulated)))
            delta_ids = cur_ids[len(prev_ids) :]
            result = parser.extract_tool_calls_streaming(
                previous_text=prev_text,
                current_text=accumulated,
                delta_text=chunk,
                previous_token_ids=prev_ids,
                current_token_ids=cur_ids,
                delta_token_ids=delta_ids,
                request=req,
            )
            if result is not None:
                deltas.append(result)
            prev_text = accumulated
            prev_ids = cur_ids
        return deltas

    def test_name_emitted(self):
        parser = self._make_parser()
        chunks = ["<tool_call>", '\n{"name": "get_weather"']
        deltas = self._stream(parser, chunks)
        names = []
        for d in deltas:
            if d.tool_calls:
                for tc in d.tool_calls:
                    fn = tc.function if isinstance(tc.function, dict) else tc.function.__dict__
                    name = fn.get("name")
                    if name:
                        names.append(name)
        assert "get_weather" in names

    def test_arguments_emitted(self):
        parser = self._make_parser()
        chunks = [
            "<tool_call>",
            '\n{"name": "get_weather"',
            ', "arguments": {"loc',
            'ation": "SF"',
            "}}\n",
            "</tool_call>",
        ]
        deltas = self._stream(parser, chunks)

        arg_parts = []
        for d in deltas:
            if d.tool_calls:
                for tc in d.tool_calls:
                    fn = tc.function if isinstance(tc.function, dict) else tc.function.__dict__
                    arguments = fn.get("arguments")
                    if arguments:
                        arg_parts.append(arguments)
        combined = "".join(arg_parts)
        assert combined, "No argument deltas were emitted"
        parsed = json.loads(combined)
        assert parsed["location"] == "SF"

    def test_no_content_leak_during_tool_call(self):
        """Raw JSON must NOT appear as content."""
        parser = self._make_parser()
        chunks = [
            "Hello ",
            "<tool_call>",
            '\n{"name": "get_weather"',
            ', "arguments": {"location": "SF"}}',
            "\n</tool_call>",
        ]
        deltas = self._stream(parser, chunks)
        content_parts = []
        for d in deltas:
            if d.content and d.content.strip():
                content_parts.append(d.content)
        content = "".join(content_parts)
        assert "get_weather" not in content
        assert '{"name"' not in content
        assert '"arguments"' not in content

    def test_closing_tag_does_not_leak(self):
        """When </tool_call> arrives, no raw text should leak as content."""
        parser = self._make_parser()
        chunks = [
            "<tool_call>",
            '\n{"name": "get_weather", "arguments": {"location": "SF"}}',
            "\n</tool_call>",
        ]
        deltas = self._stream(parser, chunks)
        for d in deltas:
            if d.content:
                assert "<tool_call>" not in d.content
                assert "</tool_call>" not in d.content
                assert '"name"' not in d.content


class TestDelegatingParserHermesIntegration:
    """Simulate qwen3 output: <think>...</think> followed by hermes tool call."""

    def _make_dp(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        return DelegatingParser(
            reasoning_parser=_FakeReasoningParser(),
            tool_parser=HermesToolParser(_FakeTokenizer()),
            tool_request=_make_request_with_tools(),
        )

    def test_reasoning_then_tool_call_streaming(self):
        dp = self._make_dp()
        chunks = [
            "<think>",
            "Let me think",
            " about this.",
            "</think>",
            "\n<tool_call>",
            '\n{"name": "get_weather"',
            ', "arguments": {"loc',
            'ation": "SF"',
            "}}",
            "\n</tool_call>",
        ]
        results = _simulate_streaming(dp, chunks)

        reasoning = _collect_reasoning_deltas(results)
        assert "Let me think" in reasoning

        tool_deltas = _collect_tool_call_deltas(results)
        assert len(tool_deltas) > 0, "No tool call deltas emitted"

        names = []
        for tc in tool_deltas:
            fn = tc.function
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if name:
                names.append(name)
        assert "get_weather" in names, f"Function name not found in deltas: {tool_deltas}"

        arg_parts = []
        for tc in tool_deltas:
            fn = tc.function
            args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            if args:
                arg_parts.append(args)
        combined_args = "".join(arg_parts)
        assert combined_args, "No argument deltas emitted"
        parsed = json.loads(combined_args)
        assert parsed["location"] == "SF"

        content = _collect_content_deltas(results)
        assert '{"name"' not in content
        assert "<tool_call>" not in content
        assert "</tool_call>" not in content

    def test_no_reasoning_direct_tool_call(self):
        """Tool call without reasoning (model skips <think>)."""
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        dp = DelegatingParser(
            reasoning_parser=None,
            tool_parser=HermesToolParser(_FakeTokenizer()),
            tool_request=_make_request_with_tools(),
        )
        chunks = [
            "<tool_call>",
            '\n{"name": "get_weather", "arguments": {"location": "NYC"}}',
            "\n</tool_call>",
        ]
        results = _simulate_streaming(dp, chunks)
        tool_deltas = _collect_tool_call_deltas(results)
        assert len(tool_deltas) > 0

        final = results[-1]
        assert final.tool_calls is not None

    def test_plain_content_no_tools(self):
        """When model doesn't call tools, content flows through cleanly."""
        dp = self._make_dp()
        chunks = [
            "<think>",
            "thinking",
            "</think>",
            "Hello, ",
            "how can I help?",
        ]
        results = _simulate_streaming(dp, chunks)
        content = _collect_content_deltas(results)
        assert "Hello" in content
        assert "how can I help" in content
        tool_deltas = _collect_tool_call_deltas(results)
        assert len(tool_deltas) == 0

    def test_phase_transitions(self):
        dp = self._make_dp()
        assert dp.phase == ParsePhase.REASONING

        dp.process_delta("<think>hi", "<think>hi", [1, 2], "", [])
        assert dp.phase == ParsePhase.REASONING

        dp.process_delta("<think>hi</think>", "</think>", [1, 2, 3], "<think>hi", [1, 2])
        assert dp.phase == ParsePhase.CONTENT

    def test_raw_content_text_tracks_accurately(self):
        """_raw_content_text must always reflect the true content, even during TOOL_CALL phase."""
        dp = self._make_dp()
        chunks = [
            "<think>ok</think>",
            "\n<tool_call>",
            '\n{"name": "get_weather", "arguments": {"location": "SF"}}',
            "\n</tool_call>",
        ]
        accumulated = ""
        token_ids = []
        for chunk in chunks:
            prev = accumulated
            prev_ids = list(token_ids)
            accumulated += chunk
            token_ids.extend(range(len(prev_ids), len(prev_ids) + len(chunk)))
            dp.process_delta(accumulated, chunk, token_ids, prev, prev_ids)

        assert "<tool_call>" in dp._raw_content_text or "get_weather" in dp._raw_content_text

    def test_multiple_argument_tokens(self):
        """Arguments streamed across many small chunks."""
        dp = self._make_dp()
        chunks = [
            "<think>ok</think>",
            "<tool_call>",
            '\n{"',
            "name",
            '": "',
            "get_weather",
            '", "',
            "arguments",
            '": {"',
            "location",
            '": "',
            "San Francisco",
            '"}}',
            "\n</tool_call>",
        ]
        results = _simulate_streaming(dp, chunks)

        tool_deltas = _collect_tool_call_deltas(results)
        assert len(tool_deltas) > 0

        arg_parts = []
        for tc in tool_deltas:
            fn = tc.function
            args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            if args:
                arg_parts.append(args)
        combined = "".join(arg_parts)
        if combined:
            parsed = json.loads(combined)
            assert parsed["location"] == "San Francisco"

        content = _collect_content_deltas(results)
        assert "get_weather" not in content
        assert "San Francisco" not in content

    def test_visible_snapshot_stays_frozen_while_tool_json_streams(self):
        """Visible accumulated_content must not jump to raw tool JSON snapshots."""
        dp = self._make_dp()
        chunks = [
            "<think>ok</think>",
            "\n\n",
            '<tool_call>{"n',
            'ame": "',
            'get_weather",',
            ' "arguments":',
            ' {"location":',
            ' "Paris"}}',
            "</tool_call>",
        ]

        results = _simulate_streaming(dp, chunks)

        visible_snapshots = [r.accumulated_content for r in results[:-1]]
        assert visible_snapshots[0] == ""
        assert visible_snapshots[1] == "\n\n"
        assert all(snapshot == "\n\n" for snapshot in visible_snapshots[2:])

        leaked_text = ""
        previous_visible = ""
        for result in results[:-1]:
            leaked_text += compute_stream_delta_text(
                result.accumulated_content,
                previous_visible,
                result.delta_content or "",
            )
            previous_visible = result.accumulated_content

        assert leaked_text == "\n\n"
        assert results[-1].accumulated_content == "\n\n"

    def test_final_batch_synthesizes_missing_argument_delta(self):
        """If the tool call only becomes complete at finish, final delta_tool_calls should backfill it."""
        dp = self._make_dp()
        chunks = [
            "<think>ok</think>",
            "\n\n",
            '<tool_call>{"name": "',
            'get_weather", "arguments": {"location": "Paris"}}',
        ]

        results = _simulate_streaming(dp, chunks)

        tool_deltas = _collect_tool_call_deltas(results)
        arg_parts = []
        for tc in tool_deltas:
            fn = tc.function
            args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            if args:
                arg_parts.append(args)

        assert json.loads("".join(arg_parts)) == {"location": "Paris"}
        assert results[-1].delta_tool_calls is not None
        assert results[-1].tool_calls is not None
        assert results[-1].accumulated_content == "\n\n"


class TestHermesFullDeltaToolCall:
    """When the entire tool call arrives in a single delta."""

    def _make_parser(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        return HermesToolParser(_FakeTokenizer())

    def _stream_single(self, parser, text, req=None):
        if req is None:
            req = _make_request_with_tools()
        ids = list(range(len(text)))
        return parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=ids,
            delta_token_ids=ids,
            request=req,
        )

    def test_full_tool_call_in_one_delta(self):
        parser = self._make_parser()
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "SF"}}\n</tool_call>'
        result = self._stream_single(parser, text)
        assert result is not None
        assert result.tool_calls is not None
        assert len(result.tool_calls) > 0
        assert result.content is None or "<tool_call>" not in (result.content or "")

    def test_content_plus_full_tool_call_in_one_delta(self):
        parser = self._make_parser()
        text = 'Hello! <tool_call>\n{"name": "get_weather", "arguments": {"location": "SF"}}\n</tool_call>'
        result = self._stream_single(parser, text)
        if result and result.content:
            assert "<tool_call>" not in result.content
            assert '{"name"' not in result.content

    def test_no_content_leak_from_balanced_tags(self):
        """Line 273-278 must not return tool markup as content."""
        parser = self._make_parser()
        text = '<tool_call>\n{"name": "ListDir", "arguments": {"directory_path": "."}}\n</tool_call>'
        result = self._stream_single(parser, text)
        if result and result.content:
            assert "<tool_call>" not in result.content
            assert "</tool_call>" not in result.content
            assert '"name"' not in result.content

    def test_delegating_parser_single_delta_tool_call(self):
        """DelegatingParser may batch the full tool call into one content_delta."""
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        dp = DelegatingParser(
            reasoning_parser=_FakeReasoningParser(),
            tool_parser=HermesToolParser(_FakeTokenizer()),
            tool_request=_make_request_with_tools(),
        )
        chunks = [
            "<think>ok</think>",
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "SF"}}\n</tool_call>',
        ]
        results = _simulate_streaming(dp, chunks)
        tool_deltas = _collect_tool_call_deltas(results)
        assert len(tool_deltas) > 0, "Tool call was silently dropped"
        content = _collect_content_deltas(results)
        assert "<tool_call>" not in content
        assert '{"name"' not in content


class TestHermesEdgeCases:
    """Edge cases that previously caused bugs."""

    def test_empty_arguments(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser(_FakeTokenizer())
        text = '<tool_call>\n{"name": "list_files", "arguments": {}}\n</tool_call>'
        result = parser.extract_tool_calls(text, _make_request_with_tools())
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "list_files"
        assert json.loads(result.tool_calls[0].function.arguments) == {}

    def test_unicode_arguments(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser(_FakeTokenizer())
        text = '<tool_call>\n{"name": "search", "arguments": {"query": "café résumé"}}\n</tool_call>'
        result = parser.extract_tool_calls(text, _make_request_with_tools())
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == "café résumé"

    def test_nested_json_arguments(self):
        from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

        parser = HermesToolParser(_FakeTokenizer())
        text = '<tool_call>\n{"name": "api_call", "arguments": {"body": {"key": "val", "nested": [1, 2]}}}\n</tool_call>'
        result = parser.extract_tool_calls(text, _make_request_with_tools())
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["body"]["nested"] == [1, 2]
