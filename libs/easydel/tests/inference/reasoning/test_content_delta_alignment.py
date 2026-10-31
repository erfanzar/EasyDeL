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

"""Tests that batch and streaming reasoning extraction produce aligned content.

The DelegatingParser uses streaming extraction for per-token deltas and batch
extraction for phase-transition detection.  If batch strips whitespace but
streaming doesn't, the prefix mismatch causes characters to be dropped from
the streamed output (e.g. "Hello!ow can I assist" instead of "Hello! How can
I assist").
"""

from easydel.inference.reasoning.basic_parsers import BaseThinkingReasoningParser


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, ids):
        return "?" * len(ids)

    def get_vocab(self):
        return {}


class _ThinkParser(BaseThinkingReasoningParser):
    start_token = "<think>"
    end_token = "</think>"

    def __init__(self):
        super().__init__(_FakeTokenizer())


class TestBatchStreamContentAlignment:
    """Batch and streaming must produce identical content strings."""

    def test_content_not_stripped_by_batch(self):
        """Content after </think> must NOT be stripped in batch mode."""
        parser = _ThinkParser()
        reasoning, content = parser.extract_reasoning("<think>ok</think>\n\nHello! How can I help?")
        assert reasoning == "ok"
        # Content must preserve the leading \n\n
        assert content is not None
        assert content.startswith("\n\nHello")

    def test_batch_and_streaming_first_content_match(self):
        """The first content from streaming must be a prefix of batch content."""
        parser = _ThinkParser()

        # Simulate streaming: </think> arrives, content follows
        full_text = "<think>thinking</think>\n\nHello! How are you?"
        delta_msg = parser.extract_reasoning_streaming(
            previous_text="<think>thinking",
            current_text=full_text,
            delta_text="</think>\n\nHello! How are you?",
            previous_token_ids=[],
            current_token_ids=list(range(len(full_text))),
            delta_token_ids=list(range(15, len(full_text))),
        )
        streaming_content = delta_msg.content if delta_msg else ""

        # Batch extraction of same text
        _, batch_content = parser.extract_reasoning(full_text)

        # Streaming content must be a prefix of (or equal to) batch content
        assert batch_content is not None
        assert batch_content.startswith(streaming_content), (
            f"Batch content {batch_content!r} does not start with streaming content {streaming_content!r}"
        )

    def test_incremental_content_deltas_no_char_loss(self):
        """Simulate token-by-token streaming and verify no characters are lost."""
        parser = _ThinkParser()

        tokens = [
            "<think>",
            "ok",
            "</think>",
            "\n\n",
            "Hello",
            "!",
            " How",
            " can",
            " I",
            " assist",
            " you",
            " today",
            "?",
        ]

        accumulated = ""
        prev_text = ""
        prev_ids = []
        content_parts = []

        for token in tokens:
            accumulated += token
            cur_ids = list(range(len(accumulated)))
            delta_ids = cur_ids[len(prev_ids) :]

            delta_msg = parser.extract_reasoning_streaming(
                previous_text=prev_text,
                current_text=accumulated,
                delta_text=token,
                previous_token_ids=prev_ids,
                current_token_ids=cur_ids,
                delta_token_ids=delta_ids,
            )

            if delta_msg and delta_msg.content is not None:
                content_parts.append(delta_msg.content)

            prev_text = accumulated
            prev_ids = cur_ids

        reassembled = "".join(content_parts)
        expected = "\n\nHello! How can I assist you today?"
        assert reassembled == expected, f"Reassembled content {reassembled!r} != expected {expected!r}"

    def test_delegating_parser_no_char_loss(self):
        """Full DelegatingParser pipeline must not lose characters."""
        from easydel.inference.parsing.delegating_parser import DelegatingParser

        dp = DelegatingParser(reasoning_parser=_ThinkParser())

        tokens = [
            "<think>",
            "ok",
            "</think>",
            "\n\n",
            "Hello",
            "!",
            " How",
            " can",
            " I",
            " assist",
            " you",
            " today",
            "?",
        ]

        accumulated = ""
        prev_text = ""
        prev_ids = []
        content_parts = []

        for token in tokens:
            accumulated += token
            cur_ids = list(range(len(accumulated)))
            result = dp.process_delta(accumulated, token, cur_ids, prev_text, prev_ids)
            if result.delta_content:
                content_parts.append(result.delta_content)
            prev_text = accumulated
            prev_ids = cur_ids

        # Final
        result = dp.process_final(accumulated, list(range(len(accumulated))))
        if result.delta_content:
            content_parts.append(result.delta_content)

        reassembled = "".join(content_parts)
        expected = "\n\nHello! How can I assist you today?"
        assert reassembled == expected, f"Reassembled {reassembled!r} != expected {expected!r}"

    def test_content_with_leading_newlines_preserved(self):
        """Leading newlines in content must be preserved exactly."""
        parser = _ThinkParser()
        _reasoning, content = parser.extract_reasoning("<think>short</think>\n\nHello world!")
        assert content == "\n\nHello world!"

    def test_content_single_newline_preserved(self):
        parser = _ThinkParser()
        _reasoning, content = parser.extract_reasoning("<think>x</think>\nResponse")
        assert content == "\nResponse"
