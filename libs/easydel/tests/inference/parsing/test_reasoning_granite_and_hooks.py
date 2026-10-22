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

"""Granite streaming split-delimiter regressions and reasoning-parser hook tests (no array computation)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from easydel.inference.openai_api_modules import ChatMessage, DeltaMessage
from easydel.inference.reasoning import ReasoningMixin
from easydel.inference.reasoning.abstract_reasoning import ReasoningParser
from easydel.inference.reasoning.basic_parsers import BaseThinkingReasoningParser
from easydel.inference.reasoning.parsers import Gemma4ReasoningParser, GraniteReasoningParser

pytestmark = pytest.mark.cpu_ok


def _stream(parser, text: str, step: int) -> tuple[str, str]:
    reasoning, content = "", ""
    previous = ""
    for start in range(0, len(text), step):
        delta = text[start : start + step]
        current = previous + delta
        msg = parser.extract_reasoning_streaming(previous, current, delta, [], [], [])
        previous = current
        if msg is not None:
            reasoning += msg.reasoning_content or ""
            content += msg.content or ""
    return reasoning, content


@pytest.mark.parametrize("step", [1, 2, 3, 5, 7, 11, 1000])
@pytest.mark.parametrize(
    ("thought", "response"),
    [
        ("Here's my thought process:", "Here's my response:"),
        ("Here is my thought process:", "Here is my response:"),
    ],
)
def test_granite_streaming_never_leaks_delimiter_fragments(step, thought, response):
    text = f"{thought}\nI Hear it thinking.\n{response}\nThe result."
    reasoning, content = _stream(GraniteReasoningParser(object()), text, step)

    assert reasoning.strip() == "I Hear it thinking."
    assert content.strip() == "The result."
    for fragment in ("Here", "thought process", "my response", "response:"):
        assert fragment not in content
        assert fragment not in reasoning


@pytest.mark.parametrize("step", [1, 4, 1000])
def test_granite_streaming_plain_text_is_all_content(step):
    reasoning, content = _stream(GraniteReasoningParser(object()), "Just a plain answer.", step)
    assert reasoning == ""
    assert content == "Just a plain answer."


def test_reasoning_parser_adjust_request_hook():
    request = SimpleNamespace(skip_special_tokens=True)

    # Base class: harmless no-op so callers can invoke it uniformly.
    assert ReasoningParser(object()).adjust_request(request) is request
    assert request.skip_special_tokens is True

    # Gemma4 keeps its channel markers by forcing special tokens through.
    assert Gemma4ReasoningParser(object()).adjust_request(request) is request
    assert request.skip_special_tokens is False


def test_openai_base_model_field_names_are_cached_per_class():
    DeltaMessage(content="x")
    ChatMessage(role="user", content="x")

    assert "name" in ChatMessage.field_names
    assert "name" not in DeltaMessage.field_names
    assert "role" in DeltaMessage.field_names


def test_openai_extensions_and_public_function_models_remain_usable():
    from easydel.inference.openai_api_modules import Function, Tool

    message = ChatMessage(role="assistant", content="answer", vendor_payload={"enabled": True})
    assert message.model_dump()["vendor_payload"] == {"enabled": True}
    tool = Tool(function=Function(name="ping", parameters={"type": "object"}))
    assert tool.model_dump() == {
        "type": "function",
        "function": {"name": "ping", "description": None, "parameters": {"type": "object"}},
    }


def test_public_reasoning_mixin_performs_registered_batch_and_stream_extraction():
    mixin = ReasoningMixin()
    mixin.reasoning_parsers = mixin.initialize_reasoning_parsers({"m": object()}, "granite", True)
    text = "Here's my thought process:consider it.Here's my response:answer"
    assert mixin.extract_reasoning_batch(text, "m") == ("consider it.", "answer")
    delta = mixin.extract_reasoning_streaming("m", "", text, text)
    assert delta.reasoning_content == "consider it."
    assert delta.content == "answer"
    assert isinstance(mixin.get_reasoning_parser_for_model("m"), GraniteReasoningParser)
    assert mixin.extract_reasoning_batch("unchanged", "other") == (None, "unchanged")


def test_public_token_boundary_api_preserves_visible_content_ids():
    tokenizer = SimpleNamespace(get_vocab=lambda: {"<think>": 1, "</think>": 2})
    parser = BaseThinkingReasoningParser(tokenizer)
    assert parser.is_reasoning_end([1, 7]) is False
    assert parser.is_reasoning_end([1, 7, 2, 9]) is True
    assert parser.is_reasoning_end_streaming([1, 7, 2], [2]) is True
    assert parser.is_reasoning_end_streaming([1, 7, 2, 9], [9]) is False
    assert parser.extract_content_ids([1, 7, 2, 9]) == [9]


def test_import_reasoning_parser_loads_external_plugin(tmp_path):
    from easydel.inference.reasoning.abstract_reasoning import ReasoningParserManager

    plugin = tmp_path / "reasoning_plugin_module.py"
    plugin.write_text(
        "from easydel.inference.reasoning.abstract_reasoning import ReasoningParser, ReasoningParserManager\n"
        "\n"
        "@ReasoningParserManager.register_module('reasoning_plugin')\n"
        "class PluginReasoningParser(ReasoningParser):\n"
        "    pass\n"
    )
    ReasoningParserManager.import_reasoning_parser(str(plugin))
    assert ReasoningParserManager.get_reasoning_parser("reasoning_plugin").__name__ == "PluginReasoningParser"

    # A missing plugin file is reported (warning) instead of raising.
    ReasoningParserManager.import_reasoning_parser(str(tmp_path / "does_not_exist.py"))


@pytest.mark.parametrize("step", [1, 2, 7, 1000])
def test_granite_streaming_releases_false_marker_prefixes(step):
    text = "Here's my thought process:Here is a useful thought.Here's my response:Here's another answer."
    reasoning, content = _stream(GraniteReasoningParser(object()), text, step)
    assert reasoning == "Here is a useful thought."
    assert content == "Here's another answer."
