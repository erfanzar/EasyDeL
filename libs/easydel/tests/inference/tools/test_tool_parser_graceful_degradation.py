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

"""Graceful degradation of auto-selected tool parsers with incompatible tokenizers.

Regression coverage for the ``mistral4 -> mistral`` auto-detection path: the
Mistral tool parser raises in ``__init__`` when the tokenizer lacks the
``[TOOL_CALLS]`` delimiter token. Instantiating it at request-add time would
crash the engine's request path; the fix is to degrade gracefully (disable
function calling with a warning) instead of raising.
"""

from __future__ import annotations

import pytest
from easydel.inference.tools import ToolParserManager
from easydel.inference.tools.auto_detect import detect_tool_parser
from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin, build_tool_parser_or_none


class _FakeTokenizer:
    """Minimal HF-like tokenizer exposing only ``get_vocab``."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab = vocab or {}
        self.chat_template = None

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_mistral4_model_type_auto_selects_mistral_parser():
    """``mistral4`` must route to the mistral parser (the failing path)."""
    assert detect_tool_parser(model_type="mistral4") == "mistral"


def test_mistral_parser_raises_without_tool_call_token():
    """Document the crash source: mistral parser requires ``[TOOL_CALLS]``."""
    parser_class = ToolParserManager.get_tool_parser("mistral")
    with pytest.raises(RuntimeError, match="tool call token"):
        parser_class(_FakeTokenizer(vocab={}))


def test_build_tool_parser_or_none_degrades_when_token_missing():
    """The guard returns None (no raise) when the tokenizer is incompatible."""
    parser_class = ToolParserManager.get_tool_parser("mistral")
    assert build_tool_parser_or_none(parser_class, _FakeTokenizer(vocab={})) is None


def test_build_tool_parser_or_none_builds_when_token_present():
    """When the delimiter token exists, a live parser is returned."""
    parser_class = ToolParserManager.get_tool_parser("mistral")
    parser = build_tool_parser_or_none(parser_class, _FakeTokenizer(vocab={"[TOOL_CALLS]": 5}))
    assert parser is not None
    assert parser.bot_token_id == 5


def test_build_tool_parser_or_none_handles_disabled_parser():
    """A None class (function calling disabled) yields None without error."""
    assert build_tool_parser_or_none(None, _FakeTokenizer(vocab={})) is None


def test_initialize_tool_parsers_skips_incompatible_model():
    """The server mixin degrades to an empty parser map instead of crashing."""

    class _Server(ToolCallingMixin):
        pass

    server = _Server()
    parsers = server.initialize_tool_parsers(
        model_processors={"mistral4-model": _FakeTokenizer(vocab={})},
        tool_parser_name="mistral",
        enable_function_calling=True,
    )
    # Incompatible tokenizer: model dropped, no exception raised.
    assert parsers == {}


def test_initialize_tool_parsers_keeps_compatible_model():
    """A compatible tokenizer still yields a live parser."""

    class _Server(ToolCallingMixin):
        pass

    server = _Server()
    parsers = server.initialize_tool_parsers(
        model_processors={"mistral4-model": _FakeTokenizer(vocab={"[TOOL_CALLS]": 5})},
        tool_parser_name="mistral",
        enable_function_calling=True,
    )
    assert set(parsers) == {"mistral4-model"}
