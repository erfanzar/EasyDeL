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

"""Nonterminal empty decoded updates must not finalize Granite's delimiter buffer."""

import pytest
from easydel.inference.parsing import DelegatingParser
from easydel.inference.reasoning import ReasoningMixin
from easydel.inference.reasoning.parsers import GraniteReasoningParser

pytestmark = pytest.mark.cpu_ok


def _consumer(entrypoint):
    parser = GraniteReasoningParser(object())
    mixin = ReasoningMixin()
    mixin.reasoning_parsers = {"m": parser}
    delegating = DelegatingParser(reasoning_parser=parser)

    def emit(previous, current, delta, previous_ids, current_ids, *, final=False):
        if entrypoint == "delegating":
            result = (
                delegating.process_final(current, current_ids)
                if final
                else delegating.process_delta(current, delta, current_ids, previous, previous_ids)
            )
            return result.delta_reasoning or "", result.delta_content or ""
        if entrypoint == "mixin":
            result = mixin.extract_reasoning_streaming(
                "m", previous, current, delta, previous_ids, current_ids, current_ids[len(previous_ids) :], final=final
            )
        elif final:
            result = parser.finalize_streaming(current, current_ids)
        else:
            result = parser.extract_reasoning_streaming(
                previous, current, delta, previous_ids, current_ids, current_ids[len(previous_ids) :]
            )
        return (result.reasoning_content or "", result.content or "") if result else ("", "")

    return emit


@pytest.mark.parametrize("entrypoint", ["parser", "mixin", "delegating"])
@pytest.mark.parametrize("advance_tokens", [False, True])
@pytest.mark.parametrize(
    ("preamble", "marker", "suffix", "expected"),
    [
        ("Preamble.", "Here's my thought process:", "reasonHere's my response:answer", ("reason", "Preamble.answer")),
        ("Preamble.", "Here is my thought process:", "reasonHere is my response:answer", ("reason", "Preamble.answer")),
        ("Here's my thought process:reason", "Here's my response:", "answer", ("reason", "answer")),
        ("Here is my thought process:reason", "Here is my response:", "answer", ("reason", "answer")),
        ("Preamble.", "Here's my response:", "answer", ("", "Preamble.answer")),
        ("Preamble.", "Here is my response:", "answer", ("", "Preamble.answer")),
    ],
)
def test_empty_updates_preserve_every_delimiter_split(entrypoint, advance_tokens, preamble, marker, suffix, expected):
    for split in range(1, len(marker)):
        emit = _consumer(entrypoint)
        prefix = preamble + marker[:split]
        reasoning, content = emit("", prefix, prefix, [], [1])
        ids = [1, 2] if advance_tokens else [1]
        # Heartbeats and tokens that decode to no text are both nonterminal.
        for _ in range(2):
            assert emit(prefix, prefix, "", [1], ids) == ("", "")
        tail = marker[split:] + suffix
        text = prefix + tail
        next_reasoning, next_content = emit(prefix, text, tail, ids, [*ids, 3])
        reasoning += next_reasoning
        content += next_content
        assert (reasoning, content) == expected
        assert emit(text, text, "", [*ids, 3], [*ids, 3], final=True) == ("", "")


@pytest.mark.parametrize("entrypoint", ["parser", "mixin", "delegating"])
@pytest.mark.parametrize(
    ("text", "safe", "tail"),
    [
        ("Answer Here", ("", "Answer "), ("", "Here")),
        ("Here's my thought process:reason Here", ("reason ", ""), ("Here", "")),
        ("Here's my thought process:reasonHere's my response:Answer Here", ("reason", "Answer Here"), ("", "")),
    ],
)
def test_only_explicit_finalization_releases_terminal_prefix_once(entrypoint, text, safe, tail):
    emit = _consumer(entrypoint)
    assert emit("", text, text, [], [1]) == safe
    assert emit(text, text, "", [1], [1, 2]) == ("", "")
    assert emit(text, text, "", [1, 2], [1, 2], final=True) == tail
    assert emit(text, text, "", [1, 2], [1, 2], final=True) == ("", "")
