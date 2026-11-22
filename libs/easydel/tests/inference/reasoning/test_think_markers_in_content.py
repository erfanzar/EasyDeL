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

"""Think markers the model writes as ordinary text are kept, in batch and streaming extraction."""

import pytest
from easydel.inference.reasoning.parsers import DeepSeekR1ReasoningParser

pytestmark = pytest.mark.cpu_ok


class _Tokenizer:
    chat_template = None

    def get_vocab(self):
        return {"<think>": 1, "</think>": 2}


def _parser(prompt_opens_reasoning=False):
    parser = DeepSeekR1ReasoningParser(_Tokenizer())
    prompt = "user turn <think>" if prompt_opens_reasoning else "user turn"
    parser.configure_prompt_context(prompt, [1] if prompt_opens_reasoning else [0])
    return parser


def _stream(parser, text, step=3):
    reasoning, content, prev = "", "", ""
    for end in range(step, len(text) + step, step):
        cur = text[:end]
        msg = parser.extract_reasoning_streaming(prev, cur, cur[len(prev) :], [], [], [])
        if msg is not None:
            reasoning += msg.reasoning_content or ""
            content += msg.content or ""
        prev = cur
    msg = parser.finalize_streaming(prev)
    if msg is not None:
        reasoning += msg.reasoning_content or ""
        content += msg.content or ""
    return reasoning, content


CASES = [
    # (prompt opens reasoning, model output, reasoning, content)
    (False, "Write <think>x</think> literally.", "", "Write <think>x</think> literally."),
    (False, "<think>plan</think>Use <think>a</think> here", "plan", "Use <think>a</think> here"),
    (True, "I'll mention <think> here</think>Done", "I'll mention <think> here", "Done"),
]


@pytest.mark.parametrize("prompt_opens,text,reasoning,content", CASES)
def test_batch_extraction_keeps_literal_markers(prompt_opens, text, reasoning, content):
    got_reasoning, got_content = _parser(prompt_opens).extract_reasoning(text)
    assert (got_reasoning or "", got_content or "") == (reasoning, content)


@pytest.mark.parametrize("prompt_opens,text,reasoning,content", CASES)
def test_streaming_extraction_keeps_literal_markers(prompt_opens, text, reasoning, content):
    assert _stream(_parser(prompt_opens), text) == (reasoning, content)
