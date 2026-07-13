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

"""Regression test for chat_template_kwargs key collisions (bug E2).

``format_chat_prompt`` passes ``add_generation_prompt`` / ``chat_template`` /
``tools`` both as explicit keyword arguments and via ``**chat_template_kwargs``.
A client body like ``{"chat_template_kwargs": {"add_generation_prompt": true}}``
therefore raised ``TypeError: got multiple values for keyword argument`` (an
HTTP 500). The client-supplied value must win and no TypeError may surface.
"""

from __future__ import annotations

from easydel.inference.esurge.engine.chat_templating import format_chat_prompt


class _RecordingTokenizer:
    """Records the kwargs apply_chat_template actually received."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=True,
        chat_template=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "add_generation_prompt": add_generation_prompt,
                "chat_template": chat_template,
                "tools": tools,
                "extra": dict(kwargs),
            }
        )
        return "PROMPT"


def test_add_generation_prompt_in_chat_template_kwargs_does_not_collide():
    tokenizer = _RecordingTokenizer()
    passed_kwargs = {"add_generation_prompt": False, "enable_thinking": True}

    # With the bug this raises TypeError (multiple values for
    # 'add_generation_prompt').
    prompt = format_chat_prompt(
        tokenizer,
        messages=[{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        chat_template_kwargs=passed_kwargs,
    )

    assert prompt == "PROMPT"
    call = tokenizer.calls[-1]
    # Client-supplied value wins over the explicit parameter.
    assert call["add_generation_prompt"] is False
    # Non-colliding kwargs still forwarded.
    assert call["extra"] == {"enable_thinking": True}
    # The caller's dict must not be mutated.
    assert passed_kwargs == {"add_generation_prompt": False, "enable_thinking": True}


def test_chat_template_and_tools_in_chat_template_kwargs_do_not_collide():
    tokenizer = _RecordingTokenizer()

    prompt = format_chat_prompt(
        tokenizer,
        messages=[{"role": "user", "content": "hi"}],
        chat_template=None,
        tools=None,
        chat_template_kwargs={
            "chat_template": "CUSTOM_TEMPLATE",
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
        },
    )

    assert prompt == "PROMPT"
    call = tokenizer.calls[-1]
    assert call["chat_template"] == "CUSTOM_TEMPLATE"
    assert call["tools"] is not None
    assert call["tools"][0]["name"] == "lookup"
