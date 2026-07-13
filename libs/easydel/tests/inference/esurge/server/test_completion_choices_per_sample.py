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

"""Regression test for /v1/completions n>1 per-sample text (bug E3).

The completion response builder used ``output.accumulated_text`` for every
choice, but ``accumulated_text`` mirrors only sample index 0 — so a request
with ``n>1`` returned the same text for every choice. Each choice must carry
its own sample's ``completion.text``.
"""

from __future__ import annotations

from types import SimpleNamespace

from easydel.inference.esurge.server.api_server import eSurgeApiServer


def test_build_completion_choices_uses_per_sample_text():
    output = SimpleNamespace(
        accumulated_text="text-0",  # mirrors sample 0 only
        outputs=[
            SimpleNamespace(text="text-0", finish_reason="stop"),
            SimpleNamespace(text="text-1", finish_reason="length"),
            SimpleNamespace(text="text-2", finish_reason=None),
        ],
    )

    choices = eSurgeApiServer._build_completion_choices(output)

    assert [c.index for c in choices] == [0, 1, 2]
    # Each choice carries its own sample's text, not the shared accumulated_text.
    assert [c.text for c in choices] == ["text-0", "text-1", "text-2"]
    # finish_reason passthrough with the "stop" fallback for None.
    assert [c.finish_reason for c in choices] == ["stop", "length", "stop"]
