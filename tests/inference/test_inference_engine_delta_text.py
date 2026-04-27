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

from easydel.inference.inference_engine_interface import BaseInferenceApiServer


def test_compute_delta_text_uses_accumulated_growth():
    delta = BaseInferenceApiServer._compute_delta_text("hello world", "hello ", "fallback")
    assert delta == "world"


def test_compute_delta_text_does_not_replay_fallback_when_text_unchanged():
    delta = BaseInferenceApiServer._compute_delta_text("same text", "same text", "same text")
    assert delta == ""


def test_compute_delta_text_keeps_fallback_for_mismatch():
    delta = BaseInferenceApiServer._compute_delta_text("new branch", "old branch", "fallback")
    assert delta == "fallback"


def test_compute_delta_text_handles_empty_reset_without_fallback():
    delta = BaseInferenceApiServer._compute_delta_text("", "previous content", "")
    assert delta == ""
