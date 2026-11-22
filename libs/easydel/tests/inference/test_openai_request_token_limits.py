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

"""OpenAI request token limits are validated at the HTTP boundary (OpenAI answers 400 for max_tokens < 1)."""

import pytest
from easydel.inference.openai_api_modules import ChatCompletionRequest, CompletionRequest, ResponsesRequest
from easydel.inference.sampling_params import SamplingParams
from pydantic import ValidationError

pytestmark = pytest.mark.cpu_ok

_MESSAGES = [{"role": "user", "content": "hi"}]


def _build(kind: str, field: str, value):
    if kind == "chat":
        return ChatCompletionRequest(model="m", messages=_MESSAGES, **{field: value})
    if kind == "completion":
        return CompletionRequest(model="m", prompt="hi", **{field: value})
    return ResponsesRequest(model="m", input="hi", **{field: value})


_FIELDS = [
    ("chat", "max_tokens"),
    ("completion", "max_tokens"),
    ("responses", "max_tokens"),
    ("responses", "max_output_tokens"),
    ("responses", "max_completion_tokens"),
]


@pytest.mark.parametrize(("kind", "field"), _FIELDS)
def test_minus_one_means_unlimited(kind, field):
    assert getattr(_build(kind, field, -1), field) is None


@pytest.mark.parametrize(("kind", "field"), _FIELDS)
@pytest.mark.parametrize("value", [0, -2, -5])
def test_non_positive_token_limits_are_rejected(kind, field, value):
    with pytest.raises(ValidationError) as excinfo:
        _build(kind, field, value)
    assert any(err["loc"] == (field,) and err["type"] == "greater_than_equal" for err in excinfo.value.errors())


@pytest.mark.parametrize(("kind", "field"), _FIELDS)
@pytest.mark.parametrize("value", [None, 1, 4096])
def test_absent_or_positive_token_limits_are_accepted(kind, field, value):
    assert getattr(_build(kind, field, value), field) == value


def test_library_sampling_params_keep_negative_means_infer():
    assert SamplingParams(max_tokens=-1).max_tokens is None
