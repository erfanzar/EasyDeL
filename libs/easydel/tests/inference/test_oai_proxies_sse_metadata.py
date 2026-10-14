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

from __future__ import annotations

from easydel.inference.oai_proxies import InferenceApiRouter
from easydel.inference.openai_api_modules import ChatCompletionRequest, CompletionRequest
from pydantic import BaseModel


class _Meta(BaseModel):
    key: str
    count: int


def test_format_metadata_sse_uses_standard_event_framing():
    payload = InferenceApiRouter._format_metadata_sse(_Meta(key="value", count=2)).decode()

    assert payload.startswith("event: metadata\ndata: ")
    assert payload.endswith("\n\n")
    assert '"key":"value"' in payload
    assert '"count":2' in payload


def test_oai_params_omit_extra_body_for_default_repetition_penalty():
    """Default/unset ``repetition_penalty`` must not add ``extra_body`` to proxied requests.

    Strict OpenAI-compatible upstreams reject unknown body fields, so the
    EasyDeL-specific key is only forwarded when the caller asked for a penalty.
    """
    router = object.__new__(InferenceApiRouter)

    completion_params = router.build_oai_params_from_request(CompletionRequest(model="test-model", prompt="hello"))
    assert "extra_body" not in completion_params

    chat_params = router.build_oai_params_from_chat_request(
        ChatCompletionRequest(model="test-model", messages=[{"role": "user", "content": "hello"}])
    )
    assert "extra_body" not in chat_params

    explicit_default = router.build_oai_params_from_request(
        CompletionRequest(model="test-model", prompt="hello", repetition_penalty=1.0)
    )
    assert "extra_body" not in explicit_default


def test_oai_params_forward_repetition_penalty_via_extra_body():
    """``repetition_penalty`` must not be a top-level ``create()`` kwarg.

    The OpenAI SDK's generated methods accept only documented parameters (plus
    ``extra_body``); a top-level ``repetition_penalty`` kwarg raises
    ``TypeError`` before any HTTP call is made, so the proxy must forward the
    EasyDeL-specific parameter through ``extra_body``.
    """
    router = object.__new__(InferenceApiRouter)

    completion_params = router.build_oai_params_from_request(
        CompletionRequest(model="test-model", prompt="hello", repetition_penalty=1.7)
    )
    assert "repetition_penalty" not in completion_params
    assert completion_params["extra_body"]["repetition_penalty"] == 1.7

    chat_params = router.build_oai_params_from_chat_request(
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            repetition_penalty=1.3,
        )
    )
    assert "repetition_penalty" not in chat_params
    assert chat_params["extra_body"]["repetition_penalty"] == 1.3
