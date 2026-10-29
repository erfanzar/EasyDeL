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

"""Regression tests for the OpenAI proxy router (no array computation)."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from easydel.inference import inference_engine_interface as iface
from easydel.inference import oai_proxies
from easydel.inference.oai_proxies import InferenceApiRouter
from easydel.inference.openai_api_modules import ChatCompletionRequest, ChatMessage, CompletionRequest, ResponsesRequest
from fastapi import HTTPException

pytestmark = pytest.mark.cpu_ok


class APIConnectionError(Exception):
    """Mimics ``openai.APIConnectionError``: no ``status_code`` attribute."""


class APITimeoutError(APIConnectionError):
    """Mimics ``openai.APITimeoutError``: no ``status_code`` attribute."""


class APIStatusError(Exception):
    """Mimics ``openai.APIStatusError``: carries the backend's HTTP status."""

    def __init__(self, status_code: int):
        super().__init__("status")
        self.status_code = status_code


@pytest.fixture
def router(monkeypatch):
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock()),
        completions=SimpleNamespace(create=AsyncMock()),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
    )
    module = SimpleNamespace(AsyncOpenAI=lambda **kwargs: client, APIError=(APIConnectionError, APIStatusError))
    monkeypatch.setitem(sys.modules, "openai", module)
    return InferenceApiRouter(api_key="test", base_url="http://localhost:8000/v1")


@pytest.mark.parametrize("endpoint", ["responses", "chat_completions", "completions"])
@pytest.mark.parametrize(
    ("error", "status"),
    [(APIStatusError(429), 429), (APIConnectionError("down"), 502), (APITimeoutError("slow"), 504)],
)
def test_handlers_return_backend_status_for_connection_errors(router, endpoint, error, status):
    if endpoint == "responses":
        router.client.responses.create.side_effect = error
        coroutine = router.responses(ResponsesRequest(model="m", input="hello"), SimpleNamespace(headers={}))
    elif endpoint == "chat_completions":
        router.client.chat.completions.create.side_effect = error
        coroutine = router.chat_completions(
            ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hello")])
        )
    else:
        router.client.completions.create.side_effect = error
        coroutine = router.completions(CompletionRequest(model="m", prompt="hello"))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(coroutine)
    assert caught.value.status_code == status
    assert router.metrics.failed_requests == 1
    assert router.metrics.successful_requests == 0


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000/v1", "http://localhost:8000"),
        ("http://localhost:8000/v1/", "http://localhost:8000"),
        ("http://localhost:8000", "http://localhost:8000"),
        ("https://host/prefix/v1", "https://host/prefix"),
        (None, None),
    ],
)
def test_server_root_url_strips_openai_v1_suffix(base_url, expected):
    router = object.__new__(InferenceApiRouter)
    router.base_url = base_url
    assert router.server_root_url == expected


def test_responses_params_only_pass_sdk_fields_as_kwargs():
    router = object.__new__(InferenceApiRouter)
    request = ResponsesRequest(
        model="m",
        input="hi",
        temperature=0.5,
        max_tokens=33,
        top_k=7,
        min_p=0.1,
        repetition_penalty=1.2,
        custom_vendor_field="x",
    )
    params = router.build_oai_params_from_responses_request(request)

    assert set(params) == {"model", "input", "temperature", "max_output_tokens", "stream", "extra_body"}
    assert params["model"] == "m"
    assert params["input"] == "hi"
    assert params["max_output_tokens"] == 33
    assert params["extra_body"] == {
        "top_k": 7,
        "min_p": 0.1,
        "repetition_penalty": 1.2,
        "custom_vendor_field": "x",
    }
    assert "max_tokens" not in params


def test_responses_params_use_messages_as_input_and_skip_neutral_penalty():
    router = object.__new__(InferenceApiRouter)
    request = ResponsesRequest(
        model="m",
        messages=[{"role": "user", "content": "hello"}],
        repetition_penalty=1.0,
    )
    params = router.build_oai_params_from_responses_request(request)
    assert params["input"] == [{"role": "user", "content": "hello"}]
    assert "extra_body" not in params


def test_oai_proxies_shares_server_scaffolding_with_engine_interface():
    assert oai_proxies.ServerStatus is iface.ServerStatus
    assert oai_proxies.ServerMetrics is iface.ServerMetrics
    assert oai_proxies.EndpointConfig is iface.EndpointConfig
    assert oai_proxies.ErrorResponse is iface.ErrorResponse
    assert oai_proxies.create_error_response is iface.create_error_response


def test_responses_handler_keeps_extensions_with_strict_keyword_backend(router):
    async def create(*, model, input, max_output_tokens, stream, extra_body):  # noqa: A002 - OpenAI SDK keyword
        return {"model": model, "input": input, "limit": max_output_tokens, "stream": stream, "body": extra_body}

    router.client.responses.create = create
    request = ResponsesRequest(
        model="m",
        input="hi",
        max_completion_tokens=12,
        top_k=5,
        reasoning=True,
        functions=[{"name": "ping", "parameters": {}}],
        extra_body={"vendor_option": "yes"},
    )
    response = asyncio.run(router.responses(request, SimpleNamespace(headers={})))
    assert response == {
        "model": "m",
        "input": "hi",
        "limit": 12,
        "stream": False,
        "body": {
            "top_k": 5,
            "reasoning": True,
            "functions": [{"name": "ping", "parameters": {}}],
            "vendor_option": "yes",
        },
    }
    assert router.metrics.successful_requests == 1


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/"])
def test_backend_auxiliary_endpoints_reach_server_root(router, monkeypatch, suffix):
    router.base_url = f"http://localhost:8000{suffix}"
    payloads = {
        "http://localhost:8000/metrics": {"requests": 7},
        "http://localhost:8000/v1/tools": {"tools": [{"name": "ping"}]},
        "http://localhost:8000/v1/tools/execute": {"result": "pong"},
    }

    class Response:
        status = 200

        def __init__(self, url):
            self.payload = payloads[url]  # An incorrectly doubled /v1 path cannot return a successful payload.

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self):
            return self.payload

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def get(self, url):
            return Response(url)

        def post(self, url, *, json):
            assert json == {"name": "ping", "arguments": {}}
            return Response(url)

    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientSession=Session))
    metrics = asyncio.run(router.get_metrics())
    tools = asyncio.run(router.list_tools())
    execution = asyncio.run(
        router.execute_tool(SimpleNamespace(json=AsyncMock(return_value={"name": "ping", "arguments": {}})))
    )
    assert json.loads(metrics.body)["backend_metrics"] == {"requests": 7}
    assert json.loads(tools.body) == {"tools": [{"name": "ping"}]}
    assert execution.status_code == 200
    assert json.loads(execution.body) == {"result": "pong"}


def test_router_registered_endpoints_and_server_launch_options(router, monkeypatch):
    paths = {route.path for route in router.app.routes}
    assert {"/v1/responses", "/v1/chat/completions", "/v1/completions", "/metrics", "/v1/tools"} <= paths
    captured = {}
    monkeypatch.setattr(iface.uvicorn, "run", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setitem(sys.modules, "uvloop", None)
    router.run(host="127.0.0.1", ssl_keyfile="key.pem", ssl_certfile="cert.pem", workers=4, reload=True)
    assert captured["app"] is router.app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8084
    assert captured["workers"] == 1
    assert captured["reload"] is True
    assert captured["ssl_keyfile"] == "key.pem"
    assert captured["ssl_certfile"] == "cert.pem"
