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

"""Comprehensive regression tests for S2 HTTP/API Server, Proxies, Stream Protocol & Auth review findings.

Covers findings S2-C4, S2-H9, S2-H10, S2-H11, S2-H12, S2-H13, S2-H14, S2-H15, S2-H16,
S2-M1, S2-M4, S2-M5, and S2-M6 using real FastAPI TestClient, real EnhancedApiKeyManager,
real ResponsesStreamAccumulator, real compute_stream_delta_text, and real RequestTrackingMiddleware.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.inference_engine_interface import (
    BaseInferenceApiServer,
    RequestTrackingMiddleware,
    ServerMetrics,
    ServerStatus,
)
from easydel.inference.oai_proxies import InferenceApiRouter
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    FunctionCallSelection,
    ResponsesRequest,
    ResponsesToolChoice,
    ToolCall,
    ToolChoiceFunction,
    ToolChoiceOption,
)
from easydel.inference.stream_protocol import (
    ResponsesStreamAccumulator,
    compute_stream_delta_text,
    iter_chat_completion_stream_responses,
    iter_responses_stream_frames,
)
from easydel.inference.typed_models import (
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseReasoningItem,
    ResponsesFinalizationOptions,
    ResponsesResponse,
)
from easydel.workers.esurge.auth import (
    ApiKeyPermissions,
    ApiKeyRole,
    PermissionDenied,
    QuotaConfig,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

pytestmark = pytest.mark.cpu_ok


class _SimpleTokenizer:
    """Minimal deterministic tokenizer implementing the surface used by eSurgeApiServer."""

    def __call__(self, text: str) -> dict[str, list[int]]:
        tokens = text.split() if isinstance(text, str) and text.strip() else ["x"]
        return {"input_ids": list(range(1, len(tokens) + 1))}

    def encode(self, text: str) -> list[int]:
        return self(text)["input_ids"]

    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


class _ScriptedEngine:
    """Lightweight real-surface eSurge stub for driving eSurgeApiServer end-to-end over HTTP."""

    def __init__(
        self,
        *,
        name: str = "test-model",
        max_model_len: int = 2048,
        max_num_seqs: int = 4,
        chat_reply: str = "ok",
        tool_calls: list[ToolCall] | None = None,
        raw_text: str | None = None,
        on_chat=None,
        on_generate=None,
    ) -> None:
        self.esurge_name = name
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.tokenizer = _SimpleTokenizer()
        self.tool_parser = "hermes"
        self.chat_reply = chat_reply
        self.tool_calls = tool_calls
        self.raw_text = raw_text
        self.on_chat = on_chat
        self.on_generate = on_generate
        self.chat_calls: list[dict] = []
        self.generate_calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.aborted: list[str] = []
        self.tokenizer_threads: list[int] = []

        orig_tok = self.tokenizer

        class _TrackingTokenizer:
            def __call__(_tok_self, text: str):
                self.tokenizer_threads.append(threading.get_ident())
                return orig_tok(text)

            def encode(_tok_self, text: str):
                return orig_tok.encode(text)

            def apply_chat_template(_tok_self, messages, **kwargs):
                return orig_tok.apply_chat_template(messages, **kwargs)

        self.tokenizer = _TrackingTokenizer()

    def _make_output(
        self,
        *,
        prompt: str,
        request_id: str,
        text: str | None = None,
        n: int = 1,
        tool_calls: list[ToolCall] | None = None,
        raw_text: str | None = None,
        finished: bool = True,
    ) -> RequestOutput:
        reply_text = self.chat_reply if text is None else text
        tc = self.tool_calls if tool_calls is None else tool_calls
        rt = self.raw_text if raw_text is None else raw_text
        outputs = [
            CompletionOutput(
                index=idx,
                text=reply_text if n == 1 else f"{reply_text}-{idx}",
                token_ids=[10, 11],
                finish_reason="stop" if finished else None,
                tool_calls=tc if idx == 0 else None,
                raw_text=rt if rt is not None else reply_text,
            )
            for idx in range(n)
        ]
        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=[1, 2, 3],
            outputs=outputs,
            finished=finished,
            accumulated_text=outputs[0].text,
            delta_text=outputs[0].text,
            raw_accumulated_text=rt if rt is not None else outputs[0].text,
            tool_calls=tc,
            num_generated_tokens=2 * n,
            tokens_per_second=10.0,
            processing_time=0.1,
            first_token_time=0.02,
        )

    def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        sampling_params=None,
        request_id=None,
        stream=False,
        chat_template=None,
        chat_template_kwargs=None,
    ):
        call = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "sampling_params": sampling_params,
            "request_id": request_id,
            "stream": stream,
            "chat_template_kwargs": chat_template_kwargs,
        }
        self.chat_calls.append(call)
        if self.on_chat is not None:
            return self.on_chat(call)
        n = int(getattr(sampling_params, "n", 1) or 1)
        out = self._make_output(prompt=str(messages), request_id=request_id or "req", n=n)
        if stream:
            return iter([out])
        return out

    def iter_chat_completion_stream(
        self,
        *,
        model: str,
        messages,
        tools=None,
        tool_choice=None,
        sampling_params=None,
        request_id=None,
        chat_template_kwargs=None,
        usage_callback=None,
    ):
        outputs = self.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            sampling_params=sampling_params,
            request_id=request_id,
            stream=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        yield from iter_chat_completion_stream_responses(outputs, model=model, usage_callback=usage_callback)

    def iter_responses_stream(
        self,
        *,
        response_id: str,
        model: str,
        messages,
        tools=None,
        tool_choice=None,
        sampling_params=None,
        request_id=None,
        include_reasoning_summary: bool = False,
        final_response_overrides=None,
        created_at: int | None = None,
        usage_callback=None,
    ):
        outputs = self.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            sampling_params=sampling_params,
            request_id=request_id,
            stream=True,
        )
        yield from iter_responses_stream_frames(
            outputs,
            response_id=response_id,
            model=model,
            include_reasoning_summary=include_reasoning_summary,
            final_response_overrides=final_response_overrides,
            created_at=created_at,
            usage_callback=usage_callback,
        )

    def generate(self, prompts, sampling_params=None, request_id=None, use_tqdm=False):
        call = {"prompts": prompts, "sampling_params": sampling_params, "request_id": request_id}
        self.generate_calls.append(call)
        if self.on_generate is not None:
            return self.on_generate(call)
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        ids = [request_id] if isinstance(request_id, str) else list(request_id or [f"r-{i}" for i in range(len(prompt_list))])
        n = int(getattr(sampling_params, "n", 1) or 1)
        return [self._make_output(prompt=p, request_id=rid, n=n) for p, rid in zip(prompt_list, ids, strict=True)]

    def stream(self, prompt, sampling_params=None, request_id=None):
        self.stream_calls.append({"prompt": prompt, "sampling_params": sampling_params, "request_id": request_id})
        n = int(getattr(sampling_params, "n", 1) or 1)
        yield self._make_output(prompt=prompt, request_id=request_id or "req", n=n)

    def abort_request(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _make_server(
    engine: _ScriptedEngine | dict[str, _ScriptedEngine] | None = None,
    *,
    require_api_key: bool = False,
    admin_key: str | None = None,
    max_concurrent_generations: int | None = None,
    enable_response_store: bool = True,
    default_store_responses: bool = True,
    **kwargs,
) -> tuple[eSurgeApiServer, _ScriptedEngine | dict[str, _ScriptedEngine]]:
    eng = engine if engine is not None else _ScriptedEngine()
    esurge_map = eng if isinstance(eng, dict) else {"test-model": eng}
    server = eSurgeApiServer(
        esurge_map=esurge_map,
        require_api_key=require_api_key,
        admin_key=admin_key,
        enable_persistence=False,
        enable_audit_logging=True,
        max_concurrent_generations=max_concurrent_generations,
        enable_response_store=enable_response_store,
        default_store_responses=default_store_responses,
        **kwargs,
    )
    server.status = ServerStatus.READY
    return server, eng


# ---------------------------------------------------------------------------
# S2-C4: Cross-tenant response-store IDOR in /v1/responses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_s2_c4_previous_response_id_cross_tenant_rejected_403(stream: bool):
    engine = _ScriptedEngine(chat_reply="secret-answer")
    server, _ = _make_server(engine, require_api_key=True)
    key_a, _ = server.auth_manager.generate_api_key(name="tenant-a", role=ApiKeyRole.USER)
    key_b, _ = server.auth_manager.generate_api_key(name="tenant-b", role=ApiKeyRole.USER)

    with TestClient(server.app) as client:
        r1 = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "My secret is 42", "store": True, "stream": stream},
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert r1.status_code == 200, r1.text
        if stream:
            completed = [
                json.loads(line.removeprefix("data: "))
                for line in r1.text.splitlines()
                if line.startswith("data: ") and '"response.completed"' in line
            ]
            assert len(completed) == 1
            resp_id = completed[0]["response"]["id"]
        else:
            resp_id = r1.json()["id"]

        # Tenant B must be denied access to Tenant A's response_id.
        stolen = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "repeat secret", "previous_response_id": resp_id},
            headers={"Authorization": f"Bearer {key_b}"},
        )
        assert stolen.status_code == 403, stolen.text

        # Tenant A can still continue its own response_id with history preserved.
        own_followup = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "follow up", "previous_response_id": resp_id},
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert own_followup.status_code == 200, own_followup.text
        seen_contents = [m["content"] for m in engine.chat_calls[-1]["messages"]]
        assert "My secret is 42" in seen_contents
        assert "follow up" in seen_contents


def test_s2_c4_conversation_id_isolated_across_tenants():
    engine = _ScriptedEngine(chat_reply="ack")
    server, _ = _make_server(engine, require_api_key=True)
    key_a, _ = server.auth_manager.generate_api_key(name="tenant-a", role=ApiKeyRole.USER)
    key_b, _ = server.auth_manager.generate_api_key(name="tenant-b", role=ApiKeyRole.USER)

    with TestClient(server.app) as client:
        r_a1 = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "TENANT_A_SECRET", "conversation": "default", "store": True},
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert r_a1.status_code == 200

        # Tenant B using the same client-chosen conversation id ("default") must not see or overwrite Tenant A.
        r_b1 = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "TENANT_B_MSG", "conversation": "default", "store": True},
            headers={"Authorization": f"Bearer {key_b}"},
        )
        assert r_b1.status_code == 200
        b_messages = [m["content"] for m in engine.chat_calls[-1]["messages"]]
        assert "TENANT_A_SECRET" not in b_messages
        assert b_messages == ["TENANT_B_MSG"]

        # Tenant A's conversation must still hold only Tenant A's prior turn.
        r_a2 = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "TENANT_A_TURN2", "conversation": "default", "store": True},
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert r_a2.status_code == 200
        a2_messages = [m["content"] for m in engine.chat_calls[-1]["messages"]]
        assert "TENANT_A_SECRET" in a2_messages
        assert "TENANT_B_MSG" not in a2_messages


# ---------------------------------------------------------------------------
# S2-H9 & S2-M1: AuthZ/RBAC, unauthenticated model enumeration, READONLY,
# /metrics endpoint path, and immediate reservation reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/completions", {"model": "nonexistent", "prompt": "hi"}),
        ("/v1/responses", {"model": "nonexistent", "input": "hi"}),
    ],
)
def test_s2_h9_unauthenticated_unknown_model_returns_401_without_enumeration(path: str, body: dict):
    server, _ = _make_server(require_api_key=True)
    with TestClient(server.app) as client:
        resp = client.post(path, json=body)
    assert resp.status_code == 401, resp.text
    assert "test-model" not in resp.text
    assert "Available" not in resp.text


def test_s2_h9_model_allowlist_blocks_unlisted_model_before_404():
    server, _ = _make_server(
        {"model-a": _ScriptedEngine(name="model-a"), "model-secret": _ScriptedEngine(name="model-secret")},
        require_api_key=True,
    )
    key, _ = server.auth_manager.generate_api_key(
        name="restricted",
        role=ApiKeyRole.USER,
        permissions=ApiKeyPermissions(allowed_models=["model-a"]),
    )
    with TestClient(server.app) as client:
        r_chat = client.post(
            "/v1/chat/completions",
            json={"model": "missing-model", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-API-Key": key},
        )
        assert r_chat.status_code == 403
        assert "model-secret" not in r_chat.text

        r_get_model = client.get("/v1/models/model-secret", headers={"X-API-Key": key})
        assert r_get_model.status_code == 403


def test_s2_h9_metrics_endpoint_path_and_metadata_tool_permissions():
    server, _ = _make_server(require_api_key=True)
    metrics_only_key, _ = server.auth_manager.generate_api_key(
        name="metrics-only",
        role=ApiKeyRole.USER,
        permissions=ApiKeyPermissions(allowed_endpoints=["/metrics"]),
    )
    no_tools_key, _ = server.auth_manager.generate_api_key(
        name="no-tools",
        role=ApiKeyRole.USER,
        permissions=ApiKeyPermissions(enable_function_calling=False),
    )

    with TestClient(server.app) as client:
        # /metrics must match the registered route path "/metrics" (not "/v1/metrics").
        assert client.get("/metrics", headers={"X-API-Key": metrics_only_key}).status_code == 200
        # Other metadata routes must enforce allowed_endpoints.
        assert client.get("/health", headers={"X-API-Key": metrics_only_key}).status_code == 403
        assert client.get("/v1/models", headers={"X-API-Key": metrics_only_key}).status_code == 403
        assert client.get("/v1/models/test-model", headers={"X-API-Key": metrics_only_key}).status_code == 403
        assert client.get("/v1/tools", headers={"X-API-Key": metrics_only_key}).status_code == 403
        assert client.post("/v1/tools/execute", json={}, headers={"X-API-Key": metrics_only_key}).status_code == 403

        # execute_tool must enforce enable_function_calling.
        r_exec = client.post("/v1/tools/execute", json={}, headers={"X-API-Key": no_tools_key})
        assert r_exec.status_code == 403
        assert "Function calling" in r_exec.json()["detail"]


def test_s2_h9_readonly_role_enforced_across_all_endpoints():
    server, _ = _make_server(require_api_key=True)
    ro_key, ro_meta = server.auth_manager.generate_api_key(name="readonly-key", role=ApiKeyRole.READONLY)

    with TestClient(server.app) as client:
        headers = {"Authorization": f"Bearer {ro_key}"}
        # Allowed read-only routes
        assert client.get("/health", headers=headers).status_code == 200
        assert client.get("/metrics", headers=headers).status_code == 200
        assert client.get("/v1/models", headers=headers).status_code == 200
        assert client.get("/v1/models/test-model", headers=headers).status_code == 200
        assert client.get("/v1/tools", headers=headers).status_code == 200

        # Denied write/generation/admin routes
        assert (
            client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                headers=headers,
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "hi"},
                headers=headers,
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/v1/responses",
                json={"model": "test-model", "input": "hi"},
                headers=headers,
            ).status_code
            == 403
        )
        assert client.post("/v1/tools/execute", json={}, headers=headers).status_code == 403
        assert client.get("/v1/admin/keys", headers=headers).status_code == 403

    # Direct auth_manager enforcement: endpoint=None or requested_tokens>0 must also fail closed for READONLY.
    with pytest.raises(PermissionDenied):
        server.auth_manager.authorize_request(ro_key, endpoint=None)
    with pytest.raises(PermissionDenied):
        server.auth_manager.authorize_request(ro_key, endpoint="/metrics", requested_tokens=1)
    assert server.auth_manager._reservations[ro_meta.key_id] == {}


def test_s2_m1_get_model_reconciles_reservation_immediately_no_leak():
    server, _ = _make_server(require_api_key=True)
    key, meta = server.auth_manager.generate_api_key(
        name="model-reader",
        role=ApiKeyRole.READONLY,
        quota=QuotaConfig(max_total_requests=3),
        permissions=ApiKeyPermissions(allowed_models=["test-model"]),
    )

    with TestClient(server.app) as client:
        for _ in range(3):
            resp = client.get("/v1/models/test-model", headers={"X-API-Key": key})
            assert resp.status_code == 200
            # Every GET /v1/models/{model_id} must immediately reconcile its zero-token reservation.
            assert server.auth_manager._reservations[meta.key_id] == {}

        # Quota of 3 requests is accurately enforced once 3 requests completed.
        assert client.get("/v1/models/test-model", headers={"X-API-Key": key}).status_code == 429

    assert server.auth_manager.get_key_by_id(meta.key_id).total_requests == 3


# ---------------------------------------------------------------------------
# S2-H10: /v1/responses OpenAI-format tool_choice round-trip & engine forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_s2_h10_responses_tool_choice_roundtrip_and_forwarding(stream: bool):
    engine = _ScriptedEngine(chat_reply="done")
    server, _ = _make_server(engine)
    flat_tool = {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
    flat_choice = {"type": "function", "name": "get_weather"}

    # Real eSurge._build_tool_parser_request must accept ResponsesToolChoice without raising ValidationError.
    parsed_choice = ResponsesToolChoice.model_validate(flat_choice)
    tool_parser_req = eSurge._build_tool_parser_request(
        prompt="weather?",
        tools=[flat_tool],
        tool_choice=parsed_choice,
    )
    assert tool_parser_req is not None
    assert tool_parser_req.tool_choice.function.name == "get_weather"

    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "What is the weather?",
                "tools": [flat_tool],
                "tool_choice": flat_choice,
                "stream": stream,
            },
        )
        assert resp.status_code == 200, resp.text
        if stream:
            completed = [
                json.loads(line.removeprefix("data: "))
                for line in resp.text.splitlines()
                if line.startswith("data: ") and '"response.completed"' in line
            ]
            assert len(completed) == 1
            payload = completed[0]["response"]
        else:
            payload = resp.json()

        assert payload["tool_choice"] == flat_choice
        revived = ResponsesResponse.model_validate(payload)
        assert isinstance(revived.tool_choice, ResponsesToolChoice)
        assert revived.tool_choice.name == "get_weather"
        assert isinstance(engine.chat_calls[-1]["tool_choice"], ResponsesToolChoice)
        assert engine.chat_calls[-1]["tool_choice"].name == "get_weather"

        # Also verify tool_choice="none" is forwarded.
        resp_none = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Just reply",
                "tools": [flat_tool],
                "tool_choice": "none",
                "stream": stream,
            },
        )
        assert resp_none.status_code == 200
        assert engine.chat_calls[-1]["tool_choice"] == "none"


# ---------------------------------------------------------------------------
# S2-H11: ResponsesStreamAccumulator output_item.added snapshot immutability
# ---------------------------------------------------------------------------


def test_s2_h11_output_item_added_snapshots_are_immutable_before_serialization():
    acc = ResponsesStreamAccumulator(
        response_id="resp_immut",
        model="test-model",
        include_reasoning_summary=True,
        final_response_overrides=ResponsesFinalizationOptions(store=False),
    )

    all_frames = list(acc.initial_frames())

    # Step 1: reasoning + first tool call chunk (name + first part of JSON args)
    all_frames.extend(
        acc.add_output(
            RequestOutput(
                request_id="r1",
                prompt="hi",
                prompt_token_ids=[1],
                outputs=[CompletionOutput(index=0, text="", token_ids=[2], reasoning_content="plan ")],
                delta_reasoning_content="plan ",
                reasoning_content="plan ",
                delta_tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=DeltaFunctionCall(name="get_weather", arguments='{"city":'),
                    )
                ],
                num_generated_tokens=1,
            )
        )
    )
    # Step 2: more reasoning + second part of JSON args + visible text
    all_frames.extend(
        acc.add_output(
            RequestOutput(
                request_id="r1",
                prompt="hi",
                prompt_token_ids=[1],
                outputs=[CompletionOutput(index=0, text="Checking.", token_ids=[2, 3], reasoning_content="plan now")],
                delta_reasoning_content="now",
                reasoning_content="plan now",
                delta_tool_calls=[
                    DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='"Paris"}')),
                ],
                delta_text="Checking.",
                accumulated_text="Checking.",
                num_generated_tokens=2,
            )
        )
    )
    # Finalize while all earlier frames are still in memory (simulating _AsyncStreamQueue buffering).
    final_frames, _ = acc.finalize(
        last_output=RequestOutput(
            request_id="r1",
            prompt="hi",
            prompt_token_ids=[1],
            outputs=[
                CompletionOutput(
                    index=0,
                    text="Checking.",
                    token_ids=[2, 3],
                    reasoning_content="plan now",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function=FunctionCall(name="get_weather", arguments='{"city":"Paris"}'),
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ],
            accumulated_text="Checking.",
            reasoning_content="plan now",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="get_weather", arguments='{"city":"Paris"}'),
                )
            ],
            finished=True,
            num_generated_tokens=2,
        ),
        prompt_tokens=1,
    )
    all_frames.extend(final_frames)

    added_events = [f.payload for f in all_frames if f.event == "response.output_item.added"]
    done_events = [f.payload for f in all_frames if f.event == "response.output_item.done"]
    arg_deltas = [f.payload.delta for f in all_frames if f.event == "response.function_call_arguments.delta"]

    assert len(added_events) == 3
    assert len(done_events) == 3

    added_by_type = {ev.item.type: ev.item for ev in added_events}
    done_by_type = {ev.item.type: ev.item for ev in done_events}

    # Every .added frame must remain in_progress with empty payload (never mutated by later steps/finalize).
    r_added = added_by_type["reasoning"]
    assert isinstance(r_added, ResponseReasoningItem)
    assert r_added.status == "in_progress"
    assert r_added.summary[0].text == ""
    assert done_by_type["reasoning"].status == "completed"
    assert done_by_type["reasoning"].summary[0].text == "plan now"

    fc_added = added_by_type["function_call"]
    assert isinstance(fc_added, ResponseFunctionCallItem)
    assert fc_added.status == "in_progress"
    assert fc_added.name == "get_weather"
    assert fc_added.arguments == ""
    # OpenAI client reconstructs args by starting from .added.item.arguments and appending all .delta chunks.
    reconstructed_args = fc_added.arguments + "".join(arg_deltas)
    assert json.loads(reconstructed_args) == {"city": "Paris"}
    assert done_by_type["function_call"].status == "completed"
    assert done_by_type["function_call"].arguments == '{"city":"Paris"}'

    msg_added = added_by_type["message"]
    assert isinstance(msg_added, ResponseMessageItem)
    assert msg_added.status == "in_progress"
    assert msg_added.content == []
    assert done_by_type["message"].status == "completed"
    assert done_by_type["message"].content[0].text == "Checking."


# ---------------------------------------------------------------------------
# S2-H12: /v1/responses streaming persists BEFORE yielding response.completed
# ---------------------------------------------------------------------------


def test_s2_h12_streaming_responses_persists_before_yielding_completed():
    engine = _ScriptedEngine(chat_reply="stored-before-yield")
    server, _ = _make_server(engine)

    async def run():
        req = ResponsesRequest(model="test-model", input="turn 1", conversation="conv-h12", store=True, stream=True)
        raw_req = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1234)})
        raw_req.is_disconnected = AsyncMock(return_value=False)

        response = await server.responses(req, raw_req)
        completed_id = None
        iterator = response.body_iterator
        try:
            async for chunk in iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else chunk
                if text.startswith("event: response.completed"):
                    data_line = next(ln for ln in text.splitlines() if ln.startswith("data: "))
                    completed_id = json.loads(data_line.removeprefix("data: "))["response"]["id"]
                    # Check persistence and token accounting IMMEDIATELY upon receiving response.completed
                    # before the generator even resumes!
                    stored_resp = await server._response_store_get_response(completed_id)
                    stored_conv = await server._response_store_get_conversation(
                        server._scoped_conversation_store_key("conv-h12", None)
                    )
                    assert stored_resp is not None, "response must be persisted BEFORE response.completed is yielded"
                    assert stored_conv is not None, "conversation must be persisted BEFORE response.completed is yielded"
                    assert server.metrics.total_tokens_generated == 2
                    break
        finally:
            await iterator.aclose()
        assert completed_id is not None
        return completed_id

    resp_id = asyncio.run(run())

    # Immediate follow-up using previous_response_id succeeds even though stream closed right on response.completed.
    with TestClient(server.app) as client:
        followup = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "turn 2", "previous_response_id": resp_id},
        )
        assert followup.status_code == 200, followup.text


# ---------------------------------------------------------------------------
# S2-H13: compute_stream_delta_text shrink duplication & O(L) complexity
# ---------------------------------------------------------------------------


def test_s2_h13_compute_stream_delta_text_shrink_does_not_duplicate():
    assert compute_stream_delta_text("hello ", "hello h", "") == ""
    assert compute_stream_delta_text("hello ", "hello h", "fallback") == ""
    assert compute_stream_delta_text("abc", "abcdefa", "") == ""
    # Genuine overlapping boundary rewrite still recovers the tail cleanly.
    assert compute_stream_delta_text("world!", "hello world", "") == "!"
    assert compute_stream_delta_text("babac", "ababa", "") == "c"


def test_s2_h13_compute_stream_delta_text_linear_time_on_large_inputs():
    n = 50_000
    prev = "a" * n + "b"
    curr = "a" * n + "c"
    t0 = time.perf_counter()
    delta = compute_stream_delta_text(curr, prev, "")
    elapsed = time.perf_counter() - t0
    assert delta == ""
    assert elapsed < 0.5, f"compute_stream_delta_text took {elapsed:.3f}s (should be linear O(L))"

    # Large overlap case also linear.
    prev2 = "x" + "a" * n
    curr2 = "a" * n + "!"
    t1 = time.perf_counter()
    delta2 = compute_stream_delta_text(curr2, prev2, "")
    elapsed2 = time.perf_counter() - t1
    assert delta2 == "!"
    assert elapsed2 < 0.5, f"overlap case took {elapsed2:.3f}s"


# ---------------------------------------------------------------------------
# S2-H14: Pre-flight 503 on streaming /v1/responses & batch /v1/completions;
# streaming /v1/chat/completions n > 1 choices
# ---------------------------------------------------------------------------


def test_s2_h14_overloaded_streaming_responses_and_batch_completions_return_503():
    server, _ = _make_server(max_concurrent_generations=1)
    # Occupy the sole generation slot.
    token = server._generation_slots.get_nowait()
    try:
        with TestClient(server.app) as client:
            r_resp = client.post(
                "/v1/responses",
                json={"model": "test-model", "input": "hello", "stream": True},
            )
            assert r_resp.status_code == 503, r_resp.text

            r_batch = client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": ["p1", "p2"], "stream": True},
            )
            assert r_batch.status_code == 503, r_batch.text
    finally:
        server._generation_slots.put_nowait(token)


def test_s2_h14_chat_stream_emits_all_n_choices():
    outputs = iter(
        [
            RequestOutput(
                request_id="req-n3",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[
                    CompletionOutput(index=0, text="A", token_ids=[10], finish_reason=None),
                    CompletionOutput(index=1, text="X", token_ids=[20], finish_reason=None),
                    CompletionOutput(index=2, text="P", token_ids=[30], finish_reason=None),
                ],
                accumulated_text="A",
                delta_text="A",
                num_generated_tokens=3,
            ),
            RequestOutput(
                request_id="req-n3",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[
                    CompletionOutput(index=0, text="AB", token_ids=[10, 11], finish_reason="stop"),
                    CompletionOutput(index=1, text="XY", token_ids=[20, 21], finish_reason="length"),
                    CompletionOutput(index=2, text="PQ", token_ids=[30, 31], finish_reason="stop"),
                ],
                finished=True,
                accumulated_text="AB",
                delta_text="B",
                num_generated_tokens=6,
            ),
        ]
    )

    chunks = list(iter_chat_completion_stream_responses(outputs, model="test-model"))
    assembled: dict[int, str] = {0: "", 1: "", 2: ""}
    roles_seen: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for chunk in chunks[:-1]:
        for choice in chunk.choices:
            if choice.delta.content:
                assembled[choice.index] += choice.delta.content
            if choice.delta.role == "assistant":
                roles_seen[choice.index] += 1

    assert assembled == {0: "AB", 1: "XY", 2: "PQ"}
    assert roles_seen == {0: 1, 1: 1, 2: 1}
    final_reasons = {choice.index: choice.finish_reason for choice in chunks[-1].choices}
    assert final_reasons == {0: "stop", 1: "length", 2: "stop"}


# ---------------------------------------------------------------------------
# S2-H15: Invalid sampling/admin params -> 400/422, top_p=0.0,
# max_completion_tokens, PATCH null boolean permissions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_params",
    [
        {"temperature": 0.0, "n": 2},
        {"n": 0},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"presence_penalty": 2.5},
        {"frequency_penalty": -2.5},
        {"repetition_penalty": 0.0},
        {"min_p": -0.1},
        {"top_k": -1},
        {"temperature": -0.5},
    ],
)
def test_s2_h15_invalid_sampling_params_rejected_422_never_500(bad_params: dict):
    server, _ = _make_server()
    with TestClient(server.app) as client:
        r_chat = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], **bad_params},
        )
        assert r_chat.status_code == 422, f"chat {bad_params}: {r_chat.status_code} {r_chat.text}"

        r_cmpl = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi", **bad_params},
        )
        assert r_cmpl.status_code == 422, f"completions {bad_params}: {r_cmpl.status_code} {r_cmpl.text}"

        r_resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "hi", **bad_params},
        )
        assert r_resp.status_code == 422, f"responses {bad_params}: {r_resp.status_code} {r_resp.text}"


def test_s2_h15_chat_max_completion_tokens_and_explicit_top_p_zero():
    engine = _ScriptedEngine()
    server, _ = _make_server(engine)

    with TestClient(server.app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 17,
            },
        )
        assert r.status_code == 200, r.text
        assert engine.chat_calls[-1]["sampling_params"].max_tokens == 17

        r_bad = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 0,
            },
        )
        assert r_bad.status_code == 422

    # Direct _create_sampling_params must not coerce top_p=0.0 into 1.0 via `or 1.0`.
    raw_req = SimpleNamespace(
        max_tokens=16,
        temperature=0.7,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=0.0,
        min_p=0.0,
        n=1,
        stop=None,
    )
    with pytest.raises(ValueError, match="top_p"):
        server._create_sampling_params(raw_req)


def test_s2_h15_admin_negative_limits_and_patch_null_boolean_permissions_rejected():
    admin_secret = "sk-admin-secret-key-0123456789"
    server, _ = _make_server(require_api_key=True, admin_key=admin_secret)
    headers = {"Authorization": f"Bearer {admin_secret}"}

    with TestClient(server.app) as client:
        # Negative limits on POST /v1/admin/keys return 422 (never 500).
        for bad_body in (
            {"name": "k1", "requests_per_minute": -1},
            {"name": "k2", "max_total_tokens": -5},
            {"name": "k3", "max_tokens_per_request": -1},
        ):
            r = client.post("/v1/admin/keys", json=bad_body, headers=headers)
            assert r.status_code == 422, r.text

        created = client.post("/v1/admin/keys", json={"name": "victim"}, headers=headers)
        assert created.status_code == 201
        key_id = created.json()["key_id"]

        # PATCH with negative limits returns 422.
        assert client.patch(f"/v1/admin/keys/{key_id}", json={"tokens_per_minute": -10}, headers=headers).status_code == 422

        # PATCH with null boolean permission must be rejected and must NOT disable streaming/function calling!
        r_null_stream = client.patch(
            f"/v1/admin/keys/{key_id}",
            json={"enable_streaming": None},
            headers=headers,
        )
        assert r_null_stream.status_code == 422, r_null_stream.text

        r_null_fc = client.patch(
            f"/v1/admin/keys/{key_id}",
            json={"enable_function_calling": None},
            headers=headers,
        )
        assert r_null_fc.status_code == 422, r_null_fc.text

        meta = server.auth_manager.get_key_by_id(key_id)
        assert meta.permissions.enable_streaming is True
        assert meta.permissions.enable_function_calling is True


# ---------------------------------------------------------------------------
# S2-H16: RequestTrackingMiddleware UUID & X-Request-ID preservation,
# _start_stream_task close wake & error iterator close, _graceful_shutdown
# ---------------------------------------------------------------------------


def test_s2_h16_request_tracking_unique_ids_and_preserves_handler_header(monkeypatch):
    monkeypatch.setattr("easydel.inference.inference_engine_interface.time.time", lambda: 1_700_000_000.123456)
    server = SimpleNamespace(metrics=ServerMetrics(), _active_requests=set())
    app = FastAPI()

    @app.get("/auto")
    async def auto_route(request: Request):
        return {"state_id": request.state.request_id}

    @app.get("/custom")
    async def custom_route():
        return JSONResponse({"ok": True}, headers={"X-Request-ID": "resp_engine_123"})

    app.add_middleware(RequestTrackingMiddleware, server=server)

    with TestClient(app) as client:
        r1 = client.get("/auto")
        r2 = client.get("/auto")
        assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]
        assert r1.headers["X-Request-ID"] == r1.json()["state_id"]

        r_custom = client.get("/custom")
        assert r_custom.headers["X-Request-ID"] == "resp_engine_123"


def test_s2_h16_stream_queue_close_wakes_get_and_producer_error_closes_iterator():
    server, _ = _make_server()

    # 1. close() wakes a consumer parked in get() promptly with ("end", None).
    block_producer = threading.Event()

    def slow_stream():
        block_producer.wait(timeout=5.0)
        yield "never"

    async def test_close_wake():
        queue = server._start_stream_task(slow_stream)
        getter = asyncio.create_task(queue.get())
        await asyncio.sleep(0.05)
        t0 = time.monotonic()
        queue.close()
        result = await asyncio.wait_for(getter, timeout=0.5)
        elapsed = time.monotonic() - t0
        block_producer.set()
        queue._producer_future.result(timeout=5.0)
        return result, elapsed

    result, elapsed = asyncio.run(test_close_wake())
    assert result == ("end", None)
    assert elapsed < 0.4

    # 2. Non-abandoned producer error still closes the iterator in finally.
    iterator_closed = threading.Event()

    class _FailingIterator:
        def __next__(self):
            raise RuntimeError("boom")

        def close(self):
            iterator_closed.set()

    async def test_error_closes():
        queue = server._start_stream_task(_FailingIterator)
        kind, err = await queue.get()
        assert kind == "error" and isinstance(err, RuntimeError)
        end_kind, _ = await queue.get()
        assert end_kind == "end"
        queue._producer_future.result(timeout=5.0)

    asyncio.run(test_error_closes())
    assert iterator_closed.is_set()

    # 3. _graceful_shutdown clears _stream_pool so subsequent pool requests work after restart.
    old_pool = server._get_stream_pool()
    asyncio.run(server._graceful_shutdown())
    assert server._stream_pool is None
    new_pool = server._get_stream_pool()
    assert new_pool is not old_pool
    assert new_pool.submit(lambda: "restarted").result(timeout=2.0) == "restarted"
    new_pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# S2-M4: oai_proxies.py (InferenceApiRouter) serialization, stream leak,
# HTTP status preservation, backend auth, and lifespan cleanup
# ---------------------------------------------------------------------------


def test_s2_m4_oai_proxies_param_serialization_and_extensions():
    router = object.__new__(InferenceApiRouter)
    chat_req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hello")],
        max_completion_tokens=64,
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice=ToolChoiceOption(type="function", function=ToolChoiceFunction(name="f")),
        function_call=FunctionCallSelection(name="legacy_f"),
        top_k=20,
        min_p=0.05,
        ignore_eos=True,
        suppress_tokens=[1, 2],
        chat_template_kwargs={"enable_thinking": False},
        repetition_penalty=1.15,
        seed=42,
        response_format={"type": "json_object"},
        stream_options={"include_usage": True},
    )
    params = router.build_oai_params_from_chat_request(chat_req)
    assert params["messages"] == [{"role": "user", "content": "hello"}]
    assert params["tool_choice"] == {"type": "function", "function": {"name": "f"}}
    assert params["function_call"] == {"name": "legacy_f"}
    assert params["max_completion_tokens"] == 64
    assert params["seed"] == 42
    assert params["response_format"] == {"type": "json_object"}
    assert params["stream_options"] == {"include_usage": True}
    assert params["extra_body"] == {
        "repetition_penalty": 1.15,
        "top_k": 20,
        "min_p": 0.05,
        "ignore_eos": True,
        "suppress_tokens": [1, 2],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_s2_m4_oai_proxies_streaming_status_and_stream_closure(monkeypatch):
    class _APIStatusError(Exception):
        def __init__(self, status_code: int):
            super().__init__(f"status {status_code}")
            self.status_code = status_code

    closed_streams: list[str] = []

    class _FakeUpstreamStream:
        def __init__(self, name: str):
            self.name = name
            self._idx = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx >= 2:
                raise StopAsyncIteration
            self._idx += 1
            return SimpleNamespace(model_dump_json=lambda **_kw: json.dumps({"chunk": self._idx}))

        async def close(self):
            closed_streams.append(self.name)

    client_closed = []
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock()),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
        completions=SimpleNamespace(create=AsyncMock()),
        close=AsyncMock(side_effect=lambda: client_closed.append(True)),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=lambda **kw: client, APIError=_APIStatusError),
    )
    router = InferenceApiRouter(api_key="sk-proxy-test", base_url="http://127.0.0.1:9999/v1")

    with TestClient(router.app) as http:
        # Upstream 429 on streaming must return HTTP 429, NOT HTTP 200.
        client.chat.completions.create.side_effect = _APIStatusError(429)
        r = http.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 429

        client.completions.create.side_effect = _APIStatusError(401)
        r2 = http.post("/v1/completions", json={"model": "m", "prompt": "hi", "stream": True})
        assert r2.status_code == 401

        client.responses.create.side_effect = _APIStatusError(400)
        r3 = http.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True})
        assert r3.status_code == 400

        # Early close of opened upstream stream must close it.
        client.chat.completions.create.side_effect = None
        client.chat.completions.create.return_value = _FakeUpstreamStream("chat-1")

        async def open_and_close_early():
            resp = await router.chat_completions(
                ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")], stream=True)
            )
            await resp.body_iterator.aclose()

        asyncio.run(open_and_close_early())
        assert closed_streams == ["chat-1"]

    # Lifespan exit closes self.client.
    assert client_closed == [True]


def test_s2_m4_oai_proxies_real_http_auth_forwarding_non_json_and_proxy_auth(monkeypatch):
    seen_auth: list[tuple[str, str | None]] = []

    class _BackendHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            seen_auth.append((self.path, self.headers.get("Authorization")))
            if self.headers.get("Authorization") != "Bearer sk-backend-secret":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"error":"unauthorized"}')
                return
            payload = {"ok": True, "path": self.path}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen_auth.append((self.path, self.headers.get("Authorization")))
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw)
            if data.get("non_json"):
                self.send_response(502)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html>Bad Gateway</html>")
                return
            body = json.dumps({"executed": data}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), _BackendHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock()),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
        completions=SimpleNamespace(create=AsyncMock()),
        close=AsyncMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=lambda **kw: client, APIError=Exception),
    )

    try:
        router = InferenceApiRouter(
            api_key="sk-backend-secret",
            base_url=f"http://127.0.0.1:{port}/v1",
            require_api_key=True,
            proxy_api_key="sk-client-token",
        )
        with TestClient(router.app) as http:
            # Missing proxy auth rejected with 401 on execute_tool & metrics.
            assert http.post("/v1/tools/execute", json={"name": "f"}).status_code == 401
            assert http.get("/metrics").status_code == 401

            # Authorized client request forwards backend Authorization header and reuses timed aiohttp session.
            # Notice: send X-API-Key for proxy auth so backend gets Bearer sk-backend-secret if no Authorization header,
            # or send Authorization: Bearer sk-backend-secret when proxy_api_key == api_key.
            router._proxy_api_key = "sk-backend-secret"
            auth_headers = {"Authorization": "Bearer sk-backend-secret"}
            r_metrics = http.get("/metrics", headers=auth_headers)
            assert r_metrics.status_code == 200
            assert r_metrics.json()["backend_metrics"]["ok"] is True

            first_session = router._http_session
            assert first_session is not None and not first_session.closed
            assert first_session.timeout.total == 30.0

            r_tools = http.get("/v1/tools", headers=auth_headers)
            assert r_tools.status_code == 200
            assert router._http_session is first_session

            r_exec = http.post("/v1/tools/execute", json={"name": "ping"}, headers=auth_headers)
            assert r_exec.status_code == 200
            assert r_exec.json()["executed"] == {"name": "ping"}

            # Non-JSON backend error reply handled gracefully (no JSONDecodeError 500 crash).
            r_non_json = http.post("/v1/tools/execute", json={"non_json": True}, headers=auth_headers)
            assert r_non_json.status_code == 502
            assert "Bad Gateway" in r_non_json.json()["error"]["message"]

        assert first_session.closed
        assert ("/metrics", "Bearer sk-backend-secret") in seen_auth
        assert ("/v1/tools", "Bearer sk-backend-secret") in seen_auth
        assert ("/v1/tools/execute", "Bearer sk-backend-secret") in seen_auth
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# S2-M5: Non-streaming content+tool_calls parity, responses normalization,
# tokenizer offloading, batch slot/reservation scaling, cancel accounting,
# stream_protocol outputs.close()
# ---------------------------------------------------------------------------


def test_s2_m5_non_streaming_chat_and_responses_preserve_parsed_content_with_tool_calls():
    tool_call = ToolCall(id="call_1", type="function", function=FunctionCall(name="lookup", arguments='{"q":"x"}'))
    engine = _ScriptedEngine(
        chat_reply="Let me look that up.",
        tool_calls=[tool_call],
        raw_text='Let me look that up.<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>',
    )
    server, _ = _make_server(engine)

    with TestClient(server.app) as client:
        r_chat = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "find x"}]},
        )
        assert r_chat.status_code == 200
        choice = r_chat.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] == "Let me look that up."
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"

        r_resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "find x"},
        )
        assert r_resp.status_code == 200
        types = [item["type"] for item in r_resp.json()["output"]]
        assert "function_call" in types
        assert "message" in types
        msg_item = next(item for item in r_resp.json()["output"] if item["type"] == "message")
        assert msg_item["content"][0]["text"] == "Let me look that up."


def test_s2_m5_responses_normalizes_text_list_parts_for_chat_template():
    engine = _ScriptedEngine(chat_reply="ok")
    server, _ = _make_server(engine)

    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}, {"type": "input_text", "text": "world"}],
                    }
                ],
            },
        )
        assert resp.status_code == 200
        sent_messages = engine.chat_calls[-1]["messages"]
        assert sent_messages[0]["content"] == "hello world"


def test_s2_m5_batch_completions_offloads_tokenizer_scales_slots_and_uses_one_disconnect_task():
    engine = _ScriptedEngine(max_num_seqs=2)
    server, _ = _make_server(engine, require_api_key=True, max_concurrent_generations=2)
    key, _ = server.auth_manager.generate_api_key(
        name="limited",
        role=ApiKeyRole.USER,
        permissions=ApiKeyPermissions(max_tokens_per_request=20),
    )

    disconnect_calls = []
    orig_disconnect = server._abort_non_stream_request_on_disconnect

    async def spy_disconnect(**kwargs):
        disconnect_calls.append(kwargs["request_id"])
        return await orig_disconnect(**kwargs)

    server._abort_non_stream_request_on_disconnect = spy_disconnect

    async def run_batch_checks():
        event_loop_thread = threading.get_ident()
        raw_req = Request({"type": "http", "headers": [(b"x-api-key", key.encode())], "client": ("127.0.0.1", 1)})
        raw_req.is_disconnected = AsyncMock(return_value=False)

        # 2 prompts * max_tokens=10 * n=1 = 20 <= 20 -> allowed.
        resp = await server.completions(
            CompletionRequest(model="test-model", prompt=["hello world", "foo bar"], max_tokens=10),
            raw_req,
        )
        assert len(resp.choices) == 2
        # Tokenizer must have executed off the event-loop thread.
        assert engine.tokenizer_threads and all(tid != event_loop_thread for tid in engine.tokenizer_threads)
        # Only ONE disconnect polling task created for the entire batch.
        assert len(disconnect_calls) == 1
        assert isinstance(disconnect_calls[0], list) and len(disconnect_calls[0]) == 2

        # 3 prompts * max_tokens=10 = 30 > max_tokens_per_request=20 -> rejected 403.
        raw_req2 = Request({"type": "http", "headers": [(b"x-api-key", key.encode())], "client": ("127.0.0.1", 1)})
        with pytest.raises(Exception) as excinfo:
            await server.completions(
                CompletionRequest(model="test-model", prompt=["a", "b", "c"], max_tokens=10),
                raw_req2,
            )
        assert getattr(excinfo.value, "status_code", None) == 403

        # With only 1 slot free out of 2, a 2-prompt batch must be rejected with 503 (rolling back cleanly).
        held = server._generation_slots.get_nowait()
        try:
            raw_req3 = Request({"type": "http", "headers": [(b"x-api-key", key.encode())], "client": ("127.0.0.1", 1)})
            with pytest.raises(Exception) as excinfo3:
                await server.completions(
                    CompletionRequest(model="test-model", prompt=["a", "b"], max_tokens=5),
                    raw_req3,
                )
            assert getattr(excinfo3.value, "status_code", None) == 503
            # Slot must not be leaked on rollback.
            assert server._generation_slots.qsize() == 1
        finally:
            server._generation_slots.put_nowait(held)

    asyncio.run(run_batch_checks())


def test_s2_m5_non_streaming_cancel_holds_slot_until_abort_finishes_and_records_partial_usage():
    started = threading.Event()
    allow_exit = threading.Event()

    def blocking_chat(call):
        started.set()
        allow_exit.wait(timeout=5.0)
        return RequestOutput(
            request_id=call["request_id"],
            prompt="hi",
            prompt_token_ids=[1, 2, 3],
            outputs=[CompletionOutput(index=0, text="part", token_ids=[4, 5], finish_reason="abort")],
            finished=True,
            accumulated_text="part",
            num_generated_tokens=2,
        )

    engine = _ScriptedEngine(on_chat=blocking_chat, max_num_seqs=1)
    server, _ = _make_server(engine, require_api_key=True, max_concurrent_generations=1)
    key, meta = server.auth_manager.generate_api_key(name="cancel-user", role=ApiKeyRole.USER)

    async def run():
        raw_req = Request({"type": "http", "headers": [(b"x-api-key", key.encode())], "client": ("127.0.0.1", 1)})
        raw_req.is_disconnected = AsyncMock(return_value=False)
        task = asyncio.create_task(
            server.chat_completions(
                ChatCompletionRequest(
                    model="test-model",
                    messages=[ChatMessage(role="user", content="hi")],
                    max_tokens=16,
                ),
                raw_req,
            )
        )
        await asyncio.to_thread(started.wait, 5.0)
        task.cancel()
        await asyncio.sleep(0.05)
        # While worker thread is still unwinding, slot MUST remain occupied (0 available) and abort requested!
        assert len(engine.aborted) == 1
        assert server._generation_slots.qsize() == 0

        allow_exit.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # After worker thread finishes, slot is freed and partial usage (3 prompt, 2 completion) is accounted.
        assert server._generation_slots.qsize() == 1

    asyncio.run(run())
    updated = server.auth_manager.get_key_by_id(meta.key_id)
    assert (updated.total_requests, updated.total_prompt_tokens, updated.total_completion_tokens) == (1, 3, 2)


def test_s2_m5_stream_protocol_closes_outputs_on_early_exit():
    closed = []

    def gen_outputs():
        try:
            yield RequestOutput(
                request_id="r",
                prompt="hi",
                prompt_token_ids=[1],
                outputs=[CompletionOutput(index=0, text="a", token_ids=[2])],
                accumulated_text="a",
                delta_text="a",
                num_generated_tokens=1,
            )
            yield RequestOutput(
                request_id="r",
                prompt="hi",
                prompt_token_ids=[1],
                outputs=[CompletionOutput(index=0, text="ab", token_ids=[2, 3], finish_reason="stop")],
                accumulated_text="ab",
                delta_text="b",
                finished=True,
                num_generated_tokens=2,
            )
        finally:
            closed.append("closed")

    chat_iter = iter_chat_completion_stream_responses(gen_outputs(), model="m")
    next(chat_iter)
    chat_iter.close()
    assert closed == ["closed"]

    resp_iter = iter_responses_stream_frames(gen_outputs(), response_id="resp_1", model="m")
    next(resp_iter)  # response.created
    next(resp_iter)  # output_item.added
    resp_iter.close()
    assert closed == ["closed", "closed"]


# ---------------------------------------------------------------------------
# S2-M6: BaseInferenceApiServer CORS, LRU/deepcopy response store, OpenAPI,
# _extract_responses_tools fallback, _parse_responses_max_tokens(0), stop=[]
# ---------------------------------------------------------------------------


def test_s2_m6_cors_credentials_disabled_for_wildcard_and_enabled_for_explicit_origins():
    wildcard_server, _ = _make_server(enable_cors=True, cors_origins=None)
    wildcard_cors = next(m for m in wildcard_server.app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert wildcard_cors.kwargs["allow_origins"] == ["*"]
    assert wildcard_cors.kwargs["allow_credentials"] is False

    explicit_server, _ = _make_server(enable_cors=True, cors_origins=["https://app.example.com"])
    explicit_cors = next(m for m in explicit_server.app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert explicit_cors.kwargs["allow_origins"] == ["https://app.example.com"]
    assert explicit_cors.kwargs["allow_credentials"] is True


def test_s2_m6_response_store_is_true_lru_and_returns_deep_copies():
    server, _ = _make_server(max_stored_responses=2, max_stored_conversations=2)

    async def run():
        await server._response_store_put_response("r1", {"nested": ["v1"]})
        await server._response_store_put_response("r2", {"nested": ["v2"]})
        # Touch r1 -> r1 becomes MRU, r2 becomes LRU.
        got_r1 = await server._response_store_get_response("r1")
        assert got_r1 == {"nested": ["v1"]}
        # Mutating returned dict must NOT affect stored copy.
        got_r1["nested"].append("mutated")
        # Insert r3 -> must evict r2 (LRU), NOT r1!
        await server._response_store_put_response("r3", {"nested": ["v3"]})
        assert await server._response_store_get_response("r2") is None
        assert await server._response_store_get_response("r1") == {"nested": ["v1"]}

        # Same LRU & deep-copy invariants for conversations.
        await server._response_store_put_conversation("c1", [{"role": "user", "content": "a"}])
        await server._response_store_put_conversation("c2", [{"role": "user", "content": "b"}])
        got_c1 = await server._response_store_get_conversation("c1")
        got_c1.append({"role": "user", "content": "mutated"})
        await server._response_store_put_conversation("c3", [{"role": "user", "content": "c"}])
        assert await server._response_store_get_conversation("c2") is None
        assert await server._response_store_get_conversation("c1") == [{"role": "user", "content": "a"}]

    asyncio.run(run())


def test_s2_m6_openapi_security_without_components_key():
    app = FastAPI(title="Minimal", version="1.0.0")

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    stub = object.__new__(eSurgeApiServer)
    stub.app = app
    stub._configure_openapi_security()
    schema = app.openapi()
    assert "BearerAuth" in schema["components"]["securitySchemes"]
    assert "ApiKeyAuth" in schema["components"]["securitySchemes"]


def test_s2_m6_responses_helpers_extract_tools_max_tokens_zero_and_stop_default():
    # Fallback branch in _extract_responses_tools must pop "type" from nested function payload.
    class _CustomTool:
        def model_dump(self, exclude_none=True):
            return {"type": "function", "name": "custom_fn", "parameters": {"type": "object"}}

    fake_req = SimpleNamespace(tools=[_CustomTool()], functions=None)
    _raw, for_template = BaseInferenceApiServer._extract_responses_tools(fake_req)
    assert for_template == [
        {"type": "function", "function": {"name": "custom_fn", "parameters": {"type": "object"}}}
    ]

    # _parse_responses_max_tokens(0) must raise ValueError instead of becoming unbounded None.
    server, _ = _make_server()
    with pytest.raises(ValueError, match="max_output_tokens"):
        server._parse_responses_max_tokens(
            SimpleNamespace(max_output_tokens=0, max_tokens=None, max_completion_tokens=None),
            None,
        )
    # -1 still means unlimited (None).
    assert server._parse_responses_max_tokens(
        SimpleNamespace(max_output_tokens=-1, max_tokens=None, max_completion_tokens=None),
        None,
    )[1] is None

    # _create_sampling_params_from_responses must default stop to [] (not None).
    sp = BaseInferenceApiServer._create_sampling_params_from_responses(
        ResponsesRequest(model="test-model", input="hi"),
        max_tokens=16,
    )
    assert sp.stop == []


# ---------------------------------------------------------------------------
# Audit #7: Shielded upstream AsyncStream closure on client disconnect in oai_proxies
# ---------------------------------------------------------------------------


def test_leak_fix_7_oai_proxies_closes_upstream_stream_on_disconnect_mid_send_and_cancel(monkeypatch):
    closed_events: list[str] = []

    class _AsyncUpstreamStream:
        def __init__(self, name: str):
            self.name = name
            self._sent = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self._sent += 1
            if self._sent > 5:
                raise StopAsyncIteration
            return SimpleNamespace(
                type="response.output_text.delta",
                model_dump_json=lambda **_kw: json.dumps({"idx": self._sent}),
            )

        async def close(self):
            # Yield across an await point to verify CancelScope(shield=True) protects teardown.
            await asyncio.sleep(0)
            closed_events.append(self.name)

    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock()),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
        completions=SimpleNamespace(create=AsyncMock()),
        close=AsyncMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=lambda **kw: client, APIError=Exception),
    )
    router = InferenceApiRouter(api_key="sk-backend", base_url="http://127.0.0.1:9999/v1")

    async def exercise_disconnect_mid_send(endpoint: str) -> None:
        stream = _AsyncUpstreamStream(endpoint)
        if endpoint == "responses":
            client.responses.create.return_value = stream
            resp = await router.responses(
                ResponsesRequest(model="m", input="hi", stream=True),
                SimpleNamespace(headers={}),
            )
        elif endpoint == "chat":
            client.chat.completions.create.return_value = stream
            resp = await router.chat_completions(
                ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")], stream=True),
                SimpleNamespace(headers={}),
            )
        else:
            client.completions.create.return_value = stream
            resp = await router.completions(
                CompletionRequest(model="m", prompt="hi", stream=True),
                SimpleNamespace(headers={}),
            )

        body_started = asyncio.Event()
        never_finish_send = asyncio.Event()

        async def receive():
            await body_started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                body_started.set()
                # Block in send() so Starlette cancels stream_response while awaiting send(),
                # NOT inside body_iterator.__anext__().
                await never_finish_send.wait()

        scope = {"type": "http", "method": "POST", "path": f"/v1/{endpoint}", "headers": []}
        await resp(scope, receive, send)

    async def run_all():
        for ep in ("responses", "chat", "completions"):
            await exercise_disconnect_mid_send(ep)

        # Also test direct task cancellation while awaiting inside __anext__().
        gate = asyncio.Event()

        class _SlowStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                gate.set()
                await asyncio.sleep(60)
                raise StopAsyncIteration

            async def close(self):
                await asyncio.sleep(0)
                closed_events.append("cancelled-anext")

        client.chat.completions.create.return_value = _SlowStream()
        resp = await router.chat_completions(
            ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")], stream=True),
            SimpleNamespace(headers={}),
        )
        task = asyncio.create_task(resp.body_iterator.__anext__())
        await gate.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_all())
    assert closed_events == ["responses", "chat", "completions", "cancelled-anext"]


# ---------------------------------------------------------------------------
# Audit #8: Proxy credential isolation & /health / /readiness authentication
# ---------------------------------------------------------------------------


def test_leak_fix_8_oai_proxies_separates_proxy_and_backend_keys_and_secures_health_readiness(monkeypatch):
    seen_headers: list[tuple[str, str | None, str | None]] = []

    class _BackendHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            seen_headers.append((self.path, self.headers.get("Authorization"), self.headers.get("X-API-Key")))
            if self.headers.get("Authorization") != "Bearer sk-upstream-secret":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"error":"unauthorized"}')
                return
            body = json.dumps({"ok": True, "path": self.path}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen_headers.append((self.path, self.headers.get("Authorization"), self.headers.get("X-API-Key")))
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.dumps({"executed": json.loads(raw)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), _BackendHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    models_list_calls = 0

    async def fake_models_list():
        nonlocal models_list_calls
        models_list_calls += 1
        return SimpleNamespace(data=[SimpleNamespace(id="m1")])

    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock()),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
        completions=SimpleNamespace(create=AsyncMock()),
        models=SimpleNamespace(list=AsyncMock(side_effect=fake_models_list)),
        close=AsyncMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=lambda **kw: client, APIError=Exception),
    )

    try:
        router = InferenceApiRouter(
            api_key="sk-upstream-secret",
            base_url=f"http://127.0.0.1:{port}/v1",
            require_api_key=True,
            proxy_api_key="sk-client-only",
        )
        with TestClient(router.app) as http:
            # 1. Client presents distinct proxy_api_key; backend receives ONLY sk-upstream-secret.
            for hdrs in ({"Authorization": "Bearer sk-client-only"}, {"X-API-Key": "sk-client-only"}):
                assert http.get("/metrics", headers=hdrs).status_code == 200
                assert http.get("/v1/tools", headers=hdrs).status_code == 200
                assert http.post("/v1/tools/execute", json={"name": "ping"}, headers=hdrs).status_code == 200

            assert len(seen_headers) == 6
            for _path, auth_hdr, x_key in seen_headers:
                assert auth_hdr == "Bearer sk-upstream-secret"
                assert x_key is None

            # 2. /health requires proxy auth and must NOT probe upstream when unauthenticated.
            assert http.get("/health").status_code == 401
            assert http.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
            assert models_list_calls == 0

            ok_health = http.get("/health", headers={"Authorization": "Bearer sk-client-only"})
            assert ok_health.status_code == 200
            assert ok_health.json()["models"]["count"] == 1
            assert models_list_calls == 1

            # 3. Unauthenticated /readiness with require_api_key=True must NOT call upstream or leak errors.
            unauth_ready = http.get("/readiness")
            assert unauth_ready.status_code == 200
            assert unauth_ready.json() == {"status": "ready"}
            assert models_list_calls == 1  # unchanged!

            # Invalid presented key on /readiness is rejected with 401.
            assert http.get("/readiness", headers={"X-API-Key": "wrong"}).status_code == 401
            assert models_list_calls == 1

            # Authenticated /readiness probes upstream and surfaces backend failure.
            client.models.list.side_effect = RuntimeError("secret-upstream-ip:5432 refused")
            auth_fail = http.get("/readiness", headers={"X-API-Key": "sk-client-only"})
            assert auth_fail.status_code == 503
            assert "secret-upstream-ip" in auth_fail.json().get("error", "")

        # 4. When require_api_key=False, unauthenticated /readiness strips backend error details.
        open_router = InferenceApiRouter(
            api_key="sk-upstream-secret",
            base_url=f"http://127.0.0.1:{port}/v1",
            require_api_key=False,
        )
        with TestClient(open_router.app) as open_http:
            open_fail = open_http.get("/readiness")
            assert open_fail.status_code == 503
            assert open_fail.json() == {"status": "not ready"}
            assert "error" not in open_fail.json()
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Audit #20: Concurrent cold-start / replacement aiohttp.ClientSession race
# ---------------------------------------------------------------------------


def test_leak_fix_20_oai_proxies_concurrent_session_init_no_leak(monkeypatch):
    created_sessions: list[object] = []

    class _FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self):
            return {"ok": True}

    class _FakeSession:
        def __init__(self, timeout=None):
            self.timeout = timeout
            self.closed = False
            created_sessions.append(self)

        def get(self, url, **_kwargs):
            return _FakeResp()

        async def close(self):
            await asyncio.sleep(0.01)
            self.closed = True

    class _FakeTimeout:
        def __init__(self, total=30.0):
            self.total = total

    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock()),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
        completions=SimpleNamespace(create=AsyncMock()),
        close=AsyncMock(),
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda **kw: client, APIError=Exception))
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientSession=_FakeSession, ClientTimeout=_FakeTimeout),
    )

    router = InferenceApiRouter(api_key="sk-test", base_url="http://127.0.0.1:8000/v1")

    async def run_concurrent():
        # 1. Cold start under 12 concurrent requests must create exactly ONE session.
        results = await asyncio.gather(
            *(router._request_backend_http("GET", "http://127.0.0.1:8000/metrics") for _ in range(12))
        )
        assert all(status == 200 and data == {"ok": True} for status, data, _ in results)
        assert len(created_sessions) == 1
        first = created_sessions[0]
        assert router._http_session is first and not first.closed

        # 2. Loop-mismatch replacement under 12 concurrent requests closes old session once & creates exactly ONE new.
        router._http_session_loop = object()
        results2 = await asyncio.gather(
            *(router._request_backend_http("GET", "http://127.0.0.1:8000/metrics") for _ in range(12))
        )
        assert all(status == 200 for status, _, _ in results2)
        assert first.closed is True
        assert len(created_sessions) == 2
        assert router._http_session is created_sessions[1]
        await router.aclose()
        assert created_sessions[1].closed is True

    asyncio.run(run_concurrent())


# ---------------------------------------------------------------------------
# Audit #21: AuthWorkerManager cleanup on AuthWorkerClient init failure + SIGKILL reap
# ---------------------------------------------------------------------------


def test_leak_fix_21_auth_worker_manager_init_failure_cleanup_and_sigkill_reap(tmp_path, monkeypatch):
    import os
    import subprocess

    from easydel.workers.esurge.auth import worker_manager as wm_mod

    # 1. AuthWorkerClient init failure must terminate+reap subprocess and unlink IPC socket file.
    mgr = wm_mod.AuthWorkerManager(ipc_dir=str(tmp_path), startup_timeout=5.0)
    spawned_procs: list[subprocess.Popen] = []
    created_paths: list[str] = []

    def fake_spawn(endpoint: str):
        path = endpoint[len("ipc://") :]
        created_paths.append(path)
        with open(path, "w") as f:
            f.write("")
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        spawned_procs.append(proc)
        return proc

    monkeypatch.setattr(mgr, "_spawn_auth_worker", fake_spawn)
    monkeypatch.setattr(
        wm_mod,
        "AuthWorkerClient",
        lambda _ep: (_ for _ in ()).throw(RuntimeError("client handshake failed")),
    )

    with pytest.raises(RuntimeError, match="client handshake failed"):
        mgr.start()

    assert len(spawned_procs) == 1
    assert spawned_procs[0].poll() is not None, "spawned worker must be terminated and reaped on client init failure"
    assert not os.path.exists(created_paths[0]), "IPC socket file must be unlinked on client init failure"
    assert mgr._auth_process is None
    assert mgr.auth_endpoint is None
    assert mgr._auth_owned is False

    # 2. _terminate_process must wait() after SIGKILL so stubborn child is never left a zombie.
    stubborn = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert stubborn.stdout is not None
    assert stubborn.stdout.readline().strip() == "ready"
    stubborn.stdout.close()

    real_wait = stubborn.wait
    calls = {"n": 0}

    def fast_timeout_wait(timeout=None):
        calls["n"] += 1
        if calls["n"] == 1 and timeout is not None:
            raise subprocess.TimeoutExpired(cmd="stubborn", timeout=timeout)
        return real_wait(timeout=timeout)

    monkeypatch.setattr(stubborn, "wait", fast_timeout_wait)
    mgr2 = wm_mod.AuthWorkerManager(ipc_dir=str(tmp_path))
    mgr2._auth_process = stubborn
    pid = stubborn.pid
    mgr2._terminate_process()
    assert mgr2._auth_process is None
    assert stubborn.returncode is not None
    # Child is already reaped: os.waitpid must raise ChildProcessError (never return (pid, status) zombie).
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


# ---------------------------------------------------------------------------
# Audit #22: Cross-tenant / admin info isolation on /metrics, /v1/models, /v1/tools, /health
# ---------------------------------------------------------------------------


def test_leak_fix_22_metrics_models_tools_health_tenant_and_admin_isolation(monkeypatch):
    from easydel.inference.esurge import metrics as metrics_mod
    from easydel.inference.esurge.serving import dashboard as dash_mod

    collector = metrics_mod.MetricsCollector(enable_detailed_logging=False)
    for rid, model_name, err in (
        ("req-secret-1", "model-secret", "secret-traceback"),
        ("req-a-1", "model-a", "tenant-a-error"),
    ):
        collector.start_request(rid, prompt_tokens=10, model_name=model_name)
        collector.complete_request(rid, finish_reason="stop", error=err)
    monkeypatch.setattr(metrics_mod, "_global_metrics_collector", collector)

    eng_a = _ScriptedEngine(name="model-a")
    eng_a.num_pending_requests = 2
    eng_a.num_running_requests = 1
    eng_secret = _ScriptedEngine(name="model-secret")
    eng_secret.num_pending_requests = 7
    eng_secret.num_running_requests = 3

    server = eSurgeApiServer(
        {"model-a": eng_a, "model-secret": eng_secret},
        require_api_key=True,
        enable_persistence=False,
        enable_cors=False,
    )
    admin_key, _ = server.auth_manager.generate_api_key(name="admin", role=ApiKeyRole.ADMIN)
    user_key, _ = server.auth_manager.generate_api_key(
        name="tenant-a-user",
        role=ApiKeyRole.USER,
        permissions=ApiKeyPermissions(allowed_models=["model-a"]),
    )
    readonly_key, _ = server.auth_manager.generate_api_key(
        name="tenant-a-ro",
        role=ApiKeyRole.READONLY,
        permissions=ApiKeyPermissions(allowed_models=["model-a"]),
    )

    with TestClient(server.app) as client:
        # 1. ADMIN sees auth_stats, recent_requests, and all loaded models.
        admin_hdrs = {"Authorization": f"Bearer {admin_key}"}
        r_admin_metrics = client.get("/metrics", headers=admin_hdrs)
        assert r_admin_metrics.status_code == 200
        am = r_admin_metrics.json()
        assert "auth_stats" in am
        assert set(am["engine_metrics"]) == {"model-a", "model-secret"}
        assert len(am["engine_metrics"]["model-secret"]["recent_requests"]) == 1
        assert am["engine_metrics"]["model-secret"]["recent_requests"][0]["request_id"] == "req-secret-1"

        # 2. USER and READONLY restricted to model-a MUST NOT see auth_stats, recent_requests, or model-secret.
        for key in (user_key, readonly_key):
            hdrs = {"Authorization": f"Bearer {key}"}

            r_metrics = client.get("/metrics", headers=hdrs)
            assert r_metrics.status_code == 200
            m = r_metrics.json()
            assert "auth_stats" not in m
            assert set(m["engine_metrics"]) == {"model-a"}
            assert "recent_requests" not in m["engine_metrics"]["model-a"]
            assert m["models_loaded"] == 1

            r_models = client.get("/v1/models", headers=hdrs)
            assert r_models.status_code == 200
            ml = r_models.json()
            assert [item["id"] for item in ml["data"]] == ["model-a"]
            assert ml["total"] == 1

            r_tools = client.get("/v1/tools", headers=hdrs)
            assert r_tools.status_code == 200
            assert set(r_tools.json()["models"]) == {"model-a"}

            r_health = client.get("/health", headers=hdrs)
            assert r_health.status_code == 200
            h = r_health.json()
            assert set(h["models"]) == {"model-a"}
            assert h["pending_requests"] == 2
            assert h["running_requests"] == 1

    # 3. Non-auth dev server keeps full metrics & model visibility intact.
    dev_server = eSurgeApiServer(
        {"model-a": eng_a, "model-secret": eng_secret},
        require_api_key=False,
        enable_persistence=False,
        enable_cors=False,
    )
    with TestClient(dev_server.app) as dev_client:
        dm = dev_client.get("/metrics").json()
        assert "auth_stats" in dm
        assert set(dm["engine_metrics"]) == {"model-a", "model-secret"}
        assert "recent_requests" in dm["engine_metrics"]["model-a"]
        assert {item["id"] for item in dev_client.get("/v1/models").json()["data"]} == {"model-a", "model-secret"}
        assert set(dev_client.get("/v1/tools").json()["models"]) == {"model-a", "model-secret"}
        assert set(dev_client.get("/health").json()["models"]) == {"model-a", "model-secret"}

    # 4. FrontMetricsMiddleware preserves redactions when merging peer front payloads.
    peer_payload = server.metrics_payload()
    for name in peer_payload["engine_metrics"]:
        peer_payload["engine_metrics"][name]["request_rows"] = collector.request_rows(name)

    async def fake_fetch_peer(_path: str):
        return peer_payload

    monkeypatch.setattr(dash_mod, "_fetch_peer", fake_fetch_peer)
    wrapped = dash_mod.FrontMetricsMiddleware(
        server.app,
        own_socket="/tmp/front-0.sock",
        sockets=["/tmp/front-0.sock", "/tmp/front-1.sock"],
        local_metrics=server.metrics_payload,
        request_rows=lambda name: collector.request_rows(name),
        window_seconds=300.0,
    )
    with TestClient(wrapped) as front_client:
        fm_user = front_client.get("/metrics", headers={"Authorization": f"Bearer {user_key}"}).json()
        assert "auth_stats" not in fm_user
        assert set(fm_user["engine_metrics"]) == {"model-a"}
        assert "recent_requests" not in fm_user["engine_metrics"]["model-a"]
        assert fm_user["models_loaded"] == 1

        fm_admin = front_client.get("/metrics", headers={"Authorization": f"Bearer {admin_key}"}).json()
        assert "auth_stats" in fm_admin
        assert set(fm_admin["engine_metrics"]) == {"model-a", "model-secret"}
        assert len(fm_admin["engine_metrics"]["model-secret"]["recent_requests"]) >= 1
