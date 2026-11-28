# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Streaming assistant turns and saved conversations.

Proves: an OpenAI-compatible SSE provider's text and tool-call deltas become
live ``step``/``text``/``tool_call``/``tool_result``/``draft``/``done`` events
in order; a non-streaming provider still works; the turn and its trace are saved
to the caller's conversation and later turns send the saved history; another
user cannot see, read, rename or delete the conversation; refusals happen
before the stream starts; a provider failure mid-turn ends with an ``error``
event that history records; conversations persist in every backend.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import AssistantMessageRecord, Principal, Role, RoleGrant
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import NotFoundError

KEY = "sk-stream-secret-0123456789abcdef"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
ALICE = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "project_submitter", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
BOB = {"X-Eyvan-User": "bob", "X-Eyvan-Roles": "project_submitter", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
SCOPE = {"org_id": "acme", "project_id": "r"}


def _sse(chunks: list[dict[str, tp.Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _delta(**delta: tp.Any) -> dict[str, tp.Any]:
    return {"choices": [{"index": 0, "delta": delta}]}


class StreamingProvider:
    """Streams scripted rounds over SSE and records every request body."""

    def __init__(self, rounds: list[httpx.Response | list[dict[str, tp.Any]]]) -> None:
        self.rounds = list(rounds)
        self.bodies: list[dict[str, tp.Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        answer = self.rounds.pop(0) if self.rounds else [_delta(content="done")]
        return answer if isinstance(answer, httpx.Response) else _sse(answer)


def _tool_round(name: str, arguments: dict[str, tp.Any], *, narration: str = "") -> list[dict[str, tp.Any]]:
    raw = json.dumps(arguments)
    half = len(raw) // 2
    chunks = [_delta(content=narration)] if narration else []
    return [
        *chunks,
        _delta(
            tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": name, "arguments": ""}}]
        ),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": raw[:half]}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": raw[half:]}}]),
    ]


def _http(provider: StreamingProvider) -> TestClient:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, team_mode=True, artifact_token_key="k" * 32))
    app.state.assistant_transport = httpx.MockTransport(provider)
    http = TestClient(app, base_url="http://127.0.0.1:8080")
    created = http.post(
        "/api/v1/assistant/providers",
        json={"name": "s", "base_url": "http://stream.test/v1", "api_key": KEY, "models": ["m1"]},
        headers=ADMIN,
    )
    assert created.status_code == 201, created.text
    return http


def _stream(http: TestClient, message: str, headers: dict[str, str] = ALICE, **extra: tp.Any) -> list[dict[str, tp.Any]]:
    with http.stream(
        "POST", "/api/v1/assistant/chat/stream", json={"message": message, **SCOPE, **extra}, headers=headers
    ) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in response.iter_lines() if line.strip()]


def test_a_turn_streams_steps_tool_calls_and_text_in_order() -> None:
    provider = StreamingProvider(
        [
            _tool_round("size_job", {"parameters": 8e9, "trainer": "sft"}, narration="Let me size that. "),
            [_delta(content="You need "), _delta(content="**a v5p-8**.")],
        ]
    )
    http = _http(provider)
    events = _stream(http, "How much TPU for an 8B SFT?")
    kinds = [event["type"] for event in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert kinds == ["start", "step", "text", "tool_call", "tool_result", "step", "text", "text", "done"]
    call = events[3]
    assert call["name"] == "size_job" and call["arguments"] == {"parameters": 8e9, "trainer": "sft"}
    assert events[4]["ok"] is True and events[4]["summary"].startswith('{"parameters": 8000000000')
    assert len(events[4]["summary"]) <= 401  # previews are bounded; the model still gets the whole result
    assert events[1] == {"type": "step", "step": 1, "max_steps": 6}
    done = events[-1]
    assert done["reply"] == "You need **a v5p-8**." and done["steps"] == 2
    assert done["tool_calls"] == [{"tool": "size_job", "ok": True}] and done["message_id"].startswith("msg_")
    assert provider.bodies[0]["stream"] is True and provider.bodies[0]["tools"]
    assert provider.bodies[1]["messages"][-1]["role"] == "tool"
    assert KEY not in json.dumps(events)


def test_turns_are_saved_and_later_turns_send_the_saved_history() -> None:
    provider = StreamingProvider([[_delta(content="first answer")], [_delta(content="second answer")]])
    http = _http(provider)
    first = _stream(http, "first question")
    conversation_id = first[0]["conversation_id"]
    assert first[0]["title"] == "first question"
    second = _stream(http, "second question", conversation_id=conversation_id)
    assert second[0]["conversation_id"] == conversation_id
    sent = [(item["role"], item["content"]) for item in provider.bodies[1]["messages"][1:]]
    assert sent == [("user", "first question"), ("assistant", "first answer"), ("user", "second question")]

    listed = http.get("/api/v1/assistant/conversations", headers=ALICE).json()["conversations"]
    assert [item["id"] for item in listed] == [conversation_id] and listed[0]["message_count"] == 4
    full = http.get(f"/api/v1/assistant/conversations/{conversation_id}", headers=ALICE).json()
    assert [(item["role"], item["content"]) for item in full["messages"]] == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ]
    assert full["messages"][1]["model"] == "m1"


def test_a_saved_turn_keeps_its_trace_and_drafts() -> None:
    provider = StreamingProvider(
        [
            _tool_round("draft_workload", {"name": "sft", "command": "python train.py", "accelerator_type": "v5p-8"}),
            [_delta(content="Drafted it.")],
        ]
    )
    http = _http(provider)
    events = _stream(http, "draft my job")
    drafts = [event for event in events if event["type"] == "draft"]
    assert len(drafts) == 1 and drafts[0]["draft"]["spec"]["name"] == "sft"
    conversation_id = events[0]["conversation_id"]
    answer = http.get(f"/api/v1/assistant/conversations/{conversation_id}", headers=ALICE).json()["messages"][-1]
    assert answer["trace"][0]["name"] == "draft_workload" and answer["trace"][0]["ok"] is True
    assert answer["drafts"][0]["spec"]["execution"]["command"] == "python train.py"


def test_conversations_are_owner_only() -> None:
    http = _http(StreamingProvider([[_delta(content="hi")]]))
    conversation_id = _stream(http, "mine")[0]["conversation_id"]
    path = f"/api/v1/assistant/conversations/{conversation_id}"
    assert http.get("/api/v1/assistant/conversations", headers=BOB).json() == {"conversations": []}
    assert http.get(path, headers=BOB).status_code == 404
    assert http.put(path, json={"title": "stolen"}, headers=BOB).status_code == 404
    assert http.delete(path, headers=BOB).status_code == 404
    assert http.get(path, headers=ADMIN).status_code == 404  # not even a fleet admin
    continued = http.post(
        "/api/v1/assistant/chat/stream", json={"message": "x", "conversation_id": conversation_id, **SCOPE}, headers=BOB
    )
    assert continued.status_code == 404

    renamed = http.put(path, json={"title": "  Sizing   chat "}, headers=ALICE)
    assert renamed.status_code == 200 and renamed.json()["title"] == "Sizing chat"
    assert http.delete(path, headers=ALICE).json() == {"deleted": conversation_id}
    assert http.get(path, headers=ALICE).status_code == 404


def test_refusals_happen_before_the_stream_and_failures_end_it_with_an_error() -> None:
    provider = StreamingProvider([httpx.Response(500, text=f"upstream broke {KEY}")])
    http = _http(provider)
    bad_model = http.post(
        "/api/v1/assistant/chat/stream", json={"message": "x", "model": "nope", **SCOPE}, headers=ALICE
    )
    assert bad_model.status_code == 400
    unknown = http.post(
        "/api/v1/assistant/chat/stream",
        json={"message": "x", "conversation_id": "conv_0000000000000000", **SCOPE},
        headers=ALICE,
    )
    assert unknown.status_code == 404

    events = _stream(http, "will fail")
    assert [event["type"] for event in events] == ["start", "step", "error"]
    assert "HTTP 500" in events[-1]["detail"] and KEY not in events[-1]["detail"]
    saved = http.get(f"/api/v1/assistant/conversations/{events[0]['conversation_id']}", headers=ALICE).json()
    assert saved["messages"][-1]["error"].startswith("the assistant provider answered HTTP 500")


def test_a_provider_that_does_not_stream_still_answers() -> None:
    plain = httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "plain reply"}}]})
    http = _http(StreamingProvider([plain]))
    events = _stream(http, "hello")
    assert [event["type"] for event in events] == ["start", "step", "text", "done"]
    assert events[-1]["reply"] == "plain reply"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "local", "sql"])
async def test_conversations_persist_in_every_backend(backend: str, tmp_path) -> None:
    engine = None
    if backend == "memory":
        repository: tp.Any = InMemoryRepository()
    elif backend == "local":
        repository = LocalJsonRepository(tmp_path / "control-plane.json")
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'plane.db'}")
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
    await repository.open()
    alice = Principal(subject="alice", grants=(RoleGrant(role=Role.PROJECT_SUBMITTER, org_id="acme", project_id="r"),))
    bob = Principal(subject="bob", grants=(RoleGrant(role=Role.PROJECT_SUBMITTER, org_id="acme", project_id="r"),))
    service = ControlPlaneService(repository)
    record = await service.start_assistant_conversation(
        alice, first_message="size an 8B model", org_id="acme", project_id="r", provider_id=None, model=None
    )
    await service.append_assistant_messages(
        alice,
        record.id,
        [
            AssistantMessageRecord(id="msg_0000000000000001", role="user", content="size an 8B model"),
            AssistantMessageRecord(
                id="msg_0000000000000002",
                role="assistant",
                content="v5p-8",
                trace=({"type": "tool", "name": "size_job"},),
            ),
        ],
    )
    with pytest.raises(NotFoundError):
        await service.get_assistant_conversation(bob, record.id)
    await repository.close()

    if backend == "memory":
        reopened = repository
    elif backend == "local":
        reopened = LocalJsonRepository(tmp_path / "control-plane.json")
        assert json.loads((tmp_path / "control-plane.json").read_text())["version"] == 10
    else:
        reopened = SqlAlchemyRepository(engine)
    await reopened.open()
    again = await ControlPlaneService(reopened).get_assistant_conversation(alice, record.id)
    assert [item.content for item in again.messages] == ["size an 8B model", "v5p-8"]
    assert again.messages[1].trace[0]["name"] == "size_job" and again.title == "size an 8B model"
    assert await ControlPlaneService(reopened).list_assistant_conversations(bob) == []
    await reopened.close()
    if engine is not None:
        await engine.dispose()
