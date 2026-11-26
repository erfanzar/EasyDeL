# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Research plans proposed by the assistant and driven over MCP.

Proves: the assistant can only *propose* a plan (nothing runs until the owner
approves); jobs given as draft fields get the same validation as a single
draft, EasyDeL included; the proposal travels in its own ``plans`` channel,
never as a workload draft, and survives in the saved conversation; over MCP
the user's own agent can approve and stop their plan, and nobody else's.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app

ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
ALICE = {
    "X-Eyvan-User": "alice",
    "X-Eyvan-Roles": "project_submitter,project_operator,project_viewer",
    "X-Eyvan-Org": "acme",
    "X-Eyvan-Project": "r",
}
BOB = {**ALICE, "X-Eyvan-User": "bob"}
SCOPE = {"org_id": "acme", "project_id": "r"}
BUDGET = {"max_jobs": 4, "max_chip_hours": 10, "max_wall_seconds": 3600}
LAUNCH = {"model": "meta-llama/Llama-3.1-8B-Instruct", "dataset": "trl-lib/ultrafeedback_binarized"}
ACCEPT = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def _call(name: str, arguments: dict[str, tp.Any], call_id: str) -> dict[str, tp.Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class ScriptedProvider:
    def __init__(self, script: list[dict[str, tp.Any]]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, tp.Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        message = self.script.pop(0) if self.script else {"role": "assistant", "content": "done"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    def tool_results(self, index: int) -> dict[str, tp.Any]:
        return {
            item["tool_call_id"]: json.loads(item["content"])
            for item in self.requests[index]["messages"]
            if item["role"] == "tool"
        }


def _client(provider: ScriptedProvider) -> TestClient:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32))
    app.state.assistant_transport = httpx.MockTransport(provider)
    http = TestClient(app, base_url="http://127.0.0.1:8080")
    created = http.post(
        "/api/v1/assistant/providers",
        json={"name": "test", "base_url": "http://provider.test/v1", "models": ["m"], "max_steps": 6},
        headers=ADMIN,
    )
    assert created.status_code == 201, created.text
    return http


def _chat(http: TestClient, text: str) -> httpx.Response:
    return http.post(
        "/api/v1/assistant/chat", json={"messages": [{"role": "user", "content": text}], **SCOPE}, headers=ALICE
    )


def _plans(http: TestClient, headers: dict[str, str] = ALICE) -> list[dict[str, tp.Any]]:
    response = http.get("/api/v1/research/plans", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["plans"]


def test_the_assistant_proposes_a_plan_that_waits_for_approval() -> None:
    jobs = [
        {"name": "lr-1e-6", "command": "echo lr=1e-6"},
        {"name": "lr-5e-7", "spec": {"name": "lr-5e-7", "execution": {"kind": "command", "command": "echo x"}}},
        {
            "name": "dpo-8b",
            "easydel_trainer": "dpo",
            "trainer_arguments": {"beta": 0.1, "loss_type": "sigmoid"},
            "easydel_launch": LAUNCH,
        },
    ]
    provider = ScriptedProvider(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call("propose_research_plan", {"title": "lr sweep", "jobs": jobs, "budget": BUDGET}, "c0")
                ],
            },
            {"role": "assistant", "content": "Proposed; approve it to start."},
        ]
    )
    http = _client(provider)
    response = _chat(http, "run these three jobs and auto-research on them")
    assert response.status_code == 200, response.text
    sent = provider.requests[0]
    assert {"propose_research_plan", "list_research_plans", "get_research_plan"} <= {
        tool["function"]["name"] for tool in sent["tools"]
    }
    assert "propose_research_plan" in sent["messages"][0]["content"]

    result = provider.tool_results(1)["c0"]
    assert result["proposed"] is True and result["approved"] is False and result["jobs"] == 3
    body = response.json()
    # Its own channel: never a workload draft the clients would offer to submit.
    assert body["drafts"] == []
    assert [plan["id"] for plan in body["plans"]] == [result["plan_id"]]

    [stored] = _plans(http)
    assert stored["id"] == result["plan_id"] and stored["state"] == "proposed" and stored["subject"] == "alice"
    detail = http.get(f"/api/v1/research/plans/{stored['id']}", headers=ALICE).json()
    commands = [job["spec"]["execution"]["command"] for job in detail["jobs"]]
    assert "python -m easydel.scripts.elarge" in commands[2] and "beta: 0.1" in commands[2]
    assert all(job["state"] == "not_submitted" for job in detail["jobs"])
    # Nothing was submitted on the user's behalf.
    assert http.get("/api/v1/workloads", params=SCOPE, headers=ALICE).json() == []


def test_a_job_the_draft_validation_refuses_proposes_nothing() -> None:
    bad = {"name": "dpo", "easydel_trainer": "dpo", "trainer_arguments": {"betta": 0.1}, "easydel_launch": LAUNCH}
    provider = ScriptedProvider(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call("propose_research_plan", {"title": "x", "jobs": [bad], "budget": BUDGET}, "c0"),
                    _call("propose_research_plan", {"title": "x", "jobs": [], "budget": BUDGET}, "c1"),
                    _call(
                        "propose_research_plan",
                        {"title": "x", "jobs": [{"name": "a", "command": "echo"}] * 5, "budget": BUDGET},
                        "c2",
                    ),
                ],
            }
        ]
    )
    http = _client(provider)
    response = _chat(http, "plan it")
    assert response.status_code == 200, response.text
    results = provider.tool_results(1)
    assert results["c0"]["proposed"] is False and results["c0"]["suggestions"] == {"betta": "beta"}
    assert [item["ok"] for item in response.json()["tool_calls"]] == [True, False, False]
    assert response.json()["plans"] == [] and _plans(http) == []


def test_a_streamed_proposal_is_an_event_and_is_kept_in_the_saved_chat() -> None:
    def sse(chunks: list[dict[str, tp.Any]]) -> httpx.Response:
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    arguments = json.dumps({"title": "sweep", "jobs": [{"name": "a", "command": "echo a"}], "budget": BUDGET})
    rounds = [
        sse(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "propose_research_plan", "arguments": arguments},
                                    }
                                ]
                            },
                        }
                    ]
                }
            ]
        ),
        sse([{"choices": [{"index": 0, "delta": {"content": "Approve it to start."}}]}]),
    ]
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32))
    app.state.assistant_transport = httpx.MockTransport(lambda request: rounds.pop(0))
    http = TestClient(app, base_url="http://127.0.0.1:8080")
    created = http.post(
        "/api/v1/assistant/providers",
        json={"name": "s", "base_url": "http://stream.test/v1", "models": ["m1"]},
        headers=ADMIN,
    )
    assert created.status_code == 201, created.text
    with http.stream(
        "POST", "/api/v1/assistant/chat/stream", json={"message": "sweep", **SCOPE}, headers=ALICE
    ) as response:
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]
    plan_events = [event for event in events if event["type"] == "plan"]
    done = events[-1]
    assert done["type"] == "done" and len(plan_events) == 1
    plan_id = plan_events[0]["plan"]["id"]
    assert [item["id"] for item in done["plans"]] == [plan_id] and done["drafts"] == []

    conversation_id = events[0]["conversation_id"]
    saved = http.get(f"/api/v1/assistant/conversations/{conversation_id}", headers=ALICE).json()
    answer = saved["messages"][-1]
    assert answer["role"] == "assistant" and [item["id"] for item in answer["plans"]] == [plan_id]
    # A turn that proposed nothing keeps its old shape (no empty "plans" key).
    assert "plans" not in saved["messages"][0]


def _mcp(http: TestClient, name: str, arguments: dict[str, tp.Any], headers: dict[str, str]) -> tuple[bool, tp.Any]:
    response = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        headers={**ACCEPT, **headers, "MCP-Protocol-Version": "2025-06-18"},
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    return result["isError"], json.loads(result["content"][0]["text"])


def test_over_mcp_the_owner_proposes_approves_and_stops_and_nobody_else_can() -> None:
    http = _client(ScriptedProvider([]))
    listed = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        headers={**ACCEPT, **ALICE},
    ).json()["result"]["tools"]
    names = {tool["name"] for tool in listed}
    assert {"propose_research_plan", "approve_research_plan", "stop_research_plan", "get_research_plan"} <= names

    failed, proposed = _mcp(
        http,
        "propose_research_plan",
        {"title": "mcp sweep", "jobs": [{"name": "a", "command": "echo a"}], "budget": BUDGET},
        ALICE,
    )
    assert failed is False and proposed["proposed"] is True
    plan_id = proposed["plan_id"]

    failed, refused = _mcp(http, "approve_research_plan", {"plan_id": plan_id}, BOB)
    assert failed is True and refused["status"] == 404
    failed, approved = _mcp(http, "approve_research_plan", {"plan_id": plan_id}, ALICE)
    assert failed is False and approved["state"] == "approved" and approved["approved_by"] == "alice"

    failed, refused = _mcp(http, "stop_research_plan", {"plan_id": plan_id}, BOB)
    assert failed is True and refused["status"] == 404
    failed, stopped = _mcp(http, "stop_research_plan", {"plan_id": plan_id, "reason": "enough"}, ALICE)
    assert failed is False and stopped["state"] in {"stopped", "stopping"}
    assert stopped["stop_reason"] == "enough"
