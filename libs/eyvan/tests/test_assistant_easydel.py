# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The assistant's EasyDeL tools through the real chat route and a scripted fake provider.

Proves: the model is told to ask for missing trainer arguments before drafting;
describe/validate answer from the shipped catalog; a draft whose arguments
eLarge would refuse or silently drop is not drafted; a valid one becomes a
runnable ``easydel.scripts.elarge`` command the user still has to confirm.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app

ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
ALICE = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "project_submitter", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
SCOPE = {"org_id": "acme", "project_id": "r"}
LAUNCH = {"model": "meta-llama/Llama-3.1-8B-Instruct", "dataset": "trl-lib/ultrafeedback_binarized"}


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


def test_the_prompt_and_tools_make_the_model_ask_before_drafting() -> None:
    provider = ScriptedProvider([{"role": "assistant", "content": "Which beta?"}])
    response = _chat(_client(provider), "I want to DPO an 8B model")
    assert response.status_code == 200, response.text
    sent = provider.requests[0]
    names = {tool["function"]["name"] for tool in sent["tools"]}
    assert {"list_easydel_trainers", "describe_easydel_trainer", "validate_easydel_arguments"} <= names
    system = sent["messages"][0]["content"]
    assert "ASK the user" in system and "Do not draft until they have answered" in system
    assert "beta" in system and "loss_type" in system


def test_describe_and_validate_answer_from_the_catalog() -> None:
    calls = [
        _call("list_easydel_trainers", {}, "c0"),
        _call("describe_easydel_trainer", {"trainer": "dpo", "query": "beta loss_type"}, "c1"),
        _call("validate_easydel_arguments", {"trainer": "dpo", "arguments": {"betta": 0.1, "loss_type": "x"}}, "c2"),
        _call("describe_easydel_trainer", {"trainer": "not-a-trainer"}, "c3"),
    ]
    provider = ScriptedProvider([{"role": "assistant", "content": None, "tool_calls": calls}])
    response = _chat(_client(provider), "what does DPO take?")
    assert response.status_code == 200, response.text
    results = provider.tool_results(1)
    assert {item["id"] for item in results["c0"]["trainers"]} >= {"dpo", "sft", "grpo"}
    fields = {item["name"]: item for item in results["c1"]["fields"]}
    assert fields["beta"]["default"] == 0.1 and fields["loss_type"]["default"] == "sigmoid"
    assert results["c2"]["ok"] is False and results["c2"]["suggestions"] == {"betta": "beta"}
    assert [item["field"] for item in results["c2"]["invalid_choices"]] == ["loss_type"]
    assert "error" in results["c3"]
    assert [item["ok"] for item in response.json()["tool_calls"]] == [True, True, True, False]


def test_an_easydel_draft_is_refused_until_its_arguments_are_valid_then_is_runnable() -> None:
    bad = {"name": "dpo-8b", "easydel_trainer": "dpo", "trainer_arguments": {"betta": 0.1}, "easydel_launch": LAUNCH}
    good = {
        "name": "dpo-8b",
        "easydel_trainer": "dpo",
        "trainer_arguments": {"beta": 0.1, "loss_type": "sigmoid", "learning_rate": 5e-7},
        "easydel_launch": LAUNCH,
        "accelerator_type": "v5p-8",
        "tpu_chips": 4,
    }
    provider = ScriptedProvider(
        [
            {"role": "assistant", "content": None, "tool_calls": [_call("draft_workload", bad, "c0")]},
            {"role": "assistant", "content": None, "tool_calls": [_call("draft_workload", good, "c1")]},
            {"role": "assistant", "content": "Drafted; confirm to submit."},
        ]
    )
    http = _client(provider)
    response = _chat(http, "DPO llama 8B with beta 0.1")
    assert response.status_code == 200, response.text
    refused = provider.tool_results(1)["c0"]
    assert refused["drafted"] is False and any("betta" in reason for reason in refused["blocking"])
    accepted = provider.tool_results(2)["c1"]
    assert accepted["drafted"] is True and accepted["submitted"] is False
    assert "trainer_type: dpo" in accepted["easydel"]["config_yaml"]

    drafts = response.json()["drafts"]
    assert len(drafts) == 1
    command = drafts[0]["spec"]["execution"]["command"]
    assert "python -m easydel.scripts.elarge --config easydel-launch.yaml" in command
    assert "beta: 0.1" in command and "meta-llama/Llama-3.1-8B-Instruct" in command
    assert drafts[0]["easydel"]["trainer"] == "dpo"
    # Nothing was submitted on the user's behalf.
    listed = http.get("/api/v1/workloads", params=SCOPE, headers={**ALICE, "X-Eyvan-Roles": "project_viewer"})
    assert listed.status_code == 200 and listed.json() == []

    # The draft is accepted by the real submit route once the user confirms it.
    submitted = http.post(
        "/api/v1/workloads",
        json={"org_id": drafts[0]["org_id"], "project_id": drafts[0]["project_id"], "spec": drafts[0]["spec"]},
        headers={**ALICE, "Idempotency-Key": "confirm-1"},
    )
    assert submitted.status_code == 201, submitted.text


def test_an_easydel_draft_declares_cpu_torch_for_the_hf_loader_and_a_user_pin_wins() -> None:
    """The live failure: eLarge loads HF checkpoints through torch and the job had none.

    The draft declares the CPU-only wheel for every supported worker Python
    (markers pick one on the host); a torch requirement the user lists replaces
    it, and other dependencies they list are kept.
    """

    easydel_src = "easydel @ git+https://github.com/erfanzar/EasyDeL@3362f13d1#subdirectory=libs/easydel"
    base = {
        "easydel_trainer": "sft",
        "trainer_arguments": {"learning_rate": 1e-4},
        "easydel_launch": {"model": "hf-internal-testing/tiny-random-LlamaForCausalLM", "dataset": "d.jsonl"},
    }
    plain = {"name": "sft-default", **base, "dependencies": ["jax[tpu]", easydel_src]}
    pinned = {"name": "sft-pinned", **base, "dependencies": ["Torch==2.9.0", easydel_src]}
    provider = ScriptedProvider(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("draft_workload", plain, "c0"), _call("draft_workload", pinned, "c1")],
            },
            {"role": "assistant", "content": "Drafted."},
        ]
    )
    http = _client(provider)
    response = _chat(http, "SFT a tiny llama")
    assert response.status_code == 200, response.text
    drafts = {draft["spec"]["name"]: draft for draft in response.json()["drafts"]}

    declared = drafts["sft-default"]["spec"]["dependencies"]
    assert declared[:2] == ["jax[tpu]", easydel_src]
    torch = [item for item in declared if item.startswith("torch @ ")]
    assert len(torch) == 6
    assert all("download.pytorch.org/whl/cpu/" in item and "%2Bcpu-" in item for item in torch)
    assert any('python_version == "3.13" and platform_machine == "x86_64"' in item for item in torch)
    assert any("CPU-only torch" in note for note in drafts["sft-default"]["easydel"]["assumptions"])

    kept = drafts["sft-pinned"]["spec"]["dependencies"]
    assert kept == ["Torch==2.9.0", easydel_src]

    # The marker requirements survive the real submit route's validation.
    submitted = http.post(
        "/api/v1/workloads",
        json={**SCOPE, "spec": drafts["sft-default"]["spec"]},
        headers={**ALICE, "Idempotency-Key": "confirm-torch"},
    )
    assert submitted.status_code == 201, submitted.text
    assert submitted.json()["spec"]["dependencies"] == declared


def test_a_plain_command_draft_gets_no_torch_and_bad_dependencies_are_refused() -> None:
    calls = [
        _call("draft_workload", {"name": "echo", "command": "echo hi", "dependencies": ["numpy"]}, "c0"),
        _call("draft_workload", {"name": "bad", "command": "echo hi", "dependencies": "numpy"}, "c1"),
    ]
    provider = ScriptedProvider([{"role": "assistant", "content": None, "tool_calls": calls}])
    response = _chat(_client(provider), "draft")
    assert response.status_code == 200, response.text
    assert [item["ok"] for item in response.json()["tool_calls"]] == [True, False]
    (draft,) = response.json()["drafts"]
    assert draft["spec"]["dependencies"] == ["numpy"]


def test_draft_needs_command_or_trainer_but_not_both() -> None:
    calls = [
        _call("draft_workload", {"name": "x"}, "c0"),
        _call(
            "draft_workload", {"name": "x", "command": "echo", "easydel_trainer": "sft", "easydel_launch": LAUNCH}, "c1"
        ),
        _call("draft_workload", {"name": "x", "easydel_trainer": "sft", "easydel_launch": {"model": "m"}}, "c2"),
    ]
    provider = ScriptedProvider([{"role": "assistant", "content": None, "tool_calls": calls}])
    response = _chat(_client(provider), "draft")
    assert response.status_code == 200, response.text
    assert [item["ok"] for item in response.json()["tool_calls"]] == [False, False, False]
    assert response.json()["drafts"] == []
