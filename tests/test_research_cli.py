# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan research` against a real in-process plane; only the network hop is replaced."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.sdk import cli
from eyvan.sdk.commands.http import EyvanApiError
from eyvan.server.api.app import create_app

PLANE = "http://127.0.0.1:8080"
ALICE = ["--user", "alice", "--role", "project_operator", "--org", "acme", "--project", "r"]
BOB = ["--user", "bob", "--role", "project_operator", "--org", "acme", "--project", "r"]
SPEC = {"name": "probe", "execution": {"kind": "command", "command": "echo hi"}}
PLAN = {
    "title": "lr sweep",
    "goal": "find a learning rate",
    "jobs": [{"name": "lr-1e-4", "spec": {**SPEC, "name": "lr-1e-4"}}, {"name": "lr-3e-4", "spec": SPEC}],
    "budget": {"max_jobs": 4, "max_chip_hours": 10, "max_wall_seconds": 3600},
}


@pytest.fixture
def plane(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    for name in ("EYVAN_URL", "EYVAN_TOKEN", "EYVAN_ORG", "EYVAN_PROJECT", "EYVAN_DEV_USER", "EYVAN_ROLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "discover_node_context", lambda *a, **k: None)
    monkeypatch.setattr(cli, "discover_node_token_or_warn", lambda *a, **k: None)
    settings = EyvanSettings(dev_auth_enabled=True, controller_enabled=False, artifact_token_key="k" * 32)
    app = create_app(settings=settings)

    class ForwardingClient:
        def __init__(self, url: str, *, headers) -> None:
            self.headers = dict(headers)
            self.http = TestClient(app, base_url=url)

        def request(self, method, path, *, params=None, json=None, headers=None, content=None):
            response = self.http.request(
                method, path, params=params, json=json, headers={**self.headers, **(headers or {})}
            )
            if response.status_code >= 400:
                raise EyvanApiError(response.status_code, response.text)
            return response.json()

    monkeypatch.setattr(cli, "EyvanClient", ForwardingClient)
    return SimpleNamespace(app=app, tmp_path=tmp_path)


def _propose(plane, runner: CliRunner, *, suffix: str = ".yaml") -> dict:
    path = plane.tmp_path / f"plan{suffix}"
    path.write_text(json.dumps(PLAN) if suffix == ".json" else yaml.safe_dump(PLAN))
    result = runner.invoke(cli.main, [*ALICE, "research", "propose", "--file", str(path), "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_propose_list_show_approve_stop_and_watch(plane, suffix: str) -> None:
    runner = CliRunner()
    plan = _propose(plane, runner, suffix=suffix)
    assert plan["state"] == "proposed" and plan["subject"] == "alice"
    assert (plan["org_id"], plan["project_id"]) == ("acme", "r") and plan["jobs_planned"] == 2

    listed = runner.invoke(cli.main, [*ALICE, "research", "list"])
    assert listed.exit_code == 0 and plan["id"] in listed.output and "lr sweep" in listed.output
    assert "jobs 0/4" in listed.output
    as_json = runner.invoke(cli.main, [*ALICE, "research", "list", "--json"])
    assert [item["id"] for item in json.loads(as_json.output)["plans"]] == [plan["id"]]

    approved = runner.invoke(cli.main, [*ALICE, "research", "approve", plan["id"]])
    assert approved.exit_code == 0, approved.output
    assert f"approved {plan['id']}" in approved.output

    shown = runner.invoke(cli.main, [*ALICE, "research", "show", plan["id"]])
    assert shown.exit_code == 0, shown.output
    assert "approved" in shown.output and "lr-1e-4" in shown.output and "not_submitted" in shown.output

    # Not terminal yet: a zero timeout stops watching without touching the plan.
    gave_up = runner.invoke(cli.main, [*ALICE, "research", "watch", plan["id"], "--timeout", "0"])
    assert gave_up.exit_code == 1 and "still approved" in gave_up.output

    stopped = runner.invoke(cli.main, [*ALICE, "research", "stop", plan["id"], "--reason", "enough"])
    assert stopped.exit_code == 0, stopped.output
    assert f"{plan['id']} is stopped" in stopped.output

    watched = runner.invoke(cli.main, [*ALICE, "research", "watch", plan["id"], "--interval", "0.1"])
    assert watched.exit_code == 0, watched.output
    assert "[stopped]" in watched.output and "stop requested by alice: enough" in watched.output
    final = runner.invoke(cli.main, [*ALICE, "research", "watch", plan["id"], "--json"])
    assert json.loads(final.output)["stop_reason"] == "enough"


def test_another_user_sees_nothing_and_cannot_act(plane) -> None:
    runner = CliRunner()
    plan = _propose(plane, runner)
    assert "no research plans" in runner.invoke(cli.main, [*BOB, "research", "list"]).output
    for command in (["show", plan["id"]], ["approve", plan["id"]], ["stop", plan["id"]]):
        result = runner.invoke(cli.main, [*BOB, "research", *command])
        assert result.exit_code != 0 and "404" in result.output + str(result.exception)


def test_propose_refuses_an_unreadable_or_over_budget_file(plane) -> None:
    runner = CliRunner()
    bad = plane.tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    result = runner.invoke(cli.main, [*ALICE, "research", "propose", "--file", str(bad)])
    assert result.exit_code != 0 and "must be a mapping" in result.output
    over = plane.tmp_path / "over.yaml"
    over.write_text(yaml.safe_dump({**PLAN, "budget": {**PLAN["budget"], "max_jobs": 1}}))
    refused = runner.invoke(cli.main, [*ALICE, "research", "propose", "--file", str(over)])
    assert refused.exit_code != 0 and "max_jobs=1" in refused.output + str(refused.exception)
