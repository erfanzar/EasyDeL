# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Selection-specific start instructions retain the selected plane's identity."""

from __future__ import annotations

import ast
import io
import subprocess

import pytest

from eyvan.artifacts import LocalArtifactStore
from eyvan.sdk import execution
from eyvan.sdk.commands.start import LOCAL_URL, python_snippet, start_flow
from eyvan.sdk.contexts import EyvanContext
from eyvan.sdk.terminal import terminal_console

REMOTE_URL = "http://192.0.2.10:8080"


class OfflineSubmitter:
    def submit(self, *_args, **_kwargs):
        pytest.fail("endpoint resolution must not submit a callable")


@pytest.mark.parametrize(
    ("selected_url", "defaults"),
    [(LOCAL_URL, "environment"), (LOCAL_URL, "context"), (LOCAL_URL, "none"), (REMOTE_URL, "local-environment")],
)
def test_selected_snippet_resolves_to_selected_endpoint(selected_url, defaults, tmp_path, monkeypatch) -> None:
    active_context = (
        EyvanContext(fleet="research", org_id="acme", project_id="models", base_url=REMOTE_URL)
        if defaults == "context"
        else None
    )
    environ = {}
    if defaults == "environment":
        environ["EYVAN_URL"] = REMOTE_URL
    elif defaults == "local-environment":
        environ["EYVAN_URL"] = LOCAL_URL

    def unexpected_io(*_args, **_kwargs):
        pytest.fail("snippet endpoint resolution must not use a transport, daemon, or private profile")

    monkeypatch.setattr(execution, "_context", None)
    monkeypatch.setattr(execution, "get_active_context", lambda _path: active_context)
    monkeypatch.setattr(execution, "discover_node_context", lambda _environment: None)
    monkeypatch.setattr(execution, "_configured_gcp_runtime_profile", lambda _endpoint: None)
    monkeypatch.setattr(execution, "_prepared_gcp_runtime_profile", unexpected_io)
    monkeypatch.setattr(execution, "ensure_loopback_control_plane", unexpected_io)
    monkeypatch.setattr("eyvan.sdk.client.sync.EyvanClient", unexpected_io)
    snippet = python_snippet(selected_url, None)
    call = next(
        node.value
        for node in ast.parse(snippet).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "init"
    )
    args = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    context = execution.init(
        *args,
        **kwargs,
        environ=environ,
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        submitter=OfflineSubmitter(),
    )

    assert context.base_url == selected_url
    assert environ == (
        {"EYVAN_URL": REMOTE_URL}
        if defaults == "environment"
        else {"EYVAN_URL": LOCAL_URL}
        if defaults == "local-environment"
        else {}
    )


def test_selected_local_flow_renders_explicit_endpoint_without_changing_environment(monkeypatch) -> None:
    monkeypatch.setenv("EYVAN_URL", REMOTE_URL)
    stream = io.StringIO()

    def run(args, *, stdin_text, timeout):
        return subprocess.CompletedProcess(list(args), 0, stdout="[]", stderr="")

    outcome = start_flow(
        terminal_console(stream),
        project="selected-project",
        runner=run,
        probe=lambda _url: True,
        choose=lambda _count: 1,
    )

    assert outcome.targets[0].url == LOCAL_URL
    assert f'eyvan.init("{LOCAL_URL}")' in stream.getvalue()
    import os

    assert os.environ["EYVAN_URL"] == REMOTE_URL


@pytest.mark.parametrize("project", ["selected-project", "selected project; echo nope"])
def test_stopped_head_recovery_keeps_discovery_identity(project) -> None:
    import shlex
    from dataclasses import replace

    from eyvan.sdk.commands.start import _next_steps

    stream = io.StringIO()
    console = terminal_console(stream)
    console.width = 240
    issued = []

    def run(args, *, stdin_text, timeout):
        issued.append(tuple(args))
        return subprocess.CompletedProcess(
            list(args),
            0,
            stdout='[{"name": "eyvan-head-research", "status": "TERMINATED", '
            '"zone": "https://x/zones/us-east5-a", '
            '"labels": {"eyvan-managed": "control-plane-head", "eyvan-head-fleet": "research"}}]',
            stderr="",
        )

    outcome = start_flow(
        console,
        project=project,
        runner=run,
        probe=lambda _url: False,
        choose=lambda _count: 1,
    )

    assert len(issued) == 1
    assert issued[0][:3] == ("compute", "instances", "list")
    assert issued[0][issued[0].index("--project") + 1] == project
    expected_start = [
        "gcloud",
        "compute",
        "instances",
        "start",
        "eyvan-head-research",
        "--project",
        project,
        "--zone",
        "us-east5-a",
    ]
    expected_reconnect = ["eyvan", "start", "--project", project]
    assert shlex.join(expected_start) in stream.getvalue()
    assert shlex.join(expected_reconnect) in stream.getvalue()
    # Display labels are not resource identity and must never be parsed to act.
    steps = _next_steps(replace(outcome.targets[0], label="a different display label"))
    assert shlex.split(steps[0][0]) == expected_start
    assert shlex.split(steps[1][0]) == expected_reconnect


def test_unscoped_head_target_does_not_offer_ambiguous_start_command() -> None:
    from eyvan.sdk.commands.start import PlaneTarget, _next_steps

    target = PlaneTarget(kind="head", label="research — guessed-name", detail="stopped", reachable=False)

    assert all(not command.startswith("gcloud") for command, _why in _next_steps(target))
