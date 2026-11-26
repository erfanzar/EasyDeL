# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discovery timeouts retain the first-run console's local choices."""

from __future__ import annotations

import importlib
import io
import json
import subprocess
from collections.abc import Sequence

import pytest
from click.testing import CliRunner

from eyvan.sdk.commands.head import HeadCommandError, HeadStep
from eyvan.sdk.commands.head_discovery import configured_project, discover_heads
from eyvan.sdk.commands.start import LOCAL_URL, discover, start_flow
from eyvan.sdk.terminal import terminal_console

_TIMEOUT_PRIVATE_DETAIL = "private-timeout-output-do-not-render"


def _timeout_runner(args: Sequence[str], *, stdin_text: str | None, timeout: float) -> subprocess.CompletedProcess[str]:
    assert stdin_text is None
    assert tuple(args[:3]) in {("config", "get-value", "project"), ("compute", "instances", "list")}
    raise subprocess.TimeoutExpired(
        ["gcloud", *args], timeout, output=_TIMEOUT_PRIVATE_DETAIL, stderr=_TIMEOUT_PRIVATE_DETAIL
    )


def test_listing_timeout_is_a_safe_typed_discovery_error() -> None:
    with pytest.raises(HeadCommandError, match="timed out") as caught:
        discover_heads("my-project", runner=_timeout_runner, timeout_s=0.25)

    assert caught.value.step == HeadStep.INSTANCE_LOOKUP
    assert _TIMEOUT_PRIVATE_DETAIL not in str(caught.value)
    assert "0.25" in str(caught.value)


def test_project_resolution_timeout_returns_unavailable_project() -> None:
    assert configured_project(_timeout_runner, timeout_s=0.25) is None


@pytest.mark.parametrize("project", ["my-project", None])
def test_discovery_timeout_retains_the_reachable_local_target(project: str | None) -> None:
    outcome = discover(project, runner=_timeout_runner, probe=lambda _url: True)

    assert outcome.project == project
    assert outcome.local_reachable
    assert outcome.heads == []
    local = next(target for target in outcome.targets if target.kind == "local")
    assert local.reachable
    assert local.url == LOCAL_URL
    assert all(target.kind != "new-local" for target in outcome.targets)
    if project is not None:
        assert outcome.discovery_error is not None
        assert "timed out" in outcome.discovery_error
        assert _TIMEOUT_PRIVATE_DETAIL not in outcome.discovery_error
    else:
        assert outcome.discovery_error is None


@pytest.mark.parametrize("project", ["my-project", None])
def test_guided_console_can_select_local_after_discovery_timeout(project: str | None) -> None:
    stream = io.StringIO()
    outcome = start_flow(
        terminal_console(stream),
        project=project,
        runner=_timeout_runner,
        probe=lambda _url: True,
        choose=lambda _count: 1,
    )

    assert outcome.targets[0].kind == "local"
    rendered = stream.getvalue()
    assert f'eyvan.init("{LOCAL_URL}")' in rendered
    assert _TIMEOUT_PRIVATE_DETAIL not in rendered
    if project is not None:
        assert "Could not list heads" in rendered
        assert "timed out" in rendered
    else:
        assert "No GCP project configured" in rendered


@pytest.mark.parametrize("project", ["my-project", None])
def test_start_json_emits_a_local_outcome_after_discovery_timeout(monkeypatch, project: str | None) -> None:
    start_module = importlib.import_module("eyvan.sdk.commands.start")

    def local_discover(selected_project: str | None):
        return discover(selected_project, runner=_timeout_runner, probe=lambda _url: True)

    monkeypatch.setattr(start_module, "discover", local_discover)
    args = ["--json"] if project is None else ["--json", "--project", project]
    result = CliRunner().invoke(start_module.start, args)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["project"] == project
    assert payload["local_reachable"] is True
    assert payload["heads"] == []
    assert _TIMEOUT_PRIVATE_DETAIL not in result.output
    if project is not None:
        assert "timed out" in payload["discovery_error"]
    else:
        assert payload["discovery_error"] is None
