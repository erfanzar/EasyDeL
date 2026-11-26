# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What ``eyvan runtime restart`` reports about the trust it is about to apply.

Callable trust is read from ``EYVAN_TRUSTED_CALLABLE_PROJECTS`` once, when a
control plane starts. The supervisor launches the replacement with the invoking
shell's environment, so restarting from a shell that does not export the
variable produces a control plane strictly less capable than the one it
replaced -- and says nothing about it.
"""

from __future__ import annotations

import pytest

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands.runtime import (
    LocalAgentState,
    RuntimeRestartResult,
    replacement_trusted_callable_projects,
)
from eyvan.server.local.daemon import LocalDaemonResult, LocalDaemonState


def _result(projects: tuple[str, ...]) -> RuntimeRestartResult:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8081")
    return RuntimeRestartResult(
        endpoint=endpoint,
        control_plane=LocalDaemonResult(
            endpoint=endpoint,
            state=LocalDaemonState.STARTED,
            pid=4242,
            log_path=None,
        ),
        agent_endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:8090"),
        agent_state=LocalAgentState.STARTED,
        trusted_callable_projects=projects,
    )


def test_a_restart_without_trust_warns_before_the_next_submission_fails() -> None:
    """Warn during restart when missing callable trust would cause the next submission to be refused."""

    warning = _result(()).callable_trust_warning

    assert warning is not None
    assert "EYVAN_TRUSTED_CALLABLE_PROJECTS" in warning
    assert "403" in warning


def test_a_restart_that_preserves_trust_does_not_warn() -> None:
    """A correctly exported environment must stay quiet, or the warning is noise."""

    assert _result(("eyvan-live/v5p16",)).callable_trust_warning is None


def test_the_restart_document_carries_the_trust_that_will_actually_apply() -> None:
    """``--json`` consumers need the effective value, not just prose."""

    document = _result(("eyvan-live/v5p16", "acme/prod")).as_document()

    assert document["trusted_callable_projects"] == ["eyvan-live/v5p16", "acme/prod"]
    assert document["callable_trust_warning"] is None


def test_the_empty_document_reports_the_warning_for_automation() -> None:
    """A scripted restart must be able to detect the narrowing without a TTY."""

    document = _result(()).as_document()

    assert document["trusted_callable_projects"] == []
    assert "403" in str(document["callable_trust_warning"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ()),
        ("   ", ()),
        ("eyvan-live/v5p16", ("eyvan-live/v5p16",)),
        (" a/b , c/d ", ("a/b", "c/d")),
        ("a/b,,c/d", ("a/b", "c/d")),
    ],
)
def test_the_replacement_trust_is_parsed_like_the_settings_it_mirrors(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: tuple[str, ...]
) -> None:
    """Disagreeing with ``EyvanSettings`` here would report trust the server will not honour."""

    monkeypatch.setenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", raw)

    assert replacement_trusted_callable_projects() == expected


def test_an_unset_variable_reads_as_no_trust_rather_than_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent must be the warned case, not a crash inside the restart path."""

    monkeypatch.delenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", raising=False)

    assert replacement_trusted_callable_projects() == ()
