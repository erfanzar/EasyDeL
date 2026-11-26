# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Network-bind security boundary tests for the cell-agent CLI."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import pytest
import uvicorn

import eyvan.worker.agent.cli as cli


def _stub_runtime(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        cli,
        "_settings_for_runtime_profile",
        lambda _profile: SimpleNamespace(
            mutual_tls_server_mode=lambda: "off",
            mutual_tls_files=lambda: None,
        ),
    )
    monkeypatch.setattr(cli, "agent_workload_identity", lambda _settings: None)
    monkeypatch.setattr(cli, "describe_agent_workload_identity", lambda _identity: "test-agent")
    monkeypatch.setattr(cli, "server_ssl_options", lambda _mode, _files: {})
    monkeypatch.setattr(cli, "create_agent_app", lambda **_kwargs: object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))
    return calls


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10", "agent.example.test"])
@pytest.mark.parametrize("source", ["flag", "env"])
def test_unauthenticated_agent_refuses_non_loopback_before_uvicorn(
    monkeypatch: pytest.MonkeyPatch, host: str, source: str
) -> None:
    """An unauthenticated listener must never be exposed beyond literal loopback."""

    arguments = ["--allow-unauthenticated"] if source == "flag" else []
    monkeypatch.setattr(sys, "argv", ["eyvan-agent", "--host", host, *arguments])
    if source == "env":
        monkeypatch.setenv("EYVAN_AGENT_ALLOW_UNAUTHENTICATED", "true")
    else:
        monkeypatch.delenv("EYVAN_AGENT_ALLOW_UNAUTHENTICATED", raising=False)
    calls = _stub_runtime(monkeypatch)

    with pytest.raises(ValueError, match=re.escape("only permitted with --host 127.0.0.1 or --host ::1")):
        cli.main()

    assert calls == []


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
@pytest.mark.parametrize("source", ["flag", "env"])
def test_unauthenticated_agent_allows_literal_loopback(monkeypatch: pytest.MonkeyPatch, host: str, source: str) -> None:
    arguments = ["--allow-unauthenticated"] if source == "flag" else []
    monkeypatch.setattr(sys, "argv", ["eyvan-agent", "--host", host, *arguments])
    if source == "env":
        monkeypatch.setenv("EYVAN_AGENT_ALLOW_UNAUTHENTICATED", "1")
    else:
        monkeypatch.delenv("EYVAN_AGENT_ALLOW_UNAUTHENTICATED", raising=False)
    calls = _stub_runtime(monkeypatch)

    cli.main()

    assert len(calls) == 1
    assert calls[0]["host"] == host


def test_authenticated_agent_preserves_network_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_AGENT_SHARED_SECRET", "test-secret")
    monkeypatch.delenv("EYVAN_AGENT_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setattr(sys, "argv", ["eyvan-agent", "--host", "0.0.0.0"])
    calls = _stub_runtime(monkeypatch)

    cli.main()

    assert len(calls) == 1
    assert calls[0]["host"] == "0.0.0.0"


@pytest.mark.parametrize("host", ["127.0.0.2", "localhost", "[::1]"])
def test_unauthenticated_agent_requires_the_exact_literal_loopback_bind(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["eyvan-agent", "--host", host, "--allow-unauthenticated"])
    calls = _stub_runtime(monkeypatch)

    with pytest.raises(ValueError):
        cli.main()

    assert calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
