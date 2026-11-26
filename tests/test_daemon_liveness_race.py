# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A process may disappear between signal-zero and the state query."""

import subprocess

import pytest

from eyvan.server.local import daemon


@pytest.mark.parametrize("query_error", [False, True])
@pytest.mark.parametrize("second_probe", ["gone", "live", "denied"])
def test_state_query_failure_rechecks_pid_without_signalling(monkeypatch, query_error, second_probe):
    calls = []

    def probe(pid, sig):
        calls.append((pid, sig))
        assert sig == 0
        if len(calls) == 2:
            if second_probe == "gone":
                raise ProcessLookupError
            if second_probe == "denied":
                raise PermissionError

    def query(*args, **kwargs):
        if query_error:
            raise OSError("state query unavailable")
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="")

    monkeypatch.setattr(daemon.os, "kill", probe)
    monkeypatch.setattr(daemon.subprocess, "run", query)
    assert daemon.process_is_alive(12345) is (second_probe != "gone")
    assert calls == [(12345, 0), (12345, 0)]
