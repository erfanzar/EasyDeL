# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Teardown reports survivors only after attempting every owned record."""

from types import SimpleNamespace

import pytest
from conftest import stop_local_daemons_recorded_under

from eyvan.server.local import daemon


def test_teardown_continues_after_a_refused_surviving_daemon(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    records_dir = tmp_path / ".eyvan"
    records_dir.mkdir()
    paths = [records_dir / "a.pid", records_dir / "b.pid"]
    for path in paths:
        path.write_text("test-owned record", encoding="utf-8")
    records = {
        paths[0]: SimpleNamespace(pid=101, endpoint=SimpleNamespace(url="http://127.0.0.1:8101")),
        paths[1]: SimpleNamespace(pid=202, endpoint=SimpleNamespace(url="http://127.0.0.1:8102")),
    }
    alive = {101, 202}
    attempted = []

    def stop(_self, endpoint) -> None:
        attempted.append(endpoint.url)
        if endpoint.url.endswith(":8101"):
            raise daemon.LocalDaemonError("simulated ownership refusal")
        alive.remove(202)

    monkeypatch.setattr(daemon, "read_owner_only_record", records.get)
    monkeypatch.setattr(daemon, "process_is_alive", lambda pid: pid in alive)
    monkeypatch.setattr(daemon.LocalDaemonSupervisor, "stop", stop)

    with pytest.raises(pytest.fail.Exception, match="101") as failure:
        stop_local_daemons_recorded_under(tmp_path)

    assert attempted == ["http://127.0.0.1:8101", "http://127.0.0.1:8102"]
    assert alive == {101}, "ownership refusal must not prevent cleanup of the later owned process"
    assert "http://127.0.0.1:8101" in str(failure.value)
    assert "202" not in str(failure.value), "a successfully stopped process must not be reported as a survivor"
