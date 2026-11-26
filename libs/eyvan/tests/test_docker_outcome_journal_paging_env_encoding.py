# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_container_executor import FakeDockerSystemd, _context, _executor, _record, _spec

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus


def _container_collection_child(state: Path, exit_code: int, *, crash: bool) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(state, daemon)
    remote_id = executor.submit(_spec(), _context())
    record = _record(executor, remote_id)
    daemon.containers[record["container_name"]]["State"].update({"Status": "exited", "ExitCode": exit_code})

    def collect(command: tuple[str, ...]):
        result = daemon(command)
        if command[:2] == ("systemctl", "reset-failed"):
            # Model Docker --rm and unit collection at the precise crash boundary.
            daemon.containers.clear()
            (state / "collection-boundary.json").write_text(json.dumps(_record(executor, remote_id)), encoding="utf-8")
            if crash:
                os._exit(77)
        return result

    executor._command_runner = collect
    executor.status(remote_id)


@pytest.mark.parametrize("exit_code, expected", [(0, "SUCCEEDED"), (9, "FAILED")])
@pytest.mark.parametrize("crash", [True, False], ids=["process-crash", "normal-control"])
def test_docker_terminal_outcome_is_durable_before_collection(
    tmp_path: Path, exit_code: int, expected: str, crash: bool
):
    script = (
        "import sys; from pathlib import Path; sys.path.insert(0, str(Path('tests').resolve())); "
        "from test_docker_outcome_journal_paging_env_encoding import _container_collection_child; "
        "_container_collection_child(Path(sys.argv[1]), int(sys.argv[2]), crash=sys.argv[3] == 'True')"
    )
    child = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path), str(exit_code), str(crash)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == (77 if crash else 0), child.stderr
    boundary = json.loads((tmp_path / "collection-boundary.json").read_text(encoding="utf-8"))
    # Startup runs actual recovery against an absent container and absent unit.
    restarted = _executor(tmp_path, FakeDockerSystemd())
    observation = restarted.status("workload-1")
    assert boundary["status"] == expected
    assert observation.status is ExecutionWorkloadStatus(expected)
    assert _record(restarted, "workload-1")["status"] == expected
    assert _record(restarted, "workload-1")["unit_collected"] is True
    assert not restarted._container_env_file_path("workload-1").exists()


@pytest.mark.parametrize("limit", [1, 2, 7])
def test_resumed_journal_pages_cover_backlog_without_skips(tmp_path: Path, monkeypatch, limit: int) -> None:
    from eyvan.worker.runtime import log_records
    from eyvan.worker.runtime.systemd_transient import SystemdCommandResult, SystemdTransientServiceExecutor

    monkeypatch.setattr(log_records, "_local_worker_address", lambda: None)
    backlog = [{"MESSAGE": f"message-{index}", "__CURSOR": f"c{index}"} for index in range(19)]

    def journal(command: tuple[str, ...]) -> SystemdCommandResult:
        after = next(part.removeprefix("--after-cursor=") for part in command if part.startswith("--after-cursor="))
        start = 0 if after == "before" else int(after.removeprefix("c")) + 1
        remaining = backlog[start:]
        lines = next(part.removeprefix("--lines=") for part in reversed(command) if part.startswith("--lines="))
        # Model journalctl's tail-N default, not a fictitious first-N runner.
        if lines != "all":
            remaining = remaining[-int(lines) :]
        return SystemdCommandResult(returncode=0, stdout="".join(json.dumps(entry) + "\n" for entry in remaining))

    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=journal, boot_id_reader=lambda: "boot-a")
    executor._record_path("journal-work").write_text(json.dumps({"unit_name": "journal-work.service"}), encoding="utf-8")
    cursor = "before"
    received = []
    for _ in range(20):
        page = executor.log_page("journal-work", after=cursor, limit=limit)
        assert not page.unavailable
        assert len(page.records) <= limit
        if not page.records:
            break
        assert page.next_cursor == page.records[-1].cursor
        received.extend(record.message for record in page.records)
        cursor = page.next_cursor
    assert received == [entry["MESSAGE"] for entry in backlog]


def test_default_journal_ingestion_stops_at_page_boundary_without_waiting_for_backlog() -> None:
    from eyvan.worker.runtime import systemd_transient

    # A real child continuously writes a large journal-shaped backlog. The
    # reader must stop after the requested entries, not capture then truncate.
    script = (
        'import os; line=b\'{"MESSAGE":"entry","__CURSOR":"cursor"}\\n\'; exec(\'while True: os.write(1, line * 1000)\')'
    )
    result = systemd_transient._systemd_journal_runner((sys.executable, "-B", "-c", script), limit=2, timeout_s=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['{"MESSAGE":"entry","__CURSOR":"cursor"}'] * 2


@pytest.mark.parametrize(
    "value, serialized",
    [
        ('"literal quotes"', '"\\"literal quotes\\""'),
        ("one\\two", '"one\\\\two"'),
        ("  keep both  ", '"  keep both  "'),
        ("\tpadding\t", '"\tpadding\t"'),
        ("'literal single quotes'", "\"'literal single quotes'\""),
        ('"$HOME`echo nope`\\end', '"\\"\\$HOME\\`echo nope\\`\\\\end"'),
    ],
)
def test_native_environment_launch_uses_systemd_value_encoding(
    tmp_path: Path, monkeypatch, value: str, serialized: str
) -> None:
    import stat

    from test_systemd_transient import FakeSystemd
    from test_systemd_transient import _spec as native_spec

    from eyvan.worker.runtime import systemd_transient

    daemon = FakeSystemd()
    executor = systemd_transient.SystemdTransientServiceExecutor(
        tmp_path, command_runner=daemon, boot_id_reader=lambda: "boot-a"
    )
    spec = native_spec()
    spec["environment"]["CONFIG"] = value
    spec["environment"].update(
        {"EYVAN_SECRET_COUNT": "1", "EYVAN_SECRET_0_NAME": "SECRET", "EYVAN_SECRET_0_TOKEN": "fake-capability"}
    )
    monkeypatch.setattr(systemd_transient, "materialize_secrets", lambda _environment: {"SECRET": value})
    remote_id = executor.submit(spec, _context())
    launch = next(command for command in daemon.commands if command[0] == "systemd-run")
    path = Path(
        next(
            part.removeprefix("--property=EnvironmentFile=")
            for part in launch
            if part.startswith("--property=EnvironmentFile=")
        )
    )
    assert path == executor._environment_file_path(remote_id)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    content = path.read_text(encoding="utf-8")
    # This is an exact EnvironmentFile format contract, not a shell/split parser
    # or a claim that a Linux manager was exercised on this Darwin test host.
    assert f"CONFIG={serialized}\n" in content
    assert f"SECRET={serialized}\n" in content
    assert "EYVAN_SECRET_" not in content
    assert "EXPERIMENT=systemd\n" in content
    assert not any(value in part for part in launch)


def test_docker_environment_file_keeps_literal_values(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd()
    executor = _executor(tmp_path, daemon)
    spec = _spec()
    values = {"QUOTED": '"literal"', "BACKSLASH": "one\\two", "PADDED": "  keep both  "}
    spec["environment"].update(values)
    remote_id = executor.submit(spec, _context())
    content = executor._container_env_file_path(remote_id).read_text(encoding="utf-8")
    assert all(f"{name}={value}\n" in content for name, value in values.items())


@pytest.mark.parametrize("systemd", [False, True])
@pytest.mark.parametrize("value", ["injected\nNAME=bad", "has\rreturn", "has\0nul"])
def test_environment_encoding_retains_line_injection_refusal(tmp_path: Path, systemd: bool, value: str) -> None:
    from eyvan.worker.runtime.systemd_transient import _write_environment_file

    path = tmp_path / "environment.env"
    with pytest.raises(ValueError, match="NUL- and newline-free"):
        _write_environment_file(path, {"NAME": value}, systemd=systemd)
    assert not path.exists()


def test_failed_terminal_durability_barrier_does_not_collect_runtime(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    daemon = FakeDockerSystemd()
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record = _record(executor, remote_id)
    daemon.containers[record["container_name"]]["State"].update({"Status": "exited", "ExitCode": 0})

    def fail_write(*_args):
        raise OSError("simulated record durability failure")

    monkeypatch.setattr(systemd_transient, "_atomic_write_json", fail_write)
    with pytest.raises(OSError, match="simulated record durability failure"):
        executor.status(remote_id)
    assert _record(executor, remote_id)["status"] == "PENDING"
    assert executor._container_env_file_path(remote_id).exists()
    assert not any(command[:2] in {("systemctl", "stop"), ("systemctl", "reset-failed")} for command in daemon.commands)


@pytest.mark.parametrize(
    "script, expected_code, detail",
    [
        ("import time; time.sleep(10)", 127, "timed out"),
        ("import os; os.write(2, b'diagnostic'); raise SystemExit(9)", 9, "diagnostic"),
        ("import os; exec('while True: os.write(1, b\"x\" * 65536)')", 127, "byte limit"),
    ],
)
def test_bounded_journal_reader_reports_failed_reads(script: str, expected_code: int, detail: str) -> None:
    from eyvan.worker.runtime.systemd_transient import _systemd_journal_runner

    result = _systemd_journal_runner((sys.executable, "-B", "-c", script), limit=2, timeout_s=0.5)
    assert result.returncode == expected_code
    assert detail in result.stderr


def test_default_log_page_uses_streaming_reader(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import log_records, systemd_transient

    monkeypatch.setattr(log_records, "_local_worker_address", lambda: None)
    executor = systemd_transient.SystemdTransientServiceExecutor(tmp_path, boot_id_reader=lambda: "boot-a")
    executor._record_path("journal-work").write_text(json.dumps({"unit_name": "journal-work.service"}), encoding="utf-8")
    real_popen = subprocess.Popen
    children = []
    script = (
        "import json,os; "
        "[os.write(1, (json.dumps({'MESSAGE': str(i), '__CURSOR': 'c' + str(i)})+'\\n').encode()) for i in range(20)]"
    )

    def local_journal(command, **kwargs):
        assert command[0] == "journalctl"
        assert "--after-cursor=before" in command
        child = real_popen((sys.executable, "-B", "-c", script), **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(systemd_transient.subprocess, "Popen", local_journal)
    page = executor.log_page("journal-work", after="before", limit=2)
    assert [record.message for record in page.records] == ["0", "1"]
    assert page.next_cursor == "c1"
    assert not page.unavailable
    assert all(child.poll() is not None for child in children)
