# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local daemon ownership must survive PID recycling and unusual profile paths."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.server.local import daemon


@dataclass
class Replacement:
    """An owned launch handle without a listener or external side effects."""

    pid: int = 999_999_997
    exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.exit_code = 0

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.exit_code or 0


def test_recycled_pid_does_not_suppress_replacement_startup(tmp_path: Path) -> None:
    """A real unrelated live process is neither adopted nor signalled."""

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19082")
    target = daemon.LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    paths = daemon.LocalDaemonPaths(tmp_path / "state")
    profile = (tmp_path / "profile.json").resolve()
    replacement = Replacement()
    health = iter([False, False, True])
    queried: list[int] = []

    def read_arguments(pid: int) -> tuple[str, ...] | None:
        queried.append(pid)
        return daemon._read_process_arguments(pid)

    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as unrelated:
        try:
            daemon.write_owner_only_record(
                paths.record_path(target),
                daemon.LocalDaemonRecord(endpoint, unrelated.pid, str(profile)),
            )
            supervisor = daemon.LocalDaemonSupervisor(
                paths=paths,
                launcher=SimpleNamespace(start=lambda _command: replacement),
                health_probe=lambda _endpoint: next(health),
                process_argument_reader=read_arguments,
            )
            result = supervisor.ensure(endpoint, runtime_profile_path=profile)
            assert result.state is daemon.LocalDaemonState.STARTED
            assert result.pid == replacement.pid
            assert replacement.poll() is None
            assert unrelated.poll() is None
            assert queried == [unrelated.pid]
            record = daemon.read_owner_only_record(paths.record_path(target))
            assert record is not None and record.pid == replacement.pid
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)


def test_matching_booting_winner_is_adopted_without_a_third_health_probe(tmp_path: Path) -> None:
    """Keep startup-race adoption, but supply actual daemon identity evidence."""

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19082")
    target = daemon.LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    paths = daemon.LocalDaemonPaths(tmp_path / "state")
    profile = (tmp_path / "profile.json").resolve()
    winner_pid = os.getpid()
    winner = daemon.LocalDaemonRecord(endpoint, winner_pid, str(profile), "winner-build")
    daemon.write_owner_only_record(paths.record_path(target), winner)
    command = daemon.LocalDaemonCommand(target, paths.log_path(target), paths.profile(target), profile)
    replacement = Replacement()
    health = iter([False, False])
    supervisor = daemon.LocalDaemonSupervisor(
        paths=paths,
        launcher=SimpleNamespace(start=lambda _command: replacement),
        health_probe=lambda _endpoint: next(health),
        process_argument_reader=lambda pid: command.argv if pid == winner_pid else None,
        process_liveness=lambda pid: pid == winner_pid,
    )
    result = supervisor.ensure(endpoint, runtime_profile_path=profile)
    assert result.state is daemon.LocalDaemonState.REUSED
    assert result.pid == winner_pid
    assert replacement.poll() == 0
    assert daemon.read_owner_only_record(paths.record_path(target)) == winner


@pytest.mark.parametrize("directory", ["plain", "profile with spaces", "profile'quote", 'profile"quote', "پروفایل"])
@pytest.mark.parametrize("action", ["stop", "restart", "rollback"])
def test_true_process_argv_preserves_profile_boundaries_and_allows_stop(
    tmp_path: Path, directory: str, action: str
) -> None:
    """Probe and stop an owned harmless sleeper, not an Eyvan listener."""

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19083")
    target = daemon.LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    paths = daemon.LocalDaemonPaths(tmp_path / "state")
    profile = (tmp_path / directory / "profile.json").resolve()
    command = daemon.LocalDaemonCommand(target, paths.log_path(target), paths.profile(target), profile)
    # Python -c consumes the code; the remaining options are inert sys.argv.
    # A ready line proves exec completed before the OS argument read.
    argv = (sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)", *command.argv[1:])
    with subprocess.Popen(argv, stdout=subprocess.PIPE, text=True) as sleeper:
        try:
            assert sleeper.stdout is not None and sleeper.stdout.readline() == "ready\n"
            arguments = daemon._read_process_arguments(sleeper.pid)
            assert arguments is not None
            assert daemon._single_option_value(arguments, "--runtime-profile") == str(profile)
            assert arguments == argv
            assert daemon._matches_local_daemon_process(
                arguments, target, runtime_profile_path=str(profile), require_local_daemon=True
            )
            daemon.write_owner_only_record(
                paths.record_path(target), daemon.LocalDaemonRecord(endpoint, sleeper.pid, str(profile))
            )
            replacement = Replacement()
            health = iter([False, False, True])
            supervisor = daemon.LocalDaemonSupervisor(
                paths=paths,
                launcher=SimpleNamespace(start=lambda _command: replacement),
                health_probe=lambda _endpoint: next(health),
            )
            if action == "stop":
                assert supervisor.stop(endpoint).state is daemon.LocalDaemonState.STOPPED
            elif action == "restart":
                result = supervisor.restart(endpoint, runtime_profile_path=profile)
                assert result.state is daemon.LocalDaemonState.STARTED
                assert result.pid == replacement.pid
            else:
                supervisor.stop_started(
                    daemon.LocalDaemonResult(endpoint, daemon.LocalDaemonState.STARTED, sleeper.pid, None),
                    runtime_profile_path=profile,
                )
            assert sleeper.wait(timeout=5) == -15
            record = daemon.read_owner_only_record(paths.record_path(target))
            if action == "restart":
                assert record is not None and record.pid == replacement.pid
            else:
                assert record is None
        finally:
            if sleeper.poll() is None:
                sleeper.terminate()
                sleeper.wait(timeout=5)


def test_darwin_argument_parser_preserves_empty_arguments_and_excludes_environment() -> None:
    arguments = ("python", "--runtime-profile", " /tmp/profile' with spaces.json ", "", "last")
    data = (
        len(arguments).to_bytes(4, sys.byteorder, signed=True)
        + b"/usr/bin/python\0\0\0"
        + b"\0".join(os.fsencode(argument) for argument in arguments)
        + b"\0PRIVATE_ENV=do-not-expose\0"
    )
    assert daemon._parse_darwin_process_arguments(data) == arguments


@pytest.mark.parametrize(
    "data",
    [b"", b"\0", (0).to_bytes(4, sys.byteorder), (1).to_bytes(4, sys.byteorder) + b"no-terminator"],
)
def test_darwin_unreadable_identity_fails_closed(data: bytes) -> None:
    assert daemon._parse_darwin_process_arguments(data) is None


def test_linux_proc_reader_preserves_argument_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    arguments = (b"python", b"--runtime-profile", b" /tmp/profile' with spaces.json ", b"", b"opaque\xff")

    def read_cmdline(path: Path) -> bytes:
        assert path == Path("/proc/12345/cmdline")
        return b"\0".join(arguments) + b"\0"

    monkeypatch.setattr(daemon.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_bytes", read_cmdline)
    assert daemon._read_process_arguments(12345) == tuple(os.fsdecode(argument) for argument in arguments)


@pytest.mark.parametrize("data", [b"", b"unterminated"])
def test_linux_incomplete_identity_fails_closed(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(daemon.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_bytes", lambda _path: data)
    assert daemon._read_process_arguments(12345) is None


def test_unreadable_proc_identity_does_not_fall_back_to_ps(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(_path: Path) -> bytes:
        raise PermissionError("process is not readable")

    monkeypatch.setattr(daemon.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_bytes", unreadable)
    assert daemon._read_process_arguments(12345) is None


def test_unsupported_platform_and_invalid_pid_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon.sys, "platform", "unsupported")
    assert daemon._read_process_arguments(12345) is None
    assert daemon._read_process_arguments(0) is None
    assert daemon._read_process_arguments(-1) is None
