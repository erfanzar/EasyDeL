# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from pathlib import Path

import pytest


def test_a_slow_start_leaves_the_server_running_instead_of_killing_it(tmp_path) -> None:
    """The failure that left a fleet with no control plane.

    Startup reconciles every cell, and each one provisioning or draining costs
    a provider round trip -- so on a real fleet the server is simply slower than
    the budget. Killing it there destroyed a reconcile in progress and left
    nothing running while TPU capacity was ACTIVE, which is strictly worse than
    waiting: the process was seconds from healthy.
    """

    import pytest

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.server.local.daemon import LocalDaemonPaths, LocalDaemonStartupError, LocalDaemonSupervisor

    killed: list[object] = []

    class _StillStarting:
        """A process that is alive and has not answered yet."""

        pid = 4242

        def poll(self) -> None:
            return None

    supervisor = LocalDaemonSupervisor(
        # Scoped to tmp_path because `ensure` *writes* a PID record for the
        # endpoint it is given, and the default paths resolve to the operator's
        # real ~/.eyvan. Without this the test stamps pid 4242 over the record of
        # whatever control plane is genuinely listening on 8080, and the next
        # native command refuses to run against its own live plane.
        paths=LocalDaemonPaths(tmp_path / "eyvan"),
        launcher=type("L", (), {"start": lambda self, command: _StillStarting()})(),
        health_probe=lambda _endpoint: False,
        startup_timeout_s=0.01,
        clock=iter([0.0, 0.0, 1.0, 2.0, 3.0]).__next__,
        sleeper=lambda _seconds: None,
        process_terminator=lambda *args, **kwargs: killed.append(args),
        process_killer=lambda *args, **kwargs: killed.append(args),
    )

    with pytest.raises(LocalDaemonStartupError) as raised:
        supervisor.ensure(ControlPlaneEndpoint("http://127.0.0.1:8080"))

    assert "still starting" in str(raised.value)
    assert "left running" in str(raised.value)
    assert killed == [], "a server that has not died must not be killed for being slow"


def test_the_startup_budget_fits_a_fleet_rather_than_one_local_pool() -> None:
    """Twenty seconds was the budget of a plane holding a single local pool."""

    from eyvan.server.local.daemon import LocalDaemonSupervisor

    assert LocalDaemonSupervisor().startup_timeout_s >= 60


#: A process id high enough that nothing on the machine can be using it, so the
#: test never depends on which pids happen to be live.
_DEAD_PID = 999_999_998


def test_a_stale_pid_beside_a_healthy_plane_is_adopted_not_refused(tmp_path) -> None:
    """A dead PID in the record is a stale record, not somebody else's process.

    The supervisor refused to proceed whenever the recorded PID was not running,
    even though the health probe had just got a typed Eyvan response from the
    endpoint and the record named that same endpoint and runtime profile. The
    record goes stale on its own -- a later start that fails overwrites a
    working plane's record with the PID it just launched, and then that PID dies
    -- so the refusal fired on a plane that was Eyvan's and serving.

    The only way past it was hand-editing the record's JSON, which happened
    three times in one session. Adopt instead.
    """

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.server.local.daemon import (
        LocalDaemonPaths,
        LocalDaemonRecord,
        LocalDaemonState,
        LocalDaemonSupervisor,
        LocalDaemonTarget,
        write_owner_only_record,
    )

    endpoint = ControlPlaneEndpoint("http://127.0.0.1:8080")
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    record_path = paths.record_path(target)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    profile = str(tmp_path / "profile.json")
    # Written through the real writer so the file carries the owner-only mode
    # the reader insists on; a hand-written file fails for the wrong reason.
    write_owner_only_record(
        record_path, LocalDaemonRecord(endpoint=endpoint, pid=_DEAD_PID, runtime_profile_path=profile)
    )

    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=type("L", (), {"start": lambda self, command: None})(),
        health_probe=lambda _endpoint: True,
        process_liveness=lambda _pid: False,
    )

    result = supervisor._existing_result(endpoint, record_path, paths.log_path(target), profile)

    assert result is not None, "a healthy Eyvan plane must not be refused over a stale pid"
    assert result.state is LocalDaemonState.REUSED
    # Not invented: which pid holds a socket is not portable to discover, and the
    # serving process rewrites its own record at startup.
    assert result.pid is None


def test_a_stale_pid_under_a_different_profile_is_still_refused(tmp_path) -> None:
    """Adoption is scoped to the profile; a foreign plane is still a refusal."""

    import pytest

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.server.local.daemon import (
        LocalDaemonConfigurationError,
        LocalDaemonPaths,
        LocalDaemonRecord,
        LocalDaemonSupervisor,
        LocalDaemonTarget,
        write_owner_only_record,
    )

    endpoint = ControlPlaneEndpoint("http://127.0.0.1:8080")
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    record_path = paths.record_path(target)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_owner_only_record(
        record_path,
        LocalDaemonRecord(endpoint=endpoint, pid=_DEAD_PID, runtime_profile_path="/some/other/profile.json"),
    )

    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=type("L", (), {"start": lambda self, command: None})(),
        health_probe=lambda _endpoint: True,
        process_liveness=lambda _pid: False,
    )

    with pytest.raises(LocalDaemonConfigurationError, match="different runtime profile"):
        supervisor._existing_result(endpoint, record_path, paths.log_path(target), str(tmp_path / "profile.json"))


def test_losing_a_startup_race_does_not_take_the_winners_record(tmp_path) -> None:
    """The failure that left a healthy plane unreachable.

    Two commands can try to start a plane at once -- `runtime deploy` followed
    immediately by anything else, where the second probes health while the first
    is still booting. The loser used to stamp its own PID over the winner's
    record, fail on the state lease, then correctly clear "its" record on the way
    out. A healthy plane was left with no record at all, and every later command
    refused to talk to it: "no owner-only PID record exists".

    Losing the race is not an error. The winner is a control plane at the
    endpoint this one wanted, so the loser stops and adopts it, and the winner's
    record is left exactly as it was.
    """

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.server.local.daemon import (
        LocalDaemonPaths,
        LocalDaemonRecord,
        LocalDaemonState,
        LocalDaemonSupervisor,
        LocalDaemonTarget,
        read_owner_only_record,
        write_owner_only_record,
    )

    endpoint = ControlPlaneEndpoint("http://127.0.0.1:8080")
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    record_path = paths.record_path(target)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    profile = str(tmp_path / "profile.json")

    # The winner: already serving, and its record says so.
    winner_pid = os.getpid()
    write_owner_only_record(
        record_path,
        LocalDaemonRecord(endpoint=endpoint, pid=winner_pid, runtime_profile_path=profile),
    )

    stopped: list[int] = []

    class _Loser:
        """The process that started second and must give the endpoint back."""

        pid = winner_pid + 1_000_000

        def poll(self) -> None:
            return None if not stopped else 1

        def terminate(self) -> None:
            stopped.append(self.pid)

        def kill(self) -> None:
            stopped.append(self.pid)

        def wait(self, timeout: float | None = None) -> int:
            return 1

    # Every probe misses: the winner is still finishing its own startup and does
    # not answer health within a timeout. That is the case that raised at a plane
    # which was healthy seconds later, so adoption must not depend on a probe.
    def _never_answers(_endpoint: object) -> bool:
        return False

    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=type("L", (), {"start": lambda self, command: _Loser()})(),
        health_probe=_never_answers,
        # A live PID is not sufficient proof: model the winning daemon's exact
        # argv rather than asking the supervisor to trust this pytest process.
        process_liveness=lambda pid: pid == winner_pid,
        process_argument_reader=lambda pid: (
            (
                "python",
                "-m",
                "eyvan.server.api.app",
                "--host",
                target.host,
                "--port",
                str(target.port),
                "--local-daemon",
                "--runtime-profile",
                profile,
            )
            if pid == winner_pid
            else None
        ),
        startup_timeout_s=0.01,
        clock=iter([0.0, 0.0, 1.0, 2.0]).__next__,
        sleeper=lambda _seconds: None,
    )

    result = supervisor.ensure(endpoint, runtime_profile_path=Path(profile))

    assert result.state is LocalDaemonState.REUSED, "the loser did not adopt the plane that won"
    surviving = read_owner_only_record(record_path)
    assert surviving is not None, "the winner's record was deleted by the loser"
    assert surviving.pid == winner_pid, "the loser stamped its own pid over the winner's record"
    assert stopped == [_Loser.pid], "the losing process must be stopped, not the verified winner"


def test_stop_ends_a_plane_this_supervisor_started(tmp_path) -> None:
    """``ensure`` without a ``stop`` is how a suite leaks a plane onto a port.

    A daemon outlives the process that started it -- that is the point of a
    daemon -- so the supervisor that records it must also be able to stop it,
    proving the same ownership it demanded before starting. A leaked plane
    keeps the port bound with a credential from a profile directory that has
    since been deleted, and the next command against that port answers 401
    from a plane nobody owns.
    """

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.server.local.daemon import (
        LocalDaemonPaths,
        LocalDaemonState,
        LocalDaemonSupervisor,
        LocalDaemonTarget,
        process_is_alive,
        read_owner_only_record,
    )

    endpoint = ControlPlaneEndpoint("http://127.0.0.1:8099")
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    supervisor = LocalDaemonSupervisor(paths=paths)
    started = supervisor.ensure(endpoint)
    assert started.state is LocalDaemonState.STARTED
    assert started.pid is not None
    pid = started.pid
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    record_path = paths.record_path(target)
    try:
        assert process_is_alive(pid)
        assert read_owner_only_record(record_path) is not None
    finally:
        stopped = supervisor.stop(endpoint)
    assert stopped.state is LocalDaemonState.STOPPED
    assert not process_is_alive(pid)
    assert read_owner_only_record(record_path) is None, "stopping must clear the record it owned"
    # A cleared record makes a second stop a no-op, not an error.
    assert supervisor.stop(endpoint).state is LocalDaemonState.STOPPED


def test_identity_resolved_before_the_plane_picks_up_the_credential_it_mints(tmp_path, monkeypatch) -> None:
    """First run, first command: the plane mints what the plane then demands.

    A CLI session resolves its identity before the command body starts the
    local plane. On a machine with no development credential yet, starting
    the plane mints one and the hardened loopback auth requires it
    immediately -- so the pre-start identity is refused by the plane the
    same command just started, and `eyvan up` failed its first-ever run on
    any new machine. The contract this pins: resolving after the plane
    exists picks up the credential, and only the stale identity fails.
    """

    import httpx

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.sdk.client.identity import resolve_control_plane_identity
    from eyvan.server.local.daemon import LocalDaemonPaths, LocalDaemonState, LocalDaemonSupervisor

    monkeypatch.setenv("HOME", str(tmp_path))
    endpoint = ControlPlaneEndpoint("http://127.0.0.1:8099")
    supervisor = LocalDaemonSupervisor(paths=LocalDaemonPaths(tmp_path / "eyvan"))
    # The role the CLI itself uses on a loopback plane: the default operator
    # role has no organization scope and is refused before authorization.
    stale = resolve_control_plane_identity(endpoint, dev_roles="fleet_admin")
    assert stale.local_development_credential is None, "a fresh machine has no credential to present"
    try:
        started = supervisor.ensure(endpoint)
        assert started.state is LocalDaemonState.STARTED
        response = httpx.get(f"{endpoint.url}/api/v1/fleets", headers=stale.headers())
        assert response.status_code == 401, "the stale identity must be refused, naming the real failure mode"
        refreshed = resolve_control_plane_identity(endpoint, dev_roles="fleet_admin")
        assert refreshed.local_development_credential is not None, "the plane minted the credential it demands"
        response = httpx.get(f"{endpoint.url}/api/v1/fleets", headers=refreshed.headers())
        assert response.status_code != 401, "the re-resolved identity must satisfy the plane it just started"
    finally:
        supervisor.stop(endpoint)


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
