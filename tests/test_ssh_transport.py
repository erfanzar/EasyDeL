# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OpenSSH transport: destinations, argv, ports and supervised tunnels, without real ssh."""

from __future__ import annotations

import socket

import pytest

from eyvan._internal.ssh import (
    LocalForward,
    SshDestination,
    SshError,
    SshOptions,
    SshTunnelManager,
    SshTunnelSpec,
    free_local_ports,
    probe_tcp,
    scp_argv,
    ssh_argv,
)

#: Above macOS/Linux default pid_max, so a stray os.kill can never hit a real process.
_FAKE_PID = 9_999_999


@pytest.mark.parametrize(
    ("text", "host", "user"),
    [
        ("host.example", "host.example", None),
        ("me@203.0.113.20", "203.0.113.20", "me"),
        ("ssh://me@host", "host", "me"),
        ("[2001:db8::1]", "2001:db8::1", None),
    ],
)
def test_destination_user_is_optional(text, host, user):
    destination = SshDestination.parse(text)
    assert (destination.host, destination.user) == (host, user)


@pytest.mark.parametrize("text", ["-oProxyCommand=x", "me@host;rm", "@host", "me@../host", "me@ho st", ""])
def test_destination_refuses_options_and_shell_syntax(text):
    with pytest.raises(ValueError):
        SshDestination.parse(text)


def test_ssh_argv_jump_forwards_and_end_of_options():
    entry = SshDestination.parse("me@203.0.113.5")
    argv = ssh_argv(
        entry.with_host("10.0.0.3"),
        "uptime",
        options=SshOptions(connect_timeout_s=7),
        jump=entry,
    )
    assert argv[0] == "ssh" and "ConnectTimeout=7" in argv
    assert argv[argv.index("-J") + 1] == "me@203.0.113.5"
    assert argv[-3:] == ("--", "me@10.0.0.3", "uptime")
    held = ssh_argv(entry, None, options=SshOptions(), forwards=[LocalForward(4100, "10.0.0.3", 8100)])
    assert "-N" in held and "ExitOnForwardFailure=yes" in held
    # Never multiplexed into an operator's ControlMaster: the process must hold the ports itself.
    assert "ControlMaster=no" in held and "ControlPath=none" in held
    assert held.index("ControlPath=none") < held.index("--")
    assert "ControlMaster=no" not in argv, "one-shot commands keep the operator's multiplexing"
    assert held[held.index("-L") + 1] == "127.0.0.1:4100:10.0.0.3:8100"
    assert held[-2:] == ("--", "me@203.0.113.5")


def test_scp_argv_targets_a_directory():
    from pathlib import Path

    argv = scp_argv(Path("/w/eyvan.whl"), SshDestination.parse("[2001:db8::1]"), "/tmp", options=SshOptions())
    assert argv[-2:] == ("/w/eyvan.whl", "[2001:db8::1]:/tmp/")


@pytest.mark.parametrize(
    "kwargs",
    [{"extra": ("ProxyCommand=sh -c x",)}, {"connect_timeout_s": 0}, {"host_key_policy": "maybe"}],
)
def test_options_refuse_unsafe_or_meaningless_values(kwargs):
    with pytest.raises(ValueError):
        SshOptions(**kwargs)


def test_os_chooses_distinct_ports():
    ports = free_local_ports(4)
    assert len(set(ports)) == 4 and all(1 <= port <= 65535 for port in ports)


def test_probe_tells_a_listener_from_a_reachable_closed_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert probe_tcp("127.0.0.1", port, 1.0) == "open"
    assert probe_tcp("127.0.0.1", port, 1.0) == "refused"


class FakeProcess:
    def __init__(self, exit_code=None):
        self.pid = _FAKE_PID
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code

    def wait(self, timeout=None):
        return 0


def _spec(port=4100, entry="me@203.0.113.5"):
    return SshTunnelSpec(
        key="attach-test",
        entry=SshDestination.parse(entry),
        forwards=(LocalForward(port, "10.0.0.2", 8100), LocalForward(port + 1, "10.0.0.3", 8100)),
    )


def _manager(tmp_path, *, exit_code=None, ports_open=True):
    started: list[list[str]] = []

    def popen(argv, **kwargs):
        started.append(argv)
        kwargs["stderr"].write(b"channel 0: open failed\nssh: connect to host 203.0.113.5 port 22: refused\n")
        return FakeProcess(exit_code)

    manager = SshTunnelManager(
        directory=tmp_path,
        startup_timeout_s=0.5,
        popen=popen,
        process_command=lambda pid: None,
        port_open=lambda host, port, timeout: ports_open,
        sleep=lambda seconds: None,
    )
    return manager, started


def test_tunnel_is_started_once_and_reused(tmp_path):
    manager, started = _manager(tmp_path)
    assert manager.ensure(_spec()) is True
    assert manager.ensure(_spec()) is False
    assert len(started) == 1 and started[0][0] == "ssh"
    assert manager.status(_spec())
    assert (tmp_path / "attach-test.json").stat().st_mode & 0o777 == 0o600


def test_changed_tunnel_replaces_the_old_process(tmp_path):
    manager, started = _manager(tmp_path)
    manager.ensure(_spec())
    assert manager.ensure(_spec(port=4200)) is True
    assert len(started) == 2


def test_tunnel_that_exits_reports_ssh_diagnostic(tmp_path):
    manager, _ = _manager(tmp_path, exit_code=255, ports_open=False)
    with pytest.raises(SshError, match="port 22: refused"):
        manager.ensure(_spec())


def test_tunnel_that_never_opens_times_out(tmp_path):
    manager, _ = _manager(tmp_path, ports_open=False)
    with pytest.raises(SshError, match="did not open"):
        manager.ensure(_spec())


def test_stop_forgets_the_record(tmp_path):
    manager, _ = _manager(tmp_path)
    manager.ensure(_spec())
    assert manager.stop("attach-test") is True
    assert not (tmp_path / "attach-test.json").exists()
    assert manager.stop("attach-test") is False
