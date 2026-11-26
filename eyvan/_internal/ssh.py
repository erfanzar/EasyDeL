# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain OpenSSH as a transport: destinations, argv, remote commands and local tunnels.

Nothing here fixes a user, key, port or timing. A destination's user is optional
so the operator's own ``~/.ssh/config`` decides it, exactly as ``ssh HOST`` does;
tunnel ports are chosen by the operating system and recorded by the caller; every
timing lives on :class:`SshOptions`.

Argv is always built as a list and passed without a shell. Remote command strings
are the only shell text, and callers build them with :func:`shlex.join`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import ipaddress
import json
import os
import re
import signal
import socket
import subprocess
import time
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from eyvan._internal.subprocess_env import provider_cli_env

_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_HOSTNAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?")
_OPTION = re.compile(r"[A-Za-z][A-Za-z0-9]*=[^\s'\"\\;|&$`<>]+")
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_POLICIES = frozenset({"yes", "no", "accept-new", "ask", "off"})


class SshError(RuntimeError):
    """An SSH transport operation failed; messages never contain input that was sent."""


@dataclass(frozen=True, slots=True)
class SshDestination:
    """One SSH destination: a host name or IP literal, with an optional user."""

    host: str
    user: str | None = None

    def __post_init__(self) -> None:
        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            if not _HOSTNAME.fullmatch(self.host) or ".." in self.host:
                raise ValueError(f"invalid SSH host {self.host!r}") from None
        if self.user is not None and not _USER.fullmatch(self.user):
            raise ValueError(f"invalid SSH user {self.user!r}")

    @classmethod
    def parse(cls, value: str) -> SshDestination:
        """Parse ``[ssh://][user@]host``; options, paths and shell syntax are refused."""

        text = value.strip().removeprefix("ssh://")
        user, separator, host = text.rpartition("@")
        if separator and not user:
            raise ValueError(f"invalid SSH destination {value!r}")
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        return cls(host=host, user=user if separator else None)

    @property
    def ip(self) -> str | None:
        """Return the canonical IP literal, or ``None`` for a host name."""

        try:
            return str(ipaddress.ip_address(self.host))
        except ValueError:
            return None

    @property
    def target(self) -> str:
        """Return the argument ``ssh`` takes."""

        return f"{self.user}@{self.host}" if self.user else self.host

    def remote_path(self, path: str) -> str:
        """Return an ``scp`` remote operand for ``path`` on this destination."""

        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.user + '@' if self.user else ''}{host}:{path}"

    def with_host(self, host: str) -> SshDestination:
        """Return the same user on another host (another VM of one slice)."""

        return SshDestination(host=host, user=self.user)


@dataclass(frozen=True, slots=True)
class SshOptions:
    """Every SSH timing and policy Eyvan sets; anything else is the operator's ssh config.

    Attributes:
        connect_timeout_s: Seconds ``ssh`` waits for a TCP connection.
        command_timeout_s: Seconds Eyvan waits for one remote command.
        keepalive_interval_s: Seconds between keepalives on long connections.
        keepalive_count: Missed keepalives before a connection is declared dead.
        host_key_policy: ``StrictHostKeyChecking`` value.
        batch_mode: Never prompt (a prompt would hang a non-interactive caller).
        extra: Additional ``KEY=VALUE`` ``-o`` options.
    """

    connect_timeout_s: int = 15
    command_timeout_s: float = 600.0
    keepalive_interval_s: int = 15
    keepalive_count: int = 3
    host_key_policy: str = "accept-new"
    batch_mode: bool = True
    extra: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("connect_timeout_s", "keepalive_interval_s", "keepalive_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"SSH option {name} must be a positive integer")
        if not self.command_timeout_s > 0:
            raise ValueError("SSH option command_timeout_s must be positive")
        if self.host_key_policy not in _POLICIES:
            raise ValueError(f"SSH host_key_policy must be one of {sorted(_POLICIES)}")
        for option in self.extra:
            if not _OPTION.fullmatch(option):
                raise ValueError(f"SSH option {option!r} must be KEY=VALUE without shell syntax")

    def argv(self) -> tuple[str, ...]:
        """Return the ``-o`` arguments these options mean."""

        options = [
            f"ConnectTimeout={self.connect_timeout_s}",
            f"ServerAliveInterval={self.keepalive_interval_s}",
            f"ServerAliveCountMax={self.keepalive_count}",
            f"StrictHostKeyChecking={self.host_key_policy}",
            *(("BatchMode=yes",) if self.batch_mode else ()),
            *self.extra,
        ]
        return tuple(part for option in options for part in ("-o", option))


@dataclass(frozen=True, slots=True)
class LocalForward:
    """``-L bind:local_port:remote_host:remote_port``, resolved on the far side."""

    local_port: int
    remote_host: str
    remote_port: int
    bind: str = "127.0.0.1"

    def __post_init__(self) -> None:
        for port in (self.local_port, self.remote_port):
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise ValueError("forwarded ports must be integers between 1 and 65535")
        SshDestination(self.remote_host)
        SshDestination(self.bind)

    @property
    def spec(self) -> str:
        remote = f"[{self.remote_host}]" if ":" in self.remote_host else self.remote_host
        return f"{self.bind}:{self.local_port}:{remote}:{self.remote_port}"


def ssh_argv(
    destination: SshDestination,
    command: str | None,
    *,
    options: SshOptions,
    jump: SshDestination | None = None,
    forwards: tp.Sequence[LocalForward] = (),
) -> tuple[str, ...]:
    """Build ``ssh`` argv; with ``command=None`` it only holds ``forwards`` open."""

    argv: list[str] = ["ssh", *options.argv()]
    if jump is not None:
        argv.extend(("-J", jump.target))
    for forward in forwards:
        argv.extend(("-L", forward.spec))
    if command is None:
        # A held tunnel must own its connection. With the operator's
        # ``ControlMaster auto``, ``ssh -N -L`` hands its forwards to a shared
        # master and exits at once, so the supervised process would
        # never be the thing holding the ports, and its death would mean nothing.
        argv.extend(("-N", "-o", "ExitOnForwardFailure=yes", "-o", "ControlMaster=no", "-o", "ControlPath=none"))
    argv.extend(("--", destination.target))
    if command is not None:
        argv.append(command)
    return tuple(argv)


def scp_argv(
    local: Path, destination: SshDestination, remote_dir: str, *, options: SshOptions, jump: SshDestination | None = None
) -> tuple[str, ...]:
    """Build ``scp`` argv copying one local file into ``remote_dir``."""

    return (
        "scp",
        *options.argv(),
        *(("-J", jump.target) if jump is not None else ()),
        "--",
        str(local),
        destination.remote_path(remote_dir.rstrip("/") + "/"),
    )


class SshRunner(tp.Protocol):
    """Run one argv; the same shape as the gcloud runner, so steps share one boundary."""

    def __call__(
        self, args: tp.Sequence[str], *, stdin_text: str | None, timeout: float
    ) -> subprocess.CompletedProcess[str]: ...


def run_ssh(args: tp.Sequence[str], *, stdin_text: str | None, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ``ssh``/``scp`` without a shell, with interpreter-steering variables removed."""

    if stdin_text is None:
        return subprocess.run(
            list(args),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=provider_cli_env(),
        )
    return subprocess.run(
        list(args),
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=provider_cli_env(),
    )


def free_local_ports(count: int, *, bind: str = "127.0.0.1") -> tuple[int, ...]:
    """Ask the operating system for ``count`` distinct free TCP ports on ``bind``.

    The sockets are held open together so the kernel cannot hand out one port twice.
    """

    if count < 1:
        raise ValueError("count must be positive")
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET6 if ":" in bind else socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(listener)
            listener.bind((bind, 0))
        return tuple(int(listener.getsockname()[1]) for listener in sockets)
    finally:
        for listener in sockets:
            listener.close()


def port_is_open(host: str, port: int, timeout: float) -> bool:
    """Return whether something accepts TCP connections at ``host:port``."""

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_tcp(host: str, port: int, timeout: float) -> str:
    """Classify a TCP connect: ``open``, ``refused`` (host reached, nothing listening) or ``filtered``.

    A firewall that drops packets looks like a timeout; a reachable host with no
    listener answers with a reset. The difference is what tells Eyvan whether a
    worker port will be reachable once a worker listens on it.
    """

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "open"
    except ConnectionRefusedError:
        return "refused"
    except OSError:
        return "filtered"


@dataclass(frozen=True, slots=True)
class SshTunnelSpec:
    """One supervised ``ssh -N`` process holding a set of local forwards open."""

    key: str
    entry: SshDestination
    forwards: tuple[LocalForward, ...]
    options: SshOptions = field(default_factory=SshOptions)

    def __post_init__(self) -> None:
        if not _KEY.fullmatch(self.key):
            raise ValueError(f"invalid tunnel key {self.key!r}")
        if not self.forwards:
            raise ValueError("a tunnel needs at least one forward")
        ports = [forward.local_port for forward in self.forwards]
        if len(set(ports)) != len(ports):
            raise ValueError("tunnel local ports must be distinct")

    def argv(self) -> tuple[str, ...]:
        return ssh_argv(self.entry, None, options=self.options, forwards=self.forwards)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.argv()).encode()).hexdigest()


def _process_command(pid: int) -> str | None:
    completed = subprocess.run(("ps", "-o", "command=", "-p", str(pid)), capture_output=True, text=True, check=False)
    text = completed.stdout.strip()
    return text if completed.returncode == 0 and text else None


def _default_tunnel_directory() -> Path:
    return Path("~/.eyvan/tunnels").expanduser()


@dataclass
class SshTunnelManager:
    """Start, reuse and stop tunnels across processes, recorded by key on disk.

    The CLI and the cell agent share one directory, so a tunnel the CLI opened is
    the one the agent later keeps alive, and either can replace a dead one.
    """

    directory: Path = field(default_factory=_default_tunnel_directory)
    startup_timeout_s: float = 30.0
    popen: tp.Callable[..., tp.Any] = subprocess.Popen
    process_command: tp.Callable[[int], str | None] = _process_command
    port_open: tp.Callable[[str, int, float], bool] = port_is_open
    clock: tp.Callable[[], float] = time.monotonic
    sleep: tp.Callable[[float], None] = time.sleep
    _children: dict[int, tp.Any] = field(default_factory=dict, init=False, repr=False)

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        if not _KEY.fullmatch(key):
            raise ValueError(f"invalid tunnel key {key!r}")
        return (self.directory / f"{key}.json", self.directory / f"{key}.lock", self.directory / f"{key}.log")

    @contextlib.contextmanager
    def _locked(self, key: str) -> tp.Iterator[None]:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _, lock_path, _ = self._paths(key)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _record(self, key: str) -> dict[str, tp.Any] | None:
        record_path, _, _ = self._paths(key)
        try:
            document = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return document if isinstance(document, dict) and isinstance(document.get("pid"), int) else None

    def _alive(self, record: dict[str, tp.Any]) -> bool:
        pid = int(record["pid"])
        child = self._children.get(pid)
        if child is not None:
            return bool(child.poll() is None)
        command = self.process_command(pid)
        # A recycled PID belongs to some other program; never treat it as ours.
        return command is not None and "ssh" in command and str(record.get("first_forward", "")) in command

    def _healthy(self, spec: SshTunnelSpec, record: dict[str, tp.Any] | None) -> bool:
        return (
            record is not None
            and record.get("fingerprint") == spec.fingerprint
            and self._alive(record)
            and all(self.port_open(forward.bind, forward.local_port, 1.0) for forward in spec.forwards)
        )

    def status(self, spec: SshTunnelSpec) -> bool:
        """Return whether ``spec``'s tunnel is running and every local port answers."""

        return self._healthy(spec, self._record(spec.key))

    def ensure(self, spec: SshTunnelSpec) -> bool:
        """Make ``spec``'s tunnel run; return ``True`` when this call started it.

        Raises:
            SshError: When the tunnel cannot be brought up; the message carries
                ssh's own last diagnostic line, never forwarded traffic.
        """

        with self._locked(spec.key):
            record = self._record(spec.key)
            if self._healthy(spec, record):
                return False
            if record is not None and self._alive(record):
                self._terminate(int(record["pid"]))
            record_path, _, log_path = self._paths(spec.key)
            with log_path.open("ab") as log:
                os.chmod(log_path, 0o600)
                process = self.popen(
                    list(spec.argv()),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    start_new_session=True,
                    close_fds=True,
                    env=provider_cli_env(),
                )
            self._children[int(process.pid)] = process
            document = {
                "pid": int(process.pid),
                "fingerprint": spec.fingerprint,
                "first_forward": spec.forwards[0].spec,
                "entry": spec.entry.target,
                "ports": [forward.local_port for forward in spec.forwards],
            }
            temporary = record_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(document), encoding="utf-8")
            os.chmod(temporary, 0o600)
            temporary.replace(record_path)
            deadline = self.clock() + self.startup_timeout_s
            while True:
                if process.poll() is not None:
                    raise SshError(f"ssh tunnel to {spec.entry.target} exited: {self._last_log_line(log_path)}")
                if all(self.port_open(forward.bind, forward.local_port, 1.0) for forward in spec.forwards):
                    return True
                if self.clock() >= deadline:
                    self._terminate(int(process.pid))
                    raise SshError(
                        f"ssh tunnel to {spec.entry.target} did not open its ports within "
                        f"{self.startup_timeout_s:.0f}s: {self._last_log_line(log_path)}"
                    )
                self.sleep(0.2)

    def stop(self, key: str) -> bool:
        """Stop the recorded tunnel for ``key``; ``False`` when none was running."""

        with self._locked(key):
            record = self._record(key)
            record_path, _, _ = self._paths(key)
            stopped = False
            if record is not None and self._alive(record):
                self._terminate(int(record["pid"]))
                stopped = True
            with contextlib.suppress(FileNotFoundError):
                record_path.unlink()
            return stopped

    def _terminate(self, pid: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
        child = self._children.pop(pid, None)
        if child is not None:
            with contextlib.suppress(Exception):
                child.wait(timeout=5)

    @staticmethod
    def _last_log_line(path: Path) -> str:
        try:
            lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
        except OSError:
            return "no diagnostic"
        return lines[-1][:300] if lines else "no diagnostic"


__all__ = (
    "LocalForward",
    "SshDestination",
    "SshError",
    "SshOptions",
    "SshRunner",
    "SshTunnelManager",
    "SshTunnelSpec",
    "free_local_ports",
    "port_is_open",
    "probe_tcp",
    "run_ssh",
    "scp_argv",
    "ssh_argv",
)
