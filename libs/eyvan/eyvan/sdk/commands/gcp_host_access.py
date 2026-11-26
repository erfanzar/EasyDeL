# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""How Eyvan reaches each host of an attached slice to install its native worker.

Two ways exist today and the connector does not care which it holds:
``gcloud compute tpus tpu-vm ssh/scp`` for nodes the operator's own credential can
see (defined beside the connector, which owns the gcloud runner), and plain
OpenSSH for nodes the operator can only log into. Plain SSH reaches the entry
host directly and every
other host of the slice through it, over the slice's internal network, so only
the one SSH login the operator already has is required.
"""

from __future__ import annotations

import subprocess
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from eyvan._internal.json_types import is_json_object
from eyvan._internal.ssh import (
    SshDestination,
    SshOptions,
    SshRunner,
    free_local_ports,
    probe_tcp,
    run_ssh,
    scp_argv,
    ssh_argv,
)
from eyvan.capacity.providers.gcp.tpu_nodes import OPTION_TUNNEL_PORTS, TpuNode
from eyvan.records import NativeContainerConfig, NativeRuntimeConfig

if tp.TYPE_CHECKING:
    from eyvan.capacity.autoscale.models import AutoscaleDefaults


def resolve_attach_runtime(
    *,
    defaults: AutoscaleDefaults | None,
    python: str | None = None,
    init_commands: tp.Sequence[str] = (),
    container_image: str | None = None,
    local_python: str | None = None,
) -> NativeRuntimeConfig:
    """Return the host setup an attached TPU gets: the same profile ``eyvan up`` uses, plus flags.

    The interpreter falls back to this machine's own ``major.minor``: a callable
    only unpickles on the interpreter that pickled it, so a worker on the image's
    Python fails every callable at dispatch, on the accelerator.
    """

    import sys

    selected_python = (
        python
        or (defaults.runtime_python if defaults is not None else None)
        or local_python
        or f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    image = container_image or (defaults.runtime_container_image if defaults is not None else None)
    # The same block generated fleets state (eyvan/capacity/autoscale/compiler.py):
    # the OCI executor refuses every TPU workload launched unprivileged, so an
    # image given for an attached TPU without it could never run a TPU job.
    container = (
        NativeContainerConfig(
            image=image,
            pull_policy="if-not-present",
            working_dir="/var/lib/eyvan/workload",
            privileged=True,
        )
        if image
        else None
    )
    return NativeRuntimeConfig(
        init_commands=(*(defaults.runtime_init_commands if defaults is not None else ()), *init_commands),
        container=container,
        python=selected_python,
    )


def runtime_document(runtime: NativeRuntimeConfig) -> dict[str, tp.Any]:
    """Return the fleet-YAML ``runtime`` block, omitting defaults; empty when nothing is declared."""

    return runtime.model_dump(mode="json", exclude_defaults=True)


@dataclass(frozen=True, slots=True)
class WorkerLayout:
    """Where the native worker's files live on an attached host.

    Attributes:
        staging_dir: Directory the worker wheel is copied into before install.
        state_dir: The native worker's state directory.
        secret_file: Root-owned file holding the worker's derived credential.
    """

    staging_dir: str = "/tmp"
    state_dir: str = "/var/lib/eyvan/native-worker"
    secret_file: str = "/etc/eyvan-native/attached-worker-secret"

    def __post_init__(self) -> None:
        for name in ("staging_dir", "state_dir", "secret_file"):
            value = getattr(self, name)
            if not value.startswith("/") or ".." in value or any(c.isspace() or c in "'\"`$;&|<>\\" for c in value):
                raise ValueError(f"worker layout {name} must be an absolute path without shell syntax")


@dataclass(frozen=True)
class GcloudHostAccess:
    """``gcloud compute tpus tpu-vm ssh/scp`` for nodes the operator's credential can read.

    Argv starts with ``gcloud`` so plans print the real command; :meth:`run`
    strips it because the gcloud runner supplies the executable itself.
    """

    node: TpuNode
    internal: bool
    runner: SshRunner

    @property
    def host_count(self) -> int:
        return len(self.node.hosts)

    def _args(self) -> tuple[str, ...]:
        return ("--zone", self.node.zone, "--project", self.node.project, *(("--internal-ip",) if self.internal else ()))

    def copy_argvs(self, local: str, remote_dir: str) -> tuple[tuple[str, ...], ...]:
        # One command per worker, never --worker=all: gcloud refuses a multi-worker
        # scp/ssh unless its key is loaded in ssh-agent ("SSH Key is not present in
        # the SSH agent"), while a single-worker command reads the key file itself.
        # The caller already runs these in parallel, exactly like exec_argv.
        target = f"{self.node.name}:{remote_dir.rstrip('/')}/"
        return tuple(
            ("gcloud", "compute", "tpus", "tpu-vm", "scp", local, target, *self._args(), f"--worker={worker}", "--quiet")
            for worker in range(self.host_count)
        )

    def exec_argv(self, worker: int, command: str) -> tuple[str, ...]:
        return (
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            self.node.name,
            *self._args(),
            f"--worker={worker}",
            "--quiet",
            "--command",
            command,
        )

    def run(self, argv: tp.Sequence[str], *, stdin_text: str | None, timeout: float) -> subprocess.CompletedProcess[str]:
        command = tuple(argv[1:]) if argv and argv[0] == "gcloud" else tuple(argv)
        return self.runner(command, stdin_text=stdin_text, timeout=timeout)


def entry_index(node: TpuNode, destination: SshDestination, reported_worker: int | None) -> int:
    """Return which worker an SSH destination is: by address first, then by what the host said."""

    address = destination.ip
    if address is not None:
        for index, host in enumerate(node.hosts):
            if address in (host.external_ip, host.internal_ip):
                return index
    if reported_worker is not None and 0 <= reported_worker < len(node.hosts):
        return reported_worker
    raise ValueError(f"cannot tell which host of TPU {node.name!r} {destination.target} is")


def choose_endpoint_mode(
    node: TpuNode,
    *,
    port: int,
    probe: tp.Callable[[str, int, float], str] = probe_tcp,
    timeout_s: float = 5.0,
) -> str:
    """Decide how this machine reaches workers it does not control the firewall for.

    Direct public addressing needs every host to have an external IP and the
    worker port to be admitted from here. A connection that is refused (host
    reached, nothing listening yet) proves the port is admitted; a timeout means
    a firewall drops it, so the workers are reached through an SSH tunnel instead.
    """

    externals = [host.external_ip for host in node.hosts]
    if externals and all(externals):
        verdicts = {probe(tp.cast(str, address), port, timeout_s) for address in externals[:1]}
        if "filtered" not in verdicts:
            return "public"
    return "tunnel"


def tunnel_ports_for(fleet_path: Path, count: int) -> tuple[int, ...]:
    """Reuse the ports a previous connect recorded for this fleet, else ask the OS for new ones.

    Stable ports keep a rerun's fleet document identical, and keep the tunnel the
    cell agent is already holding open the one it should hold.
    """

    try:
        document = yaml.safe_load(fleet_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        document = None
    pools = document.get("pools") if isinstance(document, dict) else None
    if isinstance(pools, dict):
        for pool in pools.values():
            options = pool.get("provider_options") if isinstance(pool, dict) else None
            ports = options.get(OPTION_TUNNEL_PORTS) if is_json_object(options) else None
            if (
                isinstance(ports, list)
                and len(ports) == count
                and all(isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535 for port in ports)
            ):
                return tuple(tp.cast(list[int], ports))
    return free_local_ports(count)


class HostAccess(tp.Protocol):
    """Build and run the per-host commands of one attachment."""

    @property
    def host_count(self) -> int:
        """Number of hosts, in worker order."""

    def copy_argvs(self, local: str, remote_dir: str) -> tuple[tuple[str, ...], ...]:
        """Return the commands that copy ``local`` into ``remote_dir`` on every host."""

    def exec_argv(self, worker: int, command: str) -> tuple[str, ...]:
        """Return the command that runs ``command`` on host ``worker``."""

    def run(self, argv: tp.Sequence[str], *, stdin_text: str | None, timeout: float) -> subprocess.CompletedProcess[str]:
        """Run one command this access built."""


@dataclass(frozen=True)
class SshHostAccess:
    """Plain OpenSSH: the entry host directly, every other host through it.

    Attributes:
        entry: The destination the operator can log into.
        internal_ips: Every host's internal address, in worker order.
        entry_index: Which worker the entry destination is.
        options: SSH timings and policy.
        runner: Command runner; injected by tests.
    """

    entry: SshDestination
    internal_ips: tuple[str, ...]
    entry_index: int = 0
    options: SshOptions = field(default_factory=SshOptions)
    runner: SshRunner = run_ssh

    def __post_init__(self) -> None:
        if not self.internal_ips:
            raise ValueError("SSH host access needs at least one host")
        if not 0 <= self.entry_index < len(self.internal_ips):
            raise ValueError("entry_index must name one of the hosts")

    @property
    def host_count(self) -> int:
        return len(self.internal_ips)

    def _route(self, worker: int) -> tuple[SshDestination, SshDestination | None]:
        if worker == self.entry_index:
            return self.entry, None
        return self.entry.with_host(self.internal_ips[worker]), self.entry

    def copy_argvs(self, local: str, remote_dir: str) -> tuple[tuple[str, ...], ...]:
        argvs = []
        for worker in range(self.host_count):
            destination, jump = self._route(worker)
            argvs.append(scp_argv(Path(local), destination, remote_dir, options=self.options, jump=jump))
        return tuple(argvs)

    def exec_argv(self, worker: int, command: str) -> tuple[str, ...]:
        destination, jump = self._route(worker)
        return ssh_argv(destination, command, options=self.options, jump=jump)

    def run(self, argv: tp.Sequence[str], *, stdin_text: str | None, timeout: float) -> subprocess.CompletedProcess[str]:
        return self.runner(argv, stdin_text=stdin_text, timeout=timeout)


__all__ = (
    "GcloudHostAccess",
    "HostAccess",
    "SshHostAccess",
    "WorkerLayout",
    "choose_endpoint_mode",
    "entry_index",
    "resolve_attach_runtime",
    "runtime_document",
    "tunnel_ports_for",
)
