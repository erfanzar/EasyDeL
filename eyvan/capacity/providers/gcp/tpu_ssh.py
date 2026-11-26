# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read a Cloud TPU VM node through SSH to one of its own hosts.

An operator can often SSH into a TPU in a project their own credential cannot
list. The host itself still
knows everything Eyvan needs: the metadata server names the node, its zone and
project, and the host's ``gcloud`` -- running as the TPU's own service account --
returns the same node document the TPU API returns to its owner. Parsing that
document with :func:`parse_tpu_node` keeps one definition of what a node is.

The reader exposes ``get_node`` with the same contract as :class:`TpuNodeApi`
(``None`` only for a definitive NOT_FOUND), so the attach driver can observe an
SSH-reached node exactly as it observes one it reads over REST.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
import typing as tp
from dataclasses import dataclass, field

from eyvan._internal.ssh import (
    LocalForward,
    SshDestination,
    SshOptions,
    SshRunner,
    SshTunnelSpec,
    run_ssh,
    ssh_argv,
)
from eyvan.capacity.providers.gcp.tpu_nodes import TpuApiError, TpuNode, TpuNodeParseError, parse_tpu_node

#: The GCE metadata server every Compute and TPU VM answers on. A platform
#: constant, not a configuration choice.
METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"

#: ssh's own exit status for "could not connect or authenticate".
_SSH_UNREACHABLE = 255
_COMMAND_NOT_FOUND = 127


@dataclass(frozen=True, slots=True)
class HostTpuIdentity:
    """What one TPU VM host says about the node it belongs to."""

    project: str
    zone: str
    name: str
    accelerator_type: str | None
    worker_id: int | None
    preemptible: bool | None


@dataclass(frozen=True, slots=True)
class HostResources:
    """CPU and memory one host reports about itself."""

    cpus: float
    memory_gib: float


def _identity_command(timeout_s: int) -> str:
    fetch = shlex.join(("curl", "-sf", "-m", str(timeout_s), "-H", "Metadata-Flavor:Google"))
    return "; ".join(
        (
            f"M={shlex.quote(fetch)}",
            f'echo "project=$($M {METADATA_ROOT}/project/project-id 2>/dev/null)"',
            f'echo "preemptible=$($M {METADATA_ROOT}/instance/scheduling/preemptible 2>/dev/null)"',
            f"$M {METADATA_ROOT}/instance/attributes/tpu-env 2>/dev/null || true",
        )
    )


def parse_host_identity(text: str) -> HostTpuIdentity | None:
    """Parse the identity probe; ``None`` when the host is not a TPU VM (no ``tpu-env``)."""

    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if "=" in line and line.split("=", 1)[0] in ("project", "preemptible"):
            key, value = line.split("=", 1)
            values[key] = value.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            values[key.strip().upper()] = value.strip().strip("'\"")
    name, zone, project = values.get("NODE_ID"), values.get("ZONE"), values.get("project")
    if not name or not zone or not project:
        return None
    worker = values.get("WORKER_ID", "")
    preemptible = values.get("preemptible", "").upper()
    return HostTpuIdentity(
        project=project,
        zone=zone,
        name=name,
        accelerator_type=(values.get("ACCELERATOR_TYPE") or "").lower() or None,
        worker_id=int(worker) if worker.isdecimal() else None,
        preemptible=True if preemptible == "TRUE" else False if preemptible == "FALSE" else None,
    )


@dataclass
class SshTpuNodeReader:
    """Read one TPU node by running the host's own ``gcloud`` over SSH.

    Attributes:
        entry: Any host of the slice the operator can SSH into.
        options: SSH timings and policy.
        runner: Command runner; injected by tests.
        cache_ttl_s: How long one successful read is reused. A cell agent
            observes every few seconds, and each read is an SSH round trip plus
            a provider call made with someone else's service account.
        metadata_timeout_s: Per-request timeout for the metadata server.
    """

    entry: SshDestination
    options: SshOptions = field(default_factory=SshOptions)
    runner: SshRunner = run_ssh
    cache_ttl_s: float = 15.0
    metadata_timeout_s: int = 5
    clock: tp.Callable[[], float] = time.monotonic
    _cache: dict[tuple[str, str, str], tuple[float, TpuNode | None]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _run(self, command: str) -> subprocess.CompletedProcess[str]:
        argv = ssh_argv(self.entry, command, options=self.options)
        try:
            return self.runner(argv, stdin_text=None, timeout=self.options.command_timeout_s)
        except (OSError, subprocess.SubprocessError) as exc:
            raise TpuApiError(f"could not run ssh to {self.entry.target}: {type(exc).__name__}", retryable=True) from exc

    @staticmethod
    def _last_error(result: subprocess.CompletedProcess[str]) -> str:
        lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
        errors = [line for line in lines if line.startswith("ERROR:")]
        return (errors or lines or ["no diagnostic"])[-1][:300]

    def identity(self) -> HostTpuIdentity | None:
        """Return the node this host belongs to, or ``None`` when it is not a TPU VM."""

        result = self._run(_identity_command(self.metadata_timeout_s))
        if result.returncode == _SSH_UNREACHABLE:
            raise TpuApiError(f"ssh to {self.entry.target} failed: {self._last_error(result)}", retryable=True)
        if result.returncode != 0:
            raise TpuApiError(
                f"metadata probe on {self.entry.target} failed: {self._last_error(result)}", retryable=False
            )
        return parse_host_identity(result.stdout)

    def get_node(self, project: str, zone: str, name: str) -> TpuNode | None:
        """Return the node document as the host's gcloud reads it; ``None`` only on NOT_FOUND."""

        key = (project, zone, name)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and self.clock() - cached[0] < self.cache_ttl_s:
                return cached[1]
        command = shlex.join(
            ("gcloud", "compute", "tpus", "tpu-vm", "describe", name, "--zone", zone, "--project", project)
        )
        result = self._run(f"{command} --format=json")
        if result.returncode == _SSH_UNREACHABLE:
            raise TpuApiError(f"ssh to {self.entry.target} failed: {self._last_error(result)}", retryable=True)
        if result.returncode == _COMMAND_NOT_FOUND:
            raise TpuApiError(f"gcloud is not installed on {self.entry.target}", retryable=False)
        if result.returncode != 0:
            detail = self._last_error(result)
            if "NOT_FOUND" in detail or "was not found" in detail:
                node = None
            else:
                raise TpuApiError(
                    f"gcloud on {self.entry.target} could not describe TPU {name!r}: {detail}",
                    retryable="UNAVAILABLE" in detail or "DEADLINE" in detail,
                )
        else:
            try:
                document = json.loads(result.stdout)
            except ValueError as exc:
                raise TpuNodeParseError(f"gcloud on {self.entry.target} returned invalid JSON") from exc
            node = parse_tpu_node(document, project=project)
        with self._lock:
            self._cache[key] = (self.clock(), node)
        return node


def attach_tunnel_spec(
    *,
    management_uid: str,
    entry: SshDestination,
    internal_ips: tp.Sequence[str],
    worker_port: int,
    local_ports: tp.Sequence[int],
    options: SshOptions | None = None,
) -> SshTunnelSpec:
    """Return the one tunnel an attached slice uses: local port ``i`` -> host ``i``'s worker.

    Defined once so the CLI that opens the tunnel and the cell agent that keeps it
    alive build byte-identical specs (same fingerprint), and neither replaces the
    other's process.
    """

    if len(internal_ips) != len(local_ports) or not internal_ips:
        raise ValueError("an attach tunnel needs exactly one local port per host")
    return SshTunnelSpec(
        key=f"attach-{management_uid}",
        entry=entry,
        forwards=tuple(
            LocalForward(local_port=int(port), remote_host=address, remote_port=worker_port)
            for address, port in zip(internal_ips, local_ports, strict=True)
        ),
        options=options or SshOptions(),
    )


__all__ = (
    "METADATA_ROOT",
    "HostResources",
    "HostTpuIdentity",
    "SshTpuNodeReader",
    "attach_tunnel_spec",
    "parse_host_identity",
)
