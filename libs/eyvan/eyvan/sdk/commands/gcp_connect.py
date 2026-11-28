# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inert attachment plans and an injected, step-labelled execution boundary.

TPU attachment installs a worker but never creates or destroys hardware. The
SSH route deliberately adopts an already-prepared worker: onprem-ssh owns its
fenced claim, not software installation or cloud credential derivation.
"""

from __future__ import annotations

import functools
import ipaddress
import math
import os
import re
import shlex
import subprocess
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

import eyvan.sdk.commands.head as head_commands
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.ssh import (
    SshDestination,
    SshError,
    SshOptions,
    SshTunnelManager,
    SshTunnelSpec,
    probe_tcp,
    run_ssh,
    ssh_argv,
)
from eyvan.capacity.fleet_config import EyvanFleetConfig, validate_fleet_name
from eyvan.capacity.hardware_catalog import shape_for
from eyvan.capacity.providers.gcp.native_worker_ingress import GcloudNativeWorkerPublicIngress
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile, GcpNativeRuntimeProfileStore
from eyvan.capacity.providers.gcp.runtime.supervisor import runtime_profile_path_for
from eyvan.capacity.providers.gcp.tpu_nodes import (
    ATTACH_CAPACITY_BACKEND,
    ENDPOINT_MODES,
    OPTION_SSH_ENTRY,
    OPTION_TUNNEL_PORTS,
    OPTION_WORKER_PORT,
    TpuApiError,
    TpuNode,
    TpuNodeApi,
    TpuNodeParseError,
    attach_fleet_name,
    attach_management_uid,
    validate_node_name,
)
from eyvan.capacity.providers.gcp.tpu_ssh import SshTpuNodeReader, attach_tunnel_spec
from eyvan.config import EyvanSettings
from eyvan.records import NativeRuntimeConfig
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.fleets import FleetCommandService
from eyvan.sdk.commands.gcp_host_access import (
    GcloudHostAccess,
    HostAccess,
    SshHostAccess,
    WorkerLayout,
    choose_endpoint_mode,
    entry_index,
    resolve_attach_runtime,
    runtime_document,
    tunnel_ports_for,
)
from eyvan.sdk.commands.head import HeadCommandRunner
from eyvan.sdk.commands.runtime import _deploy_native_runtime, native_worker_wheel_builder
from eyvan.sdk.commands.state import CliSession
from eyvan.worker.runtime.native_bootstrap import (
    build_native_worker_startup_script,
    derive_native_worker_token,
    render_native_runtime_setup,
)

_WHEEL = "<local-worker-wheel>"
_REMOTE_WHEEL = "<remote-worker-wheel>"
_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_ZONE = re.compile(r"^[a-z]+[0-9]?-[a-z]+[0-9]+-[a-z][a-z0-9]*$")
#: Diagnostic lines ssh/scp/gcloud print about themselves; anything else a remote
#: command printed is withheld, since a remote side may echo what it received.
_TOOL_DIAGNOSTIC = re.compile(r"^(ERROR:|ssh:|scp:|kex_exchange|Connection |.*Permission denied|.*Host key)")


class AttachError(RuntimeError):
    """An attachment step failed without exposing subprocess input or output."""

    def __init__(self, step: str, detail: str):
        self.step = step
        super().__init__(f"eyvan gcp connect {step.replace('_', ' ')} failed: {detail}")


class _FleetRefused(AttachError):
    """The plane or agent definitively refused the fleet; waiting longer cannot help."""


#: Cells the plane may replace before connect stops waiting. Bound the wait so a
#: repeatedly refused attachment cannot keep the connector waiting indefinitely.
_MAX_REPLACED_CELLS = 3

#: Provider failure codes that are worth waiting through; anything else is final.
_TRANSIENT_FAILURE_CODES = frozenset({"provider_unavailable", "stockout"})


class _QuietPhase:
    """No-op progress phase used for JSON output and tests."""

    def __enter__(self) -> _QuietPhase:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def update(self, text: str) -> None:
        """Ignore live detail."""


def _quiet_progress(label: str) -> tp.ContextManager[tp.Any]:
    return _QuietPhase()


def _resolved_local_agent_endpoint(profile_path: Path) -> ControlPlaneEndpoint:
    """Resolve the local loopback cell agent URL for ``profile_path`` (supporting custom-port planes)."""

    from eyvan._internal.platform import load_platform_config
    from eyvan.server.local.daemon import (
        DEFAULT_LOCAL_CELL_AGENT_URL,
        LocalDaemonError,
        LocalDaemonPaths,
        LocalDaemonTarget,
        process_is_alive,
        read_owner_only_record,
    )

    try:
        platform = load_platform_config(EyvanSettings.from_env().platform_config_path)
    except ValueError:
        platform = None
    if platform is not None and platform.default_agent_endpoint:
        try:
            candidate = ControlPlaneEndpoint.parse(platform.default_agent_endpoint)
        except ValueError:
            candidate = None
        if candidate is not None and LocalDaemonTarget.from_endpoint(candidate) is not None:
            return candidate

    target_profile = str(profile_path.expanduser().resolve())
    directory = LocalDaemonPaths().resolved_directory
    for pid_path in sorted(directory.glob("cell-agent-*.pid")):
        try:
            record = read_owner_only_record(pid_path)
        except (LocalDaemonError, OSError):
            continue
        if (
            record is not None
            and record.runtime_profile_path == target_profile
            and LocalDaemonTarget.from_endpoint(record.endpoint) is not None
            and process_is_alive(record.pid)
        ):
            return record.endpoint
    return ControlPlaneEndpoint.parse(DEFAULT_LOCAL_CELL_AGENT_URL)


def _runtime_needs_deploy(
    endpoint: ControlPlaneEndpoint,
    *,
    require_current: bool = False,
    agent_endpoint: ControlPlaneEndpoint | None = None,
) -> str | None:
    """Return why the local runtime cannot serve attached TPUs, or ``None`` when it can.

    The cell agent is kept at the build the local plane was launched from, so an
    agent from before ``gcp-attach`` existed rejects every attached cell with
    HTTP 400 and the plane repeatedly replaces the cell. Ask the agent
    directly; when it cannot be asked, fall back to comparing build identities.
    """

    from eyvan._internal.build_identity import build_identity
    from eyvan.capacity.providers.gcp.runtime.supervisor import HttpCellAgentCapabilitiesProbe
    from eyvan.server.local.daemon import serving_source_digest

    profile_path = runtime_profile_path_for(endpoint)
    try:
        profile = GcpNativeRuntimeProfileStore(profile_path).load()
    except FileNotFoundError:
        return None
    resolved_agent = agent_endpoint if agent_endpoint is not None else _resolved_local_agent_endpoint(profile_path)
    capabilities = HttpCellAgentCapabilitiesProbe(timeout_s=3.0)(resolved_agent, profile.agent_shared_secret)
    if capabilities is not None and capabilities.capacity_backends:
        if ATTACH_CAPACITY_BACKEND not in capabilities.capacity_backends:
            return "the local cell agent runs an Eyvan build without attached-TPU support"
        if not require_current:
            return None
    serving = serving_source_digest(endpoint)
    caller = build_identity().source_digest
    if serving is not None and caller is not None and serving != caller:
        # Capabilities name backends, not modes: an agent that knows gcp-attach can
        # still predate SSH-read nodes and tunnels, so those plans need this build.
        return "the local control plane runs an older Eyvan build than this CLI"
    return None


def _deploy_runtime(endpoint: ControlPlaneEndpoint) -> None:
    """Redeploy the local plane, agent and worker wheel from this Eyvan, as `eyvan runtime deploy` does."""

    _deploy_native_runtime(endpoint=endpoint, profile_path=runtime_profile_path_for(endpoint))


def _run_gcloud(
    args: tp.Sequence[str],
    *,
    stdin_text: str | None,
    timeout: float,
    account: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Reuse head's scrubbed runner, resolving it at call time for embedding/tests."""

    if account is None:
        return head_commands._run_gcloud(args, stdin_text=stdin_text, timeout=timeout)
    from eyvan._internal.subprocess_env import provider_cli_env

    return subprocess.run(
        ["gcloud", *args],
        input=stdin_text,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=provider_cli_env(overrides={"CLOUDSDK_CORE_ACCOUNT": account}),
    )


def _public_ingress(profile: GcpNativeRuntimeProfile, runner: HeadCommandRunner) -> GcloudNativeWorkerPublicIngress:
    """Reuse the profile-owned /32 firewall manager with the same injected runner."""

    return GcloudNativeWorkerPublicIngress(
        network_tag=profile.native_worker_public_ingress_tag,
        runner=lambda argv: runner(argv, stdin_text=None, timeout=300.0),
    )


def _worker_health(endpoint: str, token: str | None) -> None:
    """Check native-worker health without proxy credentials or redirects."""

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    route = "/api/v1/health" if token else "/healthz"
    with httpx.Client(timeout=5.0, trust_env=False, follow_redirects=False) as client:
        response = client.get(f"{endpoint}{route}", headers=headers)
    if response.status_code != 200:
        raise ValueError("worker health did not return HTTP 200")
    body = response.json()
    if not is_json_object(body) or body.get("status") != "ok" or body.get("runtime") != "native-worker":
        raise ValueError("endpoint did not identify an Eyvan native worker")


_MEMTOTAL = re.compile(r"^MemTotal:\s+(\d+)\s+kB$")


def parse_host_resources(text: str) -> tuple[float, float]:
    """Parse ``nproc`` and ``/proc/meminfo`` MemTotal output into ``(cpus, memory_gib)``.

    Memory is rounded down to a tenth of a GiB so Eyvan never advertises more
    than the host has.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].isdigit():
        raise ValueError("host did not report nproc and MemTotal")
    match = _MEMTOTAL.fullmatch(lines[1])
    cpus = int(lines[0])
    if cpus < 1 or match is None or int(match.group(1)) <= 0:
        raise ValueError("host reported unusable CPU or memory figures")
    return float(cpus), math.floor(int(match.group(1)) / (1024 * 1024) * 10) / 10


def _run_ssh(args: tp.Sequence[str], *, stdin_text: str | None, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ssh/scp; resolved at call time so tests and embedders can replace it."""

    return run_ssh(args, stdin_text=stdin_text, timeout=timeout)


def _ssh_node_reader(destination: SshDestination, options: SshOptions) -> SshTpuNodeReader:
    """Return a reader for the TPU behind ``destination``; replaceable in tests."""

    return SshTpuNodeReader(
        destination,
        options=options,
        runner=lambda args, *, stdin_text, timeout: _run_ssh(args, stdin_text=stdin_text, timeout=timeout),
    )


def _port_probe(host: str, port: int, timeout: float) -> str:
    """Classify reachability of ``host:port`` from here; replaceable in tests."""

    return probe_tcp(host, port, timeout)


def _probe_host_resources(host: str, options: SshOptions | None = None) -> tuple[float, float]:
    """Read a ``[user@]host``'s CPU count and memory over non-interactive SSH."""

    selected = options or SshOptions()
    result = _run_ssh(
        ssh_argv(SshDestination.parse(host), "nproc; grep -m1 '^MemTotal:' /proc/meminfo", options=selected),
        stdin_text=None,
        timeout=selected.command_timeout_s,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise ValueError(f"ssh exited {result.returncode}: {detail[-1][:200] if detail else 'no output'}")
    return parse_host_resources(result.stdout)


def _endpoint(host: str, port: int) -> str:
    """Format IPv4, IPv6 and DNS host names as an HTTP endpoint."""

    host = host.strip("[]")
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


@dataclass(frozen=True, slots=True)
class AttachAction:
    """HeadAction-style plan data, including non-gcloud local/API boundaries."""

    step: str
    argv: tuple[str, ...]
    summary: str
    mutates: bool = False
    stdin: bool = False

    def as_document(self) -> JsonObject:
        """Return the complete safe argv, never the bytes sent on stdin."""

        return {
            "step": self.step,
            "argv": list(self.argv),
            "command": shlex.join(self.argv),
            "summary": self.summary,
            "mutates": self.mutates,
            "stdin": self.stdin,
        }


@dataclass(frozen=True, slots=True)
class AttachPlan:
    """One immutable, printable plan with no credential or wheel-build side effects."""

    fleet: str
    fleet_path: Path
    fleet_yaml: str
    endpoints: tuple[str, ...]
    actions: tuple[AttachAction, ...]
    node: TpuNode | None = None
    endpoint_mode: str = "public"
    limitations: tuple[str, ...] = ()
    #: How each host is reached for installation (gcloud or plain SSH).
    access: HostAccess | None = None
    #: The SSH destination the node was discovered through, when not read via the API.
    ssh_entry: SshDestination | None = None
    #: Local ports, one per host, when workers are reached through an SSH tunnel.
    tunnel_ports: tuple[int, ...] = ()

    def as_document(self) -> JsonObject:
        """Return a machine-readable dry-run document."""

        return {
            "dry_run": True,
            "fleet": self.fleet,
            "fleet_path": str(self.fleet_path),
            "fleet_yaml": self.fleet_yaml,
            "endpoints": list(self.endpoints),
            "actions": [action.as_document() for action in self.actions],
            "limitations": list(self.limitations),
        }

    def render_lines(self) -> tuple[str, ...]:
        """Describe every planned boundary and the generated public fleet YAML."""

        lines = [f"Would attach fleet {self.fleet}; no changes made."]
        for index, action in enumerate(self.actions, 1):
            marker = "change" if action.mutates else "read"
            lines.extend((f"  {index}. [{marker}] {action.step}: {action.summary}", f"      {shlex.join(action.argv)}"))
            if action.stdin:
                lines.append("      stdin: <redacted / supplied only during execution>")
        lines.extend(f"Limitation: {limitation}" for limitation in self.limitations)
        lines.extend((f"Fleet YAML ({self.fleet_path}):", self.fleet_yaml))
        return tuple(lines)


class _QuietRenderer(JsonRenderer):
    """Keep FleetCommandService's intermediate response out of final JSON stdout."""

    def emit(self, value: tp.Any) -> None:
        pass


@dataclass
class GcpConnector:
    """Plan independently; execute only after the command has obtained consent."""

    session: CliSession
    api: TpuNodeApi | None = None
    runner: HeadCommandRunner | None = None
    attached_dir: Path = field(default_factory=lambda: Path("~/.eyvan/attached").expanduser())
    port: int = field(default_factory=lambda: EyvanSettings.from_env().native_worker_port)
    clock: tp.Callable[[], float] = time.monotonic
    sleep: tp.Callable[[float], None] = time.sleep
    #: Reads ``(cpus, memory_gib)`` from a plain SSH host; injected by tests.
    host_probe: tp.Callable[[str], tuple[float, float]] | None = None
    #: Wraps each visible step; the CLI supplies a spinner/checkmark printer.
    progress: tp.Callable[[str], tp.ContextManager[tp.Any]] = _quiet_progress
    #: Overrides for the local-runtime check and redeploy; resolved at call time otherwise.
    runtime_check: tp.Callable[..., str | None] | None = None
    runtime_deploy: tp.Callable[[ControlPlaneEndpoint], None] | None = None
    #: Where the worker's files go on each host.
    layout: WorkerLayout = field(default_factory=WorkerLayout)
    #: SSH timings/policy for installs and SSH node reads (the tunnel uses the
    #: agent's defaults, so both sides build the same tunnel).
    ssh_options: SshOptions = field(default_factory=SshOptions)
    #: How many hosts are copied to / installed on at once.
    install_concurrency: int = 8
    tunnels: SshTunnelManager | None = None
    #: Host setup for TPU hosts: interpreter, init commands, OCI image. The CLI
    #: fills it from the same defaults profile ``eyvan up`` derives fleets from.
    runtime: NativeRuntimeConfig = field(default_factory=lambda: resolve_attach_runtime(defaults=None))
    #: Optional gcloud account used for REST and gcloud calls without changing gcloud config.
    account: str | None = None

    def _gcloud_runner(self) -> HeadCommandRunner:
        if self.runner is not None:
            return self.runner
        if self.account is None:
            return _run_gcloud
        account = self.account
        return lambda args, *, stdin_text, timeout: _run_gcloud(
            args, stdin_text=stdin_text, timeout=timeout, account=account
        )

    def resolve(self, *, project: str, name: str, zone: str | None) -> TpuNode:
        """Resolve an exact incarnation, refusing ambiguous names and unknown zones."""

        try:
            validate_node_name(name)
            if not _PROJECT.fullmatch(project):
                raise ValueError("--project must be a valid GCP project id")
            if zone is not None and not _ZONE.fullmatch(zone):
                raise ValueError("--zone must be a valid GCP zone")
            api = self.api if self.api is not None else TpuNodeApi()
            if zone:
                node = api.get_node(project, zone, name)
            else:
                nodes, unreachable = api.list_nodes(project)
                matches = [node for node in nodes if node.name == name]
                if len(matches) > 1:
                    zones = ", ".join(sorted(node.zone for node in matches))
                    raise ValueError(f"TPU name {name!r} is ambiguous across zones: {zones}; pass --zone")
                if unreachable:
                    raise ValueError(
                        f"could not rule out an ambiguous TPU name in unreachable locations: {', '.join(unreachable)}; "
                        "pass --zone to read the intended node directly"
                    )
                node = matches[0] if matches else None
            if node is None:
                raise ValueError(f"TPU {name!r} was not found in {project}; use `eyvan gcp browse`")
            return node
        except Exception as exc:
            raise AttachError("resolve", str(exc)) from exc

    def host_access(
        self,
        node: TpuNode,
        *,
        endpoint_mode: str,
        ssh_entry: SshDestination | None = None,
        entry_worker: int | None = None,
    ) -> HostAccess:
        """Return how each host is reached for installation.

        A node the operator's credential can read is reached with ``gcloud``; a
        node found by logging into one of its hosts is reached with plain SSH, the
        entry host directly and every other host through it.
        """

        if ssh_entry is None:
            return GcloudHostAccess(node, internal=endpoint_mode != "public", runner=self._gcloud_runner())
        internal = node.endpoint_ips("internal")
        if internal is None:
            raise AttachError("resolve", f"TPU {node.name!r} does not report an internal address for every host")
        try:
            index = entry_index(node, ssh_entry, entry_worker)
        except ValueError as exc:
            raise AttachError("resolve", str(exc)) from exc
        return SshHostAccess(ssh_entry, internal, index, options=self.ssh_options, runner=_run_ssh)

    def _tunnel_spec_for(self, node: TpuNode, entry: SshDestination, ports: tp.Sequence[int]) -> SshTunnelSpec:
        """Return the exact tunnel the cell agent keeps alive for this claim (same fingerprint)."""

        internal = node.endpoint_ips("tunnel")
        if internal is None:
            raise AttachError("resolve", f"TPU {node.name!r} does not report an internal address for every host")
        return attach_tunnel_spec(
            management_uid=attach_management_uid(
                project=node.project, zone=node.zone, name=node.name, node_id=node.node_id
            ),
            entry=entry,
            internal_ips=internal,
            worker_port=self.port,
            local_ports=ports,
        )

    def plan_tpu(
        self,
        node: TpuNode,
        *,
        fleet: str | None = None,
        endpoint_mode: str = "public",
        ssh_entry: SshDestination | None = None,
        entry_worker: int | None = None,
    ) -> AttachPlan:
        """Build the complete TPU plan from a read-only observation (TPU API or SSH)."""

        if node.state != "READY":
            raise AttachError(
                "resolve", f"TPU {node.name!r} is {node.state}, not READY; wait for READY before connecting"
            )
        if node.health not in (None, "HEALTHY"):
            raise AttachError(
                "resolve", f"TPU {node.name!r} has health {node.health}; refuse to install on an unhealthy node"
            )
        if shape_for(node.accelerator_type) is None:
            raise AttachError(
                "resolve", f"unknown catalog shape {node.accelerator_type!r}; update Eyvan's hardware catalog"
            )
        if endpoint_mode not in ENDPOINT_MODES:
            raise AttachError("plan", f"--endpoint-mode must be one of {', '.join(ENDPOINT_MODES)}")
        if endpoint_mode == "tunnel" and ssh_entry is None:
            raise AttachError("plan", "tunnel endpoints need an SSH entry host; pass --ssh HOST")
        ips = node.endpoint_ips(endpoint_mode)
        if ips is None:
            raise AttachError("resolve", f"TPU {node.name!r} lacks {endpoint_mode} addresses for every host")
        name = fleet or attach_fleet_name(project=node.project, zone=node.zone, name=node.name)
        path = self._fleet_path(name)
        access = self.host_access(node, endpoint_mode=endpoint_mode, ssh_entry=ssh_entry, entry_worker=entry_worker)
        options: dict[str, tp.Any] = {
            "tpu_node": node.name,
            "tpu_node_id": node.node_id,
            "endpoint_mode": endpoint_mode,
        }
        tunnel_ports: tuple[int, ...] = ()
        if ssh_entry is not None:
            # The cell agent reads the node through the same host, and installs
            # used this worker port; both are recorded rather than assumed later.
            options[OPTION_SSH_ENTRY] = ssh_entry.target
            options[OPTION_WORKER_PORT] = self.port
        if endpoint_mode == "tunnel":
            tunnel_ports = tunnel_ports_for(path, len(ips))
            options[OPTION_TUNNEL_PORTS] = list(tunnel_ports)
            endpoints = tuple(f"http://127.0.0.1:{port}" for port in tunnel_ports)
        else:
            endpoints = tuple(_endpoint(ip, self.port) for ip in ips)
        document = {
            "version": 1,
            "name": name,
            "provider": {"type": "gcp", "project": node.project, "region": node.region, "zone": node.zone},
            "executor": "eyvan-native",
            "capacity_class": node.capacity_class.value,
            "pools": {
                node.accelerator_type: {
                    "kind": "tpu",
                    "accelerator_type": node.accelerator_type,
                    "capacity_backend": ATTACH_CAPACITY_BACKEND,
                    "min_workers": 1,
                    "max_workers": 1,
                    "provider_options": options,
                    **({"runtime": runtime_document(self.runtime)} if runtime_document(self.runtime) else {}),
                }
            },
        }
        args = self._tpu_args(node)
        actions: list[AttachAction] = [
            AttachAction("ssh_check", access.exec_argv(worker, "true"), f"Host {worker}: check SSH login", False)
            for worker in range(access.host_count)
        ]
        if endpoint_mode == "public" and ssh_entry is None:
            # Only a project the operator's own credential reads gets a firewall
            # rule; a node reached over SSH sits in somebody else's network.
            actions.extend(
                (
                    AttachAction(
                        "ingress",
                        (
                            "GcloudNativeWorkerPublicIngress.ensure",
                            node.project,
                            node.network or "default",
                            str(self.port),
                        ),
                        "Resolve laptop IPv4; describe/create/update only its managed /32 worker-port rule",
                        True,
                    ),
                    AttachAction(
                        "tag",
                        (
                            "gcloud",
                            "compute",
                            "tpus",
                            "tpu-vm",
                            "update",
                            node.name,
                            *args,
                            "--add-tags",
                            "<profile-worker-tag>",
                            "--quiet",
                        ),
                        "Add the ingress tag without replacing existing tags",
                        True,
                    ),
                )
            )
        actions.append(
            AttachAction(
                "wheel_build", ("native_worker_wheel_builder",), "Build this Eyvan's worker wheel locally", True
            )
        )
        copies = access.copy_argvs(_WHEEL, self.layout.staging_dir)
        for index, argv in enumerate(copies):
            summary = (
                "Copy the locally built wheel to every host"
                if len(copies) == 1
                else f"Host {index}: copy the locally built wheel"
            )
            actions.append(AttachAction("wheel_scp", argv, summary, True))
        # Parallel SSH children may share stdin. Send private input to each worker
        # separately so every host receives the whole token and installer.
        for worker in range(access.host_count):
            actions.extend(
                (
                    AttachAction(
                        "worker_secret",
                        access.exec_argv(worker, self._secret_command()),
                        f"Host {worker}: deliver token on stdin into a root-owned 0600 file",
                        True,
                        True,
                    ),
                    AttachAction(
                        "worker_install",
                        access.exec_argv(worker, "sudo -n bash -s"),
                        f"Host {worker}: install {_REMOTE_WHEEL}; configure/restart native-worker systemd unit",
                        True,
                        True,
                    ),
                )
            )
        if ssh_entry is not None and endpoint_mode == "tunnel":
            spec = self._tunnel_spec_for(node, ssh_entry, tunnel_ports)
            actions.append(
                AttachAction(
                    "tunnel",
                    spec.argv(),
                    f"Hold {len(tunnel_ports)} local port(s) open to every host's worker through {ssh_entry.target}; "
                    "the cell agent restarts it when it dies",
                    True,
                )
            )
        actions.extend(self._tail_actions(path, name, endpoints))
        return AttachPlan(
            name,
            path,
            yaml.safe_dump(document, sort_keys=False),
            endpoints,
            tuple(actions),
            node,
            endpoint_mode,
            access=access,
            ssh_entry=ssh_entry,
            tunnel_ports=tunnel_ports,
        )

    def find_tpu_by_address(
        self, host: str, *, project: str | None = None, zone: str | None = None
    ) -> tuple[TpuNode | None, str, tuple[str, ...]]:
        """Return the TPU VM owning ``host``'s IP, its endpoint mode, and search notes.

        Only IP literals are looked up: resolving a DNS name here would make a
        read-only lookup depend on the laptop's resolver. A project whose TPU API
        cannot be read is reported in the notes rather than ending the search,
        and a lookup that could not run at all is a note, never a silent "not a TPU".
        """

        destination = host.rsplit("@", 1)[-1].strip("[]")
        try:
            address = str(ipaddress.ip_address(destination))
        except ValueError:
            return None, "public", ()
        api = self.api if self.api is not None else TpuNodeApi()
        notes: list[str] = []
        try:
            projects = (project,) if project else tuple(item.project_id for item in api.list_projects())
        except (TpuApiError, TpuNodeParseError) as exc:
            return None, "public", (f"could not list GCP projects to check whether {address} is a TPU VM: {exc}",)
        found: dict[tuple[str, str, str], tuple[TpuNode, str]] = {}
        for project_id in projects:
            try:
                nodes, unreachable = api.list_nodes(project_id, zone=zone or "-")
            except (TpuApiError, TpuNodeParseError) as exc:
                notes.append(f"could not read TPU VMs in {project_id}: {exc}")
                continue
            if unreachable:
                notes.append(f"unreachable TPU locations in {project_id}: {', '.join(unreachable)}")
            for node in nodes:
                for tpu_host in node.hosts:
                    if address in (tpu_host.external_ip, tpu_host.internal_ip):
                        mode = "public" if tpu_host.external_ip == address else "internal"
                        found[(node.project, node.zone, node.name)] = (node, mode)
        if len(found) > 1:
            owners = ", ".join("/".join(key) for key in sorted(found))
            raise AttachError(
                "resolve", f"{address} belongs to more than one TPU VM ({owners}); use --tpu NAME --zone Z"
            )
        if found:
            node, mode = next(iter(found.values()))
            return node, mode, tuple(notes)
        return None, "public", tuple(notes)

    def discover_over_ssh(self, destination: SshDestination) -> tuple[TpuNode, int | None] | None:
        """Ask a host what it is: its TPU node and worker index, or ``None`` when it is no TPU VM.

        The host's metadata server names its node, and the host's own ``gcloud``
        (running as the TPU's service account) reads the node document, so a TPU in
        a project the operator's credential cannot list is described exactly as
        the API would describe it.
        """

        reader = _ssh_node_reader(destination, self.ssh_options)
        try:
            identity = reader.identity()
            if identity is None:
                return None
            node = reader.get_node(identity.project, identity.zone, identity.name)
        except (TpuApiError, TpuNodeParseError) as exc:
            raise AttachError("probe", f"could not read {destination.target} over SSH: {exc}") from exc
        if node is None:
            raise AttachError(
                "resolve",
                f"{destination.target} reports TPU {identity.name!r} in {identity.project}/{identity.zone}, "
                "but gcloud on it cannot find that node",
            )
        return node, identity.worker_id

    def endpoint_mode_for(self, node: TpuNode) -> str:
        """Probe whether this machine reaches the workers directly or needs an SSH tunnel."""

        return choose_endpoint_mode(node, port=self.port, probe=_port_probe)

    def plan_ssh(
        self,
        host: str,
        *,
        cpus: float | None = None,
        memory_gib: float | None = None,
        fleet: str | None = None,
        capacity_class: str = "reserved",
        region: str = "onprem",
        notes: tuple[str, ...] = (),
    ) -> AttachPlan:
        """Adopt one preinstalled CPU worker using onprem-ssh, reading its size over SSH."""

        import hashlib

        try:
            destination = SshDestination.parse(host)
        except ValueError as exc:
            raise AttachError(
                "plan", f"--ssh must be [user@]host without options, paths or shell syntax ({exc})"
            ) from exc
        if cpus is None or memory_gib is None:
            probe = self.host_probe if self.host_probe is not None else _probe_host_resources
            try:
                probed_cpus, probed_memory = probe(destination.target)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                raise AttachError(
                    "probe",
                    f"could not read CPU and memory over SSH from {destination.target} ({exc}); "
                    "check SSH access or pass --cpus and --memory-gib",
                ) from exc
            cpus = probed_cpus if cpus is None else cpus
            memory_gib = probed_memory if memory_gib is None else memory_gib
        if not math.isfinite(cpus) or not math.isfinite(memory_gib) or cpus <= 0 or memory_gib <= 0:
            raise AttachError("plan", "--ssh requires positive finite --cpus and --memory-gib describing the host")
        if capacity_class not in ("reserved", "on_demand"):
            raise AttachError("plan", "onprem-ssh never reports Spot; choose reserved or on_demand")
        name = fleet or f"ssh-{hashlib.sha256(destination.target.encode()).hexdigest()[:12]}"
        path = self._fleet_path(name)
        document = {
            "version": 1,
            "name": name,
            "provider": {"type": "onprem-ssh", "region": region},
            "executor": "eyvan-native",
            "capacity_class": capacity_class,
            "pools": {
                "host": {
                    "kind": "cpu",
                    "machine_type": "operator-owned",
                    "capacity_backend": "onprem-ssh",
                    "min_workers": 1,
                    "max_workers": 1,
                    "resources": {"CPU": cpus, "MEMORY_GIB": memory_gib},
                    "provider_options": {
                        "hosts": [destination.target],
                        "worker_port": self.port,
                        "state_dir": self.layout.state_dir,
                    },
                }
            },
        }
        endpoints = (_endpoint(destination.host, self.port),)
        return AttachPlan(
            name,
            path,
            yaml.safe_dump(document, sort_keys=False),
            endpoints,
            self._tail_actions(path, name, endpoints, health_path="/healthz"),
            limitations=(
                "SSH attachment requires an already-prepared native worker and an onprem-ssh agent with SSH access. "
                "It does not install software, create ingress, or distribute GCP credentials. "
                "Health checks liveness only; the agent's configured authentication is checked during fleet readiness.",
                *notes,
            ),
        )

    def up(self, plan: AttachPlan, *, timeout: float = 300.0) -> JsonObject:
        """Execute each boundary once and report success only after fleet readiness."""

        import math

        if not math.isfinite(timeout) or timeout <= 0:
            raise AttachError("plan", "--timeout must be a positive finite number")
        service = FleetCommandService(self.session, renderer=_QuietRenderer())
        config = self._step("fleet_validate", lambda: EyvanFleetConfig.model_validate(yaml.safe_load(plan.fleet_yaml)))
        self._step("fleet_validate", lambda: service.apply_request(config))
        self._step("fleet_apply", lambda: service.require_apply_endpoint(plan.fleet))
        # Refuse overwriting an unrelated local config before modifying workers.
        self._step("fleet_write", lambda: self._check_fleet_path(plan))
        token: str | None = None
        if plan.node is not None:
            # Before anything touches the TPU: an agent that cannot serve attached
            # TPUs refuses every cell, and nothing about the install would help.
            self._ensure_runtime_supports_attach(require_current=plan.ssh_entry is not None)
            with self.progress(f"Checking TPU {plan.node.name} is still READY"):
                self._revalidate_node(plan)
            profile = self._step("credential", self._load_profile)
            token = derive_native_worker_token(
                profile.native_worker_root_secret,
                attach_management_uid(
                    project=plan.node.project, zone=plan.node.zone, name=plan.node.name, node_id=plan.node.node_id
                ),
                provider_generation=1,
            )
            assert token is not None
            self._install_tpu(plan, profile, token)
            if plan.endpoint_mode == "tunnel":
                with self.progress(f"Opening an SSH tunnel to {len(plan.tunnel_ports)} worker(s)"):
                    try:
                        self._open_tunnel(plan)
                    except (SshError, OSError, ValueError) as exc:
                        raise AttachError("tunnel", str(exc)) from exc
        with self.progress(f"Waiting for the Eyvan worker on {len(plan.endpoints)} host(s)"):
            for endpoint in plan.endpoints:
                self._wait_worker(endpoint, token, timeout)
        self._step("fleet_write", lambda: self._write_fleet(plan))
        # Cells a previous attempt left behind are history, not evidence about this
        # run: counting them could declare a fresh apply refused before its first
        # cell exists.
        stale_cells = self._dead_cell_ids(plan.fleet)
        with self.progress(f"Applying fleet {plan.fleet}"):
            self._step("fleet_apply", lambda: service.ensure_local_runtime(config))
            self._step(
                "fleet_apply",
                lambda: service.apply(
                    config,
                    generated_payload=None,
                    selected=None,
                    autoscale_cache_dir=None,
                    dry_run=False,
                    as_json=True,
                    idempotency_key=None,
                ),
            )
        with self.progress("Waiting for the fleet to become ready") as phase:
            try:
                cells = self._wait_fleet(plan.fleet, timeout, phase=phase, ignore=stale_cells)
            except _FleetRefused:
                # A refused fleet is replaced cell after cell for as long as it is
                # enabled. Release it (attached hardware is never touched) so the
                # failure is the end of the story, not the start of a loop.
                self._release_fleet(plan.fleet)
                raise
        return {
            "fleet": plan.fleet,
            "fleet_path": str(plan.fleet_path),
            "ready": True,
            "cells": cells,
            "submit": f'init(fleet="{plan.fleet}")',
            "limitations": list(plan.limitations),
        }

    def _load_profile(self) -> GcpNativeRuntimeProfile:
        """Load only the profile serving the selected plane, never mint secrets here."""

        try:
            return GcpNativeRuntimeProfileStore(runtime_profile_path_for(self.session.base_url)).load()
        except FileNotFoundError as exc:
            raise AttachError(
                "credential", "native GCP runtime profile is missing; run `eyvan runtime init` first"
            ) from exc

    def _revalidate_node(self, plan: AttachPlan) -> None:
        """Re-read the incarnation after confirmation, before delivering any credential."""

        original = plan.node
        assert original is not None
        if plan.ssh_entry is None:
            current = self.resolve(project=original.project, zone=original.zone, name=original.name)
        else:
            try:
                read = _ssh_node_reader(plan.ssh_entry, self.ssh_options).get_node(
                    original.project, original.zone, original.name
                )
            except (TpuApiError, TpuNodeParseError) as exc:
                raise AttachError("resolve", str(exc)) from exc
            if read is None:
                raise AttachError("resolve", f"TPU {original.name!r} no longer exists; rerun `eyvan gcp connect`")
            current = read
        if current.node_id != original.node_id:
            raise AttachError("resolve", "TPU was recreated under the same name; rerun `eyvan gcp connect`")
        if (
            current.state != "READY"
            or current.health not in (None, "HEALTHY")
            or current.accelerator_type != original.accelerator_type
            or current.capacity_class != original.capacity_class
            or current.endpoint_ips(plan.endpoint_mode) != original.endpoint_ips(plan.endpoint_mode)
        ):
            raise AttachError(
                "resolve", "TPU state, shape, class or endpoints changed after planning; rerun `eyvan gcp connect`"
            )

    def _open_tunnel(self, plan: AttachPlan) -> None:
        """Start (or reuse) the tunnel the agent will keep alive, so worker health can be checked now."""

        assert plan.node is not None and plan.ssh_entry is not None
        manager = self.tunnels if self.tunnels is not None else SshTunnelManager()
        manager.ensure(self._tunnel_spec_for(plan.node, plan.ssh_entry, plan.tunnel_ports))

    def _install_tpu(self, plan: AttachPlan, profile: GcpNativeRuntimeProfile, token: str) -> None:
        node = plan.node
        assert node is not None and plan.access is not None
        access: HostAccess = plan.access
        runner = self._gcloud_runner()
        # Prove every host accepts this machine's SSH login before anything is
        # changed, so a refused key cannot leave firewall or TPU tag changes
        # behind before worker installation begins.
        self._check_ssh_login(node.name, access)
        if plan.endpoint_mode == "public" and plan.ssh_entry is None:
            with self.progress(f"Opening worker port {self.port} to this machine"):
                tag = self._step(
                    "ingress",
                    lambda: _public_ingress(profile, runner).ensure(
                        project=node.project, network=node.network, port=self.port
                    ),
                )
                # gcloud refuses an update that changes nothing ("No fields would
                # change"), so a rerun on a TPU that already carries the tag must skip it.
                if tag not in node.tags:
                    self._command(
                        "tag",
                        (
                            "compute",
                            "tpus",
                            "tpu-vm",
                            "update",
                            node.name,
                            *self._tpu_args(node),
                            "--add-tags",
                            tag,
                            "--quiet",
                        ),
                        unchanged_ok=True,
                    )
        builder = self._step("wheel_build", native_worker_wheel_builder)
        try:
            with self.progress("Building the Eyvan worker package"):
                wheel = self._step("wheel_build", builder.build)
            if not wheel.is_file() or not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", wheel.name):
                raise AttachError("wheel_build", "builder did not return a valid wheel file")
            copies = access.copy_argvs(str(wheel), self.layout.staging_dir)
            with self.progress(f"Copying it to {access.host_count} host(s)") as phase:
                self._parallel(
                    [functools.partial(self._command, "wheel_scp", argv, access=access) for argv in copies],
                    phase=phase,
                )
            script = self._step(
                "worker_install",
                lambda: build_native_worker_startup_script(
                    package=f"{self.layout.staging_dir.rstrip('/')}/{wheel.name}",
                    port=self.port,
                    state_dir=self.layout.state_dir,
                    worker_secret_file=self.layout.secret_file,
                    runtime_setup=render_native_runtime_setup(self.runtime),
                    runtime_env=self.runtime.env or None,
                    runtime_container=self.runtime.container,
                    runtime_python=self.runtime.python,
                ),
            )

            def install(worker: int) -> None:
                self._command(
                    "worker_secret", access.exec_argv(worker, self._secret_command()), stdin_text=token, access=access
                )
                self._command(
                    "worker_install", access.exec_argv(worker, "sudo -n bash -s"), stdin_text=script, access=access
                )

            with self.progress(f"Installing the Eyvan worker on {access.host_count} host(s)") as phase:
                self._parallel([functools.partial(install, worker) for worker in range(access.host_count)], phase=phase)
        finally:
            self._step("wheel_build", builder.close)

    def _check_ssh_login(self, name: str, access: HostAccess) -> None:
        """Run a no-op on every host; refuse with a plain reason before any mutation."""

        with self.progress(f"Checking SSH login on {access.host_count} host(s)") as phase:
            try:
                self._parallel(
                    [
                        functools.partial(self._command, "ssh_check", access.exec_argv(worker, "true"), access=access)
                        for worker in range(access.host_count)
                    ],
                    phase=phase,
                )
            except AttachError as exc:
                detail = str(exc).split(" failed: ", 1)[-1]
                hint = (
                    " The host refuses this machine's SSH key: make sure a key this machine holds is accepted "
                    "on every host (the guest agent installs project and instance ssh-keys)."
                    if "permission denied" in detail.casefold()
                    else ""
                )
                raise AttachError(
                    "ssh_check", f"cannot log in to {name} over SSH: {detail}. Nothing was changed.{hint}"
                ) from exc

    def _parallel(self, tasks: tp.Sequence[tp.Callable[[], None]], *, phase: tp.Any = None) -> None:
        """Run per-host steps concurrently, then report the first failure in host order."""

        if len(tasks) == 1:
            tasks[0]()
            return
        errors: list[Exception | None] = [None] * len(tasks)
        finished = 0
        with ThreadPoolExecutor(max_workers=max(1, min(self.install_concurrency, len(tasks)))) as pool:
            futures = {pool.submit(task): index for index, task in enumerate(tasks)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors[futures[future]] = exc
                finished += 1
                if phase is not None:
                    phase.update(f"{finished}/{len(tasks)} hosts")
        failed = [(index, error) for index, error in enumerate(errors) if error is not None]
        if not failed:
            return
        index, error = failed[0]
        if not isinstance(error, AttachError):
            raise error
        detail = str(error).split(" failed: ", 1)[-1]
        scope = f"{len(failed)} of {len(tasks)} hosts failed; host {index}" if len(failed) > 1 else f"host {index}"
        raise AttachError(error.step, f"{detail} ({scope})") from error

    def _command(
        self,
        step: str,
        argv: tuple[str, ...],
        *,
        stdin_text: str | None = None,
        unchanged_ok: bool = False,
        access: HostAccess | None = None,
    ) -> None:
        """Run one host or gcloud step; show its error only when nothing private was sent to it.

        A command fed a credential on stdin may have it echoed back by the remote
        side, so its output stays withheld. A command that received no stdin
        carries nothing private; preserve the tool's error to make failures
        diagnosable.
        """

        timeout = self.ssh_options.command_timeout_s
        try:
            if access is not None:
                result = access.run(argv, stdin_text=stdin_text, timeout=timeout)
            else:
                result = self._gcloud_runner()(argv, stdin_text=stdin_text, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            raise AttachError(
                step, f"command could not complete ({type(exc).__name__}); inspect the host without sharing credentials"
            ) from exc
        if result.returncode == 0:
            return
        if stdin_text is not None:
            raise AttachError(
                step, f"command exited {result.returncode}; captured output withheld because it may contain credentials"
            )
        # Only gcloud's/ssh's own diagnostic lines are shown; other remote output is not.
        lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
        # ssh's own refusal is the cause; gcloud's trailing "exited with return
        # code [255]" only restates it, so the specific line wins when present.
        detail = next((line for line in reversed(lines) if "permission denied" in line.casefold()), "") or next(
            (line for line in reversed(lines) if _TOOL_DIAGNOSTIC.match(line)), ""
        )
        if unchanged_ok and "no fields would change" in detail.casefold():
            return
        shown = detail[:300] if detail else "no tool diagnostic; other output withheld"
        raise AttachError(step, f"command exited {result.returncode}: {shown}")

    def _wait_worker(self, endpoint: str, token: str | None, timeout: float) -> None:
        deadline = self.clock() + timeout
        while True:
            try:
                _worker_health(endpoint, token)
                return
            except (httpx.HTTPError, ValueError, OSError):
                if self.clock() >= deadline:
                    raise AttachError("worker_health", f"timed out waiting for native worker at {endpoint}") from None
                self.sleep(min(1.0, max(deadline - self.clock(), 0)))

    def _ensure_runtime_supports_attach(self, *, require_current: bool = False) -> None:
        """Make sure the local agent can serve this plan, redeploying it when it cannot.

        ``require_current`` is set for SSH-read or tunnelled nodes: an agent that
        knows ``gcp-attach`` may still predate those modes, so only this build will do.
        """

        endpoint = ControlPlaneEndpoint.parse(self.session.base_url)
        if not endpoint.is_loopback:
            return
        selected = self.runtime_check if self.runtime_check is not None else _runtime_needs_deploy

        def check(target: ControlPlaneEndpoint) -> str | None:
            return selected(target, require_current=True) if require_current else selected(target)

        with self.progress("Checking the local Eyvan runtime"):
            reason = self._step("runtime", lambda: check(endpoint))
        if reason is None:
            return
        deploy = self.runtime_deploy if self.runtime_deploy is not None else _deploy_runtime
        with self.progress(f"Updating the local Eyvan runtime ({reason})"):
            try:
                deploy(endpoint)
            except Exception as exc:
                raise AttachError(
                    "runtime",
                    f"{reason}, and redeploying it failed ({type(exc).__name__}: {str(exc)[:200]}); "
                    "run `eyvan runtime deploy --yes`, then retry",
                ) from exc
            self.session.refresh_loopback_identity()
        after = self._step("runtime", lambda: check(endpoint))
        if after is not None:
            raise AttachError("runtime", f"{after} even after redeploying; run `eyvan runtime deploy --yes`")

    def _release_fleet(self, fleet: str) -> None:
        """Disable a refused fleet; best effort, since the refusal is the error to report."""

        try:
            self.session.client.request(
                "DELETE",
                f"/api/v1/fleets/{fleet}",
                headers={"Idempotency-Key": f"cli-gcp-connect-release-{fleet}"},
            )
        except Exception:
            pass

    @staticmethod
    def _refusal(rows: list[JsonObject]) -> str | None:
        """Return a definitive refusal detail from fleet status cells, or ``None``."""

        for cell in rows:
            detail = cell.get("unavailable_detail")
            code = cell.get("provider_failure_code")
            reason = cell.get("unavailable_reason")
            if cell.get("state") == "unavailable" and isinstance(detail, str) and detail:
                if "invalid configuration" in detail or reason == "provider_failure":
                    if code is None or str(code) not in _TRANSIENT_FAILURE_CODES:
                        return f"{cell.get('id', '?')}: {detail}"
        replaced = [cell for cell in rows if cell.get("state") in ("deleted", "released")]
        if len(replaced) >= _MAX_REPLACED_CELLS:
            details = {
                str(cell.get("unavailable_detail") or cell.get("unavailable_reason") or "no reason recorded")
                for cell in replaced
            }
            return f"{len(replaced)} cells were created and lost ({'; '.join(sorted(details))[:300]})"
        return None

    def _dead_cell_ids(self, fleet: str) -> frozenset[str]:
        """Return ids of the fleet's already deleted/released cells; empty when it does not exist yet."""

        try:
            status = self.session.client.request("GET", f"/api/v1/fleets/{fleet}")
        except Exception:  # a missing fleet (404) or unreadable status means there is no history to skip
            return frozenset()
        cells = status.get("cells") if is_json_object(status) else None
        if not isinstance(cells, list):
            return frozenset()
        return frozenset(
            str(cell["id"])
            for cell in cells
            if is_json_object(cell) and "id" in cell and cell.get("state") in ("deleted", "released")
        )

    def _wait_fleet(
        self, fleet: str, timeout: float, *, phase: tp.Any = None, ignore: frozenset[str] = frozenset()
    ) -> list[str]:
        deadline = self.clock() + timeout
        states = "no cell observed (unknown)"
        while True:
            status = self._step("fleet_ready", lambda: self.session.client.request("GET", f"/api/v1/fleets/{fleet}"))
            cells = status.get("cells") if is_json_object(status) else None
            rows = (
                [cell for cell in cells if is_json_object(cell) and str(cell.get("id")) not in ignore]
                if isinstance(cells, list)
                else []
            )
            live = [cell for cell in rows if cell.get("state") not in ("deleted", "released")]
            if rows:
                states = ", ".join(f"{cell.get('id', '?')}={cell.get('state', 'unknown')}" for cell in live or rows)
                if live and all(cell.get("state") == "ready" for cell in live):
                    return [str(cell.get("id", "?")) for cell in live]
                refusal = self._refusal(rows)
                if refusal is not None:
                    raise _FleetRefused(
                        "fleet_ready",
                        f"the control plane refused fleet {fleet!r}: {refusal}; the fleet was released "
                        "(the TPU and its worker are untouched)",
                    )
                if phase is not None:
                    phase.update(states)
            if self.clock() >= deadline:
                raise AttachError(
                    "fleet_ready",
                    f"timed out waiting for fleet {fleet!r}; cell state: {states}; inspect `eyvan status {fleet}`",
                )
            self.sleep(min(1.0, max(deadline - self.clock(), 0)))

    def _fleet_path(self, fleet: str) -> Path:
        try:
            validate_fleet_name(fleet)
        except ValueError as exc:
            raise AttachError("plan", str(exc)) from exc
        return self.attached_dir / f"{fleet}.yaml"

    @staticmethod
    def _attachment_identity(text: str) -> tuple[tp.Any, ...] | None:
        """Return what a fleet document attaches (name, pools, node id or hosts), or ``None`` if unreadable."""

        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError:
            return None
        if not isinstance(document, dict) or not isinstance(document.get("pools"), dict):
            return None
        pools = []
        for pool_name, pool in sorted(document["pools"].items()):
            options = pool.get("provider_options") if isinstance(pool, dict) else None
            options = options if isinstance(options, dict) else {}
            hosts = options.get("hosts")
            pools.append(
                (
                    str(pool_name),
                    pool.get("capacity_backend") if isinstance(pool, dict) else None,
                    options.get("tpu_node"),
                    options.get("tpu_node_id"),
                    tuple(hosts) if isinstance(hosts, list) else None,
                )
            )
        return (document.get("name"), tuple(pools))

    @staticmethod
    def _check_fleet_path(plan: AttachPlan) -> bool:
        """Return whether an existing file must be replaced; refuse one describing another attachment.

        A rerun against the same node may legitimately change the document (new
        runtime setup, a different endpoint mode), so the same attachment is
        updated rather than refused. A file for any other node or host is never
        overwritten.
        """

        if plan.fleet_path.is_symlink():
            raise ValueError("refusing a symlink fleet YAML destination")
        if not plan.fleet_path.exists():
            return False
        existing = plan.fleet_path.read_text()
        if existing == plan.fleet_yaml:
            return False
        current = GcpConnector._attachment_identity(existing)
        if current is None or current != GcpConnector._attachment_identity(plan.fleet_yaml):
            raise ValueError(
                f"fleet YAML {plan.fleet_path} already exists for a different attachment; choose --fleet NAME"
            )
        return True

    @staticmethod
    def _write_fleet(plan: AttachPlan) -> None:
        replace = GcpConnector._check_fleet_path(plan)
        plan.fleet_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if replace:
            temporary = plan.fleet_path.with_name(f".{plan.fleet_path.name}.tmp")
            temporary.write_text(plan.fleet_yaml)
            os.chmod(temporary, 0o600)
            temporary.replace(plan.fleet_path)
            return
        try:
            with plan.fleet_path.open("x") as stream:
                stream.write(plan.fleet_yaml)
        except FileExistsError:
            if GcpConnector._check_fleet_path(plan):
                GcpConnector._write_fleet(plan)

    @staticmethod
    def _step(step: str, action: tp.Callable[[], tp.Any]) -> tp.Any:
        try:
            return action()
        except AttachError:
            raise
        except Exception as exc:
            # Exceptions can embed credentials (HTTP headers, argv, echoed stdin).
            # Only expose known-safe local validation details, never arbitrary I/O.
            detail = (
                str(exc)
                if step in ("fleet_validate", "fleet_write") and isinstance(exc, ValueError)
                else type(exc).__name__
            )
            raise AttachError(step, detail) from exc

    @staticmethod
    def _tpu_args(node: TpuNode) -> tuple[str, ...]:
        return ("--zone", node.zone, "--project", node.project)

    @staticmethod
    def _ssh(node: TpuNode, worker: int, command: str, *, internal: bool = False) -> tuple[str, ...]:
        return (
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            node.name,
            *GcpConnector._tpu_args(node),
            *(("--internal-ip",) if internal else ()),
            f"--worker={worker}",
            "--quiet",
            "--command",
            command,
        )

    def _secret_command(self) -> str:
        # stdin is exclusively the derived credential. Use a private file at
        # creation, not chmod after a world-readable redirection.
        secret = self.layout.secret_file
        directory = secret.rsplit("/", 1)[0] or "/"
        script = (
            f"set -eu; install -d -o root -g root -m 0700 {directory}; "
            f"umask 077; cat > {secret}; "
            f"chown root:root {secret}; chmod 0600 {secret}"
        )
        return f"sudo -n sh -c {shlex.quote(script)}"

    @staticmethod
    def _tail_actions(
        path: Path, name: str, endpoints: tuple[str, ...], *, health_path: str = "/api/v1/health"
    ) -> tuple[AttachAction, ...]:
        return (
            *(
                AttachAction(
                    "worker_health",
                    ("GET", f"{endpoint}{health_path}"),
                    "Verify every host identifies as a native worker",
                )
                for endpoint in endpoints
            ),
            AttachAction("fleet_write", ("write", str(path)), "Persist public fleet YAML (no credential)", True),
            AttachAction(
                "fleet_apply",
                ("FleetCommandService.apply", name),
                "Prepare the local runtime, apply fleet and activate its context",
                True,
            ),
            AttachAction(
                "fleet_ready",
                ("GET", f"/api/v1/fleets/{name}"),
                "Poll until the fleet cell is ready or report its last state",
            ),
        )


__all__ = ("AttachError", "AttachPlan", "GcpConnector")
