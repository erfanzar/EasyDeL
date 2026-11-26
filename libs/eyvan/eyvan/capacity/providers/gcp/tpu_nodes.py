# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read existing Cloud TPU VMs, for attaching hardware Eyvan did not create.

Shared by the ``gcp-attach`` capacity driver (which observes one attached node)
and ``eyvan gcp browse``/``connect`` (which list projects and nodes). Read-only:
nothing here creates, updates or deletes a cloud resource.

The credential is whatever ``gcloud`` already holds, minted the same way the
accelerator catalogue mints it, and every request is plain HTTPS to the Cloud
TPU and Resource Manager REST APIs.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import subprocess
import threading
import time
import typing as tp
from dataclasses import dataclass, field

from eyvan._internal.identifiers import stable_id
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.providers.gcp.accelerator_catalog import AcceleratorCatalogError, gcloud_access_token
from eyvan.records import CapacityClass

#: Capacity backend name for attached, never-provisioned TPU VMs.
ATTACH_CAPACITY_BACKEND = "gcp-attach"

#: ``provider_options`` keys an attached pool carries.
OPTION_NODE_NAME = "tpu_node"
OPTION_NODE_ID = "tpu_node_id"
OPTION_ENDPOINT_MODE = "endpoint_mode"
#: SSH destination of one host of the slice, when the node is read and reached over SSH.
OPTION_SSH_ENTRY = "ssh_entry"
#: Local ports, one per host in worker order, that tunnel mode forwards to each worker.
OPTION_TUNNEL_PORTS = "tunnel_ports"
#: The native worker port the hosts were installed with.
OPTION_WORKER_PORT = "worker_port"
#: ``public``/``internal``: the agent dials the hosts' own addresses.
#: ``tunnel``: the agent dials local ports an SSH process forwards to each host's
#: internal address, for networks whose firewall admits only SSH.
ENDPOINT_MODES = ("public", "internal", "tunnel")

REQUEST_TIMEOUT_S = 30.0
PAGE_SIZE = 500
#: Statuses retried once with a freshly minted token. 403 is included because the
#: usual cause on an operator's machine is the gcloud account having changed (or
#: been fixed) since the token was cached; without a re-mint the process keeps
#: presenting the old account until it restarts.
_AUTH_RETRY = frozenset({401, 403})
#: A 403 also answers "this account may not read that project", which discovery
#: meets for every locked project on every pass. Re-mint on a 403 at most this
#: often per reader, so a listing does not start one gcloud process per project.
_FORBIDDEN_REMINT_INTERVAL_S = 60.0
_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})

_TPU_NODES = "https://tpu.googleapis.com/v2/projects/{project}/locations/{zone}/nodes"
_TPU_NODE = "https://tpu.googleapis.com/v2/projects/{project}/locations/{zone}/nodes/{name}"
_PROJECTS = "https://cloudresourcemanager.googleapis.com/v1/projects"

#: Cloud TPU node names can include underscores, so accept the API's names as returned.
_NAME = re.compile(r"^[a-z]([-_a-z0-9]*[a-z0-9])?$")
_FLEET_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_FLEET_NAME_MAX = 63


class TpuApiError(RuntimeError):
    """A Cloud TPU or Resource Manager read failed.

    ``retryable`` separates "the link failed, nothing was learned" from "the
    provider answered and refused" (permissions, a malformed request).
    """

    def __init__(self, message: str, *, retryable: bool, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class TpuNodeParseError(ValueError):
    """The Cloud TPU API returned a node document Eyvan cannot interpret."""


@dataclass(frozen=True, slots=True)
class GcpProject:
    """One project the operator's gcloud identity can see."""

    project_id: str
    name: str | None
    state: str | None


@dataclass(frozen=True, slots=True)
class TpuHost:
    """One VM of a TPU slice, in ``networkEndpoints`` order (worker index)."""

    internal_ip: str | None
    external_ip: str | None


@dataclass(frozen=True, slots=True)
class TpuNode:
    """One existing Cloud TPU VM node, as the API reported it."""

    project: str
    zone: str
    name: str
    node_id: str
    accelerator_type: str
    state: str
    health: str | None
    capacity_class: CapacityClass
    hosts: tuple[TpuHost, ...]
    network: str | None
    subnetwork: str | None
    tags: tuple[str, ...]
    labels: dict[str, str] = field(default_factory=dict)
    runtime_version: str | None = None
    service_account: str | None = None

    @property
    def spot(self) -> bool:
        """Whether the provider can reclaim this node (Spot or legacy preemptible)."""

        return self.capacity_class is CapacityClass.SPOT

    @property
    def region(self) -> str:
        return _region(self.zone)

    def endpoint_ips(self, mode: str) -> tuple[str, ...] | None:
        """Return one address per host, or ``None`` when any host lacks one.

        Partial lists are never returned: a slice with a host missing its
        address cannot run a multi-host job, and publishing the rest would
        send work to a fraction of the slice.
        """

        if mode not in ENDPOINT_MODES:
            raise ValueError(f"endpoint mode must be one of {ENDPOINT_MODES}, not {mode!r}")
        if not self.hosts:
            return None
        # A tunnel forwards to each host's internal address from inside the network.
        addresses = [host.external_ip if mode == "public" else host.internal_ip for host in self.hosts]
        if any(address is None for address in addresses):
            return None
        return tuple(tp.cast(str, address) for address in addresses)


def _region(zone: str) -> str:
    region, separator, suffix = zone.rpartition("-")
    if not separator or not region or not suffix:
        raise TpuNodeParseError(f"invalid zone {zone!r}")
    return region


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _ip(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TpuNodeParseError(f"TPU node returned an invalid {label}")
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError as exc:
        raise TpuNodeParseError(f"TPU node returned an invalid {label}") from exc


def node_capacity_class(document: JsonObject) -> CapacityClass:
    """Return the purchase class a node document states.

    ``schedulingConfig.spot`` and the legacy ``preemptible`` flag both mean the
    provider may reclaim the node; ``reserved`` means it draws on a
    reservation. Anything else is on-demand.
    """

    scheduling = document.get("schedulingConfig")
    if is_json_object(scheduling):
        if scheduling.get("spot") is True or scheduling.get("preemptible") is True:
            return CapacityClass.SPOT
        if scheduling.get("reserved") is True:
            return CapacityClass.RESERVED
    return CapacityClass.ON_DEMAND


def parse_tpu_node(document: object, *, project: str) -> TpuNode:
    """Parse one ``projects.locations.nodes`` document strictly."""

    if not is_json_object(document):
        raise TpuNodeParseError("TPU node document must be an object")
    resource = document.get("name")
    if not isinstance(resource, str):
        raise TpuNodeParseError("TPU node document has no name")
    parts = resource.split("/")
    if len(parts) != 6 or parts[0] != "projects" or parts[2] != "locations" or parts[4] != "nodes":
        raise TpuNodeParseError(f"unexpected TPU node resource name {resource!r}")
    zone, name = parts[3], parts[5]
    node_id = document.get("id")
    if not isinstance(node_id, str) or not node_id.strip():
        raise TpuNodeParseError(f"TPU node {name!r} has no id")
    accelerator_type = document.get("acceleratorType")
    if not isinstance(accelerator_type, str) or not accelerator_type.strip():
        raise TpuNodeParseError(f"TPU node {name!r} has no acceleratorType")
    state = document.get("state")
    state_value = state.strip().upper() if isinstance(state, str) and state.strip() else "STATE_UNSPECIFIED"
    hosts: list[TpuHost] = []
    raw_endpoints = document.get("networkEndpoints")
    if raw_endpoints is not None:
        if not isinstance(raw_endpoints, list):
            raise TpuNodeParseError(f"TPU node {name!r} returned invalid networkEndpoints")
        for raw in raw_endpoints:
            if not is_json_object(raw):
                raise TpuNodeParseError(f"TPU node {name!r} returned an invalid network endpoint")
            access = raw.get("accessConfig")
            external = _ip(access.get("externalIp"), label="external IP") if is_json_object(access) else None
            hosts.append(TpuHost(internal_ip=_ip(raw.get("ipAddress"), label="IP address"), external_ip=external))
    network_config = document.get("networkConfig")
    network = subnetwork = None
    if is_json_object(network_config):
        network = _optional_string(network_config.get("network"))
        subnetwork = _optional_string(network_config.get("subnetwork"))
    raw_tags = document.get("tags")
    tags = tuple(tag for tag in raw_tags if isinstance(tag, str)) if isinstance(raw_tags, list) else ()
    raw_labels = document.get("labels")
    labels = (
        {str(key): str(value) for key, value in raw_labels.items() if isinstance(value, str)}
        if is_json_object(raw_labels)
        else {}
    )
    service_account = document.get("serviceAccount")
    return TpuNode(
        project=project,
        zone=zone,
        name=name,
        node_id=node_id.strip(),
        accelerator_type=accelerator_type.strip().lower(),
        state=state_value,
        health=_optional_string(document.get("health")),
        capacity_class=node_capacity_class(document),
        hosts=tuple(hosts),
        network=network,
        subnetwork=subnetwork,
        tags=tags,
        labels=labels,
        runtime_version=_optional_string(document.get("runtimeVersion")),
        service_account=_optional_string(service_account.get("email")) if is_json_object(service_account) else None,
    )


def attach_management_uid(*, project: str, zone: str, name: str, node_id: str) -> str:
    """Return the stable identity an attached node's native workers are keyed to.

    Independent of the Eyvan cell id on purpose: the plane mints cell ids, so a
    CLI bootstrapping workers before the cell exists could not know it. The
    node's numeric ``id`` is included so a TPU deleted and recreated under the
    same name is a different identity -- its workers never inherit a credential.
    """

    return stable_id("gcp-attach", project, zone, name, node_id)


def attach_fleet_name(*, project: str, zone: str, name: str) -> str:
    """Return a stable, valid fleet name for one attached TPU node."""

    digest = hashlib.sha256(f"{project}\0{zone}\0{name}".encode()).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "tpu"
    prefix = "tpu-"
    room = _FLEET_NAME_MAX - len(prefix) - 1 - len(digest)
    candidate = f"{prefix}{slug[:room].rstrip('-')}-{digest}"
    if not _FLEET_NAME.fullmatch(candidate):  # pragma: no cover - the construction guarantees it
        raise ValueError(f"could not derive a valid fleet name for TPU {name!r}")
    return candidate


def validate_node_name(name: str) -> str:
    """Return a trimmed TPU node name, refusing anything the API could not hold."""

    value = name.strip()
    if not value or len(value) > 63 or not _NAME.fullmatch(value):
        raise ValueError(f"invalid TPU node name {name!r}")
    return value


def gcloud_active_account(timeout_s: float = 15.0) -> str | None:
    """Return the gcloud account whose token :func:`gcloud_access_token` mints, or ``None`` when unknown."""

    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=provider_cli_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    account = (result.stdout or "").strip()
    if result.returncode != 0 or not account or account == "(unset)" or len(account) > 320:
        return None
    return account


@dataclass
class TpuNodeApi:
    """Read projects and TPU nodes over REST with the operator's gcloud credential.

    Attributes:
        token_source: Bearer credential source; injected so tests never shell out.
        transport: Optional ``httpx`` transport, injected by tests.
    """

    token_source: tp.Callable[[], str] = gcloud_access_token
    transport: tp.Any = None
    _token: str | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _forbidden_remint_at: float = field(default=float("-inf"), init=False, repr=False)
    #: Names the account the credential belongs to. ``None`` means: ask gcloud
    #: when the token comes from gcloud, otherwise the account is unknown.
    account_source: tp.Callable[[], str | None] | None = None

    def account(self) -> str | None:
        """Return the identity this reader lists as, or ``None`` when it cannot be named."""

        source = self.account_source
        if source is None and self.token_source is gcloud_access_token:
            source = gcloud_active_account
        if source is None:
            return None
        try:
            return source()
        except Exception:  # naming the account is informational; never fail a listing over it
            return None

    def _may_remint(self, status: int) -> bool:
        """Return whether a ``status`` answer should be retried with a fresh token."""

        if status != 403:
            return status in _AUTH_RETRY
        with self._lock:
            now = time.monotonic()
            if now - self._forbidden_remint_at < _FORBIDDEN_REMINT_INTERVAL_S:
                return False
            self._forbidden_remint_at = now
            return True

    def _client(self) -> tp.Any:
        import httpx

        kwargs: dict[str, tp.Any] = {"timeout": REQUEST_TIMEOUT_S, "trust_env": False}
        if self.transport is not None:
            kwargs["transport"] = self.transport
        return httpx.Client(**kwargs)

    def _bearer(self, stale: str | None = None) -> str:
        with self._lock:
            if self._token is None or self._token == stale:
                try:
                    self._token = self.token_source()
                except AcceleratorCatalogError as exc:
                    raise TpuApiError(str(exc), retryable=False) from exc
            return self._token

    def _get(self, url: str, params: dict[str, tp.Any] | None = None) -> tuple[int, JsonObject]:
        import httpx

        token = self._bearer()
        try:
            with self._client() as client:
                response = client.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
                if self._may_remint(response.status_code):
                    token = self._bearer(token)
                    response = client.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            raise TpuApiError(f"request to {url.split('?')[0]} failed: {type(exc).__name__}", retryable=True) from exc
        try:
            body = response.json() if response.content else {}
        except ValueError:
            body = {}
        return response.status_code, body if is_json_object(body) else {}

    @staticmethod
    def _failure(status: int, body: JsonObject, what: str) -> TpuApiError:
        error = body.get("error")
        message = error.get("message") if is_json_object(error) else None
        detail = message if isinstance(message, str) and message else f"HTTP {status}"
        return TpuApiError(f"{what}: {detail}", retryable=status in _RETRYABLE, status=status)

    def list_projects(self) -> tuple[GcpProject, ...]:
        """Return every ACTIVE project the credential can list, sorted by id."""

        projects: list[GcpProject] = []
        page: str | None = None
        while True:
            params: dict[str, tp.Any] = {"pageSize": PAGE_SIZE}
            if page:
                params["pageToken"] = page
            status, body = self._get(_PROJECTS, params)
            if status != 200:
                raise self._failure(status, body, "could not list GCP projects")
            for item in body.get("projects") or []:
                if not is_json_object(item):
                    continue
                project_id = item.get("projectId")
                state = _optional_string(item.get("lifecycleState"))
                if isinstance(project_id, str) and project_id and state in (None, "ACTIVE"):
                    projects.append(GcpProject(project_id, _optional_string(item.get("name")), state))
            next_page = body.get("nextPageToken")
            if not isinstance(next_page, str) or not next_page:
                break
            page = next_page
        return tuple(sorted(projects, key=lambda project: project.project_id))

    def list_nodes(self, project: str, *, zone: str = "-") -> tuple[tuple[TpuNode, ...], tuple[str, ...]]:
        """Return a project's TPU nodes and any locations the API could not reach.

        Unreachable locations are returned rather than dropped: "this zone has
        no TPU" and "nobody could ask this zone" are different answers.
        """

        nodes: list[TpuNode] = []
        unreachable: list[str] = []
        page: str | None = None
        while True:
            params: dict[str, tp.Any] = {"pageSize": PAGE_SIZE}
            if page:
                params["pageToken"] = page
            status, body = self._get(_TPU_NODES.format(project=project, zone=zone), params)
            if status != 200:
                raise self._failure(status, body, f"could not list TPU nodes in {project}")
            for item in body.get("nodes") or []:
                nodes.append(parse_tpu_node(item, project=project))
            for location in body.get("unreachable") or []:
                if isinstance(location, str):
                    unreachable.append(location)
            next_page = body.get("nextPageToken")
            if not isinstance(next_page, str) or not next_page:
                break
            page = next_page
        return tuple(sorted(nodes, key=lambda node: (node.zone, node.name))), tuple(sorted(set(unreachable)))

    def get_node(self, project: str, zone: str, name: str) -> TpuNode | None:
        """Return one node, or ``None`` when the API definitively says it does not exist."""

        status, body = self._get(_TPU_NODE.format(project=project, zone=zone, name=validate_node_name(name)))
        if status == 404:
            return None
        if status != 200:
            raise self._failure(status, body, f"could not read TPU node {name!r} in {zone}")
        return parse_tpu_node(body, project=project)


__all__ = (
    "ATTACH_CAPACITY_BACKEND",
    "ENDPOINT_MODES",
    "OPTION_ENDPOINT_MODE",
    "OPTION_NODE_ID",
    "OPTION_NODE_NAME",
    "OPTION_SSH_ENTRY",
    "OPTION_TUNNEL_PORTS",
    "OPTION_WORKER_PORT",
    "GcpProject",
    "TpuApiError",
    "TpuHost",
    "TpuNode",
    "TpuNodeApi",
    "TpuNodeParseError",
    "attach_fleet_name",
    "attach_management_uid",
    "gcloud_active_account",
    "node_capacity_class",
    "parse_tpu_node",
    "validate_node_name",
)
