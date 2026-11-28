# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One command that stands up an Eyvan control-plane head on GCP.

Standing a head up by hand is fifteen steps and every one of them is a place to
stop: a required flag that is ``argparse.SUPPRESS``, three separate crash-restart
cycles for three environment variables the server reads one at a time, a missing
``uv``, and an egress mode whose legal values appear in no page. Each failure is
individually small and collectively fatal, because the operator learns about
step *n* only after paying for steps ``1..n-1``.

So this module owns the whole sequence as data. A :class:`HeadPlan` is every
action and every resource name, computed before anything is created, which makes
``--dry-run`` a real answer rather than a partial rehearsal. Every mutation goes
through one injected :class:`HeadCommandRunner`, the same seam
``capacity/providers/gcp/runtime/bootstrap.py`` uses, so the sequencing,
ownership, and refusal logic are testable without a GCP project.

**This module has not been run against real GCP.** The seam is exercised, the
argv is asserted, the refusals are proven; provider acceptance of those commands
is not established here and is not claimed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import re
import secrets
import shlex
import subprocess
import tarfile
import tempfile
import time
import typing as tp
from dataclasses import dataclass, replace
from pathlib import Path

import click
from rich.prompt import Confirm

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.plane_tls import CONTROL_PLANE_CA_FILE_ENV, HEAD_CONTROL_PLANE_CA_PATH
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.sdk.commands.autoscale import autoscale_console, is_interactive
from eyvan.sdk.commands.common import FleetConfigLoader
from eyvan.sdk.contexts import DEFAULT_ORG_ID, DEFAULT_PROJECT_ID
from eyvan.sdk.node import NODE_CONTEXT_PATH

#: Legal ``bootstrap_egress`` values, restated here because an operator who
#: guesses wrong learns the set from a rejection rather than from a page.
BOOTSTRAP_EGRESS_MODES: tuple[str, ...] = ("cloud-nat", "private-google-access", "public-internet")

#: Label proving Eyvan created a head. Adoption requires it: a VM that merely
#: has the right *name* may be someone else's, and reconfiguring it would be
#: the most expensive kind of idempotence.
HEAD_MANAGED_LABEL_KEY = "eyvan-managed"
HEAD_MANAGED_LABEL_VALUE = "control-plane-head"
HEAD_FLEET_LABEL_KEY = "eyvan-head-fleet"

#: Ownership marker on managed firewall rules, mirroring the native-worker
#: ingress convention: a rule Eyvan did not write is never widened or deleted.
HEAD_FIREWALL_DESCRIPTION_PREFIX = "eyvan-control-plane-head:v1"

#: Google's fixed IAP TCP-forwarding source range. It is not RFC 1918, and it is
#: not the internet either: only Google's IAP front end originates from it, and
#: reaching it still requires ``roles/iap.tunnelResourceAccessor``.
IAP_TCP_FORWARDING_RANGE = "35.235.240.0/20"

#: The one project a head with no identity provider trusts for callable
#: workloads. Bootstrap mode has exactly one principal, and this is its project.
BOOTSTRAP_CALLABLE_PROJECT = f"{DEFAULT_ORG_ID}/{DEFAULT_PROJECT_ID}"

DEFAULT_HEAD_MACHINE_TYPE = "e2-standard-4"
DEFAULT_HEAD_PORT = 8080
DEFAULT_HEAD_BOOT_DISK_GB = 50
DEFAULT_HEAD_IMAGE_FAMILY = "ubuntu-2404-lts-amd64"
DEFAULT_HEAD_IMAGE_PROJECT = "ubuntu-os-cloud"
#: The default is this installation itself, staged to the head. The name
#: ``eyvan`` resolves only from an index, and Eyvan is not published to one.
DEFAULT_HEAD_PACKAGE = "eyvan"
STAGED_WHEEL_PACKAGE = "@staged-wheel"
#: Egress mode assumed when only reading or deleting a head, where it is unused.
DEFAULT_HEAD_EGRESS = "cloud-nat"
DEFAULT_COMMAND_TIMEOUT_S = 600.0
#: How long the installer waits for a just-created VM to accept SSH. `instances
#: create` returns before sshd is listening, so without this every first run
#: fails on a machine that is merely young.
DEFAULT_SSH_READY_ATTEMPTS = 20
DEFAULT_SSH_READY_DELAY_S = 15.0

#: Where the head keeps the environment the server and migration both read.
HEAD_ENVIRONMENT_PATH = "/etc/eyvan/head.env"
#: The head's TLS material. The server certificate is issued by a CA the head
#: generates for itself: its address is a private VPC address, which no public CA
#: will certify. The CA certificate is public and lives at
#: ``HEAD_CONTROL_PLANE_CA_PATH``; its key never leaves ``HEAD_TLS_DIR``.
HEAD_TLS_DIR = "/etc/eyvan/tls"
HEAD_TLS_CERT_PATH = f"{HEAD_TLS_DIR}/server.pem"
HEAD_TLS_KEY_PATH = f"{HEAD_TLS_DIR}/server.key"
HEAD_TLS_CA_KEY_PATH = f"{HEAD_TLS_DIR}/ca.key"
HEAD_BOOTSTRAP_PATH = "/opt/eyvan/head-bootstrap.sh"
#: Where a staged distribution lands. A wheel must keep its PEP 427 name --
#: pip refuses "eyvan-head.whl" as not a valid wheel filename -- so the file is
#: written under its own name in a directory the installer globs.
HEAD_WHEEL_DIR = "/opt/eyvan/wheels"
#: Superseded wheels, newest first, so a bad deploy has somewhere to go back to.
#: Preserve prior wheels during staging so rollback does not require rebuilding
#: an older distribution from source.
HEAD_PREVIOUS_WHEEL_DIR = f"{HEAD_WHEEL_DIR}/previous"
#: How many superseded wheels to keep. Two is enough to undo a bad deploy and
#: still have the one before it, without letting a long-lived head accumulate
#: builds on a disk that also holds workload state.
HEAD_WHEEL_RETENTION = 2
HEAD_VENV_PATH = "/opt/eyvan/venv"
#: Where the head unpacks the operations UI it serves at its own root.
#: The server mounts this only when the directory exists, so a head built
#: without a bundle simply serves the API and says so.
HEAD_DASHBOARD_DIR = "/opt/eyvan/dashboard"
#: Where a head installs the interpreter its workers run, so that submitting from
#: the head works without the operator building a matching environment by hand.
HEAD_SUBMITTER_DIR = "/opt/eyvan/submit"

#: Where uv puts the standalone interpreter the submitter venv points at. Named
#: explicitly because uv's default lives in the installing user's home, and this
#: script installs as root -- leaving every other account with a venv whose
#: ``python`` symlink they cannot execute.
HEAD_PYTHON_DIR = "/opt/eyvan/python"
HEAD_SERVICE_NAME = "eyvan-head.service"
#: A cell agent on the head. Every capacity pool needs an agent endpoint, and a
#: loopback development plane quietly supplies one -- so a fleet that applied on
#: a laptop was refused on a real head with "no default cell agent is
#: configured". The head runs one and points the platform default at it.
HEAD_AGENT_SERVICE_NAME = "eyvan-head-agent.service"

#: How long ``eyvan head up`` watches the units before calling them running:
#: several ``RestartSec`` intervals, so a unit that exits at every start restarts
#: at least once inside the window.
HEAD_SERVICE_SETTLE_S = 12
HEAD_AGENT_PORT = 8090
PLATFORM_CONFIG_PATH = "/etc/eyvan/platform.yaml"

#: Stand-in for the wheel path in a planned upload, replaced at execution time.
#: The plan is rendered before the temporary file exists, and a dry run must be
#: able to print the command without creating one.
_WORKER_WHEEL_PLACEHOLDER = "<staged-worker-wheel>"

#: What the installer runs to record this node's own control plane. Written as
#: one call into ``eyvan.sdk.node`` rather than as generated JSON so the format
#: has exactly one producer, and that producer is the module that reads it back.
_NODE_CONTEXT_SNIPPET = (
    "from eyvan.sdk.node import HEAD_ENVIRONMENT_PATH, NODE_CONTEXT_PATH, write_node_context; "
    "write_node_context(HEAD_ENVIRONMENT_PATH, NODE_CONTEXT_PATH, fleet={fleet!r})"
)

_PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_RESOURCE_NAME_PATTERN = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_REGION_PATTERN = re.compile(r"^[a-z]+[0-9]?-[a-z]+[0-9]+$")
_ZONE_PATTERN = re.compile(r"^[a-z]+[0-9]?-[a-z]+[0-9]+-[a-z][a-z0-9]*$")
_SAFE_PASSWORD_PATTERN = re.compile(r"^[A-Za-z0-9_-]{24,}$")
_UNSAFE_VALUE_PATTERN = re.compile(r"[\s'\"`$\\]")

#: The ranges a VPC subnet may use and still be internal. RFC 1918 plus the
#: shared address space GCP permits for privately used subnets.
_INTERNAL_SUPERNETS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
)


class HeadStep(StrEnum):
    """One named boundary in head provisioning, teardown, or inspection."""

    INSTANCE_LOOKUP = "instance_lookup"
    INSTANCE_CREATE = "instance_create"
    INSTANCE_ADDRESS = "instance_address"
    SUBNET_LOOKUP = "subnet_lookup"
    FIREWALL_LOOKUP = "firewall_lookup"
    FIREWALL_CREATE = "firewall_create"
    SECRET_LOOKUP = "secret_lookup"
    CONFIGURE = "configure"
    STAGE_WORKER_PACKAGE = "stage_worker_package"
    WORKER_PACKAGE_GENERATION = "worker_package_generation"
    STAGE_WHEEL = "stage_wheel"
    STAGE_DASHBOARD = "stage_dashboard"
    INSTALL = "install"
    BOOTSTRAP = "bootstrap"
    MIGRATE = "migrate"
    SERVICE_START = "service_start"
    SERVICE_VERIFY = "service_verify"
    INSTANCE_DELETE = "instance_delete"
    FIREWALL_DELETE = "firewall_delete"


class HeadError(RuntimeError):
    """Base failure carrying the step that did not complete."""

    def __init__(self, step: HeadStep, detail: str):
        self.step = step
        self.detail = detail
        super().__init__(f"eyvan head {step.value.replace('_', ' ')} failed: {detail}")


class HeadCommandError(HeadError):
    """One gcloud command failed and its diagnostic is preserved."""


class HeadOwnershipError(HeadError):
    """A resource with the expected name exists and Eyvan does not own it."""


class HeadIngressError(HeadError):
    """Control-plane ingress could not be restricted to internal VPC ranges."""


class HeadCommandRunner(tp.Protocol):
    """Typed subprocess boundary for every gcloud command a head issues."""

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        stdin_text: str | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Run one bounded gcloud command and return its complete result."""

        ...


def _run_gcloud(
    args: tp.Sequence[str],
    *,
    stdin_text: str | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded gcloud command without inheriting the terminal."""

    return subprocess.run(
        ["gcloud", *args],
        input=stdin_text,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=provider_cli_env(),
    )


def _validate(value: str, pattern: re.Pattern[str], *, field_name: str) -> str:
    """Return a trimmed value that matches a GCP resource-name pattern."""

    if not isinstance(value, str) or value != value.strip() or not pattern.fullmatch(value):
        raise ValueError(f"{field_name} {value!r} is not a valid GCP name")
    return value


def _shell_safe(value: str, *, field_name: str) -> str:
    """Return a value safe to place in an environment file a shell also sources.

    Not a style rule. ``EYVAN_DATABASE_URL`` reaches the head as a line the
    migration step sources, so a password containing ``$`` would be expanded to
    something else entirely and the failure would surface as an authentication
    error against a database that is fine.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if _UNSAFE_VALUE_PATTERN.search(value):
        raise ValueError(
            f"{field_name} contains a character a shell would interpret; percent-encode it "
            "(quotes, backslashes, backticks, dollar signs, and whitespace are refused)"
        )
    return value


def _private_network(value: str, *, field_name: str) -> ipaddress.IPv4Network:
    """Parse one IPv4 CIDR and refuse anything that is not an internal VPC range.

    Deliberately stricter than :attr:`ipaddress.IPv4Network.is_private`, which
    also accepts the documentation ranges: those are globally routed in
    practice, and "the standard library called it private" is not a reason to
    put a control-plane port behind one.
    """

    try:
        network = ipaddress.ip_network(value.strip(), strict=False)
    except ValueError as exc:
        raise ValueError(f"{field_name} {value!r} is not an IPv4 CIDR range") from exc
    if not isinstance(network, ipaddress.IPv4Network):
        raise ValueError(f"{field_name} {value!r} must be IPv4")
    if any(network.subnet_of(supernet) for supernet in _INTERNAL_SUPERNETS):
        return network
    raise ValueError(
        f"{field_name} {value!r} is not an internal VPC range "
        f"({', '.join(str(supernet) for supernet in _INTERNAL_SUPERNETS)})"
    )


@dataclass(frozen=True, slots=True)
class HeadSpec:
    """Every name and value one head is made of, resolved before anything runs.

    A spec is complete and inert. Building it cannot create a resource, so an
    operator can print one, diff two, or keep one in review without a project.
    """

    fleet: str
    project: str
    region: str
    zone: str
    machine_type: str = DEFAULT_HEAD_MACHINE_TYPE
    port: int = DEFAULT_HEAD_PORT
    network: str = "default"
    subnetwork: str | None = None
    egress: str = DEFAULT_HEAD_EGRESS
    image_family: str = DEFAULT_HEAD_IMAGE_FAMILY
    image_project: str = DEFAULT_HEAD_IMAGE_PROJECT
    boot_disk_size_gb: int = DEFAULT_HEAD_BOOT_DISK_GB
    package: str = STAGED_WHEEL_PACKAGE
    service_account: str | None = None
    database_url: str | None = None
    artifact_store_uri: str | None = None
    oidc_issuer: str | None = None
    oidc_audience: str = "eyvan"
    #: Projects whose principals may submit serialized callables. Granted by name
    #: and never inferred from an identity provider: a callable is a pickle this
    #: fleet deserializes and runs, so authenticating a principal answers who it
    #: is, not whether its project may execute arbitrary code here.
    trusted_callable_projects: tuple[str, ...] = ()
    #: Accept managed-artifact capability tokens travelling in clear text. Off by
    #: default. This command terminates no TLS and binds a private VPC address
    #: with VPC-only ingress, so callables were refused outright -- and the
    #: refusal held the accelerator rather than releasing it. Defensible only
    #: because the plane is unreachable from outside that network, which is why
    #: it is a flag an operator types rather than an inference.
    allow_insecure_artifact_http: bool = False
    #: The interpreter this head provisions for *submitting* work, matching the
    #: one its workers run. A pickled callable cannot load on another version, so
    #: a head running 3.12 against a Cloud TPU image shipping 3.10 can submit
    #: nothing at all -- which cost two provisioned slices and four dead ranks to
    #: discover. ``None`` when the fleet declares no interpreter, or when its
    #: pools disagree and there is no single one to provide.
    submitter_python: str | None = None
    source_ranges: tuple[str, ...] = ()
    iap_ssh: bool = True
    #: Run the head itself on Spot capacity. For development and qualification
    #: planes, where an hour of a head is the cost being avoided: GCP may stop the
    #: VM at any time. The termination action is STOP, never DELETE, so the boot
    #: disk -- and the PostgreSQL on it -- survives; starting the VM again brings
    #: the enabled unit back with its state. Off by default, because a stopped
    #: head stops every fleet it controls from being reconciled.
    spot: bool = False
    #: Write ``team: true`` into the head's platform file, so the plane serves
    #: team accounts. Off by default: the platform file is then byte-identical
    #: to what earlier releases wrote.
    team: bool = False

    @property
    def installed_description(self) -> str:
        """Describe what gets installed, without leaking the internal sentinel."""

        return "this Eyvan distribution" if self.stages_wheel else self.package

    @property
    def stages_wheel(self) -> bool:
        """Whether this run ships its own distribution instead of resolving a name."""

        return self.package == STAGED_WHEEL_PACKAGE

    def host_install_spec(self) -> HostInstallSpec:
        """Return the host-local half of this head's installer."""

        return HostInstallSpec(
            port=self.port,
            install_target=f"{HEAD_WHEEL_DIR}/*.whl" if self.stages_wheel else shlex.quote(self.package),
            force_reinstall=self.stages_wheel,
            local_database=self.database_url is None,
        )

    def __post_init__(self) -> None:
        """Reject a spec that could not be provisioned before any command runs."""

        if self.submitter_python is not None and (
            not isinstance(self.submitter_python, str)
            or re.fullmatch(r"3\.(?:11|12|13)(?:\.[0-9]+)?", self.submitter_python) is None
        ):
            raise ValueError("Eyvan submitters require Python 3.11 through 3.13")
        _validate(self.fleet, _RESOURCE_NAME_PATTERN, field_name="fleet name")
        _validate(self.project, _PROJECT_ID_PATTERN, field_name="project")
        _validate(self.region, _REGION_PATTERN, field_name="region")
        _validate(self.zone, _ZONE_PATTERN, field_name="zone")
        _validate(self.network, _RESOURCE_NAME_PATTERN, field_name="network")
        if not self.zone.startswith(f"{self.region}-"):
            raise ValueError(f"zone {self.zone!r} is not in region {self.region!r}")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("control-plane port must be between 1 and 65535")
        if self.egress not in BOOTSTRAP_EGRESS_MODES:
            legal = ", ".join(BOOTSTRAP_EGRESS_MODES)
            raise ValueError(f"egress {self.egress!r} must be one of: {legal}")
        if isinstance(self.boot_disk_size_gb, bool) or not isinstance(self.boot_disk_size_gb, int):
            raise ValueError("boot disk size must be a positive integer number of GB")
        if self.boot_disk_size_gb <= 0:
            raise ValueError("boot disk size must be a positive integer number of GB")
        if self.database_url is not None and not self.database_url.startswith("postgresql"):
            raise ValueError("an external control-plane database URL must use a postgresql driver")
        # Every one of these becomes a line in a systemd EnvironmentFile that a
        # POSIX shell also sources during migration. A quote or a dollar sign in
        # one would be interpreted rather than passed, so it is refused here
        # instead of silently changing the value the server ends up reading.
        for name in ("database_url", "artifact_store_uri", "oidc_issuer", "oidc_audience", "package"):
            value = getattr(self, name)
            if value is not None:
                _shell_safe(value, field_name=name.replace("_", " "))
        for value in self.trusted_callable_projects:
            _shell_safe(value, field_name="trusted callable project")
            # EYVAN_TRUSTED_CALLABLE_PROJECTS is one comma-separated line in that
            # same file, so a comma inside an entry would silently become two
            # entries and grant code execution to a project nobody named.
            if "," in value:
                raise ValueError(f"trusted callable project {value!r} must not contain a comma")
        for value in self.source_ranges:
            _private_network(value, field_name="control-plane source range")
        if len(self.instance_name) > 63:
            raise ValueError(f"fleet name {self.fleet!r} is too long to derive a head instance name")

    @classmethod
    def from_fleet_config(
        cls,
        config: EyvanFleetConfig,
        **overrides: tp.Any,
    ) -> HeadSpec:
        """Derive a head from the fleet it will control.

        The head belongs in the fleet's own project and zone: a control plane in
        another region pays the round trip on every agent poll, and one in
        another project cannot read the artifact bucket the fleet writes to.
        """

        if config.provider.type != "gcp":
            raise ValueError(f"eyvan head up supports the gcp provider, not {config.provider.type!r}")
        project = overrides.pop("project", None) or config.provider.project
        if project is None:
            raise ValueError("fleet configuration has no provider.project and none was supplied")
        zone = overrides.pop("zone", None) or config.provider.zone
        if zone is None:
            raise ValueError("fleet configuration has no provider.zone and none was supplied")
        region = overrides.pop("region", None) or config.provider.region
        egress = overrides.pop("egress", None) or _fleet_bootstrap_egress(config)
        submitter_python = overrides.pop("submitter_python", None) or _fleet_runtime_python(config)
        if egress is None:
            legal = ", ".join(BOOTSTRAP_EGRESS_MODES)
            raise ValueError(
                f"fleet pools declare no single provider_options.bootstrap_egress; pass --egress with one of: {legal}"
            )
        supplied = {key: value for key, value in overrides.items() if value is not None}
        if submitter_python is not None:
            supplied["submitter_python"] = submitter_python
        return cls(
            fleet=config.name,
            project=project,
            region=region,
            zone=zone,
            egress=egress,
            **supplied,
        )

    @property
    def instance_name(self) -> str:
        """Return the deterministic head VM name."""

        return f"{self.fleet}-head"

    @property
    def network_tag(self) -> str:
        """Return the tag every managed head firewall rule targets."""

        return f"{self.fleet}-head"

    @property
    def firewall_rule_name(self) -> str:
        """Return the deterministic control-plane ingress rule name."""

        return f"{self.instance_name}-api"

    @property
    def ssh_firewall_rule_name(self) -> str:
        """Return the deterministic IAP SSH ingress rule name."""

        return f"{self.instance_name}-iap-ssh"

    @property
    def has_external_address(self) -> bool:
        """Return whether this head is created with an external IPv4 address."""

        return self.egress == "public-internet"

    @property
    def labels(self) -> dict[str, str]:
        """Return the ownership labels adoption is checked against."""

        return {
            HEAD_MANAGED_LABEL_KEY: HEAD_MANAGED_LABEL_VALUE,
            HEAD_FLEET_LABEL_KEY: self.fleet,
        }

    @property
    def firewall_description(self) -> str:
        """Return the exact ownership marker expected on a managed rule."""

        return f"{HEAD_FIREWALL_DESCRIPTION_PREFIX}:{self.fleet}"

    @property
    def resource_names(self) -> dict[str, str]:
        """Return every resource name this spec can cause to exist."""

        names = {
            "instance": self.instance_name,
            "network_tag": self.network_tag,
            "firewall_rule": self.firewall_rule_name,
        }
        if self.iap_ssh:
            names["ssh_firewall_rule"] = self.ssh_firewall_rule_name
        return names

    def as_document(self) -> JsonObject:
        """Return the non-secret spec for automation and dry-run output."""

        return {
            "boot_disk_size_gb": self.boot_disk_size_gb,
            "database": "external" if self.database_url else "head-local postgresql",
            "egress": self.egress,
            "external_address": self.has_external_address,
            "fleet": self.fleet,
            "iap_ssh": self.iap_ssh,
            "machine_type": self.machine_type,
            "provisioning_model": "spot" if self.spot else "standard",
            "network": self.network,
            "package": self.package,
            "port": self.port,
            "project": self.project,
            "region": self.region,
            "resources": dict(sorted(self.resource_names.items())),
            # A trust grant is the one decision in this document a reviewer most
            # needs to see before it is made, and it would otherwise appear only
            # inside an environment file on a machine that does not exist yet.
            "trusted_callable_projects": list(self.trusted_callable_projects),
            "zone": self.zone,
            # Present only when set, so the default document is unchanged.
            **({"team": True} if self.team else {}),
        }


def _fleet_bootstrap_egress(config: EyvanFleetConfig) -> str | None:
    """Return the one egress mode every pool agrees on, or ``None``.

    Inheriting it is the point: a head that cannot reach the same places its
    workers reach is a head that installs on some fleets and hangs on others,
    and the operator has no way to tell which from the error.
    """

    declared = {
        str(pool.compiled_provider_options.get("bootstrap_egress"))
        for pool in config.pools.values()
        if isinstance(pool.compiled_provider_options.get("bootstrap_egress"), str)
    }
    legal = declared & set(BOOTSTRAP_EGRESS_MODES)
    if len(legal) == 1 and len(declared) == 1:
        return legal.pop()
    return None


def _fleet_runtime_python(config: EyvanFleetConfig) -> str | None:
    """Return the interpreter every pool agrees its workers run, or ``None``.

    Same reasoning as the egress mode above, and the same consequence for getting
    it wrong: a serialized callable cannot load on an interpreter other than the
    one that pickled it, so a head whose Python differs from its workers' can
    submit nothing. Pools that disagree get ``None`` -- there is no single answer
    to provision, and guessing one would make some of the fleet unusable while
    looking configured.
    """

    declared: set[str] = set()
    for payload in config.pool_payloads():
        runtime = payload.get("runtime")
        python = runtime.get("python") if isinstance(runtime, dict) else None
        if isinstance(python, str) and python:
            declared.add(python)
    return declared.pop() if len(declared) == 1 else None


@dataclass(frozen=True, slots=True)
class HeadSecrets:
    """The credentials the server refuses to start without, generated for you.

    Three separate restarts is what the manual path costs: the server reads
    ``EYVAN_METRICS_TOKEN``, then ``EYVAN_ARTIFACT_TOKEN_KEY``, then the agent
    secret, and each one is discovered only after the previous is supplied.
    They are all generated at once here, and never printed.
    """

    metrics_token: str
    artifact_token_key: str
    agent_shared_secret: str
    database_password: str
    bootstrap_token: str
    #: Root credential every native worker's own bearer is derived from. Required
    #: by the cell agent and by the GCP capacity driver whenever the executor is
    #: ``eyvan-native``, which is every TPU fleet this command exists to serve.
    native_worker_root_secret: str

    def __post_init__(self) -> None:
        """Refuse material too short to be worth generating."""

        for name in (
            "metrics_token",
            "artifact_token_key",
            "agent_shared_secret",
            "bootstrap_token",
            "native_worker_root_secret",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) < 32:
                raise ValueError(f"{name} must be at least 32 characters")
        # The database password is interpolated into one CREATE ROLE statement on
        # the head, so its character set is a correctness property and not a
        # style preference: a quote in it would end the statement.
        if not _SAFE_PASSWORD_PATTERN.fullmatch(self.database_password):
            raise ValueError("database_password must be at least 24 URL-safe base64 characters")

    @classmethod
    def generate(cls) -> HeadSecrets:
        """Return fresh high-entropy material from the system CSPRNG."""

        return cls(
            metrics_token=secrets.token_urlsafe(32),
            artifact_token_key=secrets.token_urlsafe(32),
            agent_shared_secret=secrets.token_urlsafe(32),
            database_password=secrets.token_urlsafe(24),
            bootstrap_token=secrets.token_urlsafe(32),
            native_worker_root_secret=secrets.token_urlsafe(32),
        )

    def __repr__(self) -> str:
        """Keep generated material out of tracebacks and debug output."""

        return (
            "HeadSecrets(metrics_token=***, artifact_token_key=***, agent_shared_secret=***, "
            "database_password=***, bootstrap_token=***, native_worker_root_secret=***)"
        )

    def values(self) -> tuple[str, ...]:
        """Return every secret value, for redaction checks only."""

        # ``bootstrap_token`` belongs here and was missing: every redaction
        # assertion in the suite goes through this tuple, so the one credential
        # that authenticates a whole control plane was the one credential no
        # test could catch leaking.
        return (
            self.metrics_token,
            self.artifact_token_key,
            self.agent_shared_secret,
            self.database_password,
            self.bootstrap_token,
            self.native_worker_root_secret,
        )

    def database_url(self, spec: HeadSpec) -> str:
        """Return the external URL when one was given, else the head-local one."""

        if spec.database_url is not None:
            return spec.database_url
        return f"postgresql+asyncpg://eyvan:{self.database_password}@127.0.0.1:5432/eyvan"

    def environment(
        self,
        spec: HeadSpec,
        *,
        address: str,
        worker_package: WorkerPackage | None = None,
        serves_dashboard: bool = False,
    ) -> dict[str, str]:
        """Return the complete server environment for this head."""

        environment = {
            "EYVAN_AGENT_SHARED_SECRET": self.agent_shared_secret,
            "EYVAN_ARTIFACT_TOKEN_KEY": self.artifact_token_key,
            "EYVAN_DATABASE_URL": self.database_url(spec),
            "EYVAN_METRICS_TOKEN": self.metrics_token,
            # Read by both units this file feeds. The cell agent refuses to start
            # without it whenever the executor is eyvan-native, and the GCP
            # capacity driver refuses to provision a TPU or Compute cell without
            # it. Both head units need the same root secret for provisioning and
            # worker authentication.
            "EYVAN_NATIVE_WORKER_ROOT_SECRET": self.native_worker_root_secret,
            # HTTPS on the private address, under the head's own CA. A network
            # bind refuses plaintext without an explicit opt-out, so configure
            # TLS here. The bootstrap script issues the certificate.
            "EYVAN_PUBLIC_URL": f"https://{address}:{spec.port}",
            "EYVAN_TLS_CERT_FILE": HEAD_TLS_CERT_PATH,
            "EYVAN_TLS_KEY_FILE": HEAD_TLS_KEY_PATH,
            # Read by the server to hand the CA to the jobs it dispatches, whose
            # callbacks to this plane must verify it.
            CONTROL_PLANE_CA_FILE_ENV: str(HEAD_CONTROL_PLANE_CA_PATH),
            "EYVAN_REPOSITORY_BACKEND": "postgres",
        }
        if spec.artifact_store_uri is not None:
            environment["EYVAN_ARTIFACT_STORE_URI"] = spec.artifact_store_uri
            # The one document every agent for this fleet reconciles against.
            # Left unset, each agent keeps a private local copy, so the capacity
            # driver's writes and the control plane's reads never meet: a cell
            # reaches `ready` publishing no worker endpoints, and dispatch waits
            # forever for an address nobody wrote down, even when the worker is up.
            environment["EYVAN_FLEET_STATE_URI"] = fleet_state_uri(spec.artifact_store_uri, spec.fleet)
            # Read by the GCP capacity factory to grant each worker's service
            # account read access to the bucket holding its own wheel. Unset, the
            # driver returns before granting anything, so a worker in a project
            # whose default service account cannot already read that bucket fails
            # to download the Eyvan it was told to install -- at bootstrap, on the
            # accelerator, after it has been provisioned and billed.
            environment["EYVAN_NATIVE_WORKER_ARTIFACT_ROOT_URI"] = spec.artifact_store_uri
        if spec.allow_insecure_artifact_http:
            environment["EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP"] = "1"
        if serves_dashboard:
            # Set only when a bundle was actually staged. The server mounts the
            # UI only if this directory exists, so naming it unconditionally
            # would be harmless -- and still wrong, because the head's own
            # output would promise a dashboard that is not there.
            environment["EYVAN_DASHBOARD_DIR"] = HEAD_DASHBOARD_DIR
        if worker_package is not None:
            # Without these three the GCP capacity driver refuses every request
            # with "eyvan-native TPU capacity requires EYVAN_NATIVE_WORKER_PACKAGE",
            # which surfaces on the first submission rather than at install time.
            environment["EYVAN_NATIVE_WORKER_PACKAGE"] = worker_package.uri
            environment["EYVAN_NATIVE_WORKER_PACKAGE_SHA256"] = worker_package.sha256
            environment["EYVAN_NATIVE_WORKER_PACKAGE_GENERATION"] = worker_package.generation
        if spec.oidc_issuer is not None:
            environment["EYVAN_OIDC_ISSUER"] = spec.oidc_issuer
            if spec.oidc_audience is not None:
                environment["EYVAN_OIDC_AUDIENCE"] = spec.oidc_audience
        else:
            # Startup refuses both at once, so this only ever fills the gap.
            environment["EYVAN_BOOTSTRAP_TOKEN"] = self.bootstrap_token
        # Callable trust, decided outside that branch because it is not a
        # property of how principals authenticate. A loopback development plane
        # trusts its own default project, so the decorator API worked on a laptop
        # and was refused with 403 after authenticating on a real head. Writing
        # the default only in the bootstrap branch fixed that for exactly the
        # deployment that has no identity provider, and left it broken for the
        # one that does -- which is the deployment anyone runs for real.
        #
        # Deliberately not auto-granted under an issuer. Deserializing a callable
        # executes arbitrary code on the fleet, and `callable_trust` grants that
        # to *every* principal holding SUBMIT in a trusted project -- under OIDC,
        # a set the head operator never enumerated. So an operator names the
        # projects; bootstrap mode, which has exactly one principal, keeps
        # trusting that principal's own project as before.
        trusted = spec.trusted_callable_projects
        if not trusted and spec.oidc_issuer is None:
            trusted = (BOOTSTRAP_CALLABLE_PROJECT,)
        if trusted:
            environment["EYVAN_TRUSTED_CALLABLE_PROJECTS"] = ",".join(trusted)
        return environment

    def environment_file(
        self,
        spec: HeadSpec,
        *,
        address: str,
        worker_package: WorkerPackage | None = None,
        serves_dashboard: bool = False,
    ) -> str:
        """Render the systemd ``EnvironmentFile`` this head reads at start."""

        resolved = self.environment(
            spec,
            address=address,
            worker_package=worker_package,
            serves_dashboard=serves_dashboard,
        )
        lines = [f"{key}={value}" for key, value in sorted(resolved.items())]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class HeadAction:
    """One planned command, complete enough to read before it is run."""

    step: HeadStep
    summary: str
    args: tuple[str, ...]
    mutates: bool = False
    #: Whether this action delivers bytes on stdin instead of in its argv.
    #: Generated material is only ever delivered this way, so it never reaches
    #: a process table on either side.
    sends_stdin: bool = False
    #: Whether this action is skipped when the resource already exists and
    #: carries Eyvan's ownership marker.
    skipped_when_present: bool = False

    def __post_init__(self) -> None:
        """Refuse an action that could not be printed or run."""

        if not self.args:
            raise ValueError("a head action must carry a concrete command")
        if not self.summary.strip():
            raise ValueError("a head action must carry a summary")

    @property
    def command(self) -> str:
        """Return the exact shell-quoted gcloud invocation."""

        return shlex.join(("gcloud", *self.args))

    def as_document(self) -> JsonObject:
        """Return one automation-safe planned action."""

        return {
            "command": self.command,
            "mutates": self.mutates,
            "skipped_when_present": self.skipped_when_present,
            "step": self.step.value,
            "summary": self.summary,
            "writes_stdin": self.sends_stdin,
        }


@dataclass(frozen=True, slots=True)
class HeadPlan:
    """Every action ``eyvan head up`` would take, and every name it would use."""

    spec: HeadSpec
    actions: tuple[HeadAction, ...]

    @property
    def mutating_actions(self) -> tuple[HeadAction, ...]:
        """Return only the actions that would change the project."""

        return tuple(action for action in self.actions if action.mutates)

    def as_document(self) -> JsonObject:
        """Return the complete plan as one deterministic JSON document."""

        return {
            "actions": [action.as_document() for action in self.actions],
            "dry_run": True,
            "spec": self.spec.as_document(),
        }

    def render_lines(self) -> tuple[str, ...]:
        """Return human-readable plan output, one action per numbered line."""

        lines = [
            f"eyvan head up would provision {self.spec.instance_name} "
            f"in {self.spec.project}/{self.spec.zone} and change nothing now.",
            "",
            "Resources:",
        ]
        lines.extend(f"  {kind:20} {name}" for kind, name in sorted(self.spec.resource_names.items()))
        lines.extend(("", "Actions:"))
        for index, action in enumerate(self.actions, start=1):
            marker = "change" if action.mutates else "read"
            lines.append(f"  {index:2}. [{marker}] {action.summary}")
            lines.append(f"      {action.command}")
        return tuple(lines)


#: Printed in a plan where a value is only knowable after a read that a dry run
#: deliberately does not perform. A plan that invented a number here would be
#: the one thing a dry run must never be: plausible and wrong.
UNRESOLVED_SOURCE_RANGES = "<internal-subnet-ranges>"

_ENVIRONMENT_PRESENT = "eyvan-head-env-present"
_ENVIRONMENT_ABSENT = "eyvan-head-env-absent"


@dataclass(frozen=True, slots=True)
class WorkerPackage:
    """The wheel a native worker downloads, and the two facts that pin it.

    A TPU or Compute worker bootstraps by fetching this object and refusing it
    unless the bytes hash to ``sha256``; the generation makes the URI name one
    immutable object rather than whatever currently sits at that path. Both are
    required by the GCP capacity driver, and a head that omitted them had every
    capacity request rejected with "requires EYVAN_NATIVE_WORKER_PACKAGE" --
    after the fleet applied, so the failure surfaced on the first submission
    rather than at install time.
    """

    uri: str
    sha256: str
    generation: str


@dataclass(frozen=True, slots=True)
class _LocalWheel:
    """One built wheel and the digest the head must reproduce."""

    data: bytes
    digest: str
    filename: str


def staged_wheel_filename() -> str:
    """Return the PEP 427 name the staged wheel lands under.

    Derived from the version rather than read from the built file so that the
    dry-run plan and the executed command name the same path. pip refuses a file
    that is not named like a wheel, which is how "eyvan-head.whl" failed on a
    real head after everything else had already been installed.
    """

    from eyvan import __version__

    return f"eyvan-{__version__}-py3-none-any.whl"


def worker_package_uri(artifact_store_uri: str, digest: str, filename: str) -> str:
    """Return the immutable object name this head's worker wheel is staged under.

    Content-addressed, so two heads that install the same distribution converge
    on one object and a changed wheel can never overwrite the bytes an existing
    worker was told to expect.
    """

    return f"{artifact_store_uri.rstrip('/')}/native-worker/{digest}/{filename}"


def fleet_state_uri(artifact_store_uri: str, fleet: str) -> str:
    """Return the shared fleet-state document this head's agents reconcile against.

    Unset, every agent falls back to a private local path, so the driver's writes
    and the control plane's reads never meet: a cell reaches ``ready`` carrying no
    worker endpoints and dispatch waits forever on an address nobody published.
    """

    return f"{artifact_store_uri.rstrip('/')}/fleet-state/{fleet}.json"


def _local_dashboard_bundle() -> bytes | None:
    """Return this checkout's built dashboard as a gzipped tar, or ``None``.

    A checkout's ``dashboard/dist`` is used first, then the copy a release
    build placed inside the package (``eyvan/_dashboard``), so a head launched
    from ``pip install eyvan`` ships a UI too. ``None`` -- a checkout that has
    not run ``npm run build``, or a wheel built without staging -- says so
    rather than pretending.
    """

    package = Path(eyvan_package_root())
    dist = next(
        (
            candidate
            for candidate in (package.parent / "dashboard" / "dist", package / "_dashboard")
            if (candidate / "index.html").is_file()
        ),
        None,
    )
    if dist is None:
        return None
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        # ``arcname="."`` so the archive unpacks *into* the target directory
        # rather than creating a nested one the server would not find.
        archive.add(dist, arcname=".", filter=_reproducible_member)
    return buffer.getvalue()


def _reproducible_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip build-host identity so the same bundle hashes the same anywhere."""

    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.mtime = 0
    return member


def eyvan_package_root() -> Path:
    """Return the installed package directory, without importing at module scope."""

    import eyvan

    return Path(eyvan.__file__).resolve().parent


def _planned_worker_wheel() -> _LocalWheel:
    """Return a wheel stand-in for rendering the plan, which builds nothing.

    ``--dry-run`` has to name the object it would publish without producing the
    bytes to publish, so the digest is unknown at plan time and shown as such.
    """

    return _LocalWheel(data=b"", digest="<sha256-of-this-distribution>", filename=staged_wheel_filename())


def _local_wheel_bytes() -> _LocalWheel:
    """Return this installation's own wheel, however Eyvan was installed.

    Reuses the source that ``eyvan runtime init`` already uses, so a checkout
    builds from the tree and a wheel install repacks the distribution it is
    running. Writing a second copy of that logic here is how the two drift.
    """

    from eyvan.sdk.commands.runtime import native_worker_wheel_builder

    builder = native_worker_wheel_builder()
    try:
        path = builder.build()
        wheel_path = Path(path)
        data = wheel_path.read_bytes()
        filename = wheel_path.name
    finally:
        close = getattr(builder, "close", None)
        if callable(close):
            close()
    return _LocalWheel(data=data, digest=hashlib.sha256(data).hexdigest(), filename=filename)


def _head_tls_lines() -> tuple[str, ...]:
    """Return the bootstrap lines that give the head a CA and a server certificate.

    The head's address is read back out of its environment file, so an adopted
    head whose address changed is reissued a certificate for the new one.
    """

    return host_tls_lines(
        address_lines=(
            "# TLS for the control-plane listener, under a CA this head owns.",
            f"head_address=$(sed -n 's#^EYVAN_PUBLIC_URL=https://\\([^:/]*\\):.*#\\1#p' {HEAD_ENVIRONMENT_PATH})",
            'if [ -z "$head_address" ]; then',
            f"  echo 'EYVAN_PUBLIC_URL in {HEAD_ENVIRONMENT_PATH} must be https://ADDRESS:PORT' >&2",
            "  exit 1",
            "fi",
        ),
        group="eyvan-head",
    )


def host_tls_lines(*, address_lines: tuple[str, ...], group: str, dns_name: bool = False) -> tuple[str, ...]:
    """Return installer lines that give a host plane a CA and a server certificate.

    ``address_lines`` must set the shell variable ``head_address`` (and
    ``head_dns_name`` when ``dns_name`` is true). The CA is created once and
    kept, so clients that already trust it keep working across re-runs. The
    server certificate is reissued whenever it does not name the current
    address (or DNS name), does not chain to that CA, or expires within 30 days.

    Args:
        address_lines: Lines that set ``head_address`` (and ``head_dns_name``).
        group: Group of the service account that must read the server key.
        dns_name: Whether the certificate also names ``$head_dns_name``.

    Returns:
        The installer lines, starting and ending with a blank line.
    """

    ca = str(HEAD_CONTROL_PLANE_CA_PATH)
    name_check = (
        (
            f'  || ! openssl x509 -in {HEAD_TLS_CERT_PATH} -noout -checkhost "$head_dns_name" 2>/dev/null '
            "| grep -q 'does match' \\",
        )
        if dns_name
        else ()
    )
    san = "IP:%s,DNS:%s" if dns_name else "IP:%s"
    san_arguments = '"$head_address" "$head_dns_name"' if dns_name else '"$head_address"'
    return (
        "",
        *address_lines,
        f"install -d -o root -g {group} -m 0750 {HEAD_TLS_DIR}",
        f"if [ ! -s {HEAD_TLS_CA_KEY_PATH} ] || [ ! -s {ca} ]; then",
        "  openssl req -x509 -newkey rsa:3072 -nodes -days 3650 -subj '/CN=Eyvan control-plane head CA' "
        f"-keyout {HEAD_TLS_CA_KEY_PATH} -out {ca} "
        "-addext 'basicConstraints=critical,CA:TRUE' -addext 'keyUsage=critical,keyCertSign,cRLSign'",
        "fi",
        f"chown root:root {HEAD_TLS_CA_KEY_PATH}",
        f"chmod 0600 {HEAD_TLS_CA_KEY_PATH}",
        f"chmod 0644 {ca}",
        f'if ! openssl x509 -in {HEAD_TLS_CERT_PATH} -noout -checkip "$head_address" 2>/dev/null '
        "| grep -q 'does match' \\",
        *name_check,
        f"  || ! openssl verify -CAfile {ca} {HEAD_TLS_CERT_PATH} >/dev/null 2>&1 \\",
        f"  || ! openssl x509 -in {HEAD_TLS_CERT_PATH} -noout -checkend 2592000 >/dev/null 2>&1; then",
        f'  openssl req -new -newkey rsa:3072 -nodes -subj "/CN=$head_address" -keyout {HEAD_TLS_KEY_PATH} '
        f"-out {HEAD_TLS_DIR}/server.csr",
        f"  openssl x509 -req -in {HEAD_TLS_DIR}/server.csr -CA {ca} -CAkey {HEAD_TLS_CA_KEY_PATH} "
        f"-CAcreateserial -CAserial {HEAD_TLS_DIR}/ca.srl -days 825 -out {HEAD_TLS_CERT_PATH} "
        f"-extfile <(printf 'subjectAltName={san}\\nbasicConstraints=CA:FALSE\\n"
        f"keyUsage=digitalSignature,keyEncipherment\\nextendedKeyUsage=serverAuth\\n' {san_arguments})",
        f"  rm -f {HEAD_TLS_DIR}/server.csr",
        "fi",
        f"chown root:{group} {HEAD_TLS_KEY_PATH} {HEAD_TLS_CERT_PATH}",
        f"chmod 0640 {HEAD_TLS_KEY_PATH}",
        f"chmod 0644 {HEAD_TLS_CERT_PATH}",
        "",
    )


@dataclass(frozen=True, slots=True)
class HostInstallSpec:
    """The host-local half of an installer: no project, zone, fleet or provider.

    ``eyvan head up`` renders one from its :class:`HeadSpec`; ``eyvan team up``
    builds one directly on the machine it is installing. Every field is a value
    the rendered script embeds, so each is validated by whoever builds it.
    """

    port: int
    #: What pip installs, already shell-ready: a quoted specifier or wheel path,
    #: or the deliberately unquoted staged-wheel glob.
    install_target: str
    #: Reinstall the same version again, for a wheel whose version did not change.
    force_reinstall: bool
    local_database: bool = True
    service_user: str = "eyvan-head"
    environment_path: str = HEAD_ENVIRONMENT_PATH
    unit_description: str = "Eyvan control plane head"
    #: The word installer messages use for what is being installed.
    label: str = "head"

    @property
    def venv_python(self) -> str:
        """Return the interpreter of the plane's virtual environment."""

        return f"{HEAD_VENV_PATH}/bin/python"


def host_prelude_lines(host: HostInstallSpec) -> tuple[str, ...]:
    """Return the script header, the apt helper, the environment check and the service account."""

    user = host.service_user
    return (
        "#!/bin/bash",
        "set -euo pipefail",
        "umask 022",
        "export DEBIAN_FRONTEND=noninteractive",
        "eyvan_apt_get() {",
        "  if ! command -v apt-get >/dev/null 2>&1; then",
        f"    echo 'the Eyvan {host.label} installer requires apt-get' >&2",
        "    return 127",
        "  fi",
        '  command apt-get -o DPkg::Lock::Timeout=300 "$@"',
        "}",
        "",
        f"if [ ! -r {host.environment_path} ]; then",
        f"  echo 'the Eyvan {host.label} environment {host.environment_path} is missing' >&2",
        "  exit 1",
        "fi",
        "",
        f"if ! getent group {user} >/dev/null 2>&1; then",
        f"  groupadd --system {user}",
        "fi",
        f"if ! id -u {user} >/dev/null 2>&1; then",
        f"  useradd --system --gid {user} --home-dir /var/lib/eyvan --create-home --shell /usr/sbin/nologin {user}",
        "fi",
        f"install -d -o {user} -g {user} -m 0750 /var/lib/eyvan",
    )


def host_server_unit(host: HostInstallSpec) -> str:
    """Return the systemd unit that runs ``eyvan-server`` on every interface."""

    after = "network-online.target postgresql.service" if host.local_database else "network-online.target"
    return "\n".join(
        (
            "[Unit]",
            f"Description={host.unit_description}",
            f"After={after}",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"User={host.service_user}",
            f"Group={host.service_user}",
            f"EnvironmentFile={host.environment_path}",
            "Environment=PYTHONUNBUFFERED=1",
            f"ExecStart={HEAD_VENV_PATH}/bin/eyvan-server --host 0.0.0.0 --port {host.port}",
            "Restart=always",
            "RestartSec=2",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
        )
    )


def host_venv_lines(host: HostInstallSpec) -> tuple[str, ...]:
    """Return the lines that build the plane's venv, install Eyvan and put the CLI on PATH."""

    venv_python = host.venv_python
    target = host.install_target
    return (
        "# The virtual environment, built with the interpreter the image ships.",
        "# `uv` is deliberately not required: a missing uv is one of the failures",
        "# this command exists to remove, and python3-venv is in the base image or",
        "# one apt-get away.",
        f"if ! {venv_python} -m pip --version >/dev/null 2>&1; then",
        "  if ! python3 -m ensurepip --version >/dev/null 2>&1; then",
        "    eyvan_apt_get update",
        "    eyvan_apt_get install -y --no-install-recommends python3-venv",
        "  fi",
        f"  rm -rf {HEAD_VENV_PATH}",
        "  install -d -m 0755 /opt/eyvan",
        f"  python3 -m venv {HEAD_VENV_PATH}",
        "fi",
        *(
            (
                # A staged wheel's glob is deliberately unquoted: the wheel keeps
                # its own PEP 427 filename, which is not known at render time.
                # Installing the file rather than the name `eyvan` means no index
                # is consulted for Eyvan itself, while its dependencies still
                # resolve normally.
                f"{venv_python} -m pip install --disable-pip-version-check --upgrade {target}",
                # Then the same wheel again, forced. A rebuilt wheel usually
                # carries the *same* version, and `--upgrade` is a no-op there --
                # the plane would keep serving the previous code while reporting a
                # successful install. `--no-deps` because the line above already
                # resolved the dependency tree.
                f"{venv_python} -m pip install --disable-pip-version-check --force-reinstall --no-deps {target}",
            )
            if host.force_reinstall
            # An index install is a version specifier, so --upgrade is doing real
            # resolution work and a forced pass would only reinstall what it chose.
            else (f"{venv_python} -m pip install --disable-pip-version-check --upgrade {target}",)
        ),
        f"chmod -R a+rX {HEAD_VENV_PATH}",
        "",
        "# Put the CLI on PATH. A head that has just told an operator to run",
        "# `eyvan up <fleet.yaml>` and then answers `command not found` has",
        "# undone its own instruction, and the venv path is not something anyone",
        "# should have to be told. An existing regular file is left alone: it is",
        "# somebody else's install, and this script owns /opt, not /usr/local.",
        "install -d -m 0755 /usr/local/bin",
        "if [ ! -e /usr/local/bin/eyvan ] || [ -L /usr/local/bin/eyvan ]; then",
        f"  ln -sfn {HEAD_VENV_PATH}/bin/eyvan /usr/local/bin/eyvan",
        "else",
        "  echo 'eyvan: /usr/local/bin/eyvan is not a symlink; leaving it as it is' >&2",
        "fi",
        "",
    )


def host_postgres_lines(host: HostInstallSpec) -> tuple[str, ...]:
    """Return the lines that install a plane-local PostgreSQL and its ``eyvan`` role and database.

    The role's password is read out of the environment file by the venv's
    interpreter and handed to psql on stdin, so it never appears in an argv.
    """

    return (
        "# A head-local PostgreSQL: the control plane refuses a network bind on",
        "# any other backend, and asking an operator to bring a database before",
        "# they have a head is how a one-command install becomes fifteen.",
        "if ! command -v psql >/dev/null 2>&1; then",
        "  eyvan_apt_get update",
        "  eyvan_apt_get install -y --no-install-recommends postgresql",
        "fi",
        "systemctl enable --now postgresql",
        "# Read straight out of the environment file rather than sourcing it:",
        "# sourcing runs an operator-supplied URL through the shell, and a",
        "# database password is exactly the string that should never be parsed.",
        "eyvan_database_password=$(",
        f"  {host.venv_python} - {host.environment_path} <<'EYVAN_HEAD_PASSWORD'",
        "import sys",
        "import urllib.parse",
        "",
        'url = ""',
        'with open(sys.argv[1], encoding="utf-8") as handle:',
        "    for line in handle:",
        '        key, separator, value = line.partition("=")',
        '        if separator and key.strip() == "EYVAN_DATABASE_URL":',
        "            url = value.strip()",
        'print(urllib.parse.unquote(urllib.parse.urlsplit(url).password or ""))',
        "EYVAN_HEAD_PASSWORD",
        ")",
        'if [ -z "${eyvan_database_password}" ]; then',
        f"  echo 'the {host.label} database URL carries no password' >&2",
        "  exit 1",
        "fi",
        "# Delivered on stdin: a password in psql's argv is readable in ps.",
        "runuser -u postgres -- psql -v ON_ERROR_STOP=1 -f - <<EYVAN_HEAD_ROLE",
        "DO \\$\\$",
        "BEGIN",
        "  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'eyvan') THEN",
        "    CREATE ROLE eyvan LOGIN PASSWORD '${eyvan_database_password}';",
        "  ELSE",
        "    ALTER ROLE eyvan LOGIN PASSWORD '${eyvan_database_password}';",
        "  END IF;",
        "END",
        "\\$\\$;",
        "EYVAN_HEAD_ROLE",
        "if ! runuser -u postgres -- psql -tAc \"SELECT 1 FROM pg_database WHERE datname = 'eyvan'\" | grep -q 1; then",
        "  runuser -u postgres -- createdb --owner eyvan eyvan",
        "fi",
        "",
    )


#: Keys of :func:`host_plane_environment` whose values are credentials.
HOST_PLANE_SECRET_KEYS: frozenset[str] = frozenset(
    {"EYVAN_ARTIFACT_TOKEN_KEY", "EYVAN_DATABASE_URL", "EYVAN_METRICS_TOKEN"}
)


def host_plane_environment(
    *,
    public_url: str,
    database_url: str,
    metrics_token: str,
    artifact_token_key: str,
    artifact_store_uri: str,
) -> dict[str, str]:
    """Return the infrastructure environment of a single-host plane, and no mode setting.

    Team mode is deliberately absent: it comes from the platform file or the
    server's ``--team`` flag, never from an environment variable.
    """

    return {
        "EYVAN_ARTIFACT_STORE_URI": artifact_store_uri,
        "EYVAN_ARTIFACT_TOKEN_KEY": artifact_token_key,
        # Handed to dispatched jobs, whose callbacks to this plane verify it.
        CONTROL_PLANE_CA_FILE_ENV: str(HEAD_CONTROL_PLANE_CA_PATH),
        "EYVAN_DATABASE_URL": database_url,
        "EYVAN_METRICS_TOKEN": metrics_token,
        "EYVAN_PUBLIC_URL": public_url,
        "EYVAN_REPOSITORY_BACKEND": "postgres",
        "EYVAN_TLS_CERT_FILE": HEAD_TLS_CERT_PATH,
        "EYVAN_TLS_KEY_FILE": HEAD_TLS_KEY_PATH,
    }


def render_head_bootstrap_script(spec: HeadSpec) -> str:
    """Render the idempotent installer the head runs under ``sudo``.

    The script carries no credentials. Everything secret is read from
    ``/etc/eyvan/head.env``, which is delivered separately over stdin, so the
    installer can be printed in a dry run and read in review.
    """

    if not spec.package.strip():
        raise ValueError("head package specification must be non-empty")
    host = spec.host_install_spec()
    venv_python = host.venv_python
    unit = host_server_unit(host)
    lines = [
        *host_prelude_lines(host),
        f"chown root:eyvan-head {HEAD_ENVIRONMENT_PATH}",
        f"chmod 0640 {HEAD_ENVIRONMENT_PATH}",
        *_head_tls_lines(),
        "# Group membership rather than a copy of the credential. Anyone who can",
        "# reach this host over IAP already has sudo and can already read this",
        "# file -- which is exactly what `head up` used to tell them to do. This",
        "# removes the sudo round trip and grants no authority they lacked. It",
        "# takes effect at their next login, which is why the SDK names the group",
        "# when the read is refused.",
        'if [ -n "${SUDO_USER:-}" ]; then usermod -aG eyvan-head "${SUDO_USER}"; fi',
        "",
        *host_venv_lines(host),
    ]
    if spec.submitter_python:
        lines.extend(
            (
                "# An interpreter matching the one this fleet's workers run.",
                "#",
                "# A serialized callable cannot load on any other version, so a head",
                "# whose Python differs from its workers' can submit nothing at all:",
                "# every rank dies on the version check *after* the accelerator has",
                "# been provisioned. Observed on a live multislice v5p-16 -- two slices",
                "# built, four ranks started, all four dead on 'callable requires",
                "# Python 3.12; current runtime is 3.10'.",
                "#",
                "# uv is used only here, and only to fetch a standalone interpreter the",
                "# base image does not carry. The head's own venv is still built with",
                "# the image's python3, so a uv that cannot be fetched costs the",
                "# submitter environment and nothing else.",
                "#",
                "# Every step is inside the `if`, because this script runs under",
                "# `set -e`: a failed interpreter download would otherwise abort the",
                "# bootstrap before PostgreSQL and the unit, turning a convenience",
                "# into a head that does not come up at all. A partial venv is",
                "# removed rather than left, because the command this prints names",
                "# that path -- and a broken interpreter there is worse than none.",
                "if ! command -v uv >/dev/null 2>&1; then",
                "  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh || true",
                "fi",
                "#",
                "# The interpreter is installed under /opt rather than root's home.",
                "# uv's default is ~/.local/share/uv/python, and this script runs as",
                "# root: the venv's `python` is a symlink into a 0700 directory, so",
                "# every operator who is not root gets `Permission denied` from the",
                "# path this head just told them to use. Found by running it.",
                "if command -v uv >/dev/null 2>&1 && \\",
                f"   rm -rf {HEAD_SUBMITTER_DIR} && \\",
                f"   export UV_PYTHON_INSTALL_DIR={HEAD_PYTHON_DIR} && \\",
                # Created rather than assumed: uv only makes this directory when
                # it downloads, and a version the image already carries -- 3.12
                # on this base image -- is used in place. The chmod below would
                # then fail on a directory that never appeared and discard a
                # perfectly good environment.
                f"   install -d -m 0755 {HEAD_PYTHON_DIR} && \\",
                f"   uv venv --python {shlex.quote(spec.submitter_python)} {HEAD_SUBMITTER_DIR} && \\",
                f"   uv pip install --python {HEAD_SUBMITTER_DIR}/bin/python --quiet {HEAD_WHEEL_DIR}/*.whl && \\",
                f"   chmod -R a+rX {HEAD_PYTHON_DIR} {HEAD_SUBMITTER_DIR}; then",
                f"  echo 'eyvan: submitter interpreter ready at {HEAD_SUBMITTER_DIR}/bin/python'",
                "else",
                f"  rm -rf {HEAD_SUBMITTER_DIR}",
                f"  echo 'eyvan: no Python {spec.submitter_python} submitter was installed; this head still serves,",
                "   but @execute run on it will refuse to submit' >&2",
                "fi",
                "",
            )
        )
    if host.local_database:
        lines.extend(host_postgres_lines(host))
    agent_unit = "\n".join(
        (
            "[Unit]",
            "Description=Eyvan cell agent on the control-plane head",
            f"After=network-online.target {HEAD_SERVICE_NAME}",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            "User=eyvan-head",
            "EnvironmentFile=/etc/eyvan/head.env",
            f"ExecStart={HEAD_VENV_PATH}/bin/eyvan-agent --host 0.0.0.0 --port {HEAD_AGENT_PORT}",
            "Restart=on-failure",
            "RestartSec=3",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
        )
    )
    lines.extend(
        (
            f"cat >/etc/systemd/system/{HEAD_SERVICE_NAME} <<'EYVAN_HEAD_UNIT'",
            unit,
            "EYVAN_HEAD_UNIT",
            f"chmod 0644 /etc/systemd/system/{HEAD_SERVICE_NAME}",
            f"cat >/etc/systemd/system/{HEAD_AGENT_SERVICE_NAME} <<'EYVAN_AGENT_UNIT'",
            agent_unit,
            "EYVAN_AGENT_UNIT",
            f"chmod 0644 /etc/systemd/system/{HEAD_AGENT_SERVICE_NAME}",
            f"cat >{PLATFORM_CONFIG_PATH} <<'EYVAN_PLATFORM'",
            "version: 1",
            *(("team: true",) if spec.team else ()),
            "agents:",
            "  default:",
            f"    endpoint: http://127.0.0.1:{HEAD_AGENT_PORT}",
            "EYVAN_PLATFORM",
            f"chmod 0644 {PLATFORM_CONFIG_PATH}",
            # What makes this node answer "which control plane am I part of"
            # without being told. Run with the venv interpreter, which holds the
            # exact wheel this head will serve, and on every `up` including an
            # adopting one -- so a changed address self-heals.
            f"{venv_python} -c {shlex.quote(_NODE_CONTEXT_SNIPPET.format(fleet=spec.fleet))}",
            "systemctl daemon-reload",
            # Restarted, not merely enabled, for the reason the server unit is:
            # this script has just replaced the wheel underneath it and may have
            # rewritten the environment it reads.
            f"systemctl enable {HEAD_AGENT_SERVICE_NAME}",
            f"systemctl restart {HEAD_AGENT_SERVICE_NAME}",
            "",
        )
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class HeadUpResult:
    """What one ``eyvan head up`` actually did, without its credentials."""

    spec: HeadSpec
    address: str
    adopted_instance: bool
    created_firewall_rules: tuple[str, ...]
    source_ranges: tuple[str, ...]
    wrote_environment: bool
    service_state: str
    #: Whether this head serves the operations UI. False is ordinary -- the
    #: dashboard is a Node build tree that cannot ship inside a Python wheel --
    #: and saying so is the point: without it the head served only the API while
    #: `eyvan dashboard` cheerfully opened a 404.
    serves_dashboard: bool = False

    @property
    def dashboard_url(self) -> str | None:
        """Return where this head serves its UI, or ``None`` when it serves none."""

        return self.url if self.serves_dashboard else None

    @property
    def url(self) -> str:
        """Return the control-plane URL reachable from inside the VPC."""

        return f"https://{self.address}:{self.spec.port}"

    @property
    def next_command(self) -> str:
        """Return the exact command an operator runs next."""

        # Both audiences, because they need different commands. From the head
        # itself nothing has to be passed: the installer recorded the plane this
        # node serves, so the CLI and the SDK discover the URL and the credential.
        # From anywhere else the address is a private VPC one, so it needs a
        # tunnel and an explicit flag.
        # Naming the submitter matters as much as naming the URL: a callable
        # pickled by the wrong interpreter is refused on the accelerator, and the
        # head's own python is not the one its workers run.
        submit = f"{HEAD_SUBMITTER_DIR}/bin/python train.py" if self.spec.submitter_python else "python train.py"
        return (
            f"on the head: eyvan up <fleet.yaml> && {submit}   |   "
            f"elsewhere: {CONTROL_PLANE_CA_FILE_ENV}=<copy of {HEAD_CONTROL_PLANE_CA_PATH}> "
            f"eyvan --url {self.url} --token <token> up <fleet.yaml>"
        )

    @property
    def dashboard_note(self) -> str | None:
        """Say where the operations UI is, because nothing else does.

        The bundle is staged and served from this head's root, so the address
        already printed *is* the dashboard -- which is precisely why nobody
        finds it: an operator reads an API base URL and does not think to open
        it. Ingress is VPC-only, so a laptop needs the tunnel named here too.
        """

        if self.dashboard_url is None:
            return None
        return (
            f"Operations UI: open {self.url} in a browser. From outside the VPC, forward it first: "
            f"gcloud compute start-iap-tunnel {self.spec.instance_name} {self.spec.port} "
            f"--local-host-port=localhost:{self.spec.port} --zone {self.spec.zone} --project {self.spec.project}"
        )

    @property
    def authentication_warning(self) -> str | None:
        """Warn when the head will 401 every request it receives.

        A head that starts and answers nothing is the failure this command must
        not hide: the server has no development auth on a network bind, so with
        no issuer configured there is no credential that authenticates.
        """

        if self.spec.oidc_issuer is not None:
            return None
        return (
            "This head has no EYVAN_OIDC_ISSUER, so it authenticates with a generated bootstrap token instead. "
            "On the head itself nothing needs configuring: the CLI and the Python SDK discover this plane's "
            "address and credential from the node. From anywhere else, read it with "
            "`sudo grep EYVAN_BOOTSTRAP_TOKEN /etc/eyvan/head.env` and pass `eyvan --token <token> --url <address>`, "
            f"with {CONTROL_PLANE_CA_FILE_ENV} naming a copy of {HEAD_CONTROL_PLANE_CA_PATH}: the head serves "
            "HTTPS under a CA it generated for itself. "
            "Register an identity provider and re-run with --oidc-issuer to retire it."
        )

    @property
    def artifact_store_warning(self) -> str | None:
        """Warn when this head cannot provision the workers it will create.

        Without an artifact store there is nowhere to publish the worker wheel,
        so no `EYVAN_NATIVE_WORKER_PACKAGE` is written -- and the GCP capacity
        driver refuses every eyvan-native request without it. The head reported
        itself active anyway, and the refusal arrived on the first submission,
        which is the point furthest from the omission that caused it.
        """

        if self.spec.artifact_store_uri is not None:
            return None
        return (
            "This head has no --artifact-store-uri, so it published no native-worker wheel and named no shared "
            "fleet-state document. Every eyvan-native capacity request will be refused with "
            "'requires EYVAN_NATIVE_WORKER_PACKAGE', on the first submission rather than now. Re-run with "
            "`eyvan head up --artifact-store-uri gs://<bucket>/<prefix>`."
        )

    @property
    def wheelhouse_warning(self) -> str | None:
        """Warn when this head's egress needs dependencies it does not stage.

        Workers on ``private-google-access`` reach Google APIs and nothing else,
        so they cannot install the wheel's dependencies from PyPI. The GCP
        driver refuses an eyvan-native TPU request without
        ``EYVAN_NATIVE_WORKER_WHEELHOUSE``, and this head publishes the wheel
        and not the wheelhouse -- a known gap, pinned by
        ``test_a_private_google_access_head_does_not_yet_stage_a_wheelhouse``.

        Until it is closed the refusal is at least moved to where it can be
        acted on. It used to arrive on the first TPU request, which is the point
        furthest from the choice that caused it and the one where an operator is
        least able to relate the two.
        """

        if self.spec.egress != "private-google-access":
            return None
        return (
            "This head selected --egress private-google-access, whose workers cannot reach PyPI. It publishes the "
            "native-worker wheel but not a wheelhouse of its dependencies, so every eyvan-native TPU request will be "
            "refused with 'requires EYVAN_NATIVE_WORKER_WHEELHOUSE' -- on the first request rather than now. Stage one "
            "and set EYVAN_NATIVE_WORKER_WHEELHOUSE in /etc/eyvan/head.env, or use --egress cloud-nat."
        )

    @property
    def callable_trust_warning(self) -> str | None:
        """Warn when this head will refuse every ``@execute`` submission.

        Trust is named, never inferred: an identity provider says who a principal
        is, not whether its project may hand this fleet a pickle to deserialize.
        So a head stood up with ``--oidc-issuer`` trusts nobody until an operator
        says who -- and the loss is otherwise silent, because the submission
        authenticates and is only then refused, with a 403 naming the workload
        rather than the setting that refused it.
        """

        if self.spec.trusted_callable_projects or self.spec.oidc_issuer is None:
            return None
        return (
            "This head trusts no project for callable workloads, so every @execute submission is refused with "
            "403 after authenticating; command workloads are unaffected. Re-run with "
            "`eyvan head up --trust-callable-project <org>/<project>` to grant it."
        )

    def as_document(self) -> JsonObject:
        """Return automation-safe output that contains no generated secret."""

        return {
            "address": self.address,
            "adopted_instance": self.adopted_instance,
            "authentication_warning": self.authentication_warning,
            "artifact_store_warning": self.artifact_store_warning,
            "wheelhouse_warning": self.wheelhouse_warning,
            "submitter_python_path": (f"{HEAD_SUBMITTER_DIR}/bin/python" if self.spec.submitter_python else None),
            "callable_trust_warning": self.callable_trust_warning,
            "created_firewall_rules": list(self.created_firewall_rules),
            "dashboard_url": self.dashboard_url,
            # A path, never a value: this document is the automation-safe one.
            "node_context_path": str(NODE_CONTEXT_PATH),
            "instance": self.spec.instance_name,
            "next_command": self.next_command,
            "service_state": self.service_state,
            "source_ranges": list(self.source_ranges),
            "spec": self.spec.as_document(),
            "url": self.url,
            "wrote_environment": self.wrote_environment,
        }


@dataclass(frozen=True, slots=True)
class HeadStatusResult:
    """A head's observed instance, ingress, and service state."""

    spec: HeadSpec
    exists: bool
    instance_status: str | None
    address: str | None
    source_ranges: tuple[str, ...]
    service_state: str | None

    def as_document(self) -> JsonObject:
        """Return the observed head state as one deterministic document."""

        return {
            "address": self.address,
            "exists": self.exists,
            "instance": self.spec.instance_name,
            "instance_status": self.instance_status,
            "port": self.spec.port,
            "project": self.spec.project,
            "service_state": self.service_state,
            "source_ranges": list(self.source_ranges),
            "url": None if self.address is None else f"https://{self.address}:{self.spec.port}",
            "zone": self.spec.zone,
        }


@dataclass(frozen=True, slots=True)
class HeadDownResult:
    """What one confirmed teardown removed."""

    spec: HeadSpec
    deleted_instance: bool
    deleted_firewall_rules: tuple[str, ...]

    def as_document(self) -> JsonObject:
        """Return automation-safe teardown output."""

        return {
            "deleted_firewall_rules": list(self.deleted_firewall_rules),
            "deleted_instance": self.deleted_instance,
            "instance": self.spec.instance_name,
            "project": self.spec.project,
            "zone": self.spec.zone,
        }


@dataclass(frozen=True, slots=True)
class _ManagedFirewall:
    """One firewall rule this head owns, in the shape it is reconciled from."""

    name: str
    port: int
    source_ranges: tuple[str, ...]
    purpose: str


class HeadProvisioner:
    """Provision, inspect, and tear down one Eyvan control-plane head.

    Every provider interaction goes through the injected runner, and every
    mutation is preceded by a read that establishes ownership. The class does
    not print, does not prompt, and never returns a credential.
    """

    def __init__(
        self,
        spec: HeadSpec,
        *,
        runner: HeadCommandRunner | None = None,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        secret_factory: tp.Callable[[], HeadSecrets] | None = None,
        ssh_ready_attempts: int = DEFAULT_SSH_READY_ATTEMPTS,
        ssh_ready_delay_s: float = DEFAULT_SSH_READY_DELAY_S,
        sleep: tp.Callable[[float], None] | None = None,
        wheel_source: tp.Callable[[], _LocalWheel] | None = None,
        dashboard_source: tp.Callable[[], bytes | None] | None = None,
    ):
        if isinstance(command_timeout_s, bool) or not isinstance(command_timeout_s, (int, float)):
            raise ValueError("command_timeout_s must be a positive number")
        if command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be a positive number")
        if isinstance(ssh_ready_attempts, bool) or not isinstance(ssh_ready_attempts, int) or ssh_ready_attempts < 1:
            raise ValueError("ssh_ready_attempts must be a positive integer")
        if isinstance(ssh_ready_delay_s, bool) or not isinstance(ssh_ready_delay_s, (int, float)):
            raise ValueError("ssh_ready_delay_s must be a non-negative number")
        if ssh_ready_delay_s < 0:
            raise ValueError("ssh_ready_delay_s must be a non-negative number")
        self.ssh_ready_attempts = ssh_ready_attempts
        self.ssh_ready_delay_s = float(ssh_ready_delay_s)
        self._sleep = sleep if sleep is not None else time.sleep
        self._wheel_source = wheel_source
        self._dashboard_source = dashboard_source
        self.spec = spec
        # Resolved here rather than as a default argument, and the difference is
        # not cosmetic: a default binds the module function at import, so a test
        # that replaces ``_run_gcloud`` keeps talking to the real project. That
        # mistake creates a VM.
        self._runner = runner if runner is not None else _run_gcloud
        self._secret_factory = secret_factory if secret_factory is not None else HeadSecrets.generate
        self.command_timeout_s = float(command_timeout_s)

    # -- planning ---------------------------------------------------------

    def plan(self) -> HeadPlan:
        """Return every action ``up`` would take, without running any of them."""

        spec = self.spec
        ranges = spec.source_ranges or None
        actions = [
            self._instance_lookup_action(),
            self._instance_create_action(),
            self._address_action(),
        ]
        if ranges is None:
            actions.append(self._subnet_lookup_action())
        for rule in self._firewall_rules(ranges):
            actions.append(self._firewall_lookup_action(rule))
            actions.append(self._firewall_create_action(rule))
        actions.extend(
            (
                self._environment_lookup_action(),
                *(
                    self._worker_package_actions(_planned_worker_wheel())
                    if self.spec.artifact_store_uri is not None
                    else ()
                ),
                *(
                    [self._stage_dashboard_action()]
                    if (self._dashboard_source or _local_dashboard_bundle)() is not None
                    else []
                ),
                self._configure_action(),
                *([self._stage_wheel_action()] if self.spec.stages_wheel else []),
                self._install_action(),
                self._bootstrap_action(),
                self._migrate_action(),
                self._service_start_action(),
                self._service_settled_action(),
            )
        )
        return HeadPlan(spec=spec, actions=tuple(actions))

    # -- provisioning -----------------------------------------------------

    def up(self, *, rotate_secrets: bool = False) -> HeadUpResult:
        """Create or adopt one head and leave its control plane running."""

        spec = self.spec
        existing = self._describe_instance()
        adopted = existing is not None
        if existing is None:
            self._execute(self._instance_create_action())
        address = self._read_address()
        ranges = spec.source_ranges or self._discover_source_ranges()
        created: list[str] = []
        for rule in self._firewall_rules(ranges):
            if self._ensure_firewall(rule):
                created.append(rule.name)
        write_environment = rotate_secrets or not self._environment_present()
        # Built once and used twice: the head installs this wheel on itself, and
        # publishes the same one for workers that have no ssh path from here. Two
        # different distributions serving one fleet is a class of failure nobody
        # would think to look for.
        wheel = (self._wheel_source or _local_wheel_bytes)() if (spec.stages_wheel or write_environment) else None
        # Published on every run, not only when the environment is rewritten. An
        # adopting re-run upgrades the head's own wheel, so leaving the worker
        # package alone made the two diverge silently: the control plane ran new
        # code while every worker kept booting the build named by the environment
        # written when the head was created. The object is content-addressed, so
        # republishing an unchanged wheel converges on the object already there.
        worker_package = self._stage_worker_package(wheel) if wheel is not None else None
        dashboard = (self._dashboard_source or _local_dashboard_bundle)()
        if dashboard is not None:
            # Before the environment that names the directory, for the same
            # reason the worker wheel is staged first: the file must not describe
            # something that is not there yet.
            self._execute(
                self._stage_dashboard_action(),
                stdin_text=base64.b64encode(dashboard).decode("ascii"),
            )
        if write_environment:
            head_secrets = self._secret_factory()
            self._execute(
                self._configure_action(),
                # Staged before this, because the environment names the object and
                # its generation: writing the file first would mean a head whose
                # configuration described a wheel that did not exist yet.
                stdin_text=head_secrets.environment_file(
                    spec,
                    address=address,
                    worker_package=worker_package,
                    serves_dashboard=dashboard is not None,
                ),
            )
        if worker_package is not None and not write_environment:
            self._execute(self._refresh_worker_package_action(worker_package))
        if spec.stages_wheel:
            assert wheel is not None
            observed = self._execute(
                self._stage_wheel_action(),
                stdin_text=base64.b64encode(wheel.data).decode("ascii"),
            ).stdout.strip()
            self._verify_staged_wheel(sent=wheel.digest, observed=observed)
        self._execute(self._install_action(), stdin_text=render_head_bootstrap_script(spec))
        self._execute(self._bootstrap_action())
        self._execute(self._migrate_action())
        self._execute(self._service_start_action())
        service_state = self._verified_service_state()
        return HeadUpResult(
            spec=spec,
            address=address,
            serves_dashboard=dashboard is not None,
            adopted_instance=adopted,
            created_firewall_rules=tuple(created),
            source_ranges=tuple(ranges),
            wrote_environment=write_environment,
            service_state=service_state,
        )

    def status(self) -> HeadStatusResult:
        """Report the head's instance, ingress, and service state."""

        spec = self.spec
        instance = self._describe_instance()
        if instance is None:
            return HeadStatusResult(
                spec=spec,
                exists=False,
                instance_status=None,
                address=None,
                source_ranges=(),
                service_state=None,
            )
        status = instance.get("status")
        address = _internal_address(instance)
        rule = self._describe_firewall(spec.firewall_rule_name)
        ranges = _string_tuple(rule.get("sourceRanges")) if rule is not None else ()
        service_state: str | None = None
        if status == "RUNNING":
            result = self._run(self._service_verify_action())
            service_state = result.stdout.strip() or "unknown"
        return HeadStatusResult(
            spec=spec,
            exists=True,
            instance_status=status if isinstance(status, str) else None,
            address=address,
            source_ranges=ranges,
            service_state=service_state,
        )

    def rollback(self, *, confirmed: bool) -> str:
        """Reinstall the wheel this head ran before its current one.

        Verified by build digest rather than by unit state: ``systemctl`` calls
        a unit active as soon as the process starts, which says nothing about
        which build it is running, and the point of a rollback is precisely that
        the running build is the wrong one.

        Both units are restarted. ``eyvan-head-agent.service`` runs the same
        wheel as ``eyvan-head.service``, and a session was lost this month to
        restarting only one of them -- leaving a head whose API was rolled back
        while its cell agent was not.
        """

        if not confirmed:
            raise HeadError(
                HeadStep.INSTALL,
                "rolling back replaces the running control-plane build; pass --yes to confirm",
            )
        instance = self._describe_instance()
        if instance is None:
            raise HeadError(HeadStep.INSTALL, f"no head named {self.spec.instance_name} to roll back")
        previous = self._run(self._previous_wheel_action()).stdout.strip()
        if not previous:
            raise HeadError(
                HeadStep.INSTALL,
                f"no superseded wheel is retained in {HEAD_PREVIOUS_WHEEL_DIR}; a head keeps one only after it "
                "has been upgraded at least once since retention was added",
            )
        before = self._run(self._build_identity_action()).stdout.strip()
        self._execute(self._rollback_install_action(previous))
        self._execute(self._service_restart_action())
        after = self._run(self._build_identity_action()).stdout.strip()
        if after == before:
            raise HeadError(
                HeadStep.SERVICE_VERIFY,
                f"the head still reports build {after or 'unknown'} after reinstalling {previous}; the rollback "
                "did not take, so the units are running the build you tried to leave",
            )
        return after

    def down(self, *, confirmed: bool) -> HeadDownResult:
        """Delete one head Eyvan owns, and only after explicit confirmation."""

        if not confirmed:
            raise HeadError(
                HeadStep.INSTANCE_DELETE,
                "tearing down a control plane destroys its durable state; pass --yes to confirm",
            )
        spec = self.spec
        instance = self._describe_instance()
        deleted_instance = False
        if instance is not None:
            self._execute(self._instance_delete_action())
            deleted_instance = True
        deleted_rules: list[str] = []
        for rule in self._firewall_rules(None):
            if self._describe_firewall(rule.name) is not None:
                self._execute(self._firewall_delete_action(rule))
                deleted_rules.append(rule.name)
        return HeadDownResult(
            spec=spec,
            deleted_instance=deleted_instance,
            deleted_firewall_rules=tuple(deleted_rules),
        )

    # -- actions ----------------------------------------------------------

    def _instance_lookup_action(self) -> HeadAction:
        """Build the read that decides between adoption, creation, and refusal."""

        spec = self.spec
        return HeadAction(
            step=HeadStep.INSTANCE_LOOKUP,
            summary=f"look up {spec.instance_name} and require Eyvan's ownership label before adopting it",
            args=(
                "compute",
                "instances",
                "describe",
                spec.instance_name,
                "--project",
                spec.project,
                "--zone",
                spec.zone,
                "--format=json",
            ),
        )

    def _instance_create_action(self) -> HeadAction:
        """Build the labelled head-VM creation command."""

        spec = self.spec
        args = [
            "compute",
            "instances",
            "create",
            spec.instance_name,
            "--project",
            spec.project,
            "--zone",
            spec.zone,
            "--machine-type",
            spec.machine_type,
            "--image-family",
            spec.image_family,
            "--image-project",
            spec.image_project,
            "--boot-disk-size",
            f"{spec.boot_disk_size_gb}GB",
            "--boot-disk-type",
            "pd-balanced",
            "--network",
            spec.network,
            "--labels",
            ",".join(f"{key}={value}" for key, value in sorted(spec.labels.items())),
            "--tags",
            spec.network_tag,
            "--scopes",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        if spec.subnetwork is not None:
            args.extend(("--subnet", spec.subnetwork))
        if spec.service_account is not None:
            args.extend(("--service-account", spec.service_account))
        if not spec.has_external_address:
            args.append("--no-address")
        if spec.spot:
            args.extend(("--provisioning-model=SPOT", "--instance-termination-action=STOP"))
        args.append("--quiet")
        return HeadAction(
            step=HeadStep.INSTANCE_CREATE,
            summary=(
                f"create {spec.machine_type} head {spec.instance_name} labelled "
                f"{HEAD_MANAGED_LABEL_KEY}={HEAD_MANAGED_LABEL_VALUE}"
            ),
            args=tuple(args),
            mutates=True,
            skipped_when_present=True,
        )

    def _address_action(self) -> HeadAction:
        """Build the read that returns the head's internal IPv4 address."""

        spec = self.spec
        return HeadAction(
            step=HeadStep.INSTANCE_ADDRESS,
            summary=f"read {spec.instance_name}'s internal VPC address",
            args=(
                "compute",
                "instances",
                "describe",
                spec.instance_name,
                "--project",
                spec.project,
                "--zone",
                spec.zone,
                "--format=value(networkInterfaces[0].networkIP)",
            ),
        )

    def _subnet_lookup_action(self) -> HeadAction:
        """Build the read that yields the VPC ranges ingress is restricted to."""

        spec = self.spec
        return HeadAction(
            step=HeadStep.SUBNET_LOOKUP,
            summary=(
                f"list internal subnet ranges of network {spec.network} in {spec.region}; "
                "a range that is not RFC 1918 is refused rather than opened"
            ),
            args=(
                "compute",
                "networks",
                "subnets",
                "list",
                "--project",
                spec.project,
                f"--network={spec.network}",
                f"--regions={spec.region}",
                "--format=value(ipCidrRange)",
            ),
        )

    def _firewall_rules(self, ranges: tp.Sequence[str] | None) -> tuple[_ManagedFirewall, ...]:
        """Return every firewall rule this head manages, in reconcile order."""

        spec = self.spec
        rules = [
            _ManagedFirewall(
                name=spec.firewall_rule_name,
                port=spec.port,
                source_ranges=tuple(ranges) if ranges else (UNRESOLVED_SOURCE_RANGES,),
                purpose="control-plane API",
            )
        ]
        if spec.iap_ssh:
            rules.append(
                _ManagedFirewall(
                    name=spec.ssh_firewall_rule_name,
                    port=22,
                    source_ranges=(IAP_TCP_FORWARDING_RANGE,),
                    purpose="IAP TCP forwarding for SSH",
                )
            )
        return tuple(rules)

    def _firewall_lookup_action(self, rule: _ManagedFirewall) -> HeadAction:
        """Build the read that establishes ownership of one managed rule."""

        return HeadAction(
            step=HeadStep.FIREWALL_LOOKUP,
            summary=f"look up firewall rule {rule.name} and require Eyvan's ownership description",
            args=(
                "compute",
                "firewall-rules",
                "describe",
                rule.name,
                "--project",
                self.spec.project,
                "--format=json",
            ),
        )

    def _firewall_create_action(self, rule: _ManagedFirewall) -> HeadAction:
        """Build one narrow ingress rule; the source range is never the internet."""

        spec = self.spec
        return HeadAction(
            step=HeadStep.FIREWALL_CREATE,
            summary=f"open tcp:{rule.port} to tag {spec.network_tag} for {rule.purpose}",
            args=(
                "compute",
                "firewall-rules",
                "create",
                rule.name,
                "--project",
                spec.project,
                "--network",
                spec.network,
                "--direction=INGRESS",
                "--priority=1000",
                f"--allow=tcp:{rule.port}",
                f"--source-ranges={','.join(rule.source_ranges)}",
                f"--target-tags={spec.network_tag}",
                f"--description={spec.firewall_description}",
                "--quiet",
            ),
            mutates=True,
            skipped_when_present=True,
        )

    def _firewall_delete_action(self, rule: _ManagedFirewall) -> HeadAction:
        """Build the deletion of one Eyvan-owned firewall rule."""

        return HeadAction(
            step=HeadStep.FIREWALL_DELETE,
            summary=f"delete Eyvan-owned firewall rule {rule.name}",
            args=(
                "compute",
                "firewall-rules",
                "delete",
                rule.name,
                "--project",
                self.spec.project,
                "--quiet",
            ),
            mutates=True,
        )

    def _instance_delete_action(self) -> HeadAction:
        """Build the deletion of the head VM itself."""

        spec = self.spec
        return HeadAction(
            step=HeadStep.INSTANCE_DELETE,
            summary=f"delete head {spec.instance_name} and its boot disk",
            args=(
                "compute",
                "instances",
                "delete",
                spec.instance_name,
                "--project",
                spec.project,
                "--zone",
                spec.zone,
                "--quiet",
            ),
            mutates=True,
        )

    def _ssh_action(
        self,
        step: HeadStep,
        summary: str,
        remote_command: str,
        *,
        mutates: bool = False,
        sends_stdin: bool = False,
        skipped_when_present: bool = False,
    ) -> HeadAction:
        """Build one bounded remote command over the selected SSH transport."""

        spec = self.spec
        args = [
            "compute",
            "ssh",
            spec.instance_name,
            "--project",
            spec.project,
            "--zone",
            spec.zone,
        ]
        args.append("--tunnel-through-iap" if spec.iap_ssh else "--internal-ip")
        args.extend(("--quiet", "--command", remote_command))
        return HeadAction(
            step=step,
            summary=summary,
            args=tuple(args),
            mutates=mutates,
            sends_stdin=sends_stdin,
            skipped_when_present=skipped_when_present,
        )

    def _environment_lookup_action(self) -> HeadAction:
        """Build the probe that distinguishes a configured head from a fresh one."""

        return self._ssh_action(
            HeadStep.SECRET_LOOKUP,
            "check whether this head already holds generated credentials",
            "sudo /bin/sh -ec "
            + shlex.quote(
                f"if [ -f {HEAD_ENVIRONMENT_PATH} ]; "
                f"then echo {_ENVIRONMENT_PRESENT}; else echo {_ENVIRONMENT_ABSENT}; fi"
            ),
        )

    def _configure_action(self) -> HeadAction:
        """Build the credential delivery; the values travel on stdin only."""

        return self._ssh_action(
            HeadStep.CONFIGURE,
            (
                "generate EYVAN_METRICS_TOKEN, EYVAN_ARTIFACT_TOKEN_KEY and EYVAN_AGENT_SHARED_SECRET and "
                f"install them as {HEAD_ENVIRONMENT_PATH} (mode 0600, delivered on stdin)"
            ),
            f"sudo install -D -m 0600 -o root -g root /dev/stdin {HEAD_ENVIRONMENT_PATH}",
            mutates=True,
            sends_stdin=True,
            skipped_when_present=True,
        )

    def _worker_package_actions(self, wheel: _LocalWheel) -> tuple[HeadAction, HeadAction]:
        """Build the upload of the worker wheel and the read-back of its generation.

        A native worker is not the head: it bootstraps on a TPU or Compute VM that
        has no ssh path from here, so it fetches its Eyvan from object storage
        rather than being handed it. The head therefore has to publish the same
        distribution it installs on itself, and record the two facts the worker
        verifies it by.

        Content-addressed and never overwritten, so a re-run with an unchanged
        wheel converges on the object that already exists, and a changed wheel
        cannot replace bytes an existing worker was told to expect.
        """

        spec = self.spec
        assert spec.artifact_store_uri is not None
        uri = worker_package_uri(spec.artifact_store_uri, wheel.digest, wheel.filename)
        upload = HeadAction(
            step=HeadStep.STAGE_WORKER_PACKAGE,
            summary=f"publish the native-worker wheel to {uri}",
            # A path rather than stdin: the runner boundary carries text, and a
            # wheel is bytes. The file is written under an owner-only temporary
            # directory and removed in a `finally`, so a failed upload leaves
            # nothing behind.
            args=("storage", "cp", _WORKER_WHEEL_PLACEHOLDER, uri, "--project", spec.project, "--quiet"),
            mutates=True,
        )
        generation = HeadAction(
            step=HeadStep.WORKER_PACKAGE_GENERATION,
            summary="read the staged object's generation, which pins the URI to these exact bytes",
            args=(
                "storage",
                "objects",
                "describe",
                uri,
                "--project",
                spec.project,
                "--format=value(generation)",
                "--quiet",
            ),
        )
        return upload, generation

    def _stage_worker_package(self, wheel: _LocalWheel) -> WorkerPackage | None:
        """Publish the worker wheel and return what a worker needs to verify it.

        ``None`` when this head has no artifact store: there is nowhere to publish
        to, and inventing a location would produce a head that looks configured
        and fails on its first submission.
        """

        if self.spec.artifact_store_uri is None:
            return None
        upload, generation = self._worker_package_actions(wheel)
        with tempfile.TemporaryDirectory(prefix="eyvan-worker-wheel-") as directory:
            source = Path(directory) / wheel.filename
            source.write_bytes(wheel.data)
            resolved = replace(
                upload,
                args=tuple(str(source) if part is _WORKER_WHEEL_PLACEHOLDER else part for part in upload.args),
            )
            self._execute(resolved)
        result = self._execute(generation)
        staged = result.stdout.strip()
        if not staged.isdigit():
            raise HeadCommandError(
                HeadStep.WORKER_PACKAGE_GENERATION,
                f"GCS reported {staged!r} as the generation of the staged worker wheel; a positive integer is "
                "required, because it is what pins the package URI to the exact bytes a worker will verify",
            )
        return WorkerPackage(uri=generation.args[3], sha256=wheel.digest, generation=staged)

    def _refresh_worker_package_action(self, package: WorkerPackage) -> HeadAction:
        """Point an existing head's environment at the worker wheel just published.

        Three lines rather than the whole file, because the file holds generated
        credentials this command no longer has: an adopting run never learns the
        secrets it wrote the first time, so rewriting it wholesale would either
        rotate them or lose them.
        """

        assignments = "\n".join(
            (
                f"EYVAN_NATIVE_WORKER_PACKAGE={package.uri}",
                f"EYVAN_NATIVE_WORKER_PACKAGE_SHA256={package.sha256}",
                f"EYVAN_NATIVE_WORKER_PACKAGE_GENERATION={package.generation}",
            )
        )
        return self._ssh_action(
            HeadStep.STAGE_WORKER_PACKAGE,
            "point this head's environment at the worker wheel just published",
            "sudo /bin/sh -ec "
            + shlex.quote(
                f"sed -i '/^EYVAN_NATIVE_WORKER_PACKAGE/d' {HEAD_ENVIRONMENT_PATH}; "
                f"printf '%s\\n' {shlex.quote(assignments)} >> {HEAD_ENVIRONMENT_PATH}"
            ),
            mutates=True,
        )

    def _stage_dashboard_action(self) -> HeadAction:
        """Build the delivery of the operations UI this head will serve.

        Unpacked into a directory the server mounts at its own root, so the
        dashboard is reachable at exactly the URL the control plane already
        answers on -- no second port, no second firewall rule, and no second
        address for an operator to hold.
        """

        return self._ssh_action(
            HeadStep.STAGE_DASHBOARD,
            f"stage the operations dashboard into {HEAD_DASHBOARD_DIR}",
            "sudo /bin/sh -ec "
            + shlex.quote(
                f"rm -rf {HEAD_DASHBOARD_DIR}; install -d -m 0755 {HEAD_DASHBOARD_DIR}; "
                f"base64 -d | tar -xzf - -C {HEAD_DASHBOARD_DIR}; "
                f"chmod -R a+rX {HEAD_DASHBOARD_DIR}; test -f {HEAD_DASHBOARD_DIR}/index.html && echo staged"
            ),
            mutates=True,
            sends_stdin=True,
        )

    def _stage_wheel_action(self, filename: str | None = None) -> HeadAction:
        filename = filename or staged_wheel_filename()
        """Build the delivery of this installation's own wheel.

        The default package used to be the name ``eyvan``, which pip resolves
        from an index. Eyvan is not published to one, so a real head bootstrap
        died with "No matching distribution found for eyvan" after every other
        step had already succeeded. The distribution running this command is the
        one the head should run, so it is sent rather than looked up.

        Base64 because the stdin boundary carries text, and decoding on the host
        keeps the wheel out of argv where ``ps`` would show it.
        """

        return self._ssh_action(
            HeadStep.STAGE_WHEEL,
            f"stage this Eyvan distribution into {HEAD_WHEEL_DIR}",
            "sudo /bin/sh -ec "
            + shlex.quote(
                # Retire the running wheel instead of deleting it. `rm -rf` left
                # a head holding only the build it was running, so a bad deploy
                # could only be undone by rebuilding its predecessor from
                # source. An unmatched glob is the literal string under /bin/sh,
                # which `[ -e ]` then rejects, so the loop is a no-op on a first
                # install.
                f"umask 077; install -d -m 0755 {HEAD_WHEEL_DIR} {HEAD_PREVIOUS_WHEEL_DIR}; "
                f'for w in {HEAD_WHEEL_DIR}/*.whl; do [ -e "$w" ] && mv -f "$w" {HEAD_PREVIOUS_WHEEL_DIR}/; done; '
                f"ls -1t {HEAD_PREVIOUS_WHEEL_DIR}/*.whl 2>/dev/null "
                f"| tail -n +{HEAD_WHEEL_RETENTION + 1} | xargs rm -f; "
                f"base64 -d > {HEAD_WHEEL_DIR}/{filename}; sha256sum {HEAD_WHEEL_DIR}/{filename} | cut -d' ' -f1"
            ),
            mutates=True,
            sends_stdin=True,
        )

    def _verify_staged_wheel(self, *, sent: str, observed: str) -> None:
        """Refuse a staged wheel whose bytes are not the bytes that were sent.

        Installing an altered or truncated wheel is worse than failing to
        install one: the head would come up running something nobody built.
        """

        if observed != sent:
            raise HeadCommandError(
                HeadStep.STAGE_WHEEL,
                f"staged wheel digest {observed or '(none)'} does not match the {sent} that was sent; "
                "the upload was truncated or altered and was not installed",
            )

    def _install_action(self) -> HeadAction:
        """Build the delivery of the idempotent installer script."""

        return self._ssh_action(
            HeadStep.INSTALL,
            f"install the Eyvan head bootstrap script as {HEAD_BOOTSTRAP_PATH}",
            f"sudo install -D -m 0700 -o root -g root /dev/stdin {HEAD_BOOTSTRAP_PATH}",
            mutates=True,
            sends_stdin=True,
        )

    def _bootstrap_action(self) -> HeadAction:
        """Build the installer run that creates the venv, database, and unit."""

        return self._ssh_action(
            HeadStep.BOOTSTRAP,
            (
                f"run the installer: virtual environment, {self.spec.installed_description}, "
                "PostgreSQL, and the systemd unit"
            ),
            f"sudo {HEAD_BOOTSTRAP_PATH}",
            mutates=True,
        )

    def _migrate_action(self) -> HeadAction:
        """Build the schema migration run against the head's own environment."""

        return self._ssh_action(
            HeadStep.MIGRATE,
            "run eyvan-migrate with the head's own environment",
            "sudo /bin/sh -ec "
            + shlex.quote(f"set -a; . {HEAD_ENVIRONMENT_PATH}; set +a; exec {HEAD_VENV_PATH}/bin/eyvan-migrate"),
            mutates=True,
        )

    def _service_start_action(self) -> HeadAction:
        """Build the enable-and-restart that survives ssh exit, reboot, and a re-run.

        ``enable --now`` alone is a no-op against an already-active unit. After
        replacing a wheel or rewriting the environment, restart the server so it
        uses the installed code and current credentials rather than its startup
        state.

        ``restart`` is what makes an adopting run mean anything, and it is
        idempotent on a unit that was not running.
        """

        return self._ssh_action(
            HeadStep.SERVICE_START,
            f"enable and restart {HEAD_SERVICE_NAME} so it runs the installed wheel and the current environment",
            f"sudo systemctl enable {HEAD_SERVICE_NAME} && sudo systemctl restart {HEAD_SERVICE_NAME}",
            mutates=True,
        )

    def _service_verify_action(self) -> HeadAction:
        """Build the check that the unit is actually active."""

        return self._ssh_action(
            HeadStep.SERVICE_VERIFY,
            f"report {HEAD_SERVICE_NAME}'s state",
            f"sudo systemctl is-active {HEAD_SERVICE_NAME}",
        )

    def _service_settled_action(self) -> HeadAction:
        """Build the check that both units are active *and stay up*.

        ``systemctl is-active`` answers "active" in the seconds between two
        crashes of a ``Restart=always`` unit. Read the restart counters and wait
        out several restart intervals so a crash-looping service cannot pass.
        A unit that restarted in between, or is
        not active at the end, fails with its state and how often it restarted.
        """

        script = (
            f"plane_before=$(systemctl show -p NRestarts --value {HEAD_SERVICE_NAME}); "
            f"agent_before=$(systemctl show -p NRestarts --value {HEAD_AGENT_SERVICE_NAME}); "
            f"sleep {HEAD_SERVICE_SETTLE_S}; "
            f"plane_after=$(systemctl show -p NRestarts --value {HEAD_SERVICE_NAME}); "
            f"agent_after=$(systemctl show -p NRestarts --value {HEAD_AGENT_SERVICE_NAME}); "
            f"plane_state=$(systemctl is-active {HEAD_SERVICE_NAME} || true); "
            f"agent_state=$(systemctl is-active {HEAD_AGENT_SERVICE_NAME} || true); "
            'if [ "$plane_state" = active ] && [ "$agent_state" = active ] '
            '&& [ "$plane_before" = "$plane_after" ] && [ "$agent_before" = "$agent_after" ]; '
            "then echo active; else "
            f'echo "{HEAD_SERVICE_NAME}: ${{plane_state:-unknown}}, restart counts $plane_before -> $plane_after; '
            f'{HEAD_AGENT_SERVICE_NAME}: ${{agent_state:-unknown}}, restart counts $agent_before -> $agent_after"; '
            "exit 3; fi"
        )
        return self._ssh_action(
            HeadStep.SERVICE_VERIFY,
            f"confirm {HEAD_SERVICE_NAME} and {HEAD_AGENT_SERVICE_NAME} stay active for {HEAD_SERVICE_SETTLE_S}s",
            "sudo /bin/sh -ec " + shlex.quote(script),
        )

    def _previous_wheel_action(self) -> HeadAction:
        """Build the lookup of the most recently superseded wheel."""

        return self._ssh_action(
            HeadStep.INSTALL,
            f"find the newest superseded wheel in {HEAD_PREVIOUS_WHEEL_DIR}",
            "sudo /bin/sh -ec " + shlex.quote(f"ls -1t {HEAD_PREVIOUS_WHEEL_DIR}/*.whl 2>/dev/null | head -1"),
        )

    def _build_identity_action(self) -> HeadAction:
        """Build the read of the source digest the head is actually running.

        ``systemctl is-active`` reports that *a* process started, which is not
        the question a rollback asks. The digest is the only answer that
        distinguishes two builds of the same version.
        """

        return self._ssh_action(
            HeadStep.SERVICE_VERIFY,
            "report the head's running build digest",
            f"sudo {HEAD_VENV_PATH}/bin/python -c "
            + shlex.quote(
                "from eyvan._internal.build_identity import build_identity; print(build_identity().source_digest or '')"
            ),
        )

    def _rollback_install_action(self, wheel_path: str) -> HeadAction:
        """Build the reinstall of one retained wheel over the running venv."""

        return self._ssh_action(
            HeadStep.INSTALL,
            f"reinstall {wheel_path}",
            "sudo /bin/sh -ec "
            + shlex.quote(
                f"{HEAD_VENV_PATH}/bin/python -m pip install --disable-pip-version-check --force-reinstall "
                f"--no-deps {shlex.quote(wheel_path)}"
            ),
        )

    def _service_restart_action(self) -> HeadAction:
        """Build the restart of both units a head runs from one wheel.

        The agent runs the same distribution as the API. Restarting only the API
        leaves a head whose control plane was rolled back and whose cell agent
        was not, which is the skew this command exists to end.
        """

        return self._ssh_action(
            HeadStep.SERVICE_START,
            f"restart {HEAD_SERVICE_NAME} and {HEAD_AGENT_SERVICE_NAME}",
            f"sudo systemctl restart {HEAD_SERVICE_NAME} {HEAD_AGENT_SERVICE_NAME}",
        )

    # -- provider boundary ------------------------------------------------

    def _run(self, action: HeadAction, *, stdin_text: str | None = None) -> subprocess.CompletedProcess[str]:
        """Invoke the injected runner and translate process failures only."""

        try:
            return self._runner(action.args, stdin_text=stdin_text, timeout=self.command_timeout_s)
        except FileNotFoundError as exc:
            raise HeadCommandError(action.step, "gcloud is not installed or is not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise HeadCommandError(action.step, f"gcloud exceeded {self.command_timeout_s:g}s") from exc
        except OSError as exc:
            raise HeadCommandError(action.step, f"could not start gcloud: {exc}") from exc

    def _execute(self, action: HeadAction, *, stdin_text: str | None = None) -> subprocess.CompletedProcess[str]:
        """Run one action that must succeed, and stop the sequence when it does not."""

        result = self._run(action, stdin_text=stdin_text)
        if result.returncode != 0:
            raise HeadCommandError(action.step, _diagnostic(action, result))
        return result

    def _describe_instance(self) -> JsonObject | None:
        """Return the head VM only when Eyvan owns it, else refuse or report absence."""

        action = self._instance_lookup_action()
        result = self._run(action)
        if result.returncode != 0:
            if _is_missing(result):
                return None
            raise HeadCommandError(action.step, _diagnostic(action, result))
        document = _decode(action.step, result.stdout, resource=self.spec.instance_name)
        labels = document.get("labels")
        owned = (
            is_json_object(labels)
            and labels.get(HEAD_MANAGED_LABEL_KEY) == HEAD_MANAGED_LABEL_VALUE
            and labels.get(HEAD_FLEET_LABEL_KEY) == self.spec.fleet
        )
        if not owned:
            raise HeadOwnershipError(
                action.step,
                f"instance {self.spec.instance_name} exists in {self.spec.project}/{self.spec.zone} without "
                f"{HEAD_MANAGED_LABEL_KEY}={HEAD_MANAGED_LABEL_VALUE} and "
                f"{HEAD_FLEET_LABEL_KEY}={self.spec.fleet}; Eyvan did not create it and will not reconfigure it",
            )
        return document

    def _read_address(self) -> str:
        """Return the head's internal IPv4 address, refusing anything else."""

        action = self._address_action()
        result = self._execute(action)
        value = result.stdout.strip()
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise HeadCommandError(action.step, f"gcloud returned {value!r} instead of an internal address") from exc
        if not isinstance(address, ipaddress.IPv4Address) or not address.is_private:
            raise HeadCommandError(action.step, f"head address {value!r} is not an internal VPC IPv4 address")
        return str(address)

    def _discover_source_ranges(self) -> tuple[str, ...]:
        """Return the VPC's internal ranges, refusing to open anything wider.

        Refusal rather than a narrower guess: a subnet range Eyvan cannot prove
        is internal is one it cannot prove is *not* the internet, and the whole
        point of this rule is that the control-plane port never faces it.
        """

        action = self._subnet_lookup_action()
        result = self._execute(action)
        candidates = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if not candidates:
            raise HeadIngressError(
                action.step,
                f"network {self.spec.network} has no subnet in {self.spec.region}, so Eyvan cannot restrict "
                "control-plane ingress to internal ranges; create the subnet or pass --source-range",
            )
        for candidate in candidates:
            try:
                _private_network(candidate, field_name="discovered subnet range")
            except ValueError as exc:
                raise HeadIngressError(
                    action.step,
                    f"{exc}; Eyvan refuses to open the control-plane port to it. Pass explicit internal "
                    "--source-range values instead",
                ) from exc
        return candidates

    def _describe_firewall(self, name: str) -> JsonObject | None:
        """Return one managed rule, refusing a name another rule already owns."""

        action = self._firewall_lookup_action(_ManagedFirewall(name=name, port=0, source_ranges=(), purpose=""))
        result = self._run(action)
        if result.returncode != 0:
            if _is_missing(result):
                return None
            raise HeadCommandError(action.step, _diagnostic(action, result))
        document = _decode(action.step, result.stdout, resource=name)
        if document.get("description") != self.spec.firewall_description:
            raise HeadOwnershipError(
                action.step,
                f"firewall rule {name} exists in {self.spec.project} without the description "
                f"{self.spec.firewall_description!r}; Eyvan did not create it and will not change it",
            )
        return document

    def _ensure_firewall(self, rule: _ManagedFirewall) -> bool:
        """Create one narrow rule when absent; verify it is still narrow when present."""

        existing = self._describe_firewall(rule.name)
        if existing is not None:
            self._verify_existing_ingress(rule, existing)
            return False
        for value in rule.source_ranges:
            if value == IAP_TCP_FORWARDING_RANGE:
                continue
            try:
                _private_network(value, field_name="control-plane source range")
            except ValueError as exc:
                raise HeadIngressError(HeadStep.FIREWALL_CREATE, str(exc)) from exc
        self._execute(self._firewall_create_action(rule))
        return True

    def _verify_existing_ingress(self, rule: _ManagedFirewall, document: JsonObject) -> None:
        """Refuse to continue when an owned rule has been widened since creation."""

        for value in _string_tuple(document.get("sourceRanges")):
            if value == IAP_TCP_FORWARDING_RANGE:
                continue
            try:
                _private_network(value, field_name=f"firewall rule {rule.name} source range")
            except ValueError as exc:
                raise HeadIngressError(
                    HeadStep.FIREWALL_LOOKUP,
                    f"{exc}; Eyvan will not run a control plane behind it. Narrow or delete the rule",
                ) from exc

    def _verified_service_state(self) -> str:
        """Return the unit's state, refusing to call anything but active a success.

        ``systemctl is-active`` exits non-zero for every state that is not
        active, and the state word it prints is the most useful thing an
        operator can be handed at that moment.
        """

        action = self._service_settled_action()
        result = self._run(action)
        state = result.stdout.strip() or "unknown"
        if result.returncode != 0 or state != "active":
            raise HeadCommandError(
                action.step,
                f"{HEAD_SERVICE_NAME} is {state}, not active; read its output with "
                f"`gcloud compute ssh {self.spec.instance_name} --zone {self.spec.zone} "
                f"--command 'sudo journalctl -u {HEAD_SERVICE_NAME} -n 100'`",
            )
        return state

    def _environment_present(self) -> bool:
        """Return whether the head already holds credentials, or fail loudly.

        The distinction that matters is between "ssh worked and the file is not
        there" and "ssh did not work". Treating the second as the first would
        rotate a live head's artifact signing key on one bad network minute, and
        every capability token minted before it would stop verifying.
        """

        action = self._environment_lookup_action()
        for attempt in range(1, self.ssh_ready_attempts + 1):
            result = self._run(action)
            if result.returncode == 0:
                break
            if attempt == self.ssh_ready_attempts:
                raise HeadCommandError(
                    action.step,
                    f"{_diagnostic(action, result)} (after {attempt} attempts over "
                    f"{self.ssh_ready_delay_s * (attempt - 1):g}s)",
                )
            self._sleep(self.ssh_ready_delay_s)
        output = result.stdout.strip()
        if output.endswith(_ENVIRONMENT_PRESENT):
            return True
        if output.endswith(_ENVIRONMENT_ABSENT):
            return False
        raise HeadCommandError(
            action.step,
            f"the head answered {output!r} instead of a credential-presence marker",
        )


def _internal_address(instance: JsonObject) -> str | None:
    """Return the first internal IPv4 address a described instance reports."""

    interfaces = instance.get("networkInterfaces")
    if not isinstance(interfaces, list):
        return None
    for interface in interfaces:
        if is_json_object(interface) and isinstance(value := interface.get("networkIP"), str):
            return value
    return None


def _string_tuple(value: object) -> tuple[str, ...]:
    """Return only the string members of a decoded JSON list."""

    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _decode(step: HeadStep, payload: str, *, resource: str) -> JsonObject:
    """Decode one gcloud JSON object, refusing anything else."""

    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HeadCommandError(step, f"gcloud returned invalid JSON for {resource}") from exc
    if not is_json_object(document):
        raise HeadCommandError(step, f"gcloud returned a non-object description for {resource}")
    return document


def _is_missing(result: subprocess.CompletedProcess[str]) -> bool:
    """Recognize gcloud's not-found outcomes without masking access errors."""

    detail = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
    return result.returncode != 0 and ("not_found" in detail or "not found" in detail or "404" in detail)


def _diagnostic(action: HeadAction, result: subprocess.CompletedProcess[str]) -> str:
    """Build bounded diagnostics that name the command that failed."""

    output = f"{result.stderr or ''}\n{result.stdout or ''}".strip()
    prefix = f"{action.summary} (gcloud exited with status {result.returncode})"
    if output:
        return f"{prefix}: {output[:800]}"
    return prefix


def _spec_from_options(
    config_path: object,
    *,
    fleet: str | None,
    overrides: dict[str, tp.Any],
    require_egress: bool = True,
) -> HeadSpec:
    """Build one spec from a fleet file or from explicit flags, never from both.

    ``require_egress`` is false for ``status`` and ``down``: egress decides how a
    head is *created*, and demanding it to read or delete one would make the two
    recovery commands the hardest to run.
    """

    if not require_egress:
        overrides.setdefault("egress", DEFAULT_HEAD_EGRESS)
    if config_path is not None:
        if fleet is not None:
            raise click.ClickException("--fleet names a head without a config file; pass one or the other")
        config = FleetConfigLoader().load(Path(str(config_path)))
        try:
            return HeadSpec.from_fleet_config(config, **overrides)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    if fleet is None:
        raise click.ClickException("eyvan head needs a fleet configuration file or --fleet with --project/--zone")
    required = {name: overrides.get(name) for name in ("project", "region", "zone", "egress")}
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        flags = ", ".join(f"--{name}" for name in missing)
        raise click.ClickException(f"--fleet requires {flags}")
    supplied = {key: value for key, value in overrides.items() if value is not None}
    try:
        return HeadSpec(fleet=fleet, **supplied)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _head_overrides(**values: tp.Any) -> dict[str, tp.Any]:
    """Drop unsupplied options so a fleet file's own values still apply."""

    return {key: value for key, value in values.items() if value is not None and value != ()}


@click.group("head")
def head_group() -> None:
    """Stand up and operate one Eyvan control-plane head on GCP.

    Sequencing, ownership refusals, idempotence, ingress scope and failure
    handling are covered against an injected command boundary, and the create
    path has run against a real GCP project: a head has been provisioned,
    adopted on a re-run, and used to dispatch a multislice workload.

    What that does not cover is every option this group accepts. A flag no live
    run has passed -- another egress mode, an external database, an OIDC issuer
    -- is tested only against the injected boundary. ``--dry-run`` prints the
    exact gcloud invocations, and reading it first costs nothing.
    """


@head_group.command("up")
@click.argument("config_path", required=False, type=click.Path(dir_okay=False, path_type=str))
@click.option("--fleet", help="Name the head without a fleet file; requires --project, --region and --zone.")
@click.option("--project", help="GCP project that owns the head; defaults to the fleet's provider.project.")
@click.option("--region", help="GCP region; defaults to the fleet's provider.region.")
@click.option("--zone", help="GCP zone; defaults to the fleet's provider.zone.")
@click.option("--machine-type", help=f"Head machine type. [default: {DEFAULT_HEAD_MACHINE_TYPE}]")
@click.option("--port", type=int, help=f"Control-plane port. [default: {DEFAULT_HEAD_PORT}]")
@click.option("--network", help="VPC network the head joins. [default: default]")
@click.option("--subnet", "subnetwork", help="Subnetwork the head joins.")
@click.option(
    "--egress",
    type=click.Choice(BOOTSTRAP_EGRESS_MODES),
    help="How the head reaches package and Google endpoints; defaults to the fleet's provider_options value.",
)
@click.option("--package", help=f"Package specification installed on the head. [default: {DEFAULT_HEAD_PACKAGE}]")
@click.option("--database-url", help="Use an existing PostgreSQL instead of installing one on the head.")
@click.option("--artifact-store-uri", help="Set EYVAN_ARTIFACT_STORE_URI, normally a gs:// prefix.")
@click.option("--oidc-issuer", help="Set EYVAN_OIDC_ISSUER; without it the head refuses every request with 401.")
@click.option("--oidc-audience", help="Set EYVAN_OIDC_AUDIENCE. [default: eyvan]")
@click.option(
    "--allow-insecure-artifact-http",
    is_flag=True,
    help="Let callables carry artifact capability tokens over plain HTTP. This head terminates no TLS, "
    "so without it every @execute submission is refused. Only defensible because ingress is VPC-only.",
)
@click.option(
    "--trust-callable-project",
    "trusted_callable_projects",
    multiple=True,
    help="Let this org/project submit serialized callables; repeatable. Without it a head with an issuer "
    "refuses every @execute submission with 403.",
)
@click.option("--service-account", help="Service account attached to the head VM.")
@click.option(
    "--source-range",
    "source_ranges",
    multiple=True,
    help="Restrict control-plane ingress to these internal CIDRs instead of discovering the VPC's subnets.",
)
@click.option("--boot-disk-size", "boot_disk_size_gb", type=int, help="Head boot disk size in GB. [default: 50]")
@click.option(
    "--iap-ssh/--no-iap-ssh",
    default=True,
    show_default=True,
    help="Manage an SSH ingress rule for Google's IAP forwarding range and tunnel through it.",
)
@click.option(
    "--spot",
    is_flag=True,
    help=(
        "Run the head VM on Spot capacity, for development and qualification planes. GCP may stop it at "
        "any time; its disk and database survive, and starting the VM again resumes the plane."
    ),
)
@click.option(
    "--team",
    is_flag=True,
    help="Write `team: true` into the head's platform file so the plane serves team accounts.",
)
@click.option("--rotate-secrets", is_flag=True, help="Regenerate this head's credentials instead of keeping them.")
@click.option("--dry-run", is_flag=True, help="Print every action and resource name, changing nothing.")
@click.option("--yes", is_flag=True, help="Confirm creating GCP resources without prompting.")
@click.option("--json", "as_json", is_flag=True, help="Emit the plan or the non-secret result as JSON.")
def head_up(
    config_path: str | None,
    fleet: str | None,
    project: str | None,
    region: str | None,
    zone: str | None,
    machine_type: str | None,
    port: int | None,
    network: str | None,
    subnetwork: str | None,
    egress: str | None,
    package: str | None,
    database_url: str | None,
    artifact_store_uri: str | None,
    oidc_issuer: str | None,
    oidc_audience: str | None,
    trusted_callable_projects: tuple[str, ...],
    allow_insecure_artifact_http: bool,
    service_account: str | None,
    source_ranges: tuple[str, ...],
    boot_disk_size_gb: int | None,
    iap_ssh: bool,
    spot: bool,
    team: bool,
    rotate_secrets: bool,
    dry_run: bool,
    yes: bool,
    as_json: bool,
) -> None:
    """Provision a control-plane head and leave it running under systemd.

    Creates or adopts one labelled CPU VM, restricts ingress on the
    control-plane port to the VPC's own internal ranges, generates the
    credentials the server refuses to start without, migrates the schema, and
    enables the unit. Running it twice adopts the head it created and keeps that
    head's credentials; it refuses a same-named VM it does not own.

    This path has run against a real GCP project. Options no live run has
    passed are not qualified by that; ``--dry-run`` prints the exact gcloud
    invocations for those, and reading it first costs nothing.
    """

    spec = _spec_from_options(
        config_path,
        fleet=fleet,
        overrides=_head_overrides(
            project=project,
            region=region,
            zone=zone,
            machine_type=machine_type,
            port=port,
            network=network,
            subnetwork=subnetwork,
            egress=egress,
            package=package,
            database_url=database_url,
            artifact_store_uri=artifact_store_uri,
            oidc_issuer=oidc_issuer,
            oidc_audience=oidc_audience,
            trusted_callable_projects=tuple(trusted_callable_projects),
            allow_insecure_artifact_http=allow_insecure_artifact_http,
            service_account=service_account,
            source_ranges=tuple(source_ranges),
            boot_disk_size_gb=boot_disk_size_gb,
            iap_ssh=iap_ssh,
            spot=spot or None,
            team=team or None,
        ),
    )
    provisioner = HeadProvisioner(spec)
    if dry_run:
        plan = provisioner.plan()
        if as_json:
            click.echo(json.dumps(plan.as_document(), indent=2, sort_keys=True))
        else:
            for line in plan.render_lines():
                click.echo(line)
        return
    if not yes:
        if not is_interactive():
            raise click.ClickException("eyvan head up requires --yes outside a terminal")
        console = autoscale_console(err=as_json)
        for line in provisioner.plan().render_lines():
            console.print(line, highlight=False)
        if not Confirm.ask(f"Create the {spec.instance_name} control plane", default=False, console=console):
            raise click.Abort()
    try:
        result = provisioner.up(rotate_secrets=rotate_secrets)
    except HeadError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(result.as_document(), indent=2, sort_keys=True))
        return
    click.echo(f"Head {spec.instance_name} is {result.service_state} at {result.url}")
    click.echo(f"Control-plane ingress is restricted to {', '.join(result.source_ranges)}.")
    dashboard = result.dashboard_note
    if dashboard is not None:
        click.echo(dashboard)
    warning = result.authentication_warning
    if warning is not None:
        click.echo(warning, err=True)
    trust_warning = result.callable_trust_warning
    if trust_warning is not None:
        click.echo(trust_warning, err=True)
    store_warning = result.artifact_store_warning
    if store_warning is not None:
        click.echo(store_warning, err=True)
    wheelhouse_warning = result.wheelhouse_warning
    if wheelhouse_warning is not None:
        click.echo(wheelhouse_warning, err=True)
    click.echo(f"Next: {result.next_command}")


@head_group.command("status")
@click.argument("config_path", required=False, type=click.Path(dir_okay=False, path_type=str))
@click.option("--fleet", help="Name the head without a fleet file; requires --project, --region and --zone.")
@click.option("--project", help="GCP project that owns the head.")
@click.option("--region", help="GCP region.")
@click.option("--zone", help="GCP zone.")
@click.option("--port", type=int, help=f"Control-plane port. [default: {DEFAULT_HEAD_PORT}]")
@click.option("--network", help="VPC network the head joined. [default: default]")
@click.option("--json", "as_json", is_flag=True, help="Emit the observed head state as JSON.")
def head_status(
    config_path: str | None,
    fleet: str | None,
    project: str | None,
    region: str | None,
    zone: str | None,
    port: int | None,
    network: str | None,
    as_json: bool,
) -> None:
    """Report the head's instance state, ingress scope, and unit state."""

    spec = _spec_from_options(
        config_path,
        fleet=fleet,
        overrides=_head_overrides(project=project, region=region, zone=zone, port=port, network=network),
        require_egress=False,
    )
    try:
        result = HeadProvisioner(spec).status()
    except HeadError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(result.as_document(), indent=2, sort_keys=True))
        return
    if not result.exists:
        click.echo(f"No head named {spec.instance_name} in {spec.project}/{spec.zone}.")
        return
    click.echo(f"{spec.instance_name}: {result.instance_status} at {result.address}")
    click.echo(f"  service       {result.service_state or 'unknown'}")
    click.echo(f"  ingress       {', '.join(result.source_ranges) or 'no managed rule'}")


@head_group.command("rollback")
@click.argument("config_path", required=False, type=click.Path(dir_okay=False, path_type=str))
@click.option("--fleet", help="Name the head without a fleet file; requires --project, --region and --zone.")
@click.option("--project", help="GCP project that owns the head.")
@click.option("--region", help="GCP region.")
@click.option("--zone", help="GCP zone.")
@click.option("--port", type=int, help=f"Control-plane port. [default: {DEFAULT_HEAD_PORT}]")
@click.option("--network", help="VPC network the head joined. [default: default]")
@click.option("--yes", is_flag=True, help="Confirm replacing the running control-plane build.")
def head_rollback(
    config_path: str | None,
    fleet: str | None,
    project: str | None,
    region: str | None,
    zone: str | None,
    port: int | None,
    network: str | None,
    yes: bool,
) -> None:
    """Reinstall the wheel this head ran before its current one.

    Verified by the build digest the head reports after restarting, not by
    whether systemd calls the unit active -- a unit is active the moment its
    process starts, which says nothing about which build it is.
    """

    spec = _spec_from_options(
        config_path,
        fleet=fleet,
        overrides=_head_overrides(project=project, region=region, zone=zone, port=port, network=network),
        require_egress=False,
    )
    try:
        digest = HeadProvisioner(spec).rollback(confirmed=yes)
    except HeadError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{spec.instance_name} rolled back; now running build {digest}")


@head_group.command("down")
@click.argument("config_path", required=False, type=click.Path(dir_okay=False, path_type=str))
@click.option("--fleet", help="Name the head without a fleet file; requires --project, --region and --zone.")
@click.option("--project", help="GCP project that owns the head.")
@click.option("--region", help="GCP region.")
@click.option("--zone", help="GCP zone.")
@click.option("--network", help="VPC network the head joined. [default: default]")
@click.option("--dry-run", is_flag=True, help="Print what teardown would delete, changing nothing.")
@click.option("--yes", is_flag=True, help="Required. Confirm destroying this control plane and its state.")
@click.option("--json", "as_json", is_flag=True, help="Emit the teardown result as JSON.")
def head_down(
    config_path: str | None,
    fleet: str | None,
    project: str | None,
    region: str | None,
    zone: str | None,
    network: str | None,
    dry_run: bool,
    yes: bool,
    as_json: bool,
) -> None:
    """Delete an Eyvan-owned head and its managed firewall rules.

    The head's database lives on the head, so this destroys durable
    control-plane state. ``--yes`` is required and is never inferred from a
    terminal.
    """

    spec = _spec_from_options(
        config_path,
        fleet=fleet,
        overrides=_head_overrides(project=project, region=region, zone=zone, network=network),
        require_egress=False,
    )
    provisioner = HeadProvisioner(spec)
    if dry_run:
        names = [spec.instance_name, spec.firewall_rule_name]
        if spec.iap_ssh:
            names.append(spec.ssh_firewall_rule_name)
        if as_json:
            click.echo(json.dumps({"dry_run": True, "would_delete": sorted(names)}, indent=2, sort_keys=True))
            return
        click.echo(f"eyvan head down would delete: {', '.join(sorted(names))}")
        return
    try:
        result = provisioner.down(confirmed=yes)
    except HeadError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(result.as_document(), indent=2, sort_keys=True))
        return
    if not result.deleted_instance and not result.deleted_firewall_rules:
        click.echo(f"No Eyvan-owned head named {spec.instance_name} in {spec.project}/{spec.zone}.")
        return
    click.echo(f"Deleted {spec.instance_name} and {len(result.deleted_firewall_rules)} managed firewall rule(s).")


def register_head_commands(group: click.Group) -> None:
    """Attach control-plane head commands to the root CLI."""

    group.add_command(head_group)


__all__ = (
    "BOOTSTRAP_EGRESS_MODES",
    "DEFAULT_HEAD_MACHINE_TYPE",
    "DEFAULT_HEAD_PORT",
    "HEAD_ENVIRONMENT_PATH",
    "HEAD_FIREWALL_DESCRIPTION_PREFIX",
    "HEAD_FLEET_LABEL_KEY",
    "HEAD_MANAGED_LABEL_KEY",
    "HEAD_MANAGED_LABEL_VALUE",
    "HEAD_SERVICE_NAME",
    "IAP_TCP_FORWARDING_RANGE",
    "UNRESOLVED_SOURCE_RANGES",
    "HeadAction",
    "HeadCommandError",
    "HeadCommandRunner",
    "HeadDownResult",
    "HeadError",
    "HeadIngressError",
    "HeadOwnershipError",
    "HeadPlan",
    "HeadProvisioner",
    "HeadSecrets",
    "HeadSpec",
    "HeadStatusResult",
    "HeadStep",
    "HeadUpResult",
    "head_down",
    "head_group",
    "head_status",
    "head_up",
    "register_head_commands",
    "render_head_bootstrap_script",
)
