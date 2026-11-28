# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Narrow local-development ingress for profile-managed native GCP workers.

This module deliberately owns only the firewall rule that lets the machine
running a local Eyvan agent reach public-IP native workers.  It never opens a
project-wide rule and it is not enabled by regular fleet YAML or environment
settings.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import subprocess
import typing as tp
from dataclasses import dataclass, field

import httpx

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode

_DEFAULT_NETWORK = "default"
_FIREWALL_PRIORITY = 1000
_INGRESS_DESCRIPTION_PREFIX = "eyvan-native-worker-ingress:v1"


def _managed_rule_name(network: FirewallNetworkReference, network_tag: str, port: int) -> str:
    """Return the deterministic rule name, derived without the source range.

    The name deliberately excludes the source /32, so an existing rule can be
    found and kept even on a day when the local public IPv4 cannot be resolved.
    """

    suffix = hashlib.sha256(network.canonical_name.encode("utf-8")).hexdigest()[:12]
    return f"{network_tag}-p{port}-{suffix}"


_NETWORK_NAME_PATTERN = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_TAG_PATTERN = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_NETWORK_REFERENCE_PATTERN = re.compile(
    r"^(?:https://www\.googleapis\.com/compute/v1/)?projects/"
    r"(?P<project>[a-z][a-z0-9-]{4,28}[a-z0-9])/global/networks/"
    r"(?P<name>[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?)$"
)


class NativeWorkerEndpointMode(StrEnum):
    """Network route Eyvan may publish for a native worker."""

    PRIVATE = "private"
    PROFILE_MANAGED_PUBLIC = "profile-managed-public"


class NativeWorkerIngressError(CapacityProviderError):
    """Base error for profile-managed public native-worker ingress."""


class PublicIpv4ResolutionError(NativeWorkerIngressError):
    """Eyvan could not safely determine the local public IPv4 address.

    Resolution runs on the operator's own network, so a failure here is "could
    not find out", never "the provider errored": it is classified retryable and
    unavailable rather than a defect charged to the capacity pool.
    """

    def __init__(self, diagnostic: str, *, _endpoint_failures: tp.Sequence[str] = ()) -> None:
        super().__init__(
            diagnostic,
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
            retryable=True,
        )
        self.endpoint_failures = tuple(_endpoint_failures)


class GcloudFirewallRuleError(NativeWorkerIngressError):
    """A gcloud firewall-rule operation failed or returned unsafe state."""


class FirewallRuleConflictError(GcloudFirewallRuleError):
    """A deterministic rule name is already occupied by another rule."""


class PublicIpv4Resolver(tp.Protocol):
    """Resolve the caller's routable IPv4 address without using proxy settings."""

    def resolve(self) -> ipaddress.IPv4Address:
        """Return the caller's public IPv4 address."""

        ...


class GcloudCommandRunner(tp.Protocol):
    """Typed subprocess boundary used by the firewall manager."""

    def __call__(self, args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run one gcloud command and return its complete result."""

        ...


class NativeWorkerPublicIngress(tp.Protocol):
    """Ensure the narrow local-agent ingress rule and return its worker tag."""

    def ensure(self, *, project: str, network: str | None, port: int) -> str:
        """Ensure a rule for one project/network/worker-port combination."""

        ...


@dataclass(frozen=True, slots=True)
class FirewallNetworkReference:
    """A VPC name and the project that owns its firewall rules."""

    project: str
    name: str

    @classmethod
    def from_config(cls, *, capacity_project: str, network: str | None) -> FirewallNetworkReference:
        """Resolve a simple or fully-qualified VPC reference.

        A fully-qualified network may point at a Shared VPC host project.  A
        simple network name belongs to the capacity project.
        """

        _validate_project_id(capacity_project, field_name="capacity project")
        if network is None:
            return cls(project=capacity_project, name=_DEFAULT_NETWORK)
        if not isinstance(network, str) or not network or network != network.strip():
            raise NativeWorkerIngressError("native worker network must be a non-empty trimmed string when set")
        match = _NETWORK_REFERENCE_PATTERN.fullmatch(network)
        if match is not None:
            return cls(project=match.group("project"), name=match.group("name"))
        _validate_network_name(network)
        return cls(project=capacity_project, name=network)

    @classmethod
    def from_gcloud_value(cls, value: str) -> FirewallNetworkReference:
        """Parse the network reference returned by ``gcloud --format=json``."""

        if not isinstance(value, str) or not value or value != value.strip():
            raise FirewallRuleConflictError("managed firewall rule returned an invalid network reference")
        match = _NETWORK_REFERENCE_PATTERN.fullmatch(value)
        if match is None:
            raise FirewallRuleConflictError("managed firewall rule returned an unsupported network reference")
        return cls(project=match.group("project"), name=match.group("name"))

    @property
    def canonical_name(self) -> str:
        """Return the canonical GCP resource name used in rule derivation."""

        return f"projects/{self.project}/global/networks/{self.name}"


@dataclass(frozen=True, slots=True)
class FirewallRuleSpec:
    """Complete static and dynamic shape of one Eyvan-managed firewall rule."""

    network: FirewallNetworkReference
    network_tag: str
    port: int
    source_range: ipaddress.IPv4Network

    def __post_init__(self) -> None:
        """Validate a rule before any subprocess execution."""

        _validate_network_tag(self.network_tag)
        _validate_port(self.port)
        if self.source_range.prefixlen != 32:
            raise NativeWorkerIngressError("native worker ingress source range must be one IPv4 /32")
        if len(self.rule_name) > 63:
            raise NativeWorkerIngressError("native worker ingress tag is too long to derive a GCP firewall rule name")

    @property
    def rule_name(self) -> str:
        """Return a deterministic VPC-scoped managed firewall rule name."""

        return _managed_rule_name(self.network, self.network_tag, self.port)

    @property
    def description(self) -> str:
        """Return the exact ownership marker expected from a managed rule."""

        return f"{_INGRESS_DESCRIPTION_PREFIX}:{self.network_tag}"

    @property
    def source_range_text(self) -> str:
        """Return the exact IPv4 /32 accepted by gcloud."""

        return self.source_range.with_prefixlen

    def describe_args(self) -> tuple[str, ...]:
        """Build the read-only gcloud request for this rule."""

        return (
            "compute",
            "firewall-rules",
            "describe",
            self.rule_name,
            "--project",
            self.network.project,
            "--format=json",
        )

    def create_args(self) -> tuple[str, ...]:
        """Build the narrow ingress-rule creation command."""

        return (
            "compute",
            "firewall-rules",
            "create",
            self.rule_name,
            "--project",
            self.network.project,
            "--network",
            self.network.name,
            "--direction=INGRESS",
            f"--priority={_FIREWALL_PRIORITY}",
            f"--allow=tcp:{self.port}",
            f"--source-ranges={self.source_range_text}",
            f"--target-tags={self.network_tag}",
            f"--description={self.description}",
            "--quiet",
        )

    def update_args(self) -> tuple[str, ...]:
        """Build a strict managed-rule update for a changed local IPv4 address."""

        return (
            "compute",
            "firewall-rules",
            "update",
            self.rule_name,
            "--project",
            self.network.project,
            f"--priority={_FIREWALL_PRIORITY}",
            f"--allow=tcp:{self.port}",
            f"--source-ranges={self.source_range_text}",
            f"--target-tags={self.network_tag}",
            f"--description={self.description}",
            "--no-disabled",
            "--quiet",
        )


@dataclass(frozen=True, slots=True)
class HttpPublicIpv4Resolver:
    """Resolve a local public IPv4 from fixed HTTPS services without proxy use.

    One endpoint failing used to fail the whole resolution: a cell that was
    otherwise fine was marked PROVIDER_ERROR while its TPU kept billing, only
    because api.ipify.org had a bad minute. Resolution now walks every endpoint
    and only reports failure when *all* of them failed -- "I could not find
    out" requires more than one unanswered question.
    """

    endpoints: tuple[str, ...] = (
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://ifconfig.me/ip",
        "https://ipv4.icanhazip.com",
    )
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise ValueError("public IPv4 resolution requires at least one endpoint")
        for endpoint in self.endpoints:
            # Fail closed: an address read over plain HTTP can be steered, and
            # the answer becomes a firewall source range.
            if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
                raise ValueError("public IPv4 endpoints must be HTTPS URLs")

    def resolve(self) -> ipaddress.IPv4Address:
        """Return a strictly parsed public IPv4 address or fail closed."""

        failures: list[str] = []
        for endpoint in self.endpoints:
            try:
                value = self._fetch(endpoint)
            except PublicIpv4ResolutionError as exc:
                failures.append(f"{endpoint}: {exc.diagnostic}")
                continue
            return value
        raise PublicIpv4ResolutionError(
            "could not determine the local public IPv4 address from any discovery endpoint; "
            "Eyvan did not change firewall ingress",
            _endpoint_failures=failures,
        )

    def _fetch(self, endpoint: str) -> ipaddress.IPv4Address:
        """Read one endpoint and strictly validate its answer."""

        try:
            with httpx.Client(timeout=self.timeout_s, trust_env=False, follow_redirects=False) as client:
                response = client.get(endpoint)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise PublicIpv4ResolutionError(f"request failed: {exc}") from exc
        value = response.text.strip()
        try:
            parsed = ipaddress.ip_address(value)
        except ValueError as exc:
            raise PublicIpv4ResolutionError("discovery returned an invalid address") from exc
        if not isinstance(parsed, ipaddress.IPv4Address):
            raise PublicIpv4ResolutionError("discovery returned an IPv6 address")
        return parsed


#: How long one firewall reconciliation may take.
#:
#: Sixty seconds was too tight, and it was found the only way it could be: a
#: `firewall-rules update` on a rule with a single source range exceeded it on a
#: degraded link, which marked a cell PROVIDER_ERROR while the Compute instance
#: it had already created kept running and billing.
#:
#: Configurable and defaulted higher, matching the TPU path's reasoning at
#: ``EYVAN_GCLOUD_TPU_TIMEOUT_S``: a provider call that is slow is not a provider
#: call that failed, and the two must not share an outcome.
DEFAULT_INGRESS_TIMEOUT_S = 300.0


def _gcloud_runner(args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a bounded gcloud command, surfacing a timeout as a retryable failure.

    A raw ``TimeoutExpired`` escaping here reached the agent as an untyped 500,
    which the control plane recorded as ``provider_failure`` with no
    ``unavailable_detail`` at all -- so an operator saw that a cell had failed
    and could not learn why without reading an agent log file.

    ``PROVIDER_UNAVAILABLE`` and ``retryable=True`` are the honest
    classification: the ingress rule is reconciled on every pass, so a slow call
    costs one cycle rather than a cell.
    """

    timeout_s = max(float(os.getenv("EYVAN_GCLOUD_INGRESS_TIMEOUT_S", DEFAULT_INGRESS_TIMEOUT_S)), 1.0)
    try:
        return subprocess.run(
            ["gcloud", *args], check=False, capture_output=True, text=True, timeout=timeout_s, env=provider_cli_env()
        )
    except subprocess.TimeoutExpired as exc:
        raise CapacityProviderError(
            f"native worker ingress reconciliation timed out after {timeout_s:g}s: gcloud {' '.join(args[:3])}",
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
            retryable=True,
        ) from exc


@dataclass(frozen=True, slots=True)
class GcloudNativeWorkerPublicIngress:
    """Idempotently manage one laptop-to-native-worker GCP firewall rule."""

    network_tag: str
    public_ipv4_resolver: PublicIpv4Resolver = field(default_factory=HttpPublicIpv4Resolver)
    runner: GcloudCommandRunner = _gcloud_runner

    def __post_init__(self) -> None:
        """Reject an unsafe network tag before a worker can be provisioned."""

        _validate_network_tag(self.network_tag)

    def ensure(self, *, project: str, network: str | None, port: int) -> str:
        """Synchronize a deterministic rule to the current public IPv4 /32.

        This method does not delete rules during capacity scale-down.  The rule
        is runtime-owned and must remain available for a later scale-from-zero.

        When the local public IPv4 cannot be resolved, an existing managed rule
        is kept as it stands instead of failing the reconciliation: the rule's
        source range was good yesterday, the readiness probe is what actually
        decides reachability, and tearing the pass down here would strand
        capacity that may already be serving. With no existing rule there is
        nothing safe to keep, so the resolution error propagates.
        """

        target_network = FirewallNetworkReference.from_config(capacity_project=project, network=network)
        _validate_port(port)
        try:
            resolved = self.public_ipv4_resolver.resolve()
        except PublicIpv4ResolutionError:
            if self._existing_rule_is_managed(target_network, port):
                return self.network_tag
            raise
        source_range = ipaddress.IPv4Network(f"{resolved}/32")
        spec = FirewallRuleSpec(
            network=target_network,
            network_tag=self.network_tag,
            port=port,
            source_range=source_range,
        )
        existing = self._describe(spec)
        if existing is None:
            created = self.runner(spec.create_args())
            if created.returncode == 0:
                return self.network_tag
            # A concurrent local agent can win the deterministic create race.
            existing_after_create = self._describe(spec)
            if existing_after_create is None:
                raise _gcloud_operation_error("create", spec, created)
            self._reconcile_existing(spec, existing_after_create)
            return self.network_tag
        self._reconcile_existing(spec, existing)
        return self.network_tag

    def _existing_rule_is_managed(self, network: FirewallNetworkReference, port: int) -> bool:
        """Return whether the deterministic rule already exists and is ours and enabled.

        Anything unreadable or surprising is answered ``False``: this path only
        ever *keeps* a rule, and keeping requires knowing exactly what is kept.
        """

        name = _managed_rule_name(network, self.network_tag, port)
        result = self.runner(
            (
                "compute",
                "firewall-rules",
                "describe",
                name,
                "--project",
                network.project,
                "--format=json",
            )
        )
        if result.returncode != 0:
            return False
        try:
            raw = _decode_object(result.stdout, resource=f"firewall rule {name!r}")
        except GcloudFirewallRuleError:
            return False
        if raw.get("disabled", False) is not False:
            return False
        parsed = _parse_managed_rule(raw)
        return (
            parsed is not None
            and parsed.name == name
            and parsed.network == network
            and parsed.network_tag == self.network_tag
            and parsed.port == port
        )

    def _describe(self, spec: FirewallRuleSpec) -> JsonObject | None:
        """Return the exact rule or ``None`` when gcloud reports it absent."""

        result = self.runner(spec.describe_args())
        if result.returncode == 0:
            return _decode_object(result.stdout, resource=f"firewall rule {spec.rule_name!r}")
        if _is_not_found(result):
            return None
        raise _gcloud_operation_error("describe", spec, result)

    def _reconcile_existing(self, spec: FirewallRuleSpec, raw: JsonObject) -> None:
        """Validate ownership before optionally changing one managed source range."""

        current_source, disabled = _validate_managed_rule(raw, spec)
        if current_source == spec.source_range and not disabled:
            return
        result = self.runner(spec.update_args())
        if result.returncode != 0:
            raise _gcloud_operation_error("update", spec, result)


@dataclass(frozen=True, slots=True)
class ManagedIngressRule:
    """One firewall rule this runtime created, as found in a project."""

    name: str
    network: FirewallNetworkReference
    network_tag: str
    port: int
    source_range: str

    @property
    def description(self) -> str:
        """Return the ownership marker this rule was matched by."""

        return f"{_INGRESS_DESCRIPTION_PREFIX}:{self.network_tag}"


def list_managed_ingress_rules(
    *,
    project: str,
    runner: GcloudCommandRunner = _gcloud_runner,
) -> tuple[ManagedIngressRule, ...]:
    """Return every rule in ``project`` carrying this runtime's ownership marker.

    ``ensure`` deliberately never deletes during scale-down -- the rule has to
    outlive a scale-to-zero so the next scale-up can reach its workers.  The
    consequence is that nothing ever removed one either, so a rule outlived the
    project that needed it and accumulated silently.  Finding them is the
    prerequisite for removing them, and it is read-only.
    """

    _validate_project_id(project, field_name="project")
    result = runner(
        (
            "compute",
            "firewall-rules",
            "list",
            "--project",
            project,
            f"--filter=description~^{_INGRESS_DESCRIPTION_PREFIX}",
            "--format=json",
        )
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"gcloud returned exit status {result.returncode}"
        raise GcloudFirewallRuleError(f"could not list managed native worker firewall rules: {detail}")
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise GcloudFirewallRuleError(f"managed firewall rule listing returned invalid JSON: {exc.msg}") from exc
    if not isinstance(decoded, list):
        raise GcloudFirewallRuleError("managed firewall rule listing did not return a JSON array")

    # The gcloud filter narrows the transfer; this loop is what decides
    # ownership. A server-side filter is a convenience and can match loosely --
    # deleting on the strength of one would eventually delete somebody else's
    # rule in a shared project.
    rules: list[ManagedIngressRule] = []
    for raw in decoded:
        if not is_json_object(raw):
            continue
        parsed = _parse_managed_rule(raw)
        if parsed is not None:
            rules.append(parsed)
    return tuple(sorted(rules, key=lambda rule: rule.name))


def release_managed_ingress_rule(
    *,
    project: str,
    rule_name: str,
    runner: GcloudCommandRunner = _gcloud_runner,
) -> None:
    """Delete one rule after confirming this runtime owns it.

    The ownership check is re-read here rather than trusted from a prior list:
    between listing and deleting, a name can come to mean a different rule, and
    a firewall rule is exactly the kind of thing where deleting the wrong one is
    both silent and someone else's outage.
    """

    _validate_project_id(project, field_name="project")
    described = runner(
        (
            "compute",
            "firewall-rules",
            "describe",
            rule_name,
            "--project",
            project,
            "--format=json",
        )
    )
    if described.returncode != 0:
        if _is_not_found(described):
            return
        detail = (described.stderr or described.stdout).strip() or f"exit status {described.returncode}"
        raise GcloudFirewallRuleError(f"could not describe firewall rule {rule_name!r}: {detail}")
    raw = _decode_object(described.stdout, resource=f"firewall rule {rule_name!r}")
    if _parse_managed_rule(raw) is None:
        raise FirewallRuleConflictError(
            f"firewall rule {rule_name!r} is not owned by this Eyvan runtime; refusing to delete it"
        )
    deleted = runner(
        (
            "compute",
            "firewall-rules",
            "delete",
            rule_name,
            "--project",
            project,
            "--quiet",
        )
    )
    if deleted.returncode != 0:
        if _is_not_found(deleted):
            return
        detail = (deleted.stderr or deleted.stdout).strip() or f"exit status {deleted.returncode}"
        raise GcloudFirewallRuleError(f"could not delete managed firewall rule {rule_name!r}: {detail}")


def _parse_managed_rule(raw: JsonObject) -> ManagedIngressRule | None:
    """Return the rule when it is unambiguously ours, ``None`` otherwise."""

    description = raw.get("description")
    if not isinstance(description, str):
        return None
    marker = f"{_INGRESS_DESCRIPTION_PREFIX}:"
    if not description.startswith(marker):
        return None
    network_tag = description[len(marker) :]
    if _TAG_PATTERN.fullmatch(network_tag) is None:
        return None
    if raw.get("direction") != "INGRESS":
        return None
    name = raw.get("name")
    network_value = raw.get("network")
    if not isinstance(name, str) or not name or not isinstance(network_value, str):
        return None
    target_tags = raw.get("targetTags")
    if target_tags != [network_tag]:
        return None
    allowed = raw.get("allowed")
    if not isinstance(allowed, list) or len(allowed) != 1 or not is_json_object(allowed[0]):
        return None
    ports = allowed[0].get("ports")
    if allowed[0].get("IPProtocol") != "tcp" or not isinstance(ports, list) or len(ports) != 1:
        return None
    try:
        port = int(str(ports[0]))
    except ValueError:
        return None
    if not 1 <= port <= 65535:
        return None
    source_ranges = raw.get("sourceRanges")
    if not isinstance(source_ranges, list) or len(source_ranges) != 1 or not isinstance(source_ranges[0], str):
        return None
    try:
        network = FirewallNetworkReference.from_gcloud_value(network_value)
    except FirewallRuleConflictError:
        return None
    return ManagedIngressRule(
        name=name,
        network=network,
        network_tag=network_tag,
        port=port,
        source_range=source_ranges[0],
    )


def endpoint_mode_from_metadata(metadata: tp.Mapping[str, JsonValue]) -> NativeWorkerEndpointMode:
    """Parse the persisted native-worker route mode with a private legacy default."""

    value = metadata.get("native_worker_endpoint_mode")
    if value is None:
        return NativeWorkerEndpointMode.PRIVATE
    if not isinstance(value, str):
        raise ValueError("native worker endpoint mode must be a string")
    try:
        return NativeWorkerEndpointMode(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in NativeWorkerEndpointMode)
        raise ValueError(f"native worker endpoint mode must be one of: {choices}") from exc


def _validate_project_id(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _PROJECT_ID_PATTERN.fullmatch(value) is None:
        raise NativeWorkerIngressError(f"{field_name} must be a valid GCP project ID")


def _validate_network_name(value: str) -> None:
    if _NETWORK_NAME_PATTERN.fullmatch(value) is None:
        raise NativeWorkerIngressError("native worker network must be a valid VPC name")


def _validate_network_tag(value: str) -> None:
    if not isinstance(value, str) or _TAG_PATTERN.fullmatch(value) is None:
        raise NativeWorkerIngressError("native worker ingress tag must be a valid GCP network tag")


def _validate_port(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise NativeWorkerIngressError("native worker ingress port must be an integer between 1 and 65535")


def _decode_object(output: str, *, resource: str) -> JsonObject:
    try:
        decoded = json.loads(output)
    except json.JSONDecodeError as exc:
        raise GcloudFirewallRuleError(f"{resource} returned invalid JSON: {exc.msg}") from exc
    if not is_json_object(decoded):
        raise GcloudFirewallRuleError(f"{resource} did not return a JSON object")
    return decoded


def _is_not_found(result: subprocess.CompletedProcess[str]) -> bool:
    text = f"{result.stderr}\n{result.stdout}".lower()
    return "not_found" in text or "not found" in text


def _gcloud_operation_error(
    operation: str,
    spec: FirewallRuleSpec,
    result: subprocess.CompletedProcess[str],
) -> GcloudFirewallRuleError:
    detail = (result.stderr or result.stdout).strip()
    if not detail:
        detail = f"gcloud returned exit status {result.returncode}"
    return GcloudFirewallRuleError(
        f"could not {operation} managed native worker firewall rule {spec.rule_name!r}: {detail}"
    )


def _validate_managed_rule(raw: JsonObject, spec: FirewallRuleSpec) -> tuple[ipaddress.IPv4Network, bool]:
    """Validate exact ownership and return the existing narrow source range."""

    description = raw.get("description")
    if description != spec.description:
        raise FirewallRuleConflictError(
            f"firewall rule {spec.rule_name!r} is not owned by this Eyvan runtime; refusing to modify it"
        )
    direction = raw.get("direction")
    if direction != "INGRESS":
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} is not an ingress rule")
    network_value = raw.get("network")
    if not isinstance(network_value, str):
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} has no valid network")
    if FirewallNetworkReference.from_gcloud_value(network_value) != spec.network:
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} targets another network")
    priority = raw.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int) or priority != _FIREWALL_PRIORITY:
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} has an unexpected priority")
    target_tags = raw.get("targetTags")
    if target_tags != [spec.network_tag]:
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} has unexpected target tags")
    _validate_allowed(raw.get("allowed"), spec)
    source_range = _managed_source_range(raw.get("sourceRanges"), spec)
    disabled = raw.get("disabled", False)
    if not isinstance(disabled, bool):
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} has an invalid disabled state")
    return source_range, disabled


def _validate_allowed(value: JsonValue | None, spec: FirewallRuleSpec) -> None:
    if not isinstance(value, list) or len(value) != 1 or not is_json_object(value[0]):
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} has unexpected allow rules")
    allowed = value[0]
    protocol = allowed.get("IPProtocol")
    ports = allowed.get("ports")
    if protocol != "tcp" or ports != [str(spec.port)]:
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} is not limited to TCP {spec.port}")


def _managed_source_range(value: JsonValue | None, spec: FirewallRuleSpec) -> ipaddress.IPv4Network:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} has unexpected source ranges")
    try:
        source_range = ipaddress.ip_network(value[0], strict=True)
    except ValueError as exc:
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} has an invalid source range") from exc
    if not isinstance(source_range, ipaddress.IPv4Network) or source_range.prefixlen != 32:
        raise FirewallRuleConflictError(f"managed firewall rule {spec.rule_name!r} is not limited to one IPv4 /32")
    return source_range


__all__ = (
    "FirewallNetworkReference",
    "FirewallRuleConflictError",
    "GcloudCommandRunner",
    "GcloudFirewallRuleError",
    "GcloudNativeWorkerPublicIngress",
    "HttpPublicIpv4Resolver",
    "ManagedIngressRule",
    "NativeWorkerEndpointMode",
    "NativeWorkerIngressError",
    "NativeWorkerPublicIngress",
    "PublicIpv4ResolutionError",
    "PublicIpv4Resolver",
    "endpoint_mode_from_metadata",
    "list_managed_ingress_rules",
    "release_managed_ingress_rule",
)
