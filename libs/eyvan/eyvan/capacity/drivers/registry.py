# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registry of capacity backends, mirroring the execution-runtime registry.

Execution backends have always been registered
(:mod:`eyvan.worker.runtime.registry`), while capacity backends were selected by a
hard-coded ``if/elif/else`` in the cell agent with eager driver imports. That
asymmetry meant "add a backend" was a *registration* on one axis and an *edit to
a conditional* on the other, and it is the thing that made a second capacity
substrate awkward to introduce.

Every factory imports its driver lazily inside the function body, for the same
reason the execution registry does: selecting one backend must never import
another's dependencies. That discipline is what keeps a GCP-free deployment from
importing the GCP driver, and it is why the Ray adapter stays out of a native
install.

**Out-of-tree providers.** The same registry serves distributions Eyvan does not
ship, through the ``eyvan.capacity_providers`` entry-point group and the
versioned ABI in ``docs/providers.md``. Three properties are deliberate:

* *Discovery is opt-in.* ``EYVAN_CAPACITY_PROVIDER_PLUGINS`` is empty by
  default, so an installed distribution changes nothing until an operator names
  it. A control plane whose set of capacity backends can be extended by a
  transitive dependency is a supply-chain problem, not a feature.
* *A version mismatch refuses.* A provider declaring a major version this build
  does not implement -- or a newer minor -- is not loaded and not adapted. The
  guarantees it was written against are the ones it will fence a fleet with.
* *Refusals are reported, never swallowed.* An entry point that raises on import
  is an ``ERROR`` log line, a :class:`ProviderPluginFailure`, and the text of
  the ``ValueError`` an operator gets when they try to use that backend. The
  failure mode this replaces -- "unknown capacity backend" for a provider that
  is installed but broken -- sends the operator to the wrong file.
"""

from __future__ import annotations

import logging
import os
import typing as tp
from dataclasses import dataclass

from eyvan._internal.compat import StrEnum
from eyvan.capacity.drivers.base import CapacityDriver, CapacityProviderPlugin

if tp.TYPE_CHECKING:
    from eyvan.capacity.fleet_state import FleetStateStore
    from eyvan.config import EyvanSettings

logger = logging.getLogger(__name__)

#: A factory receives the requested backend name so one factory can serve
#: several aliases that differ only by mode (as the GCP driver does).
CapacityBackendFactory = tp.Callable[..., CapacityDriver]

#: Setuptools/PEP 621 entry-point group an out-of-tree provider publishes under.
CAPACITY_PROVIDER_ENTRY_POINT_GROUP = "eyvan.capacity_providers"

#: The ABI this Eyvan implements. ``MAJOR`` changes when a guarantee in
#: ``docs/providers.md`` is withdrawn or a signature changes; ``MINOR`` changes
#: when a guarantee is added. A provider declaring a different major, or a newer
#: minor than this, is **refused**: it was written against promises this build
#: does not make, and adapting silently is how a fleet ends up fenced by a
#: contract nobody implements.
CAPACITY_PROVIDER_ABI_VERSION = "1.0"

#: Comma-separated entry-point names an operator has chosen to trust, or ``*``
#: for every advertised provider. Empty -- the default -- disables plugin
#: discovery entirely, which is exactly today's behaviour: an installed
#: distribution cannot change which backends exist until an operator names it.
CAPACITY_PROVIDER_PLUGIN_ENV = "EYVAN_CAPACITY_PROVIDER_PLUGINS"


class ProviderPluginRefusal(StrEnum):
    """Why one advertised capacity provider was not loaded."""

    #: The entry point raised while being imported or resolved.
    IMPORT = "import"
    #: The plugin declared an ABI this build does not implement.
    ABI_VERSION = "abi_version"
    #: The published object is not a ``CapacityProviderPlugin``.
    CONTRACT = "contract"
    #: Two plugins, or a plugin and a built-in, claim the same backend name.
    CONFLICT = "conflict"


class CapacityProviderPluginError(RuntimeError):
    """A published capacity provider was refused rather than loaded."""

    def __init__(self, message: str, *, refusal: ProviderPluginRefusal) -> None:
        self.refusal = refusal
        super().__init__(message)


@dataclass(frozen=True, slots=True, order=True)
class ProviderAbiVersion:
    """A parsed ``MAJOR.MINOR`` capacity-provider ABI version."""

    major: int
    minor: int

    @classmethod
    def parse(cls, value: object) -> ProviderAbiVersion:
        """Parse a declared ABI version.

        Args:
            value: The provider's declared ``abi_version``.

        Returns:
            The parsed version.

        Raises:
            CapacityProviderPluginError: If the value is not ``"MAJOR.MINOR"``
                with non-negative integers. A version that cannot be parsed is
                a contract violation, never an assumed default -- guessing
                would mean loading a provider whose promises are unknown.
        """

        if not isinstance(value, str):
            raise CapacityProviderPluginError(
                f"capacity provider abi_version must be a 'MAJOR.MINOR' string, got {type(value).__name__}",
                refusal=ProviderPluginRefusal.CONTRACT,
            )
        parts = value.strip().split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise CapacityProviderPluginError(
                f"capacity provider abi_version {value!r} is not a 'MAJOR.MINOR' version",
                refusal=ProviderPluginRefusal.CONTRACT,
            )
        return cls(major=int(parts[0]), minor=int(parts[1]))

    def __str__(self) -> str:
        """Render the version the way a provider declares it."""

        return f"{self.major}.{self.minor}"

    def refusal_against(self, host: ProviderAbiVersion) -> str | None:
        """Return why ``host`` cannot honour this provider's ABI, or ``None``.

        Args:
            host: The ABI this Eyvan build implements.

        Returns:
            An operator-facing reason, or ``None`` when the provider is loadable.
        """

        if self.major != host.major:
            return (
                f"provider targets capacity ABI {self} but this Eyvan implements {host}; "
                "a major version difference withdraws or changes a guarantee, so it is refused"
            )
        if self.minor > host.minor:
            return (
                f"provider targets capacity ABI {self} but this Eyvan implements {host}; "
                "the provider requires guarantees this build does not make yet"
            )
        return None


HOST_PROVIDER_ABI = ProviderAbiVersion.parse(CAPACITY_PROVIDER_ABI_VERSION)


@dataclass(frozen=True, slots=True)
class ProviderPluginFailure:
    """One advertised provider that was refused, and why."""

    name: str
    value: str
    refusal: ProviderPluginRefusal
    reason: str

    def describe(self) -> str:
        """Render an operator-facing one-line diagnostic."""

        return f"{self.name} ({self.value}) refused [{self.refusal.value}]: {self.reason}"


@dataclass(frozen=True, slots=True)
class LoadedProviderPlugin:
    """One accepted provider plugin and the names it serves."""

    name: str
    value: str
    abi_version: ProviderAbiVersion
    backend_names: tuple[str, ...]
    plugin: CapacityProviderPlugin


@dataclass(frozen=True, slots=True)
class ProviderPluginDiscovery:
    """The outcome of one plugin discovery pass -- accepted **and** refused."""

    plugins: tuple[LoadedProviderPlugin, ...] = ()
    failures: tuple[ProviderPluginFailure, ...] = ()

    def require(self) -> None:
        """Raise if any advertised provider was refused.

        Raises:
            CapacityProviderPluginError: Listing every refusal. Used by callers
                that would rather not start than start with a provider missing.
        """

        if not self.failures:
            return
        detail = "; ".join(failure.describe() for failure in self.failures)
        raise CapacityProviderPluginError(
            f"{len(self.failures)} capacity provider plugin(s) were refused: {detail}",
            refusal=self.failures[0].refusal,
        )


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _local_factory(name: str, *, state: FleetStateStore, settings: EyvanSettings) -> CapacityDriver:
    del name, settings
    from eyvan.capacity.drivers.local import LocalCapacityDriver

    return LocalCapacityDriver(state)


def _manual_factory(name: str, *, state: FleetStateStore, settings: EyvanSettings) -> CapacityDriver:
    del name
    from eyvan.capacity.drivers.manual import ManualCapacityDriver

    return ManualCapacityDriver(state, native_worker_port=settings.native_worker_port)


def _onprem_ssh_factory(name: str, *, state: FleetStateStore, settings: EyvanSettings) -> CapacityDriver:
    """Build the on-premises SSH driver for operator-owned hardware."""

    del name
    from eyvan.capacity.drivers.onprem_ssh import OnPremSshCapacityDriver

    return OnPremSshCapacityDriver(
        state,
        native_worker_port=settings.native_worker_port,
        native_worker_state_dir=settings.native_worker_state_dir,
    )


def _kubernetes_factory(name: str, *, state: FleetStateStore, settings: EyvanSettings) -> CapacityDriver:
    """Build the Kubernetes driver for either of its two registered names.

    One factory, two names -- exactly as the three GCP modes share one -- because
    ``kubernetes`` and ``k8s`` are the same backend and a second factory would be
    a second place to keep the defaults in sync.
    """

    del name
    from eyvan.capacity.drivers.kubernetes import KubernetesCapacityDriver
    from eyvan.capacity.providers.kubernetes.api import KubernetesApi

    return KubernetesCapacityDriver(
        state,
        api=KubernetesApi(),
        namespace=settings.kubernetes_namespace,
        image=settings.kubernetes_worker_image,
        shared_secret=settings.native_worker_root_secret,
    )


def _aws_factory(name: str, *, state: FleetStateStore, settings: EyvanSettings) -> CapacityDriver:
    """Build the EC2 driver, refusing to guess at a region or a credential.

    Refused loudly rather than defaulted: a driver that quietly picked
    ``us-east-1`` would provision real machines in a region nobody chose, and one
    that fell back to an anonymous client would fail every call with an
    authentication error that says nothing about the missing setting.
    """

    del name
    from eyvan.capacity.drivers.aws import AwsEc2CapacityDriver
    from eyvan.capacity.providers.aws.api import HttpEc2Api
    from eyvan.capacity.providers.aws.signing import AwsCredentials

    if not settings.aws_region:
        raise ValueError("the aws-ec2 capacity backend requires EYVAN_AWS_REGION")
    if not (settings.aws_access_key_id and settings.aws_secret_access_key):
        raise ValueError(
            "the aws-ec2 capacity backend requires EYVAN_AWS_ACCESS_KEY_ID and EYVAN_AWS_SECRET_ACCESS_KEY. "
            "Instance-role credentials are not read yet, so a plane relying on one must supply them explicitly"
        )
    return AwsEc2CapacityDriver(
        state,
        api=HttpEc2Api(
            region=settings.aws_region,
            credentials=AwsCredentials(
                access_key_id=settings.aws_access_key_id,
                secret_access_key=settings.aws_secret_access_key,
                session_token=settings.aws_session_token or None,
            ),
        ),
    )


def _azure_factory(name: str, *, state: FleetStateStore, settings: EyvanSettings) -> CapacityDriver:
    """Build the Azure VM driver, refusing to guess at a subscription.

    Refused loudly rather than defaulted, for the reason the AWS factory gives:
    a driver that picked a subscription would create real machines somewhere
    nobody chose.
    """

    del name
    from eyvan.capacity.drivers.azure import AzureVmCapacityDriver
    from eyvan.capacity.providers.azure.api import HttpAzureApi

    missing = [
        variable
        for variable, value in (
            ("EYVAN_AZURE_SUBSCRIPTION_ID", settings.azure_subscription_id),
            ("EYVAN_AZURE_RESOURCE_GROUP", settings.azure_resource_group),
            ("EYVAN_AZURE_LOCATION", settings.azure_location),
            ("EYVAN_AZURE_ACCESS_TOKEN", settings.azure_access_token),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"the azure-vm capacity backend requires {', '.join(missing)}")
    return AzureVmCapacityDriver(
        state,
        api=HttpAzureApi(
            subscription_id=settings.azure_subscription_id,
            resource_group=settings.azure_resource_group,
            location=settings.azure_location,
            # A callable rather than a value, so a rotating credential is read
            # at call time. A token captured at construction works for exactly
            # one lifetime and then fails as an unrelated permissions problem.
            token=lambda: settings.azure_access_token,
        ),
        location=settings.azure_location,
    )


def _gcp_factory(name: str, *, state: FleetStateStore, settings: EyvanSettings) -> CapacityDriver:
    from eyvan.capacity.drivers.gcp import GcpCapacityDriver
    from eyvan.capacity.providers.gcp.native_worker_ingress import GcloudNativeWorkerPublicIngress

    native_worker_public_ingress = (
        GcloudNativeWorkerPublicIngress(network_tag=settings.native_worker_public_ingress_tag)
        if settings.native_worker_public_ingress_tag is not None
        else None
    )
    return GcpCapacityDriver(
        state,
        execution_backend=settings.execution_backend,
        mode=name,
        native_worker_port=settings.native_worker_port,
        native_worker_package=settings.native_worker_package,
        native_worker_package_sha256=settings.native_worker_package_sha256,
        native_worker_package_generation=settings.native_worker_package_generation,
        native_worker_wheelhouse=settings.native_worker_wheelhouse,
        native_worker_wheelhouse_sha256=settings.native_worker_wheelhouse_sha256,
        native_worker_wheelhouse_generation=settings.native_worker_wheelhouse_generation,
        native_worker_state_dir=settings.native_worker_state_dir,
        native_worker_shared_secret=settings.native_worker_root_secret,
        native_worker_artifact_root_uri=settings.native_worker_artifact_root_uri,
        native_worker_public_ingress=native_worker_public_ingress,
    )


class ProviderEntryPoint(tp.Protocol):
    """The slice of :class:`importlib.metadata.EntryPoint` this module uses."""

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> str: ...

    def load(self) -> object: ...


def _advertised_entry_points() -> tuple[ProviderEntryPoint, ...]:
    """Return every entry point advertised under the provider group."""

    from importlib.metadata import entry_points

    return tuple(entry_points(group=CAPACITY_PROVIDER_ENTRY_POINT_GROUP))


def _parse_allowlist(allowlist: str | tp.Iterable[str] | None) -> tuple[frozenset[str], bool]:
    """Return the trusted entry-point names and whether everything is trusted."""

    if allowlist is None:
        return frozenset(), False
    names = allowlist.split(",") if isinstance(allowlist, str) else list(allowlist)
    trusted = {name.strip() for name in names if name.strip()}
    return frozenset(trusted - {"*"}), "*" in trusted


def _plugin_backend_names(plugin: object) -> tuple[str, ...]:
    """Validate and normalize the backend names a plugin claims.

    Raises:
        CapacityProviderPluginError: If the declaration is missing, malformed,
            or empty.
    """

    declared = getattr(plugin, "backend_names", None)
    if isinstance(declared, str) or not isinstance(declared, tp.Sequence):
        raise CapacityProviderPluginError(
            "capacity provider backend_names must be a sequence of names, not a bare string",
            refusal=ProviderPluginRefusal.CONTRACT,
        )
    names: list[str] = []
    for entry in declared:
        if not isinstance(entry, str) or not _normalize(entry):
            raise CapacityProviderPluginError(
                "every capacity provider backend name must be a non-empty string",
                refusal=ProviderPluginRefusal.CONTRACT,
            )
        names.append(_normalize(entry))
    if not names:
        raise CapacityProviderPluginError(
            "a capacity provider plugin must declare at least one backend name",
            refusal=ProviderPluginRefusal.CONTRACT,
        )
    if len(set(names)) != len(names):
        raise CapacityProviderPluginError(
            "a capacity provider plugin declared the same backend name twice",
            refusal=ProviderPluginRefusal.CONTRACT,
        )
    return tuple(names)


def validate_provider_plugin(
    plugin: object,
    *,
    name: str = "",
    value: str = "",
    host: ProviderAbiVersion = HOST_PROVIDER_ABI,
) -> LoadedProviderPlugin:
    """Check one resolved plugin object against the published ABI.

    An external provider is expected to call this in its own test suite: it is
    the same code path the registry runs, so "my plugin loads" is a fact rather
    than a hope.

    Args:
        plugin: The object an entry point resolved to -- a module, an instance,
            anything exposing ``abi_version``, ``backend_names`` and ``create``.
        name: The advertised entry-point name, for diagnostics.
        value: The advertised entry-point target, for diagnostics.
        host: The ABI this build implements. Injectable so the version contract
            itself is testable without shipping a second Eyvan.

    Returns:
        The accepted plugin, with its parsed version and normalized names.

    Raises:
        CapacityProviderPluginError: On any contract or version violation. The
            version check happens **before** anything else is trusted, because
            a mismatched provider's other attributes mean whatever its ABI said
            they meant, not what this one does.
    """

    declared = getattr(plugin, "abi_version", None)
    if declared is None:
        raise CapacityProviderPluginError(
            "a capacity provider plugin must declare abi_version",
            refusal=ProviderPluginRefusal.CONTRACT,
        )
    version = ProviderAbiVersion.parse(declared)
    refusal = version.refusal_against(host)
    if refusal is not None:
        raise CapacityProviderPluginError(refusal, refusal=ProviderPluginRefusal.ABI_VERSION)
    names = _plugin_backend_names(plugin)
    create = getattr(plugin, "create", None)
    if not callable(create):
        raise CapacityProviderPluginError(
            "a capacity provider plugin must expose a callable create(name, *, state, settings)",
            refusal=ProviderPluginRefusal.CONTRACT,
        )
    return LoadedProviderPlugin(
        name=name or getattr(plugin, "__name__", "") or type(plugin).__name__,
        value=value,
        abi_version=version,
        backend_names=names,
        plugin=tp.cast(CapacityProviderPlugin, plugin),
    )


def discover_provider_plugins(
    *,
    allowlist: str | tp.Iterable[str] | None,
    entry_points: tp.Sequence[ProviderEntryPoint] | None = None,
    reserved_names: tp.Container[str] = (),
    host: ProviderAbiVersion = HOST_PROVIDER_ABI,
) -> ProviderPluginDiscovery:
    """Resolve advertised capacity providers the operator has trusted.

    Every outcome is reported. An entry point that raises on import becomes a
    :class:`ProviderPluginFailure` with the exception text and an ``ERROR`` log
    line; it is never dropped, and it never degrades into "that backend does not
    exist". The distinction matters because those two produce opposite operator
    actions: fix your install, versus fix your config.

    Args:
        allowlist: Entry-point names to trust, or ``"*"`` for all of them.
            Empty or ``None`` discovers nothing.
        entry_points: Advertised entry points, defaulting to the installed
            distributions'.
        reserved_names: Backend names a plugin may not claim -- the built-ins.
        host: The ABI this build implements.

    Returns:
        The accepted plugins and every refusal, in a stable order.
    """

    trusted, trust_all = _parse_allowlist(allowlist)
    if not trusted and not trust_all:
        return ProviderPluginDiscovery()
    advertised = _advertised_entry_points() if entry_points is None else tuple(entry_points)
    candidates = sorted(
        (entry for entry in advertised if trust_all or entry.name in trusted),
        key=lambda entry: (entry.name, entry.value),
    )
    plugins: list[LoadedProviderPlugin] = []
    failures: list[ProviderPluginFailure] = []
    claimed: dict[str, str] = {}
    for entry in candidates:
        try:
            resolved = entry.load()
        except BaseException as exc:  # a plugin may raise anything while it is imported
            failures.append(
                ProviderPluginFailure(
                    name=entry.name,
                    value=entry.value,
                    refusal=ProviderPluginRefusal.IMPORT,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            logger.error("capacity provider plugin %r failed to import: %s: %s", entry.name, type(exc).__name__, exc)
            continue
        try:
            accepted = validate_provider_plugin(resolved, name=entry.name, value=entry.value, host=host)
        except CapacityProviderPluginError as exc:
            failures.append(
                ProviderPluginFailure(
                    name=entry.name,
                    value=entry.value,
                    refusal=exc.refusal,
                    reason=str(exc),
                )
            )
            logger.error("capacity provider plugin %r was refused: %s", entry.name, exc)
            continue
        conflicts = [name for name in accepted.backend_names if name in reserved_names or name in claimed]
        if conflicts:
            # Refused rather than resolved by order. Whichever way "first wins"
            # falls, the fleet's backend would depend on installation order,
            # and a name that already means something in-tree would silently
            # come to mean something else.
            reason = f"backend name(s) {', '.join(sorted(conflicts))} are already served"
            failures.append(
                ProviderPluginFailure(
                    name=entry.name,
                    value=entry.value,
                    refusal=ProviderPluginRefusal.CONFLICT,
                    reason=reason,
                )
            )
            logger.error("capacity provider plugin %r was refused: %s", entry.name, reason)
            continue
        for name in accepted.backend_names:
            claimed[name] = entry.name
        plugins.append(accepted)
    missing = sorted(name for name in trusted if not any(entry.name == name for entry in candidates))
    for name in missing:
        # An operator named a provider that nothing advertises. Silence here
        # reads as "that backend does not exist", which sends them to the wrong
        # file; this says the install is what is missing.
        failures.append(
            ProviderPluginFailure(
                name=name,
                value="",
                refusal=ProviderPluginRefusal.IMPORT,
                reason=(
                    f"no distribution advertises {name!r} under the "
                    f"{CAPACITY_PROVIDER_ENTRY_POINT_GROUP!r} entry-point group"
                ),
            )
        )
        logger.error("capacity provider plugin %r is trusted but not installed", name)
    return ProviderPluginDiscovery(plugins=tuple(plugins), failures=tuple(failures))


class CapacityBackendRegistry:
    """Name-to-factory registry for capacity backends."""

    def __init__(self) -> None:
        self._factories: dict[str, CapacityBackendFactory] = {}
        self._plugins: dict[str, LoadedProviderPlugin] = {}
        self._discovery: ProviderPluginDiscovery | None = None
        self._discovered_allowlist: str | None = None

    def register(self, name: str, factory: CapacityBackendFactory, *, replace: bool = False) -> None:
        """Register one capacity backend under a normalized name."""

        key = _normalize(name)
        if not key:
            raise ValueError("capacity backend name must be non-empty")
        if not replace and key in self._factories:
            raise ValueError(f"capacity backend {key!r} is already registered")
        self._factories[key] = factory

    def load_plugins(
        self,
        *,
        allowlist: str | tp.Iterable[str] | None,
        entry_points: tp.Sequence[ProviderEntryPoint] | None = None,
        host: ProviderAbiVersion = HOST_PROVIDER_ABI,
    ) -> ProviderPluginDiscovery:
        """Discover trusted out-of-tree providers and register the accepted ones.

        Built-ins are ``reserved_names``, so a plugin can extend this registry
        but never redefine ``gcp`` or ``kubernetes`` underneath an operator.

        Args:
            allowlist: Entry-point names to trust, or ``"*"``.
            entry_points: Advertised entry points, defaulting to the installed
                distributions'.
            host: The ABI this build implements.

        Returns:
            The discovery result, including refusals.
        """

        discovery = discover_provider_plugins(
            allowlist=allowlist,
            entry_points=entry_points,
            reserved_names=frozenset(self._factories),
            host=host,
        )
        self._plugins = {name: plugin for plugin in discovery.plugins for name in plugin.backend_names}
        self._discovery = discovery
        return discovery

    def plugin_discovery(self) -> ProviderPluginDiscovery:
        """Return the last discovery result, empty when none has run."""

        return self._discovery or ProviderPluginDiscovery()

    def _resolve_plugins(self) -> ProviderPluginDiscovery:
        """Return the discovery to resolve an unknown backend against.

        An explicit :meth:`load_plugins` always counts. The environment
        allowlist only drives *implicit* discovery, and it runs at most once per
        distinct allowlist value so a reconcile loop does not rescan every
        installed distribution.
        """

        allowlist = os.getenv(CAPACITY_PROVIDER_PLUGIN_ENV, "").strip()
        if not allowlist:
            # The shipped default. Nothing is scanned, nothing is imported, and
            # an installed distribution cannot change what this registry serves.
            return self.plugin_discovery()
        if self._discovery is None or self._discovered_allowlist != allowlist:
            self._discovered_allowlist = allowlist
            return self.load_plugins(allowlist=allowlist)
        return self._discovery

    def create(self, name: str, *, state: FleetStateStore, settings: EyvanSettings) -> CapacityDriver:
        """Build the driver registered under ``name``.

        Built-in backends resolve without importing anything a plugin ships. An
        unknown name consults the operator-trusted plugins, and a name that is
        unknown *because a plugin was refused* reports the refusal rather than
        claiming the backend does not exist.

        Args:
            name: The requested capacity backend.
            state: The durable fleet-state store handed to the driver.
            settings: The agent settings handed to the driver.

        Returns:
            The constructed driver.

        Raises:
            ValueError: If no built-in and no trusted plugin serves ``name``.
        """

        key = _normalize(name)
        factory = self._factories.get(key)
        if factory is not None:
            return factory(key, state=state, settings=settings)
        discovery = self._resolve_plugins()
        plugin = self._plugins.get(key)
        if plugin is not None:
            return plugin.plugin.create(key, state=state, settings=settings)
        message = f"unknown Eyvan capacity backend {name!r}; choose one of: {', '.join(self.names())}"
        if discovery.failures:
            detail = "; ".join(failure.describe() for failure in discovery.failures)
            message = f"{message}. {len(discovery.failures)} provider plugin(s) were refused: {detail}"
        raise ValueError(message)

    def names(self) -> tuple[str, ...]:
        """Return every registered backend name in stable order.

        Includes plugin-served names only once a discovery pass has accepted
        them, so a deployment that never enabled plugins reports exactly the
        built-ins it did before the ABI existed.
        """

        return tuple(sorted({*self._factories, *self._plugins}))


DEFAULT_CAPACITY_BACKENDS = CapacityBackendRegistry()
DEFAULT_CAPACITY_BACKENDS.register("local", _local_factory)
DEFAULT_CAPACITY_BACKENDS.register("manual", _manual_factory)
DEFAULT_CAPACITY_BACKENDS.register("gcp", _gcp_factory)
DEFAULT_CAPACITY_BACKENDS.register("gcp-tpu", _gcp_factory)
DEFAULT_CAPACITY_BACKENDS.register("gcp-attach", _gcp_factory)
DEFAULT_CAPACITY_BACKENDS.register("gcp-compute", _gcp_factory)
DEFAULT_CAPACITY_BACKENDS.register("kubernetes", _kubernetes_factory)
DEFAULT_CAPACITY_BACKENDS.register("k8s", _kubernetes_factory)
DEFAULT_CAPACITY_BACKENDS.register("onprem-ssh", _onprem_ssh_factory)
DEFAULT_CAPACITY_BACKENDS.register("aws-ec2", _aws_factory)
DEFAULT_CAPACITY_BACKENDS.register("aws", _aws_factory)
DEFAULT_CAPACITY_BACKENDS.register("azure-vm", _azure_factory)
DEFAULT_CAPACITY_BACKENDS.register("azure", _azure_factory)


__all__ = (
    "CAPACITY_PROVIDER_ABI_VERSION",
    "CAPACITY_PROVIDER_ENTRY_POINT_GROUP",
    "CAPACITY_PROVIDER_PLUGIN_ENV",
    "DEFAULT_CAPACITY_BACKENDS",
    "HOST_PROVIDER_ABI",
    "CapacityBackendFactory",
    "CapacityBackendRegistry",
    "CapacityProviderPluginError",
    "LoadedProviderPlugin",
    "ProviderAbiVersion",
    "ProviderEntryPoint",
    "ProviderPluginDiscovery",
    "ProviderPluginFailure",
    "ProviderPluginRefusal",
    "discover_provider_plugins",
    "validate_provider_plugin",
)
