# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The capacity provider entry-point ABI: what loads, what refuses, and what says so.

Every plugin here is a real module written to disk and resolved through a real
:class:`importlib.metadata.EntryPoint`, so ``load()`` really imports it. One test
goes further and installs a real ``.dist-info`` on ``sys.path`` so the discovery
path an operator actually uses -- ``importlib.metadata.entry_points`` -- is the
one under test, rather than a fake list handed to the function.
"""

from __future__ import annotations

import importlib
import logging
import sys
import typing as tp
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from eyvan.capacity.drivers.onprem_ssh import OnPremSshCapacityDriver
from eyvan.capacity.drivers.registry import (
    CAPACITY_PROVIDER_ABI_VERSION,
    CAPACITY_PROVIDER_ENTRY_POINT_GROUP,
    CAPACITY_PROVIDER_PLUGIN_ENV,
    DEFAULT_CAPACITY_BACKENDS,
    CapacityBackendRegistry,
    CapacityProviderPluginError,
    ProviderAbiVersion,
    ProviderPluginRefusal,
    discover_provider_plugins,
    validate_provider_plugin,
)
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings

GROUP = CAPACITY_PROVIDER_ENTRY_POINT_GROUP

#: The smallest legal plugin: a module, two constants, one function, and not one
#: import from Eyvan. ``docs/providers.md`` publishes this shape.
PLUGIN_SOURCE = '''
abi_version = "{version}"
backend_names = {names}


class {driver}:
    """A driver stub that only has to be the object the registry returns."""

    provider = "{driver}"


def create(name, *, state, settings):
    """Build the driver for one of ``backend_names``."""

    return {driver}()
'''

IMPORT_TIME_FAILURE = """
raise RuntimeError("acme provider needs the acme SDK")
"""

SENTINEL_SOURCE = '''
import pathlib

pathlib.Path({sentinel!r}).write_text("imported")

abi_version = "1.0"
backend_names = ("sentinel",)


def create(name, *, state, settings):
    """Never reached in these tests."""

    raise AssertionError("this plugin must not be built")
'''

PLUGIN_INSTANCE_SOURCE = '''


class _Plugin:
    """The instance form of the same plugin."""

    abi_version = "{version}"
    backend_names = ("acme",)

    def create(self, name, *, state, settings):
        """Build the driver."""

        return AcmeDriver()


plugin = _Plugin()
'''


@pytest.fixture
def plugin_site(tmp_path: Path) -> tp.Iterator[tp.Callable[..., EntryPoint]]:
    """Write importable plugin modules and resolve them through real entry points."""

    site = tmp_path / "site"
    site.mkdir()
    sys.path.insert(0, str(site))
    written: list[str] = []

    def publish(module: str, source: str, *, name: str = "acme", target: str | None = None) -> EntryPoint:
        (site / f"{module}.py").write_text(source)
        written.append(module)
        importlib.invalidate_caches()
        return EntryPoint(name=name, value=target or module, group=GROUP)

    try:
        yield publish
    finally:
        sys.path.remove(str(site))
        for module in written:
            sys.modules.pop(module, None)


def _plugin(version: str = CAPACITY_PROVIDER_ABI_VERSION, names: str = '("acme",)', driver: str = "AcmeDriver") -> str:
    return PLUGIN_SOURCE.format(version=version, names=names, driver=driver)


def _state(tmp_path: Path) -> FleetStateStore:
    return FleetStateStore(LocalBackend(tmp_path / "fleet.json"))


# --------------------------------------------------------------------------- the version contract


def test_a_matching_provider_loads_and_serves_its_backend(plugin_site, tmp_path) -> None:
    """The whole point: a distribution Eyvan does not ship becomes a backend."""

    entry = plugin_site("eyvan_acme_provider", _plugin())
    registry = CapacityBackendRegistry()

    discovery = registry.load_plugins(allowlist="acme", entry_points=[entry])

    assert discovery.failures == ()
    assert discovery.plugins[0].backend_names == ("acme",)
    assert registry.names() == ("acme",)
    driver = registry.create("acme", state=_state(tmp_path), settings=EyvanSettings())
    assert type(driver).__name__ == "AcmeDriver"


@pytest.mark.parametrize(
    ("declared", "because"),
    (
        ("2.0", "a major version difference withdraws or changes a guarantee"),
        ("0.9", "a major version difference withdraws or changes a guarantee"),
        ("1.7", "the provider requires guarantees this build does not make yet"),
    ),
)
def test_a_version_mismatch_refuses_to_load_and_never_adapts(plugin_site, tmp_path, declared, because) -> None:
    """A provider written against other promises is refused, not accommodated."""

    entry = plugin_site("eyvan_acme_provider", _plugin(version=declared))
    registry = CapacityBackendRegistry()

    discovery = registry.load_plugins(allowlist="acme", entry_points=[entry])

    assert discovery.plugins == ()
    assert discovery.failures[0].refusal is ProviderPluginRefusal.ABI_VERSION
    assert because in discovery.failures[0].reason
    assert declared in discovery.failures[0].reason and CAPACITY_PROVIDER_ABI_VERSION in discovery.failures[0].reason
    assert "acme" not in registry.names()
    with pytest.raises(ValueError, match="unknown Eyvan capacity backend"):
        registry.create("acme", state=_state(tmp_path), settings=EyvanSettings())


def test_an_older_minor_version_still_loads(plugin_site) -> None:
    """Within a major version Eyvan only ever adds guarantees, so old is fine."""

    entry = plugin_site("eyvan_acme_provider", _plugin(version="1.0"))

    discovery = discover_provider_plugins(
        allowlist="acme",
        entry_points=[entry],
        host=ProviderAbiVersion(major=1, minor=4),
    )

    assert discovery.failures == ()
    assert discovery.plugins[0].abi_version == ProviderAbiVersion(major=1, minor=0)


@pytest.mark.parametrize("declared", ('"1"', '"one.zero"', "1.0", "None", '"1.0.3"'))
def test_an_unparseable_abi_version_is_refused_rather_than_guessed(plugin_site, declared) -> None:
    """Guessing a version means loading a provider whose promises are unknown."""

    source = _plugin().replace(f'abi_version = "{CAPACITY_PROVIDER_ABI_VERSION}"', f"abi_version = {declared}")
    entry = plugin_site("eyvan_acme_provider", source)

    discovery = discover_provider_plugins(allowlist="acme", entry_points=[entry])

    assert discovery.plugins == ()
    assert discovery.failures[0].refusal is ProviderPluginRefusal.CONTRACT


# --------------------------------------------------------------------------- failures are reported


def test_an_entry_point_that_raises_on_import_is_reported_not_swallowed(plugin_site, tmp_path, caplog) -> None:
    """The refusal must reach the log, the report, and the operator's error."""

    entry = plugin_site("eyvan_acme_provider", IMPORT_TIME_FAILURE)
    registry = CapacityBackendRegistry()

    with caplog.at_level(logging.ERROR, logger="eyvan.capacity.drivers.registry"):
        discovery = registry.load_plugins(allowlist="acme", entry_points=[entry])

    assert discovery.plugins == ()
    failure = discovery.failures[0]
    assert failure.refusal is ProviderPluginRefusal.IMPORT
    assert failure.reason == "RuntimeError: acme provider needs the acme SDK"
    assert "acme provider needs the acme SDK" in caplog.text
    with pytest.raises(ValueError, match="acme provider needs the acme SDK"):
        registry.create("acme", state=_state(tmp_path), settings=EyvanSettings())


def test_require_raises_with_every_refusal_at_once(plugin_site) -> None:
    """A caller that would rather not start gets one error naming all of them."""

    broken = plugin_site("eyvan_acme_provider", IMPORT_TIME_FAILURE, name="acme")
    mismatched = plugin_site("eyvan_zeta_provider", _plugin(version="2.0"), name="zeta")

    discovery = discover_provider_plugins(allowlist="acme,zeta", entry_points=[broken, mismatched])

    with pytest.raises(CapacityProviderPluginError) as raised:
        discovery.require()
    assert "2 capacity provider plugin(s) were refused" in str(raised.value)
    assert "acme" in str(raised.value) and "zeta" in str(raised.value)


def test_a_trusted_provider_that_is_not_installed_is_reported(plugin_site) -> None:
    """Silence would read as "no such backend" and send the operator to the wrong file."""

    entry = plugin_site("eyvan_acme_provider", _plugin())

    discovery = discover_provider_plugins(allowlist="acme,coreweave", entry_points=[entry])

    assert discovery.plugins[0].name == "acme"
    assert discovery.failures[0].name == "coreweave"
    assert "no distribution advertises" in discovery.failures[0].reason


@pytest.mark.parametrize(
    "mutation",
    (
        "del abi_version",
        "del create",
        "backend_names = ()",
        'backend_names = "acme"',
        'backend_names = ("acme", "acme")',
        "backend_names = (1,)",
        "create = 4",
    ),
)
def test_a_plugin_that_does_not_satisfy_the_contract_is_refused(plugin_site, mutation) -> None:
    """The three published members are the whole ABI, and all three are checked."""

    entry = plugin_site("eyvan_acme_provider", _plugin() + f"\n{mutation}\n")

    discovery = discover_provider_plugins(allowlist="acme", entry_points=[entry])

    assert discovery.plugins == ()
    assert discovery.failures[0].refusal is ProviderPluginRefusal.CONTRACT


# --------------------------------------------------------------------------- names stay owned


def test_a_plugin_cannot_shadow_a_builtin_backend(plugin_site, tmp_path) -> None:
    """A backend name that already means something must keep meaning it."""

    entry = plugin_site("eyvan_acme_provider", _plugin(names='("onprem-ssh",)'))
    registry = CapacityBackendRegistry()
    registry.register("onprem-ssh", lambda name, *, state, settings: OnPremSshCapacityDriver(state))

    discovery = registry.load_plugins(allowlist="acme", entry_points=[entry])

    assert discovery.failures[0].refusal is ProviderPluginRefusal.CONFLICT
    driver = registry.create("onprem-ssh", state=_state(tmp_path), settings=EyvanSettings())
    assert isinstance(driver, OnPremSshCapacityDriver)


def test_two_plugins_claiming_one_name_are_both_refused(plugin_site) -> None:
    """Resolving this by installation order is how a fleet changes substrate silently."""

    first = plugin_site("eyvan_acme_provider", _plugin(names='("shared",)'), name="acme")
    second = plugin_site("eyvan_zeta_provider", _plugin(names='("shared",)', driver="ZetaDriver"), name="zeta")

    discovery = discover_provider_plugins(allowlist="*", entry_points=[second, first])

    assert [plugin.name for plugin in discovery.plugins] == ["acme"]
    assert discovery.failures[0].name == "zeta"
    assert discovery.failures[0].refusal is ProviderPluginRefusal.CONFLICT


def test_discovery_order_does_not_depend_on_installation_order(plugin_site) -> None:
    """The same set of distributions must produce the same fleet, always."""

    acme = plugin_site("eyvan_acme_provider", _plugin(), name="acme")
    zeta = plugin_site("eyvan_zeta_provider", _plugin(names='("zeta",)', driver="ZetaDriver"), name="zeta")

    forward = discover_provider_plugins(allowlist="*", entry_points=[acme, zeta])
    backward = discover_provider_plugins(allowlist="*", entry_points=[zeta, acme])

    assert [plugin.name for plugin in forward.plugins] == ["acme", "zeta"]
    assert [plugin.name for plugin in backward.plugins] == ["acme", "zeta"]


# --------------------------------------------------------------------------- opt-in discovery


def test_discovery_is_off_by_default_and_imports_nothing(plugin_site, tmp_path, monkeypatch) -> None:
    """Today's behaviour: an installed distribution changes nothing by itself."""

    sentinel = tmp_path / "imported.txt"
    plugin_site("eyvan_sentinel_provider", SENTINEL_SOURCE.format(sentinel=str(sentinel)), name="sentinel")
    monkeypatch.delenv(CAPACITY_PROVIDER_PLUGIN_ENV, raising=False)

    with pytest.raises(ValueError, match="unknown Eyvan capacity backend"):
        DEFAULT_CAPACITY_BACKENDS.create("sentinel", state=_state(tmp_path), settings=EyvanSettings())

    assert not sentinel.exists(), "a disabled plugin must not be imported"
    assert DEFAULT_CAPACITY_BACKENDS.names() == (
        "aws",
        "aws-ec2",
        "azure",
        "azure-vm",
        "gcp",
        "gcp-attach",
        "gcp-compute",
        "gcp-tpu",
        "k8s",
        "kubernetes",
        "local",
        "manual",
        "onprem-ssh",
    )


def test_an_untrusted_plugin_is_never_imported(plugin_site, tmp_path) -> None:
    """The allowlist is a supply-chain boundary, not a filter over loaded code."""

    sentinel = tmp_path / "imported.txt"
    untrusted = plugin_site("eyvan_sentinel_provider", SENTINEL_SOURCE.format(sentinel=str(sentinel)), name="sentinel")
    trusted = plugin_site("eyvan_acme_provider", _plugin(), name="acme")

    discovery = discover_provider_plugins(allowlist="acme", entry_points=[untrusted, trusted])

    assert [plugin.name for plugin in discovery.plugins] == ["acme"]
    assert not sentinel.exists()


def test_the_environment_allowlist_enables_a_plugin_through_create(plugin_site, tmp_path, monkeypatch) -> None:
    """The operator-facing path: name it, and the backend exists."""

    entry = plugin_site("eyvan_acme_provider", _plugin())
    registry = CapacityBackendRegistry()
    monkeypatch.setenv(CAPACITY_PROVIDER_PLUGIN_ENV, "acme")
    monkeypatch.setattr("eyvan.capacity.drivers.registry._advertised_entry_points", lambda: (entry,))

    driver = registry.create("acme", state=_state(tmp_path), settings=EyvanSettings())

    assert type(driver).__name__ == "AcmeDriver"
    assert registry.plugin_discovery().plugins[0].name == "acme"


def test_a_star_allowlist_loads_every_advertised_provider(plugin_site) -> None:
    """Documented as trading the supply-chain guarantee away, and it must work."""

    acme = plugin_site("eyvan_acme_provider", _plugin(), name="acme")
    zeta = plugin_site("eyvan_zeta_provider", _plugin(names='("zeta",)', driver="ZetaDriver"), name="zeta")

    discovery = discover_provider_plugins(allowlist="*", entry_points=[acme, zeta])

    assert {name for plugin in discovery.plugins for name in plugin.backend_names} == {"acme", "zeta"}


# --------------------------------------------------------------------------- real distribution metadata


def test_an_installed_distribution_is_discovered_through_importlib_metadata(tmp_path, monkeypatch) -> None:
    """The path an operator's install actually takes, with no fake entry points."""

    site = tmp_path / "site-packages"
    dist = site / "eyvan_acme_provider-1.0.dist-info"
    dist.mkdir(parents=True)
    (site / "eyvan_acme_provider.py").write_text(_plugin())
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: eyvan-acme-provider\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(f"[{GROUP}]\nacme = eyvan_acme_provider\n")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    try:
        discovery = discover_provider_plugins(allowlist="acme")
    finally:
        sys.modules.pop("eyvan_acme_provider", None)

    assert discovery.failures == ()
    assert discovery.plugins[0].value == "eyvan_acme_provider"
    assert discovery.plugins[0].backend_names == ("acme",)


def test_an_entry_point_may_name_an_object_inside_a_module(plugin_site) -> None:
    """A plugin instance is as legal as a plugin module; both are published."""

    source = _plugin() + PLUGIN_INSTANCE_SOURCE.format(version=CAPACITY_PROVIDER_ABI_VERSION)
    entry = plugin_site("eyvan_acme_provider", source, target="eyvan_acme_provider:plugin")

    discovery = discover_provider_plugins(allowlist="acme", entry_points=[entry])

    assert discovery.failures == ()
    assert discovery.plugins[0].backend_names == ("acme",)


def test_validate_provider_plugin_is_the_same_gate_a_provider_can_run_itself() -> None:
    """An external provider must be able to prove it loads without installing itself."""

    class Plugin:
        abi_version = CAPACITY_PROVIDER_ABI_VERSION
        backend_names = ("Acme_Cloud",)

        def create(self, name, *, state, settings):
            """Build nothing; this test is about the gate."""

            raise NotImplementedError

    accepted = validate_provider_plugin(Plugin())

    assert accepted.backend_names == ("acme-cloud",), "names are normalized exactly as the registry normalizes them"
    with pytest.raises(CapacityProviderPluginError):
        validate_provider_plugin(Plugin(), host=ProviderAbiVersion(major=99, minor=0))
