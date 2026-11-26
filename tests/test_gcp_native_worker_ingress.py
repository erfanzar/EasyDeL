# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage for profile-managed local public ingress to native GCP workers."""

from __future__ import annotations

import ipaddress
import json
import subprocess
import typing as tp
from dataclasses import dataclass, field

import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.capacity.providers.gcp.native_worker_ingress import (
    FirewallNetworkReference,
    FirewallRuleConflictError,
    FirewallRuleSpec,
    GcloudNativeWorkerPublicIngress,
    HttpPublicIpv4Resolver,
    PublicIpv4ResolutionError,
    list_managed_ingress_rules,
    release_managed_ingress_rule,
)

_TAG = "eyvan-nw-0123456789abcdef01234567"


def _result(returncode: int = 0, *, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["gcloud"], returncode, stdout, stderr)


@dataclass(frozen=True, slots=True)
class StaticPublicIpv4Resolver:
    """Return one known address without calling a public discovery service."""

    address: ipaddress.IPv4Address

    def resolve(self) -> ipaddress.IPv4Address:
        """Return the configured address."""

        return self.address


@dataclass(slots=True)
class FirewallRunner:
    """In-memory strict gcloud firewall-rule surface for unit tests."""

    rules: dict[str, JsonObject] = field(default_factory=dict)
    calls: list[tuple[str, ...]] = field(default_factory=list)
    create_race: bool = False

    def __call__(self, args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Handle list/describe/create/update/delete commands emitted by the manager."""

        command = tuple(args)
        self.calls.append(command)
        operation = command[2]
        if operation == "list":
            # gcloud's own --filter narrows the transfer; it is not the
            # ownership decision. This fake deliberately ignores it and returns
            # every rule, so a test cannot pass on the strength of a filter the
            # production code should not be trusting either.
            return _result(stdout=json.dumps(list(self.rules.values())))
        rule_name = command[3]
        if operation == "delete":
            if rule_name not in self.rules:
                return _result(1, stderr="NOT_FOUND")
            del self.rules[rule_name]
            return _result()
        if operation == "describe":
            rule = self.rules.get(rule_name)
            return _result(stdout=json.dumps(rule)) if rule is not None else _result(1, stderr="NOT_FOUND")
        if operation == "create":
            if self.create_race:
                self.create_race = False
                self.rules[rule_name] = self._rule_from_create(command)
                return _result(1, stderr="ALREADY_EXISTS")
            self.rules[rule_name] = self._rule_from_create(command)
            return _result()
        if operation == "update":
            self.rules[rule_name] = self._rule_from_update(command, self.rules[rule_name])
            return _result()
        raise AssertionError(f"unexpected gcloud firewall operation {operation!r}")

    @staticmethod
    def _value(args: tuple[str, ...], flag: str) -> str:
        if flag in args:
            return args[args.index(flag) + 1]
        prefix = f"{flag}="
        for value in args:
            if value.startswith(prefix):
                return value.removeprefix(prefix)
        raise AssertionError(f"missing flag {flag!r}")

    @classmethod
    def _rule_from_create(cls, args: tuple[str, ...]) -> JsonObject:
        project = cls._value(args, "--project")
        network = cls._value(args, "--network")
        return {
            # Real gcloud always returns a name on describe and list, and the
            # release path keys off it, so the fake has to carry one too.
            "name": args[3],
            "description": cls._value(args, "--description"),
            "direction": "INGRESS",
            "network": f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{network}",
            "priority": int(cls._value(args, "--priority")),
            "targetTags": [cls._value(args, "--target-tags")],
            "allowed": [
                {
                    "IPProtocol": "tcp",
                    "ports": [cls._value(args, "--allow").split(":", 1)[1]],
                }
            ],
            "sourceRanges": [cls._value(args, "--source-ranges")],
            "disabled": False,
        }

    @classmethod
    def _rule_from_update(cls, args: tuple[str, ...], existing: JsonObject) -> JsonObject:
        updated = dict(existing)
        updated["description"] = cls._value(args, "--description")
        updated["priority"] = int(cls._value(args, "--priority"))
        updated["targetTags"] = [cls._value(args, "--target-tags")]
        updated["allowed"] = [
            {
                "IPProtocol": "tcp",
                "ports": [cls._value(args, "--allow").split(":", 1)[1]],
            }
        ]
        updated["sourceRanges"] = [cls._value(args, "--source-ranges")]
        updated["disabled"] = False
        return updated


def _manager(runner: FirewallRunner, address: str = "203.0.113.24") -> GcloudNativeWorkerPublicIngress:
    return GcloudNativeWorkerPublicIngress(
        network_tag=_TAG,
        public_ipv4_resolver=StaticPublicIpv4Resolver(ipaddress.IPv4Address(address)),
        runner=runner,
    )


def _existing_rule(*, source_range: str = "198.51.100.8/32") -> JsonObject:
    network = FirewallNetworkReference.from_config(capacity_project="project-a", network="research")
    return {
        "name": _rule_name(),
        "description": f"eyvan-native-worker-ingress:v1:{_TAG}",
        "direction": "INGRESS",
        "network": f"https://www.googleapis.com/compute/v1/{network.canonical_name}",
        "priority": 1000,
        "targetTags": [_TAG],
        "allowed": [{"IPProtocol": "tcp", "ports": ["8100"]}],
        "sourceRanges": [source_range],
        "disabled": False,
    }


def _rule_name(network: str = "research") -> str:
    target = FirewallNetworkReference.from_config(capacity_project="project-a", network=network)
    return FirewallRuleSpec(
        network=target,
        network_tag=_TAG,
        port=8100,
        source_range=ipaddress.IPv4Network("203.0.113.24/32"),
    ).rule_name


def test_creates_a_narrow_rule_for_only_the_current_public_ipv4() -> None:
    runner = FirewallRunner()

    tag = _manager(runner).ensure(project="project-a", network=None, port=8100)

    assert tag == _TAG
    assert [call[2] for call in runner.calls] == ["describe", "create"]
    create = runner.calls[-1]
    assert "--source-ranges=203.0.113.24/32" in create
    assert "--target-tags=eyvan-nw-0123456789abcdef01234567" in create
    assert "--allow=tcp:8100" in create
    assert all("0.0.0.0/0" not in value for value in create)
    assert runner.rules[_rule_name("default")]["sourceRanges"] == ["203.0.113.24/32"]


def test_existing_exact_rule_is_a_read_only_no_op() -> None:
    runner = FirewallRunner()
    manager = _manager(runner)
    runner.rules[_rule_name()] = _existing_rule(source_range="203.0.113.24/32")

    manager.ensure(project="project-a", network="research", port=8100)

    assert [call[2] for call in runner.calls] == ["describe"]


def test_changed_local_ip_updates_only_the_owned_narrow_rule() -> None:
    runner = FirewallRunner()
    manager = _manager(runner)
    runner.rules[_rule_name()] = _existing_rule()

    manager.ensure(project="project-a", network="research", port=8100)

    assert [call[2] for call in runner.calls] == ["describe", "update"]
    update = runner.calls[-1]
    assert "--source-ranges=203.0.113.24/32" in update
    assert all("0.0.0.0/0" not in value for value in update)
    assert runner.rules[_rule_name()]["sourceRanges"] == ["203.0.113.24/32"]


def test_refuses_to_modify_a_foreign_deterministic_rule() -> None:
    runner = FirewallRunner()
    manager = _manager(runner)
    foreign = _existing_rule()
    foreign["description"] = "not-eyvan"
    runner.rules[_rule_name()] = foreign

    with pytest.raises(FirewallRuleConflictError, match="not owned"):
        manager.ensure(project="project-a", network="research", port=8100)

    assert [call[2] for call in runner.calls] == ["describe"]


def test_create_race_recovers_only_after_validating_the_managed_rule() -> None:
    runner = FirewallRunner(create_race=True)

    _manager(runner).ensure(project="project-a", network="research", port=8100)

    assert [call[2] for call in runner.calls] == ["describe", "create", "describe"]


def test_shared_vpc_uses_the_network_host_project_for_the_rule() -> None:
    runner = FirewallRunner()

    _manager(runner).ensure(
        project="project-a",
        network="projects/network-host/global/networks/research",
        port=8100,
    )

    create = runner.calls[-1]
    assert create[create.index("--project") + 1] == "network-host"
    assert create[create.index("--network") + 1] == "research"


def _foreign_rule(name: str = "someone-elses-rule") -> JsonObject:
    """A rule that looks similar but carries no Eyvan ownership marker."""

    return {
        "name": name,
        "description": "allow ssh from the office",
        "direction": "INGRESS",
        "network": "https://www.googleapis.com/compute/v1/projects/project-a/global/networks/research",
        "priority": 1000,
        "targetTags": ["office"],
        "allowed": [{"IPProtocol": "tcp", "ports": ["22"]}],
        "sourceRanges": ["198.51.100.0/32"],
        "disabled": False,
    }


def test_listing_returns_only_rules_this_runtime_owns() -> None:
    runner = FirewallRunner(rules={_rule_name(): _existing_rule(), "someone-elses-rule": _foreign_rule()})

    found = list_managed_ingress_rules(project="project-a", runner=runner)

    assert [rule.name for rule in found] == [_rule_name()]
    assert found[0].network_tag == _TAG
    assert found[0].port == 8100


def test_release_removes_a_rule_this_runtime_created() -> None:
    runner = FirewallRunner()
    _manager(runner).ensure(project="project-a", network="research", port=8100)
    created = _rule_name()
    assert created in runner.rules

    release_managed_ingress_rule(project="project-a", rule_name=created, runner=runner)

    assert created not in runner.rules


def test_release_refuses_a_rule_this_runtime_does_not_own() -> None:
    runner = FirewallRunner(rules={"someone-elses-rule": _foreign_rule()})

    with pytest.raises(FirewallRuleConflictError):
        release_managed_ingress_rule(project="project-a", rule_name="someone-elses-rule", runner=runner)

    # The refusal has to leave the rule in place. A project here is shared with
    # other researchers, so deleting one on a failed ownership check would be
    # somebody else's outage.
    assert "someone-elses-rule" in runner.rules
    assert "delete" not in [call[2] for call in runner.calls]


def test_release_of_an_absent_rule_is_not_an_error() -> None:
    runner = FirewallRunner()

    release_managed_ingress_rule(project="project-a", rule_name=_rule_name(), runner=runner)

    assert runner.rules == {}


def test_ownership_is_rechecked_at_delete_not_taken_from_the_listing() -> None:
    """A name that changes meaning between list and delete must not be deleted."""

    runner = FirewallRunner(rules={_rule_name(): _existing_rule()})
    found = list_managed_ingress_rules(project="project-a", runner=runner)
    assert [rule.name for rule in found] == [_rule_name()]

    # The name now denotes a rule nobody here owns.
    runner.rules[_rule_name()] = _foreign_rule(name=_rule_name())

    with pytest.raises(FirewallRuleConflictError):
        release_managed_ingress_rule(project="project-a", rule_name=_rule_name(), runner=runner)

    assert _rule_name() in runner.rules


def test_create_then_list_then_release_closes_the_leak() -> None:
    """The whole point: a rule provisioning made can now be found and removed."""

    runner = FirewallRunner()
    _manager(runner).ensure(project="project-a", network="research", port=8100)

    found = list_managed_ingress_rules(project="project-a", runner=runner)
    assert len(found) == 1
    release_managed_ingress_rule(project="project-a", rule_name=found[0].name, runner=runner)

    assert list_managed_ingress_rules(project="project-a", runner=runner) == ()


@dataclass(frozen=True, slots=True)
class FailingPublicIpv4Resolver:
    """A resolver that never answers, for the keep-the-existing-rule path."""

    def resolve(self) -> ipaddress.IPv4Address:
        raise PublicIpv4ResolutionError("could not determine the local public IPv4 address")


def test_resolution_failure_keeps_an_existing_managed_rule_instead_of_failing() -> None:
    """Keep an existing ingress rule when public-IP discovery is unavailable rather than strand reachable capacity."""

    runner = FirewallRunner()
    _manager(runner).ensure(project="project-a", network="research", port=8100)
    calls_before = list(runner.calls)

    manager = GcloudNativeWorkerPublicIngress(
        network_tag=_TAG,
        public_ipv4_resolver=FailingPublicIpv4Resolver(),
        runner=runner,
    )

    assert manager.ensure(project="project-a", network="research", port=8100) == _TAG
    # Keeping the rule is one describe; nothing is created or updated.
    new_calls = runner.calls[len(calls_before) :]
    assert [call[2] for call in new_calls] == ["describe"]


def test_resolution_failure_without_an_existing_rule_fails_closed_and_retryable() -> None:
    """No rule plus no address means nothing safe exists to keep."""

    manager = GcloudNativeWorkerPublicIngress(
        network_tag=_TAG,
        public_ipv4_resolver=FailingPublicIpv4Resolver(),
        runner=FirewallRunner(),
    )

    with pytest.raises(PublicIpv4ResolutionError) as caught:
        manager.ensure(project="project-a", network="research", port=8100)

    # "I could not find out" on the operator's own network is retryable and is
    # never charged to the pool as a provider defect.
    assert caught.value.code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
    assert caught.value.retryable is True


def test_resolution_failure_does_not_keep_a_disabled_rule() -> None:
    runner = FirewallRunner()
    _manager(runner).ensure(project="project-a", network="research", port=8100)
    runner.rules[_rule_name()]["disabled"] = True

    manager = GcloudNativeWorkerPublicIngress(
        network_tag=_TAG,
        public_ipv4_resolver=FailingPublicIpv4Resolver(),
        runner=runner,
    )

    with pytest.raises(PublicIpv4ResolutionError):
        manager.ensure(project="project-a", network="research", port=8100)


def test_resolver_walks_every_endpoint_before_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    """One discovery endpoint having a bad minute is not 'could not find out'."""

    import httpx

    tried: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tried.append(request.url.host)
        if request.url.host == "api.ipify.org":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, text="203.0.113.24\n")

    original = httpx.Client

    class _MappedClient(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _MappedClient)

    resolver = HttpPublicIpv4Resolver(endpoints=("https://api.ipify.org", "https://checkip.amazonaws.com"))

    assert str(resolver.resolve()) == "203.0.113.24"
    assert tried == ["api.ipify.org", "checkip.amazonaws.com"]


def test_resolver_records_every_failure_when_all_endpoints_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="try later")

    original = httpx.Client

    class _MappedClient(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _MappedClient)

    resolver = HttpPublicIpv4Resolver(endpoints=("https://api.ipify.org", "https://checkip.amazonaws.com"))

    with pytest.raises(PublicIpv4ResolutionError) as caught:
        resolver.resolve()

    assert len(caught.value.endpoint_failures) == 2


def test_resolver_refuses_plain_http_endpoints() -> None:
    """The answer becomes a firewall source range; it must not be steerable in cleartext."""

    with pytest.raises(ValueError):
        HttpPublicIpv4Resolver(endpoints=("http://api.ipify.org",))
