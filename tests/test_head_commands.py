# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage for `eyvan head up`, the one-command GCP control-plane head.

Nothing here talks to GCP, and nothing here claims the commands were accepted by
GCP. What is proven is everything that sits between the operator and the
provider: which commands are issued and in what order, what is refused, what a
second run does *not* do, and where a generated credential is allowed to appear.

The seam is the injected runner, the same boundary
``capacity/providers/gcp/runtime/bootstrap.py`` uses. Assertions are on the exact
argv, the exact bytes delivered on stdin, and the returned values -- never on
whether a mock was called.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

import eyvan.sdk.cli as cli
import eyvan.sdk.commands.head as head_commands
from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.sdk.commands.head import (
    HEAD_DASHBOARD_DIR,
    HEAD_ENVIRONMENT_PATH,
    HEAD_FLEET_LABEL_KEY,
    HEAD_MANAGED_LABEL_KEY,
    HEAD_MANAGED_LABEL_VALUE,
    HeadCommandError,
    HeadIngressError,
    HeadOwnershipError,
    HeadProvisioner,
    HeadSecrets,
    HeadSpec,
    HeadStep,
    _LocalWheel,
    render_head_bootstrap_script,
    staged_wheel_filename,
)

FLEET_YAML = """
version: 1
name: research
provider:
  type: gcp
  project: example-project-1
  region: us-east5
  zone: us-east5-a
executor: eyvan-native
pools:
  cpu:
    kind: cpu
    machine_type: n2-standard-4
    min_workers: 0
    max_workers: 1
    resources:
      CPU: 4
      MEMORY_GIB: 16
    provider_options:
      bootstrap_egress: private-google-access
"""

#: Deterministic material so a test can assert exactly where a secret may appear.
STUB_SECRETS = HeadSecrets(
    metrics_token="metrics-" + "m" * 32,
    artifact_token_key="artifact-" + "a" * 32,
    agent_shared_secret="agent-" + "g" * 32,
    database_password="database-" + "d" * 24,
    bootstrap_token="bootstrap-" + "b" * 32,
    native_worker_root_secret="native-root-" + "n" * 32,
)


#: Tests must never build or upload a real distribution. The digest is what the
#: fake host will echo back, so the verification path is still exercised.
_FAKE_WHEEL_BYTES = b"PK\x03\x04 not a real wheel"
_FAKE_WHEEL_DIGEST = __import__("hashlib").sha256(_FAKE_WHEEL_BYTES).hexdigest()


def _fake_wheel() -> _LocalWheel:
    """Return a deterministic stand-in for this installation's wheel."""

    return _LocalWheel(data=_FAKE_WHEEL_BYTES, digest=_FAKE_WHEEL_DIGEST, filename=staged_wheel_filename())


def _spec(**overrides: tp.Any) -> HeadSpec:
    """Return one complete head spec for the tests that do not read YAML."""

    values: dict[str, tp.Any] = {
        "fleet": "research",
        "project": "example-project-1",
        "region": "us-east5",
        "zone": "us-east5-a",
        "egress": "private-google-access",
    }
    values.update(overrides)
    return HeadSpec(**values)


def _owned_instance(spec: HeadSpec, *, address: str = "10.128.0.7") -> dict[str, tp.Any]:
    """Return the instance description gcloud would return for an Eyvan head."""

    return {
        "name": spec.instance_name,
        "status": "RUNNING",
        "labels": {
            HEAD_MANAGED_LABEL_KEY: HEAD_MANAGED_LABEL_VALUE,
            HEAD_FLEET_LABEL_KEY: spec.fleet,
        },
        "networkInterfaces": [{"networkIP": address}],
    }


def _owned_firewall(spec: HeadSpec, name: str, *, ranges: tuple[str, ...]) -> dict[str, tp.Any]:
    """Return the rule description gcloud would return for a managed rule."""

    return {
        "name": name,
        "description": spec.firewall_description,
        "sourceRanges": list(ranges),
    }


@dataclass
class Call:
    """One command the provisioner issued, with the bytes it delivered."""

    args: tuple[str, ...]
    stdin_text: str | None

    @property
    def remote_command(self) -> str | None:
        """Return the remote shell command when this call was an SSH call."""

        if "--command" not in self.args:
            return None
        return self.args[self.args.index("--command") + 1]


@dataclass
class FakeGcloud:
    """An in-memory GCP project that answers the exact commands a head issues."""

    spec: HeadSpec
    instance: dict[str, tp.Any] | None = None
    firewall: dict[str, dict[str, tp.Any]] = field(default_factory=dict)
    subnets: tuple[str, ...] = ("10.128.0.0/20",)
    address: str = "10.128.0.7"
    environment_present: bool = False
    #: What GCS reports as the staged worker wheel's object generation.
    object_generation: str = "1786489052345332"
    #: Remote-command or gcloud-verb fragment that should fail, and its result.
    failures: dict[str, subprocess.CompletedProcess[str]] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        stdin_text: str | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        recorded = Call(args=tuple(args), stdin_text=stdin_text)
        self.calls.append(recorded)
        assert timeout > 0
        # A real host decodes the wheel and echoes sha256sum. The digest check is
        # part of the contract, so the double answers it instead of being bypassed.
        remote_command = args[args.index("--command") + 1] if "--command" in args else ""
        if "base64 -d" in remote_command and stdin_text is not None:
            digest = hashlib.sha256(base64.b64decode(stdin_text)).hexdigest()
            return subprocess.CompletedProcess(list(args), 0, f"{digest}\n", "")
        joined = " ".join(args)
        for fragment, result in self.failures.items():
            if fragment in joined:
                return result
        return self._dispatch(recorded)

    # -- helpers ----------------------------------------------------------

    @property
    def issued(self) -> list[tuple[str, ...]]:
        """Return every argv issued, in order."""

        return [call.args for call in self.calls]

    def issued_matching(self, *fragments: str) -> list[tuple[str, ...]]:
        """Return every argv containing all of ``fragments`` as elements."""

        return [args for args in self.issued if all(fragment in args for fragment in fragments)]

    def stdin_for(self, remote_fragment: str) -> str:
        """Return the stdin delivered to the one SSH call matching a fragment."""

        payloads = [
            call.stdin_text
            for call in self.calls
            if call.stdin_text is not None and remote_fragment in (call.remote_command or "")
        ]
        assert len(payloads) == 1, f"expected one call matching {remote_fragment!r}, got {len(payloads)}"
        assert payloads[0] is not None
        return payloads[0]

    # -- dispatch ---------------------------------------------------------

    def _dispatch(self, call: Call) -> subprocess.CompletedProcess[str]:
        args = call.args
        if args[:2] == ("storage", "cp"):
            return _ok(args, "")
        if args[:3] == ("storage", "objects", "describe"):
            return _ok(args, f"{self.object_generation}\n")
        if args[:3] == ("compute", "instances", "describe"):
            if self.instance is None:
                return _missing(args)
            if any(argument.startswith("--format=value(") for argument in args):
                return _ok(args, self.address)
            return _ok(args, json.dumps(self.instance))
        if args[:3] == ("compute", "instances", "create"):
            self.instance = _owned_instance(self.spec, address=self.address)
            return _ok(args, "")
        if args[:3] == ("compute", "instances", "delete"):
            self.instance = None
            return _ok(args, "")
        if args[:4] == ("compute", "networks", "subnets", "list"):
            return _ok(args, "\n".join(self.subnets) + "\n")
        if args[:3] == ("compute", "firewall-rules", "describe"):
            existing = self.firewall.get(args[3])
            return _ok(args, json.dumps(existing)) if existing is not None else _missing(args)
        if args[:3] == ("compute", "firewall-rules", "create"):
            self.firewall[args[3]] = {
                "name": args[3],
                "description": _flag_value(args, "--description="),
                "sourceRanges": _flag_value(args, "--source-ranges=").split(","),
            }
            return _ok(args, "")
        if args[:3] == ("compute", "firewall-rules", "delete"):
            self.firewall.pop(args[3], None)
            return _ok(args, "")
        if args[:2] == ("compute", "ssh"):
            return self._ssh(call)
        raise AssertionError(f"unexpected gcloud command: {args}")

    def _ssh(self, call: Call) -> subprocess.CompletedProcess[str]:
        remote = call.remote_command or ""
        if "eyvan-head-env-present" in remote:
            marker = "eyvan-head-env-present" if self.environment_present else "eyvan-head-env-absent"
            return _ok(call.args, marker + "\n")
        if HEAD_ENVIRONMENT_PATH in remote and "install -D" in remote:
            self.environment_present = True
            return _ok(call.args, "")
        if "is-active" in remote:
            return _ok(call.args, "active\n")
        return _ok(call.args, "")


def _ok(args: tp.Sequence[str], stdout: str) -> subprocess.CompletedProcess[str]:
    """Return a successful process result."""

    return subprocess.CompletedProcess(list(args), 0, stdout=stdout, stderr="")


def _missing(args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Return the not-found result gcloud produces for an absent resource."""

    return subprocess.CompletedProcess(list(args), 1, stdout="", stderr="ERROR: (gcloud) NOT_FOUND: was not found\n")


def _failed(detail: str, *, returncode: int = 1) -> subprocess.CompletedProcess[str]:
    """Return a failing process result with a diagnostic on stderr."""

    return subprocess.CompletedProcess([], returncode, stdout="", stderr=detail)


def _ok_with_status(stdout: str, *, returncode: int) -> subprocess.CompletedProcess[str]:
    """Return a result that prints on stdout and still exits non-zero."""

    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def _flag_value(args: tp.Sequence[str], prefix: str) -> str:
    """Return the value of one ``--flag=value`` argument."""

    for argument in args:
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    raise AssertionError(f"no argument starting with {prefix!r} in {args}")


def _environment(text: str) -> dict[str, str]:
    """Parse a rendered systemd environment file."""

    return dict(line.split("=", 1) for line in text.splitlines() if line)


@pytest.fixture(autouse=True)
def never_reach_a_real_project(monkeypatch) -> None:
    """Make the real gcloud boundary unreachable from this module.

    Not a precaution in the abstract: an earlier version of these tests bound
    the real runner as a default argument, and running the suite created a VM
    and two firewall rules in a live project. The seam is only a seam if
    forgetting to inject it fails loudly.
    """

    def refuse(args: tp.Sequence[str], *, stdin_text: str | None, timeout: float) -> tp.NoReturn:
        raise AssertionError(f"tests must not run gcloud: {' '.join(args)}")

    monkeypatch.setattr(head_commands, "_run_gcloud", refuse)


def test_a_provisioner_with_no_injected_runner_uses_the_module_boundary() -> None:
    """The guard above only works if the runner is resolved at call time.

    A default argument would bind the real function at import, which is exactly
    how the accident happened.
    """

    with pytest.raises(AssertionError, match="tests must not run gcloud"):
        HeadProvisioner(_spec()).status()


def test_a_dry_run_prints_every_action_and_resource_name_and_issues_no_command(tmp_path: Path) -> None:
    """A rehearsal that runs a command is not a rehearsal.

    The plan is what makes this command approvable before it is affordable, so
    the guarantee has to be that nothing at all reaches the provider boundary --
    not "nothing mutating", which is a judgement call a reader cannot check.
    """

    config = tmp_path / "fleet.yaml"
    config.write_text(FLEET_YAML, encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def refuse(args: tp.Sequence[str], *, stdin_text: str | None, timeout: float):
        calls.append(tuple(args))
        raise AssertionError("a dry run must not reach gcloud")

    spec = HeadSpec.from_fleet_config(EyvanFleetConfig.from_yaml(config))
    plan = HeadProvisioner(spec, runner=refuse).plan()
    rendered = "\n".join(plan.render_lines())

    assert calls == []
    for name in spec.resource_names.values():
        assert name in rendered
    steps = [action.step for action in plan.actions]
    assert steps == [
        HeadStep.INSTANCE_LOOKUP,
        HeadStep.INSTANCE_CREATE,
        HeadStep.INSTANCE_ADDRESS,
        HeadStep.SUBNET_LOOKUP,
        HeadStep.FIREWALL_LOOKUP,
        HeadStep.FIREWALL_CREATE,
        HeadStep.FIREWALL_LOOKUP,
        HeadStep.FIREWALL_CREATE,
        HeadStep.SECRET_LOOKUP,
        # Before CONFIGURE: the environment file names the directory the UI is
        # unpacked into, and must not describe something that is not there yet.
        HeadStep.STAGE_DASHBOARD,
        HeadStep.CONFIGURE,
        HeadStep.STAGE_WHEEL,
        HeadStep.INSTALL,
        HeadStep.BOOTSTRAP,
        HeadStep.MIGRATE,
        HeadStep.SERVICE_START,
        HeadStep.SERVICE_VERIFY,
    ]
    for action in plan.actions:
        assert action.command in rendered


def test_the_dry_run_cli_changes_nothing_and_names_every_resource(monkeypatch, tmp_path: Path) -> None:
    """The flag, not just the class: --dry-run is what an operator actually types."""

    config = tmp_path / "fleet.yaml"
    config.write_text(FLEET_YAML, encoding="utf-8")

    def refuse(args, *, stdin_text, timeout):
        raise AssertionError("a dry run must not reach gcloud")

    monkeypatch.setattr(head_commands, "_run_gcloud", refuse)

    invoked = CliRunner().invoke(head_commands.head_group, ["up", str(config), "--dry-run", "--json"])

    assert invoked.exit_code == 0, invoked.output
    document = json.loads(invoked.output)
    assert document["dry_run"] is True
    assert document["spec"]["resources"] == {
        "firewall_rule": "research-head-api",
        "instance": "research-head",
        "network_tag": "research-head",
        "ssh_firewall_rule": "research-head-iap-ssh",
    }
    assert [action["step"] for action in document["actions"]].count("firewall_create") == 2
    assert all(action["command"].startswith("gcloud ") for action in document["actions"])


def test_the_executed_sequence_is_exactly_the_planned_sequence() -> None:
    """A plan that diverges from the run is a plan that cannot be reviewed.

    Both are built from the same action constructors, and this is the test that
    keeps them there: the only difference permitted is the source range, which a
    dry run deliberately does not discover.
    """

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)
    planned = [action.args for action in HeadProvisioner(spec, runner=gcloud).plan().actions]

    HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    executed = gcloud.issued
    assert len(executed) == len(planned)
    for planned_args, executed_args in zip(planned, executed, strict=True):
        if head_commands.UNRESOLVED_SOURCE_RANGES in " ".join(planned_args):
            assert "--source-ranges=10.128.0.0/20" in executed_args
            continue
        assert planned_args == executed_args


def test_up_creates_one_labelled_head_and_returns_its_internal_address() -> None:
    """Adoption is only safe if creation leaves the label adoption checks for."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    result = HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    created = gcloud.issued_matching("create", "research-head")[0]
    assert "--labels" in created
    assert created[created.index("--labels") + 1] == (
        f"{HEAD_FLEET_LABEL_KEY}=research,{HEAD_MANAGED_LABEL_KEY}={HEAD_MANAGED_LABEL_VALUE}"
    )
    assert "--no-address" in created
    assert result.address == "10.128.0.7"
    assert result.url == "https://10.128.0.7:8080"
    # Two audiences, two commands. On the head the installer recorded the plane,
    # so nothing is passed; from a laptop the address is private and needs both a
    # tunnel and explicit flags.
    # This spec declares no worker interpreter, so the head provisions no second
    # Python and the submit command is the plain one.
    assert result.next_command == (
        "on the head: eyvan up <fleet.yaml> && python train.py   |   "
        "elsewhere: EYVAN_CONTROL_PLANE_CA_FILE=<copy of /etc/eyvan/control-plane-ca.pem> "
        "eyvan --url https://10.128.0.7:8080 --token <token> up <fleet.yaml>"
    )
    assert result.service_state == "active"
    assert result.adopted_instance is False


def test_control_plane_ingress_is_the_vpc_only_and_never_the_internet() -> None:
    """The one rule that must never be got wrong, asserted on the emitted argv."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec, subnets=("10.128.0.0/20", "172.16.4.0/22"))

    result = HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    api_rule = gcloud.issued_matching("create", "research-head-api")[0]
    assert _flag_value(api_rule, "--source-ranges=") == "10.128.0.0/20,172.16.4.0/22"
    assert _flag_value(api_rule, "--allow=") == "tcp:8080"
    assert _flag_value(api_rule, "--target-tags=") == "research-head"
    assert result.source_ranges == ("10.128.0.0/20", "172.16.4.0/22")
    assert "0.0.0.0/0" not in " ".join(" ".join(args) for args in gcloud.issued)


def test_a_subnet_range_that_is_not_internal_is_refused_rather_than_opened() -> None:
    """Refusing is the requirement. A VPC Eyvan cannot prove is private is one
    whose ranges it must not put in front of a control-plane port."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec, subnets=("10.128.0.0/20", "203.0.113.0/24"))

    with pytest.raises(HeadIngressError, match=re.escape("203.0.113.0/24")) as failure:
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert failure.value.step is HeadStep.SUBNET_LOOKUP
    assert "--source-range" in failure.value.detail
    assert gcloud.issued_matching("firewall-rules", "create") == []


def test_a_vpc_with_no_subnet_in_the_region_stops_before_any_ingress_rule() -> None:
    """No discovered range is not an empty allowlist; it is an unanswerable question."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec, subnets=())

    with pytest.raises(HeadIngressError, match="cannot restrict"):
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert gcloud.issued_matching("firewall-rules", "create") == []


def test_an_owned_rule_widened_to_the_internet_stops_the_run() -> None:
    """Eyvan created the rule; something else opened it. Continuing would put a
    control plane behind an allowlist that is not one."""

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        instance=_owned_instance(spec),
        firewall={spec.firewall_rule_name: _owned_firewall(spec, spec.firewall_rule_name, ranges=("0.0.0.0/0",))},
    )

    with pytest.raises(HeadIngressError, match=re.escape("0.0.0.0/0")) as failure:
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert failure.value.step is HeadStep.FIREWALL_LOOKUP
    assert gcloud.issued_matching("compute", "ssh") == []


def test_explicit_source_ranges_replace_discovery_and_must_still_be_internal() -> None:
    """An operator may pin the allowlist; they may not widen it."""

    gcloud = FakeGcloud(spec=_spec())
    pinned = _spec(source_ranges=("10.24.0.0/16",))

    HeadProvisioner(pinned, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert gcloud.issued_matching("compute", "networks", "subnets", "list") == []
    assert _flag_value(gcloud.issued_matching("create", "research-head-api")[0], "--source-ranges=") == "10.24.0.0/16"
    with pytest.raises(ValueError, match="not an internal VPC range"):
        _spec(source_ranges=("0.0.0.0/0",))
    with pytest.raises(ValueError, match="not an internal VPC range"):
        _spec(source_ranges=("8.8.8.0/24",))


def test_up_refuses_a_same_named_instance_it_did_not_create() -> None:
    """The most expensive kind of idempotence is the kind that adopts a stranger."""

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        instance={"name": spec.instance_name, "status": "RUNNING", "labels": {"team": "someone-else"}},
    )

    with pytest.raises(HeadOwnershipError, match="did not create it") as failure:
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert failure.value.step is HeadStep.INSTANCE_LOOKUP
    assert gcloud.issued_matching("compute", "instances", "create") == []
    assert gcloud.issued_matching("compute", "ssh") == []


def test_up_refuses_a_firewall_rule_name_another_rule_already_owns() -> None:
    """A rule with the right name and someone else's description is not ours."""

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        instance=_owned_instance(spec),
        firewall={
            spec.firewall_rule_name: {
                "name": spec.firewall_rule_name,
                "description": "platform team: allow prometheus",
                "sourceRanges": ["10.128.0.0/20"],
            }
        },
    )

    with pytest.raises(HeadOwnershipError, match="will not change it") as failure:
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert failure.value.step is HeadStep.FIREWALL_LOOKUP
    assert gcloud.issued_matching("firewall-rules", "create") == []


def test_a_second_up_creates_nothing_and_keeps_the_first_run_credentials() -> None:
    """Running it twice must produce one head, and must not rotate its keys.

    Rewriting ``EYVAN_ARTIFACT_TOKEN_KEY`` on a re-run would invalidate every
    capability token already minted, which looks exactly like an outage in the
    artifact path and nothing like the idempotent command that caused it.
    """

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)
    provisioner = HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS)

    first = provisioner.up()
    gcloud.calls.clear()
    second = provisioner.up()

    assert first.adopted_instance is False
    assert first.wrote_environment is True
    assert second.adopted_instance is True
    assert second.wrote_environment is False
    assert second.created_firewall_rules == ()
    assert gcloud.issued_matching("compute", "instances", "create") == []
    assert gcloud.issued_matching("firewall-rules", "create") == []
    assert [
        call for call in gcloud.calls if call.stdin_text is not None and "EYVAN_METRICS_TOKEN" in call.stdin_text
    ] == []


def test_rotate_secrets_is_the_only_way_an_adopted_head_gets_new_credentials() -> None:
    """Opt-in, because the cost of rotating is paid by tokens already in flight."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec, instance=_owned_instance(spec), environment_present=True)

    result = HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up(rotate_secrets=True)

    assert result.wrote_environment is True
    assert "EYVAN_METRICS_TOKEN" in gcloud.stdin_for(HEAD_ENVIRONMENT_PATH)


def test_the_head_environment_carries_every_setting_the_server_refuses_to_start_without() -> None:
    """Three crash-restart cycles, removed by generating all three at once.

    A network-bound Eyvan server refuses to start without ``postgres`` and a
    metrics token, and refuses to mint artifact tokens without a signing key. The
    manual path discovers those one restart at a time.
    """

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    assert environment["EYVAN_REPOSITORY_BACKEND"] == "postgres"
    assert environment["EYVAN_METRICS_TOKEN"] == STUB_SECRETS.metrics_token
    assert environment["EYVAN_ARTIFACT_TOKEN_KEY"] == STUB_SECRETS.artifact_token_key
    assert environment["EYVAN_AGENT_SHARED_SECRET"] == STUB_SECRETS.agent_shared_secret
    assert environment["EYVAN_DATABASE_URL"] == (
        f"postgresql+asyncpg://eyvan:{STUB_SECRETS.database_password}@127.0.0.1:5432/eyvan"
    )
    assert environment["EYVAN_PUBLIC_URL"] == "https://10.128.0.7:8080"


def test_generated_credentials_are_delivered_on_stdin_and_never_in_an_argument() -> None:
    """A credential in argv is readable in ``ps`` on both ends of the connection."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    for value in STUB_SECRETS.values():
        assert not any(value in argument for args in gcloud.issued for argument in args)
    configure = [call for call in gcloud.calls if call.stdin_text and "EYVAN_METRICS_TOKEN" in call.stdin_text]
    assert len(configure) == 1
    assert configure[0].remote_command == f"sudo install -D -m 0600 -o root -g root /dev/stdin {HEAD_ENVIRONMENT_PATH}"


def test_generated_credentials_are_distinct_high_entropy_values() -> None:
    """One value reused across three settings collapses three trust boundaries."""

    generated = HeadSecrets.generate()

    values = generated.values()
    # Five, and derived from the dataclass rather than written as a literal: the
    # count was 4 while the class held 5 fields, and the missing one was
    # `bootstrap_token` -- the credential that authenticates the whole control
    # plane, and therefore the one every redaction assertion below silently
    # stopped covering.
    assert len(values) == len(dataclasses.fields(generated))
    assert len(set(values)) == len(values)
    assert all(len(value) >= 24 for value in values)
    assert all(len(value) >= 32 for value in values[:3])
    assert "***" in repr(generated)
    assert not any(value in repr(generated) for value in values)


def test_the_non_secret_result_document_contains_no_generated_material() -> None:
    """Automation output is copied into logs and tickets."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    result = HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    encoded = json.dumps(result.as_document())
    assert not any(value in encoded for value in STUB_SECRETS.values())
    assert result.as_document()["url"] == "https://10.128.0.7:8080"


def test_a_head_with_no_issuer_says_so_instead_of_answering_401_silently() -> None:
    """The server starts either way; only one of those is worth reporting as ready."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    result = HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    warning = result.authentication_warning
    assert warning is not None
    assert "bootstrap token" in warning
    assert "--oidc-issuer" in warning, "the warning must name the way to retire the bootstrap token"

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    assert environment["EYVAN_BOOTSTRAP_TOKEN"] == STUB_SECRETS.bootstrap_token
    assert "EYVAN_OIDC_ISSUER" not in environment


def test_configuring_an_issuer_removes_the_warning_and_reaches_the_head() -> None:
    """The remedy the warning names has to be the one the flag performs."""

    spec = _spec(oidc_issuer="https://issuer.example.com", oidc_audience="eyvan-prod")
    gcloud = FakeGcloud(spec=spec)

    result = HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    assert environment["EYVAN_OIDC_ISSUER"] == "https://issuer.example.com"
    assert environment["EYVAN_OIDC_AUDIENCE"] == "eyvan-prod"
    assert result.authentication_warning is None


def test_an_issuer_head_grants_callable_trust_only_where_an_operator_named_it() -> None:
    """The reported defect: trust was written only in the branch with no issuer.

    ``@execute`` worked on a laptop, because a loopback development plane trusts
    its own default project, and was refused on a real head with a 403 *after*
    authenticating. Writing the default inside the bootstrap branch fixed that
    for the deployment with no identity provider and left it broken for the one
    that has one -- the deployment anyone runs for real.

    The fix is not to auto-grant under an issuer. Deserializing a callable runs
    arbitrary code on the fleet, and the trust policy grants that to every
    principal holding SUBMIT in a trusted project, which under OIDC is a set the
    operator never enumerated. So the operator names it, and is told when nobody
    has.
    """

    issuer = "https://issuer.example.com"
    granted = _spec(oidc_issuer=issuer, trusted_callable_projects=("acme/research",))
    gcloud = FakeGcloud(spec=granted)

    result = HeadProvisioner(granted, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    assert environment["EYVAN_TRUSTED_CALLABLE_PROJECTS"] == "acme/research"
    assert result.callable_trust_warning is None

    ungranted = _spec(oidc_issuer=issuer)
    gcloud = FakeGcloud(spec=ungranted)

    result = HeadProvisioner(ungranted, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    assert "EYVAN_TRUSTED_CALLABLE_PROJECTS" not in environment
    # The warning has to name the remedy, because the 403 the operator would
    # otherwise meet names the workload rather than the setting.
    warning = result.callable_trust_warning
    assert warning is not None
    assert "--trust-callable-project" in warning
    assert "command workloads are unaffected" in warning


def test_a_bootstrap_head_keeps_trusting_its_one_principal_and_an_operator_can_replace_it() -> None:
    """Bootstrap mode has exactly one principal; naming projects replaces, never widens."""

    gcloud = FakeGcloud(spec=_spec())
    result = HeadProvisioner(_spec(), runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    assert environment["EYVAN_TRUSTED_CALLABLE_PROJECTS"] == "default/default"
    assert result.callable_trust_warning is None

    replaced = _spec(trusted_callable_projects=("acme/research", "acme/eval"))
    gcloud = FakeGcloud(spec=replaced)
    HeadProvisioner(replaced, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    assert environment["EYVAN_TRUSTED_CALLABLE_PROJECTS"] == "acme/research,acme/eval"


def test_a_trusted_project_containing_a_comma_is_refused() -> None:
    """The variable is one comma-separated line in a file a shell also sources.

    A comma inside an entry would silently become two entries and grant code
    execution on this fleet to a project nobody named.
    """

    with pytest.raises(ValueError, match="must not contain a comma"):
        _spec(trusted_callable_projects=("acme/research,evil/project",))


def test_a_step_that_fails_midway_names_the_step_and_stops_the_sequence() -> None:
    """Half a head is the normal outcome of a fifteen-step manual install.

    What must not happen is a half-installed head reported as running, so the
    failure carries the step and nothing after it is attempted.
    """

    spec = _spec()
    failure_result = _failed("E: Unable to fetch archives")
    gcloud = FakeGcloud(spec=spec, failures={"--command sudo /opt/eyvan/head-bootstrap.sh": failure_result})

    with pytest.raises(HeadCommandError, match="Unable to fetch archives") as failure:
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert failure.value.step is HeadStep.BOOTSTRAP
    remote_commands = [call.remote_command for call in gcloud.calls]
    assert "sudo /opt/eyvan/head-bootstrap.sh" in remote_commands
    assert not any(remote and "eyvan-migrate" in remote for remote in remote_commands)
    assert not any(remote and "systemctl enable" in remote for remote in remote_commands)


def test_an_unreachable_head_never_becomes_a_reason_to_rotate_its_credentials() -> None:
    """The probe's failure mode matters more than the probe.

    Reading "ssh failed" as "no credentials yet" would rewrite a live head's
    signing key over one bad minute of network.
    """

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        instance=_owned_instance(spec),
        environment_present=True,
        failures={
            "eyvan-head-env-present": _failed("ssh: connect to host port 22: Connection timed out", returncode=255)
        },
    )

    with pytest.raises(HeadCommandError, match="Connection timed out") as failure:
        HeadProvisioner(
            spec,
            runner=gcloud,
            secret_factory=lambda: STUB_SECRETS,
            ssh_ready_attempts=2,
            ssh_ready_delay_s=0,
            sleep=lambda _: None,
        ).up()

    assert failure.value.step is HeadStep.SECRET_LOOKUP
    assert [call for call in gcloud.calls if call.stdin_text is not None] == []


def test_a_head_that_answers_something_unexpected_is_not_treated_as_fresh() -> None:
    """A shell that prints a banner is not a head without credentials."""

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        instance=_owned_instance(spec),
        failures={"eyvan-head-env-present": _ok([], "Welcome to Ubuntu\n")},
    )

    with pytest.raises(HeadCommandError, match="credential-presence marker"):
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()


def test_the_installed_unit_survives_ssh_exit_and_reboot() -> None:
    """`nohup ... &` is the failure this replaces; systemd is the fix, stated."""

    script = render_head_bootstrap_script(_spec())

    assert "[Install]\nWantedBy=multi-user.target" in script
    assert "Restart=always" in script
    assert "ExecStart=/opt/eyvan/venv/bin/eyvan-server --host 0.0.0.0 --port 8080" in script
    assert f"EnvironmentFile={HEAD_ENVIRONMENT_PATH}" in script
    assert "systemctl daemon-reload" in script
    plan = HeadProvisioner(_spec()).plan()
    start = next(action for action in plan.actions if action.step is HeadStep.SERVICE_START)
    # Restart, not `enable --now`: the latter is a no-op on an already-active
    # unit, so a re-run replaced the wheel and rewrote the environment while the
    # server went on serving the code and credentials it started with.
    assert start.args[-1] == "sudo systemctl enable eyvan-head.service && sudo systemctl restart eyvan-head.service"


def test_the_installer_carries_no_credential_and_does_not_require_uv() -> None:
    """`uv` missing on a fresh VM was one of the six stop-the-world failures."""

    script = render_head_bootstrap_script(_spec())

    assert not any(value in script for value in STUB_SECRETS.values())
    assert "uv build" not in script
    assert "uv pip" not in script
    assert "python3 -m venv /opt/eyvan/venv" in script
    # The staged wheel, not the name `eyvan`: that name resolves only from an
    # index, and a real bootstrap died on "No matching distribution found".
    assert "--upgrade /opt/eyvan/wheels/*.whl" in script
    assert not any(line.rstrip().endswith(" eyvan") for line in script.splitlines() if "pip install" in line)


def test_the_head_environment_carries_every_secret_its_own_units_require() -> None:
    """Found on the first real head: the agent crash-looped on a secret nobody wrote.

    `eyvan-head-agent.service` refuses to start without
    `EYVAN_NATIVE_WORKER_ROOT_SECRET` whenever the executor is `eyvan-native` --
    which is every TPU fleet this command exists to serve -- and the GCP capacity
    driver refuses to provision a TPU or Compute cell without it too. So a head
    built without it restarted its agent forever and could never have created a
    cell, while `systemctl is-active` still answered "active" between restarts.

    The Kubernetes manifests have always set it. Only this one-command path did
    not, which is the same shape as every other setting a loopback development
    plane supplies for free and a real head does not.
    """

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))

    assert environment["EYVAN_NATIVE_WORKER_ROOT_SECRET"] == STUB_SECRETS.native_worker_root_secret
    # Both units read this one file, so writing it once serves the agent and the
    # server's capacity driver alike.
    script = render_head_bootstrap_script(spec)
    assert script.count(f"EnvironmentFile={HEAD_ENVIRONMENT_PATH}") == 2


def test_the_installer_records_the_control_plane_this_node_serves() -> None:
    """So a process on the head needs neither EYVAN_URL nor EYVAN_TOKEN.

    The Ray property: whatever stands a node up writes the answer down on that
    node. Everything needed already existed -- the context format, the reader in
    `init()` -- and nothing wrote the file, so `head up` finished holding the URL
    it had just created and only pasted it into a printed command.

    It is also what keeps the SDK's endpoint non-loopback on the head, which is
    what stops `init()` from trying to start a *second* control plane there.
    """

    script = render_head_bootstrap_script(_spec())

    assert "write_node_context" in script
    assert "/etc/eyvan/node.json" not in script, "the path belongs to eyvan.sdk.node, not to a literal here"
    # Written before the unit starts, so the file is there for anything that runs
    # after this script returns.
    assert script.index("write_node_context") < script.index("systemctl daemon-reload")
    # And the operator can read the credential without a sudo round trip.
    assert 'usermod -aG eyvan-head "${SUDO_USER}"' in script


def test_a_staged_wheel_is_reinstalled_even_at_an_unchanged_version() -> None:
    """`--upgrade` alone is a no-op at an equal version, so the head kept old code.

    The observed failure: `eyvan head up` was re-run against a head after a code
    change that did not bump the version, pip reported "Requirement already
    satisfied", the unit restarted on the previous wheel, and every later
    observation described code that was not running. A staged wheel's version is
    the version of the tree that built it, so this is the *ordinary* case rather
    than an edge one.
    """

    script = render_head_bootstrap_script(_spec())
    install_lines = [line for line in script.splitlines() if "pip install" in line]

    assert any("--force-reinstall" in line and "--no-deps" in line for line in install_lines), install_lines
    # The dependency-resolving pass still has to come first: --no-deps on its own
    # would install Eyvan into a venv with nothing under it.
    upgrade_index = next(index for index, line in enumerate(install_lines) if "--upgrade" in line)
    forced_index = next(index for index, line in enumerate(install_lines) if "--force-reinstall" in line)
    assert upgrade_index < forced_index


def test_an_index_install_is_not_forced() -> None:
    """A version specifier is real resolution work; forcing it only redoes it."""

    script = render_head_bootstrap_script(_spec(package="eyvan==0.0.1"))

    assert not any("--force-reinstall" in line for line in script.splitlines() if "pip install" in line)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to parse the generated installer")
def test_the_generated_installer_is_syntactically_valid_bash() -> None:
    """It runs unattended under sudo on a machine nobody is watching."""

    for spec in (_spec(), _spec(database_url="postgresql+asyncpg://eyvan:pw@10.9.0.3:5432/eyvan")):
        script = render_head_bootstrap_script(spec)
        parsed = subprocess.run(
            ["bash", "-n"],
            input=script,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert parsed.returncode == 0, parsed.stderr


def test_an_external_database_replaces_the_head_local_postgresql() -> None:
    """A fleet with a managed database should not get a second one on the head."""

    external = _spec(database_url="postgresql+asyncpg://eyvan:pw@10.9.0.3:5432/eyvan")
    gcloud = FakeGcloud(spec=external)

    HeadProvisioner(external, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    script = render_head_bootstrap_script(external)
    assert "install -y --no-install-recommends postgresql" not in script
    assert "createdb" not in script
    assert "After=network-online.target\n" in script
    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    assert environment["EYVAN_DATABASE_URL"] == "postgresql+asyncpg://eyvan:pw@10.9.0.3:5432/eyvan"
    local = render_head_bootstrap_script(_spec())
    assert "install -y --no-install-recommends postgresql" in local
    assert "createdb --owner eyvan eyvan" in local


def test_a_database_url_a_shell_would_reinterpret_is_refused() -> None:
    """It becomes a line in an environment file the migration step sources."""

    with pytest.raises(ValueError, match="percent-encode it"):
        _spec(database_url="postgresql+asyncpg://eyvan:pa$$word@10.9.0.3:5432/eyvan")


def test_migration_runs_with_the_heads_own_environment_before_the_service_starts() -> None:
    """A control plane started against an unmigrated schema fails at first request."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    remote_commands = [call.remote_command for call in gcloud.calls if call.remote_command]
    migrate = next(command for command in remote_commands if "eyvan-migrate" in command)
    assert HEAD_ENVIRONMENT_PATH in migrate
    assert remote_commands.index(migrate) < remote_commands.index(
        "sudo systemctl enable eyvan-head.service && sudo systemctl restart eyvan-head.service"
    )


def test_public_internet_egress_keeps_an_external_address_and_still_no_public_ingress() -> None:
    """An external address is an egress decision; ingress stays the VPC's."""

    spec = _spec(egress="public-internet")
    gcloud = FakeGcloud(spec=spec)

    HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    created = gcloud.issued_matching("compute", "instances", "create")[0]
    assert "--no-address" not in created
    assert _flag_value(gcloud.issued_matching("create", "research-head-api")[0], "--source-ranges=") == "10.128.0.0/20"


def test_the_egress_modes_are_the_ones_the_capacity_path_accepts() -> None:
    """The set an operator has to guess today, stated in one importable place."""

    assert head_commands.BOOTSTRAP_EGRESS_MODES == ("cloud-nat", "private-google-access", "public-internet")
    with pytest.raises(ValueError, match="cloud-nat, private-google-access, public-internet"):
        _spec(egress="nat")


def test_a_head_inherits_project_zone_and_egress_from_the_fleet_it_will_control(tmp_path: Path) -> None:
    """The head belongs where the fleet is; nothing should have to be retyped."""

    config = tmp_path / "fleet.yaml"
    config.write_text(FLEET_YAML, encoding="utf-8")

    spec = HeadSpec.from_fleet_config(EyvanFleetConfig.from_yaml(config))

    assert (spec.project, spec.region, spec.zone) == ("example-project-1", "us-east5", "us-east5-a")
    assert spec.egress == "private-google-access"
    assert spec.instance_name == "research-head"
    assert spec.has_external_address is False


def test_pools_that_disagree_about_egress_produce_an_error_naming_the_legal_values(tmp_path: Path) -> None:
    """The failure that cost an afternoon: a rejected value whose legal set was
    documented nowhere. The message is the documentation."""

    config = tmp_path / "fleet.yaml"
    config.write_text(
        FLEET_YAML
        + """  nvidia-tesla-a100:
    kind: gpu
    machine_type: a2-highgpu-1g
    resources:
      CPU: 12
      MEMORY_GIB: 85
      GPU: 1
    provider_options:
      bootstrap_egress: public-internet
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cloud-nat, private-google-access, public-internet"):
        HeadSpec.from_fleet_config(EyvanFleetConfig.from_yaml(config))

    resolved = HeadSpec.from_fleet_config(EyvanFleetConfig.from_yaml(config), egress="cloud-nat")
    assert resolved.egress == "cloud-nat"


def test_status_reports_the_address_ingress_and_unit_state_of_a_running_head() -> None:
    """ "Is it up" has three answers and an operator needs all three."""

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        instance=_owned_instance(spec),
        firewall={
            spec.firewall_rule_name: _owned_firewall(spec, spec.firewall_rule_name, ranges=("10.128.0.0/20",)),
        },
    )

    result = HeadProvisioner(spec, runner=gcloud).status()

    assert result.exists is True
    assert result.instance_status == "RUNNING"
    assert result.address == "10.128.0.7"
    assert result.service_state == "active"
    assert result.source_ranges == ("10.128.0.0/20",)
    assert result.as_document()["url"] == "https://10.128.0.7:8080"


def test_status_reports_an_absent_head_without_failing_and_without_ssh() -> None:
    """Asking about a head that was never created is a question, not an error."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    result = HeadProvisioner(spec, runner=gcloud).status()

    assert result.exists is False
    assert result.address is None
    assert result.service_state is None
    assert gcloud.issued_matching("compute", "ssh") == []


def test_status_refuses_to_describe_a_same_named_instance_it_does_not_own() -> None:
    """Reporting someone else's VM as the fleet's control plane is worse than
    reporting nothing."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec, instance={"name": spec.instance_name, "status": "RUNNING", "labels": {}})

    with pytest.raises(HeadOwnershipError):
        HeadProvisioner(spec, runner=gcloud).status()


def test_down_without_confirmation_deletes_nothing() -> None:
    """The head holds the control plane's durable state on its own disk."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec, instance=_owned_instance(spec))

    with pytest.raises(head_commands.HeadError, match="pass --yes to confirm"):
        HeadProvisioner(spec, runner=gcloud).down(confirmed=False)

    assert gcloud.issued == []
    assert gcloud.instance is not None


def test_down_deletes_the_head_and_the_rules_it_owns() -> None:
    """Teardown that leaves the firewall rules behind is a name collision next time."""

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        instance=_owned_instance(spec),
        firewall={
            spec.firewall_rule_name: _owned_firewall(spec, spec.firewall_rule_name, ranges=("10.128.0.0/20",)),
            spec.ssh_firewall_rule_name: _owned_firewall(
                spec, spec.ssh_firewall_rule_name, ranges=(head_commands.IAP_TCP_FORWARDING_RANGE,)
            ),
        },
    )

    result = HeadProvisioner(spec, runner=gcloud).down(confirmed=True)

    assert result.deleted_instance is True
    assert result.deleted_firewall_rules == (spec.firewall_rule_name, spec.ssh_firewall_rule_name)
    assert gcloud.instance is None
    assert gcloud.firewall == {}


def test_down_refuses_to_delete_an_instance_eyvan_does_not_own() -> None:
    """A destructive command must fail closed on an ownership question."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec, instance={"name": spec.instance_name, "labels": {"owner": "platform"}})

    with pytest.raises(HeadOwnershipError):
        HeadProvisioner(spec, runner=gcloud).down(confirmed=True)

    assert gcloud.issued_matching("compute", "instances", "delete") == []


def test_the_down_command_requires_the_flag_and_not_merely_a_terminal(monkeypatch) -> None:
    """An explicit flag, so a confirmation cannot be produced by a tty."""

    def refuse(args, *, stdin_text, timeout):
        raise AssertionError("unconfirmed teardown must not reach gcloud")

    monkeypatch.setattr(head_commands, "_run_gcloud", refuse)

    invoked = CliRunner().invoke(
        head_commands.head_group,
        [
            "down",
            "--fleet",
            "research",
            "--project",
            "example-project-1",
            "--region",
            "us-east5",
            "--zone",
            "us-east5-a",
        ],
    )

    assert invoked.exit_code != 0
    assert "pass --yes to confirm" in invoked.output


def test_the_up_command_refuses_to_provision_without_confirmation_outside_a_terminal(monkeypatch) -> None:
    """Non-interactive automation must state its intent to spend money."""

    def refuse(args, *, stdin_text, timeout):
        raise AssertionError("unconfirmed provisioning must not reach gcloud")

    monkeypatch.setattr(head_commands, "_run_gcloud", refuse)
    monkeypatch.setattr(head_commands, "is_interactive", lambda: False)

    invoked = CliRunner().invoke(
        head_commands.head_group,
        [
            "up",
            "--fleet",
            "research",
            "--project",
            "example-project-1",
            "--region",
            "us-east5",
            "--zone",
            "us-east5-a",
            "--egress",
            "cloud-nat",
        ],
    )

    assert invoked.exit_code != 0
    assert "requires --yes" in invoked.output


def test_the_up_command_reports_the_address_the_ingress_and_the_next_command(monkeypatch) -> None:
    """The last line of a successful install should be the next command, typed."""

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)
    monkeypatch.setattr(head_commands, "_run_gcloud", gcloud)
    monkeypatch.setattr(head_commands.HeadSecrets, "generate", staticmethod(lambda: STUB_SECRETS))

    invoked = CliRunner().invoke(
        head_commands.head_group,
        [
            "up",
            "--fleet",
            "research",
            "--project",
            "example-project-1",
            "--region",
            "us-east5",
            "--zone",
            "us-east5-a",
            "--egress",
            "private-google-access",
            "--yes",
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    assert "https://10.128.0.7:8080" in invoked.output
    assert "restricted to 10.128.0.0/20" in invoked.output
    assert "Next: on the head: eyvan up <fleet.yaml>" in invoked.output
    assert "eyvan --url https://10.128.0.7:8080 --token <token> up <fleet.yaml>" in invoked.output
    assert not any(value in invoked.output for value in STUB_SECRETS.values())


def test_the_head_commands_are_reachable_from_the_root_cli(monkeypatch, tmp_path: Path) -> None:
    """A command nobody can type is a command that does not exist."""

    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))

    invoked = CliRunner().invoke(cli.main, ["head", "up", "--help"])

    assert invoked.exit_code == 0, invoked.output
    assert "--dry-run" in invoked.output
    # The help still states the limit of what has been qualified, now that the
    # create path has run for real: options no live run passed are covered only
    # against the injected boundary.
    assert "not qualified" in invoked.output
    for mode in head_commands.BOOTSTRAP_EGRESS_MODES:
        assert mode in invoked.output


def test_a_head_name_that_cannot_exist_is_refused_before_any_command() -> None:
    """GCP's limits are the spec's limits, checked where they can be explained."""

    with pytest.raises(ValueError, match="too long"):
        _spec(fleet="a" * 60)
    with pytest.raises(ValueError, match="not in region"):
        _spec(zone="us-central1-a")
    with pytest.raises(ValueError, match="port"):
        _spec(port=0)


def test_explicit_flags_override_the_fleet_file_without_colliding(tmp_path: Path) -> None:
    """Overrides and file values meet in one constructor call.

    The first version of that merge passed ``region`` twice and raised a
    ``TypeError`` for every operator who overrode it.
    """

    config = tmp_path / "fleet.yaml"
    config.write_text(FLEET_YAML, encoding="utf-8")

    spec = HeadSpec.from_fleet_config(
        EyvanFleetConfig.from_yaml(config),
        region="us-central1",
        zone="us-central1-b",
        project="other-project-id",
        machine_type="e2-standard-2",
        port=9443,
        egress="cloud-nat",
    )

    assert (spec.region, spec.zone, spec.project) == ("us-central1", "us-central1-b", "other-project-id")
    create = next(action for action in HeadProvisioner(spec).plan().actions if action.step is HeadStep.INSTANCE_CREATE)
    assert create.args[create.args.index("--machine-type") + 1] == "e2-standard-2"
    assert create.args[create.args.index("--zone") + 1] == "us-central1-b"
    ingress = next(action for action in HeadProvisioner(spec).plan().actions if action.step is HeadStep.FIREWALL_CREATE)
    assert "--allow=tcp:9443" in ingress.args


def test_status_and_down_do_not_demand_an_egress_mode_to_read_or_delete_a_head(monkeypatch, tmp_path: Path) -> None:
    """Recovery commands must not require a creation-time decision.

    A fleet whose pools disagree about egress can still be inspected and torn
    down; only ``up`` has to know how the head reaches the network.
    """

    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    where = ["--fleet", "research", "--project", "example-project-1", "--region", "us-east5", "--zone", "us-east5-a"]

    status = CliRunner().invoke(head_commands.head_group, ["status", *where, "--json"])
    teardown = CliRunner().invoke(head_commands.head_group, ["down", *where, "--dry-run", "--json"])

    assert status.exit_code != 0
    assert "tests must not run gcloud" in str(status.exception)
    assert teardown.exit_code == 0, teardown.output
    assert json.loads(teardown.output)["would_delete"] == [
        "research-head",
        "research-head-api",
        "research-head-iap-ssh",
    ]


def test_the_install_page_states_the_command_its_scope_and_what_the_receipt_covers() -> None:
    """The doc gap is half the failure being fixed here.

    The egress value that stopped the live attempt was rejected by a message
    naming no legal set, and the set appeared in no page. It appears in this one
    now.

    This used to assert the page said the command "has not been run against real
    GCP", which was true when it was written and stopped being true when P0 ran
    -- so the test went on enforcing a claim the project had already disproved.
    What the page must keep doing is *bound* the claim: name the receipt, and say
    that the options no live run has passed are tested only against the injected
    command boundary. Understating what is proven and overstating it both end
    with somebody building a head by hand.
    """

    page = (Path(__file__).resolve().parent.parent / "docs" / "install.md").read_text(encoding="utf-8")

    assert "eyvan head up" in page
    assert "has run against real GCP" in page
    assert "injected command boundary" in page, "the page must still bound what the receipt covers"
    for mode in head_commands.BOOTSTRAP_EGRESS_MODES:
        assert mode in page
    assert "0.0.0.0/0" in page
    assert "--rotate-secrets" in page


def test_a_new_head_is_waited_for_rather_than_failed_on_because_sshd_is_young() -> None:
    """`instances create` returns before sshd listens.

    Without a bounded wait the very first run of this command fails on every
    project, for a reason that resolves itself in under a minute.
    """

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)
    refusals = {"count": 3}
    slept: list[float] = []

    original = gcloud._ssh

    def flaky(call: Call) -> subprocess.CompletedProcess[str]:
        if "eyvan-head-env-present" in (call.remote_command or "") and refusals["count"]:
            refusals["count"] -= 1
            return _failed("ssh: connect to host 10.128.0.7 port 22: Connection refused", returncode=255)
        return original(call)

    gcloud._ssh = flaky  # type: ignore[method-assign]

    result = HeadProvisioner(
        spec,
        runner=gcloud,
        secret_factory=lambda: STUB_SECRETS,
        ssh_ready_attempts=5,
        ssh_ready_delay_s=2.5,
        sleep=slept.append,
    ).up()

    assert refusals["count"] == 0
    assert slept == [2.5, 2.5, 2.5]
    assert result.service_state == "active"
    assert result.wrote_environment is True


def test_the_wait_for_ssh_is_bounded_and_its_failure_writes_nothing() -> None:
    """A head that never comes up must not be waited on forever, or half-configured."""

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        failures={"eyvan-head-env-present": _failed("ssh: connect to host: Connection refused", returncode=255)},
    )
    slept: list[float] = []

    with pytest.raises(HeadCommandError, match="after 4 attempts") as failure:
        HeadProvisioner(
            spec,
            runner=gcloud,
            secret_factory=lambda: STUB_SECRETS,
            ssh_ready_attempts=4,
            ssh_ready_delay_s=1.0,
            sleep=slept.append,
        ).up()

    assert failure.value.step is HeadStep.SECRET_LOOKUP
    assert slept == [1.0, 1.0, 1.0]
    assert [call for call in gcloud.calls if call.stdin_text is not None] == []


def test_a_unit_that_did_not_start_is_never_reported_as_a_running_head() -> None:
    """`is-active` prints the state and exits non-zero for every state but active.

    Reporting an address for a control plane that is not listening is the one
    output that would send an operator to debug the network instead of the unit.
    """

    spec = _spec()
    gcloud = FakeGcloud(spec=spec, failures={"is-active": _ok_with_status("failed\n", returncode=3)})

    with pytest.raises(HeadCommandError, match="is failed, not active") as failure:
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert failure.value.step is HeadStep.SERVICE_VERIFY
    assert "journalctl -u eyvan-head.service" in failure.value.detail


@pytest.mark.parametrize(
    ("plane_state", "agent_state", "restarted", "success"),
    [
        ("active", "active", False, True),
        ("active", "failed", False, False),
        ("failed", "active", False, False),
        ("active", "active", True, False),
    ],
)
def test_settle_script_checks_both_units(tmp_path, plane_state, agent_state, restarted, success) -> None:
    class Capture:
        def _ssh_action(self, step, summary, remote_command, **kwargs):
            return remote_command

    remote = HeadProvisioner._service_settled_action(Capture())
    script = shlex.split(remote)[3]
    systemctl = tmp_path / "systemctl"
    systemctl.write_text(
        '#!/bin/sh\nif [ "$1" = show ]; then\n'
        f'  if [ -f "{tmp_path}/waited" ]; then echo {int(restarted)}; else echo 0; fi\n'
        "  exit 0\nfi\n"
        f'if [ "$2" = eyvan-head.service ]; then state={plane_state}; else state={agent_state}; fi\n'
        'echo "$state"\n[ "$state" = active ]\n'
    )
    systemctl.chmod(0o700)
    sleep = tmp_path / "sleep"
    sleep.write_text(f'#!/bin/sh\n: > "{tmp_path}/waited"\n')
    sleep.chmod(0o700)
    result = subprocess.run(
        ["/bin/sh", "-ec", script], env={"PATH": f"{tmp_path}:/usr/bin:/bin"}, capture_output=True, text=True
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    if success:
        assert result.stdout.strip() == "active"
    else:
        assert f"eyvan-head.service: {plane_state}" in result.stdout
        assert f"eyvan-head-agent.service: {agent_state}" in result.stdout
        assert "restart counts" in result.stdout


def test_a_unit_that_restarts_between_checks_is_not_reported_running() -> None:
    """Reject crash-looping units by comparing restart counters across a settle interval."""

    spec = _spec()
    gcloud = FakeGcloud(
        spec=spec,
        failures={"NRestarts": _ok_with_status("active, restart counts 47 0 -> 50 0\n", returncode=3)},
    )

    with pytest.raises(HeadCommandError, match=r"restart counts 47 0 -> 50 0, not active") as failure:
        HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS).up()

    assert failure.value.step is HeadStep.SERVICE_VERIFY


def test_the_head_publishes_the_worker_wheel_and_names_the_state_agents_share() -> None:
    """Both failures that stopped a live v5p-16 after the TPU was already up.

    A native worker bootstraps on a TPU with no ssh path from the head, so it
    fetches Eyvan from object storage and refuses it unless the bytes hash to
    what it was told. `head up` published nothing, so the capacity driver
    rejected every request with "requires EYVAN_NATIVE_WORKER_PACKAGE" -- after
    the fleet applied, so it surfaced on the first submission.

    `EYVAN_FLEET_STATE_URI` is the second half. Unset, each agent keeps a private
    local copy of the fleet state, so the driver's writes and the control plane's
    reads never meet: the cell reached `ready` publishing no worker endpoints and
    dispatch waited forever, while the worker sat healthy and serving.
    """

    spec = _spec(artifact_store_uri="gs://fleet-bucket/artifacts")
    gcloud = FakeGcloud(spec=spec)

    HeadProvisioner(
        spec,
        runner=gcloud,
        secret_factory=lambda: STUB_SECRETS,
        wheel_source=_fake_wheel,
    ).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))
    expected_uri = f"gs://fleet-bucket/artifacts/native-worker/{_FAKE_WHEEL_DIGEST}/{staged_wheel_filename()}"

    assert environment["EYVAN_NATIVE_WORKER_PACKAGE"] == expected_uri
    assert environment["EYVAN_NATIVE_WORKER_PACKAGE_SHA256"] == _FAKE_WHEEL_DIGEST
    assert environment["EYVAN_NATIVE_WORKER_PACKAGE_GENERATION"] == gcloud.object_generation
    assert environment["EYVAN_FLEET_STATE_URI"] == "gs://fleet-bucket/artifacts/fleet-state/research.json"

    # Content-addressed by the digest of the wheel this head installs on itself,
    # so one fleet never runs two different distributions.
    assert _FAKE_WHEEL_DIGEST in expected_uri
    uploads = gcloud.issued_matching("storage", "cp")
    # ("storage", "cp", <local path>, <destination uri>, "--project", ...)
    assert len(uploads) == 1 and uploads[0][3] == expected_uri

    # And published before the environment that names it, or the head would
    # describe an object that does not exist yet.
    upload_index = next(i for i, call in enumerate(gcloud.calls) if call.args[:2] == ("storage", "cp"))
    configure_index = next(
        i
        for i, call in enumerate(gcloud.calls)
        if call.stdin_text is not None and HEAD_ENVIRONMENT_PATH in (call.remote_command or "")
    )
    assert upload_index < configure_index


def test_a_head_with_no_artifact_store_publishes_nothing_rather_than_inventing_a_location() -> None:
    """Guessing a bucket produces a head that looks configured and fails on submission."""

    gcloud = FakeGcloud(spec=_spec())

    HeadProvisioner(_spec(), runner=gcloud, secret_factory=lambda: STUB_SECRETS, wheel_source=_fake_wheel).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))

    assert "EYVAN_NATIVE_WORKER_PACKAGE" not in environment
    assert "EYVAN_FLEET_STATE_URI" not in environment
    assert gcloud.issued_matching("storage", "cp") == []


def test_a_non_numeric_object_generation_is_refused() -> None:
    """The generation is what pins the URI to exact bytes; a blank one pins nothing."""

    spec = _spec(artifact_store_uri="gs://fleet-bucket/artifacts")
    gcloud = FakeGcloud(spec=spec, object_generation="")

    with pytest.raises(HeadCommandError, match="generation"):
        HeadProvisioner(
            spec,
            runner=gcloud,
            secret_factory=lambda: STUB_SECRETS,
            wheel_source=_fake_wheel,
        ).up()


def _dashboard_bundle() -> bytes:
    """A minimal but real gzipped tar, so the staging path is exercised for real."""

    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        payload = b"<!doctype html><title>eyvan</title>"
        info = tarfile.TarInfo("index.html")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_a_head_built_with_a_dashboard_serves_it_at_its_own_url() -> None:
    """The UI belongs on the address the plane already answers on.

    No second port and no second firewall rule: an operator holding one address
    for the API and another for the UI is one address too many, and the VPC-only
    ingress rule would have to be duplicated to match.
    """

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    result = HeadProvisioner(
        spec,
        runner=gcloud,
        secret_factory=lambda: STUB_SECRETS,
        wheel_source=_fake_wheel,
        dashboard_source=_dashboard_bundle,
    ).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))

    assert environment["EYVAN_DASHBOARD_DIR"] == HEAD_DASHBOARD_DIR
    assert result.serves_dashboard is True
    assert result.dashboard_url == result.url
    # Unpacked before the environment that names the directory.
    steps = [call.args for call in gcloud.calls]
    staged = next(i for i, a in enumerate(steps) if HEAD_DASHBOARD_DIR in " ".join(a))
    configured = next(
        i
        for i, call in enumerate(gcloud.calls)
        if call.stdin_text is not None and HEAD_ENVIRONMENT_PATH in (call.remote_command or "")
    )
    assert staged < configured


def test_the_operator_is_told_where_the_dashboard_is() -> None:
    """The address printed *is* the UI, which is exactly why nobody finds it.

    An operator reads "Head research-head is active at http://10.128.0.22:8080"
    as an API base URL and does not think to open it in a browser -- the question
    "where's the dashboard URL?" has been asked of a head that was already
    serving one. Ingress is VPC-only, so the way in from a laptop is named too.
    """

    spec = _spec()

    result = HeadProvisioner(
        spec,
        runner=FakeGcloud(spec=spec),
        secret_factory=lambda: STUB_SECRETS,
        wheel_source=_fake_wheel,
        dashboard_source=_dashboard_bundle,
    ).up()

    note = result.dashboard_note

    assert note is not None
    assert result.url in note
    assert "start-iap-tunnel" in note
    assert spec.instance_name in note


def test_no_dashboard_note_is_printed_when_none_was_staged() -> None:
    """A wheel-built head serves no UI, and must not point at one."""

    spec = _spec()

    result = HeadProvisioner(
        spec,
        runner=FakeGcloud(spec=spec),
        secret_factory=lambda: STUB_SECRETS,
        wheel_source=_fake_wheel,
        dashboard_source=lambda: None,
    ).up()

    assert result.dashboard_note is None


def test_the_cli_is_on_path_after_a_head_comes_up() -> None:
    """The head prints `eyvan up <fleet.yaml>`; that has to be runnable.

    Observed on a live head: the instruction it had just printed answered
    `command not found`, because the CLI lives in a venv under /opt. A path an
    operator has to be told is a path the head should have linked.
    """

    script = render_head_bootstrap_script(_spec())

    assert "ln -sfn /opt/eyvan/venv/bin/eyvan /usr/local/bin/eyvan" in script
    # Somebody else's install is left alone: this script owns /opt, not
    # /usr/local, and clobbering a real file there is not its business.
    assert "[ -L /usr/local/bin/eyvan ]" in script


def test_a_head_built_without_a_dashboard_says_so_instead_of_promising_one() -> None:
    """A wheel carries no Node build tree, and the head must not claim otherwise.

    Reporting ``serves_dashboard=False`` lets the CLI explain a missing
    dashboard rather than open a URL that returns 404.
    """

    spec = _spec()
    gcloud = FakeGcloud(spec=spec)

    result = HeadProvisioner(
        spec,
        runner=gcloud,
        secret_factory=lambda: STUB_SECRETS,
        wheel_source=_fake_wheel,
        dashboard_source=lambda: None,
    ).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))

    assert "EYVAN_DASHBOARD_DIR" not in environment
    assert result.serves_dashboard is False
    assert result.dashboard_url is None
    assert result.as_document()["dashboard_url"] is None


def test_an_adopting_run_republishes_the_worker_wheel_and_repoints_the_head_at_it() -> None:
    """Otherwise the control plane upgrades and every worker stays behind.

    `head up` force-reinstalls the head's own wheel on a re-run. The worker
    package was published only when the environment was rewritten -- that is, on
    a fresh head or `--rotate-secrets` -- so an ordinary re-run after a code
    change left `EYVAN_NATIVE_WORKER_PACKAGE` naming the previous build. The head
    would run new code while every TPU it provisioned booted the old one, with
    nothing reporting the split.

    The environment is patched rather than rewritten because an adopting run
    never learns the credentials it wrote the first time; rewriting the file
    wholesale would rotate or lose them.
    """

    spec = _spec(artifact_store_uri="gs://fleet-bucket/artifacts")
    gcloud = FakeGcloud(spec=spec, instance=_owned_instance(spec), environment_present=True)

    result = HeadProvisioner(
        spec,
        runner=gcloud,
        secret_factory=lambda: STUB_SECRETS,
        wheel_source=_fake_wheel,
    ).up()

    # Adopting: the environment file was not rewritten.
    assert result.adopted_instance is True
    assert result.wrote_environment is False

    # The wheel was still published...
    uploads = gcloud.issued_matching("storage", "cp")
    assert len(uploads) == 1
    expected_uri = f"gs://fleet-bucket/artifacts/native-worker/{_FAKE_WHEEL_DIGEST}/{staged_wheel_filename()}"
    assert uploads[0][3] == expected_uri

    # ...and the existing environment was repointed at it, without touching the
    # generated credentials this run does not have.
    repoint = [
        call.remote_command
        for call in gcloud.calls
        if call.remote_command and "EYVAN_NATIVE_WORKER_PACKAGE" in call.remote_command
    ]
    assert len(repoint) == 1
    assert expected_uri in repoint[0]
    assert _FAKE_WHEEL_DIGEST in repoint[0]
    assert not any(value in repoint[0] for value in STUB_SECRETS.values())


def test_a_head_grants_its_workers_read_access_to_the_bucket_holding_their_wheel() -> None:
    """Otherwise a worker cannot download the Eyvan it was told to install.

    The GCP capacity factory reads EYVAN_NATIVE_WORKER_ARTIFACT_ROOT_URI to grant
    each worker's service account read access to that bucket, and returns without
    granting anything when it is unset. A worker in a project whose default
    service account cannot already read the bucket then fails at bootstrap -- on
    the accelerator, after it has been provisioned and billed.
    """

    spec = _spec(artifact_store_uri="gs://fleet-bucket/artifacts")
    gcloud = FakeGcloud(spec=spec)

    result = HeadProvisioner(spec, runner=gcloud, secret_factory=lambda: STUB_SECRETS, wheel_source=_fake_wheel).up()

    environment = _environment(gcloud.stdin_for(HEAD_ENVIRONMENT_PATH))

    assert environment["EYVAN_NATIVE_WORKER_ARTIFACT_ROOT_URI"] == "gs://fleet-bucket/artifacts"
    assert result.artifact_store_warning is None


def test_a_head_without_an_artifact_store_says_its_capacity_requests_will_be_refused() -> None:
    """The head reported itself active and failed on the first submission instead.

    With no artifact store there is nowhere to publish the worker wheel, so no
    EYVAN_NATIVE_WORKER_PACKAGE is written -- and the capacity driver refuses
    every eyvan-native request without it. The refusal then arrived at the point
    furthest from the omission that caused it.
    """

    gcloud = FakeGcloud(spec=_spec())

    result = HeadProvisioner(_spec(), runner=gcloud, secret_factory=lambda: STUB_SECRETS, wheel_source=_fake_wheel).up()

    warning = result.artifact_store_warning
    assert warning is not None
    assert "--artifact-store-uri" in warning
    assert "EYVAN_NATIVE_WORKER_PACKAGE" in warning
    assert result.as_document()["artifact_store_warning"] == warning


def test_a_private_google_access_head_warns_before_its_first_tpu_request_is_refused() -> None:
    """The gap is known and pinned; this moves where an operator learns of it.

    Workers on that egress cannot reach PyPI, and the head stages the wheel but
    not its dependencies -- so the GCP driver refuses every eyvan-native TPU
    request. That refusal used to arrive on the first request, which is the
    point furthest from the flag that caused it.
    """

    spec = dataclasses.replace(_spec(), egress="private-google-access")

    result = HeadProvisioner(
        spec,
        runner=FakeGcloud(spec=spec),
        secret_factory=lambda: STUB_SECRETS,
        wheel_source=_fake_wheel,
    ).up()

    warning = result.wheelhouse_warning

    assert warning is not None
    assert "EYVAN_NATIVE_WORKER_WHEELHOUSE" in warning
    assert "cloud-nat" in warning
    assert result.as_document()["wheelhouse_warning"] == warning


def test_a_head_whose_workers_can_reach_pypi_is_not_warned() -> None:
    """The other egresses stage nothing extra and need nothing said."""

    for egress in ("cloud-nat", "public-internet"):
        spec = dataclasses.replace(_spec(), egress=egress)

        result = HeadProvisioner(
            spec,
            runner=FakeGcloud(spec=spec),
            secret_factory=lambda: STUB_SECRETS,
            wheel_source=_fake_wheel,
        ).up()

        assert result.wheelhouse_warning is None, egress


@pytest.mark.parametrize("version", ["3.9", "3.10", "3.14"])
def test_a_head_refuses_an_unsupported_submitter_interpreter_before_provisioning(version) -> None:
    config = EyvanFleetConfig.from_yaml(_fleet_file_declaring_python(version))
    with pytest.raises(ValueError, match=r"Python 3\.11"):
        HeadSpec.from_fleet_config(config)


def test_a_head_provisions_the_interpreter_its_own_workers_run() -> None:
    """Without it a head cannot submit to the TPUs it provisions.

    A serialized callable is pinned to the interpreter that pickled it. The head
    image ships Python 3.12 and the Cloud TPU image ships 3.10, so `@execute` on
    a head produced a callable no worker could load -- and the check runs on the
    worker, so the mismatch surfaced only after the accelerator was built. A live
    multislice v5p-16 provisioned two slices and killed all four ranks proving it.

    The fleet already states which interpreter its workers run, so the head
    installs that one alongside its own and points the operator at it.
    """

    config = EyvanFleetConfig.from_yaml(_fleet_file_declaring_python("3.11"))
    spec = HeadSpec.from_fleet_config(config)

    assert spec.submitter_python == "3.11"

    script = render_head_bootstrap_script(spec)

    assert "uv venv --python 3.11 /opt/eyvan/submit" in script
    assert "/opt/eyvan/submit/bin/python" in script
    # The head's own environment is still built with the image's interpreter, so
    # a uv that cannot be fetched costs the submitter and nothing else.
    assert "python3 -m venv /opt/eyvan/venv" in script


def test_a_fleet_that_declares_no_interpreter_provisions_no_submitter() -> None:
    """Silence is not a request: a command-only fleet needs no second Python."""

    spec = _spec()

    assert spec.submitter_python is None
    script = render_head_bootstrap_script(spec)
    assert "uv venv" not in script


def _submitter_block(script: str, root: Path) -> str:
    """Return the submitter section, rewritten to install under ``root``."""

    start = script.index("# An interpreter matching")
    end = script.index("\nfi\n\n", start) + len("\nfi\n")
    block = script[start:end]
    for absolute, relative in (
        ("/opt/eyvan/submit", "submit"),
        ("/opt/eyvan/wheels", "wheels"),
        ("/opt/eyvan/python", "python"),
    ):
        block = block.replace(absolute, f"{root}/{relative}")
    return block


#: A stub that installs the way uv does: the interpreter lands in the install
#: directory, owner-only, and the venv holds a symlink to it. The mode is what
#: makes this a reproduction rather than a mock -- a real head's submitter was
#: unusable by every account except root for exactly this reason.
_UV_LIKE_INSTALLER = """#!/bin/sh
set -e
target="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}/cpython-3.11/bin"
case "$1" in
  venv)
    mkdir -p "$target" SUBMIT_DIR/bin
    printf '#!/bin/sh\\nexit 0\\n' > "$target/python3.11"
    chmod 0700 "$target/python3.11"
    chmod 0700 "$(dirname "$target")"
    ln -sf "$target/python3.11" SUBMIT_DIR/bin/python
    ;;
esac
exit 0
"""


def _run_submitter_block(block: str, root: Path, uv: str) -> subprocess.CompletedProcess[str]:
    """Run one rendered submitter section against a stub ``uv``."""

    import os
    import stat

    binaries = root / "bin"
    binaries.mkdir()
    for name, body in (("uv", uv), ("curl", "#!/bin/sh\nexit 1\n")):
        path = binaries / name
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    (root / "wheels").mkdir()
    (root / "wheels" / "eyvan-0.0.1-py3-none-any.whl").write_text("", encoding="utf-8")
    return subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{block}\necho the-rest-of-the-bootstrap"],
        capture_output=True,
        text=True,
        # HOME points into the sandbox so a regression that stops directing the
        # interpreter install writes there instead of the developer's own home.
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "HOME": str(root / "home")},
        check=False,
    )


def test_a_submitter_that_cannot_be_built_does_not_stop_the_head() -> None:
    """The bootstrap runs under ``set -e``, and this section is optional.

    An interpreter download that fails -- no route to the CDN, a version with no
    build for this platform -- must cost the submitter environment and nothing
    else. Aborting here would leave the head with no PostgreSQL and no unit: a
    convenience turning into a control plane that does not come up.
    """

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = HeadSpec.from_fleet_config(EyvanFleetConfig.from_yaml(_fleet_file_declaring_python("3.11")))
        block = _submitter_block(render_head_bootstrap_script(spec), root)

        completed = _run_submitter_block(block, root, "#!/bin/sh\nexit 1\n")

    assert completed.returncode == 0, completed.stderr
    assert "the-rest-of-the-bootstrap" in completed.stdout
    assert "refuse to submit" in completed.stderr


def test_a_half_built_submitter_is_removed_rather_than_left() -> None:
    """The command this head prints names that path.

    An interpreter that exists but cannot import ``eyvan`` is worse than none:
    the operator runs it, and the failure arrives with no explanation of which
    of the two Pythons was wrong.
    """

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = HeadSpec.from_fleet_config(EyvanFleetConfig.from_yaml(_fleet_file_declaring_python("3.11")))
        block = _submitter_block(render_head_bootstrap_script(spec), root)

        # Creates the environment, then fails installing into it -- the shape a
        # partial network failure produces.
        completed = _run_submitter_block(
            block,
            root,
            f'#!/bin/sh\ncase "$1" in venv) mkdir -p {root}/submit/bin; exit 0;; *) exit 1;; esac\n',
        )

        assert completed.returncode == 0, completed.stderr
        assert not (root / "submit").exists()


def test_the_installed_interpreter_is_executable_by_someone_who_is_not_root() -> None:
    """The head runs this as root, and prints the path to a human who is not.

    uv installs a standalone interpreter under the installing user's home by
    default, so the venv's ``python`` is a symlink into a 0700 directory: every
    operator who follows the printed command gets ``Permission denied``. A real
    head did exactly that. Directing the install into /opt and widening it is the
    fix, and this is the property that says whether it worked.
    """

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = HeadSpec.from_fleet_config(EyvanFleetConfig.from_yaml(_fleet_file_declaring_python("3.11")))
        block = _submitter_block(render_head_bootstrap_script(spec), root)

        completed = _run_submitter_block(block, root, _UV_LIKE_INSTALLER.replace("SUBMIT_DIR", f"{root}/submit"))

        assert completed.returncode == 0, completed.stderr
        interpreter = (root / "submit" / "bin" / "python").resolve()
        assert interpreter.exists(), "the venv points at no interpreter"
        assert interpreter.stat().st_mode & 0o005 == 0o005, oct(interpreter.stat().st_mode)
        for parent in (interpreter.parent, interpreter.parent.parent):
            assert parent.stat().st_mode & 0o005 == 0o005, f"{parent} is not traversable: {oct(parent.stat().st_mode)}"


def test_a_submitter_that_builds_is_reported_and_kept() -> None:
    """The other half of the same property: success leaves the interpreter."""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = HeadSpec.from_fleet_config(EyvanFleetConfig.from_yaml(_fleet_file_declaring_python("3.11")))
        block = _submitter_block(render_head_bootstrap_script(spec), root)

        completed = _run_submitter_block(
            block,
            root,
            f'#!/bin/sh\ncase "$1" in venv) mkdir -p {root}/submit/bin; touch {root}/submit/bin/python;; esac\nexit 0\n',
        )

        assert completed.returncode == 0, completed.stderr
        assert (root / "submit" / "bin" / "python").exists()
        assert "submitter interpreter ready" in completed.stdout


def _fleet_file_declaring_python(version: str) -> str:
    """Write a minimal fleet whose pools state the interpreter their workers run."""

    import tempfile

    directory = tempfile.mkdtemp()
    path = Path(directory) / "fleet.yaml"
    path.write_text(
        f"""
version: 1
name: research
provider:
  type: gcp
  project: example-project-1
  region: us-east5
  zone: us-east5-a
executor: eyvan-native
runtime:
  python: "{version}"
pools:
  v5p-16:
    capacity_class: spot
    min_workers: 0
    max_workers: 2
    provider_options:
      bootstrap_egress: public-internet
""",
        encoding="utf-8",
    )
    return str(path)


# ------------------------------------------------------------------ rolling a head back


def _rollback_runner(
    *,
    previous: str,
    digests: list[str],
    calls: list[tuple[str, ...]],
    instance: dict[str, tp.Any] | None = None,
):
    """Answer the reads a rollback makes, recording every command issued."""

    described = (
        instance
        if instance is not None
        else {
            "status": "RUNNING",
            "networkInterfaces": [{"networkIP": "10.0.0.2"}],
            # Eyvan refuses to reconfigure an instance it did not create, so a
            # fake head has to carry the ownership labels a real one does.
            "labels": {
                head_commands.HEAD_MANAGED_LABEL_KEY: head_commands.HEAD_MANAGED_LABEL_VALUE,
                head_commands.HEAD_FLEET_LABEL_KEY: "research",
            },
        }
    )

    def run(args: tp.Sequence[str], *, stdin_text: str | None, timeout: float):
        calls.append(tuple(args))
        joined = " ".join(args)
        if "describe" in args and "instances" in joined:
            if instance is None and described is None:
                return subprocess.CompletedProcess(list(args), 1, "", "not found")
            return subprocess.CompletedProcess(list(args), 0, json.dumps(described), "")
        if "build_identity" in joined:
            return subprocess.CompletedProcess(list(args), 0, digests.pop(0), "")
        if "-1t" in joined:
            return subprocess.CompletedProcess(list(args), 0, previous, "")
        return subprocess.CompletedProcess(list(args), 0, "", "")

    return run


def test_rollback_reinstalls_the_previous_wheel_and_restarts_both_units() -> None:
    """A head whose API is rolled back and whose agent is not is still broken.

    Both units run the same distribution, and a session was lost this month to
    restarting only one of them.
    """

    calls: list[tuple[str, ...]] = []
    runner = _rollback_runner(
        previous=f"{head_commands.HEAD_PREVIOUS_WHEEL_DIR}/eyvan-0.0.1-py3-none-any.whl",
        digests=["sha256:new", "sha256:old"],
        calls=calls,
    )

    digest = HeadProvisioner(_spec(), runner=runner).rollback(confirmed=True)

    assert digest == "sha256:old"
    restarts = [" ".join(call) for call in calls if "systemctl restart" in " ".join(call)]
    assert len(restarts) == 1
    assert head_commands.HEAD_SERVICE_NAME in restarts[0]
    assert head_commands.HEAD_AGENT_SERVICE_NAME in restarts[0], "the agent runs the same wheel and must restart too"


def test_rollback_refuses_when_the_digest_did_not_change() -> None:
    """Reinstalling something and still running the old build is a failed rollback.

    Verifying by ``systemctl is-active`` would call this a success: the unit is
    active either way. The digest is the only signal that distinguishes builds.
    """

    runner = _rollback_runner(
        previous=f"{head_commands.HEAD_PREVIOUS_WHEEL_DIR}/eyvan-0.0.1-py3-none-any.whl",
        digests=["sha256:same", "sha256:same"],
        calls=[],
    )

    with pytest.raises(head_commands.HeadError, match="did not take"):
        HeadProvisioner(_spec(), runner=runner).rollback(confirmed=True)


def test_rollback_refuses_without_confirmation_and_touches_nothing() -> None:
    calls: list[tuple[str, ...]] = []
    runner = _rollback_runner(previous="", digests=[], calls=calls)

    with pytest.raises(head_commands.HeadError, match="pass --yes"):
        HeadProvisioner(_spec(), runner=runner).rollback(confirmed=False)

    assert calls == []


def test_rollback_says_so_when_no_wheel_was_retained() -> None:
    """A head upgraded before retention existed has nothing to go back to."""

    runner = _rollback_runner(previous="", digests=[], calls=[])

    with pytest.raises(head_commands.HeadError, match="no superseded wheel"):
        HeadProvisioner(_spec(), runner=runner).rollback(confirmed=True)


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow


def test_a_spot_head_is_created_on_spot_and_stops_rather_than_deleting() -> None:
    """A development plane on Spot keeps its disk, and its database, when reclaimed.

    DELETE would lose the head's PostgreSQL along with the VM; STOP keeps the
    disk, and the enabled unit resumes the plane when the VM is started again.
    """

    spec = _spec(spot=True)

    create = next(action for action in HeadProvisioner(spec).plan().actions if action.step is HeadStep.INSTANCE_CREATE)

    assert "--provisioning-model=SPOT" in create.args
    assert "--instance-termination-action=STOP" in create.args
    assert spec.as_document()["provisioning_model"] == "spot"


def test_a_head_is_standard_capacity_unless_spot_is_asked_for() -> None:
    """A production head that GCP may stop at any time must be a decision, not a default."""

    spec = _spec()

    create = next(action for action in HeadProvisioner(spec).plan().actions if action.step is HeadStep.INSTANCE_CREATE)

    assert not any("provisioning-model" in arg for arg in create.args)
    assert spec.as_document()["provisioning_model"] == "standard"
