# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage for ``sudo eyvan team up``, the on-host team-plane installer.

Nothing here touches the real machine: every process goes through an injected
runner, every file lands under a temporary filesystem root, and the GCE
metadata server and the TLS health probe are replaced. What is proven is the
plan, the rendered installer, where generated secrets may and may not appear,
the refusals, and what a second run keeps. Whether apt, systemd, PostgreSQL and
openssl on a real Debian/Ubuntu VM accept the script is not established here.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import stat
import subprocess
import typing as tp
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

import eyvan.sdk.cli as cli
import eyvan.sdk.commands.head as head_commands
import eyvan.sdk.commands.team_up as team_up
from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.sdk.commands.head import HeadSpec, render_head_bootstrap_script
from eyvan.sdk.commands.team_up import (
    TEAM_ENVIRONMENT_PATH,
    GceFacts,
    TeamUpError,
    TeamUpInstaller,
    TeamUpSpec,
    discover_gce,
    merged_platform_yaml,
    render_team_bootstrap_script,
    resolve_address,
)
from eyvan.sdk.team_credentials import display_fingerprint

ADDRESS = "203.0.113.7"
PUBLIC_URL = f"https://{ADDRESS}:8443"
INVITE_TOKEN = "invite-secret-token-0123456789abcdefghij"  # pragma: allowlist secret
INVITE_URL = f"{PUBLIC_URL}/#/accept-invite/{INVITE_TOKEN}"


@pytest.fixture(autouse=True)
def never_touch_the_real_host(monkeypatch) -> None:
    """Make every real host effect unreachable, so a forgotten injection fails loudly."""

    def refuse_run(args: tp.Sequence[str], **_: tp.Any) -> tp.NoReturn:
        raise AssertionError(f"tests must not run host commands: {list(args)}")

    def refuse_probe(*_: tp.Any) -> tp.NoReturn:
        raise AssertionError("tests must not probe a real plane")

    def refuse_gcloud(args: tp.Sequence[str], **_: tp.Any) -> tp.NoReturn:
        raise AssertionError(f"tests must not run gcloud: {list(args)}")

    monkeypatch.setattr(team_up, "run_host_command", refuse_run)
    monkeypatch.setattr(team_up, "probe_health", refuse_probe)
    monkeypatch.setattr(team_up, "read_gce_metadata", lambda path, timeout=2.0: None)
    monkeypatch.setattr(head_commands, "_run_gcloud", refuse_gcloud)


def _certificate() -> tuple[str, str]:
    """Return a PEM certificate and its expected ``SHA256:AB:..`` fingerprint."""

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ADDRESS)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC))
        .not_valid_after(datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC))
        .sign(key, hashes.SHA256())
    )
    der = certificate.public_bytes(serialization.Encoding.DER)
    pem = certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")
    return pem, display_fingerprint(hashlib.sha256(der).hexdigest())


@dataclasses.dataclass
class FakeHost:
    """A host whose processes are recorded; the installer step writes the certificate."""

    root: Path
    certificate_pem: str
    admin_exists: bool = False
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = dataclasses.field(default_factory=list)
    fail: dict[str, subprocess.CompletedProcess[str]] = dataclasses.field(default_factory=dict)

    def __call__(
        self, args: tp.Sequence[str], *, env: tp.Mapping[str, str], timeout: float, stream: bool
    ) -> subprocess.CompletedProcess[str]:
        argv = tuple(args)
        self.calls.append((argv, dict(env)))
        name = Path(argv[0]).name
        if name in self.fail:
            return self.fail[name]
        if argv[0] == "/bin/bash":
            target = self.root / "etc/eyvan/tls/server.pem"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.certificate_pem, encoding="utf-8")
            (self.root / "etc/eyvan/control-plane-ca.pem").write_text(self.certificate_pem, encoding="utf-8")
        if argv[1:2] == ("bootstrap-admin",):
            if self.admin_exists:
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    "",
                    "bootstrap-admin: this plane already has team accounts; an existing fleet admin invites people "
                    "from Settings or with `eyvan users invite`\n",
                )
            self.admin_exists = True
            document = {"email": "erfan@localhost.invalid", "expires_at": "2026-11-24T00:00:00+00:00"}
            document.update(url=INVITE_URL, username=argv[argv.index("--username") + 1])
            return subprocess.CompletedProcess(argv, 0, json.dumps(document, sort_keys=True) + "\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def argv_of(self, needle: str) -> list[tuple[str, ...]]:
        return [argv for argv, _ in self.calls if any(needle in part for part in argv)]


def _spec(**overrides: tp.Any) -> TeamUpSpec:
    values: dict[str, tp.Any] = {"address": ADDRESS, "admin": "erfan", "package": "eyvan==0.0.3"}
    values.update(overrides)
    return TeamUpSpec(**values)


def _installer(host: FakeHost, spec: TeamUpSpec | None = None, **kwargs: tp.Any) -> TeamUpInstaller:
    return TeamUpInstaller(
        spec or _spec(),
        runner=host,
        root=host.root,
        health=lambda address, port, ca, timeout: True,
        sleep=lambda seconds: None,
        **kwargs,
    )


def _environment(root: Path) -> dict[str, str]:
    text = (root / TEAM_ENVIRONMENT_PATH.lstrip("/")).read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.splitlines() if line)


def _secret_values(environment: tp.Mapping[str, str]) -> tuple[str, ...]:
    password = environment["EYVAN_DATABASE_URL"].split("eyvan:", 1)[1].split("@", 1)[0]
    return (environment["EYVAN_METRICS_TOKEN"], environment["EYVAN_ARTIFACT_TOKEN_KEY"], password)


# -- head refactor ------------------------------------------------------------

#: SHA-256 of ``render_head_bootstrap_script`` for three fixed specs, captured
#: from the base commit *before* the host-install pieces were extracted. A
#: deliberate future change to the head installer updates these on purpose.
_PRE_REFACTOR_HEAD_SCRIPTS = {
    "default": ({}, "e24526f408748edf8b26222f7eaa649f85e0024136730d990203f592901a7717"),  # pragma: allowlist secret
    "external_db": (
        {"database_url": "postgresql+asyncpg://u:p@db:5432/e", "port": 9443},
        "1840a39bb7dbaab9200b4302bd64af2e7f6db6e3b3674a6fadd095f3c7b5606e",  # pragma: allowlist secret
    ),
    "submitter": (
        {"submitter_python": "3.12", "package": "eyvan==0.0.3"},
        "8229b2f2f0cccaf5e6dac186ff253ec9a81385ad44b162333bc415eba5a8587d",  # pragma: allowlist secret
    ),
}


def _head_spec(**overrides: tp.Any) -> HeadSpec:
    return HeadSpec(fleet="demo", project="demo-project-1", region="us-central1", zone="us-central1-a", **overrides)


@pytest.mark.parametrize("name", sorted(_PRE_REFACTOR_HEAD_SCRIPTS))
def test_head_installer_is_byte_identical_to_the_pre_refactor_rendering(name: str) -> None:
    overrides, expected = _PRE_REFACTOR_HEAD_SCRIPTS[name]

    script = render_head_bootstrap_script(_head_spec(**overrides))

    assert hashlib.sha256(script.encode("utf-8")).hexdigest() == expected


def test_head_team_adds_exactly_one_platform_line_and_the_server_parses_it() -> None:
    plain = render_head_bootstrap_script(_head_spec()).splitlines()
    team = render_head_bootstrap_script(_head_spec(team=True)).splitlines()

    assert [line for line in team if line not in plain] == ["team: true"]
    assert len(team) == len(plain) + 1
    start = team.index("cat >/etc/eyvan/platform.yaml <<'EYVAN_PLATFORM'") + 1
    body = "\n".join(team[start : team.index("EYVAN_PLATFORM", start)])
    parsed = EyvanPlatformConfig.from_document(yaml.safe_load(body))
    assert parsed.team is True
    assert parsed.default_agent_endpoint == "http://127.0.0.1:8090"


def test_head_up_team_flag_reaches_the_dry_run_spec_and_is_absent_by_default() -> None:
    base = [
        "up",
        "--fleet",
        "demo",
        "--project",
        "demo-project-1",
        "--region",
        "us-central1",
        "--zone",
        "us-central1-a",
        "--egress",
        "cloud-nat",
        "--dry-run",
        "--json",
    ]

    plain = CliRunner().invoke(head_commands.head_group, base)
    team = CliRunner().invoke(head_commands.head_group, [*base, "--team"])

    assert plain.exit_code == 0, plain.output
    assert team.exit_code == 0, team.output
    assert "team" not in json.loads(plain.output)["spec"]
    assert json.loads(team.output)["spec"]["team"] is True


# -- rendering and plan -------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        _spec(),
        _spec(package=None, wheel="/tmp/eyvan-0.0.3-py3-none-any.whl", dns_name="eyvan.example.com", port=9443),
    ],
    ids=["index-package", "wheel-and-dns"],
)
def test_the_team_installer_is_valid_bash_and_carries_no_mode_or_credential(spec: TeamUpSpec) -> None:
    script = render_team_bootstrap_script(spec)

    parsed = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True, check=False)

    assert parsed.returncode == 0, parsed.stderr
    assert "TEAM_MODE" not in script
    assert f"ExecStart=/opt/eyvan/venv/bin/eyvan-server --host 0.0.0.0 --port {spec.port}\n" in script
    assert "EnvironmentFile=/etc/eyvan/plane.env" in script
    assert "User=eyvan\n" in script
    assert "chmod 0600 /etc/eyvan/plane.env" in script
    assert "createdb --owner eyvan eyvan" in script
    assert f"head_address={ADDRESS}" in script
    if spec.dns_name:
        assert "subjectAltName=IP:%s,DNS:%s" in script
        assert '-checkhost "$head_dns_name"' in script
        assert "--force-reinstall --no-deps /tmp/eyvan-0.0.3-py3-none-any.whl" in script
    else:
        assert "subjectAltName=IP:%s\\n" in script
        assert "-m pip install --disable-pip-version-check --upgrade eyvan==0.0.3\n" in script
        assert "--force-reinstall" not in script


def test_dry_run_prints_the_plan_redacted_and_changes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(team_up, "filesystem_root", tmp_path)
    monkeypatch.setattr(team_up, "effective_uid", lambda: 1000)

    invoked = CliRunner().invoke(
        team_up.team_group, ["up", "--address", ADDRESS, "--admin", "erfan", "--dry-run", "--json"]
    )

    assert invoked.exit_code == 0, invoked.output
    document = json.loads(invoked.output)
    assert document["dry_run"] is True
    assert document["spec"]["public_url"] == PUBLIC_URL
    assert document["environment"]["EYVAN_DATABASE_URL"] == "***"
    assert document["environment"]["EYVAN_METRICS_TOKEN"] == "***"
    assert document["environment"]["EYVAN_ARTIFACT_TOKEN_KEY"] == "***"
    assert document["environment"]["EYVAN_PUBLIC_URL"] == PUBLIC_URL
    assert not any("TEAM" in key for key in document["environment"])
    assert yaml.safe_load(document["platform_yaml"]) == {"version": 1, "team": True}
    assert any("bootstrap-admin --username erfan --team --json" in step for step in document["steps"])
    assert "#!/bin/bash" in document["script"]
    assert list(tmp_path.iterdir()) == []

    text = CliRunner().invoke(team_up.team_group, ["up", "--address", ADDRESS, "--admin", "erfan", "--dry-run"])
    assert text.exit_code == 0, text.output
    assert "EYVAN_DATABASE_URL=***" in text.output
    assert "changes nothing now" in text.output
    assert list(tmp_path.iterdir()) == []


def test_the_team_command_is_registered_on_the_cli_root() -> None:
    assert cli.main.commands["team"].commands["up"] is team_up.team_up


# -- refusals -----------------------------------------------------------------


def test_a_non_root_run_is_refused_before_anything_happens(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(team_up, "filesystem_root", tmp_path)
    monkeypatch.setattr(team_up, "effective_uid", lambda: 1000)

    invoked = CliRunner().invoke(team_up.team_group, ["up", "--address", ADDRESS, "--admin", "erfan"])

    assert invoked.exit_code != 0
    assert "must run as root" in invoked.output
    assert list(tmp_path.iterdir()) == []


def test_a_host_running_a_head_is_refused(tmp_path: Path) -> None:
    host = FakeHost(tmp_path, _certificate()[0])
    (tmp_path / "etc/eyvan").mkdir(parents=True)
    (tmp_path / "etc/eyvan/head.env").write_text("EYVAN_PUBLIC_URL=https://10.0.0.2:8080\n", encoding="utf-8")

    with pytest.raises(TeamUpError, match="eyvan head up --team"):
        _installer(host).up()

    assert host.calls == []
    assert not (tmp_path / TEAM_ENVIRONMENT_PATH.lstrip("/")).exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"address": "127.0.0.1"}, "not an address other machines can reach"),
        ({"address": "eyvan.example.com"}, "must be an IP address"),
        ({"admin": "Erfan Z"}, "admin user name"),
        ({"dns_name": "bad name"}, "--dns-name"),
        ({"package": None, "wheel": "/tmp/a b.whl"}, "--wheel"),
    ],
)
def test_values_the_script_could_not_carry_safely_are_refused(overrides: dict[str, tp.Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(**overrides)


def test_an_installer_without_an_injected_runner_hits_the_guarded_boundary(tmp_path: Path) -> None:
    installer = TeamUpInstaller(_spec(), root=tmp_path, health=lambda *args: True)

    with pytest.raises(AssertionError, match="must not run host commands"):
        installer.up()


# -- address detection --------------------------------------------------------


def test_gce_metadata_supplies_the_address_name_zone_and_network() -> None:
    answers = {
        "instance/name": "plane-1",
        "instance/network-interfaces/0/access-configs/0/external-ip": ADDRESS,
        "instance/zone": "projects/123/zones/us-east5-a",
        "instance/network-interfaces/0/network": "projects/123/networks/research",
    }

    facts = discover_gce(answers.get)

    assert facts == GceFacts(external_ip=ADDRESS, instance="plane-1", zone="us-east5-a", network="research")
    assert resolve_address(None, facts) == ADDRESS
    assert resolve_address("198.51.100.4", facts) == "198.51.100.4"


def test_off_gce_detection_stops_after_one_miss_and_refuses_without_an_address() -> None:
    asked: list[str] = []

    def fetch(path: str) -> str | None:
        asked.append(path)
        return None

    facts = discover_gce(fetch)

    assert facts == GceFacts()
    assert asked == ["instance/name"]
    with pytest.raises(TeamUpError, match="pass --address"):
        resolve_address(None, facts)


def test_a_gce_vm_without_an_external_ip_is_refused_without_an_address() -> None:
    facts = discover_gce({"instance/name": "plane-1"}.get)

    with pytest.raises(TeamUpError, match="no external IP"):
        resolve_address(None, facts)


# -- platform.yaml ------------------------------------------------------------


def test_the_platform_merge_keeps_agents_and_sets_team() -> None:
    existing = "version: 1\nagents:\n  default:\n    endpoint: http://127.0.0.1:8090\n"

    merged = merged_platform_yaml(existing)

    assert yaml.safe_load(merged) == {
        "version": 1,
        "team": True,
        "agents": {"default": {"endpoint": "http://127.0.0.1:8090"}},
    }
    parsed = EyvanPlatformConfig.from_document(yaml.safe_load(merged))
    assert parsed.team is True
    assert parsed.default_agent_endpoint == "http://127.0.0.1:8090"
    assert yaml.safe_load(merged_platform_yaml("version: 1\nteam: false\n")) == {"version": 1, "team": True}
    assert yaml.safe_load(merged_platform_yaml(None)) == {"version": 1, "team": True}


def test_an_unparseable_platform_file_stops_the_run_before_anything_is_written(tmp_path: Path) -> None:
    host = FakeHost(tmp_path, _certificate()[0])
    platform = tmp_path / "etc/eyvan/platform.yaml"
    platform.parent.mkdir(parents=True)
    platform.write_text("version: 1\nsurprise: 3\n", encoding="utf-8")

    with pytest.raises(TeamUpError, match="unknown Eyvan platform field"):
        _installer(host).up()

    assert host.calls == []
    assert not (tmp_path / TEAM_ENVIRONMENT_PATH.lstrip("/")).exists()
    assert platform.read_text(encoding="utf-8") == "version: 1\nsurprise: 3\n"


# -- the full run -------------------------------------------------------------


def test_up_writes_files_runs_the_steps_in_order_and_returns_the_invite(tmp_path: Path) -> None:
    pem, fingerprint = _certificate()
    host = FakeHost(tmp_path, pem)
    platform = tmp_path / "etc/eyvan/platform.yaml"
    platform.parent.mkdir(parents=True)
    platform.write_text("version: 1\nagents:\n  default:\n    endpoint: http://127.0.0.1:8090\n", encoding="utf-8")

    result = _installer(host, facts=GceFacts(instance="plane-1", zone="us-east5-a", network="research")).up()

    environment_path = tmp_path / TEAM_ENVIRONMENT_PATH.lstrip("/")
    assert stat.S_IMODE(os.stat(environment_path).st_mode) == 0o600
    environment = _environment(tmp_path)
    assert set(environment) == {
        "EYVAN_ARTIFACT_STORE_URI",
        "EYVAN_ARTIFACT_TOKEN_KEY",
        "EYVAN_CONTROL_PLANE_CA_FILE",
        "EYVAN_DATABASE_URL",
        "EYVAN_METRICS_TOKEN",
        "EYVAN_PUBLIC_URL",
        "EYVAN_REPOSITORY_BACKEND",
        "EYVAN_TLS_CERT_FILE",
        "EYVAN_TLS_KEY_FILE",
    }
    assert environment["EYVAN_PUBLIC_URL"] == PUBLIC_URL
    assert environment["EYVAN_REPOSITORY_BACKEND"] == "postgres"
    assert yaml.safe_load(platform.read_text(encoding="utf-8"))["agents"] == {
        "default": {"endpoint": "http://127.0.0.1:8090"}
    }
    assert yaml.safe_load(platform.read_text(encoding="utf-8"))["team"] is True
    installer_script = tmp_path / "opt/eyvan/team-bootstrap.sh"
    assert stat.S_IMODE(os.stat(installer_script).st_mode) == 0o700

    assert [argv for argv, _ in host.calls] == [
        ("/bin/bash", "/opt/eyvan/team-bootstrap.sh"),
        ("/opt/eyvan/venv/bin/eyvan-migrate",),
        ("systemctl", "enable", "eyvan-plane.service"),
        ("systemctl", "restart", "eyvan-plane.service"),
        ("/opt/eyvan/venv/bin/eyvan-server", "bootstrap-admin", "--username", "erfan", "--team", "--json"),
    ]
    # Settings reach migrate and bootstrap-admin through their environment only.
    for argv, env in host.calls:
        if argv[0].endswith(("eyvan-migrate", "eyvan-server")):
            assert env["EYVAN_DATABASE_URL"] == environment["EYVAN_DATABASE_URL"]
        else:
            assert not any(value in env.values() for value in _secret_values(environment))

    assert result.fingerprint == fingerprint
    assert result.invite_url == INVITE_URL
    assert result.login_command == f"eyvan login {PUBLIC_URL} --username erfan --fingerprint {fingerprint}"
    assert "--network research" in result.firewall[0]
    assert "--allow tcp:8443" in result.firewall[0]
    assert "add-tags plane-1 --zone us-east5-a" in result.firewall[1]


@pytest.mark.parametrize("leftover", [False, True])
def test_shared_directories_are_traversable_whatever_the_callers_umask(tmp_path: Path, leftover: bool) -> None:
    """The service user must reach platform.yaml and tls/: a root `umask 077` once left /etc/eyvan 0700.

    The plane then crash-looped on PermissionError reading /etc/eyvan/platform.yaml. ``leftover`` is a
    re-run over that broken directory, which the installer must repair rather than keep.
    """

    pem, _ = _certificate()
    host = FakeHost(tmp_path, pem)
    if leftover:
        for directory in ("etc/eyvan", "opt/eyvan"):
            (tmp_path / directory).mkdir(parents=True)
            os.chmod(tmp_path / directory, 0o700)
    previous = os.umask(0o077)
    try:
        _installer(host).up()
    finally:
        os.umask(previous)

    for directory in ("etc/eyvan", "opt/eyvan"):
        assert stat.S_IMODE(os.stat(tmp_path / directory).st_mode) == 0o755, directory
    assert stat.S_IMODE(os.stat(tmp_path / "etc/eyvan/platform.yaml").st_mode) == 0o644
    assert stat.S_IMODE(os.stat(tmp_path / TEAM_ENVIRONMENT_PATH.lstrip("/")).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(tmp_path / "opt/eyvan/team-bootstrap.sh").st_mode) == 0o700


def test_no_secret_reaches_an_argv_or_the_printed_output(tmp_path: Path, monkeypatch) -> None:
    pem, fingerprint = _certificate()
    host = FakeHost(tmp_path, pem)
    wheel = tmp_path / "eyvan-0.0.3-py3-none-any.whl"
    wheel.write_bytes(b"not really a wheel")
    monkeypatch.setattr(team_up, "filesystem_root", tmp_path)
    monkeypatch.setattr(team_up, "effective_uid", lambda: 0)
    monkeypatch.setattr(team_up, "run_host_command", host)
    monkeypatch.setattr(team_up, "probe_health", lambda address, port, ca, timeout: True)

    invoked = CliRunner().invoke(
        team_up.team_group, ["up", "--address", ADDRESS, "--admin", "erfan", "--wheel", str(wheel)]
    )

    assert invoked.exit_code == 0, invoked.output
    secrets = _secret_values(_environment(tmp_path))
    for argv, _ in host.calls:
        assert not any(value in part for value in secrets for part in argv), argv
    assert not any(value in invoked.output for value in secrets)
    # The invite is the one secret printed, exactly once.
    assert invoked.output.count(INVITE_TOKEN) == 1
    assert fingerprint in invoked.output
    assert f"eyvan login {PUBLIC_URL} --username erfan --fingerprint {fingerprint}" in invoked.output
    assert "gcloud compute firewall-rules create eyvan-team-plane-8443" in invoked.output
    script = (tmp_path / "opt/eyvan/team-bootstrap.sh").read_text(encoding="utf-8")
    assert not any(value in script for value in secrets)


def test_a_rerun_keeps_secrets_and_tolerates_the_existing_admin(tmp_path: Path) -> None:
    pem, _ = _certificate()
    host = FakeHost(tmp_path, pem)

    first = _installer(host).up()
    before = _environment(tmp_path)
    second = _installer(host).up()

    assert _environment(tmp_path) == before
    assert first.invite_url == INVITE_URL
    assert second.invite_url is None
    assert second.admin_existed is True
    rendered = "\n".join(second.render_lines())
    assert INVITE_TOKEN not in rendered
    assert "already has team accounts" in rendered
    assert json.loads(json.dumps(second.as_document()))["invite_url"] is None


def test_a_failing_step_is_reported_with_its_secrets_redacted(tmp_path: Path) -> None:
    pem, _ = _certificate()
    host = FakeHost(tmp_path, pem)
    _installer(host).up()
    password = _secret_values(_environment(tmp_path))[2]
    host.fail["eyvan-migrate"] = subprocess.CompletedProcess(
        ("eyvan-migrate",), 1, "", f"eyvan-migrate: password authentication failed for {password}\n"
    )

    with pytest.raises(TeamUpError) as raised:
        _installer(host).up()

    assert raised.value.step is team_up.TeamUpStep.MIGRATE
    assert password not in str(raised.value)
    assert "***" in str(raised.value)


def test_an_unhealthy_plane_times_out_without_bootstrapping(tmp_path: Path) -> None:
    pem, _ = _certificate()
    host = FakeHost(tmp_path, pem)
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    installer = TeamUpInstaller(
        _spec(),
        runner=host,
        root=tmp_path,
        health=lambda *args: False,
        sleep=sleep,
        clock=lambda: now[0],
        health_timeout_s=10.0,
    )

    with pytest.raises(TeamUpError, match="did not answer /healthz"):
        installer.up()

    assert not host.argv_of("bootstrap-admin")


def test_an_unexpected_bootstrap_failure_is_an_error_not_an_existing_admin(tmp_path: Path) -> None:
    pem, _ = _certificate()
    host = FakeHost(tmp_path, pem)
    host.fail["eyvan-server"] = subprocess.CompletedProcess(
        ("eyvan-server",), 1, "", "bootstrap-admin: team mode is off for this plane\n"
    )

    with pytest.raises(TeamUpError, match="team mode is off"):
        _installer(host).up()
