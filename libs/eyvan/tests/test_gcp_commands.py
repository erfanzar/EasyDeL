# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline CLI attachment contracts: no subprocess or network escapes the doubles."""

from __future__ import annotations

import hashlib
import json
import subprocess
import typing as tp
from pathlib import Path

import httpx
import pytest
import yaml
from click.testing import CliRunner

import eyvan.sdk.cli as cli
import eyvan.sdk.commands.gcp as commands
import eyvan.sdk.commands.gcp_connect as connect
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile, GcpNativeRuntimeProfileStore
from eyvan.capacity.providers.gcp.tpu_nodes import TpuNodeApi, attach_management_uid, parse_tpu_node
from eyvan.capacity.providers.gcp.tpu_ssh import HostTpuIdentity
from eyvan.sdk.commands.common import FleetConfigLoader
from eyvan.sdk.commands.fleets import FleetCommandService
from eyvan.worker.runtime.native_bootstrap import build_native_worker_startup_script, derive_native_worker_token

PROJECT = "example-project"
ROOT_SECRET = "worker-root-" + "r" * 40  # pragma: allowlist secret
PROFILE = GcpNativeRuntimeProfile(PROJECT, "gs://example-bucket/eyvan", "agent-" + "a" * 40, ROOT_SECRET)


def node(*, name: str = "existing", zone: str = "us-east5-a", state: str = "READY", shape: str = "v5p-8") -> dict:
    """Return the provider's node wire document, not a fabricated domain object."""

    return {
        "name": f"projects/{PROJECT}/locations/{zone}/nodes/{name}",
        "id": "12345678",
        "acceleratorType": shape,
        "state": state,
        "health": "HEALTHY",
        "schedulingConfig": {"spot": True},
        "networkConfig": {"network": "default"},
        "networkEndpoints": [{"ipAddress": "10.0.0.2", "accessConfig": {"externalIp": "203.0.113.2"}}],
    }


class FakeCliClient:
    """Return only the fleet API requests used by the real apply service."""

    calls: tp.ClassVar[list] = []
    state = "ready"

    def __init__(self, *args: object, **kwargs: object):
        pass

    def request(self, method: str, path: str, **kwargs: object) -> dict:
        self.calls.append((method, path, kwargs))
        return {"name": path.rsplit("/", 1)[-1], "pools": [], "cells": [{"id": "attached-cell", "state": self.state}]}


class FakeRunner:
    """Record argv separately from sensitive stdin; never execute either."""

    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], str | None]] = []
        self.fail: str | None = None

    def __call__(self, args: tp.Sequence[str], *, stdin_text: str | None, timeout: float):
        assert timeout > 0
        self.calls.append((tuple(args), stdin_text))
        if self.fail and self.fail in " ".join(args):
            # A hostile remote may echo everything it received. Diagnostics must not leak it.
            return subprocess.CompletedProcess(args, 1, stdin_text or "", ROOT_SECRET)
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Forbid real subprocesses and HTTP, including accidental default seams."""

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("real subprocess/network forbidden in attach tests")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(cli, "EyvanClient", FakeCliClient)
    monkeypatch.setattr(commands, "is_interactive", lambda: False)
    monkeypatch.setattr(FleetCommandService, "ensure_local_runtime", lambda *args: None)
    FakeCliClient.calls = []
    FakeCliClient.state = "ready"
    GcpNativeRuntimeProfileStore().save(PROFILE)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Wire API GETs, runner, wheel builder, ingress and worker health doubles."""

    documents = [node()]
    requests: list[httpx.Request] = []
    unreachable: list[str] = []

    def respond(request: httpx.Request):
        assert request.method == "GET"
        requests.append(request)
        if request.url.host == "cloudresourcemanager.googleapis.com":
            return httpx.Response(200, json={"projects": [{"projectId": PROJECT, "lifecycleState": "ACTIVE"}]})
        if request.url.path.endswith("/nodes"):
            return httpx.Response(200, json={"nodes": documents, "unreachable": unreachable})
        matches = [item for item in documents if item["name"] == request.url.path.removeprefix("/v2/")]
        return httpx.Response(200 if matches else 404, json=matches[0] if matches else {})

    api = TpuNodeApi(token_source=lambda: "fake-access-token", transport=httpx.MockTransport(respond))
    runner = FakeRunner()
    wheel = tmp_path / "eyvan-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-double")

    class Builder:
        def build(self):
            return wheel

        def close(self):
            pass

    class Ingress:
        def ensure(self, **kwargs):
            return "eyvan-native-test"

    checked: list[tuple[str, str | None]] = []
    monkeypatch.setattr(commands, "TpuNodeApi", lambda: api)
    monkeypatch.setattr(connect, "_run_gcloud", runner)
    monkeypatch.setattr(connect, "native_worker_wheel_builder", Builder)
    monkeypatch.setattr(connect, "_public_ingress", lambda *args: Ingress())
    monkeypatch.setattr(connect, "_worker_health", lambda endpoint, token: checked.append((endpoint, token)))
    monkeypatch.setattr(connect, "_runtime_needs_deploy", lambda endpoint, **kwargs: None)

    # Hosts reachable over SSH whose own metadata/gcloud describe a TPU node:
    # address -> (identity, node wire document). Anything else is "not a TPU VM".
    ssh_nodes: dict[str, tuple[HostTpuIdentity, dict]] = {}

    class FakeSshReader:
        def __init__(self, destination, options):
            self.destination = destination

        def identity(self):
            entry = ssh_nodes.get(self.destination.host)
            return entry[0] if entry else None

        def get_node(self, project, zone, name):
            entry = ssh_nodes.get(self.destination.host)
            return parse_tpu_node(entry[1], project=project) if entry else None

    monkeypatch.setattr(connect, "_ssh_node_reader", FakeSshReader)
    # A firewall that drops the worker port, as in a project the operator does not own.
    monkeypatch.setattr(connect, "_port_probe", lambda host, port, timeout: "filtered")
    return documents, requests, runner, checked, unreachable, ssh_nodes


def invoke(*args: str, stdin: str | None = None):
    """Invoke the composition root with deterministic explicit plane identity."""

    return CliRunner().invoke(cli.main, ["--user", "tester", "gcp", *args], input=stdin)


def test_browse_json_and_noninteractive_table(harness):
    harness[4].append("us-west4-a")
    result = invoke("browse", "--json")
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["nodes"][0]["capacity_class"] == "spot"
    assert body["unreachable"][PROJECT] == ["us-west4-a"]
    assert body["projects"][0]["project_id"] == PROJECT
    result = invoke("browse", "--project", PROJECT)
    assert result.exit_code == 0, result.output
    for value in ("existing", "us-east5-a", "v5p-8", "READY", "spot", "Hosts", "us-west4-a"):
        assert value in result.output
    assert "Select" not in result.output
    assert not harness[2].calls and not FakeCliClient.calls


def test_browse_interactive_pick_hands_off(monkeypatch, harness):
    monkeypatch.setattr(commands, "is_interactive", lambda: True)
    result = invoke("browse", "--project", PROJECT, stdin="1\ny\n")
    assert result.exit_code == 0, result.output
    assert any(call[0] == "PUT" for call in FakeCliClient.calls)
    assert 'init(fleet="' in result.output


def test_json_browse_never_prompts_even_on_tty(monkeypatch, harness):
    monkeypatch.setattr(commands, "is_interactive", lambda: True)
    result = invoke("browse", "--json")
    assert result.exit_code == 0
    json.loads(result.output)
    assert not harness[2].calls


def test_tpu_dry_run_is_complete_and_inert(harness, tmp_path):
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert {action["step"] for action in plan["actions"]} >= {
        "ingress",
        "tag",
        "wheel_build",
        "wheel_scp",
        "worker_secret",
        "worker_install",
        "worker_health",
        "fleet_write",
        "fleet_apply",
        "fleet_ready",
    }
    assert not harness[2].calls and not harness[3] and not FakeCliClient.calls
    assert all(request.method == "GET" for request in harness[1])
    assert not (tmp_path / ".eyvan" / "attached").exists()
    assert ROOT_SECRET not in result.output
    assert any(action["stdin"] for action in plan["actions"])


def test_execute_spot_yaml_real_loader_secret_only_on_stdin(harness, tmp_path):
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--fleet", "my-attach", "--yes", "--json")
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["fleet"] == "my-attach" and body["ready"] is True
    path = tmp_path / ".eyvan" / "attached" / "my-attach.yaml"
    document = yaml.safe_load(path.read_text())
    assert document["capacity_class"] == "spot"
    assert document["pools"]["v5p-8"]["provider_options"]["tpu_node_id"] == "12345678"
    config = FleetConfigLoader().load(path)
    assert config.pools["v5p-8"].capacity_backend == "gcp-attach"
    token = derive_native_worker_token(
        ROOT_SECRET,
        attach_management_uid(project=PROJECT, zone="us-east5-a", name="existing", node_id="12345678"),
        provider_generation=1,
    )
    assert token and any(stdin == token for _, stdin in harness[2].calls)
    for secret in (ROOT_SECRET, token):
        assert secret not in result.output + path.read_text() + repr([argv for argv, _ in harness[2].calls])
    assert harness[3] == [("http://203.0.113.2:8100", token)]
    # gcloud refuses a multi-worker scp/ssh unless its key is in ssh-agent, so
    # every command targets exactly one worker, and each worker gets the wheel.
    assert not any("--worker=all" in argv for argv, _ in harness[2].calls)
    scp_workers = sorted(
        next(arg for arg in argv if arg.startswith("--worker=")) for argv, _ in harness[2].calls if "scp" in argv
    )
    assert scp_workers == [f"--worker={index}" for index in range(len(harness[3]))]
    assert any(method == "PUT" for method, _, _ in FakeCliClient.calls)


def test_a_refused_ssh_login_stops_connect_before_anything_changes(monkeypatch, harness):
    runner = harness[2]
    ensured: list[dict] = []

    class Ingress:
        def ensure(self, **kwargs):
            ensured.append(kwargs)
            return "eyvan-native-test"

    def refusing(args, *, stdin_text, timeout):
        runner.calls.append((tuple(args), stdin_text))
        if "ssh" in args and args[-1] == "true":
            # Real gcloud order: ssh's refusal, then gcloud's generic exit line.
            return subprocess.CompletedProcess(
                args,
                255,
                "",
                "erfan@203.0.113.2: Permission denied (publickey).\n"
                "ERROR: (gcloud.compute.tpus.tpu-vm.ssh) [/usr/bin/ssh] exited with return code [255].\n",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(connect, "_run_gcloud", refusing)
    monkeypatch.setattr(connect, "_public_ingress", lambda *args: Ingress())
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code != 0
    assert "ssh check" in result.output and "Permission denied" in result.output
    assert "Nothing was changed" in result.output and "refuses this machine's SSH key" in result.output
    # Only the read-only login checks ran: no firewall rule, tag, copy or install.
    assert ensured == []
    assert all(argv[-1] == "true" for argv, _ in runner.calls), runner.calls
    assert not harness[3] and not any(method == "PUT" for method, _, _ in FakeCliClient.calls)


def test_the_preview_lists_a_read_only_ssh_check_first(harness):
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    actions = json.loads(result.output)["actions"]
    assert actions[0]["step"] == "ssh_check" and actions[0]["mutates"] is False
    assert not harness[2].calls


def test_a_multi_host_tpu_copies_the_wheel_one_worker_at_a_time() -> None:
    from eyvan.capacity.providers.gcp.tpu_nodes import parse_tpu_node
    from eyvan.sdk.commands.gcp_host_access import GcloudHostAccess

    node = parse_tpu_node(
        {
            "name": f"projects/{PROJECT}/locations/us-central1-a/nodes/big",
            "id": "1",
            "acceleratorType": "v5p-32",
            "state": "READY",
            "networkEndpoints": [{"ipAddress": f"10.0.0.{index}"} for index in range(2, 6)],
        },
        project=PROJECT,
    )
    copies = GcloudHostAccess(node, internal=False, runner=lambda *a, **k: None).copy_argvs("/w.whl", "/tmp")
    assert [argv[-2] for argv in copies] == ["--worker=0", "--worker=1", "--worker=2", "--worker=3"]
    assert all("--worker=all" not in argv for argv in copies)


def test_connect_proceeds_without_yes_and_still_accepts_it(harness):
    """Connecting is the command's purpose; --dry-run is how to look first, so nothing prompts."""

    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ready"] is True
    assert harness[2].calls
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes", "--json")
    assert result.exit_code == 0, result.output


def test_runtime_comes_from_the_defaults_profile_plus_flags(harness, tmp_path):
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "project": PROJECT,
                "fleet_prefix": "research",
                "runtime_python": "3.12",
                "runtime_init_commands": ["echo from-profile"],
            }
        )
    )
    result = invoke(
        "connect",
        "--tpu",
        "existing",
        "--project",
        PROJECT,
        "--defaults",
        str(defaults),
        "--init-command",
        "echo from-flag",
        "--dry-run",
        "--json",
    )
    assert result.exit_code == 0, result.output
    runtime = yaml.safe_load(json.loads(result.stdout)["fleet_yaml"])["pools"]["v5p-8"]["runtime"]
    assert runtime == {"init_commands": ["echo from-profile", "echo from-flag"], "python": "3.12"}
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--python", "3.11", "--dry-run", "--json")
    runtime = yaml.safe_load(json.loads(result.stdout)["fleet_yaml"])["pools"]["v5p-8"]["runtime"]
    assert runtime["python"] == "3.11"


def test_runtime_setup_reaches_every_hosts_install_script(harness):
    result = invoke(
        "connect", "--tpu", "existing", "--project", PROJECT, "--python", "3.12", "--init-command", "echo marker-xyz"
    )
    assert result.exit_code == 0, result.output
    scripts = [stdin for argv, stdin in harness[2].calls if stdin and "bash -s" in " ".join(argv)]
    assert scripts and all("echo marker-xyz" in script and "3.12" in script for script in scripts)


@pytest.mark.parametrize("state", ["DELETING", "PREEMPTED", "STATE_UNSPECIFIED"])
def test_non_ready_node_refused(harness, state):
    harness[0][0]["state"] = state
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--dry-run")
    assert result.exit_code != 0 and state in result.output and "resolve" in result.output
    assert not harness[2].calls


def test_unknown_shape_refused(harness):
    harness[0][0]["acceleratorType"] = "future-1024"
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--dry-run")
    assert result.exit_code != 0 and "catalog" in result.output and "future-1024" in result.output


def test_ambiguous_node_names_show_zones(harness):
    harness[0].append(node(zone="us-central2-b"))
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--dry-run")
    assert result.exit_code != 0 and "--zone" in result.output
    assert "us-east5-a" in result.output and "us-central2-b" in result.output
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--zone", "us-east5-a", "--dry-run")
    assert result.exit_code == 0, result.output


def test_readiness_timeout_names_cell_and_state(harness):
    FakeCliClient.state = "registering"
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes", "--timeout", "0.001")
    assert result.exit_code != 0
    assert "fleet ready" in result.output and "timed out" in result.output
    assert "attached-cell" in result.output and "registering" in result.output


@pytest.mark.parametrize(
    "fragment,step", [("update", "tag"), ("scp", "wheel scp"), ("cat >", "worker secret"), ("bash -s", "worker install")]
)
def test_command_failure_names_step_and_hides_echoed_secrets(harness, fragment, step):
    harness[2].fail = fragment
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code != 0 and step in result.output
    assert ROOT_SECRET not in result.output
    for _, stdin in harness[2].calls:
        if stdin and len(stdin) == 64:
            assert stdin not in result.output


def test_internal_mode_skips_ingress_and_uses_internal_ip(monkeypatch, harness):
    def forbidden(*args, **kwargs):
        raise AssertionError("internal mode must not reconcile public ingress")

    monkeypatch.setattr(connect, "_public_ingress", forbidden)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--endpoint-mode", "internal", "--yes")
    assert result.exit_code == 0, result.output
    assert harness[3][0][0] == "http://10.0.0.2:8100"
    assert not any("update" in argv for argv, _ in harness[2].calls)


def test_ssh_dry_run_yaml_is_explicit_reserved_cpu(harness):
    result = invoke(
        "connect", "--ssh", "researcher@host.example", "--cpus", "8", "--memory-gib", "32", "--dry-run", "--json"
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    config = yaml.safe_load(plan["fleet_yaml"])
    assert config["provider"]["type"] == "onprem-ssh"
    assert config["capacity_class"] == "reserved"
    assert config["pools"]["host"]["provider_options"]["hosts"] == ["researcher@host.example"]
    assert "prepared" in " ".join(plan["limitations"])
    health = [action for action in plan["actions"] if action["step"] == "worker_health"]
    assert health[0]["argv"] == ["GET", "http://host.example:8100/healthz"]
    assert not harness[2].calls and not harness[1] and not FakeCliClient.calls


def test_ssh_execute_uses_prepared_host_only(harness, tmp_path):
    result = invoke(
        "connect",
        "--ssh",
        "researcher@host.example",
        "--cpus",
        "8",
        "--memory-gib",
        "32",
        "--fleet",
        "ssh-host",
        "--yes",
    )
    assert result.exit_code == 0, result.output
    config = FleetConfigLoader().load(tmp_path / ".eyvan" / "attached" / "ssh-host.yaml")
    assert config.provider.type == "onprem-ssh"
    assert not harness[2].calls
    assert harness[3] == [("http://host.example:8100", None)]


@pytest.mark.parametrize("destination", ["-oProxyCommand=bad", "user@host;bad", "@host", "user@../host"])
def test_ssh_invalid_destination_refused(harness, destination):
    result = invoke("connect", "--ssh", destination, "--cpus", "1", "--memory-gib", "2", "--dry-run")
    assert result.exit_code != 0 and not harness[2].calls


@pytest.mark.parametrize(
    "package,digest",
    [
        # Script digests, not credentials.
        # pragma: allowlist nextline secret
        ("eyvan", "2b9fda65c148faee8e3a94cb9bdcde3aa57e1108858ba42f81b4d7b3ef2c41d9"),
        # Force-reinstall local wheels so same-version rebuilds replace installed bytes.
        # pragma: allowlist nextline secret
        ("/tmp/eyvan-1.0-py3-none-any.whl", "083356c5956df7b544ac797ed24f03d099f2376a4c0db892ba4de503e0dfd72d"),
    ],
)
def test_bootstrap_default_bytes_match_pre_edit_capture(package, digest):
    script = build_native_worker_startup_script(
        package=package,
        port=8100,
        state_dir="/var/lib/eyvan/native-worker",
        worker_secret_resource="projects/example/secrets/worker/versions/1",  # pragma: allowlist secret
    )
    assert hashlib.sha256(script.encode()).hexdigest() == digest


def test_bootstrap_file_secret_is_root_only_and_no_gcloud():
    script = build_native_worker_startup_script(
        package="/tmp/eyvan-1.0-py3-none-any.whl",
        port=8100,
        state_dir="/var/lib/eyvan/native-worker",
        worker_secret_file="/etc/eyvan-native/attached-worker-secret",
    )
    assert "gcloud" not in script
    assert (
        "install -o root -g root -m 0600 /etc/eyvan-native/attached-worker-secret /run/eyvan-native/worker-secret"
        in script
    )
    assert "/opt/eyvan-native/venv" in script and "Python 3.11" in script
    assert "eyvan-native-worker.service" in script


def test_bootstrap_refuses_two_secret_sources():
    with pytest.raises(ValueError, match="exactly one"):
        build_native_worker_startup_script(
            package="eyvan",
            port=8100,
            state_dir="/var/lib/eyvan/native-worker",
            worker_secret_resource="projects/example/secrets/worker/versions/1",  # pragma: allowlist secret
            worker_secret_file="/etc/eyvan-native/worker-secret",
        )


@pytest.mark.parametrize(
    "boundary", ["ingress", "wheel_build", "worker_health", "fleet_write", "fleet_apply", "fleet_ready", "credential"]
)
def test_injected_failure_always_names_step(monkeypatch, harness, boundary):
    def fail(*args, **kwargs):
        raise ValueError("injected failure")

    if boundary == "ingress":
        monkeypatch.setattr(connect, "_public_ingress", fail)
    elif boundary == "wheel_build":
        monkeypatch.setattr(connect, "native_worker_wheel_builder", fail)
    elif boundary == "worker_health":
        monkeypatch.setattr(connect, "_worker_health", fail)
    elif boundary == "fleet_write":
        monkeypatch.setattr(connect.GcpConnector, "_write_fleet", fail)
    elif boundary == "fleet_apply":
        monkeypatch.setattr(FleetCommandService, "apply", fail)
    elif boundary == "fleet_ready":
        original = FakeCliClient.request

        def request(self, method, *args, **kwargs):
            if method == "GET":
                fail()
            return original(self, method, *args, **kwargs)

        monkeypatch.setattr(FakeCliClient, "request", request)
    else:
        monkeypatch.setattr(GcpNativeRuntimeProfileStore, "load", fail)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes", "--timeout", "0.001")
    assert result.exit_code != 0 and boundary.replace("_", " ") in result.output
    assert ROOT_SECRET not in result.output


def test_multi_host_install_and_authenticated_health_on_every_host(harness):
    harness[0][0]["networkEndpoints"].append({"ipAddress": "10.0.0.3", "accessConfig": {"externalIp": "203.0.113.3"}})
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code == 0, result.output
    assert [endpoint for endpoint, _ in harness[3]] == ["http://203.0.113.2:8100", "http://203.0.113.3:8100"]
    stdin_workers = [argv for argv, stdin in harness[2].calls if stdin]
    assert len(stdin_workers) == 4
    assert sum("--worker=0" in argv for argv in stdin_workers) == 2
    assert sum("--worker=1" in argv for argv in stdin_workers) == 2


def test_unknown_zone_is_not_silently_empty(harness):
    harness[4].append("us-west4-a")
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--dry-run")
    assert result.exit_code != 0 and "unreachable" in result.output and "--zone" in result.output
    assert not harness[2].calls


def test_conflicting_config_is_preserved_before_worker_mutation(harness, tmp_path):
    path = tmp_path / ".eyvan" / "attached" / "my-attach.yaml"
    path.parent.mkdir()
    path.write_text("operator data\n")
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--fleet", "my-attach", "--yes")
    assert result.exit_code != 0 and "fleet write" in result.output and "different attachment" in result.output
    assert path.read_text() == "operator data\n" and not harness[2].calls


def test_rerun_with_new_runtime_updates_its_own_fleet_file(harness, tmp_path):
    """The same node's file is updated in place; only a different attachment is refused."""

    path = tmp_path / ".eyvan" / "attached" / "my-attach.yaml"
    first = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--fleet", "my-attach", "--json")
    assert first.exit_code == 0, first.output
    result = invoke(
        "connect", "--tpu", "existing", "--project", PROJECT, "--fleet", "my-attach", "--init-command", "echo new"
    )
    assert result.exit_code == 0, result.output
    document = yaml.safe_load(path.read_text())
    assert document["pools"]["v5p-8"]["runtime"]["init_commands"] == ["echo new"]
    assert path.stat().st_mode & 0o777 == 0o600
    harness[0][0]["id"] = "87654321"
    other = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--fleet", "my-attach")
    assert other.exit_code != 0 and "different attachment" in other.output


def test_internal_ssh_and_scp_use_internal_ip(harness):
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--endpoint-mode", "internal", "--yes")
    assert result.exit_code == 0, result.output
    assert all("--internal-ip" in argv for argv, _ in harness[2].calls if "ssh" in argv or "scp" in argv)


def test_missing_profile_is_actionable(monkeypatch, harness):
    def missing(*args):
        raise FileNotFoundError()

    monkeypatch.setattr(GcpNativeRuntimeProfileStore, "load", missing)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code != 0 and "credential" in result.output and "eyvan runtime init" in result.output
    assert not harness[2].calls


def test_node_recreated_after_confirmation_never_receives_secret(monkeypatch, harness):
    original = commands.GcpConnector.plan_tpu

    def plan_then_replace(self, *args, **kwargs):
        plan = original(self, *args, **kwargs)
        harness[0][0]["id"] = "87654321"
        return plan

    monkeypatch.setattr(commands.GcpConnector, "plan_tpu", plan_then_replace)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code != 0 and "recreated" in result.output
    assert not harness[2].calls


def test_ssh_with_a_tpu_ip_attaches_that_tpu_without_typed_specs(harness):
    """`--ssh <ip>` of one of the operator's TPU VMs needs nothing else typed."""

    result = invoke("connect", "--ssh", "203.0.113.2", "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    config = yaml.safe_load(plan["fleet_yaml"])
    pool = next(iter(config["pools"].values()))
    assert pool["capacity_backend"] == "gcp-attach"
    assert pool["accelerator_type"] == "v5p-8"
    assert pool["provider_options"]["tpu_node"] == "existing"
    assert pool["provider_options"]["endpoint_mode"] == "public"
    assert config["capacity_class"] == "spot"
    assert all(request.method == "GET" for request in harness[1])
    assert not harness[2].calls and not FakeCliClient.calls


def test_ssh_with_a_tpu_internal_ip_uses_internal_endpoints(harness):
    result = invoke("connect", "--ssh", "10.0.0.2", "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(json.loads(result.stdout)["fleet_yaml"])
    assert next(iter(config["pools"].values()))["provider_options"]["endpoint_mode"] == "internal"


def test_ssh_tpu_ip_refuses_host_resource_overrides(harness):
    result = invoke("connect", "--ssh", "203.0.113.2", "--cpus", "4", "--dry-run")
    assert result.exit_code != 0 and "TPU VM" in result.output


def test_ssh_bare_ip_that_is_no_tpu_is_a_plain_host_with_the_ssh_config_user(monkeypatch, harness):
    """No user is invented: a bare host is passed to ssh as-is, so ~/.ssh/config decides."""

    probed: list[str] = []

    def probe(host: str) -> tuple[float, float]:
        probed.append(host)
        return 4.0, 8.0

    monkeypatch.setattr(connect, "_probe_host_resources", probe)
    result = invoke("connect", "--ssh", "198.51.100.9", "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(json.loads(result.stdout)["fleet_yaml"])
    assert probed == ["198.51.100.9"]
    assert config["pools"]["host"]["provider_options"]["hosts"] == ["198.51.100.9"]


def test_ssh_plain_host_reads_cpu_and_memory_over_ssh(monkeypatch, harness):
    probed: list[str] = []

    def probe(host: str) -> tuple[float, float]:
        probed.append(host)
        return 16.0, 62.7

    monkeypatch.setattr(connect, "_probe_host_resources", probe)
    result = invoke("connect", "--ssh", "researcher@host.example", "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    resources = yaml.safe_load(json.loads(result.stdout)["fleet_yaml"])["pools"]["host"]["resources"]
    assert probed == ["researcher@host.example"]
    assert resources == {"CPU": 16.0, "MEMORY_GIB": 62.7}


def test_ssh_probe_failure_names_its_step(monkeypatch, harness):
    def probe(host: str) -> tuple[float, float]:
        raise ValueError("ssh exited 255: Permission denied (publickey)")

    monkeypatch.setattr(connect, "_probe_host_resources", probe)
    result = invoke("connect", "--ssh", "researcher@host.example", "--dry-run")
    assert result.exit_code != 0
    assert "probe failed" in result.output and "--cpus" in result.output


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8\nMemTotal:       32867872 kB\n", (8.0, 31.3)),
        ("  208\nMemTotal: 475000000 kB", (208.0, 452.9)),
    ],
)
def test_host_resource_parsing_rounds_memory_down(text, expected):
    assert connect.parse_host_resources(text) == expected


@pytest.mark.parametrize("text", ["", "8\n", "0\nMemTotal: 10 kB", "x\nMemTotal: 10 kB", "8\nMemFree: 10 kB"])
def test_host_resource_parsing_refuses_unusable_output(text):
    with pytest.raises(ValueError):
        connect.parse_host_resources(text)


def test_tpu_names_with_underscores_are_accepted(harness):
    """Real node names such as ``lab_spot_m`` contain underscores."""

    harness[0][0]["name"] = f"projects/{PROJECT}/locations/us-east5-a/nodes/lab_spot_m"
    result = invoke("connect", "--tpu", "lab_spot_m", "--project", PROJECT, "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(json.loads(result.stdout)["fleet_yaml"])
    assert next(iter(config["pools"].values()))["provider_options"]["tpu_node"] == "lab_spot_m"
    assert config["name"].startswith("tpu-lab-spot-m-")


def test_stale_runtime_is_redeployed_before_the_tpu_is_touched(monkeypatch, harness):
    """An agent without gcp-attach refuses every cell; fix that before installing anything."""

    order: list[str] = []
    answers = iter(["the local cell agent runs an Eyvan build without attached-TPU support", None])

    def check(endpoint):
        order.append("check")
        return next(answers)

    def deploy(endpoint):
        order.append("deploy")
        assert not harness[2].calls, "runtime must be fixed before any gcloud mutation"

    monkeypatch.setattr(connect, "_runtime_needs_deploy", check)
    monkeypatch.setattr(connect, "_deploy_runtime", deploy)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code == 0, result.output
    assert order == ["check", "deploy", "check"]
    assert harness[2].calls


def test_failed_runtime_redeploy_stops_before_any_mutation(monkeypatch, harness):
    monkeypatch.setattr(connect, "_runtime_needs_deploy", lambda endpoint: "agent too old")

    def deploy(endpoint):
        raise RuntimeError("could not stage wheel")

    monkeypatch.setattr(connect, "_deploy_runtime", deploy)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code != 0
    assert "runtime" in result.output and "eyvan runtime deploy --yes" in result.output
    assert not harness[2].calls and not FakeCliClient.calls


def test_refused_fleet_fails_fast_and_is_released(monkeypatch, harness):
    """A cell refused with a configuration error ends the wait at once, and the fleet is released."""

    def refused(self, method, path, **kwargs):
        FakeCliClient.calls.append((method, path, kwargs))
        return {
            "name": path.rsplit("/", 1)[-1],
            "pools": [],
            "cells": [
                {
                    "id": "attached-cell",
                    "state": "unavailable",
                    "unavailable_reason": "provider_failure",
                    "unavailable_detail": (
                        "invalid configuration: GCP driver cannot satisfy capacity backend 'gcp-attach'"
                    ),
                    "provider_failure_code": None,
                }
            ],
        }

    monkeypatch.setattr(FakeCliClient, "request", refused)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes", "--timeout", "600")
    assert result.exit_code != 0
    assert "refused" in result.output and "gcp-attach" in result.output and "released" in result.output
    assert any(method == "DELETE" for method, _, _ in FakeCliClient.calls)
    # One read before applying (history), one readiness read that ends the wait.
    assert sum(1 for method, path, _ in FakeCliClient.calls if method == "GET" and "/fleets/" in path) == 2


def test_replaced_cells_end_the_wait(monkeypatch, harness):
    applied = {"done": False}

    def churn(self, method, path, **kwargs):
        FakeCliClient.calls.append((method, path, kwargs))
        if method == "PUT":
            applied["done"] = True
        if not applied["done"]:
            return {"name": "x", "pools": [], "cells": []}
        cells = [{"id": f"cell-{index}", "state": "deleted", "unavailable_reason": "ABSENT"} for index in range(3)]
        return {"name": "x", "pools": [], "cells": [*cells, {"id": "cell-3", "state": "registering"}]}

    monkeypatch.setattr(FakeCliClient, "request", churn)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes", "--timeout", "600")
    assert result.exit_code != 0 and "3 cells were created and lost" in result.output


def test_dead_cells_from_an_earlier_attempt_are_ignored(monkeypatch, harness):
    """A rerun under the same fleet name must not count the previous run's dead cells."""

    history = [{"id": f"old-{index}", "state": "deleted", "unavailable_reason": "ABSENT"} for index in range(14)]

    def rerun(self, method, path, **kwargs):
        FakeCliClient.calls.append((method, path, kwargs))
        return {"name": "x", "pools": [], "cells": [*history, {"id": "new-cell", "state": "ready"}]}

    monkeypatch.setattr(FakeCliClient, "request", rerun)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["cells"] == ["new-cell"]
    assert not any(method == "DELETE" for method, _, _ in FakeCliClient.calls)


def test_transient_cell_failure_keeps_waiting(monkeypatch, harness):
    calls = {"n": 0}

    def flaky(self, method, path, **kwargs):
        FakeCliClient.calls.append((method, path, kwargs))
        if method == "GET" and "/fleets/" in path:
            calls["n"] += 1
            if calls["n"] == 1:
                cell = {
                    "id": "c",
                    "state": "unavailable",
                    "unavailable_reason": "provider_failure",
                    "unavailable_detail": "TPU API unreachable",
                    "provider_failure_code": "provider_unavailable",
                }
                return {"name": "x", "pools": [], "cells": [cell]}
        return {"name": "x", "pools": [], "cells": [{"id": "c", "state": "ready"}]}

    monkeypatch.setattr(FakeCliClient, "request", flaky)
    monkeypatch.setattr(connect.time, "sleep", lambda seconds: None)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code == 0, result.output
    assert calls["n"] == 2


def test_human_output_shows_each_step_and_a_result_panel(harness):
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code == 0, result.output
    for label in ("Checking the local Eyvan runtime", "Building the Eyvan worker package", "Applying fleet"):
        assert f"✓ {label}" in result.output
    assert "Connected" in result.output and 'init(fleet="' in result.output


def test_tag_already_present_is_skipped(harness):
    harness[0][0]["tags"] = ["eyvan-native-test"]
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code == 0, result.output
    assert not any("update" in argv for argv, _ in harness[2].calls)


def test_tag_no_change_error_is_tolerated_and_other_gcloud_errors_are_shown(monkeypatch, harness):
    messages = iter(
        [
            "ERROR: (gcloud.compute.tpus.tpu-vm.update) No fields would change as a result of this update",
            "ERROR: (gcloud.compute.tpus.tpu-vm.update) PERMISSION_DENIED: tpu.nodes.update",
        ]
    )
    original = harness[2]

    def runner(args, *, stdin_text, timeout):
        if "update" in args:
            original.calls.append((tuple(args), stdin_text))
            return subprocess.CompletedProcess(args, 1, "", next(messages))
        return original(args, stdin_text=stdin_text, timeout=timeout)

    monkeypatch.setattr(connect, "_run_gcloud", runner)
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code == 0, result.output
    result = invoke("connect", "--tpu", "existing", "--project", PROJECT, "--yes")
    assert result.exit_code != 0 and "PERMISSION_DENIED" in result.output


SSH_ENTRY = "198.51.100.20"


def _ssh_tpu_document(hosts: int = 2) -> dict:
    """A node in a project this machine's credential cannot list, as the host's gcloud returns it."""

    return {
        "name": "projects/other-project/locations/europe-west4-b/nodes/shared_v5p",
        "id": "999",
        "acceleratorType": "v5p-16",
        "state": "READY",
        "health": "HEALTHY",
        "schedulingConfig": {"reserved": True},
        "networkEndpoints": [
            {"ipAddress": f"10.9.0.{index + 2}", "accessConfig": {"externalIp": f"198.51.100.{index + 20}"}}
            for index in range(hosts)
        ],
    }


def _ssh_identity() -> HostTpuIdentity:
    return HostTpuIdentity(
        project="other-project",
        zone="europe-west4-b",
        name="shared_v5p",
        accelerator_type="v5p-16",
        worker_id=0,
        preemptible=False,
    )


def test_ssh_to_a_tpu_outside_your_projects_is_read_from_the_host_and_tunnelled(harness):
    harness[5][SSH_ENTRY] = (_ssh_identity(), _ssh_tpu_document())
    result = invoke("connect", "--ssh", SSH_ENTRY, "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    config = yaml.safe_load(plan["fleet_yaml"])
    pool = config["pools"]["v5p-16"]
    assert pool["capacity_backend"] == "gcp-attach"
    assert config["capacity_class"] == "reserved"
    assert config["provider"]["project"] == "other-project"
    options = pool["provider_options"]
    assert options["endpoint_mode"] == "tunnel"
    assert options["ssh_entry"] == SSH_ENTRY
    assert options["worker_port"] == 8100
    ports = options["tunnel_ports"]
    assert len(ports) == 2 and len(set(ports)) == 2
    assert plan["endpoints"] == [f"http://127.0.0.1:{port}" for port in ports]
    steps = [action["step"] for action in plan["actions"]]
    # Somebody else's network: no firewall rule, no tag, a tunnel instead.
    assert "ingress" not in steps and "tag" not in steps and "tunnel" in steps
    installs = [action["argv"] for action in plan["actions"] if action["step"] == "worker_install"]
    assert installs[0][0] == "ssh" and "-J" not in installs[0] and SSH_ENTRY in installs[0]
    assert "-J" in installs[1] and SSH_ENTRY in installs[1] and "10.9.0.3" in installs[1]
    assert not harness[2].calls and not FakeCliClient.calls


def test_ssh_tpu_whose_worker_port_answers_is_reached_directly(monkeypatch, harness):
    harness[5][SSH_ENTRY] = (_ssh_identity(), _ssh_tpu_document())
    monkeypatch.setattr(connect, "_port_probe", lambda host, port, timeout: "refused")
    result = invoke("connect", "--ssh", SSH_ENTRY, "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    options = yaml.safe_load(plan["fleet_yaml"])["pools"]["v5p-16"]["provider_options"]
    assert options["endpoint_mode"] == "public" and "tunnel_ports" not in options
    assert plan["endpoints"] == ["http://198.51.100.20:8100", "http://198.51.100.21:8100"]
    assert "ingress" not in [action["step"] for action in plan["actions"]]


def test_ssh_tpu_refuses_host_resource_overrides(harness):
    harness[5][SSH_ENTRY] = (_ssh_identity(), _ssh_tpu_document())
    result = invoke("connect", "--ssh", SSH_ENTRY, "--cpus", "4", "--dry-run")
    assert result.exit_code != 0 and "TPU VM" in result.output


def test_ssh_tpu_execute_installs_over_plain_ssh_and_opens_the_agents_tunnel(monkeypatch, harness, tmp_path):
    harness[5][SSH_ENTRY] = (_ssh_identity(), _ssh_tpu_document())
    ran: list[tuple[tuple[str, ...], str | None]] = []

    def run(args, *, stdin_text, timeout):
        ran.append((tuple(args), stdin_text))
        return subprocess.CompletedProcess(args, 0, "", "")

    opened: list = []

    class Tunnels:
        def ensure(self, spec):
            opened.append(spec)
            return True

    monkeypatch.setattr(connect, "_run_ssh", run)
    monkeypatch.setattr(connect, "SshTunnelManager", Tunnels)
    result = invoke("connect", "--ssh", SSH_ENTRY, "--fleet", "shared", "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ready"] is True
    uid = attach_management_uid(project="other-project", zone="europe-west4-b", name="shared_v5p", node_id="999")
    token = derive_native_worker_token(ROOT_SECRET, uid, provider_generation=1)
    assert len([argv for argv, _ in ran if argv[0] == "scp"]) == 2
    delivered = [argv for argv, stdin in ran if stdin == token]
    assert len(delivered) == 2
    assert all(token not in " ".join(argv) for argv, _ in ran)
    assert not harness[2].calls, "a node reached over SSH must never be driven through gcloud"
    [spec] = opened
    assert spec.key == f"attach-{uid}" and spec.entry.target == SSH_ENTRY
    assert [(forward.remote_host, forward.remote_port) for forward in spec.forwards] == [
        ("10.9.0.2", 8100),
        ("10.9.0.3", 8100),
    ]
    assert [endpoint for endpoint, _ in harness[3]] == [
        f"http://127.0.0.1:{forward.local_port}" for forward in spec.forwards
    ]
    config = FleetConfigLoader().load(tmp_path / ".eyvan" / "attached" / "shared.yaml")
    assert config.pools["v5p-16"].capacity_backend == "gcp-attach"


def test_ssh_tpu_rerun_reuses_its_tunnel_ports(harness, tmp_path):
    harness[5][SSH_ENTRY] = (_ssh_identity(), _ssh_tpu_document())
    path = tmp_path / ".eyvan" / "attached" / "shared.yaml"
    first = json.loads(invoke("connect", "--ssh", SSH_ENTRY, "--fleet", "shared", "--dry-run", "--json").stdout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(first["fleet_yaml"])
    second = json.loads(invoke("connect", "--ssh", SSH_ENTRY, "--fleet", "shared", "--dry-run", "--json").stdout)
    assert second["fleet_yaml"] == first["fleet_yaml"]


def test_ssh_install_failure_names_the_host(monkeypatch, harness):
    harness[5][SSH_ENTRY] = (_ssh_identity(), _ssh_tpu_document())

    def run(args, *, stdin_text, timeout):
        if args[0] == "scp" and "10.9.0.3" in " ".join(args):
            return subprocess.CompletedProcess(args, 1, "", "scp: /tmp/: Permission denied\n" + ROOT_SECRET)
        return subprocess.CompletedProcess(args, 0, "", "")

    class Tunnels:
        def ensure(self, spec):
            raise AssertionError("no tunnel after a failed install")

    monkeypatch.setattr(connect, "_run_ssh", run)
    monkeypatch.setattr(connect, "SshTunnelManager", Tunnels)
    result = invoke("connect", "--ssh", SSH_ENTRY, "--yes")
    assert result.exit_code != 0
    assert "wheel scp" in result.output and "host 1" in result.output and "Permission denied" in result.output
    assert ROOT_SECRET not in result.output


def test_health_transport_authentication_and_response_validation(monkeypatch):
    real_client = httpx.Client
    requests: list[httpx.Request] = []
    payload: dict = {"status": "ok", "runtime": "native-worker"}

    def respond(request):
        assert request.method == "GET"
        requests.append(request)
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    connect._worker_health("http://203.0.113.2:8100", "derived-token")
    assert requests[-1].url.path == "/api/v1/health"
    assert requests[-1].headers["Authorization"] == "Bearer derived-token"
    connect._worker_health("http://host.example:8100", None)
    assert requests[-1].url.path == "/healthz" and "Authorization" not in requests[-1].headers
    payload["runtime"] = "other-service"
    with pytest.raises(ValueError, match="native worker"):
        connect._worker_health("http://host.example:8100", None)
