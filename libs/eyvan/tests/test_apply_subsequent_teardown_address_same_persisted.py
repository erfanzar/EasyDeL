# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply and subsequent teardown must address the same persisted plane."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

import eyvan.sdk.cli as cli
import eyvan.sdk.client.transport as transport_module
from eyvan.sdk.contexts import EyvanContext, FileContextStore

_FLEET_YAML = """\
version: 1
name: dev-fleet
provider:
  type: gcp
  project: gcp-project
  region: us-central1
  zone: us-central1-a
agent_endpoint: https://agent.example.invalid
node_types:
  cpu-workers:
    kind: cpu
    machine_type: n2-standard-8
    resources:
      cpu: 8
      memory_gib: 32
    min_workers: 0
    max_workers: 1
    provider_options:
      service_account: eyvan-worker@gcp-project.iam.gserviceaccount.com
      bootstrap_egress: private-google-access
"""


@pytest.mark.parametrize(
    ("other_url", "override"),
    [
        ("https://other.example.invalid", None),
        ("http://127.0.0.1:8080", None),
        ("https://other.example.invalid", "https://other.example.invalid"),
        ("https://dev.example.invalid", None),
        ("https://dev.example.invalid/", None),
    ],
    ids=[
        "saved-fleet-endpoint",
        "refuse-before-loopback-runtime",
        "explicit-endpoint-override",
        "equal-context-endpoints",
        "normalized-equal-context-endpoints",
    ],
)
def test_apply_persisted_context_and_teardown_use_one_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, other_url: str, override: str | None
) -> None:
    """Exercise real CLI, adapter and HTTPX URL routing without network contact."""

    saved_url = "https://dev.example.invalid"
    config_path = tmp_path / "contexts.json"
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(config_path))
    for name in ("EYVAN_URL", "EYVAN_TOKEN", "EYVAN_ORG", "EYVAN_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "discover_node_context", lambda: None)
    monkeypatch.setattr(cli, "discover_node_token_or_warn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport_module, "control_plane_httpx_options", lambda: {})

    import eyvan.sdk.commands.fleets as fleet_commands

    runtime_calls: list[str] = []

    def unexpected_runtime(*_args: object, **_kwargs: object) -> None:
        runtime_calls.append("runtime")
        raise AssertionError("endpoint refusal must precede runtime preparation")

    monkeypatch.setattr(fleet_commands.FleetCommandService, "ensure_local_runtime", unexpected_runtime)

    store = FileContextStore(config_path)
    for fleet, url in (("dev-fleet", saved_url), ("other-fleet", other_url)):
        store.activate(EyvanContext(fleet=fleet, base_url=url, org_id="acme", project_id="training"))
    fleet_path = tmp_path / "fleet.yaml"
    fleet_path.write_text(_FLEET_YAML, encoding="utf-8")

    routed_requests: list[tuple[str, str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        routed_requests.append((request.method, str(request.url)))
        assert request.headers["X-Eyvan-Org"] == "acme"
        assert request.headers["X-Eyvan-Project"] == "training"
        assert request.method in {"PUT", "DELETE"}
        return httpx.Response(200, json=[])

    original_client = httpx.Client
    clients: list[httpx.Client] = []

    def local_client(**kwargs: object) -> httpx.Client:
        client = original_client(**kwargs, transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(transport_module.httpx, "Client", local_client)
    runner = CliRunner()
    flags = ["--user", "alice", "--role", "fleet_admin"]
    if override is not None:
        flags.extend(["--url", override])
    try:
        before = config_path.read_bytes()
        applied = runner.invoke(cli.main, [*flags, "up", str(fleet_path), "--json"])
        if override is None and other_url.rstrip("/") != saved_url:
            assert applied.exit_code != 0, applied.output
            assert "saved control-plane endpoint" in applied.output
            assert "eyvan use dev-fleet" in applied.output
            assert "--url" in applied.output
            assert routed_requests == []
            assert runtime_calls == []
            assert config_path.read_bytes() == before
            assert store.load().active_context == "other-fleet"
            selected = runner.invoke(cli.main, [*flags, "use", "dev-fleet"])
            assert selected.exit_code == 0, selected.output
            assert routed_requests == []
            applied = runner.invoke(cli.main, [*flags, "up", str(fleet_path), "--json"])
        assert applied.exit_code == 0, applied.output
        active = store.load().active
        assert active is not None
        torn_down = runner.invoke(
            cli.main, ["--user", "alice", "--role", "fleet_admin", "down", str(fleet_path), "--yes"]
        )
        assert torn_down.exit_code == 0, torn_down.output
    finally:
        for client in clients:
            client.close()

    expected_url = override or saved_url
    assert active.base_url == expected_url
    assert routed_requests == [
        ("PUT", f"{expected_url}/api/v1/fleets/dev-fleet"),
        ("DELETE", f"{expected_url}/api/v1/fleets/dev-fleet"),
    ]
