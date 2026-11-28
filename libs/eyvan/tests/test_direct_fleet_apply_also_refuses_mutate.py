# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Direct fleet apply also refuses to mutate a plane it would not persist."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.fleets import FleetCommandService
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContext, FileContextStore


@pytest.mark.parametrize("dry_run", [False, True], ids=["refuse-mutation", "allow-offline-plan"])
def test_direct_apply_endpoint_policy(tmp_path: Path, dry_run: bool, capsys: pytest.CaptureFixture[str]) -> None:
    """A saved endpoint mismatch is harmless only for an offline plan."""

    store = FileContextStore(tmp_path / "contexts.json")
    store.activate(
        EyvanContext(fleet="dev-fleet", base_url="https://saved.example.invalid", org_id="acme", project_id="training")
    )
    contexts = store.activate(
        EyvanContext(
            fleet="other-fleet", base_url="https://active.example.invalid", org_id="acme", project_id="training"
        )
    )
    before = store.resolved_path.read_bytes()
    requests: list[tuple[str, str]] = []

    class RecordingClient:
        def request(self, method: str, path: str, **_kwargs: object) -> list[object]:
            requests.append((method, path))
            return []

    session = CliSession(
        client=RecordingClient(),
        identity=ClientIdentity(dev_user="alice"),
        context_store=store,
        context_config=contexts,
        base_url="https://active.example.invalid",
        org_id=None,
        project_id=None,
        overrides=ContextOverrides(),
    )
    config = EyvanFleetConfig.model_validate(
        {
            "version": 1,
            "name": "dev-fleet",
            "provider": {
                "type": "gcp",
                "project": "test-project",
                "region": "us-central1",
                "zone": "us-central1-a",
            },
            "agent_endpoint": "https://agent.example.invalid",
            "node_types": {
                "cpu-workers": {
                    "kind": "cpu",
                    "machine_type": "n2-standard-8",
                    "resources": {"cpu": 8, "memory_gib": 32},
                    "min_workers": 0,
                    "max_workers": 1,
                    "provider_options": {
                        "service_account": "eyvan-worker@test-project.iam.gserviceaccount.com",
                        "bootstrap_egress": "private-google-access",
                    },
                }
            },
        }
    )
    service = FleetCommandService(session)
    if dry_run:
        service.apply(
            config,
            generated_payload=None,
            selected=None,
            autoscale_cache_dir=None,
            dry_run=True,
            as_json=True,
            idempotency_key=None,
        )
        assert '"fleet": "dev-fleet"' in capsys.readouterr().out
    else:
        with pytest.raises(click.ClickException, match="saved control-plane endpoint"):
            service.apply(
                config,
                generated_payload=None,
                selected=None,
                autoscale_cache_dir=None,
                dry_run=False,
                as_json=True,
                idempotency_key=None,
            )
    assert requests == []
    assert store.resolved_path.read_bytes() == before
    assert session.context_config.active_context == "other-fleet"
