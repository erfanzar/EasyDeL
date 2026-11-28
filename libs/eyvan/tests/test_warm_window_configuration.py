# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reaching the warm-window feature from outside the test suite.

``WarmWindow`` and ``warm_floor`` were written, tested and wired into the
planner and the warmth report. What was missing was any way for an operator to
set one: ``CapacityPoolUpsertRequest`` had no field, fleet YAML had no key, and
``rg warm_windows`` outside the package returned only tests. The feature worked
and nobody could use it.

Deliberately no ``pool set-warm``. ``eyvan capacity pool`` says it plainly --
pools are declared in a fleet config and applied by ``eyvan up``, so an
imperative creator would fork the source of truth. A window set through the CLI
would survive exactly until the next apply, which is a worse failure than not
having the command: it works, and then silently stops.

So the surface is the declarative one, and the loop closes through
``eyvan capacity warmth``, which already reports each pool's floor and whether a
window is currently holding it.
"""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.records import WarmWindow

_FLEET = """\
version: 1
name: research
provider:
  type: gcp
  project: gcp-project
  region: us-central1
  zone: us-central1-a
agent_endpoint: https://agent.example.com
executor: eyvan-native
node_types:
  v5p:
    kind: tpu
    accelerator_type: v5p-16
    resources:
      tpu: 8
    min_workers: 0
    max_workers: {max_workers}
    provider_options:
      bootstrap_egress: private-google-access
{warm}
"""


def _fleet(tmp_path, warm: str, *, max_workers: int = 8) -> EyvanFleetConfig:
    path = tmp_path / "fleet.yaml"
    path.write_text(_FLEET.format(warm=warm, max_workers=max_workers), encoding="utf-8")
    return EyvanFleetConfig.from_yaml(path)


def _pool_payload(fleet: EyvanFleetConfig) -> dict:
    pools = fleet.pool_payloads()
    assert len(pools) == 1
    return pools[0]


# --------------------------------------------------------------------------- fleet YAML


def test_a_warm_window_declared_in_fleet_yaml_reaches_the_pool_payload(tmp_path) -> None:
    """The whole gap in one assertion.

    The planner has always honoured a window on the pool record. Getting one
    onto that record was the part nobody could do.
    """

    fleet = _fleet(
        tmp_path,
        textwrap.indent(
            textwrap.dedent(
                """\
                warm_windows:
                  - min_cells: 4
                    start_hour: 9
                    end_hour: 17
                    days: [1, 2, 3, 4, 5]
                    utc_offset_minutes: -480
                """
            ),
            " " * 4,
        ),
    )

    assert _pool_payload(fleet)["warm_windows"] == [
        {
            "min_cells": 4,
            "start_hour": 9,
            "end_hour": 17,
            "days": [1, 2, 3, 4, 5],
            "utc_offset_minutes": -480,
        }
    ]


def test_a_fleet_without_warm_windows_is_unchanged(tmp_path) -> None:
    """Every existing fleet config keeps applying, with no window."""

    assert _pool_payload(_fleet(tmp_path, ""))["warm_windows"] == []


def test_a_window_floor_above_the_pool_ceiling_is_refused(tmp_path) -> None:
    """Caught at apply, not at 09:00 on a Monday.

    A floor above the ceiling can never be satisfied. The planner would clamp it
    silently, so the operator would see a pool that quietly ignored the number
    they wrote and no error anywhere saying why.
    """

    with pytest.raises(ValidationError, match="exceeds max_cells"):
        _fleet(
            tmp_path,
            textwrap.indent("warm_windows:\n  - min_cells: 20\n    start_hour: 9\n    end_hour: 17\n", " " * 4),
            max_workers=8,
        )


def test_a_zero_length_window_is_refused_at_the_fleet_boundary(tmp_path) -> None:
    """``WarmWindow``'s own validation has to survive being reachable from YAML.

    A window whose hours are equal never applies, so it is a typo that produces
    silence rather than capacity.
    """

    with pytest.raises(ValidationError):
        _fleet(
            tmp_path, textwrap.indent("warm_windows:\n  - min_cells: 2\n    start_hour: 9\n    end_hour: 9\n", " " * 4)
        )


# --------------------------------------------------------------------------- the API


def test_a_pool_upsert_carries_warm_windows_onto_the_record() -> None:
    """The hop between the applied payload and the record the planner reads."""

    from eyvan.server.api.schemas import CapacityPoolUpsertRequest

    request = CapacityPoolUpsertRequest.model_validate(
        {
            "id": "research/v5p",
            "fleet_name": "research",
            "node_type": "v5p",
            "provider": "gcp",
            "capacity_backend": "gcp-tpu",
            "provider_project": "gcp-project",
            "region": "us-central1",
            "zone": "us-central1-a",
            "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 8},
            "allocatable": {"cpu": 208, "memory_gib": 448, "tpu": 8},
            "accelerator_type": "v5p-16",
            "max_cells": 8,
            "warm_windows": [{"min_cells": 3, "start_hour": 9, "end_hour": 17}],
        }
    )

    record = request.to_record(default_agent_endpoint="https://agent.example.com")

    assert record.warm_windows == (WarmWindow(min_cells=3, start_hour=9, end_hour=17),)


def test_an_applied_window_raises_the_floor_the_planner_reads() -> None:
    """The point of the whole path: a configured window changes ``warm_floor``.

    Asserted through the record rather than the model, because the planner reads
    the record and a field that arrives anywhere else is decoration.
    """

    from datetime import UTC, datetime

    from eyvan.server.api.schemas import CapacityPoolUpsertRequest

    record = CapacityPoolUpsertRequest.model_validate(
        {
            "id": "research/v5p",
            "fleet_name": "research",
            "node_type": "v5p",
            "provider": "gcp",
            "capacity_backend": "gcp-tpu",
            "provider_project": "gcp-project",
            "region": "us-central1",
            "zone": "us-central1-a",
            "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 8},
            "allocatable": {"cpu": 208, "memory_gib": 448, "tpu": 8},
            "accelerator_type": "v5p-16",
            "max_cells": 8,
            "min_cells": 0,
            "warm_windows": [{"min_cells": 4, "start_hour": 9, "end_hour": 17}],
        }
    ).to_record(default_agent_endpoint="https://agent.example.com")

    assert record.warm_floor(datetime(2026, 7, 29, 10, 0, tzinfo=UTC)) == 4
    assert record.warm_floor(datetime(2026, 7, 29, 20, 0, tzinfo=UTC)) == 0


# --------------------------------------------------------------------------- the estimate


def test_an_operator_can_ask_what_a_workload_would_cost_from_the_cli(monkeypatch) -> None:
    """``POST /workloads/estimate`` existed since budgets did and nothing reached it.

    ``rg estimate eyvan/sdk/`` returned nothing, so the question it answers --
    "what will this cost, and will my budget take it" -- still required
    submitting a job and reading the resulting lease.
    """

    from click.testing import CliRunner
    from test_cli import FakeClient, _json_argument

    from eyvan.sdk import cli

    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--org",
            "acme",
            "--project",
            "research",
            "workload",
            "estimate",
            "--tpu",
            "8",
            "--accelerator-type",
            "v5p-8",
            "--duration-s",
            "7200",
        ],
    )

    assert result.exit_code == 0, result.output
    method, path, _arguments = fake.calls[0]
    assert (method, path) == ("POST", "/api/v1/workloads/estimate")
    payload = _json_argument(fake.calls[0])
    assert payload["spec"]["estimated_duration_seconds"] == 7200
    assert payload["spec"]["accelerator_type"] == "v5p-8"
    assert payload["org_id"] == "acme"


def test_the_estimate_writes_nothing_and_sends_no_idempotency_key(monkeypatch) -> None:
    """A read does not need replay protection, and carrying one implies it might write."""

    from click.testing import CliRunner
    from test_cli import FakeClient

    from eyvan.sdk import cli

    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    result = CliRunner().invoke(
        cli.main,
        ["--user", "alice", "--org", "acme", "--project", "research", "workload", "estimate"],
    )

    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1
    assert "Idempotency-Key" not in fake.calls[0][2].get("headers", {})


def test_the_typed_client_can_estimate_too() -> None:
    """The CLI is not the only caller worth having; the SDK asks the same question."""

    import httpx

    from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadEstimate
    from eyvan.sdk import EyvanClient
    from eyvan.server.api.schemas import WorkloadEstimateRequest

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=WorkloadEstimate(org_id="acme", project_id="research", estimated_duration_seconds=3600).model_dump(
                mode="json"
            ),
        )

    client = EyvanClient(client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"))
    estimate = client.estimate_workload(
        WorkloadEstimateRequest(
            org_id="acme",
            project_id="research",
            spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        )
    )

    assert estimate.estimated_duration_seconds == 3600
    assert seen[0].url.path == "/api/v1/workloads/estimate"
    client.close()
