# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provider-neutral opaque accelerator names must survive human CLI rendering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from eyvan._internal.json_types import JsonObject
from eyvan.sdk.commands.capacity import capacity
from eyvan.sdk.commands.fleets import _render_fleet_teardown, _render_planned_fleet
from eyvan.sdk.shapes import summarize_shapes
from eyvan.server.api.schemas import CapacityPoolUpsertRequest


@pytest.mark.parametrize(
    ("shapes", "expected"),
    [
        (["custom-²"], "custom-²"),
        (["custom-①"], "custom-①"),
        (["custom-2"], "custom-2"),
        (["custom-٢"], "custom-2"),
        (["custom-\uff12"], "custom-2"),
        (["custom-²", "custom-①", "custom-2", "custom-٤"], "custom-², custom-①, custom 2-4"),
        (["opaque-cpu", "cpu"], "opaque-cpu, cpu"),
        ([], ""),
    ],
)
def test_shape_summary_preserves_nondecimal_suffixes(shapes: list[str], expected: str) -> None:
    assert summarize_shapes(shapes) == expected


def _pool_payload(shape: str) -> JsonObject:
    request = CapacityPoolUpsertRequest(
        id="opaque/gpu",
        fleet_name="opaque",
        node_type="gpu",
        provider="manual",
        region="local",
        capacity={"cpu": 4, "memory_gib": 16, "gpu": 1},
        allocatable={"cpu": 4, "memory_gib": 16, "gpu": 1},
        accelerator_type=shape,
        agent_endpoint="http://127.0.0.1:8090",
    )
    return request.to_record().model_dump(mode="json")


@pytest.mark.parametrize("shape", ["custom-²", "custom-①"])
def test_valid_opaque_pool_survives_fleet_plan_and_teardown_rendering(shape: str) -> None:
    pool = _pool_payload(shape)
    planned = _render_planned_fleet({"fleet": "opaque", "pools": [pool]})
    teardown = _render_fleet_teardown({"name": "opaque", "pools": [pool], "cells": []})
    assert any(shape in line for line in planned)
    assert any(shape in line for line in teardown)
    assert any("no live cells" in line for line in teardown)


@pytest.mark.parametrize("shape", ["custom-²", "custom-①"])
def test_capacity_fleets_human_output_preserves_valid_opaque_pool(shape: str) -> None:
    pool = _pool_payload(shape)
    capacity_pool = {
        "fleet": pool["fleet_name"],
        "pool": pool["id"],
        "accelerator_type": pool["accelerator_type"],
        "capacity_class": pool["capacity_class"],
        "region": pool["region"],
        "zone": pool["zone"],
        "min_cells": pool["min_cells"],
        "max_cells": pool["max_cells"],
        "enabled": pool["enabled"],
    }

    class InventoryClient:
        def request(self, method: str, path: str, *, params: object = None) -> dict[str, object]:
            assert (method, path) == ("GET", "/api/v1/capacity")
            assert params == {"org_id": "research", "project_id": "training"}
            return {"pools": [capacity_pool]}

    session = SimpleNamespace(client=InventoryClient(), org_id="research", project_id="training")
    result = CliRunner().invoke(capacity, ["fleets"], obj=session)
    assert result.exit_code == 0, repr(result.exception)
    assert "Fleets you can submit to:" in result.output
    assert "opaque" in result.output
    assert shape in result.output
