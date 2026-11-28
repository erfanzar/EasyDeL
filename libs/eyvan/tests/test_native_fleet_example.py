# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest
import yaml

from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.server.api.schemas import FleetApplyRequest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "fleet.yaml"


def test_native_fleet_example_declares_a_supported_worker_python() -> None:
    """The built-in Ubuntu 22 TPU image need not have a supported system Python."""
    config = EyvanFleetConfig.from_yaml(EXAMPLE)
    request = FleetApplyRequest.model_validate({"pools": config.pool_payloads()})

    assert len(request.pools) == 3
    assert all(pool.executor == "eyvan-native" for pool in request.pools)
    assert all(pool.runtime.python == "3.11" for pool in request.pools)


@pytest.mark.parametrize("version", ["3.12", "3.13"])
def test_native_fleet_example_keeps_explicit_node_python_override(version: str) -> None:
    document = yaml.safe_load(EXAMPLE.read_text())
    document["node_types"]["tpu-workers"]["runtime"]["python"] = version
    config = EyvanFleetConfig.model_validate(document)
    request = FleetApplyRequest.model_validate({"pools": config.pool_payloads()})

    assert {pool.node_type: pool.runtime.python for pool in request.pools} == {
        "cpu-workers": "3.11",
        "gpu-workers": "3.11",
        "tpu-workers": version,
    }
