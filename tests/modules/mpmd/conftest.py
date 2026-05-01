"""MPMD module test configuration."""

from __future__ import annotations

import jax
import pytest

from tests.modules import conftest as spmd_conftest
from tests.modules.mpmd._scheduler_utils import get_mpmd_schedule_case


@pytest.fixture
def mpmd_schedule_kind(request: pytest.FixtureRequest) -> str | None:
    return getattr(request, "param", None)


@pytest.fixture(scope="session")
def _base_mpmd_small_model_config() -> dict:
    """Reuse module fixtures with a PP-shaped mesh for MPMD runs."""
    if jax.device_count() < 2:
        pytest.skip("MPMD module tests require at least 2 JAX devices")

    config = dict(spmd_conftest.small_model_config.__wrapped__())
    config["sharding_axis_dims"] = (2, 1, 1, -1, 1, 1)
    config["pipeline_stage_regions"] = False
    config["scan_layers"] = False
    return config


@pytest.fixture
def small_model_config(_base_mpmd_small_model_config: dict, mpmd_schedule_kind: str | None) -> dict:
    config = dict(_base_mpmd_small_model_config)
    schedule_case = get_mpmd_schedule_case(mpmd_schedule_kind)
    if schedule_case is not None:
        config["mpmd_schedule"] = schedule_case.schedule
        config["pipeline_virtual_stages"] = schedule_case.virtual_stages
        config["num_hidden_layers"] = schedule_case.num_hidden_layers
        config["num_layers"] = schedule_case.num_hidden_layers
        config["batch_size"] = max(
            int(config["batch_size"]),
            int(getattr(schedule_case.schedule, "microbatches", 1)),
        )
    else:
        config["pipeline_virtual_stages"] = 1
    return config
