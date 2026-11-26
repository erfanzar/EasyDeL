# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""GPU and CPU decorator configs alongside the TPU one.

``GpuNodeType``/``CpuNodeType`` and the gcp-compute driver already ship, so these
are an SDK surface over machinery that exists rather than a new capability. What
they must not do is fork the submission path: every consumer works against
``AcceleratorConfig``, so a GPU job travels the code a TPU job already proved.
"""

from __future__ import annotations

import pytest

from eyvan.records import CapacityClass
from eyvan.sdk.client.jobs import AcceleratorConfig, CpuConfig, GpuConfig, TpuConfig


@pytest.mark.parametrize(
    "config",
    [
        TpuConfig(type="v5p-16"),
        GpuConfig(type="nvidia-tesla-a100"),
        CpuConfig(),
    ],
)
def test_every_config_satisfies_the_shared_submission_contract(config) -> None:
    """One protocol, so fleet resolution and the submitter need no per-kind branch."""

    assert isinstance(config, AcceleratorConfig)
    assert isinstance(config.type, str) and config.type
    assert config.pods >= 1
    assert isinstance(config.capacity_class, CapacityClass)


def test_gpu_and_cpu_default_to_on_demand_while_tpu_defaults_to_spot() -> None:
    """GPU spot capacity is reclaimed far more aggressively than TPU spot.

    A silently preemptible default would surprise a caller who wrote one line, so
    the cheap-but-interruptible default is kept only where the fleet is built for it.
    """

    assert TpuConfig(type="v5p-16").capacity_class is CapacityClass.SPOT
    assert GpuConfig(type="nvidia-tesla-a100").capacity_class is CapacityClass.ON_DEMAND
    assert CpuConfig().capacity_class is CapacityClass.ON_DEMAND


def test_only_tpu_may_request_multi_worker_capacity() -> None:
    """``worker_count > 1`` requires a TPU resource request in WorkloadDefinition.

    Eyvan's multi-worker unit is one atomic Multislice queued resource, which has no
    GPU or CPU analogue. Rejecting it here makes that a clear SDK error instead of a
    validation failure deep in submit.
    """

    assert TpuConfig(type="v5p-16", pods=4).pods == 4

    with pytest.raises(ValueError, match="does not support multi-worker"):
        GpuConfig(type="nvidia-tesla-a100", pods=2)
    with pytest.raises(ValueError, match="does not support multi-worker"):
        CpuConfig(pods=2)


@pytest.mark.parametrize("factory", [TpuConfig, GpuConfig])
def test_an_unusable_accelerator_type_is_rejected(factory) -> None:
    for bad in ("", "   ", "has space", "semi;colon"):
        with pytest.raises(ValueError, match="type must contain"):
            factory(type=bad)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (TpuConfig, {"type": "v5p-16"}),
        (GpuConfig, {"type": "nvidia-tesla-a100"}),
        (CpuConfig, {}),
    ],
)
def test_retry_and_capacity_class_are_validated_identically(factory, kwargs) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        factory(**kwargs, max_retries=-1)
    with pytest.raises(ValueError, match="max_retries"):
        factory(**kwargs, max_retries=101)
    with pytest.raises(ValueError, match="capacity_class"):
        factory(**kwargs, capacity_class="spot")


def test_a_gpu_config_resolves_a_fleet_through_the_same_capacity_lookup() -> None:
    """The proof that GPU reuses the TPU path rather than forking it."""

    from eyvan.sdk.execution import _resolve_fleet_from_tpu

    looked_up: list[tuple[str, CapacityClass]] = []

    class _Pool:
        fleet = "research"
        max_cells = 4

    class _Client:
        def list_capacity(self, *, org_id, project_id, accelerator_type, capacity_class):
            looked_up.append((accelerator_type, capacity_class))
            return [_Pool()]

    class _Submitter:
        client = _Client()

    class _Context:
        org_id = "acme"
        project_id = "research"
        submitter = _Submitter()

    import eyvan.sdk.execution as execution_module

    original = execution_module.EyvanCallableSubmitter
    execution_module.EyvanCallableSubmitter = _Submitter  # type: ignore[assignment]
    try:
        fleet = _resolve_fleet_from_tpu(_Context(), GpuConfig(type="nvidia-tesla-a100"))
    finally:
        execution_module.EyvanCallableSubmitter = original  # type: ignore[assignment]

    assert fleet == "research"
    assert looked_up == [("nvidia-tesla-a100", CapacityClass.ON_DEMAND)]


def test_the_public_package_exports_all_three_configs() -> None:
    import eyvan

    assert eyvan.GpuConfig(type="nvidia-tesla-a100").type == "nvidia-tesla-a100"
    assert eyvan.CpuConfig().type == "cpu"
    assert {"TpuConfig", "GpuConfig", "CpuConfig"} <= set(eyvan.__all__)
