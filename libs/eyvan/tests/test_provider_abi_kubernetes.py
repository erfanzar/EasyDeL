# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Kubernetes driver, held to the same twelve properties as every other.

A conformance suite that one provider passes is a document. The point is that
"supported" stops being a claim somebody makes and becomes something a driver
either does or does not do -- and Kubernetes is the backend most likely to be
somebody's first Eyvan deployment, because it is the one they already have.

The six hooks are facts about a real substrate, not promises: incarnations are
counted from the Jobs the fake API server actually holds, and the leak is a Job
seeded with fencing metadata this cell never wrote.
"""

from __future__ import annotations

import typing as tp
from contextlib import contextmanager
from pathlib import Path

import pytest
from test_kubernetes_capacity import _desired, _FakeApi

from eyvan.capacity.conformance import (
    CONFORMANCE_CHECK_NAMES,
    ProviderConformanceHarness,
    run_conformance,
)
from eyvan.capacity.drivers.base import (
    CapacityDriver,
    CapacityProviderError,
    CapacityProviderFailureCode,
)
from eyvan.capacity.drivers.kubernetes import KubernetesCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.capacity.providers.kubernetes.workers import (
    CELL_LABEL,
    FENCE_ANNOTATION,
    GENERATION_LABEL,
    cell_label_value,
    indexed_job_name,
)

WORKER_IMAGE = "registry.example.com/eyvan-worker@sha256:" + "b" * 64


class _UnreachableApi:
    """An API server that answers nothing conclusively.

    Every call fails in a way that leaves the caller unable to say whether it
    took effect -- which is the state a driver must never resolve into "there is
    no capacity here".
    """

    def __init__(self, inner: _FakeApi) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> tp.Any:
        def unreachable(*_args: object, **_kwargs: object) -> tp.NoReturn:
            raise CapacityProviderError(
                "the API server could not be reached",
                code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
            )

        getattr(self._inner, name)  # fail loudly if the driver calls something that does not exist
        return unreachable


class _RefusingApi:
    """An API server that refuses conclusively, however many times it is asked."""

    def __init__(self, inner: _FakeApi) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> tp.Any:
        def refuse(*_args: object, **_kwargs: object) -> tp.NoReturn:
            raise CapacityProviderError(
                "the namespace does not exist",
                code=CapacityProviderFailureCode.CONFIGURATION,
            )

        getattr(self._inner, name)
        return refuse


class KubernetesConformanceHarness(ProviderConformanceHarness):
    """Run the conformance suite against the Kubernetes capacity driver."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self._built = 0
        self._apis: dict[int, _FakeApi] = {}

    @property
    def provider_name(self) -> str:
        """Name the provider under test."""

        return "kubernetes"

    def new_driver(self) -> CapacityDriver:
        """Return a driver over a private fake cluster and durable document."""

        self._built += 1
        api = _FakeApi()
        self._apis[id(api)] = api
        driver = KubernetesCapacityDriver(
            FleetStateStore(LocalBackend(self.tmp_path / f"fleet-{self._built}.json")),
            api=api,
            namespace="eyvan",
            image=WORKER_IMAGE,
            shared_secret="agent-secret",
        )
        return driver

    @staticmethod
    def _api(driver: CapacityDriver) -> _FakeApi:
        """Return the fake cluster a driver was built over."""

        inner = tp.cast(KubernetesCapacityDriver, driver).api
        return tp.cast(_FakeApi, getattr(inner, "_inner", inner))

    def desired_capacity(self, cell_id: str) -> dict:
        """Return a valid Kubernetes pool declaration."""

        del cell_id
        return _desired()

    def provider_incarnations(self, driver: CapacityDriver, cell_id: str) -> int:
        """Count distinct fenced incarnations of ``cell_id`` held by the cluster.

        Counted from the Jobs the API server holds, not from the fleet document:
        a Job nobody remembers is exactly the thing that keeps costing money.
        """

        api = self._api(driver)
        incarnations = set()
        for job in api.jobs.values():
            metadata = job.get("metadata")
            if not isinstance(metadata, dict):
                continue
            labels = metadata.get("labels")
            annotations = metadata.get("annotations") or {}
            if not isinstance(labels, dict) or not isinstance(annotations, dict):
                continue
            if labels.get(CELL_LABEL) != cell_label_value(cell_id):
                continue
            incarnations.add((labels.get(GENERATION_LABEL), annotations.get(FENCE_ANNOTATION)))
        return len(incarnations)

    def leak_previous_incarnation(self, driver: CapacityDriver, cell_id: str) -> None:
        """Leave a Job behind wearing fencing metadata this cell never wrote.

        What a crash between two mutations leaves in a cluster: the workload of
        a previous incarnation, still running, still billed, and belonging to
        nobody the control plane can name.
        """

        api = self._api(driver)
        # Seeded at the name this cell's own Job occupies, which is where the
        # driver looks. A leak the driver never reads is a different bug and
        # would make this check measure the harness instead of the driver.
        name = indexed_job_name(cell_id)
        api.objects.pop(f"/apis/batch/v1/namespaces/eyvan/jobs/{name}", None)
        api.jobs.pop(f"eyvan/{name}", None)
        api.seed(
            f"/apis/batch/v1/namespaces/eyvan/jobs/{name}",
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": name,
                    "namespace": "eyvan",
                    "labels": {CELL_LABEL: cell_label_value(cell_id), GENERATION_LABEL: "0"},
                    "annotations": {FENCE_ANNOTATION: "a-fingerprint-this-cell-never-held"},
                },
                "spec": {"parallelism": 1},
            },
        )

    @contextmanager
    def provider_unreachable(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """Make every API call fail without establishing an outcome."""

        target = tp.cast(KubernetesCapacityDriver, driver)
        original = target.api
        target.api = _UnreachableApi(self._api(driver))  # type: ignore[assignment]
        try:
            yield
        finally:
            target.api = original

    @contextmanager
    def provider_rejects_permanently(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """Make every API call fail the same way however often it is retried."""

        target = tp.cast(KubernetesCapacityDriver, driver)
        original = target.api
        target.api = _RefusingApi(self._api(driver))  # type: ignore[assignment]
        try:
            yield
        finally:
            target.api = original


@pytest.fixture
def harness(tmp_path: Path) -> KubernetesConformanceHarness:
    return KubernetesConformanceHarness(tmp_path)


def test_the_kubernetes_driver_passes_every_conformance_check(harness: KubernetesConformanceHarness) -> None:
    """The whole suite, and the report names anything that fails.

    Asserting the report rather than a boolean so a failure says which property
    broke -- "the Kubernetes driver is not conformant" is not a bug report.
    """

    report = run_conformance(harness)

    assert report.passed, "\n".join(result.describe() for result in report.failures)


def test_the_suite_actually_ran_every_published_check(harness: KubernetesConformanceHarness) -> None:
    """A suite that quietly skipped half its checks would pass just as loudly."""

    report = run_conformance(harness)

    assert tuple(result.check for result in report.results) == CONFORMANCE_CHECK_NAMES


def test_the_harness_counts_a_real_substrate(harness: KubernetesConformanceHarness) -> None:
    """A harness that always answered zero would pass the leak checks for free.

    The conformance contract says each hook is a fact, never a promise. This is
    that sentence as a test: applying capacity must make the counter go up, and
    releasing it must bring it back down, or the suite above is measuring
    nothing.
    """

    from eyvan.capacity.conformance import context

    driver = harness.new_driver()
    cell = "conformance-substrate"

    assert harness.provider_incarnations(driver, cell) == 0
    driver.apply_capacity(cell, harness.desired_capacity(cell), context("apply-1"))
    assert harness.provider_incarnations(driver, cell) == 1, "an applied cell must exist in the cluster"

    driver.release(cell, context("release-1"))
    assert harness.provider_incarnations(driver, cell) == 0, "a released cell must leave nothing behind"


def test_the_harness_can_actually_break_the_provider(harness: KubernetesConformanceHarness) -> None:
    """The two failure contexts must fail, or half the suite proves nothing."""

    from eyvan.capacity.conformance import context

    # A separate cell per context. After an inconclusive apply the driver
    # refuses further mutations of that cell -- an older POST may still be
    # inflight, and advancing durable ownership over it is how two incarnations
    # of one cell come to exist -- so reusing the cell would measure that
    # refusal rather than the context under test.
    unreachable_driver = harness.new_driver()
    with (
        harness.provider_unreachable(unreachable_driver),
        pytest.raises(CapacityProviderError) as unreachable,
    ):
        unreachable_driver.apply_capacity("cell-unreachable", harness.desired_capacity("x"), context("apply-1"))
    assert unreachable.value.code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE

    refusing_driver = harness.new_driver()
    with (
        harness.provider_rejects_permanently(refusing_driver),
        pytest.raises(CapacityProviderError) as refused,
    ):
        refusing_driver.apply_capacity("cell-refused", harness.desired_capacity("x"), context("apply-1"))
    assert refused.value.code is CapacityProviderFailureCode.CONFIGURATION
