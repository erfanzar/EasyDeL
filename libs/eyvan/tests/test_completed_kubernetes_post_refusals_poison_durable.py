# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Completed Kubernetes POST refusals must not poison durable create intents."""

from __future__ import annotations

import copy
import typing as tp

import httpx
import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode
from eyvan.capacity.providers.kubernetes.api import KubernetesApi
from tests.test_kubernetes_capacity import _context, _desired, _driver, _FakeApi


class _CompletedResponseApi(_FakeApi, KubernetesApi):
    """Exercise the shipped create decoder over local named-object persistence."""

    def __init__(self, stage: str, status: int) -> None:
        _FakeApi.__init__(self)
        self.stage = stage
        self.status = status
        self.reject = True

    def create(self, path: str, manifest: JsonObject) -> bool:
        return KubernetesApi.create(self, path, manifest)

    def request(self, method: str, path: str, *, json: JsonObject | None = None, **kwargs: object) -> httpx.Response:
        assert method == "POST"
        assert json is not None
        if self.reject and path.endswith(f"/{self.stage}"):
            return httpx.Response(self.status, json={"message": "private-admission-echo"})
        created = _FakeApi.create(self, path, json)
        return httpx.Response(201 if created else 409, json={})


@pytest.mark.parametrize("stage", ["services", "secrets", "jobs"])
@pytest.mark.parametrize("status", [403, 422])
@pytest.mark.parametrize("recovery", ["apply", "reconcile", "drain", "release"])
def test_definite_post_refusal_allows_durable_recovery(tmp_path, stage: str, status: int, recovery: str) -> None:
    api = _CompletedResponseApi(stage, status)
    driver = _driver(tmp_path, api)
    with pytest.raises(CapacityProviderError) as raised:
        driver.apply_capacity("cell-a", _desired(), _context())
    assert raised.value.code is CapacityProviderFailureCode.CONFIGURATION
    assert raised.value.retryable is False
    assert "private-admission-echo" not in str(raised.value)
    assert len(api.objects) == {"services": 0, "secrets": 1, "jobs": 2}[stage]
    assert driver.state.lease_holder() is None

    # Reload the actual local backend: recovery must not depend on an in-memory
    # acknowledgement or treat an absent provider GET as POST completion proof.
    api.reject = False
    recovered = _driver(tmp_path, api)
    record = recovered.state.get("cell-a")
    assert record is not None
    assert "kubernetes_pending_creates" not in record.metadata
    if recovery == "apply":
        observation = recovered.apply_capacity("cell-a", _desired(), _context())
        assert observation.state is CapacityStatus.HEALTHY
        assert len(api.objects) == 3
        record = recovered.state.get("cell-a")
        assert record is not None and record.generation == 1
        assert record.idempotency_keys == ["apply-1"]
    elif recovery == "reconcile":
        observation = recovered.reconcile("cell-a", _context(generation=2, key="reconcile-2"))
        assert observation.generation == 2
        record = recovered.state.get("cell-a")
        assert record is not None and record.desired_state == "up"
    elif recovery == "drain":
        observation = recovered.drain("cell-a", _context(generation=2, key="drain-2"))
        assert observation.state is CapacityStatus.DRAINING
        record = recovered.state.get("cell-a")
        assert record is not None and record.desired_state == "down"

    released = recovered.release("cell-a", _context(generation=3, key="release-3"))
    assert released.state is CapacityStatus.RELEASED
    assert recovered.state.get("cell-a") is None
    assert recovered.state.cell_tombstone("cell-a") is not None
    assert api.objects == {}
    assert all(uid and version for _path, uid, version in api.delete_preconditions)


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_indeterminate_post_error_remains_fail_closed(tmp_path, status: int) -> None:
    api = _CompletedResponseApi("services", status)
    driver = _driver(tmp_path, api)
    with pytest.raises(CapacityProviderError):
        driver.apply_capacity("cell-a", _desired(), _context())
    record = driver.state.get("cell-a")
    assert record is not None
    pending_before = copy.deepcopy(record.metadata["kubernetes_pending_creates"])
    api.reject = False
    for action in ("reconcile", "drain", "release"):
        with pytest.raises(CapacityProviderError, match="POST is still inflight"):
            getattr(driver, action)("cell-a", _context(generation=2, key=action))
    with pytest.raises(CapacityProviderError, match="POST is still inflight"):
        driver.apply_capacity("cell-a", _desired(), _context())
    current = driver.state.get("cell-a")
    assert current is not None
    assert current.metadata["kubernetes_pending_creates"] == pending_before
    assert api.objects == {} and api.deleted == [] and api.created == []


def test_generic_nonretryable_failure_is_not_completion_proof(tmp_path) -> None:
    class UnknownFailureApi(_FakeApi):
        def create(self, path: str, manifest: JsonObject) -> tp.NoReturn:
            raise CapacityProviderError(
                "configuration failure without a completed POST response",
                code=CapacityProviderFailureCode.CONFIGURATION,
                retryable=False,
            )

    api = UnknownFailureApi()
    driver = _driver(tmp_path, api)
    with pytest.raises(CapacityProviderError, match="configuration failure"):
        driver.apply_capacity("cell-a", _desired(), _context())
    record = driver.state.get("cell-a")
    assert record is not None
    pending = record.metadata["kubernetes_pending_creates"]
    assert isinstance(pending, dict)
    assert all(isinstance(intent, dict) and intent["status"] == "inflight" for intent in pending.values())
    with pytest.raises(CapacityProviderError, match="POST is still inflight"):
        driver.release("cell-a", _context(generation=2, key="release"))
    assert api.objects == {} and api.deleted == []
