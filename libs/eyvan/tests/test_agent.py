# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import threading

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eyvan.capacity.drivers.base import (
    CapacityProviderFailureCode,
    CapacityStatus,
    CellCapabilities,
    CellObservation,
    StaleMutationError,
)
from eyvan.capacity.fleet_state import FleetLeaseBusyError, FleetStateStore, LocalBackend, StaleFenceError
from eyvan.capacity.providers.gcp.queued_resources import GcloudError, GcloudQueueAdmissionDeniedError
from eyvan.config import EyvanSettings
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import (
    CellAgentBusyError,
    CellAgentCapacityProviderError,
    CellAgentClient,
    CellAgentStaleMutationError,
    CellAgentUnsupportedCapabilityError,
)
from eyvan.worker.agent.models import CapacityRequest, MutationModel, WorkloadMutationRequest, WorkloadRequest
from eyvan.worker.agent.service import CellAgentService


def workload_request(generation: int, token: str) -> WorkloadRequest:
    return WorkloadRequest(
        mutation=MutationModel(
            idempotency_key=f"submit-{generation}",
            generation=generation,
            fencing_token=token,
        ),
        spec={
            "submission_id": f"workload-1-a{generation}",
            "metadata": {"eyvan_workload_id": "workload-1"},
        },
    )


class FakeCapacity:
    def __init__(self, *, observed_tpu_vms: int | None = None) -> None:
        self.observed_tpu_vms = observed_tpu_vms
        self.runtime_process_count = observed_tpu_vms

    def capabilities(self):
        return CellCapabilities(
            provider="fake",
            executor="native-local",
            accelerator_types=("tpu",),
            capacity_classes=("spot",),
        )

    def observe(self, cell_id):
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.HEALTHY,
            generation=3,
            execution_endpoint="local",
            observed_tpu_vms=self.observed_tpu_vms,
            runtime_process_count=self.runtime_process_count,
        )

    def reconcile(self, cell_id, _context):
        return self.observe(cell_id)

    def release(self, cell_id, _context):
        return self.observe(cell_id)


class StaticResolver:
    def __init__(self, executor):
        self.executor = executor

    def resolve(self, _cell_id, _observation):
        return self.executor


class NotReadyResolver(StaticResolver):
    def readiness(self, _cell_id, _observation):
        return False, "runtime is still starting"


class NoopExecutor:
    def submit(self, spec, _context):
        return str(spec["submission_id"])

    def retire(self, _remote_id, _context):
        return None


def service(tmp_path, executor=None, *, owner="agent", observed_tpu_vms: int | None = None):
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state._owner = owner
    return CellAgentService(
        capacity=FakeCapacity(observed_tpu_vms=observed_tpu_vms),  # type: ignore[arg-type]
        state=state,
        executors=StaticResolver(executor or NoopExecutor()),  # type: ignore[arg-type]
    ), state


def test_cell_agent_client_preserves_driver_lease_contention() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"detail": "Eyvan fleet mutation lease is busy", "code": "driver_busy"},
        )

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = WorkloadMutationRequest(
        mutation=MutationModel(idempotency_key="reconcile-1", generation=1, fencing_token="cell:1")
    )

    with pytest.raises(CellAgentBusyError, match="lease is busy"):
        client.reconcile("cell-a", request)


def test_cell_agent_client_preserves_typed_capacity_provider_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={"detail": "GCP queued-resource admission denied", "code": "queue_admission_denied"},
        )

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = CapacityRequest(
        mutation=MutationModel(idempotency_key="apply-1", generation=1, fencing_token="cell:1"),
        desired={},
    )

    with pytest.raises(CellAgentCapacityProviderError, match="queued-resource admission denied") as error:
        client.apply_capacity("cell-a", request)
    assert error.value.code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED


def test_an_unstructured_agent_500_is_distinguishable_from_a_provider_failure() -> None:
    """The agent failing is not the capacity failing.

    An unavailable agent must not be recorded as ``CapacityStatus.PROVIDER_ERROR``:
    doing so restarts the provider phase clock and feeds the zone's belief a
    failure the provider never caused.
    """

    from eyvan.worker.agent.client import CellAgentUnavailableError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="fleet state backend is unreachable")

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = WorkloadMutationRequest(
        mutation=MutationModel(idempotency_key="reconcile-1", generation=1, fencing_token="cell:1")
    )

    with pytest.raises(CellAgentUnavailableError, match="unreachable"):
        client.reconcile("cell-a", request)


def test_a_structured_502_still_reports_the_provider_not_the_agent() -> None:
    """The 5xx rule must not swallow the one 5xx that *is* an answer."""

    from eyvan.worker.agent.client import CellAgentUnavailableError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={"detail": "GCP stockout", "code": "provider_error"},
        )

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = CapacityRequest(
        mutation=MutationModel(idempotency_key="apply-1", generation=1, fencing_token="cell:1"),
        desired={},
    )

    with pytest.raises(CellAgentCapacityProviderError):
        client.apply_capacity("cell-a", request)
    assert not issubclass(CellAgentCapacityProviderError, CellAgentUnavailableError)


def test_a_deliberate_501_is_not_read_as_the_agent_being_broken() -> None:
    """Terminal-by-design, so it must not join the retried-forever class."""

    from eyvan.worker.agent.client import CellAgentUnavailableError, CellAgentUnsupportedCapabilityError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(501, text="this backend cannot run checkpoint hooks")

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = WorkloadMutationRequest(
        mutation=MutationModel(idempotency_key="reconcile-1", generation=1, fencing_token="cell:1")
    )

    with pytest.raises(CellAgentUnsupportedCapabilityError):
        client.reconcile("cell-a", request)
    assert not issubclass(CellAgentUnsupportedCapabilityError, CellAgentUnavailableError)


def test_cell_agent_client_preserves_missing_cell_persistence_as_a_stale_mutation() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "detail": "cell 'cell-a' was removed before native worker bootstrap could be persisted; "
                "old generation was fenced",
                "code": "stale_mutation",
            },
        )

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = CapacityRequest(
        mutation=MutationModel(idempotency_key="apply-1", generation=1, fencing_token="cell:1"),
        desired={},
    )

    with pytest.raises(CellAgentStaleMutationError, match="old generation was fenced"):
        client.apply_capacity("cell-a", request)


def test_agent_returns_structured_capacity_provider_failure(tmp_path) -> None:
    class ProviderFailingCapacity(FakeCapacity):
        def apply_capacity(self, _cell_id, _desired, _context):
            raise GcloudQueueAdmissionDeniedError("GCP queued-resource admission denied")

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state._owner = "agent"
    cell_service = CellAgentService(
        capacity=ProviderFailingCapacity(),
        state=state,
        executors=StaticResolver(NoopExecutor()),
    )
    client = TestClient(
        create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")),
        headers={"Authorization": "Bearer secret"},
    )

    response = client.put(
        "/api/v1/cells/cell-a/capacity",
        json={
            "mutation": {"idempotency_key": "apply-1", "generation": 1, "fencing_token": "cell:1"},
            "desired": {},
        },
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "GCP queued-resource admission denied", "code": "queue_admission_denied"}


def test_agent_keeps_generic_provider_permission_failure_generic(tmp_path) -> None:
    class ProviderFailingCapacity(FakeCapacity):
        def apply_capacity(self, _cell_id, _desired, _context):
            raise GcloudError("PERMISSION_DENIED: missing tpu.nodes.create")

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state._owner = "agent"
    cell_service = CellAgentService(
        capacity=ProviderFailingCapacity(),
        state=state,
        executors=StaticResolver(NoopExecutor()),
    )
    client = TestClient(
        create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")),
        headers={"Authorization": "Bearer secret"},
    )

    response = client.put(
        "/api/v1/cells/cell-a/capacity",
        json={
            "mutation": {"idempotency_key": "apply-1", "generation": 1, "fencing_token": "cell:1"},
            "desired": {},
        },
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "PERMISSION_DENIED: missing tpu.nodes.create", "code": "provider_error"}


def test_agent_returns_missing_cell_persistence_as_a_stale_mutation(tmp_path) -> None:
    class MissingCellPersistenceCapacity(FakeCapacity):
        def apply_capacity(self, _cell_id, _desired, _context):
            raise StaleMutationError(
                "cell 'cell-a' was removed before native worker bootstrap could be persisted; old generation was fenced"
            )

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state._owner = "agent"
    cell_service = CellAgentService(
        capacity=MissingCellPersistenceCapacity(),
        state=state,
        executors=StaticResolver(NoopExecutor()),
    )
    client = TestClient(
        create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")),
        headers={"Authorization": "Bearer secret"},
    )
    response = client.put(
        "/api/v1/cells/cell-a/capacity",
        json={
            "mutation": {"idempotency_key": "apply-1", "generation": 1, "fencing_token": "cell:1"},
            "desired": {},
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "cell 'cell-a' was removed before native worker bootstrap could be persisted; "
        "old generation was fenced",
        "code": "stale_mutation",
    }


def test_cell_agent_client_validates_capabilities() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "provider": "gcp-tpu",
                "executor": "eyvan-native",
                "accelerator_types": ["v5p"],
                "capacity_classes": ["spot"],
                "supports_checkpoint": False,
                "supports_callables": True,
                "supports_exec": True,
                "max_gang_replicas": 4,
                "resource_enforcement": "backend",
            },
        )

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    capabilities = CellAgentClient("http://agent", client=http_client).capabilities()

    assert capabilities.provider == "gcp-tpu"
    assert capabilities.accelerator_types == ("v5p",)
    assert capabilities.max_gang_replicas == 4


def test_agent_workload_request_rejects_non_finite_nested_json() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        WorkloadRequest(
            mutation=MutationModel(idempotency_key="submit-1", generation=1, fencing_token="cell:1"),
            spec={"submission_id": "workload-1", "metadata": {"loss": float("nan")}},
        )


def test_cell_agent_rejects_a_non_string_submission_id(tmp_path) -> None:
    cell_service, _ = service(tmp_path)
    request = WorkloadRequest(
        mutation=MutationModel(idempotency_key="submit-1", generation=1, fencing_token="cell:1"),
        spec={
            "submission_id": 123,
            "metadata": {"eyvan_workload_id": "workload-1"},
        },
    )

    with pytest.raises(ValueError, match="submission_id must be a non-empty string"):
        cell_service.submit("cell-a", request)


def test_cell_agent_client_allows_native_submit_to_finish() -> None:
    observed: dict[str, dict[str, float]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, json={"remote_id": "workload-1-a1", "status": "SUBMITTED"})

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)

    submitted = client.submit("cell-a", workload_request(1, "leader-1"))

    assert submitted.remote_id == "workload-1-a1"
    assert observed["timeout"] == {"connect": 1800, "read": 1800, "write": 1800, "pool": 1800}


def test_agent_requires_configured_bearer_token(tmp_path) -> None:
    cell_service, _ = service(tmp_path)
    app = create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret"))
    client = TestClient(app)

    assert client.get("/healthz").status_code == 200
    assert client.get("/api/v1/capabilities").status_code == 401
    # A non-ASCII header must be a refused credential (401), not the 500 the
    # TypeError inside compare_digest used to raise. Sent as the raw latin-1
    # bytes a caller puts on the wire, which the ASGI server decodes to str.
    non_ascii = client.get("/api/v1/capabilities", headers={"Authorization": "Bearer tök".encode("latin-1")})
    assert non_ascii.status_code == 401
    response = client.get("/api/v1/capabilities", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json()["provider"] == "fake"


def test_agent_refuses_to_start_without_a_shared_secret(tmp_path) -> None:
    cell_service, _ = service(tmp_path)

    with pytest.raises(ValueError, match="EYVAN_AGENT_SHARED_SECRET"):
        create_agent_app(cell_service, EyvanSettings())


def test_agent_allow_unauthenticated_opt_out_serves_requests_without_credentials(tmp_path) -> None:
    cell_service, _ = service(tmp_path)
    app = create_agent_app(cell_service, EyvanSettings(), allow_unauthenticated=True)
    client = TestClient(app)

    response = client.get("/api/v1/cells/cell-a")

    assert response.status_code == 200
    assert response.json()["cell_id"] == "cell-a"


def test_agent_factory_passes_explicit_settings_to_default_service() -> None:
    with pytest.raises(ValueError, match="unknown Eyvan capacity backend"):
        create_agent_app(settings=EyvanSettings(capacity_backend="not-a-provider", agent_shared_secret="secret"))


def test_agent_rejects_unknown_execution_backend_at_startup(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown execution backend"):
        CellAgentService(
            state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
            settings=EyvanSettings(execution_backend="not-an-executor"),
        )


def test_agent_requires_a_shared_secret_for_native_execution(tmp_path) -> None:
    with pytest.raises(ValueError, match="EYVAN_NATIVE_WORKER_ROOT_SECRET"):
        CellAgentService(
            state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
            settings=EyvanSettings(),
        )


def test_agent_observation_contract(tmp_path) -> None:
    cell_service, _ = service(tmp_path)
    client = TestClient(
        create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")),
        headers={"Authorization": "Bearer secret"},
    )

    response = client.get("/api/v1/cells/cell-a")

    assert response.status_code == 200
    assert response.json() == {
        "cell_id": "cell-a",
        "state": "HEALTHY",
        "generation": 3,
        "execution_endpoint": "local",
        "detail": "",
        "metadata": {},
    }


def test_agent_observation_includes_gcp_reported_tpu_vm_count(tmp_path) -> None:
    cell_service, _ = service(tmp_path, observed_tpu_vms=8)
    client = TestClient(
        create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")),
        headers={"Authorization": "Bearer secret"},
    )

    response = client.get("/api/v1/cells/cell-a")

    assert response.status_code == 200
    assert response.json()["observed_tpu_vms"] == 8
    assert response.json()["runtime_process_count"] == 8


def test_agent_does_not_report_provider_capacity_ready_before_execution_runtime(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    cell_service = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=NotReadyResolver(NoopExecutor()),  # type: ignore[arg-type]
    )

    observation = cell_service.reconcile(
        "cell-a",
        WorkloadMutationRequest(
            mutation=MutationModel(idempotency_key="reconcile-1", generation=3, fencing_token="cell:3")
        ),
    )

    assert observation.state is CapacityStatus.EXECUTION_CONNECTING
    assert observation.detail == "runtime is still starting"


def test_agent_release_disposes_cached_execution_for_the_deleted_cell(tmp_path) -> None:
    closed: list[str] = []

    class ClosingResolver(StaticResolver):
        def close_cell(self, cell_id: str) -> None:
            closed.append(cell_id)

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    cell_service = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=ClosingResolver(NoopExecutor()),  # type: ignore[arg-type]
    )

    cell_service.release(
        "cell-a",
        WorkloadMutationRequest(
            mutation=MutationModel(idempotency_key="release-1", generation=3, fencing_token="cell:3")
        ),
    )

    assert closed == ["cell-a"]


def test_submit_persists_high_watermark_and_rejects_old_generation(tmp_path) -> None:
    submitted: list[tuple[int, str]] = []

    class FakeExecutor:
        def __init__(self, state):
            self.state = state

        def submit(self, spec, context):
            assert self.state.lease_holder() is not None
            submitted.append((context.generation, context.fencing_token))
            return str(spec["submission_id"])

        def retire(self, _remote_id, _context):
            pytest.fail("a stale generation must not retire the current job")

    first_state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    first_state._owner = "current"
    executor = FakeExecutor(first_state)
    current = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=first_state,
        executors=StaticResolver(executor),  # type: ignore[arg-type]
    )

    observation = current.submit("cell-a", workload_request(2, "leader-2"))

    assert observation.remote_id == "workload-1-a2"
    assert first_state.lease_holder() is None
    current_fence = first_state.workload_fence("workload-1")
    assert current_fence is not None and current_fence["generation"] == 2

    stale_state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    stale_state._owner = "stale"
    stale = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=stale_state,
        executors=StaticResolver(executor),  # type: ignore[arg-type]
    )
    with pytest.raises(StaleMutationError, match="older"):
        stale.submit("cell-a", workload_request(1, "leader-1"))
    assert submitted == [(2, "leader-2")]


def test_submit_holds_state_lease_through_execution_side_effect(tmp_path) -> None:
    path = tmp_path / "fleet.json"
    entered_submit = threading.Event()
    finish_submit = threading.Event()
    submitted: list[str] = []
    errors: list[BaseException] = []

    # A genuinely separate agent, so the owner is constructor-supplied: it keys
    # the mutual-exclusion lock, and relabelling after construction would leave
    # this sharing a lock with the agent under test.
    old_state = FleetStateStore(LocalBackend(path), owner="old-agent")

    class BlockingExecutor:
        def submit(self, spec, _context):
            assert old_state.lease_holder() == "old-agent"
            entered_submit.set()
            assert finish_submit.wait(timeout=2)
            submitted.append("old")
            return str(spec["submission_id"])

        def retire(self, _remote_id, _context):
            pytest.fail("the first generation has no predecessor")

    old_service = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=old_state,
        executors=StaticResolver(BlockingExecutor()),  # type: ignore[arg-type]
    )

    new_state = FleetStateStore(LocalBackend(path))
    new_state._owner = "new-agent"

    class NewExecutor:
        def submit(self, spec, _context):
            submitted.append("new")
            return str(spec["submission_id"])

        def status(self, remote_id):
            from eyvan.capacity.drivers.base import ExecutionWorkloadStatus

            return {
                "remote_id": remote_id,
                "status": ExecutionWorkloadStatus.STOPPED,
                "message": "double",
            }

        def retire(self, remote_id, context):
            assert new_state.lease_holder() == "new-agent"
            assert remote_id == "workload-1-a1"
            assert (context.generation, context.fencing_token) == (1, "leader-1")
            submitted.append("retired-old")

    new_service = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=new_state,
        executors=StaticResolver(NewExecutor()),  # type: ignore[arg-type]
    )

    def submit_old() -> None:
        try:
            old_service.submit("cell-a", workload_request(1, "leader-1"))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=submit_old)
    thread.start()
    assert entered_submit.wait(timeout=2)
    with pytest.raises(FleetLeaseBusyError, match="lease is busy"):
        new_service.submit("cell-a", workload_request(2, "leader-2"))
    finish_submit.set()
    thread.join(timeout=2)

    assert errors == []
    assert submitted == ["old"]
    new_service.submit("cell-a", workload_request(2, "leader-2"))
    assert submitted == ["old", "retired-old", "new"]


def test_submit_retires_a_launch_if_lease_ownership_is_lost_afterward(tmp_path) -> None:
    class LeaseLosingState(FleetStateStore):
        def __init__(self, backend):
            super().__init__(backend)
            self.checks = 0

        def require_lease(self):
            self.checks += 1
            if self.checks == 2:
                raise StaleFenceError("test lease was stolen")
            return super().require_lease()

    events: list[str] = []

    class CleanupExecutor:
        def submit(self, spec, _context):
            events.append("submitted")
            return str(spec["submission_id"])

        def retire(self, remote_id, _context):
            events.append(f"retired:{remote_id}")

    state = LeaseLosingState(LocalBackend(tmp_path / "fleet.json"))
    cell_service = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=StaticResolver(CleanupExecutor()),  # type: ignore[arg-type]
    )

    with pytest.raises(StaleMutationError, match="test lease was stolen"):
        cell_service.submit("cell-a", workload_request(1, "leader-1"))

    assert events == ["submitted", "retired:workload-1-a1"]


def test_one_agent_serializes_concurrent_workload_generations(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state._owner = "one-agent"
    first_entered = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    events: list[str] = []
    errors: list[BaseException] = []

    class OrderedExecutor:
        def submit(self, spec, context):
            if context.generation == 1:
                events.append("submit-1-entered")
                first_entered.set()
                assert release_first.wait(timeout=2)
            events.append(f"submit-{context.generation}")
            return str(spec["submission_id"])

        def status(self, remote_id):
            from eyvan.capacity.drivers.base import ExecutionWorkloadStatus

            return {
                "remote_id": remote_id,
                "status": ExecutionWorkloadStatus.STOPPED,
                "message": "double",
            }

        def retire(self, remote_id, context):
            events.append(f"retire-{remote_id}-g{context.generation}")

    cell_service = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=StaticResolver(OrderedExecutor()),  # type: ignore[arg-type]
    )

    def submit(generation: int, token: str) -> None:
        try:
            cell_service.submit("cell-a", workload_request(generation, token))
        except BaseException as exc:
            errors.append(exc)
        finally:
            if generation == 2:
                second_finished.set()

    first = threading.Thread(target=submit, args=(1, "leader-1"))
    second = threading.Thread(target=submit, args=(2, "leader-2"))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    assert not second_finished.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert errors == []
    assert events == [
        "submit-1-entered",
        "submit-1",
        "retire-workload-1-a1-g1",
        "submit-2",
    ]
    current_fence = state.workload_fence("workload-1")
    assert current_fence is not None and current_fence["generation"] == 2


def test_native_cross_cell_supersede_routes_to_source_agent(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state.claim_workload_fence(
        "workload-1",
        1,
        "leader-1",
        remote_id="workload-1-a1",
        cell_id="cell-a",
    )
    retired: list[tuple[str, str, int, str]] = []

    def remote_retire(cell_id, remote_id, request):
        retired.append(
            (
                cell_id,
                remote_id,
                request.mutation.generation,
                request.mutation.fencing_token,
            )
        )

    cell_service = CellAgentService(
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=StaticResolver(NoopExecutor()),  # type: ignore[arg-type]
        remote_retire=remote_retire,
    )

    result = cell_service.submit("cell-b", workload_request(2, "leader-2"))

    assert result.remote_id == "workload-1-a2"
    assert retired == [("cell-a", "workload-1-a1", 1, "leader-1")]
    current_fence = state.workload_fence("workload-1")
    assert current_fence is not None and current_fence["cell_id"] == "cell-b"


def test_cross_cell_supersede_does_not_treat_absent_capacity_as_terminal(tmp_path) -> None:
    class AbsentSourceCapacity(FakeCapacity):
        def observe(self, cell_id):
            state = CapacityStatus.ABSENT if cell_id == "cell-a" else CapacityStatus.HEALTHY
            return CellObservation(cell_id=cell_id, state=state, generation=2)

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state.claim_workload_fence(
        "workload-1",
        1,
        "leader-1",
        remote_id="workload-1-a1",
        cell_id="cell-a",
    )
    retired: list[tuple[str, str]] = []
    cell_service = CellAgentService(
        capacity=AbsentSourceCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=StaticResolver(NoopExecutor()),  # type: ignore[arg-type]
        remote_retire=lambda cell_id, remote_id, _request: retired.append((cell_id, remote_id)),
    )

    cell_service.submit("cell-b", workload_request(2, "leader-2"))

    assert retired == [("cell-a", "workload-1-a1")]


def test_cell_agent_client_preserves_an_unsupported_capability_as_terminal() -> None:
    """The agent's deliberate 501 must not arrive as a generic transport error.

    ``_raise_for_status`` typed 409 and 502 and let everything else fall through
    to ``raise_for_status()``, so "this backend does not implement checkpoint
    hooks" was indistinguishable from the agent being broken -- and the
    controller retried it on every tick against an answer that never changes.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            501,
            json={"detail": "the manual backend cannot run checkpoint hooks", "code": "unsupported_capability"},
        )

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = WorkloadMutationRequest(
        mutation=MutationModel(idempotency_key="ckpt-1", generation=1, fencing_token="cell:1")
    )

    with pytest.raises(CellAgentUnsupportedCapabilityError, match="cannot run checkpoint hooks"):
        client.stop("cell-a", "wl-1", request)
    http_client.close()


@pytest.mark.asyncio
async def test_an_unsupported_capability_settles_the_command_instead_of_retrying() -> None:
    """A permanent 501 must not spend the retry budget or hold the outbox open.

    Every other exception from an agent call becomes ``FAILED`` with exponential
    backoff, so a backend answering "I do not implement this" was re-attempted
    forever against an answer that cannot change.
    """

    from eyvan.records import (
        CellRecord,
        OutboxState,
        Principal,
        ResourceVector,
        Role,
        RoleGrant,
    )
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    class UnsupportedAgent:
        def apply_capacity(self, cell_id, request):
            raise CellAgentUnsupportedCapabilityError("this backend does not provision capacity")

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, agent_factory=lambda _cell: UnsupportedAgent())  # type: ignore[arg-type]
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-a",
            provider="manual",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(cpu=8),
            allocatable=ResourceVector(cpu=8),
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )

    assert await controller.dispatch_outbox_once() == 1

    async with repository.transaction() as transaction:
        commands = await transaction.list_outbox(limit=100)
    applies = [command for command in commands if command.topic == "capacity.apply"]
    assert applies, "the cell upsert must have enqueued a capacity apply for this to test anything"
    assert applies[0].state is OutboxState.DISCARDED, "a permanent 501 must not stay FAILED for retry"
    assert "unsupported by cell agent backend" in (applies[0].last_error or "")


def test_a_dropped_connection_is_unavailable_not_a_provider_failure() -> None:
    """Classify dropped connections as agent unavailability rather than provider failure.

    A dropped connection has no response for ``_raise_for_status`` to classify,
    so transport errors need the same typed unavailability handling as agent 5xx responses.
    """

    from eyvan.worker.agent.client import CellAgentUnavailableError

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = WorkloadMutationRequest(
        mutation=MutationModel(idempotency_key="reconcile-1", generation=1, fencing_token="cell:1")
    )

    with pytest.raises(CellAgentUnavailableError, match="could not be reached"):
        client.reconcile("cell-a", request)


def test_a_connect_timeout_is_also_unavailable() -> None:
    """A slow link is the common case on a laptop-hosted control plane."""

    from eyvan.worker.agent.client import CellAgentUnavailableError

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)

    with pytest.raises(CellAgentUnavailableError):
        client.observe("cell-a")


def test_every_agent_call_goes_through_the_one_seam() -> None:
    """Twelve `except` clauses would drift; one wrapper cannot.

    Asserts the taxonomy is structural rather than remembered: no method may
    reach the transport directly, or a transport failure on that path silently
    becomes a provider error again.
    """

    import inspect

    from eyvan.worker.agent import client as module

    source = inspect.getsource(module.CellAgentClient)
    direct = [
        line.strip()
        for line in source.splitlines()
        if "self.client." in line and "self.client.close()" not in line and "self.client = " not in line
    ]
    assert direct == ["response = self.client.request(method, url, **kwargs)"], direct


def test_a_rejected_configuration_carries_the_agents_own_sentence() -> None:
    """Preserve the agent's actionable configuration error instead of replacing it with a generic 500."""

    from eyvan.worker.agent.client import CellAgentInvalidConfigurationError

    message = "internal-only eyvan-native Compute capacity cannot use bootstrap_egress=public-internet"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": message, "code": "invalid_configuration"})

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = CapacityRequest(
        mutation=MutationModel(idempotency_key="apply-1", generation=1, fencing_token="cell:1"),
        desired={},
    )

    with pytest.raises(CellAgentInvalidConfigurationError, match="bootstrap_egress"):
        client.apply_capacity("cell-a", request)


def test_an_ordinary_400_is_not_mistaken_for_a_configuration_refusal() -> None:
    """Only the agent's typed code may claim terminality; a bare 400 may not."""

    from eyvan.worker.agent.client import CellAgentInvalidConfigurationError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "malformed request body"})

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = CapacityRequest(
        mutation=MutationModel(idempotency_key="apply-1", generation=1, fencing_token="cell:1"),
        desired={},
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.apply_capacity("cell-a", request)
    assert not issubclass(CellAgentInvalidConfigurationError, httpx.HTTPStatusError)


def test_an_agent_can_be_asked_what_it_holds(tmp_path) -> None:
    """Enumerate agent-held cells so stale desired-up records can be reconciled after database replacement."""

    agent, state = service(tmp_path)
    state.mutate_or_create_record(
        "ghost-cell",
        lambda record: record.__dict__.update(desired_state="up", generation=3, project="p", zone="z"),
    )

    inventory = agent.inventory()

    assert [entry["name"] for entry in inventory] == ["ghost-cell"]
    assert inventory[0]["desired_state"] == "up"
    assert inventory[0]["generation"] == 3


# --------------------------------------------------------------------------- which build an agent runs


def test_an_agent_reports_the_code_it_is_running(tmp_path) -> None:
    """Version is the same string in every component of this package.

    So it cannot answer the question an upgrade raises -- is the agent holding
    my capacity the build I deployed? -- and a proof has already been run
    against a stale agent and read as evidence.
    """

    cell_service, _ = service(tmp_path)
    client = TestClient(create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")))

    document = client.get("/api/v1/build", headers={"Authorization": "Bearer secret"}).json()

    assert document["source_digest"].startswith("sha256:"), document["digest_detail"]
    assert document["file_count"] > 100
    assert document["origin"] in {"wheel", "editable", "source"}


def test_the_agent_build_route_is_not_public(tmp_path) -> None:
    """A build digest and an interpreter version are a fingerprint."""

    cell_service, _ = service(tmp_path)
    client = TestClient(create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")))

    assert client.get("/api/v1/build").status_code == 401


def test_an_agent_whose_driver_cannot_be_asked_answers_null(tmp_path) -> None:
    """Three answers, and the one that matters is this one.

    A driver with no probe used to answer ``[]`` -- the same thing a backend
    with genuinely no worker says -- so a live GCE cell running an active native
    worker reported no ranks and the fleet was called in agreement without
    anything having asked it.
    """

    cell_service, _ = service(tmp_path)
    client = TestClient(create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")))

    response = client.get("/api/v1/cells/cell-a/worker-builds", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json() is None


def test_the_worker_builds_route_is_not_public(tmp_path) -> None:
    cell_service, _ = service(tmp_path)
    client = TestClient(create_agent_app(cell_service, EyvanSettings(agent_shared_secret="secret")))

    assert client.get("/api/v1/cells/cell-a/worker-builds").status_code == 401


def test_an_agent_predating_the_worker_route_reports_none(tmp_path) -> None:
    """404 keeps "cannot ask" separate from "nothing to ask"."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )

    assert client.worker_builds("cell-a") is None


def test_a_null_worker_builds_answer_reports_none_to_the_client() -> None:
    """200 with a null body is the answer of a driver that cannot be asked.

    The agent answers exactly this when its execution driver has no
    worker-builds probe, so reading it as a malformed answer put dispatch to
    every such cell into an unbounded retry.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"null")

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )

    assert client.worker_builds("cell-a") is None


def test_a_non_array_worker_builds_answer_still_raises() -> None:
    """Only the null body means "cannot be asked"; every other shape stays an error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json="cell-a")

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )

    with pytest.raises(ValueError):
        client.worker_builds("cell-a")


def test_exec_derives_its_http_timeout_from_the_command_budget() -> None:
    """The agent may hold the exec answer for the whole requested ``timeout_s``.

    With no per-request timeout the client default of 30 seconds fired first on
    a command budgeted for minutes, and the observation recorded a rank error
    while the command was still running on the agent.
    """

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"stdout": "", "stderr": "", "exit_code": 0})

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )

    client.exec_in_workload("cell-a", "wl-1", ["py-spy", "dump"], timeout_s=600.0)

    timeout = requests[0].extensions["timeout"]
    assert timeout["read"] == 630.0


def test_exec_http_timeout_stays_capped_for_out_of_contract_budgets() -> None:
    """A budget past the wire contract's ``le=600`` cannot pin the connection forever."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"stdout": "", "stderr": "", "exit_code": 0})

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )

    client.exec_in_workload("cell-a", "wl-1", ["true"], timeout_s=10_000.0)

    timeout = requests[0].extensions["timeout"]
    assert timeout["read"] == 660.0


def test_an_agent_that_predates_the_route_reports_no_build_rather_than_failing() -> None:
    """404 is an answer from a working agent, not an error.

    It is also why this is a route of its own rather than a field on
    ``AgentCapabilities``, which forbids extras: a new field there would make a
    control plane one version behind unable to decode capabilities at all.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )

    assert client.build() is None


def test_an_agent_that_answers_500_to_a_build_read_still_raises() -> None:
    """ "Cannot say" and "is broken" are different, and only the first is benign."""

    from eyvan.worker.agent.client import CellAgentUnavailableError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )

    with pytest.raises(CellAgentUnavailableError):
        client.build()


def test_an_unreachable_agent_is_not_a_missing_build() -> None:
    """An agent nobody could reach has not reported that it has no build."""

    from eyvan.worker.agent.client import CellAgentUnavailableError

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    client = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )

    with pytest.raises(CellAgentUnavailableError):
        client.build()


def test_wire_omits_unset_optionals_so_an_older_agent_still_parses() -> None:
    """A head one commit ahead must not break every allocation.

    Agent-side models are ``extra="forbid"`` on purpose. Sending an explicit
    null for an optional the agent has never heard of makes it reject the whole
    configuration -- which is how ``runtime.python: null`` killed every
    allocation and surfaced as "provider rejected capacity request" about a
    provider that was never called.
    """

    from eyvan.records import NativeRuntimeConfig
    from eyvan.worker.agent.client import _wire

    config = NativeRuntimeConfig(env={"A": "b"})
    assert config.python is None

    payload = _wire(config)

    assert "python" not in payload, "an unset optional must not be sent as an explicit null"
    assert payload["env"] == {"A": "b"}, "set fields must still be sent"
    # An agent that does know the field parses the trimmed payload identically,
    # so this costs nothing when both sides match.
    assert NativeRuntimeConfig.model_validate(payload) == config


def test_agent_capabilities_forward_the_driver_backend_declaration(tmp_path) -> None:
    """Capability-based routing reads ``capacity_backends``; dropping it here
    made every agent look like an older one that declares nothing."""

    class DeclaringCapacity(FakeCapacity):
        def capabilities(self):
            return CellCapabilities(
                provider="gcp",
                executor="eyvan-native",
                accelerator_types=("cpu", "gpu", "tpu"),
                capacity_classes=("spot", "on-demand"),
                capacity_backends=("gcp", "gcp-tpu", "gcp-compute"),
            )

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state._owner = "agent"
    service = CellAgentService(
        capacity=DeclaringCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=StaticResolver(NoopExecutor()),  # type: ignore[arg-type]
    )

    capabilities = service.capabilities()

    assert capabilities.capacity_backends == ("gcp", "gcp-tpu", "gcp-compute")
