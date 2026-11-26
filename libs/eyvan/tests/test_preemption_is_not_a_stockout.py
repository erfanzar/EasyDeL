# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Telling a reclaimed machine apart from an empty zone.

Two places reconstructed *why* a provider failed by searching its prose. One
asked whether the error text contained "quota" or "capacity"; the other simply
recorded every non-ready observation that was not an explicit quota halt as a
stockout. Both fed ``CapacityBelief``, which decides where the next workload is
placed.

So an ordinary spot reclamation -- the normal ending of a spot TPU, and the
common case on this fleet -- taught the placement engine that a zone had run out
of accelerators. It then routed away from a zone that was working, toward one
that might genuinely be full. A message like "capacity pool not found", a
configuration mistake, did the same thing.

The provider knows the difference and always did: ``GcpTpuNodeState.PREEMPTED``
has been in ``TERMINAL_NODE_STATES`` the whole time. What was missing was a way
to carry it, so it arrived as English inside a sentence about terminality.
"""

from __future__ import annotations

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, provision_outcome
from eyvan.domain.availability import ProvisionOutcome

# --------------------------------------------------------------------------- classification


def test_a_preemption_is_not_recorded_as_a_stockout() -> None:
    """The defect, stated directly.

    The provider had the capacity and took it back. That says nothing about
    whether the pool can satisfy the next request, and saying otherwise steers
    placement away from a zone that is fine.
    """

    assert provision_outcome(CapacityProviderFailureCode.PREEMPTED) is ProvisionOutcome.INTERRUPTED


def test_a_stockout_is_still_recorded_as_one() -> None:
    """The signal the belief exists for must survive being made explicit."""

    assert provision_outcome(CapacityProviderFailureCode.STOCKOUT) is ProvisionOutcome.STOCKOUT


def test_a_quota_failure_is_not_a_stockout() -> None:
    """Another zone can satisfy a stockout. Nothing satisfies an exhausted quota.

    Collapsing them sends the autoscaler looking for capacity that exists and is
    not the problem.
    """

    assert provision_outcome(CapacityProviderFailureCode.QUOTA_EXCEEDED) is ProvisionOutcome.QUOTA_ERROR


def test_a_configuration_mistake_is_not_evidence_about_capacity() -> None:
    """A rejected manifest says nothing about a zone's inventory."""

    assert provision_outcome(CapacityProviderFailureCode.CONFIGURATION) is ProvisionOutcome.PROVIDER_ERROR


def test_an_unreported_reason_defaults_to_a_provider_error_not_a_stockout() -> None:
    """The direction of the default is the whole point.

    Defaulting the unknown to "this pool is empty" is how a belief that drives
    placement gets poisoned by causes that have nothing to do with capacity --
    which is exactly what the previous code did for every status except an
    explicit quota halt.
    """

    assert provision_outcome(None, CapacityStatus.NEEDS_RECREATE) is ProvisionOutcome.PROVIDER_ERROR
    assert provision_outcome(None, CapacityStatus.UNAVAILABLE) is ProvisionOutcome.PROVIDER_ERROR
    assert provision_outcome(None, CapacityStatus.HALTED_QUOTA) is ProvisionOutcome.QUOTA_ERROR


def test_every_failure_code_has_an_outcome() -> None:
    """A code added without one would raise on the path that records a failure.

    Which is the path that runs when something is already going wrong -- the
    worst possible place for a KeyError.
    """

    for code in CapacityProviderFailureCode:
        assert isinstance(provision_outcome(code), ProvisionOutcome)


# --------------------------------------------------------------------------- the provider


def test_a_preempted_tpu_reports_a_typed_reason() -> None:
    """The observation carries the reason rather than only describing it.

    ``"TPU VM is terminal (PREEMPTED)"`` is a sentence for a human. The control
    plane cannot route on a sentence without parsing it, which is the practice
    this replaces -- so the same real observation path is driven here and the
    typed field is what is asserted.
    """

    import json

    from test_gcp_provider import UID, _multislice_qr, _multislice_record, result

    from eyvan.capacity.providers.gcp.queued_resources import GcloudTpuProvider

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr()))
        node_id = args[args.index("describe") + 1]
        return result(
            stdout=json.dumps(
                {
                    "state": "PREEMPTED" if node_id.endswith("-1") else "READY",
                    "acceleratorType": "v5p-16",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": "10.2.0.4" if node_id.endswith("-0") else "10.2.0.5"}],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(_multislice_record())

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.failure_code is CapacityProviderFailureCode.PREEMPTED
    assert provision_outcome(observation.failure_code) is ProvisionOutcome.INTERRUPTED


def test_a_terminal_state_that_is_not_preemption_reports_no_reason() -> None:
    """Only preemption is claimed, because only preemption is known.

    A TERMINATED or STOPPED node might be anything; asserting a reason for it
    would be the same guess in a new place.
    """

    import json

    from test_gcp_provider import UID, _multislice_qr, _multislice_record, result

    from eyvan.capacity.providers.gcp.queued_resources import GcloudTpuProvider

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr()))
        node_id = args[args.index("describe") + 1]
        return result(
            stdout=json.dumps(
                {
                    "state": "TERMINATED" if node_id.endswith("-1") else "READY",
                    "acceleratorType": "v5p-16",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": "10.2.0.4" if node_id.endswith("-0") else "10.2.0.5"}],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(_multislice_record())

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.failure_code is None


# --------------------------------------------------------------------------- end to end


async def _observed_cell(failure_code, *, state=CapacityStatus.NEEDS_RECREATE):
    """Reconcile one cell through the real controller and return the stored record."""

    from test_controller import FakeAgent, principal

    from eyvan.config import EyvanSettings
    from eyvan.records import CellRecord, CellState, ResourceVector, Role
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = FakeAgent()
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=lambda _cell: agent)
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="local",
            region="local",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64),
            allocatable=ResourceVector(cpu=8, memory_gib=64),
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )
    assert await controller.dispatch_outbox_once() == 1
    agent.capacity_state = state
    agent.capacity_failure_code = failure_code
    await controller.observe_cells_once()
    async with repository.transaction() as tx:
        return await tx.get_cell("cell-a")


async def test_a_reported_preemption_does_not_cool_down_a_working_zone() -> None:
    """The consequence, not just the label.

    A stockout belief sets an exponential cooldown -- 60s, then 120s, then 240s,
    up to an hour -- during which the planner will not place into that pool. On a
    spot fleet, where reclamation is the ordinary ending, every preemption used
    to extend that penalty against a zone that was working perfectly.
    """

    cell = await _observed_cell(CapacityProviderFailureCode.PREEMPTED)

    assert cell is not None
    assert cell.availability.interruptions >= 1
    assert cell.availability.cooldown_until is None
    assert cell.availability.consecutive_failures == 0
    assert cell.unavailable_reason == CapacityProviderFailureCode.PREEMPTED.value


async def test_a_reported_stockout_still_cools_the_pool_down() -> None:
    """The penalty has to survive being made conditional, or the belief is useless."""

    cell = await _observed_cell(CapacityProviderFailureCode.STOCKOUT)

    assert cell is not None
    assert cell.availability.cooldown_until is not None
    assert cell.availability.consecutive_failures >= 1
    assert cell.availability.interruptions == 0


async def test_an_agent_that_reports_no_reason_is_not_assumed_to_mean_stockout() -> None:
    """An older agent, or a provider that simply did not say.

    Before, silence *was* the stockout signal for every status but a quota halt,
    so the least informative observation produced the most consequential belief.
    """

    cell = await _observed_cell(None)

    assert cell is not None
    assert cell.availability.failure_evidence >= 1
    # A provider error cools down too, but on a 15-second base rather than 60 --
    # the difference between "try again shortly" and "assume this zone is empty".
    assert cell.availability.cooldown_until is not None
    assert cell.availability.interruptions == 0


# --------------------------------------------------------------------------- the wire


def test_the_agent_forwards_a_drivers_reason_rather_than_dropping_it(tmp_path) -> None:
    """The seam between the driver and the control plane.

    Every other test here can pass while the agent quietly drops the field --
    the control-plane tests construct an observation directly, and the driver
    tests never leave the driver. This is the one hop neither covers, and it is
    the hop the reason has to survive to matter.
    """

    from test_agent import NoopExecutor, StaticResolver

    from eyvan.capacity.drivers.base import CellObservation
    from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
    from eyvan.worker.agent.service import CellAgentService

    class PreemptedCapacity:
        def capabilities(self):
            from eyvan.capacity.drivers.base import CellCapabilities

            return CellCapabilities(
                provider="fake",
                executor="native-local",
                accelerator_types=("tpu",),
                capacity_classes=("spot",),
            )

        def observe(self, cell_id):
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.NEEDS_RECREATE,
                generation=3,
                detail="TPU VM is terminal (PREEMPTED)",
                failure_code=CapacityProviderFailureCode.PREEMPTED,
            )

        def reconcile(self, cell_id, _context):
            return self.observe(cell_id)

        def release(self, cell_id, _context):
            return self.observe(cell_id)

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state._owner = "agent"
    agent = CellAgentService(
        capacity=PreemptedCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=StaticResolver(NoopExecutor()),  # type: ignore[arg-type]
    )

    assert agent.observe("cell-a").failure_code is CapacityProviderFailureCode.PREEMPTED


def test_the_driver_persists_a_reason_so_a_later_observation_still_has_it(tmp_path) -> None:
    """``observe`` re-reads the durable record and never calls the provider.

    So a reason that lived only on the reconcile that produced it would be
    correct once and gone by the next poll -- and the control plane polls far
    more often than it reconciles.
    """

    from test_gcp_driver import FakeProvider, context, desired, make_driver

    from eyvan.capacity.providers.gcp.queued_resources import ProviderObservation

    provider = FakeProvider()
    provider.reports = [
        ProviderObservation(
            CapacityStatus.NEEDS_RECREATE,
            "TPU VM is terminal (PREEMPTED)",
            failure_code=CapacityProviderFailureCode.PREEMPTED,
        )
    ]
    _state, _provider, driver = make_driver(tmp_path, provider)

    applied = driver.apply_capacity("cell-a", desired(), context("request-1"))
    polled = driver.observe("cell-a")

    assert applied.failure_code is CapacityProviderFailureCode.PREEMPTED
    assert polled.failure_code is CapacityProviderFailureCode.PREEMPTED


def test_a_recovered_cell_stops_carrying_its_old_reason(tmp_path) -> None:
    """A sticky reason would make a healthy cell keep reporting why it once failed.

    Which is worse than losing it: the belief would keep being told about a
    preemption that already ended.
    """

    from test_gcp_driver import FakeProvider, context, desired, make_driver

    from eyvan.capacity.providers.gcp.queued_resources import ProviderObservation

    provider = FakeProvider()
    provider.reports = [
        ProviderObservation(
            CapacityStatus.NEEDS_RECREATE,
            "TPU VM is terminal (PREEMPTED)",
            failure_code=CapacityProviderFailureCode.PREEMPTED,
        ),
        ProviderObservation(CapacityStatus.HEALTHY, "ready"),
    ]
    _state, _provider, driver = make_driver(tmp_path, provider)

    driver.apply_capacity("cell-a", desired(), context("request-1"))
    recovered = driver.apply_capacity("cell-a", desired(), context("request-2"))

    assert recovered.state is CapacityStatus.HEALTHY
    assert recovered.failure_code is None
    assert driver.observe("cell-a").failure_code is None


# --------------------------------------------------------------------------- the budget


async def _lost_workload(*, failure_code, max_retries=1, max_preemption_retries=1, spent_preemptions=0):
    """Lose one running workload's cell and return the workload after recovery."""

    from test_controller import FakeAgent, principal

    from eyvan.config import EyvanSettings
    from eyvan.records import (
        CellRecord,
        CellState,
        CommandExecution,
        ResourceVector,
        Role,
        WorkloadDefinition,
        WorkloadState,
    )
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = FakeAgent()
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=lambda _cell: agent)
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="local",
            region="local",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64),
            allocatable=ResourceVector(cpu=8, memory_gib=64),
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )
    assert await controller.dispatch_outbox_once() == 1
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            checkpointable=True,
            checkpoint_hook="python -m save",
            checkpoint_output_uri="gs://ckpt",
            max_retries=max_retries,
            max_preemption_retries=max_preemption_retries,
        ),
        idempotency_key="train",
    )
    await controller.reconcile_once()
    assert await controller.dispatch_outbox_once() == 1
    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
        assert stored is not None
        await tx.put_workload(
            stored.model_copy(
                update={
                    "state": WorkloadState.RUNNING,
                    "checkpoint_uri": "gs://ckpt/step-1",
                    "preemption_count": spent_preemptions,
                }
            )
        )
    agent.capacity_state = CapacityStatus.NEEDS_RECREATE
    agent.capacity_failure_code = failure_code
    await controller.observe_cells_once()
    await controller.observe_workloads_once()
    async with repository.transaction() as tx:
        return await tx.get_workload(workload.id)


async def test_a_reclamation_spends_the_preemption_budget_not_the_capacity_one() -> None:
    """The two failures mean opposite things about whether running again can work.

    Charging both to ``max_retries`` means a spot job spends its whole allowance
    on the failure that was always going to happen, then gives up the first time
    something is genuinely wrong.
    """

    workload = await _lost_workload(failure_code=CapacityProviderFailureCode.PREEMPTED)

    assert workload is not None
    assert workload.preemption_count == 1
    assert workload.capacity_loss_count == 0


async def test_an_ordinary_capacity_loss_still_spends_the_capacity_budget() -> None:
    """The existing accounting has to survive the split, or every job's limit moved."""

    workload = await _lost_workload(failure_code=None)

    assert workload is not None
    assert workload.capacity_loss_count == 1
    assert workload.preemption_count == 0


async def test_an_exhausted_preemption_budget_stops_the_workload() -> None:
    """A separate budget is only a budget if it can run out.

    Otherwise "give preemption its own allowance" is "give preemption an
    unlimited allowance", which is how a broken image on spot bills forever.
    """

    from eyvan.records import WorkloadState

    workload = await _lost_workload(
        failure_code=CapacityProviderFailureCode.PREEMPTED,
        max_preemption_retries=1,
        spent_preemptions=1,
    )

    assert workload is not None
    assert workload.state in {WorkloadState.FAILED, WorkloadState.CANCELLED}
    assert workload.preemption_count == 1


async def test_a_reclamation_does_not_consult_the_capacity_budget() -> None:
    """A job with no capacity-loss allowance left must still survive a reclamation.

    This is the distinguishing case: with one shared budget the exhausted
    ``max_retries`` would end the workload, and the split is what keeps it alive.
    """

    from eyvan.records import WorkloadState

    workload = await _lost_workload(
        failure_code=CapacityProviderFailureCode.PREEMPTED,
        max_retries=0,
        max_preemption_retries=5,
    )

    assert workload is not None
    assert workload.state is WorkloadState.PENDING
    assert workload.preemption_count == 1


def test_a_submitter_can_set_the_preemption_budget_from_the_cli(monkeypatch) -> None:
    """A budget only the default ever reaches is a constant with extra steps."""

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
            "submit",
            "--name",
            "train",
            "--command",
            "python train.py",
            "--max-preemption-retries",
            "25",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _json_argument(fake.calls[0])["spec"]["max_preemption_retries"] == 25


def test_a_callable_config_carries_its_preemption_budget(tmp_path) -> None:
    """``@execute`` on spot is the case this budget exists for."""

    import json

    import httpx

    from eyvan.artifacts import LocalArtifactStore
    from eyvan.records import CallableExecution, WorkloadDefinition, WorkloadRecord
    from eyvan.sdk import EyvanCallableSubmitter, EyvanClient
    from eyvan.sdk.execution import CallableSubmission, TpuConfig
    from eyvan.worker.runtime.callables import dump_callable

    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri="gs://artifacts/results/run-1.result",
    )
    pending = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=execution),
        created_by="alice",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=pending.model_dump(mode="json"))

    client = EyvanClient(client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"))
    EyvanCallableSubmitter(client, store).submit_callable(
        CallableSubmission(
            invocation_id="call-1",
            fleet="research",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v5p-16", max_preemption_retries=40),
            artifact=artifact,
            result_uri="gs://artifacts/results/run-1.result",
        )
    )

    assert json.loads(requests[0].content)["max_preemption_retries"] == 40
    client.close()


# --------------------------------------------------------------------------- compute


def test_a_reclaimed_spot_vm_reports_a_typed_reason() -> None:
    """GCE has no PREEMPTED status: a reclaimed Spot VM just reaches TERMINATED.

    So the purchase class is what distinguishes it, and the class is already
    validated against the spec on the same call.
    """

    import json

    from test_gcp_compute_provider import instance_payload, record, result

    from eyvan.capacity.providers.gcp.compute_instances import GcloudComputeProvider

    provider = GcloudComputeProvider(
        runner=lambda _args: result(stdout=json.dumps(instance_payload(status="TERMINATED")))
    )

    observation = provider.ensure(record())

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.failure_code is CapacityProviderFailureCode.PREEMPTED


def test_a_suspended_spot_vm_is_not_called_a_preemption() -> None:
    """SUSPENDED is not reclamation, and claiming it is would be the same guess.

    Bounded to what the evidence supports: only Spot, and only TERMINATED.
    """

    import json

    from test_gcp_compute_provider import instance_payload, record, result

    from eyvan.capacity.providers.gcp.compute_instances import GcloudComputeProvider

    provider = GcloudComputeProvider(
        runner=lambda _args: result(stdout=json.dumps(instance_payload(status="SUSPENDED")))
    )

    assert provider.ensure(record()).failure_code is None


async def test_a_capacity_failure_records_what_actually_happened() -> None:
    """The reason is a code; the detail is the sentence an operator needs.

    A cell that died on a cell-agent config rejection reported "provider
    rejected capacity request" -- about a provider that had never been called --
    while the real pydantic error existed only in a log file on whichever
    machine ran the head. Diagnosing it took SSH and grep.
    """

    from test_controller import FakeAgent, principal

    from eyvan.config import EyvanSettings
    from eyvan.records import CellRecord, CellState, OutboxRecord, ResourceVector, Role
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=lambda _cell: FakeAgent())
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-bad-config",
            provider="local",
            region="local",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64),
            allocatable=ResourceVector(cpu=8, memory_gib=64),
        ),
        idempotency_key="cell-bad-config",
    )
    async with repository.transaction() as tx:
        cell = await tx.get_cell("cell-bad-config")
    assert cell is not None
    diagnostic = (
        "invalid configuration: 1 validation error for NativeRuntimeConfig\npython\n  Extra inputs are not permitted"
    )

    await controller._record_capacity_failure(
        OutboxRecord(
            id="cmd-1",
            topic="capacity.apply",
            aggregate_type="cell",
            aggregate_id=cell.id,
            idempotency_key="cmd-1",
            generation=cell.generation,
        ),
        diagnostic,
        failure_code=None,
    )

    async with repository.transaction() as tx:
        updated = await tx.get_cell("cell-bad-config")
    assert updated is not None
    assert updated.unavailable_detail is not None
    assert "Extra inputs are not permitted" in updated.unavailable_detail, (
        "the operator must be able to read the real failure without SSHing the head"
    )
