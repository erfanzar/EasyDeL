# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Six complete features nothing could reach.

Each was built, validated and exported, and each was walled off by exactly one
line somewhere else:

* ``GpuConfig`` and ``CpuConfig`` were complete and everything below the
  decorator was already ``AcceleratorConfig``-shaped, while ``execute()`` raised
  ``TypeError("execute requires a TpuConfig")`` -- so the decorator could reach
  no hardware but TPU.
* ``max_failure_retries`` existed on the record and both request schemas and
  appeared nowhere in ``eyvan/sdk/``, so the job-failure budget was reachable
  only by hand-writing HTTP.
* ``create_schema`` stamped ``schema_revisions`` and never ``alembic_version``,
  so a database it bootstrapped could never be migrated afterwards -- Alembic
  would replay every migration against tables that already existed.
* ``can_submit_children`` gated all of in-job submission -- capability minting,
  ``EYVAN_CHILDREN_URL``, ``submit_child()``, the 256-live ceiling, the depth
  guard, generation fencing -- and no CLI option or SDK config set it. The only
  workloads that ever carried it were services, which settlement grants it to
  on their behalf, so every path a person could type left the feature off.
* the placement breadth a submitter declares -- ``additional_capacity_classes``,
  ``allowed_regions``, ``allowed_zones``, ``allowed_accelerator_types`` and
  ``named_assets`` -- was read by the scheduler and settable by nothing. The
  first of those is the capacity portfolio, so "spot, but take on-demand rather
  than sit in a queue" was not a request the CLI could make.

* the job's checkpoint contract -- ``checkpointable``, ``checkpoint_hook``,
  ``checkpoint_output_uri`` -- was carried by ``eyvan workload submit`` and by
  neither pool request schema, so ``@execute`` and ``eyvan run`` could not set
  it. ``_checkpoint`` grants ``ON_PREEMPTION`` only when the flag and a hook are
  both present, and the callable path defaults to *spot*: every callable ever
  reclaimed there restarted from zero, with ``max_preemption_retries`` buying
  more attempts and never a resumption.

The shape repeats: a field lands on the record and both schemas, the server
learns to honour it, and the client surface is the step nobody notices is
missing, because everything downstream of it already works. ``grep <field>
eyvan/sdk/`` returning nothing is the whole diagnostic, and it is worth running
against ``WorkloadDefinition.model_fields`` after adding one. Mind the false
negatives: ``allowed_regions`` appeared to be reachable only because
``eyvan policy set --allow-region`` names the unrelated *policy* allow-list.
"""

from __future__ import annotations

import asyncio
import tempfile

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.sdk.client.jobs import AcceleratorConfig, CpuConfig, GpuConfig, TpuConfig
from eyvan.sdk.client.requests import pool_callable_request, pool_workload_request
from eyvan.sdk.execution import execute
from eyvan.server.persistence.sqlalchemy import SCHEMA_REVISION, create_schema


def _noop() -> int:
    return 0


@pytest.mark.parametrize(
    "config",
    [TpuConfig(type="v5p-8"), GpuConfig(type="nvidia-tesla-a100"), CpuConfig()],
    ids=["tpu", "gpu", "cpu"],
)
def test_the_decorator_accepts_every_config_it_ships(config: AcceleratorConfig) -> None:
    """The whole defect: two of the three exported configs were refused."""

    assert execute(config)(_noop).config is config


def test_a_config_class_is_rejected_with_a_sentence_that_names_the_mistake() -> None:
    """``@execute(TpuConfig)`` -- the missing parentheses -- is the likely typo.

    A class object satisfies a data protocol's attribute check as readily as an
    instance, so a bare protocol test would accept it and fail much later,
    somewhere that mentions none of this.
    """

    with pytest.raises(TypeError, match="TpuConfig, GpuConfig, or CpuConfig instance"):
        execute(TpuConfig)  # type: ignore[arg-type]


def test_something_that_is_not_a_config_is_still_rejected() -> None:
    """Widening the gate must not remove it."""

    with pytest.raises(TypeError, match="TpuConfig, GpuConfig, or CpuConfig instance"):
        execute("v5p-8")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        TpuConfig(type="v5p-8", max_failure_retries=2),
        GpuConfig(type="a100", max_failure_retries=2),
        CpuConfig(max_failure_retries=2),
    ],
    ids=["tpu", "gpu", "cpu"],
)
def test_every_config_carries_the_job_failure_budget(config: AcceleratorConfig) -> None:
    """It existed server-side and on no client type."""

    assert config.max_failure_retries == 2


@pytest.mark.parametrize("value", [-1, 101, True], ids=["negative", "over-limit", "bool"])
def test_an_out_of_range_failure_budget_is_rejected_client_side(value: object) -> None:
    """Same bounds as the record, reported where the caller wrote the number."""

    with pytest.raises(ValueError, match="max_failure_retries"):
        CpuConfig(max_failure_retries=value)  # type: ignore[arg-type]


def test_the_two_budgets_stay_separate_in_the_wire_payload() -> None:
    """Merging them would let a deterministically broken job burn the capacity budget.

    ``max_retries`` pays for capacity the fleet lost -- a preemption, a stockout
    -- which says nothing about the job. ``max_failure_retries`` pays for the
    job's own non-zero exit, which usually repeats. That is why the second
    defaults to zero and why one number could not serve both.
    """

    request = pool_workload_request(
        org_id="acme",
        project_id="research",
        command="python train.py",
        name=None,
        environment=None,
        metadata=None,
        max_retries=5,
        max_failure_retries=2,
    )

    assert (request.max_retries, request.max_failure_retries) == (5, 2)


def test_the_callable_payload_carries_it_too() -> None:
    """Both submit routes accept it; both builders must send it."""

    from eyvan.records import CallableExecution

    request = pool_callable_request(
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-8",
        pods=1,
        capacity_class=TpuConfig(type="v5p-8").capacity_class,
        execution=CallableExecution(
            artifact_uri="eyvan://callables/" + "0" * 64,
            sha256="0" * 64,
            python="3.13",
        ),
        name=None,
        environment=None,
        metadata=None,
        max_retries=5,
        max_failure_retries=3,
    )

    assert (request.max_retries, request.max_failure_retries) == (5, 3)


def test_the_failure_budget_defaults_to_zero_on_the_wire() -> None:
    """A job that fails on its own usually fails again; re-running costs an accelerator."""

    request = pool_workload_request(
        org_id="acme",
        project_id="research",
        command="python train.py",
        name=None,
        environment=None,
        metadata=None,
        max_retries=5,
    )

    assert request.max_failure_retries == 0


def test_a_bootstrapped_database_can_still_be_migrated() -> None:
    """Without ``alembic_version`` there is no path forward from ``create_schema``.

    Alembic reads that table to decide what has already run. Finding none it
    concludes the database is empty and replays ``0002`` onward against tables
    that already exist, which fails -- so the database is stuck on whatever
    build created it, permanently.
    """

    async def bootstrap() -> tuple[str | None, str | None]:
        with tempfile.TemporaryDirectory() as directory:
            engine = create_async_engine(f"sqlite+aiosqlite:///{directory}/eyvan.db")
            try:
                await create_schema(engine)
                async with engine.connect() as connection:
                    alembic = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
                    declared = (await connection.execute(text("SELECT revision FROM schema_revisions"))).scalar_one()
                return alembic, declared
            finally:
                await engine.dispose()

    alembic, declared = asyncio.run(bootstrap())

    assert alembic == SCHEMA_REVISION
    # Both stamps, agreeing. ``open()`` reads Alembic's first, so a disagreement
    # would refuse to serve a database this very function had just created.
    assert declared == SCHEMA_REVISION


def test_the_stamped_revision_comes_from_the_migrations_not_a_constant() -> None:
    """A hand-maintained constant is the thing that drifts.

    Taking the head from the scripts means adding a migration and forgetting to
    bump ``SCHEMA_REVISION`` fails loudly at bootstrap rather than stamping a
    revision the scripts do not have.
    """

    from alembic.script import ScriptDirectory

    from eyvan.server.persistence.migrate import build_alembic_config

    assert ScriptDirectory.from_config(build_alembic_config()).get_current_head() == SCHEMA_REVISION


def test_the_run_command_can_set_the_failure_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """``grep max_failure_retries eyvan/sdk/`` used to return nothing.

    The record and both request schemas carried it; the only way to reach it was
    to hand-write the HTTP call, which defeats having a CLI at all.
    """

    from click.testing import CliRunner
    from test_cli import FakeClient, _json_argument

    from eyvan.sdk import cli
    from eyvan.server.api.schemas import PoolWorkloadSubmitRequest

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
            "run",
            "--no-wait",
            "dev-fleet/gpu-workers",
            "--max-retries",
            "5",
            "--max-failure-retries",
            "2",
            "--",
            "python",
            "train.py",
        ],
    )

    assert result.exit_code == 0, result.output
    request = PoolWorkloadSubmitRequest.model_validate(_json_argument(fake.calls[0]))
    assert (request.max_retries, request.max_failure_retries) == (5, 2)


def test_the_submit_command_can_set_the_failure_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit-intent path needs it as much as the pool path."""

    from click.testing import CliRunner
    from test_cli import FakeClient, _json_argument

    from eyvan.sdk import cli
    from eyvan.server.api.schemas import WorkloadSubmitRequest

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
            "--max-failure-retries",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    request = WorkloadSubmitRequest.model_validate(_json_argument(fake.calls[0]))
    assert request.spec.max_failure_retries == 4


def _submit(monkeypatch: pytest.MonkeyPatch, *extra: str):
    """Run ``workload submit`` against a fake client and return the request."""

    from click.testing import CliRunner
    from test_cli import FakeClient, _json_argument

    from eyvan.sdk import cli
    from eyvan.server.api.schemas import WorkloadSubmitRequest

    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    result = CliRunner().invoke(
        cli.main,
        [
            *["--user", "alice", "--org", "acme", "--project", "research"],
            *["workload", "submit", "--name", "launcher", "--command", "python launch.py"],
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return WorkloadSubmitRequest.model_validate(_json_argument(fake.calls[0]))


def test_the_submit_command_can_request_the_child_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole of in-job submission was reachable only by hand-written HTTP.

    The server side is complete and tested: the capability is minted per attempt,
    delivered as ``EYVAN_CHILDREN_URL`` plus a token, fenced by generation, capped
    at 256 live children and at a maximum tree depth, and every child is created
    under the parent's owner rather than the workload. None of it could be turned
    on by anything a person types -- ``grep can_submit_children eyvan/sdk/``
    returned nothing, exactly as ``max_failure_retries`` once did.
    """

    assert _submit(monkeypatch, "--can-submit-children").spec.can_submit_children is True


def test_the_child_capability_stays_off_when_it_is_not_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential nobody asked for is one nobody is watching.

    The flag has to be the only thing that turns it on; a default that leaked to
    ``True`` would hand every ordinary job a token it never requested.
    """

    assert _submit(monkeypatch).spec.can_submit_children is False


def test_a_pool_submitted_spot_job_can_declare_itself_checkpointable() -> None:
    """Otherwise every ``@execute`` job on spot restarts from zero, forever.

    ``_checkpoint`` in ``scheduling`` grants ``CheckpointMode.ON_PREEMPTION``
    only when ``checkpointable`` is set *and* a hook is named. The pool path --
    which is what ``@execute`` and ``eyvan run`` use, and which defaults to spot
    for callables -- carried neither field, so the mode was unreachable there:
    ``max_preemption_retries`` bought more attempts and never a resumption.
    """

    from eyvan.records import CommandExecution, ResourceGang, ResourceVector, WorkloadRecord, WorkloadState
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CheckpointMode

    spec = ControlPlaneService._pinned_pool_workload_spec(
        pool=_pool_record(),
        name="train",
        execution=CommandExecution(kind="command", command="python train.py"),
        environment={},
        secret_mounts={},
        metadata={},
        max_retries=3,
        max_failure_retries=0,
        cell_group_count=1,
        checkpointable=True,
        checkpoint_hook="python -m train.checkpoint",
        checkpoint_output_uri="gs://ckpt/run-1",
    )

    assert spec.checkpointable is True
    assert spec.checkpoint_hook == "python -m train.checkpoint"

    record = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        created_by="alice",
        state=WorkloadState.PENDING,
        spec=spec.model_copy(update={"gang": ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1))}),
    )
    from eyvan.server.service.scheduling import _checkpoint

    assert _checkpoint(record).mode is CheckpointMode.ON_PREEMPTION


def test_a_pool_job_that_asks_for_nothing_is_still_not_checkpointed() -> None:
    """The default has to stay off: a hook that does not exist is worse than none.

    The job would be stopped for a checkpoint it never wrote.
    """

    from eyvan.records import CommandExecution
    from eyvan.server.service.control_plane import ControlPlaneService

    spec = ControlPlaneService._pinned_pool_workload_spec(
        pool=_pool_record(),
        name="train",
        execution=CommandExecution(kind="command", command="python train.py"),
        environment={},
        secret_mounts={},
        metadata={},
        max_retries=3,
        max_failure_retries=0,
        cell_group_count=1,
    )

    assert spec.checkpointable is False
    assert spec.checkpoint_hook is None


@pytest.mark.asyncio
async def test_a_callable_submitted_through_the_service_keeps_its_checkpoint_contract() -> None:
    """The same thing through the public entry point rather than the builder.

    ``submit_pool_callable`` is what the callable path actually calls, so this is
    the assertion that the fields survive the whole submission and land on the
    durable record -- not merely that a helper accepts them.
    """

    from eyvan.records import CallableExecution, Principal, Role, RoleGrant
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService
    from eyvan.server.service.callable_trust import CallableProjectTrustPolicy

    service = ControlPlaneService(
        InMemoryRepository(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.upsert_capacity_pool(admin, _pool_record(), idempotency_key="pool-0")

    workload, _ = await service.submit_pool_callable(
        Principal(
            subject="alice",
            grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
        ),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-8",
        pods=1,
        execution=CallableExecution(artifact_uri="gs://trusted/train.pkl", sha256="a" * 64, python="3.12"),
        checkpointable=True,
        checkpoint_hook="python -m train.checkpoint",
        checkpoint_output_uri="gs://ckpt/run-1",
        idempotency_key="ckpt-callable-1",
    )

    assert workload.spec.checkpointable is True
    assert workload.spec.checkpoint_hook == "python -m train.checkpoint"
    assert workload.spec.checkpoint_output_uri == "gs://ckpt/run-1"


@pytest.mark.asyncio
async def test_a_callable_can_also_ask_for_the_child_capability() -> None:
    """The half of the earlier fix that was left for a wider API change.

    ``eyvan workload submit --can-submit-children`` reached the field because the
    explicit-intent schema already carried it. Neither pool schema did, so
    ``@execute`` -- the surface most likely to want to fan out -- still could not
    ask. Both halves now go through the same builder.
    """

    from eyvan.records import CallableExecution, Principal, Role, RoleGrant
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService
    from eyvan.server.service.callable_trust import CallableProjectTrustPolicy

    service = ControlPlaneService(
        InMemoryRepository(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.upsert_capacity_pool(admin, _pool_record(), idempotency_key="pool-0")

    workload, _ = await service.submit_pool_callable(
        Principal(
            subject="alice",
            grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
        ),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-8",
        pods=1,
        execution=CallableExecution(artifact_uri="gs://trusted/train.pkl", sha256="a" * 64, python="3.12"),
        can_submit_children=True,
        idempotency_key="children-callable-1",
    )

    assert workload.spec.can_submit_children is True


def _pool_record():
    """One enabled spot pool, which is what the callable path defaults to."""

    from eyvan.records import CapacityPoolRecord, ResourceVector

    resources = ResourceVector(cpu=8, memory_gib=32, tpu=4)
    return CapacityPoolRecord(
        id="research/v5p",
        fleet_name="research",
        node_type="v5p",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="p",
        region="us-east5",
        zone="us-east5-a",
        capacity=resources,
        allocatable=resources,
        accelerator_type="v5p-8",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=4,
    )


#: Spec fields a pool submitter is not meant to set, and why. Anything else that
#: ``WorkloadDefinition`` carries has to be expressible on the pool path, because
#: that path is what ``@execute`` and ``eyvan run`` use.
_PLANE_OWNED = {
    # Written by the control plane while placing and settling the workload.
    "preferred_cells",
    "preferred_labels",
    "required_labels",
    "worker_resource_scope",
    "backend_options",
}
_POOL_DERIVED = {
    # Dictated by the pool the caller named. That is the whole point of naming
    # one: the fleet YAML owns shape, provider and machine type.
    "provider",
    "region",
    "accelerator_type",
    "gang",
    "execution",
    "worker_count",
    "capacity_class",
    "required_executor",
    "allowed_regions",
    "allowed_zones",
    "allowed_accelerator_types",
    "additional_capacity_classes",
    "cell_group_count",
}


def test_the_pool_schemas_carry_every_field_a_submitter_owns() -> None:
    """Expose workload fields through pool request schemas unless explicitly plane-owned or pool-derived.

    Explicit exemptions make new spec fields fail this check when the request
    schemas do not expose them.
    """

    from eyvan.records import WorkloadDefinition
    from eyvan.server.api.schemas import PoolCallableSubmitRequest, PoolWorkloadSubmitRequest

    owned = set(WorkloadDefinition.model_fields) - _PLANE_OWNED - _POOL_DERIVED
    # `placement` is the spec's spelling of what both submit paths expose as
    # `atomic` (bool): elastic packing by default, exact-shape when true. The
    # names differ so the SDK surface stays one boolean; this exemption is the
    # claim that they are the same knob.
    assert not owned - set(PoolWorkloadSubmitRequest.model_fields) - {"placement"}
    # A callable ships its working tree inside CallableExecution rather than as a
    # separate ref, so `workspace` is the one field this path legitimately lacks.
    assert not owned - set(PoolCallableSubmitRequest.model_fields) - {"workspace", "placement"}


def test_the_run_command_reaches_every_pool_field_the_schema_gained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threading a field to the schema is not the same as making it reachable.

    Written after doing exactly that: the checkpoint contract, the child
    capability and the band all reached ``PoolWorkloadSubmitRequest`` and the two
    clients while ``eyvan run`` still had no flags for the last three, so the
    field was settable by everything except the thing a person types. This
    asserts the whole surface at once, because the gap is always the last hop.
    """

    from click.testing import CliRunner
    from test_cli import FakeClient, _json_argument

    from eyvan.sdk import cli
    from eyvan.server.api.schemas import PoolWorkloadSubmitRequest

    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    result = CliRunner().invoke(
        cli.main,
        [
            *["--user", "alice", "--org", "acme", "--project", "research"],
            *["run", "--no-wait", "dev-fleet/gpu-workers"],
            *["--checkpointable", "--checkpoint-hook", "python -m ckpt"],
            *["--checkpoint-output-uri", "gs://ckpt/run-1", "--checkpoint-grace-seconds", "120"],
            *["--can-submit-children", "--priority-band", "batch", "--intra-band-priority", "7"],
            *["--duration-s", "60", "--named-asset", "gs://corpus/shard-0"],
            *["--", "python", "train.py"],
        ],
    )

    assert result.exit_code == 0, result.output
    request = PoolWorkloadSubmitRequest.model_validate(_json_argument(fake.calls[0]))
    assert request.checkpointable is True
    assert request.checkpoint_hook == "python -m ckpt"
    assert request.checkpoint_output_uri == "gs://ckpt/run-1"
    assert request.checkpoint_grace_seconds == 120
    assert request.can_submit_children is True
    assert request.priority_band.value == "batch"
    assert request.intra_band_priority == 7
    # Added to `eyvan run` after this test was first written, which is exactly
    # how a last-hop gap survives: the schema and the client had them already.
    assert request.estimated_duration_seconds == 60
    assert request.named_assets == ("gs://corpus/shard-0",)


def test_a_spot_first_job_can_say_it_would_rather_fall_back_than_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capacity portfolio, which nothing typeable could express.

    ``capacity_class_portfolio`` puts it plainly: the core scheduler has always
    ranked a portfolio -- ``capacity_class_rank`` is the fourth key of
    ``PlacementScore`` -- and the only thing collapsing it to one entry was the
    adapter. So the single most common thing a spot user wants, "spot, but take
    on-demand rather than sit in a queue", could be asked for only by
    hand-writing the HTTP request.
    """

    from eyvan.records import CapacityClass

    spec = _submit(
        monkeypatch,
        *["--capacity-class", "spot"],
        *["--additional-capacity-class", "on_demand"],
        *["--additional-capacity-class", "reserved"],
    ).spec

    assert spec.capacity_class_portfolio == (CapacityClass.SPOT, CapacityClass.ON_DEMAND, CapacityClass.RESERVED)


def test_a_job_can_name_the_places_it_will_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Breadth, as opposed to the pin that ``--region`` already expressed.

    The scheduler reads all three (``scheduling.py`` ranks on zone, region and
    accelerator type), and a submitter could set none of them. Naming several
    regions is what lets a job chase capacity across them instead of queueing in
    the one that happens to be full.
    """

    spec = _submit(
        monkeypatch,
        *["--allowed-region", "us-east5", "--allowed-region", "europe-west4"],
        *["--allowed-zone", "us-east5-a"],
        *["--allowed-accelerator-type", "v5p-16", "--allowed-accelerator-type", "v5p-32"],
        *["--named-asset", "gs://corpus/shard-0"],
    ).spec

    assert spec.allowed_regions == ("us-east5", "europe-west4")
    assert spec.allowed_zones == ("us-east5-a",)
    assert spec.allowed_accelerator_types == ("v5p-16", "v5p-32")
    assert spec.named_assets == ("gs://corpus/shard-0",)


def test_naming_regions_while_pinning_one_is_refused_rather_than_silently_ranked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pin is a statement about *the* region; a list beside it contradicts it.

    The record refuses the combination, and the CLI has to surface that as a
    usage error rather than posting a spec the plane will reject.
    """

    from click.testing import CliRunner
    from test_cli import FakeClient

    from eyvan.sdk import cli

    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())
    result = CliRunner().invoke(
        cli.main,
        [
            *["--user", "alice", "--org", "acme", "--project", "research"],
            *["workload", "submit", "--name", "t", "--command", "python train.py"],
            *["--region", "us-east5", "--allowed-region", "europe-west4"],
        ],
    )

    assert result.exit_code != 0


def _sent_payload(method: str, **arguments) -> dict:
    """Call one sync-client method against a mock transport, return its body."""

    import json

    import httpx

    from eyvan.records import CommandExecution as _Command
    from eyvan.records import WorkloadDefinition, WorkloadRecord
    from eyvan.sdk import EyvanClient

    stored = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="t", execution=_Command(command="python train.py")),
        created_by="alice",
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json=stored.model_dump(mode="json"))

    client = EyvanClient(client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"))
    try:
        getattr(client, method)(**arguments)
    finally:
        client.close()
    return json.loads(seen[0].content)


#: Every field threaded through the pool path today, with a value that is not its
#: default. A default-valued assertion cannot tell "wired" from "accepted and
#: dropped", and these were threaded by scripted replacement across six files.
_NON_DEFAULT = {
    "checkpointable": True,
    "checkpoint_hook": "python -m ckpt",
    "checkpoint_output_uri": "gs://ckpt/run-1",
    "checkpoint_grace_seconds": 120,
    "can_submit_children": True,
    "intra_band_priority": 7,
    "estimated_duration_seconds": 60,
    "named_assets": ("gs://corpus/shard-0",),
}


def test_the_sync_client_sends_every_pool_field_it_accepts_on_a_command() -> None:
    """A parameter a method accepts and drops is unreachable with extra steps.

    The existing payload tests pin the defaults, which is exactly the shape that
    cannot catch this: a field accepted into the signature and never passed to
    the request builder still serializes to its default and still matches.
    """

    payload = _sent_payload(
        "run",
        fleet_name="research",
        node_type="cpu",
        command="python train.py",
        org_id="acme",
        project_id="research",
        priority_band="batch",
        **_NON_DEFAULT,
    )

    for field, value in _NON_DEFAULT.items():
        expected = list(value) if isinstance(value, tuple) else value
        assert payload.get(field) == expected, f"run dropped {field}"
    assert payload.get("priority_band") == "batch"


def test_the_sync_client_sends_every_pool_field_it_accepts_on_a_callable() -> None:
    """The same for the path ``@execute`` uses."""

    from eyvan.records import CallableExecution

    payload = _sent_payload(
        "submit_pool_callable",
        fleet_name="research",
        accelerator_type="v5p-8",
        execution=CallableExecution(artifact_uri="gs://a/train.pkl", sha256="a" * 64, python="3.12"),
        org_id="acme",
        project_id="research",
        priority_band="batch",
        **_NON_DEFAULT,
    )

    for field, value in _NON_DEFAULT.items():
        expected = list(value) if isinstance(value, tuple) else value
        assert payload.get(field) == expected, f"submit_pool_callable dropped {field}"
    assert payload.get("priority_band") == "batch"


def test_an_operator_can_turn_on_exec_for_a_gcp_fleet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the operator's explicit exec opt-in through the GCP driver's admission capabilities."""

    from eyvan.capacity.drivers.gcp import GcpTpuCapacityDriver

    monkeypatch.setenv("EYVAN_CAPACITY_SUPPORTS_EXEC", "1")
    assert GcpTpuCapacityDriver().capabilities().supports_exec is True


def test_exec_stays_off_for_a_fleet_that_did_not_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running arbitrary commands beside a trainer is opt-in, not a default."""

    from eyvan.capacity.drivers.gcp import GcpTpuCapacityDriver

    monkeypatch.delenv("EYVAN_CAPACITY_SUPPORTS_EXEC", raising=False)
    assert GcpTpuCapacityDriver().capabilities().supports_exec is False
