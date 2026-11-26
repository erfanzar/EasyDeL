# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Whether the agents holding a fleet's capacity run the code that was deployed.

Every component of this package reports the same version string, so that string
answers nothing during an upgrade. A hardware proof has already been run against
a stale agent and read as evidence, and nothing in the control plane could have
shown it at the time.

The property throughout is that three answers stay three. An agent that could
not be reached, and an agent that answered but has no digest to give, are both
"not shown to agree" -- and neither may be reported as agreement, because the
run this sweep exists to protect is the one where somebody is about to trust it.
"""

from __future__ import annotations

import typing as tp

import pytest

from eyvan._internal.build_identity import build_identity
from eyvan.records import CapacityClass, CapacityPoolRecord, CellRecord, CellState, Principal, Role, RoleGrant
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.client import CellAgentUnavailableError
from eyvan.worker.agent.models import AgentBuild


def fleet_admin() -> Principal:
    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _pool(pool_id: str = "research/cpu", *, endpoint: str = "http://agent-a:8090") -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type=pool_id.rsplit("/", 1)[-1],
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 8, "memory_gib": 32},
        allocatable={"cpu": 8, "memory_gib": 32},
        capacity_class=CapacityClass.SPOT,
        executor="eyvan-native",
        agent_endpoint=endpoint,
        max_cells=4,
    )


def _cell(cell_id: str, *, pool: CapacityPoolRecord, endpoint: str, state: CellState = CellState.READY) -> CellRecord:
    return pool.make_cell(cell_id, worker_count=1).model_copy(update={"state": state, "endpoint": endpoint})


class _Agent:
    """One agent's answer to being asked what it, and its workers, are."""

    #: Distinct from ``None``, which is the agent answering "this driver cannot
    #: be asked". The default has to mean neither, or the two cases the sweep
    #: must keep apart cannot both be written down here.
    UNSET: tp.ClassVar[object] = object()

    def __init__(
        self,
        build: AgentBuild | None | Exception,
        workers: list[dict[str, object]] | None | Exception | object = UNSET,
    ) -> None:
        self._build = build
        # Default: a cell whose backend runs the workload in the agent process,
        # so there is no separate worker to ask and nothing is unknown.
        self._workers: object = [] if workers is _Agent.UNSET else workers

    def build(self) -> AgentBuild | None:
        if isinstance(self._build, Exception):
            raise self._build
        return self._build

    def worker_builds(self, cell_id: str) -> list[dict[str, object]] | None:
        del cell_id
        if isinstance(self._workers, Exception):
            raise self._workers
        return tp.cast("list[dict[str, object]] | None", self._workers)


def _agent_build(digest: str | None) -> AgentBuild:
    return AgentBuild(
        version="0.0.1",
        origin="wheel",
        source_digest=digest,
        file_count=239 if digest else 0,
        python="3.12.3",
        platform="linux-x86_64",
    )


async def _sweep(agents: dict[str, _Agent | Exception], *, cells: tp.Sequence[CellRecord], pools) -> dict:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())

    def factory(cell: CellRecord):
        answer = agents[cell.endpoint or ""]
        if isinstance(answer, Exception):
            raise answer
        return answer

    controller = EyvanController(repository, service, agent_factory=factory)  # type: ignore[arg-type]
    await service.apply_fleet(fleet_admin(), "research", tuple(pools), idempotency_key="fleet-v1")
    async with repository.transaction() as transaction:
        for cell in cells:
            await transaction.put_cell(cell)
    return await controller.fleet_builds()


@pytest.mark.asyncio
async def test_a_fleet_whose_agents_all_run_this_build_is_reported_as_agreeing() -> None:
    """The only shape that may report agreement: every agent asked, every one matching."""

    plane = build_identity().source_digest
    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep({"http://agent-a:8090": _Agent(_agent_build(plane))}, cells=[cell], pools=[pool])

    assert report["in_agreement"] is True
    assert report["plane"]["source_digest"] == plane
    assert report["agents"][0]["build"]["source_digest"] == plane


@pytest.mark.asyncio
async def test_a_worker_running_other_code_is_skew_even_when_its_agent_agrees() -> None:
    """The worker executes the workload, and its wheel outlives a re-staged head.

    Observed while this was written: the head's wheel was replaced twice and the
    running TPU workers kept the one their cell was provisioned with. An agent
    that agrees says nothing about them.
    """

    plane = build_identity().source_digest
    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep(
        {
            "http://agent-a:8090": _Agent(
                _agent_build(plane),
                workers=[{"rank": 0, "endpoint": "http://w0:8100", "build": {"source_digest": "sha256:last-week"}}],
            )
        },
        cells=[cell],
        pools=[pool],
    )

    assert report["in_agreement"] is False


@pytest.mark.asyncio
async def test_a_worker_that_could_not_say_is_not_agreement() -> None:
    """A rank that answered nothing has not agreed with anything."""

    plane = build_identity().source_digest
    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep(
        {
            "http://agent-a:8090": _Agent(
                _agent_build(plane),
                workers=[{"rank": 0, "endpoint": "http://w0:8100", "build": None}],
            )
        },
        cells=[cell],
        pools=[pool],
    )

    assert report["in_agreement"] is None


@pytest.mark.asyncio
async def test_an_agent_that_cannot_be_asked_about_its_workers_is_not_agreement() -> None:
    """An agent predating the worker route leaves the ranks unestablished."""

    plane = build_identity().source_digest
    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep(
        {"http://agent-a:8090": _Agent(_agent_build(plane), workers=RuntimeError("404"))},
        cells=[cell],
        pools=[pool],
    )

    assert report["in_agreement"] is None
    assert report["agents"][0]["workers"] is None


@pytest.mark.asyncio
async def test_an_agent_running_other_code_is_reported_as_skew() -> None:
    """The failure this exists for, stated as a fact rather than a suspicion."""

    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep(
        {"http://agent-a:8090": _Agent(_agent_build("sha256:some-other-build"))},
        cells=[cell],
        pools=[pool],
    )

    assert report["in_agreement"] is False


@pytest.mark.asyncio
async def test_an_agent_that_cannot_be_reached_is_not_agreement() -> None:
    """ "Nothing disagreed" is not "everything agreed", and this is when it matters."""

    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep(
        {"http://agent-a:8090": CellAgentUnavailableError("no route to host")},
        cells=[cell],
        pools=[pool],
    )

    assert report["in_agreement"] is None
    assert report["agents"] == []
    assert report["unreachable"][0]["endpoint"] == "http://agent-a:8090"


@pytest.mark.asyncio
async def test_an_agent_that_predates_the_route_is_not_agreement_either() -> None:
    """It answered, and what it said was that it cannot say."""

    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep({"http://agent-a:8090": _Agent(None)}, cells=[cell], pools=[pool])

    assert report["in_agreement"] is None
    assert report["agents"][0]["build"] is None
    assert report["unreachable"] == []


@pytest.mark.asyncio
async def test_one_stale_agent_outranks_every_agreeing_one() -> None:
    """Proven skew is a fact; the others being fine does not soften it."""

    plane = build_identity().source_digest
    pool_a = _pool("research/cpu", endpoint="http://agent-a:8090")
    pool_b = _pool("research/gpu", endpoint="http://agent-b:8090")

    report = await _sweep(
        {
            "http://agent-a:8090": _Agent(_agent_build(plane)),
            "http://agent-b:8090": _Agent(_agent_build("sha256:yesterday")),
        },
        cells=[
            _cell("cell-a", pool=pool_a, endpoint="http://agent-a:8090"),
            _cell("cell-b", pool=pool_b, endpoint="http://agent-b:8090"),
        ],
        pools=[pool_a, pool_b],
    )

    assert report["in_agreement"] is False


@pytest.mark.asyncio
async def test_a_fleet_with_no_cells_reports_no_agreement_and_names_the_pools() -> None:
    """A sweep that contacted nobody must not read as a clean bill of health.

    This is the same defect the orphan scan already had once: an empty list is
    indistinguishable from an empty answer, and it is read at exactly the moment
    somebody wants reassurance.
    """

    pool = _pool()

    report = await _sweep({}, cells=[], pools=[pool])

    assert report["in_agreement"] is None
    assert report["agents"] == []
    assert report["unscanned_pools"] == [pool.id]


@pytest.mark.asyncio
async def test_a_deleted_cell_is_not_probed() -> None:
    """Its agent is not holding anything, and the endpoint may be gone with it."""

    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090", state=CellState.DELETED)

    report = await _sweep({}, cells=[cell], pools=[pool])

    assert report["agents"] == []
    assert report["unreachable"] == []
    assert report["unscanned_pools"] == [pool.id]


# --------------------------------------------------------------------------- the operator's surface


class _FakeClient:
    """Answer the builds route with one prepared report."""

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def request(self, method: str, path: str, **kwargs: object) -> object:
        del method, kwargs
        self.calls.append(path)
        return self.payload

    def replace_identity_headers(self, headers: object) -> None:
        del headers


def _run_builds(monkeypatch, payload: object):
    from click.testing import CliRunner

    from eyvan.sdk import cli

    fake = _FakeClient(payload)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    result = CliRunner().invoke(
        cli.main, ["--user", "alice", "--org", "acme", "--project", "research", "capacity", "builds"]
    )
    return result, fake


def _report(**overrides: object) -> dict:
    document: dict[str, object] = {
        "plane": {"version": "0.0.1", "source_digest": "sha256:plane", "digest_detail": ""},
        "agents": [
            {
                "endpoint": "http://agent-a:8090",
                "cell_id": "cell-a",
                "pool_id": "p",
                "build": {"source_digest": "sha256:plane"},
                "workers": [],
            }
        ],
        "unreachable": [],
        "unscanned_pools": [],
        "in_agreement": True,
    }
    document.update(overrides)
    return document


def test_the_command_reaches_the_route_and_reports_agreement(monkeypatch) -> None:
    result, fake = _run_builds(monkeypatch, _report())

    assert result.exit_code == 0, result.output
    assert fake.calls == ["/api/v1/capacity/builds"]
    assert "same build" in result.output


def test_the_command_fails_on_proven_skew(monkeypatch) -> None:
    """An operator who does not read the output must still be stopped."""

    result, _fake = _run_builds(
        monkeypatch,
        _report(
            agents=[{"endpoint": "http://agent-a:8090", "pool_id": "p", "build": {"source_digest": "sha256:old"}}],
            in_agreement=False,
        ),
    )

    assert result.exit_code != 0
    assert "DIFFERENT BUILD" in result.output


def test_the_command_fails_when_agreement_could_not_be_shown(monkeypatch) -> None:
    """The answer this whole feature exists to keep separate from a pass."""

    result, _fake = _run_builds(
        monkeypatch,
        _report(
            agents=[],
            unreachable=[{"endpoint": "http://agent-a:8090", "detail": "no route to host"}],
            in_agreement=None,
        ),
    )

    assert result.exit_code != 0
    assert "could not be shown to agree" in result.output


def test_the_command_shows_a_rank_running_other_code(monkeypatch) -> None:
    """The rank is what ran the workload; naming it is the point of the sweep."""

    result, _fake = _run_builds(
        monkeypatch,
        _report(
            agents=[
                {
                    "endpoint": "http://agent-a:8090",
                    "cell_id": "cell-a",
                    "pool_id": "p",
                    "build": {"source_digest": "sha256:plane"},
                    "workers": [
                        {"rank": 0, "endpoint": "http://w0:8100", "build": {"source_digest": "sha256:plane"}},
                        {"rank": 1, "endpoint": "http://w1:8100", "build": {"source_digest": "sha256:last-week"}},
                    ],
                }
            ],
            in_agreement=False,
        ),
    )

    assert result.exit_code != 0
    assert "rank 1 http://w1:8100  DIFFERENT BUILD" in result.output


def test_the_command_says_when_a_cells_workers_could_not_be_asked(monkeypatch) -> None:
    """An agent predating the worker route leaves its ranks unestablished."""

    result, _fake = _run_builds(
        monkeypatch,
        _report(
            agents=[
                {
                    "endpoint": "http://agent-a:8090",
                    "cell_id": "cell-a",
                    "pool_id": "p",
                    "build": {"source_digest": "sha256:plane"},
                    "workers": None,
                }
            ],
            in_agreement=None,
        ),
    )

    assert result.exit_code != 0
    assert "its workers could not be asked" in result.output


@pytest.mark.asyncio
async def test_a_malformed_worker_entry_is_an_unknown_rank_not_a_crash() -> None:
    """This is a read-only operator command over data another process sent.

    An agent answering something unexpected must leave the fleet unproven, not
    raise out of the route an operator is using to find out what is wrong.
    """

    plane = build_identity().source_digest
    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep(
        {"http://agent-a:8090": _Agent(_agent_build(plane), workers=["not-a-document"])},
        cells=[cell],
        pools=[pool],
    )

    assert report["in_agreement"] is None


@pytest.mark.asyncio
async def test_a_cell_whose_driver_cannot_be_asked_is_not_agreement() -> None:
    """Found on a live CPU cell, inside the check meant to prevent exactly this.

    The agent returned an empty worker list for a driver that had no probe, the
    sweep read that as "this backend has no separate worker", and reported the
    fleet in agreement -- while a native worker was active on the VM and had
    never been asked. An empty list now means only what it says.
    """

    plane = build_identity().source_digest
    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep(
        {"http://agent-a:8090": _Agent(_agent_build(plane), workers=None)},
        cells=[cell],
        pools=[pool],
    )

    assert report["in_agreement"] is None
    assert report["agents"][0]["workers"] is None


@pytest.mark.asyncio
async def test_a_backend_with_genuinely_no_worker_can_still_agree() -> None:
    """The other side of the same distinction, so the fix is not just stricter.

    A local driver runs the workload in the agent's own process. There is no
    rank to ask, the agent's own build covers it, and that is agreement.
    """

    plane = build_identity().source_digest
    pool = _pool()
    cell = _cell("cell-a", pool=pool, endpoint="http://agent-a:8090")

    report = await _sweep(
        {"http://agent-a:8090": _Agent(_agent_build(plane), workers=[])},
        cells=[cell],
        pools=[pool],
    )

    assert report["in_agreement"] is True
