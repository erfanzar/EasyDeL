# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fenced managed staging and eviction, one cell at a time.

The shape of every pass here is the same three steps, and the order is the
point: read and reserve in a transaction, close it, do the round trip, then
reopen a transaction to record what happened. A staging fetch is minutes of
network for hundreds of gigabytes; holding a row lock across it would park a
PostgreSQL transaction for the length of a dataset pull, and every other writer
touching that cell behind it.

Between the two transactions the world can move, so every write re-reads and
re-checks the generation it was planned against. A cell that was replaced while
its dataset was being fetched gets nothing written against it: the row would
describe a disk that no longer exists.

Two phases rather than one, because they fail differently. Staging that cannot
reach a host costs a cache miss; eviction that deletes the wrong thing costs a
running job its data. So eviction plans first, deletes only when an operator has
switched deletion on, and never touches an asset a live workload declared.

**Deployment order: agents first, controller second.** Staging reaches a host
through one cell-agent wire method, ``POST /api/v1/cells/{cell_id}/stage-actions``,
which older agents do not serve. :class:`CellAgentStageExecutor` reports that as
:class:`CellStageUnsupportedError`, and every phase here treats it exactly as it
treats an unconfigured executor: nothing staged, nothing recorded, nothing
charged, nothing counted. A control plane upgraded first is therefore inert
rather than broken, and the feature appears when the agents beneath it are
upgraded.

The binding itself changes no default. A deployment that leaves
``EYVAN_CELL_STAGE_BUDGET_BYTES`` at zero -- which is every deployment until an
operator sizes a cache -- never reaches the executor at all, because a zero
budget disables the policy before any of this runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import typing as tp
import uuid
from datetime import datetime

import httpx

from eyvan._internal.observability import trace_span
from eyvan.capacity.stage_policy import CellStagePolicy
from eyvan.records import (
    MAX_CELL_STAGE_ASSET_LENGTH,
    CellRecord,
    CellStageRecord,
    CellStageState,
    CellState,
    LeaseRecord,
    WorkloadRecord,
    utcnow,
)
from eyvan.server.controller._support import logger
from eyvan.server.service.stage import (
    MAX_STAGE_ACTIONS_PER_PASS,
    CellStageError,
    CellStageEvictionPlan,
    CellStageExecutor,
    CellStageInconclusiveError,
    CellStageMeasurement,
    CellStageRefusedError,
    CellStageRequest,
    CellStageStaleError,
    StageOutcome,
    declared_assets_on_cell,
    plan_eviction,
    stage_headroom_bytes,
)
from eyvan.worker.agent.client import (
    CellAgentClient,
    CellAgentStagingUnsupportedError,
    CellAgentStaleMutationError,
    CellAgentUnavailableError,
)
from eyvan.worker.agent.models import (
    CellStageAction,
    CellStageActionRequest,
    CellStageActionResult,
    CellStageVerdict,
    MutationModel,
)

#: Cell states a stage may be acted on in. A cell that is not ready has no disk
#: worth talking to, and one being released is about to lose the bytes anyway.
_STAGEABLE_CELL_STATES = frozenset({CellState.READY})


class CellStageUnsupportedError(CellStageError):
    """There is no executor for this cell, so no action was attempted.

    The fourth answer, and the one the protocol did not have. The other three
    are all statements about an *action* -- it will never work, it did not
    finish, it was fenced out -- and every one of them writes a row. This says
    there was nothing to attempt, which is a fact about the deployment: an agent
    that predates managed staging, a backend that does not own the workload's
    disk, or a cell with no agent endpoint at all.

    It must not be spelled as any of the others. A refusal would block the asset
    permanently on evidence about the *control plane*, and an inconclusive
    result would charge the cell's budget for bytes nobody attempted to fetch --
    both of which outlive the rolling upgrade that caused them.
    """


def _measurement(result: CellStageActionResult) -> CellStageMeasurement:
    """Project one agent answer onto the executor contract, or raise its verdict."""

    if result.verdict is CellStageVerdict.REFUSED:
        raise CellStageRefusedError(result.detail or f"the agent refused to stage {result.asset!r}")
    if result.verdict is CellStageVerdict.INCONCLUSIVE:
        raise CellStageInconclusiveError(result.detail or f"staging {result.asset!r} ended without an answer")
    return CellStageMeasurement(size_bytes=result.size_bytes, digest=result.digest, detail=result.detail)


class CellAgentStageExecutor:
    """Reach one fleet's cell agents to stage, measure and evict named assets.

    The implementation of :class:`~eyvan.server.service.stage.CellStageExecutor`
    that ships, and the reason the protocol was left injectable: staging is a
    property of the asset store, so this is *an* executor rather than *the* one.
    A deployment with a different store binds its own and this class is never
    constructed.

    Endpoints come from a snapshot the caller already read, never from a fresh
    query. Every method here runs on a worker thread outside every transaction,
    where an async repository read is not available and would be a database call
    held open across a network fetch if it were. The snapshot is one leader pass
    old at most, and a cell that moved within it is caught by the generation
    check on the way out rather than by a stale endpoint on the way in.
    """

    def __init__(
        self,
        cells: tp.Mapping[str, CellRecord],
        agent: tp.Callable[[CellRecord], CellAgentClient],
    ) -> None:
        self._cells = dict(cells)
        self._agent = agent

    def _client(self, request: CellStageRequest) -> tuple[CellRecord, CellAgentClient]:
        """Resolve the agent holding one cell, refusing to guess about the rest."""

        cell = self._cells.get(request.cell_id)
        if cell is None or cell.generation != request.cell_generation:
            raise CellStageStaleError(f"cell {request.cell_id!r} is no longer at generation {request.cell_generation}")
        if not cell.endpoint:
            raise CellStageUnsupportedError(f"cell {request.cell_id!r} has no agent endpoint to stage through")
        try:
            return cell, self._agent(cell)
        except (RuntimeError, ValueError) as exc:
            raise CellStageUnsupportedError(f"cell {request.cell_id!r} has no reachable agent: {exc}") from exc

    def _act(self, request: CellStageRequest, action: CellStageAction) -> CellStageActionResult:
        """Issue one fenced staging action and translate how the agent answered."""

        cell, client = self._client(request)
        # The asset is fingerprinted rather than spelled out: an asset name may
        # be 512 characters and an idempotency key may be 256, and truncating a
        # name is how two different assets come to share one key.
        fingerprint = hashlib.sha256(request.asset.encode("utf-8")).hexdigest()[:32]
        wire = CellStageActionRequest(
            mutation=MutationModel(
                idempotency_key=f"stage:{action.value}:{request.cell_generation}:{fingerprint}",
                generation=request.cell_generation,
                fencing_token=request.fencing_token,
            ),
            action=action,
            asset=request.asset,
            max_bytes=request.max_bytes,
        )
        try:
            return client.stage_asset(cell.id, wire)
        except CellAgentStagingUnsupportedError as exc:
            raise CellStageUnsupportedError(str(exc)) from exc
        except CellAgentStaleMutationError as exc:
            raise CellStageStaleError(str(exc)) from exc
        except CellAgentUnavailableError as exc:
            # The agent did not answer, so nothing was learned. Never a refusal:
            # an unreachable agent says nothing about whether the asset could be
            # staged, and blocking on it would outlive the outage.
            raise CellStageInconclusiveError(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if 400 <= status < 500:
                # The agent read the request and rejected it. Retrying the same
                # bytes gets the same answer, which is what blocked means.
                raise CellStageRefusedError(f"the cell agent rejected this staging request: HTTP {status}") from exc
            raise CellStageInconclusiveError(f"the cell agent answered HTTP {status}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise CellStageInconclusiveError(f"the cell agent could not be asked: {exc}") from exc

    def measure(self, request: CellStageRequest) -> CellStageMeasurement:
        """Return what this asset would occupy on this cell, fetching nothing."""

        return _measurement(self._act(request, CellStageAction.MEASURE))

    def stage(self, request: CellStageRequest) -> CellStageMeasurement:
        """Materialize the asset and return the bytes the host proved present."""

        return _measurement(self._act(request, CellStageAction.STAGE))

    def evict(self, request: CellStageRequest) -> None:
        """Remove the asset, returning only once the host says the bytes are gone."""

        _measurement(self._act(request, CellStageAction.EVICT))


class _ControllerStaging:
    """Managed asset staging and eviction for the cells this leader owns.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    def _stage_policy(self) -> CellStagePolicy:
        """Return the staging bounds this deployment declared."""

        return self.settings.cell_stage_policy()

    def _stage_fence(self, cell: CellRecord) -> str:
        """Return the fencing token one cell's stage actions carry.

        Identical in shape to every other cell action's token, so an executor
        that already understands one understands this, and so a stale token is
        recognisable as stale rather than merely unfamiliar.
        """

        return f"cell:{cell.id}:{cell.generation}"

    def _resolved_stage_executor(self, cells: tp.Mapping[str, CellRecord]) -> CellStageExecutor:
        """Return the executor this pass will use, injected or agent-backed.

        An explicitly injected executor always wins: a deployment whose assets
        live somewhere the cell agent cannot reach binds its own, and silently
        replacing it would be worse than ignoring it.

        Otherwise the cell agents are the executor. This is the whole of the
        binding, and it changes no default: everything above it is already
        gated on a stage budget an operator had to set, and a fleet whose agents
        predate the staging wire method degrades to no executor at all rather
        than to a failure.
        """

        if self._stage_executor is not None:
            return self._stage_executor
        return CellAgentStageExecutor(cells, self._agent_factory)

    async def stage_cell_assets_once(self, *, now: datetime | None = None) -> dict[str, int]:
        """Materialize the assets running workloads declared on the cells they hold.

        Desired stages are derived from live leases rather than from a durable
        wish list. A lease is the control plane's own record that this workload
        is on this cell, so the set cannot drift: a workload that finished stops
        asking for its asset without anything having to remember to withdraw it.

        This is deliberately a cache warmed by use rather than by prediction. The
        workload that triggers a stage has already been placed and pays its own
        pull; what the stage buys is the *next* placement, which is exactly the
        thing the old lease-history inference was guessing at and could not
        confirm.
        """

        policy = self._stage_policy()
        outcome = StageOutcome()
        if not policy.enabled:
            return outcome.as_counts()
        moment = now or utcnow()
        async with self.repository.transaction() as transaction:
            known = {cell.id: cell for cell in await transaction.list_cells()}
            leases = await transaction.list_leases()
            workloads = await transaction.list_workloads()
            stages = await transaction.list_cell_stages()
        executor = self._resolved_stage_executor(known)
        stageable = {cell_id: cell for cell_id, cell in known.items() if cell.state in _STAGEABLE_CELL_STATES}
        wanted = _desired_stages(stageable, workloads, leases)
        by_cell: dict[str, list[CellStageRecord]] = {}
        for stage in stages:
            by_cell.setdefault(stage.cell_id, []).append(stage)

        # Every cell that holds rows, not only the ones something wants staged: a
        # cell that was replaced or retired stops appearing in the wish list, and
        # the sweep is the only thing that would ever look at its rows again.
        for cell_id, held in sorted(by_cell.items()):
            await self._discard_superseded_stages(cell_id, known.get(cell_id), held, outcome)
        for cell_id, assets in sorted(wanted.items()):
            cell = stageable[cell_id]
            for asset in sorted(assets):
                if outcome.exhausted():
                    # A bounded bite, and the next cycle takes the next one.
                    # Reported rather than silent: a pass that keeps capping is a
                    # fleet declaring more than staging can keep up with.
                    outcome.capped = 1
                    logger.info("cell staging reached its %s action budget for this pass", MAX_STAGE_ACTIONS_PER_PASS)
                    return outcome.as_counts()
                with trace_span("eyvan.cell_stage", cell=cell_id, asset=asset):
                    await self._stage_one_asset(
                        cell,
                        asset,
                        executor=executor,
                        policy=policy,
                        now=moment,
                        outcome=outcome,
                    )
        return outcome.as_counts()

    async def _discard_superseded_stages(
        self,
        cell_id: str,
        cell: CellRecord | None,
        stages: tp.Sequence[CellStageRecord],
        outcome: StageOutcome,
    ) -> None:
        """Delete rows describing a host that no longer exists.

        Not an eviction: there is nothing to delete on any disk, because the
        machine those rows measured has been replaced or retired. Leaving them
        would charge a new host's budget for an old host's disk, which is how a
        cache ends up refusing to stage anything on a cell that is in fact empty.

        This is also the only garbage collector the ledger has. A row nothing can
        confirm and nothing can delete is bounded by the life of its cell, and
        this is where that bound is applied.
        """

        for stage in stages:
            if cell is not None and stage.cell_generation == cell.generation:
                continue
            async with self.repository.transaction() as transaction:
                current = await transaction.get_cell(cell_id)
                stored = await transaction.get_cell_stage(cell_id, stage.asset)
                if stored is None:
                    continue
                if current is not None and stored.cell_generation >= current.generation:
                    continue
                await transaction.delete_cell_stage(cell_id, stored.asset)
            outcome.superseded += 1
            logger.info(
                "discarded stage %s on cell %s: recorded generation %s, cell is now %s",
                stage.asset,
                cell_id,
                stage.cell_generation,
                "absent" if cell is None else cell.generation,
            )

    async def _stage_one_asset(
        self,
        cell: CellRecord,
        asset: str,
        *,
        executor: CellStageExecutor,
        policy: CellStagePolicy,
        now: datetime,
        outcome: StageOutcome,
    ) -> None:
        """Reserve, fetch, and confirm one asset on one cell.

        Three transactions and two round trips, none of them overlapping. The
        measurement is what makes the budget exact: the size is observed before
        anything is admitted, so a cell never learns it is over budget by going
        over it.
        """

        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if current is None or current.generation != cell.generation or current.state not in _STAGEABLE_CELL_STATES:
                return
            existing = await transaction.get_cell_stage(cell.id, asset)
            if existing is not None and existing.cell_generation == current.generation:
                if existing.state is CellStageState.STAGED:
                    # Touched, not restaged. The eviction clock has to advance
                    # for an asset a job is using, or LRU gives up exactly the
                    # stage that is earning its bytes.
                    await transaction.put_cell_stage(existing.model_copy(update={"last_used_at": now}))
                    return
                if existing.state in {CellStageState.BLOCKED, CellStageState.EVICTING}:
                    return
                if existing.attempts >= policy.max_attempts:
                    # Includes a row left in ``STAGING`` by a leader that died
                    # mid-fetch. Its bytes stay charged because nobody knows what
                    # reached the disk, but it stops being retried -- an action
                    # that has spent its budget must end somewhere.
                    await transaction.put_cell_stage(
                        existing.model_copy(
                            update={
                                "state": CellStageState.BLOCKED,
                                "confirmed_at": None,
                                "detail": f"staging stopped after {existing.attempts} attempts; presence is unknown",
                                "updated_at": now,
                            }
                        )
                    )
                    outcome.blocked += 1
                    return
            live = await transaction.list_cell_stages(cell_id=cell.id)
            headroom = stage_headroom_bytes(live, cell=current, policy=policy, excluding=asset)
        if headroom <= 0:
            outcome.refused_by_budget += 1
            return

        request = CellStageRequest(
            cell_id=cell.id,
            asset=asset,
            cell_generation=cell.generation,
            fencing_token=self._stage_fence(cell),
            max_bytes=headroom,
        )
        outcome.actions += 1
        measured = await self._measure_stage(request, executor=executor, outcome=outcome, now=now)
        if measured is None:
            return
        if measured.size_bytes > headroom:
            # Refused before a byte moves, and recorded as blocked rather than
            # retried: the asset is larger than this cell may hold, and asking
            # again tomorrow gets the same answer.
            outcome.refused_by_budget += 1
            await self._record_stage(
                request,
                state=CellStageState.BLOCKED,
                reserved_bytes=0,
                detail=f"asset needs {measured.size_bytes} bytes; this cell may admit {headroom}",
                now=now,
            )
            return

        reserved = await self._reserve_stage(request, measured=measured, policy=policy, now=now)
        if reserved is None:
            outcome.refused_by_budget += 1
            return
        outcome.actions += 1
        try:
            written = await asyncio.to_thread(executor.stage, request)
        except CellStageUnsupportedError as exc:
            # Reachable only when an agent is upgraded away mid-pass, because
            # the measurement above goes through the same wire method. Recorded
            # nowhere, for the same reason it is recorded nowhere there.
            logger.info("no staging executor for cell %s: %s", cell.id, exc)
            return
        except CellStageStaleError as exc:
            logger.info("staging %s on cell %s was fenced out: %s", asset, cell.id, exc)
            return
        except CellStageRefusedError as exc:
            outcome.blocked += 1
            await self._record_stage(request, state=CellStageState.BLOCKED, reserved_bytes=0, detail=str(exc), now=now)
            return
        except (CellStageInconclusiveError, OSError) as exc:
            outcome.inconclusive += 1
            await self._record_stage(
                request,
                state=CellStageState.UNKNOWN,
                reserved_bytes=measured.size_bytes,
                detail=str(exc),
                now=now,
            )
            return
        await self._confirm_stage(request, written=written, now=now, outcome=outcome)

    async def _measure_stage(
        self,
        request: CellStageRequest,
        *,
        executor: CellStageExecutor,
        outcome: StageOutcome,
        now: datetime,
    ) -> CellStageMeasurement | None:
        """Ask what the asset would cost, outside every transaction."""

        try:
            return await asyncio.to_thread(executor.measure, request)
        except CellStageUnsupportedError as exc:
            # No executor for this cell -- an agent that predates the staging
            # wire method, or a backend that does not own the workload's disk.
            # Nothing was attempted, so nothing is written and nothing counted:
            # the same silence a deployment with no executor configured keeps.
            logger.debug("no staging executor for cell %s: %s", request.cell_id, exc)
            return None
        except CellStageStaleError as exc:
            logger.info("measuring %s on cell %s was fenced out: %s", request.asset, request.cell_id, exc)
            return None
        except CellStageRefusedError as exc:
            outcome.blocked += 1
            await self._record_stage(request, state=CellStageState.BLOCKED, reserved_bytes=0, detail=str(exc), now=now)
            return None
        except (CellStageInconclusiveError, OSError) as exc:
            outcome.inconclusive += 1
            await self._record_stage(
                request,
                state=CellStageState.UNKNOWN,
                reserved_bytes=0,
                detail=str(exc),
                now=now,
                charge_attempt=True,
            )
            return None

    async def _reserve_stage(
        self,
        request: CellStageRequest,
        *,
        measured: CellStageMeasurement,
        policy: CellStagePolicy,
        now: datetime,
    ) -> CellStageRecord | None:
        """Charge the budget before the fetch starts, re-checking fence and headroom.

        The reservation is what makes the budget hold: the bytes are charged
        while they are being fetched, so anything looking at this cell afterwards
        sees capacity that is already committed rather than capacity that merely
        looks free. Headroom is recomputed here against fresh state because the
        measurement round trip happened outside every transaction, and another
        asset may have been admitted during it.

        The attempt is charged here rather than on failure, so a leader that dies
        mid-fetch still spends one: otherwise a crash loop retries an asset
        forever and never reaches the attempt bound at all.
        """

        async with self.repository.transaction() as transaction:
            cell = await transaction.get_cell(request.cell_id)
            if cell is None or cell.generation != request.cell_generation:
                return None
            live = await transaction.list_cell_stages(cell_id=request.cell_id)
            headroom = stage_headroom_bytes(live, cell=cell, policy=policy, excluding=request.asset)
            if measured.size_bytes > headroom:
                return None
            existing = await transaction.get_cell_stage(request.cell_id, request.asset)
            if existing is not None and existing.cell_generation != request.cell_generation:
                existing = None
            if existing is not None and existing.state is CellStageState.EVICTING:
                # A staging response cannot resolve a possibly-running delete.
                return None
            # A new source length is not proof that an earlier partial copy was
            # removed. Keep its charge until the fetch confirms replacement.
            reservation_bytes = max(measured.size_bytes, 0 if existing is None else existing.reserved_bytes)
            if reservation_bytes > headroom:
                return None
            reserved = CellStageRecord(
                cell_id=request.cell_id,
                asset=request.asset,
                cell_generation=request.cell_generation,
                state=CellStageState.STAGING,
                reserved_bytes=reservation_bytes,
                observed_bytes=None if existing is None else existing.observed_bytes,
                digest=measured.digest,
                last_used_at=now,
                attempts=(0 if existing is None else existing.attempts) + 1,
                detail=None,
                created_at=now if existing is None else existing.created_at,
                updated_at=now,
            )
            await transaction.put_cell_stage(reserved)
            return reserved

    async def _confirm_stage(
        self,
        request: CellStageRequest,
        *,
        written: CellStageMeasurement,
        now: datetime,
        outcome: StageOutcome,
    ) -> None:
        """Record the bytes an executor proved present, or refuse to.

        The generation is re-checked here and not merely at reservation: the
        fetch took real time, and a row confirming an asset on a cell that has
        since been replaced is exactly the false hit this whole design exists to
        avoid.
        """

        async with self.repository.transaction() as transaction:
            cell = await transaction.get_cell(request.cell_id)
            if cell is None or cell.generation != request.cell_generation:
                logger.info(
                    "not recording stage %s: cell %s moved to generation %s",
                    request.asset,
                    request.cell_id,
                    None if cell is None else cell.generation,
                )
                return
            existing = await transaction.get_cell_stage(request.cell_id, request.asset)
            if (
                existing is None
                or existing.cell_generation != request.cell_generation
                or existing.state is CellStageState.EVICTING
            ):
                return
            await transaction.put_cell_stage(
                existing.model_copy(
                    update={
                        "state": CellStageState.STAGED,
                        "reserved_bytes": written.size_bytes,
                        "observed_bytes": written.size_bytes,
                        "digest": written.digest,
                        "confirmed_at": now,
                        "last_used_at": now,
                        "attempts": 0,
                        "detail": written.detail or None,
                        "updated_at": now,
                    }
                )
            )
        outcome.staged += 1

    @staticmethod
    def _owns_eviction(current: CellStageRecord | None, generation: int, operation: str) -> bool:
        """Match the exact durable reservation, not merely a reused asset key."""

        return (
            current is not None
            and current.cell_generation == generation
            and current.state is CellStageState.EVICTING
            and current.detail is not None
            and current.detail.startswith(f"{operation}:")
        )

    async def _restore_stage(self, stored: CellStageRecord, *, now: datetime, operation: str) -> None:
        """Put one row back as it was, after an action that never ran.

        Only for an eviction that found no executor at all. The row was marked
        ``EVICTING`` before the round trip, because a delete that is attempted
        and then loses its answer must stay unproven -- but nothing was
        attempted here, and a row left claiming an eviction is in flight would
        be planned again on every pass against an executor that will never
        answer, which is the unbounded retry this design refuses.
        """

        async with self.repository.transaction() as transaction:
            cell = await transaction.get_cell(stored.cell_id)
            current = await transaction.get_cell_stage(stored.cell_id, stored.asset)
            if cell is None or cell.generation != stored.cell_generation:
                return
            if not self._owns_eviction(current, stored.cell_generation, operation):
                return
            await transaction.put_cell_stage(stored.model_copy(update={"updated_at": now}))

    async def _record_stage(
        self,
        request: CellStageRequest,
        *,
        state: CellStageState,
        reserved_bytes: int,
        detail: str,
        now: datetime,
        charge_attempt: bool = False,
    ) -> None:
        """Write one unproven outcome, promoting to blocked at the attempt budget.

        ``UNKNOWN`` says an action ended without an answer; it is never absence,
        so the bytes stay charged. ``BLOCKED`` says this control plane has
        stopped asking, which is what keeps an impossible fetch from consuming a
        leader phase forever -- the bound the outbox's endless backoff lacks.
        """

        policy = self._stage_policy()
        async with self.repository.transaction() as transaction:
            cell = await transaction.get_cell(request.cell_id)
            if cell is None or cell.generation != request.cell_generation:
                return
            existing = await transaction.get_cell_stage(request.cell_id, request.asset)
            if existing is not None and existing.cell_generation != request.cell_generation:
                existing = None
            if existing is not None and existing.state is CellStageState.EVICTING:
                # A staging response cannot resolve a possibly-running delete.
                return None
            # Failure/refusal reports no absence: even a fetch refusal can leave
            # partial or quarantined bytes. Only confirmed replacement, eviction
            # or a superseded host may release a prior reservation.
            reserved_bytes = max(reserved_bytes, 0 if existing is None else existing.reserved_bytes)
            attempts = (0 if existing is None else existing.attempts) + int(charge_attempt)
            resolved = state
            if state is CellStageState.UNKNOWN and attempts >= policy.max_attempts:
                resolved = CellStageState.BLOCKED
                detail = f"{detail} (staging stopped after {attempts} attempts; presence is unknown)"
            await transaction.put_cell_stage(
                CellStageRecord(
                    cell_id=request.cell_id,
                    asset=request.asset,
                    cell_generation=request.cell_generation,
                    state=resolved,
                    reserved_bytes=reserved_bytes,
                    observed_bytes=None if existing is None else existing.observed_bytes,
                    digest=None if existing is None else existing.digest,
                    last_used_at=now,
                    attempts=attempts,
                    detail=detail[:2048],
                    created_at=now if existing is None else existing.created_at,
                    updated_at=now,
                )
            )

    async def _evict_cell_stages_phase(self) -> tuple[CellStageEvictionPlan, ...]:
        """Run eviction with the deletion switch this deployment configured.

        Dry-run unless an operator turned deletion on, which is the whole point
        of the switch: the phase logs exactly the plan it would execute for as
        long as the operator wants to watch it, and the plan it eventually
        executes is the one they watched.
        """

        return await self.evict_cell_stages_once(dry_run=not self.settings.cell_stage_eviction_enabled)

    async def evict_cell_stages_once(
        self,
        *,
        dry_run: bool = True,
        now: datetime | None = None,
    ) -> tuple[CellStageEvictionPlan, ...]:
        """Return what eviction would do, and do it only when asked to.

        A dry run is the same computation with the deletions withheld, not a
        second code path that could disagree with the one that deletes. It is the
        default because a cache that starts removing bytes the moment it is
        configured gives an operator no way to see what it was about to remove.

        Plans are returned for every cell examined, including those with nothing
        to give up: "this cell is over budget and every byte over it belongs to a
        running workload" is the answer an operator most needs, and it is a plan
        with no victims and a non-zero shortfall.
        """

        policy = self._stage_policy()
        if not policy.enabled:
            return ()
        moment = now or utcnow()
        async with self.repository.transaction() as transaction:
            cells = await transaction.list_cells()
            leases = await transaction.list_leases()
            workloads = await transaction.list_workloads()
            stages = await transaction.list_cell_stages()
        executor = self._resolved_stage_executor({cell.id: cell for cell in cells})
        by_cell: dict[str, list[CellStageRecord]] = {}
        for stage in stages:
            by_cell.setdefault(stage.cell_id, []).append(stage)

        plans: list[CellStageEvictionPlan] = []
        actions = 0
        for cell in sorted(cells, key=lambda record: record.id):
            held = by_cell.get(cell.id)
            if not held:
                continue
            plan = plan_eviction(
                held,
                cell=cell,
                policy=policy,
                protected=declared_assets_on_cell(cell.id, workloads, leases)
                | {stage.asset for stage in held if stage.state is CellStageState.EVICTING},
            )
            plans.append(plan)
            if plan.shortfall_bytes:
                logger.warning(
                    "cell %s is %s bytes over its stage budget and remaining assets are protected, "
                    "blocked or have unresolved evictions; nothing was evicted",
                    cell.id,
                    plan.shortfall_bytes,
                )
            if dry_run or not plan.deletes_anything:
                continue
            if actions >= MAX_STAGE_ACTIONS_PER_PASS:
                # Planning is pure and cheap, so every cell still gets a plan;
                # only the round trips are rationed. An operator reading the
                # result sees the whole picture and a partially executed pass.
                logger.info("cell stage eviction reached its %s action budget", MAX_STAGE_ACTIONS_PER_PASS)
                continue
            with trace_span("eyvan.cell_stage_eviction", cell=cell.id):
                actions += await self._execute_eviction(
                    cell,
                    plan,
                    executor=executor,
                    now=moment,
                    max_actions=MAX_STAGE_ACTIONS_PER_PASS - actions,
                )
        return tuple(plans)

    async def _execute_eviction(
        self,
        cell: CellRecord,
        plan: CellStageEvictionPlan,
        *,
        executor: CellStageExecutor,
        now: datetime,
        max_actions: int = MAX_STAGE_ACTIONS_PER_PASS,
    ) -> int:
        """Delete bounded victims and return the number of attempted executor calls.

        Protection is recomputed inside the transaction that marks each victim,
        not trusted from the plan. Between planning and deleting, a workload can
        be dispatched onto this cell declaring exactly the asset about to be
        removed, and the plan has no way to know.
        """

        actions = 0
        for victim in plan.victims:
            if actions >= max_actions:
                break
            async with self.repository.transaction() as transaction:
                current = await transaction.get_cell(cell.id)
                stored = await transaction.get_cell_stage(cell.id, victim.asset)
                if current is None or current.generation != plan.cell_generation or stored is None:
                    continue
                if stored.cell_generation != plan.cell_generation or stored.state is CellStageState.EVICTING:
                    # A lost response may leave a remote delete running. Neither
                    # another delete nor a fetch may supersede its reservation.
                    continue
                protected = declared_assets_on_cell(
                    cell.id,
                    await transaction.list_workloads(),
                    await transaction.list_leases(),
                )
                if victim.asset in protected:
                    logger.info(
                        "not evicting %s from cell %s: a running workload declared it",
                        victim.asset,
                        cell.id,
                    )
                    continue
                # Persist identity in the existing detail field: no schema or
                # wire change. Every response must still own this reservation.
                operation = f"eviction:{uuid.uuid4().hex}"
                await transaction.put_cell_stage(
                    stored.model_copy(
                        update={
                            "state": CellStageState.EVICTING,
                            "confirmed_at": None,
                            "detail": f"{operation}: in progress",
                            "updated_at": now,
                        }
                    )
                )
            request = CellStageRequest(
                cell_id=cell.id,
                asset=victim.asset,
                cell_generation=plan.cell_generation,
                fencing_token=self._stage_fence(cell),
                max_bytes=victim.reserved_bytes,
            )
            # Fresh protection/generation skips cost no round trip. A failed
            # call still spends one, because it can be just as slow as success.
            actions += 1
            try:
                await asyncio.to_thread(executor.evict, request)
            except CellStageUnsupportedError as exc:
                # Nothing was attempted, so the row must go back exactly as it
                # was. Leaving it in ``EVICTING`` would make the next pass plan
                # the same deletion against the same absent executor forever --
                # an unbounded retry loop begun by a rolling upgrade.
                logger.info("no staging executor for cell %s: %s", cell.id, exc)
                await self._restore_stage(stored, now=now, operation=operation)
                continue
            except CellStageStaleError as exc:
                # Fenced out before anything was deleted, so the row goes back
                # as it was -- which is what :class:`CellStageStaleError` has
                # always promised and what stops the next pass planning the same
                # deletion against the same refusal on every cycle.
                logger.info("evicting %s from cell %s was fenced out: %s", victim.asset, cell.id, exc)
                await self._restore_stage(stored, now=now, operation=operation)
                continue
            except (CellStageInconclusiveError, OSError) as exc:
                # The bytes may or may not be gone, so the row stays charged and
                # stays unproven. Uncharging here would be inventing evidence
                # that a disk is free.
                logger.warning("eviction of %s from cell %s was inconclusive: %s", victim.asset, cell.id, exc)
                # UNKNOWN would permit admission/fetch while a timed-out remote
                # deletion is still running. Keep its exclusion and byte charge
                # until a definitive response or host replacement resolves it.
                async with self.repository.transaction() as transaction:
                    current = await transaction.get_cell(cell.id)
                    pending = await transaction.get_cell_stage(cell.id, victim.asset)
                    if current is None or current.generation != plan.cell_generation:
                        continue
                    if pending is None or not self._owns_eviction(pending, plan.cell_generation, operation):
                        continue
                    await transaction.put_cell_stage(
                        pending.model_copy(
                            update={
                                "detail": f"{operation}: eviction was inconclusive: {exc}"[:2048],
                                "attempts": pending.attempts + 1,
                                "updated_at": now,
                            }
                        )
                    )
                continue
            async with self.repository.transaction() as transaction:
                current = await transaction.get_cell(cell.id)
                stored = await transaction.get_cell_stage(cell.id, victim.asset)
                if current is None or current.generation != plan.cell_generation:
                    continue
                if not self._owns_eviction(stored, plan.cell_generation, operation):
                    continue
                await transaction.delete_cell_stage(cell.id, victim.asset)
        return actions


def _desired_stages(
    cells: tp.Mapping[str, CellRecord],
    workloads: tp.Sequence[WorkloadRecord],
    leases: tp.Sequence[LeaseRecord],
) -> dict[str, set[str]]:
    """Return the assets each stageable cell should be holding right now.

    Read straight off live leases, so the wish list cannot outlive the work that
    justified it: a workload that ended stops asking without anything having to
    remember to withdraw its request.

    An asset longer than the ledger's key is dropped here rather than attempted.
    ``WorkloadDefinition.named_assets`` is unbounded user input while a stage row
    keys on at most :data:`MAX_CELL_STAGE_ASSET_LENGTH` characters, so one
    over-long name would fetch successfully and then fail to persist -- taking
    the whole pass, every cell and every other asset, down with it. Skipped and
    logged: that claims nothing about the asset and costs one cache entry.
    """

    wanted: dict[str, set[str]] = {}
    for cell_id in cells:
        for asset in declared_assets_on_cell(cell_id, workloads, leases):
            if len(asset) > MAX_CELL_STAGE_ASSET_LENGTH:
                logger.warning(
                    "not staging an asset on cell %s: its name is %s characters and a stage row keys on %s",
                    cell_id,
                    len(asset),
                    MAX_CELL_STAGE_ASSET_LENGTH,
                )
                continue
            wanted.setdefault(cell_id, set()).add(asset)
    return wanted


__all__ = ("CellAgentStageExecutor", "CellStageUnsupportedError", "_ControllerStaging")
