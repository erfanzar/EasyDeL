# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    _QUEUE_TREND_MAX_EVENTS,
    ACTIVE_LEASE_STATES,
    POOL_ALTERNATE_TEMPLATES_LABEL,
    POOL_ALTERNATES_LABEL,
    POOL_LABEL,
    POOL_TEMPLATE_LABEL,
    QUEUE_TREND_MAX_BUCKETS,
    QUEUE_TREND_MAX_WINDOW_S,
    QUEUE_TREND_MIN_WINDOW_S,
    BudgetStatus,
    CapacityPoolRecord,
    InputError,
    NotFoundError,
    Permission,
    PoolEstimate,
    Principal,
    QueueTrend,
    QueueTrendPoint,
    WorkloadDefinition,
    WorkloadEstimate,
    WorkloadState,
    accelerator_unit_price,
    candidate_pool_templates,
    costing,
    datetime,
    math,
    timedelta,
    utcnow,
)


class EstimatesMixin:
    """Transactional estimates operations."""

    async def estimate_workload(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        spec: WorkloadDefinition,
    ) -> WorkloadEstimate:
        """Estimate what one workload would cost, before any record exists.

        Previously the only way to learn a price was to submit and read the
        resulting lease's ``reserved_cost`` -- which means the answer to "can I
        afford this" required spending some of the budget to find out.

        Deliberately *not* a scheduling dry run. It does not consult cell
        availability, quota, fair share or the placement scorer, because a
        pre-submission estimate that claimed to predict placement would be wrong
        the moment the fleet moved, and the number an operator actually wants is
        the price of the shape, not a promise about a particular cell.
        """

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            pools = await tx.list_capacity_pools()
            budget = await tx.get_budget(org_id, project_id)
        hours = spec.estimated_duration_seconds / 3_600
        candidates: list[PoolEstimate] = []
        unpriced: list[str] = []
        for pool in pools:
            if not pool.enabled or not self._pool_admits_spec(pool, spec):
                continue
            # ``cell_group_count`` cells, each of which ``make_cell`` already builds
            # at the full ``worker_count`` width -- its capacity and therefore its
            # hourly price scale with the worker count, so multiplying by it again
            # here priced a pods=8 job at eight times its real hourly.
            cells = spec.cell_group_count
            try:
                template = pool.make_cell(f"estimate-{pool.id}", worker_count=spec.worker_count)
            except ValueError:
                # Capability declarations need not guarantee that this pool's
                # factory can materialize the requested width. An impossible
                # declared shape is not a candidate, nor a failure of the whole
                # estimate; make_cell performs no provider or live-state I/O.
                continue
            # Not costing.scale: that drops zero entries by design, because it exists
            # for accelerator-hour vectors where a zero reads as "used no GPU". A
            # scalar price of zero is a real answer an unpriced pool must be able to
            # give, so this multiplies through the money primitive directly.
            hourly = costing.cost_for_hours(template.hourly_cost(), cells)
            priced = accelerator_unit_price(pool.unit_prices) is not None
            if not priced:
                unpriced.append(pool.id)
            candidates.append(
                PoolEstimate(
                    pool_id=pool.id,
                    accelerator_type=pool.accelerator_type,
                    capacity_class=str(pool.capacity_class),
                    region=pool.region,
                    zone=pool.zone,
                    cells=cells,
                    hourly_cost=hourly,
                    estimated_cost=costing.cost_for_hours(hourly, hours),
                    price_source=f"pool:{pool.id}" if priced else None,
                )
            )
        candidates.sort(key=lambda estimate: (estimate.estimated_cost, estimate.pool_id))
        blocked: tuple[str, ...] = ()
        if budget is not None and budget.hard_limit:
            remaining = max(0.0, budget.limit - budget.spent)
            blocked = tuple(estimate.pool_id for estimate in candidates if estimate.estimated_cost > remaining)
        return WorkloadEstimate(
            org_id=org_id,
            project_id=project_id,
            estimated_duration_seconds=spec.estimated_duration_seconds,
            candidates=tuple(candidates),
            budget_limit=None if budget is None else budget.limit,
            budget_spent=None if budget is None else budget.spent,
            budget_hard_limit=True if budget is None else budget.hard_limit,
            budget_blocked_pool_ids=blocked,
            unpriced_pool_ids=tuple(sorted(unpriced)),
        )

    @staticmethod
    def _pool_admits_spec(pool: CapacityPoolRecord, spec: WorkloadDefinition) -> bool:
        """Return whether a pool could host this shape, on declared attributes only.

        Attributes the operator wrote, never live state: an estimate that skipped
        a pool because it happened to be full right now would tell an operator
        their workload is unaffordable when it is merely queued.
        """

        if spec.required_executor is not None and pool.executor != spec.required_executor:
            return False
        if spec.worker_count > 1 and not pool.hosts_multi_worker_cells:
            return False
        if spec.accelerator_type and pool.accelerator_type != spec.accelerator_type:
            return False
        if spec.capacity_class is not pool.capacity_class:
            return False
        cross_zone = bool(spec.required_labels.get(POOL_ALTERNATES_LABEL))
        if spec.region and pool.region != spec.region and not cross_zone:
            return False
        if spec.provider and pool.provider != spec.provider:
            return False
        # A cross-zone workload names its alternates and their templates as
        # labels no pool carries; asking a pool to carry them made every
        # cross-zone workload match no pool at all. Its pool pins are "one of".
        choices = {}
        if spec.required_labels.get(POOL_ALTERNATES_LABEL):
            choices = candidate_pool_templates(spec.required_labels)
        if choices:
            if pool.id not in choices:
                return False
            ignored = {POOL_ALTERNATES_LABEL, POOL_ALTERNATE_TEMPLATES_LABEL, POOL_LABEL, POOL_TEMPLATE_LABEL}
            return all(
                pool.labels.get(name) == value for name, value in spec.required_labels.items() if name not in ignored
            )
        return all(pool.labels.get(name) == value for name, value in spec.required_labels.items())

    async def queue_trend(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        window_s: float = 86_400.0,
        buckets: int = 24,
        now: datetime | None = None,
    ) -> QueueTrend:
        """Report submitted-vs-admitted counts over recent time buckets.

        Derived from the audit ledger rather than a new time-series store. The
        ledger already records ``workload.submitted`` and the admission transition
        with timestamps and a monotonic sequence, so the trend is a query over data
        that exists -- and a second store would be one more thing to prune, back up
        and keep consistent with the events it duplicates.

        Bucketed by wall clock rather than by event count, because the question is
        "is the queue growing", and a count-based bucket compresses exactly the
        quiet periods that answer it.
        """

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        if not math.isfinite(window_s) or window_s < QUEUE_TREND_MIN_WINDOW_S or window_s > QUEUE_TREND_MAX_WINDOW_S:
            raise InputError(
                "queue trend window must be finite and between "
                f"{QUEUE_TREND_MIN_WINDOW_S:g} and {QUEUE_TREND_MAX_WINDOW_S:g} seconds"
            )
        if buckets < 1 or buckets > QUEUE_TREND_MAX_BUCKETS:
            raise InputError(f"queue trend requires between 1 and {QUEUE_TREND_MAX_BUCKETS} buckets")
        moment = now or utcnow()
        start = moment - timedelta(seconds=window_s)
        width = window_s / buckets
        submitted = [0] * buckets
        admitted = [0] * buckets
        async with self.repository.transaction() as tx:
            # Filter the finite time range in storage and read newest-first. The
            # old query asked for the first 5,000 sequences, so once a project
            # crossed that lifetime count every current event disappeared from
            # this chart forever. cap+1 makes overload explicit: returning a
            # plausible partial chart would be worse than refusing it.
            events = await tx.list_events(
                org_id=org_id,
                project_id=project_id,
                entity_type="workload",
                occurred_at_or_after=start,
                occurred_before=moment,
                newest_first=True,
                limit=_QUEUE_TREND_MAX_EVENTS + 1,
            )
        if len(events) > _QUEUE_TREND_MAX_EVENTS:
            raise InputError(
                "queue trend is incomplete: the requested window contains more than "
                f"{_QUEUE_TREND_MAX_EVENTS} workload events; request a shorter window"
            )
        for event in events:
            index = min(buckets - 1, int((event.timestamp - start).total_seconds() / width))
            if event.action == "workload.submitted":
                submitted[index] += 1
            elif event.new_state == WorkloadState.ADMITTED.value:
                admitted[index] += 1
        return QueueTrend(
            observed_at=moment,
            bucket_seconds=width,
            points=tuple(
                QueueTrendPoint(
                    started_at=start + timedelta(seconds=width * index),
                    submitted=submitted[index],
                    admitted=admitted[index],
                )
                for index in range(buckets)
            ),
        )

    async def budget_status(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        now: datetime | None = None,
    ) -> BudgetStatus:
        """Report one budget's window and its projected end-of-period spend.

        The forecast is derived here rather than stored, for the reason a stored
        one fails: it changes whenever any lease starts or ends, so a persisted
        copy is wrong between every pair of writes and there is no event that
        would reliably refresh it.

        The projection assumes every currently active lease runs to the end of the
        period. That is deliberately the pessimistic reading -- a forecast that
        assumed leases end early would under-warn, and the whole purpose of this
        number is to warn before a hard limit refuses admission.
        """

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            budget = await tx.get_budget(org_id, project_id)
            if budget is None:
                raise NotFoundError(f"budget for {org_id}/{project_id} was not found")
            budget = self._rolled_budget(budget, moment)
            leases = [
                lease
                for lease in await tx.list_leases(org_id=org_id, project_id=project_id)
                if lease.state in ACTIVE_LEASE_STATES
            ]
        length = self._period_seconds(budget.period)
        started = budget.period_started_at
        ends_at = None if length is None or started is None else started + timedelta(seconds=length)
        forecast = None
        if ends_at is not None:
            hours_left = max(0.0, (ends_at - moment).total_seconds()) / 3_600
            burn = costing.total(*(lease.hourly_cost for lease in leases))
            forecast = costing.total(budget.spent, costing.cost_for_hours(burn, hours_left))
        return BudgetStatus(
            org_id=budget.org_id,
            project_id=budget.project_id,
            limit=budget.limit,
            spent=budget.spent,
            currency=budget.currency,
            period=budget.period,
            period_started_at=started,
            period_ends_at=ends_at,
            hard_limit=budget.hard_limit,
            forecast=forecast,
            alert_thresholds=budget.alert_thresholds,
            alerted_thresholds=budget.alerted_thresholds,
        )
