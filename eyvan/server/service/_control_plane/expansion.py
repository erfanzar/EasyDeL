# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    ACTIVE_LEASE_STATES,
    DEFAULT_MAX_FAN_OUT,
    MAX_GROUP_MEMBERS,
    AdmissionPlan,
    AuditEvent,
    BatchRequest,
    CellState,
    ConflictError,
    CoordinationKind,
    DagNode,
    ExpansionAdmissionRecord,
    ExpansionAdmissionState,
    ExpansionMode,
    GroupRole,
    InputError,
    LeaseRecord,
    Permission,
    Principal,
    RepositorySession,
    ResourceGang,
    WorkerResourceScope,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    _fingerprint,
    _new_id,
    _reserved_resource_hours,
    _scope,
    _validate_idempotency_key,
    child_idempotency_key,
    dag_validate,
    ensure_workload_transition,
    expand,
    expansion_admission_id,
    group_member_names,
    group_validate,
    plan_expansion,
    quota_headroom_children,
    replace,
    resource_quantities,
    secrets,
    stable_id,
    task_index,
    timedelta,
    tp,
    utcnow,
)


class ExpansionMixin:
    """Transactional expansion operations."""

    async def submit_batch(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        template: WorkloadDefinition,
        count: int,
        idempotency_key: str,
        name_prefix: str = "",
        max_fan_out: int = DEFAULT_MAX_FAN_OUT,
    ) -> tuple[WorkloadRecord, list[WorkloadRecord]]:
        """Create one parent workload plus ``count`` ordinary child workloads.

        Every task is a normal durable workload -- same admission, scheduling,
        leasing, settlement, and audit -- so batch adds a shape, not a second
        execution runtime with its own retry and preemption semantics.

        The whole expansion commits in one transaction: a partially-created
        array would leave orphan tasks holding capacity with no parent to cancel
        them. Child keys are derived from the parent's key, so a retried request
        replays the same array rather than creating a second one.

        The parent is a coordinator, not a task. It is stored with
        ``schedulable=False`` so the scheduler never places it -- otherwise a
        batch of N would occupy N+1 slices and the extra one would run the
        template command with no task index. Its state is derived from its
        children by the controller.
        """

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        self._authorize_callable_spec(template, org_id=org_id, project_id=project_id)
        request = BatchRequest(template=template, count=count, name_prefix=name_prefix)
        tasks = expand(request, max_fan_out=max_fan_out)
        key = _validate_idempotency_key(idempotency_key)
        # Derived child keys must themselves be valid. Validating the longest
        # one up front turns a mid-transaction failure on task 100 into a clear
        # rejection of the request.
        _validate_idempotency_key(child_idempotency_key(key, max(count - 1, 0)))
        scope = _scope(org_id, project_id)
        async with self.repository.transaction() as tx:
            parent, replayed = await self._store_workload(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                spec=template.model_copy(update={"name": name_prefix or template.name}),
                scope=scope,
                operation="workload.batch",
                key=key,
                fingerprint=_fingerprint(
                    {
                        "org_id": org_id,
                        "project_id": project_id,
                        "template": template.model_dump(mode="json"),
                        "count": count,
                        # Both change what the array *is*. Omitting them let a
                        # replay with a different prefix or a lowered fan-out
                        # limit silently return the original array instead of
                        # being rejected as a conflicting reuse of the key.
                        "name_prefix": name_prefix,
                        "max_fan_out": max_fan_out,
                    }
                ),
                schedulable=False,
                coordination_kind=CoordinationKind.BATCH,
            )
            if replayed:
                # Ordered by task index, not by the children's random ids:
                # callers index into this list, so children[3] must be task 3
                # on a replay exactly as it was on the original submission.
                # Filtered to batch tasks: the parent may also have acquired
                # ordinary sub-jobs, which are children but not array members.
                stored = await tx.list_child_workloads(parent.id)
                members = [child for child in stored if task_index(child.spec.environment) >= 0]
                return parent, sorted(members, key=lambda child: task_index(child.spec.environment))
            # Planned inside the submitting transaction and after the replay branch:
            # against the same snapshot the tasks are written into, so the wave a
            # caller is told about is the one their quota actually admits, and an
            # already-admitted array still replays if the ceiling has since moved.
            plan = await self._plan_expansion(
                tx,
                org_id=org_id,
                project_id=project_id,
                spec=template,
                requested=count,
                requested_ceiling=max_fan_out,
                subject="batch",
            )
            children: list[WorkloadRecord] = []
            for index, task in enumerate(tasks):
                child_key = child_idempotency_key(key, index)
                child, _ = await self._store_workload(
                    tx,
                    principal,
                    org_id=org_id,
                    project_id=project_id,
                    spec=task,
                    scope=scope,
                    operation="workload.batch_task",
                    key=child_key,
                    fingerprint=_fingerprint({"parent": parent.id, "index": index}),
                    parent_workload_id=parent.id,
                    # Tasks past the first wave are created, durably owed, and
                    # invisible to the scheduler. Not creating them would be a
                    # silently truncated fan-out; admitting them all at once is
                    # the queue flood the wave exists to bound.
                    schedulable=index < plan.admitted_now,
                )
                children.append(child)
            await self._record_expansion_admission(
                tx,
                parent,
                plan,
                kind=CoordinationKind.BATCH,
                governed_children=len(children),
            )
            return parent, children

    async def submit_dag(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        nodes: tp.Sequence[DagNode],
        idempotency_key: str,
        name_prefix: str = "",
        max_fan_out: int = DEFAULT_MAX_FAN_OUT,
    ) -> tuple[WorkloadRecord, list[WorkloadRecord]]:
        """Create one coordinator plus a node per stage, gated by dependencies.

        Nodes are created in topological order, so a node's predecessors always
        already exist and their durable ids can be resolved without a second pass.

        A node with predecessors is stored ``schedulable=False``. That is the entire
        gate: ``CoreSchedulingService.schedule`` already filters on the flag, so a
        waiting node is invisible to placement without a second admission concept.
        The controller flips it when every predecessor has succeeded.
        """

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        for _node in nodes:
            self._authorize_callable_spec(_node.template, org_id=org_id, project_id=project_id)
        order = dag_validate(nodes, max_fan_out=max_fan_out)
        key = _validate_idempotency_key(idempotency_key)
        _validate_idempotency_key(child_idempotency_key(key, max(len(nodes) - 1, 0)))
        by_name = {node.name: node for node in nodes}
        scope = _scope(org_id, project_id)

        async with self.repository.transaction() as tx:
            parent, replayed = await self._store_workload(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                spec=by_name[order[0]].template.model_copy(update={"name": name_prefix or "dag"}),
                scope=scope,
                operation="workload.dag",
                key=key,
                fingerprint=_fingerprint(
                    {
                        "org_id": org_id,
                        "project_id": project_id,
                        "name_prefix": name_prefix,
                        "max_fan_out": max_fan_out,
                        "nodes": [
                            {
                                "name": node.name,
                                "depends_on": sorted(node.depends_on),
                                "template": node.template.model_dump(mode="json"),
                            }
                            for node in sorted(nodes, key=lambda candidate: candidate.name)
                        ],
                    }
                ),
                schedulable=False,
                coordination_kind=CoordinationKind.DAG,
            )
            if replayed:
                stored = await tx.list_child_workloads(parent.id)
                return parent, sorted(stored, key=lambda child: child.spec.name)

            # After the replay branch, so a submission that was already admitted
            # replays unchanged even if an operator has lowered the ceiling since:
            # refusing a request whose children already exist would tell the caller
            # nothing was created while the pipeline runs.
            self._refuse_oversized_expansion(len(nodes), subject="DAG")
            roots = tuple(name for name in order if not by_name[name].depends_on)
            # The plan governs every node, not just the roots. Bounding the entry
            # fan-out alone left the case that actually floods a queue unbounded:
            # a DAG whose roots fit in one wave was admitted IMMEDIATE, no ledger
            # row was written, and one root fanning out to five hundred dependents
            # released all five hundred the moment it succeeded. A dependent is
            # gated by its predecessor, which says *when* it may run and nothing
            # about *how many* may run at once.
            plan = await self._plan_expansion(
                tx,
                org_id=org_id,
                project_id=project_id,
                spec=by_name[roots[0]].template if roots else by_name[order[0]].template,
                requested=max(len(order), 1),
                requested_ceiling=max_fan_out,
                subject="DAG",
            )
            ids_by_name: dict[str, str] = {}
            children: list[WorkloadRecord] = []
            admitted_roots = 0
            for index, name in enumerate(order):
                node = by_name[name]
                predecessors = tuple(ids_by_name[parent_name] for parent_name in node.depends_on)
                schedulable = not predecessors
                if schedulable:
                    schedulable = admitted_roots < plan.admitted_now
                    admitted_roots += 1
                child, _ = await self._store_workload(
                    tx,
                    principal,
                    org_id=org_id,
                    project_id=project_id,
                    spec=node.template.model_copy(update={"name": f"{name_prefix or 'dag'}-{name}"}),
                    scope=scope,
                    operation="workload.dag_node",
                    key=child_idempotency_key(key, index),
                    fingerprint=_fingerprint({"parent": parent.id, "node": name}),
                    parent_workload_id=parent.id,
                    schedulable=schedulable,
                    depends_on=predecessors,
                )
                ids_by_name[name] = child.id
                children.append(child)
            await self._record_expansion_admission(
                tx,
                parent,
                plan,
                kind=CoordinationKind.DAG,
                governed_children=len(order),
            )
            return parent, children

    async def submit_group(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        roles: tp.Sequence[GroupRole],
        idempotency_key: str,
        name_prefix: str = "",
        max_members: int = MAX_GROUP_MEMBERS,
    ) -> tuple[WorkloadRecord, list[WorkloadRecord]]:
        """Create one coordinator plus a member per role replica, all gated together.

        Every member is stored ``schedulable=False``. That is the whole gate --
        ``CoreSchedulingService.schedule`` already filters on the flag, so a waiting
        member is invisible to placement without inventing a second admission
        concept. The controller releases all of them in one transaction once every
        member is independently placeable in the same evaluation pass.
        """

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        for _role in roles:
            self._authorize_callable_spec(_role.template, org_id=org_id, project_id=project_id)
        members = group_validate(roles, max_members=max_members)
        key = _validate_idempotency_key(idempotency_key)
        _validate_idempotency_key(child_idempotency_key(key, max(members - 1, 0)))
        scope = _scope(org_id, project_id)
        names = group_member_names(roles, prefix=name_prefix)
        templates = [role.template for role in roles for _ in range(role.count)]

        async with self.repository.transaction() as tx:
            parent, replayed = await self._store_workload(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                spec=templates[0].model_copy(update={"name": name_prefix or "group"}),
                scope=scope,
                operation="workload.group",
                key=key,
                fingerprint=_fingerprint(
                    {
                        "org_id": org_id,
                        "project_id": project_id,
                        "name_prefix": name_prefix,
                        "max_members": max_members,
                        "roles": [
                            {
                                "name": role.name,
                                "count": role.count,
                                "template": role.template.model_dump(mode="json"),
                            }
                            for role in sorted(roles, key=lambda candidate: candidate.name)
                        ],
                    }
                ),
                schedulable=False,
                coordination_kind=CoordinationKind.GROUP,
            )
            if replayed:
                stored = await tx.list_child_workloads(parent.id)
                return parent, sorted(stored, key=lambda child: child.spec.name)

            # Ceiling only, never waves: a group's whole contract is that its
            # members admit together, so admitting a prefix of one would be the
            # partial hold the shape exists to prevent. After the replay branch,
            # so an already-admitted group still replays under a lowered ceiling.
            self._refuse_oversized_expansion(members, subject="group")
            children: list[WorkloadRecord] = []
            for index, (name, template) in enumerate(zip(names, templates, strict=True)):
                child, _ = await self._store_workload(
                    tx,
                    principal,
                    org_id=org_id,
                    project_id=project_id,
                    spec=template.model_copy(update={"name": name}),
                    scope=scope,
                    operation="workload.group_member",
                    key=child_idempotency_key(key, index),
                    fingerprint=_fingerprint({"parent": parent.id, "member": name}),
                    parent_workload_id=parent.id,
                    schedulable=False,
                )
                children.append(child)
            return parent, children

    # ----------------------------------------------------------------- fan-out admission

    @staticmethod
    def _expansion_demand(spec: WorkloadDefinition) -> dict[str, float]:
        """Return the whole resource vector one child of this shape would hold."""

        scale = float(spec.gang.replicas)
        if spec.worker_resource_scope is WorkerResourceScope.PER_WORKER:
            scale *= float(spec.worker_count)
        return {
            name: quantity * scale for name, quantity in resource_quantities(spec.gang.per_replica).items() if quantity
        }

    async def _expansion_quota_headroom(
        self,
        tx: RepositorySession,
        *,
        org_id: str,
        project_id: str,
        spec: WorkloadDefinition,
    ) -> int | None:
        """Return how many more children of this shape the project's quota can hold.

        ``None`` when the project has no quota, or when the quota constrains no
        dimension this child consumes. Unknown is reported as unknown: a project
        with no quota is not a project with a quota of zero, and sizing a wave
        from a guess would throttle a tenant nobody limited.

        Usage is counted from active lease gangs. A ``PER_WORKER`` workload's
        multiplier is not reconstructed here -- that would need the workload
        behind every lease -- so the estimate errs toward a *larger* wave. That
        direction is deliberate: this only sizes waves, and the scheduler remains
        the one authority that refuses a placement over quota.
        """

        quota = next(
            (
                record
                for record in await tx.list_quotas(org_id=org_id, project_id=project_id)
                if record.project_id == project_id and record.principal_id is None
            ),
            None,
        )
        if quota is None:
            return None
        constrained = set(quota.constrained_resources)
        limits = {name: quantity for name, quantity in resource_quantities(quota.limits).items() if name in constrained}
        usage: dict[str, float] = {}
        for lease in await tx.list_leases(org_id=org_id, project_id=project_id):
            if lease.state not in ACTIVE_LEASE_STATES:
                continue
            for name, quantity in resource_quantities(lease.resources.per_replica).items():
                usage[name] = usage.get(name, 0.0) + quantity * lease.resources.replicas
        return quota_headroom_children(limits, usage, self._expansion_demand(spec))

    def _refuse_oversized_expansion(self, requested: int, *, subject: str) -> None:
        """Refuse an expansion above the control plane's ceiling, before anything exists.

        Deliberately outside the submitting transaction: an expansion that is
        refused creates nothing, so the submitter is told while still holding the
        request instead of discovering it from a coordinator that will never make
        progress. The per-request limit each shape already carries is left to that
        shape's own validator, so a caller who lowered their own bound still gets
        the error their tooling has always matched on.

        Raises:
            ExpansionRefused: If ``requested`` exceeds the configured ceiling.
        """

        plan_expansion(requested, limits=replace(self.expansion_limits, wave_size=0), subject=subject)

    async def _plan_expansion(
        self,
        tx: RepositorySession,
        *,
        org_id: str,
        project_id: str,
        spec: WorkloadDefinition,
        requested: int,
        requested_ceiling: int | None,
        subject: str,
        waves: bool = True,
    ) -> AdmissionPlan:
        """Decide, before anything is created, how one expansion is admitted.

        Called inside the submitting transaction so the quota headroom that sizes
        a wave is the one this expansion is actually admitted against, and read
        from the same snapshot the children are written into.

        Raises:
            ExpansionRefused: If the expansion exceeds the admission ceiling.
        """

        limits = self.expansion_limits if waves else replace(self.expansion_limits, wave_size=0)
        headroom = (
            await self._expansion_quota_headroom(tx, org_id=org_id, project_id=project_id, spec=spec)
            if waves and limits.wave_size > 0
            else None
        )
        return plan_expansion(
            requested,
            limits=limits,
            requested_ceiling=requested_ceiling,
            quota_headroom=headroom,
            subject=subject,
        )

    async def _record_expansion_admission(
        self,
        tx: RepositorySession,
        coordinator: WorkloadRecord,
        plan: AdmissionPlan,
        *,
        kind: CoordinationKind,
        governed_children: int,
    ) -> None:
        """Persist the plan a bounded expansion was admitted under.

        Only written when the expansion is actually bounded. An immediately
        admitted expansion has no plan to resume and no children to release, so
        writing a row for it would put durable state -- and, on the local
        backend, a newer document version -- in front of every deployment that
        never turned waves on.
        """

        if plan.mode is not ExpansionMode.WAVES or governed_children < 1:
            return
        now = utcnow()
        await tx.put_expansion_admission(
            ExpansionAdmissionRecord(
                id=expansion_admission_id(coordinator.id),
                workload_id=coordinator.id,
                org_id=coordinator.org_id,
                project_id=coordinator.project_id,
                kind=kind,
                total_children=governed_children,
                wave_size=min(plan.wave_size, governed_children),
                released_children=min(plan.admitted_now, governed_children),
                ceiling=plan.ceiling,
                mode=plan.mode,
                state=ExpansionAdmissionState.ADMITTING,
                created_at=now,
                updated_at=now,
            )
        )

    async def get_expansion_admission(
        self,
        principal: Principal,
        workload_id: str,
    ) -> ExpansionAdmissionRecord | None:
        """Return the durable admission plan one expansion was accepted under.

        ``None`` means no plan governs this coordinator: it was admitted
        immediately, or it predates admission ledgers. Never reconstructed from
        current configuration -- that would report a plan the submitter was never
        told, which is exactly the silent re-planning the durable row exists to
        prevent.
        """

        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            self.authorizer.require(principal, Permission.READ, org_id=workload.org_id, project_id=workload.project_id)
            return await tx.get_expansion_admission(workload_id)

    async def place_group_atomically(
        self,
        principal: Principal,
        *,
        coordinator_id: str,
        lease_ttl_s: int,
    ) -> tuple[str, ...]:
        """Lease every member of one heterogeneous group, or none of them.

        The feasibility gate that releases a group answers "could all of these be
        placed", and then the ordinary scheduler places them one at a time. In
        between, another tenant can take a cell -- so the four TPU members lease,
        the GPU member does not, and half a job holds accelerators for hours. That
        is the exact failure a group exists to prevent, and a check that is not a
        commit cannot prevent it.

        This is the commit. It follows the shape gang dispatch already uses for a
        spread-cell pod (:meth:`create_lease_group`): decide and write inside one
        transaction, so the members are leased together or the transaction
        contributes nothing. The decision is the pure scheduler's, run over the
        snapshot this transaction is holding rather than one read earlier, and
        every member's lease carries its own fencing token and the cell generation
        it was decided against.

        Members stay ``schedulable=False`` throughout. They never become
        candidates for the ordinary scheduling pass, so there is no window in
        which one placement path could double-place a member the other already
        leased.

        Args:
            principal: The fleet-managing caller, normally the controller.
            coordinator_id: The group coordinator whose members to place.
            lease_ttl_s: Lease lifetime, owned by the controller's settings for
                the same reason every other lease it writes is.

        Returns:
            The member ids leased, or an empty tuple when the whole group could
            not be placed -- in which case nothing was written and nothing is
            held.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        if self.scheduler is None:
            raise InputError("a scheduling service is not configured")
        async with self.repository.transaction() as tx:
            coordinator = await tx.get_workload(coordinator_id)
            if (
                coordinator is None
                or coordinator.coordination_kind is not CoordinationKind.GROUP
                or coordinator.state is not WorkloadState.PENDING
            ):
                return ()
            members = await tx.list_child_workloads(coordinator.id)
            if not members:
                return ()
            if any(
                member.state is not WorkloadState.PENDING
                or member.schedulable
                or member.preemption_id is not None
                or member.lease_id is not None
                for member in members
            ):
                # Some member is already placed, already competing, or reserved by
                # a preemption plan. Whatever this group is, it is not one this
                # call may commit as a whole.
                return ()
            await self._lock_admission_scope(tx, coordinator.org_id, coordinator.project_id)
            wanted = {member.id for member in members}
            candidates: list[WorkloadRecord] = []
            for workload in await tx.list_workloads():
                if workload.id in wanted:
                    # Presented as schedulable because the scheduler filters on the
                    # very flag the gate holds down; asked as stored, every member
                    # reports unplaceable and the group never opens.
                    candidates.append(workload.model_copy(update={"schedulable": True}))
                elif workload.state is not WorkloadState.PENDING or workload.preemption_id is None:
                    candidates.append(workload)
            cells = {cell.id: cell for cell in await tx.list_cells()}
            leases = await tx.list_leases()
            # Pure CPU over the snapshot this transaction already holds: no
            # provider call, no second round trip, nothing that could block while
            # the admission scope is locked.
            result = self.scheduler.schedule(
                candidates,
                list(cells.values()),
                leases=leases,
                budgets=await tx.list_budgets(),
                quotas=await tx.list_quotas(),
                policies=await tx.list_policies(),
                grants=await tx.list_budget_grants() if self.plane_id else [],
                quota_grants=await tx.list_quota_grants() if self.plane_id else [],
                plane_id=self.plane_id,
            )
            placements = {placement.workload_id: placement for placement in result.placements}
            if not wanted <= set(placements):
                # All or nothing. Returning here writes nothing, so a group that
                # cannot be placed whole holds no cell at all.
                return ()
            # Eviction marking locks the host before checking active holders.
            # Take those same locks in stable order and refuse a changed planning
            # snapshot before inspecting its stage rows or publishing any lease.
            chosen_cells = {candidate.cell_id for member in members for candidate in placements[member.id].all_leases}
            for cell_id in sorted(chosen_cells):
                current = await tx.get_cell(cell_id)
                if current is None or current != cells.get(cell_id):
                    return ()
                cells[cell_id] = current
            now = utcnow()
            # Every member's leases are built before any of them is written. A
            # loop that wrote as it went and then found the next member's cell
            # unusable would *commit* the members it had already leased, which is
            # the partial hold this method exists to make impossible.
            proposals: list[tuple[WorkloadRecord, tuple[LeaseRecord, ...]]] = []
            for member in sorted(members, key=lambda record: record.spec.name):
                proposed = placements[member.id].all_leases
                records: list[LeaseRecord] = []
                group_id = (
                    stable_id("pod_group", member.id, member.attempt, member.generation) if len(proposed) > 1 else None
                )
                for index, candidate in enumerate(proposed):
                    cell = cells.get(candidate.cell_id)
                    if cell is None or cell.state is not CellState.READY:
                        return ()
                    try:
                        await self._require_no_asset_eviction(tx, member, cell)
                    except ConflictError:
                        return ()
                    self._require_autoscale_request_ownership(cell, member.id)
                    records.append(
                        LeaseRecord(
                            id=candidate.lease_id,
                            workload_id=member.id,
                            cell_id=candidate.cell_id,
                            cell_generation=cell.generation,
                            org_id=member.org_id,
                            project_id=member.project_id,
                            resources=ResourceGang(
                                replicas=member.spec.gang.replicas,
                                per_replica=member.spec.gang.per_replica,
                                required_topology=member.spec.gang.required_topology,
                            ),
                            hourly_cost=float(candidate.hourly_cost),
                            reserved_cost=float(candidate.reserved_cost),
                            resource_hours_reserved=_reserved_resource_hours(candidate.resource_hours_reserved),
                            generation=member.generation,
                            fencing_token=secrets.token_urlsafe(32),
                            group_id=group_id,
                            group_index=index if group_id is not None else 0,
                            expires_at=now + timedelta(seconds=lease_ttl_s),
                        )
                    )
                proposals.append((member, tuple(records)))

            placed: list[str] = []
            for member, records in proposals:
                admitted = await self._transition_workload(
                    tx,
                    principal,
                    member,
                    WorkloadState.ADMITTED,
                    "workload.admitted",
                )
                ensure_workload_transition(admitted.state, WorkloadState.LEASED)
                leader = records[0]
                for record in records:
                    await self._clear_autoscale_request_ownership(tx, cells[record.cell_id], member.id, now)
                    await tx.put_lease(record)
                leased = admitted.model_copy(
                    update={
                        "state": WorkloadState.LEASED,
                        "lease_id": leader.id,
                        "lease_ids": tuple(record.id for record in records) if len(records) > 1 else (),
                        "dispatch_reservation": None,
                        "preemption_id": None,
                        "autoscale_block_reason": None,
                        "queue_admission_denial": None,
                        "scheduling_reason": None,
                        "updated_at": now,
                    }
                )
                await tx.put_workload(leased)
                await self._outbox(
                    tx,
                    topic="workload.dispatch",
                    aggregate_type="workload",
                    aggregate_id=leased.id,
                    idempotency_key=f"dispatch:{leader.id}:{leader.generation}",
                    generation=leader.generation,
                    payload={
                        "workload_id": leased.id,
                        "lease_id": leader.id,
                        "cell_id": leader.cell_id,
                        **(
                            {
                                "lease_ids": [record.id for record in records],
                                "cell_ids": [record.cell_id for record in records],
                            }
                            if len(records) > 1
                            else {}
                        ),
                    },
                )
                for record in records:
                    await tx.append_event(
                        AuditEvent(
                            id=_new_id("evt"),
                            actor=principal.subject,
                            org_id=record.org_id,
                            project_id=record.project_id,
                            entity_type="lease",
                            entity_id=record.id,
                            action="lease.created",
                            new_state=record.state.value,
                            payload={"workload_id": record.workload_id, "cell_id": record.cell_id},
                        )
                    )
                await self._audit(
                    tx,
                    principal,
                    leased,
                    "workload.group_placed",
                    previous_state=admitted.state.value,
                    new_state=leased.state.value,
                    payload={"group_id": coordinator.id, "members": len(members), "lease_id": leader.id},
                )
                placed.append(member.id)
            return tuple(placed)

    async def list_child_workloads(self, principal: Principal, workload_id: str) -> list[WorkloadRecord]:
        """Return the direct children of one workload, batch tasks in task order.

        Backends return children in whatever order their index yields, which for
        both the in-memory and SQL stores is effectively random UUID order. That
        is fine for a job tree but wrong for an array, where the public
        ``GET /workloads/{id}/children`` response is indexed by task. Ordering
        here rather than per backend keeps the two from drifting.
        """

        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            self.authorizer.require(principal, Permission.READ, org_id=workload.org_id, project_id=workload.project_id)
            children = await tx.list_child_workloads(workload_id)
        # Batch tasks first in index order; non-array sub-jobs keep a stable
        # id ordering after them (task_index returns -1 for those).
        return sorted(children, key=lambda child: (task_index(child.spec.environment), child.id))
