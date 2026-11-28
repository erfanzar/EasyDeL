# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Literal

from eyvan.capacity.hardware_catalog import shape_for
from eyvan.domain.tpu_chips import chip_request_error
from eyvan.records import CrossRegionTransferPolicy, ResourceVector, TpuChipRequest, WorkloadSetup
from eyvan.server.service._control_plane.shared import (
    CALLABLE_SIGNING_KEY_ENV,
    DEFAULT_MAX_RETRIES,
    MAX_GCP_TPU_MULTISLICE_WORKERS,
    POOL_ALTERNATE_TEMPLATES_LABEL,
    POOL_ALTERNATES_LABEL,
    REMOTE_CALLABLE_SIGNATURE_ERROR,
    REMOTE_CALLABLE_SIGNING_KEY_ERROR,
    WORKER_COUNT_LABEL,
    CallableExecution,
    CapacityClass,
    CapacityClassUnavailableError,
    CapacityPoolRecord,
    CommandExecution,
    ConflictError,
    InputError,
    ManagedArtifactKind,
    NoMatchingCapacityPoolError,
    Permission,
    PlacementMode,
    Principal,
    PriorityBand,
    RepositorySession,
    ResourceGang,
    WorkerResourceScope,
    WorkloadDefinition,
    WorkloadRecord,
    WorkspaceRef,
    _fingerprint,
    _scope,
    _validate_idempotency_key,
    backend_capabilities,
    managed_result_uri,
    parse_managed_uri,
    parse_tpu_accelerator_type,
    tp,
)


def chip_share_resources(allocatable: ResourceVector, chips: int) -> ResourceVector:
    """Return one chip share's reservation: ``chips`` TPU and that fraction of the cell's CPU and memory.

    Host CPU and memory are split in proportion so neighbours on one host cannot
    starve each other through the scheduler's accounting.
    """

    total = allocatable.tpu
    if total <= 0 or chips > total:
        raise ValueError("a chip share must fit within the cell's TPU chips")
    fraction = chips / total
    return ResourceVector(
        cpu=allocatable.cpu * fraction,
        memory_gib=allocatable.memory_gib * fraction,
        gpu=0,
        tpu=chips,
        custom={name: amount * fraction for name, amount in allocatable.custom.items()},
    )


class CallableSubmissionMixin:
    """Transactional callable submission operations."""

    async def submit_pool_callable(
        self,
        principal: Principal,
        *,
        fleet_name: str,
        org_id: str,
        project_id: str,
        accelerator_type: str,
        pods: int,
        execution: CallableExecution,
        capacity_class: CapacityClass = CapacityClass.SPOT,
        name: str | None = None,
        environment: tp.Mapping[str, str] | None = None,
        secret_mounts: tp.Mapping[str, str] | None = None,
        metadata: tp.Mapping[str, str] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_failure_retries: int = 0,
        max_preemption_retries: int = 10,
        checkpointable: bool = False,
        checkpoint_hook: str | None = None,
        checkpoint_output_uri: str | None = None,
        can_submit_children: bool = False,
        priority_band: PriorityBand = PriorityBand.INTERACTIVE,
        intra_band_priority: int = 0,
        checkpoint_grace_seconds: int = 0,
        reclaim_policy: Literal["requeue", "stop"] = "requeue",
        estimated_duration_seconds: int = 3600,
        named_assets: tuple[str, ...] = (),
        dependencies: tuple[str, ...] = (),
        atomic: bool = False,
        tpu_chips: TpuChipRequest | None = None,
        setup: WorkloadSetup | None = None,
        cross_region_transfer_policy: CrossRegionTransferPolicy | None = None,
        idempotency_key: str,
    ) -> tuple[WorkloadRecord, bool]:
        """Queue one callable on ``pods`` workers, packed or atomic.

        ``atomic`` selects the exact-shape contract: one cell of exactly
        ``pods`` workers, one provider resource, one hardware world. The
        default packs warm cells of the same accelerator type and provisions
        only the deficit, which spans one hardware world per cell instead.
        """

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        self._require_trusted_callable_project(org_id, project_id)
        if execution.signature is not None:
            raise InputError(REMOTE_CALLABLE_SIGNATURE_ERROR)
        if environment is not None and CALLABLE_SIGNING_KEY_ENV in environment:
            raise InputError(REMOTE_CALLABLE_SIGNING_KEY_ERROR)
        if isinstance(pods, bool) or not isinstance(pods, int) or pods < 1:
            raise InputError("pods must be a positive integer")
        if pods > MAX_GCP_TPU_MULTISLICE_WORKERS:
            # Named for what the number is rather than for who set it. It is the
            # widest gang any *known* backend allocates, taken from the
            # capability table -- so calling it GCP's limit stops being true the
            # moment a second backend declares a different one.
            raise InputError(
                f"pods must not exceed {MAX_GCP_TPU_MULTISLICE_WORKERS}, "
                "the widest gang any capacity backend allocates atomically"
            )
        if not accelerator_type or accelerator_type != accelerator_type.strip():
            raise InputError("accelerator_type must be non-empty and trimmed")
        if tpu_chips is not None and (pods != 1 or atomic):
            raise InputError("a TPU chip request runs on one host; it cannot be combined with pods > 1 or atomic")

        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(org_id, project_id)
        operation = "workload.submit_pool_callable"
        fingerprint = _fingerprint(
            {
                "fleet_name": fleet_name,
                "org_id": org_id,
                "project_id": project_id,
                "accelerator_type": accelerator_type,
                "pods": pods,
                "capacity_class": capacity_class.value,
                "execution": execution.model_dump(mode="json"),
                "name": name,
                "environment": dict(environment or {}),
                "secret_mounts": dict(secret_mounts or {}),
                "metadata": dict(metadata or {}),
                "max_retries": max_retries,
                "max_failure_retries": max_failure_retries,
                "max_preemption_retries": max_preemption_retries,
                "checkpointable": checkpointable,
                "checkpoint_hook": checkpoint_hook,
                "checkpoint_output_uri": checkpoint_output_uri,
                "can_submit_children": can_submit_children,
                "priority_band": priority_band.value,
                "intra_band_priority": intra_band_priority,
                "checkpoint_grace_seconds": checkpoint_grace_seconds,
                "estimated_duration_seconds": estimated_duration_seconds,
                "named_assets": list(named_assets),
                "atomic": atomic,
                # Part of what was asked for: the same callable against a
                # different dependency set is a different job, and a replay
                # ignoring them would return one built on other packages.
                "dependencies": list(dependencies),
                # Only when asked for, so every whole-slice fingerprint (and
                # therefore every stored replay) stays what it was.
                **({"tpu_chips": tpu_chips.model_dump(mode="json")} if tpu_chips is not None else {}),
                # Likewise: a different setup step is a different job.
                **({"setup": setup.model_dump(mode="json")} if setup is not None else {}),
                **({"reclaim_policy": reclaim_policy} if reclaim_policy != "requeue" else {}),
                **(
                    {"cross_region_transfer_policy": cross_region_transfer_policy.value}
                    if cross_region_transfer_policy is not None
                    else {}
                ),
            }
        )
        async with self.repository.transaction() as tx:
            replayed = await self._replayed_workload(
                tx,
                scope=scope,
                operation=operation,
                key=key,
                fingerprint=fingerprint,
            )
            if replayed is not None:
                return replayed, True

            fleet_pools = await self._fleet_pools(tx, fleet_name)
            exact_matches = [pool for pool in fleet_pools if pool.enabled and pool.accelerator_type == accelerator_type]
            if not exact_matches:
                # Read the rest of the project only on the failure path. The
                # commonest way to arrive here is naming a zone where a fleet
                # name belongs -- `asia-east1-c` for `research-asia-east1-c` --
                # and the fix is one word the caller cannot guess and the plane
                # already holds. Paying for that listing on every successful
                # submission to produce it would be the wrong trade.
                # Every enabled pool, which is the same set `list_capacity`
                # offers a submitter: a pool belongs to a fleet, not to a
                # project, and the scope is enforced when the workload is
                # placed rather than by filtering the inventory here.
                in_scope = [pool for pool in await tx.list_capacity_pools() if pool.enabled]
                raise NoMatchingCapacityPoolError(
                    fleet=fleet_name,
                    requested_accelerator_type=accelerator_type,
                    available_enabled_accelerator_types=self._callable_accelerator_types(fleet_pools, pods=pods),
                    fleets_providing_requested_type=[
                        pool.fleet_name for pool in in_scope if pool.accelerator_type == accelerator_type
                    ],
                    available_fleets=[pool.fleet_name for pool in in_scope],
                )
            # The requested kind is derived from what the caller asked for, never
            # from what a pool declares itself to be. A pool mislabelled
            # `eyvan_resource_kind: gpu` must still not answer a request for
            # `v5p-16`, and it cannot here, because `v5p-16` is decided to be TPU
            # before any pool is consulted.
            #
            # `cpu` is the sentinel `CpuConfig` sends and nothing else does. TPU
            # is decided by parsing the accelerator type rather than by catalog
            # lookup, so a generation the catalog has not got -- a live case P1
            # treats as inconclusive rather than absent -- still takes the TPU
            # path. GPU is what remains, and it cannot be catalog-driven either:
            # the catalog carries no GPU shapes at all.
            wants_cpu = accelerator_type == "cpu"
            wants_tpu = not wants_cpu and parse_tpu_accelerator_type(accelerator_type) is not None
            wants_gpu = not wants_cpu and not wants_tpu

            def _is_complete(pool: CapacityPoolRecord) -> bool:
                if wants_cpu:
                    return self._is_complete_cpu_pool(pool)
                if wants_gpu:
                    return self._is_complete_gpu_pool(pool)
                return self._is_complete_tpu_pool(pool)

            tpu_matches = [
                pool for pool in exact_matches if _is_complete(pool) and pool.capacity_class is capacity_class
            ]
            if not tpu_matches:
                complete = [pool for pool in exact_matches if _is_complete(pool)]
                available_capacity_classes = sorted({pool.capacity_class.value for pool in complete})
                if available_capacity_classes:
                    # A class counts as existing hardware only when every pool
                    # offering it is: one provisioned pool would start spend.
                    existing_hardware = [
                        klass
                        for klass in available_capacity_classes
                        if all(
                            backend_capabilities(pool.capacity_backend).existing_hardware
                            for pool in complete
                            if pool.capacity_class.value == klass
                        )
                    ]
                    raise CapacityClassUnavailableError(
                        fleet=fleet_name,
                        requested_accelerator_type=accelerator_type,
                        requested_capacity_class=capacity_class.value,
                        available_capacity_classes=available_capacity_classes,
                        existing_hardware_capacity_classes=existing_hardware,
                    )
                kind_label = "CPU" if wants_cpu else "GPU" if wants_gpu else "TPU"
                raise ConflictError(
                    f"enabled capacity pool for {accelerator_type!r} in fleet {fleet_name!r} is not a {kind_label} pool"
                )
            # Several pools of one shape is the shape of cross-zone capacity, not
            # a misconfiguration: a fleet is scoped to a zone, so this is how a
            # workload gets somewhere else to go when one zone keeps reclaiming
            # it. They must be genuinely interchangeable to be alternates --
            # equal admission width and topology -- or the choice would silently
            # change what the workload runs on.
            candidates = sorted(tpu_matches, key=lambda candidate: candidate.id)
            pool = candidates[0]
            interchangeable = [
                candidate
                for candidate in candidates[1:]
                if candidate.allocatable == pool.allocatable and candidate.topology == pool.topology
            ]
            if len(candidates) > 1 and not interchangeable:
                raise ConflictError(
                    f"TPU accelerator type {accelerator_type!r} is ambiguous in fleet {fleet_name!r}: its enabled "
                    "pools differ in admission width or topology, so they are not interchangeable. Configure "
                    "exactly one, or make them match to use them as cross-zone alternates."
                )
            alternate_pool_ids = tuple(candidate.id for candidate in interchangeable)
            alternate_pool_templates = tuple(candidate.template_fingerprint for candidate in interchangeable)
            if pool.max_cells == 0:
                raise ConflictError(
                    f"TPU accelerator type {accelerator_type!r} in fleet {fleet_name!r} has max_workers set to zero"
                )
            if pods > 1 and not pool.hosts_multi_worker_cells:
                # Both halves matter, and only the executor half was ever checked.
                # A pods>1 request against a ``gcp-compute`` TPU pool -- which is
                # what the catalog says a ``v5p-8`` is -- passed this gate, was
                # stored, and then reached ``make_cell`` inside the
                # ``capacity-pools`` leader phase, which carries no
                # failure isolation at the time. The ValueError raised there aborted every
                # later phase of that cycle -- scheduling, settlement, billing --
                # on every tick, so one ordinary submission stopped the whole
                # control plane, and the workload stayed PENDING so nothing ever
                # cleared it.
                raise ConflictError(
                    f"TPU accelerator type {accelerator_type!r} in fleet {fleet_name!r} cannot host a pods={pods} "
                    "callable: a multi-worker request is one Cloud TPU multislice queued resource, which requires a "
                    "gcp-tpu capacity backend and the eyvan-native executor so Eyvan can create and fence it"
                )
            # Deliberately not also gated on the catalog's ``multislice_capable``.
            # That field records the shape's *default* provisioning backend -- a
            # v5p-8 is catalogued ``gcp-compute`` and therefore single-slice by
            # default -- but an operator who declares a ``gcp-tpu`` pool for one
            # has overridden exactly that, and a queued resource is then genuinely
            # creatable. Refusing here would override the operator using a field
            # that answers a different question.
            if pool.max_replica_resources is not None and pool.max_replica_resources != pool.allocatable:
                raise ConflictError(
                    f"TPU accelerator type {accelerator_type!r} in fleet {fleet_name!r} partitions workers with "
                    "max_replica_resources; callable execution requires one complete slice"
                )
            if pods > pool.max_cells:
                raise ConflictError(
                    f"TPU pool {pool.id!r} is capped at {pool.max_cells} workers, but the callable requested pods={pods}"
                )
            declared_python = pool.runtime.python if pool.runtime is not None else None
            if declared_python is not None and execution.python != declared_python:
                # Refused here because the alternative is discovering it on the
                # accelerator. A serialized callable is pinned to the interpreter
                # that pickled it, and a live multislice v5p-16 provisioned two
                # slices, dispatched to four ranks, and had every rank die with
                # "callable requires Python 3.12; current runtime is 3.10" -- a
                # fact this pool already knew, paid for with a full provisioning
                # cycle and the accelerator time that takes.
                raise ConflictError(
                    f"this callable was serialized on Python {execution.python}, and pool {pool.id!r} runs "
                    f"workloads on Python {declared_python}; a pickled callable cannot load on a different "
                    "interpreter. Submit from a matching Python, or give the pool a runtime that provides yours."
                )

            if tpu_chips is not None:
                self._require_chip_sharing_pool(pool, tpu_chips, fleet_name=fleet_name)
                # A chip share pins to this one pool: alternates are other
                # zones' slices, whose chip indexes mean different hardware.
                alternate_pool_ids = ()
                alternate_pool_templates = ()
            spec = self._pinned_pool_workload_spec(
                pool=pool,
                tpu_chips=tpu_chips,
                alternate_pool_ids=alternate_pool_ids,
                alternate_pool_templates=alternate_pool_templates,
                name=name if name is not None else f"{fleet_name}/{accelerator_type}",
                execution=execution,
                environment=dict(environment or {}),
                secret_mounts=dict(secret_mounts or {}),
                metadata=dict(metadata or {}),
                max_retries=max_retries,
                max_failure_retries=max_failure_retries,
                max_preemption_retries=max_preemption_retries,
                cell_group_count=1,
                worker_count=pods,
                # A chip share is one host of one cell by construction: atomic.
                placement=PlacementMode.ATOMIC if atomic or tpu_chips is not None else PlacementMode.ELASTIC,
                checkpointable=checkpointable,
                checkpoint_hook=checkpoint_hook,
                checkpoint_output_uri=checkpoint_output_uri,
                can_submit_children=can_submit_children,
                priority_band=priority_band,
                intra_band_priority=intra_band_priority,
                checkpoint_grace_seconds=checkpoint_grace_seconds,
                reclaim_policy=reclaim_policy,
                estimated_duration_seconds=estimated_duration_seconds,
                named_assets=named_assets,
                dependencies=dependencies,
                setup=setup,
                cross_region_transfer_policy=cross_region_transfer_policy,
            )
            return await self._store_workload(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                spec=spec,
                scope=scope,
                operation=operation,
                key=key,
                fingerprint=fingerprint,
            )

    @staticmethod
    def _require_chip_sharing_pool(pool: CapacityPoolRecord, request: TpuChipRequest, *, fleet_name: str) -> None:
        """Refuse a chip request the pool has not opted into or whose shape cannot hold it."""

        if not pool.chip_sharing:
            raise ConflictError(
                f"pool {pool.id!r} in fleet {fleet_name!r} does not allow chip sharing; set chip_sharing: true on "
                "the pool to run jobs on some of its chips, or drop chips=/chips_indexes= to use the whole slice"
            )
        shape = shape_for(pool.accelerator_type or "")
        chips_per_host = shape.chips_per_host if shape is not None else None
        if chips_per_host is None:
            raise ConflictError(
                f"chip sharing needs the chips per host of {pool.accelerator_type!r}, which the hardware catalog "
                "does not record"
            )
        problem = chip_request_error(
            request.count,
            request.indexes,
            total_chips=pool.allocatable.tpu,
            chips_per_host=chips_per_host,
        )
        if problem is not None:
            raise InputError(f"TPU chip request for pool {pool.id!r}: {problem}")

    def _require_trusted_callable_project(self, org_id: str, project_id: str) -> None:
        """Apply the project rule now, or defer it to the submitting transaction in team mode.

        With team governance the decision depends on the owner's account (tier
        and resource caps), which only a transaction can read. Every path that calls
        this stores the workload through ``_store_workload``, which runs
        ``_require_team_callable`` for the owning identity -- and that re-applies
        this exact project rule to anyone without a team account.
        """

        if getattr(self, "governance_enabled", False):
            return
        self.callable_trust.require(org_id, project_id)

    def _authorize_callable_spec(self, spec: WorkloadDefinition, *, org_id: str, project_id: str) -> None:
        """Apply the callable rules to any spec, however it arrived.

        Trust decides whether a project may have arbitrary pickled code
        deserialized and executed on this fleet, and it was enforced only on the
        direct submission route. Batch, DAG, group, service and in-job child
        submission all accept a ``WorkloadDefinition`` and all carried a
        ``CallableExecution`` straight past it -- so the gate was a front door
        beside five open windows. Reproduced: a principal whose project is
        trusted for nothing is refused by ``submit_workload`` and accepted by
        ``submit_batch`` with the same pickle.

        Written once and called by every route so a sixth cannot quietly skip it.
        """

        if not isinstance(spec.execution, CallableExecution):
            return
        self._require_trusted_callable_project(org_id, project_id)
        if spec.execution.signature is not None:
            raise InputError(REMOTE_CALLABLE_SIGNATURE_ERROR)
        if CALLABLE_SIGNING_KEY_ENV in spec.environment:
            raise InputError(REMOTE_CALLABLE_SIGNING_KEY_ERROR)

    def authorize_callable_artifact(self, principal: Principal, *, org_id: str, project_id: str) -> None:
        """Authorize managed callable storage for one trusted project.

        Synchronous, so in team mode it can only check SUBMIT; uploading bytes
        executes nothing, and every submission that would run them is gated by
        the account rule in its transaction. ``authorize_callable_upload`` is the
        complete check for callers that can await.
        """

        self.authorizer.require(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id)
        self._require_trusted_callable_project(org_id, project_id)

    async def authorize_callable_upload(self, principal: Principal, *, org_id: str, project_id: str) -> None:
        """Authorize managed callable/workspace storage, applying the team-mode account rule."""

        self.authorize_callable_artifact(principal, org_id=org_id, project_id=project_id)
        if not getattr(self, "governance_enabled", False):
            return
        async with self.repository.transaction() as tx:
            await self._require_team_callable(  # type: ignore[attr-defined]
                tx, subject=principal.subject, org_id=org_id, project_id=project_id, spec=None
            )

    @staticmethod
    def result_spec_for_generation(
        spec: WorkloadDefinition,
        *,
        workload_id: str,
        generation: int,
    ) -> WorkloadDefinition:
        """Assign a server-owned destination to one managed callable attempt."""

        execution = spec.execution
        if not isinstance(execution, CallableExecution) or not execution.result_uri:
            return spec
        if not execution.result_uri.startswith("eyvan://"):
            return spec
        try:
            reference = parse_managed_uri(execution.result_uri)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        if reference.kind is not ManagedArtifactKind.RESULTS:
            raise InputError("callable result_uri must reference an eyvan://results object")
        owned_uri = managed_result_uri(f"{workload_id}-g{generation}")
        return spec.model_copy(update={"execution": execution.model_copy(update={"result_uri": owned_uri})})

    @staticmethod
    def _is_complete_tpu_pool(pool: CapacityPoolRecord) -> bool:
        kind = pool.labels.get("eyvan_resource_kind")
        declared_tpu = kind == "tpu" if kind is not None else pool.capacity_backend in {"gcp-tpu", "gcp-attach"}
        return declared_tpu and pool.allocatable.tpu > 0 and pool.allocatable.gpu == 0

    @staticmethod
    def _is_complete_gpu_pool(pool: CapacityPoolRecord) -> bool:
        """Return whether a pool can really run a callable that asked for a GPU.

        The third answer to the question the other two ask. Deliberately not
        derived from the hardware catalog, which carries no GPU shapes at all --
        ``shape_for('nvidia-tesla-p100')`` is ``None`` -- so a catalog-driven
        gate would refuse every GPU type there is.

        A pool carrying a TPU is refused even when it declares itself GPU, for
        the reason the CPU gate refuses an accelerator it did not ask for: the
        callable would hold hardware nothing accounted to it.
        """

        kind = pool.labels.get("eyvan_resource_kind")
        declared_gpu = kind == "gpu" if kind is not None else pool.allocatable.gpu > 0
        return declared_gpu and pool.allocatable.gpu > 0 and pool.allocatable.tpu == 0

    @staticmethod
    def _is_complete_cpu_pool(pool: CapacityPoolRecord) -> bool:
        """Return whether a pool can really run a callable that asked for a CPU.

        The same question :meth:`_is_complete_tpu_pool` asks, for the kind that
        had no answer: `eyvan` ships `CpuConfig`, `@execute` accepts it, and
        every such submission was refused with "is not a TPU pool" -- after the
        callable had been serialized and uploaded.

        A pool carrying an accelerator is refused even when it declares itself
        CPU, because a callable placed there would hold a TPU or a GPU that
        nothing asked for and nothing would account to it.
        """

        kind = pool.labels.get("eyvan_resource_kind")
        declared_cpu = kind == "cpu" if kind is not None else pool.capacity_backend != "gcp-tpu"
        return declared_cpu and pool.allocatable.tpu == 0 and pool.allocatable.gpu == 0

    @classmethod
    def _callable_accelerator_types(
        cls,
        pools: tp.Iterable[CapacityPoolRecord],
        *,
        pods: int,
    ) -> tuple[str, ...]:
        """Return the unambiguous accelerator types that can accept this shape.

        Every kind, not only TPU. This list is what a caller is handed when no
        pool matched, so filtering it to TPU told someone with a GPU or CPU fleet
        that nothing was available while their pool sat there enabled -- an error
        message that describes a different fleet than the one being addressed.

        Completeness is judged per pool against the kind that pool declares,
        which is safe *here* and would not be at the placement gate: this answers
        "what does the fleet offer", where there is no requested type to lose. The
        gate answers "may this pool serve this request", and deriving the kind
        from the pool there is what let a pool named for a TPU while declaring
        something else answer a TPU request.
        """

        complete_pool_counts: dict[str, int] = {}
        compatible: set[str] = set()
        for pool in pools:
            accelerator_type = pool.accelerator_type
            if not pool.enabled or accelerator_type is None or not cls._pool_is_complete_for_its_own_type(pool):
                continue
            complete_pool_counts[accelerator_type] = complete_pool_counts.get(accelerator_type, 0) + 1
            if (
                pool.max_cells < pods
                # Multi-worker capacity is a TPU multislice allocation and has no
                # CPU or GPU analogue, so a wider request narrows the list to TPU.
                or (pods > 1 and (pool.executor != "eyvan-native" or not cls._is_complete_tpu_pool(pool)))
                or (pool.max_replica_resources is not None and pool.max_replica_resources != pool.allocatable)
            ):
                continue
            compatible.add(accelerator_type)
        return tuple(accelerator_type for accelerator_type in compatible if complete_pool_counts[accelerator_type] == 1)

    @classmethod
    def _pool_is_complete_for_its_own_type(cls, pool: CapacityPoolRecord) -> bool:
        """Return whether a pool is coherent for the kind its own type names."""

        accelerator_type = pool.accelerator_type or ""
        if accelerator_type == "cpu":
            return cls._is_complete_cpu_pool(pool)
        if parse_tpu_accelerator_type(accelerator_type) is not None:
            return cls._is_complete_tpu_pool(pool)
        return cls._is_complete_gpu_pool(pool)

    @staticmethod
    async def _fleet_pools(tx: RepositorySession, fleet_name: str) -> list[CapacityPoolRecord]:
        """Load one fleet's pools inside the caller's existing transaction."""

        return await tx.list_capacity_pools(fleet_name=fleet_name)

    @staticmethod
    def _pinned_pool_workload_spec(
        *,
        pool: CapacityPoolRecord,
        name: str,
        execution: CommandExecution | CallableExecution,
        environment: tp.Mapping[str, str],
        secret_mounts: tp.Mapping[str, str],
        metadata: tp.Mapping[str, str],
        max_retries: int,
        max_failure_retries: int,
        cell_group_count: int,
        max_preemption_retries: int = 10,
        worker_count: int = 1,
        workspace: WorkspaceRef | None = None,
        checkpointable: bool = False,
        checkpoint_hook: str | None = None,
        checkpoint_output_uri: str | None = None,
        can_submit_children: bool = False,
        priority_band: PriorityBand = PriorityBand.INTERACTIVE,
        intra_band_priority: int = 0,
        checkpoint_grace_seconds: int = 0,
        reclaim_policy: Literal["requeue", "stop"] = "requeue",
        estimated_duration_seconds: int = 3600,
        named_assets: tuple[str, ...] = (),
        dependencies: tuple[str, ...] = (),
        alternate_pool_ids: tuple[str, ...] = (),
        placement: PlacementMode = PlacementMode.ATOMIC,
        alternate_pool_templates: tuple[str, ...] = (),
        tpu_chips: TpuChipRequest | None = None,
        setup: WorkloadSetup | None = None,
        cross_region_transfer_policy: CrossRegionTransferPolicy | None = None,
    ) -> WorkloadDefinition:
        """Build a workload placed in one fleet pool, or any equivalent of it.

        The checkpoint fields are the job's, not the pool's. Everything else here
        is dictated by the pool a caller named -- provider, region, accelerator,
        gang -- but whether a job can be checkpointed is a property of the job,
        and ``_checkpoint`` in ``scheduling`` grants ``ON_PREEMPTION`` only when
        ``checkpointable`` and ``checkpoint_hook`` are both set. Omitting them
        from this path meant every pool-submitted spot job restarted from zero on
        each reclamation, however many preemption retries it was given.
        """

        pool_labels = {
            "eyvan_pool": pool.id,
            "eyvan_pool_template": pool.template_fingerprint,
        }
        if alternate_pool_ids:
            # Pools of the same shape in other zones. The planner keeps
            # ``eyvan_pool`` as the first choice and falls to these when it is
            # cold, which is the difference between a job that waits on one zone
            # and a job that goes where the capacity is.
            pool_labels[POOL_ALTERNATES_LABEL] = ",".join(alternate_pool_ids)
            if len(alternate_pool_templates) == len(alternate_pool_ids):
                pool_labels[POOL_ALTERNATE_TEMPLATES_LABEL] = ",".join(alternate_pool_templates)
        required_labels = {**pool_labels, WORKER_COUNT_LABEL: str(worker_count)}
        return WorkloadDefinition(
            name=name,
            execution=execution,
            workspace=workspace,
            gang=ResourceGang(
                replicas=1,
                per_replica=(
                    pool.allocatable if tpu_chips is None else chip_share_resources(pool.allocatable, tpu_chips.count)
                ),
                required_topology=pool.topology,
            ),
            tpu_chips=tpu_chips,
            environment=dict(environment),
            secret_mounts=dict(secret_mounts),
            metadata={**dict(metadata), **pool_labels},
            provider=pool.provider,
            region=pool.region,
            accelerator_type=pool.accelerator_type,
            capacity_class=pool.capacity_class,
            required_executor=pool.executor,
            required_labels=required_labels,
            max_retries=max_retries,
            max_failure_retries=max_failure_retries,
            max_preemption_retries=max_preemption_retries,
            cell_group_count=cell_group_count,
            worker_count=worker_count,
            worker_resource_scope=WorkerResourceScope.PER_WORKER,
            placement=placement,
            checkpointable=checkpointable,
            checkpoint_hook=checkpoint_hook,
            checkpoint_output_uri=checkpoint_output_uri,
            can_submit_children=can_submit_children,
            priority_band=priority_band,
            intra_band_priority=intra_band_priority,
            checkpoint_grace_seconds=checkpoint_grace_seconds,
            reclaim_policy=reclaim_policy,
            estimated_duration_seconds=estimated_duration_seconds,
            named_assets=named_assets,
            dependencies=dependencies,
            setup=setup,
            cross_region_transfer_policy=cross_region_transfer_policy,
        )
