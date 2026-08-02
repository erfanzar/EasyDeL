# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Planning semantics adapted from the Iris autoscaler in
# marin-community/marin (lib/iris/src/iris/cluster/controller/autoscaler —
# Copyright The Marin Authors, Apache-2.0): desired = demand + buffer,
# quota as a typed error with a categorical per-zone block, labels as the
# source of truth for ownership rediscovery.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The capacity pool: multi-zone acquire + hold + reclaim for TPU spot QRs.

One :class:`PoolSpec` declares *desired* capacity — "``count`` slices of
``v5p-128`` (plus ``buffer`` warm spares), spot, from these zones in this
order" — and :func:`plan_pool` computes the actions that reconcile observed
cloud state toward it:

- **reclaim**: ``SUSPENDED``/``FAILED`` QRs are deleted and (by way of the
  resulting deficit) replaced. A ``SUSPENDING`` QR already counts against
  nothing — its replacement is requested immediately, before GCP finishes
  tearing it down.
- **zone fallthrough, loud**: a create rejected with
  :class:`~eray.capacity.types.QuotaExhaustedError` blocks that zone for
  ``quota_block_s`` and the create retries in the next declared zone.
- **zone fallthrough, quiet**: a spot QR stuck in ``WAITING_FOR_RESOURCES``
  past ``zone_wait_timeout_s`` is deleted and re-requested in the next zone
  — the stockout mode that never raises an error at all.
- **buffer**: ``desired = count + buffer`` holds warm spares across job
  boundaries (Iris ``buffer_slices`` semantics).

Pool membership lives in GCP labels (:data:`CAPACITY_POOL_LABEL`), never in
local state: every QR gets a fresh unique name, so a replacement never waits
on its dead predecessor's name, and a restarted loop rediscovers everything
with one wildcard list.

Following eray's watcher, the planner is a **pure function** (observation →
actions) so the whole truth table is unit-testable with zero I/O;
:class:`CapacityPool` is the effects layer around it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..provision.qr import default_runtime_version
from .gcp_qr import GcpQrService
from .types import (
    CAPACITY_POOL_LABEL,
    CAPACITY_TYPE_LABEL,
    KNOWN_TPU_TYPES,
    MAX_RESOURCE_NAME_LENGTH,
    QR_DEAD_STATES,
    QR_DYING_STATES,
    QR_LIVE_STATES,
    CapacityType,
    InfraError,
    QueuedResourceInfo,
    QuotaExhaustedError,
    ResourceNotFoundError,
    TpuCreateRequest,
    generate_qr_suffix,
    validate_labels,
    validate_resource_name,
)

logger = logging.getLogger("eray.capacity.pool")

#: Suffix budget: "-YYYYMMDD-HHMM-hhhhhh" appended to the pool name.
_SUFFIX_LENGTH = 21

#: Default: how long a QR may sit in WAITING_FOR_RESOURCES before the pool
#: rotates it to the next declared zone (only when another zone is usable).
DEFAULT_ZONE_WAIT_TIMEOUT_S = 3600.0
#: Default: how long a zone stays blocked after a quota/stockout error
#: (Iris uses a categorical 5-minute block).
DEFAULT_QUOTA_BLOCK_S = 300.0


@dataclass(frozen=True)
class PoolSpec:
    """Desired-state specification for one capacity pool.

    Attributes:
        name: Pool name; becomes the :data:`CAPACITY_POOL_LABEL` value and
            the prefix of every QR name. Lowercase alphanumeric/hyphens,
            short enough to leave room for the 21-char unique suffix.
        accelerator_type: e.g. ``"v5p-128"``.
        zones: Zone preference order; creates go to the first usable zone,
            stuck/quota'd requests fall through to the next.
        project: GCP project id.
        count: Number of slices the pool should hold for use.
        buffer: Warm spares on top of ``count`` (held even at ``count=0``).
        capacity: Capacity tier (spot by default).
        runtime_version: TPU VM runtime version; None resolves the
            generation default from :mod:`eray.provision.qr`.
        labels: Extra node labels merged under the pool's ownership labels.
        metadata: TPU VM metadata key/values.
        service_account: Service-account email for the nodes.
        network: VPC network (GCP default when None).
        internal_ips: Allocate internal IPs only.
        valid_until_duration: Optional per-QR auto-expiry, e.g. ``"72h"``.
        zone_wait_timeout_s: Seconds a QR may wait for resources before
            being rotated to the next zone.
        quota_block_s: Seconds a zone is skipped after quota/stockout.
        adopt_prefix: Also treat unlabeled QRs whose name starts with this
            prefix as pool members — the migration path for capacity created
            by hand or by legacy scripts. Adopted QRs are counted, reclaimed,
            and replaced like any other member; their replacements are
            normal labeled pool QRs, so a prefix fleet converges onto labels
            one preemption at a time. Prefixes must not overlap between
            pools.
    """

    name: str
    accelerator_type: str
    zones: tuple[str, ...]
    project: str
    count: int = 1
    buffer: int = 0
    capacity: CapacityType = CapacityType.SPOT
    runtime_version: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    service_account: str | None = None
    network: str | None = None
    internal_ips: bool = False
    valid_until_duration: str | None = None
    zone_wait_timeout_s: float = DEFAULT_ZONE_WAIT_TIMEOUT_S
    quota_block_s: float = DEFAULT_QUOTA_BLOCK_S
    adopt_prefix: str | None = None

    def __post_init__(self) -> None:
        """Validate the spec eagerly (names, labels, zones, counts, type).

        A bad accelerator type or unknown family must fail here — at
        ``provision`` time, before the spec is persisted — not blow up the
        reclaim loop that manages every other healthy pool.
        """
        validate_resource_name(self.name, "capacity pool")
        if len(self.name) > MAX_RESOURCE_NAME_LENGTH - _SUFFIX_LENGTH:
            raise ValueError(
                f"pool name {self.name!r} too long: must leave room for the "
                f"{_SUFFIX_LENGTH}-char QR suffix (max {MAX_RESOURCE_NAME_LENGTH - _SUFFIX_LENGTH})"
            )
        if not self.zones:
            raise ValueError("at least one zone is required")
        if self.count < 0 or self.buffer < 0:
            raise ValueError("count and buffer must be >= 0")
        if self.accelerator_type not in KNOWN_TPU_TYPES:
            raise ValueError(
                f"unknown accelerator type {self.accelerator_type!r} (known: a v4/v5litepod/v5p/v6e size)"
            )
        self.resolved_runtime_version()  # raises ValueError for an unresolvable family
        validate_labels(self.labels)

    @property
    def desired_total(self) -> int:
        """Slices the pool should hold: ``count + buffer``."""
        return self.count + self.buffer

    def resolved_runtime_version(self) -> str:
        """The runtime version to request (explicit or generation default)."""
        return self.runtime_version or default_runtime_version(self.accelerator_type)

    def pool_labels(self) -> dict[str, str]:
        """Ownership labels stamped on every QR the pool creates."""
        return {
            **self.labels,
            CAPACITY_POOL_LABEL: self.name,
            CAPACITY_TYPE_LABEL: str(self.capacity),
        }


@dataclass(frozen=True)
class PoolAction:
    """One planned reconcile action.

    Attributes:
        kind: ``"delete"`` (name+zone+force), ``"create"`` (zone), or
            ``"note"`` (reason only — something worth surfacing, no effect).
        name: Delete target QR id.
        zone: Zone of the delete target / preferred zone for the create.
        force: Whether the delete needs ``--force`` (a node may exist).
        reason: Why the action was planned (``"dead:SUSPENDED"``,
            ``"zone-rotate"``, ``"deficit"``, ``"excess"``, ...).
    """

    kind: str
    name: str = ""
    zone: str = ""
    force: bool = True
    reason: str = ""


def _usable_zones(spec: PoolSpec, zone_blocks: dict[str, float], now: float) -> list[str]:
    """Zones in preference order that are not currently quota-blocked."""
    return [z for z in spec.zones if zone_blocks.get(z, 0.0) <= now]


def _next_zone_after(spec: PoolSpec, zone: str, usable: list[str]) -> str | None:
    """The first usable zone after ``zone`` in declaration order, wrapping.

    Args:
        spec: The pool spec (declares zone order).
        zone: Zone to rotate away from.
        usable: Currently usable zones.

    Returns:
        The rotation target, or None when no *other* usable zone exists.
    """
    candidates = [z for z in usable if z != zone]
    if not candidates:
        return None
    try:
        start = spec.zones.index(zone)
    except ValueError:
        return candidates[0]
    ordered = [spec.zones[(start + i) % len(spec.zones)] for i in range(1, len(spec.zones) + 1)]
    for z in ordered:
        if z in candidates:
            return z
    return candidates[0]


def plan_pool(
    spec: PoolSpec,
    observed: list[QueuedResourceInfo],
    *,
    now: float,
    zone_blocks: dict[str, float] | None = None,
) -> list[PoolAction]:
    """Compute the actions that move observed pool state toward the spec.

    Pure — no I/O, no clock reads — so the reclaim truth table is testable
    the way eray's watcher ``plan()`` is.

    Args:
        spec: Desired pool state.
        observed: This pool's QRs as returned by the service (already
            filtered to the pool label).
        now: Current Unix time (seconds).
        zone_blocks: zone → blocked-until epoch from previous quota errors.

    Returns:
        Actions to apply. Deletes for dead/rotated/excess QRs and creates
        for the deficit; ``note`` actions (unknown states, type mismatches,
        unfillable deficits) are interleaved where they are discovered.
    """
    zone_blocks = zone_blocks or {}
    usable = _usable_zones(spec, zone_blocks, now)
    actions: list[PoolAction] = []

    live: list[QueuedResourceInfo] = []
    for qr in observed:
        if qr.state in QR_DEAD_STATES:
            # A node may still exist under a SUSPENDED QR; force is safe and
            # required often enough that the pool always uses it.
            actions.append(PoolAction(kind="delete", name=qr.name, zone=qr.zone, force=True, reason=f"dead:{qr.state}"))
        elif qr.state in QR_DYING_STATES:
            # Not live (its replacement is requested via the deficit below),
            # not deletable yet — it is reclaimed once it lands SUSPENDED.
            continue
        elif qr.state == "DELETING":
            continue
        elif qr.state in QR_LIVE_STATES:
            live.append(qr)
        else:
            actions.append(PoolAction(kind="note", name=qr.name, zone=qr.zone, reason=f"unknown-state:{qr.state}"))

    # Guard against a spec whose accelerator type changed under live QRs:
    # non-matching pending capacity is rotated out like excess, but ACTIVE
    # capacity of the wrong type is only surfaced, never silently killed.
    mismatched = [qr for qr in live if qr.accelerator_type and qr.accelerator_type != spec.accelerator_type]
    for qr in mismatched:
        live.remove(qr)
        reason = f"type-mismatch:{qr.accelerator_type}"
        if qr.state == "ACTIVE":
            actions.append(PoolAction(kind="note", name=qr.name, zone=qr.zone, reason=reason))
        else:
            actions.append(PoolAction(kind="delete", name=qr.name, zone=qr.zone, force=False, reason=reason))

    # Quiet-stockout rotation: a request that has waited past the timeout in
    # its zone moves to the next usable zone. Only when there is somewhere
    # else to go — a single-zone pool just keeps waiting.
    rotate_targets: list[str] = []
    if len(spec.zones) > 1:
        for qr in list(live):
            if qr.state != "WAITING_FOR_RESOURCES":
                continue
            waited = now - (qr.create_time if qr.create_time is not None else now)
            if waited <= spec.zone_wait_timeout_s:
                continue
            target = _next_zone_after(spec, qr.zone, usable)
            if target is None:
                continue
            live.remove(qr)
            actions.append(PoolAction(kind="delete", name=qr.name, zone=qr.zone, force=False, reason="zone-rotate"))
            rotate_targets.append(target)

    deficit = spec.desired_total - len(live)
    if deficit < 0:
        # Scale down: never kill won capacity while requests are still
        # pending — shed newest non-ACTIVE first, then newest ACTIVE.
        def shed_order(qr: QueuedResourceInfo) -> tuple[int, float]:
            return (1 if qr.state == "ACTIVE" else 0, -(qr.create_time or 0.0))

        for qr in sorted(live, key=shed_order)[: -deficit]:
            # PROVISIONING also needs force: GCP rejects plain deletion once
            # node creation has started.
            force = qr.state in ("ACTIVE", "PROVISIONING")
            actions.append(PoolAction(kind="delete", name=qr.name, zone=qr.zone, force=force, reason="excess"))
    elif deficit > 0:
        create_zones = list(rotate_targets)
        while len(create_zones) < deficit:
            create_zones.append(usable[0] if usable else "")
        for zone in create_zones[:deficit]:
            if zone:
                actions.append(PoolAction(kind="create", zone=zone, reason="deficit"))
            else:
                actions.append(PoolAction(kind="note", reason="deficit-unfillable:all-zones-blocked"))
    elif rotate_targets:
        # Rotations freed capacity that the deficit didn't re-request (e.g.
        # desired shrank in the same pass); nothing further to do.
        pass

    return actions


@dataclass
class PoolReport:
    """Result of one reconcile pass.

    Attributes:
        pool: Pool name.
        observed: All labeled QRs seen this pass.
        actions: Actions the planner produced.
        executed: Actions actually applied (empty in dry-run).
        errors: Failures that left the pool short of its desired state
            (observe outages, unrecoverable creates/deletes).
        warnings: Recoverable events the pool absorbed (e.g. a zone
            stockout that fell through to a zone that worked) — not
            failures, and they must not fail a cron pass.
        zone_blocks: zone → blocked-until epoch after this pass.
    """

    pool: str
    observed: list[QueuedResourceInfo] = field(default_factory=list)
    actions: list[PoolAction] = field(default_factory=list)
    executed: list[PoolAction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    zone_blocks: dict[str, float] = field(default_factory=dict)

    @property
    def active(self) -> int:
        """Observed QRs currently ACTIVE."""
        return sum(1 for qr in self.observed if qr.state == "ACTIVE")

    @property
    def live(self) -> int:
        """Observed QRs holding or acquiring capacity."""
        return sum(1 for qr in self.observed if qr.is_live)


class CapacityPool:
    """Effects layer: observes cloud state, plans, and applies actions.

    Args:
        service: The queued-resource service (real or fake).
        spec: Desired pool state.
        dry_run: Plan and report, but never create or delete.
        clock: Time source (injectable for tests).
    """

    def __init__(
        self,
        service: GcpQrService,
        spec: PoolSpec,
        *,
        dry_run: bool = False,
        clock=time.time,
    ) -> None:
        self.service = service
        self.spec = spec
        self.dry_run = dry_run
        self.clock = clock
        #: zone → blocked-until epoch; populated by quota/stockout errors.
        self.zone_blocks: dict[str, float] = {}

    def observe(self) -> list[QueuedResourceInfo]:
        """List this pool's QRs (project-wide, so departed zones still report).

        Returns:
            QRs labeled with this pool's name, plus — when ``adopt_prefix``
            is set — unlabeled QRs whose name matches the prefix.
        """
        labeled = self.service.queued_resource_list(zones=None, labels={CAPACITY_POOL_LABEL: self.spec.name})
        if not self.spec.adopt_prefix:
            return labeled
        seen = {(qr.name, qr.zone) for qr in labeled}
        adopted = [
            qr
            for qr in self.service.queued_resource_list(zones=None)
            if qr.name.startswith(self.spec.adopt_prefix)
            and (qr.name, qr.zone) not in seen
            and not (qr.labels or {}).get(CAPACITY_POOL_LABEL)
        ]
        return labeled + adopted

    def reconcile(self) -> PoolReport:
        """Run one observe → plan → apply pass.

        Returns:
            The pass report (in dry-run mode, ``executed`` stays empty).
            When observation itself fails, the report carries the error and
            NO actions — a pool that cannot see its own state must not act
            on the emptiness it would otherwise mistake for "own nothing".
        """
        now = self.clock()
        try:
            observed = self.observe()
        except InfraError as exc:
            logger.warning("[%s] observe failed; taking no actions this pass: %s", self.spec.name, exc)
            return PoolReport(pool=self.spec.name, errors=[f"observe failed: {exc}"], zone_blocks=dict(self.zone_blocks))
        actions = plan_pool(self.spec, observed, now=now, zone_blocks=self.zone_blocks)
        report = PoolReport(pool=self.spec.name, observed=observed, actions=actions)
        for action in actions:
            if action.kind == "note":
                logger.info("[%s] note: %s (%s)", self.spec.name, action.reason, action.name or "-")
                continue
            if self.dry_run:
                logger.info(
                    "[%s] dry-run: would %s %s (zone=%s, reason=%s)",
                    self.spec.name,
                    action.kind,
                    action.name or "<new>",
                    action.zone,
                    action.reason,
                )
                continue
            try:
                if action.kind == "delete":
                    self._apply_delete(action)
                elif action.kind == "create":
                    self._apply_create(action, report)
                report.executed.append(action)
            except InfraError as exc:
                msg = f"{action.kind} {action.name or action.zone} failed: {exc}"
                logger.warning("[%s] %s", self.spec.name, msg)
                report.errors.append(msg)
        report.zone_blocks = dict(self.zone_blocks)
        return report

    def run(self, *, interval_s: float = 60.0, once: bool = False) -> PoolReport:
        """Run the reconcile loop.

        Args:
            interval_s: Seconds between passes.
            once: Run a single pass and return.

        Returns:
            The most recent pass report.
        """
        while True:
            report = self.reconcile()
            logger.info(
                "[%s] live=%d active=%d desired=%d actions=%d errors=%d",
                self.spec.name,
                report.live,
                report.active,
                self.spec.desired_total,
                len(report.executed),
                len(report.errors),
            )
            if once:
                return report
            time.sleep(interval_s)

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------

    def _apply_delete(self, action: PoolAction) -> None:
        """Delete a QR, tolerating disappearance.

        Args:
            action: The delete action.
        """
        try:
            self.service.queued_resource_delete(action.name, action.zone, force=action.force)
        except ResourceNotFoundError:
            pass

    def _apply_create(self, action: PoolAction, report: PoolReport) -> None:
        """Create one QR, falling through zones on quota/stockout errors.

        The preferred zone comes from the plan; a loud stockout
        (:class:`QuotaExhaustedError`) blocks that zone for
        ``spec.quota_block_s`` and immediately retries the next usable zone.
        Any other failure triggers a best-effort delete of the possibly
        half-created QR (ported Iris behavior) and surfaces as an error.

        Args:
            action: The create action (carries the preferred zone).
            report: Pass report, for error accumulation on fallthrough.

        Raises:
            QuotaExhaustedError: When every declared zone is blocked or
                stocked out.
            InfraError: On a non-quota create failure.
        """
        now = self.clock()
        tried: list[str] = []
        zone = action.zone
        # An earlier create in this same pass may have just blocked the
        # planned zone; skip straight to the first still-usable one.
        if self.zone_blocks.get(zone, 0.0) > now:
            usable = _usable_zones(self.spec, self.zone_blocks, now)
            if not usable:
                raise QuotaExhaustedError(f"all declared zones blocked (planned zone {zone})")
            zone = usable[0]
        while True:
            name = f"{self.spec.name}-{generate_qr_suffix()}"
            request = TpuCreateRequest(
                name=name,
                zone=zone,
                accelerator_type=self.spec.accelerator_type,
                runtime_version=self.spec.resolved_runtime_version(),
                capacity_type=self.spec.capacity,
                labels=self.spec.pool_labels(),
                metadata=dict(self.spec.metadata),
                service_account=self.spec.service_account,
                network=self.spec.network,
                enable_external_ip=not self.spec.internal_ips,
                valid_until_duration=self.spec.valid_until_duration,
            )
            try:
                self.service.queued_resource_create(request)
                logger.info("[%s] created %s in %s", self.spec.name, name, zone)
                return
            except QuotaExhaustedError as exc:
                tried.append(zone)
                self.zone_blocks[zone] = now + self.spec.quota_block_s
                remaining = [z for z in _usable_zones(self.spec, self.zone_blocks, now) if z not in tried]
                if not remaining:
                    raise QuotaExhaustedError(f"all declared zones exhausted or blocked (tried {tried})") from exc
                # Fallthrough that lands elsewhere is the feature working,
                # not a failure — record it as a warning so cron passes
                # exercising it still exit 0.
                report.warnings.append(f"zone {zone} exhausted; falling through: {exc}")
                zone = remaining[0]
                logger.info("[%s] zone %s exhausted; falling through to %s", self.spec.name, tried[-1], zone)
            except InfraError:
                # The create may have half-registered server-side; never
                # leak it (ported from Iris _best_effort_delete).
                try:
                    self.service.queued_resource_delete(name, zone, force=False)
                except InfraError:
                    logger.warning("[%s] cleanup of %s failed", self.spec.name, name, exc_info=True)
                raise
