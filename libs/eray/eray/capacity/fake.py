# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Portions adapted from the Iris cluster manager in marin-community/marin
# (lib/iris/src/iris/cluster/platforms/gcp/fake.py — Copyright The Marin
# Authors, Apache-2.0).
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

"""In-memory :class:`~eray.capacity.gcp_qr.GcpQrService` fake for tests.

Where the Iris fake models queued resources as instantly-ACTIVE name tuples,
this one keeps a real per-QR state machine (spot QRs spend their life in
``WAITING_FOR_RESOURCES``, which is exactly the behavior the pool's
zone-rotation logic exists for) plus the failure knobs the pool tests need:

- ``stockout_zones`` — creates raise :class:`QuotaExhaustedError` (the
  loud stockout: quota errors, immediate capacity rejection).
- ``silent_stockout_zones`` — creates succeed but never leave
  ``WAITING_FOR_RESOURCES`` (the quiet stockout every spot user knows).
- ``inject_failure`` — one-shot exceptions per operation, optionally after
  the QR was already registered (the half-created-resource case).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .types import (
    QueuedResourceInfo,
    QuotaExhaustedError,
    TpuCreateRequest,
    validate_labels,
    validate_resource_name,
)


class FakeClock:
    """Deterministic, manually-advanced clock for tests.

    Attributes:
        now: Current fake time (Unix epoch seconds).
    """

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        """Initialize the clock.

        Args:
            start: Initial epoch time.
        """
        self.now = start

    def advance(self, seconds: float) -> None:
        """Move the clock forward.

        Args:
            seconds: Seconds to advance by.
        """
        self.now += seconds

    def __call__(self) -> float:
        """Return the current fake time (so the clock is a ``clock()`` callable)."""
        return self.now


@dataclass
class _FakeQr:
    """One fake queued resource."""

    request: TpuCreateRequest
    state: str
    created_at: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class _InjectedFailure:
    """A queued one-shot failure for an operation."""

    exc: Exception
    register_before_failing: bool = False


class FakeQrService:
    """In-memory queued-resource service with failure-injection knobs.

    Attributes:
        creates: Every request passed to ``queued_resource_create`` (in
            order), including ones that then failed.
        deletes: Every ``(name, zone, force)`` delete call, including
            deletes of QRs that did not exist.
        stockout_zones: Zones where creates raise ``QuotaExhaustedError``.
        silent_stockout_zones: Zones where created QRs never progress past
            ``WAITING_FOR_RESOURCES``.
    """

    def __init__(self, project_id: str = "test-project", clock: FakeClock | None = None) -> None:
        """Initialize the fake.

        Args:
            project_id: Reported project id.
            clock: Time source; a fresh :class:`FakeClock` when omitted.
        """
        self._project_id = project_id
        self.clock = clock or FakeClock()
        self._qrs: dict[tuple[str, str], _FakeQr] = {}
        self._injected: dict[str, deque[_InjectedFailure]] = {}
        self.creates: list[TpuCreateRequest] = []
        self.deletes: list[tuple[str, str, bool]] = []
        self.stockout_zones: set[str] = set()
        self.silent_stockout_zones: set[str] = set()

    # ------------------------------------------------------------------
    # Test knobs
    # ------------------------------------------------------------------

    def inject_failure(self, operation: str, exc: Exception, *, register_before_failing: bool = False) -> None:
        """Queue a one-shot failure for the next call of an operation.

        Args:
            operation: ``"queued_resource_create"`` / ``"..._delete"`` /
                ``"..._describe"`` / ``"..._list"``.
            exc: Exception instance to raise.
            register_before_failing: For creates only — register the QR
                before raising, modeling a create RPC that times out
                client-side after the resource was actually made.
        """
        self._injected.setdefault(operation, deque()).append(
            _InjectedFailure(exc, register_before_failing=register_before_failing)
        )

    def set_state(self, name: str, zone: str, state: str) -> None:
        """Force a QR into a state.

        Args:
            name: Queued-resource id.
            zone: GCP zone.
            state: New lifecycle state (e.g. ``"SUSPENDED"``).

        Raises:
            KeyError: If the QR does not exist.
        """
        self._qrs[(name, zone)].state = state

    def activate_pending(self, zone: str | None = None) -> int:
        """Advance pending QRs to ACTIVE (except in silent-stockout zones).

        Args:
            zone: Restrict to one zone; None advances everywhere.

        Returns:
            Number of QRs activated.
        """
        n = 0
        for (_, qr_zone), qr in self._qrs.items():
            if zone is not None and qr_zone != zone:
                continue
            if qr_zone in self.silent_stockout_zones:
                continue
            if qr.state in ("ACCEPTED", "CREATING", "WAITING_FOR_RESOURCES", "PROVISIONING"):
                qr.state = "ACTIVE"
                n += 1
        return n

    def qr_names(self) -> set[str]:
        """Names of every QR currently in the fake cloud."""
        return {name for (name, _zone) in self._qrs}

    def states_by_name(self) -> dict[str, str]:
        """Mapping of QR name → state for assertion convenience."""
        return {name: qr.state for (name, _zone), qr in self._qrs.items()}

    def _check_injected(self, operation: str) -> _InjectedFailure | None:
        """Pop the next queued failure for an operation, if any."""
        queue = self._injected.get(operation)
        if queue:
            return queue.popleft()
        return None

    # ------------------------------------------------------------------
    # GcpQrService protocol
    # ------------------------------------------------------------------

    @property
    def project_id(self) -> str:
        """The fake project id."""
        return self._project_id

    def queued_resource_create(self, request: TpuCreateRequest) -> None:
        """Create a fake QR (initial state ``WAITING_FOR_RESOURCES``).

        Args:
            request: The create request.

        Raises:
            QuotaExhaustedError: When the zone is in ``stockout_zones``.
            ValueError: On invalid names/labels or duplicate ids.
            Exception: Whatever was queued via :meth:`inject_failure`.
        """
        self.creates.append(request)
        validate_resource_name(request.name, "queued resource")
        validate_labels(request.labels)

        injected = self._check_injected("queued_resource_create")
        if injected is not None and not injected.register_before_failing:
            raise injected.exc
        if request.zone in self.stockout_zones:
            raise QuotaExhaustedError(f"no more capacity in zone {request.zone}")

        key = (request.name, request.zone)
        if key in self._qrs:
            raise ValueError(f"queued resource {request.name} already exists in {request.zone}")
        self._qrs[key] = _FakeQr(
            request=request,
            state="WAITING_FOR_RESOURCES",
            created_at=self.clock(),
            labels=dict(request.labels),
        )
        if injected is not None:
            raise injected.exc

    def queued_resource_describe(self, name: str, zone: str) -> QueuedResourceInfo | None:
        """Describe a fake QR, or None when missing."""
        injected = self._check_injected("queued_resource_describe")
        if injected is not None:
            raise injected.exc
        qr = self._qrs.get((name, zone))
        if qr is None:
            return None
        return self._info(name, zone, qr)

    def queued_resource_delete(self, name: str, zone: str, *, force: bool = True) -> None:
        """Delete a fake QR (idempotent, like the real service)."""
        self.deletes.append((name, zone, force))
        injected = self._check_injected("queued_resource_delete")
        if injected is not None:
            raise injected.exc
        self._qrs.pop((name, zone), None)

    def queued_resource_list(
        self, zones: list[str] | None = None, labels: dict[str, str] | None = None
    ) -> list[QueuedResourceInfo]:
        """List fake QRs, optionally restricted by zones and labels."""
        injected = self._check_injected("queued_resource_list")
        if injected is not None:
            raise injected.exc
        results: list[QueuedResourceInfo] = []
        for (name, zone), qr in sorted(self._qrs.items()):
            if zones and zone not in zones:
                continue
            if labels and not all(qr.labels.get(k) == v for k, v in labels.items()):
                continue
            results.append(self._info(name, zone, qr))
        return results

    def _info(self, name: str, zone: str, qr: _FakeQr) -> QueuedResourceInfo:
        """Build the info view of a fake QR."""
        return QueuedResourceInfo(
            name=name,
            state=qr.state,
            zone=zone,
            labels=dict(qr.labels),
            create_time=qr.created_at,
            accelerator_type=qr.request.accelerator_type,
            node_ids=(qr.request.resolved_node_id(),),
        )
