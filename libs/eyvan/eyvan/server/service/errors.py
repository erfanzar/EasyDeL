# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Service-layer errors mapped to transport-specific responses by adapters."""

from __future__ import annotations

import typing as tp

from eyvan._internal.compat import StrEnum
from eyvan.domain.accelerators import sort_accelerator_types


class ServiceErrorCode(StrEnum):
    """Stable machine-readable codes for structured service failures."""

    NO_MATCHING_CAPACITY_POOL = "no_matching_capacity_pool"
    CAPACITY_CLASS_UNAVAILABLE = "capacity_class_unavailable"


class EyvanServiceError(Exception):
    """Base class for expected control-plane failures."""


class NotFoundError(EyvanServiceError):
    """Requested control-plane entity does not exist or is not visible."""


class ConflictError(EyvanServiceError):
    """Request conflicts with durable state or prior idempotent input."""


class AuthorizationError(EyvanServiceError):
    """Verified principal lacks the required scoped permission."""


class InputError(EyvanServiceError):
    """Request is structurally valid but violates a domain precondition."""


class UnavailableError(EyvanServiceError):
    """A dependency outcome is inconclusive and the request may be retried."""


class LogCursorRejectedError(EyvanServiceError):
    """A presented log cursor cannot address the read it was presented for.

    Raised instead of restarting the read. A decoder that answers "start from the
    beginning" for a truncated, foreign, or superseded cursor is indistinguishable
    from a first read, so the client silently re-reads the whole journal and never
    learns why -- "I could not find out", written down as a fact.

    ``reason`` is the log-cursor codec's own stable code and is meant to be routed
    on; ``detail`` is the one human sentence naming what failed. ``incompatible``
    separates the two remedies: an *invalid* cursor was corrupted in transit and a
    fresh read is the answer, while an *incompatible* one is well-formed and the
    journal moved on beneath it -- a retry, a topology change, or an archive
    handover -- which the caller may want to report differently.

    Attributes:
        reason: Stable machine-readable code from the cursor codec.
        detail: One human sentence naming the mismatch or malformation.
        incompatible: True when the cursor is well formed but addresses another read.
        workload_id: Workload the read belongs to.
        attempt: Attempt being served.
        topology: Digest of the frozen attempt topology, or ``None`` when unproven.
        expected_ranks: Width of the frozen rank space, or ``None`` when unproven.
    """

    def __init__(
        self,
        *,
        reason: str,
        detail: str,
        incompatible: bool,
        workload_id: str,
        attempt: int,
        topology: str | None = None,
        expected_ranks: int | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail
        self.incompatible = incompatible
        self.workload_id = workload_id
        self.attempt = attempt
        self.topology = topology
        self.expected_ranks = expected_ranks
        super().__init__(detail or reason)

    def as_document(self) -> dict[str, object]:
        """Return the refusal as the body an adapter can serialize unchanged."""

        return {
            "detail": self.detail or self.reason,
            "code": self.reason,
            "context": {
                "workload_id": self.workload_id,
                "attempt": self.attempt,
                "topology": self.topology,
                "expected_ranks": self.expected_ranks,
                "incompatible": self.incompatible,
            },
        }


class NoMatchingCapacityPoolError(NotFoundError):
    """No enabled pool in one fleet provides the requested accelerator type."""

    code = ServiceErrorCode.NO_MATCHING_CAPACITY_POOL

    def __init__(
        self,
        *,
        fleet: str,
        requested_accelerator_type: str,
        available_enabled_accelerator_types: tp.Iterable[str],
        fleets_providing_requested_type: tp.Iterable[str] = (),
        available_fleets: tp.Iterable[str] = (),
    ) -> None:
        self.fleet = fleet
        self.requested_accelerator_type = requested_accelerator_type
        self.available_enabled_accelerator_types = sort_accelerator_types(available_enabled_accelerator_types)
        # Which fleets *do* provide the shape, and failing that, which exist at
        # all. The commonest way to reach this error is naming a zone where a
        # fleet name belongs -- `asia-east1-c` instead of
        # `research-asia-east1-c` -- and the answer is one word the caller
        # cannot guess and the plane already knows.
        self.fleets_providing_requested_type = tuple(sorted(set(fleets_providing_requested_type)))
        self.available_fleets = tuple(sorted(set(available_fleets)))
        super().__init__(
            f"no enabled capacity pool in fleet {fleet!r} matches accelerator type {requested_accelerator_type!r}"
        )


class CapacityClassUnavailableError(ConflictError):
    """A fleet has pools for the shape, but none in the requested capacity class.

    ``existing_hardware_capacity_classes`` names the offered classes whose every
    pool is a claim on machines that already run and are billed whatever Eyvan
    does (an attached TPU, operator-owned hosts). A client may move an unnamed
    class onto those without starting spend nobody asked for; the plane states
    it because only the plane can see a pool's backend without fleet-admin
    rights. The message is unchanged from the untyped refusal it replaces.
    """

    code = ServiceErrorCode.CAPACITY_CLASS_UNAVAILABLE

    def __init__(
        self,
        *,
        fleet: str,
        requested_accelerator_type: str,
        requested_capacity_class: str,
        available_capacity_classes: tp.Iterable[str],
        existing_hardware_capacity_classes: tp.Iterable[str] = (),
    ) -> None:
        self.fleet = fleet
        self.requested_accelerator_type = requested_accelerator_type
        self.requested_capacity_class = requested_capacity_class
        self.available_capacity_classes = tuple(sorted(set(available_capacity_classes)))
        self.existing_hardware_capacity_classes = tuple(sorted(set(existing_hardware_capacity_classes)))
        super().__init__(
            f"enabled TPU pool for {requested_accelerator_type!r} in fleet {fleet!r} does not offer "
            f"{requested_capacity_class!r} capacity; available: {', '.join(self.available_capacity_classes)}"
        )


__all__ = (
    "AuthorizationError",
    "CapacityClassUnavailableError",
    "ConflictError",
    "EyvanServiceError",
    "InputError",
    "LogCursorRejectedError",
    "NoMatchingCapacityPoolError",
    "NotFoundError",
    "ServiceErrorCode",
    "UnavailableError",
)
