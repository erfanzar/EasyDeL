# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The closed external-evidence sample P8 needs, and the slot that holds one.

A control plane cannot be the source of its own resource evidence. A process
reporting its own RSS proves nothing about a process that is leaking, and it
cannot see what a provider still bills for without asking the provider. So the
resource, database, spend, inventory and cleanup figures a P8 receipt depends on
are *measured elsewhere* and posted in; this module owns the one shape that
crossing is allowed to take, and the single slot the control plane keeps.

Three properties are deliberate and none of them is convenience:

* **The schema is closed.** Exactly eight keys, four bounded integers, two
  booleans and one opaque proof binding. ``eyvan.sdk.commands.soak``'s
  ``_parse_p8_evidence`` refuses anything else, and this module refuses the same
  set on the way in, so a sample that the control plane stored is a sample the
  evaluator can read.
* **There is no way to say "unknown".** That is not an oversight. A field the
  monitor could not establish has no honest encoding here -- zero is a
  measurement and ``false`` is a claim -- so a monitor that cannot measure one
  field emits *nothing*, and the evaluator reports the ``*_evidence_missing``
  code it already has. See :func:`build_external_evidence_sample`.
* **A sample expires.** A monitor that died an hour ago must read as missing
  rather than as the last thing it happened to see, so the slot forgets a sample
  once it is older than its bound. See :class:`SoakEvidenceStore`.
"""

from __future__ import annotations

import math
import time
import typing as tp
from dataclasses import dataclass

from eyvan.soak_contract import MAX_SAFE_COUNTER, require_opaque_id

#: The one schema identifier a sample may carry. Fixed, and not a version this
#: module negotiates: the consumer compares it literally.
EXTERNAL_EVIDENCE_SCHEMA: tp.Final[str] = "eyvan.soak.external-evidence.v1"

#: The bounded non-negative integers in the closed schema.
EXTERNAL_EVIDENCE_COUNTERS: tp.Final[tuple[str, ...]] = (
    "rss_bytes",
    "fd_count",
    "database_bytes",
    "provider_inventory_count",
)

#: The booleans in the closed schema.
EXTERNAL_EVIDENCE_FLAGS: tp.Final[tuple[str, ...]] = ("spend_reconciled", "cleanup_complete")

#: Every field a monitor has to establish before it may emit anything at all.
EXTERNAL_EVIDENCE_MEASURED_FIELDS: tp.Final[tuple[str, ...]] = tuple(
    sorted((*EXTERNAL_EVIDENCE_COUNTERS, *EXTERNAL_EVIDENCE_FLAGS))
)

#: The complete closed key set, in wire order.
EXTERNAL_EVIDENCE_FIELDS: tp.Final[tuple[str, ...]] = (
    "schema",
    "proof_id",
    "rss_bytes",
    "fd_count",
    "database_bytes",
    "spend_reconciled",
    "provider_inventory_count",
    "cleanup_complete",
)

#: Largest request body the evidence route will read. The closed document is a
#: few hundred bytes; anything larger is not a sample, and the limit is enforced
#: while the stream is consumed rather than after it is buffered.
MAX_EVIDENCE_BODY_BYTES: tp.Final[int] = 4_096

#: How long a stored sample stays evidence when an operator configures nothing.
#: Five minutes is several monitor intervals of slack and far short of the hour
#: that made "the last thing it happened to see" the wrong answer.
DEFAULT_EVIDENCE_MAX_AGE_S: tp.Final[float] = 300.0


def _counter(value: object, *, name: str) -> int:
    """Return one bounded non-negative integer, or raise.

    Args:
        value: The candidate value, straight off the wire.
        name: The field name used in the refusal.

    Returns:
        The validated integer.

    Raises:
        ValueError: If the value is a bool, is not an integer, or is outside
            ``[0, MAX_SAFE_COUNTER]``. Numeric text is refused rather than
            coerced: a monitor emits JSON integers, and accepting ``"12"`` here
            would mean the stored sample and the posted one differ in type.
    """

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SAFE_COUNTER:
        raise ValueError(f"{name} must be a bounded non-negative integer")
    return value


def _flag(value: object, *, name: str) -> bool:
    """Return one strict boolean, or raise.

    Args:
        value: The candidate value, straight off the wire.
        name: The field name used in the refusal.

    Returns:
        The validated boolean.

    Raises:
        ValueError: If the value is not exactly ``True`` or ``False``. ``0``,
            ``"false"`` and ``null`` are all refused: each is a different way of
            not knowing, and none of them is a measurement.
    """

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, order=True, slots=True)
class ExternalEvidenceSample:
    """One complete, proof-bound external measurement of the control plane.

    Every field is present because the schema has no way to omit one. Building
    an instance is therefore already the assertion that all six values were
    measured; :func:`build_external_evidence_sample` is the only place that
    decides whether that assertion can be made.
    """

    proof_id: str
    rss_bytes: int
    fd_count: int
    database_bytes: int
    spend_reconciled: bool
    provider_inventory_count: int
    cleanup_complete: bool

    def __post_init__(self) -> None:
        require_opaque_id("proof_id", self.proof_id)
        for name in EXTERNAL_EVIDENCE_COUNTERS:
            _counter(getattr(self, name), name=name)
        for name in EXTERNAL_EVIDENCE_FLAGS:
            _flag(getattr(self, name), name=name)

    def as_dict(self) -> dict[str, object]:
        """Return the closed eight-key document, exactly as it travels."""

        return {
            "schema": EXTERNAL_EVIDENCE_SCHEMA,
            "proof_id": self.proof_id,
            "rss_bytes": self.rss_bytes,
            "fd_count": self.fd_count,
            "database_bytes": self.database_bytes,
            "spend_reconciled": self.spend_reconciled,
            "provider_inventory_count": self.provider_inventory_count,
            "cleanup_complete": self.cleanup_complete,
        }

    @classmethod
    def from_dict(cls, value: object) -> ExternalEvidenceSample:
        """Parse one closed document without coercing any field.

        Args:
            value: The decoded request body.

        Returns:
            The parsed sample.

        Raises:
            ValueError: If the key set is not exactly the eight closed fields,
                if the schema identifier differs, if the proof binding is not a
                canonical opaque identifier, or if any value has the wrong type
                or falls outside its bound.
        """

        if not isinstance(value, dict) or set(value) != set(EXTERNAL_EVIDENCE_FIELDS):
            raise ValueError("external evidence has an invalid closed schema")
        if value["schema"] != EXTERNAL_EVIDENCE_SCHEMA:
            raise ValueError("external evidence names an unsupported schema")
        return cls(
            proof_id=require_opaque_id("proof_id", value["proof_id"]),
            rss_bytes=_counter(value["rss_bytes"], name="rss_bytes"),
            fd_count=_counter(value["fd_count"], name="fd_count"),
            database_bytes=_counter(value["database_bytes"], name="database_bytes"),
            spend_reconciled=_flag(value["spend_reconciled"], name="spend_reconciled"),
            provider_inventory_count=_counter(
                value["provider_inventory_count"],
                name="provider_inventory_count",
            ),
            cleanup_complete=_flag(value["cleanup_complete"], name="cleanup_complete"),
        )


def build_external_evidence_sample(
    proof_id: str,
    *,
    rss_bytes: int | None,
    fd_count: int | None,
    database_bytes: int | None,
    spend_reconciled: bool | None,
    provider_inventory_count: int | None,
    cleanup_complete: bool | None,
) -> tuple[ExternalEvidenceSample | None, tuple[str, ...]]:
    """Assemble a complete sample, or refuse and name what was unknown.

    ``None`` means the monitor could not establish that field. There is no
    encoding for it in the schema, so a single ``None`` suppresses the whole
    sample: substituting a zero would be a measurement nobody took, and carrying
    the previous value forward would republish an observation as a new one.

    Args:
        proof_id: The run this sample is bound to.
        rss_bytes: Resident set size of the control-plane process, or ``None``.
        fd_count: Open file descriptors of that process, or ``None``.
        database_bytes: Size of the durable state, or ``None``.
        spend_reconciled: Whether the ledger reconciles, or ``None``.
        provider_inventory_count: Live cells the provider reports, or ``None``.
        cleanup_complete: Whether the fleet owns nothing outstanding, or ``None``.

    Returns:
        ``(sample, ())`` when every field was measured, otherwise
        ``(None, unmeasured)`` with the unmeasured field names sorted.

    Raises:
        ValueError: If ``proof_id`` is not a canonical opaque identifier, or a
            supplied value is outside its bound. A measured-but-invalid value is
            a defect in the monitor, not an unknown.
    """

    require_opaque_id("proof_id", proof_id)
    measured: dict[str, object | None] = {
        "rss_bytes": rss_bytes,
        "fd_count": fd_count,
        "database_bytes": database_bytes,
        "spend_reconciled": spend_reconciled,
        "provider_inventory_count": provider_inventory_count,
        "cleanup_complete": cleanup_complete,
    }
    unmeasured = tuple(sorted(name for name, value in measured.items() if value is None))
    if unmeasured:
        return None, unmeasured
    return (
        ExternalEvidenceSample(
            proof_id=proof_id,
            rss_bytes=_counter(rss_bytes, name="rss_bytes"),
            fd_count=_counter(fd_count, name="fd_count"),
            database_bytes=_counter(database_bytes, name="database_bytes"),
            spend_reconciled=_flag(spend_reconciled, name="spend_reconciled"),
            provider_inventory_count=_counter(provider_inventory_count, name="provider_inventory_count"),
            cleanup_complete=_flag(cleanup_complete, name="cleanup_complete"),
        ),
        (),
    )


@dataclass(frozen=True, slots=True)
class _StoredEvidence:
    """One accepted sample with both clock readings taken when it arrived."""

    sample: ExternalEvidenceSample
    monotonic_s: float
    wall_s: float


class SoakEvidenceStore:
    """The single most recent external sample, and when it stops counting.

    Deliberately in-process and not durable. It is derived observation rather
    than fleet state: a replica that restarts has not lost a record, it has
    stopped having a current measurement, and reporting "no evidence" until the
    monitor posts again is the truthful answer. That also means the monitor and
    the evaluator must address the same replica; a sample posted to one process
    is not visible from another.

    Age is the *larger* of the monotonic and wall-clock deltas. Monotonic alone
    is what let a suspended host report a fresh sample across thirty-one
    unobserved hours -- ``CLOCK_MONOTONIC`` does not advance while a machine
    sleeps. Wall alone would be steerable by an NTP correction. Taking the
    larger means a forward wall jump can only make evidence read as *missing*,
    which is inconclusive rather than a false pass.
    """

    def __init__(
        self,
        *,
        max_age_s: float = DEFAULT_EVIDENCE_MAX_AGE_S,
        monotonic: tp.Callable[[], float] = time.monotonic,
        wall: tp.Callable[[], float] = time.time,
    ) -> None:
        """Build one evidence slot.

        Args:
            max_age_s: How old a sample may be and still be evidence.
            monotonic: Suspend-insensitive clock, injected for tests.
            wall: Wall clock, injected for tests.

        Raises:
            ValueError: If ``max_age_s`` is not a finite positive number. A
                non-positive bound would discard every sample the instant it
                arrived, which reads as "the monitor is down" forever.
        """

        if isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float)):
            raise ValueError("the soak evidence freshness bound must be a number")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("the soak evidence freshness bound must be finite and positive")
        self._max_age_s = float(max_age_s)
        self._monotonic = monotonic
        self._wall = wall
        self._stored: _StoredEvidence | None = None

    @property
    def max_age_s(self) -> float:
        """How old a stored sample may be before it reads as absent."""

        return self._max_age_s

    def record(self, sample: ExternalEvidenceSample) -> None:
        """Replace the stored sample with ``sample``.

        Args:
            sample: The accepted measurement.

        Raises:
            TypeError: If ``sample`` is not an :class:`ExternalEvidenceSample`.
        """

        if not isinstance(sample, ExternalEvidenceSample):
            raise TypeError("only a parsed external evidence sample may be stored")
        # One immutable slot, replaced wholesale: a reader either sees the whole
        # previous sample or the whole new one, never a half-updated record.
        self._stored = _StoredEvidence(sample=sample, monotonic_s=self._monotonic(), wall_s=self._wall())

    def age_s(self) -> float | None:
        """Return the stored sample's age in seconds, or ``None`` if empty."""

        stored = self._stored
        if stored is None:
            return None
        return max(
            0.0,
            self._monotonic() - stored.monotonic_s,
            self._wall() - stored.wall_s,
        )

    def latest(self, *, proof_id: str) -> ExternalEvidenceSample | None:
        """Return the current sample for one run, or ``None``.

        Args:
            proof_id: The proof the caller is evaluating.

        Returns:
            The stored sample when it is bound to ``proof_id`` and is within the
            freshness bound. ``None`` otherwise -- including when a sample
            exists but belongs to a different run, because evidence measured
            during someone else's soak is not evidence about this one.
        """

        stored = self._stored
        if stored is None or stored.sample.proof_id != proof_id:
            return None
        age = self.age_s()
        if age is None or age > self._max_age_s:
            return None
        return stored.sample


__all__ = (
    "DEFAULT_EVIDENCE_MAX_AGE_S",
    "EXTERNAL_EVIDENCE_COUNTERS",
    "EXTERNAL_EVIDENCE_FIELDS",
    "EXTERNAL_EVIDENCE_FLAGS",
    "EXTERNAL_EVIDENCE_MEASURED_FIELDS",
    "EXTERNAL_EVIDENCE_SCHEMA",
    "MAX_EVIDENCE_BODY_BYTES",
    "ExternalEvidenceSample",
    "SoakEvidenceStore",
    "build_external_evidence_sample",
)
