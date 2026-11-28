# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded longitudinal observations and fail-closed P8 evaluator receipts."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import typing as tp
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import click

from eyvan._internal.workspace import create_private_directory
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.receipts import write_receipt_atomic
from eyvan.sdk.commands.state import CliSession
from eyvan.soak_contract import (
    AVAILABILITY_BUDGET_CATEGORIES,
    MAX_BUDGET_SCOPES,
    MAX_COVERAGE_ENTRIES,
    MAX_PARTICIPANTS,
    MAX_SAFE_COUNTER,
    MAX_SERVER_VIOLATIONS,
    SAFE_CAPACITY_BACKENDS,
    SAFE_CAPACITY_CLASSES,
    SAFE_COORDINATION_KINDS,
    SAFE_EXECUTION_KINDS,
    SAFE_EXECUTORS,
    SAFE_FAILURE_CATEGORIES,
    SAFE_PROVIDERS,
    SAFE_RESOURCE_CLASSES,
    BackendIdentity,
    CapabilityIdentity,
    ExecutionIdentity,
    ExpectedExecution,
    ParticipantBinding,
    ParticipantExecutionIdentity,
    ProviderIdentity,
    is_opaque_id,
    participant_manifest_sha256,
    safe_period_id,
    safe_subject_id,
)

PERSISTENCE_THRESHOLD = 3
RECEIPT_SCHEMA_ID = "eyvan.soak.receipt"
RECEIPT_SCHEMA_VERSION = 1
REQUIRED_INVARIANTS = tuple(f"I{number}" for number in range(1, 10))
VERDICT_EXIT_CODES = {"pass": 0, "fail": 1, "inconclusive": 2}

MAX_TRANSIENT_DETAILS = 100
MAX_RECEIPT_REASONS = 128
MAX_CONFIG_FILES = 64
MAX_CONFIG_BYTES = 16 * 1024 * 1024
MAX_SOAK_DURATION_S = 31 * 24 * 60 * 60
MAX_SOAK_HOURS = MAX_SOAK_DURATION_S // 3600
MAX_INTERVAL_S = MAX_SOAK_DURATION_S
MAX_OBSERVATION_SLOTS = 10_000_000

# Decimal input is proof material, so its resource bounds cannot depend on the
# process-wide Decimal context. These limits are checked on Decimal.as_tuple()
# before a power of ten or coefficient is constructed.
MAX_DECIMAL_TEXT_LENGTH = 128
MAX_DECIMAL_DIGITS = 128
MAX_DECIMAL_ABS_EXPONENT = 128

# Every persisted reason is one closed code. No exception text, CLI value,
# tenant identifier, object path, provider detail, or arbitrary extension is
# accepted into this vocabulary.
REASON_CODES: tp.Final[frozenset[str]] = frozenset(
    {
        "artifact_collection_missing",
        "artifact_creation_missing",
        "artifact_growth_bound_missing",
        "artifact_net_growth_exceeded",
        "availability_budget_exceeded",
        "candidate_identity_missing",
        "cleanup_evidence_missing",
        "observation_gap",
        "cleanup_incomplete",
        "config_execution_binding_missing",
        "configuration_snapshot_missing",
        "coverage_binding_mismatch",
        "coverage_counter_regression",
        "coverage_duplicate",
        "coverage_missing",
        "coverage_unclassified",
        "database_growth_bound_missing",
        "database_growth_evidence_missing",
        "database_growth_exceeded",
        "duration_evidence_invalid",
        "duration_shortfall",
        "evidence_truncated",
        "expected_execution_missing",
        "expected_execution_not_exercised",
        "fd_growth_bound_missing",
        "fd_growth_evidence_missing",
        "fd_growth_exceeded",
        "internal_preflight_error",
        "invalid_configuration_input",
        "invalid_expected_execution_input",
        "invalid_numeric_input",
        "invalid_output_input_alias",
        "invalid_proof_identity",
        "longitudinal_invariant_violation",
        "no_successful_observation",
        "non_transient_observation_error",
        "observation_slot_shortfall",
        "p8_cpu_mix_missing",
        "p8_evidence_binding_mismatch",
        "p8_evidence_missing",
        "p8_tpu_mix_missing",
        "persistent_invariant_violation",
        "provider_inventory_evidence_missing",
        "provider_inventory_not_empty",
        "reason_collection_truncated",
        "rss_growth_bound_missing",
        "rss_growth_evidence_missing",
        "rss_growth_exceeded",
        "run_interrupted",
        "run_not_started",
        "run_still_running",
        "runner_filesystem_error",
        "runner_internal_error",
        "source_identity_missing",
        "spend_reconciliation_evidence_missing",
        "spend_unreconciled",
    }
)


def _decimal_parts(value: Decimal, *, name: str) -> tuple[int, int]:
    """Return a signed integer coefficient and base-10 exponent.

    ``Decimal`` construction and :meth:`Decimal.as_tuple` are context-free. The
    explicit bounds are checked before accumulating the coefficient or creating
    ``10**exponent``, so hostile ``Decimal`` objects cannot bypass the CLI text
    limit and trigger unbounded integer work.
    """

    if not value.is_finite():
        raise ValueError(f"{name} is outside its finite bound")
    decimal_tuple = value.as_tuple()
    exponent = decimal_tuple.exponent
    digits = decimal_tuple.digits
    if (
        not isinstance(exponent, int)
        or not digits
        or len(digits) > MAX_DECIMAL_DIGITS
        or abs(exponent) > MAX_DECIMAL_ABS_EXPONENT
    ):
        raise ValueError(f"{name} must be a bounded finite decimal")
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    if decimal_tuple.sign:
        coefficient = -coefficient
    return coefficient, exponent


def _decimal_ratio(value: Decimal, *, name: str) -> tuple[int, int]:
    """Return the exact ``numerator, denominator`` using bounded Python ints."""

    coefficient, exponent = _decimal_parts(value, name=name)
    if exponent >= 0:
        return coefficient * 10**exponent, 1
    return coefficient, 10 ** (-exponent)


def _decimal_from_parts(coefficient: int, exponent: int, *, name: str) -> Decimal:
    """Build one exact finite Decimal without consulting the ambient context."""

    sign = int(coefficient < 0)
    coefficient = abs(coefficient)
    if coefficient == 0:
        return Decimal(0)
    while coefficient % 10 == 0:
        coefficient //= 10
        exponent += 1
    digits_text = str(coefficient)
    if len(digits_text) > MAX_DECIMAL_DIGITS or abs(exponent) > MAX_DECIMAL_ABS_EXPONENT:
        raise ValueError(f"{name} must be a bounded finite decimal")
    return Decimal((sign, tuple(int(digit) for digit in digits_text), exponent))


def _scale_decimal_exact(value: Decimal, factor: int, *, name: str) -> Decimal:
    """Multiply a finite decimal by a positive integer using Python integers."""

    if isinstance(factor, bool) or not isinstance(factor, int) or factor <= 0:
        raise ValueError("decimal scale factor must be a positive integer")
    coefficient, exponent = _decimal_parts(value, name=name)
    return _decimal_from_parts(coefficient * factor, exponent, name=name)


def _decimal_quotient_parts(dividend: Decimal, divisor: Decimal) -> tuple[int, int]:
    """Return the positive exact quotient as an integer numerator/denominator."""

    dividend_numerator, dividend_denominator = _decimal_ratio(dividend, name="dividend")
    divisor_numerator, divisor_denominator = _decimal_ratio(divisor, name="divisor")
    if dividend_numerator <= 0 or divisor_numerator <= 0:
        raise ValueError("decimal quotient operands must be positive")
    return (
        dividend_numerator * divisor_denominator,
        dividend_denominator * divisor_numerator,
    )


def _decimal_quotient_floor(dividend: Decimal, divisor: Decimal) -> int:
    numerator, denominator = _decimal_quotient_parts(dividend, divisor)
    return numerator // denominator


def _decimal_to_roundtrip_float(value: Decimal, *, name: str) -> float:
    """Return a float only if its shortest decimal rendering keeps the value.

    Receipts encode these values as Python JSON numbers and the scheduler hands
    them to a float monotonic-clock interface. A Decimal whose distinguishing
    digits disappear from the float's shortest decimal rendering is rejected
    instead of being persisted as different proof evidence.
    """

    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} cannot be represented as a finite runtime number")
    if value.is_zero():
        return 0.0
    round_trip = Decimal(str(converted))
    value_numerator, value_denominator = _decimal_ratio(value, name=name)
    round_trip_numerator, round_trip_denominator = _decimal_ratio(round_trip, name=name)
    if value_numerator * round_trip_denominator != round_trip_numerator * value_denominator:
        raise ValueError(f"{name} cannot be represented without changing its decimal value")
    return converted


def _rational_to_finite_float(numerator: int, denominator: int, *, name: str) -> float:
    try:
        converted = numerator / denominator
    except OverflowError as error:
        raise ValueError(f"{name} cannot be represented by the monotonic clock") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} cannot be represented by the monotonic clock")
    return converted


def _runtime_grid_parts(
    duration: Decimal,
    interval: Decimal,
    slots: int,
) -> tuple[float, float, int, int]:
    """Validate the exact grid at the receipt/float-clock representation seam."""

    duration_float = _decimal_to_roundtrip_float(duration, name="duration_s")
    interval_float = _decimal_to_roundtrip_float(interval, name="interval_s")
    interval_numerator, interval_denominator = _decimal_ratio(interval, name="interval_s")
    if slots > 1:
        last_offset = _rational_to_finite_float(
            interval_numerator * (slots - 1),
            interval_denominator,
            name="observation grid",
        )
        if last_offset >= duration_float:
            raise ValueError("the observation grid cannot be represented by the monotonic clock")
    return duration_float, interval_float, interval_numerator, interval_denominator


def _decimal_nonnegative_difference(left: Decimal, right: Decimal, *, name: str) -> Decimal:
    """Return ``max(0, left - right)`` exactly, without Decimal arithmetic."""

    left_coefficient, left_exponent = _decimal_parts(left, name=name)
    right_coefficient, right_exponent = _decimal_parts(right, name=name)
    common_exponent = min(left_exponent, right_exponent)
    difference = left_coefficient * 10 ** (left_exponent - common_exponent) - right_coefficient * 10 ** (
        right_exponent - common_exponent
    )
    return _decimal_from_parts(max(0, difference), common_exponent, name=name)


def _decimal_value(
    value: object,
    *,
    name: str,
    maximum: int,
    allow_zero: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    if isinstance(value, Decimal):
        parsed = value
    else:
        try:
            rendered = str(value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name} must be a finite decimal") from error
        if len(rendered) > MAX_DECIMAL_TEXT_LENGTH:
            raise ValueError(f"{name} must be a bounded finite decimal")
        try:
            parsed = Decimal(rendered)
        except (ArithmeticError, ValueError, TypeError) as error:
            raise ValueError(f"{name} must be a finite decimal") from error
    numerator, denominator = _decimal_ratio(parsed, name=name)
    minimum_ok = numerator >= 0 if allow_zero else numerator > 0
    if not minimum_ok or numerator > maximum * denominator:
        raise ValueError(f"{name} is outside its finite bound")
    return parsed


def _bounded_int(value: object, *, name: str, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a bounded non-negative integer")
    if isinstance(value, str):
        if len(value) > 32 or not value or re.fullmatch(r"[0-9]+", value) is None:
            raise ValueError(f"{name} must be a bounded non-negative integer")
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    else:
        raise ValueError(f"{name} must be a bounded non-negative integer")
    if parsed < 0 or parsed > MAX_SAFE_COUNTER:
        raise ValueError(f"{name} must be a bounded non-negative integer")
    return parsed


def observation_slot_count(duration_s: object, interval_s: object) -> int:
    """Return exact decimal slots in the half-open interval ``[0, duration)``."""

    duration = _decimal_value(duration_s, name="duration_s", maximum=MAX_SOAK_DURATION_S)
    interval = _decimal_value(interval_s, name="interval_s", maximum=MAX_INTERVAL_S)
    numerator, denominator = _decimal_quotient_parts(duration, interval)
    slots = (numerator + denominator - 1) // denominator
    if slots > MAX_OBSERVATION_SLOTS:
        raise ValueError("the requested observation grid exceeds its fixed bound")
    return slots


@dataclass(frozen=True, slots=True)
class AvailabilityPolicy:
    """Explicit budget for timeout/unreachable windows only."""

    max_failed_observations: int = 0
    max_consecutive_failed_observations: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("max_failed_observations", self.max_failed_observations),
            ("max_consecutive_failed_observations", self.max_consecutive_failed_observations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_SAFE_COUNTER:
                raise ValueError(f"{name} must be a bounded non-negative integer")

    def as_dict(self) -> dict[str, int]:
        return {
            "max_failed_observations": self.max_failed_observations,
            "max_consecutive_failed_observations": self.max_consecutive_failed_observations,
        }


@dataclass(frozen=True, slots=True)
class _FrozenConfigFile:
    sha256: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class ConfigCapture:
    """In-memory content snapshot that can be materialized content-addressably."""

    files: tuple[_FrozenConfigFile, ...]
    combined_sha256: str

    @classmethod
    def capture(cls, paths: tp.Iterable[Path]) -> ConfigCapture:
        captured: list[_FrozenConfigFile] = []
        total = 0
        for raw_path in paths:
            if len(captured) >= MAX_CONFIG_FILES:
                raise ValueError("the configuration file count exceeds its fixed bound")
            path = Path(raw_path)
            remaining = MAX_CONFIG_BYTES - total
            metadata = path.stat()
            size = metadata.st_size
            if not stat.S_ISREG(metadata.st_mode) or size < 0 or size > remaining:
                raise ValueError("the configuration snapshot exceeds its fixed byte bound")
            with path.open("rb") as handle:
                payload = handle.read(remaining + 1)
            total += len(payload)
            if total > MAX_CONFIG_BYTES:
                raise ValueError("the configuration snapshot exceeds its fixed byte bound")
            captured.append(_FrozenConfigFile(sha256=hashlib.sha256(payload).hexdigest(), payload=payload))
        if not captured:
            raise ValueError("at least one config file is required")
        manifest = json.dumps(
            [
                {"index": index, "sha256": item.sha256, "size_bytes": len(item.payload)}
                for index, item in enumerate(captured)
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(b"eyvan-soak-frozen-config-v2\0" + manifest).hexdigest()
        return cls(files=tuple(captured), combined_sha256=f"sha256:{digest}")

    @property
    def file_count(self) -> int:
        return len(self.files)

    def materialize(self, root: Path) -> Path:
        """Atomically create and verify a read-only content-addressed snapshot."""

        snapshot_root = Path(root)
        create_private_directory(snapshot_root)
        target = snapshot_root / self.combined_sha256.removeprefix("sha256:")
        manifest = {
            "schema": "eyvan.soak.config-snapshot.v1",
            "sha256": self.combined_sha256,
            "files": [
                {"name": f"{index:04d}.config", "sha256": item.sha256, "size_bytes": len(item.payload)}
                for index, item in enumerate(self.files)
            ],
        }

        def verify() -> None:
            persisted = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            if persisted != manifest:
                raise ValueError("an existing configuration snapshot has conflicting content")
            for entry, item in zip(manifest["files"], self.files, strict=True):
                payload = (target / tp.cast(str, entry["name"])).read_bytes()
                if payload != item.payload or hashlib.sha256(payload).hexdigest() != item.sha256:
                    raise ValueError("an existing configuration snapshot failed content verification")

        if target.exists():
            verify()
            return target
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=snapshot_root))
        try:
            for entry, item in zip(manifest["files"], self.files, strict=True):
                path = temporary / tp.cast(str, entry["name"])
                with path.open("xb") as handle:
                    handle.write(item.payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                path.chmod(0o444)
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            manifest_path.chmod(0o444)
            try:
                os.rename(temporary, target)
            except FileExistsError:
                verify()
            else:
                temporary = target
                target.chmod(0o555)
                directory = os.open(snapshot_root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            verify()
            return target
        finally:
            if temporary != target:
                shutil.rmtree(temporary, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class SoakRunInputs:
    """Immutable proof identity, exact participants, grid and safety bounds."""

    candidate_sha: str | None
    source_sha: str | None
    config_sha256: str | None
    config_file_count: int
    config_snapshot_materialized: bool
    proof_id: str | None
    requested_duration_s: float
    interval_s: float
    requested_duration_decimal: str
    interval_decimal: str
    expected_executions: tuple[ExpectedExecution, ...] = ()
    max_rss_growth_bytes: int | None = None
    max_fd_growth: int | None = None
    max_database_growth_bytes: int | None = None
    max_artifact_net_growth_bytes: int | None = None
    availability_policy: AvailabilityPolicy = field(default_factory=AvailabilityPolicy)

    def __post_init__(self) -> None:
        duration = _decimal_value(
            self.requested_duration_decimal,
            name="requested_duration_s",
            maximum=MAX_SOAK_DURATION_S,
        )
        interval = _decimal_value(self.interval_decimal, name="interval_s", maximum=MAX_INTERVAL_S)
        slots = observation_slot_count(duration, interval)
        duration_float, interval_float, _, _ = _runtime_grid_parts(duration, interval, slots)
        if (
            isinstance(self.requested_duration_s, bool)
            or not isinstance(self.requested_duration_s, (int, float))
            or not math.isfinite(self.requested_duration_s)
            or duration_float != self.requested_duration_s
        ):
            raise ValueError("requested duration decimal and numeric values disagree")
        if (
            isinstance(self.interval_s, bool)
            or not isinstance(self.interval_s, (int, float))
            or not math.isfinite(self.interval_s)
            or interval_float != self.interval_s
        ):
            raise ValueError("interval decimal and numeric values disagree")
        if (
            isinstance(self.config_file_count, bool)
            or not isinstance(self.config_file_count, int)
            or not 0 <= self.config_file_count <= MAX_CONFIG_FILES
        ):
            raise ValueError("config_file_count is outside its bound")
        if not isinstance(self.config_snapshot_materialized, bool):
            raise ValueError("config_snapshot_materialized must be boolean")
        if not isinstance(self.expected_executions, tuple):
            raise ValueError("expected executions must be an immutable tuple")
        if self.expected_executions != tuple(sorted(set(self.expected_executions))):
            raise ValueError("expected executions must be sorted and deduplicated")
        if len(self.expected_executions) > MAX_PARTICIPANTS:
            raise ValueError("expected executions exceed their fixed bound")
        participants = [item.participant for item in self.expected_executions]
        if len(participants) != len(set(participants)):
            raise ValueError("each exact participant may have only one expected joint identity")
        for name, value in (
            ("max_rss_growth_bytes", self.max_rss_growth_bytes),
            ("max_fd_growth", self.max_fd_growth),
            ("max_database_growth_bytes", self.max_database_growth_bytes),
            ("max_artifact_net_growth_bytes", self.max_artifact_net_growth_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_SAFE_COUNTER
            ):
                raise ValueError(f"{name} must be a bounded non-negative integer")

    @classmethod
    def capture(
        cls,
        *,
        candidate_sha: str | None,
        source_sha: str | None,
        config_sha256: str | None,
        config_file_count: int,
        requested_duration_s: object,
        interval_s: object,
        proof_id: str | None = None,
        config_snapshot_materialized: bool = False,
        expected_executions: tp.Iterable[ExpectedExecution] = (),
        max_rss_growth_bytes: int | None = None,
        max_fd_growth: int | None = None,
        max_database_growth_bytes: int | None = None,
        max_artifact_net_growth_bytes: int | None = None,
        availability_policy: AvailabilityPolicy | None = None,
    ) -> SoakRunInputs:
        duration = _decimal_value(requested_duration_s, name="requested_duration_s", maximum=MAX_SOAK_DURATION_S)
        interval = _decimal_value(interval_s, name="interval_s", maximum=MAX_INTERVAL_S)
        slots = observation_slot_count(duration, interval)
        duration_float, interval_float, _, _ = _runtime_grid_parts(duration, interval, slots)

        def normalized(value: str | None) -> str | None:
            if value is None:
                return None
            stripped = str(value).strip()
            return stripped or None

        unique = tuple(sorted(set(expected_executions)))
        return cls(
            candidate_sha=normalized(candidate_sha),
            source_sha=normalized(source_sha),
            config_sha256=normalized(config_sha256),
            config_file_count=config_file_count,
            config_snapshot_materialized=config_snapshot_materialized,
            proof_id=normalized(proof_id),
            requested_duration_s=duration_float,
            interval_s=interval_float,
            requested_duration_decimal=str(duration),
            interval_decimal=str(interval),
            expected_executions=unique,
            max_rss_growth_bytes=max_rss_growth_bytes,
            max_fd_growth=max_fd_growth,
            max_database_growth_bytes=max_database_growth_bytes,
            max_artifact_net_growth_bytes=max_artifact_net_growth_bytes,
            availability_policy=availability_policy or AvailabilityPolicy(),
        )

    @property
    def requested_observations(self) -> int:
        return observation_slot_count(self.requested_duration_decimal, self.interval_decimal)

    @property
    def participants(self) -> tuple[ParticipantBinding, ...]:
        return tuple(item.participant for item in self.expected_executions)

    @property
    def participant_manifest_sha256(self) -> str | None:
        if not is_opaque_id(self.proof_id) or not self.participants:
            return None
        return participant_manifest_sha256(tp.cast(str, self.proof_id), self.participants)

    def expected_execution_report(self) -> list[dict[str, object]]:
        if not is_opaque_id(self.proof_id):
            return []
        proof_id = tp.cast(str, self.proof_id)
        return [item.as_receipt_dict(proof_id) for item in self.expected_executions]

    def request_manifest(self) -> dict[str, object]:
        if not is_opaque_id(self.proof_id) or not is_opaque_id(self.config_sha256) or not self.participants:
            raise ValueError("a scoped observation requires a complete frozen proof manifest")
        return {
            "proof_id": self.proof_id,
            "config_sha256": self.config_sha256,
            "participants": [participant.as_request_dict() for participant in self.participants],
        }


@dataclass(frozen=True, slots=True)
class _BudgetCounter:
    spent: float
    revision: int
    period_id: str
    period_started_at: datetime | None


@dataclass(frozen=True, slots=True)
class _CoverageSample:
    proof_id: str
    manifest_sha256: str
    counts: dict[ParticipantExecutionIdentity, int]
    truncated: bool = False
    unclassified: bool = False
    duplicates_deduplicated: int = 0


@dataclass(frozen=True, slots=True)
class _P8EvidenceSample:
    proof_id: str
    rss_bytes: int
    fd_count: int
    database_bytes: int
    spend_reconciled: bool
    provider_inventory_count: int
    cleanup_complete: bool


def _violation_sort_key(item: dict[str, object]) -> tuple[str, str]:
    return str(item.get("invariant")), str(item.get("subject_id"))


@dataclass(slots=True)
class SoakState:
    """Bounded longitudinal state for one proof run."""

    ticks: int = 0
    #: Longest stretch, in seconds, between two consecutive observations. A soak
    #: measures its duration on the monotonic clock, which on macOS stops while
    #: the machine sleeps -- so a run that a laptop suspended for thirty-one
    #: hours still completes its slot count and would otherwise report a clean
    #: seventy-two hours over a period it spent thirty-one hours not watching.
    #: An unobserved stretch is exactly "I could not find out", and the receipt
    #: has to carry it rather than average it away.
    longest_observation_gap_s: float = 0.0
    unobserved_s: float = 0.0
    attempts: int = 0
    failures: int = 0
    availability_failures: int = 0
    non_transient_failures: int = 0
    failure_categories: dict[str, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0
    consecutive_availability_failures: int = 0
    max_consecutive_availability_failures: int = 0
    streaks: dict[tuple[str, str], int] = field(default_factory=dict)
    confirmed: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)
    transient: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)
    transient_observations: int = 0
    transient_details_truncated: bool = False
    violation_response_truncated: bool = False
    previous_snapshot: dict[str, object] | None = None
    first_artifact_bytes: int | None = None
    artifact_growth: int = 0
    artifact_collected: int = 0
    longitudinal_findings: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)
    longitudinal_details_truncated: bool = False
    budget_spend_truncated: bool = False
    coverage_observations: int = 0
    coverage_previous: dict[ParticipantExecutionIdentity, int] = field(default_factory=dict)
    coverage_exercised: dict[ParticipantExecutionIdentity, int] = field(default_factory=dict)
    coverage_proof_ids: set[str] = field(default_factory=set)
    coverage_manifest_ids: set[str] = field(default_factory=set)
    coverage_truncated: bool = False
    coverage_unclassified: bool = False
    coverage_counter_regressions: int = 0
    coverage_duplicates_deduplicated: int = 0
    p8_evidence_observations: int = 0
    p8_evidence_proof_ids: set[str] = field(default_factory=set)
    p8_first: _P8EvidenceSample | None = None
    p8_last: _P8EvidenceSample | None = None
    p8_max_rss_bytes: int | None = None
    p8_max_fd_count: int | None = None
    p8_max_database_bytes: int | None = None
    p8_spend_unreconciled: bool = False

    def observe(self, payload: dict[str, object]) -> None:
        parsed, snapshot, coverage, evidence, response_truncated, budget_truncated = self._validate_payload(payload)
        self.violation_response_truncated |= response_truncated
        self.budget_spend_truncated |= budget_truncated
        self.attempts += 1
        self.ticks += 1
        self.consecutive_failures = 0
        self.consecutive_availability_failures = 0
        seen: set[tuple[str, str]] = set()
        for key in parsed:
            seen.add(key)
            streak = self.streaks.get(key, 0) + 1
            self.streaks[key] = streak
            self.transient_observations += 1
            if key not in self.confirmed:
                sample = self.transient.get(key)
                if sample is None and len(self.transient) < MAX_TRANSIENT_DETAILS:
                    sample = {
                        "invariant": key[0],
                        "subject_id": key[1],
                        "observations": 0,
                        "max_consecutive_observations": 0,
                    }
                    self.transient[key] = sample
                elif sample is None:
                    self.transient_details_truncated = True
                if sample is not None:
                    observations = sample.get("observations")
                    longest = sample.get("max_consecutive_observations")
                    sample["observations"] = (observations if isinstance(observations, int) else 0) + 1
                    sample["max_consecutive_observations"] = max(
                        longest if isinstance(longest, int) else 0,
                        streak,
                    )
            if streak >= PERSISTENCE_THRESHOLD:
                prior = self.confirmed.get(key)
                longest = prior.get("consecutive_observations") if prior else 0
                if prior is not None or len(self.confirmed) < MAX_SERVER_VIOLATIONS:
                    self.confirmed[key] = {
                        "invariant": key[0],
                        "subject_id": key[1],
                        "consecutive_observations": max(streak, longest if isinstance(longest, int) else 0),
                    }
                else:
                    self.violation_response_truncated = True
                self.transient.pop(key, None)
        for key in [key for key in self.streaks if key not in seen]:
            del self.streaks[key]
        self._compare(snapshot, coverage)
        self._observe_p8_evidence(evidence)

    @staticmethod
    def _validate_payload(
        payload: dict[str, object],
    ) -> tuple[
        list[tuple[str, str]],
        dict[str, object],
        _CoverageSample | None,
        _P8EvidenceSample | None,
        bool,
        bool,
    ]:
        if not isinstance(payload, dict) or set(payload) != {
            "checked",
            "violations",
            "violations_truncated",
            "snapshot",
        }:
            raise ValueError("the invariant response has an invalid closed schema")
        checked = payload["checked"]
        violations = payload["violations"]
        snapshot = payload["snapshot"]
        if not isinstance(checked, list) or not all(isinstance(item, str) for item in checked):
            raise ValueError("the invariant response omitted checked invariant names")
        if set(checked) != set(REQUIRED_INVARIANTS) or len(checked) != len(set(checked)):
            raise ValueError("the invariant response checked set is invalid")
        if not isinstance(violations, list) or any(not isinstance(item, dict) for item in violations):
            raise ValueError("the invariant violations are invalid")
        declared_truncated = payload["violations_truncated"]
        if not isinstance(declared_truncated, bool):
            raise ValueError("violations_truncated must be boolean")
        response_truncated = declared_truncated or len(violations) > MAX_SERVER_VIOLATIONS
        parsed: set[tuple[str, str]] = set()
        for violation in violations[:MAX_SERVER_VIOLATIONS]:
            if set(violation) != {"invariant", "subject", "detail"}:
                raise ValueError("a violation contains unexpected fields")
            invariant = violation["invariant"]
            subject = violation["subject"]
            detail = violation["detail"]
            if invariant not in REQUIRED_INVARIANTS:
                raise ValueError("a violation names an unchecked invariant")
            if not isinstance(subject, str) or not subject or not isinstance(detail, str):
                raise ValueError("a violation has invalid diagnostic fields")
            # Subject/detail are privileged diagnostics, never receipt fields.
            parsed.add((tp.cast(str, invariant), safe_subject_id(invariant, subject)))
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "observed_at",
            "budget_spend",
            "budget_spend_truncated",
            "artifact_bytes",
            "workload_count",
            "cell_count",
            "coverage",
            "p8_evidence",
        }:
            raise ValueError("the invariant snapshot has an invalid closed schema")
        observed_at = snapshot["observed_at"]
        if not isinstance(observed_at, str) or len(observed_at) > 128:
            raise ValueError("the invariant observed_at field is invalid")
        for name in ("artifact_bytes", "workload_count", "cell_count"):
            value = snapshot[name]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SAFE_COUNTER:
                raise ValueError(f"the invariant snapshot has an invalid {name}")
        budget_spend, budget_truncated = _parse_budget_spend(snapshot["budget_spend"])
        declared_budget_truncated = snapshot["budget_spend_truncated"]
        if not isinstance(declared_budget_truncated, bool):
            raise ValueError("budget_spend_truncated must be boolean")
        normalized: dict[str, object] = {
            "artifact_bytes": snapshot["artifact_bytes"],
            "budget_spend": budget_spend,
            "workload_count": snapshot["workload_count"],
            "cell_count": snapshot["cell_count"],
        }
        return (
            sorted(parsed),
            normalized,
            _parse_coverage(snapshot["coverage"]),
            _parse_p8_evidence(snapshot["p8_evidence"]),
            response_truncated,
            budget_truncated or declared_budget_truncated,
        )

    def record_failure(self, error: BaseException) -> None:
        self.attempts += 1
        self.failures += 1
        category = _exception_category(error)
        self.failure_categories[category] = self.failure_categories.get(category, 0) + 1
        self.consecutive_failures += 1
        self.max_consecutive_failures = max(self.max_consecutive_failures, self.consecutive_failures)
        if category in AVAILABILITY_BUDGET_CATEGORIES:
            self.availability_failures += 1
            self.consecutive_availability_failures += 1
            self.max_consecutive_availability_failures = max(
                self.max_consecutive_availability_failures,
                self.consecutive_availability_failures,
            )
        else:
            self.non_transient_failures += 1
            self.consecutive_availability_failures = 0
        self.streaks.clear()

    def _record_longitudinal(self, code: str, scope_id: str) -> None:
        key = (code, scope_id)
        if key in self.longitudinal_findings or len(self.longitudinal_findings) < MAX_BUDGET_SCOPES:
            self.longitudinal_findings[key] = {
                "invariant": "I6",
                "subject_id": scope_id,
                "code": code,
            }
        else:
            self.longitudinal_details_truncated = True

    def _compare(self, snapshot: dict[str, object], coverage: _CoverageSample | None) -> None:
        self._compare_coverage(coverage)
        previous = self.previous_snapshot
        self.previous_snapshot = dict(snapshot)
        current_bytes = tp.cast(int, snapshot["artifact_bytes"])
        if self.first_artifact_bytes is None:
            self.first_artifact_bytes = current_bytes
        if previous is None:
            return
        before = previous.get("budget_spend")
        after = snapshot.get("budget_spend")
        if isinstance(before, dict) and isinstance(after, dict):
            for scope_id, current in sorted(after.items()):
                earlier = before.get(scope_id)
                if not isinstance(earlier, _BudgetCounter) or not isinstance(current, _BudgetCounter):
                    continue
                if current.revision < earlier.revision:
                    self._record_longitudinal("budget_revision_regressed", scope_id)
                elif current.period_id == earlier.period_id:
                    if current.period_started_at != earlier.period_started_at:
                        self._record_longitudinal("budget_period_identity_conflict", scope_id)
                    elif current.spent < earlier.spent:
                        self._record_longitudinal("budget_spend_decreased", scope_id)
                else:
                    proved_advance = (
                        earlier.period_started_at is not None
                        and current.period_started_at is not None
                        and current.period_started_at > earlier.period_started_at
                        and current.revision > earlier.revision
                    )
                    if not proved_advance:
                        self._record_longitudinal("budget_period_advance_unproved", scope_id)
        earlier_bytes = previous.get("artifact_bytes")
        if isinstance(earlier_bytes, int):
            growth = self.artifact_growth + max(0, current_bytes - earlier_bytes)
            collected = self.artifact_collected + max(0, earlier_bytes - current_bytes)
            if growth > MAX_SAFE_COUNTER or collected > MAX_SAFE_COUNTER:
                self.longitudinal_details_truncated = True
            self.artifact_growth = min(growth, MAX_SAFE_COUNTER)
            self.artifact_collected = min(collected, MAX_SAFE_COUNTER)

    def _compare_coverage(self, sample: _CoverageSample | None) -> None:
        if sample is None:
            return
        first = self.coverage_observations == 0
        self.coverage_observations += 1
        self.coverage_proof_ids.add(sample.proof_id)
        self.coverage_manifest_ids.add(sample.manifest_sha256)
        self.coverage_truncated |= sample.truncated
        self.coverage_unclassified |= sample.unclassified
        self.coverage_duplicates_deduplicated += sample.duplicates_deduplicated
        if not first:
            for identity in sorted(set(self.coverage_previous) | set(sample.counts)):
                previous = self.coverage_previous.get(identity, 0)
                current = sample.counts.get(identity, 0)
                if current < previous:
                    self.coverage_counter_regressions += 1
                    continue
                delta = current - previous
                if not delta:
                    continue
                if identity not in self.coverage_exercised and len(self.coverage_exercised) >= MAX_COVERAGE_ENTRIES:
                    self.coverage_truncated = True
                    continue
                accumulated = self.coverage_exercised.get(identity, 0) + delta
                if accumulated > MAX_SAFE_COUNTER:
                    self.coverage_truncated = True
                self.coverage_exercised[identity] = min(accumulated, MAX_SAFE_COUNTER)
        self.coverage_previous = dict(sample.counts)

    def _observe_p8_evidence(self, sample: _P8EvidenceSample | None) -> None:
        if sample is None:
            return
        self.p8_evidence_observations += 1
        self.p8_evidence_proof_ids.add(sample.proof_id)
        if self.p8_first is None:
            self.p8_first = sample
        self.p8_last = sample
        self.p8_max_rss_bytes = max(self.p8_max_rss_bytes or sample.rss_bytes, sample.rss_bytes)
        self.p8_max_fd_count = max(self.p8_max_fd_count or sample.fd_count, sample.fd_count)
        self.p8_max_database_bytes = max(
            self.p8_max_database_bytes or sample.database_bytes,
            sample.database_bytes,
        )
        self.p8_spend_unreconciled |= not sample.spend_reconciled

    def identity_report(self) -> dict[str, object]:
        executions = [
            {**identity.as_dict(), "exercise_count": count}
            for identity, count in sorted(self.coverage_exercised.items())
        ]
        return {"executions": executions}

    def artifact_report(self) -> dict[str, object]:
        ending = (self.previous_snapshot or {}).get("artifact_bytes")
        net_growth = None
        if self.first_artifact_bytes is not None and isinstance(ending, int):
            net_growth = ending - self.first_artifact_bytes
        return {
            "start_bytes": self.first_artifact_bytes,
            "end_bytes": ending if isinstance(ending, int) else None,
            "created_bytes": self.artifact_growth,
            "collected_bytes": self.artifact_collected,
            "net_growth_bytes": net_growth,
            "creation_exercised": self.artifact_growth > 0,
            "collection_exercised": self.artifact_collected > 0,
        }

    def p8_evidence_report(self) -> dict[str, object]:
        first = self.p8_first
        last = self.p8_last
        return {
            "observations": self.p8_evidence_observations,
            "rss_growth_bytes": (
                None
                if first is None or self.p8_max_rss_bytes is None
                else max(0, self.p8_max_rss_bytes - first.rss_bytes)
            ),
            "fd_growth": (
                None if first is None or self.p8_max_fd_count is None else max(0, self.p8_max_fd_count - first.fd_count)
            ),
            "database_growth_bytes": (
                None
                if first is None or self.p8_max_database_bytes is None
                else max(0, self.p8_max_database_bytes - first.database_bytes)
            ),
            "spend_reconciled": None if last is None else not self.p8_spend_unreconciled,
            "final_provider_inventory_count": None if last is None else last.provider_inventory_count,
            "cleanup_complete": None if last is None else last.cleanup_complete,
        }

    def report(self, availability_policy: AvailabilityPolicy | None = None) -> dict[str, object]:
        policy = availability_policy or AvailabilityPolicy()
        persistent = [dict(item) for item in sorted(self.confirmed.values(), key=_violation_sort_key)]
        transient = [dict(item) for item in sorted(self.transient.values(), key=_violation_sort_key)]
        longitudinal = [dict(item) for item in sorted(self.longitudinal_findings.values(), key=_violation_sort_key)]
        availability_within_policy = (
            self.availability_failures <= policy.max_failed_observations
            and self.max_consecutive_availability_failures <= policy.max_consecutive_failed_observations
        )
        truncation = {
            "violations": self.violation_response_truncated,
            "transient_details": self.transient_details_truncated,
            "longitudinal_details": self.longitudinal_details_truncated,
            "budget_scopes": self.budget_spend_truncated,
            "coverage": self.coverage_truncated,
        }
        evidence_truncated = any(truncation.values())
        artifact = self.artifact_report()
        return {
            "observations": self.ticks,
            "failed_observations": self.failures,
            "persistence_threshold": PERSISTENCE_THRESHOLD,
            "violations": persistent,
            "over_time": longitudinal,
            "artifact_bytes": artifact["end_bytes"],
            "artifact_growth": self.artifact_growth,
            "healthy": (
                self.ticks > 0
                and availability_within_policy
                and self.non_transient_failures == 0
                and not persistent
                and not longitudinal
                and not evidence_truncated
            ),
            "attempted_observations": self.attempts,
            "longest_observation_gap_s": round(self.longest_observation_gap_s, 3),
            "unobserved_s": round(self.unobserved_s, 3),
            "unreachable_observations": self.availability_failures,
            "availability_budgeted_observations": self.availability_failures,
            "non_transient_failed_observations": self.non_transient_failures,
            "failure_categories": dict(sorted(self.failure_categories.items())),
            "max_consecutive_failed_observations": self.max_consecutive_failures,
            "max_consecutive_availability_failures": self.max_consecutive_availability_failures,
            "availability_policy": policy.as_dict(),
            "availability_within_policy": availability_within_policy,
            "persistent_violations": persistent,
            "transient_violations": transient,
            "transient_violation_observations": self.transient_observations,
            "evidence_truncated": evidence_truncated,
            "truncation": truncation,
            "artifact": artifact,
            "artifact_collected": self.artifact_collected,
            "coverage_observations": self.coverage_observations,
            "coverage_counter_regressions": self.coverage_counter_regressions,
            "coverage_duplicates_deduplicated": self.coverage_duplicates_deduplicated,
            "coverage_unclassified": self.coverage_unclassified,
            "identity": self.identity_report(),
            "p8_evidence": self.p8_evidence_report(),
        }


def _exception_category(error: BaseException, *, filesystem: bool = False) -> str:
    """Map an exception chain to one fixed receipt-safe category."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int):
            if status_code in {401, 403}:
                return "authorization"
            if status_code in {408, 504}:
                return "timeout"
            if status_code >= 500:
                return "server"
            return "request"
        if isinstance(current, TimeoutError):
            return "timeout"
        if isinstance(current, PermissionError):
            return "filesystem"
        if isinstance(current, ConnectionError):
            return "unreachable"
        if isinstance(current, OSError):
            return "filesystem"
        if isinstance(current, click.ClickException):
            return "unreachable"
        if isinstance(current, (ValueError, TypeError, KeyError, json.JSONDecodeError, OverflowError)):
            return "invalid_response"
        current = current.__cause__ or current.__context__
    category = "internal"
    if category not in SAFE_FAILURE_CATEGORIES:  # pragma: no cover
        raise AssertionError("unsafe failure category")
    return category


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("a budget period timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("a budget period timestamp is invalid") from error
    if parsed.tzinfo is None or not 2000 <= parsed.year <= 9999:
        raise ValueError("a budget period timestamp must be bounded and timezone-aware")
    return parsed.astimezone(UTC)


def _parse_budget_spend(value: object) -> tuple[dict[str, _BudgetCounter], bool]:
    if not isinstance(value, list):
        raise ValueError("the invariant snapshot omitted versioned budget spend")
    truncated = len(value) > MAX_BUDGET_SCOPES
    parsed: dict[str, _BudgetCounter] = {}
    for entry in value[:MAX_BUDGET_SCOPES]:
        if not isinstance(entry, dict) or set(entry) != {
            "scope_id",
            "spent",
            "revision",
            "period_id",
            "period_started_at",
        }:
            raise ValueError("a budget counter has an invalid closed schema")
        scope_id = entry["scope_id"]
        period_id = entry["period_id"]
        spent = entry["spent"]
        revision = entry["revision"]
        raw_started = entry["period_started_at"]
        if not is_opaque_id(scope_id) or not is_opaque_id(period_id):
            raise ValueError("a budget counter has an invalid opaque identity")
        if (
            isinstance(spent, bool)
            or not isinstance(spent, (int, float))
            or not math.isfinite(spent)
            or not 0 <= spent <= MAX_SAFE_COUNTER
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or not 0 <= revision <= MAX_SAFE_COUNTER
        ):
            raise ValueError("a budget counter has invalid bounded values")
        started = _parse_timestamp(raw_started)
        expected_period_id = safe_period_id(raw_started)
        if period_id != expected_period_id:
            raise ValueError("a budget period identity does not match its stable period")
        if scope_id in parsed:
            raise ValueError("the invariant snapshot repeats a budget scope")
        parsed[tp.cast(str, scope_id)] = _BudgetCounter(
            spent=float(spent),
            revision=revision,
            period_id=tp.cast(str, period_id),
            period_started_at=started,
        )
    return parsed, truncated


def _parse_coverage(value: object) -> _CoverageSample | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "proof_id",
        "manifest_sha256",
        "executions",
        "truncated",
    }:
        raise ValueError("coverage has an invalid closed schema")
    proof_id = value["proof_id"]
    manifest_sha256 = value["manifest_sha256"]
    entries = value["executions"]
    declared_truncated = value["truncated"]
    if not is_opaque_id(proof_id) or not is_opaque_id(manifest_sha256):
        raise ValueError("coverage has an invalid proof binding")
    if not isinstance(entries, list) or not isinstance(declared_truncated, bool):
        raise ValueError("coverage has invalid executions or truncation")
    truncated = declared_truncated or len(entries) > MAX_COVERAGE_ENTRIES
    counts: dict[ParticipantExecutionIdentity, int] = {}
    duplicates = 0
    unclassified = False
    for entry in entries[:MAX_COVERAGE_ENTRIES]:
        if not isinstance(entry, dict) or set(entry) != {
            "participant_id",
            "capability",
            "backend",
            "provider",
            "execution_count",
        }:
            raise ValueError("an execution counter has an invalid closed schema")
        count = entry["execution_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count not in {0, 1}:
            raise ValueError("an exact participant execution counter must be zero or one")
        identity = ParticipantExecutionIdentity(
            participant_id=entry["participant_id"],
            identity=ExecutionIdentity.from_dict({name: entry[name] for name in ("capability", "backend", "provider")}),
        )
        unclassified |= not identity.identity.classified
        if identity in counts:
            duplicates += 1
            counts[identity] = max(counts[identity], count)
        else:
            counts[identity] = count
    return _CoverageSample(
        proof_id=tp.cast(str, proof_id),
        manifest_sha256=tp.cast(str, manifest_sha256),
        counts=dict(sorted(counts.items())),
        truncated=truncated,
        unclassified=unclassified,
        duplicates_deduplicated=duplicates,
    )


def _parse_p8_evidence(value: object) -> _P8EvidenceSample | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "proof_id",
        "rss_bytes",
        "fd_count",
        "database_bytes",
        "spend_reconciled",
        "provider_inventory_count",
        "cleanup_complete",
    }:
        raise ValueError("P8 evidence has an invalid closed schema")
    if value["schema"] != "eyvan.soak.external-evidence.v1" or not is_opaque_id(value["proof_id"]):
        raise ValueError("P8 evidence has an invalid schema or proof binding")
    numeric: dict[str, int] = {}
    for name in ("rss_bytes", "fd_count", "database_bytes", "provider_inventory_count"):
        parsed = _bounded_int(value[name], name=name)
        assert parsed is not None
        numeric[name] = parsed
    if not isinstance(value["spend_reconciled"], bool) or not isinstance(value["cleanup_complete"], bool):
        raise ValueError("P8 evidence booleans are invalid")
    return _P8EvidenceSample(
        proof_id=tp.cast(str, value["proof_id"]),
        rss_bytes=numeric["rss_bytes"],
        fd_count=numeric["fd_count"],
        database_bytes=numeric["database_bytes"],
        spend_reconciled=value["spend_reconciled"],
        provider_inventory_count=numeric["provider_inventory_count"],
        cleanup_complete=value["cleanup_complete"],
    )


def run_soak(
    observe: tp.Callable[[], dict[str, object]],
    *,
    ticks: int,
    sleep: tp.Callable[[float], None],
    interval_s: float,
    on_tick: tp.Callable[[int, SoakState], None] | None = None,
    state: SoakState | None = None,
) -> SoakState:
    """Make exactly ``ticks`` injected attempts (the unit-test primitive)."""

    if isinstance(ticks, bool) or not isinstance(ticks, int) or not 0 <= ticks <= MAX_OBSERVATION_SLOTS:
        raise ValueError("ticks is outside its fixed bound")
    if not math.isfinite(interval_s) or interval_s < 0:
        raise ValueError("interval_s must be finite and non-negative")
    current = state or SoakState()
    for tick in range(ticks):
        try:
            current.observe(observe())
        except Exception as error:
            current.record_failure(error)
        if on_tick is not None:
            on_tick(tick, current)
        if tick + 1 < ticks:
            sleep(interval_s)
    return current


#: How far past one interval two observations may sit before the stretch counts
#: as unobserved. Two intervals absorbs ordinary scheduling jitter and a slow
#: poll; anything beyond it is time the fleet genuinely was not watched.
_OBSERVATION_GAP_TOLERANCE: tp.Final[float] = 2.0


def _finite_monotonic(monotonic: tp.Callable[[], float]) -> float:
    value = monotonic()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("monotonic clock returned a non-finite value")
    return float(value)


def _utc_wall_datetime(now: tp.Callable[[], datetime]) -> datetime:
    value = now()
    if not isinstance(value, datetime):
        raise ValueError("wall clock returned a non-datetime value")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def run_soak_for_duration(
    observe: tp.Callable[[], dict[str, object]],
    *,
    duration_s: object,
    interval_s: object,
    sleep: tp.Callable[[float], None],
    monotonic: tp.Callable[[], float],
    wall_clock: tp.Callable[[], datetime] | None = None,
    on_tick: tp.Callable[[int, SoakState], None] | None = None,
    state: SoakState | None = None,
) -> SoakState:
    """Schedule monotonic slots, checking continuity against both clocks.

    A forward wall-clock correction is conservatively an observation gap, not
    proof of suspension. A backward correction cannot hide a monotonic gap.
    """

    duration = _decimal_value(duration_s, name="duration_s", maximum=MAX_SOAK_DURATION_S)
    interval = _decimal_value(interval_s, name="interval_s", maximum=MAX_INTERVAL_S)
    total_slots = observation_slot_count(duration, interval)
    duration_float, _, interval_numerator, interval_denominator = _runtime_grid_parts(
        duration,
        interval,
        total_slots,
    )

    def slot_offset(index: int) -> float:
        return _rational_to_finite_float(
            interval_numerator * index,
            interval_denominator,
            name="observation grid",
        )

    current = state or SoakState()
    started = _finite_monotonic(monotonic)
    deadline = started + duration_float
    if not math.isfinite(deadline):
        raise ValueError("the monotonic deadline overflowed")
    slot = 0
    previous: float | None = None
    previous_wall: datetime | None = None
    wall_clock = wall_clock or (lambda: datetime.now(UTC))
    interval_float = _rational_to_finite_float(interval_numerator, interval_denominator, name="observation interval")

    def record_boundary(now: float, wall: datetime) -> None:
        nonlocal previous, previous_wall
        # Some monotonic clocks exclude suspend. Fresh evidence after resume
        # cannot prove continuity across the missing wall-clock interval.
        if previous is not None and previous_wall is not None:
            gap = max(now - previous, (wall - previous_wall).total_seconds())
            if gap > interval_float * _OBSERVATION_GAP_TOLERANCE:
                current.longest_observation_gap_s = max(current.longest_observation_gap_s, gap)
                current.unobserved_s += gap - interval_float
        previous = now
        previous_wall = wall

    while slot < total_slots:
        scheduled = started + slot_offset(slot)
        now = _finite_monotonic(monotonic)
        if now < scheduled:
            sleep(scheduled - now)
        now = _finite_monotonic(monotonic)
        if now >= deadline:
            break
        low = slot
        high = total_slots
        while low < high:
            middle = (low + high) // 2
            if started + slot_offset(middle) <= now:
                low = middle + 1
            else:
                high = middle
        slot = max(slot, low - 1)
        record_boundary(now, _utc_wall_datetime(wall_clock))
        try:
            current.observe(observe())
        except Exception as error:
            current.record_failure(error)
        if on_tick is not None:
            on_tick(current.attempts - 1, current)
        slot += 1
    remaining = deadline - _finite_monotonic(monotonic)
    if remaining > 0:
        sleep(remaining)
    # There may be no next poll to expose a slow last request/checkpoint or
    # suspension during the final deadline wait. Apply the same tolerance to
    # the entire terminal stretch, without counting earlier gaps twice.
    record_boundary(_finite_monotonic(monotonic), _utc_wall_datetime(wall_clock))
    return current


def _valid_sha(value: str | None) -> bool:
    return (
        value is not None
        and re.fullmatch(
            r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|sha256:[0-9a-f]{64})",
            value,
        )
        is not None
    )


def _valid_sha256(value: str | None) -> bool:
    return is_opaque_id(value)


def _input_reason_counts(inputs: SoakRunInputs) -> dict[str, int]:
    reasons: dict[str, int] = {}

    def add(code: str, count: int = 1) -> None:
        reasons[code] = reasons.get(code, 0) + count

    if not _valid_sha(inputs.candidate_sha):
        add("candidate_identity_missing")
    if not _valid_sha(inputs.source_sha):
        add("source_identity_missing")
    if (
        not _valid_sha256(inputs.config_sha256)
        or inputs.config_file_count <= 0
        or not inputs.config_snapshot_materialized
    ):
        add("configuration_snapshot_missing")
    if not is_opaque_id(inputs.proof_id):
        add("invalid_proof_identity")
    if not inputs.expected_executions:
        add("expected_execution_missing")
    resources = {item.identity.provider.resource_class for item in inputs.expected_executions}
    if "cpu" not in resources:
        add("p8_cpu_mix_missing")
    if "tpu" not in resources:
        add("p8_tpu_mix_missing")
    if inputs.max_rss_growth_bytes is None:
        add("rss_growth_bound_missing")
    if inputs.max_fd_growth is None:
        add("fd_growth_bound_missing")
    if inputs.max_database_growth_bytes is None:
        add("database_growth_bound_missing")
    if inputs.max_artifact_net_growth_bytes is None:
        add("artifact_growth_bound_missing")
    return reasons


def _add_reason(mapping: dict[str, int], code: str, count: int = 1) -> None:
    if code not in REASON_CODES:
        code = "internal_preflight_error"
    mapping[code] = mapping.get(code, 0) + max(1, count)


def _missing_expected_executions(state: SoakState, inputs: SoakRunInputs) -> list[dict[str, object]]:
    if not is_opaque_id(inputs.proof_id):
        return []
    proof_id = tp.cast(str, inputs.proof_id)
    observed = {identity for identity, count in state.coverage_exercised.items() if count > 0}
    missing: list[dict[str, object]] = []
    for expected in inputs.expected_executions:
        key = ParticipantExecutionIdentity(
            participant_id=expected.participant.opaque_id(proof_id),
            identity=expected.identity,
        )
        if key not in observed:
            missing.append(key.as_dict())
    return missing


def _format_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError("receipt timestamps must be datetime values")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_soak_receipt(
    state: SoakState,
    inputs: SoakRunInputs,
    *,
    started_at: datetime,
    ended_at: datetime,
    actual_duration_s: object,
    status: str,
    config_unchanged: bool | None = None,
    extra_reasons: tp.Iterable[str] = (),
    extra_reason_codes: tp.Iterable[str] = (),
) -> dict[str, object]:
    """Build a deterministic receipt containing closed codes and opaque IDs only.

    ``config_unchanged`` and ``extra_reasons`` remain compatibility parameters.
    Endpoint re-reads and arbitrary text are deliberately ignored; only a frozen
    snapshot and closed ``extra_reason_codes`` can affect the verdict.
    """

    del config_unchanged, extra_reasons
    if status not in {"not_started", "running", "completed", "interrupted"}:
        raise ValueError("unsupported soak status")
    numeric_invalid = False
    shortfall_exceeded = False
    try:
        actual_decimal = _decimal_value(
            actual_duration_s,
            name="actual_duration_s",
            maximum=MAX_SOAK_DURATION_S,
            allow_zero=True,
        )
        actual = _decimal_to_roundtrip_float(actual_decimal, name="actual_duration_s")
        requested_decimal = _decimal_value(
            inputs.requested_duration_decimal,
            name="requested_duration_s",
            maximum=MAX_SOAK_DURATION_S,
        )
        shortfall_decimal = _decimal_nonnegative_difference(
            requested_decimal,
            actual_decimal,
            name="duration_shortfall_s",
        )
        shortfall = _decimal_to_roundtrip_float(shortfall_decimal, name="duration_shortfall_s")
        shortfall_numerator, shortfall_denominator = _decimal_ratio(
            shortfall_decimal,
            name="duration_shortfall_s",
        )
        shortfall_exceeded = shortfall_numerator * 1_000_000_000 > shortfall_denominator
    except ValueError:
        # An unsafe numeric value is not rounded into proof evidence. Replace it
        # with closed finite placeholders and mark the receipt inconclusive.
        actual = 0.0
        shortfall = inputs.requested_duration_s
        numeric_invalid = True
    report = state.report(inputs.availability_policy)
    artifact = state.artifact_report()
    p8 = state.p8_evidence_report()
    missing_executions = _missing_expected_executions(state, inputs)

    failures: dict[str, int] = {}
    inconclusive: dict[str, int] = dict(_input_reason_counts(inputs))
    if state.longest_observation_gap_s > 0:
        # Inconclusive, not failed. Nothing was observed to be wrong during the
        # gap -- and nothing was observed to be right either, which is the whole
        # objection: a soak's claim is that invariants held *continuously*, and
        # it cannot make that claim about hours it did not watch.
        _add_reason(inconclusive, "observation_gap")
    if state.confirmed:
        _add_reason(failures, "persistent_invariant_violation", len(state.confirmed))
    if state.longitudinal_findings:
        _add_reason(failures, "longitudinal_invariant_violation", len(state.longitudinal_findings))
    if status == "not_started":
        _add_reason(inconclusive, "run_not_started")
    elif status == "running":
        _add_reason(inconclusive, "run_still_running")
    elif status == "interrupted":
        _add_reason(inconclusive, "run_interrupted")
    if numeric_invalid:
        _add_reason(inconclusive, "duration_evidence_invalid")
    elif shortfall_exceeded:
        _add_reason(inconclusive, "duration_shortfall")
    if state.attempts < inputs.requested_observations:
        _add_reason(inconclusive, "observation_slot_shortfall", inputs.requested_observations - state.attempts)
    if state.ticks == 0:
        _add_reason(inconclusive, "no_successful_observation")
    if report["availability_within_policy"] is not True:
        _add_reason(inconclusive, "availability_budget_exceeded")
    if state.non_transient_failures:
        _add_reason(inconclusive, "non_transient_observation_error", state.non_transient_failures)
    if state.coverage_observations < state.ticks:
        _add_reason(inconclusive, "coverage_missing", state.ticks - state.coverage_observations)
    expected_proof = inputs.proof_id if is_opaque_id(inputs.proof_id) else None
    expected_manifest = inputs.participant_manifest_sha256
    if (
        expected_proof is None
        or state.coverage_proof_ids != {expected_proof}
        or expected_manifest is None
        or state.coverage_manifest_ids != {expected_manifest}
    ):
        _add_reason(inconclusive, "coverage_binding_mismatch")
    if missing_executions:
        _add_reason(inconclusive, "expected_execution_not_exercised", len(missing_executions))
    if report["evidence_truncated"] is True:
        _add_reason(inconclusive, "evidence_truncated")
    if state.coverage_counter_regressions:
        _add_reason(inconclusive, "coverage_counter_regression", state.coverage_counter_regressions)
    if state.coverage_duplicates_deduplicated:
        _add_reason(inconclusive, "coverage_duplicate", state.coverage_duplicates_deduplicated)
    if state.coverage_unclassified:
        _add_reason(inconclusive, "coverage_unclassified")
    if artifact["creation_exercised"] is not True:
        _add_reason(inconclusive, "artifact_creation_missing")
    if artifact["collection_exercised"] is not True:
        _add_reason(inconclusive, "artifact_collection_missing")
    net_growth = artifact["net_growth_bytes"]
    if inputs.max_artifact_net_growth_bytes is not None and isinstance(net_growth, int):
        if net_growth > inputs.max_artifact_net_growth_bytes:
            _add_reason(failures, "artifact_net_growth_exceeded")

    evidence_complete = state.p8_evidence_observations == state.ticks and state.ticks > 0
    if not evidence_complete:
        _add_reason(inconclusive, "p8_evidence_missing", max(1, state.ticks - state.p8_evidence_observations))
    if expected_proof is None or state.p8_evidence_proof_ids != {expected_proof}:
        _add_reason(inconclusive, "p8_evidence_binding_mismatch")
    for field_name, missing_code, exceeded_code, bound in (
        ("rss_growth_bytes", "rss_growth_evidence_missing", "rss_growth_exceeded", inputs.max_rss_growth_bytes),
        ("fd_growth", "fd_growth_evidence_missing", "fd_growth_exceeded", inputs.max_fd_growth),
        (
            "database_growth_bytes",
            "database_growth_evidence_missing",
            "database_growth_exceeded",
            inputs.max_database_growth_bytes,
        ),
    ):
        value = p8[field_name]
        if value is None:
            _add_reason(inconclusive, missing_code)
        elif bound is not None and tp.cast(int, value) > bound:
            _add_reason(failures, exceeded_code)
    if p8["spend_reconciled"] is None:
        _add_reason(inconclusive, "spend_reconciliation_evidence_missing")
    elif p8["spend_reconciled"] is not True:
        _add_reason(failures, "spend_unreconciled")
    if p8["final_provider_inventory_count"] is None:
        _add_reason(inconclusive, "provider_inventory_evidence_missing")
    elif status == "completed" and p8["final_provider_inventory_count"] != 0:
        _add_reason(failures, "provider_inventory_not_empty")
    if p8["cleanup_complete"] is None:
        _add_reason(inconclusive, "cleanup_evidence_missing")
    elif status == "completed" and p8["cleanup_complete"] is not True:
        _add_reason(failures, "cleanup_incomplete")

    reason_count = 0
    for code in extra_reason_codes:
        if reason_count >= MAX_RECEIPT_REASONS:
            _add_reason(inconclusive, "reason_collection_truncated")
            break
        reason_count += 1
        _add_reason(inconclusive, code)
    if len(failures) + len(inconclusive) > MAX_RECEIPT_REASONS:
        keep = sorted({*failures, *inconclusive})[: MAX_RECEIPT_REASONS - 1]
        failures = {code: failures[code] for code in keep if code in failures}
        inconclusive = {code: inconclusive[code] for code in keep if code in inconclusive}
        _add_reason(inconclusive, "reason_collection_truncated")

    if failures:
        verdict = "fail"
    elif inconclusive:
        verdict = "inconclusive"
    else:
        verdict = "pass"
    reasons = sorted({*failures, *inconclusive})
    valid_candidate = inputs.candidate_sha if _valid_sha(inputs.candidate_sha) else None
    valid_source = inputs.source_sha if _valid_sha(inputs.source_sha) else None
    valid_config = inputs.config_sha256 if _valid_sha256(inputs.config_sha256) else None
    valid_proof = inputs.proof_id if is_opaque_id(inputs.proof_id) else None
    expected = inputs.expected_execution_report()
    identity = state.identity_report()
    receipt = {
        **report,
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
        "run_proof_id": valid_proof,
        "participant_manifest_sha256": inputs.participant_manifest_sha256,
        "candidate_sha": valid_candidate,
        "source_sha": valid_source,
        "candidate": {"sha": valid_candidate, "source_sha": valid_source},
        "config_hash": valid_config,
        "configuration": {
            "sha256": valid_config,
            "file_count": inputs.config_file_count if valid_config else 0,
            "frozen_content_addressed_snapshot": inputs.config_snapshot_materialized and valid_config is not None,
            "execution_binding_key": "workload_metadata",
        },
        "started_at": _format_timestamp(started_at),
        "ended_at": _format_timestamp(ended_at),
        "requested_duration_s": inputs.requested_duration_s,
        "actual_duration_s": actual,
        "duration_shortfall_s": shortfall,
        "duration": {"requested_s": inputs.requested_duration_s, "actual_s": actual, "shortfall_s": shortfall},
        "observation_counts": {
            "requested": inputs.requested_observations,
            "attempted": state.attempts,
            "successful": state.ticks,
            "failed": state.failures,
            "availability_budgeted": state.availability_failures,
            "non_transient_failed": state.non_transient_failures,
            "max_consecutive_failed": state.max_consecutive_failures,
            "max_consecutive_availability_failed": state.max_consecutive_availability_failures,
        },
        "persistent_violations": [
            dict(item) for item in tp.cast(list[dict[str, object]], report["persistent_violations"])
        ],
        "transient_violations": [
            dict(item) for item in tp.cast(list[dict[str, object]], report["transient_violations"])
        ],
        "longitudinal_findings": [
            dict(item) for item in sorted(state.longitudinal_findings.values(), key=_violation_sort_key)
        ],
        "artifact": dict(artifact),
        "expected_executions": [dict(item) for item in expected],
        "missing_expected_executions": [dict(item) for item in missing_executions],
        "identity_requirements_met": bool(expected) and not missing_executions,
        "identity": identity,
        "resource_evidence": dict(p8),
        "resource_bounds": {
            "max_rss_growth_bytes": inputs.max_rss_growth_bytes,
            "max_fd_growth": inputs.max_fd_growth,
            "max_database_growth_bytes": inputs.max_database_growth_bytes,
            "max_artifact_net_growth_bytes": inputs.max_artifact_net_growth_bytes,
        },
        # Durable rows prove state-machine evidence only. They are deliberately
        # not represented as provider operation or hardware attestation.
        "execution_evidence_scope": "durable_control_plane_state",
        "hardware_attested": False,
        "reason_counts": {code: failures.get(code, inconclusive.get(code, 0)) for code in reasons},
        "failure_reason_codes": sorted(failures),
        "inconclusive_reason_codes": sorted(inconclusive),
        "reasons": reasons,
        "verdict": verdict,
        "exit_code": VERDICT_EXIT_CODES[verdict],
        "healthy": verdict == "pass",
    }
    return receipt


def build_preflight_receipt(reason_codes: tp.Iterable[str]) -> dict[str, object]:
    """Return the minimal closed receipt used before full inputs can exist."""

    counts: dict[str, int] = {}
    for code in reason_codes:
        _add_reason(counts, code)
    if not counts:
        _add_reason(counts, "internal_preflight_error")
    reasons = sorted(counts)
    return {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "not_started",
        "verdict": "inconclusive",
        "exit_code": 2,
        "healthy": False,
        "run_proof_id": None,
        "candidate_sha": None,
        "source_sha": None,
        "config_hash": None,
        "reason_counts": {code: counts[code] for code in reasons},
        "failure_reason_codes": [],
        "inconclusive_reason_codes": reasons,
        "reasons": reasons,
    }


def execute_soak(
    observe: tp.Callable[[], dict[str, object]],
    *,
    inputs: SoakRunInputs,
    output: Path | None,
    sleep: tp.Callable[[float], None] = time.sleep,
    monotonic: tp.Callable[[], float] = time.monotonic,
    now: tp.Callable[[], datetime] | None = None,
    config_is_unchanged: tp.Callable[[], bool] | None = None,
    on_tick: tp.Callable[[int, SoakState], None] | None = None,
    preflight_reasons: tp.Iterable[str] = (),
    preflight_reason_codes: tp.Iterable[str] = (),
) -> dict[str, object]:
    """Run, checkpoint and finalize without polling mutable config endpoints."""

    del config_is_unchanged, preflight_reasons
    wall_clock = now or (lambda: datetime.now(UTC))
    state = SoakState()
    closed_preflight = list(preflight_reason_codes)
    try:
        started_at = _utc_wall_datetime(wall_clock)
        started_monotonic = _finite_monotonic(monotonic)
    except Exception as error:
        category = _exception_category(error, filesystem=True)
        closed_preflight.append("runner_filesystem_error" if category == "filesystem" else "runner_internal_error")
        final = build_preflight_receipt(closed_preflight)
        if output is not None:
            write_receipt_atomic(output, final)
        return final

    def receipt(status: str, *, codes: tp.Iterable[str] = ()) -> dict[str, object]:
        try:
            ended = _utc_wall_datetime(wall_clock)
        except Exception:
            ended = started_at
        try:
            elapsed = max(_finite_monotonic(monotonic) - started_monotonic, (ended - started_at).total_seconds())
        except Exception:
            elapsed = float("nan")
        return build_soak_receipt(
            state,
            inputs,
            started_at=started_at,
            ended_at=ended,
            actual_duration_s=elapsed,
            status=status,
            extra_reason_codes=(*closed_preflight, *tuple(codes)),
        )

    if closed_preflight or _input_reason_counts(inputs):
        final = receipt("not_started")
        if output is not None:
            write_receipt_atomic(output, final)
        return final

    initial = receipt("running")
    if output is not None:
        write_receipt_atomic(output, initial)

    def checkpoint(tick: int, current: SoakState) -> None:
        if output is not None:
            write_receipt_atomic(output, receipt("running"))
        if on_tick is not None:
            on_tick(tick, current)

    status = "completed"
    run_codes: list[str] = []
    # A long soak is started in the background, and a background job inherits
    # SIGINT ignored -- so the KeyboardInterrupt path below, which is the one that
    # writes the receipt, is unreachable for exactly the runs that need it most.
    # SIGTERM is what an operator actually sends, and its default action ends the
    # process with the whole longitudinal state still in memory: the terminated
    # P8 lost the identities of three persistent violations that way. Raising
    # KeyboardInterrupt from the handler routes SIGTERM into the same graceful
    # exit, so the receipt is written either way.
    with _terminate_as_interrupt():
        try:
            run_soak_for_duration(
                observe,
                duration_s=inputs.requested_duration_decimal,
                interval_s=inputs.interval_decimal,
                sleep=sleep,
                monotonic=monotonic,
                wall_clock=wall_clock,
                on_tick=checkpoint,
                state=state,
            )
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as error:
            status = "interrupted"
            category = _exception_category(error, filesystem=True)
            run_codes.append("runner_filesystem_error" if category == "filesystem" else "runner_internal_error")
    final = receipt(status, codes=run_codes)
    if output is not None:
        write_receipt_atomic(output, final)
    return final


@contextlib.contextmanager
def _terminate_as_interrupt() -> tp.Iterator[None]:
    """Route SIGTERM into the interrupt path, and restore what was there.

    Installed only for the duration of the run, and only when this is the main
    thread of the main interpreter, because that is the only place
    :func:`signal.signal` is allowed. A handler left installed afterwards would
    change how the process behaves for every later caller, and a soak embedded in
    a larger program must not do that.
    """

    import signal as _signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise_interrupt(_number: int, _frame: object) -> None:
        raise KeyboardInterrupt

    try:
        previous = _signal.signal(_signal.SIGTERM, _raise_interrupt)
    except (ValueError, OSError):
        # No signal facility here -- a soak still runs, it just cannot be asked
        # to stop gracefully, which is exactly the state it was in before.
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(ValueError, OSError):
            _signal.signal(_signal.SIGTERM, previous)


def _parse_cli_enum(value: str, allowed: tuple[str, ...]) -> str:
    canonical = {item.replace("_", "-"): item for item in allowed if item != "other"}
    parsed = canonical.get(value.strip().lower().replace("_", "-"))
    if parsed is None:
        raise ValueError("identity contains a value outside the safe enum")
    return parsed


def _parse_expected_execution_spec(value: str) -> ExpectedExecution:
    if not isinstance(value, str) or len(value) > 1_024:
        raise ValueError("expected execution identity is too long")
    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 12 or parts[4].lower() not in {"true", "false"}:
        raise ValueError("expected execution must have twelve fixed fields")
    participant = ParticipantBinding(workload_id=parts[0], lease_id=parts[1], cell_id=parts[2])
    identity = ExecutionIdentity(
        capability=CapabilityIdentity(
            execution=_parse_cli_enum(parts[3], SAFE_EXECUTION_KINDS),
            checkpointable=parts[4].lower() == "true",
            coordination=_parse_cli_enum(parts[5], SAFE_COORDINATION_KINDS),
            required_executor=_parse_cli_enum(parts[6], SAFE_EXECUTORS),
        ),
        backend=BackendIdentity(
            capacity_backend=_parse_cli_enum(parts[7], SAFE_CAPACITY_BACKENDS),
            executor=_parse_cli_enum(parts[8], SAFE_EXECUTORS),
        ),
        provider=ProviderIdentity(
            provider=_parse_cli_enum(parts[9], SAFE_PROVIDERS),
            resource_class=_parse_cli_enum(parts[10], SAFE_RESOURCE_CLASSES),
            capacity_class=_parse_cli_enum(parts[11], SAFE_CAPACITY_CLASSES),
        ),
    )
    return ExpectedExecution(participant=participant, identity=identity)


def _write_and_emit_preflight(output: Path | None, codes: tp.Iterable[str]) -> tp.NoReturn:
    receipt = build_preflight_receipt(codes)
    if output is not None:
        write_receipt_atomic(output, receipt)
    JsonRenderer().emit(tp.cast(tp.Any, receipt))
    raise SystemExit(2)


@click.command("soak")
@click.option("--hours", type=str, default="48", show_default=True, help="Finite decimal hours to watch.")
@click.option("--interval-s", type=str, default="60", show_default=True, help="Finite decimal seconds between slots.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Atomically write the receipt as JSON.")
@click.option("--candidate-sha", envvar="EYVAN_CANDIDATE_SHA", help="Full immutable release-candidate SHA.")
@click.option("--source-sha", envvar="EYVAN_SOURCE_SHA", help="Full immutable source commit SHA.")
@click.option("--proof-id", envvar="EYVAN_SOAK_PROOF_ID", help="Opaque sha256 proof/run identifier.")
@click.option(
    "--config",
    "configs",
    type=click.Path(path_type=Path),
    multiple=True,
    help="Config bytes to freeze; repeat for the complete execution snapshot.",
)
@click.option(
    "--config-snapshot-dir",
    type=click.Path(path_type=Path),
    help="Directory in which to materialize the frozen content-addressed snapshot.",
)
@click.option(
    "--expect-execution",
    "expected_execution_specs",
    multiple=True,
    metavar=(
        "WORKLOAD,LEASE,CELL,EXECUTION,BOOL,COORDINATION,REQUIRED_EXECUTOR,"
        "CAPACITY_BACKEND,EXECUTOR,PROVIDER,RESOURCE_CLASS,CAPACITY_CLASS"
    ),
    help="Exact participant and required joint identity; repeat for the P8 mix.",
)
@click.option("--max-rss-growth-bytes", type=str, help="Required bounded RSS peak growth.")
@click.option("--max-fd-growth", type=str, help="Required bounded file-descriptor growth.")
@click.option("--max-database-growth-bytes", type=str, help="Required bounded database growth.")
@click.option("--max-artifact-net-growth-bytes", type=str, help="Required bounded final artifact net growth.")
@click.option("--max-failed-observations", type=str, default="0", show_default=True)
@click.option("--max-consecutive-failed-observations", type=str, default="0", show_default=True)
@click.pass_obj
def soak(
    session: CliSession,
    hours: str,
    interval_s: str,
    output: Path | None,
    candidate_sha: str | None,
    source_sha: str | None,
    proof_id: str | None,
    configs: tuple[Path, ...],
    config_snapshot_dir: Path | None,
    expected_execution_specs: tuple[str, ...],
    max_rss_growth_bytes: str | None,
    max_fd_growth: str | None,
    max_database_growth_bytes: str | None,
    max_artifact_net_growth_bytes: str | None,
    max_failed_observations: str,
    max_consecutive_failed_observations: str,
) -> None:
    """Observe the host-side P8 boundary and emit pass/fail/inconclusive JSON."""

    preflight: list[str] = []
    try:
        hours_decimal = _decimal_value(
            hours,
            name="hours",
            maximum=MAX_SOAK_HOURS,
        )
        duration_decimal = _scale_decimal_exact(hours_decimal, 3600, name="requested_duration_s")
        duration_decimal = _decimal_value(
            duration_decimal,
            name="requested_duration_s",
            maximum=MAX_SOAK_DURATION_S,
        )
        interval_decimal = _decimal_value(interval_s, name="interval_s", maximum=MAX_INTERVAL_S)
        slots = observation_slot_count(duration_decimal, interval_decimal)
        _runtime_grid_parts(duration_decimal, interval_decimal, slots)
        hour_decimal = _decimal_value(3600, name="one_hour_s", maximum=MAX_SOAK_DURATION_S)
        observations_per_hour = max(1, _decimal_quotient_floor(hour_decimal, interval_decimal))
        policy = AvailabilityPolicy(
            max_failed_observations=tp.cast(
                int,
                _bounded_int(max_failed_observations, name="max_failed_observations"),
            ),
            max_consecutive_failed_observations=tp.cast(
                int,
                _bounded_int(
                    max_consecutive_failed_observations,
                    name="max_consecutive_failed_observations",
                ),
            ),
        )
        rss_bound = _bounded_int(max_rss_growth_bytes, name="max_rss_growth_bytes", allow_none=True)
        fd_bound = _bounded_int(max_fd_growth, name="max_fd_growth", allow_none=True)
        database_bound = _bounded_int(
            max_database_growth_bytes,
            name="max_database_growth_bytes",
            allow_none=True,
        )
        artifact_bound = _bounded_int(
            max_artifact_net_growth_bytes,
            name="max_artifact_net_growth_bytes",
            allow_none=True,
        )
    except (ArithmeticError, ValueError):
        _write_and_emit_preflight(output, ("invalid_numeric_input",))

    output_aliases_input = False
    if output is not None:
        try:
            output_identity = output.resolve()
            output_aliases_input = any(path.resolve() == output_identity for path in configs)
        except OSError:
            preflight.append("invalid_configuration_input")
    if output_aliases_input:
        # The output path is the release gate. Even an invalid alias must replace
        # a stale pass with a closed receipt; the run never consumes that input.
        _write_and_emit_preflight(output, ("invalid_output_input_alias",))

    capture: ConfigCapture | None = None
    materialized = False
    if configs:
        try:
            capture = ConfigCapture.capture(configs)
            if config_snapshot_dir is not None:
                capture.materialize(config_snapshot_dir)
                materialized = True
        except Exception:
            preflight.append("invalid_configuration_input")
    else:
        preflight.append("configuration_snapshot_missing")
    if config_snapshot_dir is None:
        preflight.append("configuration_snapshot_missing")

    if not is_opaque_id(proof_id):
        preflight.append("invalid_proof_identity")
    expected: list[ExpectedExecution] = []
    invalid_execution = False
    for spec in expected_execution_specs:
        try:
            expected.append(_parse_expected_execution_spec(spec))
        except ValueError:
            invalid_execution = True
    if invalid_execution or len(expected) != len(set(expected)) or len(expected) > MAX_PARTICIPANTS:
        preflight.append("invalid_expected_execution_input")
    participants = [item.participant for item in expected]
    if len(participants) != len(set(participants)):
        preflight.append("invalid_expected_execution_input")

    if preflight and capture is None:
        _write_and_emit_preflight(output, preflight)
    try:
        inputs = SoakRunInputs.capture(
            candidate_sha=candidate_sha,
            source_sha=source_sha,
            config_sha256=capture.combined_sha256 if capture is not None else None,
            config_file_count=capture.file_count if capture is not None else 0,
            config_snapshot_materialized=materialized,
            proof_id=proof_id,
            requested_duration_s=duration_decimal,
            interval_s=interval_decimal,
            expected_executions=expected,
            max_rss_growth_bytes=rss_bound,
            max_fd_growth=fd_bound,
            max_database_growth_bytes=database_bound,
            max_artifact_net_growth_bytes=artifact_bound,
            availability_policy=policy,
        )
    except (TypeError, ValueError, OverflowError):
        _write_and_emit_preflight(output, (*preflight, "internal_preflight_error"))

    def observe() -> dict[str, object]:
        payload = session.client.request(
            "POST",
            "/api/v1/system/invariants/soak",
            json=tp.cast(tp.Any, inputs.request_manifest()),
        )
        if not isinstance(payload, dict):
            raise ValueError("the scoped invariants route returned an unexpected document")
        return dict(payload)

    def announce(_tick: int, state: SoakState) -> None:
        if state.attempts and state.attempts % observations_per_hour == 0:
            click.echo(
                f"[{state.ticks} successful/{state.attempts} attempted observations] "
                f"{len(state.confirmed)} persistent violation(s), {state.failures} failed",
                err=True,
            )

    receipt = execute_soak(
        observe,
        inputs=inputs,
        output=output,
        on_tick=announce,
        preflight_reason_codes=preflight,
    )
    JsonRenderer().emit(tp.cast(tp.Any, receipt))
    exit_code = receipt.get("exit_code")
    if not isinstance(exit_code, int):
        raise RuntimeError("soak receipt omitted its integer exit code")
    raise SystemExit(exit_code)


def register_soak_commands(main: click.Group) -> None:
    """Attach the soak command to the CLI root."""

    main.add_command(soak)


__all__ = (
    "MAX_TRANSIENT_DETAILS",
    "PERSISTENCE_THRESHOLD",
    "REASON_CODES",
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "AvailabilityPolicy",
    "ConfigCapture",
    "SoakRunInputs",
    "SoakState",
    "build_preflight_receipt",
    "build_soak_receipt",
    "execute_soak",
    "observation_slot_count",
    "register_soak_commands",
    "run_soak",
    "run_soak_for_duration",
    "soak",
    "write_receipt_atomic",
)
