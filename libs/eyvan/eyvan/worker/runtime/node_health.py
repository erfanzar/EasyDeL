# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recognising a broken execution *host* in the output of a failed attempt.

Eyvan has a rich vocabulary for a failed *attempt* and none at all for a bad
*machine*, so a host whose accelerator never initializes is retried until the
workload's failure budget is gone -- and, while that is happening, it holds a
lease, which keeps its cell undeletable and blocks ``eyvan down``.

The idea and the pattern set are taken from marin-community's Iris, whose worker
promotes ``FAILED`` to ``WORKER_FAILED`` on a stderr signature match so the host
is replaced rather than retried (``lib/iris/src/iris/cluster/worker/tpu_health.py``,
as reported in ``.agents/projects/eyvan-target-architecture.md`` W1). This module
is the recognizer only; what to do about a match is the control plane's decision.

Two properties matter more than coverage:

*Evidence of a broken machine, never of a broken workload.* A pattern that a
user's own code could plausibly emit -- or that describes a resource conflict, a
network partition, or an out-of-memory kill -- does not belong in the table, no
matter how often it accompanies a real bad node. A false positive here destroys
a healthy accelerator and hides the user's real bug behind an infinite
provision/destroy loop. ``_DELIBERATELY_ABSENT`` records the ones considered and
rejected, so the next reader argues with the decision rather than re-making it.

*A match is attributable to exactly one host.* One rank's dead TPU must
quarantine that rank's cell and no other, so the per-rank aggregate message the
slice and gang executors already emit is parsed back into ranks here rather than
matched as one blob.
"""

from __future__ import annotations

import json
import os
import re
import typing as tp
from dataclasses import dataclass
from datetime import datetime

from eyvan._internal.compat import UTC, StrEnum

#: Longest excerpt kept as evidence for one match. The durable field that
#: carries it is bounded at 2048 characters and also holds the code, the
#: workload, and the timestamp, so the quoted text has to leave room for them.
EVIDENCE_LIMIT = 320


class HostFaultCode(StrEnum):
    """Stable machine-readable identity for one unrecoverable host fault.

    Durable: the code is written into the cell record that records the fault and
    read back later as proof, so a value here is a wire contract with the future
    and is never renamed in place.
    """

    #: The kernel refused to hand the TPU's IOMMU group to userspace. The group
    #: is established by the host's VFIO/IOMMU configuration at boot, so nothing
    #: a retry does inside the VM changes the answer.
    TPU_IOMMU_GROUP_UNAVAILABLE = "tpu_iommu_group_unavailable"
    #: The VFIO character device backing the accelerator could not be opened.
    #: Same mechanism, reported one layer down; the device node either exists
    #: and is bindable for this boot or it does not.
    TPU_VFIO_DEVICE_UNAVAILABLE = "tpu_vfio_device_unavailable"
    #: The TPU runtime could not bring the chips up. This is the driver's own
    #: verdict about the hardware it found, reached before any user computation
    #: runs, and it does not become true again by asking a second time.
    TPU_SYSTEM_INIT_FAILED = "tpu_system_init_failed"
    #: The same verdict in the phrasing libtpu uses elsewhere. Kept as its own
    #: code rather than folded into the one above so an operator counting faults
    #: is counting messages the host actually emitted.
    TPU_INITIALIZATION_FAILED = "tpu_initialization_failed"
    #: The runtime enumerated the machine and found no accelerator at all. On a
    #: fleet where every cell is provisioned with accelerators attached, a host
    #: that reports none has lost the device, not been asked for the wrong one.
    ACCELERATOR_ABSENT = "accelerator_absent"


def _phrase(literal: str) -> re.Pattern[str]:
    """Compile one fault phrase with a right-hand word boundary.

    Case-sensitive, and anchored on the right by "not followed by another word".
    Both restrictions exist to keep a *prefix* of a fault message from matching a
    different sentence that merely starts the same way: ``Failed to initialize
    TPU system`` is a dead machine, while ``Failed to initialize TPU system
    metrics exporter`` is a telemetry sidecar that failed to start. The first is
    followed by end-of-line or punctuation, the second by a word, and that is the
    whole distinction.

    Program-emitted text keeps its capitalisation, so a lowercased or reworded
    copy of the phrase in a user's own log line is not this signature.
    """

    return re.compile(re.escape(literal) + r"(?![ \t]*[\w-])")


@dataclass(frozen=True, slots=True)
class HostFaultSignature:
    """One recognized fault phrase, its code, and why a retry cannot fix it."""

    code: HostFaultCode
    pattern: re.Pattern[str]
    #: What the machine is telling us, in one sentence. Read by operators in
    #: the fault record, so it describes the hardware state rather than the regex.
    diagnosis: str


#: The table. Order is the tie-break when two phrases match at the same offset,
#: so it is stable and additive: append, never reorder.
HOST_FAULT_SIGNATURES: tuple[HostFaultSignature, ...] = (
    HostFaultSignature(
        code=HostFaultCode.TPU_IOMMU_GROUP_UNAVAILABLE,
        pattern=_phrase("Couldn't open iommu group"),
        diagnosis="the host kernel would not hand this machine's TPU IOMMU group to userspace",
    ),
    HostFaultSignature(
        code=HostFaultCode.TPU_VFIO_DEVICE_UNAVAILABLE,
        pattern=_phrase("open(/dev/vfio"),
        diagnosis="the VFIO device node backing this machine's accelerator could not be opened",
    ),
    HostFaultSignature(
        code=HostFaultCode.TPU_SYSTEM_INIT_FAILED,
        pattern=_phrase("Failed to initialize TPU system"),
        diagnosis="the TPU runtime could not bring this machine's chips up",
    ),
    HostFaultSignature(
        code=HostFaultCode.TPU_INITIALIZATION_FAILED,
        pattern=_phrase("TPU initialization failed"),
        diagnosis="the TPU runtime reported this machine's accelerator initialization as failed",
    ),
    HostFaultSignature(
        code=HostFaultCode.ACCELERATOR_ABSENT,
        pattern=_phrase("No accelerator found"),
        diagnosis="the runtime enumerated this machine and found no accelerator attached",
    ),
)

#: Phrases considered for the table and deliberately left out, with the reason.
#:
#: Every one of them shows up in real bad-node incidents, which is exactly why
#: the list is written down: each also has a reading in which the machine is
#: fine and something else -- the user's code, a peer, a leftover process, the
#: scheduler -- is at fault, and a signature table cannot tell the two apart from
#: the text alone. Promoting any of them would replace healthy hosts.
_DELIBERATELY_ABSENT: tp.Mapping[str, str] = {
    "Device or resource busy": (
        "the usual cause is another process still holding libtpu -- a previous attempt of the "
        "same workload that has not exited yet. Replacing the host hides a leak in the workload."
    ),
    "RESOURCE_EXHAUSTED": (
        "an out-of-memory verdict about the computation that was asked for, not about the "
        "machine that was asked to run it."
    ),
    "Failed to connect to coordinator": (
        "says a peer or the network did not answer. On a cold gang start it means a rank is "
        "still booting, which is W2's deadline, not a broken host."
    ),
    "UNAVAILABLE: Socket closed": (
        "a transport failure. It describes the path to the host, and Eyvan already has a "
        "vocabulary -- LOST, INDETERMINATE -- for not being able to reach one."
    ),
    "libtpu.so already in use": (
        "another process on this host owns the accelerator. That process is usually Eyvan's "
        "own previous unit; killing the machine is the wrong half of the problem."
    ),
    "XlaRuntimeError": (
        "the base class of nearly every JAX runtime error, including ones raised by user "
        "code. Matching it would promote most failing workloads to bad machines."
    ),
}


@dataclass(frozen=True, slots=True)
class HostFault:
    """One matched host fault and the bounded excerpt that evidences it."""

    code: HostFaultCode
    #: The line the phrase was found on, trimmed and truncated. Kept verbatim
    #: rather than summarised: this is the only account of why a machine was
    #: destroyed, and a paraphrase cannot be checked against the host's journal.
    evidence: str

    @property
    def diagnosis(self) -> str:
        """Return the table's sentence about what this fault means."""

        return _SIGNATURES_BY_CODE[self.code].diagnosis


_SIGNATURES_BY_CODE: dict[HostFaultCode, HostFaultSignature] = {
    signature.code: signature for signature in HOST_FAULT_SIGNATURES
}


def _excerpt(text: str, position: int) -> str:
    """Return the bounded line containing one match offset."""

    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    line = text[start:] if end < 0 else text[start:end]
    return line.strip()[:EVIDENCE_LIMIT]


def match_host_fault(text: str | None) -> HostFault | None:
    """Return the first host fault in one blob of execution output, if any.

    "First" is by position, so a machine that logged an IOMMU failure and then a
    cascade of consequences is reported by its cause. Ties -- two phrases at the
    same offset -- fall back to table order, which keeps the answer stable for a
    given input across runs and processes.
    """

    if not text:
        return None
    best: tuple[int, int, HostFaultSignature] | None = None
    for index, signature in enumerate(HOST_FAULT_SIGNATURES):
        found = signature.pattern.search(text)
        if found is None:
            continue
        candidate = (found.start(), index, signature)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    position, _index, signature = best
    return HostFault(code=signature.code, evidence=_excerpt(text, position))


#: The per-rank shape the slice and gang executors render their observations in
#: (``rank 3=failed: ...``, joined by ``"; "``). Parsed rather than re-derived so
#: one rank's fault stays attached to one rank.
_RANK_REPORT = re.compile(r"(?:^|; )rank (\d+)=[A-Za-z_]+(?:: |(?=; )|$)")


def split_rank_reports(message: str | None) -> tuple[tuple[int, str], ...]:
    """Split one aggregate multi-rank observation into ``(rank, text)`` pairs.

    Returns ``()`` for a message that is not in that shape -- a single-process
    executor's diagnostic, or an empty one. That is not a fallback that blames
    everybody: a caller with more than one candidate host and no rank attribution
    has learned nothing about *which* host, and must treat the attribution as
    unknown.

    A rank's own text may itself contain ``; ``, so the pairs are cut at the
    positions of the rank headers rather than by splitting on the separator.
    """

    if not message:
        return ()
    headers = list(_RANK_REPORT.finditer(message))
    if not headers:
        return ()
    reports: list[tuple[int, str]] = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(message)
        reports.append((int(header.group(1)), message[header.end() : end].strip()))
    return tuple(reports)


def host_faults_by_rank(message: str | None) -> dict[int, HostFault]:
    """Return the host fault each rank reported, keyed by rank.

    Only ranks whose own text matches are present. A rank that failed for an
    ordinary reason beside a rank whose host died is not itself evidence of a
    bad machine, and must not be.
    """

    faults: dict[int, HostFault] = {}
    for rank, text in split_rank_reports(message):
        fault = match_host_fault(text)
        if fault is not None:
            faults[rank] = fault
    return faults


#: Audit ``entity_type`` every node-health decision is recorded under.
#:
#: Its own type rather than ``"cell"`` so the ledger of replacement decisions can
#: be read -- and counted against a budget -- without scanning every drain,
#: release and provision the fleet also records against cells.
HOST_FAULT_AUDIT_ENTITY = "host_fault"

#: A host was quarantined and handed to its pool for replacement. This is the
#: countable act: one of these is one replacement charged to a scope's budget.
HOST_FAULT_HOST_QUARANTINED = "cell.host_fault_quarantined"

#: A host fault was recognized and the host was *not* replaced, because the
#: scope's budget was spent or could not be established. Recorded with the same
#: evidence as a replacement: an operator has to be able to see the machine that
#: Eyvan knows is broken and deliberately left in service.
HOST_FAULT_REPLACEMENT_DECLINED = "cell.host_fault_replacement_declined"

#: The durable ``CellRecord.unavailable_reason`` a quarantined host carries.
#:
#: A free-text field rather than a new ``CapacityProviderFailureCode``, and the
#: distinction is deliberate: a host fault is observed from an *execution*, never
#: from the provider's API, so there is no provider code to add and a rolled-back
#: replica must still be able to decode the record.
HOST_FAULT_REASON = "host_fault"

_MARKER = re.compile(
    r"^code=(?P<code>[a-z0-9_]+) workload=(?P<workload>\S+) generation=(?P<generation>\d+) "
    r"at=(?P<at>\S+) evidence=(?P<evidence>.*)$",
    re.S,
)


@dataclass(frozen=True, slots=True)
class HostFaultProof:
    """A host fault read back out of a durable cell record.

    Named *proof* because that is the claim being made when it is used: this
    machine's accelerator runtime died, so a process that needed it is not still
    running on the host. The claim is only as narrow as the fields -- it is
    evidence about ``workload_id`` at ``generation`` and about nothing else.
    """

    code: HostFaultCode
    evidence: str
    workload_id: str
    generation: int
    observed_at: datetime

    @property
    def diagnosis(self) -> str:
        """Return the table's sentence about what this fault means."""

        return _SIGNATURES_BY_CODE[self.code].diagnosis


def format_host_fault_marker(
    fault: HostFault,
    *,
    workload_id: str,
    generation: int,
    observed_at: datetime,
) -> str:
    """Render one fault for ``CellRecord.unavailable_detail``.

    The execution it was observed for travels with it. Without that, a fault
    recorded for one workload would later read as proof about a different one
    that happened to land on the same cell record.
    """

    if observed_at.tzinfo is None:
        raise ValueError("a host fault marker requires a timezone-aware timestamp")
    rendered = (
        f"code={fault.code.value} workload={workload_id} generation={generation} "
        f"at={observed_at.astimezone(UTC).isoformat()} evidence={json.dumps(fault.evidence)}"
    )
    return rendered[:2048]


def parse_host_fault_marker(reason: str | None, detail: str | None) -> HostFaultProof | None:
    """Read a durable host-fault record back, or ``None`` when there is none.

    Total and forgiving: a record written by a newer control plane, truncated, or
    simply not a host fault returns ``None``. Every caller treats ``None`` as *no
    proof*, which is the safe direction -- it costs capacity, never safety.
    """

    if reason != HOST_FAULT_REASON or not detail:
        return None
    found = _MARKER.match(detail)
    if found is None:
        return None
    try:
        code = HostFaultCode(found.group("code"))
        observed_at = datetime.fromisoformat(found.group("at"))
        evidence = json.loads(found.group("evidence"))
    except ValueError:
        return None
    if not isinstance(evidence, str) or observed_at.tzinfo is None:
        return None
    return HostFaultProof(
        code=code,
        evidence=evidence,
        workload_id=found.group("workload"),
        generation=int(found.group("generation")),
        observed_at=observed_at,
    )


#: Shipped defaults for :class:`HostReplacementBudget`. Named constants rather
#: than only field defaults because ``slots=True`` makes the class attribute a
#: descriptor, so ``cls.max_replacements`` is not the value.
_DEFAULT_REPLACEMENTS = 0
_DEFAULT_WINDOW_S = 3600.0


@dataclass(frozen=True, slots=True)
class HostReplacementBudget:
    """How many hosts one pool may replace for node health in one window.

    The bound is the point. A signature match says *this* machine is broken; a
    zone with a bad image says every machine is, and without a ceiling the
    control plane answers by provisioning and destroying capacity forever, at
    full price, with a workload that never runs. Exhaustion is therefore an
    outcome that gets reported, not a condition that gets retried.

    ``max_replacements`` of zero disables promotion entirely, which is the
    shipped default: this changes what happens to a failed attempt, so an
    operator opts in rather than discovering it.
    """

    max_replacements: int = _DEFAULT_REPLACEMENTS
    window_s: float = _DEFAULT_WINDOW_S

    def __post_init__(self) -> None:
        """Reject a budget that cannot bound anything."""

        if self.max_replacements < 0:
            raise ValueError("host replacement budget must not be negative")
        if self.window_s <= 0:
            raise ValueError("host replacement window must be positive")

    @property
    def enabled(self) -> bool:
        """Return whether node-health replacement is turned on at all."""

        return self.max_replacements > 0

    @classmethod
    def from_env(cls) -> HostReplacementBudget:
        """Load the budget from ``EYVAN_HOST_REPLACEMENT_*``.

        A fallback, not the documented path. Both names now live in
        :class:`~eyvan.config.EyvanSettings` so ``docs/install.md`` can describe
        them at all -- that document is checked against the settings loader, so a
        knob read only through :func:`os.getenv` is a knob no operator can be told
        about. This reader stays for a host-side process that has no settings
        object, and both defaults remain "off".
        """

        budget = os.getenv("EYVAN_HOST_REPLACEMENT_BUDGET")
        window = os.getenv("EYVAN_HOST_REPLACEMENT_WINDOW_S")
        return cls(
            max_replacements=_DEFAULT_REPLACEMENTS if budget is None else int(budget),
            window_s=_DEFAULT_WINDOW_S if window is None else float(window),
        )


def host_replacement_budget_from_settings(settings: tp.Any) -> HostReplacementBudget:
    """Build a budget from a settings object, falling back to the environment.

    Typed as ``Any`` deliberately: ``eyvan.worker`` may not import the control
    plane or the SDK, and a settings import would cross that CI-enforced contract
    for one attribute read.
    """

    budget = getattr(settings, "host_replacement_budget", None)
    window = getattr(settings, "host_replacement_window_s", None)
    if budget is None or window is None:
        return HostReplacementBudget.from_env()
    return HostReplacementBudget(max_replacements=int(budget), window_s=float(window))


def host_replacement_scope(pool_id: str | None, provider: str, zone: str) -> str:
    """Return the key a replacement budget is counted against.

    A pool when there is one, because that is the unit an operator sized and the
    unit the autoscaler replaces into. An unpooled cell has no such unit, so its
    provider and location stand in -- a manually declared fleet in one zone still
    shares whatever is wrong with that zone.
    """

    return f"pool:{pool_id}" if pool_id else f"unpooled:{provider}/{zone}"


__all__ = (
    "EVIDENCE_LIMIT",
    "HOST_FAULT_AUDIT_ENTITY",
    "HOST_FAULT_HOST_QUARANTINED",
    "HOST_FAULT_REASON",
    "HOST_FAULT_REPLACEMENT_DECLINED",
    "HOST_FAULT_SIGNATURES",
    "HostFault",
    "HostFaultCode",
    "HostFaultProof",
    "HostFaultSignature",
    "HostReplacementBudget",
    "format_host_fault_marker",
    "host_faults_by_rank",
    "host_replacement_scope",
    "match_host_fault",
    "parse_host_fault_marker",
    "split_rank_reports",
)
