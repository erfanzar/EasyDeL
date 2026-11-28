# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Machine-decided verdicts for the hardware proofs.

``docs/proofs.md`` states each proof's pass criterion in prose, which means the
verdict is whatever a person concludes from a screen of JSON at the end of a
spend they are relieved to have finished. That is the worst possible moment to
ask someone to be sceptical, and it has already gone wrong once: a proof was
started against a stale agent and its output looked fine.

So the criteria live here as pure functions over observations. The plane is
polled by a thin command; every judgement is made by code that can be tested
without hardware, which is the same split that makes the soak runner
changeable.

**Three verdicts, not two.** A proof whose preconditions never occurred is
INCONCLUSIVE, and inconclusive exits non-zero. This is the defect shape this
package keeps producing -- a check that passes because there was nothing to
check -- and on a proof it is the most expensive version of it, because the
answer arrives after the money is spent and reads as evidence. "No cell was ever
created, so nothing contradicted the claim" is not a pass.
"""

from __future__ import annotations

import json
import os
import sys
import typing as tp
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import click

from eyvan._internal.compat import StrEnum
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.receipts import (
    build_proof_receipt,
    criterion_identity,
    file_evidence,
    write_receipt_atomic,
)
from eyvan.sdk.commands.state import CliSession
from eyvan.server.service.subjobs import MAX_LIVE_CHILDREN


class Verdict(StrEnum):
    """The outcome of one check or one proof."""

    PASS = "pass"
    FAIL = "fail"
    #: The observation could not settle the question. Distinct from FAIL because
    #: the responses differ: a failure means the behaviour is wrong, and an
    #: inconclusive means the run did not exercise it and has to be repeated.
    INCONCLUSIVE = "inconclusive"


#: Exit codes, so a scheduled run can tell the two non-passes apart.
EXIT_CODES = {Verdict.PASS: 0, Verdict.FAIL: 1, Verdict.INCONCLUSIVE: 2}

#: The proof was refused before it ran, so it has no verdict at all. Distinct
#: from all three of them: a caller that read a refused run as INCONCLUSIVE
#: would retry the spend, and one that read it as FAIL would report a defect
#: nothing observed.
EXIT_NOT_RUN = 3


@dataclass(frozen=True, slots=True)
class Check:
    """One stated criterion and what the observation said about it."""

    name: str
    verdict: Verdict
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "verdict": self.verdict.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ProofResult:
    """Every check for one proof, and the verdict they combine to."""

    proof: str
    checks: tuple[Check, ...]

    @property
    def verdict(self) -> Verdict:
        """Return the worst check's verdict.

        A failure outranks an inconclusive: if one criterion is definitively
        wrong, "we could not tell" about another does not soften it.
        """

        if any(check.verdict is Verdict.FAIL for check in self.checks):
            return Verdict.FAIL
        if not self.checks or any(check.verdict is Verdict.INCONCLUSIVE for check in self.checks):
            return Verdict.INCONCLUSIVE
        return Verdict.PASS

    def as_dict(self) -> dict[str, object]:
        return {
            "proof": self.proof,
            "verdict": self.verdict.value,
            "checks": [check.as_dict() for check in self.checks],
        }


def _passed(name: str, detail: str) -> Check:
    return Check(name, Verdict.PASS, detail)


def _failed(name: str, detail: str) -> Check:
    return Check(name, Verdict.FAIL, detail)


def _unknown(name: str, detail: str) -> Check:
    return Check(name, Verdict.INCONCLUSIVE, detail)


# --------------------------------------------------------------------------- P1


def evaluate_catalog_authority(
    pools: tp.Sequence[dict],
    cells: tp.Sequence[dict],
    *,
    accelerator_type: str,
    catalog_hosts: int | None,
) -> ProofResult:
    """P1 -- the catalog, not the provider response, decides hosts per slice.

    A provider returning one host of two yields a silently half-sized JAX
    cluster: every rank starts, the mesh is wrong, and the job runs at half
    throughput while reporting success. Only real hardware produces that case,
    which is why it is a proof rather than a test.
    """

    checks: list[Check] = []
    matching = [pool for pool in pools if pool.get("accelerator_type") == accelerator_type]
    if not matching:
        return ProofResult(
            "P1",
            (
                _unknown(
                    "pool-exists",
                    f"no pool declares accelerator_type {accelerator_type!r}; the fleet never applied",
                ),
            ),
        )
    checks.append(_passed("pool-exists", f"{len(matching)} pool(s) for {accelerator_type}"))

    if catalog_hosts is None:
        checks.append(
            _unknown(
                "catalog-records-hosts",
                f"the catalog has no host count for {accelerator_type}, so it cannot be the authority for one",
            )
        )
    else:
        checks.append(_passed("catalog-records-hosts", f"catalog says {catalog_hosts} host(s)"))

    pool_ids = {pool["id"] for pool in matching}
    members = [cell for cell in cells if cell.get("pool_id") in pool_ids and cell.get("state") != "deleted"]
    if not members:
        checks.append(_unknown("cells-exist", "the pool holds no live cell, so no observation was made"))
        return ProofResult("P1", tuple(checks))
    checks.append(_passed("cells-exist", f"{len(members)} live cell(s)"))

    unobserved = [cell["id"] for cell in members if cell.get("observed_tpu_vms") is None]
    unsized = [cell["id"] for cell in members if not _is_positive_int(cell.get("worker_count"))]
    if unobserved:
        checks.append(
            _unknown(
                "hosts-observed",
                f"{len(unobserved)} cell(s) never reported observed_tpu_vms: {', '.join(sorted(unobserved)[:3])}",
            )
        )
    elif unsized:
        # Without the slice count the observation cannot be interpreted at all:
        # four VMs is right for two slices and wrong for one, and guessing which
        # would turn this criterion into a coin toss dressed as a verdict.
        checks.append(
            _unknown(
                "hosts-observed",
                f"{len(unsized)} cell(s) report no usable worker_count, so hosts per slice cannot be derived: "
                + ", ".join(sorted(unsized)[:3]),
            )
        )
    elif catalog_hosts is not None:
        # The catalog states hosts per *slice*; a multislice cell holds
        # ``worker_count`` of them under one allocation. Comparing the total
        # against the per-slice number reported a live 2x v5p-16 as a defect --
        # four VMs where the catalog says two -- which is the shape of failure
        # this file exists to keep out of a spend's final screen.
        wrong = {
            cell["id"]: f"{cell['observed_tpu_vms']} VM(s) across {cell['worker_count']} slice(s)"
            for cell in members
            if cell["observed_tpu_vms"] != catalog_hosts * cell["worker_count"]
        }
        if wrong:
            checks.append(
                _failed(
                    "hosts-match-catalog",
                    f"expected {catalog_hosts} host(s) per slice; observed {wrong}",
                )
            )
        else:
            slices = sum(cell["worker_count"] for cell in members)
            checks.append(
                _passed(
                    "hosts-match-catalog",
                    f"every cell reports {catalog_hosts} host(s) per slice across {slices} slice(s)",
                )
            )
    return ProofResult("P1", tuple(checks))


def _is_positive_int(value: object) -> bool:
    """Return whether ``value`` is a slice count this criterion can divide by."""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_number(value: object) -> bool:
    """Return whether ``value`` is a positive count that may have decayed.

    Separate from :func:`_is_positive_int` because the availability counters are
    time-decayed floats -- one interruption reads as 1.9604, not 1 -- so an
    int-only check would treat every one of them as absent.
    """

    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


# --------------------------------------------------------------------------- P5


def _instant(value: object) -> datetime | None:
    """Parse one API timestamp, or return ``None`` for anything else."""

    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def evaluate_spot_preemption(workload: dict, cells: tp.Sequence[dict], pools: tp.Sequence[dict]) -> ProofResult:
    """P5 -- a reclaimed Spot slice is a preemption, not a provider error.

    The distinguishing case, and the reason the run sets ``--max-retries 0``:
    under the old shared budget the workload would have failed outright. A
    surviving workload whose *preemption* budget moved and whose *failure*
    budget did not establishes budget charging. Current global cell inventory
    cannot establish run-linked provider classification or cooldown history,
    so those dimensions remain INCONCLUSIVE rather than qualifying this run
    from another workload's evidence.
    """

    checks: list[Check] = []
    preemptions = int(workload.get("preemption_count", 0))
    losses = int(workload.get("capacity_loss_count", 0))

    if preemptions == 0 and losses == 0:
        # The common outcome of a short run: GCP simply never reclaimed the
        # slice. Reporting that as a pass would turn "we were not preempted"
        # into "preemption is classified correctly".
        return ProofResult(
            "P5",
            (
                _unknown(
                    "preemption-observed",
                    "the slice was never reclaimed; nothing about classification was exercised",
                ),
            ),
        )
    checks.append(_passed("preemption-observed", f"preemption_count={preemptions}, capacity_loss_count={losses}"))

    if preemptions == 0:
        checks.append(
            _failed(
                "charged-to-the-preemption-budget",
                f"capacity_loss_count moved to {losses} and preemption_count stayed at 0, "
                "so the reclamation was charged to the failure budget",
            )
        )
    elif losses:
        checks.append(
            _failed(
                "charged-to-the-preemption-budget",
                f"both budgets moved: preemption_count={preemptions}, capacity_loss_count={losses}",
            )
        )
    else:
        checks.append(_passed("charged-to-the-preemption-budget", "only preemption_count moved"))

    # These payloads are current, global inventory, not workload-linked event
    # history. A pool label or updated_at after submission cannot prove which
    # workload experienced a cell's provider event; interruptions also include
    # earlier runs. Neither a sibling's preempted code nor its cooldown can
    # qualify (or fail) this run. Preserve the public input signature, but leave
    # these dimensions unknown until an authoritative run-linked observation is
    # available. In particular, a cleared code after recovery is not a failure.
    checks.append(
        _unknown(
            "classified-as-preempted",
            "cell inventory does not link a provider classification to this workload's reclamation",
        )
    )
    checks.append(
        _unknown("no-stockout-cooldown", "cell inventory does not link cooldown evidence to this workload's reclamation")
    )

    state = workload.get("state")
    if state in {"running", "succeeded"}:
        checks.append(_passed("workload-survived", f"workload is {state}"))
    elif state == "failed":
        checks.append(_failed("workload-survived", "the workload failed rather than retrying on its preemption budget"))
    else:
        checks.append(_unknown("workload-survived", f"workload is {state}, which settles nothing yet"))
    return ProofResult("P5", tuple(checks))


# --------------------------------------------------------------------------- P6


def evaluate_warm_window(inside: dict, outside: dict, billing: dict, *, pool_id: str) -> ProofResult:
    """P6 -- a window holds a floor through its hours and releases after.

    Two observations, either side of the window's own boundary. One observation
    cannot settle it: a floor that is held and a floor that is *always* held look
    identical from inside the window, and the second is a pool that never
    scales to zero.
    """

    checks: list[Check] = []

    def _pool(report: dict) -> dict | None:
        return next((entry for entry in report.get("pools", []) if entry.get("pool_id") == pool_id), None)

    during, after = _pool(inside), _pool(outside)
    if during is None or after is None:
        return ProofResult(
            "P6",
            (_unknown("pool-observed", f"pool {pool_id!r} is absent from one of the two warmth reports"),),
        )

    if not during.get("window_active"):
        checks.append(
            _unknown(
                "window-was-active",
                "the first observation was taken outside the window, so it says nothing about holding one",
            )
        )
    else:
        checks.append(_passed("window-was-active", f"floor {during.get('floor')} in force"))

    held = int(during.get("warm_cells", 0))
    floor = int(during.get("floor", 0))
    if floor <= 0:
        checks.append(_unknown("floor-declared", "the window raised no floor, so nothing was promised"))
    elif held < floor:
        checks.append(_failed("floor-held", f"{held} of {floor} cell(s) held while the window was active"))
    else:
        checks.append(_passed("floor-held", f"{held} of {floor} cell(s) held"))

    if after.get("window_active"):
        checks.append(_unknown("window-closed", "the second observation was still inside the window"))
    elif floor <= 0:
        # A floor that was never raised cannot be observed being released. Read
        # as a pass, this reported the window working on a run where the window
        # did nothing -- and it is the *cheap* half to satisfy, because a pool
        # sitting at zero all day satisfies it every time.
        checks.append(_unknown("floor-released", "no floor was raised, so its release says nothing"))
    elif int(after.get("floor", 0)) >= floor:
        checks.append(
            _failed(
                "floor-released",
                f"the floor stayed at {after.get('floor')} after the window closed",
            )
        )
    else:
        checks.append(_passed("floor-released", f"the floor fell to {after.get('floor')} after the window closed"))

    # The cost half. A window that holds capacity and bills nothing is a fleet
    # whose pool carries no price, and reporting that as a passing cost check is
    # how an unpriced fleet reads as a free one.
    idle_seconds = float(billing.get("idle_seconds", 0))
    idle_cost = float(billing.get("idle_cost", 0))
    unpriced = int(billing.get("unpriced_cells", 0))
    if idle_seconds <= 0:
        checks.append(_unknown("idle-cost-accrued", "no idle time was billed, so the warm period was not measured"))
    elif idle_cost > 0:
        # Cost was computed, so the failure this check exists for -- a fleet that
        # reads as free because nothing carries a price -- did not occur. Any
        # unpriced cells are named anyway: on a pool that has been rebuilt they
        # are the *released* incarnations, and they drag the total down without
        # explaining themselves.
        detail = f"{idle_seconds:.0f}s idle, {idle_cost:.4f} accrued"
        checks.append(
            _passed("idle-cost-accrued", detail if not unpriced else f"{detail} ({unpriced} released cell(s) unpriced)")
        )
    elif unpriced:
        checks.append(
            _unknown(
                "idle-cost-accrued",
                f"{unpriced} cell(s) carry no price and nothing accrued, "
                "so cost cannot distinguish a free pool from an unpriced one",
            )
        )
    else:
        checks.append(_failed("idle-cost-accrued", f"{idle_seconds:.0f}s idle on a priced pool accrued no cost"))
    return ProofResult("P6", tuple(checks))


# --------------------------------------------------------------------------- P4


def _decoded_strings(value: object) -> tp.Iterator[str]:
    """Inspect decoded leaves and keys, not an escaped debug representation."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _decoded_strings(key)
            yield from _decoded_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _decoded_strings(item)


def evaluate_secret_delivery(
    events: tp.Sequence[dict],
    workload: dict,
    logs: str,
    *,
    secret_value: str,
) -> ProofResult:
    """P4 -- the value is delivered, used, and recorded nowhere.

    The leak check is the reason this is code rather than a person grepping. A
    human checks the log, remembers to check the record, and does not think to
    check the audit payloads -- which is the one place a well-meaning "context"
    field would put it.
    """

    checks: list[Check] = []
    if not secret_value:
        return ProofResult(
            "P4",
            (_unknown("value-known", "no secret value was supplied, so absence cannot be checked"),),
        )

    issued = [event for event in events if event.get("action") == "secret.issued"]
    read = [event for event in events if event.get("action") == "secret.read"]
    if not issued and not read:
        return ProofResult(
            "P4",
            (_unknown("secret-was-used", "the ledger shows no secret event; the mount was never exercised"),),
        )

    # One issuance grants all mounts, but each mount has its own redemption.
    # Both records carry attempt and generation; reads do not carry the outbox
    # identity, so duplicate issuances in the same scope cannot be disambiguated
    # and must not qualify. Legacy nameless pairs remain comparable only with
    # other nameless pairs, never as wildcards for named mounts.
    def scope(event: dict) -> tuple[object, object]:
        payload = event.get("payload") or {}
        return payload.get("attempt"), payload.get("generation")

    issuances: dict[tuple[object, object], int] = {}
    expected: set[tuple[object, object, object]] = set()
    for event in issued:
        identity = scope(event)
        issuances[identity] = issuances.get(identity, 0) + 1
        names = (event.get("payload") or {}).get("names")
        for mount in names if names is not None else (None,):
            expected.add((*identity, mount))
    repeated = sorted(str(identity[0]) for identity, count in issuances.items() if count != 1)
    if not issued:
        checks.append(_failed("one-issuance", "expected exactly one secret.issued per attempt; found none"))
    elif repeated:
        checks.append(
            _failed(
                "one-issuance",
                f"expected exactly one secret.issued per attempt; attempt(s) {', '.join(repeated)} had more",
            )
        )
    else:
        checks.append(_passed("one-issuance", f"exactly one secret.issued in each of {len(issuances)} attempt(s)"))

    redemptions: dict[tuple[object, object, object], int] = {}
    for event in read:
        key = (*scope(event), (event.get("payload") or {}).get("name"))
        redemptions[key] = redemptions.get(key, 0) + 1
    duplicates = sorted(str(key[0]) for key, count in redemptions.items() if count != 1)
    missing = expected - redemptions.keys()
    if duplicates:
        checks.append(
            _failed(
                "one-read", f"expected exactly one secret.read per mount; attempt(s) {', '.join(duplicates)} had more"
            )
        )
    elif missing:
        detail = f"{len(missing)} issued mount(s) have no secret.read in their attempt and generation"
        unfinished = workload.get("state") in {
            "pending",
            "admitted",
            "leased",
            "dispatching",
            "running",
            "checkpointing",
            "cancelling",
        }
        checks.append(_unknown("one-read", detail) if unfinished else _failed("one-read", detail))
    elif not read or not expected:
        checks.append(_failed("one-read", "expected exactly one secret.read per issued mount; found none"))
    else:
        checks.append(_passed("one-read", f"exactly one secret.read per mount in each of {len(issuances)} attempt(s)"))

    orphans = redemptions.keys() - expected
    if orphans:
        checks.append(
            _failed(
                "read-matches-an-issuance",
                f"{len(orphans)} read(s) with no issuance for that mount, attempt and generation",
            )
        )
    else:
        checks.append(_passed("read-matches-an-issuance", "every read matches an issued mount, attempt and generation"))
    read_attempts = {scope(event)[0] for event in read}
    if len(read_attempts) > 1 and not missing and not orphans and not duplicates:
        checks.append(_passed("redelivered", f"the value was issued and read afresh in {len(read_attempts)} attempts"))

    leaks = []
    if secret_value in logs:
        leaks.append("the workload log")
    if any(secret_value in value for value in _decoded_strings(workload)):
        leaks.append("the workload record")
    if any(secret_value in value for value in _decoded_strings(list(events))):
        leaks.append("an audit payload")
    if leaks:
        checks.append(_failed("value-never-recorded", f"the secret value appears in {', '.join(leaks)}"))
    elif (workload.get("spec") or {}).get("secret_mounts"):
        # Derived from the spec rather than by matching the plane's wording: a
        # workload that mounts secrets is served no log output at all, so the
        # log grep above is vacuous and passing quietly on it would credit an
        # inspection that did not happen. The stronger fact is worth stating --
        # a value cannot leak through output the plane refuses to serve -- but
        # it is a different fact from "the job did not print it".
        checks.append(
            _passed(
                "value-never-recorded",
                "the value appears in no record or audit payload; the plane serves no log output for a "
                "secret-bearing workload, so the log half holds by suppression rather than by inspection",
            )
        )
    else:
        checks.append(_passed("value-never-recorded", "the value appears in no log, record or audit payload"))
    return ProofResult("P4", tuple(checks))


# --------------------------------------------------------------------------- P7


def evaluate_kubernetes_preconditions(observations: tp.Mapping[str, str | None]) -> ProofResult:
    """P7 -- the Kueue precondition, decided against a real API server.

    ``observations`` maps a case name to the failure code the driver produced,
    or ``None`` when it allowed the configuration.

    Both halves matter and they fail differently. Refusing everything is a
    driver nobody can use with Kueue at all; refusing nothing is the hazard the
    check exists for -- Kueue eviction destroys a pod with no checkpoint hook,
    bypassing Eyvan's whole preemption protocol, and the only trace is a
    training job that lost hours with no ledger entry explaining why.
    """

    expected: tuple[tuple[str, str | None, str], ...] = (
        ("no-queue", None, "a pool naming no queue has nothing to verify"),
        ("safe-queue", None, "preemption Never/Never cannot evict Eyvan's work"),
        ("unsafe-queue", "configuration", "preemption that could evict must be refused"),
        ("missing-queue", "configuration", "a queue that does not exist suspends every Job forever"),
    )
    checks: list[Check] = []
    for case, want, why in expected:
        if case not in observations:
            checks.append(_unknown(case, f"the run did not exercise this case ({why})"))
            continue
        got = observations[case]
        if got == want:
            checks.append(_passed(case, "allowed" if want is None else f"refused with {want!r}"))
        elif want is None:
            checks.append(_failed(case, f"refused with {got!r}; {why}"))
        else:
            checks.append(_failed(case, f"allowed; {why}"))
    return ProofResult("P7", tuple(checks))


# --------------------------------------------------------------------------- P3


def evaluate_control_plane_ran_it(workload: dict, cells: tp.Sequence[dict]) -> ProofResult:
    """P9 -- Eyvan scheduled the run, rather than a human doing it by hand.

    The proof this exists to prevent is a whole day of accelerator time spent
    driving GCP with ``gcloud queued-resources`` and startup scripts. It proved
    the workload framework and said nothing about the control plane, because
    Eyvan never admitted the job, never held its lease, never observed its
    preemption -- and the resulting green run read exactly like evidence.

    Every criterion here is written against the vacuous-pass class: each demands
    a positive witness that the plane *did* something, so a fleet that never ran
    anything reports unknown rather than pass. A criterion that a fresh, empty
    control plane satisfies is not a criterion.
    """

    checks: list[Check] = []

    state = workload.get("state")
    if state != "succeeded":
        return ProofResult(
            "P9",
            (
                _unknown(
                    "workload-succeeded",
                    f"workload is {state!r}, so nothing can be concluded about how it was placed",
                ),
            ),
        )
    checks.append(_passed("workload-succeeded", f"workload {workload.get('id')!r} reached succeeded"))

    # A lease is the plane's claim on capacity. Without one the workload may
    # have run, but not because Eyvan placed it -- which is the whole question.
    labels = workload.get("required_labels") or {}
    pool_id = labels.get("eyvan_pool")
    if not pool_id:
        checks.append(
            _failed("bound-to-a-pool", "the workload carries no eyvan_pool label; it was not placed by a fleet")
        )
    else:
        checks.append(_passed("bound-to-a-pool", f"placement was bound to pool {pool_id!r}"))

    # The cells that carried it must be the plane's own, named by the pool it
    # was bound to. A hand-made TPU has no cell record at all, so this is the
    # criterion a bypass cannot fake.
    carried = [cell for cell in cells if cell.get("pool_id") == pool_id]
    if not pool_id:
        checks.append(_unknown("ran-on-a-fleet-cell", "no pool to attribute cells to"))
    elif not carried:
        checks.append(
            _failed(
                "ran-on-a-fleet-cell",
                f"no cell belongs to {pool_id!r}; the work did not run on capacity this plane created",
            )
        )
    else:
        checks.append(_passed("ran-on-a-fleet-cell", f"{len(carried)} cell(s) in {pool_id!r} carried the workload"))

    # Worker count is what makes a multislice claim checkable. One worker is a
    # single slice, whatever the run was called.
    workers = workload.get("worker_count")
    if not _is_positive_int(workers):
        checks.append(_unknown("worker-count-observed", "the workload records no worker count"))
    else:
        checks.append(_passed("worker-count-observed", f"the plane dispatched {workers} worker(s)"))

    return ProofResult("P9", tuple(checks))


def evaluate_in_job_submission(
    parent: dict,
    children: tp.Sequence[dict],
    *,
    expected_children: int,
    late_submit_status: int | None,
    ceiling_status: int | None,
) -> ProofResult:
    """P3 -- a running job submits children under its owner's identity.

    ``created_by`` is the criterion that matters most and the one most easily
    got wrong in a way nothing notices. A child attributed to
    ``workload:<id>`` competes as a fresh per-principal fair-share leaf and
    against a fresh quota scope, so a self-submitting launcher would escape the
    limits its owner is subject to simply by fanning out. The job tree would look
    correct the whole time.
    """

    checks: list[Check] = []
    if not children:
        return ProofResult(
            "P3",
            (_unknown("children-created", "the parent submitted no children; nothing about the capability was used"),),
        )
    if len(children) == expected_children:
        checks.append(_passed("children-created", f"{len(children)} child(ren) accepted"))
    else:
        checks.append(_failed("children-created", f"expected {expected_children}, found {len(children)}"))

    owner = parent.get("created_by")
    synthetic = [child["id"] for child in children if child.get("created_by") != owner]
    if synthetic:
        checks.append(
            _failed(
                "children-carry-the-owner",
                f"{len(synthetic)} child(ren) not attributed to {owner!r}: {', '.join(sorted(synthetic)[:3])}",
            )
        )
    else:
        checks.append(_passed("children-carry-the-owner", f"every child is attributed to {owner!r}"))

    orphans = [child["id"] for child in children if child.get("parent_workload_id") != parent.get("id")]
    scope = {(child.get("org_id"), child.get("project_id")) for child in children}
    if orphans:
        checks.append(_failed("children-are-parented", f"{len(orphans)} child(ren) name a different parent"))
    elif scope != {(parent.get("org_id"), parent.get("project_id"))}:
        checks.append(_failed("children-are-parented", f"children landed in {sorted(scope)}, not the parent's scope"))
    else:
        checks.append(_passed("children-are-parented", "every child is parented and in the parent's scope"))

    if late_submit_status is None:
        checks.append(_unknown("capability-expires-with-the-parent", "reuse after termination was not attempted"))
    elif late_submit_status == 409:
        checks.append(_passed("capability-expires-with-the-parent", "refused with 409 once the parent was terminal"))
    else:
        checks.append(
            _failed(
                "capability-expires-with-the-parent",
                f"a terminated parent still accepted a child (HTTP {late_submit_status})",
            )
        )

    if ceiling_status is None:
        checks.append(
            _unknown(
                "width-ceiling",
                f"the live-children ceiling was not reached; it takes {MAX_LIVE_CHILDREN} concurrent children",
            )
        )
    elif ceiling_status == 409:
        checks.append(_passed("width-ceiling", "the ceiling refused with 409"))
    else:
        checks.append(_failed("width-ceiling", f"the ceiling did not refuse (HTTP {ceiling_status})"))
    return ProofResult("P3", tuple(checks))


# --------------------------------------------------------------------------- P2


def evaluate_structured_logs(page: dict, *, expected_ranks: int) -> ProofResult:
    """P2 -- per-rank structured logs, and an unreachable rank reported in place.

    The failure this exists to prevent is a missing rank that looks like silence.
    A fan-out returning only the hosts it could reach shows a complete-looking
    log for a half-dead slice, and the reader has no way to tell.

    ``expected_ranks`` is passed rather than inferred: inferring it from the page
    would make "one rank answered" indistinguishable from "one rank exists",
    which is precisely the confusion being tested for.
    """

    checks: list[Check] = []
    records = page.get("records") or []
    if page.get("unavailable") and not records:
        return ProofResult(
            "P2",
            (
                _unknown(
                    "structured-records",
                    f"the backend returned no structured page: {str(page.get('detail'))[:120]}",
                ),
            ),
        )
    if not records:
        return ProofResult("P2", (_unknown("structured-records", "the page was empty; nothing was attributed"),))
    checks.append(_passed("structured-records", f"{len(records)} record(s)"))

    # Every producing record must preserve its metadata. One healthy host must
    # not qualify another host's message-only output. Unavailable ranks live in
    # truncated_ranks, not in synthetic records needing these fields.
    required: tuple[tuple[str, tp.Callable[[dict], bool], str], ...] = (
        ("records-carry-a-timestamp", lambda record: _instant(record.get("timestamp")) is not None, "usable timestamps"),
        (
            "records-carry-a-stream",
            lambda record: record.get("stream") in {"stdout", "stderr"},
            "stdout/stderr attribution",
        ),
        (
            "records-are-resumable",
            lambda record: isinstance(record.get("cursor"), str) and bool(record["cursor"]),
            "backend cursors",
        ),
    )
    for name, valid, detail in required:
        missing = [record for record in records if not valid(record)]
        if missing:
            ranks = ", ".join(sorted({str(record.get("rank")) for record in missing}))
            checks.append(_failed(name, f"{len(missing)} record(s) on rank(s) {ranks} lack {detail}"))
        else:
            checks.append(_passed(name, f"all {len(records)} records preserve {detail}"))

    seen = {r.get("rank") for r in records} | set(page.get("truncated_ranks") or [])
    if expected_ranks <= 1:
        checks.append(
            _unknown(
                "per-rank-attribution",
                "a single-host run cannot show attribution; the failure mode needs at least two ranks",
            )
        )
    elif len(seen) >= expected_ranks:
        checks.append(_passed("per-rank-attribution", f"{len(seen)} rank(s) accounted for"))
    else:
        checks.append(
            _failed("per-rank-attribution", f"only {len(seen)} of {expected_ranks} rank(s) appear anywhere in the page")
        )

    truncated = page.get("truncated_ranks") or []
    if expected_ranks <= 1:
        checks.append(_unknown("unreachable-rank-reported-in-place", "no second rank to lose"))
    elif truncated:
        checks.append(_passed("unreachable-rank-reported-in-place", f"rank(s) {sorted(truncated)} reported unavailable"))
    else:
        checks.append(
            _unknown(
                "unreachable-rank-reported-in-place",
                "no rank was unreachable during this run, so the reporting path was not exercised",
            )
        )
    return ProofResult("P2", tuple(checks))


# --------------------------------------------------------------------------- the command


class _ProofGroup(click.Group):
    """A group whose refusals never wear a verdict's exit code.

    Click answers a mistyped option with exit 2, which is this group's code for
    INCONCLUSIVE. A scheduled run would then record "the behaviour was not
    exercised, repeat the spend" for a command that can never work as typed.
    Refusals exit :data:`EXIT_NOT_RUN` instead, and only an evaluated criterion
    produces 0, 1, or 2.
    """

    def invoke(self, ctx: click.Context) -> object:
        try:
            return super().invoke(ctx)
        except click.UsageError as error:
            error.show()
            raise SystemExit(EXIT_NOT_RUN) from error


@click.group("proof", cls=_ProofGroup)
def proof() -> None:
    """Decide a hardware proof's stated criterion against the live plane.

    Every criterion in ``docs/proofs.md`` is evaluated here rather than read off
    a screen at the end of a spend. Exits 0 on pass, 1 on failure, 2 when the
    run did not exercise the behaviour -- because "nothing contradicted it" is
    not evidence, and it is the answer most likely to be mistaken for one -- and
    3 when the proof was refused before it ran and so has no verdict.

    ``--receipt`` writes what a later reader needs to tell whether the run
    qualifies the build in front of them: the verdict, the digest of the
    criterion that decided it, the builds of the client and the plane, and the
    dimensions the run left unestablished. ``eyvan proof verify`` reads one back.
    """


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _refuse(message: str) -> tp.NoReturn:
    """Stop before the proof runs, without minting a verdict for it."""

    click.echo(f"eyvan proof: {message}", err=True)
    raise SystemExit(EXIT_NOT_RUN)


@dataclass
class _ProofRun:
    """One invocation: what it is judging, and what it observed judging it.

    Exists so a receipt can name the criterion that produced the verdict and the
    plane it was produced against. Both are known at the call site and nowhere
    else -- the evaluator is a plain function, and the observations are whatever
    that command happened to fetch.
    """

    proof: str
    evaluator: tp.Callable[..., ProofResult]
    session: CliSession | None = None
    receipt: str | None = None
    evidence: str | None = None
    #: Values that must not be written to the evidence file even though the run
    #: read them. The verdict records whether a secret leaked; copying the leaked
    #: value into a second durable file would widen the leak to report it.
    secrets: tuple[str, ...] = ()
    started_at: str = field(default_factory=_utc_now)
    observations: dict[str, object] = field(default_factory=dict)

    def seen(self, name: str, value: tp.Any) -> tp.Any:
        """Record one fetched document and return it unchanged."""

        self.observations[name] = value
        return value

    def preflight(self) -> None:
        """Refuse an output path this run could not record its result to.

        Checked before the plane is polled, so a run that cannot record its
        result is stopped while it is still free rather than after the spend.

        A directory that does not exist is refused here for the same reason the
        taken path is. Left to the write, it surfaces as a ``FileNotFoundError``
        *after* the verdict is decided, and the non-zero exit that escapes is
        indistinguishable from a failing proof -- so a mistyped ``--evidence``
        path reports that the build failed a criterion it actually passed. That
        is the ambiguity a refusal exit code exists to prevent.
        """

        if self.receipt is not None and self.evidence is not None:
            if Path(self.receipt).resolve() == Path(self.evidence).resolve():
                _refuse("receipt and evidence paths must be distinct; raw observations must be retained separately")
        for label, path in (("receipt", self.receipt), ("evidence", self.evidence)):
            if path is None:
                continue
            if Path(path).exists():
                _refuse(f"{label} path {path} already exists; a receipt is written once, never over another run")
            parent = Path(path).parent
            if not parent.is_dir():
                _refuse(f"{label} path {path} is not writable: {parent} is not a directory")

    def _plane_build(self) -> tuple[dict[str, object] | None, str]:
        """Ask the plane what code it is running.

        A plane that predates the route answers 404, which is recorded as an
        unknown build rather than treated as an error: the proof still ran, and
        the receipt says exactly which dimension it could not establish.
        """

        if self.session is None:
            return None, "this proof does not poll a control plane"
        try:
            document = self.session.client.request("GET", "/api/v1/build")
        except click.ClickException as error:
            return None, f"the plane did not report a build: {error.message}"
        except OSError as error:
            return None, f"the plane could not be reached for its build: {error}"
        return (document, "") if isinstance(document, dict) else (None, "the plane returned a malformed build document")

    def _fleet_builds(self) -> dict[str, object] | None:
        """Ask the plane which build each of its agents runs.

        The dimension every receipt used to name and none could close. A plane
        that predates the route, or one whose agents cannot be reached, leaves
        it open exactly as before -- the receipt then says so rather than
        implying the check was made.
        """

        if self.session is None:
            return None
        try:
            document = self.session.client.request("GET", "/api/v1/capacity/builds")
        except click.ClickException:
            return None
        except OSError:
            return None
        return document if isinstance(document, dict) else None

    def _write_evidence(self) -> dict[str, object] | None:
        if self.evidence is None or not self.observations:
            return None
        from eyvan.secrets import redact_secret_material

        def redact(value: object) -> object:
            # Copy decoded leaves before serialization escapes their characters.
            # The fetched observations stay intact for evaluation.
            if isinstance(value, dict):
                return {
                    redact_secret_material(str(key), secrets=self.secrets): redact(item) for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [redact(item) for item in value]
            if value is None or isinstance(value, (bool, int, float)):
                return value
            return redact_secret_material(str(value), secrets=self.secrets)

        payload = json.dumps(redact(self.observations), indent=2, sort_keys=True, ensure_ascii=False)
        # Opened owner-only rather than narrowed afterwards: a plane document
        # carries fields this run has no way to exhaustively redact, and the
        # window between creating it under the umask and chmod-ing it is exactly
        # long enough for another process on a shared host to read it.
        descriptor = os.open(self.evidence, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return file_evidence(self.evidence)

    def record(self, result: ProofResult) -> None:
        """Write this run's evidence and receipt, if either was asked for."""

        evidence = self._write_evidence()
        if self.receipt is None:
            return
        plane_build, plane_detail = self._plane_build()
        write_receipt_atomic(
            self.receipt,
            build_proof_receipt(
                proof=result.proof,
                verdict=result.verdict.value,
                exit_status=EXIT_CODES[result.verdict],
                checks=[check.as_dict() for check in result.checks],
                criterion=criterion_identity(self.evaluator),
                plane_url=None if self.session is None else self.session.base_url,
                plane_build=plane_build,
                plane_build_detail=plane_detail,
                started_at=self.started_at,
                completed_at=_utc_now(),
                command=sys.argv,
                evidence=evidence,
                fleet_builds=self._fleet_builds(),
            ),
            replace_existing=False,
        )


def _receipt_options(command: tp.Callable[..., None]) -> tp.Callable[..., None]:
    """Attach the two output options every proof shares."""

    command = click.option(
        "--evidence",
        type=click.Path(dir_okay=False, writable=True),
        help="Write the raw documents this proof read, redacted of secret material, to this path.",
    )(command)
    return click.option(
        "--receipt",
        type=click.Path(dir_okay=False, writable=True),
        help="Write an immutable receipt naming the verdict, the criterion that decided it, and what it left "
        "unestablished. Refuses to overwrite an existing file.",
    )(command)


def _emit(result: ProofResult, run: _ProofRun | None = None) -> None:
    if run is not None:
        run.record(result)
    JsonRenderer().emit(result.as_dict())
    raise SystemExit(EXIT_CODES[result.verdict])


def _gather(proof_name: str, fetch: tp.Callable[[], ProofResult]) -> ProofResult:
    """Run ``fetch``, turning an unreachable plane into INCONCLUSIVE.

    Found by pointing the finished command at a real plane: a mistyped workload
    id exited 1, which is the code for "this proof failed". A scheduled run would
    have reported the preemption behaviour as broken when nothing had been asked
    about it.

    That is defect shape (a) from the plan -- a handler turning "I could not find
    out" into a fact -- and it is worse here than anywhere else, because the fact
    it invents is a verdict about the thing the spend existed to establish.
    """

    try:
        return fetch()
    except click.ClickException as error:
        return ProofResult(proof_name, (_unknown("plane-reachable", f"the plane could not answer: {error.message}"),))
    except OSError as error:
        return ProofResult(proof_name, (_unknown("plane-reachable", f"the plane could not be reached: {error}"),))


@proof.command("catalog")
@click.option("--accelerator-type", required=True, help="The shape the fleet declares, e.g. v5p-16.")
@_receipt_options
@click.pass_obj
def catalog_proof(session: CliSession, accelerator_type: str, receipt: str | None, evidence: str | None) -> None:
    """P1 -- pool resources and host count come from the catalog."""

    from eyvan.capacity.hardware_catalog import shape_for

    run = _ProofRun("P1", evaluate_catalog_authority, session=session, receipt=receipt, evidence=evidence)
    run.preflight()
    shape = shape_for(accelerator_type)
    _emit(
        _gather(
            "P1",
            lambda: evaluate_catalog_authority(
                run.seen("capacity-pools", session.client.request("GET", "/api/v1/capacity-pools")),
                run.seen("cells", session.client.request("GET", "/api/v1/cells")),
                accelerator_type=accelerator_type,
                catalog_hosts=None if shape is None else shape.hosts,
            ),
        ),
        run,
    )


@proof.command("preemption")
@click.argument("workload_id")
@_receipt_options
@click.pass_obj
def preemption_proof(session: CliSession, workload_id: str, receipt: str | None, evidence: str | None) -> None:
    """P5 -- a reclaimed Spot slice is charged to the preemption budget."""

    run = _ProofRun("P5", evaluate_spot_preemption, session=session, receipt=receipt, evidence=evidence)
    run.preflight()
    _emit(
        _gather(
            "P5",
            lambda: evaluate_spot_preemption(
                run.seen("workload", session.client.request("GET", f"/api/v1/workloads/{workload_id}")),
                run.seen("cells", session.client.request("GET", "/api/v1/cells")),
                run.seen("capacity-pools", session.client.request("GET", "/api/v1/capacity-pools")),
            ),
        ),
        run,
    )


@proof.command("warm-window")
@click.option("--pool", "pool_id", required=True)
@click.option(
    "--inside", type=click.Path(exists=True, dir_okay=False), required=True, help="Warmth JSON from inside the window."
)
@_receipt_options
@click.pass_obj
def warm_window_proof(session: CliSession, pool_id: str, inside: str, receipt: str | None, evidence: str | None) -> None:
    """P6 -- a warm window held its floor and released it.

    ``--inside`` is a warmth report captured earlier, because one observation
    cannot distinguish a floor that is held from one that is never released.
    Capture it with ``eyvan capacity warmth --json`` during the window.
    """

    run = _ProofRun("P6", evaluate_warm_window, session=session, receipt=receipt, evidence=evidence)
    run.preflight()
    with open(inside, encoding="utf-8") as handle:
        captured = json.load(handle)
    run.seen("warmth-inside-window", captured)
    _emit(
        _gather(
            "P6",
            lambda: evaluate_warm_window(
                captured,
                run.seen("warmth-now", session.client.request("GET", "/api/v1/capacity/warmth")),
                # Scoped to this pool. The fleet-wide report counts every cell
                # the plane has ever held, and a long-lived fleet's historical
                # unpriced cells would make the cost half inconclusive on every
                # run regardless of how the pool under test is priced -- found
                # by running P6 against a plane with 33 of them.
                run.seen(
                    "billing",
                    session.client.request("GET", "/api/v1/capacity/billing", params={"pool_id": pool_id}),
                ),
                pool_id=pool_id,
            ),
        ),
        run,
    )


@proof.command("secret")
@click.argument("workload_id")
@click.option(
    "--value-file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="File holding the secret's plaintext, for the absence check. A file rather than an argument: "
    "a value on the command line lands in shell history and in every process listing on the host.",
)
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@_receipt_options
@click.pass_obj
def secret_proof(
    session: CliSession,
    workload_id: str,
    value_file: str,
    org_id: str | None,
    project_id: str | None,
    receipt: str | None,
    evidence: str | None,
) -> None:
    """P4 -- the secret is delivered, used, and recorded nowhere.

    The absence half is why this is code. A person checks the log, remembers the
    workload record, and does not think to check the audit payloads -- which is
    the one place a well-meaning "context" field would put it.
    """

    with open(value_file, encoding="utf-8") as handle:
        value = handle.read().strip()
    run = _ProofRun(
        "P4",
        evaluate_secret_delivery,
        session=session,
        receipt=receipt,
        evidence=evidence,
        # The evidence file is the one place a leaked secret would be copied to
        # by the very run that exists to find the leak. The checks record whether
        # it leaked; the evidence does not have to carry it to prove that.
        secrets=(value,),
    )
    run.preflight()
    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _emit(
        _gather(
            "P4",
            lambda: evaluate_secret_delivery(
                run.seen(
                    "events",
                    session.client.request(
                        "GET",
                        "/api/v1/events",
                        # Filtered to this workload rather than reading the project's
                        # first thousand events. `/api/v1/events` pages oldest-first
                        # from `after_sequence`, so on any project with prior history
                        # the window never reaches the run the proof was pointed at --
                        # and the evaluator then judged a run it could not see.
                        params={
                            "org_id": scope.org_id,
                            "project_id": scope.project_id,
                            "entity_type": "workload",
                            "entity_id": workload_id,
                            "limit": "1000",
                        },
                    ),
                ),
                run.seen("workload", session.client.request("GET", f"/api/v1/workloads/{workload_id}")),
                # ``request_text``, not ``request``: the logs route is declared
                # PlainTextResponse, so decoding it as JSON fails for every
                # workload -- and for this proof's workload in particular, whose
                # body is the plane's notice that it suppresses output for
                # secret-bearing execution. Read as JSON that notice surfaced as
                # "the plane could not answer", which reported a plane that had
                # answered clearly as unreachable.
                run.seen("logs", session.client.request_text("GET", f"/api/v1/workloads/{workload_id}/logs")),
                secret_value=value,
            ),
        ),
        run,
    )


@proof.command("kubernetes")
@click.option("--api-server", required=True, help="https://<master-ip> of the cluster under test.")
@click.option("--token-file", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--ca-file", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--safe-queue", default="eyvan-safe", show_default=True)
@click.option("--unsafe-queue", default="eyvan-unsafe", show_default=True)
@click.option("--namespace", default="eyvan", show_default=True)
@_receipt_options
def kubernetes_proof(
    api_server: str,
    token_file: str,
    ca_file: str,
    safe_queue: str,
    unsafe_queue: str,
    namespace: str,
    receipt: str | None,
    evidence: str | None,
) -> None:
    """P7 -- drive the real Kubernetes transport against a real API server.

    Credentials are files rather than arguments for the same reason the secret
    proof takes a file: a bearer token on a command line lands in shell history
    and in every process listing on the host.

    Needs a cluster with Kueue installed and two LocalQueues in ``--namespace``,
    routing into two ClusterQueues -- one with preemption Never/Never, one with
    preemption enabled. The names given here are LocalQueue names, because that
    is what a Job may carry and therefore what the driver resolves from. Everything this decides
    is driver-to-API-server, so no workload has to run and no worker has to
    reach the control plane.
    """

    from eyvan.capacity.drivers.base import CapacityProviderError
    from eyvan.capacity.drivers.kubernetes import KubernetesCapacityDriver
    from eyvan.capacity.providers.kubernetes.api import KubernetesApi

    run = _ProofRun("P7", evaluate_kubernetes_preconditions, receipt=receipt, evidence=evidence)
    run.preflight()
    driver = KubernetesCapacityDriver(api=KubernetesApi(base_url=api_server, token_path=token_file, ca_path=ca_file))

    def probe(queue: str | None) -> str | None:
        try:
            driver._require_safe_queue(queue, namespace)
        except CapacityProviderError as error:
            return error.code.value
        return None

    _emit(
        _gather(
            "P7",
            lambda: evaluate_kubernetes_preconditions(
                run.seen(
                    "queue-probes",
                    {
                        "no-queue": probe(None),
                        "safe-queue": probe(safe_queue),
                        "unsafe-queue": probe(unsafe_queue),
                        "missing-queue": probe("eyvan-proof-absent-queue"),
                    },
                )
            ),
        ),
        run,
    )


@proof.command("children")
@click.argument("parent_id")
@click.option("--expected", type=int, default=3, show_default=True, help="How many children the job submitted.")
@click.option(
    "--late-submit-status",
    type=int,
    help="HTTP status from reusing the capability after the parent terminated. Omit if not attempted.",
)
@click.option("--ceiling-status", type=int, help="HTTP status from exceeding the live-children ceiling.")
@_receipt_options
@click.pass_obj
def children_proof(
    session: CliSession,
    parent_id: str,
    expected: int,
    late_submit_status: int | None,
    ceiling_status: int | None,
    receipt: str | None,
    evidence: str | None,
) -> None:
    """P3 -- a running job submits children under its owner's identity.

    The two status codes are passed in rather than provoked here: reusing a
    capability needs the token the job was given, which only the job ever holds,
    and reaching the width ceiling needs hundreds of concurrent children. Both
    are omitted rather than faked when a run did not exercise them, and both
    report inconclusive when omitted.
    """

    run = _ProofRun("P3", evaluate_in_job_submission, session=session, receipt=receipt, evidence=evidence)
    run.preflight()
    _emit(
        _gather(
            "P3",
            lambda: evaluate_in_job_submission(
                run.seen("parent", session.client.request("GET", f"/api/v1/workloads/{parent_id}")),
                run.seen("children", session.client.request("GET", f"/api/v1/workloads/{parent_id}/children")),
                expected_children=expected,
                late_submit_status=late_submit_status,
                ceiling_status=ceiling_status,
            ),
        ),
        run,
    )


@proof.command("logs")
@click.argument("workload_id")
@click.option("--ranks", type=int, default=1, show_default=True, help="How many hosts the workload actually spans.")
@click.option("--limit", type=int, default=50, show_default=True)
@_receipt_options
@click.pass_obj
def logs_proof(
    session: CliSession, workload_id: str, ranks: int, limit: int, receipt: str | None, evidence: str | None
) -> None:
    """P2 -- per-rank structured logs, with an unreachable rank visible.

    ``--ranks`` is supplied rather than inferred, because inferring it from the
    page makes "one rank answered" indistinguishable from "one rank exists" --
    the exact confusion this proof is about.
    """

    run = _ProofRun("P2", evaluate_structured_logs, session=session, receipt=receipt, evidence=evidence)
    run.preflight()
    _emit(
        _gather(
            "P2",
            lambda: evaluate_structured_logs(
                run.seen(
                    "log-page",
                    session.client.request("GET", f"/api/v1/workloads/{workload_id}/log-page", params={"limit": limit}),
                ),
                expected_ranks=ranks,
            ),
        ),
        run,
    )


# --------------------------------------------------------------------------- reading one back


#: The evaluators a receipt may name, so verification can find the criterion a
#: run recorded and hash the version this build carries.
_EVALUATORS: dict[str, tp.Callable[..., ProofResult]] = {
    "P1": evaluate_catalog_authority,
    "P2": evaluate_structured_logs,
    "P3": evaluate_in_job_submission,
    "P4": evaluate_secret_delivery,
    "P5": evaluate_spot_preemption,
    "P6": evaluate_warm_window,
    "P7": evaluate_kubernetes_preconditions,
}


def evaluate_receipt(
    document: tp.Mapping[str, object],
    *,
    digest_intact: bool | None,
    criterion_now: tp.Mapping[str, object] | None,
    build_now: tp.Mapping[str, object],
) -> ProofResult:
    """Decide whether a receipt qualifies the build in front of it.

    Three separate questions, kept separate because their answers differ:
    whether the receipt still says what it said (``digest_intact``), whether the
    criterion that produced the verdict is the criterion this build carries
    (``criterion_now``), and whether the code judged is the code running
    (``build_now``). A receipt can be perfectly intact and still fail to qualify
    anything, which is precisely the case ``docs/proofs.md`` warns about.

    A verdict that was not PASS stays not-PASS here. Verification never upgrades
    an outcome; the most it can do is refuse one.
    """

    checks: list[Check] = []

    if digest_intact is False:
        checks.append(_failed("receipt-intact", "this receipt no longer digests to the value it records"))
    elif digest_intact is None:
        checks.append(_unknown("receipt-intact", "this receipt records no digest, so editing cannot be ruled out"))
    else:
        checks.append(_passed("receipt-intact", "the receipt digests to the value it records"))

    verdict = document.get("verdict")
    if verdict == Verdict.PASS.value:
        checks.append(_passed("verdict", "the run passed its criterion"))
    elif verdict in {Verdict.FAIL.value, Verdict.INCONCLUSIVE.value}:
        checks.append(_failed("verdict", f"the run recorded {verdict}, which qualifies nothing"))
    else:
        checks.append(_unknown("verdict", f"the receipt records no verdict this build understands: {verdict!r}"))

    recorded = document.get("criterion")
    recorded_digest = recorded.get("source_digest") if isinstance(recorded, dict) else None
    if criterion_now is None:
        checks.append(
            _unknown(
                "criterion-current",
                f"this build has no evaluator for {document.get('proof')!r}, so the criterion cannot be compared",
            )
        )
    elif not isinstance(recorded_digest, str) or not recorded_digest:
        checks.append(_unknown("criterion-current", "the receipt records no criterion digest to compare"))
    elif recorded_digest == criterion_now.get("source_digest"):
        checks.append(_passed("criterion-current", "the criterion that decided this verdict is the one running here"))
    else:
        checks.append(
            _failed(
                "criterion-current",
                "the criterion changed since this run: the recorded verdict was decided by different code",
            )
        )

    recorded_candidate = document.get("candidate")
    recorded_build = recorded_candidate.get("build") if isinstance(recorded_candidate, dict) else None
    recorded_source = recorded_build.get("source_digest") if isinstance(recorded_build, dict) else None
    running = build_now.get("source_digest")
    if not isinstance(recorded_source, str) or not recorded_source or not isinstance(running, str) or not running:
        checks.append(
            _unknown("candidate-current", "one of the two builds has no source digest, so they cannot be compared")
        )
    elif recorded_source == running:
        checks.append(_passed("candidate-current", "the build that ran this proof is the build running here"))
    else:
        checks.append(
            _failed(
                "candidate-current",
                f"this receipt qualifies {recorded_source}, and this build is {running}",
            )
        )

    # Requirements are proof-specific: P7 talks directly to Kubernetes, while
    # P1/P6 judge plane metadata/accounting without workload execution. P2-P5
    # also depend on the agents/workers that ran the observed workload. The CLI
    # build alone cannot identify any of those backends.
    proof_id = document.get("proof")
    if proof_id in {"P1", "P2", "P3", "P4", "P5", "P6"}:
        plane = document.get("plane")
        plane_build = plane.get("build") if isinstance(plane, dict) else None
        plane_source = plane_build.get("source_digest") if isinstance(plane_build, dict) else None
        if not isinstance(plane_source, str) or not plane_source or not isinstance(running, str) or not running:
            checks.append(_unknown("plane-current", "the plane or candidate has no source digest to compare"))
        elif plane_source != running:
            checks.append(_failed("plane-current", "the observed control plane ran a different candidate"))
        else:
            checks.append(_passed("plane-current", "the observed control plane ran this candidate"))

    if proof_id in {"P2", "P3", "P4", "P5"}:
        fleet = document.get("fleet")
        fleet = fleet if isinstance(fleet, dict) else {}
        agreement = fleet.get("in_agreement")
        fleet_plane = fleet.get("plane")
        fleet_source = fleet_plane.get("source_digest") if isinstance(fleet_plane, dict) else None
        if agreement is False:
            checks.append(_failed("fleet-current", "the observed fleet explicitly ran different code from its plane"))
        elif isinstance(fleet_source, str) and fleet_source and running and fleet_source != running:
            checks.append(_failed("fleet-current", "the fleet build sweep observed a different control-plane candidate"))
        elif (
            agreement is not True
            or not isinstance(fleet_source, str)
            or not fleet_source
            or not running
            or not fleet.get("agents")
            or fleet.get("unreachable")
        ):
            checks.append(_unknown("fleet-current", "the receipt does not establish complete fleet build agreement"))
        else:
            checks.append(_passed("fleet-current", "the observed fleet agrees with this candidate's plane"))

    return ProofResult(f"verify:{document.get('proof')}", tuple(checks))


@proof.command("verify")
@click.argument("receipt_path", type=click.Path(exists=True, dir_okay=False))
def verify_proof(receipt_path: str) -> None:
    """Decide whether a receipt qualifies *this* build.

    ``docs/proofs.md`` says no historical run qualifies the current candidate
    without an exact-candidate receipt. This is that sentence as a command: it
    compares the build and the criterion the receipt names against the ones in
    front of it, and refuses rather than assuming the intervening changes were
    irrelevant.

    Exits with the usual proof codes, so it can gate a release the same way a
    proof gates a spend.
    """

    from eyvan._internal.build_identity import build_identity
    from eyvan.sdk.commands.receipts import read_receipt, receipt_digest_matches

    try:
        document = read_receipt(receipt_path)
    except (ValueError, OSError) as error:
        _refuse(f"{receipt_path} is not a receipt this build can read: {error}")

    evaluator = _EVALUATORS.get(str(document.get("proof")))
    _emit(
        evaluate_receipt(
            document,
            digest_intact=receipt_digest_matches(document),
            criterion_now=None if evaluator is None else criterion_identity(evaluator),
            build_now=build_identity().as_dict(),
        )
    )


def register_proof_commands(main: click.Group) -> None:
    """Attach the proof command group to the CLI root."""

    main.add_command(proof)


__all__ = (
    "EXIT_CODES",
    "EXIT_NOT_RUN",
    "Check",
    "ProofResult",
    "Verdict",
    "evaluate_catalog_authority",
    "evaluate_in_job_submission",
    "evaluate_kubernetes_preconditions",
    "evaluate_receipt",
    "evaluate_secret_delivery",
    "evaluate_spot_preemption",
    "evaluate_structured_logs",
    "evaluate_warm_window",
    "proof",
    "register_proof_commands",
)
