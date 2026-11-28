# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The criteria that decide whether a hardware spend proved anything.

``docs/proofs.md`` stated each pass criterion in prose, so the verdict was
whatever a person concluded from a screen of JSON at the end of a run they were
relieved to have finished. That is the worst possible moment to ask someone to be
sceptical, and it has already gone wrong once.

**Most of what follows asserts INCONCLUSIVE rather than PASS or FAIL**, and that
is the point rather than an accident of coverage. Every proof here has a shape
where nothing contradicted the claim because nothing exercised it: no cell was
created, the spot slice was never reclaimed, the observation was taken outside
the window. Each of those produces a clean-looking run that establishes nothing,
and a two-state verdict has no way to say so -- it says "pass".
"""

from __future__ import annotations

from eyvan.sdk.commands.proof import (
    EXIT_CODES,
    Verdict,
    evaluate_catalog_authority,
    evaluate_control_plane_ran_it,
    evaluate_in_job_submission,
    evaluate_kubernetes_preconditions,
    evaluate_secret_delivery,
    evaluate_spot_preemption,
    evaluate_structured_logs,
    evaluate_warm_window,
)


def _check(result, name: str) -> Verdict:
    return next(check.verdict for check in result.checks if check.name == name)


def _detail(result, name: str) -> str:
    return next(check.detail for check in result.checks if check.name == name)


# --------------------------------------------------------------------------- the verdict algebra


def test_a_failure_outranks_an_inconclusive() -> None:
    """One definitively wrong criterion is not softened by an unmeasured one."""

    result = evaluate_spot_preemption(
        {"preemption_count": 0, "capacity_loss_count": 2, "state": "pending"},
        [],
        [],
    )

    assert result.verdict is Verdict.FAIL


def test_each_verdict_has_a_distinct_exit_code() -> None:
    """A scheduled run has to tell "wrong" from "did not happen".

    They call for different responses: one is a defect to fix, the other is a run
    to repeat, and a single non-zero exit conflates them.
    """

    assert sorted(EXIT_CODES.values()) == [0, 1, 2]
    assert EXIT_CODES[Verdict.PASS] == 0


def test_a_proof_with_no_checks_is_not_a_pass() -> None:
    """The degenerate case, asserted because it is the one that reads as success.

    An empty check list is what a proof produces when nothing matched anything,
    and `all([])` is True.
    """

    from eyvan.sdk.commands.proof import ProofResult

    assert ProofResult("P0", ()).verdict is Verdict.INCONCLUSIVE


# --------------------------------------------------------------------------- P1


def _pool(pool_id: str = "research/v5p", accelerator: str = "v5p-16") -> dict:
    return {"id": pool_id, "accelerator_type": accelerator}


def _cell(
    cell_id: str,
    *,
    hosts: int | None = 2,
    pool: str = "research/v5p",
    state: str = "ready",
    slices: object = 1,
) -> dict:
    """One cell as the plane reports it.

    ``observed_tpu_vms`` counts every VM in the cell and ``worker_count`` counts
    the slices they are divided into, so the two coincide only on a single-slice
    cell.
    """

    return {
        "id": cell_id,
        "pool_id": pool,
        "state": state,
        "observed_tpu_vms": hosts,
        "worker_count": slices,
    }


def test_a_slice_with_the_catalog_host_count_passes() -> None:
    result = evaluate_catalog_authority(
        [_pool()], [_cell("cell-a"), _cell("cell-b")], accelerator_type="v5p-16", catalog_hosts=2
    )

    assert result.verdict is Verdict.PASS


def test_a_half_sized_slice_fails() -> None:
    """The defect the proof exists for.

    A provider returning one host of two starts every rank, builds the wrong
    mesh, and runs at half throughput while reporting success.
    """

    result = evaluate_catalog_authority(
        [_pool()], [_cell("cell-a", hosts=1)], accelerator_type="v5p-16", catalog_hosts=2
    )

    assert result.verdict is Verdict.FAIL
    assert "expected 2 host(s)" in next(check.detail for check in result.checks if check.verdict is Verdict.FAIL)


def test_a_multislice_cell_is_measured_per_slice_rather_than_in_total() -> None:
    """Found by running P1 against a live 2x v5p-16.

    The catalog states hosts per *slice*. A multislice cell holds several under
    one allocation, so its four VMs were compared against the catalog's two and
    reported as a defect -- on an allocation that was exactly right. A criterion
    that fails a correct fleet is worse than no criterion: it arrives at the end
    of a spend, and the next person turns it off.
    """

    result = evaluate_catalog_authority(
        [_pool()], [_cell("cell-a", hosts=4, slices=2)], accelerator_type="v5p-16", catalog_hosts=2
    )

    assert result.verdict is Verdict.PASS
    assert "2 slice(s)" in _detail(result, "hosts-match-catalog")


def test_a_multislice_cell_missing_a_whole_slice_still_fails() -> None:
    """The per-slice rule must not become a way to pass with half the hardware."""

    result = evaluate_catalog_authority(
        [_pool()], [_cell("cell-a", hosts=2, slices=2)], accelerator_type="v5p-16", catalog_hosts=2
    )

    assert result.verdict is Verdict.FAIL
    assert "2 VM(s) across 2 slice(s)" in _detail(result, "hosts-match-catalog")


def test_a_cell_that_reports_no_slice_count_is_inconclusive() -> None:
    """Four VMs is right for two slices and wrong for one.

    Without the count there is nothing to divide by, and assuming one would make
    every multislice cell on an older plane read as a failure.
    """

    result = evaluate_catalog_authority(
        [_pool()], [_cell("cell-a", hosts=4, slices=None)], accelerator_type="v5p-16", catalog_hosts=2
    )

    assert result.verdict is Verdict.INCONCLUSIVE
    assert _check(result, "hosts-observed") is Verdict.INCONCLUSIVE


def test_a_fleet_that_never_applied_is_inconclusive() -> None:
    """Not a pass. No pool means nothing was observed, not that hosts matched."""

    result = evaluate_catalog_authority([], [], accelerator_type="v5p-16", catalog_hosts=2)

    assert result.verdict is Verdict.INCONCLUSIVE


def test_a_pool_with_no_live_cells_is_inconclusive() -> None:
    """The run that dies during provisioning: pool applied, cell never arrived."""

    result = evaluate_catalog_authority([_pool()], [], accelerator_type="v5p-16", catalog_hosts=2)

    assert result.verdict is Verdict.INCONCLUSIVE
    assert _check(result, "cells-exist") is Verdict.INCONCLUSIVE


def test_a_cell_that_never_reported_hosts_is_inconclusive() -> None:
    """`None` is "not observed", and treating it as a mismatch would be a lie."""

    result = evaluate_catalog_authority(
        [_pool()], [_cell("cell-a", hosts=None)], accelerator_type="v5p-16", catalog_hosts=2
    )

    assert _check(result, "hosts-observed") is Verdict.INCONCLUSIVE


def test_a_catalog_with_no_host_count_cannot_be_the_authority_for_one() -> None:
    """`hosts` is optional in the catalog, so the proof must notice its absence.

    Otherwise a shape with no recorded host count silently skips the only check
    that matters and the proof reports a pass.
    """

    result = evaluate_catalog_authority([_pool()], [_cell("cell-a")], accelerator_type="v5p-16", catalog_hosts=None)

    assert result.verdict is Verdict.INCONCLUSIVE


def test_deleted_cells_do_not_decide_the_verdict() -> None:
    """A released cell's last observation says nothing about this run."""

    result = evaluate_catalog_authority(
        [_pool()],
        [_cell("cell-old", hosts=1, state="deleted"), _cell("cell-a", hosts=2)],
        accelerator_type="v5p-16",
        catalog_hosts=2,
    )

    assert result.verdict is Verdict.PASS


# --------------------------------------------------------------------------- P5


def _preempted(**overrides) -> dict:
    return {"id": "wl-1", "preemption_count": 1, "capacity_loss_count": 0, "state": "running", **overrides}


def _assert_unattributed_preemption(result) -> None:
    """Budget and survival are measured; current cell inventory is not event history."""

    assert _check(result, "charged-to-the-preemption-budget") is Verdict.PASS
    assert _check(result, "workload-survived") is Verdict.PASS
    assert _check(result, "classified-as-preempted") is Verdict.INCONCLUSIVE
    assert _check(result, "no-stockout-cooldown") is Verdict.INCONCLUSIVE
    assert "does not link" in _detail(result, "no-stockout-cooldown")
    assert result.verdict is Verdict.INCONCLUSIVE


def test_clean_budget_charging_does_not_qualify_unattributed_inventory() -> None:
    result = evaluate_spot_preemption(
        _preempted(),
        [{"id": "cell-a", "pool_id": "research/v5p", "provider_failure_code": "preempted", "availability": {}}],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)


def test_a_run_that_was_never_reclaimed_is_inconclusive() -> None:
    """The common outcome of a short spot run, and the one most easily misread.

    Nothing failed. Nothing was classified either, and "we were not preempted"
    is not "preemption is classified correctly".
    """

    result = evaluate_spot_preemption({"preemption_count": 0, "capacity_loss_count": 0, "state": "running"}, [], [])

    assert result.verdict is Verdict.INCONCLUSIVE


def test_a_reclamation_charged_to_the_failure_budget_fails() -> None:
    """Exactly the regression the phase fixed, stated as a criterion."""

    result = evaluate_spot_preemption(
        {"preemption_count": 0, "capacity_loss_count": 1, "state": "running"},
        [{"id": "cell-a", "pool_id": "p", "provider_failure_code": "preempted", "availability": {}}],
        [{"id": "p"}],
    )

    assert _check(result, "charged-to-the-preemption-budget") is Verdict.FAIL


def test_both_budgets_moving_is_also_a_failure() -> None:
    """Double-charging is not half-right; it consumes a retry budget twice."""

    result = evaluate_spot_preemption(
        _preempted(capacity_loss_count=1),
        [{"id": "cell-a", "pool_id": "p", "provider_failure_code": "preempted", "availability": {}}],
        [{"id": "p"}],
    )

    assert _check(result, "charged-to-the-preemption-budget") is Verdict.FAIL


def test_an_unattributed_cooldown_cannot_fail_this_reclamation() -> None:
    """An inventory cooldown cannot fail a workload without event attribution."""

    result = evaluate_spot_preemption(
        _preempted(),
        [
            {
                "id": "cell-a",
                "pool_id": "research/v5p",
                "provider_failure_code": "preempted",
                "availability": {"cooldown_until": "2026-08-05T12:00:00Z"},
            }
        ],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)


def test_an_unrelated_pool_cooldown_cannot_qualify_this_runs_cooldown_history() -> None:
    """Another pool's backoff must not fail this proof.

    Otherwise any fleet with one unhealthy pool anywhere fails every preemption
    run, and the proof becomes something people re-run until it is green.
    """

    result = evaluate_spot_preemption(
        _preempted(),
        [
            {"id": "cell-a", "pool_id": "research/v5p", "provider_failure_code": "preempted", "availability": {}},
            {"id": "other", "pool_id": "elsewhere", "availability": {"cooldown_until": "2026-08-05T12:00:00Z"}},
        ],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)


def test_a_reclaimed_cell_that_is_gone_cannot_vouch_for_the_absence_of_a_cooldown() -> None:
    """The reading that would have qualified the defect this proof exists to catch.

    A reclaimed cell is deleted and awaiting recreation while the proof runs, so
    it leaves the report on its own. Read naively, its departure turns "the cell
    is not cooling" into "no cell is cooling" -- and the criterion goes green on
    a build that still sets the cooldown, with no operator action in between.

    This is the exact shape of a run that establishes nothing and looks clean
    doing it, so it must settle nothing rather than settle it wrongly. The
    fleet-wide counters are deliberately still those of a real reclamation: the
    preemption happened, only the cell that witnessed it has been reaped.
    """

    result = evaluate_spot_preemption(
        _preempted(),
        [{"id": "replacement", "pool_id": "research/v5p", "availability": {}}],
        [{"id": "research/v5p"}],
    )

    assert _check(result, "no-stockout-cooldown") is Verdict.INCONCLUSIVE
    assert result.verdict is not Verdict.PASS


def test_a_replacement_cell_backing_off_for_its_own_reasons_is_not_this_proof_s_business() -> None:
    """Scope preemption-classification evidence to the reclaimed cell, excluding unrelated replacement cooldowns."""

    result = evaluate_spot_preemption(
        _preempted(),
        [
            {
                "id": "cell-0001",
                "pool_id": "research/v5p",
                "provider_failure_code": "preempted",
                "availability": {"interruptions": 1.9604, "consecutive_failures": 0, "cooldown_until": None},
            },
            {
                "id": "cell-0002",
                "pool_id": "research/v5p",
                "provider_failure_code": None,
                "availability": {
                    "interruptions": 0.0,
                    "consecutive_failures": 1,
                    "cooldown_until": "2026-08-13T19:08:05Z",
                },
            },
        ],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)


def test_an_unattributed_reclaimed_cell_cooldown_is_unknown_beside_a_healthy_replacement() -> None:
    """Neither the cooldown nor a healthy replacement identifies this workload's event."""

    result = evaluate_spot_preemption(
        _preempted(),
        [
            {
                "id": "cell-0001",
                "pool_id": "research/v5p",
                "provider_failure_code": "preempted",
                "availability": {
                    "interruptions": 1.9604,
                    "consecutive_failures": 1,
                    "cooldown_until": "2026-08-13T17:14:29Z",
                },
            },
            {"id": "cell-0002", "pool_id": "research/v5p", "availability": {"cooldown_until": None}},
        ],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)


def test_a_recreated_cells_interruption_count_cannot_attribute_a_workload_event() -> None:
    """A cleared code and cumulative interruptions do not identify this run's event."""

    result = evaluate_spot_preemption(
        _preempted(),
        [
            {
                "id": "cell-a",
                "pool_id": "research/v5p",
                "provider_failure_code": None,
                "availability": {"interruptions": 1.9604, "cooldown_until": None},
            }
        ],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)


def test_a_cell_that_was_never_interrupted_does_not_witness() -> None:
    """Otherwise a fresh replacement would vouch for an event it never saw."""

    result = evaluate_spot_preemption(
        _preempted(),
        [{"id": "replacement", "pool_id": "research/v5p", "availability": {"interruptions": 0.0}}],
        [{"id": "research/v5p"}],
    )

    assert _check(result, "no-stockout-cooldown") is Verdict.INCONCLUSIVE


def test_a_surviving_reclaimed_cell_without_a_cooldown_cannot_qualify_this_run() -> None:
    """A surviving cell's current code and empty cooldown are not run-linked history."""

    result = evaluate_spot_preemption(
        _preempted(),
        [
            {"id": "cell-a", "pool_id": "research/v5p", "provider_failure_code": "preempted", "availability": {}},
            {"id": "replacement", "pool_id": "research/v5p", "availability": {}},
        ],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)


def test_a_workload_that_failed_did_not_survive_its_preemption() -> None:
    result = evaluate_spot_preemption(
        _preempted(state="failed"),
        [{"id": "cell-a", "pool_id": "p", "provider_failure_code": "preempted", "availability": {}}],
        [{"id": "p"}],
    )

    assert _check(result, "workload-survived") is Verdict.FAIL


def test_an_unattributed_wrong_failure_code_cannot_fail_this_run() -> None:
    result = evaluate_spot_preemption(
        _preempted(),
        [{"id": "cell-a", "pool_id": "p", "provider_failure_code": "provider_error", "availability": {}}],
        [{"id": "p"}],
    )

    _assert_unattributed_preemption(result)


# --------------------------------------------------------------------------- P6


def _warmth(*, active: bool, floor: int, warm: int, pool_id: str = "research/cpu") -> dict:
    return {"pools": [{"pool_id": pool_id, "window_active": active, "floor": floor, "warm_cells": warm}]}


def _billing(**overrides) -> dict:
    return {"idle_seconds": 3600.0, "idle_cost": 2.5, "unpriced_cells": 0, **overrides}


def test_a_window_held_and_released_passes() -> None:
    result = evaluate_warm_window(
        _warmth(active=True, floor=1, warm=1),
        _warmth(active=False, floor=0, warm=0),
        _billing(),
        pool_id="research/cpu",
    )

    assert result.verdict is Verdict.PASS


def test_a_floor_that_never_released_fails() -> None:
    result = evaluate_warm_window(
        _warmth(active=True, floor=1, warm=1),
        _warmth(active=False, floor=1, warm=1),
        _billing(),
        pool_id="research/cpu",
    )

    assert _check(result, "floor-released") is Verdict.FAIL


def test_a_floor_that_was_never_raised_settles_nothing_by_falling() -> None:
    """Found by auditing every criterion for the class P5's cooldown check had.

    A window that raised no floor releases nothing, and reading that as a pass
    reported the window working on a run where it did nothing at all. It is the
    *cheap* half to satisfy, too: a pool sitting at zero all day meets it on
    every observation, so the one shape that cannot establish the claim was the
    one shape guaranteed to look like it had.
    """

    result = evaluate_warm_window(
        _warmth(active=True, floor=0, warm=0),
        _warmth(active=False, floor=0, warm=0),
        _billing(),
        pool_id="research/cpu",
    )

    assert _check(result, "floor-released") is Verdict.INCONCLUSIVE
    assert result.verdict is not Verdict.PASS


def test_a_floor_that_was_never_held_fails() -> None:
    """The window's whole promise: the 09:00 latency happens anyway without it."""

    result = evaluate_warm_window(
        _warmth(active=True, floor=2, warm=0),
        _warmth(active=False, floor=0, warm=0),
        _billing(),
        pool_id="research/cpu",
    )

    assert _check(result, "floor-held") is Verdict.FAIL


def test_an_observation_taken_outside_the_window_settles_nothing() -> None:
    """Both observations after the window is a run that measured the wrong hour."""

    result = evaluate_warm_window(
        _warmth(active=False, floor=0, warm=0),
        _warmth(active=False, floor=0, warm=0),
        _billing(),
        pool_id="research/cpu",
    )

    assert result.verdict is Verdict.INCONCLUSIVE
    assert _check(result, "window-was-active") is Verdict.INCONCLUSIVE


def test_a_second_observation_still_inside_the_window_settles_nothing() -> None:
    """Held-and-still-held is not held-and-released."""

    result = evaluate_warm_window(
        _warmth(active=True, floor=1, warm=1),
        _warmth(active=True, floor=1, warm=1),
        _billing(),
        pool_id="research/cpu",
    )

    assert _check(result, "window-closed") is Verdict.INCONCLUSIVE


def test_an_unpriced_fleet_cannot_prove_the_cost_half() -> None:
    """The finding that produced the doctor fix, as a proof criterion.

    Zero cost on an unpriced fleet is not evidence that idle cost was measured;
    it is evidence that nothing could be. Passing here is how a fleet that has
    been running for hours reads as free.
    """

    result = evaluate_warm_window(
        _warmth(active=True, floor=1, warm=1),
        _warmth(active=False, floor=0, warm=0),
        _billing(idle_cost=0.0, unpriced_cells=3),
        pool_id="research/cpu",
    )

    assert _check(result, "idle-cost-accrued") is Verdict.INCONCLUSIVE
    assert result.verdict is Verdict.INCONCLUSIVE


def test_idle_seconds_with_no_idle_cost_on_a_priced_fleet_fails() -> None:
    """Distinct from the unpriced case: here the price exists and the clock did not."""

    result = evaluate_warm_window(
        _warmth(active=True, floor=1, warm=1),
        _warmth(active=False, floor=0, warm=0),
        _billing(idle_cost=0.0),
        pool_id="research/cpu",
    )

    assert _check(result, "idle-cost-accrued") is Verdict.FAIL


def test_a_missing_pool_is_inconclusive() -> None:
    result = evaluate_warm_window(
        _warmth(active=True, floor=1, warm=1),
        _warmth(active=False, floor=0, warm=0),
        _billing(),
        pool_id="research/absent",
    )

    assert result.verdict is Verdict.INCONCLUSIVE


# --------------------------------------------------------------------------- P4


_VALUE = "hf_secretvalue123"


def _event(action: str, *, attempt: int = 1, **payload) -> dict:
    mount = {"names": ["HF_TOKEN"]} if action == "secret.issued" else {"name": "HF_TOKEN"}
    return {
        "action": action,
        "entity_type": "workload",
        "entity_id": "wl-1",
        "org_id": "acme",
        "project_id": "research",
        "payload": {"attempt": attempt, "generation": 1, **mount, **payload},
    }


def test_a_secret_used_once_and_recorded_nowhere_passes() -> None:
    result = evaluate_secret_delivery(
        [_event("secret.issued"), _event("secret.read")],
        {"id": "wl-1", "spec": {"secret_mounts": {"HF_TOKEN": "gcp-secret://p/t"}}},
        "token length 17\n",
        secret_value=_VALUE,
    )

    assert result.verdict is Verdict.PASS


def test_a_value_in_the_log_fails() -> None:
    result = evaluate_secret_delivery(
        [_event("secret.issued"), _event("secret.read")],
        {"id": "wl-1"},
        f"printing {_VALUE} for debugging\n",
        secret_value=_VALUE,
    )

    assert _check(result, "value-never-recorded") is Verdict.FAIL


def test_a_value_in_an_audit_payload_fails() -> None:
    """The place a person checking by hand does not think to look.

    A well-meaning "context" field on the issuance event is exactly how the value
    would end up in the one store that is retained for ninety days.
    """

    result = evaluate_secret_delivery(
        [_event("secret.issued", context=_VALUE), _event("secret.read")],
        {"id": "wl-1"},
        "token length 17\n",
        secret_value=_VALUE,
    )

    assert _check(result, "value-never-recorded") is Verdict.FAIL


def test_a_value_in_the_workload_record_fails() -> None:
    """The defect the whole phase exists to remove: a credential in a spec."""

    result = evaluate_secret_delivery(
        [_event("secret.issued"), _event("secret.read")],
        {"id": "wl-1", "spec": {"environment": {"HF_TOKEN": _VALUE}}},
        "",
        secret_value=_VALUE,
    )

    assert _check(result, "value-never-recorded") is Verdict.FAIL


def test_a_suppressed_log_says_so_rather_than_claiming_a_clean_grep() -> None:
    """The plane serves no output at all for a workload that mounts secrets.

    So the log grep is vacuous for every workload this proof is ever pointed at,
    and reporting it as "the value appears in no log" credits an inspection that
    did not happen. The check still passes -- a value cannot leak through output
    the plane refuses to serve -- but the receipt has to name which fact it is.
    """

    result = evaluate_secret_delivery(
        [_event("secret.issued"), _event("secret.read")],
        {"id": "wl-1", "spec": {"secret_mounts": {"HF_TOKEN": "gcp-secret://p/t"}}},
        "[eyvan] logs are suppressed for this workload because it mounts secrets; its output may contain their values.",
        secret_value=_VALUE,
    )

    assert _check(result, "value-never-recorded") is Verdict.PASS
    assert "by suppression rather than by inspection" in _detail(result, "value-never-recorded")


def test_suppression_never_masks_a_value_that_did_reach_the_log() -> None:
    """The ordering that makes the branch above safe.

    If suppression were consulted before the grep, a plane that regressed and
    served the output anyway would be reported clean by the one proof whose job
    is to catch exactly that.
    """

    result = evaluate_secret_delivery(
        [_event("secret.issued"), _event("secret.read")],
        {"id": "wl-1", "spec": {"secret_mounts": {"HF_TOKEN": "gcp-secret://p/t"}}},
        f"printing {_VALUE} for debugging\n",
        secret_value=_VALUE,
    )

    assert _check(result, "value-never-recorded") is Verdict.FAIL


def test_a_read_with_no_issuance_in_its_attempt_fails() -> None:
    """The signal a leaked token produces, and the reason both events exist."""

    result = evaluate_secret_delivery(
        [_event("secret.issued", attempt=1), _event("secret.read", attempt=2)],
        {"id": "wl-1"},
        "",
        secret_value=_VALUE,
    )

    assert _check(result, "read-matches-an-issuance") is Verdict.FAIL


def test_a_mount_that_was_never_exercised_is_inconclusive() -> None:
    """No secret event means the workload never read it -- not that it leaked nothing."""

    result = evaluate_secret_delivery([], {"id": "wl-1"}, "", secret_value=_VALUE)

    assert result.verdict is Verdict.INCONCLUSIVE


def test_two_reads_for_one_issuance_fails() -> None:
    """Per-attempt means once. A second read is a token outliving its attempt."""

    result = evaluate_secret_delivery(
        [_event("secret.issued"), _event("secret.read"), _event("secret.read")],
        {"id": "wl-1"},
        "",
        secret_value=_VALUE,
    )

    assert _check(result, "one-read") is Verdict.FAIL


def test_without_the_value_the_absence_check_cannot_run() -> None:
    """An empty needle matches every haystack, so this must refuse rather than pass."""

    result = evaluate_secret_delivery(
        [_event("secret.issued"), _event("secret.read")], {"id": "wl-1"}, "", secret_value=""
    )

    assert result.verdict is Verdict.INCONCLUSIVE


# --------------------------------------------------------------------------- the commands


def test_every_proof_evaluator_has_a_command() -> None:
    """An evaluator with no caller decides nothing, which is this repo's defect.

    The four criteria that can be machine-decided are the four commands; the
    proofs that need a host killed mid-run stay procedural on purpose, and are
    not silently listed here as if they were automated.
    """

    from eyvan.sdk import cli

    assert set(cli.main.commands["proof"].commands) == {
        "catalog",
        "preemption",
        "warm-window",
        "secret",
        "kubernetes",
        "children",
        "logs",
        # Not a criterion and not a spend: it reads a receipt an earlier run
        # wrote and decides whether that run qualifies the build in front of it.
        "verify",
    }


# --------------------------------------------------------------------------- reachability


def test_a_plane_that_cannot_answer_is_inconclusive_not_a_failure() -> None:
    """Found by pointing the finished command at a real plane.

    A mistyped workload id exited 1 -- the code for "this proof failed" -- so a
    scheduled run would have reported the preemption behaviour as broken when
    nothing had been asked about it.

    Defect shape (a) from the plan: a handler turning "I could not find out" into
    a fact. It is worse here than anywhere else, because the fact it invents is a
    verdict about the thing the spend existed to establish.
    """

    import click

    from eyvan.sdk.commands.proof import _gather

    def unreachable():
        raise click.ClickException("404: workload 'typo' was not found")

    result = _gather("P5", unreachable)

    assert result.verdict is Verdict.INCONCLUSIVE
    assert "typo" in result.checks[0].detail


def test_a_dropped_connection_is_also_inconclusive() -> None:
    """A control plane restarting mid-proof is not a proof failure."""

    from eyvan.sdk.commands.proof import _gather

    def dropped():
        raise ConnectionError("connection reset by peer")

    assert _gather("P1", dropped).verdict is Verdict.INCONCLUSIVE


def test_a_real_verdict_passes_through_the_guard_unchanged() -> None:
    """The guard must not swallow the answer it exists to protect.

    A wrapper that turned every result into inconclusive would make the proofs
    incapable of ever passing, which is a failure mode the error path alone
    cannot detect.
    """

    from eyvan.sdk.commands.proof import Check as ProofCheck
    from eyvan.sdk.commands.proof import ProofResult, _gather

    expected = ProofResult("P1", (ProofCheck("x", Verdict.PASS, "ok"),))

    assert _gather("P1", lambda: expected) is expected


def test_released_unpriced_cells_do_not_block_a_measured_cost() -> None:
    """Accept measured positive cost even when earlier released cell incarnations had no prices."""

    result = evaluate_warm_window(
        _warmth(active=True, floor=1, warm=1),
        _warmth(active=False, floor=0, warm=0),
        _billing(idle_cost=0.0304, idle_seconds=9894.0, unpriced_cells=2),
        pool_id="research/cpu",
    )

    assert _check(result, "idle-cost-accrued") is Verdict.PASS
    assert result.verdict is Verdict.PASS


def test_a_fully_priced_pool_that_accrued_nothing_still_fails() -> None:
    """The clock not running is a defect, and must not be softened to inconclusive."""

    result = evaluate_warm_window(
        _warmth(active=True, floor=1, warm=1),
        _warmth(active=False, floor=0, warm=0),
        _billing(idle_cost=0.0, idle_seconds=3600.0, unpriced_cells=0),
        pool_id="research/cpu",
    )

    assert _check(result, "idle-cost-accrued") is Verdict.FAIL


# --------------------------------------------------------------------------- P7


def _k8s(**overrides: str | None) -> dict[str, str | None]:
    base: dict[str, str | None] = {
        "no-queue": None,
        "safe-queue": None,
        "unsafe-queue": "configuration",
        "missing-queue": "configuration",
    }
    base.update(overrides)
    return base


def test_a_driver_that_refuses_the_right_queues_passes() -> None:
    assert evaluate_kubernetes_preconditions(_k8s()).verdict is Verdict.PASS


def test_admitting_an_unsafe_queue_fails() -> None:
    """The hazard the whole check exists for.

    Kueue eviction destroys a pod with no checkpoint hook, bypassing Eyvan's
    preemption protocol entirely, and the only trace is a training job that lost
    hours with nothing in the ledger explaining why.
    """

    result = evaluate_kubernetes_preconditions(_k8s(**{"unsafe-queue": None}))

    assert _check(result, "unsafe-queue") is Verdict.FAIL


def test_refusing_a_safe_queue_also_fails() -> None:
    """A driver that refuses everything is unusable with Kueue at all.

    Asserted separately because a check written only against the hazard passes
    trivially for an implementation that says no to everything.
    """

    result = evaluate_kubernetes_preconditions(_k8s(**{"safe-queue": "configuration"}))

    assert _check(result, "safe-queue") is Verdict.FAIL


def test_a_missing_queue_must_be_refused() -> None:
    """A pool pointed at a queue that does not exist suspends every Job forever.

    Reporting "no problems" for that is worse than a refusal, because the fleet
    looks configured and never runs anything.
    """

    result = evaluate_kubernetes_preconditions(_k8s(**{"missing-queue": None}))

    assert _check(result, "missing-queue") is Verdict.FAIL


def test_an_unexercised_case_is_inconclusive_not_a_pass() -> None:
    """A cluster with no Kueue cannot answer, and must not appear to."""

    observations = _k8s()
    del observations["unsafe-queue"]

    result = evaluate_kubernetes_preconditions(observations)

    assert _check(result, "unsafe-queue") is Verdict.INCONCLUSIVE
    assert result.verdict is Verdict.INCONCLUSIVE


def test_the_wrong_failure_code_is_not_a_pass() -> None:
    """The criterion is the *documented* code, not merely some refusal.

    A refusal typed as a transient provider error would be retried forever
    against a configuration that can never become safe.
    """

    result = evaluate_kubernetes_preconditions(_k8s(**{"unsafe-queue": "provider_unavailable"}))

    assert _check(result, "unsafe-queue") is Verdict.FAIL


# --------------------------------------------------------------------------- P3


_PARENT = {"id": "wl-parent", "created_by": "alice", "org_id": "acme", "project_id": "research"}


def _child(child_id: str, **overrides) -> dict:
    base = {
        "id": child_id,
        "created_by": "alice",
        "parent_workload_id": "wl-parent",
        "org_id": "acme",
        "project_id": "research",
    }
    base.update(overrides)
    return base


def _p3(children, **kw):
    kw.setdefault("expected_children", 2)
    kw.setdefault("late_submit_status", 409)
    kw.setdefault("ceiling_status", 409)
    return evaluate_in_job_submission(_PARENT, children, **kw)


def test_children_under_the_owners_identity_pass() -> None:
    assert _p3([_child("a"), _child("b")]).verdict is Verdict.PASS


def test_a_synthetic_created_by_fails() -> None:
    """The criterion the plan singled out, and the one nothing else would catch.

    A child attributed to `workload:<id>` competes as a fresh per-principal
    fair-share leaf against a fresh quota scope, so a self-submitting launcher
    escapes the limits its owner is subject to. The job tree looks correct
    throughout.
    """

    result = _p3([_child("a"), _child("b", created_by="workload:wl-parent")])

    assert _check(result, "children-carry-the-owner") is Verdict.FAIL


def test_a_child_in_another_scope_fails() -> None:
    """The capability exists to stop a job naming a scope its parent is not in."""

    result = _p3([_child("a"), _child("b", project_id="finance")])

    assert _check(result, "children-are-parented") is Verdict.FAIL


def test_an_unparented_child_fails() -> None:
    result = _p3([_child("a"), _child("b", parent_workload_id="wl-someone-else")])

    assert _check(result, "children-are-parented") is Verdict.FAIL


def test_a_terminated_parent_that_still_accepts_children_fails() -> None:
    """A capability outliving its parent is the leak the design revokes by state."""

    result = _p3([_child("a"), _child("b")], late_submit_status=201)

    assert _check(result, "capability-expires-with-the-parent") is Verdict.FAIL


def test_an_unattempted_reuse_is_inconclusive() -> None:
    result = _p3([_child("a"), _child("b")], late_submit_status=None)

    assert _check(result, "capability-expires-with-the-parent") is Verdict.INCONCLUSIVE
    assert result.verdict is Verdict.INCONCLUSIVE


def test_an_unreached_ceiling_is_inconclusive_not_a_pass() -> None:
    """256 concurrent children is impractical in most runs, and saying so is the point."""

    result = _p3([_child("a"), _child("b")], ceiling_status=None)

    assert _check(result, "width-ceiling") is Verdict.INCONCLUSIVE


def test_a_ceiling_that_admits_fails() -> None:
    result = _p3([_child("a"), _child("b")], ceiling_status=201)

    assert _check(result, "width-ceiling") is Verdict.FAIL


def test_a_parent_that_submitted_nothing_is_inconclusive() -> None:
    """No children means the capability was never used, not that it works."""

    assert _p3([]).verdict is Verdict.INCONCLUSIVE


def test_a_short_fan_out_is_a_failure_not_a_pass() -> None:
    """Two of three accepted means one was rejected, and that is a result."""

    result = _p3([_child("a")], expected_children=3)

    assert _check(result, "children-created") is Verdict.FAIL


# --------------------------------------------------------------------------- P2


def _log_record(rank: int = 0, **overrides) -> dict:
    base = {
        "message": "line",
        "stream": "stdout",
        "timestamp": "2026-08-07T02:50:43+00:00",
        "cursor": "s=abc;i=1",
        "rank": rank,
        "pid": 42,
    }
    base.update(overrides)
    return base


def _log_page(records=None, **overrides) -> dict:
    page = {
        "records": records if records is not None else [_log_record()],
        "next_cursor": "1:0=s=abc",
        "unavailable": False,
        "detail": "",
        "truncated_ranks": [],
    }
    page.update(overrides)
    return page


def test_a_two_rank_page_passes_only_once_a_rank_has_actually_been_lost() -> None:
    """P2's criterion is a rank *killed* mid-stream, not merely two healthy ones.

    A slice where nothing went wrong has not exercised the reporting path, so the
    run says so instead of passing. That is the difference between "both hosts
    answered" and "a host stopped answering and we could still see the other".
    """

    healthy = evaluate_structured_logs(_log_page([_log_record(0), _log_record(1)]), expected_ranks=2)
    assert _check(healthy, "unreachable-rank-reported-in-place") is Verdict.INCONCLUSIVE
    assert healthy.verdict is Verdict.INCONCLUSIVE

    lost = evaluate_structured_logs(_log_page([_log_record(0)], truncated_ranks=[1]), expected_ranks=2)
    assert lost.verdict is Verdict.PASS


def test_a_page_with_no_timestamps_fails() -> None:
    """The defect that made multi-host ordering meaningless.

    `merge_pages` orders by timestamp, so a page without them collapses to rank
    order and the interleaving per-rank attribution exists for is gone.
    """

    result = evaluate_structured_logs(
        _log_page([_log_record(0, timestamp=None), _log_record(1, timestamp=None)]), expected_ranks=2
    )

    assert _check(result, "records-carry-a-timestamp") is Verdict.FAIL


def test_a_page_with_no_streams_fails() -> None:
    """Without stdout/stderr this is `logs()` in a more expensive shape."""

    result = evaluate_structured_logs(
        _log_page([_log_record(0, stream="unknown"), _log_record(1, stream="unknown")]), expected_ranks=2
    )

    assert _check(result, "records-carry-a-stream") is Verdict.FAIL


def test_a_page_with_no_cursors_fails() -> None:
    """A record nobody can resume from makes `--follow` restart every poll."""

    result = evaluate_structured_logs(_log_page([_log_record(0, cursor="")]), expected_ranks=1)

    assert _check(result, "records-are-resumable") is Verdict.FAIL


def test_a_missing_rank_is_a_failure_not_silence() -> None:
    """The whole point: a rank that vanished must not read as a quiet host."""

    result = evaluate_structured_logs(_log_page([_log_record(0)]), expected_ranks=2)

    assert _check(result, "per-rank-attribution") is Verdict.FAIL


def test_a_rank_reported_unavailable_still_counts_as_accounted_for() -> None:
    """Reported-in-place is the correct outcome, not a missing rank."""

    result = evaluate_structured_logs(_log_page([_log_record(0)], truncated_ranks=[1]), expected_ranks=2)

    assert _check(result, "per-rank-attribution") is Verdict.PASS
    assert _check(result, "unreachable-rank-reported-in-place") is Verdict.PASS


def test_a_single_host_run_cannot_show_attribution() -> None:
    """Honest about what one host can prove, rather than passing trivially."""

    result = evaluate_structured_logs(_log_page(), expected_ranks=1)

    assert _check(result, "per-rank-attribution") is Verdict.INCONCLUSIVE
    assert result.verdict is Verdict.INCONCLUSIVE


def test_an_absent_backend_is_distinguished_from_an_empty_page() -> None:
    """Both are inconclusive; only one is a reason to stop running the proof.

    "This backend has no structured logs" and "this workload produced no output"
    are the distinction the whole log path is built around -- a 501 rather than
    an empty page, so a follower cannot poll forever against something that will
    never answer. A verdict that collapsed them would throw that away at the last
    step, and asserting only the verdict cannot tell them apart.
    """

    absent = evaluate_structured_logs(
        _log_page([], unavailable=True, detail="does not expose structured logs"), expected_ranks=2
    )
    silent = evaluate_structured_logs(_log_page([]), expected_ranks=2)

    assert absent.verdict is Verdict.INCONCLUSIVE
    assert silent.verdict is Verdict.INCONCLUSIVE
    assert "does not expose structured logs" in absent.checks[0].detail
    assert "does not expose structured logs" not in silent.checks[0].detail


# ------------------------------------------------------- P9: the plane, not a person


def _succeeded_workload(**overrides):
    workload = {
        "id": "wl-1",
        "state": "succeeded",
        "worker_count": 2,
        "required_labels": {"eyvan_pool": "research/v6e-8-us-central2-b"},
    }
    workload.update(overrides)
    return workload


def _fleet_cell(pool_id: str = "research/v6e-8-us-central2-b"):
    return {"id": "cell-1", "pool_id": pool_id}


def test_p9_passes_when_the_plane_placed_and_carried_the_run() -> None:
    result = evaluate_control_plane_ran_it(_succeeded_workload(), [_fleet_cell()])

    assert result.verdict is Verdict.PASS


def test_p9_is_unknown_when_nothing_ran() -> None:
    """The vacuous-pass guard: an empty plane must not report a pass."""

    result = evaluate_control_plane_ran_it(_succeeded_workload(state="pending"), [])

    assert result.verdict is Verdict.INCONCLUSIVE


def test_p9_fails_a_run_that_used_no_fleet_capacity() -> None:
    """A job driven by hand leaves no cell of the plane's own behind.

    This is the criterion a bypass cannot satisfy, and the reason the proof
    exists: a whole day of hand-run accelerator time read like evidence about
    the control plane while telling us nothing about it.
    """

    result = evaluate_control_plane_ran_it(_succeeded_workload(), [_fleet_cell(pool_id="someone-elses-pool")])

    assert result.verdict is Verdict.FAIL


def test_p9_fails_a_workload_that_was_never_bound_to_a_pool() -> None:
    result = evaluate_control_plane_ran_it(_succeeded_workload(required_labels={}), [_fleet_cell()])

    assert result.verdict is Verdict.FAIL


def test_a_secret_redelivered_to_a_second_attempt_passes() -> None:
    """Validate secret issuance and reads per attempt so redelivery on retry remains valid."""

    result = evaluate_secret_delivery(
        [
            _event("secret.issued", attempt=1),
            _event("secret.read", attempt=1),
            _event("secret.issued", attempt=2),
            _event("secret.read", attempt=2),
        ],
        {"id": "wl-1", "spec": {"secret_mounts": {"HF_TOKEN": "gcp-secret://p/t"}}},
        "",
        secret_value=_VALUE,
    )

    assert result.verdict is Verdict.PASS
    assert _check(result, "redelivered") is Verdict.PASS
    assert "2 attempt(s)" in _detail(result, "one-read")


def test_a_second_read_within_one_attempt_still_fails_when_other_attempts_are_clean() -> None:
    result = evaluate_secret_delivery(
        [
            _event("secret.issued", attempt=1),
            _event("secret.read", attempt=1),
            _event("secret.issued", attempt=2),
            _event("secret.read", attempt=2),
            _event("secret.read", attempt=2),
        ],
        {"id": "wl-1"},
        "",
        secret_value=_VALUE,
    )

    assert _check(result, "one-read") is Verdict.FAIL
    assert "attempt(s) 2" in _detail(result, "one-read")


def test_a_cooldown_left_by_an_earlier_run_does_not_decide_this_one() -> None:
    """Timestamp filtering cannot turn current inventory into run-linked event history."""

    workload = {**_preempted(), "created_at": "2026-09-24T17:22:00Z"}
    result = evaluate_spot_preemption(
        workload,
        [
            {
                "id": "earlier-run",
                "pool_id": "research/v5p",
                "provider_failure_code": "preempted",
                "updated_at": "2026-09-24T16:52:00Z",
                "availability": {"cooldown_until": "2026-09-24T16:51:07Z", "interruptions": 1.0},
            },
            {
                "id": "this-run",
                "pool_id": "research/v5p",
                "provider_failure_code": "preempted",
                "updated_at": "2026-09-24T17:28:00Z",
                "availability": {"interruptions": 1.0},
            },
        ],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)
    assert "this-run" not in _detail(result, "no-stockout-cooldown")


def test_an_updated_at_during_this_run_does_not_attribute_a_cooldown() -> None:
    workload = {**_preempted(), "created_at": "2026-09-24T17:22:00Z"}
    result = evaluate_spot_preemption(
        workload,
        [
            {
                "id": "this-run",
                "pool_id": "research/v5p",
                "provider_failure_code": "preempted",
                "updated_at": "2026-09-24T17:28:00Z",
                "availability": {"cooldown_until": "2026-09-24T17:29:00Z"},
            }
        ],
        [{"id": "research/v5p"}],
    )

    _assert_unattributed_preemption(result)
