# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What a status shows, and what it refuses to show.

`workload status` printed the durable record: sixty-odd keys, most of them null,
with the three facts somebody wants -- what state it is in, what it runs, and
why it is not running -- scattered between `autoscale_block_reason` and
`worker_resource_scope`. The record is the right thing to store and the wrong
thing to show.
"""

from __future__ import annotations

from eyvan.sdk.workload_summary import summarize

RECORD = {
    "id": "wl_ad755a46",
    "state": "failed",
    "attempt": 1,
    "created_by": "erfan",
    "created_at": "2026-08-19T11:21:26Z",
    "failure_reason": 'User does not have permission to submit requests into this queue for "v6e-16"',
    "autoscale_block_reason": "queue_admission_denied",
    "queue_admission_denial": {"accelerator_type": "v6e-16", "zone": "asia-east1-c"},
    "failure_count": 0,
    "preemption_count": 0,
    "spec": {
        "name": "__main__.dummy_execute",
        "accelerator_type": "v6e-16",
        "capacity_class": "spot",
        "cell_group_count": 1,
        "region": "asia-east1",
        "metadata": {"eyvan_pool": "research-asia-east1-c-v6e-16"},
        "gang": {"replicas": 1, "per_replica": {"cpu": 180.0, "memory_gib": 720.0, "tpu": 16, "gpu": 0}},
        "execution": {"kind": "callable", "sha256": "3fc27d137154e0dd2df58ef49cde6c05", "python": "3.13"},
    },
}


def _rendered(record: dict) -> str:
    return "\n".join(summarize(record))


def test_the_provider_sentence_leads_rather_than_eyvans_category_for_it() -> None:
    """`queue_admission_denied` is a label; the provider's sentence is the only
    line that says what to go and fix."""

    text = _rendered(RECORD)

    assert "does not have permission" in text
    assert "queue_admission_denied" not in text


def test_a_callable_is_identified_by_its_content_hash() -> None:
    """Once pickled, the hash is the code's only durable identity -- and the
    bucket path it happens to sit at today is not."""

    text = _rendered(RECORD)

    assert "sha256:3fc27d137154" in text
    assert "python 3.13" in text
    assert "gs://" not in text


def test_a_command_is_shown_as_itself() -> None:
    record = {**RECORD, "spec": {**RECORD["spec"], "execution": {"kind": "command", "command": "python train.py"}}}

    assert "python train.py" in _rendered(record)


def test_the_resources_admission_checks_are_shown_not_just_the_shape() -> None:
    """A pool matches on its resource vector, so "v6e-16" alone does not explain
    why a 720 GiB request did not fit somewhere."""

    text = _rendered(RECORD)

    assert "16 TPU" in text
    assert "180 CPU" in text
    assert "720 GiB" in text


def test_nothing_the_record_leaves_null_is_invented() -> None:
    """Sixty keys with nulls is what this replaced; printing "lease: None" for
    each would be the same defect with nicer spacing."""

    text = _rendered({"id": "wl-1", "state": "pending", "spec": {}})

    assert "None" not in text
    assert "null" not in text
    assert len(text.splitlines()) <= 3


def test_multislice_is_stated_because_it_is_what_an_atomic_request_means() -> None:
    record = {**RECORD, "spec": {**RECORD["spec"], "cell_group_count": 2}}

    assert "2, allocated together" in _rendered(record)


def test_an_unreadable_record_summarises_to_nothing_rather_than_guessing() -> None:
    """The caller falls back to the record, which beats printing a confident
    summary of something this did not understand."""

    from eyvan.sdk.workload_summary import render

    assert render("not a record") == ()
    assert render(None) == ()


def test_a_cross_zone_request_is_not_reported_as_its_first_choice() -> None:
    """Do not describe a cross-zone request's first-choice region as its actual placement."""

    record = {
        "id": "wl-x",
        "state": "running",
        "spec": {
            "region": "europe-west4",
            "metadata": {
                "eyvan_pool": "spot-all-v5p8-v5p8-europe-west4-b",
                "eyvan_pool_alternates": "spot-all-v5p8-v5p8-us-central1-a,spot-all-v5p8-v5p8-us-east5-a",
            },
        },
    }

    text = _rendered(record)

    assert "placed on" not in text
    assert "requested   spot-all-v5p8-v5p8-europe-west4-b or 2 alternate pool(s)" in text
    assert "region" not in text
