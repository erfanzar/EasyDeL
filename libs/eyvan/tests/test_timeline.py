# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import UTC


def test_dispatch_duration_is_named_where_the_stage_happened() -> None:
    """'dispatched' without its duration is a timestamp, not an answer."""

    from datetime import datetime, timedelta

    from eyvan.records import AuditEvent
    from eyvan.sdk.timeline import timeline

    base = datetime(2026, 8, 24, 9, 0, tzinfo=UTC)
    events = [
        AuditEvent(
            id="evt_1",
            actor="erfan",
            entity_type="workload",
            entity_id="wl_1",
            action="workload.dispatch_reserved",
            new_state="leased",
            timestamp=base,
        ),
        AuditEvent(
            id="evt_2",
            actor="system",
            entity_type="workload",
            entity_id="wl_1",
            action="workload.dispatched",
            previous_state="leased",
            new_state="dispatching",
            timestamp=base + timedelta(seconds=159),
            payload={"remote_id": "rw_1", "dispatch_elapsed_s": 158.734},
        ),
    ]

    from eyvan.sdk.timeline import render

    lines = [str(line) for line in render(timeline(events))]
    joined = "\n".join(lines)
    assert "159" in joined or "158" in joined
