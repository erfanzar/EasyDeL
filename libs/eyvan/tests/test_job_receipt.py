# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Terminal presentation tests for durable Eyvan workload controls."""

from __future__ import annotations

import io

import pytest

from eyvan.sdk.job_receipt import JobLifecycleState, JobLifecycleUpdate, JobSubmissionReceipt, JobTerminalOutput


class _OutputBuffer(io.StringIO):
    """In-memory output whose physical TTY capability is explicit."""

    def __init__(self, *, is_terminal: bool) -> None:
        super().__init__()
        self._is_terminal = is_terminal

    def isatty(self) -> bool:
        return self._is_terminal


_RECEIPT = JobSubmissionReceipt(
    workload_id="wl_123",
    fleet="research-us-central1-c",
    resource_shape="v6e-16 x 2 workers",
    atomic=True,
)


@pytest.mark.parametrize(
    ("is_terminal", "environment", "force_terminal", "rich_layout", "ansi"),
    (
        (True, {}, None, True, True),
        (False, {}, None, False, False),
        (True, {"NO_COLOR": "1"}, None, True, False),
        (False, {"FORCE_COLOR": "1"}, None, True, True),
        (True, {"NO_COLOR": "1", "FORCE_COLOR": "1"}, None, True, False),
        (False, {"NO_COLOR": "1", "FORCE_COLOR": "1"}, None, False, False),
        (False, {"NO_COLOR": "1", "FORCE_COLOR": "1"}, True, True, False),
        (True, {"FORCE_COLOR": "1"}, False, False, False),
    ),
    ids=(
        "tty",
        "redirected",
        "no-color-tty",
        "force-color-redirected",
        "conflicting-color-tty",
        "conflicting-color-redirected",
        "explicit-terminal-overrides-layout",
        "explicit-plain-overrides-force-color",
    ),
)
def test_receipt_rendering_has_deterministic_terminal_and_color_precedence(
    is_terminal: bool,
    environment: dict[str, str],
    force_terminal: bool | None,
    rich_layout: bool,
    ansi: bool,
) -> None:
    stream = _OutputBuffer(is_terminal=is_terminal)

    JobTerminalOutput(
        stream=stream,
        force_terminal=force_terminal,
        environment=environment,
    ).submission(_RECEIPT)

    rendered = stream.getvalue()
    assert ("Eyvan job submitted" in rendered) is rich_layout
    assert ("\x1b[" in rendered) is ansi
    if rich_layout:
        assert "State" in rendered
        assert "Submitted" in rendered
        assert "Eyvan: job wl_123 submitted" not in rendered
    else:
        assert rendered.splitlines() == list(_RECEIPT.plain_lines)


def test_terminal_receipt_uses_a_compact_colored_panel_and_lifecycle_labels() -> None:
    stream = io.StringIO()
    output = JobTerminalOutput(stream=stream, force_terminal=True, environment={})

    output.submission(_RECEIPT)
    output.lifecycle(
        JobLifecycleUpdate(
            JobLifecycleState.BLOCKED,
            "v6e-16 x 2 workers | us-central1-c",
            "GCP denied queue admission; no queued resource was created",
            "PERMISSION_DENIED: queue admission policy rejected this request",
        )
    )

    rendered = stream.getvalue()
    assert "\x1b[" in rendered
    assert "Eyvan job submitted" in rendered
    assert "v6e-16 x 2 workers" in rendered
    assert "eyvan workload status wl_123" in rendered
    assert "eyvan workload logs wl_123" in rendered
    assert "eyvan workload cancel wl_123" in rendered
    assert "BLOCKED" in rendered
    assert "GCP denied queue admission; no queued resource was created" in rendered


def test_writer_sink_remains_plain_even_when_terminal_color_is_forced() -> None:
    lines: list[str] = []

    JobTerminalOutput(
        writer=lines.append,
        force_terminal=True,
        environment={"FORCE_COLOR": "1"},
    ).submission(_RECEIPT)

    assert lines == list(_RECEIPT.plain_lines)


def test_waiting_states_redraw_one_line_instead_of_stacking_a_block_each() -> None:
    """Waiting for a spot TPU must not bury the run in its own status.

    A job sits in requesting/queued/provisioning/connecting for minutes with no
    output of its own, and each change printed a fresh two-line block. A run
    that eventually started had a screenful of its own waiting above it, which
    reads as breakage rather than progress. The waiting line is redrawn in
    place; anything that is not waiting settles it and prints normally.
    """

    stream = io.StringIO()
    output = JobTerminalOutput(stream=stream, force_terminal=True, environment={})

    for state in (JobLifecycleState.QUEUED, JobLifecycleState.PROVISIONING, JobLifecycleState.CONNECTING):
        output.lifecycle(JobLifecycleUpdate(state, "v6e-8 | us-central1-a", f"waiting on {state}", None))
    # A live region paints on refresh or on stop, so read it after it settles --
    # which is what the next non-waiting event does.
    output.lifecycle(JobLifecycleUpdate(JobLifecycleState.RUNNING, "v6e-8 | us-central1-a", "started", None))
    rendered = stream.getvalue()

    # The last thing it was waiting on survives: how a run got here is part of
    # the story, so the live display is not transient.
    assert "connecting" in rendered.lower()
    # The earlier waiting summaries were redrawn over, not each given a line.
    assert "waiting on queued" not in rendered
    assert "waiting on provisioning" not in rendered
    # And a real event lands in scrollback rather than inside the live region.
    assert "started" in rendered


def test_settle_is_safe_to_call_when_nothing_is_being_drawn() -> None:
    """The follower settles on every exit path, including ones that never waited."""

    output = JobTerminalOutput(stream=io.StringIO(), force_terminal=True, environment={})

    output.settle()
    output.settle()
