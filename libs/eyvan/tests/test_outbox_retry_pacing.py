# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Back off and report durable commands whose transient failures persist.

Transient failures must not spend the workload's retry budget, but an unchanged
attempt number must not permit once-per-second retries and identical warnings
forever.
"""

from __future__ import annotations

import pytest

from eyvan.server.controller.outbox import command_is_stuck, transient_retry_delay_s


def test_a_blip_is_retried_immediately() -> None:
    """The case the give-the-attempt-back rule exists for stays fast."""

    assert transient_retry_delay_s(0.0) == 1
    assert transient_retry_delay_s(5.0) == 1


def test_a_command_failing_for_minutes_backs_off() -> None:
    """Once a second forever is a load problem and a log-flood, not patience."""

    assert transient_retry_delay_s(60.0) > transient_retry_delay_s(5.0)
    assert transient_retry_delay_s(600.0) > transient_retry_delay_s(60.0)


def test_the_backoff_is_monotonic() -> None:
    """Age only grows, so the wait may never shrink under it."""

    delays = [transient_retry_delay_s(age) for age in (0, 15, 30, 90, 300, 900, 3600)]

    assert delays == sorted(delays)


def test_the_backoff_is_capped_low_enough_to_notice_recovery() -> None:
    """Something is waiting on these -- a cancel, a dispatch -- so an hour-old
    command still retries about once a minute rather than once an hour."""

    assert transient_retry_delay_s(3_600.0) == transient_retry_delay_s(86_400.0)
    assert transient_retry_delay_s(86_400.0) <= 60


@pytest.mark.parametrize("age", [0.0, 60.0, 299.0])
def test_a_recent_failure_is_routine(age: float) -> None:
    assert command_is_stuck(age) is False


@pytest.mark.parametrize("age", [300.0, 1_200.0])
def test_a_command_failing_this_long_is_not_routine(age: float) -> None:
    """An operator reading warnings at one a second will not notice one more.

    The level is what makes the twentieth minute look different from the first.
    """

    assert command_is_stuck(age) is True


def test_a_negative_age_is_treated_as_new() -> None:
    """Clock skew between replicas must not produce a negative wait."""

    assert transient_retry_delay_s(-10.0) == 1
