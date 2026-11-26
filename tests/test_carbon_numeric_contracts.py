# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Carbon reporting cannot turn nonfinite inputs into measured or fresh data."""

from datetime import UTC, datetime, timedelta

import pytest

from eyvan.domain.carbon import (
    CarbonIntensity,
    CarbonUnavailable,
    CarbonUnavailableReason,
    fresh_reading,
)

NOW = datetime(2026, 9, 12, tzinfo=UTC)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_intensity_is_not_a_reading(value):
    with pytest.raises(ValueError):
        CarbonIntensity("test-region", value, "test-source", NOW)


@pytest.mark.parametrize("value", [0.0, 1e-12, 410.0, 1e308])
def test_finite_nonnegative_intensity_remains_exact(value):
    reading = CarbonIntensity("test-region", value, "test-source", NOW)
    assert reading.grams_co2e_per_kwh == value
    assert fresh_reading(reading, now=NOW) is reading


@pytest.mark.parametrize("window", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_window_cannot_admit_an_old_reading(window):
    reading = CarbonIntensity("test-region", 410.0, "test-source", NOW - timedelta(days=7))
    with pytest.raises(ValueError) as error:
        fresh_reading(reading, now=NOW, max_age_s=window)
    assert "window" in str(error.value).lower()


@pytest.mark.parametrize("window", [0, -1])
def test_nonpositive_window_still_fails(window):
    reading = CarbonIntensity("test-region", 410.0, "test-source", NOW)
    with pytest.raises(ValueError, match="positive"):
        fresh_reading(reading, now=NOW, max_age_s=window)


@pytest.mark.parametrize("window", [float("nan"), float("inf"), 0])
def test_unavailable_reading_preserves_its_reason_without_a_window(window):
    absent = CarbonUnavailable(CarbonUnavailableReason.PROVIDER_ERROR, "source unavailable")
    assert fresh_reading(absent, now=NOW, max_age_s=window) is absent


@pytest.mark.parametrize(
    ("age", "available"),
    [(-60.000001, False), (-60, True), (0, True), (3600, True), (3600.000001, False)],
)
def test_freshness_limits_remain_inclusive(age, available):
    reading = CarbonIntensity("test-region", 410.0, "test-source", NOW - timedelta(seconds=age))
    result = fresh_reading(reading, now=NOW)
    if available:
        assert result is reading
    else:
        assert isinstance(result, CarbonUnavailable)
        assert result.reason is CarbonUnavailableReason.STALE


@pytest.mark.parametrize("window", [1e-6, 0.5, 1.0])
def test_small_positive_windows_accept_current_readings(window):
    reading = CarbonIntensity("test-region", 410.0, "test-source", NOW)
    assert fresh_reading(reading, now=NOW, max_age_s=window) is reading


def test_future_diagnostic_reports_positive_clock_skew():
    reading = CarbonIntensity("test-region", 410.0, "test-source", NOW + timedelta(seconds=120))
    result = fresh_reading(reading, now=NOW)
    assert isinstance(result, CarbonUnavailable)
    assert result.reason is CarbonUnavailableReason.STALE
    assert "120s in the future" in result.detail
    assert "-120s" not in result.detail
