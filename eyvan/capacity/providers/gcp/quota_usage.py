# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read-only Cloud Monitoring usage for Cloud TPU quota metrics."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import quote

import httpx

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.subprocess_env import provider_cli_env

_MONITORING_TIME_SERIES_URL = "https://monitoring.googleapis.com/v3/projects/{project}/timeSeries"
_QUOTA_USAGE_METRIC = "serviceruntime.googleapis.com/quota/allocation/usage"
_CONSUMER_QUOTA_RESOURCE = "consumer_quota"
_TPU_QUOTA_SERVICE = "tpu.googleapis.com"
#: Cloud Monitoring writes an allocation-usage point when usage changes and can
#: then stay silent for weeks while it does not. A short lookback can therefore
#: report unchanged usage as unknown. The newest point in the window is still
#: the current value; its own timestamp travels with it, so
#: callers keep seeing how old the reading is.
_DEFAULT_LOOKBACK = timedelta(days=90)


class GcpQuotaUsageError(RuntimeError):
    """Raised when Eyvan cannot read Cloud Monitoring quota usage safely."""


TokenRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Clock = Callable[[], datetime]


def _token_runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run the installed gcloud CLI only to obtain an active access token."""

    timeout_s = max(float(os.getenv("EYVAN_GCLOUD_QUOTA_TIMEOUT_S", "60")), 1.0)
    return subprocess.run(
        ["gcloud", *args], check=False, capture_output=True, text=True, timeout=timeout_s, env=provider_cli_env()
    )


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time for Monitoring intervals."""

    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class TpuQuotaUsage:
    """One latest Cloud Monitoring allocation-usage point for a quota location."""

    metric: str
    location: str
    used_quota_units: int
    observed_at: datetime

    def __post_init__(self) -> None:
        if not self.metric or self.metric != self.metric.strip():
            raise ValueError("metric must be a non-empty trimmed string")
        if not self.location or self.location != self.location.strip():
            raise ValueError("location must be a non-empty trimmed string")
        if isinstance(self.used_quota_units, bool) or not isinstance(self.used_quota_units, int):
            raise ValueError("used_quota_units must be an integer")
        if self.used_quota_units < 0:
            raise ValueError("used_quota_units must be non-negative")
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))


#: Metric reads run at once. One per generation per purchase class, so this is
#: the whole set for any real project rather than a sample of it.
_DEFAULT_CONCURRENCY = 16


class GcpTpuQuotaUsageDiscovery:
    """Read current TPU quota usage from the Cloud Monitoring REST API.

    Cloud Quotas exposes effective limits but not remaining balance. This
    reader obtains the matching allocation-usage gauge through the active
    ``gcloud`` identity without creating or changing cloud resources.
    """

    def __init__(
        self,
        *,
        token_runner: TokenRunner = _token_runner,
        client: httpx.Client | None = None,
        clock: Clock = _utc_now,
        lookback: timedelta = _DEFAULT_LOOKBACK,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        if lookback <= timedelta(0):
            raise ValueError("lookback must be positive")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        self._token_runner = token_runner
        self._client = client
        self._clock = clock
        self._lookback = lookback
        self._concurrency = concurrency

    def discover(self, *, project: str, metrics: Iterable[str]) -> tuple[TpuQuotaUsage, ...]:
        """Return the latest valid allocation usage by metric and location.

        A missing time series intentionally remains absent from the result.
        The caller must treat that as unknown rather than assuming zero usage.
        """

        normalized_project = project.strip()
        if not normalized_project:
            raise ValueError("project must be a non-empty trimmed string")
        requested_metrics = tuple(sorted({metric.strip() for metric in metrics if metric.strip()}))
        if not requested_metrics:
            return ()
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        now = now.astimezone(UTC)
        token = self._access_token()
        client = self._client or httpx.Client(timeout=30.0)
        owns_client = self._client is None
        try:
            records: dict[tuple[str, str], TpuQuotaUsage] = {}

            # One round trip per metric, run together. Sequentially this was the
            # largest remaining cost of a quota read: eight generations across
            # two purchase classes is sixteen serial round trips to Cloud
            # Monitoring, each a few seconds, for readings that do not depend on
            # each other. The merge below already tolerates any arrival order.
            def read(metric: str) -> tuple[TpuQuotaUsage, ...]:
                return tuple(
                    self._metric_usage(
                        client=client,
                        project=normalized_project,
                        metric=metric,
                        token=token,
                        start=now - self._lookback,
                        end=now,
                    )
                )

            with ThreadPoolExecutor(max_workers=min(self._concurrency, len(requested_metrics))) as pool:
                for batch in pool.map(read, requested_metrics):
                    for usage in batch:
                        key = (usage.metric, usage.location)
                        existing = records.get(key)
                        if existing is None or usage.observed_at > existing.observed_at:
                            records[key] = usage
            return tuple(sorted(records.values(), key=lambda usage: (usage.metric, usage.location)))
        finally:
            if owns_client:
                client.close()

    def _access_token(self) -> str:
        """Return the active gcloud OAuth token or a concise diagnostic."""

        command = ("auth", "print-access-token")
        try:
            result = self._token_runner(command)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GcpQuotaUsageError(f"gcloud auth token lookup failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise GcpQuotaUsageError(f"gcloud auth token lookup failed: {detail or result.returncode}")
        token = result.stdout.strip()
        if not token:
            raise GcpQuotaUsageError("gcloud auth token lookup returned no access token")
        return token

    def _metric_usage(
        self,
        *,
        client: httpx.Client,
        project: str,
        metric: str,
        token: str,
        start: datetime,
        end: datetime,
    ) -> tuple[TpuQuotaUsage, ...]:
        """Read every Monitoring page for one exact Cloud TPU quota metric."""

        endpoint = _MONITORING_TIME_SERIES_URL.format(project=quote(project, safe=""))
        filter_expression = (
            f'metric.type="{_QUOTA_USAGE_METRIC}" '
            f'AND resource.type="{_CONSUMER_QUOTA_RESOURCE}" '
            f'AND resource.labels.service="{_TPU_QUOTA_SERVICE}" '
            f'AND metric.labels.quota_metric="{metric}"'
        )
        page_token: str | None = None
        records: list[TpuQuotaUsage] = []
        while True:
            params = {
                "filter": filter_expression,
                "interval.startTime": _format_timestamp(start),
                "interval.endTime": _format_timestamp(end),
                "pageSize": "1000",
                "view": "FULL",
            }
            if page_token is not None:
                params["pageToken"] = page_token
            document = self._request_page(client=client, endpoint=endpoint, params=params, token=token)
            records.extend(_parse_usage_page(document, project=project, metric=metric))
            next_token = document.get("nextPageToken")
            if not isinstance(next_token, str) or not next_token:
                return tuple(records)
            page_token = next_token

    @staticmethod
    def _request_page(
        *,
        client: httpx.Client,
        endpoint: str,
        params: dict[str, str],
        token: str,
    ) -> JsonObject:
        """Fetch and validate one Cloud Monitoring JSON page."""

        try:
            response = client.get(endpoint, params=params, headers={"Authorization": f"Bearer {token}"})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GcpQuotaUsageError(f"Cloud Monitoring quota usage request failed: {exc}") from exc
        try:
            value = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise GcpQuotaUsageError("Cloud Monitoring quota usage request returned invalid JSON") from exc
        if not is_json_object(value):
            raise GcpQuotaUsageError("Cloud Monitoring quota usage request returned an invalid JSON document")
        return value


def _parse_usage_page(document: JsonObject, *, project: str, metric: str) -> tuple[TpuQuotaUsage, ...]:
    """Extract one latest valid point per location from a Monitoring page."""

    raw_series = document.get("timeSeries")
    if not isinstance(raw_series, list):
        return ()
    records: list[TpuQuotaUsage] = []
    for series in raw_series:
        if not is_json_object(series):
            continue
        location = _series_location(series, project=project, metric=metric)
        if location is None:
            continue
        point = _latest_usage_point(series)
        if point is None:
            continue
        used_quota_units, observed_at = point
        records.append(
            TpuQuotaUsage(
                metric=metric,
                location=location,
                used_quota_units=used_quota_units,
                observed_at=observed_at,
            )
        )
    return tuple(records)


def _series_location(series: JsonObject, *, project: str, metric: str) -> str | None:
    """Return a location only for the exact requested consumer-quota series."""

    raw_metric = series.get("metric")
    raw_resource = series.get("resource")
    if not is_json_object(raw_metric) or not is_json_object(raw_resource):
        return None
    if raw_metric.get("type") != _QUOTA_USAGE_METRIC:
        return None
    metric_labels = raw_metric.get("labels")
    if not is_json_object(metric_labels) or metric_labels.get("quota_metric") != metric:
        return None
    if raw_resource.get("type") != _CONSUMER_QUOTA_RESOURCE:
        return None
    resource_labels = raw_resource.get("labels")
    if not is_json_object(resource_labels):
        return None
    if resource_labels.get("service") != _TPU_QUOTA_SERVICE:
        return None
    if resource_labels.get("project_id") != project:
        return None
    location = resource_labels.get("location")
    return location if isinstance(location, str) and location.strip() == location and location else None


def _latest_usage_point(series: JsonObject) -> tuple[int, datetime] | None:
    """Return the newest parsed non-negative INT64 point in one time series."""

    raw_points = series.get("points")
    if not isinstance(raw_points, list):
        return None
    latest: tuple[int, datetime] | None = None
    for raw_point in raw_points:
        if not is_json_object(raw_point):
            continue
        observed_at = _point_timestamp(raw_point)
        used_quota_units = _point_value(raw_point)
        if observed_at is None or used_quota_units is None:
            continue
        if latest is None or observed_at > latest[1]:
            latest = (used_quota_units, observed_at)
    return latest


def _point_timestamp(point: JsonObject) -> datetime | None:
    """Parse one Cloud Monitoring point end timestamp as aware UTC."""

    interval = point.get("interval")
    if not is_json_object(interval):
        return None
    raw = interval.get("endTime")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = datetime.fromisoformat(f"{raw[:-1]}+00:00" if raw.endswith("Z") else raw)
    except ValueError:
        return None
    return value.astimezone(UTC) if value.tzinfo is not None else None


def _point_value(point: JsonObject) -> int | None:
    """Parse one Cloud Monitoring non-negative integer allocation usage value."""

    raw_value = point.get("value")
    if not is_json_object(raw_value):
        return None
    raw = raw_value.get("int64Value")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _format_timestamp(value: datetime) -> str:
    """Format a timezone-aware UTC timestamp for the Monitoring API."""

    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ("GcpQuotaUsageError", "GcpTpuQuotaUsageDiscovery", "TpuQuotaUsage")
