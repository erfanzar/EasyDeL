# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import subprocess
import typing as tp
from collections.abc import Sequence
from datetime import datetime

import httpx
import pytest

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject
from eyvan.capacity.providers.gcp.quota_usage import GcpQuotaUsageError, GcpTpuQuotaUsageDiscovery

_METRIC = "tpu.googleapis.com/tpu-v5p-preemptible"


def _token_runner(_command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["gcloud"], 0, "test-token\n", "")


def _series(*, location: str, points: list[JsonObject], metric: str = _METRIC) -> JsonObject:
    return {
        "metric": {
            "type": "serviceruntime.googleapis.com/quota/allocation/usage",
            "labels": {"quota_metric": metric},
        },
        "resource": {
            "type": "consumer_quota",
            "labels": {
                "location": location,
                "project_id": "project-a",
                "service": "tpu.googleapis.com",
            },
        },
        "points": points,
    }


def _point(value: str, timestamp: str) -> JsonObject:
    return {"interval": {"endTime": timestamp}, "value": {"int64Value": value}}


def test_quota_usage_discovery_keeps_region_and_zone_usage_separate_and_follows_pages() -> None:
    requests: list[httpx.Request] = []
    pages: dict[str | None, JsonObject] = {
        None: {
            "nextPageToken": "second-page",
            "timeSeries": [
                _series(
                    location="us-east5-a",
                    points=[
                        _point("2048", "2026-07-15T00:01:03Z"),
                        _point("1920", "2026-07-14T23:50:00Z"),
                    ],
                )
            ],
        },
        "second-page": {
            "timeSeries": [
                _series(location="us-east5", points=[_point("2048", "2026-07-15T00:01:03Z")]),
            ]
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=pages[request.url.params.get("pageToken")])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        records = GcpTpuQuotaUsageDiscovery(
            token_runner=_token_runner,
            client=client,
            clock=lambda: datetime(2026, 7, 15, 1, tzinfo=UTC),
        ).discover(project="project-a", metrics=(_METRIC,))
    finally:
        client.close()

    assert [(record.location, record.used_quota_units) for record in records] == [
        ("us-east5", 2048),
        ("us-east5-a", 2048),
    ]
    assert all(record.observed_at == datetime(2026, 7, 15, 0, 1, 3, tzinfo=UTC) for record in records)
    assert len(requests) == 2
    assert requests[0].headers["Authorization"] == "Bearer test-token"
    assert requests[0].url.params["view"] == "FULL"
    assert requests[0].url.params["filter"].endswith(f'metric.labels.quota_metric="{_METRIC}"')


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_an_idle_quota_keeps_its_last_usage_point_weeks_after_it_was_written() -> None:
    """Monitoring writes a usage point when usage changes, not while it stays put.

    A short query window can omit a valid older reading from an idle quota.
    The peer below honours the requested interval as Cloud Monitoring does.
    """

    stored = _series(location="us-east5-b", points=[_point("0", "2026-10-01T10:28:03Z")])

    def handler(request: httpx.Request) -> httpx.Response:
        start = _parse_time(request.url.params["interval.startTime"])
        end = _parse_time(request.url.params["interval.endTime"])
        points = [
            point
            for point in tp.cast(list[JsonObject], stored["points"])
            if start <= _parse_time(tp.cast(str, tp.cast(JsonObject, point["interval"])["endTime"])) <= end
        ]
        return httpx.Response(200, json={"timeSeries": [{**stored, "points": points}] if points else []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        records = GcpTpuQuotaUsageDiscovery(
            token_runner=_token_runner,
            client=client,
            clock=lambda: datetime(2026, 10, 23, 12, tzinfo=UTC),
        ).discover(project="project-a", metrics=(_METRIC,))
    finally:
        client.close()

    # The reading keeps its real timestamp, so every caller still sees its age.
    assert [(record.location, record.used_quota_units, record.observed_at) for record in records] == [
        ("us-east5-b", 0, datetime(2026, 10, 1, 10, 28, 3, tzinfo=UTC)),
    ]


def test_quota_usage_discovery_ignores_invalid_series_instead_of_inventing_zero_usage() -> None:
    document = {
        "timeSeries": [
            _series(location="us-east5-a", points=[_point("-1", "2026-07-15T00:01:03Z")]),
            _series(location="us-east5-b", points=[_point("not-an-int", "2026-07-15T00:01:03Z")]),
            _series(
                location="us-east5-c",
                metric="tpu.googleapis.com/tpu-v5p",
                points=[_point("0", "2026-07-15T00:01:03Z")],
            ),
            {
                **_series(location="us-east5-d", points=[_point("0", "2026-07-15T00:01:03Z")]),
                "resource": {"type": "global", "labels": {}},
            },
        ]
    }

    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=document)))
    try:
        records = GcpTpuQuotaUsageDiscovery(token_runner=_token_runner, client=client).discover(
            project="project-a", metrics=(_METRIC,)
        )
    finally:
        client.close()

    assert records == ()


def test_quota_usage_discovery_reports_monitoring_failures_without_leaking_credentials() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(403, text="denied")))
    try:
        with pytest.raises(GcpQuotaUsageError, match="Cloud Monitoring quota usage request failed") as raised:
            GcpTpuQuotaUsageDiscovery(token_runner=_token_runner, client=client).discover(
                project="project-a", metrics=(_METRIC,)
            )
    finally:
        client.close()

    assert "test-token" not in str(raised.value)
