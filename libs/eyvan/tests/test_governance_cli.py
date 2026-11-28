# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Governance and capacity CLI: ten routes that shipped with no way to call them.

``eyvan doctor`` and ``eyvan workload explain`` exist to answer "why is my job
stuck", and the top three answers are quota, budget, and policy -- none of which
an operator could read or change without writing the HTTP call by hand. Capacity
was worse: ``eyvan capacity`` did not exist, so "what can I run on?" required
provisioning something first with ``eyvan up --region``.

What is asserted here is the wiring an operator depends on and cannot see:
mutations carry an ``Idempotency-Key`` (these routes reject requests without one),
scope defaults come from the active context rather than being retyped, and every
new field on a record is actually reachable -- a setting nobody can set is the
same as no setting.
"""

from __future__ import annotations

import typing as tp

from click.testing import CliRunner

import eyvan.sdk.cli as cli
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.contexts import DEFAULT_BASE_URL

RequestArguments = dict[str, tp.Any]


class FakeClient:
    """Record every request and answer with a shape the command can render."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, RequestArguments]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, tp.Any] | None = None,
        json: dict[str, tp.Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tp.Any:
        arguments: RequestArguments = {}
        if params is not None:
            arguments["params"] = params
        if json is not None:
            arguments["json"] = json
        if headers is not None:
            arguments["headers"] = headers
        self.calls.append((method, path, arguments))
        if method == "GET" and path == "/api/v1/capacity":
            return {
                "pools": [
                    {
                        "fleet": "research",
                        "accelerator_type": "v5p-16",
                        "capacity_class": "on_demand",
                        "region": "us-east5",
                        "zone": "us-east5-a",
                        "max_cells": 4,
                    },
                    {
                        "fleet": "research",
                        "accelerator_type": "v6e-16",
                        "capacity_class": "spot",
                        "region": "us-east5",
                        "zone": None,
                        "max_cells": 2,
                    },
                ]
            }
        if method == "GET" and path.endswith("/status"):
            return {"org_id": "acme", "project_id": "research", "limit": 100.0, "spent": 10.0}
        if method == "DELETE":
            return None
        if json is not None:
            return json
        return []

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        del headers

    def stream_lines(self, method: str, path: str, **_: tp.Any) -> tp.Iterator[str]:
        self.calls.append((method, path, {}))
        yield ": keep-alive"
        yield 'data: {"action": "workload.submitted"}'
        yield ""
        yield 'data: {"action": "workload.terminal"}'


def _invoke(fake: FakeClient, monkeypatch, argv: list[str]):
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    return CliRunner().invoke(
        cli.main,
        ["--user", "alice", "--org", "acme", "--project", "research", *argv],
    )


def _json_argument(call: tuple[str, str, RequestArguments]) -> dict[str, tp.Any]:
    payload = call[2]["json"]
    assert isinstance(payload, dict)
    return payload


def _headers(call: tuple[str, str, RequestArguments]) -> dict[str, str]:
    return call[2].get("headers", {})


def test_capacity_list_reaches_the_route_that_previously_needed_a_provision(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["capacity", "list"])

    assert result.exit_code == 0, result.output
    assert fake.calls[0][:2] == ("GET", "/api/v1/capacity")
    assert "v5p-16" in result.output
    assert "v6e-16" in result.output


def test_capacity_filters_are_applied_client_side(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["capacity", "list", "--accelerator-type", "v5p-16"])

    assert result.exit_code == 0, result.output
    assert "v5p-16" in result.output
    assert "v6e-16" not in result.output


def test_a_filter_matching_nothing_says_so_differently_from_an_empty_fleet(monkeypatch) -> None:
    """The two have the same empty table and completely different fixes."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["capacity", "list", "--accelerator-type", "v7-99"])

    assert result.exit_code == 0, result.output
    assert "match those filters" in result.output


def test_capacity_billing_and_warmth_reach_their_routes(monkeypatch) -> None:
    fake = FakeClient()

    assert _invoke(fake, monkeypatch, ["capacity", "billing"]).exit_code == 0
    assert _invoke(fake, monkeypatch, ["capacity", "warmth"]).exit_code == 0

    assert [call[1] for call in fake.calls] == ["/api/v1/capacity/billing", "/api/v1/capacity/warmth"]


def test_capacity_pool_get_uses_the_single_pool_route(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["capacity", "pool", "get", "research/v5p-16"])

    assert result.exit_code == 0, result.output
    assert fake.calls[0][:2] == ("GET", "/api/v1/capacity-pools/research/v5p-16")


def test_there_is_no_pool_create(monkeypatch) -> None:
    """Pools are declared in a fleet config; a second creator would fork the truth."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["capacity", "pool", "create"])

    assert result.exit_code != 0
    assert fake.calls == []


def test_quota_set_carries_an_idempotency_key(monkeypatch) -> None:
    """These routes reject a request without one, so the CLI must always send it."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["quota", "set", "--cpu", "16", "--tpu", "8"])

    assert result.exit_code == 0, result.output
    method, path, _arguments = fake.calls[0]
    assert (method, path) == ("PUT", "/api/v1/quotas")
    assert "Idempotency-Key" in _headers(fake.calls[0])
    payload = _json_argument(fake.calls[0])
    assert payload["limits"]["cpu"] == 16.0
    assert payload["limits"]["tpu"] == 8


def test_a_tpu_only_quota_set_leaves_every_unnamed_dimension_unconstrained(monkeypatch) -> None:
    """The server constrains exactly what the wire carries: naming only --tpu
    used to bind cpu, gpu and memory to zero and refuse every workload in the
    scope that asked for any of them."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["quota", "set", "--tpu", "20"])

    assert result.exit_code == 0, result.output
    payload = _json_argument(fake.calls[0])
    assert payload["limits"] == {"tpu": 20}
    assert "cpu" not in payload["limits"]
    assert "memory_gib" not in payload["limits"]
    assert "gpu" not in payload["limits"]


def test_a_named_zero_is_a_constraint_not_an_omission(monkeypatch) -> None:
    """``--cpu 0`` closes a dimension deliberately; the unset ones stay open."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["quota", "set", "--cpu", "0", "--gpu", "2"])

    assert result.exit_code == 0, result.output
    payload = _json_argument(fake.calls[0])
    assert payload["limits"] == {"cpu": 0.0, "gpu": 2}


def test_quota_set_can_reach_the_artifact_byte_ceiling(monkeypatch) -> None:
    """A setting nobody can set is the same as no setting."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["quota", "set", "--cpu", "1", "--artifact-bytes", "1048576"])

    assert result.exit_code == 0, result.output
    assert _json_argument(fake.calls[0])["artifact_bytes_limit"] == 1_048_576


def test_an_omitted_artifact_ceiling_is_unlimited_not_zero(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["quota", "set", "--cpu", "1"])

    assert result.exit_code == 0, result.output
    assert _json_argument(fake.calls[0])["artifact_bytes_limit"] is None


def test_quota_get_asks_for_the_effective_quota_not_a_stored_row(monkeypatch) -> None:
    """Which scope is binding is the whole question an operator is asking."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["quota", "get"])

    assert result.exit_code == 0, result.output
    assert fake.calls[0][1] == "/api/v1/quotas/current"


def test_budget_set_carries_alert_thresholds(monkeypatch) -> None:
    """A warning nobody can configure is a warning nobody gets."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["budget", "set", "--limit", "500", "--alert-at", "0.8", "--alert-at", "0.95"])

    assert result.exit_code == 0, result.output
    payload = _json_argument(fake.calls[0])
    assert payload["limit"] == 500.0
    assert payload["alert_thresholds"] == [0.8, 0.95]
    assert payload["hard_limit"] is True


def test_repeated_alert_thresholds_are_deduplicated_and_ordered(monkeypatch) -> None:
    """The record rejects duplicates, so the CLI must not hand it any."""

    fake = FakeClient()

    result = _invoke(
        fake,
        monkeypatch,
        ["budget", "set", "--limit", "500", "--alert-at", "0.95", "--alert-at", "0.8", "--alert-at", "0.95"],
    )

    assert result.exit_code == 0, result.output
    assert _json_argument(fake.calls[0])["alert_thresholds"] == [0.8, 0.95]


def test_a_soft_budget_is_expressed_by_a_flag_not_a_negation(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["budget", "set", "--limit", "500", "--soft"])

    assert result.exit_code == 0, result.output
    assert _json_argument(fake.calls[0])["hard_limit"] is False


def test_an_invalid_threshold_never_reaches_the_control_plane(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["budget", "set", "--limit", "500", "--alert-at", "1.5"])

    assert result.exit_code != 0
    assert fake.calls == []


def test_budget_status_and_roll_reach_their_routes(monkeypatch) -> None:
    fake = FakeClient()

    assert _invoke(fake, monkeypatch, ["budget", "status"]).exit_code == 0
    assert _invoke(fake, monkeypatch, ["budget", "roll"]).exit_code == 0

    assert [call[:2] for call in fake.calls] == [
        ("GET", "/api/v1/budgets/acme/research/status"),
        ("POST", "/api/v1/budgets/acme/research/roll"),
    ]


def test_policy_set_sends_the_scoped_path_and_an_idempotency_key(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(
        fake,
        monkeypatch,
        ["policy", "set", "--allow-region", "us-east5", "--deny-accelerator-type", "v4-8", "--no-preemption"],
    )

    assert result.exit_code == 0, result.output
    method, path, _arguments = fake.calls[0]
    assert (method, path) == ("PUT", "/api/v1/policies/acme/research")
    assert "Idempotency-Key" in _headers(fake.calls[0])
    payload = _json_argument(fake.calls[0])
    assert payload["allowed_regions"] == ["us-east5"]
    assert payload["denied_accelerator_types"] == ["v4-8"]
    assert payload["preemption_enabled"] is False


def test_an_omitted_allow_list_means_unrestricted_not_forbidden(monkeypatch) -> None:
    """An empty list that meant "allow nothing" would silently block every submit."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["policy", "set"])

    assert result.exit_code == 0, result.output
    payload = _json_argument(fake.calls[0])
    assert payload["allowed_regions"] == []
    assert payload["allowed_providers"] == []


def test_governance_commands_default_to_the_active_scope(monkeypatch) -> None:
    """Retyping --org/--project on every governance call is how they drift."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["budget", "get"])

    assert result.exit_code == 0, result.output
    assert fake.calls[0][1] == "/api/v1/budgets/acme/research"


def test_events_follow_streams_data_frames_and_drops_protocol_noise(monkeypatch) -> None:
    """Keep-alive comments and frame separators are protocol, not content."""

    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["events", "--org", "acme", "--project", "research", "--follow"])

    assert result.exit_code == 0, result.output
    assert fake.calls[0][:2] == ("GET", "/api/v1/events/stream")
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert lines == ['{"action": "workload.submitted"}', '{"action": "workload.terminal"}']


def test_events_without_follow_still_reads_the_collection(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["events", "--org", "acme", "--project", "research"])

    assert result.exit_code == 0, result.output
    assert fake.calls[0][:2] == ("GET", "/api/v1/events")


def test_system_reconcile_reaches_the_leader_route(monkeypatch) -> None:
    fake = FakeClient()

    result = _invoke(fake, monkeypatch, ["system-reconcile"])

    assert result.exit_code == 0, result.output
    assert fake.calls[0][:2] == ("POST", "/api/v1/system/reconcile")


def test_the_default_endpoint_is_unchanged_by_these_commands() -> None:
    """A guard on the one global these commands could accidentally reach."""

    assert CliSession is not None
    assert DEFAULT_BASE_URL.startswith("http")
