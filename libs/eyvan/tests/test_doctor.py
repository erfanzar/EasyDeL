# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan doctor` preflight: pass, warn, fail, and skip must stay distinct."""

from __future__ import annotations

import typing as tp

import httpx
import pytest

from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.doctor import CheckStatus, DoctorCommandService, DoctorReport
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContext, EyvanContextConfig


class _Store:
    """Context store stub; doctor never writes through it."""

    def load(self) -> EyvanContextConfig:  # pragma: no cover - not exercised
        return EyvanContextConfig()

    def save(self, config: EyvanContextConfig) -> None:  # pragma: no cover
        raise AssertionError("doctor must not mutate saved contexts")


class _FakeClient:
    """Minimal stand-in for the CLI HTTP client.

    ``DoctorCommandService`` only calls ``request`` and consumes parsed JSON,
    so routing through a real ``EyvanClient`` would add a transport with no
    injection point and test nothing extra.
    """

    def __init__(self, handler: tp.Callable[[str], tp.Any]) -> None:
        self._handler = handler

    def request(self, method: str, path: str, **_kwargs) -> tp.Any:
        del method
        return self._handler(path)


def _session(
    handler: tp.Callable[[str], tp.Any],
    *,
    active: EyvanContext | None = None,
    org: str | None = "acme",
) -> CliSession:
    config = EyvanContextConfig(contexts=(active,), active_name=active.name) if active else EyvanContextConfig()
    return CliSession(
        client=_FakeClient(handler),
        identity=ClientIdentity(),
        context_store=_Store(),
        context_config=config,
        base_url="http://test",
        org_id=org,
        project_id="research" if org else None,
        overrides=ContextOverrides(),
    )


def _healthy(controller: str = "active", fleets: tp.Any = None) -> tp.Callable[[str], tp.Any]:
    payload = [{"name": "research"}] if fleets is None else fleets

    def handler(path: str) -> tp.Any:
        if path.endswith("/health"):
            return {"status": "ok", "controller": controller}
        # An empty fleet is the healthy default here: these tests assert check
        # wiring, and price drift needs real cells to have anything to compare.
        if path.endswith("/cells") or path.endswith("/capacity-pools"):
            return []
        return payload

    return handler


def _unreachable(path: str) -> tp.Any:
    raise httpx.ConnectError("connection refused")


def _by_name(report: DoctorReport) -> dict[str, CheckStatus]:
    return {check.name: check.status for check in report.checks}


@pytest.fixture(autouse=True)
def _isolated_runtime_profile(tmp_path, monkeypatch) -> None:
    """Point the staged-wheel check at a profile that does not exist.

    Otherwise every test in this module reports differently depending on what
    the machine running it has staged or has running.
    """

    # Whether *this* machine happens to be running a stale control plane must
    # not decide a test's outcome either.
    monkeypatch.setattr("eyvan.sdk.commands.doctor.stale_components", lambda **_kwargs: ())
    # The local-preflight family (gcloud auth, secret-manager bind, artifact
    # store, runtime profile, port ownership) reads this machine's tooling
    # and home directory; without this, the expectations below would be a
    # description of whatever machine the suite happens to run on. No gcloud
    # and no profile is the deterministic baseline.
    monkeypatch.setattr("eyvan.sdk.commands.doctor_preflights.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "eyvan.sdk.commands.doctor_preflights.DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH",
        tmp_path / "no-runtime-profile.json",
    )


def test_healthy_control_plane_with_one_fleet_passes_everything() -> None:
    report = DoctorCommandService(_session(_healthy())).report()

    assert report.ok is True
    assert _by_name(report) == {
        # A default-host preflight: a loopback server requires nothing beyond
        # what is already here. tests/test_startup_preflight.py covers the
        # network bind this check exists for.
        "server-startup-settings": CheckStatus.PASS,
        # A loopback development plane trusts its own default project, so the
        # default bind this suite uses passes rather than warning. The network
        # bind it exists for is covered below.
        "callable-trust": CheckStatus.PASS,
        # The local preflight family, all in their not-applicable state under
        # this fixture: no gcloud and no runtime profile staged. Each check's
        # real verdicts live in tests/test_doctor_local_preflights.py.
        "gcloud-authentication": CheckStatus.PASS,
        "secret-manager-bind-permission": CheckStatus.PASS,
        "artifact-store": CheckStatus.PASS,
        "runtime-profile": CheckStatus.PASS,
        "control-plane-port": CheckStatus.PASS,
        "control-plane-url": CheckStatus.PASS,
        "control-plane-reachable": CheckStatus.PASS,
        "controller-mode": CheckStatus.PASS,
        "identity-scope": CheckStatus.PASS,
        "submittable-fleet": CheckStatus.PASS,
        "budget-pricing": CheckStatus.PASS,
        "price-drift": CheckStatus.PASS,
        "pool-schedulability": CheckStatus.PASS,
        "kueue-preconditions": CheckStatus.PASS,
        "native-worker-route": CheckStatus.PASS,
        "source-drift": CheckStatus.PASS,
    }


def test_unreachable_control_plane_fails_once_and_skips_the_rest() -> None:
    """Derived failures would bury the single real cause."""

    report = DoctorCommandService(_session(_unreachable)).report()

    statuses = _by_name(report)
    assert report.ok is False
    assert statuses["control-plane-reachable"] is CheckStatus.FAIL
    assert statuses["identity-scope"] is CheckStatus.SKIP
    assert statuses["submittable-fleet"] is CheckStatus.SKIP


def test_api_only_replica_warns_rather_than_fails() -> None:
    """An API-only replica serves reads; it just will not reconcile."""

    report = DoctorCommandService(_session(_healthy(controller="api_only"))).report()

    assert report.ok is True
    assert _by_name(report)["controller-mode"] is CheckStatus.WARN
    remedy = next(check.remedy for check in report.checks if check.name == "controller-mode")
    assert "EYVAN_CONTROLLER_ENABLED" in remedy


def test_no_applied_fleet_is_a_failure_with_an_actionable_remedy() -> None:
    report = DoctorCommandService(_session(_healthy(fleets=[]))).report()

    assert report.ok is False
    check = next(check for check in report.checks if check.name == "submittable-fleet")
    assert check.status is CheckStatus.FAIL
    assert "eyvan up" in check.remedy


def test_multiple_fleets_without_a_selection_warns() -> None:
    handler = _healthy(fleets=[{"name": "a"}, {"name": "b"}])

    report = DoctorCommandService(_session(handler)).report()

    assert report.ok is True
    assert _by_name(report)["submittable-fleet"] is CheckStatus.WARN


def test_missing_scope_warns_without_failing() -> None:
    report = DoctorCommandService(_session(_healthy(), org=None)).report()

    assert report.ok is True
    assert _by_name(report)["identity-scope"] is CheckStatus.WARN


def test_unrecognized_controller_mode_warns_rather_than_assuming_health() -> None:
    """WARN, not SKIP: a SKIP now fails the run, and this may just be a newer server.

    The point is unchanged -- an unrecognized mode is never reported as PASS --
    but it must not be fatal, because a client older than its control plane is
    an ordinary situation rather than a broken environment.
    """

    report = DoctorCommandService(_session(_healthy(controller="something-new"))).report()

    assert _by_name(report)["controller-mode"] is CheckStatus.WARN
    assert report.ok is True


def test_a_control_plane_that_goes_away_mid_preflight_does_not_exit_zero() -> None:
    """The exact shape that made ``doctor`` report success on a dead fleet.

    ``EyvanUnavailableError`` surfaces as a ``ClickException``, not an
    ``EyvanApiError``, so it fell into the fleet check's generic handler and was
    reported as SKIP -- which ``ok`` counted as a pass. Health answering first
    makes this reachable: the run got past the reachability check and then
    declared everything fine.
    """

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.sdk.client.transport import EyvanUnavailableError

    def handler(path: str) -> tp.Any:
        if path.endswith("/health"):
            return {"status": "ok", "controller": "active"}
        raise EyvanUnavailableError(ControlPlaneEndpoint.parse("http://test"), "GET")

    with pytest.raises(SystemExit) as raised:
        DoctorCommandService(_session(handler)).run()

    assert raised.value.code == 1
    report = DoctorCommandService(_session(handler)).report()
    check = next(check for check in report.checks if check.name == "submittable-fleet")
    assert check.status is CheckStatus.FAIL
    assert "unreachable" in check.detail
    assert check.remedy


def test_the_real_cli_client_still_names_an_unreachable_plane(monkeypatch) -> None:
    """The typed handlers must fire through the client the CLI actually uses.

    ``http.EyvanClient`` used to re-wrap transport failures as a plain
    ``ClickException`` before they left the adapter, so every
    ``except EyvanUnavailableError`` branch in this module was dead when the
    plane went away *between* checks: the diagnosis degraded to
    "could not ...: ClickException: ...".
    """

    import eyvan.sdk.client.transport as client_transport
    import eyvan.sdk.commands.http as cli_http

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok", "controller": "active"})
        raise httpx.ConnectError("connection refused")

    raw_client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(client_transport.httpx, "Client", lambda **_kwargs: raw_client)
    session = CliSession(
        client=cli_http.EyvanClient("http://test", headers={}),
        identity=ClientIdentity(),
        context_store=_Store(),
        context_config=EyvanContextConfig(),
        base_url="http://test",
        org_id="acme",
        project_id="research",
        overrides=ContextOverrides(),
    )

    report = DoctorCommandService(session).report()

    assert report.ok is False
    fleet = _by_name(report)["submittable-fleet"]
    assert fleet is CheckStatus.FAIL
    check = next(check for check in report.checks if check.name == "submittable-fleet")
    assert "unreachable" in check.detail
    budget = next(check for check in report.checks if check.name == "budget-pricing")
    assert "became unreachable" in budget.detail


def test_a_check_that_could_not_run_is_not_a_pass() -> None:
    """The module contract says SKIP is distinct from passing; ``ok`` must agree."""

    report = DoctorReport()
    report.checks.append(
        next(
            check
            for check in DoctorCommandService(_session(_unreachable)).report().checks
            if check.status is CheckStatus.SKIP
        )
    )

    assert report.ok is False


def test_an_endpoint_that_is_not_a_control_plane_fails(tmp_path) -> None:
    """Something answered /health, but not with an Eyvan health document."""

    def handler(path: str) -> tp.Any:
        return "<html>hello from nginx</html>"

    report = DoctorCommandService(_session(handler)).report()

    assert report.ok is False
    check = next(check for check in report.checks if check.name == "controller-mode")
    assert check.status is CheckStatus.FAIL


def test_run_exits_non_zero_when_a_check_failed() -> None:
    with pytest.raises(SystemExit) as raised:
        DoctorCommandService(_session(_unreachable)).run()

    assert raised.value.code == 1


def test_report_json_is_machine_readable() -> None:
    payload = DoctorCommandService(_session(_healthy())).report().as_json()

    assert payload["ok"] is True
    assert payload["client_version"]
    assert {check["name"] for check in payload["checks"]} >= {"control-plane-reachable", "submittable-fleet"}


def test_a_hard_money_budget_over_unpriced_capacity_is_a_failure() -> None:
    """An unpriced pool costs exactly zero, so a hard budget silently does not bind.

    ``ResourceVector.dot`` returns zero for a resource with no unit price, and
    ``Scheduler._budget_allows`` is a hard admission gate — so this combination admits
    unlimited work while reporting compliance and never moving ``spent``. A budget an
    operator believes protects them, and does not, is worse than no budget.
    """

    def handler(path: str) -> tp.Any:
        if path.endswith("/health"):
            return {"status": "ok", "controller": "active"}
        if path.endswith("/budgets"):
            return [{"org_id": "acme", "project_id": "research", "limit": 500.0, "hard_limit": True}]
        if path.endswith("/capacity-pools"):
            return [
                {"id": "research/v5p-16", "enabled": True, "capacity": {"tpu": 8}, "unit_prices": {}},
                {"id": "research/cpu", "enabled": True, "capacity": {"cpu": 8}, "unit_prices": {"cpu": 0.05}},
            ]
        if path.endswith("/cells"):
            return []
        return [{"name": "research"}]

    report = DoctorCommandService(_session(handler)).report()

    assert report.ok is False
    check = next(check for check in report.checks if check.name == "budget-pricing")
    assert check.status is CheckStatus.FAIL
    assert "research/v5p-16" in check.detail
    # The priced pool is not reported: only unenforceable capacity is a finding.
    assert "research/cpu" not in check.detail
    assert "unit_prices" in check.remedy


def test_unpriced_capacity_without_a_hard_budget_is_not_a_finding() -> None:
    """Local and manual development fleets legitimately price nothing."""

    def handler(path: str) -> tp.Any:
        if path.endswith("/health"):
            return {"status": "ok", "controller": "active"}
        if path.endswith("/budgets"):
            return [{"org_id": "acme", "project_id": "research", "limit": 500.0, "hard_limit": False}]
        if path.endswith("/capacity-pools"):
            return [{"id": "research/dev", "enabled": True, "capacity": {"cpu": 8}, "unit_prices": {}}]
        if path.endswith("/cells"):
            return []
        return [{"name": "research"}]

    report = DoctorCommandService(_session(handler)).report()

    assert report.ok is True
    assert _by_name(report)["budget-pricing"] is CheckStatus.PASS


def test_kueue_preconditions_pass_when_no_pool_uses_kueue() -> None:
    """A fleet with no Kueue queues has no ClusterQueue that could evict anything.

    PASS rather than SKIP is load-bearing: ``DoctorReport.ok`` treats a SKIP as a
    failure, because a skipped check is one that could not run. This one ran and
    found the hazard cannot exist.
    """

    report = DoctorCommandService(_session(_healthy())).report()

    assert _by_name(report)["kueue-preconditions"] is CheckStatus.PASS
    assert report.ok is True


def test_a_pool_routing_through_kueue_is_reported_as_unverified() -> None:
    """Cluster-side preemption cannot be read from the CLI, and saying otherwise would lie."""

    def handler(path: str) -> tp.Any:
        if path.endswith("/health"):
            return {"status": "ok", "controller": "active"}
        if path.endswith("/capacity-pools"):
            return [
                {
                    "id": "research/k8s",
                    "enabled": True,
                    "capacity": {"cpu": 8},
                    "unit_prices": {"cpu": 0.05},
                    "provider_options": {"queue_name": "research-queue"},
                }
            ]
        if path.endswith("/cells"):
            return []
        return [{"name": "research"}]

    report = DoctorCommandService(_session(handler)).report()

    check = next(item for item in report.checks if item.name == "kueue-preconditions")
    assert check.status is CheckStatus.WARN
    assert "research-queue" in check.detail
    assert "Never" in check.remedy
    assert report.ok is True, "an unverifiable precondition warns; it does not block the CLI"


def _pools_handler(pools: list[dict[str, tp.Any]]) -> tp.Callable[[str], tp.Any]:
    """Serve one capacity-pool list; the route check reads nothing else."""

    def handler(path: str) -> tp.Any:
        if path.endswith("/capacity-pools"):
            return pools
        raise AssertionError(f"the native-worker route check must not call {path}")

    return handler


def test_a_fleet_without_native_pools_needs_no_worker_route(monkeypatch) -> None:
    """PASS, not SKIP: the check ran and found the hazard cannot exist.

    ``DoctorReport.ok`` treats SKIP as a failure because a skipped check is one
    that could not run. A fleet whose pools use no eyvan-native executor has no
    worker for this machine to reach, so there is nothing to verify -- and no
    network call is made to decide that.
    """

    from eyvan.sdk.commands.doctor import DoctorCommandService

    def _never_called() -> str | None:
        raise AssertionError("a fleet with no native pools must not probe the network")

    monkeypatch.setattr(DoctorCommandService, "_observed_public_ipv4", staticmethod(_never_called))
    session = _session(_pools_handler([{"id": "p", "executor": "native-local", "enabled": True}]))

    result = DoctorCommandService(session)._native_worker_route_check()

    assert result.status is CheckStatus.PASS
    assert "no enabled pool" in result.detail


def test_an_unobservable_address_warns_rather_than_claiming_reachability(monkeypatch) -> None:
    """Not knowing the address is not the same as knowing it is fine.

    The ingress rule is pinned to this machine's public IPv4. If that cannot be
    observed, the honest answer is that it was not verified -- reporting PASS
    would assert a route nobody checked.
    """

    from eyvan.sdk.commands.doctor import DoctorCommandService

    monkeypatch.setattr(DoctorCommandService, "_observed_public_ipv4", staticmethod(lambda: None))
    session = _session(_pools_handler([{"id": "p", "executor": "eyvan-native", "enabled": True}]))

    result = DoctorCommandService(session)._native_worker_route_check()

    assert result.status is CheckStatus.WARN
    assert "could not be determined" in result.detail


def test_the_remedy_names_the_address_the_rule_must_allow(monkeypatch) -> None:
    """The whole point: an operator should not have to walk the chain by hand."""

    from eyvan.sdk.commands.doctor import DoctorCommandService

    monkeypatch.setattr(DoctorCommandService, "_observed_public_ipv4", staticmethod(lambda: "203.0.113.7"))
    session = _session(_pools_handler([{"id": "p", "executor": "eyvan-native", "enabled": True}]))

    result = DoctorCommandService(session)._native_worker_route_check()

    assert result.status is CheckStatus.PASS
    assert "203.0.113.7" in result.detail
    assert result.remedy is not None
    assert "203.0.113.7/32" in result.remedy
    assert "EXECUTION_CONNECTING" in result.remedy


def test_the_fleet_check_reads_the_shape_the_route_actually_returns() -> None:
    """Found against a live head: `GET /api/v1/fleets` returns bare names.

    Only a list of objects was ever read, so every entry of `["research"]` was
    skipped and a plane with a fleet applied reported "no fleet is applied" --
    on every healthy plane, forever, with a non-zero exit. A preflight that
    always fails is worse than no preflight: it teaches an operator to ignore
    the one command that tells them what is wrong.

    The fixture in this module had been asserting an invented shape, which is
    how the defect survived a passing suite.
    """

    report = DoctorCommandService(_session(_healthy(fleets=["research"]))).report()

    assert _by_name(report)["submittable-fleet"] is CheckStatus.PASS
    assert report.ok is True

    # Object entries stay accepted: not trusting the shape is the point.
    mixed = DoctorCommandService(_session(_healthy(fleets=["research", {"name": "serving"}]))).report()
    detail = next(c for c in mixed.checks if c.name == "submittable-fleet")
    assert detail.status is CheckStatus.WARN
    assert "research" in detail.detail and "serving" in detail.detail


def test_an_unpriced_accelerator_is_reported_even_when_cpu_has_a_price() -> None:
    """A TPU pool priced on cpu alone bills its accelerators at nothing.

    `ResourceVector.dot` contributes zero for any resource with no price, so an
    unpriced accelerator costs exactly nothing -- and the check accepted a pool
    if *any* one of its resources was priced. The expensive resource is the one
    most likely to be missing, because it is the one nobody prices by hand, so
    the any-match test passed on precisely the pools that matter.
    """

    from eyvan.sdk.commands.doctor import _unpriced_pool_names

    partly_priced = {
        "id": "research/v5p-16",
        "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 8},
        "unit_prices": {"cpu": 0.03, "memory_gib": 0.004},
    }
    fully_priced = {
        "id": "research/priced",
        "capacity": {"cpu": 208, "tpu": 8},
        "unit_prices": {"cpu": 0.03, "tpu": 4.2},
    }

    assert _unpriced_pool_names([partly_priced]) == {"research/v5p-16"}
    assert _unpriced_pool_names([fully_priced]) == set()


# --------------------------------------------------------------------------- the one next thing


def test_the_report_names_one_thing_to_act_on() -> None:
    """A dozen lines with the important one buried is a report, not an answer.

    The checks already run in dependency order, so the first non-passing one
    carrying a remedy is the cause. Naming it is what makes `doctor` the
    command somebody standing a fleet up for the first time can follow.
    """

    from eyvan.sdk.commands.doctor import CheckResult, CheckStatus, DoctorReport

    report = DoctorReport(
        checks=[
            CheckResult("server-startup-settings", CheckStatus.PASS, "nothing needed"),
            CheckResult("submittable-fleet", CheckStatus.FAIL, "no fleet is applied", "eyvan up --region REGION"),
            CheckResult("price-drift", CheckStatus.WARN, "not authorized", "ask an admin"),
        ]
    )

    step = report.next_step
    assert step is not None
    assert step.name == "submittable-fleet", "the cause, not the symptom that follows it"


def test_a_healthy_fleet_is_told_to_do_nothing() -> None:
    """An instruction invented for a fleet with no problem is noise."""

    from eyvan.sdk.commands.doctor import CheckResult, CheckStatus, DoctorReport

    report = DoctorReport(checks=[CheckResult("submittable-fleet", CheckStatus.PASS, "active fleet 'research'")])

    assert report.next_step is None
    assert report.as_json()["next_step"] is None


def test_a_failure_with_no_remedy_never_becomes_the_next_step() -> None:
    """ "Something is wrong" is not an instruction; the next actionable one is."""

    from eyvan.sdk.commands.doctor import CheckResult, CheckStatus, DoctorReport

    report = DoctorReport(
        checks=[
            CheckResult("mystery", CheckStatus.FAIL, "it broke"),
            CheckResult("submittable-fleet", CheckStatus.FAIL, "no fleet", "eyvan up --region REGION"),
        ]
    )

    step = report.next_step
    assert step is not None and step.name == "submittable-fleet"


def test_the_next_step_is_readable_by_a_script() -> None:
    """A script bringing a fleet up wants the same answer a person gets.

    Re-deriving it from the check list means re-implementing the ordering rule,
    which is how a wrapper and the tool it wraps come to disagree.
    """

    from eyvan.sdk.commands.doctor import CheckResult, CheckStatus, DoctorReport

    report = DoctorReport(checks=[CheckResult("submittable-fleet", CheckStatus.FAIL, "no fleet", "eyvan up")])

    assert report.as_json()["next_step"] == {
        "name": "submittable-fleet",
        "status": "fail",
        "detail": "no fleet",
        "remedy": "eyvan up",
    }


def test_a_fleetless_plane_is_pointed_at_the_generated_path() -> None:
    """`up --region` reads real spot quota; a hand-written fleet guesses it.

    And it names `autoscale init`, because `up --region` refuses without a
    project in the profile -- advice that stops one step short is the reason
    somebody follows it and hits a wall.
    """

    report = DoctorCommandService(_session(_healthy(fleets=[]))).report()

    step = report.next_step
    assert step is not None and step.name == "submittable-fleet"
    assert "eyvan autoscale init" in step.remedy
    assert "eyvan up --region" in step.remedy
