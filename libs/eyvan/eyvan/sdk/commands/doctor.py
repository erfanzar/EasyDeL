# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan doctor``: preflight the local environment before blaming a workload.

``eyvan workload explain`` answers "why is this job not scheduled" from the
control plane's point of view. ``doctor`` answers the question that comes
first: is this client even talking to a working control plane, with an identity
it will accept and a fleet it can submit to.

Every check reports one of three outcomes and never guesses. ``FAIL`` means a
submission cannot work; ``WARN`` means it may work but something is
unconventional; ``SKIP`` means the check could not be run and says why, which is
deliberately distinct from passing.

One check comes before all of that and asks about this machine instead: given
the address a server would bind, which of the settings that server requires are
not configured. It is answered by the same function the server refuses to start
with, so ``doctor`` cannot report ready for an environment the server will
reject -- and it can be asked before anything has been started at all, which is
the moment the question actually matters.
"""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass, field
from enum import Enum

import click
import httpx
from pydantic import ValidationError

from eyvan._internal.client_version import client_version
from eyvan._internal.json_types import JsonValue, is_json_object
from eyvan.config import EyvanSettings
from eyvan.records import CapacityPoolRecord, CellRecord, CellState
from eyvan.sdk.client.transport import EyvanUnavailableError
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.deployment import newest_source_mtime, stale_components
from eyvan.sdk.commands.doctor_preflights import (
    check_artifact_store,
    check_control_plane_port,
    check_gcloud_authentication,
    check_runtime_profile,
    check_secret_manager_bind,
    serving_or_default_runtime_profile,
)
from eyvan.sdk.commands.http import EyvanApiError
from eyvan.sdk.commands.state import CliSession
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.api.server_startup import (
    StartupConfigurationSource,
    format_requirement_bullets,
    server_environment_settings,
    unsatisfied_startup_requirements,
)
from eyvan.server.service.pricing_resolver import audit_price_drift

#: Bind address ``eyvan-server`` uses when ``--host`` is not passed. The startup
#: check answers for the server this operator would actually start, so its
#: default has to be that server's default: a loopback plane requires nothing
#: extra, and reporting network requirements against it would be noise on every
#: development machine.
DEFAULT_SERVER_BIND_HOST = "127.0.0.1"

#: One check name, used by the CLI, the JSON report, and the divergence test.
SERVER_STARTUP_CHECK = "server-startup-settings"

#: The callable-trust check. A warning, never a refusal: a plane that trusts no
#: project serves every command workload correctly and only refuses ``@execute``.
CALLABLE_TRUST_CHECK = "callable-trust"


def _positive_resource(document: dict[str, JsonValue], name: str) -> bool:
    """Return whether a raw API resource field is a positive finite number."""

    value = document.get(name)
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and value > 0
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _pool_advertises_accelerator(pool: dict[str, JsonValue]) -> bool:
    """Recognize accelerator pools without requiring legacy rows to revalidate."""

    accelerator_type = pool.get("accelerator_type")
    if isinstance(accelerator_type, str) and accelerator_type.strip().lower() not in {"", "cpu"}:
        return True
    capacity_backend = pool.get("capacity_backend")
    if isinstance(capacity_backend, str) and capacity_backend.strip().lower().replace("_", "-") in {
        "gcp-tpu",
        "gcp-attach",
        "tpu",
    }:
        return True
    for field_name in ("capacity", "allocatable"):
        resources = pool.get(field_name)
        if is_json_object(resources) and any(_positive_resource(resources, kind) for kind in ("gpu", "tpu")):
            return True
    return False


class CheckStatus(str, Enum):  # noqa: UP042 -- preserve existing Enum string rendering
    """Outcome of one preflight check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One named preflight outcome with an actionable remedy when it is not passing."""

    name: str
    status: CheckStatus
    detail: str
    remedy: str = ""

    def as_json(self) -> dict[str, str]:
        payload = {"name": self.name, "status": self.status.value, "detail": self.detail}
        if self.remedy:
            payload["remedy"] = self.remedy
        return payload


@dataclass
class DoctorReport:
    """Aggregate preflight result."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether every check ran and none failed.

        A SKIP is not a pass. The module contract says so, but ``ok`` used to
        ignore SKIPs, so a control plane that answered ``/health`` and then went
        away before ``/fleets`` produced ``ok: true`` and exit 0 -- doctor
        reporting success for a client that cannot submit anything.

        SKIPs that merely suppress derived noise always accompany a FAIL, so
        this does not turn a clean run red.
        """

        return not any(check.status in {CheckStatus.FAIL, CheckStatus.SKIP} for check in self.checks)

    @property
    def next_step(self) -> CheckResult | None:
        """Return the one check to act on, or ``None`` when nothing needs acting on.

        The checks already run in dependency order, so the first non-passing one
        carrying a remedy is the cause rather than a symptom of it. Naming it is
        the difference between a report and an instruction: somebody standing a
        fleet up for the first time gets a wall of a dozen lines, and the only
        one that matters is buried in it.
        """

        for check in self.checks:
            if check.status is not CheckStatus.PASS and check.remedy:
                return check
        return None

    def as_json(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "ok": self.ok,
            "client_version": client_version(),
            "checks": [check.as_json() for check in self.checks],
        }
        step = self.next_step
        # Machine-readable too: a script bringing a fleet up wants the same one
        # answer a person does, and re-deriving it from the check list means
        # re-implementing the ordering rule.
        payload["next_step"] = step.as_json() if step is not None else None
        return payload


@dataclass(frozen=True, slots=True)
class DoctorCommandService:
    """Run local preflight checks against the configured control plane."""

    session: CliSession
    renderer: JsonRenderer = field(default_factory=JsonRenderer)
    server_bind_host: str = DEFAULT_SERVER_BIND_HOST
    as_json: bool = False
    #: The ``eyvan-server --team`` flag the previewed server will be started with.
    team: bool = False

    def _server_startup_check(self) -> CheckResult:
        """Report every setting a server bound to ``server_bind_host`` would refuse without.

        The one check here that answers a question about *this machine* rather
        than about a running control plane, and therefore the only one that is
        useful before anything has been started. It runs first for that reason:
        an operator standing a head up has no endpoint to reach yet, and the
        checks below would all skip.

        The answer comes from ``unsatisfied_startup_requirements``, the same
        function the server refuses with. A second list here would be a list that
        drifts, and drift on this surface reads as "doctor says I am ready" a
        moment before the server says otherwise.
        """

        try:
            bind = ServerBindPolicy.from_host(self.server_bind_host)
        except ValueError as error:
            return CheckResult(
                SERVER_STARTUP_CHECK,
                CheckStatus.FAIL,
                f"{self.server_bind_host!r} is not a usable bind address: {error}",
                "pass the address the server will bind, as in `eyvan doctor --server-host 0.0.0.0`",
            )
        if bind.is_loopback:
            # No remedy: a passing check that advertises a flag on every local run
            # is noise, and the server's own refusal already names this command.
            return CheckResult(
                SERVER_STARTUP_CHECK,
                CheckStatus.PASS,
                f"a server started here on {bind.host} needs no further configuration",
            )
        try:
            # The same resolution eyvan-server performs: EYVAN_*, then the
            # platform file's ``team``, then the flag. A team mode turned on only
            # by the file must be preflighted exactly as the server will enforce it.
            settings = server_environment_settings(team=self.team)
        except ValueError as error:
            return CheckResult(
                SERVER_STARTUP_CHECK,
                CheckStatus.FAIL,
                f"the EYVAN_* environment or platform file does not parse: {error}",
                "correct the setting named above; the server reads the same variables and file",
            )
        missing = unsatisfied_startup_requirements(
            bind,
            settings,
            source=StartupConfigurationSource.ENVIRONMENT,
        )
        if not missing:
            return CheckResult(
                SERVER_STARTUP_CHECK,
                CheckStatus.PASS,
                f"this environment satisfies every requirement of a {bind.host} bind",
            )
        noun = "setting" if len(missing) == 1 else "settings"
        names = ", ".join(requirement.environment_variable for requirement in missing)
        # "started here", not "is broken": this reads *this* machine's EYVAN_*
        # environment, which is the environment of the server an operator is
        # about to start, and says nothing about a plane running elsewhere.
        return CheckResult(
            SERVER_STARTUP_CHECK,
            CheckStatus.FAIL,
            f"a server started here on {bind.host} would refuse; "
            f"{len(missing)} {noun} unset or wrong in this environment: {names}",
            format_requirement_bullets(missing),
        )

    def _callable_trust_check(self) -> CheckResult:
        """Report whether any project on this plane may submit serialized callables.

        Deliberately not a startup requirement. ``unsatisfied_startup_requirements``
        is the list the server *refuses* to start without, and a plane that trusts
        nobody serves every command workload correctly -- refusing there would
        break deployments this setting does not concern.

        But the loss is silent, which is why it is here at all: the whole
        ``@execute`` API authenticates and is only then refused, with a 403 that
        names the workload rather than the variable that refused it. So it is a
        warning on this surface, and a warning from ``eyvan head up`` on the way
        out, and neither turns a healthy environment red.
        """

        try:
            bind = ServerBindPolicy.from_host(self.server_bind_host)
        except ValueError:
            # The startup check above already failed on this and named the cause.
            return CheckResult(CALLABLE_TRUST_CHECK, CheckStatus.SKIP, "the bind address is not usable")
        try:
            settings = EyvanSettings.from_env()
        except ValueError:
            # ``ValueError``, not pydantic's ``ValidationError``: ``EyvanSettings``
            # is a plain frozen dataclass and raises the former. Catching the
            # latter here was a handler for something that cannot happen, so a
            # malformed environment crashed `eyvan doctor` with a traceback --
            # in the one command whose entire job is diagnosing a malformed
            # environment. The startup check above had it right all along.
            return CheckResult(CALLABLE_TRUST_CHECK, CheckStatus.SKIP, "the EYVAN_* environment does not parse")
        if bind.is_loopback:
            return CheckResult(
                CALLABLE_TRUST_CHECK,
                CheckStatus.PASS,
                "a loopback development plane trusts its own default project",
            )
        if settings.trusted_callable_projects:
            named = ", ".join(settings.trusted_callable_projects)
            return CheckResult(
                CALLABLE_TRUST_CHECK,
                CheckStatus.PASS,
                f"callable workloads are trusted for: {named}",
            )
        return CheckResult(
            CALLABLE_TRUST_CHECK,
            CheckStatus.WARN,
            "no project may submit serialized callables, so every @execute submission would be refused with 403",
            "set EYVAN_TRUSTED_CALLABLE_PROJECTS=<org>/<project> on the server, or stand the head up with "
            "`eyvan head up --trust-callable-project <org>/<project>`; command workloads need none of this",
        )

    def _endpoint_check(self) -> CheckResult:
        url = self.session.base_url
        if not url:
            return CheckResult(
                "control-plane-url",
                CheckStatus.FAIL,
                "no control-plane URL is configured",
                "pass --url, set EYVAN_URL, or select a context with `eyvan use FLEET`",
            )
        return CheckResult("control-plane-url", CheckStatus.PASS, url)

    def _health_check(self) -> tuple[CheckResult, JsonValue | None]:
        try:
            payload = self.session.client.request("GET", "/api/v1/health")
        except EyvanApiError as error:
            return (
                CheckResult(
                    "control-plane-reachable",
                    CheckStatus.FAIL,
                    f"health request failed: {error}",
                    "start it with `eyvan-server` or check the URL and network path",
                ),
                None,
            )
        except Exception as error:
            return (
                CheckResult(
                    "control-plane-reachable",
                    CheckStatus.FAIL,
                    f"health request failed: {type(error).__name__}: {error}",
                    "start it with `eyvan-server` or check the URL and network path",
                ),
                None,
            )
        return CheckResult("control-plane-reachable", CheckStatus.PASS, "health endpoint answered"), payload

    @staticmethod
    def _controller_check(health: JsonValue | None) -> CheckResult:
        if not is_json_object(health):
            return CheckResult(
                "controller-mode",
                CheckStatus.FAIL,
                "health response was not a JSON object",
                "this endpoint is answering but is not an Eyvan control plane",
            )
        mode = health.get("controller")
        if mode == "active":
            return CheckResult("controller-mode", CheckStatus.PASS, "controller is reconciling")
        if mode == "api_only":
            return CheckResult(
                "controller-mode",
                CheckStatus.WARN,
                "this replica is API-only; workloads and capacity are not reconciled here",
                "another replica must run the controller, or unset EYVAN_CONTROLLER_ENABLED=0",
            )
        return CheckResult(
            "controller-mode",
            CheckStatus.WARN,
            f"unrecognized controller mode {mode!r}",
            "this client may be older than the control plane; upgrade it if submissions behave oddly",
        )

    def _identity_check(self) -> CheckResult:
        if self.session.org_id and self.session.project_id:
            return CheckResult(
                "identity-scope",
                CheckStatus.PASS,
                f"{self.session.org_id}/{self.session.project_id}",
            )
        return CheckResult(
            "identity-scope",
            CheckStatus.WARN,
            "no organization/project scope is configured; the server default will be used",
            "set --org/--project, or save one with `eyvan use FLEET --org ORG --project PROJECT`",
        )

    def _fleet_check(self) -> CheckResult:
        try:
            # Both query parameters are required by the route. Omitting them
            # made this check fail with a 422 on every invocation, in every
            # scope -- so `eyvan doctor` always exited non-zero and the one
            # question it exists to answer was never actually asked.
            payload = self.session.client.request(
                "GET",
                "/api/v1/fleets",
                params={"org_id": self.session.org_id, "project_id": self.session.project_id},
            )
        except EyvanApiError as error:
            return CheckResult(
                "submittable-fleet",
                CheckStatus.FAIL,
                f"listing fleets failed: {error}",
                "confirm the token or development identity is accepted by this control plane",
            )
        except EyvanUnavailableError as error:
            # Reaches here as a ClickException, not an EyvanApiError, so it used
            # to land in the generic branch below and be reported as SKIP --
            # which `ok` counted as a pass.
            return CheckResult(
                "submittable-fleet",
                CheckStatus.FAIL,
                f"control plane became unreachable while listing fleets: {error}",
                "check the URL, network path, and that `eyvan-server` is still running",
            )
        except Exception as error:
            return CheckResult(
                "submittable-fleet",
                CheckStatus.FAIL,
                f"could not list fleets: {type(error).__name__}: {error}",
                "re-run with a reachable control plane; a preflight that cannot run cannot pass",
            )
        names = _fleet_names(payload)
        if not names:
            return CheckResult(
                "submittable-fleet",
                CheckStatus.FAIL,
                "no fleet is applied on this control plane",
                # The generated path first: it discovers which zones actually
                # hold spot quota and writes the pools from it, which is where
                # a hand-written fleet gets the zone, shape and quota group
                # wrong. `autoscale init` is named because `up --region`
                # refuses without a project in the profile.
                "eyvan autoscale init      # once: project, fleet prefix, egress\n"
                "eyvan up --region REGION  # discover spot quota, generate, apply\n"
                "eyvan up FLEET.yaml       # or apply a fleet you wrote",
            )
        active = self.session.active_context
        if active is not None and active.fleet in names:
            return CheckResult("submittable-fleet", CheckStatus.PASS, f"active fleet {active.fleet!r}")
        if len(names) == 1:
            return CheckResult("submittable-fleet", CheckStatus.PASS, f"only fleet {names[0]!r} will be used")
        return CheckResult(
            "submittable-fleet",
            CheckStatus.WARN,
            f"{len(names)} fleets are applied and none is selected: {', '.join(names)}",
            "select one with `eyvan use FLEET`, or submit explicitly as FLEET/NODE_TYPE",
        )

    def _budget_pricing_check(self) -> CheckResult:
        """Fail when a hard money budget applies but capacity is unpriced.

        ``ResourceVector.dot`` returns zero for a resource with no unit price, so an
        unpriced pool's cells cost exactly nothing. ``Scheduler._budget_allows`` is a
        hard admission gate, so that combination admits unlimited work while reporting
        compliance and never moving ``spent`` -- a budget that silently does not apply
        is worse than no budget, because an operator believes they are protected.

        Only a *hard* budget makes this dangerous, which is why the check lives here:
        ``doctor`` can see budgets and pools together, while neither the scheduler nor
        the fleet-config parser can.
        """

        try:
            budgets = self.session.client.request("GET", "/api/v1/budgets", params={"org_id": self.session.org_id})
            pools = self.session.client.request("GET", "/api/v1/capacity-pools")
        except EyvanUnavailableError as error:
            return CheckResult("budget-pricing", CheckStatus.FAIL, f"control plane became unreachable: {error}")
        except EyvanApiError as error:
            # A viewer identity cannot read budgets; that is not a misconfiguration.
            if error.status_code in {401, 403}:
                return CheckResult(
                    "budget-pricing",
                    CheckStatus.WARN,
                    "not authorized to read budgets, so budget/pricing consistency was not checked",
                )
            return CheckResult("budget-pricing", CheckStatus.FAIL, f"could not read budgets or pools: {error}")
        except Exception as error:
            return CheckResult(
                "budget-pricing",
                CheckStatus.FAIL,
                f"could not read budgets or pools: {type(error).__name__}: {error}",
            )

        hard_scopes = {
            (budget.get("org_id"), budget.get("project_id"))
            for budget in (budgets if isinstance(budgets, list) else ())
            if is_json_object(budget) and budget.get("hard_limit") and budget.get("limit")
        }
        if not hard_scopes:
            return CheckResult("budget-pricing", CheckStatus.PASS, "no hard money budget is configured")

        unpriced = sorted(_unpriced_pool_names(pools))
        if not unpriced:
            return CheckResult("budget-pricing", CheckStatus.PASS, "every enabled pool prices its resources")
        return CheckResult(
            "budget-pricing",
            CheckStatus.FAIL,
            f"a hard money budget is set but {len(unpriced)} pool(s) declare no unit prices: {', '.join(unpriced[:5])}",
            "set unit_prices on those pools; unpriced capacity costs zero, so the budget cannot bind",
        )

    def _price_drift_check(self) -> CheckResult:
        """Warn when a cell's declared price disagrees with its pool's.

        Cost-ranked placement compares ``hourly_cost`` across candidate cells, so
        a stale or hand-edited price does not merely misreport spend -- it makes
        the scheduler systematically prefer whichever cell lies lowest, silently
        and forever. Nothing else in the package can see this: the scheduler is
        handed the prices, and a budget sums whatever it is told.

        A WARN rather than a FAIL. A negotiated rate on one pool is legitimate,
        and this check cannot tell that apart from a mistake -- only an operator
        can, which is exactly what a warning asks them to do.
        """

        try:
            pools = self.session.client.request("GET", "/api/v1/capacity-pools")
            cells = self.session.client.request("GET", "/api/v1/cells")
        except EyvanUnavailableError as error:
            return CheckResult("price-drift", CheckStatus.FAIL, f"control plane became unreachable: {error}")
        except EyvanApiError as error:
            if error.status_code in {401, 403}:
                return CheckResult(
                    "price-drift", CheckStatus.WARN, "not authorized to read capacity, so prices were not compared"
                )
            raise
        if not isinstance(pools, list) or not isinstance(cells, list):
            return CheckResult("price-drift", CheckStatus.WARN, "control plane returned an unexpected capacity shape")
        try:
            pool_records = [CapacityPoolRecord.model_validate(pool) for pool in pools]
            cell_records = [CellRecord.model_validate(cell) for cell in cells]
        except ValidationError as error:
            return CheckResult("price-drift", CheckStatus.WARN, f"could not parse capacity records: {error}")
        drift = audit_price_drift(pools=pool_records, cells=cell_records)
        if not drift:
            # "Every priced cell agrees" is vacuously true on a fleet where no
            # cell is priced, and that fleet reports $0 accrued and $0 idle
            # forever. Missing prices must not be mistaken for free capacity.
            #
            # Warn rather than fail. A fleet may be deliberately unpriced --
            # reserved capacity already paid for, or a test fleet -- and the
            # remedy is a sentence, not a refusal to operate.
            live = [cell for cell in cell_records if cell.state is not CellState.DELETED]
            unpriced = [cell for cell in live if not cell.unit_prices]
            if live and len(unpriced) == len(live):
                return CheckResult(
                    "price-drift",
                    CheckStatus.WARN,
                    f"no cell carries a price, so cost reporting is structurally zero across {len(live)} cell(s)",
                    "add `unit_prices` to the pool in your fleet config; `eyvan capacity billing` cannot "
                    "distinguish a free fleet from an unpriced one",
                )
            return CheckResult("price-drift", CheckStatus.PASS, "every priced cell agrees with its pool")
        worst = max(drift, key=lambda finding: abs(finding.relative_delta))
        return CheckResult(
            "price-drift",
            CheckStatus.WARN,
            f"{len(drift)} cell(s) declare a price their pool does not; "
            f"worst is {worst.record_id} at {worst.observed_hourly_unit} vs {worst.expected_hourly_unit} "
            f"from {worst.expected_source}",
            "re-apply the fleet config, or price the pool deliberately if the difference is intended",
        )

    def _unschedulable_pool_check(self) -> CheckResult:
        """Report enabled legacy accelerator pools with no schedulable host vector.

        Catalog and request validation prevent new rows from acquiring this
        state. Durable rows remain readable so this check can name both missing
        admission dimensions: the documented default workload needs positive CPU
        and memory, not merely an accelerator count.
        """

        name = "pool-schedulability"
        try:
            pools = self.session.client.request("GET", "/api/v1/capacity-pools")
        except EyvanUnavailableError as error:
            return CheckResult(name, CheckStatus.FAIL, f"control plane became unreachable: {error}")
        except EyvanApiError as error:
            if error.status_code in {401, 403}:
                return CheckResult(name, CheckStatus.WARN, "not authorized to read capacity pools")
            raise
        if not isinstance(pools, list):
            return CheckResult(name, CheckStatus.WARN, "control plane returned an unexpected shape")

        unschedulable: list[tuple[str, tuple[str, ...]]] = []
        for pool in pools:
            if not is_json_object(pool) or pool.get("enabled") is not True or not _pool_advertises_accelerator(pool):
                continue
            invalid: list[str] = []
            for field_name in ("capacity", "allocatable"):
                resources = pool.get(field_name)
                for resource_name in ("cpu", "memory_gib"):
                    if not is_json_object(resources) or not _positive_resource(resources, resource_name):
                        invalid.append(f"{field_name}.{resource_name}")
            if invalid:
                unschedulable.append((str(pool.get("id")), tuple(invalid)))
        unschedulable.sort(key=lambda finding: finding[0])
        if not unschedulable:
            return CheckResult(
                name,
                CheckStatus.PASS,
                "every enabled accelerator pool advertises positive CPU and memory",
            )

        examples = ", ".join(f"{pool_id} ({', '.join(fields)})" for pool_id, fields in unschedulable[:3])
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"{len(unschedulable)} enabled accelerator pool(s) have non-positive host CPU or memory: {examples}",
            "Edit the fleet YAML first: remove explicit zero `resources.cpu`, `resources.memory_gib`, "
            "`allocatable.cpu`, or `allocatable.memory_gib` overrides for cataloged shapes, or replace them "
            "with positive host values; unknown shapes require complete manual resource and allocatable vectors. "
            "Then run `eyvan up <fleet.yaml>`. If this is a quota-generated cached fleet, run "
            "`eyvan autoscale generate` before `eyvan up`, or rerun `eyvan up --quota-based --preemptible`, "
            "so the quota/catalog cache is regenerated.",
        )

    def _kueue_check(self) -> CheckResult:
        """Report a Kueue configuration that could evict Eyvan's workloads.

        Reports, and deliberately does not refuse. The refusal lives in
        ``KubernetesCapacityDriver``, which reads the ClusterQueue before
        creating a Job and raises: it runs in-cluster with a ServiceAccount,
        while this runs on an operator's machine that usually cannot reach the
        Kubernetes API at all. Saying "refuse" here would describe a gate that
        does not exist, and on a safety check that is worse than saying nothing.

        The hazard is otherwise invisible: Kueue eviction destroys a pod with no
        checkpoint hook, bypassing Eyvan's whole preemption protocol, and the
        only trace is a training job that lost hours with no ledger entry
        explaining why.

        PASS -- not SKIP -- when no pool names a queue. ``DoctorReport.ok`` treats
        a SKIP as a failure because a skipped check is one that could not run;
        this one *ran* and found the hazard cannot exist, since a fleet with no
        Kueue queues has no ClusterQueue that could evict anything.
        """

        try:
            pools = self.session.client.request("GET", "/api/v1/capacity-pools")
        except EyvanUnavailableError as error:
            return CheckResult("kueue-preconditions", CheckStatus.FAIL, f"control plane became unreachable: {error}")
        except EyvanApiError as error:
            if error.status_code in {401, 403}:
                return CheckResult("kueue-preconditions", CheckStatus.WARN, "not authorized to read capacity pools")
            raise
        if not isinstance(pools, list):
            return CheckResult("kueue-preconditions", CheckStatus.WARN, "control plane returned an unexpected shape")
        queues = sorted(
            {
                str(options["queue_name"]).strip()
                for pool in pools
                if is_json_object(pool)
                and is_json_object(options := pool.get("provider_options", {}))
                and str(options.get("queue_name", "")).strip()
            }
        )
        if not queues:
            return CheckResult(
                "kueue-preconditions",
                CheckStatus.PASS,
                "no capacity pool routes through kueue, so no ClusterQueue can evict Eyvan's work",
            )
        return CheckResult(
            "kueue-preconditions",
            CheckStatus.WARN,
            f"{len(queues)} pool queue(s) configured ({', '.join(queues[:3])}); "
            "cluster-side preemption was not verified from here",
            "run this check from inside the cluster, or confirm preemption.withinClusterQueue and "
            "reclaimWithinCohort are both 'Never' on each ClusterQueue Eyvan uses",
        )

    def _native_worker_route_check(self) -> CheckResult:
        """Verify this machine can still reach the native workers it provisions.

        ``EXECUTION_CONNECTING`` is the most expensive symptom Eyvan has: the
        provider has already granted capacity and started billing, the slice has
        booted, and the control plane sits waiting for a worker it cannot reach
        until ``provisioning_timeout_s`` drains it. On Spot TPU that can be an
        hour of queueing thrown away, and the durable record says only "eyvan-
        native execution runtime is not ready".

        The usual cause is mundane and invisible: a public-routed fleet pins its
        ingress rule to the operator's public IPv4, and that address changes when
        the laptop moves network. Everything else stays valid, so nothing looks
        wrong anywhere -- diagnosing it by hand means walking the whole chain
        from firewall rule to service-account IAM, which is what this check is.
        """

        name = "native-worker-route"
        try:
            pools = self.session.client.request("GET", "/api/v1/capacity-pools")
        except EyvanUnavailableError as error:
            return CheckResult(name, CheckStatus.FAIL, f"control plane became unreachable: {error}")
        except EyvanApiError as error:
            if error.status_code in {401, 403}:
                return CheckResult(name, CheckStatus.WARN, "not authorized to read capacity pools")
            raise
        if not isinstance(pools, list):
            return CheckResult(name, CheckStatus.WARN, "control plane returned an unexpected shape")
        native = [
            pool
            for pool in pools
            if is_json_object(pool) and str(pool.get("executor", "")) == "eyvan-native" and pool.get("enabled")
        ]
        if not native:
            return CheckResult(
                name,
                CheckStatus.PASS,
                "no enabled pool uses the eyvan-native executor, so no worker route is required",
            )
        observed = self._observed_public_ipv4()
        if observed is None:
            return CheckResult(
                name,
                CheckStatus.WARN,
                f"{len(native)} eyvan-native pool(s) enabled; this machine's public IPv4 could not be determined",
                "a public-routed fleet pins its ingress rule to this address; verify it by hand before submitting",
            )
        return CheckResult(
            name,
            CheckStatus.PASS,
            f"{len(native)} eyvan-native pool(s) enabled; this machine is {observed}",
            f"if a cell stalls in EXECUTION_CONNECTING, confirm the ingress rule still allows {observed}/32 "
            "on the native worker port -- it is pinned to this address and does not follow a network change",
        )

    @staticmethod
    def _observed_public_ipv4() -> str | None:
        """Return this machine's public IPv4, or None when it cannot be observed."""

        try:
            with httpx.Client(timeout=5.0) as client:
                value = client.get("https://api.ipify.org").text.strip()
        except (httpx.HTTPError, OSError):
            return None
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return None
        return str(address) if address.version == 4 else None

    def _source_drift_check(self) -> CheckResult:
        """Report every local component running code older than the source tree.

        A source checkout has three separately deployed surfaces -- the control
        plane, the cell agent, and the staged worker wheel -- and nothing links
        any of them to the source they were built from. Each has silently run
        stale code during this package's own development, including a fix that
        lived in the agent while only the TPU hosts were redeployed. The failure
        is invisible: the workload succeeds, on code nobody is looking at.
        """

        if newest_source_mtime() is None:
            return CheckResult("source-drift", CheckStatus.PASS, "installed package, not a source checkout")

        stale = stale_components(runtime_profile_path=serving_or_default_runtime_profile(self.session.base_url))
        if not stale:
            return CheckResult("source-drift", CheckStatus.PASS, "every local component is newer than the source")
        return CheckResult(
            "source-drift",
            CheckStatus.WARN,
            f"running older code than the source tree: {', '.join(stale)}",
            remedy="eyvan runtime deploy --yes",
        )

    def _local_preflight_checks(self) -> tuple[CheckResult, ...]:
        """Map the local-machine preflights onto doctor check results.

        These ask about this machine — credentials, permissions, stores,
        profile, port — not about a running control plane, so they run before
        the endpoint check and still answer when nothing has been started.
        Each preflight returns (name, status, detail, remedy); the mapping
        keeps the status vocabulary in one place.
        """

        try:
            settings = EyvanSettings.from_env()
        except ValueError as error:
            # The startup check above reports the malformed environment in
            # full; these checks read the same variables and would only
            # re-crash on it.
            return (
                CheckResult(
                    "local-preflight",
                    CheckStatus.SKIP,
                    f"the EYVAN_* environment does not parse: {error}",
                    "",
                ),
            )
        outcomes = (
            check_gcloud_authentication(),
            check_secret_manager_bind(),
            check_artifact_store(settings),
            check_runtime_profile(self.session.base_url),
            check_control_plane_port(self.session.base_url),
        )
        status_by_name = {status.value: status for status in CheckStatus}
        results = []
        for name, status, detail, remedy in outcomes:
            results.append(CheckResult(name, status_by_name[status], detail, remedy))
        return tuple(results)

    def report(self) -> DoctorReport:
        """Run every check in dependency order and collect the outcomes."""

        report = DoctorReport()
        # First, and never skipped: it is the only check that answers before a
        # control plane exists, which is exactly when an operator is standing one
        # up and every check below would report "unreachable".
        report.checks.append(self._server_startup_check())
        # Second, and for the same reason: it reads the same environment, so it
        # also answers with no control plane running. Placed before the endpoint
        # check so an unreachable plane cannot skip it, and after the startup
        # check so ``checks[0]`` stays the refusal list.
        report.checks.append(self._callable_trust_check())
        report.checks.extend(self._local_preflight_checks())
        endpoint = self._endpoint_check()
        report.checks.append(endpoint)
        if endpoint.status is CheckStatus.FAIL:
            return report
        health_check, health = self._health_check()
        report.checks.append(health_check)
        if health_check.status is CheckStatus.FAIL:
            # Every remaining check needs the control plane; reporting them as
            # failures too would bury the one cause under derived noise.
            report.checks.append(CheckResult("identity-scope", CheckStatus.SKIP, "control plane is unreachable"))
            report.checks.append(CheckResult("submittable-fleet", CheckStatus.SKIP, "control plane is unreachable"))
            return report
        report.checks.append(self._controller_check(health))
        report.checks.append(self._identity_check())
        report.checks.append(self._fleet_check())
        report.checks.append(self._budget_pricing_check())
        report.checks.append(self._price_drift_check())
        report.checks.append(self._unschedulable_pool_check())
        report.checks.append(self._kueue_check())
        report.checks.append(self._native_worker_route_check())
        report.checks.append(self._source_drift_check())
        return report

    def run(self) -> None:
        """Emit the report and exit non-zero when a check failed.

        The readable report used to go to stderr, and only when stderr was a
        terminal, while stdout always received the JSON. So the answer a person
        wanted vanished the moment they piped the command anywhere, and what
        survived was the document. Report to stdout, and keep the document
        behind ``--json`` like every other read command here.
        """

        report = self.report()
        if self.as_json:
            self.renderer.emit(report.as_json())
        else:
            for line in self.render(report):
                click.echo(line)
        if not report.ok:
            raise SystemExit(1)

    @staticmethod
    def render(report: DoctorReport) -> tuple[str, ...]:
        """Render one report as the lines a person reads."""

        lines: list[str] = []
        step = report.next_step
        if step is not None:
            lines.append(f"  next: {step.detail}")
            lines.extend(f"        {line}" for line in step.remedy.splitlines())
            lines.append("")
        for check in report.checks:
            lines.append(f"  {check.status.value.upper():5} {check.name}  {check.detail}")
            # A remedy that lists several settings is several lines; printing it
            # as one would wrap the whole preflight into a wall. Identical output
            # for every single-line remedy.
            prefix = "        -> "
            for line in check.remedy.splitlines():
                lines.append(f"{prefix}{line}")
                prefix = "           "
        failed = sum(1 for check in report.checks if check.status is CheckStatus.FAIL)
        warned = sum(1 for check in report.checks if check.status is CheckStatus.WARN)
        summary = f"{len(report.checks)} checks"
        if failed:
            summary += f", {failed} failed"
        if warned:
            summary += f", {warned} warned"
        lines.append("")
        lines.append(f"  {summary if failed or warned else summary + ', all passing'}")
        return tuple(lines)


def _unpriced_pool_names(payload: JsonValue) -> set[str]:
    """Return enabled pools that advertise resources with no unit price.

    Shape is not trusted: a pool that cannot be parsed is skipped rather than reported,
    because inventing a finding from a malformed row would be worse than missing one.
    """

    if not isinstance(payload, list):
        return set()
    unpriced: set[str] = set()
    for pool in payload:
        if not is_json_object(pool) or pool.get("enabled") is False:
            continue
        prices = pool.get("unit_prices")
        priced = (
            {name for name, price in prices.items() if isinstance(price, int | float) and price}
            if is_json_object(prices)
            else set()
        )
        resources = pool.get("capacity")
        if not is_json_object(resources):
            continue
        wanted = {
            name for name, amount in resources.items() if name != "custom" and isinstance(amount, int | float) and amount
        }
        # Every resource the pool advertises must be priced, not merely one of
        # them. `ResourceVector.dot` contributes zero for any resource with no
        # price, so a TPU pool priced on `cpu` alone bills its accelerators at
        # nothing -- and an any-match test called that healthy. The expensive
        # resource is exactly the one most likely to be missing, because it is
        # the one nobody thinks to price by hand.
        if wanted - priced:
            name = pool.get("id") or pool.get("node_type")
            unpriced.add(name if isinstance(name, str) else "<unnamed pool>")
    return unpriced


def _fleet_names(payload: JsonValue) -> tuple[str, ...]:
    """Extract fleet names from a list response without trusting its shape."""

    if not isinstance(payload, list):
        return ()
    names: list[str] = []
    for entry in payload:
        # The route returns bare names -- `["research"]` -- and only objects were
        # ever read, so every entry was skipped and a correctly configured plane
        # was reported as having no fleet at all. That is the worst shape a
        # preflight failure can take: it fires on every healthy plane, exits
        # non-zero forever, and teaches an operator to stop reading it. Objects
        # are still accepted because "without trusting its shape" is the point.
        if isinstance(entry, str):
            name: JsonValue | None = entry
        elif is_json_object(entry):
            name = entry.get("name") or entry.get("fleet")
        else:
            continue
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return tuple(names)


@click.command("doctor")
@click.option(
    "--server-host",
    default=DEFAULT_SERVER_BIND_HOST,
    show_default=True,
    help="Bind address to preflight a server for; the same value `eyvan-server --host` takes.",
)
@click.option(
    "--team",
    is_flag=True,
    help="Preflight a server started with `eyvan-server --team` (the platform file's `team: true` is read anyway).",
)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def doctor(session: CliSession, server_host: str, team: bool, as_json: bool) -> None:
    """Check credentials, connectivity, and fleet readiness before submitting."""

    DoctorCommandService(session, server_bind_host=server_host, as_json=as_json, team=team).run()


def register_doctor_commands(group: click.Group) -> None:
    """Attach the doctor command to the root command group."""

    group.add_command(doctor)


__all__ = (
    "CALLABLE_TRUST_CHECK",
    "DEFAULT_SERVER_BIND_HOST",
    "SERVER_STARTUP_CHECK",
    "CheckResult",
    "CheckStatus",
    "DoctorCommandService",
    "DoctorReport",
    "doctor",
    "register_doctor_commands",
)
