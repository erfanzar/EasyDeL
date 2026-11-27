# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FastAPI transport for the Eyvan control-plane service."""

from __future__ import annotations

import argparse
import asyncio
import collections
import hmac
import ipaddress
import json
import logging
import os
import secrets
import typing as tp
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from eyvan._internal.build_identity import build_identity
from eyvan._internal.client_version import CLIENT_VERSION_HEADER, is_client_supported, stale_client_message
from eyvan._internal.correlation import canonical_correlation_id
from eyvan._internal.json_types import JsonObject, model_json
from eyvan._internal.observability import EyvanMetrics
from eyvan._internal.platform import EyvanPlatformConfig, load_platform_config, refuse_removed_team_variable
from eyvan.artifacts._capability_tokens import InvalidCapabilityToken
from eyvan.artifacts.managed import (
    ArtifactAction,
    ArtifactTokenSigner,
    InvalidArtifactToken,
    ManagedArtifactInventoryError,
    ManagedArtifactInventoryStore,
    ManagedArtifactVersionedStore,
    callable_uri,
    managed_artifact_store_from_uri,
    result_uri,
    workspace_uri,
)
from eyvan.artifacts.secret_capability import SecretCapabilitySigner
from eyvan.artifacts.transfers import ManagedArtifactAccess, progress_report_uri
from eyvan.artifacts.workload_capability import WorkloadCapabilitySigner
from eyvan.capacity.autoscale.service import TpuQuotaDiscovery
from eyvan.capacity.providers.gcp.quota_discovery import (
    GcloudQuotaError,
    GcpTpuQuotaDiscovery,
    TpuQuotaAvailability,
)
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfileStore
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.identity.assistant_keys import AssistantKeySealer
from eyvan.identity.rotation import (
    CredentialRotationBlocked,
    RotatingCredential,
)
from eyvan.records import (
    ArtifactInventoryReport,
    ArtifactReferenceRecord,
    ArtifactUsage,
    AuditEvent,
    BudgetRecord,
    BudgetStatus,
    CallableExecution,
    CapacityClass,
    CapacityPoolRecord,
    CarbonReport,
    CellBillingReport,
    CellRecord,
    CoordinationKind,
    CredentialRotationRecord,
    EffectiveQuota,
    LaunchLatencyReport,
    LeaseRecord,
    ManagedArtifactRecord,
    OrganizationRecord,
    Permission,
    PolicyRecord,
    Principal,
    ProjectRecord,
    ProjectUsageReport,
    QueueTrend,
    QuotaRecord,
    Role,
    RoleGrantRecord,
    ServiceDecisionRecord,
    UserProgressReport,
    WarmthReport,
    WebhookRecord,
    WorkloadEstimate,
    WorkloadExplanation,
    WorkloadProgress,
    WorkloadRecord,
    WorkloadSpendRecord,
    WorkloadState,
    utcnow,
)
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.contexts import DEFAULT_ORG_ID, DEFAULT_PROJECT_ID
from eyvan.secrets import MAX_SECRET_BYTES, SecretUnavailable, read_secret
from eyvan.server.api.assistant import register_assistant_routes
from eyvan.server.api.attachments import is_loopback_development, register_attachment_routes
from eyvan.server.api.auth import (
    DEFAULT_DEVELOPMENT_SCOPE,
    OIDCTokenVerifier,
    RequestAuthenticator,
    TokenVerifier,
    authenticated_principal,
)
from eyvan.server.api.dashboard_sessions import (
    DashboardSessionStore,
    register_dashboard_auth_routes,
)
from eyvan.server.api.mcp_routes import register_mcp_routes
from eyvan.server.api.research_routes import register_research_routes
from eyvan.server.api.research_runner import ResearchRunner
from eyvan.server.api.schemas import (
    ArtifactHoldRequest,
    ArtifactInventoryReindexRequest,
    BatchSubmitRequest,
    BatchSubmitResponse,
    BudgetGrantIssueRequest,
    BudgetPutRequest,
    BuildResponse,
    CapacityClassUnavailableErrorContext,
    CapacityClassUnavailableErrorResponse,
    CapacityPoolResponse,
    CapacityPoolUpsertRequest,
    CapacityResponse,
    CellUpsertRequest,
    ChildWorkloadSubmitRequest,
    DagSubmitRequest,
    DagSubmitResponse,
    EffectiveProjectScope,
    FleetApplyRequest,
    FleetInitScriptRequest,
    FleetStatusResponse,
    GovernanceSummaryResponse,
    GroupSubmitRequest,
    GroupSubmitResponse,
    HealthResponse,
    NoMatchingCapacityPoolErrorContext,
    NoMatchingCapacityPoolErrorResponse,
    OrganizationRegisterRequest,
    PlanePeerPutRequest,
    PlatformAgentNotConfiguredError,
    PolicyPutRequest,
    PoolCallableSubmitRequest,
    PoolWorkloadSubmitRequest,
    PreemptionForceFinalizeRequest,
    ProgressReportRequest,
    ProjectRegisterRequest,
    QuotaGrantIssueRequest,
    QuotaPutRequest,
    ReadOnlyQueryRequest,
    ReadOnlyQueryResponse,
    RoleGrantIssueRequest,
    ServiceEndpointRequest,
    ServiceHealthResponse,
    SessionResponse,
    StorageStatusResponse,
    TpuQuotaEntry,
    TpuQuotaFailure,
    TpuQuotaSnapshot,
    TpuQuotaStatus,
    WebhookCreateRequest,
    WorkloadEstimateRequest,
    WorkloadExecRequest,
    WorkloadSubmitRequest,
)
from eyvan.server.api.server_bind import ServerBindPolicy, uvicorn_tls_kwargs
from eyvan.server.api.server_logging import ServerLogLevel, server_log_config
from eyvan.server.api.server_startup import (
    NON_LOCAL_REPOSITORY_REQUIRES_ARTIFACT_TOKEN_KEY,
    DevelopmentAuthMode,
    ServerStartupSettings,
)
from eyvan.server.api.sse import KEEPALIVE_FRAME, sse_frame, sse_response
from eyvan.server.api.team_routes import register_team_routes
from eyvan.server.api.tracking_proxy import register_tracking_routes
from eyvan.server.controller import EyvanController
from eyvan.server.introspection import allocations_as_json, thread_stacks_as_json
from eyvan.server.invariants import Invariant, Violation, check_fleet, snapshot_fleet
from eyvan.server.local.auth import LocalDevelopmentCredentialStore
from eyvan.server.local.daemon import (
    DEFAULT_LOCAL_CELL_AGENT_URL,
    LocalDaemonRecord,
    record_publication_lock,
    runtime_profile_identifier,
    write_owner_only_record,
)
from eyvan.server.local.profile import LocalArtifactTokenKeyStore
from eyvan.server.persistence.base import ReadOnlyQueryUnsupported, Repository, TransientRepositoryError
from eyvan.server.persistence.leader import PostgresAdvisoryLeader
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.control_plane import (
    DEFAULT_ARTIFACT_GRACE_PERIOD_S,
    QUEUE_TREND_MAX_BUCKETS,
    QUEUE_TREND_MAX_WINDOW_S,
    QUEUE_TREND_MIN_WINDOW_S,
    ArtifactCollection,
    ControlPlaneService,
)
from eyvan.server.service.credential_rotation import (
    SECRET_CAPABILITY_CREDENTIAL,
    CredentialRotationService,
)
from eyvan.server.service.dag import DagNode
from eyvan.server.service.errors import (
    AuthorizationError,
    CapacityClassUnavailableError,
    ConflictError,
    EyvanServiceError,
    InputError,
    NoMatchingCapacityPoolError,
    NotFoundError,
    UnavailableError,
)
from eyvan.server.service.expansion import ExpansionLimits
from eyvan.server.service.federation import BudgetGrantRecord, PlanePeerRecord, QuotaGrantRecord
from eyvan.server.service.groups import GroupRole
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.services import ServicePolicy, service_policy
from eyvan.server.service.workload_progress import WorkloadProgressResolver
from eyvan.soak_contract import (
    MAX_BUDGET_SCOPES,
    MAX_COVERAGE_ENTRIES,
    MAX_PARTICIPANTS,
    MAX_SAFE_COUNTER,
    MAX_SERVER_VIOLATIONS,
    ParticipantBinding,
    is_opaque_id,
    participant_manifest_sha256,
    safe_scope_id,
)
from eyvan.soak_evidence import MAX_EVIDENCE_BODY_BYTES, ExternalEvidenceSample, SoakEvidenceStore
from eyvan.storage import StorageError, select_preferred_gcp_project, storage_status
from eyvan.worker.agent.client import CellAgentUnavailableError, CellAgentUnsupportedCapabilityError

logger = logging.getLogger(__name__)

UNMATCHED_ROUTE_LABEL = "<unmatched>"

#: How far behind its cursor an audit follower re-scans, so an event whose
#: transaction committed after a higher-sequence one is still delivered.
_AUDIT_STREAM_LOOKBACK = 256
#: How long a quiet log stream waits before polling again. journald has no
#: notification channel, so following is polling; a second is short enough to
#: read as live and long enough that a hundred idle followers are not a hundred
#: reads a second against every cell agent.
_LOG_STREAM_POLL_S = 1.0

#: Bound on remembered event ids used to suppress the duplicates that lookback
#: necessarily re-reads. Comfortably larger than the lookback window.
_AUDIT_STREAM_DEDUPE_LIMIT = 4096

#: How long one provider TPU quota reading is reused before it is read again.
#:
#: Cloud Monitoring's allocation-usage gauge is minutes behind reality anyway,
#: so a shorter window would buy no accuracy and would let one dashboard turn a
#: page refresh into a burst of `gcloud` calls. Every response carries the age
#: of its reading, so reuse is stated rather than hidden.
#: How long shutdown waits for in-flight assistant turn saves before closing the store.
ASSISTANT_SAVE_DRAIN_S = 30.0
TPU_QUOTA_CACHE_TTL_S = 120.0

#: How long a request waits for a provider read before answering without it.
#:
#: Quota discovery lists accelerator types per zone and can take minutes on a
#: project with many quota zones. A browser must not hang on that: past the
#: deadline the request answers with the previous reading and its real age, or
#: with a stated failure, while the read continues in the background and lands
#: in the cache for the next caller.
TPU_QUOTA_READ_DEADLINE_S = 25.0


class SecretRotationRequest(BaseModel):
    """Body of the signed-credential rotation route.

    Kept materialless for wire compatibility. The route refuses rotation until
    protected shared credential distribution and startup recovery exist; an
    overlap alone cannot make a process-local key replacement deployment-wide.
    """

    #: Seconds the superseded generation keeps verifying. Unset means one secret
    #: capability TTL, the lifetime an outstanding token can already have.
    overlap_s: float | None = Field(default=None, ge=0)


#: Served at "/" by a plane that hosts no dashboard, in place of the framework's
#: bare 404. Deliberately one self-contained document with no linked asset: the
#: plane that cannot serve a UI bundle cannot serve a stylesheet either.
NO_DASHBOARD_PAGE = """<!doctype html>
<meta charset="utf-8"><title>Eyvan control plane</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{color-scheme:light dark}
 body{margin:0;min-height:100vh;display:grid;place-items:center;
      font:16px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}
 main{max-width:34rem;padding:2rem}
 h1{font-size:1.1rem;font-weight:600;margin:0 0 .25rem;letter-spacing:.01em}
 p{margin:0 0 1rem;opacity:.75}
 code{font:0.875rem ui-monospace,SFMono-Regular,Menlo,monospace;
      padding:.1rem .35rem;border-radius:.25rem;background:color-mix(in srgb,currentColor 8%,transparent)}
 ul{margin:0;padding:0;list-style:none;display:grid;gap:.6rem}
 li{display:grid;gap:.15rem}
 span{font-size:.875rem;opacity:.6}
</style>
<main>
  <h1>Eyvan control plane</h1>
  <p>This plane serves the API. It is not hosting the dashboard bundle.</p>
  <ul>
    <li><code>eyvan dashboard-dev</code><span>run the UI against this plane from a source checkout</span></li>
    <li><code>/healthz</code><span>whether this plane is up</span></li>
    <li><code>/docs</code><span>the API this plane does serve</span></li>
  </ul>
</main>
"""


def dashboard_bundle() -> Path | None:
    """Return the dashboard bundle this plane should serve, if it has one.

    A head and the container image both point ``EYVAN_DASHBOARD_DIR`` at a
    built bundle, and that stays authoritative. What it adds is the fallback
    that makes a *development* plane behave like the head it stands in for: a
    source checkout keeps ``dashboard/dist`` beside the package, and serving it
    is the difference between the URL every command prints being the UI and
    being a 404.

    An installed wheel has no such sibling, so release builds copy the bundle
    into the package as ``eyvan/_dashboard`` (``scripts/stage_dashboard.py``).
    A checkout's own build wins over that copy, so a developer sees what they
    just built. With neither, it returns ``None`` and the plane says so at "/"
    rather than pretending.
    """

    configured = os.environ.get("EYVAN_DASHBOARD_DIR")
    if configured:
        directory = Path(configured)
        return directory if directory.is_dir() else None
    here = Path(__file__).resolve()
    # .../eyvan/server/api/app.py -> the checkout root that holds dashboard/,
    # then the copy a release build placed inside the package itself.
    for bundled in (here.parents[3] / "dashboard" / "dist", here.parents[2] / "_dashboard"):
        if (bundled / "index.html").is_file():
            return bundled
    return None


@dataclass(frozen=True, slots=True)
class _TpuQuotaReading:
    """One provider quota read for a single purchase class, or its failure."""

    capacity_class: CapacityClass
    refreshed_at: datetime
    rows: tuple[TpuQuotaAvailability, ...] = ()
    failure: str | None = None
    cached: bool = False


def _read_tpu_quota(
    discovery: TpuQuotaDiscovery,
    *,
    project: str,
    zones: tuple[str, ...],
    capacity_class: CapacityClass,
    account: str | None = None,
) -> _TpuQuotaReading:
    """Read one purchase class, turning a provider failure into a stated reason.

    A raised error here would become a 500 for a read-only panel, which reads
    as "the control plane is broken" when the truth is almost always "this host
    cannot talk to the provider". The reason is carried instead.
    """

    import inspect

    kwargs: dict[str, tp.Any] = {
        "project": project,
        "zones": zones,
        "preemptible": capacity_class is CapacityClass.SPOT,
    }
    if account is not None:
        params = inspect.signature(discovery.discover).parameters
        if "account" in params or any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
            kwargs["account"] = account
    try:
        rows = discovery.discover(**kwargs)
    except (GcloudQuotaError, OSError, ValueError) as exc:
        return _TpuQuotaReading(capacity_class=capacity_class, refreshed_at=utcnow(), failure=str(exc))
    return _TpuQuotaReading(capacity_class=capacity_class, refreshed_at=utcnow(), rows=tuple(rows))


class TpuQuotaCache:
    """Single-flight, age-stating cache over read-only provider quota discovery.

    Two properties matter and neither is optional. Concurrent callers share one
    provider read, so a dashboard with several tabs open cannot multiply into a
    burst of `gcloud` invocations. And a reading that is served again is
    returned marked as reused, so the route can report its true age instead of
    presenting it as a fresh observation.
    """

    def __init__(self, *, ttl_s: float = TPU_QUOTA_CACHE_TTL_S, deadline_s: float = TPU_QUOTA_READ_DEADLINE_S):
        if ttl_s < 0:
            raise ValueError("ttl_s must not be negative")
        if deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        self.ttl_s = ttl_s
        self.deadline_s = deadline_s
        self._readings: dict[tuple[str, tuple[str, ...], CapacityClass, str | None], _TpuQuotaReading] = {}
        self._inflight: dict[tuple[str, tuple[str, ...], CapacityClass, str | None], asyncio.Task[_TpuQuotaReading]] = {}
        self._lock = asyncio.Lock()

    async def read(
        self,
        *,
        discovery: TpuQuotaDiscovery,
        project: str,
        zones: tuple[str, ...],
        capacity_class: CapacityClass,
        refresh: bool = False,
        account: str | None = None,
    ) -> _TpuQuotaReading:
        """Return a fresh reading, a shared in-flight one, or a dated older one."""

        normalized_account = account.strip() if isinstance(account, str) and account.strip() else None
        key = (project, zones, capacity_class, normalized_account)
        async with self._lock:
            reading = self._readings.get(key)
            if reading is not None and not refresh and (utcnow() - reading.refreshed_at).total_seconds() < self.ttl_s:
                return replace(reading, cached=True)
            task = self._inflight.get(key)
            if task is None or task.done():
                task = asyncio.create_task(
                    self._refresh(
                        key,
                        discovery=discovery,
                        project=project,
                        zones=zones,
                        capacity_class=capacity_class,
                        account=normalized_account,
                    )
                )
                self._inflight[key] = task
        try:
            return await asyncio.wait_for(asyncio.shield(task), self.deadline_s)
        except TimeoutError:
            # The read is still running and will populate the cache. Answering
            # with the previous reading and its real age beats hanging, and
            # beats an empty table that reads as "no capacity anywhere".
            if reading is not None:
                return replace(reading, cached=True)
            return _TpuQuotaReading(
                capacity_class=capacity_class,
                refreshed_at=utcnow(),
                failure=(
                    f"provider quota discovery for {project!r} did not answer within "
                    f"{self.deadline_s:.0f}s; it is still running and a later read will report it"
                ),
            )

    async def _refresh(
        self,
        key: tuple[str, tuple[str, ...], CapacityClass, str | None],
        *,
        discovery: TpuQuotaDiscovery,
        project: str,
        zones: tuple[str, ...],
        capacity_class: CapacityClass,
        account: str | None = None,
    ) -> _TpuQuotaReading:
        """Perform one blocking provider read off the event loop and retain it."""

        reading = await asyncio.to_thread(
            _read_tpu_quota,
            discovery,
            project=project,
            zones=zones,
            capacity_class=capacity_class,
            account=account,
        )
        async with self._lock:
            self._readings[key] = reading
        return reading


def _route_label(request: Request) -> str:
    """Return a bounded Prometheus ``operation`` label for one request.

    Only a matched route template is safe to use as a label value. An unmatched
    path is caller-controlled, so labelling with it lets any scanner mint an
    unbounded number of Prometheus time series and grow the registry without
    limit -- reachable before authentication.
    """

    path = getattr(request.scope.get("route"), "path", None)
    return path if isinstance(path, str) else UNMATCHED_ROUTE_LABEL


#: Response header naming how one fan-out expansion was admitted.
#:
#: A header rather than a response field on purpose. The submitter has to learn
#: whether their fan-out is competing now or in waves -- a silently truncated or
#: silently deferred expansion is worse than a refusal -- and a header carries
#: that without changing a single byte of a response body that older clients
#: already parse.
EXPANSION_ADMISSION_HEADER = "Eyvan-Expansion-Admission"
CORRELATION_HEADER = "X-Eyvan-Correlation-ID"


def _set_correlation_header(response: Response, enabled: bool, identifier: str | None) -> None:
    """Expose a server-owned correlation identity without changing response bodies."""

    if enabled and identifier is not None:
        response.headers[CORRELATION_HEADER] = canonical_correlation_id(identifier)


async def _report_expansion_admission(
    service: ControlPlaneService,
    principal: Principal,
    response: Response,
    coordinator_id: str,
    *,
    children: int,
) -> None:
    """Tell the submitter which admission their expansion actually got.

    Read back from the durable ledger rather than recomputed, so a replayed
    submission reports the plan the original was admitted under instead of one
    derived from configuration as it stands now.
    """

    admission = await service.get_expansion_admission(principal, coordinator_id)
    response.headers[EXPANSION_ADMISSION_HEADER] = (
        admission.describe() if admission is not None else f"admitted {children} children immediately"
    )


def _artifact_token_key(settings: EyvanSettings) -> bytes:
    """Resolve an explicit key, a stable local key, or an ephemeral in-memory development key."""

    if settings.artifact_token_key is not None:
        return settings.artifact_token_key.encode()
    if settings.repository_backend is RepositoryBackend.LOCAL:
        return LocalArtifactTokenKeyStore(settings.local_artifact_token_key_path).load_or_create()
    if settings.repository_backend is RepositoryBackend.MEMORY:
        logger.warning(
            "EYVAN_ARTIFACT_TOKEN_KEY is not set; managed-artifact tokens minted by this in-memory "
            "development process stop verifying as soon as it exits"
        )
        return secrets.token_bytes(32)
    # The same sentence the network-bind preflight lists, from the same constant.
    # A loopback bind reaches only this refusal, so the two must not drift: an
    # operator who fixed what `eyvan doctor` reported would otherwise meet a
    # differently worded version of a requirement they had already satisfied.
    raise ValueError(NON_LOCAL_REPOSITORY_REQUIRES_ARTIFACT_TOKEN_KEY)


def create_app(
    *,
    settings: EyvanSettings | None = None,
    repository: Repository | None = None,
    service: ControlPlaneService | None = None,
    verifier: TokenVerifier | None = None,
    metrics: EyvanMetrics | None = None,
    metrics_registry: CollectorRegistry | None = None,
    artifact_access: ManagedArtifactAccess | None = None,
    server_bind: ServerBindPolicy | None = None,
    automatic_loopback_auth: bool = False,
    platform_config: EyvanPlatformConfig | None = None,
    soak_evidence: SoakEvidenceStore | None = None,
    quota_discovery: TpuQuotaDiscovery | None = None,
    server_startup: ServerStartupSettings | None = None,
) -> FastAPI:
    """Build an application with injectable auth, service, and persistence adapters.

    Programmatic builds default to a loopback bind; network exposure requires an explicit
    ``server_bind`` (or the ``create_network_app`` factory) and a production configuration.

    ``quota_discovery`` is constructed on first use rather than here, so a plane
    that never opens the capacity panel never builds a provider client and an
    injected one is the only thing an offline test can reach.

    ``server_startup`` is the already-validated startup policy ``eyvan-server``
    resolved from its environment, platform file and flags; it replaces
    ``settings`` and carries its own bind.
    """

    # Pin this process's identity to the tree it started from; see build_identity.
    build_identity()
    if server_startup is not None:
        if settings is not None or automatic_loopback_auth:
            raise ValueError("server_startup already carries the settings; pass one or the other")
        if server_bind is not None and server_bind != server_startup.bind:
            raise ValueError("server_bind must match the bind server_startup was validated for")
        server_bind = server_startup.bind
    server_bind = server_bind or ServerBindPolicy.from_host("127.0.0.1")
    quota_cache = TpuQuotaCache(ttl_s=TPU_QUOTA_CACHE_TTL_S, deadline_s=TPU_QUOTA_READ_DEADLINE_S)
    resolved_quota_discoveries: dict[str | None, TpuQuotaDiscovery] = {}

    def resolve_quota_discovery(account: str | None = None) -> TpuQuotaDiscovery:
        """Return the injected provider reader, or build one cached reader per GCP account."""

        if quota_discovery is not None:
            return quota_discovery
        normalized = account.strip() if isinstance(account, str) and account.strip() else None
        instance = resolved_quota_discoveries.get(normalized)
        if instance is None:
            instance = GcpTpuQuotaDiscovery(account=normalized)
            resolved_quota_discoveries[normalized] = instance
        return instance

    if automatic_loopback_auth:
        if settings is None:
            raise ValueError("automatic loopback authentication requires explicit local daemon settings")
        if not server_bind.is_loopback or not settings.dev_auth_enabled:
            raise ValueError("automatic loopback authentication requires a development-authenticated loopback bind")
        startup = ServerStartupSettings(
            bind=server_bind,
            settings=settings,
            development_auth_mode=DevelopmentAuthMode.LOOPBACK_DEFAULT,
        )
    elif server_startup is not None:
        startup = server_startup
    elif settings is not None:
        startup = ServerStartupSettings.from_settings(server_bind, settings)
    else:
        # The reload factories land here: EYVAN_* plus the platform file's ``team``.
        startup = ServerStartupSettings.from_environment(server_bind)
    settings = startup.settings
    local_development_credential = (
        LocalDevelopmentCredentialStore().load_or_create()
        if startup.development_auth_mode is DevelopmentAuthMode.LOOPBACK_DEFAULT
        else None
    )
    platform = platform_config or load_platform_config(settings.platform_config_path)
    loopback_development = is_loopback_development(server_bind, settings)
    if platform.default_agent_endpoint is None and loopback_development:
        # Keep every other operator setting when filling in the local agent.
        platform = replace(platform, default_agent_endpoint=DEFAULT_LOCAL_CELL_AGENT_URL)
    engine: AsyncEngine | None = None
    if repository is None:
        if settings.repository_backend is RepositoryBackend.MEMORY:
            repository = InMemoryRepository()
        elif settings.repository_backend is RepositoryBackend.LOCAL:
            repository = LocalJsonRepository(settings.local_state_path)
        elif settings.repository_backend is RepositoryBackend.POSTGRES:
            engine = create_async_engine(settings.database_url, pool_pre_ping=True)
            repository = SqlAlchemyRepository(engine)
        else:
            raise ValueError("unsupported Eyvan repository backend")
    if service is None:
        trusted_callable_projects = set(settings.trusted_callable_projects)
        if loopback_development:
            trusted_callable_projects.add(f"{DEFAULT_ORG_ID}/{DEFAULT_PROJECT_ID}")
        service = ControlPlaneService(
            repository,
            callable_trust=CallableProjectTrustPolicy.from_projects(trusted_callable_projects),
            scheduler=CoreSchedulingService(),
            # Without this the setting, the service parameter and every federation
            # test all existed while no deployed plane ever had an identity: a
            # plane with an empty ``plane_id`` refuses to issue grants and skips
            # consulting them at admission, so federation was configurable,
            # tested, and off everywhere.
            plane_id=settings.plane_id,
            correlation_enabled=settings.correlation_enabled,
            # Per-user rules, approval holds and reclaim exist only where accounts do.
            governance_enabled=settings.team_mode,
            # The seam that makes the fan-out ceiling an operator control rather
            # than a constant: without it the setting exists and no submission is
            # ever judged against it.
            expansion_limits=ExpansionLimits(
                ceiling=settings.max_expansion_children,
                wave_size=settings.expansion_wave_size,
            ),
        )
    if verifier is None and settings.oidc_issuer:
        verifier = OIDCTokenVerifier(
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
            jwks_url=settings.oidc_jwks_url,
        )
    registry = metrics_registry or CollectorRegistry(auto_describe=True)
    metrics = metrics or EyvanMetrics(registry=registry)
    soak_evidence = soak_evidence or SoakEvidenceStore(max_age_s=settings.soak_evidence_max_age_s)
    if settings.secret_token_ttl_s <= 0:
        raise ValueError("EYVAN_SECRET_TOKEN_TTL_S must be positive")
    token_key = _artifact_token_key(settings)
    if getattr(service, "assistant_sealer", None) is None:
        # Assistant provider keys are sealed under a key derived from this same
        # server secret, so they need no second secret and no new setting.
        service.assistant_sealer = AssistantKeySealer(token_key)
    service.default_reclaim_grace_s = settings.default_reclaim_grace_s
    # Secret and child capabilities are control-plane delivery mechanisms, not
    # properties of the injected artifact-store adapter.  Constructing them only
    # while constructing that adapter made an otherwise valid injection silently
    # launch declared mounts as missing variables.
    workload_capability = WorkloadCapabilitySigner(token_key)
    # A keyring rather than the raw key, so this replica can be handed new
    # material without a restart. Identical behaviour until something rotates:
    # one generation, no retiring epochs, the same signature over the same
    # bytes. The fence that stops two rotations interleaving is durable and
    # lives in ``eyvan.server.service.credential_rotation``; this is only the
    # in-process half that decides what a signature is made with.
    secret_capability_credential = RotatingCredential(token_key)
    secret_capability = SecretCapabilitySigner(
        secret_capability_credential,
        ttl_seconds=settings.secret_token_ttl_s,
    )
    credential_rotation = CredentialRotationService(repository)
    if artifact_access is None:
        if settings.artifact_token_ttl_s <= 0:
            raise ValueError("EYVAN_ARTIFACT_TOKEN_TTL_S must be positive")
        artifact_access = ManagedArtifactAccess(
            store=managed_artifact_store_from_uri(settings.artifact_store_uri),
            signer=ArtifactTokenSigner(
                token_key,
                default_ttl_seconds=settings.artifact_token_ttl_s,
                max_ttl_seconds=settings.artifact_token_ttl_s,
                clock_skew_seconds=30,
            ),
            public_url=settings.public_url,
            token_ttl_seconds=settings.artifact_token_ttl_s,
        )

    if service.inventory_store is not None and service.inventory_store is not artifact_access.store:
        raise ValueError("the service inventory and transfer API must use the same managed artifact store")
    # Transfer, inventory, lifecycle reads and conditional deletion must name
    # exactly one physical namespace. A store lacking versions can remain an
    # explicitly injected legacy SDK object, but it cannot back managed API I/O.
    if service.artifact_store is not None and service.artifact_store is not artifact_access.store:
        raise ValueError("the service lifecycle and transfer API must use the same managed artifact store")
    if service.artifact_store is None and isinstance(artifact_access.store, ManagedArtifactVersionedStore):
        service.artifact_store = artifact_access.store
    if service.inventory_store is None and isinstance(artifact_access.store, ManagedArtifactInventoryStore):
        service.inventory_store = artifact_access.store

    if engine is not None:
        leader = PostgresAdvisoryLeader(engine)
    else:
        leader = None
    controller = EyvanController(
        repository,
        service,
        settings=settings,
        leader=leader,
        metrics=metrics,
        artifact_access=artifact_access,
        workload_capability=workload_capability,
        secret_capability=secret_capability,
    )
    workload_progress = WorkloadProgressResolver(service, repository)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if engine is not None and settings.auto_create_schema:
            await create_schema(engine)
        await repository.open()
        try:
            historical_rotation = await credential_rotation.current(SECRET_CAPABILITY_CREDENTIAL)
            if historical_rotation is not None:
                # The old endpoint persisted a ledger but not its randomly
                # generated key. Even a committed row cannot restore the key,
                # and an abandoned latest row does not prove earlier keys were
                # never replaced. Refuse before serving or starting dispatch;
                # do not resurrect a retired key or ignore an uncertain epoch.
                raise CredentialRotationBlocked(
                    "secret credential rotation history cannot be recovered without protected shared "
                    "credential versions; operator recovery is required before this replica can start"
                )
        except BaseException:
            await controller.close()
            await repository.close()
            if engine is not None:
                await engine.dispose()
            raise
        stop = asyncio.Event()
        task = (
            asyncio.create_task(controller.run(stop), name="eyvan-controller") if settings.controller_enabled else None
        )
        # Research plans run beside the controller, only while this replica holds its leadership.
        _app.state.research_runner = ResearchRunner(
            service, app=_app, leader=controller.leader, read_logs=controller.workload_logs
        )
        research_task = (
            asyncio.create_task(_app.state.research_runner.run(stop), name="eyvan-research")
            if settings.controller_enabled
            else None
        )
        if startup.development_auth_mode is DevelopmentAuthMode.LOOPBACK_DEFAULT:
            logger.info(
                "Eyvan loopback development auth enabled; owner credential and local identity headers are required"
            )
        elif startup.development_auth_mode is DevelopmentAuthMode.DISABLED and server_bind.is_loopback:
            logger.warning(
                "Eyvan loopback development auth is disabled; local CLI and SDK identity headers will be rejected"
            )
        if task is None:
            logger.warning(
                "Eyvan controller mode=api_only; this process will serve the API without reconciling workloads "
                "or capacity"
            )
        else:
            logger.info("Eyvan controller mode=active; workload scheduling and capacity reconciliation are active")
        try:
            yield
        finally:
            stop.set()
            if research_task is not None:
                await research_task
            if task is not None:
                await task
            # Assistant turns save in background tasks; let them land before the
            # store closes, so a turn is never dropped and no save outlives the
            # store's ownership lease (a successor process may already own it).
            pending = set(getattr(_app.state, "assistant_pending_saves", ()) or ())
            if pending:
                _done, unfinished = await asyncio.wait(pending, timeout=ASSISTANT_SAVE_DRAIN_S)
                if unfinished:
                    logger.warning("%d assistant save(s) did not finish before shutdown", len(unfinished))
            await controller.close()
            await repository.close()
            if engine is not None:
                await engine.dispose()

    app = FastAPI(title="Eyvan Fleet API", version="1.0.0", lifespan=lifespan)
    app.state.service = service
    dashboard_sessions = DashboardSessionStore(
        session_ttl_s=settings.dashboard_session_ttl_s,
        transaction_ttl_s=settings.dashboard_auth_transaction_ttl_s,
    )
    app.state.dashboard_sessions = dashboard_sessions
    app.state.authenticator = RequestAuthenticator(
        settings,
        verifier=verifier,
        local_development_credential=local_development_credential,
        dashboard_sessions=dashboard_sessions,
    )
    # The dashboard's backend-for-frontend auth: login/callback/logout. The
    # routes self-check the OIDC configuration, so mounting them on a plane
    # without one only makes the endpoints answer 404.
    register_dashboard_auth_routes(app, settings=settings, store=dashboard_sessions)
    register_tracking_routes(app, settings=settings, service=service)
    register_attachment_routes(app, settings=settings, service=service, server_bind=server_bind)
    register_team_routes(app, settings=settings, service=service)
    register_assistant_routes(app, settings=settings, service=service)
    register_research_routes(app, settings=settings, service=service)
    register_mcp_routes(app, settings=settings, service=service)
    # The seam that makes durable grants real. Without it the records exist, the
    # routes issue them, and no request is ever authorized by one.
    app.state.grant_resolver = service.active_grants_for
    app.state.workload_capability = workload_capability
    app.state.secret_capability = secret_capability
    # Material never crosses the API. The rotation route is deliberately a
    # refusal: a durable fence alone cannot distribute or recover the keyring.
    app.state.secret_capability_credential = secret_capability_credential
    app.state.credential_rotation = credential_rotation
    app.state.metrics = metrics
    app.state.metrics_registry = registry
    app.state.controller = controller
    app.state.artifact_access = artifact_access
    app.state.platform = platform
    app.state.server_startup = startup

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        if request.scope["path"].startswith("/v1/"):
            request.scope["path"] = "/api" + request.scope["path"]
        try:
            response = await call_next(request)
        except Exception:
            metrics.api_requests.labels(operation=_route_label(request), result="error").inc()
            raise
        metrics.api_requests.labels(operation=_route_label(request), result=str(response.status_code)).inc()
        return response

    @app.middleware("http")
    async def enforce_client_version(request: Request, call_next):
        """Reject an Eyvan SDK older than this deployment supports.

        Both path spellings are checked because Starlette runs a
        later-registered middleware outermost, so this may see a request before
        ``observe_request`` rewrites the ``/v1/`` compatibility alias.
        """

        minimum = settings.min_client_version
        if minimum and request.scope["path"].startswith(("/api/v1/", "/v1/")):
            reported = request.headers.get(CLIENT_VERSION_HEADER)
            if not is_client_supported(reported, minimum):
                return JSONResponse(
                    status_code=status.HTTP_426_UPGRADE_REQUIRED,
                    content={
                        "detail": stale_client_message(reported, minimum),
                        "code": "client_too_old",
                        "context": {"client_version": reported, "minimum_client_version": minimum},
                    },
                )
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Keep dashboard HTML strict and scope the framework docs separately.

        FastAPI documentation uses CDN assets and inline initialization; those
        exceptions apply only to its registered documentation routes, never to
        the dashboard bundle or a catch-all HTML fallback.
        """

        response = await call_next(request)
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            documentation_endpoints = {
                route.endpoint
                for route in app.routes
                if getattr(route, "name", None) in {"swagger_ui_html", "redoc_html", "swagger_ui_redirect"}
                and getattr(route, "path", None) in {app.docs_url, app.redoc_url, app.swagger_ui_oauth2_redirect_url}
            }
            is_documentation = request.scope.get("endpoint") in documentation_endpoints
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                # blob: only for images and audio/video: the Runs media views fetch
                # tracked media with the bearer and display it from object URLs.
                # Scripts, frames and objects stay as strict as before.
                "img-src 'self' data: blob:; "
                "media-src 'self' blob:; "
                "connect-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "frame-ancestors 'none'"
            )
            if is_documentation:
                response.headers["Content-Security-Policy"] = (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                    "font-src 'self' https://fonts.gstatic.com; "
                    "img-src 'self' data: https://fastapi.tiangolo.com; "
                    "connect-src 'self'; "
                    "object-src 'none'; "
                    "base-uri 'self'; "
                    "form-action 'self'; "
                    "frame-ancestors 'none'"
                )
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    @app.exception_handler(EyvanServiceError)
    async def handle_service_error(request: Request, exc: EyvanServiceError) -> JSONResponse:
        del request
        if isinstance(exc, AuthorizationError):
            status_code = status.HTTP_403_FORBIDDEN
        elif isinstance(exc, NotFoundError):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(exc, ConflictError):
            status_code = status.HTTP_409_CONFLICT
        elif isinstance(exc, UnavailableError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            status_code = status.HTTP_400_BAD_REQUEST
        if isinstance(exc, (AuthorizationError, NotFoundError)):
            logger.info("Eyvan API request rejected (%s): %s", type(exc).__name__, exc)
        else:
            logger.warning("Eyvan API service error (%s): %s", type(exc).__name__, exc)
        if isinstance(exc, NoMatchingCapacityPoolError):
            response = NoMatchingCapacityPoolErrorResponse(
                detail=str(exc),
                context=NoMatchingCapacityPoolErrorContext(
                    fleet=exc.fleet,
                    requested_accelerator_type=exc.requested_accelerator_type,
                    available_enabled_accelerator_types=exc.available_enabled_accelerator_types,
                    fleets_providing_requested_type=exc.fleets_providing_requested_type,
                    available_fleets=exc.available_fleets,
                ),
            )
            return JSONResponse(status_code=status_code, content=model_json(response))
        if isinstance(exc, CapacityClassUnavailableError):
            class_response = CapacityClassUnavailableErrorResponse(
                detail=str(exc),
                context=CapacityClassUnavailableErrorContext(
                    fleet=exc.fleet,
                    requested_accelerator_type=exc.requested_accelerator_type,
                    requested_capacity_class=exc.requested_capacity_class,
                    available_capacity_classes=exc.available_capacity_classes,
                    existing_hardware_capacity_classes=exc.existing_hardware_capacity_classes,
                ),
            )
            return JSONResponse(status_code=status_code, content=model_json(class_response))
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{key: value for key, value in error.items() if key not in {"input", "ctx"}} for error in exc.errors()]
        path = request.scope["path"]
        if path.startswith("/v1/"):
            path = "/api" + path
        legacy_scoped_endpoints = {
            ("POST", "/api/v1/artifacts/holds"),
            ("DELETE", "/api/v1/artifacts/holds"),
            ("GET", "/api/v1/artifacts/references"),
        }
        missing = {
            str(error["loc"][-1]) for error in exc.errors() if error.get("type") == "missing" and error.get("loc")
        }
        if (request.method, path) in legacy_scoped_endpoints and missing & {"org_id", "project_id"}:
            return JSONResponse(
                status_code=status.HTTP_426_UPGRADE_REQUIRED,
                headers={"X-Eyvan-Required-Client-Feature": "scoped-artifact-identity"},
                content={
                    "detail": (
                        "artifact hold, release, and reference requests now require explicit org_id and project_id; "
                        "upgrade the Eyvan client or resend the request with both scope fields"
                    ),
                    "code": "artifact_scope_required",
                    "context": {"required_fields": ["org_id", "project_id"]},
                },
            )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": errors})

    @app.exception_handler(ValidationError)
    async def handle_model_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        del request
        errors = [{key: value for key, value in error.items() if key not in {"input", "ctx"}} for error in exc.errors()]
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": errors})

    @app.exception_handler(SecretUnavailable)
    async def handle_secret_unavailable(request: Request, exc: SecretUnavailable) -> JSONResponse:
        """Report a retryable store failure without reflecting resolver output.

        Provider and custom-store exceptions are an untrusted diagnostic
        boundary: some clients include response fragments or request headers in
        their text. The detailed cause stays attached for local debugging, but
        neither logs nor the worker-facing response may receive it.
        """

        del request, exc
        logger.warning("Eyvan secret resolution failed")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "5"},
            content={"detail": "secret mount could not be resolved"},
        )

    @app.exception_handler(ManagedArtifactInventoryError)
    async def handle_artifact_inventory_error(request: Request, exc: ManagedArtifactInventoryError) -> JSONResponse:
        del request
        logger.warning("Managed artifact inventory was inconclusive: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "managed artifact inventory was inconclusive; no metadata was changed",
                "code": "artifact_inventory_inconclusive",
            },
        )

    @app.exception_handler(TransientRepositoryError)
    async def handle_transient_repository_error(request: Request, exc: TransientRepositoryError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "1"},
            content={"detail": str(exc)},
        )

    @app.exception_handler(CellAgentUnsupportedCapabilityError)
    async def handle_unsupported_capability_error(
        request: Request, exc: CellAgentUnsupportedCapabilityError
    ) -> JSONResponse:
        del request
        logger.info("Eyvan API operation unsupported by cell backend: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content={"detail": str(exc), "code": "unsupported_capability"},
        )

    @app.exception_handler(CellAgentUnavailableError)
    async def handle_cell_agent_unavailable_error(request: Request, exc: CellAgentUnavailableError) -> JSONResponse:
        del request
        logger.warning("Eyvan cell agent unreachable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "5"},
            content={
                "detail": (
                    "The cell agent could not be reached right now; check cell health on the Fleet page "
                    "or retry in a few seconds."
                ),
                "code": "cell_agent_unavailable",
            },
        )

    @app.exception_handler(StorageError)
    async def handle_storage_error(request: Request, exc: StorageError) -> JSONResponse:
        del request
        logger.warning("Eyvan storage configuration error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc), "code": "storage_unavailable"},
        )

    @app.get("/healthz", response_model=HealthResponse, tags=["system"])
    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Liveness: this process is running and its event loop answers.

        Deliberately does **not** touch the database, even though a probe that
        did would look more thorough. Liveness failure means *restart me*, and a
        liveness probe that depends on a shared database restarts every replica
        at once the moment that database blips -- turning a recoverable outage
        into a total one, then preventing recovery by stampeding reconnections.

        Whether this replica can usefully serve is a different question with a
        different consequence, and it is answered by ``/readyz``.
        """

        return HealthResponse(controller=startup.controller_mode)

    @app.get("/api/v1/build", response_model=BuildResponse, tags=["system"])
    async def build(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> BuildResponse:
        """Report the code this plane is running, not only its version string.

        Every component of this system ships one version, so an operator cannot
        use it to answer the question an upgrade actually raises: is the plane
        this proof polled the build the receipt claims? A hardware proof has
        already been run against a stale agent and read as evidence.

        Authenticated, because a build digest and an interpreter version are a
        fingerprint. It costs one cached tree walk per process.
        """

        del principal
        identity = build_identity()
        return BuildResponse(
            version=identity.version,
            origin=identity.origin,
            origin_detail=identity.origin_detail,
            source_digest=identity.source_digest,
            digest_detail=identity.digest_detail,
            file_count=identity.file_count,
            python=identity.python,
            platform=identity.platform,
            controller=startup.controller_mode,
        )

    @app.get("/api/v1/process/stacks", tags=["system"])
    async def process_stacks(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[dict[str, tp.Any]]:
        """Return every live thread's stack in this control-plane process.

        The one answer that identifies a wedged scheduler. Every external signal
        looks identical when the leader holds its lease and advances nothing --
        the API answers, health is green -- and the difference is which frame the
        process is sitting in.

        Taken as a snapshot rather than by suspending the process: a consistent
        dump bought with a stop-the-world pause would be an outage taken to
        investigate a stall.
        """

        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        return thread_stacks_as_json()

    @app.get("/api/v1/process/allocations", tags=["system"])
    async def process_allocations(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, tp.Any]:
        """Return the largest tracked allocations in this process.

        ``already_tracing`` is the half that decides whether the numbers mean
        anything: ``tracemalloc`` sees only what was allocated after it started,
        so a snapshot from a tracker this call had to switch on covers
        milliseconds rather than the leak that has been growing since Tuesday.
        Read it before reading the sites.
        """

        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        return allocations_as_json()

    @app.post("/api/v1/query", response_model=ReadOnlyQueryResponse, tags=["system"])
    async def readonly_query(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        request: ReadOnlyQueryRequest,
    ) -> ReadOnlyQueryResponse:
        """Answer one read-only statement against this plane's own database.

        The triage surface the typed endpoints cannot cover: during an incident
        the question is usually a join nobody anticipated, and the alternative to
        this is an operator opening ``psql`` on the head with write access and no
        ledger entry.

        Read-ness is enforced by the transaction rather than by inspecting the
        SQL, and the authorization is ``MANAGE_FLEET`` because the scope of a raw
        read is the whole database rather than the caller's grants.
        """

        try:
            result = await service.run_readonly_query(
                principal,
                request.statement,
                limit=request.limit,
                timeout_ms=settings.readonly_query_timeout_ms,
            )
        except ReadOnlyQueryUnsupported as error:
            raise HTTPException(status_code=501, detail=str(error)) from error
        except DBAPIError as error:
            # The statement was wrong -- bad column, syntax, or a write the
            # read-only transaction refused. That is the caller's mistake rather
            # than this plane failing, so it must not read as a 500.
            raise HTTPException(status_code=400, detail=str(error.orig or error)) from error
        return ReadOnlyQueryResponse(columns=result.columns, rows=result.rows, truncated=result.truncated)

    @app.get("/api/v1/session", response_model=SessionResponse, tags=["system"])
    async def session_authority(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        org_id: tp.Annotated[str | None, Query(min_length=1, max_length=128)] = None,
        project_id: tp.Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    ) -> SessionResponse:
        """Return the caller's effective authority after all grants are resolved.

        The dependency has already verified the request credential and unioned
        its claim grants with unexpired durable grants. The dashboard consumes
        this response rather than decoding an unverified JWT payload and
        pretending that payload includes the server's durable authority.

        Registered projects are returned only when this principal has at least
        one permission there. ``org_id``/``project_id`` may name a preferred
        legacy scope that predates the tenant registry; it is included only if
        the same authorizer that guards API operations says it is effective.
        """

        if (org_id is None) != (project_id is None):
            raise InputError("session scope requires both org_id and project_id")

        authorizer = service.authorizer
        candidates: dict[tuple[str, str], ProjectRecord | None] = {
            (grant.org_id, grant.project_id): None
            for grant in principal.grants
            if grant.org_id is not None and grant.project_id is not None
        }
        async with repository.transaction() as tx:
            organizations = {organization.id: organization for organization in await tx.list_organizations()}
            registered_projects = await tx.list_projects()
        for project in registered_projects:
            if authorizer.permissions(principal, org_id=project.org_id, project_id=project.id):
                candidates[project.key] = project
        if org_id is not None and project_id is not None:
            preferred = (org_id, project_id)
            if authorizer.permissions(principal, org_id=org_id, project_id=project_id):
                candidates.setdefault(preferred, None)

        scopes: list[EffectiveProjectScope] = []
        for (scope_org, scope_project), project in sorted(candidates.items()):
            permissions = authorizer.permissions(principal, org_id=scope_org, project_id=scope_project)
            if not permissions:
                continue
            roles = tuple(
                sorted(
                    {
                        grant.role
                        for grant in principal.grants
                        if grant.role is Role.FLEET_ADMIN
                        or (
                            grant.org_id == scope_org
                            and (grant.role is Role.ORG_ADMIN or grant.project_id == scope_project)
                        )
                    },
                    key=lambda role: role.value,
                )
            )
            organization = organizations.get(scope_org)
            scopes.append(
                EffectiveProjectScope(
                    org_id=scope_org,
                    project_id=scope_project,
                    display_name="" if project is None else project.display_name,
                    registered=project is not None,
                    enabled=(organization is None or organization.enabled) and (project is None or project.enabled),
                    # Registration is scoped per organization. A completely
                    # unregistered org is the supported pre-registry world and
                    # keeps accepting work; once the org exists, a missing or
                    # disabled project is history-only even when the caller's
                    # grant still authorizes reading it.
                    accepts_new_work=(
                        organization is None or (organization.enabled and project is not None and project.enabled)
                    ),
                    roles=roles,
                    permissions=permissions,
                )
            )
        return SessionResponse(
            subject=principal.subject,
            display_name=principal.display_name,
            grants=principal.grants,
            global_permissions=authorizer.permissions(principal),
            scopes=tuple(scopes),
        )

    @app.get("/readyz", response_model=HealthResponse, tags=["system"], include_in_schema=False)
    async def readiness(response: Response) -> HealthResponse:
        """Readiness: this replica can reach its database *now*.

        The gap this closes: the deployment probed ``/healthz`` for readiness,
        and ``/healthz`` returned a value computed at startup. A replica whose
        database connection had died therefore reported 200 forever and stayed in
        the Service's endpoints, absorbing traffic that could only fail.

        503 rather than an exception so the body still names the controller mode:
        an operator reading a failing probe wants to know which replica this is
        as much as that it is failing.
        """

        try:
            await repository.ping()
        except Exception as exc:
            logger.warning("Eyvan readiness probe failed: %s", exc)
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(status="unavailable", controller=startup.controller_mode)
        return HealthResponse(controller=startup.controller_mode)

    async def artifact_body(request: Request, *, limit: int) -> bytes:
        if limit <= 0:
            raise ValueError("managed artifact size limits must be positive")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
                if declared_length < 0:
                    raise ValueError
                if declared_length > limit:
                    raise InputError(f"managed artifact exceeds the {limit}-byte limit")
            except ValueError as exc:
                raise InputError("Content-Length must be a non-negative integer") from exc
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > limit:
                raise InputError(f"managed artifact exceeds the {limit}-byte limit")
            payload.extend(chunk)
        return bytes(payload)

    async def managed_get(uri: str, *, org_id: str, project_id: str) -> bytes:
        return await service.read_managed_artifact(org_id=org_id, project_id=project_id, uri=uri)

    async def managed_put(
        uri: str,
        payload: bytes,
        *,
        org_id: str,
        project_id: str,
        sha256: str | None = None,
        actor: str,
    ) -> ManagedArtifactRecord:
        """Reserve, conditionally write, and activate one scoped object."""

        return await service.upload_managed_artifact(
            org_id=org_id,
            project_id=project_id,
            uri=uri,
            payload=payload,
            sha256=sha256,
            actor=actor,
        )

    def verify_artifact_token(
        token: str,
        action: ArtifactAction,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
    ) -> None:
        try:
            artifact_access.signer.verify(
                token,
                action,
                uri,
                org_id=org_id,
                project_id=project_id,
                workload_id=workload_id,
            )
        except InvalidArtifactToken as exc:
            raise AuthorizationError("invalid managed artifact transfer authorization") from exc

    async def verify_and_audit_artifact_read(
        token: str,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
        kind: str,
    ) -> None:
        """Verify a read capability and record the attempt, refused or granted.

        Both outcomes are audited. A refusal is the more interesting of the two: a
        rejected token is the signal that one leaked or outlived its attempt, and it
        is invisible if only successes are recorded.
        """

        try:
            verify_artifact_token(
                token,
                ArtifactAction.READ,
                uri,
                org_id=org_id,
                project_id=project_id,
                workload_id=workload_id,
            )
        except AuthorizationError:
            # Record the denial against the WORKLOAD's durable scope: the
            # caller here is unauthenticated by definition, so their claimed
            # org/project must never choose which tenant's audit ledger an
            # event lands in. A workload id that does not resolve records
            # nothing -- there is no owner to attribute the event to.
            async with repository.transaction() as transaction:
                workload = await transaction.get_workload(workload_id)
            if workload is not None:
                await service.record_artifact_access(
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    workload_id=workload_id,
                    kind=kind,
                    object_id=uri,
                    action="artifact.read_denied",
                )
            raise

    async def current_result_uri(workload_id: str) -> str:
        """Return the signing scope URI this workload's current attempt owns.

        Read live rather than passed in by the caller: managed ``eyvan://``
        attempts receive a fresh URI per generation from
        ``result_spec_for_generation``, and external-result attempts (such as
        ``gs://``) derive ``progress_report_uri(workload.id, workload.generation)``
        so a retried attempt's bumped generation revokes its predecessor's token
        identically.

        Raises:
            AuthorizationError: If the workload has no result destination, which
                is the case for every command workload -- those have no artifact
                token either, so there is nothing this route could verify.
        """

        async with repository.transaction() as transaction:
            workload = await transaction.get_workload(workload_id)
        execution = workload.spec.execution if workload is not None else None
        uri = getattr(execution, "result_uri", None)
        if workload is None or not isinstance(uri, str) or not uri:
            raise AuthorizationError("this workload has no managed result to authenticate against")
        if uri.startswith("eyvan://"):
            return uri
        return progress_report_uri(workload.id, workload.generation)

    async def authorize_current_result_target(
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
    ) -> None:
        # The audit event is written in the *same* transaction as the authorization
        # read, so a granted upload cannot be committed unaudited.
        async with repository.transaction() as transaction:
            workload = await transaction.get_workload(workload_id)
            execution = workload.spec.execution if workload is not None else None
            granted = (
                workload is not None
                and workload.org_id == org_id
                and workload.project_id == project_id
                and workload.state
                in {
                    WorkloadState.LEASED,
                    WorkloadState.DISPATCHING,
                    WorkloadState.RUNNING,
                    WorkloadState.CHECKPOINTING,
                    WorkloadState.CANCELLING,
                }
                and isinstance(execution, CallableExecution)
                and execution.result_uri == uri
            )
            await service.record_artifact_access(
                org_id=org_id,
                project_id=project_id,
                workload_id=workload_id,
                kind="results",
                object_id=uri,
                action="artifact.write" if granted else "artifact.write_denied",
                tx=transaction,
            )
        if not granted:
            raise AuthorizationError("managed result destination is not current for this workload attempt")

    @app.get("/api/v1/capacity/queue-trend", response_model=QueueTrend, tags=["capacity"])
    async def queue_trend(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        window_s: tp.Annotated[
            float,
            Query(ge=QUEUE_TREND_MIN_WINDOW_S, le=QUEUE_TREND_MAX_WINDOW_S),
        ] = 86_400.0,
        buckets: tp.Annotated[int, Query(ge=1, le=QUEUE_TREND_MAX_BUCKETS)] = 24,
    ) -> QueueTrend:
        """Report submitted-vs-admitted counts over recent time buckets.

        Derived from the audit ledger, which already records both with timestamps
        and a monotonic sequence -- so this is a query over data that exists rather
        than a second time-series store to prune, back up, and keep consistent.
        """

        return await service.queue_trend(
            principal, org_id=org_id, project_id=project_id, window_s=window_s, buckets=buckets
        )

    @app.get("/api/v1/capacity/carbon", response_model=CarbonReport, tags=["capacity"])
    async def carbon_report(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CarbonReport:
        """Report grid intensity for the regions this fleet occupies.

        Every entry carries ``no_provider`` unless an operator has configured a
        source, and none ships. That is the answer, not a gap in it: a carbon
        feature whose unconfigured state is an empty response cannot be told
        apart from a broken one, and one whose unconfigured state is *zero*
        claims every grid is clean.

        Reporting only. Nothing in the placement path consults this, and a test
        asserts the scheduler and the planner do not import the module behind
        it -- a number nobody has validated would otherwise steer scheduling
        while looking deliberate.
        """

        return await service.carbon_report(principal)

    @app.get("/api/v1/capacity/launch-latency", response_model=LaunchLatencyReport, tags=["capacity"])
    async def launch_latency(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> LaunchLatencyReport:
        """Report how long each pool takes to hand over usable capacity.

        The same ``provision_latencies_s`` the scheduler ranks pools on, which is
        why every figure carries its sample count: a p95 over three provisions is
        not a p95, and presenting one as equivalent to a p95 over three hundred
        invites the confident wrong decision this exists to prevent.
        """

        return await service.launch_latency_report(principal)

    @app.get("/api/v1/capacity/warmth", response_model=WarmthReport, tags=["capacity"])
    async def warmth_report(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WarmthReport:
        """Report each pool's warm floor and whether it is currently held.

        Pair this with ``/capacity/billing``: warm capacity is deliberately idle,
        so its cost must be subtracted from the fleet's idle cost before that
        number means anything. Without the pairing, the only safe reaction to an
        unexplained idle figure is to delete the warm floor -- undoing the fix.
        """

        return await service.warmth_report(principal)

    @app.get("/api/v1/usage", response_model=ProjectUsageReport, tags=["capacity"])
    async def project_usage(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        org_id: str,
        project_id: str,
        since: datetime | None = None,
    ) -> ProjectUsageReport:
        """Report what each person in a project used, and what it cost.

        The fleet's spending was answerable per pool and never per person, so a
        shared project could see its bill and not its breakdown. Nothing new is
        measured: a lease carries its settled cost and accelerator hours, and
        its workload carries who submitted it.

        Read permission, not project administration. The people sharing a
        budget are exactly who should be able to see how it is being spent.
        """

        return await service.project_usage(principal, org_id=org_id, project_id=project_id, since=since)

    @app.get("/api/v1/capacity/billing", response_model=CellBillingReport, tags=["capacity"])
    async def cell_billing_report(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        pool_id: str | None = None,
    ) -> CellBillingReport:
        """Report accrued and idle cell cost across the fleet or one pool.

        The idle figure is not derivable from budgets: an idle cell holds no lease,
        so nothing settles and nothing is charged to a project. This is the only
        place a fleet's unused-but-paid-for capacity is visible.
        """

        return await service.cell_billing_report(principal, pool_id=pool_id)

    @app.get("/api/v1/capacity/quota", response_model=TpuQuotaSnapshot, tags=["capacity"])
    async def tpu_quota_snapshot(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        project: tp.Annotated[str | None, Query(min_length=1, max_length=128)] = None,
        zone: tp.Annotated[list[str] | None, Query()] = None,
        refresh: bool = False,
    ) -> TpuQuotaSnapshot:
        """Report provider TPU quota, live usage, and free balance per zone and type.

        The numbers ``eyvan autoscale generate`` already reads, answered for the
        operator choosing where to run instead of only for the generator. Two
        provider reads back it -- Spot and on-demand are separate quota metrics --
        and either may fail on its own, so a failed class is reported beside the
        rows that succeeded rather than collapsing the whole snapshot.

        A zone whose Cloud Monitoring usage series is missing is reported as
        ``usage_unknown`` with no free balance at all. It is never reported as
        zero usage, because a quota limit presented as free capacity is the one
        answer here that costs money to act on.

        Read-only and cached for ``TPU_QUOTA_CACHE_TTL_S``: every response states
        when the provider was actually read, so a cached number is never
        presented as a live one. Authorized exactly as the pool listing it reads
        its provider projects from.
        """

        pools = await service.list_capacity_pools(principal)
        projects = sorted(
            {
                pool.provider_project
                for pool in pools
                if pool.provider == "gcp" and pool.provider_project is not None and pool.provider_project.strip()
            }
        )
        explicit_project = (project or "").strip()
        if explicit_project:
            selected: str | None = explicit_project
        elif len(projects) > 1:
            selected = await asyncio.to_thread(select_preferred_gcp_project, projects)
        else:
            selected = projects[0] if projects else None
        moment = utcnow()
        if selected is None:
            return TpuQuotaSnapshot(
                observed_at=moment,
                project=None,
                available_projects=projects,
                status=TpuQuotaStatus.UNCONFIGURED,
                ttl_s=TPU_QUOTA_CACHE_TTL_S,
            )
        zones = tuple(sorted({value.strip() for value in (zone or []) if value.strip()}))
        selected_account = next(
            (
                pool.provider_account.strip()
                for pool in pools
                if pool.provider == "gcp"
                and pool.provider_project == selected
                and pool.provider_account is not None
                and pool.provider_account.strip()
            ),
            None,
        )
        # Concurrently, so the deadline the cache states is the deadline this
        # endpoint honours. Awaited in sequence, two independent provider reads
        # each bounded at 25s made the worst case 50s -- and a bound a caller
        # can exceed twice over is not a bound: the dashboard budgeted for the
        # stated one and timed out on every cold read.
        readings = list(
            await asyncio.gather(
                *(
                    quota_cache.read(
                        discovery=resolve_quota_discovery(account=selected_account),
                        project=selected,
                        zones=zones,
                        capacity_class=capacity_class,
                        refresh=refresh,
                        account=selected_account,
                    )
                    for capacity_class in (CapacityClass.SPOT, CapacityClass.ON_DEMAND)
                )
            )
        )
        observed_at = utcnow()
        entries = [
            TpuQuotaEntry.from_availability(row, capacity_class=reading.capacity_class, observed_at=observed_at)
            for reading in readings
            for row in reading.rows
        ]
        entries.sort(key=lambda entry: (entry.zone, entry.accelerator_type, entry.capacity_class))
        failures = [
            TpuQuotaFailure.from_reason(
                capacity_class=reading.capacity_class,
                reason=reading.failure,
                project=selected,
            )
            for reading in readings
            if reading.failure is not None
        ]
        refreshed_at = min(reading.refreshed_at for reading in readings)
        return TpuQuotaSnapshot(
            observed_at=observed_at,
            project=selected,
            available_projects=projects,
            # One failed purchase class is a partial answer, not an unusable one.
            # Only a snapshot with nothing readable at all is unavailable.
            status=TpuQuotaStatus.UNAVAILABLE if len(failures) == len(readings) else TpuQuotaStatus.OK,
            refreshed_at=refreshed_at,
            age_s=max((observed_at - refreshed_at).total_seconds(), 0.0),
            cached=any(reading.cached for reading in readings),
            ttl_s=TPU_QUOTA_CACHE_TTL_S,
            entries=entries,
            failures=failures,
        )

    @app.get("/api/v1/storage/status", response_model=StorageStatusResponse, tags=["storage"])
    async def get_storage_status(
        request: Request,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> StorageStatusResponse:
        """Return non-secret Eyvan storage configuration and detected GCP context.

        Authenticated like ``/api/v1/build`` and ``/api/v1/tracking/health``:
        exposes configured bucket names, locations, and detected GCP region so
        the dashboard Settings and Overview pages can display active storage
        without revealing any credentials or tenant data.
        """

        del principal
        config_path = getattr(request.app.state, "storage_config_path", None)
        payload = await asyncio.to_thread(storage_status, config_path=config_path)
        return StorageStatusResponse.model_validate(payload)

    @app.get("/api/v1/governance/summary", response_model=GovernanceSummaryResponse, tags=["governance"])
    async def get_governance_summary(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> GovernanceSummaryResponse:
        """Return plane-wide team quota, borrowing, and active reclaim summary (fleet admins)."""

        payload = await service.governance_summary(principal)
        return GovernanceSummaryResponse.model_validate(payload)

    @app.get("/api/v1/artifacts", response_model=list[ManagedArtifactRecord], tags=["artifacts"])
    async def list_managed_artifacts(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        kind: str | None = None,
        limit: int = 1000,
    ) -> list[ManagedArtifactRecord]:
        """List indexed artifacts in one project."""

        return await service.list_managed_artifacts(
            principal, org_id=org_id, project_id=project_id, kind=kind, limit=limit
        )

    @app.get(
        "/api/v1/artifacts/inventory",
        response_model=ArtifactInventoryReport,
        tags=["artifacts"],
    )
    async def inventory_managed_artifacts(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> ArtifactInventoryReport:
        """Dry-run one bounded physical-versus-index inventory page."""

        return await service.inventory_managed_artifacts(
            principal,
            org_id=org_id,
            project_id=project_id,
            cursor=cursor,
            limit=limit,
        )

    @app.post(
        "/api/v1/artifacts/inventory/reindex",
        response_model=ArtifactInventoryReport,
        tags=["artifacts"],
    )
    async def reindex_managed_artifacts(
        request: ArtifactInventoryReindexRequest,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> ArtifactInventoryReport:
        """Add metadata for physical-only objects after explicit scope acknowledgement.

        This endpoint never deletes bytes and never rewrites an existing index
        row or discrepancy. Run the GET report first.
        """

        return await service.inventory_managed_artifacts(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            cursor=request.cursor,
            limit=request.limit,
            reindex_missing=True,
            acknowledge_scope=request.acknowledge_scope,
        )

    @app.get(
        "/api/v1/artifacts/collectable",
        response_model=list[ManagedArtifactRecord],
        tags=["artifacts"],
    )
    async def collectable_artifacts(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[ManagedArtifactRecord]:
        """List artifacts nothing currently protects.

        A candidate list, not a deletion queue or a race-free deletion claim.
        Compare it with physical inventory; serialized, version-conditional
        deletion belongs to the follow-up artifact state machine.
        """

        return await service.collectable_artifacts(principal, org_id=org_id, project_id=project_id)

    @app.post("/api/v1/artifacts/collect", response_model=ArtifactCollection, tags=["artifacts"])
    async def collect_artifacts(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        dry_run: bool = True,
        grace_period_s: float = DEFAULT_ARTIFACT_GRACE_PERIOD_S,
        limit: int = 500,
    ) -> ArtifactCollection:
        """Run one bounded collection pass; dry-run reports without deleting.

        Deletion here is the store's claim/version-delete/finalize sequence,
        never an unlink: a workload that re-validated its artifact between
        listing and sweep is refused rather than destroyed.
        """

        return await service.collect_artifacts_once(
            principal=principal,
            org_id=org_id,
            project_id=project_id,
            dry_run=dry_run,
            grace_period_s=grace_period_s,
            limit=limit,
        )

    @app.post("/api/v1/artifacts/holds", response_model=ManagedArtifactRecord, tags=["artifacts"])
    async def hold_artifact(
        request: ArtifactHoldRequest,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> ManagedArtifactRecord:
        """Claim one artifact so no future collection may reclaim it.

        Attributed to the calling principal: "who said to keep this" is the only
        question ever asked of a hold, and a typed-in owner cannot be asked.
        """

        return await service.hold_artifact(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            uri=request.uri,
            reason=request.reason,
            expires_at=request.expires_at,
        )

    @app.delete("/api/v1/artifacts/holds", response_model=ManagedArtifactRecord, tags=["artifacts"])
    async def release_artifact_hold(
        org_id: str,
        project_id: str,
        uri: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> ManagedArtifactRecord:
        """Release the hold on one scoped artifact, whoever placed it."""

        return await service.release_artifact_hold(principal, org_id=org_id, project_id=project_id, uri=uri)

    @app.get("/api/v1/artifacts/usage", response_model=ArtifactUsage, tags=["artifacts"])
    async def artifact_usage(
        org_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        project_id: str | None = None,
    ) -> ArtifactUsage:
        """Report indexed artifact bytes and the quota they are measured against."""

        return await service.artifact_usage(principal, org_id=org_id, project_id=project_id)

    @app.get(
        "/api/v1/artifacts/references",
        response_model=list[ArtifactReferenceRecord],
        tags=["artifacts"],
    )
    async def artifact_references(
        org_id: str,
        project_id: str,
        uri: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        active_only: bool = False,
    ) -> list[ArtifactReferenceRecord]:
        """List the workloads that claim one scoped artifact."""

        return await service.artifact_references(
            principal,
            org_id=org_id,
            project_id=project_id,
            uri=uri,
            active_only=active_only,
        )

    @app.put("/api/v1/artifacts/callables/{digest}", tags=["artifacts"])
    async def upload_callable_artifact(
        digest: str,
        raw_request: Request,
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, str]:
        await service.authorize_callable_upload(principal, org_id=org_id, project_id=project_id)
        try:
            uri = callable_uri(digest)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        payload = await artifact_body(raw_request, limit=settings.artifact_max_callable_bytes)
        await managed_put(
            uri,
            payload,
            org_id=org_id,
            project_id=project_id,
            sha256=digest,
            actor=principal.subject,
        )
        return {"uri": uri, "sha256": digest}

    @app.put("/api/v1/artifacts/workspaces/{digest}", tags=["artifacts"])
    async def upload_workspace_artifact(
        digest: str,
        raw_request: Request,
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, str]:
        """Store one content-addressed workspace bundle.

        A bundle carries executable application code, so it is gated by the same
        callable-trust check as the callable itself rather than a weaker one.
        """

        await service.authorize_callable_upload(principal, org_id=org_id, project_id=project_id)
        try:
            uri = workspace_uri(digest)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        payload = await artifact_body(raw_request, limit=settings.artifact_max_workspace_bytes)
        await managed_put(
            uri,
            payload,
            org_id=org_id,
            project_id=project_id,
            sha256=digest,
            actor=principal.subject,
        )
        return {"uri": uri, "sha256": digest}

    @app.get(
        "/api/v1/artifacts/workspaces/{digest}/content",
        response_class=Response,
        include_in_schema=False,
    )
    async def transfer_workspace_artifact(
        digest: str,
        org_id: str,
        project_id: str,
        workload_id: str,
        token: tp.Annotated[str, Header(alias="X-Eyvan-Artifact-Token", min_length=1)],
    ) -> Response:
        try:
            uri = workspace_uri(digest)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        await verify_and_audit_artifact_read(
            token,
            uri,
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
            kind="workspaces",
        )
        payload = await managed_get(uri, org_id=org_id, project_id=project_id)
        await service.record_artifact_access(
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
            kind="workspaces",
            object_id=uri,
            action="artifact.read",
        )
        return Response(payload, media_type="application/gzip", headers={"Cache-Control": "private, no-store"})

    @app.get("/api/v1/artifacts/results/{result_id}", response_class=Response, tags=["artifacts"])
    async def download_result_artifact(
        result_id: str,
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> Response:
        await service.authorize_result_download(principal, org_id=org_id, project_id=project_id, result_id=result_id)
        try:
            uri = result_uri(result_id)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        return Response(
            await managed_get(uri, org_id=org_id, project_id=project_id),
            media_type="application/octet-stream",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get(
        "/api/v1/artifacts/callables/{digest}/content",
        response_class=Response,
        include_in_schema=False,
    )
    async def transfer_callable_artifact(
        digest: str,
        org_id: str,
        project_id: str,
        workload_id: str,
        token: tp.Annotated[str, Header(alias="X-Eyvan-Artifact-Token", min_length=1)],
    ) -> Response:
        try:
            uri = callable_uri(digest)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        await verify_and_audit_artifact_read(
            token,
            uri,
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
            kind="callables",
        )
        payload = await managed_get(uri, org_id=org_id, project_id=project_id)
        await service.record_artifact_access(
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
            kind="callables",
            object_id=uri,
            action="artifact.read",
        )
        return Response(payload, media_type="application/octet-stream", headers={"Cache-Control": "private, no-store"})

    @app.put("/api/v1/artifacts/results/{result_id}/content", include_in_schema=False)
    async def transfer_result_artifact(
        result_id: str,
        raw_request: Request,
        org_id: str,
        project_id: str,
        workload_id: str,
        token: tp.Annotated[str, Header(alias="X-Eyvan-Artifact-Token", min_length=1)],
    ) -> dict[str, str]:
        try:
            uri = result_uri(result_id)
        except ValueError as exc:
            raise InputError(str(exc)) from exc

        try:
            # ``authorize_current_result_target`` audits both of *its* outcomes,
            # but it is never reached when the token itself fails -- so the one
            # refusal that most needs recording, a capability presented against
            # an address it was not issued for, was the silent one. Same verb as
            # the authorization denial, so "a refused write" stays one predicate
            # over the ledger rather than two.
            verify_artifact_token(
                token,
                ArtifactAction.WRITE,
                uri,
                org_id=org_id,
                project_id=project_id,
                workload_id=workload_id,
            )
        except AuthorizationError:
            # Same durable-scope rule as the read path: the caller here is
            # unauthenticated by definition, so their claimed org/project must
            # never choose which tenant's audit ledger an event lands in. A
            # workload id that does not resolve records nothing -- there is no
            # owner to attribute the event to.
            async with repository.transaction() as transaction:
                workload = await transaction.get_workload(workload_id)
            if workload is not None:
                await service.record_artifact_access(
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    workload_id=workload_id,
                    kind="results",
                    object_id=uri,
                    action="artifact.write_denied",
                )
            raise
        await authorize_current_result_target(
            uri,
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
        )
        payload = await artifact_body(raw_request, limit=settings.artifact_max_result_bytes)
        await managed_put(
            uri,
            payload,
            org_id=org_id,
            project_id=project_id,
            actor=f"workload:{workload_id}",
        )
        return {"uri": uri}

    @app.get("/api/v1/secrets/content", response_class=Response, include_in_schema=False)
    async def transfer_secret(
        org_id: str,
        project_id: str,
        workload_id: str,
        name: str,
        token: tp.Annotated[str, Header(alias="X-Eyvan-Secret-Token", min_length=1)],
    ) -> Response:
        """Resolve one declared secret mount for the attempt that holds its capability.

        Out of the schema deliberately. This is not an API a person calls: it is
        redeemed by a cell agent holding a capability the control plane minted at
        dispatch, and publishing it as a documented endpoint invites exactly the
        wrong integration.

        Its own header, not ``X-Eyvan-Artifact-Token``. A redaction or logging
        rule written for one header does not silently cover a second one, and a
        capability that reads out a *value* deserves to be named distinctly from
        one that reads an artifact.
        """

        if secret_capability is None:
            raise AuthorizationError("secret mounts are not configured on this control-plane replica")
        resolved_scope = await service.secret_redemption_scope(workload_id)
        if resolved_scope is None:
            # There is no authenticated entity to own an audit event.  In
            # particular, do not persist the caller's query scope/name merely
            # because it presented an invalid capability.
            raise AuthorizationError("invalid secret capability")
        actual_org_id, actual_project_id = resolved_scope
        try:
            claims = secret_capability.verify(
                token,
                workload_id=workload_id,
                org_id=actual_org_id,
                project_id=actual_project_id,
            )
        except InvalidCapabilityToken:
            await service.record_secret_rejection(
                workload_id=workload_id,
                reason="invalid_capability",
            )
            raise AuthorizationError("invalid secret capability") from None
        if org_id != actual_org_id or project_id != actual_project_id:
            await service.record_secret_rejection(
                workload_id=workload_id,
                verified_name=claims.name,
                reason="out_of_scope",
            )
            raise AuthorizationError("secret capability was issued for a different project")
        if claims.name != name:
            await service.record_secret_rejection(
                workload_id=workload_id,
                verified_name=claims.name,
                reason="mount_mismatch",
            )
            raise AuthorizationError("secret capability was issued for a different mount")
        reference, allowed_root = await service.authorize_secret_mount(
            org_id=actual_org_id,
            project_id=actual_project_id,
            workload_id=workload_id,
            attempt=claims.attempt,
            generation=claims.generation,
            dispatch_fence=claims.dispatch_fence,
            name=claims.name,
        )
        try:
            # Off the event loop: resolving a reference shells out to the
            # provider's CLI, and file resolution repeats the matched policy root
            # in the same dirfd walk that obtains the file descriptor.
            value = await asyncio.to_thread(read_secret, reference, allowed_roots=(allowed_root,))
        except Exception:
            # Provider and custom resolver exceptions are untrusted diagnostics;
            # ordinary failures all become the same retryable response.
            raise SecretUnavailable("secret mount could not be resolved") from None
        if not isinstance(value, bytes):
            raise SecretUnavailable("secret mount returned an invalid payload")
        if len(value) > MAX_SECRET_BYTES:
            # Enforce the limit here as well as on the current worker. Otherwise
            # an older agent would still receive (and launch with) an oversized
            # value even after the worker-side hardening rolls out.
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"secret mount exceeds the {MAX_SECRET_BYTES}-byte environment limit",
            )
        return Response(
            value,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @app.post("/api/v1/secrets/rotate", response_model=CredentialRotationRecord, tags=["system"])
    async def rotate_secret_credential(
        request: SecretRotationRequest,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CredentialRotationRecord:
        """Refuse unsupported deployment-wide signing credential rotation.

        No protected shared credential-version store or startup hydration exists
        yet. Installing random material in this process would strand dispatches
        on other replicas and resurrect retired keys on restart. A ledger commit
        cannot repair that, and an uncertain commit must not leave a signer live.
        Keep the authorized route discoverable, but change neither key nor ledger.
        """

        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        del request
        raise UnavailableError(
            "secret credential rotation is unavailable until protected shared credential distribution "
            "and startup recovery are implemented; no signing material or rotation ledger was changed"
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics(raw_request: Request) -> Response:
        """Serve the Prometheus exposition to an authenticated scraper.

        This endpoint had no ``Depends`` at all while exporting
        ``eyvan_project_spend_usd`` labeled by (org, project) -- every tenant's
        spend, readable by anyone who could reach the port.

        A bearer token rather than an OIDC identity because a scraper has no
        identity to present, and the realistic alternative to a token is leaving
        it open. When no token is configured the endpoint stays open, which is
        only reachable on a loopback bind: ``ServerStartupSettings`` refuses to
        start a network-bound server without one.
        """

        expected = settings.metrics_token
        if expected:
            authorization = raw_request.headers.get("Authorization", "")
            scheme, _, presented = authorization.partition(" ")
            # Byte comparison: compare_digest rejects non-ASCII str input with
            # a TypeError, and a garbage bearer must answer 401, not 500.
            token_matches = hmac.compare_digest(
                presented.encode("utf-8"),
                expected.encode("utf-8"),
            )
            if scheme.lower() != "bearer" or not token_matches:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="metrics token required",
                    headers={"Cache-Control": "no-store"},
                )
        # The exposition is labelled by tenant and carries spend figures.
        return Response(
            generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/dashboard/config", tags=["system"])
    async def dashboard_config() -> dict[str, str | bool | None]:
        """Serve the browser's identity configuration at runtime.

        The dashboard's OIDC issuer and client id were compiled into the bundle
        by ``VITE_`` build arguments, so one image could serve exactly one
        identity provider -- and an operator with two environments had to build
        two images that differed in nothing else.

        Unauthenticated on purpose, and safe: an issuer URL and a public OAuth
        client id are both things the browser is about to send to the identity
        provider in a redirect anyway. Nothing here is a secret, and requiring a
        token to learn how to obtain a token is a loop.

        Null when unconfigured rather than absent, so a caller can tell "this
        control plane runs without OIDC" from "this control plane is too old to
        answer".

        ``development_auth`` names the third state, which the browser could not
        previously see. A plane with no OIDC is not necessarily misconfigured:
        the loopback development plane every laptop runs has none by design, and
        the dashboard used to greet it by naming two build-time ``VITE_``
        variables of a bundle the reader never built. Reported as the mode's own
        name so the page can say what is true of *this* plane rather than
        recite the general case.
        """

        return {
            "issuer": settings.oidc_issuer,
            "client_id": settings.oidc_dashboard_client_id,
            "audience": settings.oidc_audience,
            "development_auth": startup.development_auth_mode.value,
            # Availability describes the plane, not permission to use its attachment routes.
            "local_attachments_available": loopback_development,
            # The scope this plane puts a dashboard session in. Reported because
            # the dashboard used to compile in "acme"/"research" as its fallback
            # while the CLI defaults to "default"/"default", so a job submitted
            # with `eyvan run` was invisible on the page: both are unregistered
            # free-form scopes and the plane answers for either, so nothing
            # failed -- the dashboard simply asked about a different project.
            "default_org": DEFAULT_DEVELOPMENT_SCOPE,
            "default_project": DEFAULT_DEVELOPMENT_SCOPE,
        }

    @app.get("/api/v1/system/invariants", tags=["system"])
    async def fleet_invariants(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, tp.Any]:
        """Report which durability invariants the fleet is currently violating.

        The point-in-time half only. Budget monotonicity and storage growth are
        statements about a *sequence* of observations, so a single request
        cannot settle them and this does not pretend to -- a soak run holds the
        previous snapshot and compares.

        An empty ``violations`` list is a real claim, not a default: every check
        has a test that makes it fire, so silence here means the checks ran and
        found nothing rather than that nothing ran.
        """

        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        violations = await check_fleet(repository, audit_retention_s=settings.audit_retention_s)
        # Deduplicate before bounding so a repeated subject cannot evict distinct
        # evidence. Sorting also makes the cap deterministic across repository
        # implementations with different row orders.
        unique_violations: dict[tuple[str, str], Violation] = {}
        for violation in violations:
            key = (violation.invariant.value, violation.subject)
            prior = unique_violations.get(key)
            if prior is None or violation.detail < prior.detail:
                unique_violations[key] = violation
        ordered_violations = [unique_violations[key] for key in sorted(unique_violations)]
        violations_truncated = len(ordered_violations) > MAX_SERVER_VIOLATIONS
        ordered_violations = ordered_violations[:MAX_SERVER_VIOLATIONS]

        # The snapshot travels with the verdict so a long-running observer makes
        # one call per tick rather than two that describe different moments --
        # and a budget compared against an artifact count from a second later is
        # a comparison of two fleets.
        snapshot = await snapshot_fleet(repository)
        budget_items = sorted(snapshot.budget_spend.items())
        budget_spend_truncated = len(budget_items) > MAX_BUDGET_SCOPES
        budget_items = budget_items[:MAX_BUDGET_SCOPES]
        execution_items = sorted(snapshot.execution_lease_counts.items())
        coverage_truncated = len(execution_items) > MAX_COVERAGE_ENTRIES
        execution_items = execution_items[:MAX_COVERAGE_ENTRIES]
        if any(count > MAX_SAFE_COUNTER for _, count in execution_items):
            coverage_truncated = True

        return {
            "checked": [invariant.value for invariant in Invariant],
            "violations": [
                {"invariant": violation.invariant.value, "subject": violation.subject, "detail": violation.detail}
                for violation in ordered_violations
            ],
            "violations_truncated": violations_truncated,
            "snapshot": {
                "observed_at": snapshot.observed_at.isoformat(),
                "budget_spend": [
                    {
                        "scope_id": safe_scope_id(org, project),
                        "spent": observation.spent,
                        "revision": observation.revision,
                        "period_id": observation.period_id,
                        "period_started_at": (
                            observation.period_started_at.isoformat()
                            if observation.period_started_at is not None
                            else None
                        ),
                    }
                    for (org, project), observation in budget_items
                ],
                "budget_spend_truncated": budget_spend_truncated,
                "artifact_bytes": snapshot.artifact_bytes,
                "workload_count": snapshot.workload_count,
                "cell_count": snapshot.cell_count,
                "coverage": {
                    "executions": [
                        {**identity.as_dict(), "execution_count": min(count, MAX_SAFE_COUNTER)}
                        for identity, count in execution_items
                    ],
                    "truncated": coverage_truncated,
                },
                # The generic control-plane endpoint does not own process/DB or
                # provider-inventory measurement. A P8 evaluator must therefore
                # report the corresponding explicit missing-evidence codes.
                "p8_evidence": None,
            },
        }

    @app.post("/api/v1/system/soak-evidence", tags=["system"])
    async def record_soak_evidence(
        raw_request: Request,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, tp.Any]:
        """Accept one externally measured, proof-bound P8 sample.

        The control plane cannot be the source of its own resource evidence: a
        process reporting its own RSS proves nothing about a process that is
        leaking, and no amount of durable state tells it what a provider still
        bills for. So this route only *stores* what a separate monitor measured
        and posted, and :func:`proof_scoped_fleet_invariants` republishes it
        unchanged. Nothing here infers, defaults, or completes a sample.

        The body is bounded while it is read rather than after it is buffered,
        the eight closed keys must be exactly right, and every value must
        already have the type it will be republished with -- numeric text is
        refused rather than coerced, because a stored ``12`` and a posted
        ``"12"`` would be two different pieces of evidence.

        Only the most recent sample is kept, and only in this process. It is a
        measurement, not fleet state: a replica that restarts has not lost a
        record, it has stopped having a current one, and reporting no evidence
        until the monitor posts again is the truthful answer.
        """

        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        received = bytearray()
        async for chunk in raw_request.stream():
            received.extend(chunk)
            if len(received) > MAX_EVIDENCE_BODY_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"a soak evidence sample may not exceed {MAX_EVIDENCE_BODY_BYTES} bytes",
                )
        try:
            payload = json.loads(bytes(received))
        except ValueError as error:
            raise InputError("the soak evidence sample is not a JSON document") from error
        try:
            sample = ExternalEvidenceSample.from_dict(payload)
        except ValueError as error:
            # The parser's message names only a closed field name, never a
            # submitted value, so a hostile body cannot echo itself back.
            raise InputError(f"the soak evidence sample was refused: {error}") from error
        soak_evidence.record(sample)
        return {"accepted": True, "proof_id": sample.proof_id}

    @app.post("/api/v1/system/invariants/soak", tags=["system"])
    async def proof_scoped_fleet_invariants(
        payload: dict[str, tp.Any],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, tp.Any]:
        """Return invariant state plus coverage for one exact frozen proof manifest."""

        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        if set(payload) != {"proof_id", "config_sha256", "participants"}:
            raise InputError("the soak manifest has unexpected or missing fields")
        proof_id = payload["proof_id"]
        config_sha256 = payload["config_sha256"]
        raw_participants = payload["participants"]
        if not is_opaque_id(proof_id) or not is_opaque_id(config_sha256):
            raise InputError("the soak proof and configuration identifiers must be canonical sha256 values")
        if not isinstance(raw_participants, list) or not 0 < len(raw_participants) <= MAX_PARTICIPANTS:
            raise InputError("the soak manifest must contain a bounded non-empty participant list")
        try:
            participants = tuple(ParticipantBinding.from_dict(item) for item in raw_participants)
        except ValueError as error:
            raise InputError("the soak manifest contains an invalid participant binding") from error
        if len(participants) != len(set(participants)):
            raise InputError("the soak manifest contains a duplicate participant binding")
        manifest_sha256 = participant_manifest_sha256(proof_id, participants)

        violations = await check_fleet(repository, audit_retention_s=settings.audit_retention_s)
        unique_violations: dict[tuple[str, str], Violation] = {}
        for violation in violations:
            key = (violation.invariant.value, violation.subject)
            prior = unique_violations.get(key)
            if prior is None or violation.detail < prior.detail:
                unique_violations[key] = violation
        ordered_violations = [unique_violations[key] for key in sorted(unique_violations)]
        violations_truncated = len(ordered_violations) > MAX_SERVER_VIOLATIONS
        ordered_violations = ordered_violations[:MAX_SERVER_VIOLATIONS]

        snapshot = await snapshot_fleet(
            repository,
            proof_id=proof_id,
            config_sha256=config_sha256,
            participants=participants,
        )
        budget_items = sorted(snapshot.budget_spend.items())
        budget_spend_truncated = len(budget_items) > MAX_BUDGET_SCOPES
        budget_items = budget_items[:MAX_BUDGET_SCOPES]
        execution_items = sorted(snapshot.participant_execution_counts.items())
        coverage_truncated = len(execution_items) > MAX_COVERAGE_ENTRIES
        execution_items = execution_items[:MAX_COVERAGE_ENTRIES]
        if any(count > MAX_SAFE_COUNTER for _, count in execution_items):
            coverage_truncated = True
        if snapshot.participant_manifest_sha256 != manifest_sha256:
            raise RuntimeError("proof-scoped snapshot returned a mismatched participant manifest")
        evidence = soak_evidence.latest(proof_id=proof_id)

        return {
            "checked": [invariant.value for invariant in Invariant],
            "violations": [
                {"invariant": violation.invariant.value, "subject": violation.subject, "detail": violation.detail}
                for violation in ordered_violations
            ],
            "violations_truncated": violations_truncated,
            "snapshot": {
                "observed_at": snapshot.observed_at.isoformat(),
                "budget_spend": [
                    {
                        "scope_id": safe_scope_id(org, project),
                        "spent": observation.spent,
                        "revision": observation.revision,
                        "period_id": observation.period_id,
                        "period_started_at": (
                            observation.period_started_at.isoformat()
                            if observation.period_started_at is not None
                            else None
                        ),
                    }
                    for (org, project), observation in budget_items
                ],
                "budget_spend_truncated": budget_spend_truncated,
                "artifact_bytes": snapshot.artifact_bytes,
                "workload_count": snapshot.workload_count,
                "cell_count": snapshot.cell_count,
                "coverage": {
                    "proof_id": proof_id,
                    "manifest_sha256": manifest_sha256,
                    "executions": [
                        {**identity.as_dict(), "execution_count": min(count, MAX_SAFE_COUNTER)}
                        for identity, count in execution_items
                    ],
                    "truncated": coverage_truncated,
                },
                # Republished exactly as an external monitor measured it, and
                # only for this run: a sample bound to another proof, or one
                # older than the freshness bound, reads as absent rather than as
                # the last thing something happened to see. The evaluator then
                # reports its explicit *_evidence_missing codes, which is what a
                # dead monitor honestly means. With no monitor running this is
                # None, exactly as it was before the route existed.
                "p8_evidence": (evidence.as_dict() if evidence is not None else None),
            },
        }

    @app.post("/api/v1/system/reconcile", tags=["system"])
    async def reconcile_once(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, int]:
        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        result = await controller.reconcile_if_leader()
        if result is None:
            raise ConflictError("this replica does not hold scheduler leadership")
        return {
            "placements": len(result.placements),
            "preemptions": len(result.preemptions),
            "infeasibilities": len(result.infeasibilities),
        }

    @app.post(
        "/api/v1/workloads", response_model=WorkloadRecord, status_code=status.HTTP_201_CREATED, tags=["workloads"]
    )
    async def submit_workload(
        request: WorkloadSubmitRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadRecord:
        workload, replayed = await service.submit_workload(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            spec=request.spec,
            idempotency_key=idempotency_key,
            parent_workload_id=request.parent_workload_id,
        )
        if replayed:
            response.status_code = status.HTTP_200_OK
            response.headers["Idempotent-Replay"] = "true"
        if settings.correlation_enabled:
            response.headers["X-Eyvan-Correlation-ID"] = canonical_correlation_id(workload.id)
        return workload

    @app.post(
        "/api/v1/batches",
        response_model=BatchSubmitResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["workloads"],
    )
    async def submit_batch(
        request: BatchSubmitRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> BatchSubmitResponse:
        """Fan a template out into ``count`` ordinary durable workloads.

        The whole array commits in one transaction, so a partially created batch
        cannot exist. ``parent`` is a coordinator that is never scheduled;
        cancelling it cancels every task.
        """

        parent, tasks = await service.submit_batch(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            template=request.template,
            count=request.count,
            idempotency_key=idempotency_key,
            name_prefix=request.name_prefix,
            max_fan_out=request.max_fan_out,
        )
        await _report_expansion_admission(service, principal, response, parent.id, children=len(tasks))
        _set_correlation_header(response, settings.correlation_enabled, parent.id)
        # No Idempotent-Replay header: submit_batch is replay-transparent by
        # construction, returning the same parent and the same tasks in the same
        # order, so a caller cannot act differently on the two cases.
        return BatchSubmitResponse(parent=parent, tasks=tasks)

    @app.post("/api/v1/workloads/estimate", response_model=WorkloadEstimate, tags=["workloads"])
    async def estimate_workload(
        request: WorkloadEstimateRequest,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadEstimate:
        """Estimate what a workload would cost, before any record exists.

        Not a scheduling dry run: it consults declared pool attributes and prices,
        never live cell availability, because an estimate that claimed to predict
        placement would be wrong the moment the fleet moved. No ``Idempotency-Key``
        -- it writes nothing.
        """

        return await service.estimate_workload(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            spec=request.spec,
        )

    @app.post(
        "/api/v1/groups", response_model=GroupSubmitResponse, status_code=status.HTTP_201_CREATED, tags=["workloads"]
    )
    async def submit_group(
        request: GroupSubmitRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> GroupSubmitResponse:
        """Submit differently-shaped workloads that must admit together.

        Every member is created unschedulable and all are released in one
        transaction once each is independently placeable in the same pass. This is
        a feasibility gate, not a reservation: capacity can still be taken between
        that answer and the members' leases. What it prevents is half a group
        holding accelerators while the rest is structurally infeasible.
        """

        parent, members = await service.submit_group(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            roles=[GroupRole(name=role.name, template=role.template, count=role.count) for role in request.roles],
            idempotency_key=idempotency_key,
            name_prefix=request.name_prefix,
            max_members=request.max_members,
        )
        _set_correlation_header(response, settings.correlation_enabled, parent.id)
        return GroupSubmitResponse(parent=parent, members=members)

    @app.post(
        "/api/v1/dags",
        response_model=DagSubmitResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["workloads"],
    )
    async def submit_dag(
        request: DagSubmitRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> DagSubmitResponse:
        """Submit a small pipeline of dependent workloads in one transaction.

        Each node is an ordinary durable workload. A node with predecessors is
        created unschedulable and released once every predecessor has succeeded; if
        one fails or is cancelled, its dependents are cancelled rather than left
        waiting forever. Dependencies order execution only -- results travel as
        ``eyvan://`` result URIs, not through the control plane.
        """

        parent, nodes = await service.submit_dag(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            nodes=[
                DagNode(name=node.name, template=node.template, depends_on=tuple(node.depends_on))
                for node in request.nodes
            ],
            idempotency_key=idempotency_key,
            name_prefix=request.name_prefix,
            max_fan_out=request.max_fan_out,
        )
        await _report_expansion_admission(service, principal, response, parent.id, children=len(nodes))
        _set_correlation_header(response, settings.correlation_enabled, parent.id)
        return DagSubmitResponse(parent=parent, nodes=nodes)

    @app.post(
        "/api/v1/fleets/{fleet_name}/node-types/{node_type}/workloads",
        response_model=WorkloadRecord,
        status_code=status.HTTP_201_CREATED,
        tags=["workloads"],
    )
    async def submit_pool_workload(
        fleet_name: str,
        node_type: str,
        request: PoolWorkloadSubmitRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadRecord:
        workload, replayed = await service.submit_pool_workload(
            principal,
            fleet_name=fleet_name,
            node_type=node_type,
            org_id=request.org_id,
            project_id=request.project_id,
            command=request.command,
            pods=request.pods,
            zone=request.zone,
            name=request.name,
            environment=request.environment,
            secret_mounts=request.secret_mounts,
            metadata=request.metadata,
            max_retries=request.max_retries,
            max_failure_retries=request.max_failure_retries,
            max_preemption_retries=request.max_preemption_retries,
            workspace=request.workspace,
            checkpointable=request.checkpointable,
            checkpoint_hook=request.checkpoint_hook,
            checkpoint_output_uri=request.checkpoint_output_uri,
            can_submit_children=request.can_submit_children,
            priority_band=request.priority_band,
            intra_band_priority=request.intra_band_priority,
            checkpoint_grace_seconds=request.checkpoint_grace_seconds,
            reclaim_policy=request.reclaim_policy,
            estimated_duration_seconds=request.estimated_duration_seconds,
            named_assets=request.named_assets,
            dependencies=request.dependencies,
            tpu_chips=request.tpu_chips,
            setup=request.setup,
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.status_code = status.HTTP_200_OK
            response.headers["Idempotent-Replay"] = "true"
        if settings.correlation_enabled:
            response.headers["X-Eyvan-Correlation-ID"] = canonical_correlation_id(workload.id)
        return workload

    @app.post(
        "/api/v1/fleets/{fleet_name}/callables",
        response_model=WorkloadRecord,
        responses={
            status.HTTP_404_NOT_FOUND: {
                "model": NoMatchingCapacityPoolErrorResponse,
                "description": "The selected fleet has no compatible pool for the requested accelerator type.",
            }
        },
        status_code=status.HTTP_201_CREATED,
        tags=["workloads"],
    )
    async def submit_pool_callable(
        fleet_name: str,
        request: PoolCallableSubmitRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadRecord:
        workload, replayed = await service.submit_pool_callable(
            principal,
            fleet_name=fleet_name,
            org_id=request.org_id,
            project_id=request.project_id,
            accelerator_type=request.accelerator_type,
            pods=request.pods,
            capacity_class=request.capacity_class,
            execution=request.execution,
            name=request.name,
            environment=request.environment,
            secret_mounts=request.secret_mounts,
            metadata=request.metadata,
            max_retries=request.max_retries,
            max_failure_retries=request.max_failure_retries,
            max_preemption_retries=request.max_preemption_retries,
            checkpointable=request.checkpointable,
            checkpoint_hook=request.checkpoint_hook,
            checkpoint_output_uri=request.checkpoint_output_uri,
            can_submit_children=request.can_submit_children,
            priority_band=request.priority_band,
            intra_band_priority=request.intra_band_priority,
            checkpoint_grace_seconds=request.checkpoint_grace_seconds,
            reclaim_policy=request.reclaim_policy,
            estimated_duration_seconds=request.estimated_duration_seconds,
            named_assets=request.named_assets,
            dependencies=request.dependencies,
            atomic=request.atomic,
            tpu_chips=request.tpu_chips,
            setup=request.setup,
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.status_code = status.HTTP_200_OK
            response.headers["Idempotent-Replay"] = "true"
        if settings.correlation_enabled:
            response.headers["X-Eyvan-Correlation-ID"] = canonical_correlation_id(workload.id)
        return workload

    @app.get("/api/v1/workloads", response_model=list[WorkloadRecord], tags=["workloads"])
    async def list_workloads(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        state_filter: tp.Annotated[WorkloadState | None, Query(alias="state")] = None,
    ) -> list[WorkloadRecord]:
        return await service.list_workloads(principal, org_id=org_id, project_id=project_id, state=state_filter)

    @app.get("/api/v1/workloads/{workload_id}", response_model=WorkloadRecord, tags=["workloads"])
    async def get_workload(
        workload_id: str, response: Response, principal: tp.Annotated[Principal, Depends(authenticated_principal)]
    ) -> WorkloadRecord | Response:
        workload = await service.get_workload(principal, workload_id, for_status=True)
        if settings.correlation_enabled:
            response.headers["X-Eyvan-Correlation-ID"] = canonical_correlation_id(workload.id)
        if not service.authorizer.allows(
            principal, Permission.READ, org_id=workload.org_id, project_id=workload.project_id
        ):
            # Preserve the SDK record contract (including the current execution/result URI),
            # without exposing lease, endpoint or dispatch details to a submit-only owner.
            fields = {
                "id",
                "org_id",
                "project_id",
                "created_by",
                "spec",
                "state",
                "attempt",
                "generation",
                "created_at",
                "updated_at",
                "failure_reason",
                "scheduling_reason",
            }
            return Response(
                workload.model_dump_json(include=fields),
                media_type="application/json",
                headers={**response.headers, "Cache-Control": "private, no-store"},
            )
        return workload

    @app.get("/api/v1/workloads/{workload_id}/children", response_model=list[WorkloadRecord], tags=["workloads"])
    async def list_child_workloads(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[WorkloadRecord]:
        """Return the direct children of one workload in a job tree."""

        return await service.list_child_workloads(principal, workload_id)

    @app.get("/api/v1/workloads/{workload_id}/progress", response_model=WorkloadProgress, tags=["workloads"])
    async def get_workload_progress(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadProgress:
        """Return autoscaler and provider lifecycle progress for one workload."""

        return await workload_progress.resolve(principal, workload_id)

    @app.get(
        "/api/v1/workloads/{workload_id}/explain",
        response_model=WorkloadExplanation,
        tags=["workloads"],
    )
    async def explain_workload(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadExplanation:
        """Return why one workload is or is not currently placeable.

        Recomputed on demand from the pure scheduler, so it reflects the fleet
        as it is now rather than the verdict from the last scheduling pass.
        """

        return await service.explain_workload(principal, workload_id)

    @app.get(
        "/api/v1/workloads/{workload_id}/cost",
        response_model=WorkloadSpendRecord,
        tags=["workloads"],
    )
    async def workload_cost(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        on_demand_hourly: tp.Annotated[float | None, Query(ge=0)] = None,
    ) -> WorkloadSpendRecord:
        """Return what one run used and was charged, with the alternative.

        Authorized as a read of the workload it is about. The client used to
        assemble this itself, which meant listing capacity pools for their unit
        prices -- a fleet read -- so the one command that answers "what did this
        cost me" was refused to everybody who could not administer the fleet.
        """

        return await service.workload_cost(principal, workload_id, on_demand_hourly=on_demand_hourly)

    @app.post(
        "/api/v1/workloads/{workload_id}/progress-reports",
        response_model=UserProgressReport,
        include_in_schema=False,
    )
    async def report_progress(
        workload_id: str,
        request: ProgressReportRequest,
        org_id: str,
        project_id: str,
        token: tp.Annotated[str, Header(alias="X-Eyvan-Artifact-Token", min_length=1)],
    ) -> UserProgressReport:
        """Record one rank's progress, authenticated as the running attempt.

        The token is the attempt's *result-write* capability, reused rather than
        minted separately: it is already scoped to this workload, already revoked
        by a generation bump, and already delivered to every rank. A second token
        kind would be a second thing to rotate and a second way to get it wrong.

        No ``Idempotency-Key``: the write is an upsert keyed on ``(workload, rank)``,
        so it is idempotent by overwrite.
        """

        verify_artifact_token(
            token,
            ArtifactAction.WRITE,
            await current_result_uri(workload_id),
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
        )
        report = await service.record_progress_report(
            workload_id=workload_id,
            rank=request.rank,
            attempt=request.attempt,
            step=request.step,
            total=request.total,
            message=request.message,
            metrics=request.metrics,
        )
        # Only the fraction becomes a series, and only when the job declared a
        # total. The metrics dict is displayed and never exported: its keys come
        # from user code, and one mistyped name in a training loop would mint an
        # unbounded label set. Series for finished workloads are dropped by the
        # export phase, which is what keeps ``workload`` from growing forever.
        if report.fraction is not None:
            metrics.workload_progress.labels(workload=workload_id, rank=str(report.rank)).set(report.fraction)
        return report

    # response_model=None: agent inventory entries are backend-shaped JSON.
    @app.get("/api/v1/capacity/orphans", tags=["capacity"], response_model=None)
    async def capacity_orphans(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, object]:
        """List cells an agent still holds that no control-plane record claims."""

        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        return await controller.capacity_orphans()

    # response_model=None: this is a report about builds, and pinning a schema
    # to it would make an agent that reports a field this plane predates
    # undecodable -- the exact skew the route exists to surface.
    @app.get("/api/v1/capacity/builds", tags=["capacity"], response_model=None)
    async def capacity_builds(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> dict[str, object]:
        """Report which build every reachable cell agent runs, beside this plane's."""

        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        return await controller.fleet_builds()

    @app.get(
        "/api/v1/workloads/{workload_id}/logs",
        response_class=PlainTextResponse,
        tags=["workloads"],
    )
    async def workload_logs(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> str:
        await service.get_workload(principal, workload_id)
        try:
            return await controller.workload_logs(workload_id)
        except CellAgentUnavailableError:
            # An unreachable cell is routine -- spot churn, a rebooting VM -- not
            # a server fault, and a 500 here blanked the dashboard's whole detail
            # page for a workload whose record, events and leases all answered.
            # The archive still serves this attempt's history through log-page.
            return (
                "cell agent unreachable; live output is unavailable right now. "
                "Archived output remains available via 'eyvan workload logs' "
                "once the run settles or through the archive."
            )

    # response_model=None: the page is a plain JSON document whose record shape
    # LogRecord already defines; a response model would restate it.
    @app.put(
        "/api/v1/workloads/{workload_id}/endpoint",
        response_model=WorkloadRecord,
        tags=["workloads"],
        include_in_schema=False,
    )
    async def register_service_endpoint(
        workload_id: str,
        request: ServiceEndpointRequest,
        token: tp.Annotated[str, Header(alias="X-Eyvan-Workload-Token", min_length=1)],
    ) -> WorkloadRecord:
        """Publish where this replica can be reached, on its own authority.

        The same capability that authorizes child submission, because it answers
        the same question: is this request coming from the live attempt of this
        workload. A separate credential would be a second thing to deliver and
        rotate for no additional guarantee.
        """

        if workload_capability is None:
            raise ConflictError("endpoint registration requires a configured artifact token key")
        async with repository.transaction() as transaction:
            workload = await transaction.get_workload(workload_id)
        if workload is None:
            raise NotFoundError(f"workload {workload_id!r} was not found")
        try:
            workload_capability.verify(token, workload_id=workload_id, generation=workload.generation)
        except InvalidCapabilityToken as exc:
            raise AuthorizationError("invalid workload capability") from exc
        return await service.register_service_endpoint(workload=workload, endpoint=request.endpoint)

    # response_model=None: one entry per live replica, a plain JSON list.
    @app.get("/api/v1/workloads/{workload_id}/endpoints", tags=["workloads"], response_model=None)
    async def service_endpoints(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[JsonObject]:
        """List the replicas of a service that are currently able to serve."""

        return await service.service_endpoints(principal, workload_id)

    async def managed_service(principal: Principal, workload_id: str) -> tuple[WorkloadRecord, ServicePolicy]:
        """Resolve one workload as a managed service, or refuse it as absent.

        Reads through the ordinary authorized workload lookup, so a caller with
        no READ on the owning project is refused before the record's shape is
        disclosed. An ordinary workload is a 404 rather than an empty document:
        "this job has no replicas" and "this job is not the kind of thing that
        has replicas" are different answers, and returning the empty one for
        both is how a mistyped id reads as a healthy service with nothing in it.
        """

        record = await service.get_workload(principal, workload_id)
        policy = (
            None if record.coordination_kind is not CoordinationKind.SERVICE else service_policy(record.spec.metadata)
        )
        if policy is None:
            raise NotFoundError(f"workload {workload_id!r} is not a managed service")
        return record, policy

    @app.get(
        "/api/v1/workloads/{workload_id}/service-health",
        response_model=ServiceHealthResponse,
        tags=["workloads"],
    )
    async def get_service_health(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> ServiceHealthResponse:
        """Report each replica's durable health, with UNKNOWN kept distinct.

        Read-only, and it never converts an absence into a verdict: a service
        that declares no health path comes back ``probing_enabled=false`` with
        every replica ``unknown`` and ``probed=false``, which says nothing was
        asked -- not that anything failed.
        """

        record, policy = await managed_service(principal, workload_id)
        return ServiceHealthResponse.build(
            service=record,
            policy=policy,
            replicas=await service.service_health(principal, workload_id),
        )

    @app.get(
        "/api/v1/workloads/{workload_id}/service-decisions",
        response_model=list[ServiceDecisionRecord],
        tags=["workloads"],
    )
    async def list_service_decisions(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        limit: tp.Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> list[ServiceDecisionRecord]:
        """List what was observed, what was decided, and why -- newest first.

        The decision carries the observation it was taken from rather than a
        pointer to state that has since moved on, so an operator reading it
        tomorrow reconstructs the call instead of guessing at it.
        """

        await managed_service(principal, workload_id)
        return await service.service_decisions(principal, workload_id, limit=limit)

    @app.post(
        "/api/v1/workloads/{workload_id}/children",
        response_model=WorkloadRecord,
        status_code=201,
        tags=["workloads"],
        include_in_schema=False,
    )
    async def submit_child_workload(
        workload_id: str,
        request: ChildWorkloadSubmitRequest,
        token: tp.Annotated[str, Header(alias="X-Eyvan-Workload-Token", min_length=1)],
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
    ) -> WorkloadRecord:
        """Submit a child workload on a running job's own authority.

        A new header rather than reusing ``X-Eyvan-Artifact-Token``: a redaction
        rule written for one header cannot be assumed to cover another, and a
        submit capability appearing in a log because a proxy only knew about the
        artifact one is the kind of leak nobody notices.

        Not a second authentication path on ``POST /workloads`` either. That
        would mean either synthesizing a ``Principal`` -- inventing a grant no
        identity provider issued -- or duplicating the authorization decision
        inside the most security-sensitive route in the package.
        """

        if workload_capability is None:
            raise ConflictError("in-job submission requires a configured artifact token key")
        async with repository.transaction() as transaction:
            parent = await transaction.get_workload(workload_id)
        if parent is None:
            raise NotFoundError(f"workload {workload_id!r} was not found")
        parent_generation = parent.generation
        try:
            workload_capability.verify(token, workload_id=workload_id, generation=parent_generation)
        except InvalidCapabilityToken as exc:
            raise AuthorizationError("invalid workload capability") from exc
        child, _replayed = await service.submit_child_workload(
            parent=parent,
            spec=request.spec,
            idempotency_key=idempotency_key,
            # Verified against the parent as this transaction read it; the admit
            # re-reads the parent in its own transaction and refuses when the
            # generation has moved, so a token from a superseded attempt cannot
            # attach a child to the current one.
            expected_parent_generation=parent_generation,
        )
        return child

    # response_model=None: results are backend-shaped JSON per rank.
    @app.post("/api/v1/workloads/{workload_id}/exec", tags=["workloads"], response_model=None)
    async def workload_exec(
        workload_id: str,
        request: WorkloadExecRequest,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> JsonObject:
        """Run one diagnostic command beside every rank of a running workload.

        ``OPERATE`` rather than ``READ``: this starts a process on a host that
        is running someone's job. It is not a mutation of the workload, but it
        is not an observation either, and treating it as one would let a viewer
        run commands on the fleet.
        """

        workload = await service.get_workload(principal, workload_id)
        service.authorizer.require(principal, Permission.OPERATE, org_id=workload.org_id, project_id=workload.project_id)
        return await controller.workload_exec(
            workload_id, request.command, timeout_s=request.timeout_s, rank=request.rank
        )

    @app.get("/api/v1/workloads/{workload_id}/log-page", tags=["workloads"], response_model=None)
    async def workload_log_page(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        after: str = "",
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> JsonObject:
        """Return one resumable page of structured logs, merged across ranks.

        Separate from ``/logs`` rather than a format parameter on it: that route
        is declared ``PlainTextResponse`` and every consumer reads it as bytes.

        ``after`` is the ``next_cursor`` from a previous page, which carries the
        attempt it came from. A cursor from a superseded attempt is reported as
        ``attempt_changed`` and the read restarts, because the alternative --
        continuing quietly -- shows the new attempt's first lines appended to the
        old attempt's tail and reads as a job that resumed.
        """

        await service.get_workload(principal, workload_id)
        return await controller.workload_log_page(workload_id, after=after, limit=limit)

    # response_model=None: the payload is backend-shaped arbitrary JSON, and the
    # recursive JsonObject alias cannot be turned into a Pydantic response model.
    @app.get("/api/v1/workloads/{workload_id}/process-status", tags=["workloads"], response_model=None)
    async def workload_process_status(
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> JsonObject:
        """Return live process detail for one workload attempt.

        Reading process state is an observation, not a mutation, so it needs
        only the READ that ``get_workload`` already enforces for this scope.
        """

        await service.get_workload(principal, workload_id)
        return await controller.workload_process_status(workload_id)

    @app.post("/api/v1/workloads/{workload_id}/cancel", response_model=WorkloadRecord, tags=["workloads"])
    async def cancel_workload(
        workload_id: str,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        cascade: tp.Annotated[bool, Query()] = True,
    ) -> WorkloadRecord:
        """Cancel one workload, and by default every descendant under it.

        ``cascade=false`` cancels exactly this workload. That is the escape hatch
        for a stored tree deeper than the traversal limit: without it such a
        parent could never be cancelled at all, because every attempt failed on
        the walk before reaching the workload the caller actually named.
        """

        workload, replayed = await service.cancel_workload(
            principal,
            workload_id,
            idempotency_key=idempotency_key,
            cascade=cascade,
        )
        if replayed:
            response.status_code = status.HTTP_200_OK
            response.headers["Idempotent-Replay"] = "true"
        if settings.correlation_enabled:
            response.headers["X-Eyvan-Correlation-ID"] = canonical_correlation_id(workload.id)
        return workload

    @app.post("/api/v1/workloads/{workload_id}/kick", response_model=WorkloadRecord, tags=["workloads"])
    async def kick_workload(
        workload_id: str,
        response: Response,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadRecord:
        """Interrupt one attempt and let it reschedule on its preemption budget.

        The move an operator wants when an attempt is wedged on a bad host but
        the job itself is fine: stop this attempt, put the workload back in the
        queue, and let placement pick somewhere else. Cancelling would end the
        job, and waiting for a timeout wastes the accelerator until it fires.

        Charged to the *preemption* budget rather than the failure budget. A
        capacity loss is charged against ``max_retries``, which a spot job sets
        to 0 precisely because it expects to be interrupted -- so charging a kick
        there would kill the job an operator was trying to move. The audit action
        stays distinct from a provider reclamation, because who interrupted the
        attempt is exactly what a later reader needs.
        """

        workload, _lease, _budget = await service.settle_preempted_workload(
            principal,
            workload_id,
            scheduling_reason=f"kicked by {principal.subject}; waiting to resume",
            action="workload.kicked",
            provider_preempted=True,
        )
        _set_correlation_header(response, settings.correlation_enabled, workload.id)
        return workload

    @app.post("/api/v1/workloads/{workload_id}/checkpoint", response_model=WorkloadRecord, tags=["workloads"])
    async def checkpoint_workload(
        workload_id: str,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadRecord:
        workload, replayed = await service.checkpoint_workload(principal, workload_id, idempotency_key=idempotency_key)
        if replayed:
            response.status_code = status.HTTP_200_OK
            response.headers["Idempotent-Replay"] = "true"
        if settings.correlation_enabled:
            response.headers["X-Eyvan-Correlation-ID"] = canonical_correlation_id(workload.id)
        return workload

    @app.post("/api/v1/workloads/{workload_id}/retry", response_model=WorkloadRecord, tags=["workloads"])
    async def retry_workload(
        workload_id: str,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadRecord:
        workload, replayed = await service.retry_workload(principal, workload_id, idempotency_key=idempotency_key)
        if replayed:
            response.status_code = status.HTTP_200_OK
            response.headers["Idempotent-Replay"] = "true"
        if settings.correlation_enabled:
            response.headers["X-Eyvan-Correlation-ID"] = canonical_correlation_id(workload.id)
        return workload

    @app.post(
        "/api/v1/workloads/{workload_id}/force-finalize",
        response_model=WorkloadRecord,
        tags=["workloads"],
    )
    async def force_finalize_stuck_workload(
        workload_id: str,
        request: PreemptionForceFinalizeRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadRecord:
        """Settle a workload whose stop the worker will never confirm.

        Break glass. It gives up the cell rather than returning it, because the
        process may still be running on a host nobody could stop.
        """

        workload, replayed = await service.force_finalize_stuck_workload(
            principal, workload_id, reason=request.reason, idempotency_key=idempotency_key
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        _set_correlation_header(response, settings.correlation_enabled, workload.id)
        return workload

    @app.post(
        "/api/v1/preemptions/{command_id}/force-finalize",
        response_model=WorkloadRecord,
        tags=["workloads"],
    )
    async def force_finalize_preemption(
        command_id: str,
        request: PreemptionForceFinalizeRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WorkloadRecord:
        workload, replayed = await service.force_finalize_preemption(
            principal,
            command_id,
            reason=request.reason,
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.status_code = status.HTTP_200_OK
            response.headers["Idempotent-Replay"] = "true"
        if settings.correlation_enabled:
            response.headers["X-Eyvan-Correlation-ID"] = canonical_correlation_id(workload.id)
        return workload

    @app.post("/api/v1/cells", response_model=CellRecord, tags=["cells"])
    async def upsert_cell(
        request: CellUpsertRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CellRecord:
        cell, replayed = await service.upsert_cell(principal, request.to_record(), idempotency_key=idempotency_key)
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        _set_correlation_header(response, settings.correlation_enabled, cell.autoscale_request_workload_id)
        return cell

    @app.post("/api/v1/capacity-pools", response_model=CapacityPoolRecord, tags=["fleets"])
    async def upsert_capacity_pool(
        request: CapacityPoolUpsertRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CapacityPoolRecord:
        try:
            record = request.to_record(default_agent_endpoint=platform.default_agent_endpoint)
        except PlatformAgentNotConfiguredError as exc:
            raise InputError(str(exc)) from exc
        pool, replayed = await controller.upsert_capacity_pool(
            principal,
            record,
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return pool

    @app.get("/api/v1/capacity-pools", response_model=list[CapacityPoolRecord], tags=["fleets"])
    async def list_capacity_pools(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        fleet_name: str | None = None,
    ) -> list[CapacityPoolRecord]:
        return await service.list_capacity_pools(principal, fleet_name=fleet_name)

    @app.get("/api/v1/capacity-pools/{pool_id}", response_model=CapacityPoolRecord, tags=["fleets"])
    async def get_capacity_pool(
        pool_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CapacityPoolRecord:
        """Show one capacity pool record."""

        return await service.get_capacity_pool(principal, pool_id=pool_id)

    @app.delete("/api/v1/capacity-pools/{pool_id}", response_model=CapacityPoolRecord, tags=["fleets"])
    async def disable_capacity_pool(
        pool_id: str,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CapacityPoolRecord:
        pool, replayed = await controller.disable_capacity_pool(
            principal,
            pool_id,
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return pool

    @app.put("/api/v1/fleets/{fleet_name}", response_model=list[CapacityPoolRecord], tags=["fleets"])
    async def apply_fleet(
        fleet_name: str,
        request: FleetApplyRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[CapacityPoolRecord]:
        try:
            records = request.to_records(default_agent_endpoint=platform.default_agent_endpoint)
        except PlatformAgentNotConfiguredError as exc:
            raise InputError(str(exc)) from exc
        pools, replayed = await controller.apply_fleet(
            principal,
            fleet_name,
            records,
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return pools

    @app.get("/api/v1/capacity", response_model=CapacityResponse, tags=["capacity"])
    async def list_capacity(
        org_id: tp.Annotated[str, Query(min_length=1, max_length=128)],
        project_id: tp.Annotated[str, Query(min_length=1, max_length=128)],
        accelerator_type: tp.Annotated[str | None, Query(min_length=1, max_length=128)] = None,
        capacity_class: tp.Annotated[CapacityClass | None, Query()] = None,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)] = None,  # type: ignore[assignment]
    ) -> CapacityResponse:
        """List enabled pools available to a workload scope, optionally filtered by TPU shape."""

        pools = await service.list_capacity(
            principal,
            org_id=org_id,
            project_id=project_id,
            accelerator_type=accelerator_type,
            capacity_class=capacity_class,
        )
        return CapacityResponse(
            pools=[
                CapacityPoolResponse(
                    fleet=pool.fleet_name,
                    pool=pool.id,
                    accelerator_type=pool.accelerator_type or pool.node_type,
                    capacity_class=pool.capacity_class,
                    region=pool.region,
                    zone=pool.zone,
                    min_cells=pool.min_cells,
                    max_cells=pool.max_cells,
                    enabled=pool.enabled,
                )
                for pool in pools
            ]
        )

    @app.get("/api/v1/fleets", response_model=list[str], tags=["fleets"])
    async def list_submit_fleets(
        org_id: tp.Annotated[str, Query(min_length=1, max_length=128)],
        project_id: tp.Annotated[str, Query(min_length=1, max_length=128)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[str]:
        """List runnable fleet names for one authorized workload scope."""

        return await service.list_submit_fleets(principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/fleets/{fleet_name}", response_model=FleetStatusResponse, tags=["fleets"])
    async def fleet_status(
        fleet_name: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> FleetStatusResponse:
        pools = await service.list_capacity_pools(principal, fleet_name=fleet_name)
        if not pools:
            raise NotFoundError(f"fleet {fleet_name!r} was not found")
        pool_ids = {pool.id for pool in pools}
        cells = [cell for cell in await service.list_cells(principal) if cell.pool_id in pool_ids]
        return FleetStatusResponse(name=fleet_name, pools=pools, cells=cells)

    # response_model=None: a small JSON document built by the service.
    @app.get("/api/v1/fleets/{fleet_name}/init-script", tags=["fleets"], response_model=None)
    async def get_fleet_init_script(
        fleet_name: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> JsonObject:
        """Show a fleet's bash init script and whether its pools can run one (fleet admins)."""

        return await service.get_fleet_init_script(principal, fleet_name)

    @app.put("/api/v1/fleets/{fleet_name}/init-script", tags=["fleets"], response_model=None)
    async def set_fleet_init_script(
        fleet_name: str,
        request: FleetInitScriptRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> JsonObject:
        """Set (or, empty, remove) a fleet's bash init script; new cells run it before their worker starts."""

        document, replayed = await controller.set_fleet_init_script(
            principal, fleet_name, request.script, idempotency_key=idempotency_key
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return document

    @app.delete("/api/v1/fleets/{fleet_name}", response_model=list[CapacityPoolRecord], tags=["fleets"])
    async def disable_fleet(
        fleet_name: str,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[CapacityPoolRecord]:
        pools, replayed = await controller.disable_fleet(
            principal,
            fleet_name,
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return pools

    @app.get("/api/v1/cells", response_model=list[CellRecord], tags=["cells"])
    async def list_cells(principal: tp.Annotated[Principal, Depends(authenticated_principal)]) -> list[CellRecord]:
        return await service.list_cells(principal)

    @app.get("/api/v1/cells/{cell_id}", response_model=CellRecord, tags=["cells"])
    async def get_cell(
        cell_id: str,
        response: Response,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CellRecord:
        cell = await service.get_cell(principal, cell_id)
        _set_correlation_header(response, settings.correlation_enabled, cell.autoscale_request_workload_id)
        return cell

    @app.post("/api/v1/cells/{cell_id}/drain", response_model=CellRecord, tags=["cells"])
    async def drain_cell(
        cell_id: str,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CellRecord:
        cell, replayed = await controller.drain_cell(principal, cell_id, idempotency_key=idempotency_key)
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        _set_correlation_header(response, settings.correlation_enabled, cell.autoscale_request_workload_id)
        return cell

    @app.delete("/api/v1/cells/{cell_id}", response_model=CellRecord, tags=["cells"])
    async def delete_cell(
        cell_id: str,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> CellRecord:
        cell, replayed = await controller.delete_cell(principal, cell_id, idempotency_key=idempotency_key)
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        _set_correlation_header(response, settings.correlation_enabled, cell.autoscale_request_workload_id)
        return cell

    @app.post(
        "/api/v1/webhooks",
        response_model=WebhookRecord,
        status_code=status.HTTP_201_CREATED,
        tags=["governance"],
    )
    async def create_webhook(
        request: WebhookCreateRequest,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WebhookRecord:
        """Register where this project's lifecycle events should be pushed.

        The response is the only time the signing key is visible, and it is the
        one the caller supplied: reads redact it, because a signature anybody
        can forge proves nothing about who sent a delivery.
        """

        return await service.create_webhook(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            url=request.url,
            secret=request.secret,
            entity_types=request.entity_types,
            actions=request.actions,
            allow_private_targets=settings.webhook_allow_private_targets,
        )

    @app.get("/api/v1/webhooks", response_model=list[WebhookRecord], tags=["governance"])
    async def list_webhooks(
        org_id: tp.Annotated[str, Query(min_length=1, max_length=128)],
        project_id: tp.Annotated[str, Query(min_length=1, max_length=128)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[WebhookRecord]:
        """List this project's subscriptions, with their signing keys removed."""

        return await service.list_webhooks(principal, org_id=org_id, project_id=project_id)

    @app.delete("/api/v1/webhooks/{webhook_id}", response_model=WebhookRecord, tags=["governance"])
    async def delete_webhook(
        webhook_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> WebhookRecord:
        """Stop delivering to one subscription and forget it."""

        return await service.delete_webhook(principal, webhook_id)

    @app.get("/api/v1/leases", response_model=list[LeaseRecord], tags=["leases"])
    async def list_leases(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        workload_id: str | None = None,
    ) -> list[LeaseRecord]:
        return await service.list_leases(principal, org_id=org_id, project_id=project_id, workload_id=workload_id)

    @app.get("/api/v1/leases/{lease_id}", response_model=LeaseRecord, tags=["leases"])
    async def get_lease(
        lease_id: str,
        response: Response,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> LeaseRecord:
        lease = await service.get_lease(principal, lease_id)
        _set_correlation_header(response, settings.correlation_enabled, lease.workload_id)
        return lease

    @app.put("/api/v1/policies/{org_id}/{project_id}", response_model=PolicyRecord, tags=["governance"])
    async def put_policy(
        org_id: str,
        project_id: str,
        request: PolicyPutRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> PolicyRecord:
        policy, replayed = await service.put_policy(
            principal, request.to_record(org_id, project_id), idempotency_key=idempotency_key
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return policy

    @app.get("/api/v1/policies", response_model=list[PolicyRecord], tags=["governance"])
    async def list_policies(
        org_id: str, principal: tp.Annotated[Principal, Depends(authenticated_principal)]
    ) -> list[PolicyRecord]:
        return await service.list_policies(principal, org_id=org_id)

    @app.get("/api/v1/policies/{org_id}/{project_id}", response_model=PolicyRecord, tags=["governance"])
    async def get_policy(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> PolicyRecord:
        return await service.get_policy(principal, org_id=org_id, project_id=project_id)

    @app.post("/api/v1/grants", response_model=RoleGrantRecord, status_code=201, tags=["tenancy"])
    async def issue_role_grant(
        request: RoleGrantIssueRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> RoleGrantRecord:
        """Grant a role durably, without an identity-provider change."""

        grant, replayed = await service.issue_role_grant(principal, request.to_record(), idempotency_key=idempotency_key)
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
            response.status_code = 200
        return grant

    @app.get("/api/v1/grants", response_model=list[RoleGrantRecord], tags=["tenancy"])
    async def list_role_grants(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        subject: str | None = None,
        org_id: str | None = None,
    ) -> list[RoleGrantRecord]:
        """List durable grants the caller is entitled to see."""

        return await service.list_role_grants(principal, subject=subject, org_id=org_id)

    @app.delete("/api/v1/grants", status_code=204, tags=["tenancy"])
    async def revoke_role_grant(
        subject: str,
        role: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        org_id: str = "",
        project_id: str = "",
    ) -> Response:
        """Withdraw a durable grant, effective on the subject's next request."""

        removed = await service.revoke_role_grant(principal, (subject, role, org_id, project_id))
        if not removed:
            raise NotFoundError(f"no grant of {role!r} to {subject!r} in scope {org_id}/{project_id}")
        return Response(status_code=204)

    @app.put("/api/v1/federation/peers/{plane_id}", response_model=PlanePeerRecord, tags=["federation"])
    async def register_plane_peer(
        plane_id: str,
        request: PlanePeerPutRequest,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> PlanePeerRecord:
        """Register or update a peer control plane.

        Peers are configured, never discovered: a plane that could join by
        asserting it is a tenant's home plane could claim that tenant's whole
        budget, and nothing in this design has the standing to refuse it.
        """

        return await service.put_plane_peer(
            principal,
            plane_id=plane_id,
            endpoint=request.endpoint,
            description=request.description,
            enabled=request.enabled,
        )

    @app.get("/api/v1/federation/peers", response_model=list[PlanePeerRecord], tags=["federation"])
    async def list_plane_peers(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[PlanePeerRecord]:
        """List the control planes this one knows about."""

        return await service.list_plane_peers(principal)

    @app.post(
        "/api/v1/federation/budget-grants",
        response_model=BudgetGrantRecord,
        status_code=201,
        tags=["federation"],
    )
    async def issue_budget_grant(
        request: BudgetGrantIssueRequest,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> BudgetGrantRecord:
        """Grant a peer plane a slice of this tenant's budget.

        The home plane refuses to issue past its own limit, counting both what it
        has already granted away and what it has itself spent. That one check is
        the entire cross-plane guarantee -- no plane consults another at
        admission time, because the slices were bounded when they were cut.
        """

        return await service.issue_budget_grant(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            grantee_plane_id=request.grantee_plane_id,
            amount=request.amount,
            expires_at=request.expires_at,
            idempotency_key=idempotency_key,
        )

    @app.get("/api/v1/federation/budget-grants", response_model=list[BudgetGrantRecord], tags=["federation"])
    async def list_budget_grants(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        org_id: str | None = None,
        project_id: str | None = None,
        include_inactive: bool = False,
    ) -> list[BudgetGrantRecord]:
        """List the budget slices this plane has issued or been issued."""

        return await service.list_budget_grants(
            principal,
            org_id=org_id,
            project_id=project_id,
            include_inactive=include_inactive,
        )

    @app.delete("/api/v1/federation/budget-grants/{grant_id}", response_model=BudgetGrantRecord, tags=["federation"])
    async def revoke_budget_grant(
        grant_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> BudgetGrantRecord:
        """Withdraw a peer's authority to spend.

        Returns the grant rather than 204: revocation stops *new* admission and
        says nothing about what already ran, and ``reported_spend`` on the
        returned record is what that distinction looks like in numbers.
        """

        return await service.revoke_budget_grant(principal, grant_id=grant_id)

    @app.post(
        "/api/v1/federation/quota-grants",
        response_model=QuotaGrantRecord,
        status_code=201,
        tags=["federation"],
    )
    async def issue_quota_grant(
        request: QuotaGrantIssueRequest,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> QuotaGrantRecord:
        """Grant a peer plane a slice of this tenant's quota.

        Headroom has to hold in every dimension at once, and a refusal names all
        the dimensions that overran rather than the first.
        """

        return await service.issue_quota_grant(
            principal,
            org_id=request.org_id,
            project_id=request.project_id,
            grantee_plane_id=request.grantee_plane_id,
            limits=request.limits,
            expires_at=request.expires_at,
            idempotency_key=idempotency_key,
        )

    @app.get("/api/v1/federation/quota-grants", response_model=list[QuotaGrantRecord], tags=["federation"])
    async def list_quota_grants(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        org_id: str | None = None,
        project_id: str | None = None,
        include_inactive: bool = False,
    ) -> list[QuotaGrantRecord]:
        """List the quota slices this plane has issued or been issued."""

        return await service.list_quota_grants(
            principal,
            org_id=org_id,
            project_id=project_id,
            include_inactive=include_inactive,
        )

    @app.delete("/api/v1/federation/quota-grants/{grant_id}", response_model=QuotaGrantRecord, tags=["federation"])
    async def revoke_quota_grant(
        grant_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> QuotaGrantRecord:
        """Withdraw a peer's authority to place work against this tenant's quota.

        Stops new admission and does not disturb what is already running: a quota
        is a concurrent ceiling, and evicting live work to enforce a ceiling
        change would make revocation an outage rather than an administrative act.
        """

        return await service.revoke_quota_grant(principal, grant_id=grant_id)

    @app.put("/api/v1/organizations/{org_id}", response_model=OrganizationRecord, tags=["tenancy"])
    async def register_organization(
        org_id: str,
        request: OrganizationRegisterRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> OrganizationRecord:
        """Register a tenant this control plane serves."""

        organization, replayed = await service.register_organization(
            principal,
            request.to_record(org_id, created_by=principal.subject),
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return organization

    @app.get("/api/v1/organizations", response_model=list[OrganizationRecord], tags=["tenancy"])
    async def list_organizations(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[OrganizationRecord]:
        """List tenants the caller has any grant for; everything, for a fleet admin."""

        return await service.list_organizations(principal)

    @app.get("/api/v1/organizations/{org_id}", response_model=OrganizationRecord, tags=["tenancy"])
    async def get_organization(
        org_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> OrganizationRecord:
        return await service.get_organization(principal, org_id)

    @app.put("/api/v1/organizations/{org_id}/projects/{project_id}", response_model=ProjectRecord, tags=["tenancy"])
    async def register_project(
        org_id: str,
        project_id: str,
        request: ProjectRegisterRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> ProjectRecord:
        project, replayed = await service.register_project(
            principal,
            request.to_record(org_id, project_id, created_by=principal.subject),
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return project

    @app.get("/api/v1/organizations/{org_id}/projects", response_model=list[ProjectRecord], tags=["tenancy"])
    async def list_projects(
        org_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> list[ProjectRecord]:
        return await service.list_projects(principal, org_id=org_id)

    @app.put("/api/v1/budgets/{org_id}/{project_id}", response_model=BudgetRecord, tags=["governance"])
    async def put_budget(
        org_id: str,
        project_id: str,
        request: BudgetPutRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> BudgetRecord:
        budget, replayed = await service.put_budget(
            principal, request.to_record(org_id, project_id), idempotency_key=idempotency_key
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return budget

    @app.get("/api/v1/budgets", response_model=list[BudgetRecord], tags=["governance"])
    async def list_budgets(
        org_id: str, principal: tp.Annotated[Principal, Depends(authenticated_principal)]
    ) -> list[BudgetRecord]:
        return await service.list_budgets(principal, org_id=org_id)

    @app.get("/api/v1/budgets/{org_id}/{project_id}", response_model=BudgetRecord, tags=["governance"])
    async def get_budget(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> BudgetRecord:
        return await service.get_budget(principal, org_id=org_id, project_id=project_id)

    @app.get(
        "/api/v1/budgets/{org_id}/{project_id}/status",
        response_model=BudgetStatus,
        tags=["governance"],
    )
    async def budget_status(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> BudgetStatus:
        """Report a budget's period window and its projected end-of-period spend.

        The forecast assumes every active lease runs to the end of the window --
        the pessimistic reading, deliberately, because this number exists to warn
        before a hard limit starts refusing admission.
        """

        return await service.budget_status(principal, org_id=org_id, project_id=project_id)

    @app.post(
        "/api/v1/budgets/{org_id}/{project_id}/roll",
        response_model=BudgetRecord,
        tags=["governance"],
    )
    async def roll_budget_period(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> BudgetRecord:
        """Close an elapsed budget period without waiting for the next settlement.

        Not a force-reset: a period that has not elapsed is left alone, because
        zeroing it would erase spend the limit is still meant to bind. Idempotent
        -- rolling twice in one period is the second call returning the record
        unchanged.
        """

        return await service.roll_budget_period(principal, org_id=org_id, project_id=project_id)

    @app.put("/api/v1/quotas", response_model=QuotaRecord, tags=["governance"])
    async def put_quota(
        request: QuotaPutRequest,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
    ) -> QuotaRecord:
        quota, replayed = await service.put_quota(principal, request.to_record(), idempotency_key=idempotency_key)
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        return quota

    @app.get("/api/v1/quotas", response_model=list[QuotaRecord], tags=["governance"])
    async def list_quotas(
        org_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        project_id: str | None = None,
        principal_id: str | None = None,
    ) -> list[QuotaRecord]:
        return await service.list_quotas(principal, org_id=org_id, project_id=project_id, principal_id=principal_id)

    @app.get("/api/v1/quotas/current", response_model=EffectiveQuota, tags=["governance"])
    async def get_quota(
        org_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        project_id: str | None = None,
        principal_id: str | None = None,
    ) -> EffectiveQuota:
        """Return every quota that applies here, rather than one stored row.

        A scope with nothing declared is reported as declaring nothing, which
        is what admission does with it. This route used to 404 when no exact
        row existed, which called an unconfigured fleet an error.
        """

        return await service.get_quota(principal, org_id=org_id, project_id=project_id, principal_id=principal_id)

    @app.delete("/api/v1/quotas", status_code=status.HTTP_204_NO_CONTENT, tags=["governance"])
    async def delete_quota(
        org_id: str,
        response: Response,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        project_id: str | None = None,
        principal_id: str | None = None,
    ) -> Response:
        replayed = await service.delete_quota(
            principal,
            org_id=org_id,
            project_id=project_id,
            principal_id=principal_id,
            idempotency_key=idempotency_key,
        )
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @app.get("/api/v1/events", response_model=list[AuditEvent], tags=["events"])
    async def list_events(
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        after_sequence: int = Query(default=0, ge=0),
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[AuditEvent]:
        return await service.list_events(
            principal,
            org_id=org_id,
            project_id=project_id,
            after_sequence=after_sequence,
            entity_type=entity_type,
            entity_id=entity_id,
            limit=limit,
        )

    @app.get("/api/v1/fleet/events", response_model=list[AuditEvent], tags=["events"])
    async def list_fleet_events(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        after_sequence: int = Query(default=0, ge=0),
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[AuditEvent]:
        return await service.list_fleet_events(
            principal,
            after_sequence=after_sequence,
            entity_type=entity_type,
            entity_id=entity_id,
            limit=limit,
        )

    @app.get("/api/v1/events/stream", tags=["events"])
    async def stream_events(
        raw_request: Request,
        org_id: str,
        project_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        after_sequence: int = Query(default=0, ge=0),
        last_event_id: tp.Annotated[str | None, Header(alias="Last-Event-ID")] = None,
        follow: bool = True,
    ) -> StreamingResponse:
        if last_event_id is not None:
            try:
                after_sequence = max(after_sequence, int(last_event_id))
            except ValueError as exc:
                raise InputError("Last-Event-ID must be an integer") from exc
        service.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)

        async def event_source() -> tp.AsyncIterator[str]:
            # Repositories serialize audit allocation through commit/rollback:
            # a higher visible sequence cannot hide a later lower commit.
            # Retain the existing bounded lookback/reconnect wire behavior for
            # compatibility, but completeness now comes from that repository
            # contract, not from a finite lookback. Rollback gaps need no resync.
            cursor = after_sequence
            delivered: collections.OrderedDict[str, None] = collections.OrderedDict()

            def remember(event_id: str) -> bool:
                """Record an id and report whether it is newly delivered."""

                if event_id in delivered:
                    return False
                delivered[event_id] = None
                while len(delivered) > _AUDIT_STREAM_DEDUPE_LIMIT:
                    delivered.popitem(last=False)
                return True

            while True:
                events = await service.list_events(
                    principal,
                    org_id=org_id,
                    project_id=project_id,
                    after_sequence=max(0, cursor - _AUDIT_STREAM_LOOKBACK),
                    limit=1000,
                )
                for event in events:
                    cursor = max(cursor, event.sequence)
                    if not remember(event.id):
                        continue
                    yield sse_frame(
                        data=event.model_dump(mode="json"),
                        event=event.action,
                        identifier=str(event.sequence),
                    )
                if not follow or await raw_request.is_disconnected():
                    return
                new_events = await service.wait_for_events(
                    principal,
                    org_id=org_id,
                    project_id=project_id,
                    after_sequence=cursor,
                    timeout=15.0,
                )
                if not new_events:
                    yield KEEPALIVE_FRAME
                else:
                    await asyncio.sleep(0)

        return sse_response(event_source())

    @app.get("/api/v1/workloads/{workload_id}/logs/stream", tags=["workloads"])
    async def stream_workload_logs(
        raw_request: Request,
        workload_id: str,
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        after: str = "",
        last_event_id: tp.Annotated[str | None, Header(alias="Last-Event-ID")] = None,
        follow: bool = True,
        limit: int = Query(default=200, ge=1, le=1000),
        include_status: bool = False,
    ) -> StreamingResponse:
        """Follow one workload's structured logs as they are produced.

        A distinct route rather than a reuse of ``/events/stream``: that stream
        is scoped to (org, project) and keyed on the audit table's BIGSERIAL,
        while log records are per-(workload, attempt, rank) and have no global
        sequence at all. They share this module's transport and nothing else.

        ``Last-Event-ID`` is the same ``"{attempt}:{rank}=cursor|..."`` the
        paged route returns, so a browser reconnecting resumes every rank where
        it left off rather than replaying from the tail.

        ``include_status=true`` opts into bounded ``log.page_status`` events,
        including on empty pages, distinguishing unavailable, suppressed or
        incomplete output from healthy silence. Unchanged status is not repeated.
        The default retains the legacy record-only wire format during rollout.

        Authorization happens here, before the generator exists. A check inside
        it would run after the response was committed, and the caller would see
        ``200`` followed by silence -- indistinguishable from a quiet job.
        """

        await service.get_workload(principal, workload_id)
        cursor = last_event_id or after
        # Preflight the cursor here, for the same reason authorization happens
        # here: the generator runs after the response is committed, so a refusal
        # raised inside it reaches the caller as ``200`` followed by silence --
        # which reads exactly like a quiet job. A browser reconnecting with a
        # corrupted ``Last-Event-ID`` has to be told its position is unusable.
        # Decoded against exactly the read the page would serve, so the preflight
        # and the served page can never disagree about whether a cursor is usable.
        await controller.validate_log_cursor(workload_id, cursor)

        async def log_source() -> tp.AsyncIterator[str]:
            position = cursor
            previous_status: JsonObject | None = None
            while True:
                page = await controller.workload_log_page(workload_id, after=position, limit=limit)
                records = page.get("records")
                advanced = page.get("next_cursor")
                frames: list[tuple[object, str]] = []
                if page.get("attempt_changed"):
                    # Exactly one typed event, then the cursor resets. Silently
                    # continuing would show the new attempt's first lines
                    # appended to the old attempt's tail, and the reader would
                    # conclude the job resumed where it left off.
                    frames.append(({"attempt": page.get("attempt")}, "log.attempt_changed"))
                if isinstance(records, list) and records:
                    frames.extend((record, "log.record") for record in records)
                if include_status:
                    status_document: JsonObject = {
                        field: page.get(field)
                        for field in (
                            "attempt",
                            "unavailable",
                            "detail",
                            "truncated_ranks",
                            "topology_state",
                            "topology",
                            "expected_ranks",
                            "missing_ranks",
                            "completeness_unknown",
                        )
                    }
                    # Status is a bounded diagnostic, not another log page.
                    # State whether lists/text were clipped so clipping itself
                    # cannot make a partial diagnostic look complete.
                    status_truncated = False
                    for field in ("truncated_ranks", "missing_ranks"):
                        ranks = status_document[field]
                        if isinstance(ranks, list) and len(ranks) > 1000:
                            status_document[field] = ranks[:1000]
                            status_truncated = True
                    detail = status_document["detail"]
                    if isinstance(detail, str) and len(detail) > 1024:
                        status_document["detail"] = detail[:1024]
                        status_truncated = True
                    status_document["status_truncated"] = status_truncated
                    if status_document != previous_status:
                        frames.append((status_document, "log.page_status"))
                        previous_status = status_document
                # The id a reconnecting reader hands back as ``Last-Event-ID``.
                # This route reads that header and resumes from it, so without
                # writing it the resume was half a protocol: every drop replayed
                # the attempt from the beginning and duplicated what the reader
                # already had.
                #
                # It rides the page's *last* frame, and it is the page cursor
                # rather than a record's own. A record's cursor addresses one
                # rank's journal in one backend's private encoding; the position
                # that can be resumed from is the joined one, and it is only
                # true once every record on the page has been delivered. A drop
                # mid-page therefore resumes at the page boundary before it and
                # redelivers that page -- at-least-once, which is the honest
                # guarantee for a stream that cannot acknowledge.
                resume = advanced if isinstance(advanced, str) and advanced else None
                for index, (data, event) in enumerate(frames):
                    yield sse_frame(
                        data=data,
                        event=event,
                        identifier=resume if index == len(frames) - 1 else None,
                    )
                if isinstance(advanced, str) and advanced:
                    position = advanced
                if not follow or await raw_request.is_disconnected():
                    return
                if not (isinstance(records, list) and records):
                    # Nothing new. The keep-alive is what tells every buffer
                    # between here and the client that the stream is alive.
                    yield KEEPALIVE_FRAME
                    await asyncio.sleep(_LOG_STREAM_POLL_S)
                else:
                    await asyncio.sleep(0)

        return sse_response(log_source())

    dashboard_dir = dashboard_bundle()
    if dashboard_dir is not None:
        app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")
    else:
        # Registered only when nothing is mounted, because a route at "/" would
        # otherwise shadow the dashboard's own index. Every Eyvan command prints
        # this plane's URL, so somebody follows it eventually, and a bare
        # `{"detail":"Not Found"}` reads as a broken plane rather than as one
        # that serves an API and no UI.
        @app.get("/", include_in_schema=False, response_class=HTMLResponse)
        async def dashboard_absent() -> str:
            """Say what this plane is, and where its UI went."""

            return NO_DASHBOARD_PAGE

    return app


def create_loopback_app() -> FastAPI:
    """Build the zero-configuration development app for a loopback server bind."""

    return create_app(server_bind=ServerBindPolicy.from_host("127.0.0.1"))


def create_network_app() -> FastAPI:
    """Build a network-exposed app without implicit callable trust."""

    return create_app(server_bind=ServerBindPolicy.from_host("0.0.0.0"))


def _local_daemon_settings(
    settings: EyvanSettings,
    *,
    server_bind: ServerBindPolicy,
    port: int,
    public_url: str,
    local_state_path: Path,
    local_artifact_token_key_path: Path,
    artifact_store_uri: str,
) -> EyvanSettings:
    """Build the isolated, local-only settings used by an internally supervised daemon."""

    endpoint = ControlPlaneEndpoint.parse(public_url)
    parsed = urlsplit(endpoint.url)
    if parsed.scheme != "http" or parsed.path:
        raise ValueError("a local Eyvan daemon public URL must be a path-free HTTP URL")
    if parsed.hostname is None:
        raise ValueError("a local Eyvan daemon public URL must include a literal loopback host")
    try:
        public_address = ipaddress.ip_address(parsed.hostname)
        bind_address = ipaddress.ip_address(server_bind.host)
    except ValueError as exc:
        raise ValueError("a local Eyvan daemon requires literal loopback bind and public addresses") from exc
    if not server_bind.is_loopback or not public_address.is_loopback or public_address != bind_address:
        raise ValueError("a local Eyvan daemon public URL must exactly match its literal loopback bind address")
    public_port = parsed.port or 80
    if public_port != port:
        raise ValueError("a local Eyvan daemon public URL port must match its server port")
    return replace(
        settings,
        repository_backend=RepositoryBackend.LOCAL,
        local_state_path=local_state_path,
        public_url=endpoint.url,
        dev_auth_enabled=True,
        artifact_store_uri=artifact_store_uri,
        artifact_token_key=None,
        local_artifact_token_key_path=local_artifact_token_key_path,
    )


def create_local_daemon_app(
    *,
    server_bind: ServerBindPolicy,
    port: int,
    public_url: str,
    local_state_path: Path,
    local_artifact_token_key_path: Path,
    artifact_store_uri: str,
    runtime_profile_path: Path | None = None,
    settings: EyvanSettings | None = None,
    team: bool = False,
) -> FastAPI:
    """Build one local daemon with endpoint-matched transfer URLs and isolated persistence.

    ``team`` is the supervisor's ``--team``. A supervised local daemon is an
    isolated loopback profile, so it takes team mode only from that flag, not
    from a host-wide platform file.
    """

    if runtime_profile_path is not None:
        if settings is not None:
            raise ValueError("a local Eyvan daemon accepts runtime_profile_path or settings, not both")
        runtime_profile = GcpNativeRuntimeProfileStore(runtime_profile_path).load()
        configured_settings = EyvanSettings.from_gcp_native_runtime_profile(runtime_profile)
        configured_artifact_store_uri = runtime_profile.artifact_root_uri
    else:
        configured_settings = EyvanSettings.from_env() if settings is None else settings
        configured_artifact_store_uri = artifact_store_uri
    daemon_settings = _local_daemon_settings(
        configured_settings,
        server_bind=server_bind,
        port=port,
        public_url=public_url,
        local_state_path=local_state_path,
        local_artifact_token_key_path=local_artifact_token_key_path,
        artifact_store_uri=configured_artifact_store_uri,
    )
    if team:
        daemon_settings = replace(daemon_settings, team_mode=True)
    platform_config = (
        EyvanPlatformConfig(default_agent_endpoint=DEFAULT_LOCAL_CELL_AGENT_URL)
        if runtime_profile_path is not None
        else None
    )
    return create_app(
        settings=daemon_settings,
        server_bind=server_bind,
        automatic_loopback_auth=True,
        platform_config=platform_config,
    )


def main() -> None:
    """Run the development API server."""

    import sys

    if sys.argv[1:2] == ["bootstrap-admin"]:
        # Host-side: create the first fleet-admin invite of a team-mode plane.
        from eyvan.server.api.bootstrap_admin import main as bootstrap_main

        raise SystemExit(bootstrap_main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description="Run the Eyvan control plane")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--insecure-plaintext",
        action="store_true",
        help="allow a network bind to serve plaintext HTTP (equivalent to EYVAN_INSECURE_PLAINTEXT_BIND=1)",
    )
    parser.add_argument(
        "--team",
        action="store_true",
        help="run team accounts (users, invites, sign-in); the same as `team: true` in the platform file",
    )
    parser.add_argument("--local-daemon", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--public-url", help=argparse.SUPPRESS)
    parser.add_argument("--local-state-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--local-artifact-token-key-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--artifact-store-uri", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-profile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--local-daemon-record-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-level",
        type=ServerLogLevel,
        choices=tuple(ServerLogLevel),
        default=ServerLogLevel.INFO,
    )
    args = parser.parse_args()
    # Validate distribution before serving: a combined CA/key file must never
    # reach a job environment. Keep the error free of certificate/key bytes.
    from eyvan._internal.plane_tls import ControlPlaneCAError, control_plane_ca_b64

    try:
        control_plane_ca_b64()
    except ControlPlaneCAError as exc:
        parser.error(str(exc))
    if args.insecure_plaintext and (os.getenv("EYVAN_TLS_CERT_FILE") or os.getenv("EYVAN_TLS_KEY_FILE")):
        parser.error("--insecure-plaintext conflicts with EYVAN_TLS_CERT_FILE/EYVAN_TLS_KEY_FILE")
    if args.reload and (args.team or args.insecure_plaintext):
        # A reload worker is a fresh process that rebuilds the app from an import
        # string; flags given here cannot reach it without writing os.environ,
        # which is exactly what neither flag may do. The platform file's
        # ``team: true`` still applies under --reload.
        parser.error("--reload does not support --team or --insecure-plaintext")
    server_bind = ServerBindPolicy.from_host(args.host)
    required_local_options = (
        args.public_url,
        args.local_state_path,
        args.local_artifact_token_key_path,
        args.artifact_store_uri,
    )
    if not args.local_daemon:
        if any(option is not None for option in (*required_local_options, args.runtime_profile)):
            parser.error("local daemon options require --local-daemon")
        if args.reload:
            uvicorn.run(
                server_bind.application_factory,
                factory=True,
                host=server_bind.host,
                port=args.port,
                reload=True,
                log_config=server_log_config(args.log_level),
                log_level=args.log_level.value,
                **uvicorn_tls_kwargs(EyvanSettings.from_env()),
            )
            return
        # Settings are resolved once (EYVAN_*, platform file, flags), validated
        # once, and the very app built from them is what uvicorn serves.
        try:
            startup = ServerStartupSettings.from_environment(
                server_bind, team=args.team, insecure_plaintext=args.insecure_plaintext
            )
            application = create_app(server_startup=startup)
        except ValueError as exc:
            raise SystemExit(f"eyvan-server: {exc}") from exc
        uvicorn.run(
            application,
            host=server_bind.host,
            port=args.port,
            log_config=server_log_config(args.log_level),
            log_level=args.log_level.value,
            **uvicorn_tls_kwargs(startup.settings),
        )
        return
    if args.reload:
        parser.error("--local-daemon does not support --reload")
    if any(option is None for option in required_local_options):
        parser.error("--local-daemon requires its internal public URL and local profile paths")
    assert args.public_url is not None
    assert args.local_state_path is not None
    assert args.local_artifact_token_key_path is not None
    assert args.artifact_store_uri is not None
    try:
        refuse_removed_team_variable()
        application = create_local_daemon_app(
            server_bind=server_bind,
            port=args.port,
            public_url=args.public_url,
            local_state_path=args.local_state_path,
            local_artifact_token_key_path=args.local_artifact_token_key_path,
            artifact_store_uri=args.artifact_store_uri,
            runtime_profile_path=args.runtime_profile,
            team=args.team,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.local_daemon_record_path is not None:
        # Claim ownership as the process that is about to serve. The supervisor
        # writes this file first, from the PID it launched, so a process that
        # dies before this point is still cleaned up -- but only this process
        # knows which PID ends up holding the port. A record naming any other
        # process makes every later native command refuse with "not owned by the
        # selected native GCP runtime profile", and the only way out was editing
        # the record's JSON by hand.
        with record_publication_lock(args.local_daemon_record_path):
            write_owner_only_record(
                args.local_daemon_record_path,
                LocalDaemonRecord(
                    endpoint=ControlPlaneEndpoint.parse(args.public_url),
                    pid=os.getpid(),
                    runtime_profile_path=runtime_profile_identifier(
                        args.runtime_profile.expanduser().resolve() if args.runtime_profile is not None else None
                    ),
                    # The build this very process pinned when its app was created:
                    # the one its cell agent must match (serving_source_digest).
                    source_digest=build_identity().source_digest,
                ),
            )
    if args.runtime_profile is not None:
        # A local GCP plane keeps every TPU the operator's account can see listed
        # as a ready fleet name. Read-only GETs on a daemon thread, started here
        # rather than in the app factory so only the real process ever calls GCP.
        from eyvan.capacity.providers.gcp.tpu_discovery import PeriodicDiscovery

        PeriodicDiscovery(
            on_error=lambda exc: logger.warning("Eyvan TPU discovery refresh failed; keeping the last list: %s", exc)
        ).start()
    uvicorn.run(
        application,
        host=server_bind.host,
        port=args.port,
        log_config=server_log_config(args.log_level),
        log_level=args.log_level.value,
        **uvicorn_tls_kwargs(EyvanSettings.from_env()),
    )


if __name__ == "__main__":
    main()
