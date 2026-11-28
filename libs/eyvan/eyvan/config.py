# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Process configuration loaded from environment variables."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

from eyvan._internal.client_version import validate_minimum_client_version
from eyvan._internal.compat import StrEnum
from eyvan.capacity.providers.gcp.runtime.profile import (
    DEFAULT_GCP_NATIVE_FLEET_STATE_PATH,
    GcpNativeRuntimeProfile,
)
from eyvan.capacity.stage_policy import CellStagePolicy, StageEvictionPolicy
from eyvan.identity.mtls import MutualTlsFiles, MutualTlsMode, parse_mutual_tls_mode
from eyvan.soak_evidence import DEFAULT_EVIDENCE_MAX_AGE_S
from eyvan.worker.runtime._multihost_site import GangInitializationPolicy

DEFAULT_LOCAL_CONTROL_PLANE_PATH = Path("~/.eyvan/control-plane.json")
"""Where a loopback control plane keeps its durable document by default."""

DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH = Path("~/.eyvan/artifact-token.key")
"""Where a loopback control plane keeps its artifact-token signing key."""


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _optional_path(name: str) -> Path | None:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else None


class RepositoryBackend(StrEnum):
    """Supported control-plane persistence backends."""

    MEMORY = "memory"
    LOCAL = "local"
    POSTGRES = "postgres"


def _repository_backend(value: str) -> RepositoryBackend:
    try:
        return RepositoryBackend(value)
    except ValueError as exc:
        choices = ", ".join(backend.value for backend in RepositoryBackend)
        raise ValueError(f"EYVAN_REPOSITORY_BACKEND must be one of: {choices}") from exc


@dataclass(frozen=True, slots=True)
class EyvanSettings:
    """Runtime settings shared by the API, scheduler, and cell agent."""

    # Loopback-development default, password-less on purpose: a static shipped
    # credential would be one every installation shares, and the local default
    # relies on the host's trust/peer auth. Any real deployment sets
    # EYVAN_DATABASE_URL (or a secret reference) with its own credentials.
    database_url: str = "postgresql+asyncpg://eyvan@127.0.0.1:5432/eyvan"
    # The in-memory backend is a loopback development default; eyvan.server.api.server_startup refuses to
    # start a network-bound server unless EYVAN_REPOSITORY_BACKEND=postgres is set explicitly.
    repository_backend: RepositoryBackend = RepositoryBackend.MEMORY
    local_state_path: Path = DEFAULT_LOCAL_CONTROL_PLANE_PATH
    auto_create_schema: bool = False
    controller_enabled: bool = True
    public_url: str = "http://127.0.0.1:8080"
    #: Optional read-only tracker proxy and direct-to-store job tracking.
    tracker_url: str | None = None
    tracker_token: str | None = field(default=None, repr=False)
    tracking_store_uri: str | None = None
    #: TLS material for the API server's listener. A network-bound bind refuses
    #: to start without both files (or an explicit insecure opt-in), because
    #: OIDC bearer tokens, bootstrap credentials, and capability tokens would
    #: otherwise cross the network in cleartext. Loopback binds are unaffected.
    tls_cert_file: Path | None = None
    tls_key_file: Path | None = None
    allow_insecure_plaintext_bind: bool = False
    oidc_issuer: str | None = None
    #: Possession-based identity for a head that has no issuer yet. Unlike
    #: dev auth it trusts a generated secret, not a caller-supplied header.
    bootstrap_token: str | None = None
    oidc_audience: str = "eyvan"
    oidc_jwks_url: str | None = None
    #: The public OAuth client the *browser* uses. Distinct from the API's
    #: audience: the dashboard is a public client doing PKCE, and the API
    #: validates tokens minted for it. Baking this into the image at build time
    #: meant one image could serve exactly one identity provider.
    oidc_dashboard_client_id: str | None = None
    dev_auth_enabled: bool = False
    #: Opt-in team accounts (users, invites, password sign-in, personal API
    #: tokens). False keeps every existing route, auth path and wire shape.
    #: Never read from the environment: ``eyvan-server --team`` or ``team: true``
    #: in the platform file turn it on (``eyvan._internal.platform.resolve_team_mode``).
    team_mode: bool = False
    #: Opt-in durable security audit; false preserves the existing event stream.
    auth_audit_enabled: bool = False
    #: Opt-in correlation envelopes/headers; disabled preserves legacy wire shapes.
    correlation_enabled: bool = False
    # Unset accepts every client. See eyvan._internal.client_version for why the gate is one-sided.
    min_client_version: str | None = None
    trusted_callable_projects: tuple[str, ...] = ()
    artifact_store_uri: str = "~/.eyvan/managed-artifacts"
    artifact_token_key: str | None = None
    local_artifact_token_key_path: Path = DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH
    artifact_token_ttl_s: int = 604800
    artifact_max_callable_bytes: int = 64 * 1024 * 1024
    artifact_max_result_bytes: int = 256 * 1024 * 1024
    artifact_max_workspace_bytes: int = 64 * 1024 * 1024
    #: How long a secret-read capability stays valid. Short next to the artifact
    #: TTL because a secret capability hands back a *value*: once read, no later
    #: revocation can un-read it.
    #:
    #: Not shorter than an hour, though. The token is minted at dispatch and
    #: presented when the cell launches the unit, and between those two moments
    #: sit provisioning, an image pull and a workspace download. A TTL tuned to
    #: how long the fetch takes would expire during a slow bootstrap and fail
    #: the workload for a reason that has nothing to do with its secrets. The
    #: binding that actually revokes is the attempt, not this.
    secret_token_ttl_s: int = 3600
    #: Ceiling on one ad-hoc read-only query. Bounded so a mistyped join
    #: cannot hold a database connection for the length of an incident.
    readonly_query_timeout_ms: int = 15_000
    scheduler_interval_s: float = 2.0
    scheduler_idle_interval_s: float = 15.0
    """Longest wait between leader cycles while the plane is idle.

    Applies only to the single-process memory and local JSON repositories.
    After a cycle that committed nothing, with no live workload and no
    service to reconcile, the loop waits up to this long, waking at once on
    any commit and no later than the next due interval phase. Any value not
    above ``scheduler_interval_s`` (including ``0``) keeps the fixed cadence,
    the behaviour before this setting existed.
    """
    cell_reconcile_interval_s: float = 15.0
    cell_reconcile_concurrency: int = 8
    cell_heartbeat_persist_interval_s: float = 300.0
    """How often an observation that changed nothing but a cell's heartbeat is persisted.

    Each such write re-serializes the whole local JSON state, every
    ``cell_reconcile_interval_s``, for a timestamp only the dashboard shows. Any
    other change to the cell is still written at once. ``0`` persists every
    observation, the behaviour before this setting existed.
    """
    cell_agent_backoff_max_s: float = 300.0
    """Longest wait between asks of a cell agent that cannot be reached.

    Each consecutive connection failure doubles the wait from
    ``cell_reconcile_interval_s`` up to this ceiling; any answer resets it.
    ``0`` asks every interval, the behaviour before this setting existed.
    """
    lease_ttl_s: int = 120
    indeterminate_execution_grace_s: int = 900
    """How long an execution status may stay unreadable before its host is replaced.

    An INDETERMINATE observation is not a verdict about the workload, so the
    controller keeps polling -- but polling forever means a lease and an
    accelerator are held by something nobody can read. After this window the
    cell is drained: the host is the only thing that can be reclaimed with
    certainty, and replacing it destroys any process that did survive.
    """

    checkpoint_grace_s: int = 300
    preemption_stop_grace_s: int = 300
    #: Default graceful checkpoint window (10 minutes) granted to a borrowed
    #: workload before forced termination when capacity is reclaimed.
    default_reclaim_grace_s: int = 600
    outbox_claim_ttl_s: int = 90
    #: How many durable commands one replica may deliver at once. Delivery used
    #: to be strictly serial, so a single preemption's checkpoint-and-stop --
    #: minutes, by design, because it waits for a real training step to finish --
    #: blocked every other mutation on that replica: capacity applies, dispatches,
    #: cancellations. Ordering is preserved per aggregate, so raising this
    #: parallelises unrelated cells and workloads and never reorders one.
    outbox_dispatch_concurrency: int = 8
    # Retention windows. 0 disables pruning for that table, which is the
    # historical behaviour: nothing was ever deleted and idempotency keys were
    # therefore honored forever.
    outbox_retention_s: int = 7 * 24 * 3600
    audit_retention_s: int = 90 * 24 * 3600
    idempotency_retention_s: int = 30 * 24 * 3600
    #: Backend-for-frontend dashboard sessions: how long an ``eyvan_session``
    #: cookie stays valid, and how long a half-finished sign-in transaction is
    #: kept before the identity provider's redirect must be restarted.
    dashboard_session_ttl_s: int = 8 * 3600
    dashboard_auth_transaction_ttl_s: int = 600
    #: How old the externally measured P8 sample may be and still be published
    #: as ``p8_evidence``. Past it the soak route reports ``null`` again, which
    #: is what a dead monitor should read as: a sample from an hour ago is the
    #: last thing something happened to see, not a statement about now. Nothing
    #: changes for a fleet with no monitor -- with no sample stored the response
    #: is byte-identical to today's.
    soak_evidence_max_age_s: float = DEFAULT_EVIDENCE_MAX_AGE_S
    #: This control plane's identity among its peers. Empty means "not federated",
    #: which is the default and the only mode most fleets ever need: one plane per
    #: region with projects pinned to one region already works by construction.
    plane_id: str = ""
    #: Namespace the Kubernetes capacity backend creates cells in. One namespace
    #: per agent: the Job names are derived from cell ids, so two agents sharing a
    #: namespace would collide on names as surely as they would on fleet state.
    kubernetes_namespace: str = "eyvan"
    kubernetes_worker_image: str = ""
    retention_interval_s: float = 3600.0
    artifact_collection_interval_s: float = 3600.0
    artifact_destructive_collection_enabled: bool = False
    artifact_recovery_grace_s: float = 900.0
    """How often artifact lifecycle recovery and dry-run collection execute.

    Collection was implemented, tested, and called by nothing, so managed
    artifact storage grew without bound in production while the index, quota
    accounting, holds and reverse references all worked.
    """

    artifact_retention_s: int = 7 * 24 * 3600
    #: How often a running workload's new output is copied into an immutable
    #: chunk, and how much of it is taken per pass. This is the bound on what a
    #: preemption can destroy: everything older than one interval is already in
    #: storage when the machine goes away. Zero disables it, which is the former
    #: behaviour -- logs existed only on the cell until the release path.
    #:
    #: On by default, unlike the other housekeeping phases. Off by default is
    #: right for a phase that changes what a fleet *does*; this one only reads a
    #: journal that is already being read on demand, and the state it prevents
    #: (a preempted attempt with no logs at all) is the one operators hit most.
    log_archive_interval_s: float = 60.0
    log_archive_chunk_lines: int = 2000
    #: How many workloads are captured at once. Bounded because each capture
    #: fans out to every cell of its attempt, and a fleet with a thousand
    #: running workloads would otherwise open a thousand reads from the leader
    #: on every interval.
    log_archive_concurrency: int = 8
    #: How many of a workload's candidate zones may be asked for capacity at
    #: once. One is a serial search: ask the best zone, and if it will not grant
    #: try the next one a reconcile interval later. Higher races them to reach
    #: zones with available Spot capacity without waiting for each earlier
    #: request to fail.
    #:
    #: One by default, because every extra request is a machine somebody is
    #: billed for until the race resolves. Bounded by the workload's own
    #: declared alternates as well, so this can never widen a search beyond
    #: pools the submission path already proved interchangeable.
    capacity_race_width: int = 1
    #: How long an attempt's archived logs are kept. Longer than the general
    #: artifact window on purpose: a callable can be rebuilt from its source,
    #: while logs are the only record of what a run did, and the question they
    #: answer is usually asked well after the run. Zero keeps them forever.
    log_retention_s: int = 30 * 24 * 3600
    """How long an unreferenced artifact is kept before it may be collected.

    Distinct from ``grace_period_s``, which is an anti-race window around a
    single collection pass. 0 disables age-based collection entirely.
    """

    metrics_export_interval_s: float = 60.0
    #: How often services are brought back to their declared replica count.
    #: Faster than the housekeeping phases because this is the loop that
    #: replaces a dead replica -- its interval is the floor on how long a
    #: service runs short. Zero disables it.
    service_reconcile_interval_s: float = 5.0
    """How often fleet-state, capacity and spend gauges are refreshed. 0 disables.

    Gauges describe a level, not a rate, so they are refreshed on their own slow
    interval rather than on every scheduling pass.
    """

    #: How often service replicas are probed for health. **Zero, which disables
    #: it**, and that is deliberate: probing is what makes a replica replaceable
    #: for being unhealthy, so switching it on by default would change what an
    #: upgrade does to a running fleet. An operator turns it on when they have
    #: decided what their replicas' health endpoints mean.
    service_health_interval_s: float = 0.0
    #: How many replicas may be probed at once. Bounded because a fleet with a
    #: thousand replicas would otherwise open a thousand sockets from the leader
    #: on every pass.
    service_health_concurrency: int = 8
    #: How often a replica whose lease, cell generation or health says it no
    #: longer owns its host is replaced. Zero disables it, which is today's
    #: behaviour: nothing replaced such a replica at all.
    service_replica_lifecycle_interval_s: float = 0.0
    #: How often services are resized from their durable load observations. Zero
    #: disables it, and it is off by default *twice* -- here and per service --
    #: because an autoscaler nobody asked for spends money nobody budgeted.
    service_autoscale_interval_s: float = 0.0
    #: How many explained decisions are kept per service. The audit is bounded in
    #: the transaction that appends to it rather than by a retention sweep,
    #: because three sweeps in this package were implemented, tested, and never
    #: called.
    service_decision_history: int = 50

    grant_expiry_interval_s: float = 300.0
    """How often expired federated budget grants are swept. 0 disables it.

    A grant's expiry is mandatory, so without the sweep headroom is never
    returned to the granting plane and a tenant can outspend its federated
    budget indefinitely.
    """
    #: How often the leader advances cell billing clocks. Slower than reconcile
    #: because the accrual charges elapsed time rather than a fixed tick, so a
    #: longer interval costs resolution, never accuracy. Zero disables it.
    cell_billing_interval_s: float = 300.0
    #: How often managed asset staging and stage eviction run. Zero disables
    #: both, which is the default and today's behaviour: locality stays the soft
    #: preference it has always been until an operator sizes a cache.
    cell_stage_interval_s: float = 0.0
    #: AWS region the EC2 capacity backend provisions in. Required for that
    #: backend and unused by every other: a fleet with no AWS pools never reads
    #: it, and one with them cannot be built without it.
    aws_region: str = ""
    #: Static AWS credentials, for a control plane that is not running with an
    #: instance role. Empty is the ordinary case and the better one -- a plane on
    #: EC2 or EKS gets rotating credentials from its role, and a long-lived key
    #: in a settings file is a key in a backup.
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    #: Azure subscription, resource group and region the azure-vm backend
    #: provisions in. Required for that backend and unused by every other.
    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_location: str = ""
    #: Bearer token for ARM. Supplied rather than acquired: a managed identity, a
    #: client credential and a federated token are three different auth stories,
    #: and a transport that picked one would pick it for every operator.
    azure_access_token: str = ""
    #: How often the leader pushes committed audit events to project webhooks.
    #: Thirty seconds because a notification is not a control loop: the events
    #: are already durable, and a receiver's chat provider rate-limits long
    #: before this does. Zero disables delivery entirely, which is also what a
    #: fleet with no subscriptions costs.
    webhook_delivery_interval_s: float = 30.0
    #: Whether a webhook may point at loopback or a private range. Off, because
    #: a subscription is a request *this plane* makes: an internal address turns
    #: one into a way to reach what the submitter cannot. On for a development
    #: plane, and for a fleet whose receiver genuinely shares its network.
    webhook_allow_private_targets: bool = False
    #: Hard ceiling on the bytes Eyvan may own on one cell's disk. Zero is the
    #: off switch, and it is off by construction: nothing is staged, so nothing
    #: has to be reclaimed and no budget can be exceeded.
    cell_stage_budget_bytes: int = 0
    #: Largest single asset staging may fetch. Separate from the budget because
    #: it answers a different question -- the budget is how much of a host Eyvan
    #: may use, this is how large one asset may be before staging it is somebody's
    #: explicit decision rather than a cache's. Must not exceed the budget.
    cell_stage_max_bytes: int = 0
    #: Which staged asset is given up first. Named rather than defaulted at the
    #: policy layer; the setting carries a value only so the loader has one, and
    #: an unknown name is refused rather than silently replaced.
    cell_stage_eviction_policy: str = "least_recently_used"
    #: Whether the eviction phase may actually delete. Off by default, so the
    #: phase computes and logs exactly the plan it would execute and removes
    #: nothing -- a cache that starts deleting the moment it is switched on gives
    #: an operator no way to see what it was about to do first.
    cell_stage_eviction_enabled: bool = False
    #: Staging attempts that may end without an answer before a stage is blocked.
    #: Bounds a fetch that can never succeed without treating one timeout as a
    #: verdict about the asset.
    cell_stage_max_attempts: int = 3
    retention_batch_limit: int = 1000
    #: Hard ceiling on how many children one batch, DAG, or group submission may
    #: create. Enforced at admission, where the submitter is told and nothing has
    #: been created yet. The default is the limit this package has always
    #: enforced, so an operator who configures nothing sees no change.
    max_expansion_children: int = 1000
    #: How many children of one admitted expansion may compete for capacity at
    #: once. Zero -- the default, and today's behaviour -- means every child is
    #: schedulable immediately. A positive value creates all of them and releases
    #: them in waves, so one large array cannot bury every other tenant's work in
    #: the queue it shares.
    expansion_wave_size: int = 0
    #: Whether a heterogeneous group is leased as one transaction. Off by
    #: default, which is the shipped behaviour: the group's members are released
    #: to the ordinary scheduler once they all *could* be placed, and can still
    #: be placed partially if capacity moves in between. On, the members are
    #: leased together or not at all.
    atomic_group_placement: bool = False
    agent_shared_secret: str | None = field(default=None, repr=False)
    #: Bearer token a Prometheus scraper presents at ``/metrics``. A separate
    #: credential rather than an OIDC identity because a scraper has none, and
    #: because the alternative in practice is leaving the endpoint open: it
    #: exports ``eyvan_project_spend_usd`` labeled by (org, project), so anyone
    #: who can reach the port reads every tenant's spend. Required for a
    #: network-bound server, which is what makes the exposure impossible rather
    #: than merely discouraged.
    metrics_token: str | None = field(default=None, repr=False)
    native_worker_root_secret: str | None = field(default=None, repr=False)
    fleet_state_uri: str | None = None
    capacity_backend: str = "gcp"
    execution_backend: str = "eyvan-native"
    #: Operator consent to run diagnostic commands beside a running workload on
    #: this fleet's cells. Off by default, and deliberately an admission choice
    #: rather than a capability one: the native runtime has implemented
    #: ``exec_in_workload`` all along, and the GCP driver declared the capability
    #: absent with a comment saying an operator turns it on -- which nothing
    #: could do, so `eyvan workload exec` was refused on every GCP fleet.
    capacity_supports_exec: bool = False
    agent_state_dir: str = "~/.eyvan/agent"
    #: Operator consent to send managed-artifact capability tokens over plain
    #: HTTP. Off by default. It exists because the alternative was worse: the
    #: refusal was unconditional, so a control plane that terminates no TLS --
    #: which is every head `eyvan head up` builds -- could never run a callable
    #: at all, and said so only by holding the accelerator forever.
    allow_insecure_artifact_http: bool = False
    native_worker_port: int = 8100
    native_worker_package: str | None = None
    native_worker_package_sha256: str | None = None
    native_worker_package_generation: str | None = None
    native_worker_artifact_root_uri: str | None = None
    native_worker_public_ingress_tag: str | None = None
    native_worker_wheelhouse: str | None = None
    native_worker_wheelhouse_sha256: str | None = None
    native_worker_wheelhouse_generation: str | None = None
    native_worker_state_dir: str = "/var/lib/eyvan/native-worker"
    platform_config_path: str = "/etc/eyvan/platform.yaml"
    #: Ask the provider metadata path who this host is instead of trusting a
    #: long-lived key file. Off by default: a probe that runs everywhere would
    #: add a network call to every process start, and on a host with no metadata
    #: service the honest answer is the one nobody asked for.
    provider_metadata_identity_enabled: bool = False
    provider_metadata_identity_timeout_s: float = 2.0
    #: Mutual TLS between control plane, cell agent, and native worker. ``off``
    #: everywhere is today's deployment; see :mod:`eyvan.identity.mtls` for the
    #: enablement order, which must be followed outward-in or the control plane
    #: loses its own agents mid-rollout.
    mutual_tls_mode: str = "off"
    mutual_tls_client_enabled: bool = False
    mutual_tls_trust_bundle: str = ""
    mutual_tls_certificate: str = ""
    mutual_tls_private_key: str = ""
    #: Seconds a superseded signing generation keeps verifying after a rotation
    #: commits. Long enough for a capability minted just before the rotation to
    #: still be redeemed by a cell that is mid-bootstrap, and no longer.
    credential_rotation_overlap_s: float = 300.0
    #: How long a multi-host rank waits at the JAX registration barrier for its
    #: peers. JAX's own default is 300s, which a cold gang restart after a
    #: preemption routinely exceeds while a subset of hosts is still syncing an
    #: environment or reading from GCS -- and the hosts that *did* arrive abort
    #: the whole gang. Iris raised this to 1800 after losing v5p-64 gangs to it.
    #:
    #: The default here is the value this package has always applied on the host
    #: (``_multihost_site.DEFAULT_INITIALIZATION_TIMEOUT_S``), so configuring
    #: nothing changes nothing; what is new is that an operator can raise it
    #: without rebuilding a wheel, and that a submitter cannot lower it.
    gang_init_deadline_s: int = 600
    #: How long a joined rank may go without heartbeating before the
    #: coordination service declares it dead. Also today's shipped value: an
    #: eSurge job on a two-host v5p-16 was killed by JAX's 100s default while
    #: one host was still loading a 0.6B model.
    gang_heartbeat_timeout_s: int = 900
    #: Whether a rank that never joined the gang ends its own workload.
    #:
    #: The initialization hook runs from ``sitecustomize``, and ``site.py``
    #: catches every ordinary exception raised there, prints one line and lets
    #: the interpreter carry on -- so with fatality off, a gang that never forms
    #: runs on as an unwitting single-process job and can exit 0. On (the
    #: default), that rank exits with an Eyvan-reserved status, and the
    #: executors report the distinct terminal reason "the gang never formed"
    #: instead of an ordinary workload failure.
    gang_init_fatal: bool = True
    #: Host replacements one scope may make inside ``host_replacement_window_s``.
    #: Zero -- the default -- means a machine that names an unrecoverable fault in
    #: its own output is still only retried, which is today's behaviour. The bound
    #: exists because each replacement costs a provision *and* a destroy, so a
    #: zone of bad hosts must not become an infinite provision loop.
    host_replacement_budget: int = 0
    host_replacement_window_s: int = 3600
    #: Sources a cell may stage a named asset from, comma separated. Empty -- the
    #: default -- reads as "no executor", not as a refusal: a refusal is terminal
    #: and would block every asset in a fleet over one unset variable. Named
    #: assets are user input, so an executor that fetched anything would be an
    #: arbitrary-read primitive; only these prefixes are reachable.
    cell_stage_sources: str = ""
    #: Where a cell keeps staged bytes. Empty uses the agent's own state directory.
    cell_stage_dir: str = ""

    def mutual_tls_files(self) -> MutualTlsFiles:
        """Return this process's mutual-TLS paths, never its key material."""

        return MutualTlsFiles(
            trust_bundle=self.mutual_tls_trust_bundle,
            certificate=self.mutual_tls_certificate,
            private_key=self.mutual_tls_private_key,
        )

    def mutual_tls_server_mode(self) -> MutualTlsMode:
        """Return what this process requires of peers connecting to it."""

        return parse_mutual_tls_mode(self.mutual_tls_mode)

    def __post_init__(self) -> None:
        """Validate settings that participate in durable controller ownership."""

        if self.tracker_url is not None:
            parsed_tracker = urlsplit(self.tracker_url)
            if (
                parsed_tracker.scheme not in {"http", "https"}
                or not parsed_tracker.hostname
                or parsed_tracker.username is not None
                or parsed_tracker.password is not None
                or parsed_tracker.query
                or parsed_tracker.fragment
            ):
                raise ValueError("tracker_url must be an http(s) URL without credentials, query or fragment")
            try:
                _ = parsed_tracker.port
            except ValueError as exc:
                raise ValueError("tracker_url must have a valid port") from exc
        if self.outbox_dispatch_concurrency <= 0:
            raise ValueError("outbox_dispatch_concurrency must be positive")
        if self.outbox_claim_ttl_s <= 0:
            raise ValueError("outbox_claim_ttl_s must be positive")
        if self.dashboard_session_ttl_s <= 0:
            raise ValueError("dashboard_session_ttl_s must be positive")
        if self.dashboard_auth_transaction_ttl_s <= 0:
            raise ValueError("dashboard_auth_transaction_ttl_s must be positive")
        for name in ("outbox_retention_s", "audit_retention_s", "idempotency_retention_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative; use 0 to disable pruning")
        if self.retention_batch_limit <= 0:
            raise ValueError("retention_batch_limit must be positive")
        if self.service_health_concurrency <= 0:
            raise ValueError("service_health_concurrency must be positive")
        if self.service_decision_history <= 0:
            # Zero would append a decision and delete it in the same transaction,
            # so every scaling action would be unexplainable the moment it
            # happened -- worse than no audit, because the table would look like
            # one.
            raise ValueError("service_decision_history must be positive")
        for name in (
            "service_health_interval_s",
            "service_replica_lifecycle_interval_s",
            "service_autoscale_interval_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative; use 0 to disable the phase")
        if not math.isfinite(self.credential_rotation_overlap_s) or self.credential_rotation_overlap_s < 0:
            raise ValueError("credential_rotation_overlap_s must be finite and non-negative")
        timeout_s = self.provider_metadata_identity_timeout_s
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("provider_metadata_identity_timeout_s must be finite and positive")
        # Refuse an unrecognized spelling at construction rather than at the
        # first connection: a typo in ``required`` that silently reads as off is
        # exactly the failure mutual TLS is being added to prevent.
        parse_mutual_tls_mode(self.mutual_tls_mode)
        if self.max_expansion_children <= 0:
            raise ValueError("max_expansion_children must be positive")
        if self.expansion_wave_size < 0:
            raise ValueError("expansion_wave_size must not be negative; use 0 to disable waves")
        if not math.isfinite(self.artifact_collection_interval_s) or self.artifact_collection_interval_s <= 0:
            raise ValueError("artifact_collection_interval_s must be finite and positive")
        if not math.isfinite(self.artifact_recovery_grace_s) or self.artifact_recovery_grace_s < 0:
            raise ValueError("artifact_recovery_grace_s must be finite and non-negative")
        for name in ("artifact_retention_s", "log_retention_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative; use 0 to disable collection")
        if not math.isfinite(self.log_archive_interval_s) or self.log_archive_interval_s < 0:
            raise ValueError("log_archive_interval_s must be finite and non-negative; use 0 to disable it")
        if self.log_archive_chunk_lines <= 0:
            raise ValueError("log_archive_chunk_lines must be positive")
        if self.log_archive_concurrency <= 0:
            raise ValueError("log_archive_concurrency must be positive")
        if self.capacity_race_width < 1:
            raise ValueError("capacity_race_width must be at least 1; 1 asks one zone at a time")
        # The remaining interval knobs: a NaN or negative value here would
        # become the controller's next sleep, or silently disable a phase.
        for name in (
            "scheduler_interval_s",
            "scheduler_idle_interval_s",
            "cell_reconcile_interval_s",
            "cell_heartbeat_persist_interval_s",
            "cell_agent_backoff_max_s",
            "service_reconcile_interval_s",
            "retention_interval_s",
            "metrics_export_interval_s",
            "grant_expiry_interval_s",
            "cell_billing_interval_s",
            "webhook_delivery_interval_s",
            "soak_evidence_max_age_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be positive")
        for name in (
            "checkpoint_grace_s",
            "preemption_stop_grace_s",
            "default_reclaim_grace_s",
            "indeterminate_execution_grace_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.cell_reconcile_concurrency <= 0:
            raise ValueError("cell_reconcile_concurrency must be positive")
        if self.gang_init_deadline_s <= 0:
            raise ValueError("gang_init_deadline_s must be positive")
        if self.host_replacement_window_s <= 0:
            raise ValueError("host_replacement_window_s must be positive")
        if not 1 <= self.native_worker_port <= 65535:
            raise ValueError("native_worker_port must be between 1 and 65535")
        self._validate_cell_stage_settings()
        validate_minimum_client_version(self.min_client_version)

    def _validate_cell_stage_settings(self) -> None:
        """Refuse stage bounds that would let a cache bound nothing.

        Every check here is the difference between a budget and a number. A
        negative ceiling, a per-action limit above the whole budget, or an
        eviction policy nobody recognises all produce a configuration that looks
        like it constrains a disk and does not.
        """

        if not math.isfinite(self.cell_stage_interval_s) or self.cell_stage_interval_s < 0:
            raise ValueError("cell_stage_interval_s must be finite and non-negative; use 0 to disable staging")
        for name in ("cell_stage_budget_bytes", "cell_stage_max_bytes"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must not be negative; use 0 to disable staging")
        if self.cell_stage_max_bytes > self.cell_stage_budget_bytes:
            raise ValueError("cell_stage_max_bytes must not exceed cell_stage_budget_bytes")
        if self.cell_stage_max_attempts < 1:
            raise ValueError("cell_stage_max_attempts must be positive")
        if self.cell_stage_eviction_policy not in {policy.value for policy in StageEvictionPolicy}:
            choices = ", ".join(sorted(policy.value for policy in StageEvictionPolicy))
            raise ValueError(f"cell_stage_eviction_policy must be one of: {choices}")

    def cell_stage_policy(self) -> CellStagePolicy:
        """Return the validated staging bounds this deployment declared.

        A method rather than a field so the policy is constructed from settings
        that have already been validated together, and so a deployment that never
        enables staging carries no policy object it does not use.
        """

        return CellStagePolicy(
            budget_bytes=self.cell_stage_budget_bytes,
            max_stage_bytes=self.cell_stage_max_bytes,
            eviction=StageEvictionPolicy(self.cell_stage_eviction_policy),
            max_attempts=self.cell_stage_max_attempts,
        )

    def gang_initialization_policy(self) -> GangInitializationPolicy:
        """Return the coordination deadlines Eyvan injects into every gang rank.

        A method rather than a field so the policy is built from settings that
        have already been read together, and so its own validation -- a
        non-positive deadline means "abort every gang immediately" -- runs where
        the executor asks for it rather than in a constructor that a host-side
        process would also pay for.

        Returns:
            The validated policy the slice and gang executors inject.

        Raises:
            ValueError: If either deadline is not a positive number of seconds.
        """

        return GangInitializationPolicy(
            initialization_timeout_s=self.gang_init_deadline_s,
            heartbeat_timeout_s=self.gang_heartbeat_timeout_s,
            fatal=self.gang_init_fatal,
        )

    @classmethod
    def from_gcp_native_runtime_profile(cls, profile: GcpNativeRuntimeProfile) -> EyvanSettings:
        """Build local API and agent settings from one owner-only GCP runtime profile."""

        if not isinstance(profile, GcpNativeRuntimeProfile):
            raise TypeError("GCP native runtime settings require a GcpNativeRuntimeProfile")
        # Every other setting is the operator's, read from the environment
        # exactly as ``from_env`` reads it; the profile overrides only what it
        # owns. Building from bare defaults would silently ignore operator
        # settings such as the callable allow-list, race width, and exec policy.
        return replace(
            cls.from_env(),
            agent_shared_secret=profile.agent_shared_secret,
            native_worker_root_secret=profile.native_worker_root_secret,
            # The profile's agent runs on this machine beside its control plane,
            # so fleet state defaults to a local file: the document is a cache
            # of provider-owned facts plus Eyvan's fences, and a bucket write on
            # that cache's critical path could block use of already-active
            # capacity. An explicit EYVAN_FLEET_STATE_URI --
            # which is how a remote head shares one document across agents --
            # still wins.
            fleet_state_uri=os.getenv("EYVAN_FLEET_STATE_URI") or str(DEFAULT_GCP_NATIVE_FLEET_STATE_PATH),
            capacity_backend="gcp",
            execution_backend="eyvan-native",
            artifact_store_uri=profile.artifact_root_uri,
            native_worker_package=profile.native_worker_wheel_uri,
            # The pin and generation describe an environment-provided package; the
            # profile's own wheel is content-addressed by its URI, and pairing it
            # with another package's digest would fail every bootstrap.
            native_worker_package_sha256=None,
            native_worker_package_generation=None,
            native_worker_artifact_root_uri=profile.artifact_root_uri,
            native_worker_public_ingress_tag=profile.native_worker_public_ingress_tag,
        )

    @property
    def tls_enabled(self) -> bool:
        """Return whether both TLS material pieces are configured."""

        return self.tls_cert_file is not None and self.tls_key_file is not None

    @classmethod
    def from_env(cls) -> EyvanSettings:
        """Load settings from ``EYVAN_*`` environment variables."""

        defaults = cls()

        def truthy(name: str, default: bool = False) -> bool:
            raw = os.getenv(name)
            return default if raw is None else raw.lower() in {"1", "true", "yes", "on"}

        return cls(
            database_url=os.getenv("EYVAN_DATABASE_URL", defaults.database_url),
            repository_backend=_repository_backend(
                os.getenv("EYVAN_REPOSITORY_BACKEND", defaults.repository_backend.value)
            ),
            auto_create_schema=truthy("EYVAN_AUTO_CREATE_SCHEMA"),
            controller_enabled=truthy("EYVAN_CONTROLLER_ENABLED", defaults.controller_enabled),
            public_url=os.getenv("EYVAN_PUBLIC_URL", defaults.public_url),
            tracker_url=os.getenv("EYVAN_TRACKER_URL") or None,
            tracker_token=os.getenv("EYVAN_TRACKER_TOKEN") or None,
            tracking_store_uri=os.getenv("EYVAN_TRACKING_STORE_URI") or None,
            tls_cert_file=_optional_path("EYVAN_TLS_CERT_FILE"),
            tls_key_file=_optional_path("EYVAN_TLS_KEY_FILE"),
            allow_insecure_plaintext_bind=truthy(
                "EYVAN_INSECURE_PLAINTEXT_BIND", defaults.allow_insecure_plaintext_bind
            ),
            oidc_issuer=os.getenv("EYVAN_OIDC_ISSUER"),
            bootstrap_token=os.getenv("EYVAN_BOOTSTRAP_TOKEN"),
            oidc_dashboard_client_id=os.getenv("EYVAN_OIDC_DASHBOARD_CLIENT_ID"),
            oidc_audience=os.getenv("EYVAN_OIDC_AUDIENCE", defaults.oidc_audience),
            oidc_jwks_url=os.getenv("EYVAN_OIDC_JWKS_URL"),
            dev_auth_enabled=truthy("EYVAN_DEV_AUTH_ENABLED"),
            auth_audit_enabled=truthy("EYVAN_AUTH_AUDIT_ENABLED", defaults.auth_audit_enabled),
            correlation_enabled=truthy("EYVAN_CORRELATION_ENABLED", defaults.correlation_enabled),
            min_client_version=os.getenv("EYVAN_MIN_CLIENT_VERSION") or None,
            trusted_callable_projects=_csv(os.getenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", "")),
            artifact_store_uri=os.getenv("EYVAN_ARTIFACT_STORE_URI", defaults.artifact_store_uri),
            artifact_token_key=os.getenv("EYVAN_ARTIFACT_TOKEN_KEY"),
            artifact_token_ttl_s=int(os.getenv("EYVAN_ARTIFACT_TOKEN_TTL_S", defaults.artifact_token_ttl_s)),
            artifact_max_callable_bytes=int(
                os.getenv("EYVAN_ARTIFACT_MAX_CALLABLE_BYTES", defaults.artifact_max_callable_bytes)
            ),
            artifact_max_result_bytes=int(
                os.getenv("EYVAN_ARTIFACT_MAX_RESULT_BYTES", defaults.artifact_max_result_bytes)
            ),
            secret_token_ttl_s=int(os.getenv("EYVAN_SECRET_TOKEN_TTL_S", defaults.secret_token_ttl_s)),
            readonly_query_timeout_ms=int(
                os.getenv("EYVAN_READONLY_QUERY_TIMEOUT_MS", defaults.readonly_query_timeout_ms)
            ),
            artifact_max_workspace_bytes=int(
                os.getenv("EYVAN_ARTIFACT_MAX_WORKSPACE_BYTES", defaults.artifact_max_workspace_bytes)
            ),
            scheduler_interval_s=float(os.getenv("EYVAN_SCHEDULER_INTERVAL_S", defaults.scheduler_interval_s)),
            scheduler_idle_interval_s=float(
                os.getenv("EYVAN_SCHEDULER_IDLE_INTERVAL_S", defaults.scheduler_idle_interval_s)
            ),
            cell_reconcile_interval_s=float(
                os.getenv("EYVAN_CELL_RECONCILE_INTERVAL_S", defaults.cell_reconcile_interval_s)
            ),
            cell_reconcile_concurrency=int(
                os.getenv("EYVAN_CELL_RECONCILE_CONCURRENCY", defaults.cell_reconcile_concurrency)
            ),
            cell_heartbeat_persist_interval_s=float(
                os.getenv("EYVAN_CELL_HEARTBEAT_PERSIST_INTERVAL_S", defaults.cell_heartbeat_persist_interval_s)
            ),
            cell_agent_backoff_max_s=float(
                os.getenv("EYVAN_CELL_AGENT_BACKOFF_MAX_S", defaults.cell_agent_backoff_max_s)
            ),
            lease_ttl_s=int(os.getenv("EYVAN_LEASE_TTL_S", defaults.lease_ttl_s)),
            checkpoint_grace_s=int(os.getenv("EYVAN_CHECKPOINT_GRACE_S", defaults.checkpoint_grace_s)),
            preemption_stop_grace_s=int(os.getenv("EYVAN_PREEMPTION_STOP_GRACE_S", defaults.preemption_stop_grace_s)),
            default_reclaim_grace_s=int(os.getenv("EYVAN_DEFAULT_RECLAIM_GRACE_S", defaults.default_reclaim_grace_s)),
            outbox_claim_ttl_s=int(os.getenv("EYVAN_OUTBOX_CLAIM_TTL_S", defaults.outbox_claim_ttl_s)),
            outbox_dispatch_concurrency=int(
                os.getenv("EYVAN_OUTBOX_DISPATCH_CONCURRENCY", defaults.outbox_dispatch_concurrency)
            ),
            outbox_retention_s=int(os.getenv("EYVAN_OUTBOX_RETENTION_S", defaults.outbox_retention_s)),
            audit_retention_s=int(os.getenv("EYVAN_AUDIT_RETENTION_S", defaults.audit_retention_s)),
            idempotency_retention_s=int(os.getenv("EYVAN_IDEMPOTENCY_RETENTION_S", defaults.idempotency_retention_s)),
            dashboard_session_ttl_s=int(os.getenv("EYVAN_DASHBOARD_SESSION_TTL_S", defaults.dashboard_session_ttl_s)),
            dashboard_auth_transaction_ttl_s=int(
                os.getenv("EYVAN_DASHBOARD_AUTH_TRANSACTION_TTL_S", defaults.dashboard_auth_transaction_ttl_s)
            ),
            soak_evidence_max_age_s=float(os.getenv("EYVAN_SOAK_EVIDENCE_MAX_AGE_S", defaults.soak_evidence_max_age_s)),
            plane_id=os.getenv("EYVAN_PLANE_ID", defaults.plane_id),
            kubernetes_namespace=os.getenv("EYVAN_KUBERNETES_NAMESPACE", defaults.kubernetes_namespace),
            kubernetes_worker_image=os.getenv("EYVAN_KUBERNETES_WORKER_IMAGE", defaults.kubernetes_worker_image),
            retention_interval_s=float(os.getenv("EYVAN_RETENTION_INTERVAL_S", defaults.retention_interval_s)),
            artifact_collection_interval_s=float(
                os.getenv("EYVAN_ARTIFACT_COLLECTION_INTERVAL_S", defaults.artifact_collection_interval_s)
            ),
            artifact_destructive_collection_enabled=truthy(
                "EYVAN_ARTIFACT_DESTRUCTIVE_COLLECTION_ENABLED",
                defaults.artifact_destructive_collection_enabled,
            ),
            artifact_recovery_grace_s=float(
                os.getenv("EYVAN_ARTIFACT_RECOVERY_GRACE_S", defaults.artifact_recovery_grace_s)
            ),
            artifact_retention_s=int(os.getenv("EYVAN_ARTIFACT_RETENTION_S", defaults.artifact_retention_s)),
            log_retention_s=int(os.getenv("EYVAN_LOG_RETENTION_S", defaults.log_retention_s)),
            log_archive_interval_s=float(os.getenv("EYVAN_LOG_ARCHIVE_INTERVAL_S", defaults.log_archive_interval_s)),
            log_archive_chunk_lines=int(os.getenv("EYVAN_LOG_ARCHIVE_CHUNK_LINES", defaults.log_archive_chunk_lines)),
            log_archive_concurrency=int(os.getenv("EYVAN_LOG_ARCHIVE_CONCURRENCY", defaults.log_archive_concurrency)),
            capacity_race_width=int(os.getenv("EYVAN_CAPACITY_RACE_WIDTH", defaults.capacity_race_width)),
            service_reconcile_interval_s=float(
                os.getenv("EYVAN_SERVICE_RECONCILE_INTERVAL_S", defaults.service_reconcile_interval_s)
            ),
            service_health_interval_s=float(
                os.getenv("EYVAN_SERVICE_HEALTH_INTERVAL_S", defaults.service_health_interval_s)
            ),
            service_health_concurrency=int(
                os.getenv("EYVAN_SERVICE_HEALTH_CONCURRENCY", defaults.service_health_concurrency)
            ),
            service_replica_lifecycle_interval_s=float(
                os.getenv(
                    "EYVAN_SERVICE_REPLICA_LIFECYCLE_INTERVAL_S",
                    defaults.service_replica_lifecycle_interval_s,
                )
            ),
            service_autoscale_interval_s=float(
                os.getenv("EYVAN_SERVICE_AUTOSCALE_INTERVAL_S", defaults.service_autoscale_interval_s)
            ),
            service_decision_history=int(os.getenv("EYVAN_SERVICE_DECISION_HISTORY", defaults.service_decision_history)),
            metrics_export_interval_s=float(
                os.getenv("EYVAN_METRICS_EXPORT_INTERVAL_S", defaults.metrics_export_interval_s)
            ),
            grant_expiry_interval_s=float(os.getenv("EYVAN_GRANT_EXPIRY_INTERVAL_S", defaults.grant_expiry_interval_s)),
            cell_billing_interval_s=float(os.getenv("EYVAN_CELL_BILLING_INTERVAL_S", defaults.cell_billing_interval_s)),
            cell_stage_interval_s=float(os.getenv("EYVAN_CELL_STAGE_INTERVAL_S", defaults.cell_stage_interval_s)),
            azure_subscription_id=os.getenv("EYVAN_AZURE_SUBSCRIPTION_ID", defaults.azure_subscription_id),
            azure_resource_group=os.getenv("EYVAN_AZURE_RESOURCE_GROUP", defaults.azure_resource_group),
            azure_location=os.getenv("EYVAN_AZURE_LOCATION", defaults.azure_location),
            azure_access_token=os.getenv("EYVAN_AZURE_ACCESS_TOKEN", defaults.azure_access_token),
            aws_region=os.getenv("EYVAN_AWS_REGION", defaults.aws_region),
            aws_access_key_id=os.getenv("EYVAN_AWS_ACCESS_KEY_ID", defaults.aws_access_key_id),
            aws_secret_access_key=os.getenv("EYVAN_AWS_SECRET_ACCESS_KEY", defaults.aws_secret_access_key),
            aws_session_token=os.getenv("EYVAN_AWS_SESSION_TOKEN", defaults.aws_session_token),
            webhook_delivery_interval_s=float(
                os.getenv("EYVAN_WEBHOOK_DELIVERY_INTERVAL_S", defaults.webhook_delivery_interval_s)
            ),
            webhook_allow_private_targets=truthy(
                "EYVAN_WEBHOOK_ALLOW_PRIVATE_TARGETS",
                defaults.webhook_allow_private_targets,
            ),
            cell_stage_budget_bytes=int(os.getenv("EYVAN_CELL_STAGE_BUDGET_BYTES", defaults.cell_stage_budget_bytes)),
            cell_stage_max_bytes=int(os.getenv("EYVAN_CELL_STAGE_MAX_BYTES", defaults.cell_stage_max_bytes)),
            cell_stage_eviction_policy=os.getenv(
                "EYVAN_CELL_STAGE_EVICTION_POLICY", defaults.cell_stage_eviction_policy
            ),
            cell_stage_eviction_enabled=truthy(
                "EYVAN_CELL_STAGE_EVICTION_ENABLED",
                defaults.cell_stage_eviction_enabled,
            ),
            capacity_supports_exec=truthy(
                "EYVAN_CAPACITY_SUPPORTS_EXEC",
                defaults.capacity_supports_exec,
            ),
            cell_stage_max_attempts=int(os.getenv("EYVAN_CELL_STAGE_MAX_ATTEMPTS", defaults.cell_stage_max_attempts)),
            retention_batch_limit=int(os.getenv("EYVAN_RETENTION_BATCH_LIMIT", defaults.retention_batch_limit)),
            max_expansion_children=int(os.getenv("EYVAN_MAX_EXPANSION_CHILDREN", defaults.max_expansion_children)),
            expansion_wave_size=int(os.getenv("EYVAN_EXPANSION_WAVE_SIZE", defaults.expansion_wave_size)),
            atomic_group_placement=truthy("EYVAN_ATOMIC_GROUP_PLACEMENT", defaults.atomic_group_placement),
            agent_shared_secret=os.getenv("EYVAN_AGENT_SHARED_SECRET"),
            metrics_token=os.getenv("EYVAN_METRICS_TOKEN"),
            native_worker_root_secret=os.getenv("EYVAN_NATIVE_WORKER_ROOT_SECRET"),
            fleet_state_uri=os.getenv("EYVAN_FLEET_STATE_URI"),
            capacity_backend=os.getenv("EYVAN_CAPACITY_BACKEND", defaults.capacity_backend),
            execution_backend=os.getenv("EYVAN_EXECUTION_BACKEND", defaults.execution_backend),
            agent_state_dir=os.getenv("EYVAN_AGENT_STATE_DIR", defaults.agent_state_dir),
            allow_insecure_artifact_http=truthy("EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP"),
            native_worker_port=int(os.getenv("EYVAN_NATIVE_WORKER_PORT", defaults.native_worker_port)),
            native_worker_package=os.getenv("EYVAN_NATIVE_WORKER_PACKAGE"),
            native_worker_package_sha256=os.getenv("EYVAN_NATIVE_WORKER_PACKAGE_SHA256") or None,
            native_worker_package_generation=os.getenv("EYVAN_NATIVE_WORKER_PACKAGE_GENERATION") or None,
            native_worker_wheelhouse=os.getenv("EYVAN_NATIVE_WORKER_WHEELHOUSE"),
            native_worker_wheelhouse_sha256=os.getenv("EYVAN_NATIVE_WORKER_WHEELHOUSE_SHA256") or None,
            native_worker_wheelhouse_generation=os.getenv("EYVAN_NATIVE_WORKER_WHEELHOUSE_GENERATION") or None,
            native_worker_state_dir=os.getenv("EYVAN_NATIVE_WORKER_STATE_DIR", defaults.native_worker_state_dir),
            # Both are read by _gcp_factory (drivers/registry.py) and were set only by
            # from_gcp_native_runtime_profile, so a locally-profiled agent had them and an
            # env-configured production agent silently got None: no artifact root for its
            # native workers, and no ingress network tag.
            native_worker_artifact_root_uri=os.getenv("EYVAN_NATIVE_WORKER_ARTIFACT_ROOT_URI"),
            native_worker_public_ingress_tag=os.getenv("EYVAN_NATIVE_WORKER_PUBLIC_INGRESS_TAG"),
            platform_config_path=os.getenv("EYVAN_PLATFORM_CONFIG_PATH", defaults.platform_config_path),
            provider_metadata_identity_enabled=truthy("EYVAN_PROVIDER_METADATA_IDENTITY_ENABLED"),
            provider_metadata_identity_timeout_s=float(
                os.getenv("EYVAN_PROVIDER_METADATA_IDENTITY_TIMEOUT_S", defaults.provider_metadata_identity_timeout_s)
            ),
            mutual_tls_mode=os.getenv("EYVAN_MUTUAL_TLS_MODE", defaults.mutual_tls_mode),
            mutual_tls_client_enabled=truthy("EYVAN_MUTUAL_TLS_CLIENT_ENABLED"),
            mutual_tls_trust_bundle=os.getenv("EYVAN_MUTUAL_TLS_TRUST_BUNDLE", defaults.mutual_tls_trust_bundle),
            mutual_tls_certificate=os.getenv("EYVAN_MUTUAL_TLS_CERTIFICATE", defaults.mutual_tls_certificate),
            mutual_tls_private_key=os.getenv("EYVAN_MUTUAL_TLS_PRIVATE_KEY", defaults.mutual_tls_private_key),
            credential_rotation_overlap_s=float(
                os.getenv("EYVAN_CREDENTIAL_ROTATION_OVERLAP_S", defaults.credential_rotation_overlap_s)
            ),
            gang_init_deadline_s=int(os.getenv("EYVAN_GANG_INIT_DEADLINE_S", defaults.gang_init_deadline_s)),
            gang_heartbeat_timeout_s=int(os.getenv("EYVAN_GANG_HEARTBEAT_TIMEOUT_S", defaults.gang_heartbeat_timeout_s)),
            gang_init_fatal=truthy("EYVAN_GANG_INIT_FATAL", defaults.gang_init_fatal),
            host_replacement_budget=int(os.getenv("EYVAN_HOST_REPLACEMENT_BUDGET", defaults.host_replacement_budget)),
            host_replacement_window_s=int(
                os.getenv("EYVAN_HOST_REPLACEMENT_WINDOW_S", defaults.host_replacement_window_s)
            ),
            cell_stage_sources=os.getenv("EYVAN_CELL_STAGE_SOURCES", defaults.cell_stage_sources),
            cell_stage_dir=os.getenv("EYVAN_CELL_STAGE_DIR", defaults.cell_stage_dir),
        )
