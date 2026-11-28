# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public control-plane service facade."""

from __future__ import annotations

from eyvan.server.service._control_plane.accounts import AccountsMixin
from eyvan.server.service._control_plane.artifact_admission import ArtifactAdmissionMixin
from eyvan.server.service._control_plane.artifact_collection import ArtifactCollectionMixin
from eyvan.server.service._control_plane.artifact_inventory import ArtifactInventoryMixin
from eyvan.server.service._control_plane.artifact_io import ArtifactIoMixin
from eyvan.server.service._control_plane.artifact_recovery import ArtifactRecoveryMixin
from eyvan.server.service._control_plane.assistant_conversations import AssistantConversationsMixin
from eyvan.server.service._control_plane.assistant_providers import AssistantProvidersMixin
from eyvan.server.service._control_plane.audit import AuditMixin
from eyvan.server.service._control_plane.callable_submission import CallableSubmissionMixin
from eyvan.server.service._control_plane.cells import CellsMixin
from eyvan.server.service._control_plane.child_submission import ChildSubmissionMixin
from eyvan.server.service._control_plane.estimates import EstimatesMixin
from eyvan.server.service._control_plane.events import EventsMixin
from eyvan.server.service._control_plane.expansion import ExpansionMixin
from eyvan.server.service._control_plane.federation import FederationMixin
from eyvan.server.service._control_plane.fleets import FleetsMixin
from eyvan.server.service._control_plane.governance import GovernanceMixin
from eyvan.server.service._control_plane.lease_settlement import LeaseSettlementMixin
from eyvan.server.service._control_plane.leases import LeasesMixin
from eyvan.server.service._control_plane.policies import PoliciesMixin
from eyvan.server.service._control_plane.reporting import ReportingMixin
from eyvan.server.service._control_plane.research import ResearchPlansMixin
from eyvan.server.service._control_plane.scheduling import SchedulingMixin
from eyvan.server.service._control_plane.services import ServicesMixin
from eyvan.server.service._control_plane.shared import (
    _ACTIVE_CAPACITY_APPLY_STATES as _ACTIVE_CAPACITY_APPLY_STATES,
)
from eyvan.server.service._control_plane.shared import (
    _CELL_LIFECYCLE_TOPIC_BY_STATE as _CELL_LIFECYCLE_TOPIC_BY_STATE,
)
from eyvan.server.service._control_plane.shared import (
    _EXPLAIN_MAX_CELLS as _EXPLAIN_MAX_CELLS,
)
from eyvan.server.service._control_plane.shared import (
    _LIVE_OUTBOX_STATES as _LIVE_OUTBOX_STATES,
)
from eyvan.server.service._control_plane.shared import (
    _MANAGED_RESULT_PREFIX as _MANAGED_RESULT_PREFIX,
)
from eyvan.server.service._control_plane.shared import (
    _POOL_LABEL as _POOL_LABEL,
)
from eyvan.server.service._control_plane.shared import (
    _QUERY_AUDIT_STATEMENT_LIMIT as _QUERY_AUDIT_STATEMENT_LIMIT,
)
from eyvan.server.service._control_plane.shared import (
    _QUEUE_TREND_MAX_EVENTS as _QUEUE_TREND_MAX_EVENTS,
)
from eyvan.server.service._control_plane.shared import (
    ACTIVE_LEASE_STATES as ACTIVE_LEASE_STATES,
)
from eyvan.server.service._control_plane.shared import (
    BILLABLE_CELL_STATES as BILLABLE_CELL_STATES,
)
from eyvan.server.service._control_plane.shared import (
    CALLABLE_SIGNING_KEY_ENV as CALLABLE_SIGNING_KEY_ENV,
)
from eyvan.server.service._control_plane.shared import (
    CHILD_REVISION as CHILD_REVISION,
)
from eyvan.server.service._control_plane.shared import (
    CONTENT_ADDRESSED_KINDS as CONTENT_ADDRESSED_KINDS,
)
from eyvan.server.service._control_plane.shared import (
    DEFAULT_ARTIFACT_GRACE_PERIOD_S as DEFAULT_ARTIFACT_GRACE_PERIOD_S,
)
from eyvan.server.service._control_plane.shared import (
    DEFAULT_ARTIFACT_RECOVERY_GRACE_S as DEFAULT_ARTIFACT_RECOVERY_GRACE_S,
)
from eyvan.server.service._control_plane.shared import (
    DEFAULT_MAX_FAN_OUT as DEFAULT_MAX_FAN_OUT,
)
from eyvan.server.service._control_plane.shared import (
    DEFAULT_MAX_RETRIES as DEFAULT_MAX_RETRIES,
)
from eyvan.server.service._control_plane.shared import (
    MAX_GCP_TPU_MULTISLICE_WORKERS as MAX_GCP_TPU_MULTISLICE_WORKERS,
)
from eyvan.server.service._control_plane.shared import (
    MAX_GROUP_MEMBERS as MAX_GROUP_MEMBERS,
)
from eyvan.server.service._control_plane.shared import (
    MAX_JOB_TREE_DEPTH as MAX_JOB_TREE_DEPTH,
)
from eyvan.server.service._control_plane.shared import (
    MAX_LIVE_CHILDREN as MAX_LIVE_CHILDREN,
)
from eyvan.server.service._control_plane.shared import (
    MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE as MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE,
)
from eyvan.server.service._control_plane.shared import (
    POOL_ALTERNATES_LABEL as POOL_ALTERNATES_LABEL,
)
from eyvan.server.service._control_plane.shared import (
    QUEUE_TREND_MAX_BUCKETS as QUEUE_TREND_MAX_BUCKETS,
)
from eyvan.server.service._control_plane.shared import (
    QUEUE_TREND_MAX_WINDOW_S as QUEUE_TREND_MAX_WINDOW_S,
)
from eyvan.server.service._control_plane.shared import (
    QUEUE_TREND_MIN_WINDOW_S as QUEUE_TREND_MIN_WINDOW_S,
)
from eyvan.server.service._control_plane.shared import (
    REMOTE_CALLABLE_SIGNATURE_ERROR as REMOTE_CALLABLE_SIGNATURE_ERROR,
)
from eyvan.server.service._control_plane.shared import (
    REMOTE_CALLABLE_SIGNING_KEY_ERROR as REMOTE_CALLABLE_SIGNING_KEY_ERROR,
)
from eyvan.server.service._control_plane.shared import (
    STUCK_CANCEL_AFTER_S as STUCK_CANCEL_AFTER_S,
)
from eyvan.server.service._control_plane.shared import (
    TERMINAL_WORKLOAD_STATES as TERMINAL_WORKLOAD_STATES,
)
from eyvan.server.service._control_plane.shared import (
    WORKER_COUNT_LABEL as WORKER_COUNT_LABEL,
)
from eyvan.server.service._control_plane.shared import (
    AdmissionPlan as AdmissionPlan,
)
from eyvan.server.service._control_plane.shared import (
    ArtifactCollection as ArtifactCollection,
)
from eyvan.server.service._control_plane.shared import (
    ArtifactHold as ArtifactHold,
)
from eyvan.server.service._control_plane.shared import (
    ArtifactInventoryEntry as ArtifactInventoryEntry,
)
from eyvan.server.service._control_plane.shared import (
    ArtifactInventoryReport as ArtifactInventoryReport,
)
from eyvan.server.service._control_plane.shared import (
    ArtifactInventoryStatus as ArtifactInventoryStatus,
)
from eyvan.server.service._control_plane.shared import (
    ArtifactReferenceRecord as ArtifactReferenceRecord,
)
from eyvan.server.service._control_plane.shared import (
    ArtifactUsage as ArtifactUsage,
)
from eyvan.server.service._control_plane.shared import (
    AuditEvent as AuditEvent,
)
from eyvan.server.service._control_plane.shared import (
    AuthorizationError as AuthorizationError,
)
from eyvan.server.service._control_plane.shared import (
    Authorizer as Authorizer,
)
from eyvan.server.service._control_plane.shared import (
    AutoscaleBlockReason as AutoscaleBlockReason,
)
from eyvan.server.service._control_plane.shared import (
    BaseModel as BaseModel,
)
from eyvan.server.service._control_plane.shared import (
    BatchRequest as BatchRequest,
)
from eyvan.server.service._control_plane.shared import (
    BudgetGrantRecord as BudgetGrantRecord,
)
from eyvan.server.service._control_plane.shared import (
    BudgetRecord as BudgetRecord,
)
from eyvan.server.service._control_plane.shared import (
    BudgetStatus as BudgetStatus,
)
from eyvan.server.service._control_plane.shared import (
    CallableExecution as CallableExecution,
)
from eyvan.server.service._control_plane.shared import (
    CallableProjectTrustPolicy as CallableProjectTrustPolicy,
)
from eyvan.server.service._control_plane.shared import (
    CandidateZone as CandidateZone,
)
from eyvan.server.service._control_plane.shared import (
    CapacityClass as CapacityClass,
)
from eyvan.server.service._control_plane.shared import (
    CapacityPoolRecord as CapacityPoolRecord,
)
from eyvan.server.service._control_plane.shared import (
    CarbonReport as CarbonReport,
)
from eyvan.server.service._control_plane.shared import (
    CarbonSource as CarbonSource,
)
from eyvan.server.service._control_plane.shared import (
    CarbonUnavailable as CarbonUnavailable,
)
from eyvan.server.service._control_plane.shared import (
    CellBillingAccrual as CellBillingAccrual,
)
from eyvan.server.service._control_plane.shared import (
    CellBillingReport as CellBillingReport,
)
from eyvan.server.service._control_plane.shared import (
    CellRecord as CellRecord,
)
from eyvan.server.service._control_plane.shared import (
    CellRejection as CellRejection,
)
from eyvan.server.service._control_plane.shared import (
    CellState as CellState,
)
from eyvan.server.service._control_plane.shared import (
    CommandExecution as CommandExecution,
)
from eyvan.server.service._control_plane.shared import (
    ConflictError as ConflictError,
)
from eyvan.server.service._control_plane.shared import (
    CoordinationKind as CoordinationKind,
)
from eyvan.server.service._control_plane.shared import (
    DagNode as DagNode,
)
from eyvan.server.service._control_plane.shared import (
    EffectiveQuota as EffectiveQuota,
)
from eyvan.server.service._control_plane.shared import (
    ExpansionAdmissionRecord as ExpansionAdmissionRecord,
)
from eyvan.server.service._control_plane.shared import (
    ExpansionAdmissionState as ExpansionAdmissionState,
)
from eyvan.server.service._control_plane.shared import (
    ExpansionLimits as ExpansionLimits,
)
from eyvan.server.service._control_plane.shared import (
    ExpansionMode as ExpansionMode,
)
from eyvan.server.service._control_plane.shared import (
    FederatedBudgetStatus as FederatedBudgetStatus,
)
from eyvan.server.service._control_plane.shared import (
    GcpQueueAdmissionDenialTarget as GcpQueueAdmissionDenialTarget,
)
from eyvan.server.service._control_plane.shared import (
    GrantState as GrantState,
)
from eyvan.server.service._control_plane.shared import (
    GroupRole as GroupRole,
)
from eyvan.server.service._control_plane.shared import (
    HostFaultProof as HostFaultProof,
)
from eyvan.server.service._control_plane.shared import (
    IdempotencyRecord as IdempotencyRecord,
)
from eyvan.server.service._control_plane.shared import (
    InputError as InputError,
)
from eyvan.server.service._control_plane.shared import (
    InvalidManagedArtifactURI as InvalidManagedArtifactURI,
)
from eyvan.server.service._control_plane.shared import (
    InvalidTransition as InvalidTransition,
)
from eyvan.server.service._control_plane.shared import (
    JobTreeTooDeepError as JobTreeTooDeepError,
)
from eyvan.server.service._control_plane.shared import (
    JobTreeTooWideError as JobTreeTooWideError,
)
from eyvan.server.service._control_plane.shared import (
    JsonObject as JsonObject,
)
from eyvan.server.service._control_plane.shared import (
    JsonValue as JsonValue,
)
from eyvan.server.service._control_plane.shared import (
    LaunchLatencyReport as LaunchLatencyReport,
)
from eyvan.server.service._control_plane.shared import (
    LeaseRecord as LeaseRecord,
)
from eyvan.server.service._control_plane.shared import (
    LeaseState as LeaseState,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactConflictError as ManagedArtifactConflictError,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactDeleteInconclusiveError as ManagedArtifactDeleteInconclusiveError,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactDeleteStatus as ManagedArtifactDeleteStatus,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactInventoryError as ManagedArtifactInventoryError,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactInventoryStore as ManagedArtifactInventoryStore,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactKind as ManagedArtifactKind,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactObjectMetadata as ManagedArtifactObjectMetadata,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactOperationInconclusiveError as ManagedArtifactOperationInconclusiveError,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactRecord as ManagedArtifactRecord,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactRef as ManagedArtifactRef,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactState as ManagedArtifactState,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactVersionedStore as ManagedArtifactVersionedStore,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactVersionMismatchError as ManagedArtifactVersionMismatchError,
)
from eyvan.server.service._control_plane.shared import (
    ManagedArtifactWriteInconclusiveError as ManagedArtifactWriteInconclusiveError,
)
from eyvan.server.service._control_plane.shared import (
    NoCarbonSource as NoCarbonSource,
)
from eyvan.server.service._control_plane.shared import (
    NoMatchingCapacityPoolError as NoMatchingCapacityPoolError,
)
from eyvan.server.service._control_plane.shared import (
    NotFoundError as NotFoundError,
)
from eyvan.server.service._control_plane.shared import (
    OrganizationRecord as OrganizationRecord,
)
from eyvan.server.service._control_plane.shared import (
    OutboxRecord as OutboxRecord,
)
from eyvan.server.service._control_plane.shared import (
    OutboxState as OutboxState,
)
from eyvan.server.service._control_plane.shared import (
    Permission as Permission,
)
from eyvan.server.service._control_plane.shared import (
    PlacementMode as PlacementMode,
)
from eyvan.server.service._control_plane.shared import (
    PlanePeerRecord as PlanePeerRecord,
)
from eyvan.server.service._control_plane.shared import (
    PolicyRecord as PolicyRecord,
)
from eyvan.server.service._control_plane.shared import (
    PoolEstimate as PoolEstimate,
)
from eyvan.server.service._control_plane.shared import (
    PoolLaunchLatency as PoolLaunchLatency,
)
from eyvan.server.service._control_plane.shared import (
    PoolWarmth as PoolWarmth,
)
from eyvan.server.service._control_plane.shared import (
    Principal as Principal,
)
from eyvan.server.service._control_plane.shared import (
    PriorityBand as PriorityBand,
)
from eyvan.server.service._control_plane.shared import (
    ProjectRecord as ProjectRecord,
)
from eyvan.server.service._control_plane.shared import (
    ProjectUsageReport as ProjectUsageReport,
)
from eyvan.server.service._control_plane.shared import (
    QueueTrend as QueueTrend,
)
from eyvan.server.service._control_plane.shared import (
    QueueTrendPoint as QueueTrendPoint,
)
from eyvan.server.service._control_plane.shared import (
    QuotaGrantRecord as QuotaGrantRecord,
)
from eyvan.server.service._control_plane.shared import (
    QuotaRecord as QuotaRecord,
)
from eyvan.server.service._control_plane.shared import (
    ReadOnlyQueryResult as ReadOnlyQueryResult,
)
from eyvan.server.service._control_plane.shared import (
    RegionCarbon as RegionCarbon,
)
from eyvan.server.service._control_plane.shared import (
    Repository as Repository,
)
from eyvan.server.service._control_plane.shared import (
    RepositorySession as RepositorySession,
)
from eyvan.server.service._control_plane.shared import (
    ResourceGang as ResourceGang,
)
from eyvan.server.service._control_plane.shared import (
    ResourceHours as ResourceHours,
)
from eyvan.server.service._control_plane.shared import (
    ResourceLimits as ResourceLimits,
)
from eyvan.server.service._control_plane.shared import (
    Role as Role,
)
from eyvan.server.service._control_plane.shared import (
    RoleGrant as RoleGrant,
)
from eyvan.server.service._control_plane.shared import (
    RoleGrantRecord as RoleGrantRecord,
)
from eyvan.server.service._control_plane.shared import (
    SchedulingResult as SchedulingResult,
)
from eyvan.server.service._control_plane.shared import (
    SchedulingService as SchedulingService,
)
from eyvan.server.service._control_plane.shared import (
    ScopeQuota as ScopeQuota,
)
from eyvan.server.service._control_plane.shared import (
    ServiceDecisionKind as ServiceDecisionKind,
)
from eyvan.server.service._control_plane.shared import (
    ServiceDecisionRecord as ServiceDecisionRecord,
)
from eyvan.server.service._control_plane.shared import (
    ServiceObservation as ServiceObservation,
)
from eyvan.server.service._control_plane.shared import (
    ServicePolicy as ServicePolicy,
)
from eyvan.server.service._control_plane.shared import (
    ServiceReplicaHealthState as ServiceReplicaHealthState,
)
from eyvan.server.service._control_plane.shared import (
    SubjectUsageRecord as SubjectUsageRecord,
)
from eyvan.server.service._control_plane.shared import (
    UnavailableError as UnavailableError,
)
from eyvan.server.service._control_plane.shared import (
    UserProgressReport as UserProgressReport,
)
from eyvan.server.service._control_plane.shared import (
    WarmthReport as WarmthReport,
)
from eyvan.server.service._control_plane.shared import (
    WebhookRecord as WebhookRecord,
)
from eyvan.server.service._control_plane.shared import (
    WorkerResourceScope as WorkerResourceScope,
)
from eyvan.server.service._control_plane.shared import (
    WorkloadDefinition as WorkloadDefinition,
)
from eyvan.server.service._control_plane.shared import (
    WorkloadDispatchReservation as WorkloadDispatchReservation,
)
from eyvan.server.service._control_plane.shared import (
    WorkloadEstimate as WorkloadEstimate,
)
from eyvan.server.service._control_plane.shared import (
    WorkloadExplanation as WorkloadExplanation,
)
from eyvan.server.service._control_plane.shared import (
    WorkloadRecord as WorkloadRecord,
)
from eyvan.server.service._control_plane.shared import (
    WorkloadSpendRecord as WorkloadSpendRecord,
)
from eyvan.server.service._control_plane.shared import (
    WorkloadState as WorkloadState,
)
from eyvan.server.service._control_plane.shared import (
    WorkloadTransitionUpdates as WorkloadTransitionUpdates,
)
from eyvan.server.service._control_plane.shared import (
    WorkspaceRef as WorkspaceRef,
)
from eyvan.server.service._control_plane.shared import (
    _artifact_record_update as _artifact_record_update,
)
from eyvan.server.service._control_plane.shared import (
    _capacity_apply_payload as _capacity_apply_payload,
)
from eyvan.server.service._control_plane.shared import (
    _changed_cell_capacity_identity as _changed_cell_capacity_identity,
)
from eyvan.server.service._control_plane.shared import (
    _decimal_sum as _decimal_sum,
)
from eyvan.server.service._control_plane.shared import (
    _declares_a_price as _declares_a_price,
)
from eyvan.server.service._control_plane.shared import (
    _dump_limits as _dump_limits,
)
from eyvan.server.service._control_plane.shared import (
    _explain_base as _explain_base,
)
from eyvan.server.service._control_plane.shared import (
    _explain_from_scheduling_result as _explain_from_scheduling_result,
)
from eyvan.server.service._control_plane.shared import (
    _fingerprint as _fingerprint,
)
from eyvan.server.service._control_plane.shared import (
    _flex_start_refusal as _flex_start_refusal,
)
from eyvan.server.service._control_plane.shared import (
    _host_fault_proof as _host_fault_proof,
)
from eyvan.server.service._control_plane.shared import (
    _is_active_capacity_apply as _is_active_capacity_apply,
)
from eyvan.server.service._control_plane.shared import (
    _is_finite_nonnegative_number as _is_finite_nonnegative_number,
)
from eyvan.server.service._control_plane.shared import (
    _multislice_refusal as _multislice_refusal,
)
from eyvan.server.service._control_plane.shared import (
    _new_id as _new_id,
)
from eyvan.server.service._control_plane.shared import (
    _on_demand_price as _on_demand_price,
)
from eyvan.server.service._control_plane.shared import (
    _pool_declaration as _pool_declaration,
)
from eyvan.server.service._control_plane.shared import (
    _reserved_resource_hours as _reserved_resource_hours,
)
from eyvan.server.service._control_plane.shared import (
    _resource_quantities as _resource_quantities,
)
from eyvan.server.service._control_plane.shared import (
    _scope as _scope,
)
from eyvan.server.service._control_plane.shared import (
    _validate_idempotency_key as _validate_idempotency_key,
)
from eyvan.server.service._control_plane.shared import (
    accelerator_unit_price as accelerator_unit_price,
)
from eyvan.server.service._control_plane.shared import (
    artifact_reference_id as artifact_reference_id,
)
from eyvan.server.service._control_plane.shared import (
    asyncio as asyncio,
)
from eyvan.server.service._control_plane.shared import (
    backend_capabilities as backend_capabilities,
)
from eyvan.server.service._control_plane.shared import (
    candidate_zone_report as candidate_zone_report,
)
from eyvan.server.service._control_plane.shared import (
    cell_matches_template as cell_matches_template,
)
from eyvan.server.service._control_plane.shared import (
    cell_provides_pool_supply as cell_provides_pool_supply,
)
from eyvan.server.service._control_plane.shared import (
    child_idempotency_key as child_idempotency_key,
)
from eyvan.server.service._control_plane.shared import (
    collect_descendants as collect_descendants,
)
from eyvan.server.service._control_plane.shared import (
    costing as costing,
)
from eyvan.server.service._control_plane.shared import (
    dag_validate as dag_validate,
)
from eyvan.server.service._control_plane.shared import (
    datetime as datetime,
)
from eyvan.server.service._control_plane.shared import (
    ensure_lease_transition as ensure_lease_transition,
)
from eyvan.server.service._control_plane.shared import (
    ensure_workload_transition as ensure_workload_transition,
)
from eyvan.server.service._control_plane.shared import (
    exceeds_quota_headroom as exceeds_quota_headroom,
)
from eyvan.server.service._control_plane.shared import (
    expand as expand,
)
from eyvan.server.service._control_plane.shared import (
    expansion_admission_id as expansion_admission_id,
)
from eyvan.server.service._control_plane.shared import (
    federated_spend as federated_spend,
)
from eyvan.server.service._control_plane.shared import (
    fresh_reading as fresh_reading,
)
from eyvan.server.service._control_plane.shared import (
    group_member_names as group_member_names,
)
from eyvan.server.service._control_plane.shared import (
    group_validate as group_validate,
)
from eyvan.server.service._control_plane.shared import (
    hashlib as hashlib,
)
from eyvan.server.service._control_plane.shared import (
    is_managed_uri as is_managed_uri,
)
from eyvan.server.service._control_plane.shared import (
    issuable_amount as issuable_amount,
)
from eyvan.server.service._control_plane.shared import (
    json as json,
)
from eyvan.server.service._control_plane.shared import (
    locality_index as locality_index,
)
from eyvan.server.service._control_plane.shared import (
    logger as logger,
)
from eyvan.server.service._control_plane.shared import (
    logging as logging,
)
from eyvan.server.service._control_plane.shared import (
    managed_result_uri as managed_result_uri,
)
from eyvan.server.service._control_plane.shared import (
    matching_secret_reference_root as matching_secret_reference_root,
)
from eyvan.server.service._control_plane.shared import (
    math as math,
)
from eyvan.server.service._control_plane.shared import (
    model_json as model_json,
)
from eyvan.server.service._control_plane.shared import (
    outstanding_total as outstanding_total,
)
from eyvan.server.service._control_plane.shared import (
    p95_latency as p95_latency,
)
from eyvan.server.service._control_plane.shared import (
    parent_depth as parent_depth,
)
from eyvan.server.service._control_plane.shared import (
    parse_host_fault_marker as parse_host_fault_marker,
)
from eyvan.server.service._control_plane.shared import (
    parse_managed_uri as parse_managed_uri,
)
from eyvan.server.service._control_plane.shared import (
    parse_tpu_accelerator_type as parse_tpu_accelerator_type,
)
from eyvan.server.service._control_plane.shared import (
    plan_expansion as plan_expansion,
)
from eyvan.server.service._control_plane.shared import (
    preferred_cells_for as preferred_cells_for,
)
from eyvan.server.service._control_plane.shared import (
    providers_hosting_multi_worker_cells as providers_hosting_multi_worker_cells,
)
from eyvan.server.service._control_plane.shared import (
    quota_headroom as quota_headroom,
)
from eyvan.server.service._control_plane.shared import (
    quota_headroom_children as quota_headroom_children,
)
from eyvan.server.service._control_plane.shared import (
    replace as replace,
)
from eyvan.server.service._control_plane.shared import (
    replica_health_state as replica_health_state,
)
from eyvan.server.service._control_plane.shared import (
    resource_quantities as resource_quantities,
)
from eyvan.server.service._control_plane.shared import (
    scaled_service_metadata as scaled_service_metadata,
)
from eyvan.server.service._control_plane.shared import (
    secrets as secrets,
)
from eyvan.server.service._control_plane.shared import (
    service_metadata as service_metadata,
)
from eyvan.server.service._control_plane.shared import (
    service_policy as service_policy,
)
from eyvan.server.service._control_plane.shared import (
    spend_of as spend_of,
)
from eyvan.server.service._control_plane.shared import (
    stable_id as stable_id,
)
from eyvan.server.service._control_plane.shared import (
    storage_scope as storage_scope,
)
from eyvan.server.service._control_plane.shared import (
    task_index as task_index,
)
from eyvan.server.service._control_plane.shared import (
    timedelta as timedelta,
)
from eyvan.server.service._control_plane.shared import (
    tp as tp,
)
from eyvan.server.service._control_plane.shared import (
    utcnow as utcnow,
)
from eyvan.server.service._control_plane.shared import (
    uuid as uuid,
)
from eyvan.server.service._control_plane.shared import (
    validate_capacity_pool_quota_groups as validate_capacity_pool_quota_groups,
)
from eyvan.server.service._control_plane.shared import (
    validate_parent as validate_parent,
)
from eyvan.server.service._control_plane.shared import (
    validate_webhook_target as validate_webhook_target,
)
from eyvan.server.service._control_plane.submission import SubmissionMixin
from eyvan.server.service._control_plane.team_requests import TeamRequestsMixin
from eyvan.server.service._control_plane.tenancy import TenancyMixin
from eyvan.server.service._control_plane.transactions import TransactionsMixin
from eyvan.server.service._control_plane.webhooks import WebhooksMixin
from eyvan.server.service._control_plane.workload_lifecycle import WorkloadLifecycleMixin
from eyvan.server.service._control_plane.workload_settlement import WorkloadSettlementMixin
from eyvan.server.service._control_plane.workload_storage import WorkloadStorageMixin


class ControlPlaneService(
    AccountsMixin,
    AssistantProvidersMixin,
    AssistantConversationsMixin,
    ResearchPlansMixin,
    TeamRequestsMixin,
    GovernanceMixin,
    SchedulingMixin,
    SubmissionMixin,
    CallableSubmissionMixin,
    WorkloadStorageMixin,
    ExpansionMixin,
    WorkloadLifecycleMixin,
    WorkloadSettlementMixin,
    CellsMixin,
    FleetsMixin,
    LeasesMixin,
    WebhooksMixin,
    PoliciesMixin,
    TenancyMixin,
    ServicesMixin,
    ChildSubmissionMixin,
    EstimatesMixin,
    FederationMixin,
    EventsMixin,
    TransactionsMixin,
    LeaseSettlementMixin,
    ReportingMixin,
    ArtifactAdmissionMixin,
    ArtifactIoMixin,
    ArtifactRecoveryMixin,
    ArtifactInventoryMixin,
    ArtifactCollectionMixin,
    AuditMixin,
):
    """Own authorization, lifecycle validation, idempotency, and audit atomicity."""

    def __init__(
        self,
        repository: Repository,
        *,
        authorizer: Authorizer | None = None,
        callable_trust: CallableProjectTrustPolicy = CallableProjectTrustPolicy(),
        scheduler: SchedulingService | None = None,
        #: Only the artifact collector needs it, and only to *delete*. Optional so
        #: every deployment that does not run a collector -- which is all of them
        #: until an operator opts in -- carries no path to removing bytes at all.
        artifact_store: ManagedArtifactVersionedStore | None = None,
        #: Read-only/additive inventory can be exposed before the deletion state
        #: machine. It is deliberately separate from ``artifact_store`` so wiring
        #: API uploads for inventory does not silently enable the current unsafe
        #: collector path in a deployment.
        inventory_store: ManagedArtifactInventoryStore | None = None,
        #: This plane's identity among peers. Empty means "not federated", which is
        #: the default: one plane per region with projects pinned to one region
        #: already works, and a plane with no id simply cannot issue grants.
        plane_id: str = "",
        #: Where grid intensity comes from. ``NoCarbonSource`` by default and in
        #: every shipped configuration: a provider Eyvan invented would publish
        #: numbers nobody can check, and the reporting path is the only place a
        #: real one may ever be consulted.
        carbon: CarbonSource | None = None,
        #: How large a fan-out one request may create, and how much of it may
        #: compete at once. The default is exactly what this package enforced
        #: before admission control existed -- the historical 1000-child ceiling
        #: and no waves -- so a deployment that configures nothing is unchanged.
        expansion_limits: ExpansionLimits = ExpansionLimits(),
        correlation_enabled: bool = False,
        #: Team governance (per-user fleet rules, approval holds, reclaim). Off by
        #: default; the API enables it only on a team-mode plane.
        governance_enabled: bool = False,
    ):
        self.governance_enabled = governance_enabled
        self.correlation_enabled = correlation_enabled
        self.repository = repository
        self.authorizer = authorizer or Authorizer()
        self.callable_trust = callable_trust
        self.scheduler = scheduler
        self.artifact_store = artifact_store
        self.inventory_store = inventory_store
        self.plane_id = plane_id
        self.carbon = carbon or NoCarbonSource()
        self.expansion_limits = expansion_limits

    @staticmethod
    async def _current_lease(tx: RepositorySession, workload: WorkloadRecord) -> LeaseRecord | None:
        leases = await ControlPlaneService._current_leases(tx, workload)
        return leases[0] if leases else None
