// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  cloneMock,
  createMockAttachment,
  mockAttachmentJobs,
  mockAttachmentPlan,
  mockDiscoveredTpus,
  createMockWorkload,
  getMockWorkload,
  getMockRun,
  getMockRunSeries,
  getMockRunHistograms,
  getMockRunEpisodes,
  getMockEpisodeCurve,
  getMockEpisodeBreakdown,
  getMockRunRollouts,
  getMockRunRollout,
  getMockRunCompletions,
  getMockRunCompletion,
  getMockCompletionStats,
  getMockRunTable,
  getMockRunMedia,
  getMockRunBlob,
  mockRuns,
  mockAuditEvents,
  mockBudgets,
  mockAvailableShapes,
  mockCells,
  mockLeases,
  mockPolicies,
  mockWorkloads,
} from "./mockData";
import {
  apiAuthHeaders,
  configuredScope,
  invalidateRefusedLocalCredential,
  isDemoMode,
  rejectSession,
  type AuthScope,
} from "../auth/oidc";
import { EventStreamHttpError, followEventStream, type StreamEvent } from "../lib/eventStream";
import type {
  AttachmentInput,
  AttachmentJob,
  AttachmentPlan,
  AuditEvent,
  DiscoveredTpus,
  FleetInitScriptDocument,
  AssistantChatStreamInput,
  AssistantConversation,
  AssistantConversationSummary,
  AssistantDraft,
  AssistantMessage,
  AssistantStreamEvent,
  ClaudeCodeLoginStart,
  ClaudeCodeStatus,
  AssistantModelDiscovery,
  AssistantModelDiscoveryInput,
  AssistantPreset,
  AssistantProvider,
  AssistantProviderInput,
  AssistantProviderList,
  AssistantProviderUpdate,
  AssistantStatus,
  AssistantTurn,
  ResearchPlan,
  ResearchPlanState,
  ResearchPlanSummary,
  OpenRouterSignInInput,
  OpenRouterSignInStart,
  TeamRequest,
  TeamTier,
  TeamInvite,
  TeamInviteCreated,
  TeamInviteInput,
  TeamToken,
  TeamTokenCreated,
  TeamUser,
  TeamUserRules,
  GovernanceSummary,
  ReclaimPolicy,
  AvailableShape,
  BillingReport,
  CapacityPoolSummary,
  ProjectUsage,
  Budget,
  CapacityLease,
  Cell,
  ControlPlaneHealth,
  EffectiveProjectScope,
  LaunchLatencyReport,
  LogEntry,
  OverviewData,
  PriorityBand,
  RunDetail,
  RunRollout,
  RunRolloutsOptions,
  RunRolloutsResponse,
  RunCompletion,
  RunCompletionsOptions,
  RunCompletionsResponse,
  CompletionStatsOptions,
  CompletionStatsResponse,
  RunTableOptions,
  RunTableResponse,
  RunMediaOptions,
  RunMediaResponse,
  RunHistogramsOptions,
  RunHistogramsResponse,
  RunEpisodesOptions,
  RunEpisodesResponse,
  EpisodeCurveOptions,
  EpisodeCurveResponse,
  EpisodeBreakdownOptions,
  EpisodeBreakdownResponse,
  RunSeriesOptions,
  RunSeriesResponse,
  RunStatus,
  RunSummary,
  SchedulingPolicy,
  SessionAuthority,
  SessionPermission,
  StorageStatus,
  SubmitWorkloadInput,
  TpuQuotaSnapshot,
  WarmthReport,
  WorkloadExplanation,
  WorkloadProcessStatus,
  WorkloadSpend,
  TimelineEvent,
  WorkloadDetail,
  WorkloadStatus,
  WorkloadSummary,
} from "../types";

interface RawResources {
  cpu: number;
  memory_gib: number;
  gpu: number;
  tpu: number;
  custom: Record<string, number>;
}

interface RawWorkload {
  id: string;
  org_id: string;
  project_id: string;
  state: string;
  attempt: number;
  /**
   * How the plane accounted for every attempt after the first. A spot
   * reclamation and a capacity loss are charged to different budgets because
   * they mean opposite things about whether running again can work, and only
   * one of them is the expected ending of a spot job. Optional: a plane that
   * predates the split reports neither, and absent must read as unknown rather
   * than as zero preemptions.
   */
  preemption_count?: number;
  capacity_loss_count?: number;
  failure_reason?: string | null;
  lease_id?: string;
  checkpoint_uri?: string;
  scheduling_reason?: string;
  reclaim_deadline_at?: string | null;
  reclaim_requested_by?: string | null;
  reclaim_policy?: ReclaimPolicy | null;
  created_by: string;
  created_at: string;
  updated_at: string;
  spec: {
    name: string;
    execution: { kind: "command"; command: string } | { kind: "callable"; artifact_uri: string };
    gang: { replicas: number; per_replica: RawResources; required_topology: Record<string, string> };
    metadata: Record<string, string>;
    accelerator_type?: string;
    priority_band: PriorityBand;
    /** Slices allocated together (pods=N). Absent on planes predating multislice. */
    cell_group_count?: number;
    worker_count?: number;
    checkpoint_grace_seconds?: number | null;
    reclaim_policy?: ReclaimPolicy | null;
  };
}

interface RawCell {
  id: string;
  provider: string;
  provider_project?: string;
  provider_account?: string | null;
  region: string;
  zone?: string;
  state: string;
  capacity: RawResources;
  allocatable: RawResources;
  accelerator_type?: string;
  capacity_class?: string;
  /** Closed code naming why the cell is unavailable; present only then. */
  unavailable_reason?: string | null;
  /** The provider's own diagnostic behind `unavailable_reason`. */
  unavailable_detail?: string | null;
  labels: Record<string, string>;
  topology?: Record<string, string>;
  last_heartbeat_at?: string;
  availability?: {
    cooldown_until?: string | null;
    consecutive_failures?: number;
    interruptions?: number;
  };
}

/** One advertised pool from `/capacity`, readable with `submit` alone. */
interface RawAvailablePool {
  fleet: string;
  pool: string;
  accelerator_type: string;
  capacity_class: string;
  region: string;
  zone?: string | null;
  enabled: boolean;
}

/** One run's totalled spend, as the workload cost route reports it. */
interface RawWorkloadSpend {
  workload_id: string;
  state: string;
  accelerator_hours: Record<string, number>;
  settled: number;
  priced: boolean;
  reserved: number;
  leases: number;
  shape?: string | null;
  capacity_class?: string | null;
  on_demand_hourly?: number | null;
  on_demand_source: string;
}

interface RawLease {
  id: string;
  workload_id: string;
  project_id: string;
  cell_id: string;
  state: string;
  resources: { replicas: number; slices?: number; per_replica: RawResources };
  hourly_cost: number;
  expires_at: string;
}

interface RawBudget {
  org_id: string;
  project_id: string;
  spent: number;
  limit: number;
  period: string;
  hard_limit: boolean;
  accelerator_hour_limits: Record<string, number>;
  accelerator_hours_spent: Record<string, number>;
  currency: string;
}

interface RawQueueTrend {
  observed_at: string;
  bucket_seconds: number;
  points: Array<{ started_at: string; submitted: number; admitted: number }>;
}

interface RawSessionScope {
  org_id: string;
  project_id: string;
  display_name: string;
  registered: boolean;
  enabled: boolean;
  accepts_new_work: boolean;
  roles: string[];
  permissions: SessionPermission[];
}

interface RawSessionAuthority {
  subject: string;
  display_name?: string | null;
  global_permissions: SessionPermission[];
  scopes: RawSessionScope[];
}

interface RawBudgetStatus {
  org_id: string;
  project_id: string;
  limit: number;
  spent: number;
  forecast: number | null;
}

interface RawPolicy {
  org_id: string;
  project_id: string;
  allowed_regions: string[];
  allowed_capacity_classes: string[];
  preemption_enabled: boolean;
  allowed_providers: string[];
  denied_accelerator_types: string[];
  max_priority: number;
  updated_at: string;
}

interface RawEvent {
  id: string;
  sequence: number;
  timestamp: string;
  actor: string;
  entity_type: string;
  entity_id: string;
  action: string;
  previous_state?: string;
  new_state?: string;
  payload: Record<string, unknown>;
}

/** Subset of the server WorkloadProgress contract used by the dashboard. */
interface RawWorkloadProgress {
  workload_id: string;
  workload_state: string;
  phase: string;
  scheduling_reason?: string | null;
  failure_reason?: string | null;
  request: { cells: number; workers_per_cell: number };
  capacity: { ready_workers: number };
}

export interface WorkloadFilters {
  search?: string;
  status?: string;
  priority?: string;
}

/**
 * Render a FastAPI error `detail` as text. Validation failures (422) send a
 * list of `{loc, msg}` objects; interpolating that printed "[object Object]".
 */
export function formatErrorDetail(detail: unknown): string {
  if (detail === undefined || detail === null || detail === "") return "";
  if (typeof detail === "string") return `: ${detail}`;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (item && typeof item === "object" && "msg" in item) {
        const msg = String((item as { msg: unknown }).msg).replace(/^Value error, /, "");
        const loc = (item as { loc?: unknown }).loc;
        const field = Array.isArray(loc) ? loc.filter((part) => part !== "body").join(".") : "";
        return field ? `${field}: ${msg}` : msg;
      }
      return typeof item === "string" ? item : JSON.stringify(item);
    });
    return messages.length ? `: ${messages.join("; ")}` : "";
  }
  return `: ${JSON.stringify(detail)}`;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Read a newline-delimited body as it arrives, one complete line at a time.
 * A chunk can end mid-line, or mid-character, so the decoder streams and the
 * unfinished tail waits for the next chunk; a final line without a newline is
 * still delivered when the body ends.
 */
export async function readNdjsonLines(response: Response, onLine: (line: string) => void): Promise<void> {
  const decoder = new TextDecoder();
  let buffer = "";
  const emit = (final: boolean) => {
    const lines = buffer.split("\n");
    buffer = final ? "" : (lines.pop() ?? "");
    for (const raw of lines) {
      const line = raw.trim();
      if (line) onLine(line);
    }
  };
  const reader = response.body?.getReader();
  if (!reader) {
    buffer = await response.text();
    emit(true);
    return;
  }
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      emit(false);
    }
    buffer += decoder.decode();
    emit(true);
  } catch (error) {
    void reader.cancel().catch(() => undefined);
    throw error;
  }
}

/** One NDJSON line of the assistant stream; a line that is not an event object is refused. */
export function parseAssistantStreamEvent(line: string): AssistantStreamEvent {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch {
    throw new ApiError("The assistant sent an unreadable update");
  }
  if (!parsed || typeof parsed !== "object" || typeof (parsed as { type?: unknown }).type !== "string") {
    throw new ApiError("The assistant sent an unreadable update");
  }
  return parsed as AssistantStreamEvent;
}

export function workloadStatus(value: string): WorkloadStatus {
  const states: Record<string, WorkloadStatus> = {
    pending: "queued",
    admitted: "planning",
    leased: "provisioning",
    dispatching: "provisioning",
    running: "running",
    checkpointing: "checkpointing",
    cancelling: "preempting",
    succeeded: "completed",
    failed: "failed",
    // A cancellation is not a failure. Collapsing them told an operator their
    // job had broken when they had stopped it themselves, and inflated the
    // Failed tile that is the first thing read on this page.
    cancelled: "cancelled",
  };
  return states[value] ?? "queued";
}

function mapWorkload(raw: RawWorkload): WorkloadDetail {
  const resources = raw.spec.gang.per_replica;
  const acceleratorKind = resources.tpu ? "tpu" : resources.gpu ? "gpu" : undefined;
  const accelerator = acceleratorKind ? (raw.spec.accelerator_type ?? acceleratorKind.toUpperCase()) : undefined;
  return {
    id: raw.id,
    name: raw.spec.name,
    org: raw.org_id,
    project: raw.project_id,
    user: raw.created_by,
    status: workloadStatus(raw.state),
    priority: raw.spec.priority_band,
    resources: {
      cpu: resources.cpu,
      memoryGb: resources.memory_gib,
      accelerator,
      acceleratorKind,
      acceleratorCount: resources.tpu || resources.gpu || undefined,
      replicas: raw.spec.gang.replicas,
      // Replicas are the product of both counts: cells × workers-per-cell.
      // The callable path stores `pods` as worker_count with one cell group,
      // so reading either field alone reported x1 for every SDK submission.
      slices:
        (typeof raw.spec.cell_group_count === "number" ? raw.spec.cell_group_count : 1) *
        (typeof raw.spec.worker_count === "number" ? raw.spec.worker_count : 1),
      topology: Object.values(raw.spec.gang.required_topology).join(", ") || undefined,
    },
    submittedAt: raw.created_at,
    startedAt: raw.state === "running" ? raw.updated_at : undefined,
    queueReason: raw.scheduling_reason,
    command: raw.spec.execution.kind === "command" ? raw.spec.execution.command : raw.spec.execution.artifact_uri,
    attempt: raw.attempt,
    preemptionCount: raw.preemption_count,
    capacityLossCount: raw.capacity_loss_count,
    failureReason: raw.failure_reason ?? undefined,
    checkpointUri: raw.checkpoint_uri,
    leaseId: raw.lease_id,
    reclaimDeadlineAt: raw.reclaim_deadline_at ?? undefined,
    reclaimRequestedBy: raw.reclaim_requested_by ?? undefined,
    reclaimPolicy: raw.reclaim_policy ?? raw.spec.reclaim_policy ?? undefined,
    checkpointGraceSeconds:
      typeof raw.spec.checkpoint_grace_seconds === "number" ? raw.spec.checkpoint_grace_seconds : undefined,
    // Populated by getWorkload from this workload's leases; the list view has no
    // allocation to show.
    allocation: [],
    timeline: [],
    logs: [],
    labels: raw.spec.metadata,
  };
}

/** Summarize one lease's per-replica resources for the allocation panel. */
function describeResources(resources: RawResources, replicas: number): string {
  const parts: string[] = [];
  if (resources.tpu) parts.push(`${resources.tpu} TPU`);
  if (resources.gpu) parts.push(`${resources.gpu} GPU`);
  if (resources.cpu) parts.push(`${resources.cpu} CPU`);
  const summary = parts.join(", ") || "no reserved resources";
  return replicas > 1 ? `${replicas} x ${summary}` : summary;
}

/** Map a durable lease state onto the three states the panel renders. */
/** Map a durable cell state to how the fleet page should read it.
 *
 * The previous expression was `ready ? healthy : draining ? maintenance :
 * degraded`, which made every other state degraded. Two of them are not:
 * `registering` is the normal state of a cell queued for capacity -- on a Spot
 * TPU that is the majority of its life -- and `deleted` is a cell that no longer
 * exists. Rendering both in red made a healthy fleet waiting on a Spot grant
 * look broken, which is the fastest way to make an operator distrust the page.
 */
export function cellStatus(state: string): "healthy" | "starting" | "maintenance" | "degraded" {
  if (state === "ready") return "healthy";
  if (state === "registering") return "starting";
  if (state === "draining" || state === "deleting") return "maintenance";
  return "degraded";
}

function leaseAllocationState(state: string): "ready" | "starting" | "draining" {
  if (state === "active") return "ready";
  if (state === "revoking" || state === "released" || state === "expired") return "draining";
  return "starting";
}

function resourceRows(raw: RawResources): Array<{ name: string; total: number }> {
  const resources: Array<[string, number]> = [
    ["CPU", raw.cpu],
    ["Memory GiB", raw.memory_gib],
    ["TPU", raw.tpu],
    ["GPU", raw.gpu],
    ...Object.entries(raw.custom),
  ];
  return resources.filter(([, total]) => total > 0).map(([name, total]) => ({ name, total }));
}

function describeTopology(topology: Record<string, string> | undefined): string {
  const entries = Object.entries(topology ?? {}).sort(([left], [right]) => left.localeCompare(right));
  return entries.map(([key, value]) => `${key}=${value}`).join(", ") || "Unspecified";
}

function mapPolicy(policy: RawPolicy): SchedulingPolicy {
  const constraints = [
    policy.allowed_regions.join(", ") || "all regions",
    policy.allowed_capacity_classes.join(", ") || "all capacity classes",
    policy.allowed_providers.join(", ") || "all providers",
  ].join("; ");
  return {
    id: `${policy.org_id}/${policy.project_id}`,
    name: `Placement policy (${policy.project_id})`,
    scope: policy.project_id,
    type: "placement",
    summary: `${constraints}; preemption ${policy.preemption_enabled ? "enabled" : "disabled"}`,
    preemptionEnabled: policy.preemption_enabled,
    updatedAt: policy.updated_at,
    configuration: {
      allowedProviders: policy.allowed_providers,
      allowedRegions: policy.allowed_regions,
      allowedCapacityClasses: policy.allowed_capacity_classes,
      deniedAcceleratorTypes: policy.denied_accelerator_types,
      maxPriority: policy.max_priority,
    },
  };
}

const LOG_LEVELS: Record<string, NonNullable<LogEntry["level"]>> = {
  TRACE: "DEBUG",
  DEBUG: "DEBUG",
  INFO: "INFO",
  WARN: "WARN",
  WARNING: "WARN",
  ERROR: "ERROR",
  CRITICAL: "ERROR",
  FATAL: "ERROR",
};

/** Extract timestamp, level, and source from a log line, leaving them unset when absent. */
function parseLogLine(line: string): Pick<LogEntry, "timestamp" | "level" | "source" | "message"> {
  let rest = line;
  let timestamp: string | undefined;
  const stamp = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)(Z|[+-]\d{2}:?\d{2})?\s+/.exec(rest);
  if (stamp) {
    timestamp = `${stamp[1]}T${stamp[2].replace(",", ".")}${stamp[3] ?? "Z"}`;
    rest = rest.slice(stamp[0].length);
  }
  let level: LogEntry["level"];
  const levelMatch = /^(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)[\s:]+/i.exec(rest);
  if (levelMatch) {
    level = LOG_LEVELS[levelMatch[1].toUpperCase()];
    rest = rest.slice(levelMatch[0].length);
  }
  let source: string | undefined;
  const sourceMatch = /^\[([^\]]+)\]\s+/.exec(rest);
  if (sourceMatch) {
    source = sourceMatch[1];
    rest = rest.slice(sourceMatch[0].length);
  }
  return { timestamp, level, source, message: rest || line };
}

/** One structured log record as the follow stream frames it. */
interface RawLogRecord {
  message: string;
  stream?: string;
  timestamp?: string | null;
  rank?: number;
  pid?: number | null;
}

/**
 * Project one streamed record onto the entry the log view renders.
 *
 * The rank leads the source column because in a gang it is the identity that
 * matters: sixteen hosts printing the same line is normal, and one of them
 * printing something different is the entire diagnosis. A bracketed tag inside
 * the line is kept alongside it rather than instead of it -- the program's own
 * idea of where the line came from does not replace which process said it.
 *
 * The record's capture time is a fallback, not the preference. A line that
 * stamps itself dates the event; the record dates the read, and on a host that
 * has been buffering those differ by however long the buffer held.
 */
function mapLogRecord(record: RawLogRecord, id: string): LogEntry {
  const parsed = parseLogLine(record.message);
  const rank = typeof record.rank === "number" ? `rank ${record.rank}` : undefined;
  const source = [rank, parsed.source].filter(Boolean).join(" · ") || undefined;
  const stream = record.stream === "stdout" || record.stream === "stderr" ? record.stream : ("unknown" as const);
  return {
    id,
    timestamp: parsed.timestamp ?? record.timestamp ?? undefined,
    level: parsed.level,
    source,
    stream,
    message: parsed.message,
  };
}

/** Derive a capacity-acquisition percentage from the workload progress contract. */
function mapProgressPercent(progress: RawWorkloadProgress): number | undefined {
  if (progress.phase === "terminal") {
    return progress.workload_state === "succeeded" ? 100 : undefined;
  }
  const requestedWorkers = progress.request.cells * progress.request.workers_per_cell;
  if (requestedWorkers <= 0) return undefined;
  const ready = Math.min(progress.capacity.ready_workers, requestedWorkers);
  return Math.round((ready / requestedWorkers) * 100);
}

/** Compute the overview from already-fetched dashboard data so each resource is requested once per refresh. */
export function computeOverview(
  workloads: WorkloadSummary[],
  cells: Cell[],
  budgets: Budget[],
  // Passed in rather than derived here: the trend comes from the audit ledger,
  // which this function does not have and should not fetch.
  queueTrend: OverviewData["queueTrend"] = [],
): OverviewData {
  const queued = workloads.filter((item) => item.status === "queued").length;
  const running = workloads.filter((item) => item.status === "running").length;
  const terminal = workloads.filter((item) => ["completed", "failed", "cancelled"].includes(item.status));
  const resourceBreakdown = new Map<string, { used: number; total: number; pending: number }>();
  cells
    .flatMap((cell) => cell.resources)
    .forEach((resource) => {
      const current = resourceBreakdown.get(resource.name) ?? { used: 0, total: 0, pending: 0 };
      current.used += resource.used;
      current.total += resource.total;
      resourceBreakdown.set(resource.name, current);
    });
  workloads
    .filter((workload) => ["queued", "planning"].includes(workload.status))
    .forEach((workload) => {
      const pendingResources: Array<[string, number]> = [
        ["CPU", workload.resources.cpu * workload.resources.replicas],
        ["Memory GiB", workload.resources.memoryGb * workload.resources.replicas],
        [
          workload.resources.acceleratorKind === "gpu" ? "GPU" : "TPU",
          (workload.resources.acceleratorCount ?? 0) * workload.resources.replicas,
        ],
      ];
      pendingResources.forEach(([name, pending]) => {
        const current = resourceBreakdown.get(name) ?? { used: 0, total: 0, pending: 0 };
        current.pending += pending;
        resourceBreakdown.set(name, current);
      });
    });
  const priorityQueue = (["production", "interactive", "batch"] as PriorityBand[]).map((priority) => {
    const queuedInBand = workloads.filter((item) => item.status === "queued" && item.priority === priority);
    const oldestSubmittedAt = queuedInBand.reduce(
      (oldest, item) => Math.min(oldest, Date.parse(item.submittedAt)),
      Date.now(),
    );
    return {
      priority,
      count: queuedInBand.length,
      oldestMinutes: queuedInBand.length ? Math.max(0, Math.floor((Date.now() - oldestSubmittedAt) / 60_000)) : 0,
    };
  });
  return {
    queued,
    running,
    fleetUtilization: cells.length
      ? Math.round(cells.reduce((sum, cell) => sum + cell.utilization, 0) / cells.length)
      : 0,
    monthlySpend: budgets.reduce((sum, budget) => sum + budget.spent, 0),
    monthlyBudget: budgets.reduce((sum, budget) => sum + budget.limit, 0),
    successRate: terminal.length
      ? (terminal.filter((item) => item.status === "completed").length / terminal.length) * 100
      : undefined,
    queueTrend,
    priorityQueue,
    fleetBreakdown: [...resourceBreakdown].map(([resource, values]) => ({ resource, ...values })),
    alerts: budgets
      .filter((budget) => budget.forecast !== undefined && budget.forecast > budget.limit)
      .map((budget) => ({
        id: budget.id,
        severity: "warning",
        title: "Budget forecast above limit",
        detail: budget.owner,
        timestamp: new Date().toISOString(),
      })),
  };
}

/** Budget for a control-plane read that is answered from its own store. */
const CONTROL_PLANE_TIMEOUT_MS = 3_500;

/** Complete authenticated media transfers may contain up to 64 MiB. */
const MEDIA_TRANSFER_TIMEOUT_MS = 120_000;

/**
 * Budget for a read the control plane forwards to a worker.
 *
 * Logs are fetched from the host that produced them, so the latency is a
 * network round trip to a TPU VM rather than a database read -- 12s is ordinary
 * for a live slice. Sharing the control-plane budget meant the workload detail
 * page timed out on every workload whose logs were remote, which is every
 * workload that has actually run somewhere.
 */
// Forwarded log reads go browser -> plane -> cell agent. A dead or churning
// agent used to hold the whole detail page hostage for thirty seconds; the
// archive serves this attempt's history anyway, so five seconds is plenty
// before falling back to it.
const WORKER_FORWARDED_TIMEOUT_MS = 5_000;

/**
 * Budget for a read the control plane answers by calling the cloud provider.
 *
 * Provider quota discovery lists accelerator types in every quota zone a
 * project can see: `gcloud` calls, not a database read, and the better part of
 * a minute on a cold cache. On the control-plane budget the quota page timed
 * out on every first visit and showed "Eyvan API request timed out" -- the read
 * itself was fine and finishing, and nothing was ever waiting for it.
 *
 * Comfortably above the server's own deadline, which answers with the previous
 * reading or a stated failure rather than hanging. This budget exists to
 * outlast that answer, not to replace it.
 */
const PROVIDER_READ_TIMEOUT_MS = 45_000;

export class EyvanApiClient {
  private activeScope: AuthScope;

  constructor(
    private readonly baseUrl = "/api/v1",
    readonly forceDemo = isDemoMode(),
    initialScope: AuthScope = configuredScope(),
  ) {
    this.activeScope = { ...initialScope };
  }

  getScope(): AuthScope {
    return { ...this.activeScope };
  }

  setScope(scope: AuthScope): void {
    if (!scope.orgId || !scope.projectId) throw new Error("dashboard scope requires an organization and project");
    this.activeScope = { ...scope };
  }

  private request<T>(path: string, init: RequestInit = {}, timeoutMs = CONTROL_PLANE_TIMEOUT_MS): Promise<T> {
    return this.requestBody(path, init, timeoutMs, async (response) => (await response.json()) as T);
  }

  private attachmentRequest<T>(path: string, init: RequestInit = {}, timeoutMs?: number): Promise<T> {
    // Public literal paths keep the browser route inventory auditable; the
    // shared requester still applies the configured API base and auth headers.
    return this.request<T>(path.replace(/^\/api\/v1/, ""), init, timeoutMs);
  }

  async planAttachment(input: AttachmentInput, signal?: AbortSignal): Promise<AttachmentPlan> {
    if (this.forceDemo) return cloneMock(mockAttachmentPlan(input));
    return this.attachmentRequest(
      "/api/v1/attachments/plan",
      { method: "POST", body: JSON.stringify(input), signal },
      130_000,
    );
  }

  async startAttachment(input: AttachmentInput): Promise<AttachmentJob> {
    if (this.forceDemo) return createMockAttachment(input);
    return this.attachmentRequest("/api/v1/attachments", { method: "POST", body: JSON.stringify(input) });
  }

  async getAttachment(id: string, signal?: AbortSignal): Promise<AttachmentJob> {
    if (this.forceDemo) {
      const job = mockAttachmentJobs.find((candidate) => candidate.id === id);
      if (!job) throw new ApiError("Attachment not found", 404);
      return cloneMock(job);
    }
    return this.attachmentRequest(`/api/v1/attachments/${encodeURIComponent(id)}`, { signal });
  }

  async listAttachments(signal?: AbortSignal): Promise<AttachmentJob[]> {
    if (this.forceDemo) return cloneMock(mockAttachmentJobs);
    const response = await this.attachmentRequest<{ jobs: AttachmentJob[] }>("/api/v1/attachments", { signal });
    return response.jobs;
  }

  async listDiscoveredTpus(signal?: AbortSignal): Promise<DiscoveredTpus> {
    if (this.forceDemo) return cloneMock(mockDiscoveredTpus);
    return this.attachmentRequest<DiscoveredTpus>("/api/v1/attachments/discovered", { signal });
  }

  /**
   * Read non-secret regional & monitoring GCS storage configuration plus detected
   * GCP context. Returns `null` when speaking to an older control plane that does
   * not expose `/api/v1/storage/status`.
   */
  async getStorageStatus(signal?: AbortSignal): Promise<StorageStatus | null> {
    if (this.forceDemo) {
      return {
        configured: true,
        auto_create: false,
        config_path: "~/.eyvan/storage.json",
        project: "demo-project",
        monitoring: {
          bucket: "eyvan-monitoring-demo",
          uri: "gs://eyvan-monitoring-demo/tracking",
          location: "us-central1",
          prefix: "tracking",
        },
        regional: {
          "us-central2": {
            bucket: "eyvan_us-central2",
            uri: "gs://eyvan_us-central2",
            location: "us-central2",
          },
        },
        context: {
          on_gcp: false,
          project: "demo-project",
          zone: null,
          region: "us-central2",
          accelerator_type: null,
          source: "demo",
        },
      };
    }
    try {
      return await this.attachmentRequest<StorageStatus>("/api/v1/storage/status", { signal });
    } catch (error) {
      if (error instanceof ApiError && (error.status === 404 || error.status === 405 || error.status === 501)) {
        return null;
      }
      throw error;
    }
  }

  // A fleet's bash init script (fleet admins). Full literal paths, like the
  // attachment routes, so the browser route inventory stays auditable. Demo
  // mode has no fleet behind it, so it refuses rather than pretending to save.
  async getFleetInitScript(fleet: string, signal?: AbortSignal): Promise<FleetInitScriptDocument> {
    if (this.forceDemo) throw new ApiError("Demo mode has no fleet init script", 400);
    return this.attachmentRequest<FleetInitScriptDocument>(`/api/v1/fleets/${encodeURIComponent(fleet)}/init-script`, {
      signal,
    });
  }

  async setFleetInitScript(fleet: string, script: string, signal?: AbortSignal): Promise<FleetInitScriptDocument> {
    if (this.forceDemo) throw new ApiError("Demo mode has no fleet to change", 400);
    return this.attachmentRequest<FleetInitScriptDocument>(`/api/v1/fleets/${encodeURIComponent(fleet)}/init-script`, {
      method: "PUT",
      headers: { "Idempotency-Key": globalThis.crypto?.randomUUID?.() ?? `dashboard-${Date.now()}` },
      body: JSON.stringify({ script }),
      signal,
    });
  }

  // Team accounts. Full literal paths, like the attachment routes, so the
  // browser route inventory stays auditable. Demo mode has no team accounts.

  async listTeamUsers(signal?: AbortSignal): Promise<TeamUser[]> {
    if (this.forceDemo) return [];
    return (await this.attachmentRequest<{ users: TeamUser[] }>("/api/v1/team/users", { signal })).users;
  }

  /**
   * Read team governance quota, borrowing, and active-reclaim summary.
   * Returns `null` on older planes without `/api/v1/governance/summary` or when
   * the caller lacks fleet governance read access.
   */
  async getGovernanceSummary(signal?: AbortSignal): Promise<GovernanceSummary | null> {
    if (this.forceDemo) return null;
    try {
      return await this.attachmentRequest<GovernanceSummary>("/api/v1/governance/summary", { signal });
    } catch (error) {
      if (
        error instanceof ApiError &&
        (error.status === 404 || error.status === 405 || error.status === 501 || error.status === 403)
      ) {
        return null;
      }
      throw error;
    }
  }

  async setTeamUserDisabled(subject: string, disabled: boolean): Promise<TeamUser> {
    const name = encodeURIComponent(subject);
    return this.attachmentRequest<TeamUser>(
      disabled ? `/api/v1/team/users/${name}/disable` : `/api/v1/team/users/${name}/enable`,
      { method: "POST" },
    );
  }

  /** Update only the supplied rules (fleet admins); null clears that rule. */
  async setTeamUserRules(subject: string, rules: Partial<TeamUserRules>): Promise<TeamUser> {
    return this.attachmentRequest<TeamUser>(`/api/v1/team/users/${encodeURIComponent(subject)}/rules`, {
      method: "PUT",
      body: JSON.stringify(rules),
    });
  }

  /** Move one account to a tier (fleet admins); trusted/user need the project scope to grant roles in. */
  async setTeamUserTier(
    subject: string,
    tier: TeamTier,
    orgId: string | null,
    projectId: string | null,
  ): Promise<TeamUser> {
    return this.attachmentRequest<TeamUser>(`/api/v1/team/users/${encodeURIComponent(subject)}/tier`, {
      method: "PUT",
      body: JSON.stringify({ tier, org_id: orgId, project_id: projectId }),
    });
  }

  async listTeamInvites(signal?: AbortSignal): Promise<TeamInvite[]> {
    if (this.forceDemo) return [];
    return (await this.attachmentRequest<{ invites: TeamInvite[] }>("/api/v1/team/invites", { signal })).invites;
  }

  async createTeamInvite(input: TeamInviteInput): Promise<TeamInviteCreated> {
    const body: Record<string, string> = { email: input.email, subject: input.subject };
    if (input.role) body.role = input.role;
    if (input.tier) body.tier = input.tier;
    if (input.org_id) body.org_id = input.org_id;
    if (input.project_id) body.project_id = input.project_id;
    return this.attachmentRequest<TeamInviteCreated>("/api/v1/team/invites", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async revokeTeamInvite(inviteId: string): Promise<TeamInvite> {
    return this.attachmentRequest<TeamInvite>(`/api/v1/team/invites/${encodeURIComponent(inviteId)}`, {
      method: "DELETE",
    });
  }

  async listTeamTokens(signal?: AbortSignal): Promise<TeamToken[]> {
    if (this.forceDemo) return [];
    return (await this.attachmentRequest<{ tokens: TeamToken[] }>("/api/v1/team/tokens", { signal })).tokens;
  }

  async createTeamToken(name: string, expiresInDays: number | null): Promise<TeamTokenCreated> {
    return this.attachmentRequest<TeamTokenCreated>("/api/v1/team/tokens", {
      method: "POST",
      body: JSON.stringify(expiresInDays ? { name, expires_in_days: expiresInDays } : { name }),
    });
  }

  async revokeTeamToken(tokenId: string): Promise<TeamToken> {
    return this.attachmentRequest<TeamToken>(`/api/v1/team/tokens/${encodeURIComponent(tokenId)}`, {
      method: "DELETE",
    });
  }

  // Fleet assistant. The plane holds the provider key and runs every tool as
  // this user; drafts come back unsubmitted.

  async getAssistantStatus(signal?: AbortSignal): Promise<AssistantStatus> {
    if (this.forceDemo) return { enabled: false, model: null };
    return this.attachmentRequest<AssistantStatus>("/api/v1/assistant/status", { signal });
  }

  async listAssistantProviders(signal?: AbortSignal): Promise<AssistantProviderList> {
    if (this.forceDemo) return { providers: [], can_manage: false };
    return this.attachmentRequest<AssistantProviderList>("/api/v1/assistant/providers", { signal });
  }

  async createAssistantProvider(input: AssistantProviderInput): Promise<AssistantProvider> {
    return this.attachmentRequest<AssistantProvider>("/api/v1/assistant/providers", {
      method: "POST",
      body: JSON.stringify(input),
    });
  }

  async updateAssistantProvider(id: string, changes: AssistantProviderUpdate): Promise<AssistantProvider> {
    return this.attachmentRequest<AssistantProvider>(`/api/v1/assistant/providers/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify(changes),
    });
  }

  async deleteAssistantProvider(id: string): Promise<{ deleted: string }> {
    return this.attachmentRequest<{ deleted: string }>(`/api/v1/assistant/providers/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  }

  async listAssistantPresets(signal?: AbortSignal): Promise<AssistantPreset[]> {
    if (this.forceDemo) return [];
    return (await this.attachmentRequest<{ presets: AssistantPreset[] }>("/api/v1/assistant/presets", { signal }))
      .presets;
  }

  /** Ask the provider for its models; a provider failure answers with suggestions, not an error. */
  async discoverAssistantModels(
    body: AssistantModelDiscoveryInput,
    signal?: AbortSignal,
  ): Promise<AssistantModelDiscovery> {
    if (this.forceDemo) return { models: [], source: "suggested", detail: null };
    return this.attachmentRequest<AssistantModelDiscovery>(
      "/api/v1/assistant/models/discover",
      { method: "POST", body: JSON.stringify(body), signal },
      PROVIDER_READ_TIMEOUT_MS,
    );
  }

  /** Begin OpenRouter's OAuth flow; the caller sends the browser to `authorize_url`. */
  async startOpenRouterSignIn(body: OpenRouterSignInInput): Promise<OpenRouterSignInStart> {
    return this.attachmentRequest<OpenRouterSignInStart>("/api/v1/assistant/oauth/openrouter/start", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async assistantChat(
    messages: AssistantMessage[],
    selection?: { provider_id: string; model: string } | null,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<AssistantTurn> {
    return this.attachmentRequest<AssistantTurn>(
      "/api/v1/assistant/chat",
      {
        method: "POST",
        body: JSON.stringify({
          messages,
          org_id: scope.orgId,
          project_id: scope.projectId,
          ...(selection ? { provider_id: selection.provider_id, model: selection.model } : {}),
        }),
        signal,
      },
      150_000,
    );
  }

  /**
   * One chat turn, streamed. Only the new message is sent; the plane keeps the
   * conversation and answers with NDJSON events, each handed to `onEvent` as
   * its line completes. A refusal before streaming starts throws its detail.
   */
  async assistantChatStream(
    body: AssistantChatStreamInput,
    onEvent: (event: AssistantStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    if (this.forceDemo) throw new ApiError("Demo mode has no assistant", 400);
    const path = "/api/v1/assistant/chat/stream".replace(/^\/api\/v1/, "");
    const payload: AssistantChatStreamInput = {
      ...body,
      org_id: body.org_id ?? this.activeScope.orgId,
      project_id: body.project_id ?? this.activeScope.projectId,
    };
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: this.requestHeaders({ Accept: "application/x-ndjson" }),
        body: JSON.stringify(payload),
        signal,
      });
    } catch (error) {
      if (signal?.aborted) throw error;
      throw new ApiError(
        error instanceof Error ? `Unable to reach Eyvan API: ${error.message}` : "Unable to reach Eyvan API",
      );
    }
    if (!response.ok) throw await this.refusal(path, response);
    try {
      await readNdjsonLines(response, (line) => onEvent(parseAssistantStreamEvent(line)));
    } catch (error) {
      if (signal?.aborted || error instanceof ApiError) throw error;
      throw new ApiError(
        error instanceof Error
          ? `The assistant's answer was interrupted: ${error.message}`
          : "The assistant's answer was interrupted",
      );
    }
  }

  /** The caller's saved conversations, newest first. */
  async listAssistantConversations(signal?: AbortSignal): Promise<AssistantConversationSummary[]> {
    if (this.forceDemo) return [];
    return (
      await this.attachmentRequest<{ conversations: AssistantConversationSummary[] }>(
        "/api/v1/assistant/conversations",
        { signal },
      )
    ).conversations;
  }

  async getAssistantConversation(id: string, signal?: AbortSignal): Promise<AssistantConversation> {
    if (this.forceDemo) throw new ApiError("Conversation not found", 404);
    return this.attachmentRequest<AssistantConversation>(`/api/v1/assistant/conversations/${encodeURIComponent(id)}`, {
      signal,
    });
  }

  async renameAssistantConversation(id: string, title: string): Promise<AssistantConversationSummary> {
    return this.attachmentRequest<AssistantConversationSummary>(
      `/api/v1/assistant/conversations/${encodeURIComponent(id)}`,
      { method: "PUT", body: JSON.stringify({ title }) },
    );
  }

  async deleteAssistantConversation(id: string): Promise<{ deleted: string }> {
    return this.attachmentRequest<{ deleted: string }>(`/api/v1/assistant/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  }

  /** What the plane host's Claude Code CLI says about itself (fleet admins). */
  async getClaudeCodeStatus(signal?: AbortSignal): Promise<ClaudeCodeStatus> {
    if (this.forceDemo) {
      return {
        installed: false,
        version: null,
        logged_in: false,
        auth_method: null,
        subscription_type: null,
        detail: null,
      };
    }
    return this.attachmentRequest<ClaudeCodeStatus>(
      "/api/v1/assistant/claude-code/status",
      { signal },
      PROVIDER_READ_TIMEOUT_MS,
    );
  }

  /** Start the CLI's sign-in on the plane host; 409 while one is already running. */
  async startClaudeCodeLogin(): Promise<ClaudeCodeLoginStart> {
    if (this.forceDemo) return { started: false, url: null };
    return this.attachmentRequest<ClaudeCodeLoginStart>(
      "/api/v1/assistant/claude-code/login",
      { method: "POST" },
      PROVIDER_READ_TIMEOUT_MS,
    );
  }

  /** Submit a draft the user confirmed: an ordinary submission under their own identity. */
  async submitAssistantDraft(draft: AssistantDraft): Promise<{ id: string }> {
    return this.attachmentRequest<{ id: string }>("/api/v1/workloads", {
      method: "POST",
      headers: { "Idempotency-Key": globalThis.crypto?.randomUUID?.() ?? `assistant-${Date.now()}` },
      body: JSON.stringify(draft),
    });
  }

  // Research plans. The assistant only proposes one; its owner approves it
  // here and the plane then submits the jobs. A plan is never a workload draft.

  /** Your plans (fleet admins see everyone's), oldest first. */
  async listResearchPlans(state?: ResearchPlanState, signal?: AbortSignal): Promise<ResearchPlanSummary[]> {
    if (this.forceDemo) return [];
    const query = state ? `?state=${encodeURIComponent(state)}` : "";
    return (
      await this.attachmentRequest<{ plans: ResearchPlanSummary[] }>(`/api/v1/research/plans${query}`, { signal })
    ).plans;
  }

  async getResearchPlan(planId: string, signal?: AbortSignal): Promise<ResearchPlan> {
    if (this.forceDemo) throw new ApiError("Research plan not found", 404);
    return this.attachmentRequest<ResearchPlan>(`/api/v1/research/plans/${encodeURIComponent(planId)}`, { signal });
  }

  /** Approve a proposed plan (its owner only); nothing ran before this. */
  async approveResearchPlan(planId: string): Promise<ResearchPlan> {
    if (this.forceDemo) throw new ApiError("Demo mode has no research plan to approve", 400);
    return this.attachmentRequest<ResearchPlan>(`/api/v1/research/plans/${encodeURIComponent(planId)}/approve`, {
      method: "POST",
    });
  }

  /** Stop a plan (owner or fleet admin); the plane cancels its running workloads. */
  async stopResearchPlan(planId: string, reason = ""): Promise<ResearchPlan> {
    if (this.forceDemo) throw new ApiError("Demo mode has no research plan to stop", 400);
    return this.attachmentRequest<ResearchPlan>(`/api/v1/research/plans/${encodeURIComponent(planId)}/stop`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    });
  }

  async listTeamRequests(signal?: AbortSignal): Promise<TeamRequest[]> {
    if (this.forceDemo) return [];
    return (await this.attachmentRequest<{ requests: TeamRequest[] }>("/api/v1/team/requests", { signal })).requests;
  }

  async fileTeamRequest(
    kind: TeamRequest["kind"],
    summary: string,
    scope: AuthScope = this.activeScope,
  ): Promise<TeamRequest> {
    return this.attachmentRequest<TeamRequest>("/api/v1/team/requests", {
      method: "POST",
      body: JSON.stringify({ kind, summary, org_id: scope.orgId, project_id: scope.projectId }),
    });
  }

  async resolveTeamRequest(
    requestId: string,
    resolution: "approved" | "declined" | "withdrawn",
    note = "",
  ): Promise<TeamRequest> {
    return this.attachmentRequest<TeamRequest>(`/api/v1/team/requests/${encodeURIComponent(requestId)}/resolve`, {
      method: "POST",
      body: JSON.stringify({ resolution, note }),
    });
  }

  private requestBlob(path: string, signal?: AbortSignal): Promise<Blob> {
    return this.requestBody(path, { signal, headers: { Accept: "*/*" } }, MEDIA_TRANSFER_TIMEOUT_MS, (response) =>
      response.blob(),
    );
  }

  /** The headers every JSON request carries: content negotiation plus this session's credential. */
  private requestHeaders(extra?: HeadersInit): HeadersInit {
    return {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...apiAuthHeaders(),
      ...(extra as Record<string, string> | undefined),
    };
  }

  /** Turn a non-2xx answer into an ApiError carrying the plane's `detail`, retiring a refused session. */
  private async refusal(path: string, response: Response): Promise<ApiError> {
    let detail = "";
    try {
      detail = formatErrorDetail(((await response.json()) as { detail?: unknown }).detail);
    } catch {
      // The HTTP status remains actionable when an upstream returns a non-JSON error body.
    }
    // Tracking has a separate upstream credential. Its refusals belong to
    // the tracking panel, never the browser's control-plane session.
    // Compared by first path segment rather than a quoted path prefix: the
    // route guard in tests/test_dashboard_calls_routes_that_exist.py treats
    // every quoted absolute path in this file as a request it must serve.
    const isTrackingRead = path.split("/")[1] === "tracking";
    if (response.status === 401 && !isTrackingRead) {
      rejectSession();
      // A refused loopback credential retires itself here: the gate let
      // this page through on the value it held, and the plane has just
      // said that value will never be accepted. Without this, a rotated
      // or reset credential turns every panel into the same 401 wall the
      // gate exists to prevent -- the working command one terminal line
      // away and never named.
      invalidateRefusedLocalCredential(detail);
    }
    return new ApiError(`Eyvan API request failed with ${response.status}${detail}`, response.status);
  }

  /** JSON and binary reads share authentication, cancellation and refusal handling. */
  private async requestBody<T>(
    path: string,
    init: RequestInit,
    timeoutMs: number,
    read: (response: Response) => Promise<T>,
  ): Promise<T> {
    const controller = new AbortController();
    const externalSignal = init.signal;
    const cancel = () => controller.abort();
    if (externalSignal?.aborted) cancel();
    else externalSignal?.addEventListener("abort", cancel, { once: true });
    const timeout = globalThis.setTimeout(cancel, timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: this.requestHeaders(init.headers),
        signal: controller.signal,
      });
      if (!response.ok) throw await this.refusal(path, response);
      return await read(response);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (controller.signal.aborted) throw new ApiError("Eyvan API request timed out");
      throw new ApiError(
        error instanceof Error ? `Unable to reach Eyvan API: ${error.message}` : "Unable to reach Eyvan API",
      );
    } finally {
      externalSignal?.removeEventListener("abort", cancel);
      globalThis.clearTimeout(timeout);
    }
  }

  private async requestText(
    path: string,
    timeoutMs = CONTROL_PLANE_TIMEOUT_MS,
    externalSignal?: AbortSignal,
  ): Promise<string> {
    const controller = new AbortController();
    const cancel = () => controller.abort();
    if (externalSignal?.aborted) cancel();
    else externalSignal?.addEventListener("abort", cancel, { once: true });
    const timeout = globalThis.setTimeout(cancel, timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}${path}`, { headers: apiAuthHeaders(), signal: controller.signal });
      if (!response.ok) {
        if (response.status === 401) {
          rejectSession();
        }
        throw new ApiError(`Eyvan API request failed with ${response.status}`, response.status);
      }
      return await response.text();
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (controller.signal.aborted) throw new ApiError("Eyvan API request timed out");
      throw new ApiError(
        error instanceof Error ? `Unable to reach Eyvan API: ${error.message}` : "Unable to reach Eyvan API",
      );
    } finally {
      externalSignal?.removeEventListener("abort", cancel);
      globalThis.clearTimeout(timeout);
    }
  }

  private scopeSnapshot(scope: AuthScope = this.activeScope): AuthScope {
    return { orgId: scope.orgId, projectId: scope.projectId };
  }

  private scope(path: string, scope: AuthScope): string {
    const join = path.includes("?") ? "&" : "?";
    return `${path}${join}org_id=${encodeURIComponent(scope.orgId)}&project_id=${encodeURIComponent(scope.projectId)}`;
  }

  async listRuns(
    options: { status?: RunStatus; limit?: number } = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunSummary[]> {
    const immutableScope = this.scopeSnapshot(scope);
    if (this.forceDemo) {
      return cloneMock(mockRuns)
        .filter((run) => !options.status || run.status === options.status)
        .slice(0, options.limit ?? 200)
        .map((run) => ({ ...run, org_id: immutableScope.orgId, project_id: immutableScope.projectId }));
    }
    const query = new URLSearchParams();
    if (options.status) query.set("status", options.status);
    if (options.limit !== undefined) query.set("limit", String(options.limit));
    const result = await this.request<{ runs: RunSummary[] }>(this.scope(`/tracking/runs?${query}`, immutableScope), {
      signal,
    });
    return result.runs;
  }

  async getRun(runId: string, scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<RunDetail> {
    const immutableScope = this.scopeSnapshot(scope);
    if (this.forceDemo) {
      const run = getMockRun(runId);
      if (!run) throw new ApiError("Run not found", 404);
      return { ...run, org_id: immutableScope.orgId, project_id: immutableScope.projectId };
    }
    return this.request<RunDetail>(this.scope(`/tracking/runs/${encodeURIComponent(runId)}`, immutableScope), {
      signal,
    });
  }

  /**
   * Read one tracking resource. Callers pass the full literal path so the
   * route guard in tests/test_dashboard_calls_routes_that_exist.py can see it.
   */
  private trackingRead<T>(
    path: string,
    options: Record<string, string | number | boolean | undefined>,
    scope: AuthScope,
    signal?: AbortSignal,
  ): Promise<T> {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(options)) if (value !== undefined) query.set(key, String(value));
    return this.request<T>(this.scope(`${path}?${query}`, this.scopeSnapshot(scope)), { signal });
  }

  private demoRun(runId: string): RunDetail {
    const run = getMockRun(runId);
    if (!run) throw new ApiError("Run not found", 404);
    return run;
  }

  async listRunRollouts(
    runId: string,
    options: RunRolloutsOptions = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunRolloutsResponse> {
    if (this.forceDemo) return getMockRunRollouts(this.demoRun(runId), options);
    return this.trackingRead(
      `/tracking/runs/${encodeURIComponent(runId)}/rollouts`,
      {
        kind: options.kind,
        actor: options.actor,
        limit: options.limit ?? 50,
        offset: options.offset ?? 0,
        include_replayed: options.includeReplayed ?? false,
      },
      scope,
      signal,
    );
  }

  async getRunRollout(
    runId: string,
    rolloutId: string,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunRollout> {
    if (this.forceDemo) {
      const rollout = getMockRunRollout(this.demoRun(runId), rolloutId);
      if (!rollout) throw new ApiError("Rollout not found", 404);
      return rollout;
    }
    return this.trackingRead(
      `/tracking/runs/${encodeURIComponent(runId)}/rollouts/${encodeURIComponent(rolloutId)}`,
      {},
      scope,
      signal,
    );
  }

  async listRunCompletions(
    runId: string,
    options: RunCompletionsOptions = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunCompletionsResponse> {
    if (this.forceDemo) return getMockRunCompletions(this.demoRun(runId), options);
    return this.trackingRead(
      `/tracking/runs/${encodeURIComponent(runId)}/completions`,
      {
        step: options.step,
        limit: options.limit ?? 50,
        offset: options.offset ?? 0,
        sort: options.sort ?? "recent",
        include_replayed: options.includeReplayed ?? false,
      },
      scope,
      signal,
    );
  }

  async getRunCompletion(
    runId: string,
    completionId: string,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunCompletion> {
    if (this.forceDemo) {
      const group = getMockRunCompletion(this.demoRun(runId), completionId);
      if (!group) throw new ApiError("Completion group not found", 404);
      return group;
    }
    return this.trackingRead(
      `/tracking/runs/${encodeURIComponent(runId)}/completions/${encodeURIComponent(completionId)}`,
      {},
      scope,
      signal,
    );
  }

  async getCompletionStats(
    runId: string,
    options: CompletionStatsOptions = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<CompletionStatsResponse> {
    if (this.forceDemo) return getMockCompletionStats(this.demoRun(runId), options.points ?? 200);
    return this.trackingRead(
      `/tracking/runs/${encodeURIComponent(runId)}/completions/stats`,
      { points: options.points ?? 200 },
      scope,
      signal,
    );
  }

  async getRunTable(
    runId: string,
    options: RunTableOptions = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunTableResponse> {
    if (this.forceDemo) {
      const table = getMockRunTable(this.demoRun(runId), options);
      if (!table) throw new ApiError("Table not found", 404);
      return table;
    }
    return this.trackingRead(`/tracking/runs/${encodeURIComponent(runId)}/tables`, { ...options }, scope, signal);
  }

  async listRunMedia(
    runId: string,
    options: RunMediaOptions = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunMediaResponse> {
    if (this.forceDemo) return getMockRunMedia(this.demoRun(runId), options);
    return this.trackingRead(
      `/tracking/runs/${encodeURIComponent(runId)}/media`,
      { key: options.key, limit: options.limit ?? 100, offset: options.offset ?? 0 },
      scope,
      signal,
    );
  }

  async fetchRunBlob(
    runId: string,
    digest: string,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<Blob> {
    if (this.forceDemo) {
      const blob = getMockRunBlob(this.demoRun(runId), digest);
      if (!blob) throw new ApiError("Blob not found", 404);
      return blob;
    }
    return this.requestBlob(
      this.scope(
        `/tracking/runs/${encodeURIComponent(runId)}/blobs/${encodeURIComponent(digest)}`,
        this.scopeSnapshot(scope),
      ),
      signal,
    );
  }

  async getRunHistograms(
    runId: string,
    options: RunHistogramsOptions,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunHistogramsResponse> {
    const immutableScope = this.scopeSnapshot(scope);
    if (this.forceDemo) {
      const run = getMockRun(runId);
      if (!run) throw new ApiError("Run not found", 404);
      return getMockRunHistograms(run, options);
    }
    const query = new URLSearchParams({
      key: options.key,
      x: options.x ?? "step",
      points: String(options.points ?? 200),
      bins: String(options.bins ?? 64),
    });
    if (options.rank !== undefined) query.set("rank", String(options.rank));
    return this.request<RunHistogramsResponse>(
      this.scope(`/tracking/runs/${encodeURIComponent(runId)}/histograms?${query}`, immutableScope),
      { signal },
    );
  }

  async listRunEpisodes(
    runId: string,
    options: RunEpisodesOptions = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunEpisodesResponse> {
    const immutableScope = this.scopeSnapshot(scope);
    if (this.forceDemo) {
      const run = getMockRun(runId);
      if (!run) throw new ApiError("Run not found", 404);
      return getMockRunEpisodes(run, options);
    }
    const query = new URLSearchParams({
      limit: String(options.limit ?? 100),
      offset: String(options.offset ?? 0),
      include_replayed: String(options.includeReplayed ?? false),
    });
    if (options.kind !== undefined) query.set("kind", options.kind);
    if (options.actor !== undefined) query.set("actor", options.actor);
    if (options.env !== undefined) query.set("env", options.env);
    return this.request<RunEpisodesResponse>(
      this.scope(`/tracking/runs/${encodeURIComponent(runId)}/episodes?${query}`, immutableScope),
      { signal },
    );
  }

  async getEpisodeCurve(
    runId: string,
    options: EpisodeCurveOptions = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<EpisodeCurveResponse> {
    const immutableScope = this.scopeSnapshot(scope);
    if (this.forceDemo) {
      const run = getMockRun(runId);
      if (!run) throw new ApiError("Run not found", 404);
      return getMockEpisodeCurve(run, options);
    }
    const query = new URLSearchParams({
      metric: options.metric ?? "return",
      kind: options.kind ?? "train",
      x: options.x ?? "step",
      points: String(options.points ?? 200),
    });
    if (options.env !== undefined) query.set("env", options.env);
    if (options.actor !== undefined) query.set("actor", options.actor);
    return this.request<EpisodeCurveResponse>(
      this.scope(`/tracking/runs/${encodeURIComponent(runId)}/episodes/curve?${query}`, immutableScope),
      { signal },
    );
  }

  async getEpisodeBreakdown(
    runId: string,
    options: EpisodeBreakdownOptions = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<EpisodeBreakdownResponse> {
    const immutableScope = this.scopeSnapshot(scope);
    if (this.forceDemo) {
      const run = getMockRun(runId);
      if (!run) throw new ApiError("Run not found", 404);
      return getMockEpisodeBreakdown(run, options);
    }
    const query = new URLSearchParams({ kind: options.kind ?? "train", last: String(options.last ?? 200) });
    return this.request<EpisodeBreakdownResponse>(
      this.scope(`/tracking/runs/${encodeURIComponent(runId)}/episodes/breakdown?${query}`, immutableScope),
      { signal },
    );
  }

  async getRunSeries(
    runId: string,
    options: RunSeriesOptions,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<RunSeriesResponse> {
    const immutableScope = this.scopeSnapshot(scope);
    const x = options.x ?? "step";
    const points = options.points ?? 1000;
    if (options.keys.length > 32) throw new Error("Request at most 32 tracking keys at a time");
    if (this.forceDemo) {
      const run = getMockRun(runId);
      if (!run) throw new ApiError("Run not found", 404);
      return getMockRunSeries(run, options.keys, x, points);
    }
    const query = new URLSearchParams({ x, points: String(points) });
    for (const key of options.keys) query.append("key", key);
    if (options.rank !== undefined) query.set("rank", String(options.rank));
    if (options.includeReplayed !== undefined) query.set("include_replayed", String(options.includeReplayed));
    return this.request<RunSeriesResponse>(
      this.scope(`/tracking/runs/${encodeURIComponent(runId)}/series?${query}`, immutableScope),
      { signal },
    );
  }

  async getSessionAuthority(preferred: AuthScope = this.activeScope, signal?: AbortSignal): Promise<SessionAuthority> {
    const preferredScope = this.scopeSnapshot(preferred);
    if (this.forceDemo) {
      const scope: EffectiveProjectScope = {
        ...preferredScope,
        displayName: "Demo project",
        registered: true,
        enabled: true,
        acceptsNewWork: true,
        roles: ["fleet_admin"],
        permissions: ["read", "submit", "operate", "manage_project", "manage_own_capacity", "manage_fleet"],
      };
      return {
        subject: "dashboard-demo",
        displayName: "Demo dataset",
        globalPermissions: ["read", "submit", "operate", "manage_project", "manage_own_capacity", "manage_fleet"],
        scopes: [scope],
      };
    }
    const raw = await this.request<RawSessionAuthority>(this.scope("/session", preferredScope), {
      method: "GET",
      signal,
    });
    return {
      subject: raw.subject,
      displayName: raw.display_name ?? undefined,
      globalPermissions: raw.global_permissions,
      scopes: raw.scopes.map((scope) => ({
        orgId: scope.org_id,
        projectId: scope.project_id,
        displayName: scope.display_name,
        registered: scope.registered,
        enabled: scope.enabled,
        acceptsNewWork: scope.accepts_new_work,
        roles: scope.roles,
        permissions: scope.permissions,
      })),
    };
  }

  /**
   * Follow this tenant's audit stream, calling `onEvent` as changes commit.
   *
   * A change *notification*, not a data channel. The stream carries audit
   * events for one (org, project); the dashboard's own model spans fleet
   * capacity too, so an event means "refetch", and the slow poll beside it
   * still covers what this stream cannot see -- another tenant's work landing
   * on a shared cell, or a cell going unhealthy without an audit event.
   *
   * Resolves only when `signal` aborts. In demo mode it resolves immediately:
   * there is no plane to stream from, and pretending otherwise would show a
   * "live" indicator over mock data.
   */
  async followEvents(
    onEvent: (event: StreamEvent) => void,
    signal: AbortSignal,
    onConnectionChange?: (connected: boolean) => void,
    scope: AuthScope = this.activeScope,
  ): Promise<void> {
    if (this.forceDemo) return;
    const requestScope = this.scopeSnapshot(scope);
    try {
      await followEventStream({
        url: `${this.baseUrl}${this.scope("/events/stream", requestScope)}`,
        headers: apiAuthHeaders(),
        signal,
        onEvent,
        onConnectionChange,
      });
    } catch (streamError) {
      if (streamError instanceof EventStreamHttpError) {
        if (streamError.status === 401) rejectSession();
        throw new ApiError(streamError.message, streamError.status);
      }
      throw streamError;
    }
  }

  /**
   * Follow one workload's output as it is produced, record by record.
   *
   * A data channel, unlike `followEvents`: these frames carry the lines
   * themselves, so nothing is refetched and no page read can substitute. That
   * is why it is a separate stream -- the audit stream is keyed on a global
   * sequence for one (org, project), while log records are per
   * (workload, attempt, rank) and have no global order at all.
   *
   * `onAttemptChanged` is not a notification the caller may ignore. A retry
   * replaces the remote execution, so the lines that follow belong to a
   * different process; appending them to what is already on screen would show
   * a job that resumed where it left off, which is the one story the reader
   * must not be told.
   *
   * Resolves only when `signal` aborts. In demo mode it resolves immediately:
   * there is no process to follow, and replaying canned lines under a "live"
   * indicator is a lie the reader cannot detect.
   */
  async followWorkloadLogs(
    id: string,
    handlers: {
      onRecord: (entry: LogEntry) => void;
      onAttemptChanged: (attempt: number) => void;
      onConnectionChange?: (connected: boolean) => void;
    },
    signal: AbortSignal,
  ): Promise<void> {
    if (this.forceDemo) return;
    // Only ever incremented, and never derived from the record's own content:
    // two ranks emit identical lines at the same millisecond routinely, and a
    // key collision makes React reuse one row for both.
    let delivered = 0;
    try {
      await followEventStream({
        url: `${this.baseUrl}/workloads/${encodeURIComponent(id)}/logs/stream`,
        headers: apiAuthHeaders(),
        signal,
        onConnectionChange: handlers.onConnectionChange,
        onEvent: (event) => {
          if (event.event === "log.attempt_changed") {
            const attempt = (event.data as { attempt?: number } | null)?.attempt;
            if (typeof attempt === "number") handlers.onAttemptChanged(attempt);
            return;
          }
          if (event.event !== "log.record") return;
          const record = event.data as RawLogRecord | null;
          if (!record || typeof record.message !== "string") return;
          delivered += 1;
          handlers.onRecord(mapLogRecord(record, `${id}-stream-${delivered}`));
        },
      });
    } catch (streamError) {
      if (streamError instanceof EventStreamHttpError) {
        if (streamError.status === 401) rejectSession();
        throw new ApiError(streamError.message, streamError.status);
      }
      throw streamError;
    }
  }

  async getHealth(signal?: AbortSignal): Promise<ControlPlaneHealth> {
    if (this.forceDemo) return { status: "demo", service: "eyvan" };
    return this.request<ControlPlaneHealth>("/health", { signal });
  }

  async getOverview(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<OverviewData> {
    const requestScope = this.scopeSnapshot(scope);
    const [workloads, cells, budgets, queueTrend] = await Promise.all([
      this.listWorkloads({}, requestScope, signal),
      this.listCells(undefined, requestScope, signal),
      this.listBudgets(requestScope, signal),
      this.getQueueTrend(requestScope, signal),
    ]);
    return computeOverview(workloads, cells, budgets, queueTrend);
  }

  async getQueueTrend(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<OverviewData["queueTrend"]> {
    if (this.forceDemo) return [];
    const requestScope = this.scopeSnapshot(scope);
    // Seven explicit 24-hour buckets match the seven chart columns. These are
    // submission/admission event counts, not instantaneous queue depth.
    const raw = await this.request<RawQueueTrend>(
      this.scope("/capacity/queue-trend?window_s=86400&buckets=7", requestScope),
      { method: "GET", signal },
    );
    return raw.points.map((point) => ({
      label: new Date(point.started_at).toISOString().slice(11, 16),
      submitted: point.submitted,
      admitted: point.admitted,
    }));
  }

  async listWorkloads(
    filters: WorkloadFilters = {},
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<WorkloadSummary[]> {
    const requestScope = this.scopeSnapshot(scope);
    const workloads: WorkloadSummary[] = this.forceDemo
      ? cloneMock(mockWorkloads)
      : (
          await this.request<RawWorkload[]>(this.scope("/workloads", requestScope), {
            method: "GET",
            signal,
          })
        ).map(mapWorkload);
    const search = filters.search?.toLowerCase();
    return (
      workloads
        .filter(
          (workload) =>
            (!search ||
              workload.name.toLowerCase().includes(search) ||
              workload.id.toLowerCase().includes(search) ||
              workload.user.toLowerCase().includes(search)) &&
            (!filters.status || workload.status === filters.status) &&
            (!filters.priority || workload.priority === filters.priority),
        )
        // Newest first. The control plane returns insertion order, so a fleet that
        // has run for a while buried the job you just submitted under every job
        // you had ever submitted -- and the one you want is almost always the last
        // one. Ties break on id so the order is stable across polls rather than
        // shuffling under the reader.
        .sort((left, right) => {
          const delta = Date.parse(right.submittedAt) - Date.parse(left.submittedAt);
          return delta !== 0 && Number.isFinite(delta) ? delta : right.id.localeCompare(left.id);
        })
    );
  }

  async getWorkload(id: string, scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<WorkloadDetail> {
    // Capture the caller's UI scope before the first await. The resource route
    // resolves its own immutable scope server-side; the snapshot fences this
    // composition from a later project-switch mutating the active client.
    const requestScope = this.scopeSnapshot(scope);
    void requestScope;
    if (this.forceDemo) {
      const mock = getMockWorkload(id);
      if (!mock) throw new ApiError(`Workload ${id} was not found`, 404);
      return cloneMock(mock);
    }
    const raw = await this.request<RawWorkload>(`/workloads/${encodeURIComponent(id)}`, {
      method: "GET",
      signal,
    });
    const detail = mapWorkload(raw);
    const workloadScope = { orgId: raw.org_id, projectId: raw.project_id };
    const [events, logs, progress, leases] = await Promise.all([
      // Also best-effort. The timeline is context; losing it should cost the
      // timeline, not the page.
      this.request<RawEvent[]>(
        this.scope(`/events?entity_type=workload&entity_id=${encodeURIComponent(id)}`, workloadScope),
        {
          method: "GET",
          signal,
        },
      ).catch((error: unknown) => {
        if (signal?.aborted) throw error;
        return [] as RawEvent[];
      }),
      // Best-effort, and slow on purpose: this is forwarded to the worker. It
      // used to share the 3.5s control-plane budget *and* sit in this
      // `Promise.all` with no fallback, so one slow log fetch blanked a page
      // whose workload, events and progress had all answered in under half a
      // second. An empty detail page reads as "this workload does not exist".
      this.requestText(`/workloads/${encodeURIComponent(id)}/logs`, WORKER_FORWARDED_TIMEOUT_MS, signal).catch(
        (error: unknown) => {
          if (signal?.aborted) throw error;
          return `logs unavailable: ${error instanceof Error ? error.message : "unknown error"}`;
        },
      ),
      // Progress is best-effort: an older control plane without the endpoint must not break the detail page.
      this.request<RawWorkloadProgress>(`/workloads/${encodeURIComponent(id)}/progress`, {
        method: "GET",
        signal,
      }).catch((error: unknown) => {
        if (signal?.aborted) throw error;
        return undefined;
      }),
      // Allocation comes from this workload's own leases. It used to be hardcoded
      // to [], so the panel rendered nothing against a live server while the mock
      // data populated it — the page looked implemented and was not.
      this.request<RawLease[]>(this.scope(`/leases?workload_id=${encodeURIComponent(id)}`, workloadScope), {
        method: "GET",
        signal,
      }).catch((error: unknown) => {
        if (signal?.aborted) throw error;
        return [];
      }),
    ]);
    detail.allocation = leases.map((lease) => ({
      host: lease.cell_id,
      rack: lease.state,
      resource: describeResources(lease.resources.per_replica, lease.resources.replicas),
      state: leaseAllocationState(lease.state),
    }));
    detail.timeline = events.map((event): TimelineEvent => {
      const payload = (event as { payload?: Record<string, unknown> }).payload ?? {};
      const elapsed = payload["dispatch_elapsed_s"];
      const stage = typeof elapsed === "number" ? `dispatch took ${elapsed < 1 ? "<1" : Math.round(elapsed)}s` : "";
      return {
        id: event.id,
        timestamp: event.timestamp,
        title: event.action,
        detail: `${event.previous_state ?? "new"} -> ${event.new_state ?? "recorded"}${stage ? ` (${stage})` : ""}`,
        severity: event.action.includes("failed") ? "critical" : "info",
        actor: event.actor,
      };
    });
    detail.logs = logs
      .split("\n")
      .filter(Boolean)
      .map((line, index): LogEntry => ({ id: `${id}-log-${index}`, ...parseLogLine(line) }));
    if (progress) {
      detail.progress = mapProgressPercent(progress);
      detail.queueReason ??= progress.failure_reason ?? progress.scheduling_reason ?? undefined;
      detail.phase = progress.phase;
    }
    return detail;
  }

  /**
   * Why one workload is or is not currently placeable.
   *
   * `scheduling_reason` is one flat sentence, and it is what the dashboard has
   * always shown. This is the per-cell detail behind it: which cells were
   * considered, which constraint each failed, and whether any are still being
   * built -- the difference between "waiting for capacity" and "can never be
   * placed", which the flat reason cannot express.
   */
  async explainWorkload(
    id: string,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<WorkloadExplanation> {
    const requestScope = this.scopeSnapshot(scope);
    void requestScope;
    return this.request<WorkloadExplanation>(`/workloads/${encodeURIComponent(id)}/explain`, { signal });
  }

  /**
   * What one run used and was charged, totalled by the control plane.
   *
   * Computed there rather than here: the honest answer needs pool unit prices,
   * which are a fleet read, so a browser assembling it would either be wrong or
   * be limited to fleet administrators. It is authorized as a read of the
   * workload, which is the permission the question implies.
   */
  async getWorkloadCost(id: string, scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<WorkloadSpend> {
    void this.scopeSnapshot(scope);
    const raw = await this.request<RawWorkloadSpend>(`/workloads/${encodeURIComponent(id)}/cost`, { signal });
    return {
      workloadId: raw.workload_id,
      state: raw.state,
      acceleratorHours: raw.accelerator_hours ?? {},
      settled: raw.settled,
      priced: raw.priced,
      reserved: raw.reserved,
      leases: raw.leases,
      shape: raw.shape ?? undefined,
      capacityClass: raw.capacity_class ?? undefined,
      onDemandHourly: raw.on_demand_hourly ?? undefined,
      onDemandSource: raw.on_demand_source,
    };
  }

  /**
   * Live process detail for a running attempt.
   *
   * Answers "it says RUNNING and nothing is happening", which no other view
   * does: the control group and pids are what a diagnostic needs to attach to.
   */
  async workloadProcessStatus(
    id: string,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<WorkloadProcessStatus> {
    const requestScope = this.scopeSnapshot(scope);
    void requestScope;
    return this.request<WorkloadProcessStatus>(`/workloads/${encodeURIComponent(id)}/process-status`, { signal });
  }

  /**
   * Each pool's warm floor and whether it is currently held.
   *
   * Read beside billing. Warm capacity is deliberately idle, and without the
   * pairing the only safe reaction to an unexplained idle figure is to delete
   * the warm floor -- undoing the fix that put it there.
   */
  /** Provisioning latency per pool, from the samples the scheduler ranks on. */
  async getLaunchLatency(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<LaunchLatencyReport> {
    if (this.forceDemo) {
      throw new ApiError("Launch latency is unavailable in demo mode; no live readings are requested.");
    }
    const requestScope = this.scopeSnapshot(scope);
    void requestScope;
    return this.request<LaunchLatencyReport>("/capacity/launch-latency", { signal });
  }

  async getWarmth(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<WarmthReport> {
    if (this.forceDemo) {
      throw new ApiError("Warm capacity is unavailable in demo mode; no live readings are requested.");
    }
    const requestScope = this.scopeSnapshot(scope);
    void requestScope;
    return this.request<WarmthReport>("/capacity/warmth", { signal });
  }

  /** Accrued and idle cell cost across the fleet. */
  async getBilling(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<BillingReport> {
    if (this.forceDemo) {
      throw new ApiError("Fleet billing is unavailable in demo mode; no live readings are requested.");
    }
    const requestScope = this.scopeSnapshot(scope);
    void requestScope;
    return this.request<BillingReport>("/capacity/billing", { signal });
  }

  /**
   * Provider TPU quota, live usage, and free balance per zone and type.
   *
   * The numbers `eyvan autoscale generate` has always read, for the person
   * choosing where to run. Deliberately not merged into any other panel: a
   * quota ceiling is a property of the cloud project, not of this fleet's
   * cells, and the two disagree constantly.
   *
   * `refresh` re-reads the provider instead of reusing a cached reading. It is
   * an explicit action because a read costs a burst of provider calls, and
   * because the snapshot already states its own age.
   */
  async getTpuQuota(
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
    options: { project?: string; refresh?: boolean } = {},
  ): Promise<TpuQuotaSnapshot> {
    // Demo data has no provider reading. Refuse instead of mixing live quota
    // (including an explicit refresh) or fabricated zero usage into the sample.
    if (this.forceDemo) {
      throw new ApiError("Provider quota is unavailable in demo mode; no live readings are requested.");
    }
    const requestScope = this.scopeSnapshot(scope);
    void requestScope;
    const query = new URLSearchParams();
    if (options.project) query.set("project", options.project);
    if (options.refresh) query.set("refresh", "true");
    const suffix = query.size ? `?${query.toString()}` : "";
    return this.request<TpuQuotaSnapshot>(`/capacity/quota${suffix}`, { signal }, PROVIDER_READ_TIMEOUT_MS);
  }

  async listCells(
    knownLeases?: CapacityLease[],
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<Cell[]> {
    if (this.forceDemo) return cloneMock(mockCells);
    const requestScope = this.scopeSnapshot(scope);
    const [raw, leases] = knownLeases
      ? [await this.request<RawCell[]>("/cells", { method: "GET", signal }), knownLeases]
      : await Promise.all([
          this.request<RawCell[]>("/cells", { method: "GET", signal }),
          this.listLeases(requestScope, signal, true),
        ]);
    // A deleted cell is capacity that no longer exists. The control plane keeps
    // the record for audit and replay, but listing it as fleet inventory means
    // one churny afternoon leaves a Fleet page that is mostly tombstones -- and
    // every one of them counted against the health summary.
    return raw
      .filter((cell) => cell.state !== "deleted")
      .map((cell) => {
        const activeLeases = leases.filter(
          (lease) => lease.cell === cell.id && ["active", "activating"].includes(lease.state),
        );
        const resources = resourceRows(cell.capacity).map((item) => ({
          ...item,
          used: activeLeases.reduce((sum, lease) => sum + (lease.resourceTotals?.[item.name] ?? 0), 0),
        }));
        const utilization = Math.round(
          Math.max(0, ...resources.map((resource) => (resource.total ? (resource.used / resource.total) * 100 : 0))),
        );
        return {
          id: cell.id,
          name: cell.id,
          capacityClass: cell.capacity_class as Cell["capacityClass"],
          backend: `${cell.provider}${cell.provider_project ? ` / ${cell.provider_project}` : ""}`,
          providerAccount: cell.provider_account ?? undefined,
          region: cell.region,
          zone: cell.zone ?? "-",
          status: cellStatus(cell.state),
          state: cell.state,
          // Surfaced where the fleet page decides something needs attention:
          // the code names the class of problem, the detail carries the
          // provider's own words, so the page answers "why" without a log.
          unavailableReason: cell.unavailable_reason ?? undefined,
          unavailableDetail: cell.unavailable_detail ?? undefined,
          availability: cell.availability
            ? {
                cooldownUntil: cell.availability.cooldown_until ?? undefined,
                consecutiveFailures: cell.availability.consecutive_failures ?? 0,
                interruptions: cell.availability.interruptions ?? 0,
              }
            : undefined,
          utilization,
          activeWorkloads: new Set(activeLeases.map((lease) => lease.workloadId)).size,
          // Topology is the durable placement/failure-domain contract. The
          // accelerator type is a resource model, not a topology, and presenting
          // it here made distinct slices look interchangeable.
          topology: describeTopology(cell.topology),
          resources,
          lastHeartbeat: cell.last_heartbeat_at ?? new Date(0).toISOString(),
        };
      });
  }

  async listLeases(
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
    includeFleetMetadata = true,
  ): Promise<CapacityLease[]> {
    if (this.forceDemo) return cloneMock(mockLeases);
    const requestScope = this.scopeSnapshot(scope);
    // Cell metadata is fleet-wide. Project readers must not probe that endpoint
    // merely to decorate a scoped lease with an optional capacity class.
    const cellsPromise = includeFleetMetadata
      ? this.request<RawCell[]>("/cells", { method: "GET", signal })
      : Promise.resolve([] as RawCell[]);
    const [raw, cells] = await Promise.all([
      this.request<RawLease[]>(this.scope("/leases", requestScope), { method: "GET", signal }),
      cellsPromise,
    ]);
    const cellClasses = new Map(cells.map((cell) => [cell.id, cell.capacity_class as CapacityLease["class"]]));
    return raw.map((lease) => ({
      id: lease.id,
      workloadId: lease.workload_id,
      owner: lease.project_id,
      cell: lease.cell_id,
      resource: lease.resources.per_replica.tpu ? "TPU" : lease.resources.per_replica.gpu ? "GPU" : "CPU",
      amount: `${lease.resources.replicas} x ${lease.resources.per_replica.tpu || lease.resources.per_replica.gpu || lease.resources.per_replica.cpu}`,
      state:
        lease.state === "pending"
          ? "activating"
          : lease.state === "released" || lease.state === "expired"
            ? "expired"
            : lease.state === "quarantined"
              ? "revoking"
              : (lease.state as CapacityLease["state"]),
      expiresAt: lease.expires_at,
      spendPerHour: lease.hourly_cost,
      resourceTotals: Object.fromEntries(
        resourceRows(lease.resources.per_replica).map((resource) => [
          resource.name,
          resource.total * lease.resources.replicas,
        ]),
      ),
      class: cellClasses.get(lease.cell_id),
    }));
  }

  /**
   * Recent fleet-wide capacity history, newest first.
   *
   * The fleet ledger rather than the project one: a cell's events carry no
   * org or project, because capacity belongs to the fleet and not to whoever
   * happens to be running on it. A project-scoped read returns none of them,
   * which is why this page was previously unable to show what a zone did.
   */
  async listCapacityEvents(limit = 200, signal?: AbortSignal): Promise<AuditEvent[]> {
    if (this.forceDemo) return cloneMock(mockAuditEvents).filter((event) => event.category === "capacity");
    const raw = await this.request<RawEvent[]>(`/fleet/events?entity_type=cell&limit=${limit}`, { signal });
    return raw
      .map((event) => ({
        id: event.id,
        sequence: event.sequence,
        timestamp: event.timestamp,
        action: event.action,
        actor: event.actor,
        target: event.entity_id,
        category: "capacity" as const,
        severity: event.action.includes("failed") ? ("critical" as const) : ("info" as const),
        detail: `${event.previous_state ?? "new"} -> ${event.new_state ?? "recorded"}`,
      }))
      .sort((left, right) => (right.sequence ?? 0) - (left.sequence ?? 0));
  }

  async listBudgets(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<Budget[]> {
    if (this.forceDemo) return cloneMock(mockBudgets);
    const requestScope = this.scopeSnapshot(scope);
    let raw: RawBudget[];
    try {
      raw = [
        await this.request<RawBudget>(
          `/budgets/${encodeURIComponent(requestScope.orgId)}/${encodeURIComponent(requestScope.projectId)}`,
          { method: "GET", signal },
        ),
      ];
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) raw = [];
      else throw error;
    }
    const forecasts = await Promise.all(
      raw.map(async (budget) => {
        try {
          const status = await this.request<RawBudgetStatus>(
            `/budgets/${encodeURIComponent(budget.org_id)}/${encodeURIComponent(budget.project_id)}/status`,
            { method: "GET", signal },
          );
          return status.forecast ?? undefined;
        } catch (error) {
          // A cancellation or authority denial must abort the coherent snapshot;
          // only a genuinely unavailable forecast is optional decoration.
          if (signal?.aborted || (error instanceof ApiError && error.status === 403)) throw error;
          return undefined;
        }
      }),
    );
    return raw.map((budget, index) => ({
      id: `${budget.org_id}/${budget.project_id}`,
      owner: budget.project_id,
      scope: "project",
      spent: budget.spent,
      limit: budget.limit,
      forecast: forecasts[index],
      period: budget.period,
      enforcement: budget.hard_limit ? "hard" : "soft",
    }));
  }

  async listPolicies(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<SchedulingPolicy[]> {
    if (this.forceDemo) return cloneMock(mockPolicies);
    const requestScope = this.scopeSnapshot(scope);
    try {
      const raw = await this.request<RawPolicy>(
        `/policies/${encodeURIComponent(requestScope.orgId)}/${encodeURIComponent(requestScope.projectId)}`,
        { method: "GET", signal },
      );
      return [mapPolicy(raw)];
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) return [];
      throw error;
    }
  }

  async listAuditEvents(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<AuditEvent[]> {
    if (this.forceDemo) return cloneMock(mockAuditEvents);
    const requestScope = this.scopeSnapshot(scope);
    // Project audit is a project read. Calling /fleet/events first and degrading
    // its inevitable 403 made every non-fleet reader generate a forbidden call
    // and then presented an empty ledger as if the project had no history.
    const raw = await this.request<RawEvent[]>(this.scope("/events?limit=1000", requestScope), {
      method: "GET",
      signal,
    });
    return raw.map((event) => ({
      id: event.id,
      sequence: event.sequence,
      timestamp: event.timestamp,
      action: event.action,
      actor: event.actor,
      target: event.entity_id,
      category:
        event.entity_type === "workload"
          ? "workload"
          : ["cell", "lease"].includes(event.entity_type)
            ? "capacity"
            : ["policy", "quota"].includes(event.entity_type)
              ? "policy"
              : event.entity_type === "budget"
                ? "budget"
                : "system",
      severity: event.action.includes("failed") ? "critical" : "info",
      detail: `${event.previous_state ?? "new"} -> ${event.new_state ?? "recorded"}`,
    }));
  }

  /**
   * Every shape this scope may submit to, across applied fleets.
   *
   * Readable with `submit` alone, unlike the pool records behind it, and that
   * is the point: choosing what to run on should not require permission to
   * manage the fleet. It is also the only honest source for the form's
   * choices -- a hard-coded list of accelerator types is a list of shapes some
   * other fleet had.
   */
  async listAvailableShapes(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<AvailableShape[]> {
    if (this.forceDemo) return cloneMock(mockAvailableShapes);
    const requestScope = this.scopeSnapshot(scope);
    const payload = await this.request<{ pools: RawAvailablePool[] }>(this.scope("/capacity", requestScope), {
      signal,
    });
    return payload.pools
      .filter((pool) => pool.enabled)
      .map((pool) => ({
        fleet: pool.fleet,
        pool: pool.pool,
        shape: pool.accelerator_type,
        capacityClass: pool.capacity_class as AvailableShape["capacityClass"],
        region: pool.region,
        zone: pool.zone ?? undefined,
      }));
  }

  /**
   * Submit one command against a fleet shape, the way `eyvan run` does.
   *
   * The pool supplies the infrastructure contract: CPU, memory, accelerator
   * count, topology and provider all come from the node type this lands on.
   * This used to build a spec by hand against the advanced `/workloads` route,
   * which meant the browser inventing every one of those -- it sent
   * `provider: "gcp"` on every submission, which is a hard constraint that
   * makes the job unplaceable on any other cloud, and a `required_topology` of
   * `slice: "single-slice"` whose values no pool declares, so the default
   * submission was refused with TOPOLOGY_MISMATCH against a value the user
   * never typed.
   */
  async submitWorkload(
    input: SubmitWorkloadInput,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<WorkloadDetail> {
    if (this.forceDemo) return createMockWorkload(input);
    const requestScope = this.scopeSnapshot(scope);
    const payload = {
      org_id: requestScope.orgId,
      project_id: requestScope.projectId,
      command: input.command,
      pods: input.pods,
      zone: input.zone,
      name: input.name,
      environment: input.environment,
      metadata: input.labels,
      priority_band: input.priority,
      estimated_duration_seconds: input.estimatedDurationSeconds,
      checkpointable: input.checkpointable,
      checkpoint_hook: input.checkpointHook,
      checkpoint_output_uri: input.checkpointUri,
    };
    const raw = await this.request<RawWorkload>(
      `/fleets/${encodeURIComponent(input.fleet)}/node-types/${encodeURIComponent(input.shape)}/workloads`,
      {
        method: "POST",
        headers: { "Idempotency-Key": globalThis.crypto?.randomUUID?.() ?? `dashboard-${Date.now()}` },
        body: JSON.stringify(payload),
        signal,
      },
    );
    return mapWorkload(raw);
  }

  async workloadAction(
    id: string,
    action: "cancel" | "checkpoint" | "retry" | "kick",
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<WorkloadDetail> {
    const requestScope = this.scopeSnapshot(scope);
    if (this.forceDemo) {
      const mock = getMockWorkload(id);
      if (!mock) throw new ApiError(`Workload ${id} was not found`, 404);
      return {
        ...cloneMock(mock),
        // A kick ends this attempt and requeues the job, so it reads as queued
        // exactly like a retry -- the difference is which budget pays for it,
        // and that is the control plane's to record rather than the mock's.
        status: action === "cancel" ? "preempting" : action === "checkpoint" ? "checkpointing" : "queued",
      };
    }
    const raw = await this.request<RawWorkload>(`/workloads/${encodeURIComponent(id)}/${action}`, {
      method: "POST",
      headers: { "Idempotency-Key": globalThis.crypto?.randomUUID?.() ?? `dashboard-${Date.now()}` },
      signal,
    });
    if (raw.org_id !== requestScope.orgId || raw.project_id !== requestScope.projectId) {
      throw new ApiError("Workload mutation returned a resource outside the immutable requested scope", 409);
    }
    return mapWorkload(raw);
  }

  /**
   * Stop new placement on one cell and ask its agent to drain.
   *
   * Returns the state the control plane moved it to rather than a remapped
   * `Cell`: a cell's rendered shape is a join against its active leases, and
   * rebuilding that from one record would put a figure on the page that the
   * next refresh disagrees with. The caller refreshes inventory instead.
   *
   * Not offered in demo mode. A drain that appears to work against a dataset
   * with no fleet behind it teaches the gesture without its consequence.
   */
  async drainCell(cellId: string, signal?: AbortSignal): Promise<{ id: string; state: string }> {
    if (this.forceDemo) throw new ApiError("Demo mode has no fleet to drain", 400);
    const raw = await this.request<RawCell>(`/cells/${encodeURIComponent(cellId)}/drain`, {
      method: "POST",
      headers: { "Idempotency-Key": globalThis.crypto?.randomUUID?.() ?? `dashboard-${Date.now()}` },
      signal,
    });
    return { id: raw.id, state: raw.state };
  }

  /**
   * The pools this project may submit to, as registered rather than as running.
   *
   * The fleet page could only ever show cells -- what is provisioned right now
   * -- so a fleet with twenty pools and nothing running looked like an empty
   * fleet, and there was no way to see what had been applied at all.
   */
  async listCapacityPools(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<CapacityPoolSummary[]> {
    if (this.forceDemo) return [];
    const raw = await this.request<Record<string, unknown>[]>(
      `/capacity-pools?org_id=${encodeURIComponent(scope.orgId)}&project_id=${encodeURIComponent(scope.projectId)}`,
      { signal },
    );
    return (Array.isArray(raw) ? raw : []).map((pool) => ({
      id: String(pool.id ?? ""),
      fleetName: String(pool.fleet_name ?? ""),
      acceleratorType: (pool.accelerator_type as string | null) ?? null,
      capacityClass: String(pool.capacity_class ?? ""),
      zone: (pool.zone as string | null) ?? null,
      region: (pool.region as string | null) ?? null,
      maxCells: Number(pool.max_cells ?? 0),
      minCells: Number(pool.min_cells ?? 0),
      enabled: pool.enabled !== false,
      provider: typeof pool.provider === "string" && pool.provider ? pool.provider : null,
      providerProject:
        typeof pool.provider_project === "string" && pool.provider_project ? pool.provider_project : null,
      providerAccount:
        typeof pool.provider_account === "string" && pool.provider_account ? pool.provider_account : null,
    }));
  }

  /**
   * Who in this project used what, and what anybody can say it cost.
   *
   * The cost is zero wherever a pool declares no unit price, and the caller has
   * to be able to tell that from free -- hence `unpricedLeases` travelling
   * beside it rather than being folded in.
   */
  async getProjectUsage(scope: AuthScope = this.activeScope, signal?: AbortSignal): Promise<ProjectUsage> {
    // Demo mode is fully self-contained like its sibling reads: no real
    // network call, an empty (honest) usage answer instead of an error that
    // blanks the Billing page.
    if (this.forceDemo) {
      return { orgId: scope.orgId, projectId: scope.projectId, subjects: [] };
    }
    const raw = await this.request<Record<string, unknown>>(
      `/usage?org_id=${encodeURIComponent(scope.orgId)}&project_id=${encodeURIComponent(scope.projectId)}`,
      { signal },
    );
    const subjects = Array.isArray(raw.subjects) ? (raw.subjects as Record<string, unknown>[]) : [];
    return {
      orgId: String(raw.org_id ?? scope.orgId),
      projectId: String(raw.project_id ?? scope.projectId),
      subjects: subjects.map((entry) => ({
        subject: String(entry.subject ?? "unknown"),
        workloads: Number(entry.workloads ?? 0),
        settledWorkloads: Number(entry.settled_workloads ?? 0),
        leases: Number(entry.leases ?? 0),
        acceleratorHours: Number(entry.accelerator_hours ?? 0),
        runningSeconds: Number(entry.running_seconds ?? 0),
        settledCost: Number(entry.settled_cost ?? 0),
        unpricedLeases: Number(entry.unpriced_leases ?? 0),
      })),
    };
  }

  /**
   * Drain a whole fleet to zero.
   *
   * The route is DELETE and its name is "disable": it turns every pool off and
   * sets its ceiling to zero so the autoscaler releases what is running. The
   * fleet's definition survives, which is why the UI offers this as stopping a
   * fleet rather than deleting one -- re-applying the same config brings it
   * back without re-generating anything.
   */
  async disableFleet(fleetName: string, signal?: AbortSignal): Promise<number> {
    if (this.forceDemo) throw new ApiError("Demo mode has no fleet to stop", 400);
    const pools = await this.request<unknown[]>(`/fleets/${encodeURIComponent(fleetName)}`, {
      method: "DELETE",
      headers: { "Idempotency-Key": globalThis.crypto?.randomUUID?.() ?? `dashboard-${Date.now()}` },
      signal,
    });
    return Array.isArray(pools) ? pools.length : 0;
  }

  async updatePolicy(
    policy: SchedulingPolicy,
    preemptionEnabled: boolean,
    scope: AuthScope = this.activeScope,
    signal?: AbortSignal,
  ): Promise<SchedulingPolicy> {
    if (this.forceDemo) return { ...cloneMock(policy), preemptionEnabled, updatedAt: new Date().toISOString() };
    const requestScope = this.scopeSnapshot(scope);
    const configuration = policy.configuration;
    if (!configuration) throw new ApiError(`Policy ${policy.id} is missing its editable configuration`);
    const raw = await this.request<RawPolicy>(
      `/policies/${encodeURIComponent(requestScope.orgId)}/${encodeURIComponent(requestScope.projectId)}`,
      {
        method: "PUT",
        headers: { "Idempotency-Key": globalThis.crypto?.randomUUID?.() ?? `dashboard-${Date.now()}` },
        body: JSON.stringify({
          allowed_providers: configuration.allowedProviders,
          allowed_regions: configuration.allowedRegions,
          allowed_capacity_classes: configuration.allowedCapacityClasses,
          denied_accelerator_types: configuration.deniedAcceleratorTypes,
          max_priority: configuration.maxPriority,
          preemption_enabled: preemptionEnabled,
        }),
        signal,
      },
    );
    if (raw.org_id !== requestScope.orgId || raw.project_id !== requestScope.projectId) {
      throw new ApiError("Policy mutation returned a resource outside the immutable requested scope", 409);
    }
    return mapPolicy(raw);
  }
}

export const apiClient = new EyvanApiClient();
