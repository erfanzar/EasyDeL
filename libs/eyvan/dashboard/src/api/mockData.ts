// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import type {
  AttachmentInput,
  AttachmentJob,
  AttachmentPlan,
  AuditEvent,
  AvailableShape,
  Budget,
  CapacityLease,
  Cell,
  CollaborationFeedResponse,
  CollaborationNote,
  CollaborationNoteInput,
  DashboardData,
  DiscoveredTpus,
  DurableRoleGrant,
  ExperimentCreateInput,
  ExperimentLeaderboardEntry,
  ExperimentLeaderboardResponse,
  ExperimentRecord,
  ExperimentUpdatePatch,
  LogEntry,
  ManagedArtifact,
  ProjectRecord,
  ProjectRegisterInput,
  RoleGrantIssueInput,
  RunCollaborationOverlay,
  RunCompareInput,
  RunCompareMetricSummaryRow,
  RunCompareResponse,
  RunCompareSeriesData,
  RunDetail,
  BlobRef,
  RunRollout,
  RunRolloutsOptions,
  RunRolloutsResponse,
  RunCompletion,
  RunCompletionsOptions,
  RunCompletionsResponse,
  CompletionStatsResponse,
  RunSharingPatch,
  RunTableOptions,
  RunTableResponse,
  RunMediaOptions,
  RunMediaResponse,
  RunHistogramsOptions,
  RunHistogramsResponse,
  RunEpisodesOptions,
  RunEpisodesResponse,
  RunEpisode,
  EpisodeCurveOptions,
  EpisodeCurveResponse,
  EpisodeBreakdownOptions,
  EpisodeBreakdownResponse,
  RunSeries,
  RunSeriesResponse,
  RunSummary,
  SchedulingPolicy,
  StorageRegionCheckInput,
  StorageRegionCheckResult,
  SubmitWorkloadInput,
  TimelineEvent,
  WebhookCreateInput,
  WebhookSubscription,
  WorkloadDetail,
  WorkloadSummary,
} from "../types";

export function mockAttachmentPlan(input: AttachmentInput): AttachmentPlan {
  return {
    dry_run: true,
    fleet: input.fleet || "tpu-demo",
    fleet_path: "~/.eyvan/fleets/tpu-demo.yaml",
    fleet_yaml: "",
    endpoints: ["http://127.0.0.1:18001", "http://127.0.0.1:18002"],
    actions: [
      { step: "install", argv: [], summary: "Install the Eyvan worker on 2 hosts", mutates: true, stdin: true },
      { step: "apply", argv: [], summary: "Register the existing TPU as a fleet", mutates: true, stdin: false },
    ],
    limitations: ["Demo only: no machine is contacted or changed."],
    summary: { shape: "v5p-8", hosts: 2, capacity_class: "spot", endpoint_mode: "tunnel" },
  };
}

export const mockAttachmentJobs: AttachmentJob[] = [];

export const mockDiscoveredTpus: DiscoveredTpus = {
  discovered_at: Date.now() / 1000,
  projects: ["demo-research", "demo-lab"],
  tpus: [
    {
      fleet: "tpu-train-a-1a2b3c4d",
      project: "demo-research",
      zone: "us-central1-a",
      name: "train-a",
      node_id: "1",
      accelerator_type: "v5p-8",
      capacity_class: "on_demand",
      state: "READY",
      hosts: 1,
      connected: true,
      cell_state: "ready",
      last_error: null,
      next_attempt_at: null,
    },
    {
      fleet: "tpu-spot-b-5e6f7a8b",
      project: "demo-lab",
      zone: "us-east5-a",
      name: "spot-b",
      node_id: "2",
      accelerator_type: "v6e-16",
      capacity_class: "spot",
      state: "READY",
      hosts: 4,
      connected: false,
    },
  ],
  skipped: {},
};

export function createMockAttachment(input: AttachmentInput): AttachmentJob {
  const fleet = input.fleet || "tpu-demo";
  const job: AttachmentJob = {
    id: `demo-${Date.now()}`,
    target: input.target,
    status: "succeeded",
    progress: ["Demo: worker installation simulated", "Demo: fleet registration simulated"],
    result: { fleet, ready: true, cells: [], submit: `init(fleet="${fleet}")`, limitations: [] },
    error: null,
    started_at: new Date().toISOString(),
    finished_at: new Date().toISOString(),
  };
  mockAttachmentJobs.unshift(job);
  return cloneMock(job);
}

export const mockWorkloads: WorkloadSummary[] = [
  {
    id: "wrk-9f31",
    name: "llama-70b-sft-v4",
    org: "demo",
    project: "foundation-models",
    user: "maya.chen",
    status: "running",
    priority: "batch",
    resources: {
      cpu: 192,
      memoryGb: 1536,
      accelerator: "TPU v5p",
      acceleratorCount: 64,
      replicas: 8,
      topology: "8x NVLink / same region",
    },
    cell: "iad-gpu-02",
    submittedAt: "2026-07-10T05:42:00Z",
    startedAt: "2026-07-10T06:01:00Z",
    progress: 68,
    budgetSpent: 4820,
    budgetLimit: 9200,
  },
  {
    id: "wrk-a284",
    name: "reward-model-sweep",
    org: "demo",
    project: "alignment",
    user: "samir.patel",
    status: "queued",
    priority: "batch",
    resources: {
      cpu: 96,
      memoryGb: 768,
      accelerator: "TPU v6e",
      acceleratorCount: 32,
      replicas: 4,
      topology: "same zone",
    },
    submittedAt: "2026-07-10T07:15:00Z",
    budgetSpent: 0,
    budgetLimit: 2800,
    queueReason: "Waiting for 4 contiguous TPU v6e hosts in us-east",
  },
  {
    id: "wrk-c711",
    name: "online-policy-eval",
    org: "demo",
    project: "evals",
    user: "nora.haddad",
    status: "provisioning",
    priority: "interactive",
    resources: {
      cpu: 48,
      memoryGb: 384,
      accelerator: "TPU v5e",
      acceleratorCount: 8,
      replicas: 8,
    },
    cell: "fra-gpu-01",
    submittedAt: "2026-07-10T08:03:00Z",
    progress: 18,
    budgetSpent: 42,
    budgetLimit: 950,
  },
  {
    id: "wrk-1bd0",
    name: "nightly-tokenization",
    org: "demo",
    project: "data-platform",
    user: "alex.kim",
    status: "completed",
    priority: "batch",
    resources: { cpu: 512, memoryGb: 2048, replicas: 64 },
    cell: "ord-cpu-03",
    submittedAt: "2026-07-09T23:00:00Z",
    startedAt: "2026-07-09T23:04:00Z",
    progress: 100,
    budgetSpent: 184,
    budgetLimit: 500,
  },
  {
    id: "wrk-834c",
    name: "vision-pretrain-2b",
    org: "demo",
    project: "multimodal",
    user: "irene.okafor",
    status: "checkpointing",
    priority: "batch",
    resources: {
      cpu: 128,
      memoryGb: 1024,
      accelerator: "TPU v5p",
      acceleratorCount: 64,
      replicas: 16,
      topology: "4x4 slice",
    },
    cell: "gcp-uscentral-tpu-01",
    submittedAt: "2026-07-09T19:32:00Z",
    startedAt: "2026-07-09T19:46:00Z",
    progress: 41,
    budgetSpent: 3390,
    budgetLimit: 7800,
  },
  {
    id: "wrk-57ee",
    name: "serve-canary-bench",
    org: "demo",
    project: "inference",
    user: "liam.jones",
    status: "failed",
    priority: "interactive",
    resources: {
      cpu: 32,
      memoryGb: 256,
      accelerator: "TPU v5p",
      acceleratorCount: 4,
      replicas: 4,
    },
    cell: "iad-gpu-01",
    submittedAt: "2026-07-10T06:56:00Z",
    startedAt: "2026-07-10T06:59:00Z",
    progress: 24,
    budgetSpent: 38,
    budgetLimit: 400,
  },
  {
    id: "wrk-2ae1",
    name: "safety-redteam-live",
    org: "demo",
    project: "safety",
    user: "zoe.martin",
    status: "running",
    priority: "production",
    resources: {
      cpu: 64,
      memoryGb: 512,
      accelerator: "TPU v5p",
      acceleratorCount: 8,
      replicas: 8,
      topology: "single slice",
    },
    cell: "iad-gpu-01",
    submittedAt: "2026-07-10T08:21:00Z",
    startedAt: "2026-07-10T08:22:00Z",
    progress: 52,
    budgetSpent: 622,
    budgetLimit: 1600,
  },
  {
    id: "wrk-d409",
    name: "embedding-refresh",
    org: "demo",
    project: "retrieval",
    user: "ravi.singh",
    status: "planning",
    priority: "batch",
    resources: {
      cpu: 128,
      memoryGb: 512,
      accelerator: "TPU v4",
      acceleratorCount: 16,
      replicas: 16,
    },
    submittedAt: "2026-07-10T08:28:00Z",
    progress: 5,
    budgetSpent: 0,
    budgetLimit: 700,
  },
  {
    id: "wrk-7c92",
    name: "code-model-distill",
    org: "demo",
    project: "foundation-models",
    user: "maya.chen",
    status: "preempting",
    priority: "batch",
    resources: {
      cpu: 96,
      memoryGb: 768,
      accelerator: "TPU v6e",
      acceleratorCount: 16,
      replicas: 4,
    },
    cell: "iad-gpu-01",
    submittedAt: "2026-07-09T14:10:00Z",
    startedAt: "2026-07-09T14:26:00Z",
    progress: 77,
    budgetSpent: 1910,
    budgetLimit: 2400,
  },
  {
    id: "wrk-316f",
    name: "dataset-quality-map",
    org: "demo",
    project: "data-platform",
    user: "alex.kim",
    status: "running",
    priority: "batch",
    resources: { cpu: 256, memoryGb: 1024, replicas: 32 },
    cell: "ord-cpu-03",
    submittedAt: "2026-07-10T04:10:00Z",
    startedAt: "2026-07-10T04:12:00Z",
    progress: 83,
    budgetSpent: 92,
    budgetLimit: 250,
  },
];

const sharedTimeline: TimelineEvent[] = [
  {
    id: "evt-1",
    timestamp: "2026-07-10T06:01:32Z",
    title: "Gang committed",
    detail: "All 8 workers passed the launch barrier in iad-gpu-02.",
    severity: "success",
    actor: "cell-scheduler",
  },
  {
    id: "evt-2",
    timestamp: "2026-07-10T05:59:18Z",
    title: "Capacity lease activated",
    detail: "Lease lse-8c14 activated with 64 TPU v5p accelerators.",
    severity: "info",
    actor: "capacity-controller",
  },
  {
    id: "evt-3",
    timestamp: "2026-07-10T05:44:05Z",
    title: "Placement selected",
    detail: "iad-gpu-02 won on locality, scarcity, and expected cost.",
    severity: "info",
    actor: "global-scheduler",
  },
  {
    id: "evt-4",
    timestamp: "2026-07-10T05:42:00Z",
    title: "Workload admitted",
    detail: "Standard-band admission passed budget and fair-share checks.",
    severity: "success",
    actor: "admission-controller",
  },
];

const sharedLogs: LogEntry[] = [
  {
    id: "log-1",
    timestamp: "2026-07-10T08:35:44.102Z",
    level: "INFO",
    source: "trainer",
    message: "step=4820 loss=1.842 grad_norm=0.918 tokens_per_second=184220",
  },
  {
    id: "log-2",
    timestamp: "2026-07-10T08:35:32.731Z",
    level: "INFO",
    source: "checkpoint",
    message: "async checkpoint 4800 committed to gs://eyvan-checkpoints/llama-70b-sft-v4",
  },
  {
    id: "log-3",
    timestamp: "2026-07-10T08:34:51.443Z",
    level: "WARN",
    source: "input-pipeline",
    message: "shard 119 read latency 842ms exceeded 750ms target",
  },
  {
    id: "log-4",
    timestamp: "2026-07-10T08:34:42.209Z",
    level: "INFO",
    source: "trainer",
    message: "step=4810 loss=1.849 grad_norm=0.925 tokens_per_second=183904",
  },
  {
    id: "log-5",
    timestamp: "2026-07-10T08:33:59.809Z",
    level: "DEBUG",
    source: "scheduler",
    message: "lease heartbeat accepted generation=12 expires_in=119s",
  },
];

function detailFor(workload: WorkloadSummary): WorkloadDetail {
  const accelerator = workload.resources.accelerator ?? "CPU";
  const allocationCount = Math.min(workload.resources.replicas, 8);
  return {
    ...workload,
    command: "python -m easydel.scripts.elarge --config configs/llama_70b_sft.yaml",
    checkpointUri: "gs://eyvan-checkpoints/llama-70b-sft-v4/step-4800",
    leaseId: workload.cell ? `lse-${workload.id.slice(4)}` : undefined,
    labels: {
      environment: "production",
      cost_center: workload.project,
      checkpointable: "true",
    },
    allocation: Array.from({ length: allocationCount }, (_, index) => ({
      host: `${workload.cell ?? "pending"}-worker-${String(index + 1).padStart(2, "0")}`,
      rack: `rack-${Math.floor(index / 4) + 3}`,
      resource: `${Math.max(1, Math.floor((workload.resources.acceleratorCount ?? 0) / allocationCount))}x ${accelerator}`,
      state: workload.status === "provisioning" && index > 4 ? "starting" : "ready",
    })),
    timeline: sharedTimeline.map((event) => ({ ...event, id: `${workload.id}-${event.id}` })),
    logs: sharedLogs.map((entry) => ({ ...entry, id: `${workload.id}-${entry.id}` })),
  };
}

export const mockCells: Cell[] = [
  {
    id: "cell-iad-gpu-01",
    name: "iad-gpu-01",
    backend: "AWS EC2",
    region: "us-east",
    zone: "iad-a",
    status: "healthy",
    utilization: 86,
    activeWorkloads: 12,
    queuedWorkloads: 3,
    topology: "8 racks / EFA fabric",
    resources: [
      { name: "TPU v5p", used: 112, total: 128 },
      { name: "TPU v6e", used: 40, total: 64 },
      { name: "CPU", used: 1830, total: 2304 },
    ],
    lastHeartbeat: "2026-07-10T08:36:18Z",
  },
  {
    id: "cell-iad-gpu-02",
    name: "iad-gpu-02",
    backend: "Kubernetes",
    region: "us-east",
    zone: "iad-c",
    status: "healthy",
    utilization: 74,
    activeWorkloads: 8,
    queuedWorkloads: 1,
    topology: "4 TPU islands / NVLink",
    resources: [
      { name: "TPU v5p", used: 88, total: 128 },
      { name: "TPU v5e", used: 28, total: 32 },
      { name: "CPU", used: 970, total: 1536 },
    ],
    lastHeartbeat: "2026-07-10T08:36:17Z",
  },
  {
    id: "cell-fra-gpu-01",
    name: "fra-gpu-01",
    backend: "Kubernetes",
    region: "eu-central",
    zone: "fra-b",
    status: "degraded",
    utilization: 63,
    activeWorkloads: 6,
    queuedWorkloads: 2,
    topology: "6 racks / RoCE",
    resources: [
      { name: "TPU v5e", used: 54, total: 96 },
      { name: "TPU v4", used: 22, total: 32 },
      { name: "CPU", used: 620, total: 1024 },
    ],
    lastHeartbeat: "2026-07-10T08:36:02Z",
  },
  {
    id: "cell-gcp-tpu-01",
    name: "gcp-uscentral-tpu-01",
    backend: "GKE / TPU",
    region: "us-central",
    zone: "usc1-b",
    status: "healthy",
    utilization: 81,
    activeWorkloads: 5,
    queuedWorkloads: 1,
    topology: "v5p slices / 4x4-8x8",
    resources: [
      { name: "TPU v5p", used: 192, total: 256 },
      { name: "CPU", used: 410, total: 512 },
      { name: "RAM TB", used: 8, total: 12 },
    ],
    lastHeartbeat: "2026-07-10T08:36:13Z",
  },
  {
    id: "cell-ord-cpu-03",
    name: "ord-cpu-03",
    backend: "Bare metal",
    region: "us-central",
    zone: "ord-2",
    status: "maintenance",
    utilization: 46,
    activeWorkloads: 11,
    queuedWorkloads: 0,
    topology: "12 racks / 100GbE",
    resources: [
      { name: "CPU", used: 2040, total: 4608 },
      { name: "RAM TB", used: 18, total: 36 },
      { name: "NVMe TB", used: 220, total: 384 },
    ],
    lastHeartbeat: "2026-07-10T08:35:58Z",
  },
];

export const mockLeases: CapacityLease[] = [
  {
    id: "lse-9f31",
    workloadId: "wrk-9f31",
    owner: "foundation-models",
    cell: "iad-gpu-02",
    resource: "TPU v5p",
    amount: "64 accelerators / 8 hosts",
    class: "reserved",
    state: "active",
    expiresAt: "2026-07-10T10:36:00Z",
    spendPerHour: 724,
  },
  {
    id: "lse-2ae1",
    workloadId: "wrk-2ae1",
    owner: "safety",
    cell: "iad-gpu-01",
    resource: "TPU v5p",
    amount: "8 accelerators / 1 host",
    class: "on_demand",
    state: "active",
    expiresAt: "2026-07-10T09:02:00Z",
    spendPerHour: 91,
  },
  {
    id: "lse-c711",
    workloadId: "wrk-c711",
    owner: "evals",
    cell: "fra-gpu-01",
    resource: "TPU v5e",
    amount: "8 accelerators / 8 hosts",
    class: "spot",
    state: "activating",
    expiresAt: "2026-07-10T09:20:00Z",
    spendPerHour: 38,
  },
  {
    id: "lse-warm-8a",
    owner: "platform",
    cell: "iad-gpu-01",
    resource: "TPU v6e",
    amount: "16 accelerators / 2 hosts",
    class: "reserved",
    state: "revoking",
    expiresAt: "2026-07-10T08:45:00Z",
    spendPerHour: 104,
  },
];

export const mockBudgets: Budget[] = [
  {
    id: "bdg-org",
    owner: "Eyvan Research",
    scope: "organization",
    spent: 124380,
    limit: 180000,
    forecast: 168200,
    period: "July 2026",
    enforcement: "soft",
  },
  {
    id: "bdg-foundation",
    owner: "foundation-models",
    scope: "project",
    spent: 52740,
    limit: 68000,
    forecast: 73400,
    period: "July 2026",
    enforcement: "hard",
  },
  {
    id: "bdg-alignment",
    owner: "alignment",
    scope: "project",
    spent: 21980,
    limit: 42000,
    forecast: 36750,
    period: "July 2026",
    enforcement: "soft",
  },
  {
    id: "bdg-multimodal",
    owner: "multimodal",
    scope: "project",
    spent: 18640,
    limit: 30000,
    forecast: 31500,
    period: "July 2026",
    enforcement: "hard",
  },
  {
    id: "bdg-maya",
    owner: "maya.chen",
    scope: "user",
    spent: 11220,
    limit: 18000,
    forecast: 16200,
    period: "July 2026",
    enforcement: "soft",
  },
];

export const mockPolicies: SchedulingPolicy[] = [
  {
    id: "pol-priority",
    name: "Production priority bands",
    scope: "all cells",
    type: "priority",
    summary: "production > interactive > batch; aging after 4 hours",
    preemptionEnabled: true,
    crossRegionTransferPolicy: "allow",
    crossRegionAllowedBuckets: [],
    crossRegionAllowMultiregion: true,
    updatedAt: "2026-07-08T13:40:00Z",
  },
  {
    id: "pol-drf",
    name: "Hierarchical fair share",
    scope: "organization / project / user",
    type: "fairness",
    summary: "Weighted DRF with revocable borrowing and reservation accounting",
    preemptionEnabled: true,
    crossRegionTransferPolicy: "allow",
    crossRegionAllowedBuckets: [],
    crossRegionAllowMultiregion: true,
    updatedAt: "2026-07-07T09:22:00Z",
  },
  {
    id: "pol-preempt",
    name: "Whole-gang preemption",
    scope: "TPU and TPU pools",
    type: "preemption",
    summary: "Checkpoint grace 180s; max 2 preemptions per workload per hour",
    preemptionEnabled: true,
    crossRegionTransferPolicy: "warn",
    crossRegionAllowedBuckets: [],
    crossRegionAllowMultiregion: true,
    updatedAt: "2026-07-09T17:12:00Z",
  },
  {
    id: "pol-locality",
    name: "Dataset locality preference",
    scope: "training workloads",
    type: "placement",
    summary: "Prefer cached shards when transfer delta exceeds 90 seconds",
    preemptionEnabled: true,
    crossRegionTransferPolicy: "ban",
    crossRegionAllowedBuckets: ["example-shared-datasets"],
    crossRegionAllowMultiregion: true,
    updatedAt: "2026-07-06T11:04:00Z",
  },
  {
    id: "pol-carbon",
    name: "Low-carbon batch placement",
    scope: "batch band",
    type: "placement",
    summary: "Apply regional carbon score after deadline and data constraints",
    preemptionEnabled: false,
    crossRegionTransferPolicy: "allow",
    crossRegionAllowedBuckets: [],
    crossRegionAllowMultiregion: true,
    updatedAt: "2026-07-02T08:00:00Z",
  },
];

export const mockAuditEvents: AuditEvent[] = [
  {
    id: "aud-7712",
    timestamp: "2026-07-10T08:35:12Z",
    action: "lease.renewed",
    actor: "cell-scheduler/iad-gpu-02",
    target: "lse-9f31",
    category: "capacity",
    severity: "info",
    detail: "Lease generation advanced from 11 to 12 for 64 TPU v5p accelerators.",
    requestId: "req-e8d12c",
  },
  {
    id: "aud-7711",
    timestamp: "2026-07-10T08:31:40Z",
    action: "workload.preemption_started",
    actor: "global-scheduler",
    target: "wrk-7c92",
    category: "workload",
    severity: "warning",
    detail: "Batch gang selected as a complete victim for production workload wrk-2ae1.",
    requestId: "req-f9a203",
  },
  {
    id: "aud-7709",
    timestamp: "2026-07-10T08:28:03Z",
    action: "workload.submitted",
    actor: "ravi.singh",
    target: "wrk-d409",
    category: "workload",
    severity: "success",
    detail: "Embedding refresh admitted to the batch priority band.",
    requestId: "req-b61dd4",
  },
  {
    id: "aud-7706",
    timestamp: "2026-07-10T08:12:55Z",
    action: "cell.degraded",
    actor: "health-controller",
    target: "fra-gpu-01",
    category: "system",
    severity: "critical",
    detail: "Provision success for TPU v5e fell below 70% over the last 15 minutes.",
    requestId: "req-6cd18b",
  },
  {
    id: "aud-7704",
    timestamp: "2026-07-10T07:58:20Z",
    action: "budget.threshold_crossed",
    actor: "budget-controller",
    target: "foundation-models",
    category: "budget",
    severity: "warning",
    detail: "Monthly forecast reached 108% of the hard project budget.",
    requestId: "req-52c137",
  },
  {
    id: "aud-7698",
    timestamp: "2026-07-10T07:40:11Z",
    action: "policy.updated",
    actor: "admin@eyvan",
    target: "pol-preempt",
    category: "policy",
    severity: "info",
    detail: "Checkpoint grace increased from 120 to 180 seconds.",
    requestId: "req-43ce80",
  },
];

export const mockDashboardData: DashboardData = {
  health: { status: "demo", service: "eyvan" },
  overview: {
    queued: 18,
    running: 42,
    fleetUtilization: 74,
    monthlySpend: 124380,
    monthlyBudget: 180000,
    successRate: 97.8,
    queueTrend: [
      { label: "02:00", submitted: 7, admitted: 4 },
      { label: "03:00", submitted: 10, admitted: 6 },
      { label: "04:00", submitted: 8, admitted: 8 },
      { label: "05:00", submitted: 14, admitted: 7 },
      { label: "06:00", submitted: 12, admitted: 9 },
      { label: "07:00", submitted: 21, admitted: 11 },
      { label: "08:00", submitted: 18, admitted: 13 },
    ],
    priorityQueue: [
      { priority: "production", count: 1, oldestMinutes: 2 },
      { priority: "interactive", count: 3, oldestMinutes: 11 },
      { priority: "batch", count: 14, oldestMinutes: 286 },
    ],
    fleetBreakdown: [
      { resource: "TPU v5p", used: 208, total: 256, pending: 16 },
      { resource: "TPU v6e", used: 40, total: 64, pending: 32 },
      { resource: "TPU v4", used: 192, total: 256, pending: 64 },
      { resource: "CPU cores", used: 5870, total: 9984, pending: 384 },
    ],
    alerts: [
      {
        id: "alert-fra",
        severity: "critical",
        title: "TPU v5e capacity degraded in fra-gpu-01",
        detail: "Provision success is 64%; new interactive work is preferring us-east.",
        timestamp: "2026-07-10T08:12:55Z",
      },
      {
        id: "alert-budget",
        severity: "warning",
        title: "Foundation models forecast above budget",
        detail: "Current forecast is $73.4k against a $68k hard limit.",
        timestamp: "2026-07-10T07:58:20Z",
      },
    ],
  },
  workloads: mockWorkloads,
  cells: mockCells,
  leases: mockLeases,
  budgets: mockBudgets,
  policies: mockPolicies,
  auditEvents: mockAuditEvents,
  access: { fleetInventory: true, fleetAudit: true },
};

export function getMockWorkload(id: string): WorkloadDetail | undefined {
  const workload = mockWorkloads.find((item) => item.id === id);
  return workload ? detailFor(workload) : undefined;
}

/**
 * Shapes the demo fleet advertises.
 *
 * Two zones of one shape, because that is the case the submit form has to get
 * right: they are alternates, so the zone field offers both and neither is
 * required.
 */
export const mockAvailableShapes: AvailableShape[] = [
  {
    fleet: "research",
    pool: "research-v6e-8-us-east5-a",
    shape: "v6e-8",
    capacityClass: "spot",
    region: "us-east5",
    zone: "us-east5-a",
  },
  {
    fleet: "research",
    pool: "research-v6e-8-us-east5-b",
    shape: "v6e-8",
    capacityClass: "spot",
    region: "us-east5",
    zone: "us-east5-b",
  },
  {
    fleet: "research",
    pool: "research-cpu-16",
    shape: "cpu-16",
    capacityClass: "on_demand",
    region: "us-east5",
    zone: "us-east5-a",
  },
];

export function createMockWorkload(input: SubmitWorkloadInput): WorkloadDetail {
  const id = `wrk-${Math.random().toString(16).slice(2, 6)}`;
  // The pool supplies the infrastructure contract, so the demo cannot know the
  // per-replica shape a real submission would land on. Stated as what was
  // asked for -- the shape name and the pod count -- rather than invented.
  const workload: WorkloadSummary = {
    id,
    name: input.name ?? `${input.fleet}/${input.shape}`,
    org: "demo",
    project: "demo",
    user: "dashboard",
    status: "planning",
    priority: input.priority,
    resources: { cpu: 0, memoryGb: 0, replicas: input.pods, accelerator: input.shape },
    submittedAt: new Date().toISOString(),
    progress: 2,
    budgetSpent: 0,
    budgetLimit: 0,
  };
  // Register the workload so subsequent detail lookups and actions resolve it.
  mockWorkloads.unshift(workload);
  const detail = detailFor(workload);
  detail.command = input.command;
  detail.labels = input.labels;
  return detail;
}

export function cloneMock<T>(value: T): T {
  return structuredClone(value);
}

const RUN_STARTED = 1792627200;
export const mockRuns: RunSummary[] = [
  {
    run_id: "pong-seed-3",
    org_id: "demo",
    project_id: "rl",
    name: "ppo-atari/pong-s3",
    group: "sweep-3",
    tags: ["pong", "seed=3"],
    status: "running",
    workload_id: mockWorkloads[0].id,
    attempts: 3,
    started_at: RUN_STARTED,
    updated_at: RUN_STARTED + 4200,
    last_step: 330,
    keys_count: 4,
    summary: { best_return: 21, "loss/policy": 0.031 },
    owner: "alice",
    visibility: "team",
    shared_with: [],
    experiment_id: "exp-pong-speedrun",
    experiment: "Atari Pong Speedrun",
    pinned: true,
    notes_count: 2,
    duration_s: 4200,
    latest_metrics: { "loss/policy": 0.031, "reward/mean": 20.4, "system/sps": 9120 },
  },
  {
    run_id: "grpo-math",
    org_id: "demo",
    project_id: "rl",
    name: "grpo/math-7b",
    group: "grpo-math",
    tags: ["grpo", "math"],
    status: "running",
    workload_id: null,
    attempts: 1,
    started_at: RUN_STARTED,
    updated_at: RUN_STARTED + 4200,
    last_step: 260,
    keys_count: 4,
    summary: { mean_reward: 0.71 },
    owner: "bob",
    visibility: "team",
    shared_with: [],
    experiment_id: "exp-grpo-math",
    experiment: "GRPO Math Reasoning",
    pinned: false,
    notes_count: 1,
    duration_s: 4200,
    latest_metrics: { "reward/mean": 0.71, "policy/kl": 0.018 },
  },
  {
    run_id: "pong-seed-1",
    org_id: "demo",
    project_id: "rl",
    name: "ppo-atari/pong-s1",
    group: "sweep-3",
    tags: ["pong", "seed=1"],
    status: "finished",
    workload_id: null,
    attempts: 1,
    started_at: RUN_STARTED,
    updated_at: RUN_STARTED + 3600,
    last_step: 400,
    keys_count: 4,
    summary: { best_return: 21 },
    owner: "bob",
    visibility: "shared",
    shared_with: ["alice"],
    experiment_id: "exp-pong-speedrun",
    experiment: "Atari Pong Speedrun",
    pinned: false,
    notes_count: 0,
    duration_s: 3600,
    latest_metrics: { "loss/policy": 0.038, "reward/mean": 21.0, "system/sps": 8940 },
  },
  {
    run_id: "diverged",
    org_id: "demo",
    project_id: "rl",
    name: "ppo-atari/lr-1e3",
    group: "sweep-3",
    tags: ["lr=1e-3"],
    status: "failed",
    workload_id: null,
    attempts: 1,
    started_at: RUN_STARTED,
    updated_at: RUN_STARTED + 2400,
    last_step: 118,
    keys_count: 4,
    summary: { reason: "KL guard crossed" },
    owner: "carol",
    visibility: "team",
    shared_with: [],
    experiment_id: "exp-pong-speedrun",
    experiment: "Atari Pong Speedrun",
    pinned: false,
    notes_count: 1,
    duration_s: 2400,
    latest_metrics: { "loss/policy": 0.89, "reward/mean": -14.2 },
  },
  {
    run_id: "offline-eval",
    org_id: "demo",
    project_id: "rl",
    name: "offline-eval",
    group: null,
    tags: ["offline"],
    status: "unknown",
    workload_id: null,
    attempts: 1,
    started_at: RUN_STARTED,
    updated_at: RUN_STARTED + 1200,
    last_step: null,
    keys_count: 0,
    summary: {},
    owner: "alice",
    visibility: "private",
    shared_with: [],
    experiment_id: null,
    experiment: null,
    pinned: false,
    notes_count: 0,
    duration_s: 1200,
    latest_metrics: {},
  },
];

export function getMockRun(runId: string): RunDetail | undefined {
  const run = mockRuns.find((candidate) => candidate.run_id === runId);
  if (!run) return undefined;
  const grpo = run.run_id === "grpo-math";
  return cloneMock({
    ...run,
    config: grpo
      ? { algo: "grpo", model: "math-7b", group_size: 8, max_tokens: 1024 }
      : { algo: "ppo", env: "Pong-v5", seed: 3, learning_rate: 0.00025, num_envs: 64 },
    keys: grpo
      ? ["reward/mean", "groups/zero_variance_frac", "policy/kl", "system/tokens_per_second"]
      : run.keys_count
        ? ["loss/policy", "loss/value", "reward/mean", "system/sps"]
        : [],
    axes: ["env_steps"],
    hist_keys: run.keys_count ? ["advantages", "policy/logits"] : [],
    episodes: { train: run.keys_count && !grpo ? 10560 : 0, eval: run.keys_count && !grpo ? 80 : 0 },
    episode_axes: ["env_steps"],
    envs: run.keys_count ? ["Pong-v5"] : [],
    actors: run.keys_count ? 32 : 0,
    rollouts: run.run_id.startsWith("pong") ? 4 : 0,
    completion_groups: run.run_id === "grpo-math" ? 32 : 0,
    table_keys: run.keys_count ? ["eval/cases"] : [],
    media_keys: run.keys_count ? ["eval/frame"] : [],
    attempt_list: Array.from({ length: run.attempts }, (_, i) => ({
      attempt: i + 1,
      started_at: run.started_at + i * 1400,
      first_step: i * 110,
      last_step: i === run.attempts - 1 ? run.last_step : (i + 1) * 110,
      status:
        i === run.attempts - 1 ? (run.status === "running" || run.status === "unknown" ? null : run.status) : "aborted",
    })),
    stats: { records: 1320, dropped: { keys: 0, rate: 2, invalid: 0, queue: 0 }, spilled: 1 },
    errors: [],
  });
}

export function getMockRunSeries(run: RunDetail, keys: string[], x: string, points: number): RunSeriesResponse {
  const series = keys
    .filter((key) => run.keys.includes(key))
    .map((key): RunSeries => {
      const total = (run.last_step ?? 0) + 1;
      const n = Math.min(total, Math.max(1, Math.floor(points)));
      const xFor = (step: number) =>
        x === "time" ? run.started_at + step * 12 : x === "env_steps" ? step * 8192 : step;
      return {
        key,
        rank: 0,
        total,
        downsampled: total > n,
        nonfinite: 0,
        points: Array.from({ length: n }, (_, i) => {
          const step = n === 1 ? 0 : (i * (total - 1)) / (n - 1);
          const y =
            key === "system/sps"
              ? 9000 + Math.sin(step / 13) * 300
              : key === "reward/mean"
                ? Math.min(21, -20 + step / 7) + Math.sin(step / 11)
                : Math.exp(-step / 100) * (key === "loss/value" ? 10 : 0.4) + Math.sin(step / 9) * 0.02;
          const spread = total > n ? Math.abs(y) * 0.08 : 0;
          return [xFor(step), y, y - spread, y + spread];
        }),
        attempts: run.attempt_list.map((attempt) => ({
          attempt: attempt.attempt,
          first_x: xFor(attempt.first_step ?? 0),
        })),
        replayed: [],
      };
    });
  return { run_id: run.run_id, x, series };
}

function trackingX(run: RunDetail, step: number, x: string): number | null {
  return x === "time" ? run.started_at + step * 12 : x === "env_steps" ? step * 8192 : x === "step" ? step : null;
}

export function getMockRunHistograms(run: RunDetail, options: RunHistogramsOptions): RunHistogramsResponse {
  const x = options.x ?? "step";
  const total =
    run.hist_keys?.includes(options.key) && trackingX(run, 0, x) !== null && (options.rank ?? 0) === 0 ? 67 : 0;
  const n = Math.min(total, options.points ?? 200);
  const bins = options.bins ?? 64;
  const edges = total ? Array.from({ length: bins + 1 }, (_, i) => -4 + (i * 8) / bins) : [];
  return {
    run_id: run.run_id,
    key: options.key,
    rank: options.rank ?? 0,
    x,
    edges,
    total,
    downsampled: total > n,
    columns: Array.from({ length: n }, (_, i) => {
      const p = n === 1 ? 1 : i / (n - 1);
      const step = Math.round(p * (run.last_step ?? 0));
      const std = (options.key === "advantages" ? 1.4 : 2) * Math.exp(-p * 3) + 0.35;
      const counts = edges
        .slice(1)
        .map((right, j) =>
          Math.round((180 * Math.exp(-(((right + edges[j]) / 2 - 0.05) ** 2) / (2 * std ** 2))) / std),
        );
      return {
        x: trackingX(run, step, x)!,
        step,
        attempt: Math.min(run.attempts, 1 + Math.floor(p * run.attempts)),
        counts,
        n: counts.reduce((a, b) => a + b, 0),
        under: 0,
        over: 0,
        nonfinite: 0,
        min: -3 * std,
        max: 3 * std,
        mean: 0.05,
        std,
      };
    }),
  };
}

function demoEpisodes(run: RunDetail): RunEpisode[] {
  if (!run.episodes?.train) return [];
  const episodes: RunEpisode[] = [];
  const end = run.last_step ?? 330;
  for (let i = 0; i < 80; i++) {
    const p = i / 79;
    const step = Math.round(p * end);
    const center = -21 + (41 * (1 - Math.exp(-4 * p))) / (1 - Math.exp(-4));
    const spread = 0.6 + 8 * Math.sin(Math.PI * p) ** 2;
    for (let actor = 0; actor < 32; actor++) {
      const ret =
        actor === 23
          ? Math.min(4.1, center)
          : Math.max(-21, Math.min(21, center + Math.sin(actor * 2.1 + i * 0.8) * spread));
      episodes.push({
        t: run.started_at + step * 12 + actor / 100,
        step,
        ret,
        length: Math.round(1100 - center * 22 + Math.cos(actor + i) * 70),
        success: ret > 15,
        kind: "train",
        actor: String(actor),
        env: "Pong-v5",
        seed: 3,
        policy_version: step,
        learner_version: step + (actor === 23 ? 6 : actor % 5 === 0 ? 4 : actor % 3),
        checkpoint: null,
        x: { env_steps: step * 8192 },
        extra: {},
        attempt: Math.min(run.attempts, 1 + Math.floor(p * run.attempts)),
        rank: Math.floor(actor / 8),
        replayed: false,
      });
    }
    if (i % 10 === 9) {
      for (let sample = 0; sample < 10; sample++) {
        episodes.push({
          ...episodes[episodes.length - 1],
          kind: "eval",
          actor: null,
          checkpoint: `ckpt-${String(step).padStart(4, "0")}`,
          ret: Math.min(21, center + 0.5 + Math.sin(sample) * 0.6),
          t: run.started_at + step * 12 + 1 + sample / 100,
          rank: 0,
        });
      }
    }
  }
  if (run.attempts > 1)
    episodes.push(...episodes.slice(-32).map((episode) => ({ ...episode, attempt: 1, replayed: true })));
  return episodes;
}

export function getMockRunEpisodes(run: RunDetail, options: RunEpisodesOptions = {}): RunEpisodesResponse {
  const all = demoEpisodes(run)
    .filter(
      (episode) =>
        (!options.kind || episode.kind === options.kind) &&
        (options.actor === undefined || episode.actor === options.actor) &&
        (options.env === undefined || episode.env === options.env) &&
        (options.includeReplayed || !episode.replayed),
    )
    .sort((a, b) => b.t - a.t || b.attempt - a.attempt);
  const offset = options.offset ?? 0;
  return { episodes: all.slice(offset, offset + (options.limit ?? 100)), total: all.length };
}

function finiteMean(values: number[]): number | null {
  return values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : null;
}
function percentile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = p * (sorted.length - 1);
  return sorted[Math.floor(index)] + (sorted[Math.ceil(index)] - sorted[Math.floor(index)]) * (index % 1);
}

export function getMockEpisodeCurve(run: RunDetail, options: EpisodeCurveOptions = {}): EpisodeCurveResponse {
  const metric = options.metric ?? "return",
    kind = options.kind ?? "train",
    x = options.x ?? "step";
  const all = demoEpisodes(run).filter(
    (episode) =>
      !episode.replayed &&
      episode.kind === kind &&
      (options.env === undefined || episode.env === options.env) &&
      (options.actor === undefined || episode.actor === options.actor),
  );
  const eligible = all.filter((episode) => trackingX(run, episode.step!, x) !== null);
  const n = Math.min(options.points ?? 200, eligible.length);
  return {
    run_id: run.run_id,
    metric,
    kind,
    x,
    total: eligible.length,
    nonfinite: 0,
    replayed_excluded: demoEpisodes(run).filter((episode) => episode.replayed).length,
    buckets: Array.from({ length: n }, (_, i) => {
      const batch = eligible.slice(Math.floor((i * eligible.length) / n), Math.floor(((i + 1) * eligible.length) / n));
      const values = batch.map((episode) => Number(metric === "return" ? episode.ret : episode.length));
      const xs = batch.map((episode) => trackingX(run, episode.step!, x)!);
      return {
        x: finiteMean(xs)!,
        x_min: Math.min(...xs),
        x_max: Math.max(...xs),
        count: batch.length,
        mean: finiteMean(values),
        min: Math.min(...values),
        max: Math.max(...values),
        p5: percentile(values, 0.05),
        p25: percentile(values, 0.25),
        p50: percentile(values, 0.5),
        p75: percentile(values, 0.75),
        p95: percentile(values, 0.95),
        success_rate: finiteMean(batch.map((episode) => Number(episode.success))),
      };
    }),
  };
}

export function getMockEpisodeBreakdown(
  run: RunDetail,
  options: EpisodeBreakdownOptions = {},
): EpisodeBreakdownResponse {
  const episodes = demoEpisodes(run),
    kind = options.kind ?? "train";
  const selected = episodes.filter((episode) => !episode.replayed && episode.kind === kind);
  const actors = [...new Set(selected.map((episode) => episode.actor))]
    .sort((a, b) => (a === null ? 1 : b === null ? -1 : Number(a) - Number(b)))
    .map((actor) => {
      const all = selected.filter((episode) => episode.actor === actor);
      const window = all.slice(-(options.last ?? 200));
      const values = window.map((episode) => Number(episode.ret));
      const lags = window.map((episode) => episode.learner_version! - episode.policy_version!);
      return {
        actor,
        count: all.length,
        window: window.length,
        mean_return: finiteMean(values),
        p50_return: percentile(values, 0.5),
        last_return: values.at(-1) ?? null,
        mean_length: finiteMean(window.map((episode) => episode.length!)),
        success_rate: finiteMean(window.map((episode) => Number(episode.success))),
        mean_lag: finiteMean(lags),
        max_lag: lags.length ? Math.max(...lags) : null,
        last_t: window.at(-1)!.t,
      };
    });
  const train = episodes.filter((episode) => !episode.replayed && episode.kind === "train");
  const lags = train.map((episode) => episode.learner_version! - episode.policy_version!);
  const evaluations = episodes.filter((episode) => !episode.replayed && episode.kind === "eval");
  const evals = [...new Set(evaluations.map((episode) => episode.checkpoint))].map((checkpoint) => {
    const batch = evaluations.filter((episode) => episode.checkpoint === checkpoint),
      first = batch[0];
    const values = batch.map((episode) => Number(episode.ret)),
      mean = finiteMean(values)!;
    return {
      checkpoint,
      policy_version: first.policy_version,
      step: first.step,
      count: batch.length,
      mean_return: mean,
      std_return: Math.sqrt(finiteMean(values.map((v) => (v - mean) ** 2))!),
      min_return: Math.min(...values),
      max_return: Math.max(...values),
      success_rate: finiteMean(batch.map((episode) => Number(episode.success))),
      mean_length: finiteMean(batch.map((episode) => episode.length!)),
      first_t: first.t,
      best: false,
    };
  });
  const best = [...evals].sort((a, b) => b.mean_return - a.mean_return)[0];
  if (best) best.best = true;
  return {
    actors,
    envs: selected.length
      ? [
          {
            env: "Pong-v5",
            count: selected.length,
            mean_return: finiteMean(selected.map((episode) => Number(episode.ret))),
            success_rate: finiteMean(selected.map((episode) => Number(episode.success))),
          },
        ]
      : [],
    lag: {
      counts: Array.from({ length: 16 }, (_, i) => lags.filter((lag) => lag === i).length),
      over: lags.filter((lag) => lag >= 16).length,
      negative: lags.filter((lag) => lag < 0).length,
      count: lags.length,
      p50: percentile(lags, 0.5),
      p95: percentile(lags, 0.95),
      max: lags.length ? Math.max(...lags) : null,
    },
    evals,
    replayed_excluded: episodes.filter((episode) => episode.replayed).length,
  };
}

// A valid, one-pixel PNG keeps demo media deterministic and network-free.
const DEMO_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a0ioAAAAASUVORK5CYII=";
const demoBlob: BlobRef = { sha256: "d".repeat(64), content_type: "image/png", bytes: 68, available: true };

function demoRollouts(run: RunDetail): RunRollout[] {
  return Array.from({ length: run.rollouts ?? 0 }, (_, i) => {
    const length = 180 + i * 20;
    return {
      id: `a1-r0-demo-${i}-0`,
      episode_id: `pong-${812 + i}`,
      t: run.started_at + 3000 + i * 30,
      step: 250 + i,
      actor: String(i),
      env: "Pong-v5",
      kind: i === 3 ? "eval" : "train",
      policy_version: 260,
      length,
      attempt: 1,
      rank: 0,
      seq: i,
      line: 0,
      replayed: false,
      ret: 18 - i,
      video: null,
      fps: 30,
      x: { env_steps: (250 + i) * 8192 },
      action_names: ["NOOP", "FIRE", "RIGHT", "LEFT"],
      actions: Array.from({ length }, (_, t) => Math.floor(t / 8) % 4),
      tracks: {
        reward: Array.from({ length }, (_, t) => (t % 31 === 30 ? 1 : t % 57 === 56 ? -1 : 0)),
        value: Array.from({ length }, (_, t) => 0.6 + Math.sin(t / 18 + i) * 0.3),
        entropy: Array.from({ length }, (_, t) => 0.7 + Math.cos(t / 23) * 0.2),
        done: Array.from({ length }, (_, t) => Number(t === length - 1)),
      },
    };
  });
}
export function getMockRunRollouts(run: RunDetail, options: RunRolloutsOptions = {}): RunRolloutsResponse {
  const all = demoRollouts(run)
    .filter(
      (item) =>
        (!options.kind || item.kind === options.kind) && (options.actor === undefined || item.actor === options.actor),
    )
    .reverse();
  return {
    total: all.length,
    rollouts: all.slice(options.offset ?? 0, (options.offset ?? 0) + (options.limit ?? 50)).map((item) => {
      const { tracks, actions, ...summary } = item;
      void actions;
      return { ...summary, track_names: Object.keys(tracks) };
    }),
  };
}
export function getMockRunRollout(run: RunDetail, id: string) {
  return demoRollouts(run).find((item) => item.id === id);
}

function demoCompletions(run: RunDetail): RunCompletion[] {
  return Array.from({ length: run.completion_groups ?? 0 }, (_, i) => {
    const step = 230 + Math.floor(i / 8) * 10,
      zero = i % 8 === 7;
    const responses = Array.from({ length: 8 }, (_, j) => {
      const correct = zero ? 0 : Number((i + j) % 3 !== 0),
        format = zero ? 1 : Number(j !== 2);
      return {
        text:
          j === 0
            ? "<b>x</b>\nThis is literal model output, not markup."
            : `Let us reason carefully. ${j % 3 === 0 ? "The answer is 41." : "The answer is 42."}\n\\boxed{${j % 3 === 0 ? 41 : 42}}`,
        rewards: { correct, format: format * 0.2 },
        reward: correct + format * 0.2,
        advantage: 0,
        tokens: j === 6 ? 1024 : 160 + j * 32,
        kl: 0.01 + j * 0.003,
        truncated: j === 6,
      };
    });
    const mean = responses.reduce((sum, item) => sum + item.reward, 0) / 8;
    const std = Math.sqrt(responses.reduce((sum, item) => sum + (item.reward - mean) ** 2, 0) / 8);
    responses.forEach((item) => {
      item.advantage = std < 1e-8 ? 0 : (item.reward - mean) / std;
    });
    return {
      id: `a1-r0-demo-${i}-0`,
      group: `g-${808 + i}`,
      step,
      t: run.started_at + step * 12 + i,
      prompt: "What is 6 × 7? Explain your reasoning. <prompt>",
      answer: "42",
      policy_version: step,
      responses,
      k: 8,
      mean_reward: mean,
      std_reward: std,
      pass_rate: zero ? 1 : responses.filter((item) => item.reward > 0).length / 8,
      zero_variance: zero,
      mean_tokens: responses.reduce((sum, item) => sum + item.tokens, 0) / 8,
      truncated: 1,
      attempt: 1,
      rank: 0,
      replayed: false,
    };
  });
}
export function getMockRunCompletions(run: RunDetail, options: RunCompletionsOptions = {}): RunCompletionsResponse {
  const all = demoCompletions(run);
  const selected = all
    .filter((item) => options.step === undefined || item.step === options.step)
    .sort((a, b) =>
      options.sort === "zero_variance"
        ? Number(b.zero_variance) - Number(a.zero_variance) || b.t - a.t
        : options.sort === "reward"
          ? (b.mean_reward ?? 0) - (a.mean_reward ?? 0) || b.t - a.t
          : b.t - a.t,
    );
  return {
    groups: selected.slice(options.offset ?? 0, (options.offset ?? 0) + (options.limit ?? 50)).map((item) => {
      const { responses, answer, policy_version, ...summary } = item;
      void responses;
      void answer;
      void policy_version;
      return summary;
    }),
    total: selected.length,
    steps: [...new Set(all.map((item) => item.step!))],
  };
}
export function getMockRunCompletion(run: RunDetail, id: string) {
  return demoCompletions(run).find((item) => item.id === id);
}
export function getMockCompletionStats(run: RunDetail, points = 200): CompletionStatsResponse {
  const all = demoCompletions(run);
  return {
    steps: [...new Set(all.map((item) => item.step!))].slice(0, points).map((step) => {
      const batch = all.filter((item) => item.step === step);
      return {
        step,
        groups: batch.length,
        mean_reward: finiteMean(batch.map((item) => item.mean_reward!)),
        zero_variance_frac: batch.filter((item) => item.zero_variance).length / batch.length,
        mean_tokens: finiteMean(batch.map((item) => item.mean_tokens!)),
        truncated_frac: 1 / 8,
        components: { correct: 0.65, format: 0.175 },
      };
    }),
  };
}
export function getMockRunTable(run: RunDetail, options: RunTableOptions = {}): RunTableResponse | undefined {
  if (!options.key) return { keys: (run.table_keys ?? []).map((key) => ({ key, count: 2, last_step: 260 })) };
  if (!run.table_keys?.includes(options.key) || (options.step !== undefined && ![250, 260].includes(options.step)))
    return undefined;
  return {
    key: options.key,
    step: options.step ?? 260,
    t: run.started_at + 3100,
    steps: [250, 260],
    columns: ["case", "prediction", "passed", "frame"],
    rows: [
      ["pong-01", "<b>plain text</b>", true, demoBlob],
      ["pong-02", "42", false, null],
    ],
  };
}
export function getMockRunMedia(run: RunDetail, options: RunMediaOptions = {}): RunMediaResponse {
  const all = (run.media_keys ?? [])
    .filter((key) => !options.key || key === options.key)
    .map((key) => ({
      key,
      step: 260,
      t: run.started_at + 3100,
      caption: "Demo frame · tiny PNG · no video in demo",
      blob: { ...demoBlob },
    }));
  return { media: all.slice(options.offset ?? 0, (options.offset ?? 0) + (options.limit ?? 100)), total: all.length };
}
export function getMockRunBlob(run: RunDetail, digest: string): Blob | undefined {
  if (!run.media_keys?.length || digest !== demoBlob.sha256) return undefined;
  return new Blob([Uint8Array.from(atob(DEMO_PNG), (char) => char.charCodeAt(0))], { type: "image/png" });
}

export const mockExperiments: ExperimentRecord[] = [
  {
    id: "exp-pong-speedrun",
    org_id: "demo",
    project_id: "rl",
    name: "Atari Pong Speedrun",
    subject: "rl-speedrun",
    kind: "speedrun",
    description: "Fastest wall-clock and step count to reach mean reward >= 20.0 on Pong-v5.",
    metric_key: "reward/mean",
    goal: "max",
    target_value: 20.0,
    group_filter: "sweep-3",
    tag_filter: [],
    visibility: "team",
    shared_with: [],
    owner: "alice",
    pinned_baseline_run_id: "pong-seed-3",
    created_at: "2026-07-08T10:00:00Z",
    updated_at: "2026-07-10T08:30:00Z",
    notes_count: 2,
    run_count: 3,
    running_count: 1,
    leader_run_id: "pong-seed-1",
    leader_owner: "bob",
    leader_value: 21.0,
  },
  {
    id: "exp-grpo-math",
    org_id: "demo",
    project_id: "rl",
    name: "GRPO Math Reasoning",
    subject: "post-training",
    kind: "experiment",
    description: "Reward shaping and KL penalty sweeps for 7B math reasoning checkpoints.",
    metric_key: "reward/mean",
    goal: "max",
    target_value: null,
    group_filter: "grpo-math",
    tag_filter: ["grpo"],
    visibility: "team",
    shared_with: [],
    owner: "bob",
    pinned_baseline_run_id: null,
    created_at: "2026-07-09T14:20:00Z",
    updated_at: "2026-07-10T08:10:00Z",
    notes_count: 1,
    run_count: 1,
    running_count: 1,
    leader_run_id: "grpo-math",
    leader_owner: "bob",
    leader_value: 0.71,
  },
];

export const mockNotes: CollaborationNote[] = [
  {
    id: "note-1",
    org_id: "demo",
    project_id: "rl",
    target_type: "experiment",
    target_id: "exp-pong-speedrun",
    author: "alice",
    body: "Pinned pong-seed-3 as baseline. Cosine LR decay shaved 180s off time-to-20.",
    metric_ref: "reward/mean",
    step_ref: 280,
    created_at: "2026-07-10T07:55:00Z",
  },
  {
    id: "note-2",
    org_id: "demo",
    project_id: "rl",
    target_type: "run",
    target_id: "pong-seed-3",
    author: "bob",
    body: "Value loss stabilizes cleanly after attempt 2 resume.",
    metric_ref: "loss/policy",
    step_ref: 220,
    created_at: "2026-07-10T08:15:00Z",
  },
];

export const mockWebhooks: WebhookSubscription[] = [
  {
    id: "whk-demo-1",
    org_id: "demo",
    project_id: "rl",
    url: "https://hooks.example.org/eyvan/events",
    entity_types: ["workload", "budget"],
    actions: ["workload.failed", "budget.threshold_crossed"],
    enabled: true,
    created_at: "2026-07-08T12:00:00Z",
    updated_at: "2026-07-08T12:00:00Z",
  },
];

export const mockManagedArtifacts: ManagedArtifact[] = [
  {
    uri: "gs://example-bucket/workspaces/sha256-a1b2c3d4.tar.gz",
    kind: "workspace",
    org_id: "demo",
    project_id: "rl",
    state: "active",
    record_generation: 1,
    size_bytes: 18_420_112,
    sha256: "a1b2c3d4e5f60718293a4b5c6d7e8f90",
    object_version: "1720601000000",
    created_at: "2026-07-10T06:00:00Z",
    updated_at: "2026-07-10T06:00:00Z",
  },
];

export const mockProjects: ProjectRecord[] = [
  {
    org_id: "demo",
    id: "rl",
    display_name: "RL Research",
    owner: "alice",
    enabled: true,
    created_at: "2026-07-01T00:00:00Z",
    updated_at: "2026-07-01T00:00:00Z",
  },
  {
    org_id: "demo",
    id: "foundation-models",
    display_name: "Foundation Models",
    owner: "bob",
    enabled: true,
    created_at: "2026-07-02T00:00:00Z",
    updated_at: "2026-07-02T00:00:00Z",
  },
];

export const mockRoleGrants: DurableRoleGrant[] = [
  {
    subject: "alice",
    grant: { role: "project_admin", org_id: "demo", project_id: "rl" },
    reason: "Lead researcher",
    granted_by: "admin",
    granted_at: "2026-07-01T00:00:00Z",
  },
  {
    subject: "bob",
    grant: { role: "project_operator", org_id: "demo", project_id: "rl" },
    reason: "Contributor",
    granted_by: "alice",
    granted_at: "2026-07-03T00:00:00Z",
  },
];

function flattenConfig(input: Record<string, unknown>, prefix = ""): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(input)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      Object.assign(out, flattenConfig(v as Record<string, unknown>, key));
    } else {
      out[key] = v;
    }
  }
  return out;
}

export function getMockCompareRuns(input: RunCompareInput, orgId = "demo", projectId = "rl"): RunCompareResponse {
  const uniqueIds = [...new Set(input.runIds)];
  const runs = uniqueIds.map((id) => getMockRun(id)).filter((item): item is RunDetail => Boolean(item));
  const x = input.x ?? "step";
  const points = input.points ?? 200;
  const allKeySet = new Set<string>();
  for (const run of runs) {
    for (const k of run.keys) allKeySet.add(k);
  }
  const allKeys = [...allKeySet].sort();
  const commonKeys = runs.length ? allKeys.filter((key) => runs.every((run) => run.keys.includes(key))) : [];
  const selectedKeys = input.keys?.length
    ? input.keys.filter((key) => allKeySet.has(key))
    : (commonKeys.length ? commonKeys : allKeys).slice(0, 12);

  const seriesByKey: Record<string, Record<string, RunCompareSeriesData>> = {};
  const metricSummary: RunCompareMetricSummaryRow[] = [];
  for (const key of selectedKeys) {
    const perRun: Record<string, RunCompareSeriesData> = {};
    const summaryValues: RunCompareMetricSummaryRow["values"] = {};
    for (let idx = 0; idx < runs.length; idx++) {
      const run = runs[idx];
      if (!run.keys.includes(key)) continue;
      const resp = getMockRunSeries(run, [key], x, points);
      const item = resp.series[0];
      if (!item) continue;
      // Slight deterministic offset per run index so overlaid curves remain visually distinct in demo mode.
      const shift = idx * 0.05;
      const xs = item.points.map((pt) => pt[0]);
      const ys = item.points.map((pt) => (pt[1] === null ? null : Number((pt[1] * (1 - shift)).toFixed(4))));
      perRun[run.run_id] = { x: xs, y: ys };
      const finite = ys.filter((v): v is number => v !== null && Number.isFinite(v));
      summaryValues[run.run_id] = {
        latest: finite.length ? finite[finite.length - 1] : null,
        min: finite.length ? Math.min(...finite) : null,
        max: finite.length ? Math.max(...finite) : null,
        last_step: run.last_step,
      };
    }
    seriesByKey[key] = perRun;
    metricSummary.push({ key, values: summaryValues });
  }

  const flattenedByRun = new Map(runs.map((run) => [run.run_id, flattenConfig(run.config)]));
  const configKeys = [...new Set(runs.flatMap((run) => Object.keys(flattenedByRun.get(run.run_id) ?? {})))].sort();
  const configDiff = configKeys
    .map((key) => {
      const values: Record<string, unknown> = {};
      const serialized = new Set<string>();
      for (const run of runs) {
        const val = flattenedByRun.get(run.run_id)?.[key];
        values[run.run_id] = val;
        serialized.add(JSON.stringify(val));
      }
      return { key, values, differs: serialized.size > 1 };
    })
    .sort((a, b) => Number(b.differs) - Number(a.differs) || a.key.localeCompare(b.key));

  return {
    org_id: orgId,
    project_id: projectId,
    x,
    runs,
    common_keys: commonKeys,
    all_keys: allKeys,
    selected_keys: selectedKeys,
    series_by_key: seriesByKey,
    config_diff: configDiff,
    metric_summary: metricSummary,
  };
}

export function updateMockRunSharing(
  runId: string,
  patch: RunSharingPatch,
  orgId = "demo",
  projectId = "rl",
): RunCollaborationOverlay {
  const run = mockRuns.find((item) => item.run_id === runId);
  if (run) {
    if (patch.visibility !== undefined) run.visibility = patch.visibility;
    if (patch.shared_with !== undefined) run.shared_with = [...patch.shared_with];
    if ("experiment_id" in patch) {
      run.experiment_id = patch.experiment_id ?? null;
      const exp = mockExperiments.find((candidate) => candidate.id === run.experiment_id);
      run.experiment = exp ? exp.name : null;
    }
    if (patch.pinned !== undefined) run.pinned = patch.pinned;
    if (patch.owner !== undefined) run.owner = patch.owner;
  }
  return {
    org_id: orgId,
    project_id: projectId,
    run_id: runId,
    owner: run?.owner ?? patch.owner ?? "alice",
    visibility: run?.visibility ?? patch.visibility ?? "team",
    shared_with: [...(run?.shared_with ?? patch.shared_with ?? [])],
    experiment_id: run?.experiment_id ?? patch.experiment_id ?? null,
    pinned: Boolean(run?.pinned ?? patch.pinned),
    updated_at: new Date().toISOString(),
    updated_by: "dashboard-demo",
  };
}

export function createMockExperiment(input: ExperimentCreateInput, orgId = "demo", projectId = "rl"): ExperimentRecord {
  const now = new Date().toISOString();
  const created: ExperimentRecord = {
    id: `exp-${Math.random().toString(16).slice(2, 10)}`,
    org_id: orgId,
    project_id: projectId,
    name: input.name,
    subject: input.subject ?? "",
    kind: input.kind ?? "experiment",
    description: input.description ?? "",
    metric_key: input.metric_key ?? null,
    goal: input.goal ?? "min",
    target_value: input.target_value ?? null,
    group_filter: input.group_filter ?? null,
    tag_filter: [...(input.tag_filter ?? [])],
    visibility: input.visibility ?? "team",
    shared_with: [...(input.shared_with ?? [])],
    owner: "alice",
    pinned_baseline_run_id: input.pinned_baseline_run_id ?? null,
    created_at: now,
    updated_at: now,
    notes_count: 0,
    run_count: 0,
    running_count: 0,
  };
  mockExperiments.unshift(created);
  return cloneMock(created);
}

export function updateMockExperiment(expId: string, patch: ExperimentUpdatePatch): ExperimentRecord | undefined {
  const existing = mockExperiments.find((item) => item.id === expId);
  if (!existing) return undefined;
  Object.assign(existing, {
    ...patch,
    updated_at: new Date().toISOString(),
  });
  return cloneMock(existing);
}

export function deleteMockExperiment(expId: string): boolean {
  const index = mockExperiments.findIndex((item) => item.id === expId);
  if (index === -1) return false;
  mockExperiments.splice(index, 1);
  return true;
}

export function getMockExperimentLeaderboard(expId: string): ExperimentLeaderboardResponse | undefined {
  const exp = mockExperiments.find((item) => item.id === expId);
  if (!exp) return undefined;
  const matched = mockRuns.filter(
    (run) =>
      run.experiment_id === exp.id ||
      (exp.group_filter && run.group === exp.group_filter) ||
      (exp.tag_filter.length > 0 && exp.tag_filter.every((tag) => run.tags.includes(tag))) ||
      run.experiment === exp.name,
  );
  const metricKey = exp.metric_key;
  const rawEntries = matched.map((run) => {
    const val =
      metricKey && run.latest_metrics && typeof run.latest_metrics[metricKey] === "number"
        ? run.latest_metrics[metricKey]
        : metricKey && typeof run.summary[metricKey] === "number"
          ? (run.summary[metricKey] as number)
          : null;
    const reached =
      val !== null &&
      exp.target_value !== null &&
      (exp.goal === "min" ? val <= exp.target_value : val >= exp.target_value);
    const stepsToTarget = reached && run.last_step !== null ? Math.max(1, Math.round(run.last_step * 0.78)) : null;
    const timeToTarget = reached && run.duration_s ? Math.round(run.duration_s * 0.72) : null;
    return {
      run,
      best_value: val,
      latest_value: val,
      reached_target: reached,
      steps_to_target: stepsToTarget,
      time_to_target_s: timeToTarget,
    };
  });

  rawEntries.sort((a, b) => {
    if (exp.target_value !== null && a.reached_target !== b.reached_target) return a.reached_target ? -1 : 1;
    if (exp.target_value !== null && a.reached_target && b.reached_target) {
      const ta = a.time_to_target_s ?? Number.POSITIVE_INFINITY;
      const tb = b.time_to_target_s ?? Number.POSITIVE_INFINITY;
      if (ta !== tb) return ta - tb;
    }
    if (a.best_value === null && b.best_value === null) return b.run.updated_at - a.run.updated_at;
    if (a.best_value === null) return 1;
    if (b.best_value === null) return -1;
    return exp.goal === "min" ? a.best_value - b.best_value : b.best_value - a.best_value;
  });

  const baselineRunId = exp.pinned_baseline_run_id ?? matched.find((r) => r.pinned)?.run_id ?? null;
  const baselineEntry = baselineRunId ? rawEntries.find((item) => item.run.run_id === baselineRunId) : undefined;
  const baselineValue = baselineEntry?.best_value ?? null;

  const entries: ExperimentLeaderboardEntry[] = rawEntries.map((item, idx) => ({
    rank: idx + 1,
    run_id: item.run.run_id,
    name: item.run.name,
    owner: item.run.owner ?? null,
    status: item.run.status,
    visibility: item.run.visibility,
    pinned: Boolean(item.run.pinned),
    is_baseline: item.run.run_id === baselineRunId,
    is_record: idx === 0 && item.best_value !== null,
    best_value: item.best_value,
    latest_value: item.latest_value,
    delta_vs_baseline:
      item.best_value !== null && baselineValue !== null ? Number((item.best_value - baselineValue).toFixed(4)) : null,
    reached_target: item.reached_target,
    steps_to_target: item.steps_to_target,
    time_to_target_s: item.time_to_target_s,
    last_step: item.run.last_step,
    duration_s: item.run.duration_s ?? null,
    updated_at: item.run.updated_at,
    tags: item.run.tags,
    group: item.run.group,
  }));

  const byOwner = new Map<
    string,
    { owner: string; run_count: number; best_value: number | null; reached_target_count: number; running_count: number }
  >();
  for (const entry of entries) {
    const owner = entry.owner ?? "unassigned";
    const cur = byOwner.get(owner) ?? {
      owner,
      run_count: 0,
      best_value: null,
      reached_target_count: 0,
      running_count: 0,
    };
    cur.run_count += 1;
    if (entry.status === "running") cur.running_count += 1;
    if (entry.reached_target) cur.reached_target_count += 1;
    if (entry.best_value !== null) {
      cur.best_value =
        cur.best_value === null
          ? entry.best_value
          : exp.goal === "min"
            ? Math.min(cur.best_value, entry.best_value)
            : Math.max(cur.best_value, entry.best_value);
    }
    byOwner.set(owner, cur);
  }

  return cloneMock({
    experiment: exp,
    metric_key: exp.metric_key,
    goal: exp.goal,
    target_value: exp.target_value,
    baseline_run_id: baselineRunId,
    baseline_value: baselineValue,
    tracker_available: true,
    entries,
    researcher_breakdown: [...byOwner.values()],
    notes_count: mockNotes.filter((n) => n.target_type === "experiment" && n.target_id === exp.id).length,
  });
}

export function getMockCollaborationFeed(limit = 30, orgId = "demo", projectId = "rl"): CollaborationFeedResponse {
  const runs = cloneMock(mockRuns).slice(0, limit);
  const active = runs.filter((run) => run.status === "running");
  const researchersMap = new Map<
    string,
    { owner: string; total_runs: number; running_runs: number; shared_runs: number; last_active_at: number | null }
  >();
  for (const run of runs) {
    const owner = run.owner ?? "unassigned";
    const cur = researchersMap.get(owner) ?? {
      owner,
      total_runs: 0,
      running_runs: 0,
      shared_runs: 0,
      last_active_at: null,
    };
    cur.total_runs += 1;
    if (run.status === "running") cur.running_runs += 1;
    if (run.visibility === "team" || run.visibility === "shared") cur.shared_runs += 1;
    if (cur.last_active_at === null || run.updated_at > cur.last_active_at) cur.last_active_at = run.updated_at;
    researchersMap.set(owner, cur);
  }
  return {
    org_id: orgId,
    project_id: projectId,
    tracker_available: true,
    active_runs: active,
    recent_runs: runs,
    experiments: cloneMock(mockExperiments),
    recent_notes: cloneMock(mockNotes).slice(0, limit),
    researchers: [...researchersMap.values()],
  };
}

export function createMockNote(input: CollaborationNoteInput, orgId = "demo", projectId = "rl"): CollaborationNote {
  const note: CollaborationNote = {
    id: `note-${Math.random().toString(16).slice(2, 10)}`,
    org_id: orgId,
    project_id: projectId,
    target_type: input.target_type,
    target_id: input.target_id,
    author: "alice",
    body: input.body,
    metric_ref: input.metric_ref ?? null,
    step_ref: input.step_ref ?? null,
    created_at: new Date().toISOString(),
  };
  mockNotes.unshift(note);
  return cloneMock(note);
}

export function deleteMockNote(noteId: string): boolean {
  const index = mockNotes.findIndex((item) => item.id === noteId);
  if (index === -1) return false;
  mockNotes.splice(index, 1);
  return true;
}

export function checkMockStorageRegion(input: StorageRegionCheckInput): StorageRegionCheckResult {
  const match = /^gs:\/\/([^/]+)/i.exec(input.uri.trim());
  const bucket = match ? match[1] : "";
  const hostRegion = input.region?.trim() || "us-central2";
  const allowlisted = Boolean(bucket && input.allowed_buckets?.includes(bucket));
  const bucketLocation = bucket.includes("eu")
    ? "europe-west4"
    : bucket.includes("multi") || bucket.includes("us-multi")
      ? "us"
      : "us-central2";
  const locationType = bucketLocation === "us" ? "multi-region" : "region";
  const isSame =
    bucketLocation === hostRegion || (!input.strict && locationType === "multi-region" && hostRegion.startsWith("us-"));
  const allowed = allowlisted || isSame;
  return {
    uri: input.uri,
    bucket,
    host_region: hostRegion,
    bucket_location: bucketLocation,
    location_type: locationType,
    allowed,
    allowlisted,
    reason: allowlisted
      ? `Bucket ${bucket} is explicitly allowlisted`
      : allowed
        ? `Bucket ${bucket} (${bucketLocation}) is local to ${hostRegion}`
        : `Cross-region transfer: bucket ${bucket} is in ${bucketLocation}, outside ${hostRegion}`,
  };
}

export function createMockWebhook(input: WebhookCreateInput, orgId = "demo", projectId = "rl"): WebhookSubscription {
  const record: WebhookSubscription = {
    id: `whk-${Math.random().toString(16).slice(2, 8)}`,
    org_id: orgId,
    project_id: projectId,
    url: input.url,
    entity_types: [...(input.entity_types ?? [])],
    actions: [...(input.actions ?? [])],
    enabled: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  mockWebhooks.unshift(record);
  return cloneMock(record);
}

export function registerMockProject(input: ProjectRegisterInput): ProjectRecord {
  const now = new Date().toISOString();
  const existing = mockProjects.find((p) => p.org_id === input.orgId && p.id === input.projectId);
  const record: ProjectRecord = {
    org_id: input.orgId,
    id: input.projectId,
    display_name: input.displayName,
    owner: input.owner,
    labels: input.labels ?? {},
    enabled: input.enabled ?? true,
    created_at: existing?.created_at ?? now,
    updated_at: now,
    created_by: "dashboard-demo",
  };
  if (existing) Object.assign(existing, record);
  else mockProjects.push(record);
  return cloneMock(record);
}

export function issueMockRoleGrant(input: RoleGrantIssueInput): DurableRoleGrant {
  const grant: DurableRoleGrant = {
    subject: input.subject,
    grant: {
      role: input.role,
      org_id: input.role === "fleet_admin" ? null : (input.orgId ?? null),
      project_id: input.role === "fleet_admin" || input.role === "org_admin" ? null : (input.projectId ?? null),
    },
    reason: input.reason ?? "",
    granted_by: "dashboard-demo",
    granted_at: new Date().toISOString(),
  };
  const idx = mockRoleGrants.findIndex(
    (g) =>
      g.subject === grant.subject &&
      g.grant.role === grant.grant.role &&
      g.grant.org_id === grant.grant.org_id &&
      g.grant.project_id === grant.grant.project_id,
  );
  if (idx >= 0) mockRoleGrants[idx] = grant;
  else mockRoleGrants.unshift(grant);
  return cloneMock(grant);
}
