// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

export interface AttachmentInput {
  target: string;
  kind: "ssh" | "tpu";
  fleet: string | null;
  project: string | null;
  zone: string | null;
}

export interface AttachmentPlan {
  dry_run: true;
  fleet: string;
  fleet_path: string;
  fleet_yaml: string;
  endpoints: string[];
  actions: Array<{ step: string; argv: string[]; command?: string; summary: string; mutates: boolean; stdin: boolean }>;
  limitations: string[];
  summary: { shape: string | null; hosts: number; capacity_class: string | null; endpoint_mode: string | null };
}

export interface AttachmentJob {
  id: string;
  target: string;
  status: "running" | "succeeded" | "failed";
  progress: string[];
  result: { fleet: string; ready: boolean; cells: unknown[]; submit: string; limitations: string[] } | null;
  error: string | null;
  started_at: string;
  finished_at: string | null;
}

export interface DiscoveredTpu {
  fleet: string;
  project: string;
  zone: string;
  name: string;
  node_id: string;
  accelerator_type: string;
  capacity_class: string;
  state: string;
  hosts: number;
  connected: boolean;
  /** What the connected fleet's cells are doing; null when the TPU is not connected. */
  cell_state?: "ready" | "connecting" | "failing" | "waiting" | null;
  last_error?: string | null;
  next_attempt_at?: string | null;
}

export interface DiscoveredTpus {
  discovered_at: number | null;
  /** The gcloud account the listing ran as; null or absent when it could not be named. */
  account?: string | null;
  projects: string[];
  tpus: DiscoveredTpu[];
  skipped: Record<string, string>;
}

export interface TeamUser {
  subject: string;
  email: string;
  display_name: string;
  disabled: boolean;
  created_at: string;
  created_by: string;
  /** Governance rules; a null field means "no rule". Older planes omit the whole object. */
  rules?: TeamUserRules;
  /** Derived by the plane: admin, trusted or user. Older planes omit it. */
  tier?: TeamTier;
  trusted?: boolean;
  /** Optional explicit override for idle-capacity borrowing; null falls back to tier default. */
  can_borrow?: boolean | null;
}

export type TeamTier = "admin" | "trusted" | "user";

export type ReclaimPolicy = "requeue" | "stop";

export interface TeamUserRules {
  allowed_fleets: string[] | null;
  approval_tpu_threshold: number | null;
  guaranteed_tpu_units: number | null;
  /** Hard cap on TPU units in use or queued; null means no limit. */
  max_tpu_units: number | null;
  max_cpu_units: number | null;
  max_gpu_units: number | null;
  /** Optional override for borrowing idle TPU capacity; null uses the account's tier default. */
  can_borrow?: boolean | null;
}

export interface GovernanceUserSummary {
  subject: string;
  tier?: TeamTier | string;
  guaranteed_tpu_units: number | null;
  max_tpu_units: number | null;
  can_borrow?: boolean | null;
  in_use_tpu_units: number;
  borrowed_tpu_units: number;
}

export interface ActiveReclaim {
  workload_id: string;
  owner: string;
  requested_by?: string | null;
  deadline_at?: string | null;
  policy?: ReclaimPolicy | null;
  state: string;
}

export interface GovernanceSummary {
  enabled: boolean;
  total_guaranteed_tpu_units: number;
  total_in_use_tpu_units: number;
  total_borrowed_tpu_units: number;
  users: GovernanceUserSummary[];
  active_reclaims: ActiveReclaim[];
}

export interface TeamRoleGrant {
  role: string;
  org_id: string | null;
  project_id: string | null;
}

export interface TeamInvite {
  id: string;
  email: string;
  subject: string;
  roles: TeamRoleGrant[];
  created_by: string;
  created_at: string;
  expires_at: string;
  accepted_at: string | null;
  revoked: boolean;
}

export interface TeamToken {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  expires_at: string | null;
  last_used_at: string | null;
  revoked_at: string | null;
}

export interface TeamInviteInput {
  email: string;
  subject: string;
  /** Send either a tier or a custom role, never both. */
  role?: string | null;
  tier?: TeamTier;
  org_id: string | null;
  project_id: string | null;
}

export interface TeamInviteCreated {
  invite: TeamInvite;
  url: string;
  token: string;
}

export interface TeamTokenCreated {
  token: string;
  record: TeamToken;
}

export interface AssistantStatus {
  enabled: boolean;
  model: string | null;
  providers?: number;
}

/** Provider keys are write-only; admin-only fields are omitted for other users. */
export interface AssistantProvider {
  id: string;
  name: string;
  models: string[];
  default_model: string;
  base_url?: string;
  max_steps?: number;
  enabled?: boolean;
  has_api_key?: boolean;
  /** The preset the provider was created from; null for hand-configured providers. */
  preset?: string | null;
  created_by?: string;
  created_at?: string;
  updated_at?: string;
}

export interface AssistantProviderList {
  providers: AssistantProvider[];
  can_manage: boolean;
}

/** A provider the plane knows how to talk to; admins pick one and paste a key. */
export interface AssistantPreset {
  id: string;
  label: string;
  /** Empty for `custom`, whose URL the admin types. */
  base_url: string;
  editable_url: boolean;
  key_required: boolean;
  /** Where an admin gets a key, when the provider has a page for it. */
  key_url: string | null;
  /** `claude-code` signs the plane host's Claude Code CLI in; no key is stored. */
  sign_in: "openrouter" | "claude-code" | null;
  suggested_models: string[];
}

/** What the plane host's Claude Code CLI reports about itself. */
export interface ClaudeCodeStatus {
  installed: boolean;
  version: string | null;
  logged_in: boolean;
  auth_method: string | null;
  subscription_type: string | null;
  detail: string | null;
}

/** The sign-in runs in a browser on the plane host; `url` is set when it can be opened from here. */
export interface ClaudeCodeLoginStart {
  started: boolean;
  url: string | null;
}

/** With a preset the server fills name, base URL and models it can infer. */
export interface AssistantProviderInput {
  preset?: string;
  name?: string;
  base_url?: string;
  api_key?: string;
  models?: string[];
  default_model?: string;
  max_steps?: number;
  enabled?: boolean;
}

export interface AssistantModelDiscoveryInput {
  preset?: string;
  base_url?: string;
  api_key?: string;
  /** Use this stored provider's key, so an edit never needs the key again. */
  provider_id?: string;
}

export interface AssistantModelDiscovery {
  models: string[];
  /** `suggested` when the live listing failed; `detail` then says why. */
  source: "live" | "suggested";
  detail: string | null;
}

export interface OpenRouterSignInInput {
  models?: string[];
  default_model?: string;
  name?: string;
}

export interface OpenRouterSignInStart {
  authorize_url: string;
  expires_at: string;
}

export type AssistantProviderUpdate = Partial<AssistantProviderInput> & { clear_api_key?: true };

export interface AssistantMessage {
  role: "user" | "assistant";
  content: string;
}

/** A submission the assistant prepared; nothing runs until the user confirms it. */
export interface AssistantDraft {
  org_id: string;
  project_id: string;
  spec: Record<string, unknown> & { name: string };
}

export interface TeamRequest {
  id: string;
  kind: "capacity" | "approval";
  subject: string;
  org_id: string | null;
  project_id: string | null;
  summary: string;
  details: Record<string, unknown>;
  state: "open" | "approved" | "declined" | "withdrawn";
  created_at: string;
  resolved_at: string | null;
  resolved_by: string | null;
  resolution_note: string;
}

export type ResearchPlanState =
  | "proposed"
  | "approved"
  | "running"
  | "stopping"
  | "finished"
  | "stopped"
  | "failed"
  | "over_budget";

/** What a research plan may spend; the plane enforces it, never the model. */
export interface ResearchBudget {
  max_jobs: number;
  max_chip_hours: number;
  max_wall_seconds: number;
  max_rounds: number | null;
}

export interface ResearchSpend {
  chip_hours: number;
  jobs_submitted: number;
  rounds: number;
}

/**
 * The list view of a research plan (`plan_summary`). A plan is never a
 * workload draft: it is approved as a whole and the plane submits its jobs.
 */
export interface ResearchPlanSummary {
  id: string;
  title: string;
  subject: string;
  org_id: string;
  project_id: string;
  state: ResearchPlanState;
  iterate: boolean;
  provider_id: string | null;
  model: string | null;
  current_round: number;
  jobs_planned: number;
  budget: ResearchBudget;
  spent: ResearchSpend;
  stop_reason: string | null;
  approved_at: string | null;
  approved_by: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string | null;
  updated_at: string | null;
}

/** One job of a plan with its workload's current (raw) state, `not_submitted`, `refused` or `unknown`. */
export interface ResearchPlanJob {
  index: number;
  name: string;
  round: number;
  spec: Record<string, unknown>;
  state: string;
  workload_id?: string;
  failure_reason?: string | null;
  error?: string | null;
  submitted_at?: string | null;
}

export interface ResearchPlanNote {
  t: string | null;
  kind: string;
  text: string;
}

/** One plan in full (`plan_document`). */
export interface ResearchPlan extends ResearchPlanSummary {
  goal: string;
  jobs: ResearchPlanJob[];
  workload_ids: string[];
  remaining: { jobs: number; chip_hours: number };
  notes: ResearchPlanNote[];
}

export interface AssistantTurn {
  reply: string;
  steps: number;
  tool_calls: Array<{ tool: string; ok: boolean }>;
  drafts: AssistantDraft[];
  requests: TeamRequest[];
  /** Research plans proposed this turn; each waits for its owner's approval. */
  plans?: ResearchPlanSummary[];
  provider_id?: string;
  model?: string;
}

/** One streamed chat turn; the plane fills org/project from the client's scope when omitted. */
export interface AssistantChatStreamInput {
  message: string;
  conversation_id?: string;
  org_id?: string;
  project_id?: string;
  provider_id?: string;
  model?: string;
}

/** One NDJSON line of `POST /api/v1/assistant/chat/stream`. */
export type AssistantStreamEvent =
  | { type: "start"; conversation_id: string; provider_id?: string; model?: string; title?: string }
  | { type: "step"; step: number; max_steps: number }
  | { type: "thinking"; delta: string }
  | { type: "text"; delta: string }
  | { type: "tool_call"; id: string; name: string; arguments: Record<string, unknown> | null }
  | { type: "tool_result"; id: string; name: string; ok: boolean; summary: string }
  | { type: "draft"; draft: AssistantDraft }
  | { type: "request"; request: TeamRequest }
  | { type: "plan"; plan: ResearchPlanSummary }
  | {
      type: "done";
      reply: string;
      steps: number;
      tool_calls: Array<{ tool: string; ok: boolean }>;
      drafts: AssistantDraft[];
      requests: TeamRequest[];
      plans?: ResearchPlanSummary[];
      message_id?: string;
    }
  | { type: "error"; detail: string };

export interface AssistantConversationSummary {
  id: string;
  title: string;
  provider_id: string | null;
  model: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

/** A tool call saved with an assistant message. */
export interface AssistantTraceEntry {
  type: "tool";
  name: string;
  ok: boolean;
  summary: string;
  arguments: Record<string, unknown> | null;
}

export interface AssistantConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  trace?: AssistantTraceEntry[];
  drafts?: AssistantDraft[];
  requests?: TeamRequest[];
  /** Omitted when the turn proposed no plan. */
  plans?: ResearchPlanSummary[];
  provider_id?: string | null;
  model?: string | null;
}

export interface AssistantConversation {
  id: string;
  title: string;
  provider_id: string | null;
  model: string | null;
  created_at: string;
  updated_at: string;
  messages: AssistantConversationMessage[];
}

export type RunStatus = "running" | "finished" | "failed" | "crashed" | "cancelled" | "killed" | "aborted" | "unknown";

/** Tracker wire types intentionally retain the store's snake_case field names. */
export interface RunSummary {
  run_id: string;
  org_id: string;
  project_id: string;
  name: string;
  group: string | null;
  tags: string[];
  status: RunStatus;
  workload_id: string | null;
  attempts: number;
  started_at: number;
  updated_at: number;
  last_step: number | null;
  keys_count: number;
  summary: Record<string, unknown>;
}

export interface RunDetail extends RunSummary {
  config: Record<string, unknown>;
  keys: string[];
  axes: string[];
  /** Additive phase-2 fields may be absent on an older tracker. */
  hist_keys?: string[];
  episodes?: { train: number; eval: number };
  episode_axes?: string[];
  envs?: string[];
  actors?: number;
  /** Additive phase-3 fields may be absent on an older tracker. */
  rollouts?: number;
  completion_groups?: number;
  table_keys?: string[];
  media_keys?: string[];
  attempt_list: Array<{
    attempt: number;
    started_at: number;
    first_step: number | null;
    last_step: number | null;
    status: string | null;
  }>;
  stats: {
    records: number;
    dropped: { keys: number; rate: number; invalid: number; queue: number };
    spilled: number;
  };
  errors: Array<{ segment: string; error: string }>;
}

export type TrackingPoint = [number, number | null, number | null, number | null];

export interface RunSeries {
  key: string;
  rank: number;
  points: TrackingPoint[];
  downsampled: boolean;
  total: number;
  nonfinite: number;
  attempts: Array<{ attempt: number; first_x: number }>;
  replayed: Array<{ attempt: number; points: Array<[number, number | null]> }>;
}

export interface RunSeriesResponse {
  run_id: string;
  x: string;
  series: RunSeries[];
}

export interface RunSeriesOptions {
  keys: string[];
  x?: string;
  points?: number;
  rank?: number;
  includeReplayed?: boolean;
}

export interface RunHistogramsOptions {
  key: string;
  x?: string;
  points?: number;
  bins?: number;
  rank?: number;
}

export interface HistogramColumn {
  x: number;
  step: number;
  attempt: number;
  counts: number[];
  n: number;
  under: number;
  over: number;
  nonfinite: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
}

export interface RunHistogramsResponse {
  run_id: string;
  key: string;
  rank: number;
  x: string;
  edges: number[];
  columns: HistogramColumn[];
  total: number;
  downsampled: boolean;
}

export type EpisodeKind = "train" | "eval";
export type TrackingValue = number | "nan" | "inf" | "-inf";

export interface RunEpisode {
  t: number;
  step: number | null;
  ret: TrackingValue;
  length: number | null;
  success: boolean | null;
  kind: EpisodeKind;
  actor: string | null;
  env: string | null;
  seed: number | null;
  policy_version: number | null;
  learner_version: number | null;
  checkpoint: string | null;
  x: Record<string, number>;
  extra: Record<string, TrackingValue>;
  attempt: number;
  rank: number;
  replayed: boolean;
}

export interface RunEpisodesOptions {
  kind?: EpisodeKind;
  actor?: string;
  env?: string;
  limit?: number;
  offset?: number;
  includeReplayed?: boolean;
}

export interface RunEpisodesResponse {
  episodes: RunEpisode[];
  total: number;
}

export interface EpisodeCurveOptions {
  metric?: "return" | "length";
  kind?: EpisodeKind;
  x?: string;
  points?: number;
  env?: string;
  actor?: string;
}

export interface EpisodeBucket {
  x: number;
  x_min: number;
  x_max: number;
  count: number;
  mean: number | null;
  min: number | null;
  max: number | null;
  p5: number | null;
  p25: number | null;
  p50: number | null;
  p75: number | null;
  p95: number | null;
  success_rate: number | null;
}

export interface EpisodeCurveResponse {
  run_id: string;
  metric: "return" | "length";
  kind: EpisodeKind;
  x: string;
  buckets: EpisodeBucket[];
  total: number;
  nonfinite: number;
  replayed_excluded: number;
}

export interface EpisodeActor {
  actor: string | null;
  count: number;
  window: number;
  mean_return: number | null;
  p50_return: number | null;
  last_return: number | null;
  mean_length: number | null;
  success_rate: number | null;
  mean_lag: number | null;
  max_lag: number | null;
  last_t: number;
}

export interface EpisodeEval {
  checkpoint: string | null;
  policy_version: number | null;
  step: number | null;
  count: number;
  mean_return: number | null;
  std_return: number | null;
  min_return: number | null;
  max_return: number | null;
  success_rate: number | null;
  mean_length: number | null;
  first_t: number;
  best: boolean;
}

export interface EpisodeBreakdownOptions {
  kind?: EpisodeKind;
  last?: number;
}

export interface EpisodeBreakdownResponse {
  actors: EpisodeActor[];
  envs: Array<{ env: string | null; count: number; mean_return: number | null; success_rate: number | null }>;
  lag: {
    counts: number[];
    over: number;
    negative: number;
    count: number;
    p50: number | null;
    p95: number | null;
    max: number | null;
  };
  evals: EpisodeEval[];
  replayed_excluded: number;
}

export interface BlobRef {
  sha256: string;
  content_type: string;
  bytes: number;
  available?: boolean;
}

export interface RunRolloutSummary {
  id: string;
  t: number;
  step: number | null;
  episode_id: string;
  actor: string | null;
  env: string | null;
  kind: EpisodeKind;
  policy_version: number | null;
  length: number;
  track_names: string[];
  action_names: string[] | null;
  ret: number | null;
  video: BlobRef | null;
  fps: number | null;
  x: Record<string, number>;
  attempt: number;
  rank: number;
  seq: number;
  line: number;
  replayed: boolean;
}
export interface RunRollout extends Omit<RunRolloutSummary, "track_names"> {
  tracks: Record<string, Array<TrackingValue | null>>;
  actions: Array<number | null> | null;
}
export interface RunRolloutsOptions {
  kind?: EpisodeKind;
  actor?: string;
  limit?: number;
  offset?: number;
  includeReplayed?: boolean;
}
export interface RunRolloutsResponse {
  rollouts: RunRolloutSummary[];
  total: number;
}
export type CompletionSort = "recent" | "zero_variance" | "reward";
export interface RunCompletionsOptions {
  step?: number;
  limit?: number;
  offset?: number;
  sort?: CompletionSort;
  includeReplayed?: boolean;
}
export interface CompletionSummary {
  id: string;
  group: string;
  step: number | null;
  t: number;
  prompt: string;
  k: number;
  mean_reward: number | null;
  std_reward: number | null;
  pass_rate: number | null;
  zero_variance: boolean;
  mean_tokens: number | null;
  truncated: number;
  attempt: number;
  rank: number;
  replayed: boolean;
}
export interface CompletionResponse {
  text: string;
  rewards: Record<string, TrackingValue>;
  reward: TrackingValue;
  advantage: TrackingValue;
  tokens: number | null;
  kl: number | null;
  truncated: boolean | null;
}
export interface RunCompletion extends CompletionSummary {
  answer: string | null;
  policy_version: number | null;
  responses: CompletionResponse[];
}
export interface RunCompletionsResponse {
  groups: CompletionSummary[];
  total: number;
  steps: number[];
}
export interface CompletionStatsOptions {
  points?: number;
}
export interface CompletionStatsResponse {
  steps: Array<{
    step: number;
    groups: number;
    mean_reward: number | null;
    zero_variance_frac: number;
    mean_tokens: number | null;
    truncated_frac: number;
    components: Record<string, number | null>;
  }>;
}
export type TableCell = TrackingValue | string | boolean | null | BlobRef;
export interface RunTableOptions {
  key?: string;
  step?: number;
}
export interface RunTable {
  key: string;
  step: number | null;
  t: number;
  columns: string[];
  rows: TableCell[][];
  steps: number[];
}
export interface RunTableKeys {
  keys: Array<{ key: string; count: number; last_step: number | null }>;
}
export type RunTableResponse = RunTable | RunTableKeys;
export interface RunMediaOptions {
  key?: string;
  limit?: number;
  offset?: number;
}
export interface RunMediaItem {
  key: string;
  step: number | null;
  t: number;
  caption: string | null;
  blob: BlobRef;
}
export interface RunMediaResponse {
  media: RunMediaItem[];
  total: number;
}

export type WorkloadStatus =
  | "queued"
  | "planning"
  | "provisioning"
  | "running"
  | "checkpointing"
  | "preempting"
  | "completed"
  | "cancelled"
  | "failed";

export type PriorityBand = "production" | "interactive" | "batch";

export type Severity = "info" | "success" | "warning" | "critical";

export type SessionPermission =
  | "read"
  | "submit"
  | "operate"
  | "manage_project"
  | "manage_own_capacity"
  | "manage_fleet";

export interface EffectiveProjectScope {
  orgId: string;
  projectId: string;
  displayName: string;
  registered: boolean;
  enabled: boolean;
  /** Server truth: false for disabled or registry-invalid scopes, true for legacy scopes. */
  acceptsNewWork: boolean;
  roles: string[];
  permissions: SessionPermission[];
}

export interface SessionAuthority {
  subject: string;
  displayName?: string;
  globalPermissions: SessionPermission[];
  scopes: EffectiveProjectScope[];
}

export interface ResourceRequest {
  cpu: number;
  memoryGb: number;
  accelerator?: string;
  /** Resource dimension, independent of the provider's accelerator model name. */
  acceleratorKind?: "gpu" | "tpu";
  acceleratorCount?: number;
  replicas: number;
  slices?: number;
  topology?: string;
}

export interface WorkloadSummary {
  id: string;
  name: string;
  org: string;
  project: string;
  user: string;
  status: WorkloadStatus;
  priority: PriorityBand;
  resources: ResourceRequest;
  cell?: string;
  submittedAt: string;
  startedAt?: string;
  progress?: number;
  /** Per-workload spend is not reported by the control plane; only demo data sets these. */
  budgetSpent?: number;
  budgetLimit?: number;
  queueReason?: string;
  /** Capacity-acquisition phase from the control plane's progress contract. */
  phase?: string;
  /**
   * Which attempt is running, and what the earlier ones were charged to.
   *
   * The timeline carries the events, but counting them by hand is not the
   * question an operator has in front of a spot job: it is whether this job is
   * about to run out of retries, and whether what keeps killing it is the cloud
   * taking the capacity back or the pool failing to give any. Those are
   * separate budgets, and only one of them is the expected ending of a spot job.
   *
   * Optional throughout, because a plane that predates the split reports
   * neither, and absent must render as unknown rather than as zero.
   */
  attempt?: number;
  preemptionCount?: number;
  capacityLossCount?: number;
  failureReason?: string;
  /** ISO timestamp when an active graceful reclaim checkpoint window ends. */
  reclaimDeadlineAt?: string | null;
  /** Principal who requested reclamation of borrowed capacity. */
  reclaimRequestedBy?: string | null;
  /** Post-checkpoint action for a reclaimed job: requeue elsewhere or stop cleanly. */
  reclaimPolicy?: ReclaimPolicy | null;
  /** Optional configured checkpoint grace window in seconds for progress scaling. */
  checkpointGraceSeconds?: number;
}

export interface TimelineEvent {
  id: string;
  timestamp: string;
  title: string;
  detail: string;
  severity: Severity;
  actor?: string;
}

export interface LogEntry {
  id: string;
  /** Set when consecutive ranks printed this exact line: how many did. */
  repeat?: number;
  /** Present only when the log line carries a parseable timestamp. */
  timestamp?: string;
  /** Present only when the log line carries a recognized level token. */
  level?: "DEBUG" | "INFO" | "WARN" | "ERROR";
  /** Present only when the log line carries a bracketed source tag. */
  source?: string;
  /**
   * Which of the process's two streams produced the line.
   *
   * Populated only by the structured follow, because only it knows: the
   * one-shot text endpoint returns lines already merged, and inferring the
   * stream from the text would invent an attribution. Deliberately not folded
   * into `level` -- plenty of healthy programs write progress to stderr, and
   * rendering every one of those lines as an error is how a working job comes
   * to look like a failing one.
   */
  stream?: "stdout" | "stderr" | "unknown";
  message: string;
}

/**
 * What one run used and was charged, as the control plane totals it.
 *
 * `priced` is the field that decides whether `settled` means anything. An
 * unpriced pool settles to zero, so rendering `$0.00` would present the absence
 * of a price as the absence of a bill -- and that is the number somebody puts
 * in a budget.
 */
export interface WorkloadSpend {
  workloadId: string;
  state: string;
  /** Measured rather than priced, so known even when the money is not. */
  acceleratorHours: Record<string, number>;
  settled: number;
  priced: boolean;
  reserved: number;
  leases: number;
  shape?: string;
  capacityClass?: string;
  /** Absent unless somebody declared a price for the same shape on demand. */
  onDemandHourly?: number;
  /** Where that price came from, or why there is none. Always worth showing. */
  onDemandSource: string;
}

export interface WorkloadDetail extends WorkloadSummary {
  command: string;
  checkpointUri?: string;
  leaseId?: string;
  allocation: Array<{
    host: string;
    rack: string;
    resource: string;
    state: "ready" | "starting" | "draining";
  }>;
  timeline: TimelineEvent[];
  logs: LogEntry[];
  labels: Record<string, string>;
}

export interface OverviewData {
  queued: number;
  running: number;
  fleetUtilization: number;
  monthlySpend: number;
  monthlyBudget: number;
  successRate?: number;
  queueTrend: Array<{ label: string; submitted: number; admitted: number }>;
  priorityQueue: Array<{ priority: PriorityBand; count: number; oldestMinutes: number }>;
  fleetBreakdown: Array<{
    resource: string;
    used: number;
    total: number;
    pending: number;
  }>;
  alerts: Array<{
    id: string;
    severity: Severity;
    title: string;
    detail: string;
    timestamp: string;
  }>;
}

export interface Cell {
  id: string;
  name: string;
  backend: string;
  /** Optional cloud credential account owning this cell's provider project. */
  providerAccount?: string;
  region: string;
  zone: string;
  status: "healthy" | "starting" | "degraded" | "maintenance";
  /**
   * The control plane's own word for what this cell is doing.
   *
   * Kept beside `status` rather than folded into it, because the four statuses
   * are presentation tones and two distinct states share one of them: a cell
   * being drained and a cell being released both read as `maintenance`, and an
   * operator deciding what to do about a machine needs to know which. Absent in
   * demo data, which has no plane to have said anything.
   */
  state?: string;
  /** The capacity class the cell was provisioned as; absent on older records. */
  capacityClass?: "reserved" | "on_demand" | "spot";
  /**
   * Why the cell is not ready, in the plane's own words.
   *
   * `unavailableReason` is a closed code other paths compare on;
   * `unavailableDetail` is the provider's own diagnostic — the difference
   * between "provider_error" and the quota message that says which quota.
   * Present only when the plane has reported a cause; both absent on a
   * healthy or merely provisioning cell.
   */
  unavailableReason?: string;
  unavailableDetail?: string;
  utilization: number;
  activeWorkloads: number;
  /** The control plane does not report a per-cell queue depth; only demo data sets this. */
  queuedWorkloads?: number;
  /**
   * What the planner believes about this machine's zone, and why it may not be
   * asking for more there.
   *
   * A cell in cooldown is one the planner is deliberately not asking, which is
   * indistinguishable on every other view from a zone that simply has no
   * capacity. That difference is the whole content of a bad spot afternoon.
   */
  availability?: {
    cooldownUntil?: string;
    consecutiveFailures: number;
    /** Decayed count, so a reclamation an hour ago weighs less than one now. */
    interruptions: number;
  };
  topology: string;
  resources: Array<{
    name: string;
    used: number;
    total: number;
  }>;
  lastHeartbeat: string;
}

export interface CapacityLease {
  id: string;
  workloadId?: string;
  owner: string;
  cell: string;
  resource: string;
  amount: string;
  /**
   * The class of the cell this lease landed on, joined client-side.
   *
   * Narrowed to the server's real `CapacityClass`: the previous union invented
   * `borrowed` and `warm`, which no server enum has, so the value could never be
   * populated and the borrowed-capacity count was permanently zero.
   */
  class?: "reserved" | "on_demand" | "spot";
  state: "active" | "activating" | "revoking" | "expired";
  expiresAt: string;
  spendPerHour: number;
  resourceTotals?: Record<string, number>;
}

export interface Budget {
  id: string;
  owner: string;
  scope: "organization" | "project" | "user";
  spent: number;
  limit: number;
  forecast?: number;
  period: string;
  enforcement: "hard" | "soft";
}

export interface SchedulingPolicy {
  id: string;
  name: string;
  scope: string;
  type: "priority" | "fairness" | "preemption" | "placement";
  summary: string;
  /**
   * Whether this project's workloads may be preempted for higher-priority work.
   *
   * Named for what it controls. It was `enabled`, which read as "is this policy
   * in force" -- so a toggle labelled that way silently turned preemption off
   * for the project rather than disabling the policy.
   */
  preemptionEnabled: boolean;
  updatedAt: string;
  configuration?: {
    allowedProviders: string[];
    allowedRegions: string[];
    allowedCapacityClasses: string[];
    deniedAcceleratorTypes: string[];
    maxPriority: number;
  };
}

export interface AuditEvent {
  id: string;
  sequence?: number;
  timestamp: string;
  action: string;
  actor: string;
  target: string;
  category: "workload" | "capacity" | "policy" | "budget" | "system";
  severity: Severity;
  detail: string;
  requestId?: string;
}

export interface ControlPlaneHealth {
  status: string;
  service: string;
}

/**
 * One shape this scope may submit to, as the control plane advertises it.
 *
 * `shape` is the name a submission uses -- the accelerator type for a pool that
 * declares one, the pool's own node type otherwise -- which is exactly what
 * `eyvan run --tpu v6e-8` passes. The rest is what the pool declares, not
 * anything the submitter chooses: a pool supplies its own CPU, memory,
 * accelerator count and topology, which is why none of those are asked for.
 */
export interface AvailableShape {
  fleet: string;
  pool: string;
  shape: string;
  capacityClass: "spot" | "on_demand" | "reserved";
  region: string;
  zone?: string;
}

export interface SubmitWorkloadInput {
  /** Which applied fleet to run in. Chosen for you when there is only one. */
  fleet: string;
  /** The hardware to run on, as `eyvan run --tpu` spells it. */
  shape: string;
  /** Whole cells allocated together and atomically. The multislice knob. */
  pods: number;
  /**
   * A constraint, not a preference: with it set only a pool in that zone may
   * run this job, and cross-zone alternates are withheld.
   */
  zone?: string;
  name?: string;
  priority: PriorityBand;
  command: string;
  estimatedDurationSeconds: number;
  checkpointable: boolean;
  checkpointHook?: string;
  checkpointUri?: string;
  environment: Record<string, string>;
  labels: Record<string, string>;
}

export interface DashboardData {
  health: ControlPlaneHealth;
  overview: OverviewData;
  workloads: WorkloadSummary[];
  cells: Cell[];
  leases: CapacityLease[];
  budgets: Budget[];
  policies: SchedulingPolicy[];
  auditEvents: AuditEvent[];
  access: {
    /** True only when the fleet-wide cell inventory answered successfully. */
    fleetInventory: boolean;
    /** True only when the fleet-wide audit ledger answered successfully. */
    fleetAudit: boolean;
  };
}

/** Per-cell placement detail behind a workload's flat scheduling reason. */
export interface CellRejection {
  cell_id: string;
  reason: string;
}

export interface WorkloadExplanation {
  workload_id: string;
  state: string;
  placeable: boolean;
  placement_cell_id: string | null;
  code: string | null;
  message: string | null;
  cells: CellRejection[];
  considered_cell_count: number;
  /**
   * Cells the autoscaler is building for this workload.
   *
   * Load-bearing for the panel: without it a workload waiting on capacity looks
   * identical to one that can never be placed, because both report that no cell
   * satisfies their constraints and only one is a problem.
   */
  provisioning_cell_ids: string[];
  scheduling_reason: string | null;
}

export interface WorkloadProcessStatus {
  remote_id?: string | null;
  control_group?: string | null;
  pids?: number[];
  [key: string]: unknown;
}

export interface PoolWarmth {
  pool_id: string;
  floor: number;
  window_active: boolean;
  warm_cells: number;
  warm_idle_cells: number;
  /** Floor cells requested but not ready yet; absent from planes that predate it. */
  provisioning_cells?: number;
  hourly_cost: number;
}

/** One registered capacity pool, as the control plane holds it. */
export interface CapacityPoolSummary {
  id: string;
  fleetName: string;
  acceleratorType: string | null;
  capacityClass: string;
  zone: string | null;
  region: string | null;
  maxCells: number;
  minCells: number;
  enabled: boolean;
  /** Optional cloud provider name (e.g. "gcp"). */
  provider?: string | null;
  /** Optional provider project ID (e.g. "example-project-a"). */
  providerProject?: string | null;
  /** Optional credential account email/identity associated with this pool's project. */
  providerAccount?: string | null;
}

/** One person's settled usage in a project. */
export interface SubjectUsage {
  subject: string;
  workloads: number;
  settledWorkloads: number;
  leases: number;
  acceleratorHours: number;
  runningSeconds: number;
  settledCost: number;
  /** Leases on pools declaring no unit price. Their cost reads zero and is not. */
  unpricedLeases: number;
}

export interface ProjectUsage {
  orgId: string;
  projectId: string;
  subjects: SubjectUsage[];
}

export interface WarmthReport {
  observed_at: string;
  pools: PoolWarmth[];
}

export interface PoolLaunchLatency {
  pool_id: string;
  accelerator_type?: string | null;
  /** Null when the pool has never provisioned successfully -- not zero. */
  p95_seconds: number | null;
  median_seconds: number | null;
  slowest_seconds: number | null;
  samples: number;
  cells_in_cooldown: number;
}

export interface LaunchLatencyReport {
  observed_at: string;
  pools: PoolLaunchLatency[];
}

export interface BillingReport {
  observed_at: string;
  accrued_cost: number;
  idle_cost: number;
  /** Named rather than left to the index signature: the billing page reads
   *  them, and `unpriced_cells` is what separates "free" from "not costed". */
  cells: number;
  billed_seconds: number;
  idle_seconds: number;
  unpriced_cells: number;
  [key: string]: unknown;
}

/**
 * What one quota row is able to say about free capacity.
 *
 * `usage_unknown` is the reason this union exists rather than a nullable
 * number. Cloud Quotas reports a limit for a zone whether or not Cloud
 * Monitoring has a matching usage series, so a row rendered as zero usage
 * claims the whole quota is free -- the one wrong answer here that costs money
 * to act on.
 */
export type TpuQuotaState = "known" | "usage_unknown" | "unlimited";

export type TpuQuotaStatus = "ok" | "unavailable" | "unconfigured";

export interface TpuQuotaEntry {
  zone: string;
  region: string;
  generation: string;
  accelerator_type: string;
  capacity_class: "reserved" | "on_demand" | "spot";
  metric: string;
  /** Quota units one slice of this type consumes; the public accelerator suffix. */
  chips_per_slice: number;
  /** Null means the provider reports no ceiling -- never zero. */
  limit_chips: number | null;
  /** Null means nobody could measure it. Never render this as 0. */
  in_use_chips: number | null;
  free_chips: number | null;
  /** How many whole slices of this type the free balance fits. */
  free_slices: number | null;
  state: TpuQuotaState;
  usage_observed_at: string | null;
  /** How old the usage observation was when the snapshot was assembled. */
  usage_age_s: number | null;
}

export type TpuQuotaFailureCode =
  | "permission_denied"
  | "auth_required"
  | "api_disabled"
  | "project_not_found"
  | "timeout"
  | "gcloud_missing"
  | "quota_read_failed";

export interface TpuQuotaFailure {
  capacity_class: "reserved" | "on_demand" | "spot";
  reason: string;
  /** Optional structured diagnostic fields returned by newer control planes. */
  code?: TpuQuotaFailureCode | string | null;
  summary?: string | null;
  permission?: string | null;
  account?: string | null;
  remediation?: string | null;
}

export interface StorageGcpContext {
  on_gcp: boolean;
  project: string | null;
  zone: string | null;
  region: string | null;
  accelerator_type: string | null;
  source: string;
}

export interface StorageRegionalBucket {
  bucket: string;
  uri: string;
  location: string;
}

export interface StorageMonitoringBucket {
  bucket: string;
  uri: string;
  location: string;
  prefix: string;
}

export interface StorageStatus {
  configured: boolean;
  auto_create: boolean;
  config_path: string;
  project: string | null;
  monitoring: StorageMonitoringBucket | null;
  regional: Record<string, StorageRegionalBucket>;
  context: StorageGcpContext;
}

/** A fleet's dashboard-managed bash init script, as the init-script route answers. */
export interface FleetInitScriptDocument {
  fleet: string;
  /** The script; "" when the fleet has none. */
  script: string;
  /** False when the fleet's pools carry different scripts (hand-edited YAML). */
  consistent: boolean;
  /** False when a pool can never run one (attached TPUs); see `unsupported_reasons`. */
  supported: boolean;
  unsupported_reasons: string[];
  /** Init commands declared in the fleet YAML; they also run, and are read-only here. */
  other_init_commands: string[];
  pools: string[];
  max_bytes: number;
}

export interface TpuQuotaSnapshot {
  observed_at: string;
  provider: string;
  project: string | null;
  available_projects: string[];
  status: TpuQuotaStatus;
  /** When the provider was actually read; null when it never was. */
  refreshed_at: string | null;
  /** Distance between the provider read and this response. */
  age_s: number | null;
  cached: boolean;
  ttl_s: number;
  entries: TpuQuotaEntry[];
  failures: TpuQuotaFailure[];
}
