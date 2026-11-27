// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { TpuQuotaFailure } from "../types";

export type ErrorTone = "danger" | "warning" | "info";

export type DiagnosticCode =
  | "permission_denied"
  | "auth_required"
  | "api_disabled"
  | "project_not_found"
  | "timeout"
  | "gcloud_missing"
  | "tracker_unconfigured"
  | "connection_refused"
  | "not_found"
  | "conflict"
  | "server_error"
  | "demo_unavailable"
  | "quota_read_failed"
  | "unknown";

export interface DiagnosticBadge {
  label: string;
  value: string;
}

export interface ParsedError {
  code: DiagnosticCode;
  tone: ErrorTone;
  title: string;
  summary: string;
  raw: string;
  hasTechnicalDetails: boolean;
  httpStatus?: number;
  project?: string;
  account?: string;
  permission?: string;
  service?: string;
  remediation?: string;
  command?: string;
  badges: DiagnosticBadge[];
}

export interface DiagnoseContext {
  project?: string | null;
  scope?: string;
  httpStatus?: number;
  fallbackTitle?: string;
  /** Structured fields from backend responses when available. */
  code?: string | null;
  summary?: string | null;
  permission?: string | null;
  account?: string | null;
  remediation?: string | null;
}

export interface GroupedQuotaFailure {
  key: string;
  classes: Array<TpuQuotaFailure["capacity_class"]>;
  scopeLabel: string;
  diagnostic: ParsedError;
}

const CAPACITY_CLASS_ORDER: Record<TpuQuotaFailure["capacity_class"], number> = {
  spot: 0,
  on_demand: 1,
  reserved: 2,
};

const CAPACITY_CLASS_LABELS: Record<TpuQuotaFailure["capacity_class"], string> = {
  spot: "Spot",
  on_demand: "On-demand",
  reserved: "Reserved",
};

/**
 * Strip Google RPC protobuf/YAML/base64 diagnostic noise (`@type`, `ErrorInfo`,
 * `DebugInfo`, `error_info_id`, `troubleshooter_url`, stack traces) so primary
 * UI summaries stay readable in one scan.
 */
export function stripRawNoise(text: string): string {
  let cleaned = text.replace(/\r\n/g, "\n").trim();
  // Remove Google RPC trailing structured metadata blocks (`- '@type': type.googleapis.com/...` or JSON-ish blocks).
  cleaned = cleaned.replace(/\s*(?:-\s*)?['"]?@type['"]?\s*:\s*type\.googleapis\.com\/[\s\S]*$/i, "");
  // Remove inline base64 error_info_id / troubleshooter_url / links blocks if they appear before @type.
  cleaned = cleaned.replace(/\berror_info_id\s*:\s*\S+/gi, "");
  cleaned = cleaned.replace(/\btroubleshooter_url\s*:\s*\S+/gi, "");
  cleaned = cleaned.replace(/\bhttps?:\/\/console\.cloud\.google\.com\/[^\s)\]"']+/gi, "");
  // Strip verbose gcloud command prefix like `ERROR: (gcloud.beta.quotas.info.list) `.
  cleaned = cleaned.replace(/^gcloud\s+[^:]+?\s+failed:\s*/i, "");
  cleaned = cleaned.replace(/^ERROR:\s*\(gcloud\.[^)]+\)\s*/i, "");
  // Collapse whitespace and trim trailing punctuation clutter.
  cleaned = cleaned.replace(/\s+/g, " ").trim();
  cleaned = cleaned.replace(/[\s,-]+$/, "").trim();
  return cleaned;
}

function extractHttpStatus(raw: string, explicit?: number): number | undefined {
  if (typeof explicit === "number" && Number.isFinite(explicit)) return explicit;
  const match = /\b(?:failed with|HTTP|status|answered)\s+(\d{3})\b/i.exec(raw);
  if (match) return Number(match[1]);
  return undefined;
}

function extractPermission(raw: string): string | undefined {
  const quoted = /Permission\s+['"]([^'"]+)['"]\s+denied/i.exec(raw);
  if (quoted) return quoted[1];
  const yamlField = /\bpermission\s*:\s*['"]?([a-z][a-z0-9._-]+)['"]?/i.exec(raw);
  if (yamlField) return yamlField[1];
  const lacks = /\blacks?\s+([a-z_]+(?:\.[a-z_]+)*)\s+permission\b/i.exec(raw);
  if (lacks) return lacks[1];
  return undefined;
}

function extractAccount(raw: string): string | undefined {
  const authAs = /authenticated as\s+([^\s,;'"()]+)/i.exec(raw);
  if (authAs) return authAs[1].replace(/[.:]+$/, "");
  const email = /\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b/.exec(raw);
  if (email) return email[1].replace(/[.:]+$/, "");
  return undefined;
}

function extractProject(raw: string, fallback?: string | null): string | undefined {
  const resource = /\/projects\/([a-z][a-z0-9-]{2,62})\b/i.exec(raw);
  if (resource) return resource[1];
  const flag = /--project(?:=|\s+)['"]?([a-z][a-z0-9-]{2,62})['"]?/i.exec(raw);
  if (flag) return flag[1];
  const named = /\bproject\s+['"`]([a-z][a-z0-9-]{2,62})['"`]/i.exec(raw);
  if (named) return named[1];
  const trimmed = fallback?.trim();
  return trimmed || undefined;
}

function extractService(raw: string): string | undefined {
  const fromPath = /\/services\/([a-z0-9.-]+\.googleapis\.com)\b/i.exec(raw);
  if (fromPath) return fromPath[1];
  const fromApi = /\bAPI\s+\[([a-z0-9.-]+\.googleapis\.com)\]/i.exec(raw);
  if (fromApi) return fromApi[1];
  const fromDomain = /\bdomain\s*:\s*([a-z0-9.-]+\.googleapis\.com)\b/i.exec(raw);
  if (fromDomain) return fromDomain[1];
  return undefined;
}

function extractCommandFromText(text: string | null | undefined): string | undefined {
  if (!text) return undefined;
  const backtick = /`((?:gcloud|eyvan)\s+[^`]+)`/.exec(text);
  if (backtick) return backtick[1].trim();
  const bare = /\b((?:gcloud|eyvan)\s+(?:[a-z0-9._:=/"'@-]+\s*){1,14})/i.exec(text);
  if (bare) return bare[1].trim().replace(/[.,;]+$/, "");
  return undefined;
}

function normalizeCode(rawCode?: string | null): DiagnosticCode | undefined {
  if (!rawCode) return undefined;
  const key = rawCode.trim().toLowerCase();
  const known: Record<string, DiagnosticCode> = {
    permission_denied: "permission_denied",
    auth_required: "auth_required",
    api_disabled: "api_disabled",
    project_not_found: "project_not_found",
    timeout: "timeout",
    gcloud_missing: "gcloud_missing",
    tracker_unconfigured: "tracker_unconfigured",
    connection_refused: "connection_refused",
    not_found: "not_found",
    conflict: "conflict",
    server_error: "server_error",
    demo_unavailable: "demo_unavailable",
    quota_read_failed: "quota_read_failed",
  };
  return known[key];
}

function isGcpContext(raw: string, permission?: string, service?: string): boolean {
  return Boolean(
    permission?.includes(".") ||
      service ||
      /\bgcloud\b|googleapis\.com|cloudquotas|IAM_PERMISSION_DENIED|Cloud Monitoring|Cloud Quotas/i.test(raw),
  );
}

function hasRawProtobufOrStackNoise(raw: string): boolean {
  return (
    raw.length > 180 ||
    raw.includes("\n") ||
    /@type|type\.googleapis\.com|error_info_id|troubleshooter_url|Traceback \(most recent|ErrorInfo|DebugInfo/i.test(
      raw,
    )
  );
}

function buildBadges(parts: {
  scope?: string;
  project?: string;
  account?: string;
  permission?: string;
  service?: string;
  httpStatus?: number;
}): DiagnosticBadge[] {
  const badges: DiagnosticBadge[] = [];
  if (parts.scope) badges.push({ label: "Scope", value: parts.scope });
  if (parts.project) badges.push({ label: "Project", value: parts.project });
  if (parts.account) badges.push({ label: "Account", value: parts.account });
  if (parts.permission) badges.push({ label: "Permission", value: parts.permission });
  if (parts.service && !parts.permission) badges.push({ label: "API", value: parts.service });
  if (parts.httpStatus !== undefined) badges.push({ label: "HTTP", value: String(parts.httpStatus) });
  return badges;
}

/**
 * Parse any backend, gcloud, HTTP, or runtime error into a crisp, actionable
 * engineering diagnostic. Understands both structured backend fields and raw
 * legacy stderr strings.
 */
export function diagnoseError(input: unknown, context: DiagnoseContext = {}): ParsedError {
  const rawMessage =
    typeof input === "string"
      ? input
      : input instanceof Error
        ? input.message
        : input !== null && typeof input === "object" && "message" in input
          ? String((input as { message: unknown }).message)
          : String(input ?? "Unexpected error");

  const raw = rawMessage.trim() || "Unexpected error";
  const errorStatus =
    input !== null &&
    typeof input === "object" &&
    "status" in input &&
    typeof (input as { status: unknown }).status === "number"
      ? (input as { status: number }).status
      : undefined;

  const httpStatus = extractHttpStatus(raw, context.httpStatus ?? errorStatus);
  const permission = context.permission?.trim() || extractPermission(raw);
  const account = context.account?.trim() || extractAccount(raw);
  const project = extractProject(raw, context.project);
  const service = extractService(raw);
  const gcp = isGcpContext(raw, permission, service);
  const explicitCode = normalizeCode(context.code);
  const structuredSummary = context.summary ? stripRawNoise(context.summary) : undefined;
  const structuredRemediation = context.remediation?.trim() || undefined;
  const cleaned = stripRawNoise(raw);

  // Detect diagnostic code from structured field first, then from raw patterns.
  let code: DiagnosticCode = explicitCode ?? "unknown";
  if (code === "unknown") {
    if (/unavailable in demo mode/i.test(raw)) {
      code = "demo_unavailable";
    } else if (/no tracker configured/i.test(raw) || (httpStatus === 503 && /tracker/i.test(raw))) {
      code = "tracker_unconfigured";
    } else if (/SERVICE_DISABLED|API has not been used|not enabled on project|Enable it by visiting/i.test(raw)) {
      code = "api_disabled";
    } else if (
      /PERMISSION_DENIED|IAM_PERMISSION_DENIED|Permission\s+['"][^'"]+['"]\s+denied/i.test(raw) ||
      httpStatus === 403
    ) {
      code = "permission_denied";
    } else if (
      /UNAUTHENTICATED|gcloud auth|token lookup failed|Reauthentication failed|credentials? (?:not found|expired)|not authenticated|login required/i.test(
        raw,
      ) ||
      httpStatus === 401
    ) {
      code = "auth_required";
    } else if (/gcloud.*not found|No such file or directory.*gcloud|ENOENT.*gcloud/i.test(raw)) {
      code = "gcloud_missing";
    } else if (/timed out|did not answer within|DEADLINE_EXCEEDED|ETIMEDOUT/i.test(raw) || httpStatus === 504) {
      code = "timeout";
    } else if (
      /Unable to reach|Failed to fetch|ECONNREFUSED|Connection refused|NetworkError|ERR_CONNECTION|unreachable/i.test(
        raw,
      )
    ) {
      code = "connection_refused";
    } else if (gcp && /NOT_FOUND|project .* not found|does not exist/i.test(raw)) {
      code = "project_not_found";
    } else if (httpStatus === 404) {
      code = "not_found";
    } else if (httpStatus === 409) {
      code = "conflict";
    } else if (httpStatus !== undefined && httpStatus >= 500) {
      code = "server_error";
    }
  }

  let tone: ErrorTone = "danger";
  let title = context.fallbackTitle ?? "Request failed";
  let summary = structuredSummary || cleaned || raw;
  let remediation = structuredRemediation;
  let command = extractCommandFromText(structuredRemediation);

  switch (code) {
    case "demo_unavailable": {
      tone = "info";
      title = "Unavailable in demo mode";
      summary = structuredSummary || raw;
      break;
    }
    case "tracker_unconfigured": {
      tone = "info";
      title = "No tracker configured";
      summary =
        structuredSummary ||
        "Start a local tracker on this machine, or set EYVAN_TRACKER_URL on the control plane, then retry.";
      remediation = remediation ?? "Run the local tracking server or open the dashboard via the CLI:";
      command = command ?? "eyvan track up --local";
      break;
    }
    case "permission_denied": {
      tone = "warning";
      if (gcp) {
        title = "GCP IAM permission denied";
        if (!structuredSummary) {
          const subject = account ? `Account ${account}` : "The active gcloud account";
          const permPart = permission ? ` lacks ${permission}` : " does not have permission to read quotas";
          const projPart = project ? ` on project ${project}` : "";
          summary = `${subject}${permPart}${projPart}.`;
        }
        remediation =
          remediation ??
          "Switch to another GCP project this account can read, switch gcloud accounts, or grant Cloud Quotas Viewer.";
        if (!command && project) {
          const member = account ? `user:${account}` : "user:<email>";
          command = `gcloud projects add-iam-policy-binding ${project} --member="${member}" --role="roles/cloudquotas.viewer"`;
        }
      } else {
        title = "Permission denied";
        if (!structuredSummary) {
          summary = raw.replace(/^Eyvan API request failed with 403:\s*/i, "");
        }
        remediation = remediation ?? "Verify your project role or fleet permissions with a fleet administrator.";
      }
      break;
    }
    case "auth_required": {
      tone = "warning";
      if (gcp) {
        title = "Google Cloud authentication required";
        if (!structuredSummary) {
          summary = account
            ? `Credentials for ${account} could not be refreshed or are missing.`
            : "The server's gcloud session could not mint an access token.";
        }
        remediation = remediation ?? "Sign in with gcloud on the control-plane host, then re-read the provider.";
        command = command ?? "gcloud auth login";
      } else {
        title = "Authentication required";
        if (!structuredSummary) {
          summary = raw.replace(/^Eyvan API request failed with 401:\s*/i, "");
        }
        remediation = remediation ?? "Re-open the dashboard from your terminal to refresh the local credential.";
        command = command ?? "eyvan dashboard";
      }
      break;
    }
    case "api_disabled": {
      tone = "warning";
      title = "Google Cloud API disabled";
      const apiName = service ?? "cloudquotas.googleapis.com";
      if (!structuredSummary) {
        summary = project
          ? `The ${apiName} API is not enabled on project ${project}.`
          : `The ${apiName} API is not enabled on the target GCP project.`;
      }
      remediation = remediation ?? `Enable ${apiName} on this project, then re-read the provider.`;
      command =
        command ??
        (project ? `gcloud services enable ${apiName} --project=${project}` : `gcloud services enable ${apiName}`);
      break;
    }
    case "project_not_found": {
      tone = "warning";
      title = "GCP project not found";
      if (!structuredSummary) {
        summary = project
          ? `Project ${project} could not be found or is inaccessible to the active account.`
          : "The requested GCP project could not be found.";
      }
      remediation = remediation ?? "Check the project ID or switch to a project declared by this fleet.";
      break;
    }
    case "gcloud_missing": {
      tone = "warning";
      title = "gcloud CLI unavailable";
      if (!structuredSummary) {
        summary = "The control-plane host could not run the gcloud CLI needed for provider fallback.";
      }
      remediation = remediation ?? "Install the Google Cloud SDK on the control-plane host and authenticate.";
      command = command ?? "gcloud auth login";
      break;
    }
    case "timeout": {
      tone = "warning";
      title = "Request timed out";
      if (!structuredSummary) {
        summary = hasRawProtobufOrStackNoise(raw) ? "The upstream service did not respond before the deadline." : raw;
      }
      remediation = remediation ?? "The provider or plane may still be finishing a cold read. Wait a moment and retry.";
      break;
    }
    case "connection_refused": {
      tone = "danger";
      title = "Control plane unreachable";
      if (!structuredSummary) {
        summary = hasRawProtobufOrStackNoise(raw) ? "Could not connect to the Eyvan API server." : raw;
      }
      remediation = remediation ?? "Confirm the Eyvan server is running and reachable, then retry.";
      command = command ?? "eyvan dashboard";
      break;
    }
    case "not_found": {
      tone = "warning";
      title = "Resource not found";
      if (!structuredSummary) {
        summary = raw.replace(/^Eyvan API request failed with 404:\s*/i, "");
      }
      break;
    }
    case "conflict": {
      tone = "warning";
      title = "State conflict";
      if (!structuredSummary) {
        summary = raw.replace(/^Eyvan API request failed with 409:\s*/i, "");
      }
      break;
    }
    case "server_error": {
      tone = "danger";
      title = httpStatus === 503 ? "Service unavailable" : "Control-plane error";
      if (!structuredSummary) {
        summary = hasRawProtobufOrStackNoise(raw) ? `The server answered with HTTP ${httpStatus ?? 500}.` : raw;
      }
      remediation = remediation ?? "Retry in a few seconds or inspect the control-plane logs if this persists.";
      break;
    }
    case "quota_read_failed": {
      tone = "warning";
      title = "Provider quota read failed";
      if (!structuredSummary) {
        summary = cleaned.length > 200 ? `${cleaned.slice(0, 197).trimEnd()}…` : cleaned;
      }
      break;
    }
    default: {
      tone = "danger";
      if (!structuredSummary && hasRawProtobufOrStackNoise(raw)) {
        summary = cleaned.length > 200 ? `${cleaned.slice(0, 197).trimEnd()}…` : cleaned;
      }
      break;
    }
  }

  // Only show the collapsible "Technical details" block when there is meaningful
  // additional raw detail beyond the displayed summary.
  const hasTechnicalDetails =
    hasRawProtobufOrStackNoise(raw) ||
    (raw.toLowerCase() !== summary.toLowerCase() && !summary.toLowerCase().includes(raw.toLowerCase()));

  const badges = buildBadges({
    scope: context.scope,
    project,
    account,
    permission,
    service,
    httpStatus: httpStatus !== undefined && !summary.includes(String(httpStatus)) ? httpStatus : undefined,
  });

  return {
    code,
    tone,
    title,
    summary,
    raw,
    hasTechnicalDetails,
    httpStatus,
    project,
    account,
    permission,
    service,
    remediation,
    command,
    badges,
  };
}

export function formatCapacityClasses(classes: ReadonlyArray<TpuQuotaFailure["capacity_class"]>): string {
  const unique = [...new Set(classes)].sort((a, b) => CAPACITY_CLASS_ORDER[a] - CAPACITY_CLASS_ORDER[b]);
  const labels = unique.map((item) => CAPACITY_CLASS_LABELS[item] ?? item);
  if (labels.length === 0) return "Quota";
  if (labels.length === 1) return labels[0];
  if (labels.length === 2) return `${labels[0]} & ${labels[1]}`;
  return `${labels.slice(0, -1).join(", ")} & ${labels[labels.length - 1]}`;
}

function dedupeKeyForQuotaFailure(diagnostic: ParsedError): string {
  if (
    diagnostic.code === "permission_denied" ||
    diagnostic.code === "auth_required" ||
    diagnostic.code === "api_disabled" ||
    diagnostic.code === "project_not_found" ||
    diagnostic.code === "gcloud_missing"
  ) {
    return [
      diagnostic.code,
      diagnostic.project ?? "",
      diagnostic.account ?? "",
      diagnostic.permission ?? "",
      diagnostic.service ?? "",
      diagnostic.summary.toLowerCase(),
    ].join("|");
  }
  return [diagnostic.code, diagnostic.summary.toLowerCase(), stripRawNoise(diagnostic.raw).toLowerCase()].join("|");
}

/**
 * Parse and deduplicate provider quota failures so identical root-cause errors
 * across Spot and On-demand render as ONE clean card instead of repeated walls.
 */
export function groupQuotaFailures(
  failures: ReadonlyArray<TpuQuotaFailure>,
  options: { project?: string | null } = {},
): GroupedQuotaFailure[] {
  const groups = new Map<
    string,
    {
      classes: Array<TpuQuotaFailure["capacity_class"]>;
      first: ParsedError;
      rawByClass: Array<{ cls: TpuQuotaFailure["capacity_class"]; raw: string }>;
    }
  >();

  for (const failure of failures) {
    const parsed = diagnoseError(failure.reason, {
      project: options.project,
      code: failure.code,
      summary: failure.summary,
      permission: failure.permission,
      account: failure.account,
      remediation: failure.remediation,
      fallbackTitle: "Provider quota could not be read",
    });
    const key = dedupeKeyForQuotaFailure(parsed);
    const existing = groups.get(key);
    if (existing) {
      if (!existing.classes.includes(failure.capacity_class)) {
        existing.classes.push(failure.capacity_class);
      }
      existing.rawByClass.push({ cls: failure.capacity_class, raw: failure.reason.trim() });
    } else {
      groups.set(key, {
        classes: [failure.capacity_class],
        first: parsed,
        rawByClass: [{ cls: failure.capacity_class, raw: failure.reason.trim() }],
      });
    }
  }

  return [...groups.entries()].map(([key, group]) => {
    const sortedClasses = [...group.classes].sort((a, b) => CAPACITY_CLASS_ORDER[a] - CAPACITY_CLASS_ORDER[b]);
    const scopeLabel = formatCapacityClasses(sortedClasses);
    const uniqueRaw = [...new Set(group.rawByClass.map((entry) => entry.raw).filter(Boolean))];
    const combinedRaw =
      uniqueRaw.length === 1
        ? uniqueRaw[0]
        : group.rawByClass.map((entry) => `[${CAPACITY_CLASS_LABELS[entry.cls]}] ${entry.raw}`).join("\n\n");

    const baseSummary = group.first.summary.endsWith(".") ? group.first.summary : `${group.first.summary}.`;
    const lead = `${scopeLabel} quota could not be read, so no zone below reports it.`;
    const summary = baseSummary.toLowerCase().includes("quota could not be read")
      ? baseSummary
      : `${lead} ${baseSummary}`;

    const badges = buildBadges({
      scope: scopeLabel,
      project: group.first.project,
      account: group.first.account,
      permission: group.first.permission,
      service: group.first.service,
      httpStatus: group.first.httpStatus,
    });

    return {
      key,
      classes: sortedClasses,
      scopeLabel,
      diagnostic: {
        ...group.first,
        tone: group.first.tone === "info" ? "warning" : group.first.tone,
        summary,
        raw: combinedRaw,
        hasTechnicalDetails: Boolean(combinedRaw && (group.first.hasTechnicalDetails || combinedRaw !== summary)),
        badges,
      },
    };
  });
}
