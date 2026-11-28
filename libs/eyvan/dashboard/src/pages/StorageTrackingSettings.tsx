// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import {
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Database,
  HardDrive,
  Package,
  RefreshCw,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

import { ApiError } from "../api/client";
import type { AuthScope } from "../auth/oidc";
import { ErrorCard } from "../components/ErrorCard";
import { StatusBadge } from "../components/StatusBadge";
import { formatDateTime } from "../lib/format";
import type {
  ManagedArtifact,
  RunSummary,
  StorageRegionCheckInput,
  StorageRegionCheckResult,
  StorageStatus,
} from "../types";

export interface StorageSettingsClient {
  getStorageStatus?: (signal?: AbortSignal) => Promise<StorageStatus | null>;
  listRuns?: (options?: { limit?: number }, scope?: AuthScope, signal?: AbortSignal) => Promise<RunSummary[]>;
  checkStorageRegion?: (input: StorageRegionCheckInput, signal?: AbortSignal) => Promise<StorageRegionCheckResult>;
  listManagedArtifacts?: (scope?: AuthScope, signal?: AbortSignal) => Promise<ManagedArtifact[]>;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

interface StorageTrackingSettingsProps {
  client: StorageSettingsClient;
  defaultOrg: string;
  defaultProject: string;
  onNotify?: (message: string) => void;
}

type TrackerHealth =
  | { state: "idle" }
  | { state: "checking" }
  | { state: "ok"; runCount: number }
  | { state: "unconfigured" }
  | { state: "error"; detail: string };

const CLI_COMMANDS: Array<{ id: string; label: string; description: string; command: string }> = [
  {
    id: "storage-init",
    label: "Initialize storage",
    description: "Create or link regional (eyvan_<region>) and monitoring GCS buckets",
    command: "eyvan storage init",
  },
  {
    id: "track-pull",
    label: "Live-tail remote runs",
    description: "Sync GCS tracking segments into ~/.eyvan/tracking every 5s",
    command: "eyvan track pull --watch 5",
  },
  {
    id: "track-up",
    label: "Local tracker",
    description: "Run the local experiment tracker bound to ~/.eyvan/tracking",
    command: "eyvan track up --local",
  },
];

async function copyText(value: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Fall back when Clipboard API is restricted.
  }
  try {
    const area = document.createElement("textarea");
    area.value = textFallback(value);
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(area);
    return ok;
  } catch {
    return false;
  }
}

function textFallback(value: string): string {
  return value;
}

export function StorageTrackingSettings({
  client,
  defaultOrg,
  defaultProject,
  onNotify,
}: StorageTrackingSettingsProps) {
  const [status, setStatus] = useState<StorageStatus | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [loading, setLoading] = useState(Boolean(client.getStorageStatus));
  const [error, setError] = useState<Error | null>(null);
  const [tracker, setTracker] = useState<TrackerHealth>({ state: "idle" });
  const [revision, setRevision] = useState(0);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof globalThis.setTimeout> | null>(null);

  // Interactive cross-region URI checker state.
  const [checkUri, setCheckUri] = useState("");
  const [checkRegion, setCheckRegion] = useState("");
  const [checkStrict, setCheckStrict] = useState(false);
  const [checkingRegion, setCheckingRegion] = useState(false);
  const [checkResult, setCheckResult] = useState<StorageRegionCheckResult | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);

  // Collapsible managed artifacts table state.
  const [artifactsOpen, setArtifactsOpen] = useState(false);
  const [artifacts, setArtifacts] = useState<ManagedArtifact[] | null>(null);
  const [artifactsLoading, setArtifactsLoading] = useState(false);
  const [artifactsError, setArtifactsError] = useState<string | null>(null);

  useEffect(
    () => () => {
      if (timerRef.current !== null) globalThis.clearTimeout(timerRef.current);
    },
    [],
  );

  const load = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    if (!client.getStorageStatus) {
      setLoading(false);
      setUnsupported(true);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setUnsupported(false);
    client
      .getStorageStatus(controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return;
        if (value === null) {
          setStatus(null);
          setUnsupported(true);
        } else {
          setStatus(value);
          setUnsupported(false);
        }
      })
      .catch((failure: unknown) => {
        if (controller.signal.aborted) return;
        setError(failure instanceof Error ? failure : new Error("Unable to read storage status"));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [client, revision]);

  useEffect(() => {
    if (!client.listRuns) {
      setTracker({ state: "idle" });
      return;
    }
    const controller = new AbortController();
    setTracker({ state: "checking" });
    client
      .listRuns({ limit: 1 }, { orgId: defaultOrg, projectId: defaultProject }, controller.signal)
      .then((runs) => {
        if (!controller.signal.aborted) setTracker({ state: "ok", runCount: runs.length });
      })
      .catch((failure: unknown) => {
        if (controller.signal.aborted) return;
        if (failure instanceof ApiError && failure.status === 503) {
          setTracker({ state: "unconfigured" });
        } else {
          setTracker({
            state: "error",
            detail: failure instanceof Error ? failure.message : "Tracker unreachable",
          });
        }
      });
    return () => controller.abort();
  }, [client, defaultOrg, defaultProject, revision]);

  useEffect(() => {
    if (!artifactsOpen || !client.listManagedArtifacts) return;
    const controller = new AbortController();
    setArtifactsLoading(true);
    setArtifactsError(null);
    client
      .listManagedArtifacts({ orgId: defaultOrg, projectId: defaultProject }, controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) setArtifacts(items);
      })
      .catch((failure: unknown) => {
        if (controller.signal.aborted) return;
        setArtifactsError(failure instanceof Error ? failure.message : "Unable to load managed artifacts");
      })
      .finally(() => {
        if (!controller.signal.aborted) setArtifactsLoading(false);
      });
    return () => controller.abort();
  }, [artifactsOpen, client, defaultOrg, defaultProject, revision]);

  const handleCheckRegion = async (event: FormEvent) => {
    event.preventDefault();
    if (!client.checkStorageRegion) return;
    const uri = checkUri.trim();
    if (!uri) {
      setCheckError("Enter a gs:// URI to check.");
      return;
    }
    setCheckingRegion(true);
    setCheckError(null);
    setCheckResult(null);
    try {
      const result = await client.checkStorageRegion({
        uri,
        region: checkRegion.trim() || status?.context.region || undefined,
        strict: checkStrict,
      });
      setCheckResult(result);
    } catch (failure) {
      setCheckError(failure instanceof Error ? failure.message : "Unable to check storage region locality");
    } finally {
      setCheckingRegion(false);
    }
  };

  const handleCopy = async (id: string, value: string, message: string) => {
    const ok = await copyText(value);
    if (!ok) return;
    setCopiedId(id);
    onNotify?.(message);
    if (timerRef.current !== null) globalThis.clearTimeout(timerRef.current);
    timerRef.current = globalThis.setTimeout(() => setCopiedId(null), 1800);
  };

  const regionalEntries = Object.entries(status?.regional ?? {}).sort(([left], [right]) => left.localeCompare(right));

  return (
    <section className="panel data-panel" aria-label="Storage and experiment tracking">
      <div className="panel-header">
        <Database size={16} aria-hidden="true" />
        <div>
          <h2>Storage &amp; experiment tracking</h2>
          <p>Regional GCS buckets (eyvan_&lt;region&gt;), monitoring store, and local tracker status</p>
        </div>
        <button className="button button-ghost" type="button" disabled={loading} onClick={load}>
          <RefreshCw className={loading ? "spin" : undefined} size={14} aria-hidden="true" />
          {loading ? "Checking" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="storage-panel-pad">
          <ErrorCard
            compact
            error={error}
            context={{ fallbackTitle: "Unable to read storage status" }}
            actions={[{ label: "Retry", onClick: load }]}
            onCopied={onNotify}
          />
        </div>
      )}

      {!error && loading && !status && (
        <p className="panel-note" role="status">
          Reading storage configuration and GCP context…
        </p>
      )}

      {!error && unsupported && (
        <p className="panel-note">
          This control plane does not expose <code>/api/v1/storage/status</code> yet. Run{" "}
          <code>eyvan storage status</code> in your terminal to inspect configured GCS buckets.
        </p>
      )}

      {!error && status && (
        <div className="storage-grid">
          <div className="storage-card">
            <div className="storage-card-head">
              <h3>GCS storage config</h3>
              <StatusBadge
                value={status.configured ? "configured" : "uninitialized"}
                tone={status.configured ? "healthy" : "warning"}
                chip
              />
            </div>
            <dl className="storage-kv">
              <div>
                <dt>Config file</dt>
                <dd className="monospace">{status.config_path}</dd>
              </div>
              <div>
                <dt>Default project</dt>
                <dd className="monospace">{status.project ?? status.context.project ?? "Not detected"}</dd>
              </div>
              <div>
                <dt>Detected region</dt>
                <dd className="monospace">
                  {status.context.region ?? "—"}
                  {status.context.zone ? ` (${status.context.zone})` : ""}
                </dd>
              </div>
              <div>
                <dt>Context source</dt>
                <dd className="monospace">
                  {status.context.source}
                  {status.context.on_gcp ? " · on GCP" : ""}
                </dd>
              </div>
              <div>
                <dt>Auto-create</dt>
                <dd>{status.auto_create ? "Enabled" : "Disabled (explicit init)"}</dd>
              </div>
            </dl>
          </div>

          <div className="storage-card">
            <div className="storage-card-head">
              <h3>Monitoring &amp; tracker</h3>
              {tracker.state === "ok" && <StatusBadge value="tracker_online" tone="healthy" chip />}
              {tracker.state === "unconfigured" && <StatusBadge value="tracker_offline" tone="warning" chip />}
              {tracker.state === "error" && <StatusBadge value="tracker_unreachable" tone="degraded" chip />}
              {tracker.state === "checking" && <StatusBadge value="checking" tone="starting" chip />}
            </div>
            <dl className="storage-kv">
              <div>
                <dt>Monitoring bucket</dt>
                <dd>
                  {status.monitoring ? (
                    <span className="storage-uri-row">
                      <code className="monospace">{status.monitoring.uri}</code>
                      <button
                        type="button"
                        className="icon-button icon-button-small"
                        aria-label="Copy monitoring bucket URI"
                        title="Copy URI"
                        onClick={() =>
                          void handleCopy("mon-uri", status.monitoring!.uri, "Monitoring URI copied to clipboard")
                        }
                      >
                        {copiedId === "mon-uri" ? <Check size={12} /> : <Copy size={12} />}
                      </button>
                    </span>
                  ) : (
                    <span className="muted">Not configured — run eyvan storage init</span>
                  )}
                </dd>
              </div>
              {status.monitoring && (
                <div>
                  <dt>Location / prefix</dt>
                  <dd className="monospace">
                    {status.monitoring.location} · /{status.monitoring.prefix || ""}
                  </dd>
                </div>
              )}
              <div>
                <dt>Experiment tracker</dt>
                <dd>
                  {tracker.state === "ok"
                    ? "Connected and serving run streams"
                    : tracker.state === "unconfigured"
                      ? "Not configured (HTTP 503) — start with eyvan track up --local"
                      : tracker.state === "error"
                        ? tracker.detail
                        : tracker.state === "checking"
                          ? "Checking tracker health…"
                          : "—"}
                </dd>
              </div>
            </dl>
          </div>
        </div>
      )}

      {!error && status && (
        <div className="storage-regional">
          <div className="storage-subhead">
            <HardDrive size={13} aria-hidden="true" />
            <span>Regional storage buckets ({regionalEntries.length})</span>
          </div>
          {regionalEntries.length === 0 ? (
            <p className="panel-note">
              No regional bucket recorded yet. Run <code>eyvan storage init --region &lt;region&gt;</code> to bind a
              region-local bucket (<code>eyvan_&lt;region&gt;</code>) for checkpoints and datasets.
            </p>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Region</th>
                    <th>Bucket</th>
                    <th>URI</th>
                    <th aria-label="Copy" />
                  </tr>
                </thead>
                <tbody>
                  {regionalEntries.map(([region, entry]) => {
                    const key = `reg-${region}`;
                    return (
                      <tr key={region}>
                        <td className="monospace">{region}</td>
                        <td className="monospace">{entry.bucket}</td>
                        <td className="monospace">{entry.uri}</td>
                        <td className="numeric">
                          <button
                            type="button"
                            className="button button-ghost"
                            aria-label={`Copy ${region} bucket URI`}
                            onClick={() => void handleCopy(key, entry.uri, `Copied ${entry.uri}`)}
                          >
                            {copiedId === key ? (
                              <Check size={12} aria-hidden="true" />
                            ) : (
                              <Copy size={12} aria-hidden="true" />
                            )}
                            {copiedId === key ? "Copied" : "Copy"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {client.checkStorageRegion && (
        <div className="storage-region-checker" aria-label="Cross-region URI checker">
          <div className="storage-subhead">
            <ShieldCheck size={13} aria-hidden="true" />
            <span>Cross-region URI checker</span>
          </div>
          <form className="storage-checker-form" onSubmit={(event) => void handleCheckRegion(event)}>
            <label className="field storage-checker-uri">
              <span>GCS URI</span>
              <input
                type="text"
                aria-label="GCS URI to check"
                placeholder="gs://example-bucket/checkpoints"
                value={checkUri}
                disabled={checkingRegion}
                onChange={(event) => setCheckUri(event.target.value)}
              />
            </label>
            <label className="field storage-checker-region">
              <span>Target TPU region</span>
              <input
                type="text"
                aria-label="Target TPU region"
                placeholder={status?.context.region ?? "e.g. europe-west4"}
                value={checkRegion}
                disabled={checkingRegion}
                onChange={(event) => setCheckRegion(event.target.value)}
              />
            </label>
            <label className="collab-checkbox-field storage-checker-strict">
              <input
                type="checkbox"
                checked={checkStrict}
                disabled={checkingRegion}
                onChange={(event) => setCheckStrict(event.target.checked)}
              />
              <span>Strict same-region</span>
            </label>
            <button className="button button-primary" type="submit" disabled={checkingRegion}>
              {checkingRegion ? "Checking…" : "Check region"}
            </button>
          </form>
          {checkError && (
            <div className="storage-panel-pad">
              <div className="form-error" role="alert">
                {checkError}
              </div>
            </div>
          )}
          {checkResult && (
            <div
              className={`storage-check-verdict ${checkResult.allowed ? "storage-check-allowed" : "storage-check-blocked"}`}
              role="status"
            >
              <div className="storage-check-verdict-head">
                <StatusBadge
                  value={checkResult.allowed ? "Allowed" : "Blocked"}
                  tone={checkResult.allowed ? "healthy" : "critical"}
                  chip
                />
                {checkResult.allowlisted && <StatusBadge value="Allowlisted" tone="info" dot={false} chip />}
                <span className="monospace">
                  {checkResult.bucket} ({checkResult.bucket_location ?? "unknown"}
                  {checkResult.location_type ? ` · ${checkResult.location_type}` : ""}) → {checkResult.host_region}
                </span>
              </div>
              <p>{checkResult.reason}</p>
            </div>
          )}
        </div>
      )}

      {client.listManagedArtifacts && (
        <div className="storage-artifacts-section" aria-label="Managed artifacts">
          <div className="storage-subhead">
            <button
              type="button"
              className="metric-group-toggle"
              aria-expanded={artifactsOpen}
              onClick={() => setArtifactsOpen((open) => !open)}
            >
              {artifactsOpen ? (
                <ChevronDown size={13} aria-hidden="true" />
              ) : (
                <ChevronRight size={13} aria-hidden="true" />
              )}
              <Package size={13} aria-hidden="true" />
              <span>Managed artifacts ({artifacts?.length ?? "click to load"})</span>
            </button>
          </div>
          {artifactsOpen && (
            <div className="storage-artifacts-body">
              {artifactsLoading && (
                <p className="panel-note" role="status">
                  Loading managed artifacts…
                </p>
              )}
              {artifactsError && (
                <div className="storage-panel-pad">
                  <div className="form-error" role="alert">
                    {artifactsError}
                  </div>
                </div>
              )}
              {!artifactsLoading && !artifactsError && artifacts && artifacts.length === 0 && (
                <p className="panel-note">
                  No managed artifacts recorded for {defaultOrg}/{defaultProject}.
                </p>
              )}
              {!artifactsLoading && !artifactsError && artifacts && artifacts.length > 0 && (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>URI</th>
                        <th>Kind</th>
                        <th>State</th>
                        <th className="numeric">Size</th>
                        <th className="numeric">Gen</th>
                        <th>Updated</th>
                      </tr>
                    </thead>
                    <tbody>
                      {artifacts.map((art) => (
                        <tr key={`${art.uri}:${art.record_generation}`}>
                          <td className="monospace" title={art.uri}>
                            {art.uri}
                          </td>
                          <td>{art.kind}</td>
                          <td>
                            <StatusBadge
                              value={art.state}
                              tone={
                                art.state === "active" ? "healthy" : art.state === "deleted" ? "expired" : "warning"
                              }
                              dot={false}
                            />
                          </td>
                          <td className="numeric">{formatBytes(art.size_bytes)}</td>
                          <td className="numeric">v{art.record_generation}</td>
                          <td className="nowrap">{formatDateTime(art.updated_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <div className="storage-commands">
        <div className="storage-subhead">
          <Terminal size={13} aria-hidden="true" />
          <span>CLI quick commands</span>
        </div>
        <div className="storage-command-list">
          {CLI_COMMANDS.map((item) => {
            const copied = copiedId === item.id;
            return (
              <div className="storage-command-row" key={item.id}>
                <div className="storage-command-meta">
                  <strong>{item.label}</strong>
                  <small>{item.description}</small>
                </div>
                <div className="storage-command-code">
                  <code className="monospace">{item.command}</code>
                  <button
                    type="button"
                    className="button button-secondary"
                    aria-label={`Copy ${item.command}`}
                    onClick={() => void handleCopy(item.id, item.command, `Copied "${item.command}"`)}
                  >
                    {copied ? <Check size={12} aria-hidden="true" /> : <Copy size={12} aria-hidden="true" />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
