// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Clock, RotateCcw, Square } from "lucide-react";
import { useEffect, useState } from "react";

import { formatDateTime } from "../lib/format";
import type { ReclaimPolicy } from "../types";

/** Default checkpoint grace window used when the workload does not specify one. */
const DEFAULT_RECLAIM_GRACE_SECONDS = 600;

/** Remaining seconds under which the countdown switches from amber to urgent red. */
const URGENT_THRESHOLD_SECONDS = 60;

/** Format remaining milliseconds as a compact human countdown, e.g. `8m 42s`, `45s`, `1h 04m 12s`. */
export function formatReclaimCountdown(remainingMs: number): string {
  if (!Number.isFinite(remainingMs) || remainingMs <= 0) return "0s";
  const totalSeconds = Math.ceil(remainingMs / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  }
  return `${seconds}s`;
}

/** Human-readable description of what happens to a reclaimed job once its checkpoint completes. */
export function reclaimPolicyLabel(policy?: ReclaimPolicy | null): string | null {
  if (policy === "requeue") return "Auto-resume elsewhere";
  if (policy === "stop") return "Stop after checkpoint";
  return null;
}

/** Parse an ISO deadline timestamp into epoch milliseconds, or null when invalid. */
function parseDeadlineMs(deadlineAt: string | null | undefined): number | null {
  if (!deadlineAt) return null;
  const parsed = Date.parse(deadlineAt);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Tick once per second while `deadlineMs` is still in the future; stops automatically at 0. */
function useLiveDeadlineNow(deadlineMs: number | null): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    setNow(Date.now());
    if (deadlineMs === null || Date.now() >= deadlineMs) return;
    const timer = globalThis.setInterval(() => {
      const current = Date.now();
      setNow(current);
      if (current >= deadlineMs) {
        globalThis.clearInterval(timer);
      }
    }, 1000);
    return () => globalThis.clearInterval(timer);
  }, [deadlineMs]);

  return now;
}

export interface ReclaimBannerProps {
  /** ISO timestamp when the checkpoint grace window ends. */
  deadlineAt: string;
  /** Who triggered the reclaim (e.g. returning guaranteed owner). */
  requestedBy?: string | null;
  /** Post-checkpoint action: requeue elsewhere or stop after checkpoint. */
  policy?: ReclaimPolicy | null;
  /**
   * Role of the workload displaying this banner:
   * - `victim`: borrowed job currently saving its checkpoint before release.
   * - `waiting`: guaranteed owner's queued job waiting for borrowed capacity to free up.
   */
  role?: "victim" | "waiting";
  /** Optional scheduler explanation (e.g. naming the borrower being reclaimed from). */
  reason?: string | null;
  /** Configured checkpoint grace duration in seconds; defaults to 600s (10 min). */
  graceSeconds?: number;
}

export function ReclaimBanner({
  deadlineAt,
  requestedBy,
  policy,
  role = "victim",
  reason,
  graceSeconds = DEFAULT_RECLAIM_GRACE_SECONDS,
}: ReclaimBannerProps) {
  const deadlineMs = parseDeadlineMs(deadlineAt);
  const now = useLiveDeadlineNow(deadlineMs);
  if (deadlineMs === null) return null;

  const remainingMs = Math.max(0, deadlineMs - now);
  const remainingSec = remainingMs / 1000;
  const expired = remainingMs <= 0;
  const urgent = !expired && remainingSec <= URGENT_THRESHOLD_SECONDS;
  const totalWindowSec = Math.max(graceSeconds > 0 ? graceSeconds : DEFAULT_RECLAIM_GRACE_SECONDS, remainingSec, 1);
  const remainingPct = expired ? 0 : Math.max(0, Math.min(100, Math.round((remainingSec / totalWindowSec) * 100)));
  const policyText = reclaimPolicyLabel(policy);
  const countdown = formatReclaimCountdown(remainingMs);

  const headline =
    role === "waiting"
      ? expired
        ? "Reclaim window elapsed · scheduling shortly"
        : `Reclaiming borrowed capacity · ${countdown} remaining`
      : expired
        ? "Reclaim deadline reached · finalizing release"
        : `Reclaiming · ${countdown} remaining`;

  return (
    <div
      className={`reclaim-banner${urgent || expired ? " reclaim-banner-urgent" : ""}${role === "waiting" ? " reclaim-banner-waiting" : ""}`}
      role="status"
      aria-live="off"
      data-testid="reclaim-banner"
    >
      <div className="reclaim-banner-top">
        <span className="reclaim-banner-icon" aria-hidden="true">
          <Clock size={15} />
        </span>
        <div className="reclaim-banner-copy">
          <strong className="reclaim-banner-headline">{headline}</strong>
          <div className="reclaim-banner-meta">
            {requestedBy && (
              <span>
                Requested by <strong className="monospace">{requestedBy}</strong>
              </span>
            )}
            {reason && <span className="reclaim-banner-reason">{reason}</span>}
            <span>Deadline {formatDateTime(deadlineAt)}</span>
          </div>
        </div>
        {policyText && (
          <span
            className={`reclaim-policy-chip${policy === "stop" ? " reclaim-policy-stop" : " reclaim-policy-requeue"}`}
          >
            {policy === "stop" ? <Square size={11} aria-hidden="true" /> : <RotateCcw size={11} aria-hidden="true" />}
            {policyText}
          </span>
        )}
      </div>
      <div
        className="reclaim-progress-track"
        role="progressbar"
        aria-label="Reclaim checkpoint window remaining"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={remainingPct}
      >
        {remainingPct > 0 && (
          <span
            className={`reclaim-progress-fill${urgent ? " reclaim-progress-urgent" : ""}`}
            style={{ width: `${remainingPct}%` }}
          />
        )}
      </div>
    </div>
  );
}

export interface ReclaimBadgeProps {
  /** ISO timestamp when the checkpoint grace window ends. */
  deadlineAt: string;
  /** Who triggered the reclaim. */
  requestedBy?: string | null;
  /** Post-checkpoint policy. */
  policy?: ReclaimPolicy | null;
  /** Optional role label override: waiting owner vs borrowed victim. */
  role?: "victim" | "waiting";
  /** Include the compact policy label beside the countdown when true. */
  showPolicy?: boolean;
}

export function ReclaimBadge({
  deadlineAt,
  requestedBy,
  policy,
  role = "victim",
  showPolicy = false,
}: ReclaimBadgeProps) {
  const deadlineMs = parseDeadlineMs(deadlineAt);
  const now = useLiveDeadlineNow(deadlineMs);
  if (deadlineMs === null) return null;

  const remainingMs = Math.max(0, deadlineMs - now);
  const remainingSec = remainingMs / 1000;
  const expired = remainingMs <= 0;
  const urgent = !expired && remainingSec <= URGENT_THRESHOLD_SECONDS;
  const countdown = formatReclaimCountdown(remainingMs);
  const policyText = reclaimPolicyLabel(policy);

  const label =
    role === "waiting"
      ? expired
        ? "Reclaim ready"
        : `Waiting reclaim · ${countdown}`
      : expired
        ? "Reclaiming · 0s"
        : `Reclaiming · ${countdown}`;

  const tooltipParts = [
    role === "waiting"
      ? "Waiting for borrowed capacity to checkpoint and release"
      : "Graceful checkpoint reclaim active",
    requestedBy ? `Requested by ${requestedBy}` : null,
    policyText,
    `Deadline ${formatDateTime(deadlineAt)}`,
  ].filter(Boolean);

  return (
    <span className="reclaim-badge-group">
      <span
        className={`reclaim-badge${urgent || expired ? " reclaim-badge-urgent" : ""}`}
        title={tooltipParts.join(" · ")}
        data-testid="reclaim-badge"
      >
        <Clock size={11} aria-hidden="true" />
        <span>{label}</span>
      </span>
      {showPolicy && policyText && (
        <span
          className={`reclaim-policy-inline${policy === "stop" ? " reclaim-policy-stop" : " reclaim-policy-requeue"}`}
        >
          {policyText}
        </span>
      )}
    </span>
  );
}
