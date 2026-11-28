// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { CrossRegionTransferPolicy } from "../types";

export const CROSS_REGION_TRANSFER_BAN_PREFIX = "cross-region-transfer-banned";

/** Normalize a raw cross-region transfer policy value from older or newer servers. */
export function normalizeCrossRegionTransferPolicy(value: unknown): CrossRegionTransferPolicy {
  return value === "ban" || value === "warn" || value === "allow" ? value : "allow";
}

/** Return true when a workload reason carries the cross-region transfer ban prefix. */
export function isCrossRegionTransferBanReason(reason: string | null | undefined): boolean {
  return typeof reason === "string" && reason.trim().toLowerCase().startsWith(CROSS_REGION_TRANSFER_BAN_PREFIX);
}

/**
 * Extract the cross-region transfer ban reason when a workload was rejected at
 * dispatch or killed at runtime for a cross-region GCS transfer violation.
 */
export function getCrossRegionTransferBanReason(workload: {
  failureReason?: string;
  queueReason?: string;
}): string | null {
  for (const candidate of [workload.failureReason, workload.queueReason]) {
    if (isCrossRegionTransferBanReason(candidate)) {
      return candidate!.trim();
    }
  }
  return null;
}

/** Strip the leading `cross-region-transfer-banned:` machine prefix for human-readable detail. */
export function formatCrossRegionBanDetail(reason: string): string {
  const stripped = reason.replace(/^cross-region-transfer-banned\s*:?\s*/i, "").trim();
  return stripped || reason;
}

/** Compact badge label for a cross-region transfer enforcement mode. */
export function crossRegionPolicyLabel(policy: CrossRegionTransferPolicy): string {
  if (policy === "ban") return "Banned";
  if (policy === "warn") return "Warn";
  return "Allowed";
}
