// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { PriorityBand, Severity, WorkloadStatus } from "../types";
import { titleCase } from "../lib/format";

type BadgeTone =
  | WorkloadStatus
  | PriorityBand
  | Severity
  | "healthy"
  | "starting"
  | "degraded"
  | "maintenance"
  | "active"
  | "activating"
  | "revoking"
  | "expired"
  | "reserved"
  | "on_demand"
  | "spot"
  | "hard"
  | "soft";

interface StatusBadgeProps {
  value: string;
  tone?: BadgeTone;
  dot?: boolean;
  /**
   * Whether the state gets a filled background.
   *
   * Off by default, because a table column of filled pills is a column of
   * colored blocks: the eye reads the shape, stops reading the word, and every
   * row shouts equally. A washed chip is for the one place there is exactly one
   * of them -- a record's own heading.
   */
  chip?: boolean;
}

export function StatusBadge({ value, tone, dot = true, chip = false }: StatusBadgeProps) {
  const className = `status-badge${chip ? " status-badge-chip" : ""} status-${tone ?? value.toLowerCase()}`;
  return (
    <span className={className}>
      {dot && <span className="status-dot" aria-hidden="true" />}
      {titleCase(value)}
    </span>
  );
}
