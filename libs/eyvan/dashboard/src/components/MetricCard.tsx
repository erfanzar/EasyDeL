// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

type Tone = "neutral" | "positive" | "warning" | "critical" | "accent";

interface MetricCardProps {
  label: string;
  value: string;
  detail: string;
  /**
   * The number's qualifier, beside it rather than under it.
   *
   * Not a trend arrow: most of what qualifies a figure on this console is not
   * a percentage change over some unstated window -- it is "oldest 10h 33m",
   * "1 lost", "68% of the daily pace". A free-text delta says the true thing;
   * an arrow would have to invent a comparison to point at.
   */
  delta?: string;
  deltaTone?: Tone;
  tone?: Tone;
}

export function MetricCard({ label, value, detail, delta, deltaTone, tone = "neutral" }: MetricCardProps) {
  return (
    <article className={`metric-card metric-${tone}`}>
      <div className="metric-heading">
        <span className="metric-dot" aria-hidden="true" />
        <span>{label}</span>
      </div>
      <div className="metric-figure">
        <strong className="metric-value">{value}</strong>
        {delta !== undefined && <span className={`metric-delta metric-delta-${deltaTone ?? tone}`}>{delta}</span>}
      </div>
      <div className="metric-detail">{detail}</div>
    </article>
  );
}
