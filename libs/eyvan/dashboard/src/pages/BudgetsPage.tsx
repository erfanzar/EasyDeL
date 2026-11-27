// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Edit3, Shield, ShieldAlert } from "lucide-react";
import { useState } from "react";

import { EmptyState } from "../components/EmptyState";
import { ErrorCard } from "../components/ErrorCard";
import { MetricCard } from "../components/MetricCard";
import { ProgressBar } from "../components/ProgressBar";
import { StatusBadge } from "../components/StatusBadge";
import { crossRegionPolicyLabel, normalizeCrossRegionTransferPolicy } from "../lib/crossRegionPolicy";
import { formatDateTime, formatMoney, percent } from "../lib/format";
import type { Budget, CrossRegionTransferPolicy, PolicyUpdatePatch, SchedulingPolicy } from "../types";

interface BudgetsPageProps {
  budgets: Budget[];
  policies: SchedulingPolicy[];
  onPolicyToggle: (policy: SchedulingPolicy, preemptionEnabled: boolean) => Promise<SchedulingPolicy>;
  onCrossRegionPolicyChange?: (
    policy: SchedulingPolicy,
    crossRegionTransferPolicy: CrossRegionTransferPolicy,
  ) => Promise<SchedulingPolicy>;
  onPolicyUpdate?: (policy: SchedulingPolicy, patch: PolicyUpdatePatch) => Promise<SchedulingPolicy>;
  canManageProject: boolean;
}

export function BudgetsPage({
  budgets,
  policies,
  onPolicyToggle,
  onCrossRegionPolicyChange,
  onPolicyUpdate,
  canManageProject,
}: BudgetsPageProps) {
  const [pendingPolicy, setPendingPolicy] = useState<string | null>(null);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const orgBudget = budgets.find((budget) => budget.scope === "organization") ?? budgets[0];
  const hardBudgets = budgets.filter((budget) => budget.enforcement === "hard");
  const overForecast = budgets.filter((budget) => budget.forecast !== undefined && budget.forecast > budget.limit);

  const togglePolicy = async (policy: SchedulingPolicy) => {
    setPendingPolicy(policy.id);
    setMutationError(null);
    try {
      await onPolicyToggle(policy, !policy.preemptionEnabled);
    } catch (error) {
      setMutationError(error instanceof Error ? error.message : `Unable to update ${policy.name}`);
    } finally {
      setPendingPolicy(null);
    }
  };

  const changeCrossRegionPolicy = async (policy: SchedulingPolicy, nextMode: CrossRegionTransferPolicy) => {
    const updater = onCrossRegionPolicyChange
      ? (target: SchedulingPolicy, mode: CrossRegionTransferPolicy) => onCrossRegionPolicyChange(target, mode)
      : onPolicyUpdate
        ? (target: SchedulingPolicy, mode: CrossRegionTransferPolicy) =>
            onPolicyUpdate(target, { crossRegionTransferPolicy: mode })
        : null;
    if (!updater) return;
    setPendingPolicy(policy.id);
    setMutationError(null);
    try {
      await updater(policy, nextMode);
    } catch (error) {
      setMutationError(error instanceof Error ? error.message : `Unable to update ${policy.name}`);
    } finally {
      setPendingPolicy(null);
    }
  };

  return (
    <div className="page-stack">
      <section className="metric-grid metric-grid-four" aria-label="Budget metrics">
        <MetricCard
          label="Scoped spend"
          value={orgBudget ? formatMoney(orgBudget.spent, true) : "Not set"}
          detail={
            orgBudget
              ? `of ${formatMoney(orgBudget.limit, true)} for ${orgBudget.owner}`
              : "No budget returned for this scope"
          }
          tone="warning"
        />
        <MetricCard
          label="Current forecast"
          value={orgBudget?.forecast !== undefined ? formatMoney(orgBudget.forecast, true) : "Unavailable"}
          detail={
            orgBudget?.forecast !== undefined
              ? `${percent(orgBudget.forecast, orgBudget.limit)}% of ${orgBudget.owner} limit`
              : "Not reported by the control plane"
          }
          tone="warning"
        />
        <MetricCard label="Hard budgets" value={hardBudgets.length.toString()} detail="reject admission at limit" />
        <MetricCard
          label="Forecast alerts"
          value={overForecast.length.toString()}
          detail="scopes projected above limit"
          tone={overForecast.length ? "critical" : "positive"}
        />
      </section>

      <section className="budget-layout">
        <div className="budget-list-section">
          <div className="section-toolbar">
            <div>
              <h2>Budget scopes</h2>
              <p>Current spend, forecast, and admission enforcement</p>
            </div>
          </div>
          {budgets.length === 0 ? (
            <section className="panel">
              <EmptyState
                title="No budgets configured for this scope"
                detail="Define project or organization budgets via the Eyvan CLI or API to enforce spend limits."
              />
            </section>
          ) : (
            <div className="budget-card-list">
              {budgets.map((budget) => {
                const spentPercent = percent(budget.spent, budget.limit);
                const forecastPercent =
                  budget.forecast === undefined ? undefined : percent(budget.forecast, budget.limit);
                return (
                  <article className="budget-card" key={budget.id}>
                    <div className="budget-card-heading">
                      <span className="budget-scope">{budget.scope}</span>
                      <StatusBadge value={budget.enforcement} tone={budget.enforcement} dot={false} chip />
                      <span title="Budget changes are available through the Eyvan API and SDK">
                        <button
                          className="icon-button icon-button-small"
                          type="button"
                          aria-label={`Editing ${budget.owner} budget is unavailable in the dashboard`}
                          disabled
                        >
                          <Edit3 size={14} />
                        </button>
                      </span>
                    </div>
                    <div className="budget-owner">{budget.owner}</div>
                    <div className="budget-amount-row">
                      <strong>{formatMoney(budget.spent)}</strong>
                      <span>of {formatMoney(budget.limit)}</span>
                    </div>
                    {/* The forecast is a tick above the bar, not a second fill:
                        it has not been spent, and a filled track would say it
                        has. */}
                    <div className="dual-progress">
                      <ProgressBar
                        value={spentPercent}
                        label={`${budget.owner} current budget usage`}
                        tone={spentPercent > 90 ? "red" : spentPercent > 75 ? "amber" : "accent"}
                      />
                      {forecastPercent !== undefined && (
                        <span
                          className="forecast-marker"
                          style={{ left: `${Math.min(forecastPercent, 100)}%` }}
                          title={`Forecast ${forecastPercent}%`}
                        />
                      )}
                    </div>
                    <div className="budget-card-footer">
                      <span>{spentPercent}% spent</span>
                      <span
                        className={
                          budget.forecast !== undefined && budget.forecast > budget.limit ? "text-critical" : ""
                        }
                      >
                        {budget.forecast === undefined
                          ? "forecast unavailable"
                          : `forecast ${formatMoney(budget.forecast)}`}
                      </span>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </div>

        <aside className="panel priority-policy-panel">
          <div className="panel-header">
            <div>
              <h2>Priority bands</h2>
              <p>Admission and preemption order</p>
            </div>
          </div>
          <ol className="priority-band-list">
            <li>
              <span className="band-rank">1</span>
              <span>
                <StatusBadge value="production" tone="critical" dot={false} />
                <small>Production workloads and urgent operational jobs</small>
              </span>
              <strong>Highest priority</strong>
            </li>
            <li>
              <span className="band-rank">2</span>
              <span>
                <StatusBadge value="interactive" tone="interactive" dot={false} />
                <small>User-facing evals and short experiments</small>
              </span>
              <strong>Middle priority</strong>
            </li>
            <li>
              <span className="band-rank">3</span>
              <span>
                <StatusBadge value="batch" tone="batch" dot={false} />
                <small>Training, backfill, and scheduled pipelines</small>
              </span>
              <strong>Base priority</strong>
            </li>
          </ol>
          <div className="policy-callout">
            <Shield size={17} aria-hidden="true" />
            <span>
              <strong>Whole-gang preemption</strong>
              <small>When enabled, victims checkpoint together before capacity is reassigned.</small>
            </span>
          </div>
          <div className="policy-callout policy-callout-critical">
            <ShieldAlert size={17} aria-hidden="true" />
            <span>
              <strong>Cross-region transfer: Ban</strong>
              <small>
                Refuses cross-region GCS URIs at dispatch and immediately kills running workloads (exit 95,
                non-retryable) if they read or write a bucket outside their TPU&apos;s region.
              </small>
            </span>
          </div>
        </aside>
      </section>

      <section className="panel data-panel">
        <div className="panel-header">
          <div>
            <h2>Scheduling policies</h2>
            <p>Versioned rules applied by admission and placement</p>
          </div>
        </div>
        {mutationError && (
          <div className="panel-alert-wrap">
            <ErrorCard
              compact
              error={mutationError}
              context={{ fallbackTitle: "Policy update failed" }}
              onDismiss={() => setMutationError(null)}
            />
          </div>
        )}
        {policies.length === 0 ? (
          <EmptyState
            title="No scheduling policies configured"
            detail="Default admission and placement rules apply until a project policy is created."
          />
        ) : (
          <div className="table-scroll">
            <table className="data-table policies-table">
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>Type</th>
                  <th>Scope</th>
                  <th>Behavior</th>
                  <th>Cross-region transfer</th>
                  <th>Updated</th>
                  <th>Preemptible</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((policy) => {
                  const crossRegionMode = normalizeCrossRegionTransferPolicy(policy.crossRegionTransferPolicy);
                  const crossRegionTone =
                    crossRegionMode === "ban" ? "critical" : crossRegionMode === "warn" ? "warning" : "standard";
                  const allowedBuckets = policy.crossRegionAllowedBuckets ?? [];
                  const isPending = pendingPolicy === policy.id;
                  const canEditCrossRegion = canManageProject && Boolean(onCrossRegionPolicyChange || onPolicyUpdate);
                  return (
                    <tr key={policy.id}>
                      <td>
                        <span className="resource-cell">
                          <strong>{policy.name}</strong>
                          <small className="monospace">{policy.id}</small>
                        </span>
                      </td>
                      <td>
                        <StatusBadge value={policy.type} dot={false} />
                      </td>
                      <td>{policy.scope}</td>
                      <td>{policy.summary}</td>
                      <td>
                        <div className="cross-region-policy-cell">
                          <div className="cross-region-policy-row">
                            <StatusBadge value={crossRegionPolicyLabel(crossRegionMode)} tone={crossRegionTone} />
                            {canEditCrossRegion && (
                              <label className={`select-field policy-select-field policy-select-${crossRegionMode}`}>
                                <span className="sr-only">Cross-region transfer policy for {policy.name}</span>
                                <select
                                  aria-label={`Cross-region transfer policy for ${policy.name}`}
                                  value={crossRegionMode}
                                  disabled={isPending}
                                  onChange={(event) => {
                                    const nextMode = event.target.value as CrossRegionTransferPolicy;
                                    if (nextMode !== crossRegionMode) {
                                      void changeCrossRegionPolicy(policy, nextMode);
                                    }
                                  }}
                                >
                                  <option value="allow">Allow</option>
                                  <option value="warn">Warn</option>
                                  <option value="ban">Ban &amp; kill</option>
                                </select>
                              </label>
                            )}
                          </div>
                          {isPending && (
                            <small className="policy-pending-hint" role="status">
                              Saving…
                            </small>
                          )}
                          {allowedBuckets.length > 0 && (
                            <small
                              className="policy-bucket-hint"
                              title={`Allowlisted buckets: ${allowedBuckets.join(", ")}`}
                            >
                              {allowedBuckets.length} allowlisted bucket{allowedBuckets.length === 1 ? "" : "s"}
                            </small>
                          )}
                          {crossRegionMode !== "allow" && policy.crossRegionAllowMultiregion === false && (
                            <small className="policy-bucket-hint">Multi-region blocked</small>
                          )}
                        </div>
                      </td>
                      <td>{formatDateTime(policy.updatedAt)}</td>
                      <td>
                        {canManageProject ? (
                          <label className="switch-control">
                            <input
                              type="checkbox"
                              checked={policy.preemptionEnabled}
                              disabled={isPending}
                              onChange={() => void togglePolicy(policy)}
                            />
                            <span aria-hidden="true" />
                            <span className="sr-only">Toggle {policy.name}</span>
                          </label>
                        ) : (
                          <span className="muted">Read only</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
