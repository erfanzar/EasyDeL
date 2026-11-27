// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertCircle, Edit3, Plus, Shield, ShieldAlert, X } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";

import { EmptyState } from "../components/EmptyState";
import { ErrorCard } from "../components/ErrorCard";
import { MetricCard } from "../components/MetricCard";
import { ProgressBar } from "../components/ProgressBar";
import { StatusBadge } from "../components/StatusBadge";
import { crossRegionPolicyLabel, normalizeCrossRegionTransferPolicy } from "../lib/crossRegionPolicy";
import { formatDateTime, formatMoney, percent } from "../lib/format";
import type { Budget, BudgetPutInput, CrossRegionTransferPolicy, PolicyUpdatePatch, SchedulingPolicy } from "../types";

interface BudgetsPageProps {
  budgets: Budget[];
  policies: SchedulingPolicy[];
  onPolicyToggle: (policy: SchedulingPolicy, preemptionEnabled: boolean) => Promise<SchedulingPolicy>;
  onCrossRegionPolicyChange?: (
    policy: SchedulingPolicy,
    crossRegionTransferPolicy: CrossRegionTransferPolicy,
  ) => Promise<SchedulingPolicy>;
  onPolicyUpdate?: (policy: SchedulingPolicy, patch: PolicyUpdatePatch) => Promise<SchedulingPolicy>;
  onSaveBudget?: (input: BudgetPutInput) => Promise<Budget>;
  orgId?: string;
  projectId?: string;
  canManageProject: boolean;
}

function splitCsv(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatHourLimits(limits?: Record<string, number>): string {
  if (!limits) return "";
  return Object.entries(limits)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

function parseHourLimits(raw: string): Record<string, number> | null {
  const out: Record<string, number> = {};
  const parts = splitCsv(raw);
  for (const part of parts) {
    const eq = part.indexOf("=");
    if (eq <= 0) return null;
    const key = part.slice(0, eq).trim();
    const num = Number(part.slice(eq + 1).trim());
    if (!key || !Number.isFinite(num) || num < 0) return null;
    out[key] = num;
  }
  return out;
}

interface BudgetModalProps {
  open: boolean;
  orgId: string;
  projectId: string;
  initial?: Budget | null;
  onClose: () => void;
  onSave: (input: BudgetPutInput) => Promise<void>;
}

function BudgetFormModal({ open, orgId, projectId, initial, onClose, onSave }: BudgetModalProps) {
  const [limit, setLimit] = useState(initial ? String(initial.limit) : "10000");
  const [period, setPeriod] = useState(
    initial && ["monthly", "weekly", "daily"].includes(initial.period) ? initial.period : "monthly",
  );
  const [hardLimit, setHardLimit] = useState(initial ? initial.enforcement === "hard" : true);
  const [thresholds, setThresholds] = useState(
    initial?.alertThresholds?.length ? initial.alertThresholds.join(", ") : "0.5, 0.8, 0.95",
  );
  const [hourLimits, setHourLimits] = useState(formatHourLimits(initial?.acceleratorHourLimits));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setLimit(initial ? String(initial.limit) : "10000");
    setPeriod(initial && ["monthly", "weekly", "daily"].includes(initial.period) ? initial.period : "monthly");
    setHardLimit(initial ? initial.enforcement === "hard" : true);
    setThresholds(initial?.alertThresholds?.length ? initial.alertThresholds.join(", ") : "0.5, 0.8, 0.95");
    setHourLimits(formatHourLimits(initial?.acceleratorHourLimits));
    setError(null);
  }, [open, initial]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    globalThis.addEventListener("keydown", onKeyDown);
    return () => globalThis.removeEventListener("keydown", onKeyDown);
  }, [open, busy, onClose]);

  if (!open) return null;

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const parsedLimit = Number(limit);
    if (!Number.isFinite(parsedLimit) || parsedLimit < 0) {
      setError("Budget limit must be a non-negative number.");
      return;
    }
    const parsedThresholds: number[] = [];
    for (const token of splitCsv(thresholds)) {
      const val = Number(token);
      if (!Number.isFinite(val) || val <= 0 || val > 1) {
        setError("Alert thresholds must be fractions in (0, 1], e.g. 0.5, 0.8, 0.95.");
        return;
      }
      parsedThresholds.push(val);
    }
    const parsedHours = parseHourLimits(hourLimits);
    if (parsedHours === null) {
      setError("Accelerator hour limits must use key=hours pairs, e.g. tpu=500, gpu=100.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onSave({
        orgId,
        projectId,
        limit: parsedLimit,
        period,
        hardLimit,
        alertThresholds: parsedThresholds,
        acceleratorHourLimits: parsedHours,
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save budget");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-layer" role="presentation">
      <button
        className="modal-scrim"
        type="button"
        aria-label="Close budget dialog"
        disabled={busy}
        onClick={onClose}
      />
      <div className="modal modal-narrow" role="dialog" aria-modal="true" aria-labelledby="budget-modal-title">
        <div className="modal-header">
          <div>
            <h2 id="budget-modal-title">{initial ? `Edit budget (${initial.owner})` : "Configure project budget"}</h2>
            <p>
              Scope:{" "}
              <span className="monospace">
                {orgId}/{projectId}
              </span>
            </p>
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label="Close budget dialog"
            disabled={busy}
            onClick={onClose}
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>
        <form onSubmit={(event) => void handleSubmit(event)}>
          <div className="modal-body collab-form-grid">
            {error && (
              <div className="form-error" role="alert">
                <AlertCircle size={15} aria-hidden="true" />
                {error}
              </div>
            )}
            <div className="collab-form-row">
              <label className="field">
                <span>Limit (USD)</span>
                <input
                  type="number"
                  min={0}
                  step="any"
                  required
                  value={limit}
                  disabled={busy}
                  onChange={(event) => setLimit(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Period</span>
                <select value={period} disabled={busy} onChange={(event) => setPeriod(event.target.value)}>
                  <option value="monthly">Monthly</option>
                  <option value="weekly">Weekly</option>
                  <option value="daily">Daily</option>
                </select>
              </label>
            </div>
            <label className="collab-checkbox-field">
              <input
                type="checkbox"
                checked={hardLimit}
                disabled={busy}
                onChange={(event) => setHardLimit(event.target.checked)}
              />
              <span>Hard limit (reject new workload admission when budget is exhausted)</span>
            </label>
            <label className="field">
              <span>Alert thresholds (0–1, comma-separated)</span>
              <input
                type="text"
                placeholder="0.5, 0.8, 0.95"
                value={thresholds}
                disabled={busy}
                onChange={(event) => setThresholds(event.target.value)}
              />
            </label>
            <label className="field">
              <span>Accelerator hour limits (optional key=hours)</span>
              <input
                type="text"
                placeholder="tpu=500, gpu=120"
                value={hourLimits}
                disabled={busy}
                onChange={(event) => setHourLimits(event.target.value)}
              />
            </label>
          </div>
          <div className="modal-footer">
            <button className="button button-secondary" type="button" disabled={busy} onClick={onClose}>
              Cancel
            </button>
            <button className="button button-primary" type="submit" disabled={busy}>
              {busy ? "Saving…" : "Save budget"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

interface PolicyModalProps {
  open: boolean;
  orgId: string;
  projectId: string;
  initial?: SchedulingPolicy | null;
  onClose: () => void;
  onSave: (target: SchedulingPolicy, patch: PolicyUpdatePatch) => Promise<void>;
}

function PolicyFormModal({ open, orgId, projectId, initial, onClose, onSave }: PolicyModalProps) {
  const cfg = initial?.configuration;
  const [allowedRegions, setAllowedRegions] = useState((cfg?.allowedRegions ?? []).join(", "));
  const [allowedProviders, setAllowedProviders] = useState((cfg?.allowedProviders ?? []).join(", "));
  const [allowedClasses, setAllowedClasses] = useState((cfg?.allowedCapacityClasses ?? []).join(", "));
  const [deniedAccelerators, setDeniedAccelerators] = useState((cfg?.deniedAcceleratorTypes ?? []).join(", "));
  const [maxPriority, setMaxPriority] = useState(String(cfg?.maxPriority ?? 1000));
  const [preemptionEnabled, setPreemptionEnabled] = useState(initial?.preemptionEnabled ?? true);
  const [crossRegionPolicy, setCrossRegionPolicy] = useState<CrossRegionTransferPolicy>(
    normalizeCrossRegionTransferPolicy(initial?.crossRegionTransferPolicy),
  );
  const [allowedBuckets, setAllowedBuckets] = useState((initial?.crossRegionAllowedBuckets ?? []).join(", "));
  const [allowMultiregion, setAllowMultiregion] = useState(initial?.crossRegionAllowMultiregion ?? true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const c = initial?.configuration;
    setAllowedRegions((c?.allowedRegions ?? []).join(", "));
    setAllowedProviders((c?.allowedProviders ?? []).join(", "));
    setAllowedClasses((c?.allowedCapacityClasses ?? []).join(", "));
    setDeniedAccelerators((c?.deniedAcceleratorTypes ?? []).join(", "));
    setMaxPriority(String(c?.maxPriority ?? 1000));
    setPreemptionEnabled(initial?.preemptionEnabled ?? true);
    setCrossRegionPolicy(normalizeCrossRegionTransferPolicy(initial?.crossRegionTransferPolicy));
    setAllowedBuckets((initial?.crossRegionAllowedBuckets ?? []).join(", "));
    setAllowMultiregion(initial?.crossRegionAllowMultiregion ?? true);
    setError(null);
  }, [open, initial]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    globalThis.addEventListener("keydown", onKeyDown);
    return () => globalThis.removeEventListener("keydown", onKeyDown);
  }, [open, busy, onClose]);

  if (!open) return null;

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const parsedPriority = Number.parseInt(maxPriority, 10);
    if (!Number.isFinite(parsedPriority) || parsedPriority < 0) {
      setError("Max priority must be a non-negative integer.");
      return;
    }
    const target: SchedulingPolicy = initial ?? {
      id: `${orgId}/${projectId}`,
      name: `Placement policy (${projectId})`,
      scope: projectId,
      type: "placement",
      summary: "",
      preemptionEnabled,
      crossRegionTransferPolicy: crossRegionPolicy,
      crossRegionAllowedBuckets: splitCsv(allowedBuckets),
      crossRegionAllowMultiregion: allowMultiregion,
      updatedAt: new Date().toISOString(),
      configuration: {
        allowedProviders: splitCsv(allowedProviders),
        allowedRegions: splitCsv(allowedRegions),
        allowedCapacityClasses: splitCsv(allowedClasses),
        deniedAcceleratorTypes: splitCsv(deniedAccelerators),
        allowedSecretReferences: [],
        maxPriority: parsedPriority,
      },
    };
    setBusy(true);
    setError(null);
    try {
      await onSave(target, {
        allowedRegions: splitCsv(allowedRegions),
        allowedProviders: splitCsv(allowedProviders),
        allowedCapacityClasses: splitCsv(allowedClasses),
        deniedAcceleratorTypes: splitCsv(deniedAccelerators),
        maxPriority: parsedPriority,
        preemptionEnabled,
        crossRegionTransferPolicy: crossRegionPolicy,
        crossRegionAllowedBuckets: splitCsv(allowedBuckets),
        crossRegionAllowMultiregion: allowMultiregion,
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save scheduling policy");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-layer" role="presentation">
      <button
        className="modal-scrim"
        type="button"
        aria-label="Close scheduling policy dialog"
        disabled={busy}
        onClick={onClose}
      />
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="policy-modal-title">
        <div className="modal-header">
          <div>
            <h2 id="policy-modal-title">{initial ? `Edit ${initial.name}` : "Configure project scheduling policy"}</h2>
            <p>
              Scope:{" "}
              <span className="monospace">
                {orgId}/{projectId}
              </span>
            </p>
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label="Close scheduling policy dialog"
            disabled={busy}
            onClick={onClose}
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>
        <form onSubmit={(event) => void handleSubmit(event)}>
          <div className="modal-body collab-form-grid">
            {error && (
              <div className="form-error" role="alert">
                <AlertCircle size={15} aria-hidden="true" />
                {error}
              </div>
            )}
            <div className="collab-form-row">
              <label className="field">
                <span>Allowed regions (comma-separated)</span>
                <input
                  type="text"
                  placeholder="e.g. europe-west4, us-central2"
                  value={allowedRegions}
                  disabled={busy}
                  onChange={(event) => setAllowedRegions(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Allowed providers (comma-separated)</span>
                <input
                  type="text"
                  placeholder="e.g. gcp"
                  value={allowedProviders}
                  disabled={busy}
                  onChange={(event) => setAllowedProviders(event.target.value)}
                />
              </label>
            </div>
            <div className="collab-form-row">
              <label className="field">
                <span>Allowed capacity classes</span>
                <input
                  type="text"
                  placeholder="spot, on_demand, reserved"
                  value={allowedClasses}
                  disabled={busy}
                  onChange={(event) => setAllowedClasses(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Denied accelerator types</span>
                <input
                  type="text"
                  placeholder="e.g. v4"
                  value={deniedAccelerators}
                  disabled={busy}
                  onChange={(event) => setDeniedAccelerators(event.target.value)}
                />
              </label>
            </div>
            <div className="collab-form-row">
              <label className="field">
                <span>Max priority</span>
                <input
                  type="number"
                  min={0}
                  required
                  value={maxPriority}
                  disabled={busy}
                  onChange={(event) => setMaxPriority(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Cross-region transfer policy</span>
                <select
                  value={crossRegionPolicy}
                  disabled={busy}
                  onChange={(event) => setCrossRegionPolicy(event.target.value as CrossRegionTransferPolicy)}
                >
                  <option value="allow">Allow</option>
                  <option value="warn">Warn</option>
                  <option value="ban">Ban &amp; kill</option>
                </select>
              </label>
            </div>
            <label className="field">
              <span>Cross-region allowlisted buckets (comma-separated)</span>
              <input
                type="text"
                placeholder="example-monitoring-bucket, example-shared-bucket"
                value={allowedBuckets}
                disabled={busy}
                onChange={(event) => setAllowedBuckets(event.target.value)}
              />
            </label>
            <label className="collab-checkbox-field">
              <input
                type="checkbox"
                checked={preemptionEnabled}
                disabled={busy}
                onChange={(event) => setPreemptionEnabled(event.target.checked)}
              />
              <span>Preemption enabled for workloads in this project</span>
            </label>
            <label className="collab-checkbox-field">
              <input
                type="checkbox"
                checked={allowMultiregion}
                disabled={busy}
                onChange={(event) => setAllowMultiregion(event.target.checked)}
              />
              <span>Allow continental multi-region buckets (uncheck for strict same-region)</span>
            </label>
          </div>
          <div className="modal-footer">
            <button className="button button-secondary" type="button" disabled={busy} onClick={onClose}>
              Cancel
            </button>
            <button className="button button-primary" type="submit" disabled={busy}>
              {busy ? "Saving…" : "Save policy"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function BudgetsPage({
  budgets,
  policies,
  onPolicyToggle,
  onCrossRegionPolicyChange,
  onPolicyUpdate,
  onSaveBudget,
  orgId = "default",
  projectId = "default",
  canManageProject,
}: BudgetsPageProps) {
  const [pendingPolicy, setPendingPolicy] = useState<string | null>(null);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const [budgetModal, setBudgetModal] = useState<{ open: boolean; target: Budget | null }>({
    open: false,
    target: null,
  });
  const [policyModal, setPolicyModal] = useState<{ open: boolean; target: SchedulingPolicy | null }>({
    open: false,
    target: null,
  });
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

  const canEditBudget = canManageProject && Boolean(onSaveBudget);
  const canEditFullPolicy = canManageProject && Boolean(onPolicyUpdate);

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
            {canEditBudget && (
              <button
                className="button button-primary"
                type="button"
                onClick={() => setBudgetModal({ open: true, target: budgets[0] ?? null })}
              >
                <Plus size={14} aria-hidden="true" />
                Configure budget
              </button>
            )}
          </div>
          {budgets.length === 0 ? (
            <section className="panel">
              <EmptyState
                title="No budgets configured for this scope"
                detail="Define project or organization budgets to enforce spend limits and alert thresholds."
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
                      {canEditBudget ? (
                        <button
                          className="icon-button icon-button-small"
                          type="button"
                          aria-label={`Edit ${budget.owner} budget`}
                          title={`Edit ${budget.owner} budget`}
                          onClick={() => setBudgetModal({ open: true, target: budget })}
                        >
                          <Edit3 size={14} aria-hidden="true" />
                        </button>
                      ) : (
                        <span title="Project admin role required to edit budgets">
                          <button
                            className="icon-button icon-button-small"
                            type="button"
                            aria-label={`Editing ${budget.owner} budget is unavailable`}
                            disabled
                          >
                            <Edit3 size={14} aria-hidden="true" />
                          </button>
                        </span>
                      )}
                    </div>
                    <div className="budget-owner">{budget.owner}</div>
                    <div className="budget-amount-row">
                      <strong>{formatMoney(budget.spent)}</strong>
                      <span>of {formatMoney(budget.limit)}</span>
                    </div>
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
          {canEditFullPolicy && (
            <button
              className="button button-secondary"
              type="button"
              onClick={() => setPolicyModal({ open: true, target: policies[0] ?? null })}
            >
              <Plus size={14} aria-hidden="true" />
              Configure policy
            </button>
          )}
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
                  {canEditFullPolicy && <th aria-label="Edit policy" />}
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
                      {canEditFullPolicy && (
                        <td className="numeric">
                          <button
                            className="button button-ghost"
                            type="button"
                            aria-label={`Edit ${policy.name}`}
                            disabled={isPending}
                            onClick={() => setPolicyModal({ open: true, target: policy })}
                          >
                            <Edit3 size={13} aria-hidden="true" />
                            Edit
                          </button>
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {canEditBudget && onSaveBudget && (
        <BudgetFormModal
          open={budgetModal.open}
          orgId={orgId}
          projectId={projectId}
          initial={budgetModal.target}
          onClose={() => setBudgetModal({ open: false, target: null })}
          onSave={async (input) => {
            await onSaveBudget(input);
          }}
        />
      )}

      {canEditFullPolicy && onPolicyUpdate && (
        <PolicyFormModal
          open={policyModal.open}
          orgId={orgId}
          projectId={projectId}
          initial={policyModal.target}
          onClose={() => setPolicyModal({ open: false, target: null })}
          onSave={async (target, patch) => {
            await onPolicyUpdate(target, patch);
          }}
        />
      )}
    </div>
  );
}
