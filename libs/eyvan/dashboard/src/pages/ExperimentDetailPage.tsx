// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ArrowLeft, Edit3, FlaskConical, GitCompareArrows, RefreshCw, Star, Trash2, Trophy } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { EmptyState } from "../components/EmptyState";
import { ErrorCard } from "../components/ErrorCard";
import { LoadingState } from "../components/PageState";
import { TeamNotesPanel } from "../components/TeamNotesPanel";
import { formatDurationSeconds } from "../lib/format";
import { formatChartValue } from "../lib/metricGrouping";
import type {
  CollaborationNote,
  CollaborationNoteInput,
  ExperimentLeaderboardResponse,
  ExperimentRecord,
  ExperimentUpdatePatch,
  RunVisibility,
} from "../types";
import { ExperimentFormModal } from "./ExperimentsPage";
import { RunStatusBadge, RunVisibilityBadge } from "./RunsPage";

export interface ExperimentDetailPageProps {
  experimentId: string;
  loadLeaderboard: (expId: string, signal: AbortSignal) => Promise<ExperimentLeaderboardResponse>;
  onUpdateExperiment: (expId: string, patch: ExperimentUpdatePatch) => Promise<ExperimentRecord>;
  onDeleteExperiment: (expId: string) => Promise<void>;
  loadNotes: (targetType: "run" | "experiment", targetId: string, signal: AbortSignal) => Promise<CollaborationNote[]>;
  onCreateNote: (input: CollaborationNoteInput) => Promise<CollaborationNote>;
  onDeleteNote: (noteId: string) => Promise<void>;
  onBack: () => void;
  onOpenRun: (runId: string) => void;
  onCompareRuns: (runIds: string[]) => void;
  currentSubject?: string | null;
  canAdminister?: boolean;
  onNotify?: (message: string) => void;
  refreshRevision?: number;
}

export function ExperimentDetailPage({
  experimentId,
  loadLeaderboard,
  onUpdateExperiment,
  onDeleteExperiment,
  loadNotes,
  onCreateNote,
  onDeleteNote,
  onBack,
  onOpenRun,
  onCompareRuns,
  currentSubject,
  canAdminister = false,
  onNotify,
  refreshRevision = 0,
}: ExperimentDetailPageProps) {
  const [board, setBoard] = useState<ExperimentLeaderboardResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [revision, setRevision] = useState(0);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [editing, setEditing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [pinningId, setPinningId] = useState<string | null>(null);
  const [visBusy, setVisBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    loadLeaderboard(experimentId, controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) {
          setBoard(value);
          setError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason : new Error("Unable to load experiment leaderboard"));
        }
      });
    return () => controller.abort();
  }, [experimentId, loadLeaderboard, refreshRevision, revision]);

  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);

  if (error && !board) {
    return (
      <div className="page-stack">
        <section className="panel">
          <div className="panel-alert-wrap">
            <ErrorCard
              error={error}
              context={{ fallbackTitle: "Unable to load experiment" }}
              actions={[
                { label: "Back to experiments", onClick: onBack },
                { label: "Retry", onClick: () => setRevision((v) => v + 1) },
              ]}
            />
          </div>
        </section>
      </div>
    );
  }

  if (!board) return <LoadingState label="Loading experiment leaderboard" />;

  const { experiment } = board;
  const canEdit = Boolean(canAdminister || !currentSubject || experiment.owner === currentSubject);
  const goalOp = board.goal === "min" ? "≤" : "≥";
  const topCompareIds = board.entries.slice(0, 4).map((e) => e.run_id);

  const toggleSelect = (runId: string) => {
    setSelectedIds((prev) => (prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId]));
  };

  const pinBaseline = async (runId: string) => {
    setPinningId(runId);
    setActionError(null);
    try {
      const nextBaseline = board.baseline_run_id === runId ? null : runId;
      await onUpdateExperiment(experiment.id, { pinned_baseline_run_id: nextBaseline });
      onNotify?.(nextBaseline ? `Pinned ${runId} as experiment baseline` : "Cleared pinned baseline");
      setRevision((v) => v + 1);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Unable to update baseline");
    } finally {
      setPinningId(null);
    }
  };

  const changeVisibility = async (visibility: RunVisibility) => {
    setVisBusy(true);
    setActionError(null);
    try {
      const updated = await onUpdateExperiment(experiment.id, { visibility });
      setBoard((prev) => (prev ? { ...prev, experiment: updated } : prev));
      onNotify?.(`Experiment visibility set to ${updated.visibility}`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Unable to update visibility");
    } finally {
      setVisBusy(false);
    }
  };

  const handleDelete = async () => {
    setDeleting(true);
    setDeleteError(null);
    try {
      await onDeleteExperiment(experiment.id);
      setConfirmDelete(false);
      onNotify?.(`Deleted ${experiment.name}`);
      onBack();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Unable to delete experiment");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="page-stack">
      <div className="tracking-meta">
        <span>{experiment.kind === "speedrun" ? "Speedrun" : "Experiment"}</span>
        <span>{experiment.id}</span>
        {board.metric_key && (
          <span>
            Metric · {board.metric_key} ({board.goal}
            {board.target_value !== null ? ` ${goalOp} ${board.target_value}` : ""})
          </span>
        )}
        <span>{board.entries.length} runs</span>
        <button
          type="button"
          className="tracking-meta-action"
          onClick={() => setRevision((v) => v + 1)}
          aria-label="Refresh leaderboard"
        >
          <RefreshCw size={11} aria-hidden="true" />
          Refresh
        </button>
      </div>

      <section className="panel exp-detail-header">
        <div className="exp-detail-header-top">
          <div className="exp-detail-title-block">
            <button className="text-button" type="button" onClick={onBack}>
              <ArrowLeft size={13} aria-hidden="true" />
              All experiments
            </button>
            <div className="exp-detail-title-row">
              <span className={`experiment-kind-badge experiment-kind-${experiment.kind}`}>
                {experiment.kind === "speedrun" ? (
                  <Trophy size={12} aria-hidden="true" />
                ) : (
                  <FlaskConical size={12} aria-hidden="true" />
                )}
                {experiment.kind === "speedrun" ? "Speedrun" : "Experiment"}
              </span>
              <h2>{experiment.name}</h2>
              {experiment.subject && <span className="experiment-subject-pill">{experiment.subject}</span>}
            </div>
            {experiment.description && <p className="exp-detail-desc">{experiment.description}</p>}
            <div className="exp-detail-meta-row">
              <span>
                Owner <strong className="collab-owner-pill">{experiment.owner}</strong>
              </span>
              {experiment.group_filter && (
                <span>
                  Group filter <code className="monospace">{experiment.group_filter}</code>
                </span>
              )}
              {board.baseline_run_id && (
                <span>
                  Baseline <code className="monospace">{board.baseline_run_id}</code>
                  {board.baseline_value !== null ? ` (${formatChartValue(board.baseline_value)})` : ""}
                </span>
              )}
            </div>
          </div>

          <div className="exp-detail-actions">
            {canEdit ? (
              <label className="select-field collab-compact-select">
                <span className="sr-only">Experiment visibility</span>
                <select
                  aria-label="Experiment visibility"
                  value={experiment.visibility}
                  disabled={visBusy}
                  onChange={(event) => void changeVisibility(event.target.value as RunVisibility)}
                >
                  <option value="team">Team</option>
                  <option value="shared">Shared</option>
                  <option value="private">Private</option>
                </select>
              </label>
            ) : (
              <RunVisibilityBadge visibility={experiment.visibility} sharedWith={experiment.shared_with} />
            )}

            {topCompareIds.length >= 2 && (
              <button type="button" className="button button-primary" onClick={() => onCompareRuns(topCompareIds)}>
                <GitCompareArrows size={13} aria-hidden="true" />
                Compare top runs
              </button>
            )}
            {canEdit && (
              <>
                <button type="button" className="button button-secondary" onClick={() => setEditing(true)}>
                  <Edit3 size={13} aria-hidden="true" />
                  Edit
                </button>
                <button
                  type="button"
                  className="button button-secondary"
                  aria-label="Delete experiment"
                  onClick={() => {
                    setDeleteError(null);
                    setConfirmDelete(true);
                  }}
                >
                  <Trash2 size={13} aria-hidden="true" />
                  Delete
                </button>
              </>
            )}
          </div>
        </div>
        {actionError && (
          <div className="form-error" role="alert">
            {actionError}
          </div>
        )}
      </section>

      {!board.tracker_available && (
        <ErrorCard
          compact
          tone="warning"
          role="status"
          title="Experiment tracker offline"
          summary="Live runs could not be queried from the tracker; metadata and notes remain available."
          error="Tracker unavailable"
          actions={[{ label: "Retry", onClick: () => setRevision((v) => v + 1) }]}
        />
      )}

      {board.researcher_breakdown.length > 0 && (
        <section className="panel exp-researchers-strip" aria-label="Researcher breakdown">
          <div className="storage-subhead">
            <span>Researcher breakdown ({board.researcher_breakdown.length})</span>
          </div>
          <div className="exp-researcher-grid">
            {board.researcher_breakdown.map((item) => (
              <div key={item.owner} className="exp-researcher-card">
                <div className="exp-researcher-card-head">
                  <span className="collab-owner-pill">{item.owner}</span>
                  {item.running_count > 0 && <span className="collab-researcher-live">{item.running_count} live</span>}
                </div>
                <div className="exp-researcher-stats">
                  <span>
                    <strong>{item.run_count}</strong> {item.run_count === 1 ? "run" : "runs"}
                  </span>
                  <span>
                    Best <strong className="monospace">{formatChartValue(item.best_value)}</strong>
                  </span>
                  {board.target_value !== null && (
                    <span>
                      <strong>{item.reached_target_count}</strong> hit target
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="panel data-panel" aria-label="Speedrun leaderboard">
        <div className="panel-header">
          <div>
            <h2>{experiment.kind === "speedrun" ? "Speedrun leaderboard" : "Experiment runs"}</h2>
            <p>
              {board.metric_key
                ? `Ranked by ${board.target_value !== null ? `time to reach ${board.metric_key} ${goalOp} ${board.target_value}, then ` : ""}${board.goal}(${board.metric_key})`
                : "Matched runs sorted by recency"}
            </p>
          </div>
          {selectedIds.length >= 2 && (
            <button type="button" className="button button-primary" onClick={() => onCompareRuns(selectedIds)}>
              <GitCompareArrows size={13} aria-hidden="true" />
              Compare selected ({selectedIds.length})
            </button>
          )}
        </div>

        {board.entries.length ? (
          <div className="table-scroll">
            <table className="data-table leaderboard-table">
              <thead>
                <tr>
                  <th className="runs-select-col">
                    <span className="sr-only">Select</span>
                  </th>
                  <th>Rank</th>
                  <th>Run</th>
                  <th>Researcher</th>
                  <th>Status</th>
                  <th className="numeric">Best</th>
                  <th className="numeric">Latest</th>
                  <th className="numeric">Δ vs baseline</th>
                  <th>Target progress</th>
                  <th className="numeric">Duration</th>
                  {canEdit && <th aria-label="Baseline" />}
                </tr>
              </thead>
              <tbody>
                {board.entries.map((entry) => {
                  const isBaseline = Boolean(entry.is_baseline || entry.run_id === board.baseline_run_id);
                  const delta = entry.delta_vs_baseline;
                  const deltaImproved = delta !== null && delta !== 0 && (board.goal === "min" ? delta < 0 : delta > 0);
                  const deltaRegressed =
                    delta !== null && delta !== 0 && (board.goal === "min" ? delta > 0 : delta < 0);
                  const targetCell =
                    board.target_value === null ? (
                      <span className="muted">—</span>
                    ) : entry.reached_target ? (
                      <span className="leaderboard-target-hit">
                        Reached
                        {entry.time_to_target_s !== null ? ` in ${Math.round(entry.time_to_target_s)}s` : ""}
                        {entry.steps_to_target !== null ? ` / ${entry.steps_to_target.toLocaleString()} steps` : ""}
                      </span>
                    ) : (
                      <span className="leaderboard-target-pending">In progress</span>
                    );
                  return (
                    <tr key={entry.run_id} className={entry.is_record ? "leaderboard-record-row" : undefined}>
                      <td className="runs-select-col">
                        <input
                          type="checkbox"
                          aria-label={`Select ${entry.name} for comparison`}
                          checked={selectedSet.has(entry.run_id)}
                          onChange={() => toggleSelect(entry.run_id)}
                        />
                      </td>
                      <td>
                        <span className={`leaderboard-rank${entry.is_record ? " leaderboard-rank-gold" : ""}`}>
                          {entry.is_record && <Trophy size={11} aria-hidden="true" />}#{entry.rank}
                        </span>
                      </td>
                      <td className="run-name-cell">
                        <div className="run-name-row">
                          <button
                            type="button"
                            className="text-button run-name"
                            onClick={() => onOpenRun(entry.run_id)}
                          >
                            {entry.name}
                          </button>
                          {isBaseline && <span className="compare-baseline-tag">baseline</span>}
                        </div>
                        <small className="run-id">{entry.run_id}</small>
                      </td>
                      <td>
                        {entry.owner ? (
                          <span className="collab-owner-pill">{entry.owner}</span>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td>
                        <RunStatusBadge status={entry.status} />
                      </td>
                      <td className="numeric monospace">
                        <strong>{formatChartValue(entry.best_value)}</strong>
                      </td>
                      <td className="numeric monospace">{formatChartValue(entry.latest_value)}</td>
                      <td className="numeric monospace">
                        {delta !== null ? (
                          <span
                            className={`compare-delta${deltaImproved ? " compare-delta-good" : deltaRegressed ? " compare-delta-bad" : ""}`}
                          >
                            {delta >= 0 ? "+" : ""}
                            {formatChartValue(delta, true)}
                          </span>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td>{targetCell}</td>
                      <td className="numeric nowrap">
                        {typeof entry.duration_s === "number" && Number.isFinite(entry.duration_s)
                          ? formatDurationSeconds(entry.duration_s)
                          : "—"}
                      </td>
                      {canEdit && (
                        <td className="numeric">
                          <button
                            type="button"
                            className={`button button-ghost${isBaseline ? " collab-pin-active" : ""}`}
                            aria-label={
                              isBaseline ? `Unpin ${entry.name} as baseline` : `Pin ${entry.name} as baseline`
                            }
                            disabled={pinningId === entry.run_id}
                            onClick={() => void pinBaseline(entry.run_id)}
                          >
                            <Star size={12} fill={isBaseline ? "currentColor" : "none"} aria-hidden="true" />
                            {isBaseline ? "Baseline" : "Pin"}
                          </button>
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No matching runs yet"
            detail="Link runs to this experiment from Run Detail, or tag/group runs to match its filter."
          />
        )}
      </section>

      <TeamNotesPanel
        targetType="experiment"
        targetId={experiment.id}
        loadNotes={loadNotes}
        onCreateNote={onCreateNote}
        onDeleteNote={onDeleteNote}
        currentSubject={currentSubject}
        canAdminister={canAdminister}
        availableMetrics={board.metric_key ? [board.metric_key] : []}
        refreshRevision={refreshRevision + revision}
      />

      <ExperimentFormModal
        open={editing}
        initial={experiment}
        onClose={() => setEditing(false)}
        onSubmit={async (input) => {
          const updated = await onUpdateExperiment(experiment.id, input);
          setBoard((prev) => (prev ? { ...prev, experiment: updated } : prev));
          setRevision((v) => v + 1);
          onNotify?.(`Updated ${updated.name}`);
        }}
      />

      <ConfirmDialog
        open={confirmDelete}
        title={`Delete ${experiment.name}?`}
        consequence="This removes the experiment metadata and leaderboard configuration. Individual tracked runs remain untouched."
        confirmLabel="Delete experiment"
        destructive
        busy={deleting}
        error={deleteError}
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => void handleDelete()}
      />
    </div>
  );
}
