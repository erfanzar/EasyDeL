// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useRef, useState } from "react";
import { type BlobLoader, useRunBlob } from "../components/BlobMedia";
import { EmptyState } from "../components/EmptyState";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import type { EpisodeKind, RunDetail, RunRollout, RunRolloutsOptions, RunRolloutsResponse } from "../types";

export interface RolloutLoaders {
  loadRollouts: (id: string, options: RunRolloutsOptions, signal: AbortSignal) => Promise<RunRolloutsResponse>;
  loadRollout: (id: string, itemId: string, signal: AbortSignal) => Promise<RunRollout>;
  loadBlob: BlobLoader;
}
const COLORS = ["--amber", "--blue", "--ok", "--red", "--text-soft", "--violet"];
const color = (action: number) => `var(${COLORS[((action % COLORS.length) + COLORS.length) % COLORS.length]})`;
const fmt = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: 3 })
    : value == null
      ? "unknown"
      : String(value);
const LEFT = 74,
  RIGHT = 12,
  ROW = 50;

function RolloutTracks({
  rollout,
  position,
  seek,
}: {
  rollout: RunRollout;
  position: number;
  seek: (step: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const names = [
    ...new Set([
      ...["reward", "value"].filter((key) => key in rollout.tracks),
      ...(rollout.actions ? ["action"] : []),
      ...Object.keys(rollout.tracks),
    ]),
  ];
  const namesKey = names.join("\n");
  const maxStep = Math.max(0, rollout.length - 1);
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const draw = () => {
      const trackNames = namesKey.split("\n").filter(Boolean);
      const width = Math.max(220, element.parentElement?.clientWidth ?? 0),
        height = Math.max(1, trackNames.length) * ROW + 22;
      const ratio = globalThis.devicePixelRatio || 1;
      element.width = Math.round(width * ratio);
      element.height = Math.round(height * ratio);
      element.style.width = `${width}px`;
      element.style.height = `${height}px`;
      const ctx = element.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const style = getComputedStyle(element),
        token = (name: string) => style.getPropertyValue(name).trim();
      const plotWidth = Math.max(1, width - LEFT - RIGHT),
        safeLen = Math.max(1, rollout.length),
        sx = (step: number) => LEFT + (step / Math.max(1, rollout.length - 1)) * plotWidth;
      ctx.font = `10px ${token("--mono")}`;
      trackNames.forEach((name, row) => {
        const top = row * ROW;
        ctx.fillStyle = token("--text-soft");
        ctx.fillText(name, 0, top + 26, LEFT - 8);
        ctx.strokeStyle = token("--border");
        ctx.beginPath();
        ctx.moveTo(LEFT, top + ROW - 1);
        ctx.lineTo(width - RIGHT, top + ROW - 1);
        ctx.stroke();
        if (name === "action" && rollout.actions) {
          rollout.actions.forEach((action, step) => {
            if (action === null) return;
            ctx.fillStyle = token(COLORS[((action % COLORS.length) + COLORS.length) % COLORS.length]);
            ctx.fillRect(LEFT + (step / safeLen) * plotWidth, top + 12, Math.max(0.6, plotWidth / safeLen), ROW - 24);
          });
          return;
        }
        const values = rollout.tracks[name] ?? [],
          finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
        if (!finite.length) return;
        const lo = name === "reward" ? Math.min(0, ...finite) : Math.min(...finite),
          hi = name === "reward" ? Math.max(0, ...finite) : Math.max(...finite);
        // Scaling before subtraction avoids overflow; constants need no
        // artificial ±0.5 expansion (which rounds away at large magnitudes).
        const scale = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
        const sy = (value: number) =>
          top + 5 + (lo === hi ? 0.5 : (hi / scale - value / scale) / (hi / scale - lo / scale)) * (ROW - 10);
        ctx.strokeStyle = token(name === "entropy" ? "--violet" : "--amber");
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        let connected = false;
        values.forEach((value, step) => {
          if (typeof value !== "number" || !Number.isFinite(value)) {
            connected = false;
            return;
          }
          if (name === "reward") {
            ctx.moveTo(sx(step), sy(0));
            ctx.lineTo(sx(step), sy(value));
          } else {
            if (connected) ctx.lineTo(sx(step), sy(value));
            else ctx.moveTo(sx(step), sy(value));
            connected = true;
          }
        });
        ctx.stroke();
      });
      ctx.strokeStyle = token("--amber");
      ctx.beginPath();
      ctx.moveTo(sx(position), 0);
      ctx.lineTo(sx(position), trackNames.length * ROW);
      ctx.stroke();
      ctx.fillStyle = token("--text-soft");
      ctx.fillText("0", LEFT, height - 4);
      ctx.fillText(String(maxStep), width - RIGHT - 28, height - 4);
    };
    draw();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(draw);
    if (element.parentElement) observer?.observe(element.parentElement);
    const theme = new MutationObserver(draw);
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    globalThis.addEventListener("resize", draw);
    return () => {
      observer?.disconnect();
      theme.disconnect();
      globalThis.removeEventListener("resize", draw);
    };
  }, [rollout, position, namesKey, maxStep]);
  return (
    <canvas
      ref={canvas}
      className="rollout-tracks"
      role="img"
      aria-label={`Per-step tracks: ${names.join(", ")}. Click to seek, arrows to step, space to play.`}
      tabIndex={0}
      onClick={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const width = bounds.width || event.currentTarget.clientWidth || 220;
        const plotWidth = Math.max(1, width - LEFT - RIGHT);
        seek(Math.round(((event.clientX - bounds.left - LEFT) / plotWidth) * maxStep));
      }}
    />
  );
}

function RolloutPlayer({
  runId,
  rollout,
  loadBlob,
  onRetry,
}: {
  runId: string;
  rollout: RunRollout;
  loadBlob: BlobLoader;
  revision: number;
  onRetry: () => void;
}) {
  const maxStep = Math.max(0, rollout.length - 1);
  const [position, setPosition] = useState(0),
    [playing, setPlaying] = useState(false);
  const [playError, setPlayError] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const [blobRetry, setBlobRetry] = useState(0);
  const media = useRunBlob(runId, rollout.video, loadBlob, blobRetry);
  const secondsPerStep = () => {
    const seconds = rollout.fps != null && rollout.fps > 0 ? 1 / rollout.fps : NaN;
    if (Number.isFinite(seconds) && seconds > 0) return seconds;
    const duration = video.current?.duration;
    return duration != null && Number.isFinite(duration) && duration > 0
      ? duration / Math.max(1, rollout.length - 1)
      : 0;
  };
  const playVideo = () => {
    void video.current?.play().catch(() => {
      setPlaying(false);
      setPlayError(true);
    });
  };
  const seek = (next: number) => {
    const step = Math.max(0, Math.min(maxStep, next));
    setPosition(step);
    const seconds = secondsPerStep(),
      target = step * seconds;
    if (video.current && seconds > 0 && Number.isFinite(target) && target >= 0) video.current.currentTime = target;
  };
  useEffect(() => {
    if (!playing || media?.url) return;
    const timer = globalThis.setTimeout(
      () => {
        if (position >= maxStep) setPlaying(false);
        else setPosition(position + 1);
      },
      rollout.fps != null && Number.isFinite(rollout.fps) && rollout.fps > 0
        ? Math.max(4, Math.min(60_000, 1000 / rollout.fps))
        : 1000 / 30,
    );
    return () => globalThis.clearTimeout(timer);
  }, [playing, position, media?.url, maxStep, rollout.fps]);
  const toggle = () => {
    if (rollout.video && rollout.video.available !== false && !media?.url) return;
    if (playing) {
      video.current?.pause();
      setPlaying(false);
      return;
    }
    if (position === maxStep) seek(0);
    setPlayError(false);
    setPlaying(true);
    playVideo();
  };
  const currentAction = rollout.actions ? rollout.actions[position] : null;
  const currentActionLabel =
    currentAction == null ? "unknown" : (rollout.action_names?.[currentAction] ?? `action ${currentAction}`);

  return (
    <section
      className="run-chart-cell rollout-player"
      aria-label="Rollout playback"
      onKeyDown={(event) => {
        if ((event.target as HTMLElement).closest("button,input,select,textarea,video")) return;
        if (event.key === " " || event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault();
          if (event.key === " ") toggle();
          else seek(position + (event.key === "ArrowRight" ? 1 : -1));
        }
      }}
    >
      <div className="chart-cell-heading">
        <span className="cell-index">r.02</span>
        <h4 title={rollout.episode_id}>Per-step tracks · {rollout.episode_id}</h4>
        <span>click to seek · space plays</span>
      </div>
      <div className="rollout-video">
        {!rollout.video || rollout.video.available === false ? (
          <p className="rollout-video-empty">Video not uploaded · no video in demo or this sampled trajectory.</p>
        ) : media?.error ? (
          <TrackingState
            error={media.error}
            onRetry={() => {
              setBlobRetry((value) => value + 1);
              onRetry();
            }}
          />
        ) : media?.url ? (
          <video
            ref={video}
            className="tracking-media"
            src={media.url}
            controls
            aria-label="Rollout video"
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            onLoadedMetadata={() => {
              seek(position);
              if (playing) playVideo();
            }}
            onTimeUpdate={() => {
              const seconds = secondsPerStep();
              if (video.current && seconds > 0)
                setPosition(Math.max(0, Math.min(maxStep, Math.round(video.current.currentTime / seconds))));
            }}
          />
        ) : (
          <p className="run-panel-empty">Loading video…</p>
        )}
        {playError && <p className="episode-note">Video playback could not start. Use the video controls to retry.</p>}
      </div>
      <RolloutTracks rollout={rollout} position={position} seek={seek} />
      <div className="rollout-controls">
        <button
          type="button"
          className="button button-secondary"
          aria-label={playing ? "Pause rollout" : "Play rollout"}
          onClick={toggle}
          disabled={!!rollout.video && rollout.video.available !== false && !media?.url}
        >
          {playing ? "Ⅱ pause" : "▶ play"}
        </button>
        <input
          type="range"
          aria-label="Rollout step"
          min={0}
          max={maxStep}
          value={position}
          onChange={(event) => seek(Number(event.target.value))}
        />
      </div>
      <div className="chart-readout">
        <output data-testid="rollout-position">
          Step {position} / {maxStep}
        </output>
        {rollout.actions && (
          <span>
            action{" "}
            <strong>
              {currentActionLabel} · {fmt(currentAction)}
            </strong>
          </span>
        )}
        {Object.entries(rollout.tracks).map(([name, values]) => (
          <span key={name}>
            {name} <strong>{fmt(values[position])}</strong>
          </span>
        ))}
      </div>
      {rollout.actions && rollout.actions.some((action) => action !== null) && (
        <div className="action-legend" aria-label="Action legend">
          {[...new Set(rollout.actions.filter((action): action is number => action !== null))]
            .sort((a, b) => a - b)
            .map((action) => (
              <span key={action}>
                <i style={{ background: color(action) }} />
                {rollout.action_names?.[action] ?? `action ${action}`}
              </span>
            ))}
        </div>
      )}
    </section>
  );
}

export function RunRollouts({
  run,
  loadRollouts,
  loadRollout,
  loadBlob,
  revision,
  onRetry,
}: RolloutLoaders & { run: RunDetail; revision: number; onRetry: () => void }) {
  const [selected, setSelected] = useState("");
  const [kind, setKind] = useState<"" | EpisodeKind>("");
  const [offset, setOffset] = useState(0);
  const [list, setList] = useState<{ query: string; data?: RunRolloutsResponse; error?: Error } | null>(null);
  const query = JSON.stringify([run.run_id, kind, offset]);
  useEffect(() => {
    const controller = new AbortController();
    loadRollouts(run.run_id, { ...(kind ? { kind } : {}), limit: 50, offset }, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setList({ query, data });
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setList({ query, error: reason instanceof Error ? reason : new Error("Unable to load rollouts") });
      });
    return () => controller.abort();
  }, [run.run_id, kind, offset, query, loadRollouts, revision]);
  const id =
    list?.query === query
      ? (list.data?.rollouts.find((item) => item.id === selected)?.id ?? list.data?.rollouts[0]?.id)
      : undefined;
  const detailQuery = JSON.stringify([run.run_id, id]);
  const [detail, setDetail] = useState<{ query: string; data?: RunRollout; error?: Error } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    if (id)
      loadRollout(run.run_id, id, controller.signal)
        .then((data) => {
          if (!controller.signal.aborted) setDetail({ query: detailQuery, data });
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted)
            setDetail({
              query: detailQuery,
              error: reason instanceof Error ? reason : new Error("Unable to load rollout"),
            });
        });
    return () => controller.abort();
  }, [run.run_id, id, detailQuery, loadRollout, revision]);
  if (list?.query !== query) return <LoadingState label="Loading rollouts" />;
  if (list.error) return <TrackingState error={list.error} onRetry={onRetry} />;
  const total = list.data?.total ?? 0;
  const pageStart = list.data?.rollouts.length ? offset + 1 : 0;
  const pageEnd = offset + (list.data?.rollouts.length ?? 0);

  return (
    <div className="page-stack">
      <div className="tracking-toolbar">
        <label>
          Kind{" "}
          <select
            aria-label="Rollout kind"
            value={kind}
            onChange={(event) => {
              setKind(event.target.value as "" | EpisodeKind);
              setSelected("");
              setOffset(0);
            }}
          >
            <option value="">All kinds</option>
            <option value="train">train</option>
            <option value="eval">eval</option>
          </select>
        </label>
        <span>
          {total.toLocaleString()} sampled {total === 1 ? "trajectory" : "trajectories"}
        </span>
      </div>
      {!total ? (
        <EmptyState
          title={kind ? `No ${kind} rollouts` : "No rollouts yet"}
          detail={
            kind
              ? "Switch the kind filter back to All kinds to inspect other sampled trajectories."
              : "Log sampled trajectories with run.rollout()."
          }
        />
      ) : (
        <>
          <div className="run-section-header">
            <h3>r.0 · rollouts</h3>
            <span>
              Showing {pageStart}–{pageEnd} of {total.toLocaleString()}
            </span>
          </div>
          <div className="rollout-layout">
            <section className="run-chart-cell rollout-sidebar" aria-label="Sampled episodes">
              <div className="chart-cell-heading">
                <span className="cell-index">r.01</span>
                <h4>Sampled episodes</h4>
              </div>
              <div className="tracking-item-list rollout-item-list">
                {list.data!.rollouts.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    aria-pressed={id === item.id}
                    onClick={() => setSelected(item.id)}
                  >
                    <span className="rollout-item-id" title={item.episode_id}>
                      {item.episode_id}
                    </span>
                    <small>
                      {item.kind} · actor {item.actor ?? "unknown"} · v{item.policy_version ?? "?"}
                    </small>
                    <span>
                      return {fmt(item.ret)} · {item.length} steps
                    </span>
                  </button>
                ))}
              </div>
              <div className="tracking-toolbar rollout-pagination">
                <button
                  type="button"
                  className="button button-secondary"
                  disabled={!offset}
                  onClick={() => setOffset(Math.max(0, offset - 50))}
                >
                  Previous
                </button>
                <button
                  type="button"
                  className="button button-secondary"
                  disabled={offset + 50 >= total}
                  onClick={() => setOffset(offset + 50)}
                >
                  Next
                </button>
              </div>
            </section>
            {detail?.query !== detailQuery ? (
              <LoadingState label="Loading rollout" />
            ) : detail.error ? (
              <TrackingState error={detail.error} onRetry={onRetry} />
            ) : detail.data ? (
              <RolloutPlayer
                key={`${run.run_id}-${id}`}
                runId={run.run_id}
                rollout={detail.data}
                loadBlob={loadBlob}
                revision={revision}
                onRetry={onRetry}
              />
            ) : null}
          </div>
        </>
      )}
    </div>
  );
}
