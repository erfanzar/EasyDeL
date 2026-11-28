// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useRef, useState } from "react";
import { BlobMedia, type BlobLoader } from "../components/BlobMedia";
import { EmptyState } from "../components/EmptyState";
import { LoadingState } from "../components/PageState";
import { TrackingState } from "../components/TrackingState";
import type {
  BlobRef,
  RunDetail,
  RunMediaOptions,
  RunMediaResponse,
  RunTable,
  RunTableKeys,
  RunTableOptions,
  RunTableResponse,
  TableCell,
} from "../types";

export interface MediaLoaders {
  loadTable: (id: string, options: RunTableOptions, signal: AbortSignal) => Promise<RunTableResponse>;
  loadMedia: (id: string, options: RunMediaOptions, signal: AbortSignal) => Promise<RunMediaResponse>;
  loadBlob: BlobLoader;
}
function isBlob(value: TableCell): value is BlobRef {
  return typeof value === "object" && value !== null && "sha256" in value;
}
function MediaViewer({
  runId,
  blob,
  caption,
  loadBlob,
  revision,
  onRetry,
  onClose,
}: {
  runId: string;
  blob: BlobRef;
  caption: string;
  loadBlob: BlobLoader;
  revision: number;
  onRetry: () => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    // Native modal focus containment in browsers; open fallback for jsdom.
    if (element && typeof element.showModal === "function") element.showModal();
    else element?.setAttribute("open", "");
    return () => {
      if (element?.open && typeof element.close === "function") element.close();
    };
  }, []);
  return (
    <dialog
      ref={dialog}
      className="tracking-media-dialog"
      aria-label="Media viewer"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <div className="panel-header">
        <h2>Media viewer</h2>
        <button type="button" className="button button-secondary" onClick={onClose}>
          Close viewer
        </button>
      </div>
      <BlobMedia
        runId={runId}
        blob={blob}
        key={blob.sha256}
        caption={caption}
        loadBlob={loadBlob}
        revision={revision}
        onRetry={onRetry}
      />
    </dialog>
  );
}

export function RunMedia({
  run,
  loadTable,
  loadMedia,
  loadBlob,
  revision,
  onRetry,
}: MediaLoaders & { run: RunDetail; revision: number; onRetry: () => void }) {
  const [selected, setSelected] = useState(""),
    [step, setStep] = useState("");
  const [offset, setOffset] = useState(0),
    [viewer, setViewer] = useState<{ runId: string; blob: BlobRef; caption: string } | null>(null);
  const query = JSON.stringify([run.run_id, offset]);
  const [result, setResult] = useState<{
    query: string;
    media?: RunMediaResponse;
    keys?: RunTableKeys;
    error?: Error;
  } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      loadMedia(run.run_id, { limit: 100, offset }, controller.signal),
      loadTable(run.run_id, {}, controller.signal),
    ])
      .then(([media, tables]) => {
        if (!controller.signal.aborted) setResult({ query, media, keys: "keys" in tables ? tables : { keys: [] } });
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setResult((prev) =>
            prev?.query === query && prev.media
              ? prev
              : { query, error: reason instanceof Error ? reason : new Error("Unable to load media") },
          );
      });
    return () => controller.abort();
  }, [run.run_id, offset, query, loadMedia, loadTable, revision]);
  const keys = [
    ...new Set([
      ...(run.table_keys ?? []),
      ...(result?.query === query ? (result.keys?.keys.map((item) => item.key) ?? []) : []),
    ]),
  ];
  const key = keys.includes(selected) ? selected : keys[0];
  const tableQuery = JSON.stringify([run.run_id, key, step]);
  const [table, setTable] = useState<{ query: string; data?: RunTable; error?: Error } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    if (key)
      loadTable(run.run_id, { key, ...(step === "" ? {} : { step: Number(step) }) }, controller.signal)
        .then((data) => {
          if (!controller.signal.aborted && "columns" in data) setTable({ query: tableQuery, data });
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted)
            setTable((prev) =>
              prev?.query === tableQuery && prev.data
                ? prev
                : {
                    query: tableQuery,
                    error: reason instanceof Error ? reason : new Error("Unable to load table"),
                  },
            );
        });
    return () => controller.abort();
  }, [run.run_id, key, step, tableQuery, loadTable, revision]);
  if (result?.query !== query) return <LoadingState label="Loading media and tables" />;
  if (result.error) return <TrackingState error={result.error} onRetry={onRetry} />;
  const data = table?.query === tableQuery ? table.data : undefined;
  const media = result.media!;
  const identities = new Map<string, number>();
  const cards = media.media.map((item) => {
    const identity = JSON.stringify([item.key, item.step, item.t, item.blob.sha256]);
    // Older trackers expose no segment ID. Disambiguate identical metadata
    // within that identity only, never by position in the whole newest-first list.
    const occurrence = identities.get(identity) ?? 0;
    identities.set(identity, occurrence + 1);
    return { item, key: `${identity}-${occurrence}` };
  });
  return (
    <div className="page-stack">
      <div className="run-section-header">
        <h3>m.0 · tables</h3>
        <span>Latest immutable snapshot</span>
      </div>
      {key ? (
        <section className="run-chart-cell histogram-panel">
          <div className="tracking-toolbar">
            <label>
              Table key{" "}
              <select
                aria-label="Table key"
                value={key}
                onChange={(event) => {
                  setSelected(event.target.value);
                  setStep("");
                }}
              >
                {keys.map((key) => (
                  <option key={key} value={key}>
                    {key}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Step{" "}
              <select aria-label="Table step" value={step} onChange={(event) => setStep(event.target.value)}>
                <option value="">Latest</option>
                {(data?.steps ?? []).map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <span>
              Step {data?.step ?? "unknown"}
              {data ? ` · ${data.rows.length.toLocaleString()} rows · ${data.columns.length} cols` : ""}
            </span>
          </div>
          {table?.query !== tableQuery ? (
            <LoadingState label="Loading table" />
          ) : table.error ? (
            <TrackingState error={table.error} onRetry={onRetry} />
          ) : data ? (
            <div className="table-scroll">
              <table className="data-table tracking-table">
                <thead>
                  <tr>
                    {data.columns.map((name) => (
                      <th key={name}>{name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((row, i) => (
                    <tr key={i}>
                      {row.map((cell, j) => (
                        <td key={j}>
                          {isBlob(cell) ? (
                            <button
                              type="button"
                              className="run-tag blob-chip"
                              aria-label={`Open blob row ${i + 1} column ${data.columns[j]}`}
                              onClick={() => setViewer({ runId: run.run_id, blob: cell, caption: "Table blob" })}
                            >
                              blob · {cell.content_type}
                            </button>
                          ) : cell === null ? (
                            "—"
                          ) : (
                            String(cell)
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {!data.rows.length && <p className="run-panel-empty">No rows in this table snapshot.</p>}
            </div>
          ) : null}
        </section>
      ) : (
        <EmptyState title="No tables yet" detail="Log evaluation cases with run.table()." />
      )}
      <div className="run-section-header">
        <h3>m.1 · media</h3>
        <span>{media.total} items · authenticated blob reads</span>
      </div>
      {!media.total ? (
        <EmptyState title="No media yet" detail="Log images, video or audio with run.media()." />
      ) : (
        <>
          <div className="tracking-media-grid">
            {cards.map(({ item, key }, index) => (
              <article className="run-chart-cell histogram-panel" key={key}>
                <div className="chart-cell-heading">
                  <span className="cell-index">m.{String(index + 1).padStart(2, "0")}</span>
                  <h4>{item.key}</h4>
                  <span>step {item.step ?? "unknown"}</span>
                </div>
                <button
                  type="button"
                  className="button button-secondary"
                  aria-label={`Open media ${item.caption ?? item.key}`}
                  onClick={() => setViewer({ runId: run.run_id, blob: item.blob, caption: item.caption ?? item.key })}
                >
                  Open media
                </button>
                {item.blob.available === false && <p className="run-panel-empty">Media not uploaded · unavailable</p>}
                {item.caption && <p className="media-caption">{item.caption}</p>}
                <p className="episode-note">
                  {item.blob.content_type} · {item.blob.bytes.toLocaleString()} bytes
                </p>
              </article>
            ))}
          </div>
          <div className="tracking-toolbar">
            <button type="button" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 100))}>
              Previous media
            </button>
            <button type="button" disabled={offset + 100 >= media.total} onClick={() => setOffset(offset + 100)}>
              Next media
            </button>
          </div>
        </>
      )}
      {viewer?.runId === run.run_id && (
        <MediaViewer
          key={JSON.stringify([run.run_id, viewer.blob.sha256])}
          runId={run.run_id}
          blob={media.media.find((item) => item.blob.sha256 === viewer.blob.sha256)?.blob ?? viewer.blob}
          caption={viewer.caption}
          loadBlob={loadBlob}
          revision={revision}
          onRetry={onRetry}
          onClose={() => setViewer(null)}
        />
      )}
    </div>
  );
}
