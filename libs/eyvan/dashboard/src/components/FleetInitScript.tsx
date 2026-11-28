// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertCircle, AlertTriangle, RefreshCw, SquareTerminal, X } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";

import type { FleetInitScriptDocument } from "../types";
import { ConfirmDialog } from "./ConfirmDialog";

/** The two calls this editor makes; the real `EyvanApiClient` satisfies it. */
export interface FleetInitScriptClient {
  getFleetInitScript: (fleet: string, signal?: AbortSignal) => Promise<FleetInitScriptDocument>;
  setFleetInitScript: (fleet: string, script: string, signal?: AbortSignal) => Promise<FleetInitScriptDocument>;
}

interface FleetInitScriptProps {
  fleet: string;
  client: FleetInitScriptClient;
  onClose?: () => void;
}

const message = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);
const encoder = new TextEncoder();
/** The limit is in UTF-8 bytes on the server, so count bytes, not characters. */
const byteLength = (text: string) => encoder.encode(text).length;

/**
 * A fleet's bash init script, edited by a fleet administrator.
 *
 * Every answer carries a ticket. Only the newest request -- a load, a reload or
 * a save -- may write what the editor shows, so a slow read that started before
 * a save cannot put the old script back over the one the plane just stored,
 * and a read for a fleet the operator has already left cannot land on the next.
 */
export function FleetInitScript({ fleet, client, onClose }: FleetInitScriptProps) {
  const editorId = useId();
  const counterId = useId();
  const sequence = useRef(0);
  const loadController = useRef<AbortController | null>(null);
  const [document, setDocument] = useState<FleetInitScriptDocument | null>(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<"save" | "remove" | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(() => {
    loadController.current?.abort();
    const controller = new AbortController();
    loadController.current = controller;
    const ticket = ++sequence.current;
    setLoading(true);
    setLoadError(null);
    client
      .getFleetInitScript(fleet, controller.signal)
      .then((next) => {
        if (ticket !== sequence.current) return;
        setDocument(next);
        setDraft(next.script);
      })
      .catch((error: unknown) => {
        if (ticket === sequence.current) setLoadError(message(error, "Could not read this fleet's init script"));
      })
      .finally(() => {
        if (ticket === sequence.current) setLoading(false);
      });
  }, [client, fleet]);

  useEffect(() => {
    setDocument(null);
    setDraft("");
    setNotice(null);
    setConfirm(null);
    setSaveError(null);
    load();
    return () => {
      loadController.current?.abort();
      // Invalidate every request still in flight for this fleet.
      sequence.current += 1;
    };
  }, [load]);

  const submit = (script: string, kind: "save" | "remove") => {
    loadController.current?.abort();
    const ticket = ++sequence.current;
    setSaving(true);
    setSaveError(null);
    setNotice(null);
    client
      .setFleetInitScript(fleet, script)
      .then((next) => {
        if (ticket !== sequence.current) return;
        setDocument(next);
        setDraft(next.script);
        setLoading(false);
        setLoadError(null);
        setConfirm(null);
        setNotice(
          kind === "remove"
            ? `Removed. New hosts of ${fleet} start without it; existing cells are replaced as they become idle.`
            : `Saved. New hosts of ${fleet} run this script; existing cells are replaced as they become idle.`,
        );
      })
      .catch((error: unknown) => {
        // The dialog stays open holding the plane's own words.
        if (ticket === sequence.current) setSaveError(message(error, "Could not save the init script"));
      })
      .finally(() => {
        if (ticket === sequence.current) setSaving(false);
      });
  };

  const bytes = byteLength(draft);
  const maxBytes = document?.max_bytes ?? 0;
  const overLimit = document !== null && bytes > maxBytes;
  const editable = document !== null && document.supported && !saving;
  const changed = document !== null && draft !== document.script;
  const canSave = editable && changed && !overLimit;
  const canRemove = document !== null && document.script !== "" && !saving;

  return (
    <section className="panel data-panel fleet-init-script" aria-label={`Init script for ${fleet}`}>
      <div className="panel-header">
        <div>
          <h2>Init script · {fleet}</h2>
          <p>Bash run as root on every new host of this fleet, before its worker starts</p>
        </div>
        <button className="button button-ghost" type="button" onClick={load} disabled={loading || saving}>
          <RefreshCw className={loading ? "spin" : undefined} size={15} aria-hidden="true" />
          Reload
        </button>
        {onClose && (
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close init script">
            <X size={16} aria-hidden="true" />
          </button>
        )}
      </div>
      <div className="fleet-init-body">
        <p className="form-note">
          Runs with <code>bash</code> as root on each <strong>new</strong> host, after the worker environment exists and
          before the worker starts, with stdin from <code>/dev/null</code>. A non-zero exit fails that host&apos;s
          setup. Saving a changed script changes the pool template: existing cells are replaced as they become idle (a
          cell running a job is never drained for this) and new cells run the new script. Re-applying the fleet YAML (
          <code>eyvan up</code>) keeps this script.
        </p>
        {loadError && (
          <div className="form-error" role="alert">
            <AlertCircle size={16} aria-hidden="true" />
            {loadError}
          </div>
        )}
        {document && !document.supported && (
          <div className="fleet-init-warning" role="alert">
            <AlertTriangle size={16} aria-hidden="true" />
            <div>
              <strong>This fleet cannot run an init script.</strong>
              <ul>
                {document.unsupported_reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
        {document && !document.consistent && (
          <div className="fleet-init-warning" role="status">
            <AlertTriangle size={16} aria-hidden="true" />
            <span>
              The pools of this fleet carry different init scripts (their YAML was edited by hand). One of them is
              shown; saving writes this script to every pool.
            </span>
          </div>
        )}
        {document === null && loading && <p className="muted">Reading the init script…</p>}
        {document && (
          <>
            <div className="field">
              <label htmlFor={editorId}>
                <span>
                  <SquareTerminal size={13} aria-hidden="true" /> Script (bash)
                </span>
              </label>
              <textarea
                id={editorId}
                className="fleet-init-editor"
                value={draft}
                onChange={(event) => {
                  setDraft(event.target.value);
                  setNotice(null);
                }}
                disabled={!editable}
                spellCheck={false}
                autoComplete="off"
                autoCapitalize="off"
                wrap="off"
                rows={14}
                placeholder={document.supported ? "pip install -r requirements.txt" : undefined}
                aria-describedby={counterId}
              />
            </div>
            <div className="fleet-init-meta">
              <span id={counterId} className={overLimit ? "fleet-init-over" : undefined}>
                {bytes} / {maxBytes} bytes{overLimit ? " — over the limit" : ""}
              </span>
              {changed && <span>Unsaved changes</span>}
            </div>
            {document.other_init_commands.length > 0 && (
              <section className="fleet-init-yaml" aria-label="Also runs (from fleet YAML)">
                <h3>Also runs (from fleet YAML)</h3>
                <p className="muted">Declared in the fleet YAML; read-only here. Change them there.</p>
                <ul>
                  {document.other_init_commands.map((command) => (
                    <li key={command}>
                      <code>{command}</code>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </>
        )}
        {notice && (
          <div className="inline-notice" role="status">
            {notice}
          </div>
        )}
      </div>
      <div className="fleet-init-actions">
        <button
          className="button button-secondary"
          type="button"
          disabled={!canRemove}
          onClick={() => {
            setSaveError(null);
            setConfirm("remove");
          }}
        >
          Remove script
        </button>
        <button
          className="button button-primary"
          type="button"
          disabled={!canSave}
          onClick={() => {
            setSaveError(null);
            // Saving an emptied editor is a removal, and is confirmed as one.
            setConfirm(draft.trim() === "" ? "remove" : "save");
          }}
        >
          Save script
        </button>
      </div>
      <ConfirmDialog
        open={confirm !== null}
        title={confirm === "remove" ? `Remove the init script of ${fleet}?` : `Save the init script of ${fleet}?`}
        confirmLabel={confirm === "remove" ? "Remove" : "Save"}
        destructive={confirm === "remove"}
        busy={saving}
        error={saveError}
        consequence={
          confirm === "remove" ? (
            <p>
              Every pool of <strong>{fleet}</strong> drops the script and gets a new template. Existing idle cells are
              replaced; a cell running a job is never drained for this. New cells start without it. Init commands from
              the fleet YAML still run.
            </p>
          ) : (
            <p>
              Every pool of <strong>{fleet}</strong> gets this script and a new template. Existing idle cells are
              replaced; a cell running a job is never drained for this. Every new cell runs the new script before its
              worker starts, and a non-zero exit fails that host&apos;s setup.
            </p>
          )
        }
        onConfirm={() => submit(confirm === "remove" ? "" : draft, confirm ?? "save")}
        onCancel={() => {
          setConfirm(null);
          setSaveError(null);
        }}
      />
    </section>
  );
}
