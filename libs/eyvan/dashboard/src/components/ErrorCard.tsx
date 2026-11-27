// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertTriangle, Check, ChevronRight, Copy, Info, ShieldAlert, Terminal, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { diagnoseError, type DiagnoseContext, type ErrorTone, type ParsedError } from "../lib/errorDiagnostics";

export interface ErrorCardAction {
  label: string;
  onClick: () => void;
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
}

export interface ErrorCardCommand {
  label?: string;
  command: string;
}

export interface ErrorCardProps {
  error: ParsedError | Error | string;
  title?: string;
  summary?: string;
  tone?: ErrorTone;
  role?: "alert" | "status";
  compact?: boolean;
  context?: DiagnoseContext;
  actions?: ErrorCardAction[];
  extraCommands?: ErrorCardCommand[];
  onCopied?: (label: string) => void;
  onDismiss?: () => void;
  className?: string;
}

function isParsedError(value: ParsedError | Error | string): value is ParsedError {
  return (
    typeof value === "object" &&
    value !== null &&
    "code" in value &&
    "badges" in value &&
    "summary" in value &&
    "raw" in value
  );
}

async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Fall back to a temporary textarea when Clipboard API is unavailable.
  }
  try {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(area);
    return ok;
  } catch {
    return false;
  }
}

export function ErrorCard({
  error,
  title,
  summary,
  tone,
  role,
  compact = false,
  context,
  actions,
  extraCommands,
  onCopied,
  onDismiss,
  className,
}: ErrorCardProps) {
  const diagnostic = useMemo(() => (isParsedError(error) ? error : diagnoseError(error, context)), [context, error]);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof globalThis.setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timerRef.current !== null) globalThis.clearTimeout(timerRef.current);
    },
    [],
  );

  const effectiveTone = tone ?? diagnostic.tone;
  const effectiveTitle = title ?? diagnostic.title;
  const effectiveSummary = summary ?? diagnostic.summary;
  const effectiveRole = role ?? (effectiveTone === "info" ? "status" : "alert");

  const commands = useMemo<ErrorCardCommand[]>(() => {
    const list: ErrorCardCommand[] = [];
    const indexByCommand = new Map<string, number>();
    const add = (item: ErrorCardCommand) => {
      const normalized = item.command.trim();
      if (!normalized) return;
      const existingIndex = indexByCommand.get(normalized);
      if (existingIndex !== undefined) {
        if (!list[existingIndex].label && item.label) {
          list[existingIndex] = { label: item.label, command: normalized };
        }
        return;
      }
      indexByCommand.set(normalized, list.length);
      list.push({ label: item.label, command: normalized });
    };
    if (diagnostic.command) add({ command: diagnostic.command });
    for (const extra of extraCommands ?? []) add(extra);
    return list;
  }, [diagnostic.command, extraCommands]);

  const handleCopy = async (key: string, value: string, label: string) => {
    const ok = await copyToClipboard(value);
    if (!ok) return;
    setCopiedKey(key);
    onCopied?.(label);
    if (timerRef.current !== null) globalThis.clearTimeout(timerRef.current);
    timerRef.current = globalThis.setTimeout(() => setCopiedKey(null), 1800);
  };

  const Icon = effectiveTone === "info" ? Info : diagnostic.code === "permission_denied" ? ShieldAlert : AlertTriangle;

  return (
    <div
      className={`error-card error-card-${effectiveTone}${compact ? " error-card-compact" : ""}${className ? ` ${className}` : ""}`}
      role={effectiveRole}
    >
      <div className="error-card-head">
        <span className="error-card-icon" aria-hidden="true">
          <Icon size={15} />
        </span>
        <div className="error-card-heading">
          <strong className="error-card-title">{effectiveTitle}</strong>
          <p className="error-card-summary">{effectiveSummary}</p>
        </div>
        {onDismiss && (
          <button
            className="icon-button icon-button-small error-card-dismiss"
            type="button"
            aria-label="Dismiss message"
            title="Dismiss"
            onClick={onDismiss}
          >
            <X size={13} aria-hidden="true" />
          </button>
        )}
      </div>

      {diagnostic.badges.length > 0 && (
        <div className="error-card-badges" aria-label="Error metadata">
          {diagnostic.badges.map((badge) => (
            <span className="error-card-badge" key={`${badge.label}:${badge.value}`}>
              <span className="error-card-badge-key">{badge.label}</span>
              <span className="error-card-badge-val monospace">{badge.value}</span>
            </span>
          ))}
        </div>
      )}

      {(diagnostic.remediation || commands.length > 0 || (actions && actions.length > 0)) && (
        <div className="error-card-remediation">
          {diagnostic.remediation && <p className="error-card-hint">{diagnostic.remediation}</p>}
          {commands.length > 0 && (
            <div className="error-card-commands">
              {commands.map((item, index) => {
                const key = `cmd-${index}`;
                const copied = copiedKey === key;
                return (
                  <div className="error-card-command" key={item.command}>
                    <Terminal size={12} className="error-card-command-icon" aria-hidden="true" />
                    {item.label && <span className="error-card-command-label">{item.label}</span>}
                    <code className="monospace">{item.command}</code>
                    <button
                      type="button"
                      className="button button-secondary error-card-copy"
                      onClick={() => void handleCopy(key, item.command, "Command copied to clipboard")}
                    >
                      {copied ? <Check size={12} aria-hidden="true" /> : <Copy size={12} aria-hidden="true" />}
                      {copied ? "Copied" : "Copy command"}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
          {actions && actions.length > 0 && (
            <div className="error-card-actions">
              {actions.map((action) => (
                <button
                  key={action.label}
                  type="button"
                  className={`button button-${action.variant ?? "secondary"}`}
                  disabled={action.disabled}
                  onClick={action.onClick}
                >
                  {action.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {diagnostic.hasTechnicalDetails && (
        <details className="error-card-details">
          <summary>
            <ChevronRight size={13} className="error-card-caret" aria-hidden="true" />
            <span>Technical details</span>
          </summary>
          <div className="error-card-details-body">
            <div className="error-card-details-bar">
              <span>Raw diagnostic output</span>
              <button
                type="button"
                className="button button-ghost error-card-copy"
                onClick={() => void handleCopy("raw", diagnostic.raw, "Raw diagnostics copied to clipboard")}
              >
                {copiedKey === "raw" ? <Check size={12} aria-hidden="true" /> : <Copy size={12} aria-hidden="true" />}
                {copiedKey === "raw" ? "Copied" : "Copy raw diagnostics"}
              </button>
            </div>
            <pre className="error-card-raw monospace">{diagnostic.raw}</pre>
          </div>
        </details>
      )}
    </div>
  );
}
