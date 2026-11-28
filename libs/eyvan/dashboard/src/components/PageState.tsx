// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { LoaderCircle } from "lucide-react";

import { ErrorCard } from "./ErrorCard";

interface LoadingStateProps {
  /** What is being waited on. Defaults to the control plane's own state. */
  label?: string;
  /**
   * Why it is slow, for a wait long enough that a bare spinner reads as a hang.
   * Omitted for the ordinary sub-second load, where explaining the wait would
   * be noise.
   */
  detail?: string;
}

export function LoadingState({ label = "Loading control-plane state", detail }: LoadingStateProps = {}) {
  return (
    <div className="page-state" role="status">
      <LoaderCircle className="spin" size={22} aria-hidden="true" />
      <span>{label}</span>
      {detail && <small>{detail}</small>}
    </div>
  );
}

interface ErrorStateProps {
  message: string;
  error?: Error;
  title?: string;
  onRetry: () => void;
  onCopied?: (label: string) => void;
}

export function ErrorState({ message, error, title, onRetry, onCopied }: ErrorStateProps) {
  return (
    <div className="page-state-card">
      <ErrorCard
        error={error ?? message}
        title={title}
        context={{ fallbackTitle: "Unable to load Eyvan state" }}
        actions={[{ label: "Retry", onClick: onRetry, variant: "secondary" }]}
        onCopied={onCopied}
      />
    </div>
  );
}
