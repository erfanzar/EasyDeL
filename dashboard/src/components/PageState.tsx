// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertTriangle, LoaderCircle, RefreshCw } from "lucide-react";

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
  onRetry: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="page-state page-state-error" role="alert">
      <AlertTriangle size={22} aria-hidden="true" />
      <strong>Unable to load Eyvan state</strong>
      <span>{message}</span>
      <button className="button button-secondary" type="button" onClick={onRetry}>
        <RefreshCw size={16} aria-hidden="true" />
        Retry
      </button>
    </div>
  );
}
