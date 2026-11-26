// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ApiError } from "../api/client";
import { ErrorState } from "./PageState";

export function TrackingState({ error, onRetry }: { error: Error; onRetry: () => void }) {
  if (error instanceof ApiError && error.status === 503) {
    return (
      <section className="page-state tracker-empty" role="status">
        <strong>
          No tracker configured — run <code>eyvan track up --local</code>
        </strong>
        <p>Start a tracker on this machine, or configure EYVAN_TRACKER_URL on the control plane, then retry.</p>
        <button type="button" className="button button-secondary" onClick={onRetry}>
          Retry
        </button>
      </section>
    );
  }
  return <ErrorState message={error.message} onRetry={onRetry} />;
}
