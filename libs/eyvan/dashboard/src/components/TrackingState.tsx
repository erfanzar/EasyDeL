// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { ApiError } from "../api/client";
import { ErrorCard } from "./ErrorCard";
import { ErrorState } from "./PageState";

export function TrackingState({ error, onRetry }: { error: Error; onRetry: () => void }) {
  if (error instanceof ApiError && error.status === 503) {
    return (
      <section className="page-state-card tracker-empty" aria-label="Tracker setup">
        <ErrorCard
          error={error}
          tone="info"
          role="status"
          title="No tracker configured"
          summary="Start a local tracker on this machine, or configure EYVAN_TRACKER_URL on the control plane, then retry."
          context={{
            code: "tracker_unconfigured",
            httpStatus: 503,
            remediation: "Run the local tracker or launch the dashboard with automatic local bootstrapping:",
          }}
          extraCommands={[
            { label: "Local tracker", command: "eyvan track up --local" },
            { label: "Auto-bootstrap", command: "eyvan dashboard" },
          ]}
          actions={[{ label: "Retry", onClick: onRetry, variant: "secondary" }]}
        />
      </section>
    );
  }
  return <ErrorState error={error} message={error.message} onRetry={onRetry} />;
}
