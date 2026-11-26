// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { workloadStatus } from "../api/client";
import { pipelinePosition, StagePipeline } from "./StagePipeline";

describe("the pipeline consumes the API adapter's statuses", () => {
  it.each(["succeeded", "failed", "cancelled"])("renders %s as settled despite an old acquisition phase", (raw) => {
    const status = workloadStatus(raw);
    expect(pipelinePosition(status, "provisioning").activeLabel).toBe("Settled");
    const html = renderToStaticMarkup(
      <StagePipeline status={status} phase="provisioning" timeline={[]} variant="rail" />,
    );
    expect(html).not.toContain('aria-current="step"');
    expect(html).not.toContain('class="spin"');
  });

  it.each([
    ["pending", "Admitted"],
    ["admitted", "Admitted"],
    ["leased", "Provisioning"],
    ["dispatching", "Provisioning"],
    ["running", "Running"],
    ["checkpointing", "Running"],
    ["cancelling", "Running"],
  ])("keeps %s past the submission stage", (raw, label) => {
    expect(pipelinePosition(workloadStatus(raw)).activeLabel).toBe(label);
  });

  it("uses a more precise live phase when one is available", () => {
    expect(pipelinePosition(workloadStatus("dispatching"), "waiting_for_dispatch").activeLabel).toBe("Dispatched");
  });
});
