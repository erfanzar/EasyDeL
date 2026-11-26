// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { workloadStatus } from "./api/client";
import { StagePipeline } from "./components/StagePipeline";

afterEach(cleanup);

describe("ledger pipeline terminal progress", () => {
  it.each(["succeeded", "failed", "cancelled"])(
    "renders adapted %s as the final stage despite a stale acquisition phase",
    (raw) => {
      const { container } = render(
        <StagePipeline status={workloadStatus(raw)} phase="provisioning" timeline={[]} variant="ledger" />,
      );

      expect(screen.getByText("Settled").tagName).toBe("STRONG");
      expect(screen.getByText("stage 8 of 8")).toBeTruthy();
      expect(screen.queryByText("Submitted")).toBeNull();
      expect(screen.queryByText("stage 0 of 8")).toBeNull();
      expect(container.querySelectorAll(".stage-tick-done, .stage-tick-failed")).toHaveLength(8);
      expect(container.querySelector(".stage-now-done, .stage-now-failed")).not.toBeNull();
    },
  );

  it("retains raw succeeded compatibility", () => {
    render(<StagePipeline status="succeeded" phase="provisioning" timeline={[]} variant="ledger" />);

    expect(screen.getByText("Settled")).toBeTruthy();
    expect(screen.getByText("stage 8 of 8")).toBeTruthy();
  });

  it("keeps live workloads at their current phase", () => {
    const { container } = render(
      <StagePipeline status={workloadStatus("running")} phase="provisioning" timeline={[]} variant="ledger" />,
    );

    expect(screen.getByText("Provisioning")).toBeTruthy();
    expect(screen.getByText("stage 4 of 8")).toBeTruthy();
    expect(container.querySelector(".stage-now-done")).toBeNull();
    expect(container.querySelectorAll(".stage-tick-active")).toHaveLength(1);
  });
});
