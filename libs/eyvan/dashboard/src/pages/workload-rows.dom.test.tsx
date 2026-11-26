// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { WorkloadsPage } from "./WorkloadsPage";
import type { WorkloadSummary } from "../types";

const WORKLOAD = {
  id: "wl_1",
  name: "trainer",
  status: "running",
  priority: "production",
  project: "default",
  user: "erfan",
  submittedAt: "2026-08-18T00:00:00Z",
  resources: { cpu: 2, memoryGib: 4, replicas: 1, acceleratorCount: 0 },
} as unknown as WorkloadSummary;

function renderPage(onOpenWorkload: (id: string) => void) {
  return render(
    <WorkloadsPage workloads={[WORKLOAD]} onOpenWorkload={onOpenWorkload} onSubmit={() => {}} canSubmit={false} />,
  );
}

describe("the workloads table", () => {
  it("opens the workload when the row is clicked, not only its name", () => {
    const opened = vi.fn();
    const { container } = renderPage(opened);

    const statusCell = container.querySelector("tbody tr td:nth-child(2)");
    fireEvent.click(statusCell!);

    expect(opened).toHaveBeenCalledWith("wl_1");
  });

  it("opens it exactly once when the name itself is clicked", () => {
    // The name stays a real button for keyboard and screen-reader users, so its
    // click also reaches the row -- one gesture must still be one navigation.
    const opened = vi.fn();
    const { container } = renderPage(opened);

    fireEvent.click(container.querySelector(".table-primary-link")!);

    expect(opened).toHaveBeenCalledTimes(1);
  });

  it("marks the row as clickable so the cursor does not lie", () => {
    const { container } = renderPage(vi.fn());

    expect(container.querySelector("tbody tr")?.className).toContain("row-clickable");
  });
});
