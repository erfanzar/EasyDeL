// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { OverviewData } from "../types";
import { OverviewPage } from "./OverviewPage";

// A fleet spent $3.68 over seventy-eight hours while this card read "$0".
// It was not wrong about its own definition -- the headline sums budget.spent,
// and no budget existed -- but an operator reads it as what the fleet cost, and
// it under-reports silently and only ever in that direction.
const snapshot = (overrides: Partial<OverviewData> = {}): OverviewData =>
  ({
    queued: 0,
    running: 0,
    fleetUtilization: 0,
    monthlySpend: 0,
    monthlyBudget: 0,
    successRate: undefined,
    queueTrend: [],
    priorityQueue: [],
    fleetBreakdown: [],
    alerts: [],
    ...overrides,
  }) as OverviewData;

afterEach(cleanup);

describe("the overview spend card", () => {
  it("does not report zero spend when no budget defines what spend means", () => {
    render(
      <OverviewPage
        overview={snapshot()}
        workloads={[]}
        onOpenWorkload={() => {}}
        onShowWorkloads={() => {}}
        fleetAvailable
      />,
    );

    expect(screen.getByText("No budget")).toBeDefined();
    expect(screen.queryByText("$0")).toBeNull();
  });

  it("says where the real number is", () => {
    render(
      <OverviewPage
        overview={snapshot()}
        workloads={[]}
        onOpenWorkload={() => {}}
        onShowWorkloads={() => {}}
        fleetAvailable
      />,
    );

    expect(screen.getByText(/cell cost still accrues/)).toBeDefined();
  });

  it("still reports spend against a budget that exists", () => {
    render(
      <OverviewPage
        overview={snapshot({ monthlySpend: 40, monthlyBudget: 100 })}
        workloads={[]}
        onOpenWorkload={() => {}}
        onShowWorkloads={() => {}}
        fleetAvailable
      />,
    );

    expect(screen.getByText(/40% of/)).toBeDefined();
  });

  it("is labelled as budgeted spend, not total spend", () => {
    render(
      <OverviewPage
        overview={snapshot({ monthlySpend: 40, monthlyBudget: 100 })}
        workloads={[]}
        onOpenWorkload={() => {}}
        onShowWorkloads={() => {}}
        fleetAvailable
      />,
    );

    expect(screen.getByText("Budgeted spend")).toBeDefined();
  });
});
