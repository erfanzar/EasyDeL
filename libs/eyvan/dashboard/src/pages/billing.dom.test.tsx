// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { BillingPage } from "./BillingPage";
import type { BillingReport, ProjectUsage } from "../types";

afterEach(cleanup);

const PRICED: BillingReport = {
  observed_at: "2026-08-19T12:00:00Z",
  cells: 4,
  billed_seconds: 7200,
  idle_seconds: 1800,
  accrued_cost: 42.5,
  idle_cost: 3.25,
  unpriced_cells: 0,
};

const USAGE: ProjectUsage = {
  orgId: "default",
  projectId: "default",
  subjects: [
    {
      subject: "erfan",
      workloads: 5,
      settledWorkloads: 4,
      leases: 4,
      acceleratorHours: 12.5,
      runningSeconds: 3600,
      settledCost: 0,
      unpricedLeases: 4,
    },
  ],
};

function renderPage(billing: BillingReport, usage: ProjectUsage = USAGE) {
  return render(<BillingPage loadBilling={async () => billing} loadUsage={async () => usage} />);
}

describe("the billing page", () => {
  it("does not display a previous project's late response after switching projects", async () => {
    let resolveOld!: (billing: BillingReport) => void;
    const oldBilling = new Promise<BillingReport>((resolve) => {
      resolveOld = resolve;
    });
    const page = render(<BillingPage loadBilling={() => oldBilling} loadUsage={async () => USAGE} />);
    page.rerender(
      <BillingPage
        loadBilling={async () => ({ ...PRICED, accrued_cost: 8 })}
        loadUsage={async () => ({ ...USAGE, subjects: [] })}
      />,
    );
    await screen.findByText("$8.00");
    await act(async () => resolveOld(PRICED));

    expect(screen.getByText("$8.00")).toBeTruthy();
    expect(screen.queryByText("$42.50")).toBeNull();
    expect(screen.getByText("Nobody has submitted work yet")).toBeTruthy();
  });

  it("hides previous project metrics while the newly selected project is loading", async () => {
    let resolveNext!: (billing: BillingReport) => void;
    const nextBilling = new Promise<BillingReport>((resolve) => {
      resolveNext = resolve;
    });
    const page = renderPage(PRICED);
    await screen.findByText("$42.50");

    page.rerender(<BillingPage loadBilling={() => nextBilling} loadUsage={async () => ({ ...USAGE, subjects: [] })} />);
    expect(screen.queryByText("$42.50")).toBeNull();
    expect(screen.getByRole("status")).toBeTruthy();

    await act(async () => resolveNext({ ...PRICED, accrued_cost: 8 }));
    expect(screen.getByText("$8.00")).toBeTruthy();
  });

  it("reports measured hours even when nothing is costed", async () => {
    // The hours are real whether or not a price exists, and hiding them would
    // make an uncosted fleet look like an idle one.
    renderPage({ ...PRICED, unpriced_cells: 4, accrued_cost: 0, idle_cost: 0 });

    await waitFor(() => expect(screen.getByText("2.0h")).toBeTruthy());
    const summary = screen.getByRole("region", { name: "What the fleet has run" });
    expect(within(summary).getByText("Not costed")).toBeTruthy();
  });

  it("says why the cost is zero rather than printing a confident nothing", async () => {
    renderPage({ ...PRICED, unpriced_cells: 4, accrued_cost: 0, idle_cost: 0 });

    await waitFor(() => expect(screen.getByText("Nothing here is costed")).toBeTruthy());
    expect(screen.getByText(/4 of 4 cell/)).toBeTruthy();
    // The fix, named: a grant is priced at zero, not left unpriced.
    expect(screen.getByText(/unit_prices/)).toBeTruthy();
  });

  it("shows a real total once prices are declared", async () => {
    renderPage(PRICED);

    const summary = await screen.findByRole("region", { name: "What the fleet has run" });
    expect(within(summary).getByText("$42.50")).toBeTruthy();
    expect(screen.queryByText("Nothing here is costed")).toBeNull();
  });

  it("attributes usage to whoever submitted it", async () => {
    renderPage(PRICED);

    await waitFor(() => expect(screen.getByRole("region", { name: "Usage by person" })).toBeTruthy());
    const row = within(screen.getByRole("region", { name: "Usage by person" })).getAllByRole("row")[1];

    expect(row.textContent).toContain("erfan");
    expect(row.textContent).toContain("12.50");
  });

  it("marks a person's own uncosted leases rather than showing them as free", async () => {
    renderPage(PRICED);

    await waitFor(() => expect(screen.getByText(/4 unpriced lease/)).toBeTruthy());
  });

  it("says so plainly when nobody has run anything", async () => {
    renderPage(PRICED, { orgId: "default", projectId: "default", subjects: [] });

    await waitFor(() => expect(screen.getByText("Nobody has submitted work yet")).toBeTruthy());
  });
});
