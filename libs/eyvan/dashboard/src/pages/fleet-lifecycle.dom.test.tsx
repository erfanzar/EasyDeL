// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { FleetPage } from "./FleetPage";
import { mockCells } from "../api/mockData";
import type { CapacityPoolSummary } from "../types";

const POOLS: CapacityPoolSummary[] = [
  {
    id: "research-v6e-8",
    fleetName: "research",
    acceleratorType: "v6e-8",
    capacityClass: "spot",
    zone: "asia-east1-c",
    region: "asia-east1",
    maxCells: 10000,
    minCells: 0,
    enabled: true,
  },
  {
    id: "research-v6e-64",
    fleetName: "research",
    acceleratorType: "v6e-64",
    capacityClass: "spot",
    zone: "asia-east1-c",
    region: "asia-east1",
    maxCells: 10000,
    minCells: 0,
    enabled: true,
  },
  {
    id: "laptop-cpu",
    fleetName: "laptop",
    acceleratorType: null,
    capacityClass: "on_demand",
    zone: null,
    region: "local",
    maxCells: 2,
    minCells: 1,
    enabled: true,
  },
];

function renderPage(overrides: Partial<Parameters<typeof FleetPage>[0]> = {}) {
  return render(
    <FleetPage
      cells={[]}
      leases={[]}
      onRefresh={async () => {}}
      loadWarmth={async () => ({ observed_at: "", pools: [] })}
      loadBilling={async () => ({ observed_at: "", total_accrued: 0, total_idle: 0, cells: [] }) as never}
      loadLaunchLatency={async () => ({ observed_at: "", pools: [] })}
      loadPools={async () => POOLS}
      {...overrides}
    />,
  );
}

afterEach(cleanup);

function appliedFleets(): HTMLElement {
  return screen.getByRole("region", { name: "Applied fleets" });
}

describe("applied fleets", () => {
  it("lists what is registered even when nothing is running", async () => {
    renderPage();

    await waitFor(() => expect(appliedFleets()).toBeTruthy());
    const rows = within(appliedFleets()).getAllByRole("row").slice(1);

    expect(rows).toHaveLength(2);
    expect(rows.map((row) => row.textContent)).toEqual([
      expect.stringContaining("laptop"),
      expect.stringContaining("research"),
    ]);
  });

  it("groups a fleet's pools rather than listing each one", async () => {
    renderPage();

    await waitFor(() => expect(appliedFleets()).toBeTruthy());
    const research = within(appliedFleets())
      .getAllByRole("row")
      .find((row) => row.textContent?.includes("research"))!;

    // One entry per generation with its size range, rather than the first three
    // shapes and a count -- that hid whole generations, because shapes sort by
    // generation and the tail of the list fell inside the count.
    expect(research.textContent).toContain("v6e 8-64");
    expect(research.textContent).toContain("asia-east1-c");
  });

  it("offers no stop control to a principal who may not manage the fleet", async () => {
    // The route would refuse them anyway; offering the button teaches a gesture
    // that ends in a 403.
    renderPage({ onStopFleet: undefined });

    await waitFor(() => expect(appliedFleets()).toBeTruthy());

    expect(within(appliedFleets()).queryByRole("button", { name: "Stop" })).toBeNull();
  });

  it("confirms before stopping, and names what stopping does", async () => {
    const onStopFleet = vi.fn(async () => {});
    renderPage({ onStopFleet });

    await waitFor(() => expect(appliedFleets()).toBeTruthy());
    fireEvent.click(within(appliedFleets()).getAllByRole("button", { name: "Stop" })[0]);

    // Not fired on the click that opened the dialog.
    expect(onStopFleet).not.toHaveBeenCalled();
    expect(screen.getByText(/ceiling set to zero/)).toBeTruthy();
    // The definition surviving is the difference between this and a delete, and
    // it is what makes the button safe to press.
    expect(screen.getByText(/definition survives/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Stop fleet" }));
    await waitFor(() => expect(onStopFleet).toHaveBeenCalledWith("laptop"));
  });

  it("reports a refused stop instead of appearing to succeed", async () => {
    const onStopFleet = vi.fn(async () => {
      throw new Error("This session may not manage fleet capacity");
    });
    renderPage({ onStopFleet });

    await waitFor(() => expect(appliedFleets()).toBeTruthy());
    fireEvent.click(within(appliedFleets()).getAllByRole("button", { name: "Stop" })[0]);
    fireEvent.click(screen.getByRole("button", { name: "Stop fleet" }));

    await waitFor(() => expect(screen.getByText(/may not manage fleet capacity/)).toBeTruthy());
  });

  it("keeps a deferred stop rejection visible after a successful pool reload", async () => {
    let rejectStop!: (reason: Error) => void;
    const stopping = new Promise<void>((_resolve, reject) => {
      rejectStop = reject;
    });
    const loadPools = vi.fn(async () => POOLS);
    renderPage({ onStopFleet: () => stopping, loadPools });
    await waitFor(() => expect(loadPools).toHaveBeenCalledTimes(1));
    fireEvent.click(within(appliedFleets()).getAllByRole("button", { name: "Stop" })[0]);
    fireEvent.click(screen.getByRole("button", { name: "Stop fleet" }));
    await screen.findByRole("button", { name: "Stopping" });
    await waitFor(() => expect(loadPools).toHaveBeenCalledTimes(2));

    await act(async () => rejectStop(new Error("DELETE was rejected; fleet still enabled")));
    await waitFor(() => expect(loadPools).toHaveBeenCalledTimes(3));
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByRole("alert").textContent).toContain("DELETE was rejected; fleet still enabled");
    expect(within(appliedFleets()).queryByText(/pool\(s\) disabled/)).toBeNull();
    expect(screen.queryByText(/inventory reloaded/)).toBeNull();
  });

  it.each(["provider_error", undefined])(
    "shows a provider diagnostic even without a reason code (%s)",
    async (reason) => {
      renderPage({
        cells: [
          {
            ...mockCells[0],
            status: "degraded",
            unavailableReason: reason,
            unavailableDetail: "Quota refused <script>alert(1)</script>",
          },
        ],
      });
      await waitFor(() => expect(screen.getByText(/Quota refused/)).toBeTruthy());
      expect(document.querySelector(".cell-card-error script")).toBeNull();
    },
  );

  it("displays provider_project and provider_account cleanly across multi-project GCP pools", async () => {
    const multiProjectPools: CapacityPoolSummary[] = [
      {
        id: "spot-v6e-8",
        fleetName: "spot-fleet",
        acceleratorType: "v6e-8",
        capacityClass: "spot",
        zone: "us-central2-b",
        region: "us-central2",
        maxCells: 8,
        minCells: 0,
        enabled: true,
        provider: "gcp",
        providerProject: "example-project-a",
        providerAccount: "operator@example.com",
      },
      {
        id: "acme-v5p-128",
        fleetName: "acme-reserved",
        acceleratorType: "v5p-128",
        capacityClass: "reserved",
        zone: "us-east5-a",
        region: "us-east5",
        maxCells: 4,
        minCells: 1,
        enabled: true,
        provider: "gcp",
        providerProject: "acme-tpus",
        providerAccount: "alice@example.com",
      },
    ];

    renderPage({ loadPools: async () => multiProjectPools });

    const section = await waitFor(() => appliedFleets());
    const auditBar = await within(section).findByLabelText("Configured GCP projects and accounts");
    expect(auditBar.textContent).toContain("example-project-a");
    expect(auditBar.textContent).toContain("operator@example.com");
    expect(auditBar.textContent).toContain("acme-tpus");
    expect(auditBar.textContent).toContain("alice@example.com");

    const rows = within(section).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2);
    const acmeRow = rows.find((row) => row.textContent?.includes("acme-reserved"))!;
    const spotRow = rows.find((row) => row.textContent?.includes("spot-fleet"))!;
    expect(acmeRow.textContent).toContain("acme-tpus · alice@example.com");
    expect(spotRow.textContent).toContain("example-project-a · operator@example.com");
  });
});
