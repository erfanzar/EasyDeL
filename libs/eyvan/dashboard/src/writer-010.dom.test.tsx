// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { FleetPage } from "./pages/FleetPage";
import type { BillingReport, CapacityPoolSummary, LaunchLatencyReport, WarmthReport } from "./types";

const billing = (overrides: Partial<BillingReport> = {}): BillingReport => ({
  observed_at: "2026-10-01T12:00:00Z",
  accrued_cost: 5,
  idle_cost: 2,
  cells: 1,
  billed_seconds: 3600,
  idle_seconds: 1800,
  unpriced_cells: 0,
  ...overrides,
});

const warmth = (held = 1): WarmthReport => ({
  observed_at: "2026-10-01T12:00:00Z",
  pools: [
    {
      pool_id: "owned-pool",
      floor: held,
      window_active: false,
      warm_cells: held,
      warm_idle_cells: held,
      provisioning_cells: 0,
      hourly_cost: held * 2,
    },
  ],
});

const latency = (seconds = 30): LaunchLatencyReport => ({
  observed_at: "2026-10-01T12:00:00Z",
  pools: [
    {
      pool_id: "owned-pool",
      p95_seconds: seconds,
      median_seconds: seconds,
      slowest_seconds: seconds,
      samples: 3,
      cells_in_cooldown: 0,
    },
  ],
});

const pools: CapacityPoolSummary[] = [
  {
    id: "owned-pool",
    fleetName: "owned-fleet",
    acceleratorType: null,
    capacityClass: "spot",
    zone: null,
    region: "local",
    maxCells: 2,
    minCells: 1,
    enabled: true,
  },
];

function props() {
  return {
    cells: [],
    leases: [],
    onRefresh: vi.fn(async () => {}),
    loadWarmth: vi.fn(async (_signal: AbortSignal) => warmth()),
    loadBilling: vi.fn(async (_signal: AbortSignal) => billing()),
    loadLaunchLatency: vi.fn(async (_signal: AbortSignal) => latency()),
    loadPools: vi.fn(async (_signal: AbortSignal) => pools),
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

afterEach(cleanup);

describe("Fleet report refresh revision", () => {
  it("reloads all reports with stable loaders after a same-scope refresh", async () => {
    const loaders = props();
    const page = render(<FleetPage {...loaders} refreshRevision={0} />);
    await screen.findByText("$5 accrued");
    await screen.findByText("p95 30s over 3 provisions");
    await screen.findByText("owned-fleet");

    loaders.loadBilling.mockResolvedValue(billing({ accrued_cost: 99, idle_cost: 44 }));
    loaders.loadWarmth.mockResolvedValue(warmth(2));
    loaders.loadLaunchLatency.mockResolvedValue(latency(300));
    page.rerender(<FleetPage {...loaders} refreshRevision={1} />);

    await screen.findByText("$99 accrued");
    expect(screen.getByText("$44 idle")).toBeTruthy();
    expect(screen.getByText("2/2 held, 2 idle")).toBeTruthy();
    expect(screen.getByText("p95 300s over 3 provisions")).toBeTruthy();
    for (const loader of [loaders.loadPools, loaders.loadBilling, loaders.loadWarmth, loaders.loadLaunchLatency]) {
      expect(loader).toHaveBeenCalledTimes(2);
    }
  });

  it("does not let superseded report replies overwrite the refreshed panels", async () => {
    const oldBilling = deferred<BillingReport>();
    const oldWarmth = deferred<WarmthReport>();
    const oldLatency = deferred<LaunchLatencyReport>();
    const loaders = props();
    loaders.loadBilling.mockReturnValueOnce(oldBilling.promise);
    loaders.loadWarmth.mockReturnValueOnce(oldWarmth.promise);
    loaders.loadLaunchLatency.mockReturnValueOnce(oldLatency.promise);
    const page = render(<FleetPage {...loaders} refreshRevision={0} />);

    loaders.loadBilling.mockResolvedValue(billing({ accrued_cost: 99 }));
    loaders.loadWarmth.mockResolvedValue(warmth(2));
    loaders.loadLaunchLatency.mockResolvedValue(latency(300));
    page.rerender(<FleetPage {...loaders} refreshRevision={1} />);
    await screen.findByText("$99 accrued");
    expect(loaders.loadBilling.mock.calls[0][0].aborted).toBe(true);

    await act(async () => {
      oldBilling.resolve(billing());
      oldWarmth.resolve(warmth());
      oldLatency.resolve(latency());
    });
    expect(screen.queryByText("$5 accrued")).toBeNull();
    expect(screen.getByText("2/2 held, 2 idle")).toBeTruthy();
    expect(screen.getByText("p95 300s over 3 provisions")).toBeTruthy();
  });

  it("reports refresh failures and clears them on a later successful revision", async () => {
    const loaders = props();
    const page = render(<FleetPage {...loaders} refreshRevision={0} />);
    await screen.findByText("$5 accrued");
    loaders.loadBilling.mockRejectedValueOnce(new Error("billing-refresh-failed"));
    loaders.loadLaunchLatency.mockRejectedValueOnce(new Error("latency-refresh-failed"));
    page.rerender(<FleetPage {...loaders} refreshRevision={1} />);
    await screen.findByText("billing-refresh-failed");
    await screen.findByText("latency-refresh-failed");

    page.rerender(<FleetPage {...loaders} refreshRevision={2} />);
    await waitFor(() => expect(loaders.loadBilling).toHaveBeenCalledTimes(3));
    expect(screen.queryByText("billing-refresh-failed")).toBeNull();
    expect(screen.queryByText("latency-refresh-failed")).toBeNull();
  });
});

describe("Fleet pool listing states", () => {
  it("shows an initial pool failure with no fleets and recovers on refresh", async () => {
    const loaders = props();
    loaders.loadPools.mockRejectedValueOnce(new Error("pool-list-503"));
    const page = render(<FleetPage {...loaders} refreshRevision={0} />);
    await screen.findByText("$5 accrued");
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("pool-list-503");
    expect(screen.queryByText("No fleets are applied.")).toBeNull();

    page.rerender(<FleetPage {...loaders} refreshRevision={1} />);
    await screen.findByText("owned-fleet");
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("distinguishes a pending listing from a successful empty listing", async () => {
    const pending = deferred<CapacityPoolSummary[]>();
    const loaders = props();
    loaders.loadPools.mockReturnValueOnce(pending.promise);
    render(<FleetPage {...loaders} />);
    const region = screen.getByRole("region", { name: "Applied fleets" });
    expect(within(region).getByText("Reading capacity pools…")).toBeTruthy();
    expect(within(region).queryByText("No fleets are applied.")).toBeNull();

    await act(async () => pending.resolve([]));
    expect(within(region).getByText("No fleets are applied.")).toBeTruthy();
    expect(within(region).queryByText("Reading capacity pools…")).toBeNull();
  });

  it("offers a local retry after an initial failure", async () => {
    const loaders = props();
    loaders.loadPools.mockRejectedValueOnce(new Error("pool-list-503"));
    render(<FleetPage {...loaders} />);
    await screen.findByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: "Retry pool listing" }));
    await screen.findByText("owned-fleet");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(loaders.loadPools).toHaveBeenCalledTimes(2);
  });
});

describe("Warm capacity ordering", () => {
  it("lists pools holding capacity first, most held first, then A-Z", async () => {
    const loaders = props();
    const row = (pool_id: string, held: number, floor = held) => ({
      pool_id,
      floor,
      window_active: false,
      warm_cells: held,
      warm_idle_cells: 0,
      provisioning_cells: 0,
      hourly_cost: 0,
    });
    loaders.loadWarmth.mockResolvedValue({
      observed_at: "2026-10-01T12:00:00Z",
      pools: [row("b-zero", 0), row("a-zero", 0), row("z-one", 1), row("m-two", 2), row("c-floor", 0, 1)],
    });
    render(<FleetPage {...loaders} />);
    const panel = screen.getByRole("region", { name: "Warm capacity and cost" });
    await within(panel).findByText("m-two");
    const order = Array.from(panel.querySelectorAll("dt")).map((node) => node.textContent);
    expect(order).toEqual(["m-two", "z-one", "c-floor", "a-zero", "b-zero"]);
  });
});

describe("Fleet unknown-price contract", () => {
  it("shows fully unpriced capacity as not costed while retaining measured hours", async () => {
    const loaders = props();
    loaders.loadBilling.mockResolvedValue(billing({ unpriced_cells: 1, accrued_cost: 0, idle_cost: 0 }));
    render(<FleetPage {...loaders} />);
    const panel = screen.getByRole("region", { name: "Warm capacity and cost" });
    await within(panel).findByText("Accrued: Not costed");
    expect(within(panel).getByText("Idle: Not costed")).toBeTruthy();
    expect(within(panel).queryByText("$0 accrued")).toBeNull();
    expect(within(panel).queryByText("$0 idle")).toBeNull();
    expect(within(panel).getByText(/1 of 1 cell\(s\) have no declared unit price/)).toBeTruthy();
    expect(within(panel).getByText(/1.0h billed, 0.5h idle/)).toBeTruthy();
  });

  it("qualifies partial totals instead of presenting them as the whole fleet cost", async () => {
    const loaders = props();
    loaders.loadBilling.mockResolvedValue(billing({ cells: 2, unpriced_cells: 1 }));
    render(<FleetPage {...loaders} />);
    const panel = screen.getByRole("region", { name: "Warm capacity and cost" });
    await within(panel).findByText("$5 accrued (priced cells only)");
    expect(within(panel).getByText("$2 idle (priced cells only)")).toBeTruthy();
    expect(within(panel).getByText(/1 of 2 cell\(s\) have no declared unit price/)).toBeTruthy();
  });

  it("preserves declared zero prices as known zero cost", async () => {
    const loaders = props();
    loaders.loadBilling.mockResolvedValue(billing({ accrued_cost: 0, idle_cost: 0, unpriced_cells: 0 }));
    render(<FleetPage {...loaders} />);
    const panel = screen.getByRole("region", { name: "Warm capacity and cost" });
    await within(panel).findByText("$0 accrued");
    expect(within(panel).getByText("$0 idle")).toBeTruthy();
    expect(within(panel).queryByText(/Not costed|priced cells only|no declared unit price/)).toBeNull();
  });
});
