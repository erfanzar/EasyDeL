// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "../api/client";
import { getMockWorkload, mockWorkloads } from "../api/mockData";
import type { CapacityLease, Cell, WorkloadDetail, WorkloadSpend } from "../types";
import { FleetPage } from "./FleetPage";
import { WorkloadDetailPage } from "./WorkloadDetailPage";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function workload(): WorkloadDetail {
  // The shipped mock rather than a hand-built literal. A partial fixture makes
  // the page throw on a field this test does not care about, and the resulting
  // failure points at the fixture instead of at the panel.
  const detail = getMockWorkload(mockWorkloads[0].id);
  if (!detail) throw new Error("the mock dataset has no workloads");
  return detail;
}

const loadExplanation = (id: string, signal: AbortSignal) =>
  apiClient.explainWorkload(id, apiClient.getScope(), signal);

const fleetLoaders = {
  loadWarmth: (signal: AbortSignal) => apiClient.getWarmth(apiClient.getScope(), signal),
  loadBilling: (signal: AbortSignal) => apiClient.getBilling(apiClient.getScope(), signal),
  loadLaunchLatency: (signal: AbortSignal) => apiClient.getLaunchLatency(apiClient.getScope(), signal),
};

describe("the placement verdict panel", () => {
  it("shows the per-cell detail behind the flat reason", async () => {
    vi.spyOn(apiClient, "explainWorkload").mockResolvedValue({
      workload_id: "wl-1",
      state: "pending",
      placeable: false,
      placement_cell_id: null,
      code: "no_matching_cell",
      message: "no cell satisfies all hard workload constraints",
      cells: [{ cell_id: "cell-a", reason: "insufficient tpu" }],
      considered_cell_count: 7,
      provisioning_cell_ids: ["cell-b"],
      scheduling_reason: "waiting for capacity",
    });

    render(
      <WorkloadDetailPage
        workload={workload()}
        onBack={() => {}}
        onAction={async () => {}}
        canOperate
        loadExplanation={loadExplanation}
        loadCost={() => new Promise(() => {})}
      />,
    );

    await waitFor(() => expect(screen.getByText("Placement verdict")).toBeDefined());
    expect(screen.getByText("no cell satisfies all hard workload constraints")).toBeDefined();
    expect(screen.getByText("cell-a")).toBeDefined();
    // The number that separates "waiting for capacity" from "can never be
    // placed" -- both report the same code, and only one is a problem.
    expect(screen.getByText("Being provisioned")).toBeDefined();
  });

  it("reports a failure rather than rendering an empty panel", async () => {
    // The claim the panel was added for, asserted where it is true or false.
    vi.spyOn(apiClient, "explainWorkload").mockRejectedValue(new Error("Eyvan API request timed out"));

    render(
      <WorkloadDetailPage
        workload={workload()}
        onBack={() => {}}
        onAction={async () => {}}
        canOperate
        loadExplanation={loadExplanation}
        loadCost={() => new Promise(() => {})}
      />,
    );

    await waitFor(() => expect(screen.getByText("Eyvan API request timed out")).toBeDefined());
  });
});

describe("why a workload is on the attempt it is on", () => {
  it("separates capacity the cloud took back from capacity the pool could not give", async () => {
    // Both send a job round again, and an operator reads them opposite ways: a
    // spot job reclaimed twice is behaving exactly as bought, while one that
    // lost capacity twice is running out of pool. The plane has kept these in
    // separate budgets since Phase 8 and the page showed neither, so the only
    // way to answer "why is this on attempt 3" was to count timeline entries
    // by eye -- which cannot distinguish them at all.
    const reclaimed = { ...workload(), attempt: 3, preemptionCount: 2, capacityLossCount: 0 };

    render(
      <WorkloadDetailPage
        workload={reclaimed}
        onBack={() => {}}
        onAction={async () => {}}
        canOperate
        loadExplanation={loadExplanation}
        loadCost={() => new Promise(() => {})}
      />,
    );

    const runtime = screen.getByText("Reclaimed").closest("dl");
    expect(runtime).not.toBeNull();
    expect(within(runtime as HTMLElement).getByText("Attempt").nextElementSibling?.textContent).toBe("3");
    expect(within(runtime as HTMLElement).getByText("Reclaimed").nextElementSibling?.textContent).toBe("2");
    expect(within(runtime as HTMLElement).getByText("Capacity lost").nextElementSibling?.textContent).toBe("0");
  });

  it("says unknown for a plane that does not report the counts", async () => {
    // Rendering 0 here would tell an operator a repeatedly-reclaimed job had
    // never been touched. No answer is the honest one.
    const older = { ...workload(), attempt: undefined, preemptionCount: undefined, capacityLossCount: undefined };

    render(
      <WorkloadDetailPage
        workload={older}
        onBack={() => {}}
        onAction={async () => {}}
        canOperate
        loadExplanation={loadExplanation}
        loadCost={() => new Promise(() => {})}
      />,
    );

    const runtime = screen.getByText("Reclaimed").closest("dl") as HTMLElement;
    expect(within(runtime).getByText("Reclaimed").nextElementSibling?.textContent).toBe("Unknown");
    expect(within(runtime).getByText("Capacity lost").nextElementSibling?.textContent).toBe("Unknown");
  });
});

describe("the warm capacity and cost panel", () => {
  const cells: Cell[] = [];
  const leases: CapacityLease[] = [];

  it("reports warm capacity beside idle cost", async () => {
    vi.spyOn(apiClient, "getWarmth").mockResolvedValue({
      observed_at: "2026-07-29T12:00:00Z",
      pools: [
        { pool_id: "research/v5p", floor: 2, window_active: true, warm_cells: 2, warm_idle_cells: 1, hourly_cost: 8 },
      ],
    });
    vi.spyOn(apiClient, "getBilling").mockResolvedValue({
      observed_at: "2026-07-29T12:00:00Z",
      cells: 1,
      billed_seconds: 3600,
      idle_seconds: 0,
      unpriced_cells: 0,
      accrued_cost: 120.5,
      idle_cost: 30.25,
    });

    render(<FleetPage cells={cells} leases={leases} onRefresh={async () => {}} {...fleetLoaders} />);

    expect(screen.getByText("Warm capacity and cost")).toBeDefined();
    // The heading is present during loading; wait for the async pool row.
    expect(await screen.findByText("research/v5p")).toBeDefined();
    // Warm capacity is deliberately idle. Without the pairing the only safe
    // reaction to an unexplained idle figure is to delete the warm floor.
    expect(screen.getByText(/window active/)).toBeDefined();
  });

  it("names a cost failure rather than showing nothing", async () => {
    vi.spyOn(apiClient, "getWarmth").mockRejectedValue(new Error("Unable to reach Eyvan API"));
    vi.spyOn(apiClient, "getBilling").mockRejectedValue(new Error("Unable to reach Eyvan API"));

    render(<FleetPage cells={cells} leases={leases} onRefresh={async () => {}} {...fleetLoaders} />);

    await waitFor(() => expect(screen.getByText("Unable to reach Eyvan API")).toBeDefined());
  });

  it("says so when no pool declares a warm floor", async () => {
    // Distinct from a failure, and it has to look distinct: "nothing is warm"
    // is a fact about the fleet, while an empty panel is a fact about the page.
    vi.spyOn(apiClient, "getWarmth").mockResolvedValue({ observed_at: "2026-07-29T12:00:00Z", pools: [] });
    vi.spyOn(apiClient, "getBilling").mockResolvedValue({
      observed_at: "2026-07-29T12:00:00Z",
      cells: 1,
      billed_seconds: 3600,
      idle_seconds: 0,
      unpriced_cells: 0,
      accrued_cost: 0,
      idle_cost: 0,
    });

    render(<FleetPage cells={cells} leases={leases} onRefresh={async () => {}} {...fleetLoaders} />);

    await waitFor(() => expect(screen.getByText("No pool declares a warm floor.")).toBeDefined());
  });
});

describe("the launch latency panel", () => {
  const cells: Cell[] = [];
  const leases: CapacityLease[] = [];

  function quietCost(): void {
    vi.spyOn(apiClient, "getWarmth").mockResolvedValue({ observed_at: "2026-07-29T12:00:00Z", pools: [] });
    vi.spyOn(apiClient, "getBilling").mockResolvedValue({
      observed_at: "2026-07-29T12:00:00Z",
      cells: 1,
      billed_seconds: 3600,
      idle_seconds: 0,
      unpriced_cells: 0,
      accrued_cost: 0,
      idle_cost: 0,
    });
  }

  it("shows the percentile with the evidence behind it", async () => {
    // The sample count is rendered because a p95 over three provisions is not a
    // p95, and a panel that hid the difference would invite exactly the
    // confident wrong decision this exists to prevent.
    quietCost();
    vi.spyOn(apiClient, "getLaunchLatency").mockResolvedValue({
      observed_at: "2026-07-29T12:00:00Z",
      pools: [
        {
          pool_id: "research/v5p",
          accelerator_type: "v5p-8",
          p95_seconds: 412,
          median_seconds: 180,
          slowest_seconds: 900,
          samples: 37,
          cells_in_cooldown: 0,
        },
      ],
    });

    render(<FleetPage cells={cells} leases={leases} onRefresh={async () => {}} {...fleetLoaders} />);

    await waitFor(() => expect(screen.getByText("Launch latency")).toBeDefined());
    expect(screen.getByText("p95 412s over 37 provisions")).toBeDefined();
  });

  it("says a pool is unmeasured rather than rendering it as instant", async () => {
    // The failure this panel most easily introduces: a pool that has never
    // provisioned sorts to the top of any latency table and reads as the
    // fastest option on the page.
    quietCost();
    vi.spyOn(apiClient, "getLaunchLatency").mockResolvedValue({
      observed_at: "2026-07-29T12:00:00Z",
      pools: [
        {
          pool_id: "research/v6e",
          accelerator_type: "v6e-16",
          p95_seconds: null,
          median_seconds: null,
          slowest_seconds: null,
          samples: 0,
          cells_in_cooldown: 0,
        },
      ],
    });

    render(<FleetPage cells={cells} leases={leases} onRefresh={async () => {}} {...fleetLoaders} />);

    await waitFor(() => expect(screen.getByText("never provisioned - no measurement")).toBeDefined());
  });

  it("names a cooldown, because a pool nobody asks looks fast", async () => {
    quietCost();
    vi.spyOn(apiClient, "getLaunchLatency").mockResolvedValue({
      observed_at: "2026-07-29T12:00:00Z",
      pools: [
        {
          pool_id: "research/v5p",
          accelerator_type: "v5p-8",
          p95_seconds: 60,
          median_seconds: 60,
          slowest_seconds: 60,
          samples: 4,
          cells_in_cooldown: 3,
        },
      ],
    });

    render(<FleetPage cells={cells} leases={leases} onRefresh={async () => {}} {...fleetLoaders} />);

    await waitFor(() => expect(screen.getByText("p95 60s over 4 provisions, 3 in cooldown")).toBeDefined());
  });

  it("reports a failure rather than rendering an empty panel", async () => {
    quietCost();
    vi.spyOn(apiClient, "getLaunchLatency").mockRejectedValue(new Error("Eyvan API request timed out"));

    render(<FleetPage cells={cells} leases={leases} onRefresh={async () => {}} {...fleetLoaders} />);

    await waitFor(() => expect(screen.getByText("Eyvan API request timed out")).toBeDefined());
  });
});

describe("what this run cost", () => {
  const spend = (overrides: Partial<WorkloadSpend> = {}): WorkloadSpend => ({
    workloadId: "wl-1",
    state: "succeeded",
    acceleratorHours: { tpu: 8 },
    settled: 9.6,
    priced: true,
    reserved: 0,
    leases: 1,
    shape: "v5p-8",
    capacityClass: "spot",
    onDemandHourly: 4,
    onDemandSource: "declared by pool pool-od",
    ...overrides,
  });

  const renderCost = (value: WorkloadSpend | Error) =>
    render(
      <WorkloadDetailPage
        workload={workload()}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => (value instanceof Error ? Promise.reject(value) : Promise.resolve(value))}
      />,
    );

  it("prints the sentence a spot-first scheduler exists to say", async () => {
    renderCost(spend());

    expect(await screen.findByText(/8.0 TPU-hours/)).toBeTruthy();
    expect(screen.getByText("$9.60")).toBeTruthy();
    expect(screen.getByText(/\$32.00 \(declared by pool pool-od\)/)).toBeTruthy();
    expect(screen.getByText(/\$22.40 \(70% less\)/)).toBeTruthy();
  });

  it("says unknown rather than free when no pool declares a price", async () => {
    // An unpriced pool settles to zero. Rendering $0.00 presents the absence of
    // a price as the absence of a bill, and that is the number a team budgets
    // against.
    renderCost(spend({ priced: false, settled: 0 }));

    expect(await screen.findByText(/no pool this run used declares a unit price/)).toBeTruthy();
    expect(screen.queryByText("$0.00")).toBeNull();
  });

  it("claims no saving against a run whose own cost is unknown", async () => {
    // Subtracting a real price from an unknown one produces a fictional saving.
    renderCost(spend({ priced: false, settled: 0 }));

    await screen.findByText(/no pool this run used declares a unit price/);
    expect(screen.queryByText(/% less/)).toBeNull();
  });

  it("says why there is no comparison instead of leaving it blank", async () => {
    renderCost(spend({ onDemandHourly: undefined, onDemandSource: "no on-demand pool for v5p-8 declares a price" }));

    expect(await screen.findByText(/Unknown — no on-demand pool for v5p-8 declares a price/)).toBeTruthy();
  });

  it("distinguishes a run that never held a lease from one that cost nothing", async () => {
    renderCost(spend({ leases: 0, acceleratorHours: {}, settled: 0, priced: false }));

    expect(await screen.findByText(/never held a lease/)).toBeTruthy();
  });

  it("names a failure rather than rendering an empty cost panel", async () => {
    // An empty cost panel reads as "nothing is being spent", which is the most
    // expensive thing it could say incorrectly.
    renderCost(new Error("Eyvan API request failed with 503"));

    expect(await screen.findByText(/503/)).toBeTruthy();
  });
});
