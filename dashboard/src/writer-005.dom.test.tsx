// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { computeOverview, EyvanApiClient } from "./api/client";
import { BillingPage } from "./pages/BillingPage";
import { CapacityPage } from "./pages/CapacityPage";
import { FleetPage } from "./pages/FleetPage";

const scope = { orgId: "org-a", projectId: "project-a" };
const observedAt = "2026-09-01T00:00:00Z";
const resources = (cpu: number, gpu: number, tpu = 0) => ({ cpu, gpu, tpu, memory_gib: 0, custom: {} });
const rawCell = (cpu: number, gpu: number) => ({
  id: "cell-gpu",
  provider: "gcp-compute",
  region: "us-central1",
  state: "ready",
  capacity: resources(cpu, gpu),
  allocatable: resources(cpu, gpu),
  labels: {},
  last_heartbeat_at: observedAt,
});
const rawLease = {
  id: "lease-gpu",
  workload_id: "running-gpu",
  project_id: "project-a",
  cell_id: "cell-gpu",
  state: "active",
  resources: { replicas: 1, per_replica: resources(0, 4) },
  hourly_cost: 2,
  expires_at: observedAt,
};
const rawWorkload = {
  id: "pending-gpu",
  org_id: "org-a",
  project_id: "project-a",
  state: "pending",
  attempt: 1,
  created_by: "alice",
  created_at: observedAt,
  updated_at: observedAt,
  spec: {
    name: "GPU training",
    execution: { kind: "command", command: "train" },
    gang: { replicas: 2, per_replica: resources(0, 4), required_topology: {} },
    metadata: {},
    accelerator_type: "nvidia-a100",
    priority_band: "batch",
  },
};

function stubRoutes(routes: Record<string, unknown>) {
  const fetch = vi.fn(async (input: string | URL | Request) => {
    const url = new URL(String(input), "http://localhost");
    if (!(url.pathname in routes)) throw new Error(`Unexpected request: ${url.pathname}`);
    return new Response(JSON.stringify(routes[url.pathname]), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("writer-005 GPU projection contract", () => {
  it.each([0, 16])("preserves GPU demand and occupancy on a host with %i CPUs", async (cpu) => {
    const fetch = stubRoutes({
      "/api/v1/workloads": [rawWorkload],
      "/api/v1/leases": [rawLease],
      "/api/v1/cells": [rawCell(cpu, 8)],
    });
    const client = new EyvanApiClient("/api/v1", false, scope);
    const workloads = await client.listWorkloads();
    const leases = await client.listLeases(scope, undefined, false);
    const cells = await client.listCells(leases);
    const overview = computeOverview(workloads, cells, []);

    expect(workloads[0].resources).toMatchObject({
      accelerator: "nvidia-a100",
      acceleratorKind: "gpu",
      acceleratorCount: 4,
    });
    expect(leases[0]).toMatchObject({ resource: "GPU", amount: "1 x 4", resourceTotals: { GPU: 4 } });
    expect(cells[0].resources).toContainEqual({ name: "GPU", used: 4, total: 8 });
    expect(cells[0].utilization).toBe(50);
    expect(overview.fleetBreakdown).toContainEqual({ resource: "GPU", used: 4, total: 8, pending: 8 });
    expect(overview.fleetBreakdown.find((item) => item.resource === "TPU")?.pending ?? 0).toBe(0);
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it.each([
    { name: "CPU", perReplica: resources(2, 0), kind: undefined, accelerator: undefined, count: undefined },
    { name: "TPU", perReplica: resources(0, 0, 4), kind: "tpu", accelerator: "nvidia-a100", count: 4 },
    { name: "GPU", perReplica: resources(0, 4), kind: "gpu", accelerator: "GPU", count: 4 },
  ])("preserves $name accounting without inferring kind from the model name", async (item) => {
    const fetch = stubRoutes({
      "/api/v1/workloads": [
        {
          ...rawWorkload,
          spec: {
            ...rawWorkload.spec,
            accelerator_type: item.name === "GPU" ? undefined : rawWorkload.spec.accelerator_type,
            gang: { ...rawWorkload.spec.gang, per_replica: item.perReplica },
          },
        },
      ],
      "/api/v1/leases": [{ ...rawLease, resources: { replicas: 1, per_replica: item.perReplica } }],
    });
    const client = new EyvanApiClient("/api/v1", false, scope);
    const workloads = await client.listWorkloads();
    const leases = await client.listLeases(scope, undefined, false);
    const overview = computeOverview(workloads, [], []);
    const amount = item.perReplica.cpu || item.perReplica.tpu || item.perReplica.gpu;

    expect(workloads[0].resources).toMatchObject({
      accelerator: item.accelerator,
      acceleratorKind: item.kind,
      acceleratorCount: item.count,
    });
    expect(leases[0]).toMatchObject({ resource: item.name, amount: `1 x ${amount}` });
    expect(leases[0].resourceTotals).toEqual({ [item.name]: amount });
    expect(overview.fleetBreakdown).toContainEqual({ resource: item.name, used: 0, total: 0, pending: 2 * amount });
    if (item.name === "TPU") {
      const { acceleratorKind: _kind, ...legacyResources } = workloads[0].resources;
      const legacy = computeOverview([{ ...workloads[0], resources: legacyResources }], [], []);
      expect(legacy.fleetBreakdown).toContainEqual({ resource: "TPU", used: 0, total: 0, pending: 8 });
    }
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});

describe("writer-005 fleet occupancy contract", () => {
  it.each(["empty selected project", "fleet manager without project read"])(
    "does not claim global idle occupancy with %s",
    async () => {
      const fetch = stubRoutes({ "/api/v1/cells": [rawCell(8, 0)] });
      const client = new EyvanApiClient("/api/v1", false, scope);
      const cells = await client.listCells([]);
      const never = () => new Promise<never>(() => {});
      render(
        <FleetPage
          cells={cells}
          leases={[]}
          onRefresh={async () => {}}
          loadWarmth={never}
          loadBilling={never}
          loadLaunchLatency={never}
          onDrainCell={async () => {}}
        />,
      );
      fireEvent.click(screen.getByRole("button", { name: "Drain" }));
      const dialog = await screen.findByRole("dialog");

      expect(within(dialog).queryByText("No workload currently holds capacity here.")).toBeNull();
      expect(within(dialog).getByText("No selected-project workloads currently hold capacity here.")).toBeTruthy();
      expect(within(dialog).getByText(/Other-project occupancy is unknown/)).toBeTruthy();
      expect(screen.getByText("Selected-project workloads")).toBeTruthy();
      expect(fetch).toHaveBeenCalledTimes(1);
    },
  );
});

const auxiliaryReads = [
  { name: "billing", path: "/api/v1/capacity/billing", read: (client: EyvanApiClient) => client.getBilling() },
  { name: "warmth", path: "/api/v1/capacity/warmth", read: (client: EyvanApiClient) => client.getWarmth() },
  {
    name: "launch latency",
    path: "/api/v1/capacity/launch-latency",
    read: (client: EyvanApiClient) => client.getLaunchLatency(),
  },
  {
    name: "quota refresh",
    path: "/api/v1/capacity/quota",
    read: (client: EyvanApiClient) =>
      client.getTpuQuota(scope, undefined, { project: "provider-project", refresh: true }),
  },
];

describe("writer-005 explicit demo isolation", () => {
  it.each(auxiliaryReads)("refuses live $name readings in demo mode without fetch", async ({ read }) => {
    const fetch = vi.fn().mockRejectedValue(new Error("Live fetch attempted"));
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", true, scope);

    await expect(read(client)).rejects.toMatchObject({
      name: "ApiError",
      message: expect.stringContaining("demo mode"),
    });
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(auxiliaryReads)("preserves live $name requests and returned data", async ({ path, read }) => {
    const report = { observed_at: observedAt, live_marker: "returned unchanged" };
    const fetch = stubRoutes({ [path]: report });
    const client = new EyvanApiClient("/api/v1", false, scope);

    expect(await read(client)).toEqual(report);
    expect(fetch).toHaveBeenCalledTimes(1);
    if (path.endsWith("/quota")) {
      expect(String(fetch.mock.calls[0][0])).toBe("/api/v1/capacity/quota?project=provider-project&refresh=true");
    }
  });

  it.each(["Fleet", "Billing", "Capacity"])(
    "mounts demo %s with an explicit unavailable notice, not live data",
    async (page) => {
      const fetch = vi.fn().mockRejectedValue(new Error("Live fetch attempted"));
      vi.stubGlobal("fetch", fetch);
      const client = new EyvanApiClient("/api/v1", true, scope);
      if (page === "Fleet") {
        render(
          <FleetPage
            cells={await client.listCells()}
            leases={await client.listLeases()}
            onRefresh={async () => {}}
            loadWarmth={(signal) => client.getWarmth(scope, signal)}
            loadBilling={(signal) => client.getBilling(scope, signal)}
            loadLaunchLatency={(signal) => client.getLaunchLatency(scope, signal)}
          />,
        );
      } else if (page === "Billing") {
        render(
          <BillingPage
            loadBilling={(signal) => client.getBilling(scope, signal)}
            loadUsage={(signal) => client.getProjectUsage(scope, signal)}
          />,
        );
      } else {
        render(<CapacityPage loadQuota={(signal, options) => client.getTpuQuota(scope, signal, options)} />);
      }

      expect((await screen.findAllByText(/unavailable in demo mode/)).length).toBeGreaterThan(0);
      expect(fetch).not.toHaveBeenCalled();
    },
  );
});
