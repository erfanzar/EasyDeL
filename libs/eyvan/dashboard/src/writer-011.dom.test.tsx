// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { workloadStatus, type EyvanApiClient } from "./api/client";
import { getMockWorkload, mockWorkloads } from "./api/mockData";
import { NO_AUTH_CAPABILITIES } from "./auth/oidc";
import { WorkloadDetailPage, type LogFollowHandlers } from "./pages/WorkloadDetailPage";
import type { SessionAuthority, WorkloadDetail, WorkloadExplanation, WorkloadSpend } from "./types";

function workload(): WorkloadDetail {
  const detail = getMockWorkload(mockWorkloads[0].id);
  if (!detail) throw new Error("the mock dataset has no workloads");
  return { ...detail, status: "queued" };
}

function explanation(message: string): WorkloadExplanation {
  return {
    workload_id: workload().id,
    state: "pending",
    placeable: false,
    placement_cell_id: null,
    code: "no_matching_cell",
    message,
    cells: [],
    considered_cell_count: 1,
    provisioning_cell_ids: [],
    scheduling_reason: null,
  };
}

function spend(leases: number): WorkloadSpend {
  return {
    workloadId: workload().id,
    state: "pending",
    acceleratorHours: { tpu: leases },
    settled: leases * 5,
    reserved: 0,
    priced: true,
    leases,
    onDemandSource: "no on-demand price declared",
  };
}

const actions = { onBack: () => {}, onAction: () => {}, canOperate: false };
const quietReads = {
  loadExplanation: () => new Promise<WorkloadExplanation>(() => {}),
  loadCost: () => new Promise<WorkloadSpend>(() => {}),
};

function field(label: string): string | null | undefined {
  return screen.getByText(label, { selector: "dt" }).nextElementSibling?.textContent;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
  vi.restoreAllMocks();
});

describe("writer-011: supplemental reads follow successful detail snapshots", () => {
  it("refreshes stable loaders for unchanged pending capacity and later completed spend", async () => {
    const loadExplanation = vi.fn(async () => explanation("original capacity rejection"));
    const loadCost = vi.fn(async () => spend(0));
    const detail = workload();
    const page = (value: WorkloadDetail) => (
      <WorkloadDetailPage workload={value} {...actions} loadExplanation={loadExplanation} loadCost={loadCost} />
    );
    const view = render(page(detail));
    await screen.findByText("original capacity rejection");
    await screen.findByText(/This run has never held a lease/);

    // An unrelated render is not a successful new detail snapshot.
    view.rerender(page(detail));
    expect(loadExplanation).toHaveBeenCalledTimes(1);
    expect(loadCost).toHaveBeenCalledTimes(1);

    loadExplanation.mockResolvedValue(explanation("new capacity evidence"));
    view.rerender(page({ ...detail }));
    await screen.findByText("new capacity evidence");
    expect(loadExplanation).toHaveBeenCalledTimes(2);

    loadCost.mockResolvedValue(spend(1));
    view.rerender(page({ ...detail, status: "completed", attempt: 2 }));
    await screen.findByText(/across 1 lease/);
    expect(screen.queryByText(/This run has never held a lease/)).toBeNull();
    expect(screen.queryByText("Placement verdict")).toBeNull();
    expect(loadCost).toHaveBeenCalledTimes(3);
  });

  it("aborts previous snapshot reads and refuses their late results", async () => {
    const oldExplanation = deferred<WorkloadExplanation>();
    const oldSpend = deferred<WorkloadSpend>();
    const signals: AbortSignal[] = [];
    const loadExplanation = vi.fn((_id: string, signal: AbortSignal) => {
      signals.push(signal);
      return signals.length === 1 ? oldExplanation.promise : Promise.resolve(explanation("current capacity"));
    });
    const costSignals: AbortSignal[] = [];
    const loadCost = vi.fn((_id: string, signal: AbortSignal) => {
      costSignals.push(signal);
      return costSignals.length === 1 ? oldSpend.promise : Promise.resolve(spend(1));
    });
    const detail = workload();
    const page = (value: WorkloadDetail) => (
      <WorkloadDetailPage workload={value} {...actions} loadExplanation={loadExplanation} loadCost={loadCost} />
    );
    const view = render(page(detail));
    view.rerender(page({ ...detail, attempt: 2 }));
    await screen.findByText("current capacity");
    await screen.findByText(/across 1 lease/);
    expect(signals[0].aborted).toBe(true);
    expect(costSignals[0].aborted).toBe(true);
    await act(async () => {
      oldExplanation.resolve(explanation("obsolete capacity"));
      oldSpend.resolve(spend(0));
    });
    expect(screen.queryByText("obsolete capacity")).toBeNull();
    expect(screen.queryByText(/This run has never held a lease/)).toBeNull();
  });

  it("refreshes panels through the real App background event path without changing workload id", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("unexpected network request"));
    let currentDetail = { ...workload(), org: "acme", project: "research" };
    let currentExplanation = explanation("initial App capacity");
    let currentSpend = spend(0);
    const session: SessionAuthority = {
      subject: "alice",
      globalPermissions: [],
      scopes: [
        {
          orgId: "acme",
          projectId: "research",
          displayName: "Research",
          registered: true,
          enabled: true,
          acceptsNewWork: true,
          roles: ["project_viewer"],
          permissions: ["read"],
        },
      ],
    };
    let notify: (() => void) | undefined;
    // Only this read-only route's client surface is needed; do not fall back
    // to a real transport for any missing method.
    const client = {
      getSessionAuthority: vi.fn(async () => session),
      getHealth: vi.fn(async () => ({ status: "ok", service: "eyvan" })),
      listWorkloads: vi.fn(async () => [currentDetail]),
      listLeases: vi.fn(async () => []),
      listBudgets: vi.fn(async () => []),
      listPolicies: vi.fn(async () => []),
      getQueueTrend: vi.fn(async () => []),
      listAuditEvents: vi.fn(async () => []),
      getWorkload: vi.fn(async () => ({ ...currentDetail })),
      explainWorkload: vi.fn(async () => currentExplanation),
      getWorkloadCost: vi.fn(async () => currentSpend),
      followEvents: vi.fn(async (onEvent: () => void, signal: AbortSignal) => {
        notify = onEvent;
        await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
      }),
    };
    window.history.replaceState(null, "", `/#workloads/${currentDetail.id}`);
    render(
      <App
        client={client as unknown as EyvanApiClient}
        identityProvider={() => ({
          displayName: "Alice",
          detail: "alice",
          mode: "oidc",
          roles: [],
          scope: { orgId: "acme", projectId: "research" },
          capabilities: NO_AUTH_CAPABILITIES,
        })}
      />,
    );
    await screen.findByText("initial App capacity");
    await screen.findByText(/This run has never held a lease/);
    await waitFor(() => expect(notify).toBeDefined());

    currentExplanation = explanation("refreshed App capacity");
    act(() => notify?.());
    await screen.findByText("refreshed App capacity", {}, { timeout: 3000 });
    currentDetail = { ...currentDetail, status: "completed" };
    currentSpend = spend(1);
    act(() => notify?.());
    await screen.findByText(/across 1 lease/, {}, { timeout: 3000 });
    expect(screen.queryByText(/This run has never held a lease/)).toBeNull();
    expect(screen.queryByText("Placement verdict")).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe("writer-011: accelerator HBM uses reported chips rather than quota-unit suffixes", () => {
  it.each([
    ["v5p-8", 4, 1, "380 GB"],
    ["v4-8", 4, 1, "128 GB"],
    ["v5e-8", 8, 1, "128 GB"],
    ["v6e-8", 8, 1, "256 GB"],
    // A v5p-16 slice spans two hosts but has eight chips, not sixteen.
    ["v5p-16", 8, 1, "760 GB"],
    ["v5p-16", 8, 2, "760 GB × 2"],
    ["v99-8", 8, 1, "Unknown"],
    ["v5p-8", undefined, 1, "Unknown"],
    ["v5p-8", 0, 1, "Unknown"],
    ["v5p-8", Number.NaN, 1, "Unknown"],
  ])("reports %s with %s chips and %s slices as %s", (accelerator, acceleratorCount, slices, expected) => {
    const detail = workload();
    render(
      <WorkloadDetailPage
        workload={{
          ...detail,
          resources: { ...detail.resources, accelerator, acceleratorCount, replicas: 1, slices },
        }}
        {...actions}
        {...quietReads}
      />,
    );
    expect(field("Accelerator HBM")).toBe(expected);
  });
});

describe("writer-011: live tail pinning survives a saturated buffer", () => {
  it.each([
    [2000, 0, 5100],
    [1999, 0, 5100],
    [2000, 1200, 3300],
  ])("pins %s records at %s pixels from the bottom to %s", async (records, distance, expected) => {
    const state: { handlers?: LogFollowHandlers } = {};
    render(
      <WorkloadDetailPage
        workload={workload()}
        {...actions}
        {...quietReads}
        followLogs={async (handlers, signal) => {
          state.handlers = handlers;
          await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
        }}
      />,
    );
    fireEvent.click(screen.getByRole("tab", { name: /^Logs/ }));
    fireEvent.click(screen.getByRole("button", { name: "Follow" }));
    await waitFor(() => expect(state.handlers).toBeDefined());
    act(() => {
      for (let index = 0; index < records; index += 1) {
        state.handlers?.onRecord({ id: `line-${index}`, message: `line ${index}` });
      }
    });
    const stream = screen.getByRole("log");
    Object.defineProperty(stream, "scrollHeight", { value: 5000, configurable: true });
    Object.defineProperty(stream, "clientHeight", { value: 500, configurable: true });
    stream.scrollTop = 4500 - distance;
    fireEvent.scroll(stream);
    // jsdom does not lay out wrapping: model the taller tail's measured height.
    Object.defineProperty(stream, "scrollHeight", { value: 5100, configurable: true });
    act(() => state.handlers?.onRecord({ id: "taller-tail", message: "latest wrapped\nmultiline output" }));
    expect(screen.getByText(/latest wrapped/)).toBeDefined();
    expect(stream.scrollTop).toBe(expected);
    if (records === 2000) {
      expect(screen.queryByText("line 0")).toBeNull();
      expect(screen.getByText(/1 earlier\s+lines dropped/)).toBeDefined();
    }
  });
});

describe("writer-011: settled placement labels follow normalized public statuses", () => {
  it.each([
    ["succeeded", undefined, "Released"],
    ["failed", undefined, "Released"],
    ["cancelled", undefined, "Released"],
    ["pending", undefined, "Pending"],
    ["succeeded", "assigned-cell", "assigned-cell"],
  ])("labels %s with cell %s as %s", (state, cell, expected) => {
    const status = workloadStatus(state);
    if (state === "succeeded") expect(status).toBe("completed");
    render(<WorkloadDetailPage workload={{ ...workload(), status, cell }} {...actions} {...quietReads} />);
    expect(field("Placement")).toBe(expected);
  });
});
