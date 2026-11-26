// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { ApiError, type EyvanApiClient } from "./api/client";
import { mockCells, mockWorkloads } from "./api/mockData";
import type { SessionAuthority, SessionPermission, WorkloadSummary } from "./types";

function authority(permissions: SessionPermission[], enabled = true): SessionAuthority {
  return {
    subject: "alice",
    displayName: "Alice",
    globalPermissions: [],
    scopes: [
      {
        orgId: "acme",
        projectId: "research",
        displayName: "Research",
        registered: true,
        enabled,
        acceptsNewWork: enabled,
        roles: [],
        permissions,
      },
    ],
  };
}

const operator = authority(["read", "submit", "operate", "manage_fleet"]);

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function fakeClient() {
  const calls = {
    getSessionAuthority: vi.fn(async (): Promise<SessionAuthority> => operator),
    getHealth: vi.fn(async () => ({ status: "ok", service: "eyvan" })),
    listWorkloads: vi.fn(
      async (): Promise<WorkloadSummary[]> => [{ ...mockWorkloads[0], org: "acme", project: "research" }],
    ),
    listLeases: vi.fn(async () => []),
    listBudgets: vi.fn(async () => []),
    listPolicies: vi.fn(async () => []),
    getQueueTrend: vi.fn(async () => []),
    listCells: vi.fn(async () => mockCells.slice(0, 1)),
    listAuditEvents: vi.fn(async () => []),
    followEvents: vi.fn(async (_event: () => void, signal: AbortSignal) => {
      await new Promise<void>((resolve) => {
        if (signal.aborted) resolve();
        else signal.addEventListener("abort", () => resolve(), { once: true });
      });
    }),
    listAvailableShapes: vi.fn(async () => []),
    listCapacityPools: vi.fn(async () => [
      {
        id: "pool-a",
        fleetName: "research",
        acceleratorType: "v6e-8",
        capacityClass: "spot",
        zone: "us-east5-a",
        region: "us-east5",
        maxCells: 1,
        minCells: 0,
        enabled: true,
      },
    ]),
    getProjectUsage: vi.fn(async () => ({ orgId: "acme", projectId: "research", subjects: [] })),
    getWarmth: vi.fn(async () => ({ observed_at: "2026-01-01T00:00:00Z", pools: [] })),
    getBilling: vi.fn(async () => ({ observed_at: "2026-01-01T00:00:00Z", accrued_cost: 0, idle_cost: 0 })),
    getLaunchLatency: vi.fn(async () => ({ observed_at: "2026-01-01T00:00:00Z", pools: [] })),
    drainCell: vi.fn(async () => undefined),
    disableFleet: vi.fn(async () => undefined),
  };
  return { client: calls as unknown as EyvanApiClient, calls };
}

function renderApp(client: EyvanApiClient) {
  return render(<App client={client} identityProvider={() => undefined} />);
}

afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
  vi.restoreAllMocks();
});

const requestCases = [
  ["shapes", "listAvailableShapes"],
  ["usage", "getProjectUsage"],
  ["pools", "listCapacityPools"],
  ["stop", "disableFleet"],
  ["drain", "drainCell"],
] as const;

type RequestKind = (typeof requestCases)[number][0];

async function triggerRequest(kind: RequestKind) {
  if (kind === "shapes") {
    fireEvent.click(await screen.findByRole("button", { name: "Submit workload" }));
    return;
  }
  const panel = kind === "usage" ? "Billing" : "Fleet";
  fireEvent.click(await screen.findByRole("button", { name: panel }));
  await screen.findByRole("heading", { name: panel });
  if (kind === "stop") {
    fireEvent.click(await screen.findByRole("button", { name: "Stop" }));
    fireEvent.click(screen.getByRole("button", { name: "Stop fleet" }));
  } else if (kind === "drain") {
    fireEvent.click(await screen.findByRole("button", { name: "Drain" }));
    const dialog = screen.getByRole("dialog", { name: `Drain ${mockCells[0].name}` });
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: mockCells[0].name } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Drain cell" }));
  }
}

describe("typed denial boundaries for fleet actions and auxiliary reads", () => {
  it.each(requestCases)("fences current-scope %s denial before slow session reconciliation", async (kind, method) => {
    const { client, calls } = fakeClient();
    renderApp(client);
    await screen.findByRole("button", { name: "Submit workload" });
    const reconciliation = deferred<SessionAuthority>();
    calls.getSessionAuthority.mockReturnValueOnce(reconciliation.promise);
    calls[method].mockRejectedValueOnce(new ApiError("grant revoked", 403));
    await triggerRequest(kind);
    await waitFor(() => expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull());
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("button", { name: "Fleet" })).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
    // A reconciliation outage must not resurrect the last successful grant.
    await act(async () => reconciliation.reject(new ApiError("session unavailable", 503)));
    await screen.findByText("session unavailable");
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(calls[method]).toHaveBeenCalledTimes(1);
  });

  it.each(requestCases)("preserves current authority on an ordinary %s outage", async (kind, method) => {
    const { client, calls } = fakeClient();
    renderApp(client);
    await screen.findByRole("button", { name: "Submit workload" });
    calls[method].mockRejectedValueOnce(new ApiError("request unavailable", 503));
    await triggerRequest(kind);
    await act(async () => {
      await Promise.resolve();
    });
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Fleet" })).toBeDefined();
    if (kind !== "pools") await screen.findAllByText("request unavailable");
    if (kind === "shapes") expect(screen.getByRole("dialog", { name: "Run a job" })).toBeDefined();
    if (kind === "drain") expect(screen.getByRole("dialog", { name: `Drain ${mockCells[0].name}` })).toBeDefined();
  });

  it.each(requestCases)("ignores obsolete-scope %s denial", async (kind, method) => {
    const { client, calls } = fakeClient();
    const otherScope = { ...operator.scopes[0], projectId: "other", displayName: "Other" };
    calls.getSessionAuthority.mockResolvedValue({ ...operator, scopes: [...operator.scopes, otherScope] });
    renderApp(client);
    await screen.findByRole("button", { name: "Submit workload" });
    const oldRequest = deferred<never>();
    calls[method].mockReturnValueOnce(oldRequest.promise);
    await triggerRequest(kind);
    await waitFor(() => expect(calls[method]).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByRole("combobox", { name: "Project scope" }), {
      target: { value: JSON.stringify(["acme", "other"]) },
    });
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(calls.listWorkloads).toHaveBeenCalledTimes(2));
    await act(async () => oldRequest.reject(new ApiError("old scope revoked", 403)));
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined();
    expect((screen.getByRole("combobox", { name: "Project scope" }) as HTMLSelectElement).value).toBe(
      JSON.stringify(["acme", "other"]),
    );
  });
});

describe("authoritative session changes before dashboard data", () => {
  it.each([true, false])("fences revocation during a pending/failed data read (enabled=%s)", async (enabled) => {
    const { client, calls } = fakeClient();
    renderApp(client);
    fireEvent.click(await screen.findByRole("button", { name: "Submit workload" }));
    await screen.findByRole("dialog", { name: "Run a job" });

    const pending = deferred<WorkloadSummary[]>();
    calls.getSessionAuthority.mockResolvedValue(authority(["read"], enabled));
    calls.listWorkloads.mockReturnValueOnce(pending.promise);
    fireEvent(document, new Event("visibilitychange"));
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(calls.listWorkloads).toHaveBeenCalledTimes(2));
    // The session response, not the later data response, removes contradicted authority.
    expect(screen.queryByRole("dialog", { name: "Run a job" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Fleet" })).toBeNull();

    await act(async () => pending.reject(new ApiError("data temporarily unavailable", 503)));
    await screen.findByText(/Live refresh failed; showing the last successful snapshot/);
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();
    expect(screen.queryByRole("dialog", { name: "Run a job" })).toBeNull();

    fireEvent(document, new Event("visibilitychange"));
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(screen.queryByText(/Live refresh failed/)).toBeNull());
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();
  });
});
