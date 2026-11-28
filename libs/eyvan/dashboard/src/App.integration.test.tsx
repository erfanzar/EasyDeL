// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { ApiError, type EyvanApiClient } from "./api/client";
import { COALESCE_MS } from "./lib/useLiveRefresh";
import {
  getMockWorkload,
  mockAuditEvents,
  mockBudgets,
  mockCells,
  mockLeases,
  mockPolicies,
  mockWorkloads,
} from "./api/mockData";
import {
  __resetRuntimeOidcConfigForTests,
  loadRuntimeOidcConfig,
  NO_AUTH_CAPABILITIES,
  type AuthIdentity,
  type AuthScope,
} from "./auth/oidc";
import type {
  CapacityPoolSummary,
  EffectiveProjectScope,
  ProjectUsage,
  SchedulingPolicy,
  SessionAuthority,
  SessionPermission,
  WorkloadDetail,
  WorkloadSummary,
} from "./types";

const PERMISSIONS: Record<string, SessionPermission[]> = {
  project_viewer: ["read"],
  project_submitter: ["submit"],
  project_operator: ["read", "submit", "operate"],
  project_admin: ["read", "submit", "operate", "manage_project"],
  org_admin: ["read", "submit", "operate", "manage_project", "manage_own_capacity"],
  fleet_admin: ["read", "submit", "operate", "manage_project", "manage_own_capacity", "manage_fleet"],
};

function effectiveScope(
  role: string,
  scope: AuthScope = { orgId: "acme", projectId: "research" },
  overrides: Partial<EffectiveProjectScope> = {},
): EffectiveProjectScope {
  return {
    ...scope,
    displayName: `${scope.orgId}/${scope.projectId}`,
    registered: true,
    enabled: true,
    acceptsNewWork: true,
    roles: [role],
    permissions: PERMISSIONS[role] ?? [],
    ...overrides,
  };
}

function authority(role: string, scopes = [effectiveScope(role)]): SessionAuthority {
  return {
    subject: "alice",
    displayName: "Alice",
    globalPermissions: role === "fleet_admin" ? [...PERMISSIONS.fleet_admin] : [],
    scopes,
  };
}

function identity(): AuthIdentity {
  return {
    displayName: "Alice",
    detail: "alice@example.com",
    mode: "oidc",
    roles: [],
    scope: { orgId: "acme", projectId: "research" },
    capabilities: NO_AUTH_CAPABILITIES,
  };
}

function scopedWorkload(scope: AuthScope, name = `${scope.orgId}-${scope.projectId}-workload`): WorkloadDetail {
  const detail = getMockWorkload(mockWorkloads[0].id);
  if (!detail) throw new Error("mock workload detail is missing");
  return {
    ...structuredClone(detail),
    id: `wl-${scope.orgId}-${scope.projectId}`,
    name,
    org: scope.orgId,
    project: scope.projectId,
  };
}

function policyFor(scope: AuthScope): SchedulingPolicy {
  return {
    ...mockPolicies[0],
    id: `${scope.orgId}/${scope.projectId}`,
    name: `Placement policy (${scope.orgId}/${scope.projectId})`,
    scope: scope.projectId,
    type: "placement",
    configuration: {
      allowedProviders: ["gcp"],
      allowedRegions: [],
      allowedCapacityClasses: [],
      deniedAcceleratorTypes: [],
      maxPriority: 1000,
    },
  };
}

/** One measured zone and one nobody could measure, which is the realistic case. */
function quotaSnapshot() {
  return {
    observed_at: "2026-01-01T00:00:00Z",
    provider: "gcp",
    project: "gcp-project",
    available_projects: ["gcp-project"],
    status: "ok" as const,
    refreshed_at: "2026-01-01T00:00:00Z",
    age_s: 0,
    cached: false,
    ttl_s: 120,
    entries: [
      {
        zone: "us-central2-b",
        region: "us-central2",
        generation: "v6e",
        accelerator_type: "v6e-8",
        capacity_class: "spot" as const,
        metric: "tpu.googleapis.com/tpu-v6e-preemptible",
        chips_per_slice: 8,
        limit_chips: 32,
        in_use_chips: 16,
        free_chips: 16,
        free_slices: 2,
        state: "known" as const,
        usage_observed_at: "2026-01-01T00:00:00Z",
        usage_age_s: 120,
      },
      {
        zone: "europe-west4-a",
        region: "europe-west4",
        generation: "v6e",
        accelerator_type: "v6e-8",
        capacity_class: "spot" as const,
        metric: "tpu.googleapis.com/tpu-v6e-preemptible",
        chips_per_slice: 8,
        limit_chips: 64,
        in_use_chips: null,
        free_chips: null,
        free_slices: null,
        state: "usage_unknown" as const,
        usage_observed_at: null,
        usage_age_s: null,
      },
    ],
    failures: [],
  };
}

type FakeOverrides = Partial<{
  getSessionAuthority: EyvanApiClient["getSessionAuthority"];
  listWorkloads: EyvanApiClient["listWorkloads"];
  listPolicies: EyvanApiClient["listPolicies"];
  submitWorkload: EyvanApiClient["submitWorkload"];
  listAvailableShapes: EyvanApiClient["listAvailableShapes"];
  updatePolicy: EyvanApiClient["updatePolicy"];
  getWorkload: EyvanApiClient["getWorkload"];
  workloadAction: EyvanApiClient["workloadAction"];
  followEvents: EyvanApiClient["followEvents"];
}>;

function fakeClient(session: SessionAuthority, overrides: FakeOverrides = {}) {
  const streamCallbacks: Array<() => void> = [];
  const client = {
    getSessionAuthority: vi.fn(overrides.getSessionAuthority ?? (async () => session)),
    getHealth: vi.fn(async () => ({ status: "ok", service: "eyvan" })),
    listWorkloads: vi.fn(
      overrides.listWorkloads ??
        (async (_filters, scope) => [scopedWorkload(scope ?? { orgId: "acme", projectId: "research" })]),
    ),
    listLeases: vi.fn(async () => mockLeases.slice(0, 1)),
    listBudgets: vi.fn(async () => mockBudgets.slice(0, 1)),
    listPolicies: vi.fn(
      overrides.listPolicies ?? (async (scope) => [policyFor(scope ?? { orgId: "acme", projectId: "research" })]),
    ),
    getQueueTrend: vi.fn(async () => [{ label: "10:00", submitted: 1, admitted: 1 }]),
    listCells: vi.fn(async () => mockCells.slice(0, 1)),
    // The fleet page reads registered pools, which is a different question
    // from which cells exist: a fleet with twenty pools and nothing running
    // is the ordinary case and used to be indistinguishable from no fleet.
    listCapacityPools: vi.fn(async (): Promise<CapacityPoolSummary[]> => []),
    listDiscoveredTpus: vi.fn(async () => {
      throw new Error("Discovery index unavailable");
    }),
    listAttachments: vi.fn(async () => []),
    getProjectUsage: vi.fn(async (): Promise<ProjectUsage> => ({ orgId: "acme", projectId: "research", subjects: [] })),
    listAuditEvents: vi.fn(async () => mockAuditEvents.slice(0, 1)),
    followEvents: vi.fn(
      overrides.followEvents ??
        (async (onEvent: () => void, signal: AbortSignal, onConnectionChange?: (connected: boolean) => void) => {
          streamCallbacks.push(onEvent);
          onConnectionChange?.(true);
          await new Promise<void>((resolve) => {
            if (signal.aborted) resolve();
            else signal.addEventListener("abort", () => resolve(), { once: true });
          });
        }),
    ),
    getWorkload: vi.fn(
      overrides.getWorkload ??
        (async (id, scope) => ({
          ...scopedWorkload(scope ?? { orgId: "acme", projectId: "research" }),
          id,
        })),
    ),
    explainWorkload: vi.fn(async (id: string) => ({
      workload_id: id,
      state: "running",
      placeable: true,
      placement_cell_id: "cell-a",
      code: null,
      message: null,
      cells: [],
      considered_cell_count: 1,
      provisioning_cell_ids: [],
      scheduling_reason: null,
    })),
    workloadProcessStatus: vi.fn(async () => ({})),
    getWarmth: vi.fn(async () => ({ observed_at: "2026-01-01T00:00:00Z", pools: [] })),
    getBilling: vi.fn(async () => ({ observed_at: "2026-01-01T00:00:00Z", accrued_cost: 0, idle_cost: 0 })),
    getLaunchLatency: vi.fn(async () => ({ observed_at: "2026-01-01T00:00:00Z", pools: [] })),
    getTpuQuota: vi.fn(async () => quotaSnapshot()),
    listAvailableShapes: vi.fn(
      overrides.listAvailableShapes ??
        (async () => [
          {
            fleet: "research",
            pool: "research-v6e-8-us-east5-a",
            shape: "v6e-8",
            capacityClass: "spot" as const,
            region: "us-east5",
            zone: "us-east5-a",
          },
        ]),
    ),
    submitWorkload: vi.fn(
      overrides.submitWorkload ??
        (async (input, scope) =>
          scopedWorkload(scope ?? { orgId: "acme", projectId: "research" }, input.name ?? input.shape)),
    ),
    workloadAction: vi.fn(
      overrides.workloadAction ??
        (async (id, action, scope) => ({
          ...scopedWorkload(scope ?? { orgId: "acme", projectId: "research" }),
          id,
          status: action === "cancel" ? "preempting" : action === "checkpoint" ? "checkpointing" : "queued",
        })),
    ),
    updatePolicy: vi.fn(
      overrides.updatePolicy ??
        (async (policy, update, scope) => {
          const patch = typeof update === "boolean" ? { preemptionEnabled: update } : update;
          return {
            ...policy,
            id: `${scope?.orgId}/${scope?.projectId}`,
            scope: scope?.projectId ?? policy.scope,
            preemptionEnabled: patch.preemptionEnabled ?? policy.preemptionEnabled,
            crossRegionTransferPolicy: patch.crossRegionTransferPolicy ?? policy.crossRegionTransferPolicy,
            crossRegionAllowedBuckets: patch.crossRegionAllowedBuckets ?? policy.crossRegionAllowedBuckets,
            crossRegionAllowMultiregion: patch.crossRegionAllowMultiregion ?? policy.crossRegionAllowMultiregion,
          };
        }),
    ),
  };
  return { client: client as unknown as EyvanApiClient, calls: client, streamCallbacks };
}

function renderApp(client: EyvanApiClient) {
  return render(<App client={client} identityProvider={identity} />);
}

async function showPanel(name: string) {
  // findByRole retries; getByRole asserts the render already happened. The
  // navigation button appears when control-plane state resolves, so querying it
  // synchronously races a slow machine -- which is how this failed on CI while
  // passing on every developer machine.
  fireEvent.click(await screen.findByRole("button", { name }));
  await waitFor(() => expect(screen.getByRole("heading", { name })).toBeDefined());
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

afterEach(() => {
  cleanup();
  __resetRuntimeOidcConfigForTests();
  window.history.replaceState(null, "", "/");
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("App composes server session authority", () => {
  it("refreshes registered inventory without requesting host-only discovery on a remote plane", async () => {
    await loadRuntimeOidcConfig(async () => Response.json({ local_attachments_available: false }));
    window.location.hash = "#/fleet";
    const { client, calls } = fakeClient(authority("fleet_admin"));
    renderApp(client);
    await screen.findByText("eyvan gcp connect --tpu NAME");
    expect(screen.queryByRole("button", { name: "Add TPU" })).toBeNull();
    expect(screen.queryByRole("dialog", { name: "Add TPU" })).toBeNull();
    expect(screen.queryByRole("region", { name: "TPUs in your Google Cloud" })).toBeNull();
    const reads = calls.listCells.mock.calls.length;
    const poolReads = calls.listCapacityPools.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Refresh inventory" }));
    await screen.findByText("Fleet inventory reloaded from Eyvan.");
    expect(calls.listCells.mock.calls.length).toBeGreaterThan(reads);
    expect(calls.listCapacityPools.mock.calls.length).toBeGreaterThan(poolReads);
    expect(calls.listDiscoveredTpus).not.toHaveBeenCalled();
    expect(calls.listAttachments).not.toHaveBeenCalled();
  });

  it.each(["fleet_admin", "project_admin"])(
    "respects %s authority when local attachments are available",
    async (role) => {
      await loadRuntimeOidcConfig(async () => Response.json({ local_attachments_available: true }));
      window.location.hash = "#/fleet";
      const { client, calls } = fakeClient(authority(role));
      renderApp(client);
      if (role === "fleet_admin") {
        await screen.findByText("Discovery index unavailable");
        fireEvent.click(screen.getByRole("button", { name: "Add TPU" }));
        await screen.findByRole("dialog", { name: "Add TPU" });
        expect(calls.listAttachments).toHaveBeenCalled();
      } else {
        await screen.findByRole("region", { name: "Operations metrics" });
        expect(screen.queryByRole("button", { name: "Add TPU" })).toBeNull();
        expect(calls.listDiscoveredTpus).not.toHaveBeenCalled();
        expect(calls.listAttachments).not.toHaveBeenCalled();
      }
    },
  );

  it.each(["project_viewer", "project_operator", "project_admin", "org_admin"])(
    "does not mount fleet-wide billing for a %s deep link",
    async (role) => {
      window.location.hash = "#/billing";
      const { client, calls } = fakeClient(authority(role));
      renderApp(client);

      await screen.findByRole("button", { name: "Sign out Alice" });
      expect(screen.queryByRole("button", { name: "Billing" })).toBeNull();
      await screen.findByRole("region", { name: "Operations metrics" });
      expect(calls.getBilling).not.toHaveBeenCalled();
    },
  );

  it("refreshes same-scope billing and usage without refetching on unrelated renders", async () => {
    const { client, calls } = fakeClient(authority("fleet_admin"));
    calls.getBilling.mockResolvedValue({
      observed_at: "2026-01-01T00:00:00Z",
      accrued_cost: 12,
      idle_cost: 0,
    });
    const page = renderApp(client);
    await showPanel("Billing");
    await screen.findByText("$12.00");
    await screen.findByText("Nobody has submitted work yet");
    const initialBilling = calls.getBilling.mock.calls.length;
    const initialUsage = calls.getProjectUsage.mock.calls.length;
    page.rerender(<App client={client} identityProvider={identity} />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(calls.getBilling).toHaveBeenCalledTimes(initialBilling);
    expect(calls.getProjectUsage).toHaveBeenCalledTimes(initialUsage);

    calls.getBilling.mockResolvedValue({ observed_at: "2026-01-01T00:00:01Z", accrued_cost: 24, idle_cost: 0 });
    calls.getProjectUsage.mockResolvedValue({
      orgId: "acme",
      projectId: "research",
      subjects: [
        {
          subject: "new-researcher",
          workloads: 1,
          settledWorkloads: 1,
          leases: 1,
          acceleratorHours: 1,
          runningSeconds: 3600,
          settledCost: 24,
          unpricedLeases: 0,
        },
      ],
    });
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(calls.getSessionAuthority.mock.calls.length).toBeGreaterThan(1));
    await act(async () => {
      await Promise.resolve();
    });
    expect(within(screen.getByRole("region", { name: "What the fleet has run" })).getByText("$24.00")).toBeTruthy();
    expect(screen.getByText("new-researcher")).toBeTruthy();
    expect(screen.queryByText("Nobody has submitted work yet")).toBeNull();
  });

  it("refreshes same-scope applied pools from the visible inventory action", async () => {
    const { client, calls } = fakeClient(authority("fleet_admin"));
    const pool: CapacityPoolSummary = {
      id: "pool-1",
      fleetName: "refreshed-research-fleet",
      acceleratorType: "v6e-8",
      capacityClass: "spot",
      zone: "us-east5-a",
      region: "us-east5",
      maxCells: 1,
      minCells: 0,
      enabled: true,
    };
    calls.listCapacityPools.mockResolvedValue([pool]);
    const page = renderApp(client);
    await showPanel("Fleet");
    await screen.findByText(pool.fleetName);
    const initialReads = calls.listCapacityPools.mock.calls.length;
    page.rerender(<App client={client} identityProvider={identity} />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(calls.listCapacityPools).toHaveBeenCalledTimes(initialReads);

    calls.listCapacityPools.mockResolvedValue([{ ...pool, enabled: false, maxCells: 0 }]);
    fireEvent.click(screen.getByRole("button", { name: "Refresh inventory" }));
    await screen.findByText("Fleet inventory reloaded from Eyvan.");
    await screen.findByText("1 pool(s) disabled");
  });

  it("offers sign-out for a server-resolved OIDC identity without browser tokens", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "https://issuer.example.com");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "dashboard");
    try {
      const { client } = fakeClient(authority("project_viewer"));
      render(<App client={client} identityProvider={() => undefined} />);
      await waitFor(() => expect(screen.getByRole("button", { name: "Sign out Alice" })).toBeTruthy());
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it.each([
    ["project_viewer", false, false, false],
    ["project_submitter", true, false, false],
    ["project_operator", true, false, false],
    ["project_admin", true, true, false],
    ["org_admin", true, true, false],
    ["fleet_admin", true, true, true],
  ])("renders real panels and actions for %s", async (role, canSubmit, canManageProject, canManageFleet) => {
    const { client } = fakeClient(authority(role));
    renderApp(client);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Operations overview" })).toBeDefined());
    expect(screen.queryByRole("button", { name: "Submit workload" }) !== null).toBe(canSubmit);
    expect(screen.queryByRole("button", { name: "Fleet" }) !== null).toBe(canManageFleet);

    await showPanel("Workloads");
    expect(screen.queryByRole("button", { name: "New workload" }) !== null).toBe(canSubmit);

    if (role !== "project_submitter") {
      await showPanel("Budgets & policies");
      expect(screen.queryAllByRole("checkbox").length > 0).toBe(canManageProject);
      await showPanel("Audit events");
      expect(screen.getByText(mockAuditEvents[0].action)).toBeDefined();
    }

    if (canManageFleet) {
      await showPanel("Fleet");
      expect(screen.getByText("Fleet topology")).toBeDefined();
    }
    // Quota is resolved from capacity pool records, so it is gated by the
    // authority that can read them and never offered to anyone else.
    expect(screen.queryByRole("button", { name: "Quota & capacity" }) !== null).toBe(canManageFleet);
  });

  it("answers where a slice fits, and never shows an unmeasured zone as empty", async () => {
    const { client, calls } = fakeClient(authority("fleet_admin"));
    renderApp(client);

    await showPanel("Quota & capacity");

    await waitFor(() => expect(calls.getTpuQuota).toHaveBeenCalled());
    const measured = within(screen.getByRole("region", { name: "Free capacity by zone" }));
    expect(measured.getByText("us-central2-b")).toBeDefined();
    expect(measured.getByText("2 x v6e-8")).toBeDefined();
    // The zone whose usage nobody could read is on the page and carries no free
    // balance -- neither hidden, nor sorted among the measured rows as a zero.
    expect(measured.queryByText("europe-west4-a")).toBeNull();
    const unknown = within(screen.getByRole("region", { name: "Zones with no measurable free balance" }));
    expect(unknown.getByText("europe-west4-a")).toBeDefined();
    expect(unknown.getByText("64")).toBeDefined();
  });

  it.each([
    ["project_viewer", false],
    ["project_operator", true],
    ["project_admin", true],
    ["org_admin", true],
    ["fleet_admin", true],
  ])("uses resource-scoped session authority for %s detail actions", async (role, canOperate) => {
    window.location.hash = "#/workloads/wl-deep";
    const { client, calls } = fakeClient(authority(role));
    renderApp(client);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Workload detail" })).toBeDefined());
    await waitFor(() => expect(calls.getWorkload).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("acme-research-workload")).toBeDefined());
    expect(screen.queryByRole("button", { name: "Cancel" }) !== null).toBe(canOperate);
    expect(calls.getSessionAuthority).toHaveBeenCalledWith(
      expect.objectContaining({ orgId: "acme", projectId: "research" }),
      expect.any(AbortSignal),
    );
  });

  it("keeps a disabled registered project selectable and readable but suppresses every tenant mutation", async () => {
    const disabled = effectiveScope("project_admin", undefined, {
      enabled: false,
      acceptsNewWork: false,
    });
    const { client } = fakeClient(authority("project_admin", [disabled]));
    renderApp(client);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Operations overview" })).toBeDefined());
    const selector = screen.getByRole("combobox", { name: "Project scope" }) as HTMLSelectElement;
    expect(selector.options[0].disabled).toBe(false);
    expect(selector.options[0].textContent).toContain("history only");
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();

    await showPanel("Budgets & policies");
    expect(screen.queryByRole("checkbox")).toBeNull();
    expect(screen.getByText("Read only")).toBeDefined();
  });
});

describe("App scope and authority race guards", () => {
  it("never lets a deferred A refresh overwrite a fast B scope", async () => {
    const a = { orgId: "acme", projectId: "research" };
    const b = { orgId: "other", projectId: "research" };
    const session = authority("project_viewer", [
      effectiveScope("project_viewer", a),
      effectiveScope("project_viewer", b),
    ]);
    const staleA = deferred<WorkloadSummary[]>();
    let aReads = 0;
    const { client, calls } = fakeClient(session, {
      listWorkloads: async (_filters, scope) => {
        if (scope?.orgId === "acme") {
          aReads += 1;
          if (aReads > 1) return staleA.promise;
          return [scopedWorkload(a, "A current")];
        }
        return [scopedWorkload(b, "B current")];
      },
    });
    renderApp(client);
    await waitFor(() => expect(screen.getByText("A current")).toBeDefined());

    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() => expect(calls.listWorkloads).toHaveBeenCalledTimes(2));
    fireEvent.change(screen.getByRole("combobox", { name: "Project scope" }), {
      target: { value: JSON.stringify([b.orgId, b.projectId]) },
    });
    await waitFor(() => expect(screen.getByText("B current")).toBeDefined());

    await act(async () => staleA.resolve([scopedWorkload(a, "A stale")]));
    expect(screen.queryByText("A stale")).toBeNull();
    expect(screen.getByText("B current")).toBeDefined();
  });

  it("binds a pending policy mutation to immutable A, never same-named project B", async () => {
    const a = { orgId: "acme", projectId: "research" };
    const b = { orgId: "other", projectId: "research" };
    const session = authority("project_admin", [
      effectiveScope("project_admin", a),
      effectiveScope("project_admin", b),
    ]);
    const pending = deferred<SchedulingPolicy>();
    const updatePolicy = vi.fn(async (policy, update, scope) => {
      await pending.promise;
      const patch = typeof update === "boolean" ? { preemptionEnabled: update } : update;
      return {
        ...policy,
        id: `${scope?.orgId}/${scope?.projectId}`,
        preemptionEnabled: patch.preemptionEnabled ?? policy.preemptionEnabled,
        crossRegionTransferPolicy: patch.crossRegionTransferPolicy ?? policy.crossRegionTransferPolicy,
      };
    });
    const { client } = fakeClient(session, { updatePolicy });
    renderApp(client);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Operations overview" })).toBeDefined());
    await showPanel("Budgets & policies");

    fireEvent.click(screen.getAllByRole("checkbox")[0]);
    await waitFor(() => expect(updatePolicy).toHaveBeenCalled());
    expect(updatePolicy.mock.calls[0][2]).toEqual(expect.objectContaining(a));

    fireEvent.change(screen.getByRole("combobox", { name: "Project scope" }), {
      target: { value: JSON.stringify([b.orgId, b.projectId]) },
    });
    await waitFor(() => expect(screen.getByText(`Placement policy (${b.orgId}/${b.projectId})`)).toBeDefined());
    await act(async () => pending.resolve(policyFor(a)));
    await showPanel("Budgets & policies");

    expect(screen.getByText(`Placement policy (${b.orgId}/${b.projectId})`)).toBeDefined();
    expect(updatePolicy.mock.calls[0][2]).not.toEqual(expect.objectContaining(b));
  });

  it("refreshes durable additions and revocations and reconciles the selected scope", async () => {
    const listen = vi.spyOn(document, "addEventListener");
    const a = { orgId: "acme", projectId: "research" };
    const b = { orgId: "acme", projectId: "platform" };
    const sessions = [
      authority("project_operator", [effectiveScope("project_operator", a)]),
      authority("project_viewer", [effectiveScope("project_viewer", a), effectiveScope("project_operator", b)]),
      authority("project_operator", [effectiveScope("project_operator", b)]),
    ];
    let index = 0;
    const { client, calls } = fakeClient(sessions[0], {
      getSessionAuthority: async () => sessions[Math.min(index++, sessions.length - 1)],
    });
    renderApp(client);
    await waitFor(() => expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined());
    // Capability rendering precedes passive-effect subscription. Dispatch the
    // refresh only after its listener is installed, including on a busy host.
    await waitFor(() => expect(listen).toHaveBeenCalledWith("visibilitychange", expect.any(Function)));

    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull());
    expect(screen.getByRole("combobox", { name: "Project scope" }).querySelectorAll("option")).toHaveLength(2);

    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined());
    expect((screen.getByRole("combobox", { name: "Project scope" }) as HTMLSelectElement).value).toBe(
      JSON.stringify([b.orgId, b.projectId]),
    );
  });

  it("closes and resets an A submission form when reconciliation automatically selects B", async () => {
    const a = { orgId: "acme", projectId: "research" };
    const b = { orgId: "other", projectId: "platform" };
    const sessions = [
      authority("project_operator", [effectiveScope("project_operator", a), effectiveScope("project_operator", b)]),
      authority("project_operator", [effectiveScope("project_operator", b)]),
    ];
    let sessionCall = 0;
    const { client, calls } = fakeClient(sessions[0], {
      getSessionAuthority: async () => sessions[Math.min(sessionCall++, sessions.length - 1)],
    });
    renderApp(client);
    await waitFor(() => expect(screen.getByText("acme-research-workload")).toBeDefined());

    fireEvent.click(screen.getByRole("button", { name: "Submit workload" }));
    const aDialog = screen.getByRole("dialog", { name: "Run a job" });
    fireEvent.change(within(aDialog).getByLabelText("Name"), { target: { value: "A draft" } });
    expect((within(aDialog).getByLabelText("Project") as HTMLInputElement).value).toBe(a.projectId);

    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() =>
      expect((screen.getByRole("combobox", { name: "Project scope" }) as HTMLSelectElement).value).toBe(
        JSON.stringify([b.orgId, b.projectId]),
      ),
    );
    expect(screen.queryByRole("dialog", { name: "Run a job" })).toBeNull();
    expect(calls.submitWorkload).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Submit workload" }));
    const bDialog = screen.getByRole("dialog", { name: "Run a job" });
    expect((within(bDialog).getByLabelText("Name") as HTMLInputElement).value).toBe("");
    expect((within(bDialog).getByLabelText("Project") as HTMLInputElement).value).toBe(b.projectId);
    fireEvent.change(within(bDialog).getByLabelText("Name"), { target: { value: "B submission" } });
    // The command is required, and the shape preselects once the fleet read
    // lands -- which is a real wait, not a test artefact: the form cannot offer
    // hardware it has not been told about.
    fireEvent.change(within(bDialog).getByLabelText("Command"), { target: { value: "python train.py" } });
    await waitFor(() =>
      expect((within(bDialog).getByLabelText("Shape") as HTMLSelectElement).value).toBe("research/v6e-8"),
    );
    fireEvent.click(within(bDialog).getByRole("button", { name: "Run" }));

    await waitFor(() => expect(calls.submitWorkload).toHaveBeenCalledTimes(1));
    expect(calls.submitWorkload.mock.calls[0][0]).toEqual(expect.objectContaining({ name: "B submission" }));
    expect(calls.submitWorkload.mock.calls[0][1]).toEqual(b);
  });

  it("drops a pending A submission result without closing a newly opened B form", async () => {
    const a = { orgId: "acme", projectId: "research" };
    const b = { orgId: "other", projectId: "research" };
    const sessions = [
      authority("project_operator", [effectiveScope("project_operator", a), effectiveScope("project_operator", b)]),
      authority("project_operator", [effectiveScope("project_operator", b)]),
    ];
    const pending = deferred<WorkloadDetail>();
    let sessionCall = 0;
    const { client, calls } = fakeClient(sessions[0], {
      getSessionAuthority: async () => sessions[Math.min(sessionCall++, sessions.length - 1)],
      submitWorkload: async () => pending.promise,
    });
    renderApp(client);
    await waitFor(() => expect(screen.getByText("acme-research-workload")).toBeDefined());

    fireEvent.click(screen.getByRole("button", { name: "Submit workload" }));
    const aDialog = screen.getByRole("dialog", { name: "Run a job" });
    fireEvent.change(within(aDialog).getByLabelText("Name"), { target: { value: "A pending" } });
    fireEvent.change(within(aDialog).getByLabelText("Command"), { target: { value: "python train.py" } });
    await waitFor(() =>
      expect((within(aDialog).getByLabelText("Shape") as HTMLSelectElement).value).toBe("research/v6e-8"),
    );
    fireEvent.click(within(aDialog).getByRole("button", { name: "Run" }));
    await waitFor(() => expect(calls.submitWorkload).toHaveBeenCalledTimes(1));
    expect(calls.submitWorkload.mock.calls[0][1]).toEqual(a);

    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() =>
      expect((screen.getByRole("combobox", { name: "Project scope" }) as HTMLSelectElement).value).toBe(
        JSON.stringify([b.orgId, b.projectId]),
      ),
    );
    expect(screen.queryByRole("dialog", { name: "Run a job" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Submit workload" }));
    const bDialog = screen.getByRole("dialog", { name: "Run a job" });
    fireEvent.change(within(bDialog).getByLabelText("Name"), { target: { value: "B draft" } });

    await act(async () => pending.resolve(scopedWorkload(a, "A pending")));
    const currentDialog = screen.getByRole("dialog", { name: "Run a job" });
    expect((within(currentDialog).getByLabelText("Name") as HTMLInputElement).value).toBe("B draft");
    expect((within(currentDialog).getByLabelText("Project") as HTMLInputElement).value).toBe(b.projectId);
    expect(screen.queryByText(/A pending submitted to/)).toBeNull();
  });

  it("settles an unauthorized cross-project deep link after exactly one fresh session reconciliation", async () => {
    window.location.hash = "#/workloads/wl-outside-authority";
    const operator = authority("project_operator");
    const submitter = authority("project_submitter");
    const reconciliation = deferred<SessionAuthority>();
    let sessionCall = 0;
    const { client, calls } = fakeClient(operator, {
      getSessionAuthority: async () => {
        sessionCall += 1;
        return sessionCall === 1 ? operator : reconciliation.promise;
      },
      getWorkload: async () => {
        throw new ApiError("Workload belongs to a project you cannot access", 403);
      },
    });
    renderApp(client);

    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2));
    // The denial fences all stale presentation authority before /session settles.
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();

    await act(async () => reconciliation.resolve(structuredClone(submitter)));
    await waitFor(() => expect(screen.getByText("Workload belongs to a project you cannot access")).toBeDefined());
    await act(async () => {
      await new Promise((resolve) => globalThis.setTimeout(resolve, 50));
    });

    expect(screen.queryByText("Loading control-plane state")).toBeNull();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2);
    expect(calls.getWorkload).toHaveBeenCalledTimes(1);
  });

  it("keeps a typed detail denial inert across retained and fresh follower events", async () => {
    window.location.hash = "#/workloads/wl-retained-denial";
    const operator = authority("project_operator");
    const reconciliation = deferred<SessionAuthority>();
    const followerEvents: Array<Parameters<EyvanApiClient["followEvents"]>[0]> = [];
    let sessionCall = 0;
    let detailCall = 0;
    const followEvents = vi.fn(
      async (
        onEvent: Parameters<EyvanApiClient["followEvents"]>[0],
        signal: AbortSignal,
        onConnectionChange?: Parameters<EyvanApiClient["followEvents"]>[2],
      ) => {
        followerEvents.push(onEvent);
        onConnectionChange?.(true);
        // A new follower receives retained history before any fresh event.
        onEvent({ id: `retained-${followerEvents.length}`, event: "workload.submitted", data: {} });
        await new Promise<void>((resolve) => {
          if (signal.aborted) resolve();
          else signal.addEventListener("abort", () => resolve(), { once: true });
        });
      },
    );
    const { client, calls } = fakeClient(operator, {
      getSessionAuthority: async () => {
        sessionCall += 1;
        if (sessionCall === 2) return reconciliation.promise;
        return structuredClone(operator);
      },
      getWorkload: async () => {
        detailCall += 1;
        if (detailCall === 1) throw new ApiError("retained detail denial", 403);
        return { ...scopedWorkload({ orgId: "acme", projectId: "research" }), id: "wl-retained-denial" };
      },
      followEvents,
    });
    renderApp(client);

    await waitFor(() => {
      expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2);
      expect(calls.getWorkload).toHaveBeenCalledTimes(1);
      expect(followEvents).toHaveBeenCalledTimes(1);
    });
    // Keep reconciliation pending past the live-refresh coalescer. The retained
    // event from the now-fenced follower must not survive its abort.
    await act(async () => {
      await new Promise((resolve) => globalThis.setTimeout(resolve, COALESCE_MS + 50));
    });
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2);
    expect(calls.getWorkload).toHaveBeenCalledTimes(1);
    expect(followEvents).toHaveBeenCalledTimes(1);

    await act(async () => reconciliation.resolve(structuredClone(operator)));
    await waitFor(() => expect(screen.getByText("retained detail denial")).toBeDefined());
    await waitFor(() => expect(followEvents).toHaveBeenCalledTimes(2));
    // The replacement follower immediately replays retained history. It may
    // reconcile the dashboard once, but it must not replay the denied detail.
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(3), { timeout: 1_500 });
    await waitFor(() => expect(calls.listWorkloads).toHaveBeenCalledTimes(3));
    expect(calls.getWorkload).toHaveBeenCalledTimes(1);
    expect(followEvents).toHaveBeenCalledTimes(2);

    act(() => {
      followerEvents.at(-1)?.({ id: "fresh-1", event: "workload.updated", data: {} });
    });
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(4), { timeout: 1_500 });
    await waitFor(() => expect(calls.listWorkloads).toHaveBeenCalledTimes(4));
    await act(async () => {
      await new Promise((resolve) => globalThis.setTimeout(resolve, COALESCE_MS + 50));
    });

    expect(calls.getWorkload).toHaveBeenCalledTimes(1);
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(4);
    expect(followEvents).toHaveBeenCalledTimes(2);
    expect(screen.getByText("retained detail denial")).toBeDefined();

    // Retry is the sole operation that clears the preserved denial.
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(calls.getWorkload).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Workload detail" })).toBeDefined());
  });

  it("does not replay a denied detail from the 30-second fallback poll", async () => {
    vi.useFakeTimers();
    window.location.hash = "#/workloads/wl-poll-denial";
    const operator = authority("project_operator");
    const { client, calls } = fakeClient(operator, {
      getSessionAuthority: async () => structuredClone(operator),
      getWorkload: async () => {
        throw new ApiError("poll detail denial", 403);
      },
    });
    renderApp(client);

    await vi.waitFor(() => {
      expect(screen.getByText("poll detail denial")).toBeDefined();
      expect(calls.getWorkload).toHaveBeenCalledTimes(1);
      expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2);
      expect(calls.followEvents).toHaveBeenCalledTimes(1);
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    // The dashboard sweep still reconciles /session; the paired detail poll is
    // inert until the user explicitly retries the preserved denial.
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(3);
    expect(calls.getWorkload).toHaveBeenCalledTimes(1);
    expect(calls.followEvents).toHaveBeenCalledTimes(1);
    expect(screen.getByText("poll detail denial")).toBeDefined();
  });

  it("ignores a stale deep-link 403 after navigation without fencing the current scope", async () => {
    window.location.hash = "#/workloads/wl-stale-denial";
    const operator = authority("project_operator");
    const staleDetail = deferred<WorkloadDetail>();
    const { client, calls } = fakeClient(operator, {
      getWorkload: async () => staleDetail.promise,
    });
    renderApp(client);
    await waitFor(() => expect(calls.getWorkload).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "Workloads" }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Workloads" })).toBeDefined());
    await act(async () => staleDetail.reject(new ApiError("stale cross-project denial", 403)));

    expect(screen.queryByText("stale cross-project denial")).toBeNull();
    expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined();
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(1);
  });

  it("re-resolves /session after a 403 before rendering any stale capability", async () => {
    const operator = authority("project_operator");
    const viewer = authority("project_viewer");
    let sessionCall = 0;
    let workloadCall = 0;
    const { client, calls } = fakeClient(operator, {
      getSessionAuthority: async () => (sessionCall++ < 1 ? operator : viewer),
      listWorkloads: async (_filters, scope) => {
        workloadCall += 1;
        if (workloadCall === 2) throw new ApiError("revoked", 403);
        return [scopedWorkload(scope ?? { orgId: "acme", projectId: "research" })];
      },
    });
    renderApp(client);
    await waitFor(() => expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined());

    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull());
  });

  it("kicks a running attempt without ending the job", async () => {
    // The action a wedged host needs and the dashboard did not have: cancel
    // ends the job, waiting for a timeout keeps the accelerator busy doing
    // nothing. Only the CLI could do this, which is the wrong place for a
    // decision somebody makes while looking at a stuck run.
    window.location.hash = "#/workloads/wl-1";
    const operator = authority("project_operator");
    const { client, calls } = fakeClient(operator);
    renderApp(client);

    await waitFor(() => expect(screen.getByRole("button", { name: "Kick" })).toBeDefined());
    fireEvent.click(screen.getByRole("button", { name: "Kick" }));

    await waitFor(() => expect(calls.workloadAction).toHaveBeenCalledTimes(1));
    expect(calls.workloadAction.mock.calls[0]?.[1]).toBe("kick");
  });

  it("drops stale workload actions when a mutation 403 refreshes durable authority", async () => {
    window.location.hash = "#/workloads/wl-revoked";
    const operator = authority("project_operator");
    const viewer = authority("project_viewer");
    let current = operator;
    const { client, calls } = fakeClient(operator, {
      getSessionAuthority: async () => current,
      workloadAction: async () => {
        current = viewer;
        throw new ApiError("operate permission revoked", 403);
      },
    });
    renderApp(client);

    await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeDefined());
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(calls.workloadAction).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull());
    expect(calls.getSessionAuthority.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it.each(["approve", "stop"] as const)(
    "re-resolves authority when the plane refuses to %s a research plan",
    async (action) => {
      window.location.hash = "#/research/rp_0123456789abcdef";
      const operator = authority("project_operator");
      const { client, calls } = fakeClient(operator);
      const plan = {
        id: "rp_0123456789abcdef",
        title: "lr sweep",
        subject: "alice",
        org_id: "acme",
        project_id: "research",
        state: action === "approve" ? "proposed" : "running",
        iterate: false,
        provider_id: null,
        model: null,
        current_round: 0,
        jobs_planned: 1,
        budget: { max_jobs: 1, max_chip_hours: 1, max_wall_seconds: 600, max_rounds: null },
        spent: { chip_hours: 0, jobs_submitted: 0, rounds: 0 },
        stop_reason: null,
        approved_at: null,
        approved_by: null,
        started_at: null,
        finished_at: null,
        created_at: "2026-10-28T10:00:00+00:00",
        updated_at: "2026-10-28T10:00:00+00:00",
        goal: "",
        jobs: [],
        workload_ids: [],
        remaining: { jobs: 1, chip_hours: 1 },
        notes: [],
      };
      const refused = vi.fn(async () => {
        throw new ApiError("research permission revoked", 403);
      });
      Object.assign(client, {
        listResearchPlans: vi.fn(async () => []),
        getResearchPlan: vi.fn(async () => plan),
        approveResearchPlan: action === "approve" ? refused : vi.fn(),
        stopResearchPlan: action === "stop" ? refused : vi.fn(),
      });
      renderApp(client);

      const label = action === "approve" ? "Approve" : "Stop";
      fireEvent.click(await screen.findByRole("button", { name: label }));
      const before = calls.getSessionAuthority.mock.calls.length;
      fireEvent.click(screen.getByRole("button", { name: `Confirm ${action}` }));
      await waitFor(() => expect(refused).toHaveBeenCalledTimes(1));
      await waitFor(() => expect(calls.getSessionAuthority.mock.calls.length).toBeGreaterThan(before));
    },
  );

  it("keeps one scoped SSE subscription when refresh decodes fresh scope objects", async () => {
    const session = authority("project_viewer");
    const followEvents = vi.fn(
      async (
        onEvent: Parameters<EyvanApiClient["followEvents"]>[0],
        signal: AbortSignal,
        onConnectionChange?: Parameters<EyvanApiClient["followEvents"]>[2],
      ) => {
        onConnectionChange?.(true);
        // The endpoint replays retained history when no Last-Event-ID is sent.
        // Re-subscribing after this refresh would replay it again and create a
        // self-sustaining refresh/reconnect loop.
        onEvent({ id: "1", event: "workload.submitted", data: { sequence: 1 } });
        await new Promise<void>((resolve) => {
          if (signal.aborted) resolve();
          else signal.addEventListener("abort", () => resolve(), { once: true });
        });
      },
    );
    const { client, calls } = fakeClient(session, {
      // Match production decoding: every /session response materializes new
      // authority and scope objects even when their values are unchanged.
      getSessionAuthority: async () => structuredClone(session),
      followEvents,
    });
    renderApp(client);

    await waitFor(() => expect(followEvents).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2), { timeout: 1_500 });
    await act(async () => {
      await new Promise((resolve) => globalThis.setTimeout(resolve, 550));
    });

    expect(followEvents).toHaveBeenCalledTimes(1);
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2);
  });

  it("fences stale controls on a dashboard 403 before failed reconciliation completes", async () => {
    const operator = authority("project_operator");
    const reconciliation = deferred<SessionAuthority>();
    let sessionCall = 0;
    let workloadCall = 0;
    const { client, calls } = fakeClient(operator, {
      getSessionAuthority: async () => {
        sessionCall += 1;
        if (sessionCall < 3) return structuredClone(operator);
        return reconciliation.promise;
      },
      listWorkloads: async (_filters, scope) => {
        workloadCall += 1;
        if (workloadCall === 2) throw new ApiError("authority revoked", 403);
        return [scopedWorkload(scope ?? { orgId: "acme", projectId: "research" })];
      },
    });
    renderApp(client);
    await waitFor(() => expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined());
    fireEvent.click(screen.getByRole("button", { name: "Submit workload" }));
    expect(screen.getByRole("dialog", { name: "Run a job" })).toBeDefined();

    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(3));

    // /session is still pending: the first typed denial, not its eventual
    // reconciliation result, must remove every stale mutation affordance.
    expect(screen.queryByRole("dialog", { name: "Run a job" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();

    await act(async () => reconciliation.reject(new ApiError("session unavailable", 503)));
    await waitFor(() => expect(screen.getByText("session unavailable")).toBeDefined());
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();
  });

  it("fences detail actions and an open modal when SSE denial reconciliation fails", async () => {
    window.location.hash = "#/workloads/wl-stream-revoked";
    const operator = authority("project_operator");
    const streamDenied = deferred<void>();
    const reconciliation = deferred<SessionAuthority>();
    let sessionCall = 0;
    const followEvents = vi.fn(async () => {
      await streamDenied.promise;
      throw new ApiError("event stream denied", 403);
    });
    const { client, calls } = fakeClient(operator, {
      getSessionAuthority: async () => {
        sessionCall += 1;
        if (sessionCall < 3) return structuredClone(operator);
        return reconciliation.promise;
      },
      followEvents,
    });
    renderApp(client);

    await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeDefined());
    fireEvent.click(screen.getByRole("button", { name: "Submit workload" }));
    expect(screen.getByRole("dialog", { name: "Run a job" })).toBeDefined();

    await act(async () => streamDenied.resolve());
    await waitFor(() => expect(calls.getSessionAuthority).toHaveBeenCalledTimes(3));
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(screen.queryByRole("dialog", { name: "Run a job" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();

    await act(async () => reconciliation.reject(new ApiError("session reconciliation failed", 503)));
    await waitFor(() => expect(screen.getByText("session reconciliation failed")).toBeDefined());
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();
  });

  it("retries bootstrap by resolving /session again", async () => {
    const session = authority("project_viewer");
    let attempts = 0;
    const { client, calls } = fakeClient(session, {
      getSessionAuthority: async () => {
        attempts += 1;
        if (attempts === 1) throw new ApiError("session unavailable", 503);
        return session;
      },
    });
    renderApp(client);

    await waitFor(() => expect(screen.getByText("session unavailable")).toBeDefined());
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Operations overview" })).toBeDefined());
    expect(calls.getSessionAuthority).toHaveBeenCalledTimes(2);
  });
});

describe("submit-only and wildcard deep links", () => {
  it("shows a durable accessible submit-only acknowledgement and issues no detail or poll reads", async () => {
    const session = authority("project_submitter");
    const { client, calls } = fakeClient(session);
    renderApp(client);
    await waitFor(() => expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined());

    fireEvent.click(screen.getByRole("button", { name: "Submit workload" }));
    const dialog = screen.getByRole("dialog", { name: "Run a job" });
    fireEvent.change(within(dialog).getByLabelText("Name"), { target: { value: "submit-only-job" } });
    fireEvent.change(within(dialog).getByLabelText("Command"), { target: { value: "python train.py" } });
    await waitFor(() =>
      expect((within(dialog).getByLabelText("Shape") as HTMLSelectElement).value).toBe("research/v6e-8"),
    );
    fireEvent.click(within(dialog).getByRole("button", { name: "Run" }));

    await waitFor(() => expect(screen.getByText("Workload submitted")).toBeDefined());
    expect(screen.getByText(/submit-only-job.*cannot read workload detail/)).toBeDefined();
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await waitFor(() => expect(calls.getSessionAuthority.mock.calls.length).toBeGreaterThan(1));

    expect(calls.listWorkloads).not.toHaveBeenCalled();
    expect(calls.listLeases).not.toHaveBeenCalled();
    expect(calls.getQueueTrend).not.toHaveBeenCalled();
    expect(calls.getWorkload).not.toHaveBeenCalled();
    expect(calls.explainWorkload).not.toHaveBeenCalled();
  });

  it("uses an authenticated scoped session query for an org-admin deep link into legacy acme/other", async () => {
    window.location.hash = "#/workloads/wl-other";
    const research = effectiveScope("org_admin", { orgId: "acme", projectId: "research" }, { registered: false });
    const other = effectiveScope("org_admin", { orgId: "acme", projectId: "other" }, { registered: false });
    const initial = authority("org_admin", [research]);
    const scoped = authority("org_admin", [research, other]);
    const { client, calls } = fakeClient(initial, {
      getSessionAuthority: async (preferred) => (preferred?.projectId === "other" ? scoped : initial),
      getWorkload: async () => ({
        ...scopedWorkload({ orgId: "acme", projectId: "other" }, "legacy other"),
        id: "wl-other",
      }),
    });
    renderApp(client);

    await waitFor(() => expect(screen.getByText("legacy other")).toBeDefined());
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDefined();
    expect(calls.getSessionAuthority).toHaveBeenCalledWith(
      { orgId: "acme", projectId: "other" },
      expect.any(AbortSignal),
    );
  });
});
