// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { cellStatus, computeOverview, EyvanApiClient, workloadStatus } from "./client";
import type { LogEntry } from "../types";
import { configuredScope, consumeSessionNotice } from "../auth/oidc";

/** The scope every unqualified request carries. Read, not restated: the
 *  literal here used to be "acme"/"research" while the CLI defaulted to
 *  "default"/"default", and the dashboard queried a project nobody used. */
const SCOPE = configuredScope();

describe("EyvanApiClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("reports an unreachable live API instead of substituting demo data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    const client = new EyvanApiClient("/api/v1", false);

    await expect(client.listWorkloads()).rejects.toEqual(
      expect.objectContaining({ name: "ApiError", message: "Unable to reach Eyvan API: offline" }),
    );
  });

  it("reports server failures instead of substituting demo data", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() =>
        Promise.resolve(
          new Response(JSON.stringify({ detail: "database unavailable" }), {
            status: 503,
            headers: { "content-type": "application/json" },
          }),
        ),
      ),
    );
    const client = new EyvanApiClient("/api/v1", false);

    await expect(client.listCells()).rejects.toEqual(
      expect.objectContaining({
        name: "ApiError",
        status: 503,
        message: "Eyvan API request failed with 503: database unavailable",
      }),
    );
  });

  it("uses sample data only when demo mode is explicit", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", true);

    const workloads = await client.listWorkloads();

    expect(workloads.length).toBeGreaterThan(0);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("maps the versioned workload contract without granting default development identity", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([rawWorkload]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const workloads = await client.listWorkloads();

    expect(workloads[0]).toMatchObject({ id: "wl-1", status: "running", priority: "production" });
    expect(fetch.mock.calls[0][0]).toContain(`org_id=${SCOPE.orgId}`);
    expect(fetch.mock.calls[0][1].headers).not.toHaveProperty("X-Eyvan-User");
    expect(fetch.mock.calls[0][1].headers).not.toHaveProperty("Authorization");
  });

  it("maps server-resolved authority and switches every scoped request explicitly", async () => {
    const fetch = vi.fn().mockImplementation((url: string) => {
      if (String(url).includes("/session")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              subject: "alice",
              display_name: "Alice",
              global_permissions: [],
              scopes: [
                {
                  org_id: "acme",
                  project_id: "research",
                  display_name: "Research",
                  registered: true,
                  enabled: true,
                  accepts_new_work: true,
                  roles: ["project_viewer"],
                  permissions: ["read"],
                },
                {
                  org_id: "other",
                  project_id: "training",
                  display_name: "Training",
                  registered: true,
                  enabled: true,
                  accepts_new_work: true,
                  roles: ["project_submitter"],
                  permissions: ["submit"],
                },
              ],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      return Promise.resolve(
        new Response(JSON.stringify([]), { status: 200, headers: { "content-type": "application/json" } }),
      );
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const authority = await client.getSessionAuthority();
    client.setScope(authority.scopes[1]);
    await client.listWorkloads();

    expect(authority.scopes[1]).toMatchObject({
      orgId: "other",
      projectId: "training",
      roles: ["project_submitter"],
      permissions: ["submit"],
    });
    expect(fetch.mock.calls[0][0]).toContain(`org_id=${SCOPE.orgId}`);
    expect(fetch.mock.calls[1][0]).toContain("org_id=other&project_id=training");
  });

  it("reads project audit through the scoped route without probing fleet authority", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            id: "event-1",
            sequence: 7,
            timestamp: "2026-01-01T00:00:00Z",
            actor: "alice",
            entity_type: "workload",
            entity_id: "wl-1",
            action: "workload.submitted",
            payload: {},
          },
        ]),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    await client.listAuditEvents({ orgId: "other", projectId: "research" });

    expect(fetch.mock.calls[0][0]).toBe("/api/v1/events?limit=1000&org_id=other&project_id=research");
    expect(fetch.mock.calls.some(([url]) => String(url).includes("/fleet/events"))).toBe(false);
  });

  it("uses the workload's own scope for deep-link event and lease composition", async () => {
    const urls: string[] = [];
    const fetch = vi.fn().mockImplementation((url: string) => {
      urls.push(String(url));
      if (String(url).endsWith("/workloads/wl-1")) {
        return Promise.resolve(
          new Response(JSON.stringify({ ...rawWorkload, org_id: "other", project_id: "training" }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      if (String(url).endsWith("/logs")) return Promise.resolve(new Response("", { status: 200 }));
      if (String(url).endsWith("/progress")) {
        return Promise.resolve(new Response("missing", { status: 404 }));
      }
      return Promise.resolve(
        new Response(JSON.stringify([]), { status: 200, headers: { "content-type": "application/json" } }),
      );
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const detail = await client.getWorkload("wl-1");

    expect(detail).toMatchObject({ org: "other", project: "training" });
    const composed = urls.filter((url) => url.includes("/events?") || url.includes("/leases?"));
    expect(composed).toHaveLength(2);
    expect(composed.every((url) => url.includes("org_id=other&project_id=training"))).toBe(true);
  });

  it("does not fabricate per-workload budget or progress figures", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([rawWorkload]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const workloads = await client.listWorkloads();

    expect(workloads[0].budgetSpent).toBeUndefined();
    expect(workloads[0].budgetLimit).toBeUndefined();
    expect(workloads[0].progress).toBeUndefined();
  });

  it("keeps a reclaimed attempt distinct from a capacity failure", async () => {
    // The two counts are the plane's answer to "why is this on attempt 3", and
    // they mean opposite things: a spot job reclaimed twice is behaving as
    // designed, while one that lost capacity twice is running out of pool. The
    // client used to declare `attempt` on the wire and then drop it along with
    // both counts, so the operations UI could show neither.
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([{ ...rawWorkload, attempt: 3, preemption_count: 2, capacity_loss_count: 0 }]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);

    const workloads = await new EyvanApiClient("/api/v1", false).listWorkloads();

    expect(workloads[0]).toMatchObject({ attempt: 3, preemptionCount: 2, capacityLossCount: 0 });
  });

  it("reports an older plane's missing counts as unknown rather than as zero", async () => {
    // A plane that predates the budget split sends neither field. Defaulting
    // them to 0 would render "reclaimed 0 times" for a job that may have been
    // reclaimed repeatedly -- a confident wrong answer in place of no answer.
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([rawWorkload]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);

    const workloads = await new EyvanApiClient("/api/v1", false).listWorkloads();

    expect(workloads[0].preemptionCount).toBeUndefined();
    expect(workloads[0].capacityLossCount).toBeUndefined();
  });

  it("sends development identity headers only when explicitly enabled", async () => {
    vi.stubEnv("VITE_EYVAN_DEV_AUTH", "true");
    vi.stubEnv("VITE_EYVAN_DEV_USER", "local-operator");
    vi.stubEnv("VITE_EYVAN_DEV_ROLE", "operator");
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([rawWorkload]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);

    await new EyvanApiClient("/api/v1", false).listWorkloads();

    expect(fetch.mock.calls[0][1].headers).toMatchObject({
      "X-Eyvan-User": "local-operator",
      "X-Eyvan-Roles": "operator",
    });
  });

  it("updates preemption through the versioned policy API without discarding placement constraints", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ...rawPolicy, preemption_enabled: false }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);
    const policy = {
      id: "acme/research",
      name: "Checkpoint-aware preemption",
      scope: "research",
      type: "preemption" as const,
      summary: "us-central2; spot",
      preemptionEnabled: true,
      crossRegionTransferPolicy: "allow" as const,
      crossRegionAllowedBuckets: [],
      crossRegionAllowMultiregion: true,
      updatedAt: rawPolicy.updated_at,
      configuration: {
        allowedProviders: ["gcp"],
        allowedRegions: ["us-central2"],
        allowedCapacityClasses: ["spot"],
        deniedAcceleratorTypes: ["v4"],
        maxPriority: 500,
      },
    };

    await expect(client.updatePolicy(policy, false)).resolves.toMatchObject({ preemptionEnabled: false });

    expect(fetch.mock.calls[0][0]).toBe(`/api/v1/policies/${SCOPE.orgId}/${SCOPE.projectId}`);
    expect(fetch.mock.calls[0][1].method).toBe("PUT");
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toMatchObject({
      allowed_regions: ["us-central2"],
      denied_accelerator_types: ["v4"],
      preemption_enabled: false,
    });
  });

  it("uses both components of the immutable selected scope for a same-project cross-org update", async () => {
    const fetch = vi.fn().mockResolvedValue(
      // Echoes BOTH components of the requested scope. The client refuses a
      // mutation answered for a different one, which is the guard this test
      // drives the request through.
      new Response(JSON.stringify({ ...rawPolicy, org_id: "other", project_id: "research" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);
    const policy = {
      id: "other/research",
      name: "Placement policy",
      scope: "research",
      type: "placement" as const,
      summary: "scoped",
      preemptionEnabled: true,
      crossRegionTransferPolicy: "allow" as const,
      crossRegionAllowedBuckets: [],
      crossRegionAllowMultiregion: true,
      updatedAt: rawPolicy.updated_at,
      configuration: {
        allowedProviders: ["gcp"],
        allowedRegions: [],
        allowedCapacityClasses: [],
        deniedAcceleratorTypes: [],
        maxPriority: 1000,
      },
    };

    await client.updatePolicy(policy, false, { orgId: "other", projectId: "research" });

    expect(fetch.mock.calls[0][0]).toBe("/api/v1/policies/other/research");
  });

  it("maps policies from server fields instead of hardcoding preemption", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(rawPolicy), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const policies = await client.listPolicies();

    expect(policies[0]).toMatchObject({
      id: `${SCOPE.orgId}/${SCOPE.projectId}`,
      type: "placement",
      preemptionEnabled: true,
      summary: expect.stringContaining("us-central2"),
    });
    expect(policies[0].name).not.toBe("Checkpoint-aware preemption");
  });

  it("reads budget and policy through project-scoped routes", async () => {
    const fetch = vi.fn().mockImplementation((url: string) => {
      const response = (body: unknown) =>
        new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
      if (url === `/api/v1/budgets/${SCOPE.orgId}/${SCOPE.projectId}`) return Promise.resolve(response(rawBudget));
      if (url === `/api/v1/budgets/${SCOPE.orgId}/${SCOPE.projectId}/status`) {
        return Promise.resolve(
          response({ org_id: SCOPE.orgId, project_id: SCOPE.projectId, limit: 1000, spent: 100, forecast: 250 }),
        );
      }
      if (url === `/api/v1/policies/${SCOPE.orgId}/${SCOPE.projectId}`) return Promise.resolve(response(rawPolicy));
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const [budgets, policies] = await Promise.all([client.listBudgets(), client.listPolicies()]);

    expect(budgets[0]).toMatchObject({ owner: SCOPE.projectId, forecast: 250 });
    expect(policies[0]).toMatchObject({ scope: SCOPE.projectId });
    // The point of the test: scoped routes, never a query-string list.
    expect(fetch.mock.calls.map(([url]) => url)).not.toContain(`/api/v1/budgets?org_id=${SCOPE.orgId}`);
    expect(fetch.mock.calls.map(([url]) => url)).not.toContain(`/api/v1/policies?org_id=${SCOPE.orgId}`);
  });

  it("fetches each resource once when computing the overview from shared data", async () => {
    const fetch = vi.fn().mockImplementation((url: string) => {
      const response = (body: unknown, status = 200) =>
        new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
      if (url === "/api/v1/workloads") return Promise.resolve(response([rawWorkload]));
      if (url === "/api/v1/cells") return Promise.resolve(response([rawCell]));
      if (url === `/api/v1/budgets/${SCOPE.orgId}/${SCOPE.projectId}`) return Promise.resolve(response(rawBudget));
      if (url === `/api/v1/leases?org_id=${SCOPE.orgId}&project_id=${SCOPE.projectId}`)
        return Promise.resolve(response([rawLease]));
      return Promise.resolve(response([]));
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const [workloads, leases, budgets] = await Promise.all([
      client.listWorkloads(),
      client.listLeases(),
      client.listBudgets(),
    ]);
    const cells = await client.listCells(leases);
    const overview = computeOverview(workloads, cells, budgets);

    const leaseCalls = fetch.mock.calls.filter(([path]) => String(path).startsWith("/api/v1/leases"));
    expect(leaseCalls).toHaveLength(1);
    expect(overview.queued).toBe(0);
    expect(cells[0].queuedWorkloads).toBeUndefined();
    expect(leases[0].class).toBeUndefined();
  });

  it("maps durable cell topology instead of substituting the accelerator type", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify([rawCell]), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    const client = new EyvanApiClient("/api/v1", false);

    const cells = await client.listCells([]);

    expect(cells[0].topology).toBe("fabric=ici, slice=4x4");
    expect(cells[0].topology).not.toContain(rawCell.accelerator_type);
  });

  it("fetches and maps workload progress on the detail page", async () => {
    const progress = {
      workload_id: "wl-1",
      workload_state: "running",
      phase: "running",
      request: { cells: 4, workers_per_cell: 2 },
      capacity: { ready_workers: 5 },
    };
    const fetch = vi.fn().mockImplementation((url: string) => {
      const response = (body: unknown, status = 200) =>
        new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
      if (url === "/api/v1/workloads/wl-1") return Promise.resolve(response(rawWorkload));
      if (url === "/api/v1/workloads/wl-1/progress") return Promise.resolve(response(progress));
      if (url.startsWith("/api/v1/events")) return Promise.resolve(response([]));
      return Promise.resolve(new Response("", { status: 200 }));
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const detail = await client.getWorkload("wl-1");

    expect(detail.progress).toBe(63); // round(5 / 8 * 100) = 63
  });

  it("keeps the detail page usable when the progress endpoint is absent", async () => {
    const fetch = vi.fn().mockImplementation((url: string) => {
      const response = (body: unknown, status = 200) =>
        new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
      if (url === "/api/v1/workloads/wl-1") return Promise.resolve(response(rawWorkload));
      if (url === "/api/v1/workloads/wl-1/progress") return Promise.resolve(new Response("Not Found", { status: 404 }));
      if (url.startsWith("/api/v1/events")) return Promise.resolve(response([]));
      return Promise.resolve(new Response("", { status: 200 }));
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    await expect(client.getWorkload("wl-1")).resolves.toMatchObject({ id: "wl-1" });
  });

  it("parses timestamps, levels, and sources from log lines when present", async () => {
    const fetch = vi.fn().mockImplementation((url: string) => {
      const response = (body: unknown, status = 200) =>
        new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
      if (url === "/api/v1/workloads/wl-1") return Promise.resolve(response(rawWorkload));
      if (url.startsWith("/api/v1/events")) return Promise.resolve(response([]));
      if (url === "/api/v1/workloads/wl-1/logs") {
        return Promise.resolve(
          new Response(
            [
              "2026-01-01T00:00:10.123Z INFO [trainer] step=10 loss=2.1",
              "2026-01-01T00:00:11Z WARN shard latency high",
              "plain executor line without metadata",
            ].join("\n"),
            { status: 200 },
          ),
        );
      }
      return Promise.resolve(new Response("", { status: 200 }));
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const detail = await client.getWorkload("wl-1");

    expect(detail.logs[0]).toMatchObject({
      timestamp: "2026-01-01T00:00:10.123Z",
      level: "INFO",
      source: "trainer",
      message: "step=10 loss=2.1",
    });
    expect(detail.logs[1]).toMatchObject({ timestamp: "2026-01-01T00:00:11Z", level: "WARN", source: undefined });
    expect(detail.logs[2]).toMatchObject({
      timestamp: undefined,
      level: undefined,
      source: undefined,
      message: "plain executor line without metadata",
    });
  });

  it("sets a re-login notice when the control plane rejects the session", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "unauthorized" }), {
          status: 401,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    const client = new EyvanApiClient("/api/v1", false);

    await expect(client.listWorkloads()).rejects.toEqual(expect.objectContaining({ status: 401 }));

    expect(consumeSessionNotice()).toMatch(/rejected/i);
  });

  it("registers demo workloads so detail lookups and actions resolve", async () => {
    const client = new EyvanApiClient("/api/v1", true);

    const workload = await client.submitWorkload({
      fleet: "research",
      shape: "v6e-8",
      pods: 1,
      name: "demo-run",
      priority: "batch",
      command: "python demo.py",
      estimatedDurationSeconds: 3600,
      checkpointable: false,
      environment: {},
      labels: {},
    });

    await expect(client.getWorkload(workload.id)).resolves.toMatchObject({ id: workload.id, name: "demo-run" });
    await expect(client.workloadAction(workload.id, "cancel")).resolves.toMatchObject({ status: "preempting" });
  });
  it("populates the allocation panel from the workload's own leases", async () => {
    // This used to be hardcoded to [], so the panel rendered nothing against a
    // live server while the mock data filled it in: the page looked implemented
    // and was not.
    const routes: Record<string, unknown> = {
      "/workloads/wl-1": rawWorkload,
      "/events": [],
      "/workloads/wl-1/progress": undefined,
      "/leases": [rawLease],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        if (url.includes("/logs")) return Promise.resolve(new Response("", { status: 200 }));
        // Longest first: "/workloads/wl-1" would otherwise shadow
        // "/workloads/wl-1/progress".
        const key = Object.keys(routes)
          .sort((a, b) => b.length - a.length)
          .find((route) => url.includes(route));
        const body = key ? routes[key] : undefined;
        return Promise.resolve(
          new Response(JSON.stringify(body ?? null), {
            status: body === undefined ? 404 : 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }),
    );

    const detail = await new EyvanApiClient("/api/v1", false).getWorkload("wl-1");

    expect(detail.allocation).toEqual([{ host: "cell-1", rack: "active", resource: "8 TPU, 8 CPU", state: "ready" }]);
  });
});

const rawWorkload = {
  id: "wl-1",
  org_id: "acme",
  project_id: "research",
  state: "running",
  attempt: 1,
  created_by: "alice",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:01:00Z",
  spec: {
    name: "train",
    execution: { kind: "command", command: "python train.py" },
    gang: {
      replicas: 1,
      per_replica: { cpu: 8, memory_gib: 64, gpu: 0, tpu: 8, custom: {} },
      required_topology: { slice: "single-slice" },
    },
    metadata: {},
    accelerator_type: "v5p-8",
    priority_band: "production",
  },
};

const rawPolicy = {
  org_id: SCOPE.orgId,
  project_id: SCOPE.projectId,
  allowed_providers: ["gcp"],
  allowed_regions: ["us-central2"],
  allowed_capacity_classes: ["spot"],
  denied_accelerator_types: ["v4"],
  max_priority: 500,
  preemption_enabled: true,
  updated_at: "2026-01-01T00:00:00Z",
};

const rawCell = {
  id: "cell-1",
  provider: "gcp",
  provider_project: "demo",
  region: "us-central",
  zone: "usc1-a",
  state: "ready",
  capacity: { cpu: 100, memory_gib: 1024, gpu: 0, tpu: 16, custom: {} },
  allocatable: { cpu: 80, memory_gib: 800, gpu: 0, tpu: 12, custom: {} },
  accelerator_type: "v5p-8",
  labels: {},
  topology: { slice: "4x4", fabric: "ici" },
};

const rawBudget = {
  org_id: SCOPE.orgId,
  project_id: SCOPE.projectId,
  spent: 100,
  limit: 1000,
  period: "monthly",
  hard_limit: false,
  accelerator_hour_limits: {},
  accelerator_hours_spent: {},
  currency: "USD",
};

const rawLease = {
  id: "lse-1",
  workload_id: "wl-1",
  project_id: "research",
  cell_id: "cell-1",
  state: "active",
  resources: { replicas: 1, per_replica: { cpu: 8, memory_gib: 64, gpu: 0, tpu: 8, custom: {} } },
  hourly_cost: 12,
  expires_at: "2026-01-02T00:00:00Z",
};

describe("dead field revival", () => {
  it("populates queueTrend from the ledger-derived route instead of hard-coding []", async () => {
    const fetch = vi.fn().mockImplementation((url: string) => {
      if (String(url).includes("/capacity/queue-trend")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              observed_at: "2026-08-01T00:00:00Z",
              bucket_seconds: 3600,
              points: [
                { started_at: "2026-07-31T22:00:00Z", submitted: 4, admitted: 2 },
                { started_at: "2026-07-31T23:00:00Z", submitted: 1, admitted: 5 },
              ],
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      return Promise.resolve(
        new Response(JSON.stringify([]), { status: 200, headers: { "content-type": "application/json" } }),
      );
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const trend = await client.getQueueTrend();

    expect(trend).toEqual([
      { label: "22:00", submitted: 4, admitted: 2 },
      { label: "23:00", submitted: 1, admitted: 5 },
    ]);
    expect(fetch.mock.calls[0][0]).toContain("window_s=86400&buckets=7");
  });

  it("keeps a failed trend request fatal instead of presenting empty history", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response("nope", { status: 500 }));
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    await expect(client.getQueueTrend()).rejects.toMatchObject({ status: 500 });
  });

  it("joins a lease's capacity class from the cell it landed on", async () => {
    const fetch = vi.fn().mockImplementation((url: string) => {
      const body = String(url).includes("/cells")
        ? [
            {
              id: "cell-a",
              provider: "gcp",
              region: "us-east5",
              state: "ready",
              capacity: { cpu: 8, memory_gib: 32, gpu: 0, tpu: 8, custom: {} },
              allocatable: { cpu: 8, memory_gib: 32, gpu: 0, tpu: 8, custom: {} },
              capacity_class: "spot",
              labels: {},
            },
          ]
        : [
            {
              id: "lease-1",
              workload_id: "wl-1",
              project_id: "research",
              cell_id: "cell-a",
              state: "active",
              resources: { replicas: 1, per_replica: { cpu: 1, memory_gib: 1, gpu: 0, tpu: 8, custom: {} } },
              expires_at: "2026-08-02T00:00:00Z",
              hourly_cost: 12,
            },
          ];
      return Promise.resolve(
        new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } }),
      );
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const leases = await client.listLeases();

    expect(leases[0].class).toBe("spot");
  });

  it("does not recurse between listLeases and listCells", async () => {
    // listCells resolves leases to compute utilization, so a naive join in
    // listLeases would loop forever. Bounded call count is the assertion.
    const fetch = vi
      .fn()
      .mockImplementation(() =>
        Promise.resolve(
          new Response(JSON.stringify([]), { status: 200, headers: { "content-type": "application/json" } }),
        ),
      );
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    await client.listLeases();

    expect(fetch.mock.calls.length).toBeLessThanOrEqual(2);
  });
});

describe("workload status mapping", () => {
  it("keeps a cancellation distinct from a failure", () => {
    // Seen live on 2026-08-01: four workloads the operator cancelled were
    // rendered "Failed", and the Failed tile read 5 when one job had failed.
    // "Your job broke" and "you stopped it" are different facts, and the tile
    // is the first thing read on the page.
    expect(workloadStatus("cancelled")).toBe("cancelled");
    expect(workloadStatus("failed")).toBe("failed");
  });

  it("still reports an unknown state as queued rather than inventing one", () => {
    expect(workloadStatus("some-future-state")).toBe("queued");
  });
});

describe("cell status mapping", () => {
  it("does not call a cell waiting for capacity degraded", () => {
    // On a Spot TPU pool `registering` is where a cell spends most of its life:
    // the queued resource exists and Google has not granted it yet. Rendering
    // that in red made a healthy fleet look broken, which is the fastest way to
    // make an operator stop trusting the page.
    expect(cellStatus("registering")).toBe("starting");
    expect(cellStatus("ready")).toBe("healthy");
  });

  it("keeps a genuinely broken cell degraded", () => {
    expect(cellStatus("unavailable")).toBe("degraded");
  });

  it("treats teardown as maintenance rather than failure", () => {
    expect(cellStatus("draining")).toBe("maintenance");
    expect(cellStatus("deleting")).toBe("maintenance");
  });
});

describe("the panels the dashboard never had", () => {
  function jsonFetch(payload: unknown) {
    return vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify(payload), { status: 200 })));
  }

  it("asks the explain route for the per-cell detail behind a flat reason", async () => {
    // The dashboard showed `scheduling_reason` alone -- one sentence. The
    // scheduler's actual verdict, per cell, existed and no page requested it.
    const fetchMock = jsonFetch({
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
    vi.stubGlobal("fetch", fetchMock);

    const explanation = await new EyvanApiClient("/api/v1", false).explainWorkload("wl-1");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/workloads/wl-1/explain");
    expect(explanation.cells).toHaveLength(1);
    // The field that distinguishes "waiting for capacity" from "can never be
    // placed" -- both of which report the same code.
    expect(explanation.provisioning_cell_ids).toEqual(["cell-b"]);
    expect(explanation.considered_cell_count).toBe(7);
  });

  it("escapes a workload id rather than pasting it into a path", async () => {
    // An id is server-generated today, but a path built by concatenation is a
    // request-forgery waiting for the day one is not.
    const fetchMock = jsonFetch({ workload_id: "a/b", state: "pending", placeable: false, cells: [] });
    vi.stubGlobal("fetch", fetchMock);

    await new EyvanApiClient("/api/v1", false).explainWorkload("a/b");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/workloads/a%2Fb/explain");
  });

  it("asks for live process detail, which answers 'running and nothing is happening'", async () => {
    const fetchMock = jsonFetch({ remote_id: "unit-1", control_group: "/eyvan.slice/x", pids: [42] });
    vi.stubGlobal("fetch", fetchMock);

    const status = await new EyvanApiClient("/api/v1", false).workloadProcessStatus("wl-1");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/workloads/wl-1/process-status");
    expect(status.pids).toEqual([42]);
  });

  it("reads warmth so idle cost can be told apart from waste", async () => {
    // Warm capacity is deliberately idle. Without this the only safe reaction
    // to an unexplained idle figure is to delete the warm floor.
    const fetchMock = jsonFetch({
      observed_at: "2026-07-29T12:00:00Z",
      pools: [
        { pool_id: "research/v5p", floor: 2, window_active: true, warm_cells: 2, warm_idle_cells: 1, hourly_cost: 8 },
      ],
    });
    vi.stubGlobal("fetch", fetchMock);

    const warmth = await new EyvanApiClient("/api/v1", false).getWarmth();

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/capacity/warmth");
    expect(warmth.pools[0].window_active).toBe(true);
    expect(warmth.pools[0].warm_idle_cells).toBe(1);
  });

  it("reads billing, which is the only view of unused-but-paid-for capacity", async () => {
    const fetchMock = jsonFetch({ observed_at: "2026-07-29T12:00:00Z", accrued_cost: 120.5, idle_cost: 30.25 });
    vi.stubGlobal("fetch", fetchMock);

    const billing = await new EyvanApiClient("/api/v1", false).getBilling();

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/capacity/billing");
    expect(billing.idle_cost).toBe(30.25);
  });

  it("surfaces a failing panel as an error rather than as empty data", async () => {
    // An empty explain panel reads as "nothing to report", which is the one
    // thing it must never say when the request failed.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => Promise.resolve(new Response("{}", { status: 503 }))),
    );

    await expect(new EyvanApiClient("/api/v1", false).getWarmth()).rejects.toEqual(
      expect.objectContaining({ name: "ApiError", status: 503 }),
    );
  });
});

describe("the workload detail page under a slow worker", () => {
  const raw = {
    id: "wl-1",
    org_id: "acme",
    project_id: "research",
    state: "running",
    attempt: 1,
    created_by: "alice",
    created_at: "2026-08-07T12:00:00Z",
    updated_at: "2026-08-07T12:00:00Z",
    spec: {
      name: "train",
      execution: { kind: "command" as const, command: "python train.py" },
      gang: { replicas: 1, per_replica: { cpu: 1, memory_gib: 1, gpu: 0, tpu: 0, custom: {} }, required_topology: {} },
      metadata: {},
      priority_band: "interactive" as const,
    },
  };

  const json = (body: unknown) =>
    new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });

  function route(overrides: Record<string, () => Response | Promise<Response>>) {
    return vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      for (const [fragment, handler] of Object.entries(overrides)) {
        if (url.includes(fragment)) return handler();
      }
      // Shape per endpoint. A catch-all returning the workload made `/leases`
      // answer with an object, and the failure looked like a code bug.
      if (url.includes("/leases")) return json([]);
      if (url.includes("/events")) return json([]);
      if (url.includes("/logs")) return new Response("log text", { status: 200 });
      if (url.includes("/progress"))
        return json({
          workload_id: "wl-1",
          workload_state: "running",
          phase: "running",
          request: { cells: 1, workers_per_cell: 1 },
          capacity: { ready_workers: 1 },
        });
      return json(raw);
    });
  }

  it("renders the workload even when the log fetch fails", async () => {
    // The bug this replaces: `/logs` is forwarded to the worker that produced
    // them, so on a live slice it takes ~12s against a 3.5s client budget. It sat
    // in the same `Promise.all` as the workload, its events and its progress --
    // all of which answered in under half a second -- with no fallback, so the
    // page rendered "Unable to load Eyvan state" for a workload that existed.
    vi.stubGlobal("fetch", route({ "/logs": () => Promise.reject(new Error("Eyvan API request timed out")) }));

    const detail = await new EyvanApiClient("/api/v1", false).getWorkload("wl-1");

    expect(detail.id).toBe("wl-1");
    expect(detail.name).toBe("train");
  });

  it("says why the logs are missing rather than showing an empty pane", async () => {
    // An empty log pane reads as "this job printed nothing", which is a
    // different and more alarming claim than "we could not fetch them".
    vi.stubGlobal("fetch", route({ "/logs": () => Promise.reject(new Error("boom")) }));

    const detail = await new EyvanApiClient("/api/v1", false).getWorkload("wl-1");

    expect(JSON.stringify(detail.logs)).toContain("logs unavailable");
  });

  it("survives a failed event fetch too", async () => {
    // The timeline is context. Losing it should cost the timeline, not the page.
    vi.stubGlobal("fetch", route({ "/events": () => Promise.reject(new Error("boom")) }));

    const detail = await new EyvanApiClient("/api/v1", false).getWorkload("wl-1");

    expect(detail.id).toBe("wl-1");
  });

  it("gives the worker-forwarded read a longer, but bounded, budget", async () => {
    // The two reads have different physics: one is answered from the plane's
    // own store, the other is a network round trip to a TPU VM that may be
    // gone. The forwarded read earns extra patience -- but it must stay
    // bounded, because its failure holds the detail page's loading gate and
    // the archive serves this attempt's history the moment it gives up.
    const seen: number[] = [];
    const original = globalThis.setTimeout;
    vi.stubGlobal("setTimeout", ((fn: () => void, ms?: number) => {
      if (typeof ms === "number") seen.push(ms);
      return original(fn, ms);
    }) as typeof globalThis.setTimeout);
    vi.stubGlobal("fetch", route({}));

    await new EyvanApiClient("/api/v1", false).getWorkload("wl-1");

    expect(Math.max(...seen)).toBeGreaterThan(Math.min(...seen));
    expect(Math.max(...seen)).toBeLessThanOrEqual(5_000);
  });
});

describe("workload ordering", () => {
  function listOf(...stamps: Array<[string, string]>) {
    return stamps.map(([id, created]) => ({
      id,
      org_id: "acme",
      project_id: "research",
      state: "succeeded",
      attempt: 1,
      created_by: "alice",
      created_at: created,
      updated_at: created,
      spec: {
        name: id,
        execution: { kind: "command" as const, command: "x" },
        gang: {
          replicas: 1,
          per_replica: { cpu: 1, memory_gib: 1, gpu: 0, tpu: 0, custom: {} },
          required_topology: {},
        },
        metadata: {},
        priority_band: "interactive" as const,
      },
    }));
  }

  it("lists the newest job first", async () => {
    // The control plane returns insertion order, so the job you just submitted
    // was buried under every job you had ever submitted.
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify(
              listOf(
                ["oldest", "2026-08-01T00:00:00Z"],
                ["newest", "2026-08-07T00:00:00Z"],
                ["middle", "2026-08-04T00:00:00Z"],
              ),
            ),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
      ),
    );

    const list = await new EyvanApiClient("/api/v1", false).listWorkloads();

    expect(list.map((w) => w.id)).toEqual(["newest", "middle", "oldest"]);
  });

  it("orders identical timestamps deterministically", async () => {
    // Otherwise the rows shuffle on every poll and the page is unreadable while
    // a burst of jobs lands in the same second -- which is what a fan-out is.
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify(
              listOf(
                ["wl-a", "2026-08-07T00:00:00Z"],
                ["wl-c", "2026-08-07T00:00:00Z"],
                ["wl-b", "2026-08-07T00:00:00Z"],
              ),
            ),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
      ),
    );

    const client = new EyvanApiClient("/api/v1", false);
    const first = (await client.listWorkloads()).map((w) => w.id);
    const second = (await client.listWorkloads()).map((w) => w.id);

    expect(first).toEqual(second);
    expect(first).toEqual(["wl-c", "wl-b", "wl-a"]);
  });
});

describe("provider TPU quota", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function quotaResponse() {
    return new Response(
      JSON.stringify({
        observed_at: "2026-07-29T12:05:00Z",
        provider: "gcp",
        project: "gcp-project",
        available_projects: ["gcp-project"],
        status: "ok",
        refreshed_at: "2026-07-29T12:04:00Z",
        age_s: 60,
        cached: true,
        ttl_s: 120,
        entries: [
          {
            zone: "us-central2-b",
            region: "us-central2",
            generation: "v6e",
            accelerator_type: "v6e-8",
            capacity_class: "spot",
            metric: "tpu.googleapis.com/tpu-v6e-preemptible",
            chips_per_slice: 8,
            limit_chips: 32,
            in_use_chips: 16,
            free_chips: 16,
            free_slices: 2,
            state: "known",
            usage_observed_at: "2026-07-29T12:00:00Z",
            usage_age_s: 300,
          },
        ],
        failures: [],
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  }

  it("reads the quota route without asking for a re-read", async () => {
    const fetch = vi.fn().mockResolvedValue(quotaResponse());
    vi.stubGlobal("fetch", fetch);

    const snapshot = await new EyvanApiClient("/api/v1", false).getTpuQuota();

    // Quota is a property of the cloud project, not of a tenant scope, so the
    // request carries no org/project selector -- and the reading it returns
    // states its own age rather than being presented as live.
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/capacity/quota");
    expect(snapshot.cached).toBe(true);
    expect(snapshot.age_s).toBe(60);
    expect(snapshot.entries[0]).toMatchObject({ free_chips: 16, free_slices: 2, state: "known" });
  });

  it("asks for a re-read only when the caller does", async () => {
    const fetch = vi.fn().mockResolvedValue(quotaResponse());
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    await client.getTpuQuota(client.getScope(), undefined, { refresh: true, project: "other-project" });

    expect(fetch.mock.calls[0][0]).toBe("/api/v1/capacity/quota?project=other-project&refresh=true");
  });

  it("keeps a missing usage as null rather than decoding it to zero", async () => {
    // The decode boundary matters as much as the render: a `?? 0` here would
    // make every renderer downstream honest and still wrong.
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          observed_at: "2026-07-29T12:05:00Z",
          provider: "gcp",
          project: "gcp-project",
          available_projects: [],
          status: "ok",
          refreshed_at: "2026-07-29T12:05:00Z",
          age_s: 0,
          cached: false,
          ttl_s: 120,
          entries: [
            {
              zone: "europe-west4-a",
              region: "europe-west4",
              generation: "v6e",
              accelerator_type: "v6e-8",
              capacity_class: "spot",
              metric: "tpu.googleapis.com/tpu-v6e-preemptible",
              chips_per_slice: 8,
              limit_chips: 64,
              in_use_chips: null,
              free_chips: null,
              free_slices: null,
              state: "usage_unknown",
              usage_observed_at: null,
              usage_age_s: null,
            },
          ],
          failures: [{ capacity_class: "on_demand", reason: "gcloud auth token lookup failed" }],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetch);

    const snapshot = await new EyvanApiClient("/api/v1", false).getTpuQuota();

    expect(snapshot.entries[0].free_chips).toBeNull();
    expect(snapshot.entries[0].limit_chips).toBe(64);
    expect(snapshot.failures[0].reason).toBe("gcloud auth token lookup failed");
  });

  it("reports a forbidden quota read as a typed error rather than an empty snapshot", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "'alice' lacks manage_fleet permission" }), {
          status: 403,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    await expect(new EyvanApiClient("/api/v1", false).getTpuQuota()).rejects.toEqual(
      expect.objectContaining({ name: "ApiError", status: 403 }),
    );
  }); // ------------------------------------------------------------------ following one workload's output

  /** A response whose body streams the given SSE text, then ends. */
  function streamingResponse(text: string): Response {
    const encoder = new TextEncoder();
    return new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(text));
          controller.close();
        },
      }),
      { status: 200 },
    );
  }

  /**
   * Follow a scripted stream until `expected` callbacks have landed, then stop.
   *
   * The abort is the test's, not the client's: a follow resolves only when its
   * caller aborts, because a stream that ended is a connection that dropped and
   * the right response to that is to reconnect. Waiting for it to finish on its
   * own would be waiting for the bug.
   */
  async function follow(text: string, expected: number) {
    const controller = new AbortController();
    const entries: LogEntry[] = [];
    const attempts: number[] = [];
    const stop = () => {
      if (entries.length + attempts.length >= expected) controller.abort();
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamingResponse(text)));
    await new EyvanApiClient("/api/v1", false).followWorkloadLogs(
      "wl-1",
      {
        onRecord: (entry) => {
          entries.push(entry);
          stop();
        },
        onAttemptChanged: (attempt) => {
          attempts.push(attempt);
          stop();
        },
      },
      controller.signal,
    );
    return { entries, attempts };
  }

  it("names the rank that produced each followed line", async () => {
    // The identity that matters in a gang. Sixteen hosts printing the same line
    // is normal; one printing something different is the whole diagnosis, and a
    // merged view cannot express it.
    const { entries } = await follow(
      'event: log.record\ndata: {"message":"step 1","stream":"stdout","rank":0,"timestamp":"2026-08-16T10:00:00Z"}\n\n' +
        'event: log.record\ndata: {"message":"step 1","stream":"stderr","rank":3,"timestamp":"2026-08-16T10:00:01Z"}\n\n',
      2,
    );

    expect(entries.map((entry) => entry.source)).toEqual(["rank 0", "rank 3"]);
    expect(entries.map((entry) => entry.stream)).toEqual(["stdout", "stderr"]);
    // Two ranks, the same text: distinct keys, or React renders one row twice.
    expect(new Set(entries.map((entry) => entry.id)).size).toBe(2);
  });

  it("prefers the timestamp the line carries over the time it was read", async () => {
    // On a host that has been buffering, those differ by however long the
    // buffer held -- and the reader is trying to date the event, not the read.
    const { entries } = await follow(
      'event: log.record\ndata: {"message":"2026-08-16T09:00:00Z INFO [trainer] loss=1.2","rank":1,"timestamp":"2026-08-16T11:30:00Z"}\n\n',
      1,
    );

    expect(entries[0].timestamp).toBe("2026-08-16T09:00:00Z");
    expect(entries[0].level).toBe("INFO");
    // The program's own tag is kept beside the rank, not instead of it.
    expect(entries[0].source).toBe("rank 1 · trainer");
    expect(entries[0].message).toBe("loss=1.2");
  });

  it("hands an attempt change to the caller rather than folding it into the output", async () => {
    // The lines after it come from a different process. Delivered as ordinary
    // records they read as the same run continuing, which is the one conclusion
    // a follower must not reach.
    const { entries, attempts } = await follow(
      'event: log.attempt_changed\ndata: {"attempt":2}\n\n' +
        'event: log.record\ndata: {"message":"restarted","rank":0}\n\n',
      2,
    );

    expect(attempts).toEqual([2]);
    expect(entries.map((entry) => entry.message)).toEqual(["restarted"]);
  });

  it("does not stream canned lines under a live indicator in demo mode", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    let connected: boolean | undefined;

    await new EyvanApiClient("/api/v1", true).followWorkloadLogs(
      "wl-1",
      { onRecord: () => {}, onAttemptChanged: () => {}, onConnectionChange: (value) => (connected = value) },
      new AbortController().signal,
    );

    expect(fetch).not.toHaveBeenCalled();
    expect(connected).toBeUndefined();
  });
  // ------------------------------------------------------------------ taking a cell out of service

  it("sends an idempotency key with a drain so a retried click drains once", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "cell-a", state: "draining" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);

    const result = await new EyvanApiClient("/api/v1", false).drainCell("cell-a");

    expect(result).toEqual({ id: "cell-a", state: "draining" });
    const [path, init] = fetch.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/api/v1/cells/cell-a/drain");
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["Idempotency-Key"]).toBeTruthy();
  });

  it("refuses to drain in demo mode rather than miming it", async () => {
    // A gesture that appears to work against a dataset with no fleet behind it
    // teaches the action without its consequence.
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    await expect(new EyvanApiClient("/api/v1", true).drainCell("cell-a")).rejects.toEqual(
      expect.objectContaining({ name: "ApiError" }),
    );
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe("how long a read is given", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("gives provider quota longer than a control-plane read", async () => {
    // The defect: quota discovery lists accelerator types in every zone a
    // project can see and takes tens of seconds cold, but ran on the 3.5s
    // budget meant for a database read. The page showed "Eyvan API request
    // timed out" on every first visit while the read itself was fine.
    vi.useFakeTimers();
    let settle: (value: Response) => void = () => {};
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise<Response>((resolve, reject) => {
            settle = resolve;
            init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
          }),
      ),
    );
    const client = new EyvanApiClient("/api/v1", false);
    const pending = client.getTpuQuota();

    // Well past the control-plane budget, and the request is still alive.
    await vi.advanceTimersByTimeAsync(20_000);
    settle(
      new Response(JSON.stringify({ entries: [], status: "ok" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    await expect(pending).resolves.toMatchObject({ status: "ok" });
  });

  it("still bounds it rather than waiting forever", async () => {
    // A budget that never fires is a hung tab. The server answers with the
    // previous reading or a stated failure at its own deadline; this outlasts
    // that answer, it does not replace it.
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
          }),
      ),
    );
    const client = new EyvanApiClient("/api/v1", false);
    const pending = client.getTpuQuota();
    const settled = pending.then(
      () => "resolved",
      (error) => (error as Error).message,
    );

    await vi.advanceTimersByTimeAsync(120_000);

    await expect(settled).resolves.toContain("timed out");
  });
});

describe("EyvanApiClient loopback credential refusals", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  function memoryStorage(initial: Record<string, string> = {}): Storage {
    const values = new Map(Object.entries(initial));
    return {
      get length() {
        return values.size;
      },
      clear: () => values.clear(),
      getItem: (key) => values.get(key) ?? null,
      key: (index) => [...values.keys()][index] ?? null,
      removeItem: (key) => {
        values.delete(key);
      },
      setItem: (key, value) => {
        values.set(key, value);
      },
    };
  }

  function refusal(detail: string): Response {
    return new Response(JSON.stringify({ detail }), {
      status: 401,
      headers: { "content-type": "application/json" },
    });
  }

  it("retires the held credential when the plane refuses exactly that", async () => {
    const store = memoryStorage({ "eyvan.local.credential": "stale-value" });
    vi.stubGlobal("sessionStorage", store);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(refusal("local Eyvan development credential is required")));
    const client = new EyvanApiClient("/api/v1", false);

    await expect(client.listCells()).rejects.toMatchObject({ status: 401 });

    expect(store.getItem("eyvan.local.credential")).toBeNull();
  });

  it("keeps the credential when a 401 is about something else", async () => {
    const store = memoryStorage({ "eyvan.local.credential": "held-value" });
    vi.stubGlobal("sessionStorage", store);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(refusal("invalid bearer token")));
    const client = new EyvanApiClient("/api/v1", false);

    await expect(client.listCells()).rejects.toMatchObject({ status: 401 });

    expect(store.getItem("eyvan.local.credential")).toBe("held-value");
  });
});

describe("EyvanApiClient tracking read budget and clean URLs", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("gives proxied tracking reads a 15s budget separate from DB reads", async () => {
    const seen: number[] = [];
    const original = globalThis.setTimeout;
    vi.stubGlobal("setTimeout", ((fn: () => void, ms?: number) => {
      if (typeof ms === "number") seen.push(ms);
      return original(fn, ms);
    }) as typeof globalThis.setTimeout);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ groups: [], steps: [], total: 0 }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    const client = new EyvanApiClient("/api/v1", false);

    await client.listRunCompletions("run-1", {}, { orgId: "default", projectId: "proj" });

    expect(seen).toContain(15_000);
  });

  it("omits empty query separator before org_id/project_id on trackingRead", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "comp-1", responses: [] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new EyvanApiClient("/api/v1", false);

    await client.getRunCompletion("run-1", "comp-1", { orgId: "default", projectId: "proj" });

    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toBe("/api/v1/tracking/runs/run-1/completions/comp-1?org_id=default&project_id=proj");
    expect(url).not.toContain("?&");
  });
});
