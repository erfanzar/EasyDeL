// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EyvanApiClient } from "../api/client";
import { getMockWorkload, mockDashboardData, mockWorkloads } from "../api/mockData";
import type { GovernanceSummary, WorkloadDetail, WorkloadSummary } from "../types";
import { OverviewPage } from "./OverviewPage";
import { WorkloadDetailPage } from "./WorkloadDetailPage";
import { WorkloadsPage } from "./WorkloadsPage";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

function sampleDetail(overrides: Partial<WorkloadDetail> = {}): WorkloadDetail {
  const base = getMockWorkload(mockWorkloads[0].id);
  if (!base) throw new Error("missing mock workload");
  return { ...structuredClone(base), ...overrides };
}

describe("live reclaim visibility across WorkloadDetailPage, WorkloadsPage, and OverviewPage", () => {
  it("renders a live ticking ReclaimBanner on WorkloadDetailPage for both victim and waiting-owner jobs", () => {
    vi.useFakeTimers();
    const base = new Date("2026-11-27T14:00:00Z").getTime();
    vi.setSystemTime(base);
    const deadlineAt = new Date(base + 522_000).toISOString(); // 8m 42s

    const victim = sampleDetail({
      id: "wl-alice-borrowed",
      name: "alice-qwen-borrow",
      user: "alice",
      status: "checkpointing",
      reclaimDeadlineAt: deadlineAt,
      reclaimRequestedBy: "bob",
      reclaimPolicy: "requeue",
      checkpointGraceSeconds: 600,
    });

    const { rerender } = render(
      <WorkloadDetailPage
        workload={victim}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => new Promise(() => {})}
      />,
    );

    const victimBanner = screen.getByTestId("reclaim-banner");
    expect(victimBanner.textContent).toContain("Reclaiming · 8m 42s remaining");
    expect(victimBanner.textContent).toContain("bob");
    expect(victimBanner.textContent).toContain("Auto-resume elsewhere");

    act(() => {
      vi.advanceTimersByTime(1_000);
    });
    expect(victimBanner.textContent).toContain("Reclaiming · 8m 41s remaining");

    const waitingOwner = sampleDetail({
      id: "wl-bob-owner",
      name: "bob-v5p-128",
      user: "bob",
      status: "queued",
      queueReason: "Reclaiming borrowed capacity from alice",
      reclaimDeadlineAt: deadlineAt,
      reclaimPolicy: "stop",
      checkpointGraceSeconds: 600,
    });

    rerender(
      <WorkloadDetailPage
        workload={waitingOwner}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => new Promise(() => {})}
      />,
    );

    const waitingBanner = screen.getByTestId("reclaim-banner");
    expect(waitingBanner.textContent).toContain("Reclaiming borrowed capacity · 8m 41s remaining");
    expect(waitingBanner.textContent).toContain("Reclaiming borrowed capacity from alice");
    expect(waitingBanner.textContent).toContain("Stop after checkpoint");

    // Once terminal, the countdown banner must not linger.
    rerender(
      <WorkloadDetailPage
        workload={{ ...victim, status: "cancelled" }}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => new Promise(() => {})}
      />,
    );
    expect(screen.queryByTestId("reclaim-banner")).toBeNull();
  });

  it("renders a compact ticking ReclaimBadge on WorkloadsPage rows", () => {
    vi.useFakeTimers();
    const base = new Date("2026-11-27T14:00:00Z").getTime();
    vi.setSystemTime(base);
    const deadlineAt = new Date(base + 305_000).toISOString();

    const reclaimingWorkload: WorkloadSummary = {
      ...mockWorkloads[0],
      id: "wl-reclaim-row",
      name: "borrowed-train",
      status: "checkpointing",
      reclaimDeadlineAt: deadlineAt,
      reclaimRequestedBy: "bob",
      reclaimPolicy: "requeue",
    };
    const opened = vi.fn();

    render(
      <WorkloadsPage workloads={[reclaimingWorkload]} onOpenWorkload={opened} onSubmit={() => {}} canSubmit={false} />,
    );

    const badge = screen.getByTestId("reclaim-badge");
    expect(badge.textContent).toContain("Reclaiming · 5m 05s");

    act(() => {
      vi.advanceTimersByTime(2_000);
    });
    expect(badge.textContent).toContain("Reclaiming · 5m 03s");

    fireEvent.click(badge);
    expect(opened).toHaveBeenCalledWith("wl-reclaim-row");
  });

  it("shows active reclaims on OverviewPage from both workloads and governance summary and opens on click", async () => {
    const deadline = new Date(Date.now() + 420_000).toISOString();
    const govSummary: GovernanceSummary = {
      enabled: true,
      total_guaranteed_tpu_units: 256,
      total_in_use_tpu_units: 128,
      total_borrowed_tpu_units: 64,
      users: [],
      active_reclaims: [
        {
          workload_id: "wl-other-project",
          owner: "alice",
          requested_by: "bob",
          deadline_at: deadline,
          policy: "stop",
          state: "checkpointing",
        },
      ],
    };
    const scopedReclaim: WorkloadSummary = {
      ...mockWorkloads[0],
      id: "wl-scoped",
      name: "scoped-borrow",
      user: "alice",
      status: "checkpointing",
      reclaimDeadlineAt: deadline,
      reclaimRequestedBy: "admin",
      reclaimPolicy: "requeue",
    };
    const opened = vi.fn();

    render(
      <OverviewPage
        overview={mockDashboardData.overview}
        workloads={[scopedReclaim]}
        loadGovernanceSummary={async () => govSummary}
        onOpenWorkload={opened}
        onShowWorkloads={() => {}}
        fleetAvailable
      />,
    );

    const strip = await screen.findByRole("region", { name: "Active reclaims" });
    await waitFor(() => expect(strip.textContent).toContain("wl-other-project"));
    expect(strip.textContent).toContain("scoped-borrow");
    expect(strip.textContent).toContain("Auto-resume elsewhere");
    expect(strip.textContent).toContain("Stop after checkpoint");

    fireEvent.click(within(strip).getByRole("button", { name: /wl-other-project/ }));
    expect(opened).toHaveBeenCalledWith("wl-other-project");
  });
});

describe("EyvanApiClient governance, reclaim, and multi-project pool mappings", () => {
  it("handles GET /api/v1/governance/summary on 200 and degrades to null on 404 or 403", async () => {
    const payload: GovernanceSummary = {
      enabled: true,
      total_guaranteed_tpu_units: 128,
      total_in_use_tpu_units: 64,
      total_borrowed_tpu_units: 16,
      users: [],
      active_reclaims: [],
    };
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(payload), { status: 200, headers: { "content-type": "application/json" } }),
      )
      .mockResolvedValueOnce(new Response("Not Found", { status: 404 }))
      .mockResolvedValueOnce(new Response("Forbidden", { status: 403 }));
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    await expect(client.getGovernanceSummary()).resolves.toEqual(payload);
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/governance/summary");
    await expect(client.getGovernanceSummary()).resolves.toBeNull();
    await expect(client.getGovernanceSummary()).resolves.toBeNull();
  });

  it("maps optional workload reclaim fields and capacity pool provider_project + provider_account", async () => {
    const fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes("/workloads")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                id: "wl-rec-1",
                org_id: "acme",
                project_id: "research",
                state: "checkpointing",
                attempt: 1,
                created_by: "alice",
                created_at: "2026-11-27T13:00:00Z",
                updated_at: "2026-11-27T13:05:00Z",
                reclaim_deadline_at: "2026-11-27T13:15:00Z",
                reclaim_requested_by: "bob",
                reclaim_policy: "requeue",
                spec: {
                  name: "borrowed-train",
                  execution: { kind: "command", command: "python train.py" },
                  gang: {
                    replicas: 1,
                    per_replica: { cpu: 8, memory_gib: 64, gpu: 0, tpu: 64, custom: {} },
                    required_topology: {},
                  },
                  metadata: {},
                  accelerator_type: "v5p-128",
                  priority_band: "batch",
                  checkpoint_grace_seconds: 600,
                },
              },
            ]),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      if (url.includes("/capacity-pools")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                id: "spot-v6e-8",
                fleet_name: "spot-fleet",
                accelerator_type: "v6e-8",
                capacity_class: "spot",
                zone: "us-central2-b",
                region: "us-central2",
                max_cells: 4,
                min_cells: 0,
                enabled: true,
                provider: "gcp",
                provider_project: "example-project-a",
                provider_account: "operator@example.com",
              },
            ]),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        );
      }
      return Promise.resolve(new Response("[]", { status: 200 }));
    });
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false);

    const workloads = await client.listWorkloads();
    expect(workloads[0]).toMatchObject({
      id: "wl-rec-1",
      reclaimDeadlineAt: "2026-11-27T13:15:00Z",
      reclaimRequestedBy: "bob",
      reclaimPolicy: "requeue",
      checkpointGraceSeconds: 600,
    });

    const pools = await client.listCapacityPools();
    expect(pools[0]).toMatchObject({
      id: "spot-v6e-8",
      provider: "gcp",
      providerProject: "example-project-a",
      providerAccount: "operator@example.com",
    });
  });
});
