// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EyvanApiClient } from "../api/client";
import { getMockWorkload, mockBudgets, mockWorkloads } from "../api/mockData";
import type { ApiPolicyRecord, SchedulingPolicy, WorkloadDetail, WorkloadSummary } from "../types";
import { BudgetsPage } from "./BudgetsPage";
import { WorkloadDetailPage } from "./WorkloadDetailPage";
import { WorkloadsPage } from "./WorkloadsPage";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function samplePolicy(overrides: Partial<SchedulingPolicy> = {}): SchedulingPolicy {
  return {
    id: "acme/research",
    name: "Placement policy (research)",
    scope: "research",
    type: "placement",
    summary: "europe-west4; spot; gcp; preemption enabled",
    preemptionEnabled: true,
    crossRegionTransferPolicy: "allow",
    crossRegionAllowedBuckets: [],
    crossRegionAllowMultiregion: true,
    updatedAt: "2026-07-10T08:00:00Z",
    configuration: {
      allowedProviders: ["gcp"],
      allowedRegions: ["europe-west4"],
      allowedCapacityClasses: ["spot"],
      deniedAcceleratorTypes: ["v4"],
      allowedSecretReferences: ["gcp-secret://projects/acme/secrets/"],
      maxPriority: 500,
    },
    ...overrides,
  };
}

function sampleDetail(overrides: Partial<WorkloadDetail> = {}): WorkloadDetail {
  const base = getMockWorkload(mockWorkloads[0].id);
  if (!base) throw new Error("missing mock workload");
  return { ...structuredClone(base), ...overrides };
}

describe("cross-region GCS transfer policy UI and client", () => {
  it("renders ban, warn, and allow policy modes with allowlisted bucket hints and callout", () => {
    const policies: SchedulingPolicy[] = [
      samplePolicy({
        id: "acme/research",
        name: "Placement policy (research)",
        scope: "research",
        crossRegionTransferPolicy: "ban",
        crossRegionAllowedBuckets: ["example-us-bucket", "example-eu-bucket"],
        crossRegionAllowMultiregion: false,
      }),
      samplePolicy({
        id: "acme/evals",
        name: "Placement policy (evals)",
        scope: "evals",
        crossRegionTransferPolicy: "warn",
        crossRegionAllowedBuckets: ["example-eu-bucket"],
        crossRegionAllowMultiregion: true,
      }),
      samplePolicy({
        id: "acme/sandbox",
        name: "Placement policy (sandbox)",
        scope: "sandbox",
        crossRegionTransferPolicy: "allow",
        crossRegionAllowedBuckets: [],
        crossRegionAllowMultiregion: true,
      }),
    ];

    render(
      <BudgetsPage
        budgets={mockBudgets.slice(0, 1)}
        policies={policies}
        onPolicyToggle={async (policy) => policy}
        onCrossRegionPolicyChange={async (policy) => policy}
        canManageProject={false}
      />,
    );

    const rows = screen.getAllByRole("row").slice(1);
    expect(rows).toHaveLength(3);

    expect(within(rows[0]).getByText("Banned")).toBeDefined();
    expect(within(rows[0]).getByText("2 allowlisted buckets")).toBeDefined();
    expect(within(rows[0]).getByText("Multi-region blocked")).toBeDefined();

    expect(within(rows[1]).getByText("Warn")).toBeDefined();
    expect(within(rows[1]).getByText("1 allowlisted bucket")).toBeDefined();

    expect(within(rows[2]).getByText("Allowed")).toBeDefined();
    expect(within(rows[2]).queryByText(/allowlisted bucket/)).toBeNull();

    // Non-admin sees read-only badges with no select or checkbox edit controls.
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.queryByRole("checkbox")).toBeNull();

    // Side-panel callout explains enforcement semantics.
    expect(screen.getByText("Cross-region transfer: Ban")).toBeDefined();
    expect(screen.getByText(/exit 95, non-retryable/i)).toBeDefined();
  });

  it("lets an admin change cross-region transfer policy from allow to ban and preserves all existing policy fields in PUT", async () => {
    const serverRecord: ApiPolicyRecord = {
      org_id: "acme",
      project_id: "research",
      allowed_providers: ["gcp"],
      allowed_regions: ["europe-west4"],
      allowed_capacity_classes: ["spot"],
      denied_accelerator_types: ["v4"],
      allowed_secret_references: ["gcp-secret://projects/acme/secrets/"],
      max_priority: 500,
      preemption_enabled: true,
      cross_region_transfer_policy: "ban",
      cross_region_allowed_buckets: ["example-eu-bucket"],
      cross_region_allow_multiregion: false,
      updated_at: "2026-07-10T09:15:00Z",
    };
    const pendingFetch = deferred<Response>();
    const fetch = vi.fn().mockReturnValue(pendingFetch.promise);
    vi.stubGlobal("fetch", fetch);

    const client = new EyvanApiClient("/api/v1", false, { orgId: "acme", projectId: "research" });
    const initialPolicy = samplePolicy({
      preemptionEnabled: true,
      crossRegionTransferPolicy: "allow",
      crossRegionAllowedBuckets: ["example-eu-bucket"],
      crossRegionAllowMultiregion: false,
    });

    let currentPolicy = initialPolicy;
    const onCrossRegionPolicyChange = vi.fn(async (policy: SchedulingPolicy, mode: "allow" | "warn" | "ban") => {
      const updated = await client.updatePolicy(
        policy,
        { crossRegionTransferPolicy: mode },
        { orgId: "acme", projectId: "research" },
      );
      currentPolicy = updated;
      return updated;
    });

    const { rerender } = render(
      <BudgetsPage
        budgets={mockBudgets.slice(0, 1)}
        policies={[currentPolicy]}
        onPolicyToggle={async (policy) => policy}
        onCrossRegionPolicyChange={onCrossRegionPolicyChange}
        canManageProject
      />,
    );

    const select = screen.getByRole("combobox", {
      name: "Cross-region transfer policy for Placement policy (research)",
    }) as HTMLSelectElement;
    expect(select.value).toBe("allow");
    expect(select.disabled).toBe(false);

    fireEvent.change(select, { target: { value: "ban" } });

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    expect(select.disabled).toBe(true);
    expect(screen.getByText("Saving…")).toBeDefined();

    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("/api/v1/policies/acme/research");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body)).toEqual({
      allowed_providers: ["gcp"],
      allowed_regions: ["europe-west4"],
      allowed_capacity_classes: ["spot"],
      denied_accelerator_types: ["v4"],
      allowed_secret_references: ["gcp-secret://projects/acme/secrets/"],
      max_priority: 500,
      preemption_enabled: true,
      cross_region_transfer_policy: "ban",
      cross_region_allowed_buckets: ["example-eu-bucket"],
      cross_region_allow_multiregion: false,
    });

    await act(async () => {
      pendingFetch.resolve(
        new Response(JSON.stringify(serverRecord), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    });

    await waitFor(() => expect(select.disabled).toBe(false));
    expect(currentPolicy).toMatchObject({
      preemptionEnabled: true,
      crossRegionTransferPolicy: "ban",
      crossRegionAllowedBuckets: ["example-eu-bucket"],
      crossRegionAllowMultiregion: false,
    });

    rerender(
      <BudgetsPage
        budgets={mockBudgets.slice(0, 1)}
        policies={[currentPolicy]}
        onPolicyToggle={async (policy) => policy}
        onCrossRegionPolicyChange={onCrossRegionPolicyChange}
        canManageProject
      />,
    );
    expect(screen.getByText("Banned")).toBeDefined();
    expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("ban");
  });

  it("shows an ErrorCard when updating cross-region transfer policy fails", async () => {
    render(
      <BudgetsPage
        budgets={mockBudgets.slice(0, 1)}
        policies={[samplePolicy()]}
        onPolicyToggle={async (policy) => policy}
        onCrossRegionPolicyChange={async () => {
          throw new Error("Eyvan API request failed with 500: policy store unavailable");
        }}
        canManageProject
      />,
    );

    const select = screen.getByRole("combobox", {
      name: "Cross-region transfer policy for Placement policy (research)",
    });
    fireEvent.change(select, { target: { value: "ban" } });

    await waitFor(() => expect(screen.getByText(/policy store unavailable/i)).toBeDefined());
  });

  it("defaults omitted cross-region policy fields when reading from an older server", async () => {
    const legacyRecord: ApiPolicyRecord = {
      org_id: "acme",
      project_id: "research",
      allowed_providers: ["gcp"],
      allowed_regions: ["us-central2"],
      allowed_capacity_classes: ["spot"],
      denied_accelerator_types: [],
      max_priority: 1000,
      preemption_enabled: true,
      updated_at: "2026-07-01T00:00:00Z",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(legacyRecord), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    const client = new EyvanApiClient("/api/v1", false, { orgId: "acme", projectId: "research" });

    const policies = await client.listPolicies();
    expect(policies).toHaveLength(1);
    expect(policies[0]).toMatchObject({
      crossRegionTransferPolicy: "allow",
      crossRegionAllowedBuckets: [],
      crossRegionAllowMultiregion: true,
    });
  });

  it("renders a critical policy-ban banner on WorkloadDetailPage when failureReason or queueReason starts with cross-region-transfer-banned", () => {
    const killedWorkload = sampleDetail({
      id: "wl-cross-region-kill",
      name: "eu-train-bad-bucket",
      user: "alice",
      status: "failed",
      failureReason:
        "cross-region-transfer-banned: TPU in europe-west4 attempted read on gs://example-us-bucket/ckpt (bucket location: us-central2). Use a same-region bucket.",
    });

    const { rerender } = render(
      <WorkloadDetailPage
        workload={killedWorkload}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => new Promise(() => {})}
      />,
    );

    const banner = screen.getByTestId("cross-region-ban-banner");
    expect(banner.getAttribute("role")).toBe("alert");
    expect(banner.textContent).toContain("Killed & banned: Cross-region GCS transfer blocked");
    expect(banner.textContent).toContain("gs://example-us-bucket/ckpt");
    expect(banner.textContent).toContain("eyvan.storage_path()");

    // Dispatch-time rejection via scheduling_reason (mapped to queueReason) also renders the banner.
    const dispatchBlocked = sampleDetail({
      id: "wl-cross-region-dispatch",
      name: "eu-dispatch-blocked",
      user: "alice",
      status: "queued",
      failureReason: undefined,
      queueReason:
        "cross-region-transfer-banned: Checkpoint URI gs://example-us-bucket/out is in us-central2, outside europe-west4.",
    });

    rerender(
      <WorkloadDetailPage
        workload={dispatchBlocked}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => new Promise(() => {})}
      />,
    );

    const dispatchBanner = screen.getByTestId("cross-region-ban-banner");
    expect(dispatchBanner.textContent).toContain("Dispatch blocked: Cross-region GCS transfer banned");
    expect(dispatchBanner.textContent).toContain("gs://example-us-bucket/out");
    expect(screen.queryByText(/^Waiting:/)).toBeNull();

    // Ordinary failures do not render the cross-region ban banner.
    rerender(
      <WorkloadDetailPage
        workload={sampleDetail({ status: "failed", failureReason: "exit code 1: OOM", queueReason: undefined })}
        onBack={() => {}}
        onAction={() => {}}
        canOperate
        loadExplanation={() => new Promise(() => {})}
        loadCost={() => new Promise(() => {})}
      />,
    );
    expect(screen.queryByTestId("cross-region-ban-banner")).toBeNull();
  });

  it("renders a concise Cross-region banned badge on WorkloadsPage rows", () => {
    const bannedRow: WorkloadSummary = {
      ...mockWorkloads[0],
      id: "wl-banned-row",
      name: "eu-banned-run",
      user: "alice",
      status: "failed",
      failureReason: "cross-region-transfer-banned: TPU in europe-west4 attempted read on gs://example-us-bucket/data.",
    };
    const normalRow: WorkloadSummary = {
      ...mockWorkloads[0],
      id: "wl-ok-row",
      name: "eu-ok-run",
      user: "alice",
      status: "running",
      failureReason: undefined,
      queueReason: undefined,
    };

    render(
      <WorkloadsPage
        workloads={[bannedRow, normalRow]}
        onOpenWorkload={() => {}}
        onSubmit={() => {}}
        canSubmit={false}
      />,
    );

    const badges = screen.getAllByTestId("cross-region-ban-badge");
    expect(badges).toHaveLength(1);
    expect(badges[0].textContent).toContain("Cross-region banned");
    expect(badges[0].getAttribute("title")).toContain("gs://example-us-bucket/data");
  });
});
