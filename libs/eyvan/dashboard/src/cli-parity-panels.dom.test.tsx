// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EyvanApiClient } from "./api/client";
import { mockBudgets } from "./api/mockData";
import { BudgetsPage } from "./pages/BudgetsPage";
import { FleetPage } from "./pages/FleetPage";
import { SettingsPage } from "./pages/SettingsPage";
import type { AssistantSettingsClient } from "./pages/AssistantProvidersSettings";
import type { TeamClient } from "./pages/UsersPage";
import type { Cell, SchedulingPolicy, StorageStatus } from "./types";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function sampleCell(overrides: Partial<Cell> = {}): Cell {
  return {
    id: "cell-eu-01",
    name: "europe-west4-a-v6e-01",
    backend: "gcp-tpu",
    region: "europe-west4",
    zone: "europe-west4-a",
    status: "healthy",
    state: "ready",
    utilization: 50,
    activeWorkloads: 1,
    topology: "v6e-16",
    resources: [{ name: "TPU", used: 8, total: 16 }],
    lastHeartbeat: "2026-07-10T09:00:00Z",
    ...overrides,
  };
}

function emptyTeamAndAssistantClient(): TeamClient & AssistantSettingsClient {
  return {
    listTeamUsers: vi.fn(async () => []),
    setTeamUserDisabled: vi.fn(),
    listTeamInvites: vi.fn(async () => []),
    createTeamInvite: vi.fn(),
    revokeTeamInvite: vi.fn(),
    listTeamTokens: vi.fn(async () => []),
    createTeamToken: vi.fn(),
    revokeTeamToken: vi.fn(),
    listAssistantProviders: vi.fn(async () => ({ providers: [], can_manage: true })),
    listAssistantPresets: vi.fn(async () => []),
    discoverAssistantModels: vi.fn(),
    createAssistantProvider: vi.fn(),
    updateAssistantProvider: vi.fn(),
    deleteAssistantProvider: vi.fn(),
    startOpenRouterSignIn: vi.fn(),
    getClaudeCodeStatus: vi.fn(),
    startClaudeCodeLogin: vi.fn(),
  };
}

const SAMPLE_STORAGE: StorageStatus = {
  configured: true,
  auto_create: false,
  config_path: "/home/alice/.eyvan/storage.json",
  project: "example-project-a",
  monitoring: {
    bucket: "example-monitoring",
    uri: "gs://example-monitoring/tracking",
    location: "us-central1",
    prefix: "tracking",
  },
  regional: {
    "europe-west4": {
      bucket: "example-eu-bucket",
      uri: "gs://example-eu-bucket",
      location: "europe-west4",
    },
  },
  context: {
    on_gcp: false,
    project: "example-project-a",
    zone: "europe-west4-a",
    region: "europe-west4",
    accelerator_type: null,
    source: "storage-config",
  },
};

describe("CLI-to-Dashboard Gap Closers", () => {
  it("supports creating/editing Budgets and full Scheduling Policies via modals on BudgetsPage", async () => {
    const fetchMock = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
      if (url.includes("/budgets/")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        return new Response(
          JSON.stringify({
            org_id: "acme",
            project_id: "research",
            limit: body.limit,
            spent: 0,
            period: body.period,
            hard_limit: body.hard_limit,
            alert_thresholds: body.alert_thresholds,
            accelerator_hour_limits: body.accelerator_hour_limits,
            accelerator_hours_spent: {},
            currency: "USD",
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      const body = JSON.parse(String(init?.body ?? "{}"));
      return new Response(
        JSON.stringify({
          org_id: "acme",
          project_id: "research",
          ...body,
          updated_at: "2026-07-10T10:00:00Z",
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new EyvanApiClient("/api/v1", false, { orgId: "acme", projectId: "research" });

    const onSaveBudget = vi.fn((input) => client.putBudget(input));
    const onPolicyUpdate = vi.fn((policy: SchedulingPolicy, patch) =>
      client.updatePolicy(policy, patch, { orgId: "acme", projectId: "research" }),
    );

    render(
      <BudgetsPage
        budgets={[]}
        policies={[]}
        orgId="acme"
        projectId="research"
        onPolicyToggle={async (policy) => policy}
        onPolicyUpdate={onPolicyUpdate}
        onSaveBudget={onSaveBudget}
        canManageProject
      />,
    );

    // Configure budget even when budgets.length === 0.
    fireEvent.click(screen.getByRole("button", { name: "Configure budget" }));
    const budgetDialog = screen.getByRole("dialog", { name: "Configure project budget" });
    fireEvent.change(within(budgetDialog).getByLabelText("Limit (USD)"), { target: { value: "25000" } });
    fireEvent.change(within(budgetDialog).getByLabelText("Period"), { target: { value: "weekly" } });
    fireEvent.change(within(budgetDialog).getByLabelText("Alert thresholds (0–1, comma-separated)"), {
      target: { value: "0.5, 0.9" },
    });
    fireEvent.change(within(budgetDialog).getByLabelText("Accelerator hour limits (optional key=hours)"), {
      target: { value: "tpu=400" },
    });
    fireEvent.click(within(budgetDialog).getByRole("button", { name: "Save budget" }));

    await waitFor(() => expect(onSaveBudget).toHaveBeenCalledTimes(1));
    const [budgetUrl, budgetInit] = fetchMock.mock.calls[0];
    expect(budgetUrl).toBe("/api/v1/budgets/acme/research");
    expect(budgetInit.method).toBe("PUT");
    expect(budgetInit.headers["Idempotency-Key"]).toBeTruthy();
    expect(JSON.parse(budgetInit.body)).toMatchObject({
      limit: 25000,
      period: "weekly",
      hard_limit: true,
      alert_thresholds: [0.5, 0.9],
      accelerator_hour_limits: { tpu: 400 },
    });

    // Configure full scheduling policy when policies.length === 0.
    fireEvent.click(screen.getByRole("button", { name: "Configure policy" }));
    const policyDialog = screen.getByRole("dialog", { name: "Configure project scheduling policy" });
    fireEvent.change(within(policyDialog).getByLabelText("Allowed regions (comma-separated)"), {
      target: { value: "europe-west4, us-central2" },
    });
    fireEvent.change(within(policyDialog).getByLabelText("Allowed providers (comma-separated)"), {
      target: { value: "gcp" },
    });
    fireEvent.change(within(policyDialog).getByLabelText("Cross-region transfer policy"), {
      target: { value: "ban" },
    });
    fireEvent.change(within(policyDialog).getByLabelText("Cross-region allowlisted buckets (comma-separated)"), {
      target: { value: "example-eu-bucket" },
    });
    fireEvent.click(
      within(policyDialog).getByLabelText("Allow continental multi-region buckets (uncheck for strict same-region)"),
    );
    fireEvent.click(within(policyDialog).getByRole("button", { name: "Save policy" }));

    await waitFor(() => expect(onPolicyUpdate).toHaveBeenCalledTimes(1));
    const [policyUrl, policyInit] = fetchMock.mock.calls[1];
    expect(policyUrl).toBe("/api/v1/policies/acme/research");
    expect(policyInit.method).toBe("PUT");
    expect(JSON.parse(policyInit.body)).toMatchObject({
      allowed_regions: ["europe-west4", "us-central2"],
      allowed_providers: ["gcp"],
      cross_region_transfer_policy: "ban",
      cross_region_allowed_buckets: ["example-eu-bucket"],
      cross_region_allow_multiregion: false,
    });
  });

  it("enables existing budget card edit button for project admins", () => {
    render(
      <BudgetsPage
        budgets={mockBudgets.slice(0, 1)}
        policies={[]}
        orgId="acme"
        projectId="research"
        onPolicyToggle={async (policy) => policy}
        onSaveBudget={async () => mockBudgets[0]}
        canManageProject
      />,
    );
    const editBtn = screen.getByRole("button", { name: `Edit ${mockBudgets[0].owner} budget` });
    expect(editBtn.hasAttribute("disabled")).toBe(false);
    fireEvent.click(editBtn);
    expect(screen.getByRole("dialog", { name: `Edit budget (${mockBudgets[0].owner})` })).toBeDefined();
  });

  it("offers Delete cell ONLY for draining/maintenance or degraded cells on FleetPage, never healthy cells", async () => {
    const onDeleteCell = vi.fn(async () => undefined);
    const healthy = sampleCell({ id: "cell-healthy", name: "healthy-v6e-01", status: "healthy", state: "ready" });
    const draining = sampleCell({
      id: "cell-draining",
      name: "draining-v6e-02",
      status: "maintenance",
      state: "draining",
      activeWorkloads: 0,
    });
    const degraded = sampleCell({
      id: "cell-degraded",
      name: "degraded-v6e-03",
      status: "degraded",
      state: "unreachable",
      activeWorkloads: 0,
    });

    const never = () => new Promise<never>(() => {});
    render(
      <FleetPage
        cells={[healthy, draining, degraded]}
        leases={[]}
        onRefresh={async () => undefined}
        loadWarmth={never}
        loadBilling={never}
        loadLaunchLatency={never}
        onDrainCell={async () => undefined}
        onDeleteCell={onDeleteCell}
      />,
    );

    expect(screen.queryByRole("button", { name: "Delete cell healthy-v6e-01" })).toBeNull();
    expect(screen.getByRole("button", { name: "Delete cell degraded-v6e-03" })).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Delete cell draining-v6e-02" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete cell draining-v6e-02?" });
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete cell" }));

    await waitFor(() => expect(onDeleteCell).toHaveBeenCalledWith("cell-draining"));
    expect(await screen.findByText(/draining-v6e-02 was removed from active fleet inventory/)).toBeDefined();
  });

  it("runs Storage Cross-region URI check, Managed Artifacts load, Webhooks CRUD, and Projects/Role Grants in SettingsPage", async () => {
    const checkStorageRegion = vi.fn(async () => ({
      uri: "gs://example-us-bucket/ckpt",
      bucket: "example-us-bucket",
      host_region: "europe-west4",
      bucket_location: "us-central2",
      location_type: "region",
      allowed: false,
      allowlisted: false,
      reason: "Bucket example-us-bucket is in us-central2, outside europe-west4.",
    }));
    const listManagedArtifacts = vi.fn(async () => [
      {
        uri: "gs://example-eu-bucket/workspaces/sha256-abc.tar.gz",
        kind: "workspace",
        org_id: "acme",
        project_id: "research",
        state: "active",
        record_generation: 2,
        size_bytes: 2048,
        sha256: "abc",
        object_version: "1",
        created_at: "2026-07-10T08:00:00Z",
        updated_at: "2026-07-10T08:05:00Z",
      },
    ]);
    const listWebhooks = vi.fn(async () => [
      {
        id: "whk-1",
        org_id: "acme",
        project_id: "research",
        url: "https://hooks.example.org/old",
        entity_types: ["workload"],
        actions: ["workload.failed"],
      },
    ]);
    const createWebhook = vi.fn(async (input) => ({
      id: "whk-2",
      org_id: "acme",
      project_id: "research",
      url: input.url,
      entity_types: input.entity_types ?? [],
      actions: input.actions ?? [],
    }));
    const deleteWebhook = vi.fn(async (id) => ({
      id,
      org_id: "acme",
      project_id: "research",
      url: "https://hooks.example.org/old",
      entity_types: [],
      actions: [],
    }));
    const registerProject = vi.fn(async (input) => ({
      org_id: input.orgId,
      id: input.projectId,
      display_name: input.displayName,
      owner: input.owner,
      enabled: true,
    }));
    const issueRoleGrant = vi.fn(async (input) => ({
      subject: input.subject,
      grant: { role: input.role, org_id: input.orgId ?? null, project_id: input.projectId ?? null },
      reason: input.reason ?? "",
    }));
    const revokeRoleGrant = vi.fn(async () => undefined);

    render(
      <SettingsPage
        client={{
          ...emptyTeamAndAssistantClient(),
          getStorageStatus: async () => SAMPLE_STORAGE,
          listRuns: async () => [],
          checkStorageRegion,
          listManagedArtifacts,
          listWebhooks,
          createWebhook,
          deleteWebhook,
          listProjects: async () => [],
          registerProject,
          listRoleGrants: async () => [],
          issueRoleGrant,
          revokeRoleGrant,
        }}
        teamMode={false}
        canManageFleet
        defaultOrg="acme"
        defaultProject="research"
      />,
    );

    // 1. Storage cross-region URI checker
    const storageRegion = await screen.findByRole("region", { name: "Storage and experiment tracking" });
    fireEvent.change(within(storageRegion).getByLabelText("GCS URI to check"), {
      target: { value: "gs://example-us-bucket/ckpt" },
    });
    fireEvent.change(within(storageRegion).getByLabelText("Target TPU region"), {
      target: { value: "europe-west4" },
    });
    fireEvent.click(within(storageRegion).getByRole("button", { name: "Check region" }));
    await waitFor(() => expect(checkStorageRegion).toHaveBeenCalled());
    expect(await within(storageRegion).findByText("Blocked")).toBeDefined();
    expect(within(storageRegion).getByText(/outside europe-west4/)).toBeDefined();

    // 2. Managed artifacts collapsible loader
    fireEvent.click(within(storageRegion).getByRole("button", { name: /Managed artifacts/ }));
    await waitFor(() => expect(listManagedArtifacts).toHaveBeenCalled());
    expect(await within(storageRegion).findByText("gs://example-eu-bucket/workspaces/sha256-abc.tar.gz")).toBeDefined();

    // 3. Webhooks CRUD
    const webhooksRegion = await screen.findByRole("region", { name: "Webhook subscriptions" });
    expect(await within(webhooksRegion).findByText("https://hooks.example.org/old")).toBeDefined();
    fireEvent.change(within(webhooksRegion).getByLabelText("Webhook URL"), {
      target: { value: "https://hooks.example.org/new" },
    });
    fireEvent.change(within(webhooksRegion).getByLabelText("Webhook signing secret"), {
      target: { value: "super-secret-signing-key-1234" },
    });
    fireEvent.click(within(webhooksRegion).getByRole("button", { name: "Add webhook" }));
    await waitFor(() => expect(createWebhook).toHaveBeenCalled());
    expect(await within(webhooksRegion).findByText("https://hooks.example.org/new")).toBeDefined();

    fireEvent.click(
      within(webhooksRegion).getByRole("button", { name: "Delete webhook https://hooks.example.org/old" }),
    );
    await waitFor(() => expect(deleteWebhook).toHaveBeenCalledWith("whk-1"));

    // 4. Projects & Role Grants
    const tenancyRegion = screen.getByRole("region", { name: "Projects and role grants" });
    fireEvent.change(within(tenancyRegion).getByLabelText("New project ID"), { target: { value: "evals" } });
    fireEvent.change(within(tenancyRegion).getByLabelText("Project owner"), { target: { value: "alice" } });
    fireEvent.click(within(tenancyRegion).getByRole("button", { name: "Register project" }));
    await waitFor(() =>
      expect(registerProject).toHaveBeenCalledWith(expect.objectContaining({ orgId: "acme", projectId: "evals" })),
    );

    fireEvent.change(within(tenancyRegion).getByLabelText("Grant subject"), { target: { value: "carol" } });
    fireEvent.click(within(tenancyRegion).getByRole("button", { name: "Grant role" }));
    await waitFor(() => expect(issueRoleGrant).toHaveBeenCalledWith(expect.objectContaining({ subject: "carol" })));
    fireEvent.click(await within(tenancyRegion).findByRole("button", { name: "Revoke project_operator from carol" }));
    await waitFor(() => expect(revokeRoleGrant).toHaveBeenCalled());
  });
});
