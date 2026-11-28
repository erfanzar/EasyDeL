// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "../api/client";
import { getMockWorkload, mockBudgets, mockDashboardData, mockPolicies, mockWorkloads } from "../api/mockData";
import { capabilitiesForPermissions, type AuthIdentity } from "../auth/oidc";
import type { EffectiveProjectScope, SessionPermission } from "../types";
import { Shell } from "../components/Shell";
import { SubmitWorkloadModal } from "../components/SubmitWorkloadModal";
import { BudgetsPage } from "./BudgetsPage";
import { OverviewPage } from "./OverviewPage";
import { WorkloadDetailPage } from "./WorkloadDetailPage";
import { WorkloadsPage } from "./WorkloadsPage";

const PERMISSIONS_BY_ROLE: Record<string, SessionPermission[]> = {
  project_viewer: ["read"],
  project_submitter: ["submit"],
  project_operator: ["read", "submit", "operate"],
  project_admin: ["read", "submit", "operate", "manage_project"],
  org_admin: ["read", "submit", "operate", "manage_project", "manage_own_capacity"],
  fleet_admin: ["read", "submit", "operate", "manage_project", "manage_own_capacity", "manage_fleet"],
};

function identity(role: string): AuthIdentity {
  return {
    displayName: "Alice",
    detail: role,
    mode: "oidc",
    roles: [role],
    scope: { orgId: "acme", projectId: "research" },
    capabilities: capabilitiesForPermissions(PERMISSIONS_BY_ROLE[role] ?? []),
  };
}

function effectiveScope(role: string): EffectiveProjectScope {
  return {
    orgId: "acme",
    projectId: "research",
    displayName: "Research",
    registered: true,
    enabled: true,
    acceptsNewWork: true,
    roles: [role],
    permissions: PERMISSIONS_BY_ROLE[role] ?? [],
  };
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("capability-aware actions", () => {
  it.each([
    ["project_viewer", false, false],
    ["project_submitter", true, false],
    ["project_operator", true, false],
    ["project_admin", true, false],
    ["org_admin", true, false],
    ["fleet_admin", true, true],
  ])("renders server permissions for %s", (role, canSubmit, canManageFleet) => {
    const resolved = identity(role);
    const scope = effectiveScope(role);
    render(
      <Shell
        route={{ section: "overview" }}
        mobileNavigationOpen={false}
        onMobileNavigationChange={() => {}}
        onNavigate={() => {}}
        onSubmit={() => {}}
        sectionAvailability={{ fleet: canManageFleet, audit: true }}
        identity={resolved}
        scopes={[scope]}
        selectedScope={scope}
        onScopeChange={() => {}}
      >
        <span>scoped content</span>
      </Shell>,
    );

    expect(screen.queryByRole("button", { name: "Submit workload" }) !== null).toBe(canSubmit);
    expect(screen.queryByRole("button", { name: "Fleet" }) !== null).toBe(canManageFleet);
  });

  it("switches among server-effective project scopes", () => {
    const first = effectiveScope("project_viewer");
    const second = {
      ...effectiveScope("project_operator"),
      orgId: "acme",
      projectId: "platform",
      displayName: "Platform",
    };
    const onScopeChange = vi.fn();
    render(
      <Shell
        route={{ section: "overview" }}
        mobileNavigationOpen={false}
        onMobileNavigationChange={() => {}}
        onNavigate={() => {}}
        onSubmit={() => {}}
        sectionAvailability={{ fleet: false, audit: true }}
        identity={identity("project_viewer")}
        scopes={[first, second]}
        selectedScope={first}
        onScopeChange={onScopeChange}
      >
        <span>scoped content</span>
      </Shell>,
    );

    fireEvent.change(screen.getByRole("combobox", { name: "Project scope" }), {
      target: { value: JSON.stringify([second.orgId, second.projectId]) },
    });
    expect(onScopeChange).toHaveBeenCalledWith(second);
  });

  it("hides fleet navigation and submit controls from a project viewer", () => {
    const viewer = identity("project_viewer");
    render(
      <Shell
        route={{ section: "workloads" }}
        mobileNavigationOpen={false}
        onMobileNavigationChange={() => {}}
        onNavigate={() => {}}
        onSubmit={() => {}}
        sectionAvailability={{ fleet: false, audit: false }}
        identity={viewer}
        scopes={[effectiveScope("project_viewer")]}
        selectedScope={effectiveScope("project_viewer")}
        onScopeChange={() => {}}
      >
        <WorkloadsPage
          workloads={mockWorkloads.slice(0, 1)}
          onOpenWorkload={() => {}}
          onSubmit={() => {}}
          canSubmit={viewer.capabilities.submitWorkloads}
        />
      </Shell>,
    );

    expect(screen.queryByRole("button", { name: "Submit workload" })).toBeNull();
    expect(screen.queryByRole("button", { name: "New workload" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Fleet" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Audit events" })).toBeNull();
    expect(screen.getByText(mockWorkloads[0].name)).toBeDefined();
  });

  it("shows submission to an operator but keeps fleet navigation unavailable", () => {
    const operator = identity("project_operator");
    render(
      <Shell
        route={{ section: "workloads" }}
        mobileNavigationOpen={false}
        onMobileNavigationChange={() => {}}
        onNavigate={() => {}}
        onSubmit={() => {}}
        sectionAvailability={{ fleet: false, audit: false }}
        identity={operator}
        scopes={[effectiveScope("project_operator")]}
        selectedScope={effectiveScope("project_operator")}
        onScopeChange={() => {}}
      >
        <span>project content</span>
      </Shell>,
    );

    expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Fleet" })).toBeNull();
  });

  it("removes workload mutations from a viewer while leaving the detail readable", () => {
    const viewer = identity("project_viewer");
    const workload = getMockWorkload(mockWorkloads[0].id);
    if (!workload) throw new Error("the mock dataset has no workload detail");
    vi.spyOn(apiClient, "explainWorkload").mockResolvedValue({
      workload_id: workload.id,
      state: "running",
      placeable: true,
      placement_cell_id: "cell-1",
      code: null,
      message: null,
      cells: [],
      considered_cell_count: 1,
      provisioning_cell_ids: [],
      scheduling_reason: null,
    });

    render(
      <WorkloadDetailPage
        workload={workload}
        onBack={() => {}}
        onAction={() => {}}
        canOperate={viewer.capabilities.operateWorkloads}
        loadExplanation={(id, signal) => apiClient.explainWorkload(id, viewer.scope, signal)}
        loadCost={() => new Promise(() => {})}
      />,
    );

    expect(screen.getByText(workload.name)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Checkpoint" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("binds submission to the configured project instead of offering a cross-scope write", () => {
    render(
      <SubmitWorkloadModal
        open
        project="platform"
        shapes={[]}
        onClose={() => {}}
        onSubmit={async () => {
          throw new Error("not submitted in this test");
        }}
      />,
    );

    const project = screen.getByLabelText("Project") as HTMLInputElement;
    expect(project.value).toBe("platform");
    expect(project.readOnly).toBe(true);
  });

  it("does not present a forbidden fleet as zero while project summaries remain visible", () => {
    render(
      <OverviewPage
        overview={mockDashboardData.overview}
        workloads={mockWorkloads.slice(0, 1)}
        onOpenWorkload={() => {}}
        onShowWorkloads={() => {}}
        fleetAvailable={false}
      />,
    );

    expect(screen.getByText("Restricted")).toBeDefined();
    expect(screen.getByText("Fleet inventory is restricted")).toBeDefined();
    expect(screen.queryByText("TPU v5p")).toBeNull();
    expect(screen.getByText("Budgeted spend")).toBeDefined();
    expect(screen.getByText(mockWorkloads[0].name)).toBeDefined();
  });

  it("renders an all-zero queue window as an empty state with no fake bars", () => {
    const { container } = render(
      <OverviewPage
        overview={{
          ...mockDashboardData.overview,
          queueTrend: mockDashboardData.overview.queueTrend.map((point) => ({
            ...point,
            submitted: 0,
            admitted: 0,
          })),
        }}
        workloads={[]}
        onOpenWorkload={() => {}}
        onShowWorkloads={() => {}}
        fleetAvailable={false}
      />,
    );

    expect(screen.getByText("No submission or admission activity")).toBeDefined();
    expect(container.querySelectorAll(".bar")).toHaveLength(0);
  });

  it("renders policies read-only unless the role can manage the project", () => {
    const operator = identity("project_operator");
    render(
      <BudgetsPage
        budgets={mockBudgets.slice(0, 1)}
        policies={mockPolicies.slice(0, 1)}
        onPolicyToggle={async (policy) => policy}
        canManageProject={operator.capabilities.manageProject}
      />,
    );

    expect(screen.getByText(mockPolicies[0].name)).toBeDefined();
    expect(screen.queryByRole("checkbox")).toBeNull();
    expect(screen.getByText("Read only")).toBeDefined();
  });
});
