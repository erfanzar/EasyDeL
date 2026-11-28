// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { loadDashboardData } from "./App";
import { ApiError } from "./api/client";
import { mockAuditEvents, mockBudgets, mockCells, mockLeases, mockPolicies, mockWorkloads } from "./api/mockData";

function dashboardClient() {
  return {
    getHealth: vi.fn(async () => ({ status: "ok", service: "eyvan" })),
    listWorkloads: vi.fn(async () => mockWorkloads.slice(0, 1)),
    listLeases: vi.fn(async () => mockLeases.slice(0, 1)),
    listBudgets: vi.fn(async () => mockBudgets.slice(0, 1)),
    listPolicies: vi.fn(async () => mockPolicies.slice(0, 1)),
    getQueueTrend: vi.fn(async () => [{ label: "10:00", submitted: 7, admitted: 3 }]),
    listCells: vi.fn(async () => mockCells.slice(0, 1)),
    listAuditEvents: vi.fn(async () => mockAuditEvents.slice(0, 1)),
  };
}

describe("loadDashboardData", () => {
  it("loads project audit without issuing any fleet inventory read for a project reader", async () => {
    const client = dashboardClient();

    const data = await loadDashboardData(
      client,
      { canRead: true, canManageFleet: false },
      { orgId: "acme", projectId: "research" },
    );

    expect(data.workloads).toHaveLength(1);
    expect(data.budgets).toHaveLength(1);
    expect(data.auditEvents).toHaveLength(1);
    expect(data.cells).toEqual([]);
    expect(data.access).toEqual({ fleetInventory: false, fleetAudit: true });
    expect(client.listCells).not.toHaveBeenCalled();
    expect(client.listAuditEvents).toHaveBeenCalledWith({ orgId: "acme", projectId: "research" }, undefined);
  });

  it("loads fleet inventory only when server authority includes fleet management", async () => {
    const client = dashboardClient();

    const data = await loadDashboardData(
      client,
      { canRead: true, canManageFleet: true },
      { orgId: "acme", projectId: "research" },
    );

    expect(data.cells).toHaveLength(1);
    expect(data.access).toEqual({ fleetInventory: true, fleetAudit: true });
    expect(client.listCells).toHaveBeenCalledOnce();
  });

  it("keeps a typed 403 visible so App can refresh /session instead of preserving stale authority", async () => {
    const client = dashboardClient();
    client.listAuditEvents.mockRejectedValue(new ApiError("authority changed", 403));

    await expect(
      loadDashboardData(client, { canRead: true, canManageFleet: false }, { orgId: "acme", projectId: "research" }),
    ).rejects.toMatchObject({ status: 403 });
  });

  it("does not issue read requests for a submit-only effective scope", async () => {
    const client = dashboardClient();

    const data = await loadDashboardData(client, { canRead: false, canManageFleet: false });

    expect(data.health.status).toBe("ok");
    expect(data.workloads).toEqual([]);
    expect(data.access).toEqual({ fleetInventory: false, fleetAudit: false });
    expect(client.listWorkloads).not.toHaveBeenCalled();
    expect(client.listLeases).not.toHaveBeenCalled();
    expect(client.getQueueTrend).not.toHaveBeenCalled();
  });

  it("keeps queue history failures fatal rather than converting them to empty history", async () => {
    const client = dashboardClient();
    client.getQueueTrend.mockRejectedValue(new ApiError("trend unavailable", 503));

    await expect(loadDashboardData(client)).rejects.toMatchObject({ status: 503 });
  });

  it("does not turn an untyped or non-authorization failure into empty fleet data", async () => {
    const client = dashboardClient();
    const untypedFailure = { status: 403, message: "not an ApiError" };
    client.listCells.mockRejectedValue(untypedFailure);

    await expect(loadDashboardData(client)).rejects.toBe(untypedFailure);
  });
});
