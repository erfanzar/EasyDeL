// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import { accessibleRoute, parseRoute, routeHash } from "./routing";

describe("dashboard routing", () => {
  it("keeps project-scoped routes when fleet panels are unavailable", () => {
    const availability = { fleet: false, audit: false };

    expect(accessibleRoute({ section: "workloads" }, availability)).toEqual({ section: "workloads" });
    expect(accessibleRoute({ section: "workloads", workloadId: "wl/1" }, availability)).toEqual({
      section: "workloads",
      workloadId: "wl/1",
    });
    expect(accessibleRoute({ section: "budgets" }, availability)).toEqual({ section: "budgets" });
  });

  it("falls back from a forbidden fleet deep link without treating navigation as authority", () => {
    expect(accessibleRoute(parseRoute("#/fleet"), { fleet: false })).toEqual({ section: "overview" });
    expect(accessibleRoute(parseRoute("#/audit"), { audit: false })).toEqual({ section: "overview" });
    expect(routeHash({ section: "fleet" })).toBe("#/fleet");
  });

  it("preserves a fleet route when the independently loaded endpoint is available", () => {
    expect(accessibleRoute({ section: "fleet" }, { fleet: true })).toEqual({ section: "fleet" });
  });
});

describe("hash queries", () => {
  it("ignores a notice query after the route", () => {
    expect(parseRoute("#/settings?assistant=openrouter-connected")).toEqual({ section: "settings" });
    expect(parseRoute("#/settings?assistant_error=denied")).toEqual({ section: "settings" });
  });
});

describe("every section the rail offers", () => {
  it("parses back to itself, so pressing it and reloading land in the same place", () => {
    for (const section of ["workloads", "fleet", "capacity", "billing", "budgets", "audit"] as const) {
      expect(parseRoute(routeHash({ section }))).toEqual({ section });
    }
  });
});

describe("run tracking routes", () => {
  it("round trips the list and encoded run identifiers", () => {
    expect(parseRoute("#/runs")).toEqual({ section: "runs" });
    const route = { section: "runs", runId: "ppo/seed 3" } as const;
    expect(routeHash(route)).toBe("#/runs/ppo%2Fseed%203");
    expect(parseRoute(routeHash(route))).toEqual(route);
    expect(parseRoute("#/runs/%E0%A4%A")).toEqual({ section: "runs" });
  });

  it("round trips episode and histogram tabs while preserving the default charts route", () => {
    for (const tab of ["episodes", "histograms"] as const) {
      const route = { section: "runs", runId: "ppo/seed 3", tab } as const;
      expect(routeHash(route)).toBe(`#/runs/ppo%2Fseed%203/${tab}`);
      expect(parseRoute(routeHash(route))).toEqual(route);
    }
    expect(parseRoute("#/runs/run-1/unknown")).toEqual({ section: "runs", runId: "run-1" });
    expect(parseRoute("#/runs/run-1/charts")).toEqual({ section: "runs", runId: "run-1" });
    expect(parseRoute("#/workloads/wl-1/episodes")).toEqual({ section: "workloads", workloadId: "wl-1" });
  });

  it("honours read availability on run deep links", () => {
    expect(accessibleRoute(parseRoute("#/runs/run-1"), { runs: false })).toEqual({ section: "overview" });
  });
});

describe("the quota section", () => {
  it("is a first-class route so a zone choice can be linked to", () => {
    expect(parseRoute("#/capacity")).toEqual({ section: "capacity" });
    expect(routeHash({ section: "capacity" })).toBe("#/capacity");
  });

  it("falls back when the identity cannot read the pools the quota is resolved from", () => {
    // The route resolves its provider project from capacity pool records, so a
    // reader without fleet authority must not mount it and issue a 403.
    expect(accessibleRoute(parseRoute("#/capacity"), { capacity: false })).toEqual({ section: "overview" });
    expect(accessibleRoute(parseRoute("#/capacity"), { capacity: true })).toEqual({ section: "capacity" });
  });
});
