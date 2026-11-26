// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";
import { EyvanApiClient } from "./client";
import { mockRuns, getMockRun, getMockRunSeries } from "./mockData";

const scope = { orgId: "org / 1", projectId: "rl&eval" };
afterEach(() => vi.unstubAllGlobals());

describe("tracking API", () => {
  it("uses scoped plane paths and preserves the JSON tracking contract", async () => {
    const detail = getMockRun(mockRuns[0].run_id)!;
    const series = getMockRunSeries(detail, detail.keys.slice(0, 2), "step", 1000);
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ runs: mockRuns })))
      .mockResolvedValueOnce(new Response(JSON.stringify(detail)))
      .mockResolvedValueOnce(new Response(JSON.stringify(series)));
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false, scope);
    expect(await client.listRuns({ status: "unknown" })).toEqual(mockRuns);
    expect(await client.getRun("run / 1")).toEqual(detail);
    expect(
      await client.getRunSeries("run / 1", { keys: ["loss/policy", "reward/mean"], x: "env_steps", points: 500 }),
    ).toEqual(series);
    const list = new URL(fetch.mock.calls[0][0], "http://plane.test");
    expect(list.pathname).toBe("/api/v1/tracking/runs");
    expect(list.searchParams.get("status")).toBe("unknown");
    for (const [url, init] of fetch.mock.calls) {
      const parsed = new URL(url, "http://plane.test");
      expect(parsed.searchParams.get("org_id")).toBe(scope.orgId);
      expect(parsed.searchParams.get("project_id")).toBe(scope.projectId);
      expect(init.headers.Accept).toBe("application/json");
      expect(init.signal).toBeInstanceOf(AbortSignal);
    }
    const query = new URL(fetch.mock.calls[2][0], "http://plane.test");
    expect(query.pathname).toBe("/api/v1/tracking/runs/run%20%2F%201/series");
    expect(query.searchParams.getAll("key")).toEqual(["loss/policy", "reward/mean"]);
    expect(query.searchParams.get("x")).toBe("env_steps");
    expect(query.searchParams.get("points")).toBe("500");
  });

  it("builds scoped, encoded phase-2 URLs and preserves every response field", async () => {
    const responses = [
      {
        run_id: "run / 1",
        key: "advantages",
        columns: [],
        edges: [],
        total: 0,
        downsampled: false,
        rank: 0,
        x: "time",
      },
      { episodes: [{ ret: "nan", replayed: true, actor: "actor / 3" }], total: 1 },
      { run_id: "run / 1", buckets: [], total: 0, nonfinite: 2, replayed_excluded: 3 },
      { actors: [], envs: [], lag: { counts: Array(16).fill(0) }, evals: [], replayed_excluded: 3 },
    ];
    const fetch = vi.fn();
    responses.forEach((response) => fetch.mockResolvedValueOnce(new Response(JSON.stringify(response))));
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false, scope);
    expect(
      await client.getRunHistograms("run / 1", { key: "policy/advantages", x: "time", points: 42, bins: 32, rank: 0 }),
    ).toEqual(responses[0]);
    expect(
      await client.listRunEpisodes("run / 1", {
        kind: "eval",
        actor: "actor / 3",
        env: "Pong&v5",
        limit: 50,
        offset: 0,
        includeReplayed: true,
      }),
    ).toEqual(responses[1]);
    expect(
      await client.getEpisodeCurve("run / 1", {
        metric: "length",
        kind: "eval",
        x: "env_steps",
        points: 99,
        env: "Pong&v5",
        actor: "actor / 3",
      }),
    ).toEqual(responses[2]);
    expect(await client.getEpisodeBreakdown("run / 1", { kind: "eval", last: 75 })).toEqual(responses[3]);
    const paths = ["histograms", "episodes", "episodes/curve", "episodes/breakdown"];
    fetch.mock.calls.forEach(([url], i) => {
      const parsed = new URL(url, "http://plane.test");
      expect(parsed.pathname).toBe(`/api/v1/tracking/runs/run%20%2F%201/${paths[i]}`);
      expect(parsed.searchParams.get("org_id")).toBe(scope.orgId);
      expect(parsed.searchParams.get("project_id")).toBe(scope.projectId);
    });
    const query = (i: number) => new URL(fetch.mock.calls[i][0], "http://plane.test").searchParams;
    expect(Object.fromEntries(query(0))).toMatchObject({
      key: "policy/advantages",
      x: "time",
      points: "42",
      bins: "32",
      rank: "0",
    });
    expect(Object.fromEntries(query(1))).toMatchObject({
      kind: "eval",
      actor: "actor / 3",
      env: "Pong&v5",
      limit: "50",
      offset: "0",
      include_replayed: "true",
    });
    expect(Object.fromEntries(query(2))).toMatchObject({
      metric: "length",
      kind: "eval",
      x: "env_steps",
      points: "99",
      env: "Pong&v5",
      actor: "actor / 3",
    });
    expect(query(3).get("last")).toBe("75");
  });

  it("keeps tracker unavailability actionable rather than replacing it with demo data", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "No tracker configured" }), { status: 503 })),
    );
    await expect(new EyvanApiClient("/api/v1", false).listRuns()).rejects.toMatchObject({ status: 503 });
  });

  it("uses scoped deterministic demo runs without network calls", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", true, scope);
    const runs = await client.listRuns();
    expect(runs.length).toBeGreaterThan(0);
    expect(runs[0]).toMatchObject({ org_id: scope.orgId, project_id: scope.projectId });
    const run = await client.getRun(runs[0].run_id);
    const series = await client.getRunSeries(run.run_id, { keys: run.keys, x: "time", points: 10 });
    expect(series.x).toBe("time");
    expect(series.series[0].points.length).toBe(10);
    expect(series.series[0].downsampled).toBe(true);
    expect(fetch).not.toHaveBeenCalled();
  });
});
