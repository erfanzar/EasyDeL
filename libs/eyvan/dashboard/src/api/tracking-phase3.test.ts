// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";
import { EyvanApiClient } from "./client";
import { parseRoute, routeHash } from "../routing";

vi.mock("../auth/oidc", async (original) => ({
  ...(await original<typeof import("../auth/oidc")>()),
  apiAuthHeaders: () => ({ Authorization: "Bearer tracking-test" }),
}));
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
const scope = { orgId: "org / 1", projectId: "rl&eval" };

describe("phase-3 tracking transport", () => {
  function delayedBody(ms: number) {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        const signal = init.signal!;
        signals.push(signal);
        return {
          ok: true,
          blob: () =>
            new Promise<Blob>((resolve, reject) => {
              const timer = setTimeout(() => resolve(new Blob(["video"])), ms);
              signal.addEventListener(
                "abort",
                () => {
                  clearTimeout(timer);
                  reject(new DOMException("Aborted", "AbortError"));
                },
                { once: true },
              );
            }),
          json: () =>
            new Promise((_resolve, reject) => {
              signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
            }),
        };
      }),
    );
    return signals;
  }

  it("37: permits four-second media bodies without extending DB read deadlines", async () => {
    vi.useFakeTimers();
    const signals = delayedBody(4000);
    const client = new EyvanApiClient("/api/v1", false, scope);
    const media = client.fetchRunBlob("run", "digest");
    const read = expect(client.listRuns()).rejects.toThrow("timed out");
    const body = expect(media).resolves.toBeInstanceOf(Blob);
    await vi.advanceTimersByTimeAsync(4000);
    await read;
    await body;
    expect(signals[0].aborted).toBe(false);
    expect(signals[1].aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each(["external", "deadline"])("37: media %s cancellation aborts body reads and clears timers", async (mode) => {
    vi.useFakeTimers();
    const signals = delayedBody(150000);
    const controller = new AbortController();
    const client = new EyvanApiClient("/api/v1", false, scope);
    const result = expect(client.fetchRunBlob("run", "digest", scope, controller.signal)).rejects.toThrow("timed out");
    await vi.advanceTimersByTimeAsync(0);
    if (mode === "external") controller.abort();
    else {
      await vi.advanceTimersByTimeAsync(119999);
      expect(signals[0].aborted).toBe(false);
      await vi.advanceTimersByTimeAsync(1);
    }
    await result;
    expect(signals[0].aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("round trips encoded run routes for all new tabs", () => {
    for (const tab of ["rollouts", "completions", "media"] as const) {
      const route = { section: "runs", runId: "run / 1", tab } as const;
      expect(parseRoute(routeHash(route))).toEqual(route);
    }
  });

  it("encodes identifiers, scope and list parameters on every plane route", async () => {
    const fetch = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ marker: "wire" })));
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false, scope);
    const results = [
      await client.listRunRollouts("run / 1", {
        kind: "eval",
        actor: "a / 1",
        limit: 12,
        offset: 3,
        includeReplayed: true,
      }),
      await client.getRunRollout("run / 1", "roll / 2"),
      await client.listRunCompletions("run / 1", {
        step: 0,
        limit: 8,
        offset: 2,
        sort: "zero_variance",
        includeReplayed: true,
      }),
      await client.getRunCompletion("run / 1", "group / 2"),
      await client.getCompletionStats("run / 1", { points: 42 }),
      await client.getRunTable("run / 1", { key: "eval/cases & more", step: 0 }),
      await client.getRunTable("run / 1"),
      await client.listRunMedia("run / 1", { key: "eval/image", limit: 9, offset: 1 }),
    ];
    expect(results.every((result) => "marker" in result)).toBe(true);
    const suffixes = [
      "rollouts",
      "rollouts/roll%20%2F%202",
      "completions",
      "completions/group%20%2F%202",
      "completions/stats",
      "tables",
      "tables",
      "media",
    ];
    fetch.mock.calls.forEach(([url], i) => {
      const parsed = new URL(url, "http://plane.test");
      expect(parsed.pathname).toBe(`/api/v1/tracking/runs/run%20%2F%201/${suffixes[i]}`);
      expect(parsed.searchParams.get("org_id")).toBe(scope.orgId);
      expect(parsed.searchParams.get("project_id")).toBe(scope.projectId);
    });
    const query = (i: number) => Object.fromEntries(new URL(fetch.mock.calls[i][0], "http://plane.test").searchParams);
    expect(query(0)).toMatchObject({
      kind: "eval",
      actor: "a / 1",
      limit: "12",
      offset: "3",
      include_replayed: "true",
    });
    expect(query(2)).toMatchObject({
      step: "0",
      limit: "8",
      offset: "2",
      sort: "zero_variance",
      include_replayed: "true",
    });
    expect(query(4)).toMatchObject({ points: "42" });
    expect(query(5)).toMatchObject({ key: "eval/cases & more", step: "0" });
    expect(query(6)).not.toHaveProperty("key");
    expect(query(7)).toMatchObject({ key: "eval/image", limit: "9", offset: "1" });
  });

  it("fetches binary bytes with the same bearer, credentials and cancellation path as JSON", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ runs: [] })))
      .mockResolvedValueOnce(new Response(new Uint8Array([1, 2, 3]), { headers: { "Content-Type": "image/png" } }));
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", false, scope);
    await client.listRuns();
    const controller = new AbortController();
    const blob = await client.fetchRunBlob("run / 1", "sha / 2", scope, controller.signal);
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe("image/png");
    expect([...new Uint8Array(await blob.arrayBuffer())]).toEqual([1, 2, 3]);
    const [url, init] = fetch.mock.calls[1];
    expect(new URL(url, "http://plane.test").pathname).toBe("/api/v1/tracking/runs/run%20%2F%201/blobs/sha%20%2F%202");
    expect(new URL(url, "http://plane.test").searchParams.get("project_id")).toBe(scope.projectId);
    expect(init.headers.Authorization).toBe(fetch.mock.calls[0][1].headers.Authorization);
    expect(init.credentials).toBe(fetch.mock.calls[0][1].credentials);
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });
});
