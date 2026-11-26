// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { EyvanApiClient } from "./api/client";
import { accessibleRoute, parseRoute, routeHash } from "./routing";

const malformedHashes = ["#/workloads/%", "#/workloads/%E0%A4%A", "#/workloads/%FF"];

function replaceHash(hash: string) {
  window.history.replaceState(null, "", hash);
}

function demoClient() {
  return new EyvanApiClient("/api/v1", true, { orgId: "acme", projectId: "research" });
}

const noBrowserIdentity = () => undefined;

afterEach(() => {
  cleanup();
  replaceHash("#/overview");
  vi.restoreAllMocks();
});

describe("malformed workload deep links", () => {
  it.each(malformedHashes)("falls back to the workload list for %s", (hash) => {
    expect(parseRoute(hash)).toEqual({ section: "workloads" });
    expect(accessibleRoute(parseRoute(hash), { workloads: false })).toEqual({ section: "overview" });
  });

  it.each(["wl/1", "wl%1", "wl-日本語"])("preserves a valid encoded workload ID %s", (workloadId) => {
    expect(parseRoute(routeHash({ section: "workloads", workloadId }))).toEqual({
      section: "workloads",
      workloadId,
    });
  });

  it.each(malformedHashes)("mounts a usable App from %s without loading an invalid detail", async (hash) => {
    replaceHash(hash);
    const client = demoClient();
    const getWorkload = vi.spyOn(client, "getWorkload");
    render(<App client={client} identityProvider={noBrowserIdentity} />);

    expect(await screen.findByRole("heading", { name: "Workloads" })).toBeDefined();
    expect(await screen.findByRole("table")).toBeDefined();
    expect(getWorkload).not.toHaveBeenCalled();
  });

  it.each(malformedHashes)("handles a later hashchange to %s without an uncaught error", async (hash) => {
    replaceHash("#/overview");
    const client = demoClient();
    const getWorkload = vi.spyOn(client, "getWorkload");
    render(<App client={client} identityProvider={noBrowserIdentity} />);
    expect(await screen.findByRole("heading", { name: "Operations overview" })).toBeDefined();

    const errors: unknown[] = [];
    const onError = (event: ErrorEvent) => {
      errors.push(event.error);
      event.preventDefault();
    };
    window.addEventListener("error", onError);
    try {
      act(() => {
        replaceHash(hash);
        window.dispatchEvent(new HashChangeEvent("hashchange"));
      });
      expect(errors).toEqual([]);
      expect(await screen.findByRole("heading", { name: "Workloads" })).toBeDefined();
      expect(await screen.findByRole("table")).toBeDefined();
      expect(getWorkload).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener("error", onError);
    }
  });
});
