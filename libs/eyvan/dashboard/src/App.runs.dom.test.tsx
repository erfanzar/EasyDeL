// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { ApiError, EyvanApiClient } from "./api/client";
import { mockRuns } from "./api/mockData";

vi.mock("./components/LineChart", () => ({
  LineChart: ({ series }: { series: { key: string } }) => <div role="img" aria-label={`${series.key} chart`} />,
}));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  window.location.hash = "";
});

describe("Runs navigation and authority integration", () => {
  it.each([401, 403])(
    "36: full App keeps tracking %s local without fencing or reconciling authority",
    async (status) => {
      const client = new EyvanApiClient("/api/v1", true);
      const session = await client.getSessionAuthority();
      // A mistaken reconciliation must remain observable instead of repeatedly
      // re-mounting Runs and spinning through the same refusal.
      const authority = vi
        .spyOn(client, "getSessionAuthority")
        .mockResolvedValueOnce(session)
        .mockImplementation(() => new Promise(() => {}));
      const list = vi.spyOn(client, "listRuns").mockRejectedValue(new ApiError("Tracking permission denied", status));
      render(<App client={client} />);
      fireEvent.click(await screen.findByRole("button", { name: "Runs" }));
      await waitFor(() => expect(list).toHaveBeenCalled());
      await act(async () => {});
      expect(authority).toHaveBeenCalledTimes(1);
      expect(await screen.findByText("Tracking permission denied")).toBeDefined();
      expect(screen.getByRole("button", { name: "Submit workload" })).toBeDefined();
    },
  );

  it("opens runs, detail, and the linked workload without changing the existing routes", async () => {
    const client = new EyvanApiClient("/api/v1", true);
    render(<App client={client} />);
    fireEvent.click(await screen.findByRole("button", { name: "Runs" }));
    fireEvent.click(await screen.findByRole("button", { name: mockRuns[0].name }));
    expect(await screen.findByRole("img", { name: "loss/policy chart" })).toBeDefined();
    expect(window.location.hash).toBe("#/runs/pong-seed-3");
    fireEvent.click(screen.getByRole("button", { name: mockRuns[0].workload_id! }));
    expect(await screen.findByRole("heading", { name: "Workload detail" })).toBeDefined();
    await waitFor(() => expect(window.location.hash).toContain("#/workloads/"));
  });

  it("loads a run deep link and keeps a tracker 503 local to Runs", async () => {
    const client = new EyvanApiClient("/api/v1", true);
    window.location.hash = "#/runs/pong-seed-3";
    vi.spyOn(client, "getRun").mockRejectedValue(new ApiError("no tracker", 503));
    render(<App client={client} />);
    expect(await screen.findByText(/No tracker configured/)).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Workloads" }));
    expect(await screen.findByRole("heading", { name: "Workloads" })).toBeDefined();
  });

  it("does not mount tracking reads for a submit-only identity's run deep link", async () => {
    const client = new EyvanApiClient("/api/v1", true);
    const authority = await client.getSessionAuthority();
    authority.globalPermissions = [];
    authority.scopes[0].permissions = ["submit"];
    vi.spyOn(client, "getSessionAuthority").mockResolvedValue(authority);
    const getRun = vi.spyOn(client, "getRun");
    const listRuns = vi.spyOn(client, "listRuns");
    window.location.hash = "#/runs/pong-seed-3";
    render(<App client={client} />);
    await screen.findByRole("button", { name: "Submit workload" });
    expect(screen.queryByRole("button", { name: "Runs" })).toBeNull();
    expect(getRun).not.toHaveBeenCalled();
    expect(listRuns).not.toHaveBeenCalled();
  });
});
