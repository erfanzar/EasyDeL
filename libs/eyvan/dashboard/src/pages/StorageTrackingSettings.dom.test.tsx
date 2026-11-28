// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, EyvanApiClient } from "../api/client";
import type { StorageStatus } from "../types";
import { SettingsPage } from "./SettingsPage";
import { StorageTrackingSettings } from "./StorageTrackingSettings";
import type { AssistantSettingsClient } from "./AssistantProvidersSettings";
import type { TeamClient } from "./UsersPage";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const SAMPLE_STORAGE: StorageStatus = {
  configured: true,
  auto_create: false,
  config_path: "/Users/operator/.eyvan/storage.json",
  project: "example-project-a",
  monitoring: {
    bucket: "eyvan-monitoring-example-project-a",
    uri: "gs://eyvan-monitoring-example-project-a/tracking",
    location: "us-central1",
    prefix: "tracking",
  },
  regional: {
    "us-central2": {
      bucket: "eyvan_us-central2",
      uri: "gs://eyvan_us-central2",
      location: "us-central2",
    },
    "europe-west4": {
      bucket: "eyvan_europe-west4",
      uri: "gs://eyvan_europe-west4",
      location: "europe-west4",
    },
  },
  context: {
    on_gcp: false,
    project: "example-project-a",
    zone: "us-central2-b",
    region: "us-central2",
    accelerator_type: null,
    source: "storage-config",
  },
};

describe("StorageTrackingSettings", () => {
  it("renders configured regional and monitoring buckets, tracker health, and copyable CLI commands", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    const onNotify = vi.fn();

    render(
      <StorageTrackingSettings
        client={{
          getStorageStatus: vi.fn(async () => SAMPLE_STORAGE),
          listRuns: vi.fn(async () => []),
        }}
        defaultOrg="default"
        defaultProject="default"
        onNotify={onNotify}
      />,
    );

    const panel = screen.getByRole("region", { name: "Storage and experiment tracking" });
    expect(await within(panel).findByText("gs://eyvan-monitoring-example-project-a/tracking")).toBeDefined();
    expect(within(panel).getByText("eyvan_us-central2")).toBeDefined();
    expect(within(panel).getByText("eyvan_europe-west4")).toBeDefined();
    expect(within(panel).getByText("example-project-a")).toBeDefined();
    expect(await within(panel).findByText("Tracker Online")).toBeDefined();

    // Copy CLI command button works with feedback.
    fireEvent.click(within(panel).getByRole("button", { name: "Copy eyvan track pull --watch 5" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("eyvan track pull --watch 5"));
    expect(onNotify).toHaveBeenCalledWith('Copied "eyvan track pull --watch 5"');

    // Copy regional bucket URI works too.
    fireEvent.click(within(panel).getByRole("button", { name: "Copy us-central2 bucket URI" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("gs://eyvan_us-central2"));
  });

  it("handles 404 from older servers gracefully without rendering an error alert", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input) === "/api/v1/storage/status") {
        return new Response(JSON.stringify({ detail: "Not Found" }), { status: 404 });
      }
      return new Response(JSON.stringify({ detail: "No tracker configured" }), { status: 503 });
    });

    const client = new EyvanApiClient("/api/v1", false);
    render(<StorageTrackingSettings client={client} defaultOrg="default" defaultProject="default" />);

    expect(await screen.findByText(/does not expose/)).toBeDefined();
    expect(screen.queryByRole("alert")).toBeNull();
    // Quick CLI commands stay available even on older servers.
    expect(screen.getByText("eyvan storage init")).toBeDefined();
    expect(screen.getByText("eyvan track pull --watch 5")).toBeDefined();
  });

  it("surfaces tracker 503 status cleanly and recovers storage read errors via Retry", async () => {
    const getStorageStatus = vi
      .fn<() => Promise<StorageStatus | null>>()
      .mockRejectedValueOnce(new ApiError("Eyvan API request failed with 500: boom", 500))
      .mockResolvedValueOnce(SAMPLE_STORAGE);

    render(
      <StorageTrackingSettings
        client={{
          getStorageStatus,
          listRuns: vi.fn(async () => {
            throw new ApiError("No tracker configured", 503);
          }),
        }}
        defaultOrg="default"
        defaultProject="default"
      />,
    );

    expect(await screen.findByRole("alert")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("eyvan_us-central2")).toBeDefined();
    expect(await screen.findByText("Tracker Offline")).toBeDefined();
  });

  it("mounts inside SettingsPage for fleet admins", async () => {
    const teamClient: TeamClient = {
      listTeamUsers: vi.fn(async () => []),
      setTeamUserDisabled: vi.fn(),
      listTeamInvites: vi.fn(async () => []),
      createTeamInvite: vi.fn(),
      revokeTeamInvite: vi.fn(),
      listTeamTokens: vi.fn(async () => []),
      createTeamToken: vi.fn(),
      revokeTeamToken: vi.fn(),
    };
    const assistantClient: AssistantSettingsClient = {
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

    render(
      <SettingsPage
        client={{
          ...teamClient,
          ...assistantClient,
          getStorageStatus: vi.fn(async () => SAMPLE_STORAGE),
          listRuns: vi.fn(async () => []),
        }}
        teamMode={false}
        canManageFleet
        defaultOrg="default"
        defaultProject="default"
      />,
    );

    expect(await screen.findByRole("region", { name: "Storage and experiment tracking" })).toBeDefined();
    expect(await screen.findByText("eyvan_us-central2")).toBeDefined();
  });
});
