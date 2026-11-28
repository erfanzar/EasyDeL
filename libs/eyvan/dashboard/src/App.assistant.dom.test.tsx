// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { EyvanApiClient } from "./api/client";
import { DOCK_OPEN_KEY } from "./components/assistantDockState";

vi.mock("./components/LineChart", () => ({
  LineChart: () => <div />,
}));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
  sessionStorage.clear();
  window.location.hash = "";
});

const dock = () => screen.queryByRole("complementary", { name: "Assistant panel" });

describe("assistant dock in the app shell", () => {
  it("toggles from the topbar and Ctrl/Cmd+J, and remembers being open", async () => {
    render(<App client={new EyvanApiClient("/api/v1", true)} />);
    const toggle = await screen.findByRole("button", { name: "Assistant" });
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    expect(dock()).toBeNull();
    // The rail no longer has an Assistant section.
    expect(screen.queryByRole("button", { name: "Assistant", pressed: undefined })).toBe(toggle);

    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    expect(dock()).not.toBeNull();
    await screen.findByText("No assistant provider is configured yet.");
    expect(localStorage.getItem(DOCK_OPEN_KEY)).toBe("1");

    fireEvent.keyDown(window, { key: "j", ctrlKey: true });
    expect(dock()).toBeNull();
    expect(localStorage.getItem(DOCK_OPEN_KEY)).toBe("0");
    fireEvent.keyDown(window, { key: "j", metaKey: true });
    expect(dock()).not.toBeNull();
    cleanup();

    render(<App client={new EyvanApiClient("/api/v1", true)} />);
    await screen.findByRole("button", { name: "Assistant", pressed: true });
    expect(dock()).not.toBeNull();
  });

  it("opens the dock over the overview for an old #/assistant link", async () => {
    window.location.hash = "#/assistant";
    render(<App client={new EyvanApiClient("/api/v1", true)} />);
    await waitFor(() => expect(dock()).not.toBeNull());
    await waitFor(() => expect(window.location.hash).toBe("#/overview"));
  });
});
