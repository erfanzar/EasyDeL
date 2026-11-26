// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TeamToken } from "../types";
import { ConnectAgentPanel } from "./ConnectAgentPanel";

const RECORD: TeamToken = {
  id: "tok_9",
  name: "agent-codex",
  prefix: "eyv_abcd",
  created_at: "2026-10-27T10:00:00+00:00",
  expires_at: null,
  last_used_at: null,
  revoked_at: null,
};

afterEach(cleanup);

describe("ConnectAgentPanel", () => {
  it("mints a named token for the chosen client and shows its config once", async () => {
    const client = { createTeamToken: vi.fn(async () => ({ token: "eyv_minted", record: RECORD })) };
    const onMinted = vi.fn();
    render(<ConnectAgentPanel client={client} onMinted={onMinted} origin="https://plane.example.com" />);

    expect(screen.queryByTestId("secret-once")).toBeNull();
    fireEvent.change(screen.getByLabelText("Agent"), { target: { value: "codex" } });
    fireEvent.click(screen.getByRole("button", { name: /Create token and show config/ }));

    await waitFor(() => expect(screen.getByTestId("secret-once").textContent).toContain("[mcp_servers.eyvan]"));
    expect(client.createTeamToken).toHaveBeenCalledWith(expect.stringMatching(/^agent-codex-\d{8}-\d{4}$/), 90);
    expect(screen.getByText(/Token expires: in 90 days/)).toBeTruthy();
    const text = screen.getByTestId("secret-once").textContent ?? "";
    expect(text).toContain("https://plane.example.com/api/v1/mcp");
    expect(text).toContain("Bearer eyv_minted");
    expect(screen.getByText("eyvan mcp install --client codex")).toBeTruthy();
    expect(onMinted).toHaveBeenCalled();
  });

  it("passes null for a token that never expires", async () => {
    const client = { createTeamToken: vi.fn(async () => ({ token: "eyv_minted", record: RECORD })) };
    render(<ConnectAgentPanel client={client} origin="https://plane.example.com" />);
    fireEvent.change(screen.getByLabelText("Expires"), { target: { value: "never" } });
    fireEvent.click(screen.getByRole("button", { name: /Create token and show config/ }));
    await waitFor(() => expect(client.createTeamToken).toHaveBeenCalledWith(expect.any(String), null));
    await screen.findByText(/Token expires: never/);
  });

  it("shows an error and no config when minting fails", async () => {
    const client = {
      createTeamToken: vi.fn(async () => {
        throw new Error("nope");
      }),
    };
    render(<ConnectAgentPanel client={client} origin="https://plane.example.com" />);
    fireEvent.click(screen.getByRole("button", { name: /Create token and show config/ }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("nope"));
    expect(screen.queryByTestId("secret-once")).toBeNull();
  });
});
