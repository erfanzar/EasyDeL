// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EyvanApiClient } from "../api/client";
import type { GovernanceSummary, TeamInvite, TeamToken, TeamUser } from "../types";
import { inviteLoginCommand, UsersPage, type TeamClient } from "./UsersPage";

const USER: TeamUser = {
  subject: "alice",
  email: "alice@example.com",
  display_name: "Alice",
  disabled: false,
  created_at: "2026-10-27T10:00:00+00:00",
  created_by: "root",
};
const TOKEN: TeamToken = {
  id: "tok_1",
  name: "laptop",
  prefix: "eyv_abcd",
  created_at: "2026-10-27T10:00:00+00:00",
  expires_at: null,
  last_used_at: null,
  revoked_at: null,
};
const INVITE: TeamInvite = {
  id: "inv_1",
  email: "carol@example.com",
  subject: "carol",
  roles: [{ role: "project_viewer", org_id: "acme", project_id: "r" }],
  created_by: "alice",
  created_at: "2026-10-27T10:00:00+00:00",
  expires_at: "2999-01-01T00:00:00+00:00",
  accepted_at: null,
  revoked: false,
};

function mockClient(): TeamClient {
  return {
    listTeamUsers: vi.fn(async () => [USER]),
    setTeamUserDisabled: vi.fn(async () => ({ ...USER, disabled: true })),
    listTeamInvites: vi.fn(async () => [INVITE]),
    createTeamInvite: vi.fn(async () => ({
      invite: { ...INVITE, id: "inv_2", email: "bob@example.com", subject: "bob" },
      url: "https://plane.example.com/#/accept-invite/secret-link",
      token: "secret-link",
    })),
    revokeTeamInvite: vi.fn(async () => ({ ...INVITE, revoked: true })),
    listTeamTokens: vi.fn(async () => [TOKEN]),
    createTeamToken: vi.fn(async () => ({ token: "eyv_newsecret", record: { ...TOKEN, id: "tok_2", name: "ci" } })),
    revokeTeamToken: vi.fn(async () => ({ ...TOKEN, revoked_at: "2026-10-27T11:00:00+00:00" })),
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Settings → Users & tokens", () => {
  it("lets a fleet admin see users and invites, invite someone and copy the link once", async () => {
    const client = mockClient();
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);

    const users = await screen.findByRole("region", { name: "Users" });
    await within(users).findByText("alice@example.com");
    await within(screen.getByRole("region", { name: "Invites" })).findByText("carol@example.com");

    const invite = screen.getByRole("region", { name: "Invite a user" });
    fireEvent.change(within(invite).getByLabelText("Email"), { target: { value: "Bob@example.com" } });
    fireEvent.click(within(invite).getByRole("button", { name: /Create invite/ }));

    await waitFor(() =>
      expect(within(invite).getByTestId("secret-once").textContent).toBe(
        "https://plane.example.com/#/accept-invite/secret-link",
      ),
    );
    expect(client.createTeamInvite).toHaveBeenCalledWith({
      email: "Bob@example.com",
      subject: "bob",
      tier: "user",
      org_id: "acme",
      project_id: "r",
    });
  });

  it("invites with a tier, or with a custom role under Advanced, never both", async () => {
    const client = mockClient();
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);
    const invite = await screen.findByRole("region", { name: "Invite a user" });
    fireEvent.change(within(invite).getByLabelText("Email"), { target: { value: "ann@example.com" } });
    fireEvent.change(within(invite).getByLabelText("Tier"), { target: { value: "admin" } });
    fireEvent.click(within(invite).getByRole("button", { name: /Create invite/ }));
    await waitFor(() =>
      expect(client.createTeamInvite).toHaveBeenCalledWith({
        email: "ann@example.com",
        subject: "ann",
        tier: "admin",
        org_id: null,
        project_id: null,
      }),
    );

    fireEvent.change(within(invite).getByLabelText("Email"), { target: { value: "dan@example.com" } });
    fireEvent.change(within(invite).getByLabelText("Role"), { target: { value: "project_viewer" } });
    fireEvent.click(within(invite).getByRole("button", { name: /Create invite/ }));
    await waitFor(() =>
      expect(client.createTeamInvite).toHaveBeenLastCalledWith({
        email: "dan@example.com",
        subject: "dan",
        role: "project_viewer",
        org_id: "acme",
        project_id: "r",
      }),
    );
  });

  it("changes a user's tier inline with the default project scope", async () => {
    const client = mockClient();
    vi.mocked(client.listTeamUsers).mockResolvedValue([{ ...USER, tier: "user" }]);
    client.setTeamUserTier = vi.fn(async () => ({ ...USER, tier: "trusted" as const }));
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);
    const select = await screen.findByLabelText("Tier for alice");
    fireEvent.change(select, { target: { value: "trusted" } });
    await waitFor(() => expect(client.setTeamUserTier).toHaveBeenCalledWith("alice", "trusted", "acme", "r"));
  });

  it("asks before making someone admin: confirm applies, cancel changes nothing", async () => {
    const client = mockClient();
    vi.mocked(client.listTeamUsers).mockResolvedValue([{ ...USER, tier: "user" }]);
    client.setTeamUserTier = vi.fn(async () => ({ ...USER, tier: "admin" as const }));
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);
    const select = (await screen.findByLabelText("Tier for alice")) as HTMLSelectElement;

    fireEvent.change(select, { target: { value: "admin" } });
    const dialog = await screen.findByRole("alertdialog", { name: "Confirm admin for alice" });
    expect(dialog.textContent).toContain("Make alice a fleet admin?");
    expect(client.setTeamUserTier).not.toHaveBeenCalled();
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(client.setTeamUserTier).not.toHaveBeenCalled();
    expect((screen.getByLabelText("Tier for alice") as HTMLSelectElement).value).toBe("user");

    fireEvent.change(screen.getByLabelText("Tier for alice"), { target: { value: "admin" } });
    fireEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Confirm" }));
    await waitFor(() => expect(client.setTeamUserTier).toHaveBeenCalledWith("alice", "admin", "acme", "r"));
  });

  it("summarizes resource caps including zero and saves fractional CPU alongside GPU and TPU caps", async () => {
    const client = mockClient();
    const withCap: TeamUser = {
      ...USER,
      rules: {
        allowed_fleets: null,
        approval_tpu_threshold: null,
        guaranteed_tpu_units: null,
        max_tpu_units: 16,
        max_cpu_units: 4,
        max_gpu_units: 0,
      },
    };
    vi.mocked(client.listTeamUsers).mockResolvedValue([withCap]);
    client.setTeamUserRules = vi.fn(async () => withCap);
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);
    const users = await screen.findByRole("region", { name: "Users" });
    await waitFor(() =>
      expect(within(users).getByTestId("rules-alice").textContent).toBe("max 16 TPU; max 4 CPU; max 0 GPU"),
    );
    fireEvent.click(within(users).getByRole("button", { name: "Rules" }));
    const form = await within(users).findByRole("form", { name: "Rules for alice" });
    fireEvent.change(within(form).getByLabelText(/Max TPU units/), { target: { value: "32" } });
    fireEvent.change(within(form).getByLabelText(/Max CPU units/), { target: { value: "0.5" } });
    fireEvent.change(within(form).getByLabelText(/Max GPU units/), { target: { value: "3" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save rules" }));
    await waitFor(() =>
      expect(client.setTeamUserRules).toHaveBeenCalledWith("alice", {
        max_tpu_units: 32,
        max_cpu_units: 0.5,
        max_gpu_units: 3,
      }),
    );
  });

  it.each([
    { denied: true, toggle: false, expected: { max_cpu_units: 0 } },
    { denied: true, toggle: true, expected: { allowed_fleets: null, max_cpu_units: 0 } },
    { denied: false, toggle: true, expected: { allowed_fleets: [], max_cpu_units: 0 } },
  ])("patches only edits when deny-all starts $denied and toggles $toggle", async ({ denied, toggle, expected }) => {
    const client = mockClient();
    const account = {
      ...USER,
      rules: {
        allowed_fleets: denied ? [] : null,
        max_cpu_units: 4,
        max_gpu_units: 2,
        max_tpu_units: 8,
        guaranteed_tpu_units: null,
        approval_tpu_threshold: null,
      },
    };
    vi.mocked(client.listTeamUsers).mockResolvedValue([account]);
    client.setTeamUserRules = vi.fn(async () => account);
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);
    fireEvent.click(await screen.findByRole("button", { name: "Rules" }));
    const form = within(await screen.findByRole("form", { name: "Rules for alice" }));
    const checkbox = form.getByRole("checkbox", { name: "Deny all fleets" }) as HTMLInputElement;
    expect(checkbox.checked).toBe(denied);
    if (toggle) fireEvent.click(checkbox);
    fireEvent.change(form.getByLabelText(/Max CPU units/), { target: { value: "0" } });
    fireEvent.click(form.getByRole("button", { name: "Save rules" }));
    await waitFor(() => expect(client.setTeamUserRules).toHaveBeenCalledWith("alice", expected));
  });

  it("disables a user and revokes an open invite", async () => {
    const client = mockClient();
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);

    const users = await screen.findByRole("region", { name: "Users" });
    fireEvent.click(await within(users).findByRole("button", { name: "Disable" }));
    await waitFor(() => expect(client.setTeamUserDisabled).toHaveBeenCalledWith("alice", true));

    const invites = screen.getByRole("region", { name: "Invites" });
    fireEvent.click(await within(invites).findByRole("button", { name: "Revoke" }));
    await waitFor(() => expect(client.revokeTeamInvite).toHaveBeenCalledWith("inv_1"));
  });

  it("shows a non-admin only their own tokens, and a new token exactly once", async () => {
    const client = mockClient();
    render(<UsersPage client={client} canAdminister={false} defaultOrg="acme" defaultProject="r" />);

    const tokens = await screen.findByRole("region", { name: "My API tokens" });
    await within(tokens).findByText("laptop");
    expect(screen.queryByRole("region", { name: "Invite a user" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Users" })).toBeNull();
    expect(client.listTeamUsers).not.toHaveBeenCalled();
    expect(client.listTeamInvites).not.toHaveBeenCalled();

    fireEvent.change(within(tokens).getByLabelText("Token name"), { target: { value: "ci" } });
    fireEvent.click(within(tokens).getByRole("button", { name: /Create token/ }));
    await waitFor(() => expect(within(tokens).getByTestId("secret-once").textContent).toBe("eyv_newsecret"));
    expect(client.createTeamToken).toHaveBeenCalledWith("ci", null);

    fireEvent.click(within(tokens).getAllByRole("button", { name: "Revoke" })[0]);
    await waitFor(() => expect(client.revokeTeamToken).toHaveBeenCalledWith("tok_1"));
  });

  it("surfaces a refused read instead of an empty page", async () => {
    const client = mockClient();
    vi.mocked(client.listTeamUsers).mockRejectedValueOnce(new Error("Eyvan API request failed with 403"));
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);

    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("403"));
  });

  it("shows each user's rules and lets an admin replace them", async () => {
    const client = mockClient();
    const withRules: TeamUser = {
      ...USER,
      rules: {
        allowed_fleets: ["research"],
        approval_tpu_threshold: 8,
        guaranteed_tpu_units: null,
        max_tpu_units: null,
        max_cpu_units: null,
        max_gpu_units: null,
      },
    };
    vi.mocked(client.listTeamUsers).mockResolvedValue([withRules]);
    client.setTeamUserRules = vi.fn(async () => withRules);
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);

    const users = await screen.findByRole("region", { name: "Users" });
    await waitFor(() =>
      expect(within(users).getByTestId("rules-alice").textContent).toBe("fleets: research; approval above 8 TPU"),
    );
    fireEvent.click(within(users).getByRole("button", { name: "Rules" }));
    const form = await within(users).findByRole("form", { name: "Rules for alice" });
    fireEvent.change(within(form).getByLabelText(/Allowed fleets/), { target: { value: "research, prod" } });
    fireEvent.change(within(form).getByLabelText(/Needs approval above/), { target: { value: "" } });
    fireEvent.change(within(form).getByLabelText(/Guaranteed TPU units/), { target: { value: "32" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save rules" }));
    await waitFor(() =>
      expect(client.setTeamUserRules).toHaveBeenCalledWith("alice", {
        allowed_fleets: ["research", "prod"],
        approval_tpu_threshold: null,
        guaranteed_tpu_units: 32,
      }),
    );
  });

  it("refuses a non-number rule without calling the plane", async () => {
    const client = mockClient();
    client.setTeamUserRules = vi.fn(async () => USER);
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);

    const users = await screen.findByRole("region", { name: "Users" });
    fireEvent.click(await within(users).findByRole("button", { name: "Rules" }));
    const form = await within(users).findByRole("form", { name: "Rules for alice" });
    expect(within(users).getByTestId("rules-alice").textContent).toBe("no rules");
    fireEvent.change(within(form).getByLabelText(/Needs approval above/), { target: { value: "lots" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save rules" }));
    await within(form).findByRole("alert");
    expect(client.setTeamUserRules).not.toHaveBeenCalled();
  });

  it("renders with the real API client class (methods are called on the client, not detached)", async () => {
    const client = new EyvanApiClient("/api/v1", true);
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);

    await screen.findByText("No accounts yet. Invite someone above.");
    expect(screen.getByText("No personal tokens.")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("renders the team quota and borrowing KPI strip, tier borrowing badges, and usage-vs-guarantee bars", async () => {
    const client = mockClient();
    const admin: TeamUser = {
      ...USER,
      subject: "admin",
      email: "admin@example.com",
      tier: "admin",
      rules: {
        allowed_fleets: null,
        approval_tpu_threshold: null,
        guaranteed_tpu_units: 64,
        max_tpu_units: null,
        max_cpu_units: null,
        max_gpu_units: null,
      },
    };
    const alice: TeamUser = {
      ...USER,
      subject: "alice",
      email: "alice@example.com",
      tier: "trusted",
      rules: {
        allowed_fleets: null,
        approval_tpu_threshold: null,
        guaranteed_tpu_units: 64,
        max_tpu_units: 256,
        max_cpu_units: null,
        max_gpu_units: null,
      },
    };
    const bob: TeamUser = {
      ...USER,
      subject: "bob",
      email: "bob@example.com",
      tier: "user",
      rules: {
        allowed_fleets: null,
        approval_tpu_threshold: null,
        guaranteed_tpu_units: 128,
        max_tpu_units: 128,
        max_cpu_units: null,
        max_gpu_units: null,
      },
    };
    const summary: GovernanceSummary = {
      enabled: true,
      total_guaranteed_tpu_units: 256,
      total_in_use_tpu_units: 192,
      total_borrowed_tpu_units: 64,
      users: [
        {
          subject: "admin",
          tier: "admin",
          guaranteed_tpu_units: 64,
          max_tpu_units: null,
          can_borrow: null,
          in_use_tpu_units: 32,
          borrowed_tpu_units: 0,
        },
        {
          subject: "alice",
          tier: "trusted",
          guaranteed_tpu_units: 64,
          max_tpu_units: 256,
          can_borrow: true,
          in_use_tpu_units: 128,
          borrowed_tpu_units: 64,
        },
        {
          subject: "bob",
          tier: "user",
          guaranteed_tpu_units: 128,
          max_tpu_units: 128,
          can_borrow: false,
          in_use_tpu_units: 32,
          borrowed_tpu_units: 0,
        },
      ],
      active_reclaims: [
        {
          workload_id: "wl-alice-borrowed",
          owner: "alice",
          requested_by: "bob",
          deadline_at: new Date(Date.now() + 500_000).toISOString(),
          policy: "requeue",
          state: "checkpointing",
        },
      ],
    };
    vi.mocked(client.listTeamUsers).mockResolvedValue([admin, alice, bob]);
    client.getGovernanceSummary = vi.fn(async () => summary);

    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);

    const kpis = await screen.findByRole("region", { name: "Team quota and borrowing summary" });
    expect(within(kpis).getByText("Guaranteed TPUs")).toBeTruthy();
    expect(within(kpis).getByText("256")).toBeTruthy();
    expect(within(kpis).getByText("192")).toBeTruthy();
    expect(within(kpis).getByText("64")).toBeTruthy();
    expect(within(kpis).getByText("reclaimable")).toBeTruthy();
    expect(within(kpis).getByText("Active reclaims")).toBeTruthy();

    const activeReclaims = await screen.findByRole("region", { name: "Active team reclaims" });
    expect(activeReclaims.textContent).toContain("wl-alice-borrowed");
    expect(activeReclaims.textContent).toContain("bob");
    expect(activeReclaims.textContent).toContain("Auto-resume elsewhere");

    expect(screen.getByTestId("tier-badge-admin").textContent).toBe("Admin");
    expect(screen.getByTestId("tier-badge-admin").getAttribute("title")).toContain("Fleet admin");
    expect(screen.getByTestId("tier-badge-alice").textContent).toBe("Trusted · can borrow");
    expect(screen.getByTestId("tier-badge-alice").getAttribute("title")).toContain("borrow idle TPU");
    expect(screen.getByTestId("tier-badge-bob").textContent).toBe("User · own quota");
    expect(screen.getByTestId("tier-badge-bob").getAttribute("title")).toContain("own guaranteed TPU quota");

    const aliceUsage = screen.getByTestId("usage-alice");
    expect(aliceUsage.textContent).toContain("128");
    expect(aliceUsage.textContent).toContain("/ 64 TPU");
    expect(aliceUsage.textContent).toContain("+64 borrowed");
    expect(aliceUsage.querySelector(".usage-bar-own")).not.toBeNull();
    expect(aliceUsage.querySelector(".usage-bar-borrowed")).not.toBeNull();
    expect(aliceUsage.querySelector(".usage-bar-guarantee-mark")).not.toBeNull();
  });

  it("degrades gracefully when governance summary returns null (404-safe older plane)", async () => {
    const client = mockClient();
    client.getGovernanceSummary = vi.fn(async () => null);
    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);

    await screen.findByRole("region", { name: "Users" });
    expect(screen.queryByRole("region", { name: "Team quota and borrowing summary" })).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("edits can_borrow rule (Default by tier / Allow / Deny) without sending spurious unchanged fields", async () => {
    const client = mockClient();
    const userWithTier: TeamUser = {
      ...USER,
      tier: "user",
      rules: {
        allowed_fleets: null,
        approval_tpu_threshold: null,
        guaranteed_tpu_units: 64,
        max_tpu_units: 128,
        max_cpu_units: null,
        max_gpu_units: null,
        can_borrow: null,
      },
    };
    vi.mocked(client.listTeamUsers).mockResolvedValue([userWithTier]);
    client.setTeamUserRules = vi.fn(async () => ({
      ...userWithTier,
      rules: { ...userWithTier.rules!, can_borrow: true },
    }));

    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" />);
    const users = await screen.findByRole("region", { name: "Users" });
    fireEvent.click(within(users).getByRole("button", { name: "Rules" }));
    const form = await within(users).findByRole("form", { name: "Rules for alice" });

    const borrowSelect = within(form).getByLabelText(/Can borrow idle capacity/) as HTMLSelectElement;
    expect(borrowSelect.value).toBe("");
    fireEvent.change(borrowSelect, { target: { value: "allow" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save rules" }));
    await waitFor(() => expect(client.setTeamUserRules).toHaveBeenCalledWith("alice", { can_borrow: true }));
  });

  it("shows a one-click copyable eyvan login --invite CLI onboarding command after creating an invite", async () => {
    const client = mockClient();
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    const onNotify = vi.fn();

    render(<UsersPage client={client} canAdminister defaultOrg="acme" defaultProject="r" onNotify={onNotify} />);
    const invite = await screen.findByRole("region", { name: "Invite a user" });
    fireEvent.change(within(invite).getByLabelText("Email"), { target: { value: "bob@example.com" } });
    fireEvent.click(within(invite).getByRole("button", { name: /Create invite/ }));

    const cmd = await within(invite).findByTestId("invite-login-command");
    expect(cmd.textContent).toBe("eyvan login https://plane.example.com --invite secret-link");

    fireEvent.click(within(invite).getByRole("button", { name: "Copy CLI command" }));
    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith("eyvan login https://plane.example.com --invite secret-link"),
    );
    expect(onNotify).toHaveBeenCalledWith("Copied CLI login command");
    expect(inviteLoginCommand("https://tpu.example.com/eyvan/#/accept-invite/tok_1", "tok_1")).toBe(
      "eyvan login https://tpu.example.com/eyvan --invite tok_1",
    );
  });
});
