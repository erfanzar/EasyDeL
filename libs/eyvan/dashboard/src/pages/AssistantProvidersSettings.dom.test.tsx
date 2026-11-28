// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EyvanApiClient } from "../api/client";
import type { AssistantPreset, AssistantProvider, ClaudeCodeStatus } from "../types";
import {
  AssistantProvidersSettings,
  claudeCodeStatusText,
  presetBadge,
  type AssistantSettingsClient,
} from "./AssistantProvidersSettings";
import { SettingsPage, assistantHashNotice } from "./SettingsPage";
import type { TeamClient } from "./UsersPage";

const OPENAI: AssistantPreset = {
  id: "openai",
  label: "OpenAI",
  base_url: "https://api.openai.com/v1",
  editable_url: false,
  key_required: true,
  key_url: "https://platform.openai.com/api-keys",
  sign_in: null,
  suggested_models: ["gpt-a", "gpt-b", "gpt-missing"],
};
const OPENROUTER: AssistantPreset = {
  id: "openrouter",
  label: "OpenRouter",
  base_url: "https://openrouter.ai/api/v1",
  editable_url: false,
  key_required: true,
  key_url: "https://openrouter.ai/keys",
  sign_in: "openrouter",
  suggested_models: ["or/one"],
};
const OLLAMA: AssistantPreset = {
  id: "ollama",
  label: "Ollama",
  base_url: "http://127.0.0.1:11434/v1",
  editable_url: true,
  key_required: false,
  key_url: null,
  sign_in: null,
  suggested_models: [],
};
const CUSTOM: AssistantPreset = {
  id: "custom",
  label: "Custom",
  base_url: "",
  editable_url: true,
  key_required: false,
  key_url: null,
  sign_in: null,
  suggested_models: [],
};
const PRESETS = [OPENAI, OPENROUTER, OLLAMA, CUSTOM];

const STORED: AssistantProvider = {
  id: "provider_1",
  name: "Team OpenAI",
  preset: "openai",
  base_url: OPENAI.base_url,
  models: ["gpt-a"],
  default_model: "gpt-a",
  max_steps: 8,
  enabled: true,
  has_api_key: true,
};

const SECRET = "sk-write-only-secret-123";

function mockClient(providers: AssistantProvider[] = []) {
  return {
    listAssistantProviders: vi.fn(async () => ({ providers, can_manage: true })),
    listAssistantPresets: vi.fn(async () => PRESETS),
    discoverAssistantModels: vi.fn<AssistantSettingsClient["discoverAssistantModels"]>(async () => ({
      models: ["gpt-a", "gpt-b", "gpt-c"],
      source: "live" as const,
      detail: null,
    })),
    createAssistantProvider: vi.fn(async () => STORED),
    updateAssistantProvider: vi.fn(async () => STORED),
    deleteAssistantProvider: vi.fn(async () => ({ deleted: STORED.id })),
    startOpenRouterSignIn: vi.fn(async () => ({
      authorize_url: "https://openrouter.ai/auth?callback=x",
      expires_at: "2026-10-28T12:00:00Z",
    })),
    getClaudeCodeStatus: vi.fn<AssistantSettingsClient["getClaudeCodeStatus"]>(async () => SIGNED_OUT),
    startClaudeCodeLogin: vi.fn<AssistantSettingsClient["startClaudeCodeLogin"]>(async () => ({
      started: true,
      url: "https://claude.ai/oauth/authorize?code=x",
    })),
  } satisfies AssistantSettingsClient;
}

const CLAUDE_CODE: AssistantPreset = {
  id: "claude-code",
  label: "Claude Code",
  base_url: "",
  editable_url: false,
  key_required: false,
  key_url: null,
  sign_in: "claude-code",
  suggested_models: ["sonnet", "opus", "haiku"],
};
const SIGNED_OUT: ClaudeCodeStatus = {
  installed: true,
  version: "2.1.0",
  logged_in: false,
  auth_method: null,
  subscription_type: null,
  detail: null,
};
const SIGNED_IN: ClaudeCodeStatus = {
  ...SIGNED_OUT,
  logged_in: true,
  auth_method: "claude.ai",
  subscription_type: "max",
};

async function choosePreset(label: string) {
  fireEvent.click(await screen.findByRole("button", { name: "Add provider" }));
  const grid = await screen.findByRole("group", { name: "Provider presets" });
  fireEvent.click(within(grid).getByRole("button", { name: label }));
}

const checkedModels = () =>
  within(screen.getByRole("group", { name: "Models" }))
    .getAllByRole("checkbox")
    .filter((box) => (box as HTMLInputElement).checked)
    .map((box) => box.closest("label")?.textContent);

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  window.location.hash = "";
});

describe("Assistant providers settings", () => {
  it("adds a preset provider from a key: discover, pre-check suggestions, create without a base URL", async () => {
    const client = mockClient();
    const { container } = render(<AssistantProvidersSettings client={client} />);
    await choosePreset("OpenAI");
    const form = screen.getByRole("form", { name: "Add OpenAI" });
    const key = within(form).getByLabelText("API key") as HTMLInputElement;
    expect(key.type).toBe("password");
    expect(key.autocomplete).toBe("off");
    expect(
      within(form)
        .getByRole("link", { name: /Get a key/ })
        .getAttribute("href"),
    ).toBe(OPENAI.key_url);
    const url = within(form).getByLabelText("Base URL") as HTMLInputElement;
    expect(url.readOnly).toBe(true);
    expect(url.value).toBe(OPENAI.base_url);
    expect(client.discoverAssistantModels).not.toHaveBeenCalled();

    fireEvent.change(key, { target: { value: SECRET } });
    await waitFor(
      () =>
        expect(client.discoverAssistantModels).toHaveBeenCalledWith(
          { preset: "openai", api_key: SECRET },
          expect.any(AbortSignal),
        ),
      { timeout: 2000 },
    );
    await within(form).findByText("3 models from the provider");
    expect(checkedModels()).toEqual(["gpt-a", "gpt-b"]);
    expect((within(form).getByLabelText("Default model") as HTMLSelectElement).value).toBe("gpt-a");

    fireEvent.change(within(form).getByLabelText("Add model id"), { target: { value: "gpt-extra" } });
    fireEvent.click(within(form).getByRole("button", { name: "Add model" }));
    fireEvent.change(within(form).getByLabelText("Default model"), { target: { value: "gpt-b" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save provider" }));

    await waitFor(() =>
      expect(client.createAssistantProvider).toHaveBeenCalledWith({
        preset: "openai",
        api_key: SECRET,
        models: ["gpt-a", "gpt-b", "gpt-extra"],
        default_model: "gpt-b",
      }),
    );
    await screen.findByText("OpenAI added.");
    expect(screen.queryByRole("form", { name: "Add OpenAI" })).toBeNull();
    expect(client.listAssistantProviders).toHaveBeenCalledTimes(2);
    expect(container.innerHTML).not.toContain(SECRET);
  });

  it("lists models at once for presets that need no key", async () => {
    const client = mockClient();
    client.discoverAssistantModels.mockResolvedValueOnce({
      models: ["llama3"],
      source: "suggested",
      detail: "connection refused",
    });
    render(<AssistantProvidersSettings client={client} />);
    await choosePreset("Ollama");
    await waitFor(() =>
      expect(client.discoverAssistantModels).toHaveBeenCalledWith(
        { preset: "ollama", base_url: OLLAMA.base_url },
        expect.any(AbortSignal),
      ),
    );
    await screen.findByText(/Showing suggested models — connection refused/);
    expect(checkedModels()).toEqual(["llama3"]);
  });

  it("requires a base URL for a custom provider", async () => {
    const client = mockClient();
    render(<AssistantProvidersSettings client={client} />);
    await choosePreset("Custom");
    const form = screen.getByRole("form", { name: "Add Custom" });
    const url = within(form).getByLabelText("Base URL") as HTMLInputElement;
    expect(url.readOnly).toBe(false);
    expect(url.required).toBe(true);
    expect(within(form).getByRole("button", { name: "Load models" })).toHaveProperty("disabled", true);
    fireEvent.click(within(form).getByRole("button", { name: "Save provider" }));
    expect((await within(form).findByRole("alert")).textContent).toContain("base URL");
    expect(client.createAssistantProvider).not.toHaveBeenCalled();

    fireEvent.change(url, { target: { value: "https://llm.internal/v1" } });
    await waitFor(
      () =>
        expect(client.discoverAssistantModels).toHaveBeenCalledWith(
          { preset: "custom", base_url: "https://llm.internal/v1" },
          expect.any(AbortSignal),
        ),
      { timeout: 2000 },
    );
    await within(form).findByText("3 models from the provider");
    fireEvent.click(within(form).getByRole("button", { name: "Save provider" }));
    await waitFor(() =>
      expect(client.createAssistantProvider).toHaveBeenCalledWith({
        preset: "custom",
        base_url: "https://llm.internal/v1",
        models: ["gpt-a", "gpt-b", "gpt-c"],
        default_model: "gpt-a",
      }),
    );
  });

  it("edits with the stored key: discover by provider_id, no api_key, and an explicit key removal", async () => {
    const client = mockClient([STORED]);
    render(<AssistantProvidersSettings client={client} />);
    const row = await screen.findByRole("listitem", { name: "Team OpenAI" });
    expect(within(row).getByText("key set")).toBeTruthy();
    expect(within(row).getByText("OpenAI")).toBeTruthy();
    expect(within(row).getByText(/1 model · default gpt-a/)).toBeTruthy();
    fireEvent.click(within(row).getByRole("button", { name: "Edit" }));
    const form = screen.getByRole("form", { name: "Edit Team OpenAI" });
    const key = within(form).getByLabelText("API key") as HTMLInputElement;
    expect(key.value).toBe("");
    expect(key.placeholder).toBe("Stored key is kept when left empty");
    expect(checkedModels()).toEqual(["gpt-a"]);
    expect(client.discoverAssistantModels).not.toHaveBeenCalled();

    fireEvent.click(within(form).getByRole("button", { name: "Load models" }));
    await waitFor(() =>
      expect(client.discoverAssistantModels).toHaveBeenCalledWith({ provider_id: STORED.id }, expect.any(AbortSignal)),
    );
    await within(form).findByText("3 models from the provider");
    // The admin's existing choice survives a fresh listing.
    expect(checkedModels()).toEqual(["gpt-a"]);
    fireEvent.click(within(within(form).getByRole("group", { name: "Models" })).getByLabelText("gpt-c"));
    fireEvent.click(within(form).getByLabelText("Remove stored key"));
    fireEvent.click(within(form).getByRole("button", { name: "Save provider" }));
    await waitFor(() =>
      expect(client.updateAssistantProvider).toHaveBeenCalledWith(STORED.id, {
        models: ["gpt-a", "gpt-c"],
        default_model: "gpt-a",
        clear_api_key: true,
      }),
    );
  });

  it("toggles and deletes after confirmation, and shows server errors in an alert", async () => {
    const client = mockClient([STORED]);
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    render(<AssistantProvidersSettings client={client} />);
    const row = await screen.findByRole("listitem", { name: "Team OpenAI" });
    fireEvent.click(within(row).getByLabelText("Enable Team OpenAI"));
    await waitFor(() => expect(client.updateAssistantProvider).toHaveBeenCalledWith(STORED.id, { enabled: false }));
    fireEvent.click(within(row).getByRole("button", { name: "Delete" }));
    expect(client.deleteAssistantProvider).not.toHaveBeenCalled();
    fireEvent.click(within(row).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(client.deleteAssistantProvider).toHaveBeenCalledWith(STORED.id));
    expect(confirm).toHaveBeenCalledTimes(2);

    client.updateAssistantProvider.mockRejectedValueOnce(new Error("Provider name already exists"));
    fireEvent.click(within(await screen.findByRole("listitem", { name: "Team OpenAI" })).getByLabelText(/Enable/));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Provider name already exists"));
  });

  it("signs in with OpenRouter using the checked models and sends the browser there", async () => {
    const client = mockClient();
    client.discoverAssistantModels.mockResolvedValue({ models: ["or/one", "or/two"], source: "live", detail: null });
    const assignLocation = vi.fn();
    render(<AssistantProvidersSettings client={client} assignLocation={assignLocation} />);
    await choosePreset("OpenRouter");
    const form = screen.getByRole("form", { name: "Add OpenRouter" });
    expect(within(form).queryByLabelText("API key")).toBeNull();
    await waitFor(() =>
      expect(client.discoverAssistantModels).toHaveBeenCalledWith({ preset: "openrouter" }, expect.any(AbortSignal)),
    );
    await within(form).findByText("2 models from the provider");
    fireEvent.click(within(within(form).getByRole("group", { name: "Models" })).getByLabelText("or/two"));
    fireEvent.click(within(form).getByRole("button", { name: "Sign in with OpenRouter" }));
    await waitFor(() =>
      expect(client.startOpenRouterSignIn).toHaveBeenCalledWith({
        models: ["or/one", "or/two"],
        default_model: "or/one",
      }),
    );
    expect(assignLocation).toHaveBeenCalledWith("https://openrouter.ai/auth?callback=x");

    fireEvent.click(within(form).getByRole("button", { name: "or paste an API key" }));
    expect(within(form).getByLabelText("API key")).toBeTruthy();
  });

  it("words the Claude Code status and badges the preset Sign in", () => {
    expect(claudeCodeStatusText(SIGNED_IN)).toBe("Signed in · Claude Max via claude.ai");
    expect(claudeCodeStatusText(SIGNED_OUT)).toBe("Not signed in");
    expect(claudeCodeStatusText({ ...SIGNED_OUT, installed: false })).toBe(
      "Claude Code isn't installed on the plane host",
    );
    expect(presetBadge(CLAUDE_CODE).text).toBe("Sign in");
  });

  it("says when Claude Code is not installed and offers no sign-in", async () => {
    const client = mockClient();
    client.listAssistantPresets.mockResolvedValue([...PRESETS, CLAUDE_CODE]);
    client.getClaudeCodeStatus.mockResolvedValue({
      ...SIGNED_OUT,
      installed: false,
      version: null,
      detail: "claude: command not found",
    });
    render(<AssistantProvidersSettings client={client} />);
    await choosePreset("Claude Code");
    const card = await screen.findByRole("group", { name: "Claude Code" });
    expect((await within(card).findByRole("status")).textContent).toBe("Claude Code isn't installed on the plane host");
    expect(within(card).getByText("claude: command not found")).toBeTruthy();
    expect(within(card).queryByRole("button", { name: "Sign in with Claude" })).toBeNull();
    expect(screen.queryByLabelText("API key")).toBeNull();
    expect(screen.queryByRole("button", { name: "Save provider" })).toBeNull();
    expect(client.discoverAssistantModels).not.toHaveBeenCalled();
  });

  it("signs Claude Code in on the plane host, polls until signed in, then saves without a key", async () => {
    const client = mockClient();
    client.listAssistantPresets.mockResolvedValue([...PRESETS, CLAUDE_CODE]);
    client.getClaudeCodeStatus
      .mockResolvedValueOnce(SIGNED_OUT)
      .mockResolvedValueOnce(SIGNED_OUT)
      .mockResolvedValue(SIGNED_IN);
    client.discoverAssistantModels.mockResolvedValue({
      models: ["sonnet", "opus", "haiku", "claude-sonnet-4-5"],
      source: "live",
      detail: null,
    });
    render(<AssistantProvidersSettings client={client} claudeCodePollMs={20} />);
    await choosePreset("Claude Code");
    const form = screen.getByRole("form", { name: "Add Claude Code" });
    expect(within(form).getByText("Sign in")).toBeTruthy();
    const card = within(form).getByRole("group", { name: "Claude Code" });
    expect((await within(card).findByRole("status")).textContent).toBe("Not signed in");
    expect(within(form).queryByLabelText("API key")).toBeNull();
    expect(within(form).queryByLabelText("Base URL")).toBeNull();
    expect(within(form).queryByRole("group", { name: "Models" })).toBeNull();
    expect(within(card).getByRole("button", { name: "Check again" })).toBeTruthy();

    fireEvent.click(within(card).getByRole("button", { name: "Sign in with Claude" }));
    const link = await within(card).findByRole("link", { name: /Open sign-in page/ });
    expect(link.getAttribute("href")).toBe("https://claude.ai/oauth/authorize?code=x");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toContain("noopener");
    expect(within(card).getByText(/completes on the plane host/)).toBeTruthy();
    expect(client.startClaudeCodeLogin).toHaveBeenCalledTimes(1);

    await within(card).findByText("Signed in · Claude Max via claude.ai", undefined, { timeout: 2000 });
    expect(client.getClaudeCodeStatus.mock.calls.length).toBeGreaterThanOrEqual(3);
    await waitFor(() =>
      expect(client.discoverAssistantModels).toHaveBeenCalledWith({ preset: "claude-code" }, expect.any(AbortSignal)),
    );
    await within(form).findByText("4 models from the provider");
    expect(checkedModels()).toEqual(["sonnet", "opus", "haiku"]);
    expect(within(form).queryByLabelText("API key")).toBeNull();
    fireEvent.click(within(form).getByRole("button", { name: "Save provider" }));
    await waitFor(() =>
      expect(client.createAssistantProvider).toHaveBeenCalledWith({
        preset: "claude-code",
        models: ["sonnet", "opus", "haiku"],
        default_model: "sonnet",
      }),
    );
  });

  it("keeps waiting when a Claude Code sign-in is already running", async () => {
    const client = mockClient();
    client.listAssistantPresets.mockResolvedValue([...PRESETS, CLAUDE_CODE]);
    client.startClaudeCodeLogin.mockRejectedValueOnce(
      Object.assign(new Error("Eyvan API request failed with 409"), { status: 409 }),
    );
    render(<AssistantProvidersSettings client={client} claudeCodePollMs={20} />);
    await choosePreset("Claude Code");
    const card = await screen.findByRole("group", { name: "Claude Code" });
    fireEvent.click(await within(card).findByRole("button", { name: "Sign in with Claude" }));
    await within(card).findByText(/already running on the plane host/);
    expect(within(card).queryByRole("alert")).toBeNull();
    client.getClaudeCodeStatus.mockResolvedValue(SIGNED_IN);
    fireEvent.click(within(card).getByRole("button", { name: "Check again" }));
    await within(card).findByText("Signed in · Claude Max via claude.ai");
  });

  it("works through the real API client class without detaching its methods", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const path = String(input);
      let body: unknown;
      if (path === "/api/v1/assistant/providers" && init?.method === "POST") body = STORED;
      else if (path === "/api/v1/assistant/providers") body = { providers: [], can_manage: true };
      else if (path === "/api/v1/assistant/presets") body = { presets: PRESETS };
      else if (path === "/api/v1/assistant/models/discover") body = { models: ["gpt-a"], source: "live", detail: null };
      else throw new Error(`Unexpected request: ${path}`);
      return new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });
    });
    render(<AssistantProvidersSettings client={new EyvanApiClient("/api/v1", false)} />);
    await choosePreset("OpenAI");
    const form = screen.getByRole("form", { name: "Add OpenAI" });
    fireEvent.change(within(form).getByLabelText("API key"), { target: { value: SECRET } });
    await within(form).findByText("1 models from the provider", undefined, { timeout: 2000 });
    fireEvent.click(within(form).getByRole("button", { name: "Save provider" }));
    await screen.findByText("OpenAI added.");
    const create = fetch.mock.calls.find(
      ([path, init]) => String(path) === "/api/v1/assistant/providers" && init?.method === "POST",
    );
    expect(JSON.parse(String(create?.[1]?.body))).toEqual({
      preset: "openai",
      api_key: SECRET,
      models: ["gpt-a"],
      default_model: "gpt-a",
    });
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("Settings page", () => {
  const teamClient = (): TeamClient => ({
    listTeamUsers: vi.fn(async () => []),
    setTeamUserDisabled: vi.fn(),
    listTeamInvites: vi.fn(async () => []),
    createTeamInvite: vi.fn(),
    revokeTeamInvite: vi.fn(),
    listTeamTokens: vi.fn(async () => []),
    createTeamToken: vi.fn(),
    revokeTeamToken: vi.fn(),
  });

  it("parses the OpenRouter return hash", () => {
    expect(assistantHashNotice("#/settings?assistant=openrouter-connected")?.tone).toBe("ok");
    expect(assistantHashNotice("#/settings?assistant_error=denied")).toEqual({
      tone: "error",
      text: expect.stringContaining("(denied)"),
    });
    expect(assistantHashNotice("#/settings")).toBeNull();
  });

  it("announces a finished OpenRouter sign-in once and cleans the hash", async () => {
    window.location.hash = "#/settings?assistant=openrouter-connected";
    const onProvidersChanged = vi.fn();
    render(
      <SettingsPage
        client={{ ...teamClient(), ...mockClient() }}
        teamMode={false}
        canManageFleet
        defaultOrg="acme"
        defaultProject="r"
        onProvidersChanged={onProvidersChanged}
      />,
    );
    expect(await screen.findByText(/OpenRouter is connected/)).toBeTruthy();
    expect(window.location.hash).toBe("#/settings");
    expect(onProvidersChanged).toHaveBeenCalled();
    expect(screen.getByRole("region", { name: "Assistant providers" })).toBeTruthy();
    expect(screen.queryByRole("region", { name: "My API tokens" })).toBeNull();
  });

  it("shows a sign-in failure as an alert", async () => {
    window.location.hash = "#/settings?assistant_error=state_mismatch";
    render(
      <SettingsPage
        client={{ ...teamClient(), ...mockClient() }}
        teamMode={false}
        canManageFleet
        defaultOrg="acme"
        defaultProject="r"
      />,
    );
    expect((await screen.findByRole("alert")).textContent).toContain("state_mismatch");
    expect(window.location.hash).toBe("#/settings");
  });

  it("shows no provider settings to a user who cannot manage the fleet", async () => {
    const client = { ...teamClient(), ...mockClient() };
    render(<SettingsPage client={client} teamMode canManageFleet={false} defaultOrg="acme" defaultProject="r" />);
    await screen.findByRole("region", { name: "My API tokens" });
    expect(screen.queryByRole("region", { name: "Assistant providers" })).toBeNull();
    expect(client.listAssistantProviders).not.toHaveBeenCalled();
    expect(client.listAssistantPresets).not.toHaveBeenCalled();
  });
});
