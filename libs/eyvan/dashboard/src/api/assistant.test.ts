// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import { EyvanApiClient } from "./client";
import { configuredScope } from "../auth/oidc";
import type { AssistantMessage, AssistantProvider, AssistantProviderInput } from "../types";

const PROVIDER: AssistantProvider = {
  id: "provider_1",
  name: "Primary",
  base_url: "https://api.example.com/v1",
  models: ["quick", "careful"],
  default_model: "careful",
  enabled: true,
  has_api_key: true,
};

function mockResponse(body: unknown, status = 200) {
  const fetch = vi.fn().mockImplementation(
    async () =>
      new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

afterEach(() => vi.unstubAllGlobals());

describe("assistant provider API", () => {
  it("returns no providers in demo mode without issuing a request", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", true);
    await expect(client.listAssistantProviders()).resolves.toEqual({ providers: [], can_manage: false });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("reads the provider list with the configured API base and forwards cancellation", async () => {
    const body = { providers: [PROVIDER], can_manage: true };
    const fetch = mockResponse(body);
    const controller = new AbortController();
    const client = new EyvanApiClient("https://plane.example/api/v1", false);
    const result = client.listAssistantProviders(controller.signal);
    expect(fetch.mock.calls[0][0]).toBe("https://plane.example/api/v1/assistant/providers");
    const signal = fetch.mock.calls[0][1].signal as AbortSignal;
    controller.abort();
    expect(signal.aborted).toBe(true);
    await expect(result).resolves.toEqual(body);
  });

  it("creates a provider with its write-only key", async () => {
    const fetch = mockResponse(PROVIDER, 201);
    const input: AssistantProviderInput = {
      name: "Primary",
      base_url: "https://api.example.com/v1",
      models: ["quick", "careful"],
      default_model: "careful",
      max_steps: 8,
      enabled: true,
      api_key: "write-only-secret",
    };
    const result = await new EyvanApiClient("/api/v1", false).createAssistantProvider(input);
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/providers");
    expect(fetch.mock.calls[0][1].method).toBe("POST");
    expect(JSON.parse(fetch.mock.calls[0][1].body as string)).toEqual(input);
    expect(result).toEqual(PROVIDER);
    expect(result).not.toHaveProperty("api_key");
  });

  it("updates only supplied fields and encodes the provider id", async () => {
    const fetch = mockResponse(PROVIDER);
    await new EyvanApiClient("/api/v1", false).updateAssistantProvider("provider/1", { name: "Renamed" });
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/providers/provider%2F1");
    expect(fetch.mock.calls[0][1].method).toBe("PUT");
    expect(JSON.parse(fetch.mock.calls[0][1].body as string)).toEqual({ name: "Renamed" });
  });

  it("sends explicit key removal separately from a replacement key", async () => {
    const fetch = mockResponse(PROVIDER);
    const client = new EyvanApiClient("/api/v1", false);
    await client.updateAssistantProvider(PROVIDER.id, { clear_api_key: true });
    await client.updateAssistantProvider(PROVIDER.id, { api_key: "replacement-secret" });
    expect(JSON.parse(fetch.mock.calls[0][1].body as string)).toEqual({ clear_api_key: true });
    expect(JSON.parse(fetch.mock.calls[1][1].body as string)).toEqual({ api_key: "replacement-secret" });
  });

  it("deletes the requested provider", async () => {
    const fetch = mockResponse({ deleted: PROVIDER.id });
    await expect(new EyvanApiClient("/api/v1", false).deleteAssistantProvider(PROVIDER.id)).resolves.toEqual({
      deleted: PROVIDER.id,
    });
    expect(fetch.mock.calls[0][0]).toBe(`/api/v1/assistant/providers/${PROVIDER.id}`);
    expect(fetch.mock.calls[0][1].method).toBe("DELETE");
    expect(fetch.mock.calls[0][1].body).toBeUndefined();
  });

  it("preserves the additive provider count in assistant status", async () => {
    const body = { enabled: true, model: "careful", providers: 2 };
    const fetch = mockResponse(body);
    await expect(new EyvanApiClient("/api/v1", false).getAssistantStatus()).resolves.toEqual(body);
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/status");
  });

  it.each([undefined, null])("keeps unselected chat requests working (%s)", async (selection) => {
    const turn = { reply: "ok", steps: 1, tool_calls: [], drafts: [], requests: [] };
    const fetch = mockResponse(turn);
    const messages: AssistantMessage[] = [{ role: "user", content: "hi" }];
    const client = new EyvanApiClient("/api/v1", false);
    if (selection === undefined) await expect(client.assistantChat(messages)).resolves.toEqual(turn);
    else await expect(client.assistantChat(messages, selection)).resolves.toEqual(turn);
    const scope = configuredScope();
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/chat");
    expect(fetch.mock.calls[0][1].method).toBe("POST");
    expect(JSON.parse(fetch.mock.calls[0][1].body as string)).toEqual({
      messages,
      org_id: scope.orgId,
      project_id: scope.projectId,
    });
  });

  it("returns empty preset and discovery answers in demo mode without a request", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", true);
    await expect(client.listAssistantPresets()).resolves.toEqual([]);
    await expect(client.discoverAssistantModels({ preset: "openai" })).resolves.toEqual({
      models: [],
      source: "suggested",
      detail: null,
    });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("reads presets, discovers models and starts the OpenRouter sign-in at their literal paths", async () => {
    const fetch = mockResponse({ presets: [{ id: "openai" }] });
    const client = new EyvanApiClient("/api/v1", false);
    await expect(client.listAssistantPresets()).resolves.toEqual([{ id: "openai" }]);
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/presets");

    mockResponse({ models: ["a"], source: "live", detail: null });
    await client.discoverAssistantModels({ provider_id: PROVIDER.id });
    const discover = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(discover[0]).toBe("/api/v1/assistant/models/discover");
    expect(discover[1]?.method).toBe("POST");
    expect(JSON.parse(String(discover[1]?.body))).toEqual({ provider_id: PROVIDER.id });

    mockResponse({ authorize_url: "https://openrouter.ai/auth", expires_at: "2026-10-28T12:00:00Z" });
    await client.startOpenRouterSignIn({ models: ["a"], default_model: "a" });
    const start = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(start[0]).toBe("/api/v1/assistant/oauth/openrouter/start");
    expect(JSON.parse(String(start[1]?.body))).toEqual({ models: ["a"], default_model: "a" });
  });

  it("creates a provider from a preset with only its key", async () => {
    const fetch = mockResponse(PROVIDER, 201);
    await new EyvanApiClient("/api/v1", false).createAssistantProvider({ preset: "openai", api_key: "k" });
    expect(JSON.parse(fetch.mock.calls[0][1].body as string)).toEqual({ preset: "openai", api_key: "k" });
  });

  it("forwards a chat abort signal", async () => {
    const fetch = mockResponse({ reply: "ok", steps: 1, tool_calls: [], drafts: [], requests: [] });
    const controller = new AbortController();
    const pending = new EyvanApiClient("/api/v1", false).assistantChat(
      [{ role: "user", content: "hi" }],
      null,
      undefined,
      controller.signal,
    );
    const signal = fetch.mock.calls[0][1].signal as AbortSignal;
    controller.abort();
    expect(signal.aborted).toBe(true);
    await pending.catch(() => undefined);
  });

  it("streams NDJSON events across chunks split mid-line and several lines per chunk", async () => {
    const lines = [
      { type: "start", conversation_id: "conv_1", provider_id: "p", model: "m", title: "t" },
      { type: "step", step: 1, max_steps: 6 },
      { type: "text", delta: "héllo ✓" },
      { type: "done", reply: "héllo ✓", steps: 1, tool_calls: [], drafts: [], requests: [], message_id: "x" },
    ];
    const bytes = new TextEncoder().encode(lines.map((line) => JSON.stringify(line)).join("\n"));
    // Odd chunk boundaries: mid-line, mid multi-byte character, and the last line has no newline.
    const cuts = [5, 6, 60, 61, 95, 96, 97, bytes.length];
    let at = 0;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const cut of cuts) {
          controller.enqueue(bytes.slice(at, cut));
          at = cut;
        }
        controller.close();
      },
    });
    const fetch = vi.fn(async () => new Response(body, { headers: { "content-type": "application/x-ndjson" } }));
    vi.stubGlobal("fetch", fetch);
    const events: unknown[] = [];
    const controller = new AbortController();
    await new EyvanApiClient("https://plane.example/api/v1", false).assistantChatStream(
      { message: "hi", conversation_id: "conv_1", provider_id: "p", model: "m" },
      (event) => events.push(event),
      controller.signal,
    );
    expect(events).toEqual(lines);
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://plane.example/api/v1/assistant/chat/stream");
    expect(init.method).toBe("POST");
    expect(init.signal).toBe(controller.signal);
    expect(new Headers(init.headers).get("accept")).toBe("application/x-ndjson");
    const scope = configuredScope();
    expect(JSON.parse(String(init.body))).toEqual({
      message: "hi",
      conversation_id: "conv_1",
      provider_id: "p",
      model: "m",
      org_id: scope.orgId,
      project_id: scope.projectId,
    });
  });

  it("parses several events delivered in one chunk", async () => {
    const text = '{"type":"step","step":1,"max_steps":2}\n\n{"type":"text","delta":"a"}\n{"type":"text","delta":"b"}\n';
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(text)),
    );
    const events: unknown[] = [];
    await new EyvanApiClient("/api/v1", false).assistantChatStream({ message: "x" }, (event) => events.push(event));
    expect(events).toEqual([
      { type: "step", step: 1, max_steps: 2 },
      { type: "text", delta: "a" },
      { type: "text", delta: "b" },
    ]);
  });

  it("throws the plane's detail when the stream is refused before it starts", async () => {
    mockResponse({ detail: "unknown provider" }, 404);
    const onEvent = vi.fn();
    const pending = new EyvanApiClient("/api/v1", false).assistantChatStream({ message: "x" }, onEvent);
    await expect(pending).rejects.toMatchObject({ status: 404, message: expect.stringContaining("unknown provider") });
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("reads, renames and deletes conversations and drives Claude Code at literal paths", async () => {
    const client = new EyvanApiClient("/api/v1", false);
    let fetch = mockResponse({ conversations: [{ id: "c1" }] });
    await expect(client.listAssistantConversations()).resolves.toEqual([{ id: "c1" }]);
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/conversations");

    fetch = mockResponse({ id: "c/1", messages: [] });
    await client.getAssistantConversation("c/1");
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/conversations/c%2F1");

    fetch = mockResponse({ id: "c1", title: "New" });
    await client.renameAssistantConversation("c1", "New");
    expect(fetch.mock.calls[0][1].method).toBe("PUT");
    expect(JSON.parse(fetch.mock.calls[0][1].body as string)).toEqual({ title: "New" });

    fetch = mockResponse({ deleted: "c1" });
    await expect(client.deleteAssistantConversation("c1")).resolves.toEqual({ deleted: "c1" });
    expect(fetch.mock.calls[0][1].method).toBe("DELETE");

    fetch = mockResponse({ installed: true, logged_in: false });
    await client.getClaudeCodeStatus();
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/claude-code/status");

    fetch = mockResponse({ started: true, url: null });
    await expect(client.startClaudeCodeLogin()).resolves.toEqual({ started: true, url: null });
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/assistant/claude-code/login");
    expect(fetch.mock.calls[0][1].method).toBe("POST");
  });

  it("answers conversations and Claude Code with empty values in demo mode", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const client = new EyvanApiClient("/api/v1", true);
    await expect(client.listAssistantConversations()).resolves.toEqual([]);
    await expect(client.getClaudeCodeStatus()).resolves.toMatchObject({ installed: false, logged_in: false });
    await expect(client.startClaudeCodeLogin()).resolves.toEqual({ started: false, url: null });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("sends the selected provider and model with an explicit chat scope", async () => {
    const fetch = mockResponse({ reply: "ok", steps: 1, tool_calls: [], drafts: [], requests: [] });
    const client = new EyvanApiClient("/api/v1", false);
    const messages: AssistantMessage[] = [{ role: "user", content: "hi" }];
    await client.assistantChat(
      messages,
      { provider_id: PROVIDER.id, model: "quick" },
      {
        orgId: "other",
        projectId: "research",
      },
    );
    expect(JSON.parse(fetch.mock.calls[0][1].body as string)).toEqual({
      messages,
      org_id: "other",
      project_id: "research",
      provider_id: PROVIDER.id,
      model: "quick",
    });
  });
});
