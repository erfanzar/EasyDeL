// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EyvanApiClient } from "../api/client";
import type {
  AssistantConversation,
  AssistantConversationSummary,
  AssistantProvider,
  AssistantStreamEvent,
  ResearchPlan,
  ResearchPlanSummary,
  TeamRequest,
} from "../types";
import {
  ACTIVE_CONVERSATION_KEY,
  AssistantDock,
  type AssistantDockClient,
  type AssistantDockProps,
} from "./AssistantDock";

const PRIMARY: AssistantProvider = {
  id: "provider_1",
  name: "Primary",
  models: ["quick", "careful"],
  default_model: "careful",
  enabled: true,
};
const OTHER: AssistantProvider = {
  id: "provider_2",
  name: "Other",
  models: ["small", "large"],
  default_model: "small",
};
const DISABLED: AssistantProvider = {
  id: "provider_3",
  name: "Off",
  models: ["x"],
  default_model: "x",
  enabled: false,
};

const REQUEST: TeamRequest = {
  id: "req_1",
  kind: "capacity",
  subject: "alice",
  org_id: "acme",
  project_id: "r",
  summary: "need a v5p-32",
  details: {},
  state: "open",
  created_at: "2026-10-27T10:00:00+00:00",
  resolved_at: null,
  resolved_by: null,
  resolution_note: "",
};

const DRAFT = { org_id: "acme", project_id: "r", spec: { name: "llama-sft", execution: { kind: "command" } } };

const PLAN: ResearchPlanSummary = {
  id: "rp_0123456789abcdef",
  title: "lr sweep",
  subject: "alice",
  org_id: "acme",
  project_id: "r",
  state: "proposed",
  iterate: true,
  provider_id: null,
  model: null,
  current_round: 0,
  jobs_planned: 2,
  budget: { max_jobs: 2, max_chip_hours: 1, max_wall_seconds: 600, max_rounds: 3 },
  spent: { chip_hours: 0, jobs_submitted: 0, rounds: 0 },
  stop_reason: null,
  approved_at: null,
  approved_by: null,
  started_at: null,
  finished_at: null,
  created_at: "2026-10-28T10:00:00+00:00",
  updated_at: "2026-10-28T10:00:00+00:00",
};
const planDocument = (state: ResearchPlan["state"], extra: Partial<ResearchPlan> = {}): ResearchPlan => ({
  ...PLAN,
  state,
  goal: "",
  jobs: [],
  workload_ids: [],
  remaining: { jobs: 2, chip_hours: 1 },
  notes: [],
  ...extra,
});

const iso = (ms: number) => new Date(ms).toISOString();
const NOW = Date.now();

const SUMMARY: AssistantConversationSummary = {
  id: "conv_today",
  title: "Quota question",
  provider_id: PRIMARY.id,
  model: "careful",
  created_at: iso(NOW - 2_000),
  updated_at: iso(NOW - 1_000),
  message_count: 2,
};
const SUMMARIES: AssistantConversationSummary[] = [
  SUMMARY,
  { ...SUMMARY, id: "conv_yesterday", title: "Fleet check", updated_at: iso(NOW - 86_400_000) },
  { ...SUMMARY, id: "conv_old", title: "Old sizing", updated_at: iso(NOW - 10 * 86_400_000) },
];

const CONVERSATION: AssistantConversation = {
  id: "conv_9",
  title: "Saved",
  provider_id: PRIMARY.id,
  model: "careful",
  created_at: iso(NOW - 5_000),
  updated_at: iso(NOW - 1_000),
  messages: [
    { id: "m1", role: "user", content: "Saved question", created_at: iso(NOW - 5_000) },
    {
      id: "m2",
      role: "assistant",
      content: "Saved answer",
      created_at: iso(NOW - 4_000),
      trace: [{ type: "tool", name: "get_quota", ok: true, summary: "8 chips free", arguments: { org_id: "acme" } }],
      drafts: [DRAFT],
    },
  ],
};

/** A whole turn: start, one step, the answer text, done. */
const answer = (reply: string, extra: AssistantStreamEvent[] = []): AssistantStreamEvent[] => [
  { type: "start", conversation_id: "conv_1", provider_id: PRIMARY.id, model: "careful", title: "t" },
  { type: "step", step: 1, max_steps: 6 },
  ...extra,
  { type: "text", delta: reply },
  { type: "done", reply, steps: 1, tool_calls: [], drafts: [], requests: [], message_id: "msg_1" },
];

const notFound = () => Object.assign(new Error("Eyvan API request failed with 404"), { status: 404 });

type StreamCall = AssistantDockClient["assistantChatStream"];

function mockClient(events: AssistantStreamEvent[] = answer("ok"), providers = [PRIMARY, OTHER, DISABLED]) {
  return {
    listAssistantProviders: vi.fn(async () => ({ providers, can_manage: false })),
    assistantChatStream: vi.fn<StreamCall>(async (_body, onEvent) => {
      for (const event of events) onEvent(event);
    }),
    listAssistantConversations: vi.fn<AssistantDockClient["listAssistantConversations"]>(async () => SUMMARIES),
    getAssistantConversation: vi.fn<AssistantDockClient["getAssistantConversation"]>(async () => {
      throw notFound();
    }),
    renameAssistantConversation: vi.fn<AssistantDockClient["renameAssistantConversation"]>(async (id, title) => ({
      ...SUMMARY,
      id,
      title,
    })),
    deleteAssistantConversation: vi.fn(async (id: string) => ({ deleted: id })),
    submitAssistantDraft: vi.fn(async () => ({ id: "wl_1" })),
    approveResearchPlan: vi.fn<AssistantDockClient["approveResearchPlan"]>(async () => planDocument("approved")),
    stopResearchPlan: vi.fn<AssistantDockClient["stopResearchPlan"]>(async () => planDocument("stopping")),
    listTeamRequests: vi.fn(async () => [REQUEST]),
    resolveTeamRequest: vi.fn(async () => ({ ...REQUEST, state: "approved" as const })),
  } satisfies AssistantDockClient;
}

/** Hold the next stream open so a test can feed it events one at a time. */
function holdStream(client: ReturnType<typeof mockClient>) {
  const held: { onEvent?: (event: AssistantStreamEvent) => void; signal?: AbortSignal; finish?: () => void } = {};
  client.assistantChatStream.mockImplementationOnce(
    (_body, onEvent, signal) =>
      new Promise<void>((resolve, reject) => {
        held.onEvent = onEvent;
        held.signal = signal;
        held.finish = resolve;
        signal?.addEventListener("abort", () => reject(new DOMException("The operation was aborted.", "AbortError")));
      }),
  );
  return {
    emit: async (...events: AssistantStreamEvent[]) => {
      await act(async () => {
        for (const event of events) held.onEvent?.(event);
      });
    },
    finish: async () => {
      await act(async () => held.finish?.());
    },
    get signal() {
      return held.signal;
    },
  };
}

function renderDock(client: AssistantDockClient, props: Partial<AssistantDockProps> = {}) {
  return render(
    <AssistantDock
      client={client}
      open
      width={420}
      onWidthChange={() => undefined}
      onClose={() => undefined}
      canAdminister={false}
      teamMode={false}
      {...props}
    />,
  );
}

async function ask(text: string) {
  const box = await screen.findByLabelText("Message the assistant");
  fireEvent.change(box, { target: { value: text } });
  fireEvent.keyDown(box, { key: "Enter" });
}

const log = () => screen.getByRole("log", { name: "Conversation" });

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  sessionStorage.clear();
  localStorage.clear();
});

describe("AssistantDock", () => {
  it("is a labelled region, hidden when closed, and reads providers only once opened", async () => {
    const client = mockClient();
    const { rerender } = renderDock(client, { open: false });
    expect(screen.queryByRole("complementary", { name: "Assistant panel" })).toBeNull();
    expect(client.listAssistantProviders).not.toHaveBeenCalled();
    rerender(
      <AssistantDock
        client={client}
        open
        width={420}
        onWidthChange={() => undefined}
        onClose={() => undefined}
        canAdminister={false}
        teamMode={false}
      />,
    );
    expect(await screen.findByRole("complementary", { name: "Assistant panel" })).toBeTruthy();
    expect(client.listAssistantProviders).toHaveBeenCalledTimes(1);
  });

  it("lists enabled providers as provider · model and streams with the chosen selection", async () => {
    const client = mockClient(answer("Use a v5p-8."));
    renderDock(client);
    const picker = await screen.findByRole("button", { name: "Assistant model" });
    expect(picker.textContent).toContain("Primary");
    expect(picker.textContent).toContain("careful");
    fireEvent.click(picker);
    const list = screen.getByRole("listbox", { name: "Assistant models" });
    expect(
      within(list)
        .getAllByRole("option")
        .map((option) => option.getAttribute("aria-label")),
    ).toEqual(["Primary · quick", "Primary · careful", "Other · small", "Other · large"]);
    fireEvent.change(screen.getByRole("combobox", { name: "Search models" }), { target: { value: "lar" } });
    expect(within(list).getAllByRole("option")).toHaveLength(1);
    fireEvent.click(within(list).getByRole("option", { name: "Other · large" }));
    expect(screen.queryByRole("listbox")).toBeNull();

    await ask("how big a TPU for llama 8b?");
    await screen.findByText("Use a v5p-8.");
    // Only the new message travels; the plane keeps the conversation.
    expect(client.assistantChatStream).toHaveBeenCalledWith(
      { message: "how big a TPU for llama 8b?", provider_id: OTHER.id, model: "large" },
      expect.any(Function),
      expect.any(AbortSignal),
    );
  });

  it("sends on Enter but not on Shift+Enter, and starter suggestions send too", async () => {
    const client = mockClient();
    renderDock(client);
    const box = (await screen.findByLabelText("Message the assistant")) as HTMLTextAreaElement;
    expect(box.tagName).toBe("TEXTAREA");
    fireEvent.change(box, { target: { value: "line one" } });
    fireEvent.keyDown(box, { key: "Enter", shiftKey: true });
    expect(client.assistantChatStream).not.toHaveBeenCalled();
    fireEvent.keyDown(box, { key: "Enter" });
    await waitFor(() => expect(client.assistantChatStream).toHaveBeenCalledTimes(1));
    await screen.findByText("ok");
    expect(box.value).toBe("");
    cleanup();

    const fresh = mockClient();
    localStorage.clear();
    renderDock(fresh);
    fireEvent.click(await screen.findByRole("button", { name: "What's my quota?" }));
    await waitFor(() => expect(fresh.assistantChatStream.mock.calls[0][0].message).toBe("What's my quota?"));
  });

  it("streams the step, each tool from running to done, thinking, narration and the answer", async () => {
    const client = mockClient();
    const stream = holdStream(client);
    renderDock(client);
    await ask("quota?");
    await waitFor(() => expect(client.assistantChatStream).toHaveBeenCalled());
    await stream.emit(
      { type: "start", conversation_id: "conv_1", model: "careful", title: "quota?" },
      { type: "step", step: 1, max_steps: 6 },
      { type: "thinking", delta: "Check the " },
      { type: "thinking", delta: "quota first." },
      { type: "text", delta: "Let me look." },
      { type: "tool_call", id: "call_1", name: "get_quota", arguments: { org_id: "acme", project_id: "r" } },
    );
    expect(within(log()).getByText("Step 1/6")).toBeTruthy();
    const running = within(log()).getByRole("button", { name: "Tool get_quota: running" });
    expect(running.textContent).toContain("org_id=acme project_id=r");
    // Text of a round that ended in a tool call is narration, kept dim in the trace.
    expect(within(log()).getByText("Let me look.").closest(".dock-narration")).not.toBeNull();
    // Thinking starts collapsed even while it streams; the user opens it.
    const thinking = within(log()).getByRole("button", { name: "Thinking" });
    expect(thinking.getAttribute("aria-expanded")).toBe("false");
    expect(within(log()).queryByText("Check the quota first.")).toBeNull();
    fireEvent.click(thinking);
    expect(thinking.getAttribute("aria-expanded")).toBe("true");
    expect(within(log()).getByText("Check the quota first.")).toBeTruthy();

    await stream.emit(
      { type: "tool_result", id: "call_1", name: "get_quota", ok: true, summary: "8 chips free" },
      { type: "step", step: 2, max_steps: 6 },
      { type: "text", delta: "You have " },
    );
    const succeeded = within(log()).getByRole("button", { name: "Tool get_quota: succeeded" });
    expect(succeeded.textContent).toContain("✓");
    expect(succeeded.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(succeeded);
    expect(succeeded.getAttribute("aria-expanded")).toBe("true");
    expect(within(log()).getByText(/"org_id": "acme"/)).toBeTruthy();
    expect(within(log()).getByText("8 chips free")).toBeTruthy();
    expect(within(log()).getByText("Step 2/6")).toBeTruthy();
    // The answer shows as it arrives, before the turn is done.
    expect(within(log()).getByText("You have")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Stop generating" })).toBeTruthy();

    await stream.emit(
      { type: "text", delta: "**8** chips." },
      {
        type: "done",
        reply: "You have **8** chips.",
        steps: 2,
        tool_calls: [{ tool: "get_quota", ok: true }],
        drafts: [],
        requests: [],
        message_id: "msg_2",
      },
    );
    await stream.finish();
    const expander = await within(log()).findByRole("button", { name: "Used 1 tool · 2 steps" });
    expect(expander.getAttribute("aria-expanded")).toBe("false");
    expect(within(log()).queryByRole("button", { name: /^Tool get_quota/ })).toBeNull();
    expect(within(log()).queryByText("Step 2/6")).toBeNull();
    expect(within(log()).getByText("8").tagName).toBe("STRONG");
    fireEvent.click(expander);
    expect(within(log()).getByRole("button", { name: "Tool get_quota: succeeded" })).toBeTruthy();
    await screen.findByRole("button", { name: "Send message" });
  });

  it("marks a failed tool with ✗", async () => {
    const client = mockClient(
      answer("Could not read it.", [
        { type: "tool_call", id: "c1", name: "get_quota", arguments: null },
        { type: "tool_result", id: "c1", name: "get_quota", ok: false, summary: "403 forbidden" },
      ]),
    );
    renderDock(client);
    await ask("quota?");
    fireEvent.click(await within(log()).findByRole("button", { name: "Used 1 tool · 1 step" }));
    const row = within(log()).getByRole("button", { name: "Tool get_quota: failed" });
    expect(row.textContent).toContain("✗");
  });

  it("stops by aborting the stream and marks the answer stopped", async () => {
    const client = mockClient();
    const stream = holdStream(client);
    renderDock(client);
    await ask("long question");
    await waitFor(() => expect(client.assistantChatStream).toHaveBeenCalled());
    await stream.emit({ type: "tool_call", id: "c1", name: "list_fleets", arguments: null });
    fireEvent.click(screen.getByRole("button", { name: "Stop generating" }));
    expect(stream.signal?.aborted).toBe(true);
    await screen.findByRole("button", { name: "Send message" });
    expect(within(log()).getByText("Stopped")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows a streamed error and a refused request inline", async () => {
    const client = mockClient([
      { type: "start", conversation_id: "conv_1" },
      { type: "error", detail: "provider timed out" },
    ]);
    renderDock(client);
    await ask("hi");
    await waitFor(() => expect(within(log()).getByRole("alert").textContent).toContain("provider timed out"));
    cleanup();

    const refused = mockClient();
    refused.assistantChatStream.mockRejectedValueOnce(new Error("Eyvan API request failed with 404: unknown provider"));
    renderDock(refused);
    await ask("hi");
    await waitFor(() => expect(within(log()).getByRole("alert").textContent).toContain("unknown provider"));
  });

  it("shows a streamed draft at once and never submits it until the user confirms", async () => {
    const client = mockClient(
      answer("Draft ready.", [{ type: "draft", draft: DRAFT }]).map((event) =>
        event.type === "done" ? { ...event, drafts: [DRAFT] } : event,
      ),
    );
    const onOpenWorkload = vi.fn();
    renderDock(client, { onOpenWorkload });
    await ask("draft my job");
    await screen.findByText("Draft ready.");
    // Reported both as an event and in the summary, shown once.
    expect(screen.getAllByTestId("assistant-draft")).toHaveLength(1);
    const card = screen.getByTestId("assistant-draft");
    expect(within(card).getByText("llama-sft")).toBeTruthy();
    fireEvent.click(within(card).getByRole("button", { name: "Submit" }));
    expect(client.submitAssistantDraft).not.toHaveBeenCalled();
    fireEvent.click(within(card).getByRole("button", { name: "Cancel" }));
    fireEvent.click(within(card).getByRole("button", { name: "Submit" }));
    fireEvent.click(within(card).getByRole("button", { name: "Confirm submit" }));
    await within(card).findByText("Submitted wl_1");
    expect(client.submitAssistantDraft).toHaveBeenCalledWith(DRAFT);
    fireEvent.click(within(card).getByRole("button", { name: "Open" }));
    expect(onOpenWorkload).toHaveBeenCalledWith("wl_1");
  });

  it("a dismissed draft is never submitted, and filed requests render as notes", async () => {
    const draft = { org_id: "acme", project_id: "r", spec: { name: "x" } };
    const client = mockClient(
      answer("Filed.", [
        { type: "draft", draft },
        { type: "request", request: REQUEST },
      ]).map((event) => (event.type === "done" ? { ...event, requests: [REQUEST] } : event)),
    );
    renderDock(client);
    await ask("draft");
    const card = await screen.findByTestId("assistant-draft");
    fireEvent.click(within(card).getByRole("button", { name: "Dismiss" }));
    await within(card).findByText("Dismissed");
    expect(client.submitAssistantDraft).not.toHaveBeenCalled();
    expect(screen.getAllByTestId("assistant-request")).toHaveLength(1);
    expect(screen.getByTestId("assistant-request").textContent).toContain("need a v5p-32");
  });

  it("renders a streamed research plan as its own card, shown once, never as a draft", async () => {
    const client = mockClient(
      answer("Approve it to start.", [{ type: "plan", plan: PLAN }]).map((event) =>
        event.type === "done" ? { ...event, plans: [PLAN] } : event,
      ),
    );
    renderDock(client, { currentSubject: "alice" });
    await ask("sweep the learning rate");
    await screen.findByText("Approve it to start.");
    expect(screen.getAllByTestId("assistant-plan")).toHaveLength(1);
    expect(screen.queryByTestId("assistant-draft")).toBeNull();
    const card = screen.getByTestId("assistant-plan");
    expect(within(card).getByText("lr sweep")).toBeTruthy();
    expect(card.textContent).toContain("2 jobs");
    expect(within(card).getByTestId("plan-budget").textContent).toBe("2 jobs · 1 chip-hour · 10 min · 3 rounds");
    expect(within(card).getByTestId("research-state").textContent).toBe("proposed");
    expect(card.textContent).toContain("yes — may plan further rounds");
    expect(card.textContent).toContain("Nothing runs until you approve.");
    expect(within(card).getByRole("link", { name: "Open in Research" }).getAttribute("href")).toBe(
      "#/research/rp_0123456789abcdef",
    );
    // A proposed plan offers no Stop; only the owner's Approve.
    expect(within(card).queryByRole("button", { name: "Stop" })).toBeNull();
  });

  it("approves a plan after confirmation, never through the draft submission path, then stops it", async () => {
    const client = mockClient(answer("Proposed.", [{ type: "plan", plan: PLAN }]));
    renderDock(client, { currentSubject: "alice" });
    await ask("plan it");
    const card = await screen.findByTestId("assistant-plan");
    fireEvent.click(within(card).getByRole("button", { name: "Approve" }));
    expect(client.approveResearchPlan).not.toHaveBeenCalled();
    expect(card.textContent).toContain("under your identity");
    fireEvent.click(within(card).getByRole("button", { name: "Confirm approve" }));
    await waitFor(() => expect(within(card).getByTestId("research-state").textContent).toBe("approved"));
    expect(client.approveResearchPlan).toHaveBeenCalledWith(PLAN.id);
    expect(client.submitAssistantDraft).not.toHaveBeenCalled();

    fireEvent.click(within(card).getByRole("button", { name: "Stop" }));
    fireEvent.click(within(card).getByRole("button", { name: "Confirm stop" }));
    await waitFor(() => expect(within(card).getByTestId("research-state").textContent).toBe("stopping"));
    expect(client.stopResearchPlan).toHaveBeenCalledWith(PLAN.id);
    expect(client.submitAssistantDraft).not.toHaveBeenCalled();
  });

  it("shows why an approval failed and keeps the plan approvable", async () => {
    const client = mockClient(answer("Proposed.", [{ type: "plan", plan: PLAN }]));
    client.approveResearchPlan.mockRejectedValueOnce(new Error("Eyvan API request failed with 409: already approved"));
    renderDock(client, { currentSubject: "alice" });
    await ask("plan it");
    const card = await screen.findByTestId("assistant-plan");
    fireEvent.click(within(card).getByRole("button", { name: "Approve" }));
    fireEvent.click(within(card).getByRole("button", { name: "Confirm approve" }));
    expect((await within(card).findByRole("alert")).textContent).toContain("already approved");
    expect(within(card).getByRole("button", { name: "Approve" })).toBeTruthy();
    expect(client.submitAssistantDraft).not.toHaveBeenCalled();
  });

  it("does not offer Approve on someone else's plan", async () => {
    const client = mockClient(answer("Proposed.", [{ type: "plan", plan: PLAN }]));
    renderDock(client, { currentSubject: "bob" });
    await ask("plan it");
    const card = await screen.findByTestId("assistant-plan");
    expect(within(card).queryByRole("button", { name: "Approve" })).toBeNull();
    expect(card.textContent).toContain("Only the person who proposed this plan can approve it.");
  });

  it("re-renders a saved conversation's plans as cards", async () => {
    localStorage.setItem(ACTIVE_CONVERSATION_KEY, "conv_9");
    const client = mockClient();
    client.getAssistantConversation.mockResolvedValueOnce({
      ...CONVERSATION,
      messages: [
        CONVERSATION.messages[0],
        { ...CONVERSATION.messages[1], drafts: undefined, plans: [{ ...PLAN, state: "running" }] },
      ],
    });
    renderDock(client, { currentSubject: "alice" });
    const card = await screen.findByTestId("assistant-plan");
    expect(within(card).getByText("lr sweep")).toBeTruthy();
    expect(within(card).getByTestId("research-state").textContent).toBe("running");
    expect(within(card).getByRole("button", { name: "Stop" })).toBeTruthy();
    expect(within(card).queryByRole("button", { name: "Approve" })).toBeNull();
    expect(screen.queryByTestId("assistant-draft")).toBeNull();
  });

  it("refreshes a saved plan card from the plane, so an already-approved plan is not offered for approval", async () => {
    localStorage.setItem(ACTIVE_CONVERSATION_KEY, "conv_9");
    const client = {
      ...mockClient(),
      getResearchPlan: vi.fn(async () => ({ ...PLAN, state: "running" as const }) as unknown as ResearchPlan),
    };
    client.getAssistantConversation.mockResolvedValueOnce({
      ...CONVERSATION,
      messages: [
        CONVERSATION.messages[0],
        { ...CONVERSATION.messages[1], drafts: undefined, plans: [{ ...PLAN, state: "proposed" }] },
      ],
    });
    renderDock(client, { currentSubject: "alice" });
    const card = await screen.findByTestId("assistant-plan");
    await waitFor(() => expect(within(card).getByTestId("research-state").textContent).toBe("running"));
    expect(client.getResearchPlan).toHaveBeenCalledTimes(1);
    expect(client.getResearchPlan).toHaveBeenCalledWith(PLAN.id, expect.anything());
    expect(within(card).queryByRole("button", { name: "Approve" })).toBeNull();
    expect(within(card).getByRole("button", { name: "Stop" })).toBeTruthy();
  });

  it("says a saved plan card's reading failed and withholds Approve", async () => {
    localStorage.setItem(ACTIVE_CONVERSATION_KEY, "conv_9");
    const client = {
      ...mockClient(),
      getResearchPlan: vi.fn(async (): Promise<ResearchPlan> => {
        throw new Error("Eyvan API request failed with 503");
      }),
    };
    client.getAssistantConversation.mockResolvedValueOnce({
      ...CONVERSATION,
      messages: [
        CONVERSATION.messages[0],
        { ...CONVERSATION.messages[1], drafts: undefined, plans: [{ ...PLAN, state: "proposed" }] },
      ],
    });
    renderDock(client, { currentSubject: "alice" });
    const card = await screen.findByTestId("assistant-plan");
    const note = await within(card).findByRole("status");
    expect(note.textContent).toContain("Couldn't read this plan's current state; showing it as proposed at");
    expect(within(card).queryByRole("button", { name: "Approve" })).toBeNull();
    expect(within(card).getByRole("button", { name: "Withdraw" })).toBeTruthy();
  });

  it("tells an admin to set up a provider and everyone else to ask an admin", async () => {
    const onOpenSettings = vi.fn();
    renderDock(mockClient(answer("ok"), []), { canAdminister: true, onOpenSettings });
    await screen.findByText("No assistant provider is configured yet.");
    fireEvent.click(screen.getByRole("button", { name: "Set up a provider" }));
    expect(onOpenSettings).toHaveBeenCalled();
    expect(screen.queryByLabelText("Message the assistant")).toBeNull();
    cleanup();

    renderDock(mockClient(answer("ok"), [DISABLED]), { canAdminister: false });
    await screen.findByText("No assistant provider is configured yet.");
    expect(screen.getByText("Ask a fleet admin to add one.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Set up a provider" })).toBeNull();
  });

  it("sends the conversation id from start with the next message, and New chat forgets it", async () => {
    const client = mockClient(answer("first"));
    renderDock(client);
    await ask("one");
    await screen.findByText("first");
    expect(client.assistantChatStream.mock.calls[0][0]).toEqual({
      message: "one",
      provider_id: PRIMARY.id,
      model: "careful",
    });
    expect(localStorage.getItem(ACTIVE_CONVERSATION_KEY)).toBe("conv_1");
    await ask("two");
    await waitFor(() => expect(client.assistantChatStream).toHaveBeenCalledTimes(2));
    expect(client.assistantChatStream.mock.calls[1][0]).toEqual({
      message: "two",
      conversation_id: "conv_1",
      provider_id: PRIMARY.id,
      model: "careful",
    });

    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    expect(screen.queryByText("first")).toBeNull();
    expect(localStorage.getItem(ACTIVE_CONVERSATION_KEY)).toBeNull();
    await ask("three");
    await waitFor(() => expect(client.assistantChatStream).toHaveBeenCalledTimes(3));
    expect(client.assistantChatStream.mock.calls[2][0]).not.toHaveProperty("conversation_id");
  });

  it("resumes the remembered conversation, and starts fresh when it is gone", async () => {
    localStorage.setItem(ACTIVE_CONVERSATION_KEY, "conv_9");
    const client = mockClient();
    client.getAssistantConversation.mockResolvedValueOnce(CONVERSATION);
    renderDock(client);
    expect(await screen.findByText("Saved answer")).toBeTruthy();
    expect(screen.getByText("Saved question")).toBeTruthy();
    expect(client.getAssistantConversation).toHaveBeenCalledWith("conv_9", expect.any(AbortSignal));
    cleanup();

    renderDock(mockClient());
    await waitFor(() => expect(localStorage.getItem(ACTIVE_CONVERSATION_KEY)).toBeNull());
    expect(await screen.findByText("Ask about this plane")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("lists saved chats by day and opens, renames and deletes them", async () => {
    const client = mockClient();
    renderDock(client);
    const toggle = await screen.findByRole("button", { name: "Chat history" });
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    const history = await screen.findByRole("region", { name: "Saved chats" });
    const today = await within(history).findByRole("group", { name: "Today" });
    const item = within(today).getByRole("listitem", { name: "Quota question" });
    expect(item.textContent).toContain("just now");
    expect(within(history).getByRole("group", { name: "Yesterday" }).textContent).toContain("Fleet check");
    expect(within(history).getByRole("group", { name: "Earlier" }).textContent).toContain("Old sizing");

    fireEvent.click(within(item).getByRole("button", { name: "Rename Quota question" }));
    const title = within(history).getByRole("textbox", { name: "Chat title" });
    fireEvent.change(title, { target: { value: "Quota for acme" } });
    fireEvent.click(within(history).getByRole("button", { name: "Save title" }));
    await within(history).findByRole("listitem", { name: "Quota for acme" });
    expect(client.renameAssistantConversation).toHaveBeenCalledWith("conv_today", "Quota for acme");

    const old = within(history).getByRole("listitem", { name: "Old sizing" });
    fireEvent.click(within(old).getByRole("button", { name: "Delete Old sizing" }));
    expect(client.deleteAssistantConversation).not.toHaveBeenCalled();
    fireEvent.click(within(old).getByRole("button", { name: "Confirm delete" }));
    await waitFor(() => expect(client.deleteAssistantConversation).toHaveBeenCalledWith("conv_old"));
    await waitFor(() => expect(within(history).queryByRole("listitem", { name: "Old sizing" })).toBeNull());

    client.getAssistantConversation.mockResolvedValueOnce({ ...CONVERSATION, id: "conv_today" });
    fireEvent.click(within(history).getByRole("button", { name: "Open chat Quota for acme" }));
    expect(await within(log()).findByText("Saved answer")).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Saved chats" })).toBeNull();
    expect(client.getAssistantConversation).toHaveBeenCalledWith("conv_today", expect.any(AbortSignal));
    expect(localStorage.getItem(ACTIVE_CONVERSATION_KEY)).toBe("conv_today");
    // The saved trace is the collapsed expander; the saved draft is a card.
    const expander = within(log()).getByRole("button", { name: "Used 1 tool" });
    expect(within(log()).queryByRole("button", { name: /^Tool get_quota/ })).toBeNull();
    fireEvent.click(expander);
    expect(within(log()).getByRole("button", { name: "Tool get_quota: succeeded" })).toBeTruthy();
    const card = within(log()).getByRole("group", { name: "Drafted workload llama-sft" });
    expect(within(card).getByRole("button", { name: "Submit" })).toBeTruthy();
  });

  it("starts fresh when the open chat is deleted from the history", async () => {
    localStorage.setItem(ACTIVE_CONVERSATION_KEY, "conv_today");
    const client = mockClient();
    client.getAssistantConversation.mockResolvedValueOnce({ ...CONVERSATION, id: "conv_today" });
    renderDock(client);
    await screen.findByText("Saved answer");
    fireEvent.click(screen.getByRole("button", { name: "Chat history" }));
    const item = await screen.findByRole("listitem", { name: "Quota question" });
    fireEvent.click(within(item).getByRole("button", { name: "Delete Quota question" }));
    fireEvent.click(within(item).getByRole("button", { name: "Confirm delete" }));
    await waitFor(() => expect(localStorage.getItem(ACTIVE_CONVERSATION_KEY)).toBeNull());
  });

  it("renders replies as text: HTML in a reply is never interpreted", async () => {
    const reply = [
      'Look: <img src="x" onerror="alert(1)"> and <script>alert(2)</script>',
      "",
      "- use `v5p-8`",
      "- **spot** capacity",
      "",
      "```bash",
      "eyvan submit <job>",
      "```",
    ].join("\n");
    const client = mockClient(answer(reply));
    const { container } = renderDock(client);
    await ask("hi");
    await screen.findByText(/Look:/);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(container.textContent).toContain('<img src="x" onerror="alert(1)">');
    expect(within(log()).getAllByRole("listitem")).toHaveLength(2);
    expect(log().querySelector("li code")?.textContent).toBe("v5p-8");
    expect(log().querySelector("li strong")?.textContent).toBe("spot");
    expect(log().querySelector("pre code")?.textContent).toBe("eyvan submit <job>");
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    fireEvent.click(within(log()).getByRole("button", { name: "Copy code" }));
    expect(writeText).toHaveBeenCalledWith("eyvan submit <job>");
  });

  it("closes on Escape and from its close button", async () => {
    const onClose = vi.fn();
    renderDock(mockClient(), { onClose });
    await screen.findByLabelText("Message the assistant");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Close assistant" }));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("resizes from the keyboard on its left-edge handle", async () => {
    const onWidthChange = vi.fn();
    renderDock(mockClient(), { onWidthChange });
    const handle = await screen.findByRole("separator", { name: "Resize assistant panel" });
    fireEvent.keyDown(handle, { key: "ArrowLeft" });
    expect(onWidthChange).toHaveBeenCalledWith(436);
  });

  it("shows requests in a second tab where admins approve and filers withdraw", async () => {
    const admin = mockClient();
    renderDock(admin, { canAdminister: true, teamMode: true });
    fireEvent.click(await screen.findByRole("tab", { name: "Requests" }));
    const panel = screen.getByRole("tabpanel", { name: "Requests" });
    await within(panel).findByText("need a v5p-32");
    fireEvent.click(within(panel).getByRole("button", { name: "Approve" }));
    await waitFor(() => expect(admin.resolveTeamRequest).toHaveBeenCalledWith(REQUEST.id, "approved"));
    cleanup();

    const filer = mockClient();
    renderDock(filer, { teamMode: true });
    fireEvent.click(await screen.findByRole("tab", { name: "Requests" }));
    const mine = screen.getByRole("tabpanel", { name: "Requests" });
    await within(mine).findByText("need a v5p-32");
    expect(within(mine).queryByRole("button", { name: "Approve" })).toBeNull();
    fireEvent.click(within(mine).getByRole("button", { name: "Withdraw" }));
    await waitFor(() => expect(filer.resolveTeamRequest).toHaveBeenCalledWith(REQUEST.id, "withdrawn"));
  });

  it("renders with the real API client class in demo mode", async () => {
    renderDock(new EyvanApiClient("/api/v1", true), { canAdminister: true, onOpenSettings: () => undefined });
    await screen.findByText("No assistant provider is configured yet.");
    fireEvent.click(screen.getByRole("button", { name: "Chat history" }));
    await screen.findByText("No saved chats yet.");
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("streams through the real API client class without detaching its methods", async () => {
    const encoder = new TextEncoder();
    const body = () => {
      const bytes = encoder.encode(
        answer("real answer")
          .map((event) => `${JSON.stringify(event)}\n`)
          .join(""),
      );
      return new ReadableStream<Uint8Array>({
        start(controller) {
          // Chunks that end mid-line, as a network does.
          for (let index = 0; index < bytes.length; index += 7) controller.enqueue(bytes.slice(index, index + 7));
          controller.close();
        },
      });
    };
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const path = String(input);
      if (path === "/api/v1/assistant/providers") {
        return new Response(JSON.stringify({ providers: [PRIMARY], can_manage: false }), {
          headers: { "content-type": "application/json" },
        });
      }
      if (path === "/api/v1/assistant/chat/stream" && init?.method === "POST") {
        return new Response(body(), { headers: { "content-type": "application/x-ndjson" } });
      }
      throw new Error(`Unexpected request: ${path}`);
    });
    renderDock(new EyvanApiClient("/api/v1", false));
    await ask("hello");
    await screen.findByText("real answer");
    const call = fetch.mock.calls.find(([path]) => String(path) === "/api/v1/assistant/chat/stream");
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      message: "hello",
      provider_id: PRIMARY.id,
      model: "careful",
    });
    expect(new Headers(call?.[1]?.headers).get("accept")).toBe("application/x-ndjson");
    expect(localStorage.getItem(ACTIVE_CONVERSATION_KEY)).toBe("conv_1");
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
