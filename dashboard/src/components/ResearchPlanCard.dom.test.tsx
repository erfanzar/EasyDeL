// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ResearchPlanSummary } from "../types";
import {
  ACTIVE_PLAN_REFRESH_MS,
  ResearchPlanCard,
  budgetSummary,
  type ResearchPlanCardProps,
} from "./ResearchPlanCard";

const PLAN: ResearchPlanSummary = {
  id: "rp_0123456789abcdef",
  title: "lr sweep",
  subject: "alice",
  org_id: "acme",
  project_id: "r",
  state: "proposed",
  iterate: false,
  provider_id: null,
  model: null,
  current_round: 0,
  jobs_planned: 2,
  budget: { max_jobs: 2, max_chip_hours: 1, max_wall_seconds: 600, max_rounds: null },
  spent: { chip_hours: 0, jobs_submitted: 0, rounds: 0 },
  stop_reason: null,
  approved_at: null,
  approved_by: null,
  started_at: null,
  finished_at: null,
  created_at: "2026-10-28T10:00:00+00:00",
  updated_at: "2026-10-28T10:00:00+00:00",
};
const at = (state: ResearchPlanSummary["state"], updatedAt: string, extra: Partial<ResearchPlanSummary> = {}) => ({
  ...PLAN,
  state,
  updated_at: updatedAt,
  ...extra,
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function renderCard(overrides: Partial<ResearchPlanCardProps> = {}) {
  const props: ResearchPlanCardProps = {
    plan: PLAN,
    canApprove: true,
    canStop: true,
    approve: vi.fn(async () => at("approved", "2026-10-28T10:05:00+00:00")),
    stop: vi.fn(async () => at("stopping", "2026-10-28T10:06:00+00:00")),
    ...overrides,
  };
  const view = render(<ResearchPlanCard {...props} />);
  return { props, view, card: () => screen.getByTestId("assistant-plan") };
}

const state = () => within(screen.getByTestId("assistant-plan")).getByTestId("research-state").textContent;

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.useRealTimers();
  Object.defineProperty(document, "hidden", { configurable: true, value: false });
});

describe("ResearchPlanCard reading freshness", () => {
  it("ignores a refresh that started before Approve and resolved after it", async () => {
    const slow = deferred<ResearchPlanSummary>();
    const refresh = vi.fn(() => slow.promise);
    renderCard({ refresh });
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm approve" }));
    await waitFor(() => expect(state()).toBe("approved"));
    await act(async () => slow.resolve(PLAN));
    expect(state()).toBe("approved");
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
    expect(screen.getByRole("button", { name: "Stop" })).toBeTruthy();
  });

  it("ignores a reading older than the one shown even when it was not aborted", async () => {
    const refresh = vi.fn(async () => at("running", "2026-10-28T10:00:30+00:00"));
    const { view, props } = renderCard({ plan: at("stopping", "2026-10-28T10:06:00+00:00"), refresh });
    await waitFor(() => expect(refresh).toHaveBeenCalled());
    await act(async () => undefined);
    expect(state()).toBe("stopping");
    // A parent re-render with an older copy does not roll the card back either.
    view.rerender(<ResearchPlanCard {...props} plan={at("running", "2026-10-28T10:01:00+00:00")} refresh={refresh} />);
    expect(state()).toBe("stopping");
  });

  it("tells you a saved card's reading failed and hides Approve until one succeeds", async () => {
    const refresh = vi
      .fn<NonNullable<ResearchPlanCardProps["refresh"]>>()
      .mockRejectedValueOnce(new Error("Eyvan API request failed with 503"))
      .mockResolvedValueOnce(at("proposed", "2026-10-28T10:00:00+00:00"));
    renderCard({ saved: true, refresh });
    const note = await screen.findByRole("status");
    expect(note.textContent).toContain("Couldn't read this plan's current state; showing it as proposed at");
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
    // Withdraw stays: the plane refuses it safely if the plan already moved on.
    expect(screen.getByRole("button", { name: "Withdraw" })).toBeTruthy();
    fireEvent.click(within(note).getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: "Approve" })).toBeTruthy();
    expect(screen.queryByText(/Couldn't read this plan's current state/)).toBeNull();
  });

  it("says so when a saved card has no way to read the current state", async () => {
    renderCard({ saved: true, plan: at("running", PLAN.updated_at!) });
    expect(screen.getByRole("status").textContent).toContain("Couldn't read this plan's current state");
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
    expect(screen.getByRole("button", { name: "Stop" })).toBeTruthy();
  });

  it("hides Approve on a saved card while its first reading is still in flight", async () => {
    const pending = deferred<ResearchPlanSummary>();
    renderCard({ saved: true, refresh: () => pending.promise });
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
    await act(async () => pending.resolve(PLAN));
    expect(screen.getByRole("button", { name: "Approve" })).toBeTruthy();
  });

  it("polls an active plan while mounted and visible, and stops once it settles", async () => {
    vi.useFakeTimers();
    const refresh = vi
      .fn<NonNullable<ResearchPlanCardProps["refresh"]>>()
      .mockResolvedValueOnce(at("running", "2026-10-28T10:02:00+00:00"))
      .mockResolvedValueOnce(at("running", "2026-10-28T10:03:00+00:00"))
      .mockResolvedValueOnce(at("finished", "2026-10-28T10:04:00+00:00"))
      .mockResolvedValue(at("finished", "2026-10-28T10:04:00+00:00"));
    const onUpdated = vi.fn();
    const { view } = renderCard({ plan: at("approved", "2026-10-28T10:01:00+00:00"), refresh, onUpdated });
    await act(async () => undefined);
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(state()).toBe("running");
    await act(async () => vi.advanceTimersByTimeAsync(ACTIVE_PLAN_REFRESH_MS));
    expect(refresh).toHaveBeenCalledTimes(2);

    // A hidden tab does not poll.
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await act(async () => vi.advanceTimersByTimeAsync(3 * ACTIVE_PLAN_REFRESH_MS));
    expect(refresh).toHaveBeenCalledTimes(2);
    Object.defineProperty(document, "hidden", { configurable: true, value: false });
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await act(async () => undefined);
    expect(refresh).toHaveBeenCalledTimes(3);
    expect(state()).toBe("finished");
    expect(onUpdated).toHaveBeenLastCalledWith(expect.objectContaining({ state: "finished" }));

    // Settled: no more reads.
    await act(async () => vi.advanceTimersByTimeAsync(3 * ACTIVE_PLAN_REFRESH_MS));
    expect(refresh).toHaveBeenCalledTimes(3);
    view.unmount();
  });

  it("stops polling when the card unmounts", async () => {
    vi.useFakeTimers();
    const refresh = vi.fn(async () => at("running", "2026-10-28T10:02:00+00:00"));
    const { view } = renderCard({ plan: at("running", "2026-10-28T10:01:00+00:00"), refresh });
    await act(async () => undefined);
    expect(refresh).toHaveBeenCalledTimes(1);
    view.unmount();
    await act(async () => vi.advanceTimersByTimeAsync(3 * ACTIVE_PLAN_REFRESH_MS));
    expect(refresh).toHaveBeenCalledTimes(1);
  });
});

describe("ResearchPlanCard actions", () => {
  it("withdraws a proposed plan for its owner after confirmation", async () => {
    const stop = vi.fn(async () => at("stopped", "2026-10-28T10:07:00+00:00", { stop_reason: "stopped by alice" }));
    renderCard({ stop });
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Withdraw" }));
    expect(stop).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Confirm withdraw" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm withdraw" }));
    await waitFor(() => expect(state()).toBe("stopped"));
    expect(stop).toHaveBeenCalledWith(PLAN.id);
    expect(screen.queryByRole("button", { name: "Withdraw" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  });

  it("does not offer Withdraw to someone else, nor on a running plan", () => {
    renderCard({ canApprove: false, canStop: true });
    expect(screen.queryByRole("button", { name: "Withdraw" })).toBeNull();
    cleanup();
    renderCard({ plan: at("running", PLAN.updated_at!) });
    expect(screen.queryByRole("button", { name: "Withdraw" })).toBeNull();
  });

  it("moves focus to the confirm button when Approve or Stop asks for confirmation", () => {
    renderCard();
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Confirm approve" }));
    cleanup();
    renderCard({ plan: at("running", PLAN.updated_at!) });
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Confirm stop" }));
  });

  it.each([
    ["failed", "Failed"],
    ["over_budget", "Over budget"],
    ["stopped", "Stopped"],
  ] as const)("labels a %s plan's reason %s", (planState, label) => {
    renderCard({ plan: at(planState, PLAN.updated_at!, { stop_reason: "why it ended" }) });
    const reason = screen.getByText("why it ended");
    expect(reason.previousElementSibling?.textContent).toBe(label);
  });
});

describe("budgetSummary", () => {
  it("names the plane's default round limit", () => {
    expect(budgetSummary(PLAN.budget)).toBe("2 jobs · 1 chip-hour · 10 min · up to 20 rounds (default)");
    expect(budgetSummary({ ...PLAN.budget, max_rounds: 1 })).toContain("1 round");
  });

  it("never renders a non-finite chip-hour budget as a number", () => {
    expect(budgetSummary({ ...PLAN.budget, max_chip_hours: Number.NaN })).toContain("unknown chip-hours");
  });
});
