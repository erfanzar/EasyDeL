// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EyvanApiClient } from "../api/client";
import { parseRoute, routeHash } from "../routing";
import type { ResearchPlan, ResearchPlanSummary } from "../types";
import { ResearchPage, ResearchPlanPage } from "./ResearchPage";

const SUMMARY: ResearchPlanSummary = {
  id: "rp_0123456789abcdef",
  title: "lr sweep",
  subject: "alice",
  org_id: "acme",
  project_id: "r",
  state: "running",
  iterate: false,
  provider_id: null,
  model: null,
  current_round: 0,
  jobs_planned: 2,
  budget: { max_jobs: 2, max_chip_hours: 1, max_wall_seconds: 600, max_rounds: null },
  spent: { chip_hours: 0.5, jobs_submitted: 1, rounds: 0 },
  stop_reason: null,
  approved_at: "2026-10-28T10:01:00+00:00",
  approved_by: "alice",
  started_at: "2026-10-28T10:01:05+00:00",
  finished_at: null,
  created_at: "2026-10-28T10:00:00+00:00",
  updated_at: "2026-10-28T10:02:00+00:00",
};

const DOCUMENT: ResearchPlan = {
  ...SUMMARY,
  goal: "find the best learning rate",
  jobs: [
    {
      index: 0,
      name: "lr-a",
      round: 0,
      spec: {},
      state: "running",
      workload_id: "wl_a",
      submitted_at: SUMMARY.started_at,
    },
    { index: 1, name: "lr-b", round: 0, spec: {}, state: "not_submitted" },
  ],
  workload_ids: ["wl_a"],
  remaining: { jobs: 1, chip_hours: 0.5 },
  notes: [
    { t: "2026-10-28T10:00:00+00:00", kind: "proposed", text: "alice proposed 2 job(s)" },
    { t: "2026-10-28T10:01:00+00:00", kind: "approved", text: "approved by alice" },
  ],
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

type DetailProps = Parameters<typeof ResearchPlanPage>[0];

function renderDetail(overrides: Partial<DetailProps> = {}) {
  const props: DetailProps & { rerender?: (ui: ReactElement) => void } = {
    planId: DOCUMENT.id,
    loadPlan: vi.fn(async () => DOCUMENT),
    onBack: vi.fn(),
    onOpenWorkload: vi.fn(),
    approve: vi.fn(async () => ({ ...DOCUMENT, state: "approved" as const })),
    stop: vi.fn(async () => ({ ...DOCUMENT, state: "stopping" as const })),
    currentSubject: "alice",
    canAdminister: false,
    ...overrides,
  };
  const view = render(<ResearchPlanPage {...props} />);
  props.rerender = view.rerender;
  return props;
}

function rerenderDetail(props: ReturnType<typeof renderDetail>, overrides: Partial<DetailProps>) {
  const { rerender, ...rest } = props;
  rerender?.(<ResearchPlanPage {...rest} {...overrides} />);
}

const heading = () => document.querySelector(".research-heading [data-testid=research-state]")?.textContent;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

describe("research routes", () => {
  it("round trips the list and an encoded plan id", () => {
    expect(parseRoute("#/research")).toEqual({ section: "research" });
    const route = { section: "research", planId: "rp_0123456789abcdef" } as const;
    expect(routeHash(route)).toBe("#/research/rp_0123456789abcdef");
    expect(parseRoute(routeHash(route))).toEqual(route);
    expect(parseRoute("#/research/%E0%A4%A")).toEqual({ section: "research" });
  });
});

describe("ResearchPage", () => {
  it("lists plans with state, owner, jobs and spend, and opens one", async () => {
    const onOpenPlan = vi.fn();
    render(<ResearchPage loadPlans={async () => [SUMMARY]} onOpenPlan={onOpenPlan} />);
    const row = (await screen.findByRole("button", { name: "lr sweep" })).closest("tr") as HTMLElement;
    expect(within(row).getByTestId("research-state").textContent).toBe("running");
    expect(row.textContent).toContain("alice");
    expect(row.textContent).toContain("1 / 2");
    expect(row.textContent).toContain("0.5 / 1");
    fireEvent.click(within(row).getByRole("button", { name: "lr sweep" }));
    expect(onOpenPlan).toHaveBeenCalledWith(SUMMARY.id);
  });

  it("says when there are no plans, and shows a load failure with retry", async () => {
    render(<ResearchPage loadPlans={async () => []} onOpenPlan={() => undefined} />);
    expect(await screen.findByText("No research plans yet")).toBeTruthy();
    cleanup();

    const loadPlans = vi
      .fn<(signal: AbortSignal) => Promise<ResearchPlanSummary[]>>()
      .mockRejectedValueOnce(new Error("Eyvan API request failed with 500"))
      .mockResolvedValue([SUMMARY]);
    render(<ResearchPage loadPlans={loadPlans} onOpenPlan={() => undefined} />);
    expect((await screen.findByRole("alert")).textContent).toContain("failed with 500");
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: "lr sweep" })).toBeTruthy();
  });
});

describe("ResearchPlanPage", () => {
  it("shows jobs with workload links, spend against budget and notes", async () => {
    const props = renderDetail();
    const jobs = await screen.findAllByTestId("research-job");
    expect(jobs).toHaveLength(2);
    expect(jobs[0].textContent).toContain("lr-a");
    expect(jobs[0].textContent).toContain("running");
    expect(jobs[1].textContent).toContain("not submitted");
    fireEvent.click(within(jobs[0]).getByRole("button", { name: "wl_a" }));
    expect(props.onOpenWorkload).toHaveBeenCalledWith("wl_a");

    const spend = screen.getByRole("region", { name: "Spend against budget" });
    expect(within(spend).getByRole("progressbar", { name: "Jobs used" }).getAttribute("aria-valuenow")).toBe("50");
    expect(within(spend).getByRole("progressbar", { name: "Chip-hours used" }).getAttribute("aria-valuenow")).toBe(
      "50",
    );
    expect(spend.textContent).toContain("1 of 2 submitted");
    expect(spend.textContent).toContain("of 10 min");
    // No round limit was set, so no rounds meter.
    expect(within(spend).queryByRole("progressbar", { name: "Rounds used" })).toBeNull();

    const notes = screen.getByRole("region", { name: "Notes" });
    expect(notes.textContent).toContain("alice proposed 2 job(s)");
    expect(notes.textContent).toContain("approved by alice");
    expect(screen.getByText("find the best learning rate")).toBeTruthy();
  });

  it("stops a running plan after confirmation and shows the new state", async () => {
    const props = renderDetail();
    await screen.findAllByTestId("research-job");
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(props.stop).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Confirm stop" }));
    await waitFor(() => expect(screen.getAllByTestId("research-state")[0].textContent).toBe("stopping"));
    expect(props.stop).toHaveBeenCalledWith(DOCUMENT.id);
  });

  it("approves a proposed plan as its owner, and shows a refusal", async () => {
    const proposed = {
      ...DOCUMENT,
      state: "proposed" as const,
      approved_at: null,
      approved_by: null,
      started_at: null,
    };
    const approve = vi
      .fn<(id: string) => Promise<ResearchPlan>>()
      .mockRejectedValueOnce(new Error("Eyvan API request failed with 403: may not submit"))
      .mockResolvedValueOnce({ ...proposed, state: "approved" });
    renderDetail({ loadPlan: async () => proposed, approve });
    await screen.findAllByTestId("research-job");
    expect(screen.getByText(/Nothing runs until you approve/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm approve" }));
    expect((await screen.findByRole("alert")).textContent).toContain("may not submit");
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm approve" }));
    await waitFor(() => expect(screen.getAllByTestId("research-state")[0].textContent).toBe("approved"));
    expect(approve).toHaveBeenCalledTimes(2);
  });

  it("lets a fleet admin stop someone else's plan but not approve it", async () => {
    const proposed = { ...DOCUMENT, state: "proposed" as const };
    renderDetail({ loadPlan: async () => proposed, currentSubject: "root", canAdminister: true });
    await screen.findAllByTestId("research-job");
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
    cleanup();
    renderDetail({ currentSubject: "root", canAdminister: true });
    await screen.findAllByTestId("research-job");
    expect(screen.getByRole("button", { name: "Stop" })).toBeTruthy();
  });

  it("shows a plan that cannot be read", async () => {
    renderDetail({ loadPlan: async () => Promise.reject(new Error("Eyvan API request failed with 404")) });
    expect((await screen.findByRole("alert")).textContent).toContain("failed with 404");
  });

  it("ignores a detail read that started before Stop and resolved after it", async () => {
    const slow = deferred<ResearchPlan>();
    const loadPlan = vi
      .fn<(id: string, signal: AbortSignal) => Promise<ResearchPlan>>()
      .mockResolvedValueOnce(DOCUMENT)
      .mockImplementationOnce(() => slow.promise)
      .mockResolvedValue(DOCUMENT);
    const stop = vi.fn(async () => ({
      ...DOCUMENT,
      state: "stopping" as const,
      updated_at: "2026-10-28T10:06:00+00:00",
    }));
    const props = renderDetail({ loadPlan, stop });
    await screen.findAllByTestId("research-job");
    // A live refresh starts a second read, which is still in flight when Stop answers.
    rerenderDetail(props, { refreshRevision: 1 });
    await waitFor(() => expect(loadPlan).toHaveBeenCalledTimes(2));
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm stop" }));
    await waitFor(() => expect(heading()).toBe("stopping"));
    await act(async () => slow.resolve(DOCUMENT));
    expect(heading()).toBe("stopping");
    // A later read older than the shown one (not aborted) is ignored as well.
    rerenderDetail(props, { refreshRevision: 2, loadPlan, stop });
    await waitFor(() => expect(loadPlan).toHaveBeenCalledTimes(3));
    await act(async () => undefined);
    expect(heading()).toBe("stopping");
  });

  it("clears a stale refresh-failure banner after a successful action", async () => {
    const loadPlan = vi
      .fn<(id: string, signal: AbortSignal) => Promise<ResearchPlan>>()
      .mockResolvedValueOnce(DOCUMENT)
      .mockRejectedValue(new Error("Eyvan API request failed with 503"));
    const props = renderDetail({ loadPlan });
    await screen.findAllByTestId("research-job");
    rerenderDetail(props, { refreshRevision: 1, loadPlan });
    expect((await screen.findByRole("alert")).textContent).toContain("Refresh failed");
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm stop" }));
    await waitFor(() => expect(heading()).toBe("stopping"));
    expect(screen.queryByText(/Refresh failed/)).toBeNull();
  });

  it("keeps each note's row when a newer note arrives", async () => {
    const newer: ResearchPlan = {
      ...DOCUMENT,
      updated_at: "2026-10-28T10:03:00+00:00",
      notes: [...DOCUMENT.notes, { t: "2026-10-28T10:03:00+00:00", kind: "submitted", text: "submitted lr-b" }],
    };
    const loadPlan = vi
      .fn<(id: string, signal: AbortSignal) => Promise<ResearchPlan>>()
      .mockResolvedValueOnce(DOCUMENT)
      .mockResolvedValue(newer);
    const props = renderDetail({ loadPlan });
    const before = (await screen.findByText("approved by alice")).closest("li");
    rerenderDetail(props, { refreshRevision: 1, loadPlan });
    await screen.findByText("submitted lr-b");
    expect(screen.getByText("approved by alice").closest("li")).toBe(before);
  });

  it("shows spend just under the budget as under it, and an unreadable figure as unknown", async () => {
    const spend = (plan: ResearchPlan) => {
      cleanup();
      renderDetail({ loadPlan: async () => plan });
      return screen.findByRole("region", { name: "Spend against budget" });
    };
    let region = await spend({ ...DOCUMENT, spent: { ...DOCUMENT.spent, chip_hours: 0.996 } });
    const chip = within(region).getByRole("progressbar", { name: "Chip-hours used" });
    expect(region.textContent).not.toContain("1 of 1 chip-hour");
    expect(region.textContent).toContain("0.996 of 1 chip-hour");
    expect(chip.querySelector(".progress-red")).toBeNull();

    region = await spend({ ...DOCUMENT, spent: { ...DOCUMENT.spent, chip_hours: 1 } });
    expect(
      within(region).getByRole("progressbar", { name: "Chip-hours used" }).querySelector(".progress-red"),
    ).not.toBeNull();

    region = await spend({ ...DOCUMENT, spent: { ...DOCUMENT.spent, chip_hours: Number.NaN } });
    const unknown = within(region).getByRole("progressbar", { name: "Chip-hours used" });
    expect(unknown.hasAttribute("aria-valuenow")).toBe(false);
    expect(unknown.getAttribute("aria-valuetext")).toBe("unknown");
    expect(region.textContent).toContain("unknown of 1 chip-hour");
    expect(region.textContent).not.toContain("NaN");
  });

  it("labels a failed plan's reason as a failure", async () => {
    renderDetail({ loadPlan: async () => ({ ...DOCUMENT, state: "failed" as const, stop_reason: "job lr-a failed" }) });
    const reason = await screen.findByText("job lr-a failed");
    expect(reason.parentElement?.textContent).toBe("Failed job lr-a failed");
  });
});

describe("ResearchPage counts", () => {
  it('does not say "1 plans"', async () => {
    render(<ResearchPage loadPlans={async () => [SUMMARY]} onOpenPlan={() => undefined} />);
    await screen.findByRole("button", { name: "lr sweep" });
    expect(screen.getByText("1 plan")).toBeTruthy();
    expect(screen.queryByText("1 plans")).toBeNull();
  });
});

describe("research client methods", () => {
  it("call the research routes, with the stop reason in the body", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const path = String(input);
      if (path === "/api/v1/research/plans?state=running") {
        return new Response(JSON.stringify({ plans: [SUMMARY] }), { headers: { "content-type": "application/json" } });
      }
      if (path.startsWith("/api/v1/research/plans/")) {
        return new Response(JSON.stringify(DOCUMENT), { headers: { "content-type": "application/json" } });
      }
      return new Response(JSON.stringify({ detail: `unexpected ${init?.method ?? "GET"} ${path}` }), { status: 500 });
    });
    const client = new EyvanApiClient("/api/v1", false);
    expect(await client.listResearchPlans("running")).toEqual([SUMMARY]);
    expect((await client.getResearchPlan(DOCUMENT.id)).jobs).toHaveLength(2);
    await client.approveResearchPlan(DOCUMENT.id);
    await client.stopResearchPlan(DOCUMENT.id, "enough");
    const calls = fetch.mock.calls.map(([path, init]) => [String(path), init?.method ?? "GET", init?.body]);
    expect(calls).toEqual([
      ["/api/v1/research/plans?state=running", "GET", undefined],
      [`/api/v1/research/plans/${DOCUMENT.id}`, "GET", undefined],
      [`/api/v1/research/plans/${DOCUMENT.id}/approve`, "POST", undefined],
      [`/api/v1/research/plans/${DOCUMENT.id}/stop`, "POST", JSON.stringify({ reason: "enough" })],
    ]);
  });

  it("never sends approve or stop in demo mode", async () => {
    const fetch = vi.spyOn(globalThis, "fetch");
    const demo = new EyvanApiClient("/api/v1", true);
    await expect(demo.approveResearchPlan("rp_x")).rejects.toMatchObject({ status: 400 });
    await expect(demo.stopResearchPlan("rp_x")).rejects.toMatchObject({ status: 400 });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("surfaces the plane's refusal detail", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "research plan rp_x is already approved" }), {
        status: 409,
        headers: { "content-type": "application/json" },
      }),
    );
    await expect(new EyvanApiClient("/api/v1", false).approveResearchPlan("rp_x")).rejects.toMatchObject({
      status: 409,
      message: expect.stringContaining("already approved"),
    });
  });
});
