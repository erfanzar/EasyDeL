// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "../api/client";
import { formatAgeSeconds } from "../lib/format";
import type { AuditEvent, Cell, TpuQuotaEntry, TpuQuotaSnapshot } from "../types";
import { CapacityPage } from "./CapacityPage";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function entry(overrides: Partial<TpuQuotaEntry> = {}): TpuQuotaEntry {
  return {
    zone: "us-central2-b",
    region: "us-central2",
    generation: "v6e",
    accelerator_type: "v6e-8",
    capacity_class: "spot",
    metric: "tpu.googleapis.com/tpu-v6e-preemptible",
    chips_per_slice: 8,
    limit_chips: 32,
    in_use_chips: 16,
    free_chips: 16,
    free_slices: 2,
    state: "known",
    usage_observed_at: "2026-07-29T12:00:00Z",
    usage_age_s: 300,
    ...overrides,
  };
}

function snapshot(overrides: Partial<TpuQuotaSnapshot> = {}): TpuQuotaSnapshot {
  return {
    observed_at: "2026-07-29T12:05:00Z",
    provider: "gcp",
    project: "gcp-project",
    available_projects: ["gcp-project"],
    status: "ok",
    refreshed_at: "2026-07-29T12:04:30Z",
    age_s: 30,
    cached: false,
    ttl_s: 120,
    entries: [entry()],
    failures: [],
    ...overrides,
  };
}

function renderPage(value: TpuQuotaSnapshot, calls: Array<{ refresh?: boolean } | undefined> = []) {
  return render(
    <CapacityPage
      loadQuota={async (_signal, options) => {
        calls.push(options);
        return value;
      }}
    />,
  );
}

/** The measured table, addressed by the label that describes what it contains. */
function measuredTable(): HTMLElement {
  return screen.getByRole("region", { name: "Free capacity by zone" });
}

function unknownTable(): HTMLElement {
  return screen.getByRole("region", { name: "Zones with no measurable free balance" });
}

describe("the quota page", () => {
  it("reports limit, in use, and free for each zone, type, and purchase class", async () => {
    renderPage(
      snapshot({
        entries: [
          entry(),
          entry({
            zone: "europe-west4-a",
            region: "europe-west4",
            capacity_class: "on_demand",
            metric: "tpu.googleapis.com/tpu-v6e",
            limit_chips: 8,
            in_use_chips: 0,
            free_chips: 8,
            free_slices: 1,
          }),
        ],
      }),
    );

    await waitFor(() => expect(screen.getByText("Provider quota")).toBeDefined());
    const rows = within(measuredTable()).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2);
    // Sorted by free capacity, and Spot is kept apart from on-demand because
    // they are different provider quotas that cannot be spent interchangeably.
    const cells = within(rows[0])
      .getAllByRole("cell")
      .map((cell) => cell.textContent);
    expect(cells).toEqual(["us-central2-bus-central2", "v6e-8", "Spot", "32", "16", "16", "2 x v6e-8", "5m ago"]);
    expect(within(rows[1]).getAllByRole("cell")[2].textContent).toBe("On Demand");
  });

  it("never renders an unmeasured zone as empty, and keeps it out of the sorted table", async () => {
    // The single most important behaviour on this page. A limit with no usage
    // series is not a free quota, and a zero here reads as "all 64 are yours".
    renderPage(
      snapshot({
        entries: [
          entry(),
          entry({
            zone: "europe-west4-a",
            region: "europe-west4",
            limit_chips: 64,
            in_use_chips: null,
            free_chips: null,
            free_slices: null,
            state: "usage_unknown",
            usage_observed_at: null,
            usage_age_s: null,
          }),
        ],
      }),
    );

    await waitFor(() => expect(screen.getByText("Usage unknown")).toBeDefined());
    const measuredZones = within(measuredTable())
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getAllByRole("cell")[0].textContent);
    expect(measuredZones).toEqual(["us-central2-bus-central2"]);

    const unknownRow = within(unknownTable()).getAllByRole("row")[1];
    const unknownCells = within(unknownRow)
      .getAllByRole("cell")
      .map((cell) => cell.textContent);
    expect(unknownCells[0]).toContain("europe-west4-a");
    // The limit is still worth knowing; the balance is not invented.
    expect(unknownCells[3]).toBe("64");
    expect(unknownCells[4]).toBe("unknown");
    expect(unknownCells.some((text) => text === "0")).toBe(false);
    expect(unknownCells[5]).toContain("usage unavailable");
  });

  it("counts unmeasured zones separately instead of folding them into free capacity", async () => {
    renderPage(
      snapshot({
        entries: [
          entry({ free_chips: 16 }),
          entry({
            zone: "europe-west4-a",
            region: "europe-west4",
            limit_chips: 64,
            in_use_chips: null,
            free_chips: null,
            free_slices: null,
            state: "usage_unknown",
            usage_observed_at: null,
            usage_age_s: null,
          }),
        ],
      }),
    );

    await waitFor(() => expect(screen.getByText("Free chips")).toBeDefined());
    const freeCard = screen.getByText("Free chips").closest("article");
    if (!freeCard) throw new Error("the free-capacity metric card is missing");
    // 16, not 80: the unmeasured limit contributes nothing to a free total.
    expect(within(freeCard).getByText("16")).toBeDefined();
    const unknownCard = screen.getByText("Zones with unknown usage").closest("article");
    if (!unknownCard) throw new Error("the unknown-usage metric card is missing");
    expect(within(unknownCard).getByText("1")).toBeDefined();
    expect(screen.getByText("not empty and not full -- nobody could tell")).toBeDefined();
  });

  it("answers a fit question without letting an unmeasured zone disappear", async () => {
    // "We could not tell" is a different answer from "it does not fit", and only
    // one of them should stop an operator trying that zone.
    renderPage(
      snapshot({
        entries: [
          entry({ zone: "us-central2-b", free_chips: 16, free_slices: 2 }),
          entry({ zone: "us-east5-a", region: "us-east5", free_chips: 8, in_use_chips: 24, free_slices: 1 }),
          entry({
            zone: "europe-west4-a",
            region: "europe-west4",
            limit_chips: 64,
            in_use_chips: null,
            free_chips: null,
            free_slices: null,
            state: "usage_unknown",
            usage_observed_at: null,
            usage_age_s: null,
          }),
        ],
      }),
    );

    await waitFor(() => expect(screen.getByText("Provider quota")).toBeDefined());
    fireEvent.change(screen.getByRole("combobox", { name: "Only zones that fit this accelerator type" }), {
      target: { value: "v6e-8" },
    });
    fireEvent.change(screen.getByRole("spinbutton", { name: "How many of that accelerator type" }), {
      target: { value: "2" },
    });

    const zones = within(measuredTable())
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getAllByRole("cell")[0].textContent);
    expect(zones).toEqual(["us-central2-bus-central2"]);
    // The zone with a measured balance of one slice is genuinely excluded.
    expect(within(measuredTable()).queryByText("us-east5-a")).toBeNull();
    // The unmeasured zone is still on the page, under its own heading.
    expect(within(unknownTable()).getByText("europe-west4-a")).toBeDefined();
  });

  it("says nothing fits while naming the zones nobody could measure", async () => {
    renderPage(
      snapshot({
        entries: [
          entry({ free_chips: 8, in_use_chips: 24, free_slices: 1 }),
          entry({
            zone: "europe-west4-a",
            region: "europe-west4",
            limit_chips: 64,
            in_use_chips: null,
            free_chips: null,
            free_slices: null,
            state: "usage_unknown",
            usage_observed_at: null,
            usage_age_s: null,
          }),
        ],
      }),
    );

    await waitFor(() => expect(screen.getByText("Provider quota")).toBeDefined());
    fireEvent.change(screen.getByRole("combobox", { name: "Only zones that fit this accelerator type" }), {
      target: { value: "v6e-8" },
    });
    fireEvent.change(screen.getByRole("spinbutton", { name: "How many of that accelerator type" }), {
      target: { value: "4" },
    });

    expect(screen.getByText("No measured zone fits 4 x v6e-8")).toBeDefined();
    expect(screen.getByText(/could not be measured and are listed below. They are not proven full./)).toBeDefined();
  });

  it("shows the age of the provider reading rather than presenting it as live", async () => {
    renderPage(snapshot({ cached: true, age_s: 95 }));

    await waitFor(() => expect(screen.getByText(/Provider read 1m ago/)).toBeDefined());
    expect(screen.getByText(/cached reading, not re-read for this view/)).toBeDefined();
  });

  it("marks a reading served past its own refresh window", async () => {
    // The deadline path: the provider did not answer in time, so the previous
    // reading was returned. It must not look like a fresh one.
    renderPage(snapshot({ cached: true, age_s: 900, ttl_s: 120 }));

    await waitFor(() => expect(screen.getByText("Older than the 120s refresh window")).toBeDefined());
  });

  it("shows how old each usage observation is, per row", async () => {
    renderPage(snapshot({ entries: [entry({ usage_age_s: 7200 })] }));

    await waitFor(() => expect(screen.getByText("2h ago")).toBeDefined());
  });

  it("names a failed purchase class instead of quietly reporting only the other one", async () => {
    renderPage(
      snapshot({
        entries: [entry({ capacity_class: "on_demand" })],
        failures: [{ capacity_class: "spot", reason: "gcloud beta quotas info list failed: PERMISSION_DENIED" }],
      }),
    );

    await waitFor(() =>
      expect(screen.getByText(/Spot quota could not be read, so no zone below reports it/)).toBeDefined(),
    );
    expect(screen.getByText(/PERMISSION_DENIED/)).toBeDefined();
  });

  it("reports an unreadable snapshot rather than an empty table", async () => {
    renderPage(
      snapshot({
        status: "unavailable",
        entries: [],
        failures: [
          { capacity_class: "spot", reason: "gcloud auth token lookup failed" },
          { capacity_class: "on_demand", reason: "gcloud auth token lookup failed" },
        ],
      }),
    );

    await waitFor(() => expect(screen.getAllByText(/gcloud auth token lookup failed/)).toHaveLength(2));
    // No summary tile either. "Free chips: 0" beside a failure banner is the
    // same lie as a zero in a row, one level up.
    expect(screen.queryByText("Free chips")).toBeNull();
    expect(screen.getByText("No provider quota could be read")).toBeDefined();
    expect(
      screen.getByText(/No zone is counted, listed, or shown as empty, because none of them were read/),
    ).toBeDefined();
  });

  it("separates an unconfigured fleet from a fleet with no quota", async () => {
    renderPage(
      snapshot({
        status: "unconfigured",
        project: null,
        available_projects: [],
        entries: [],
        refreshed_at: null,
        age_s: null,
      }),
    );

    await waitFor(() => expect(screen.getByText(/No GCP capacity pool declares a provider project/)).toBeDefined());
    expect(screen.getByText("The provider has not been read")).toBeDefined();
    // No table and no summary tile, rather than an empty one that reads as
    // "this fleet has no capacity anywhere".
    expect(screen.queryByRole("region", { name: "Free capacity by zone" })).toBeNull();
    expect(screen.queryByText("Free chips")).toBeNull();
    expect(screen.getByText("This fleet declares no provider project")).toBeDefined();
  });

  it("reports a request failure instead of rendering a fleet with no capacity", async () => {
    render(
      <CapacityPage
        loadQuota={async () => {
          throw new Error("Eyvan API request timed out");
        }}
      />,
    );

    await waitFor(() => expect(screen.getByText("Eyvan API request timed out")).toBeDefined());
    expect(screen.getByText("No zone is shown as empty, because none of them were read.")).toBeDefined();
  });

  it("re-reads the provider only when asked to", async () => {
    const calls: Array<{ refresh?: boolean } | undefined> = [];
    renderPage(snapshot(), calls);

    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0]).toEqual({ refresh: false });

    fireEvent.click(screen.getByRole("button", { name: /Re-read provider/ }));

    await waitFor(() => expect(calls).toHaveLength(2));
    expect(calls[1]).toEqual({ refresh: true });
  });

  it("keeps a refreshed quota when an earlier scope's request completes last", async () => {
    let resolveOld!: (value: TpuQuotaSnapshot) => void;
    let resolveRefresh!: (value: TpuQuotaSnapshot) => void;
    const page = render(
      <CapacityPage loadQuota={() => new Promise<TpuQuotaSnapshot>((resolve) => (resolveOld = resolve))} />,
    );
    page.rerender(
      <CapacityPage
        loadQuota={(_signal, options) =>
          options?.refresh
            ? new Promise<TpuQuotaSnapshot>((resolve) => (resolveRefresh = resolve))
            : Promise.resolve(snapshot({ entries: [entry({ zone: "new-scope-zone" })] }))
        }
      />,
    );
    await screen.findByText("new-scope-zone");

    fireEvent.click(screen.getByRole("button", { name: /Re-read provider/ }));
    await act(async () => resolveOld(snapshot({ entries: [entry({ zone: "old-zone" })] })));
    expect(screen.queryByText("old-zone")).toBeNull();
    await act(async () => resolveRefresh(snapshot({ entries: [entry({ zone: "fresh-zone" })] })));
    expect(screen.getByText("fresh-zone")).toBeTruthy();
    expect(screen.queryByText("old-zone")).toBeNull();
  });

  it("hides an earlier scope's quota while its replacement read is pending", async () => {
    const page = render(
      <CapacityPage loadQuota={async () => snapshot({ entries: [entry({ zone: "old-project-zone" })] })} />,
    );
    await screen.findByText("old-project-zone");

    page.rerender(<CapacityPage loadQuota={() => new Promise<TpuQuotaSnapshot>(() => {})} />);

    expect(screen.queryByText("old-project-zone")).toBeNull();
    expect(screen.queryByRole("region", { name: "Free capacity by zone" })).toBeNull();
  });

  it("keeps an unlimited quota out of the free totals it has no number for", async () => {
    renderPage(
      snapshot({
        entries: [
          entry({
            zone: "us-east5-a",
            region: "us-east5",
            limit_chips: null,
            in_use_chips: null,
            free_chips: null,
            free_slices: null,
            state: "unlimited",
            usage_observed_at: null,
            usage_age_s: null,
          }),
        ],
      }),
    );

    await waitFor(() => expect(screen.getByText("Usage unknown")).toBeDefined());
    expect(screen.getByText(/no quota ceiling/)).toBeDefined();
    expect(screen.getByText("No zone reports a measured free balance")).toBeDefined();
  });
});

describe("age formatting", () => {
  it("says unknown rather than inventing a fresh observation", () => {
    // A row whose usage nobody could read has no age. "0s ago" would present
    // the absence of a measurement as the most recent possible one.
    expect(formatAgeSeconds(null)).toBe("unknown");
    expect(formatAgeSeconds(undefined)).toBe("unknown");
    expect(formatAgeSeconds(Number.NaN)).toBe("unknown");
    expect(formatAgeSeconds(-1)).toBe("unknown");
  });

  it("rounds down so an age is never reported as newer than it is", () => {
    expect(formatAgeSeconds(0)).toBe("0s ago");
    expect(formatAgeSeconds(59.9)).toBe("59s ago");
    expect(formatAgeSeconds(60)).toBe("1m ago");
    expect(formatAgeSeconds(5399)).toBe("89m ago");
    expect(formatAgeSeconds(7200)).toBe("2h ago");
  });
});

describe("the page and the client method agree", () => {
  it("loads through the real client method, including the re-read option", async () => {
    // Not a redundant wiring test: the page receives a loader from App, so a
    // signature drift between that loader and `apiClient.getTpuQuota` would be
    // invisible to every test that stubs the loader directly.
    const spy = vi.spyOn(apiClient, "getTpuQuota").mockResolvedValue(snapshot());

    render(
      <CapacityPage loadQuota={(signal, options) => apiClient.getTpuQuota(apiClient.getScope(), signal, options)} />,
    );

    await waitFor(() => expect(screen.getByText("Provider quota")).toBeDefined());
    expect(spy.mock.calls[0][2]).toEqual({ refresh: false });

    fireEvent.click(screen.getByRole("button", { name: /Re-read provider/ }));

    await waitFor(() => expect(spy.mock.calls).toHaveLength(2));
    expect(spy.mock.calls[1][2]).toEqual({ refresh: true });
  });
});

describe("what each zone has been doing", () => {
  const cell = (overrides: Partial<Cell> = {}): Cell => ({
    id: "cell-a",
    name: "cell-a",
    backend: "gcp-tpu",
    region: "us-east5",
    zone: "us-east5-b",
    status: "healthy",
    utilization: 50,
    activeWorkloads: 1,
    topology: "v6e-8",
    resources: [],
    lastHeartbeat: "2026-08-17T09:00:00Z",
    ...overrides,
  });

  const renderZones = (cells: Cell[], events: AuditEvent[] | Error = []) =>
    render(
      <CapacityPage
        loadQuota={async () => snapshot({ status: "unconfigured", entries: [] })}
        cells={cells}
        loadCapacityEvents={async () => (events instanceof Error ? Promise.reject(events) : events)}
      />,
    );

  it("names a zone the planner is deliberately not asking", async () => {
    // A cooldown is a decision, and on every other view it is indistinguishable
    // from a zone with no capacity. The two call for opposite reactions: wait,
    // or go somewhere else.
    const future = new Date(Date.now() + 600_000).toISOString();
    renderZones([cell({ availability: { cooldownUntil: future, consecutiveFailures: 3, interruptions: 2 } })]);

    expect(await screen.findByText(/cooling until/)).toBeTruthy();
  });

  it("does not report an expired cooldown as one still in force", async () => {
    const past = new Date(Date.now() - 600_000).toISOString();
    renderZones([cell({ availability: { cooldownUntil: past, consecutiveFailures: 0, interruptions: 0 } })]);

    // The row exists and reports nothing withheld, which is the distinction:
    // an expired cooldown is not a cooldown.
    const rows = await screen.findAllByRole("row");
    const zone = rows.find((row) => row.textContent?.includes("us-east5-b"));
    expect(zone?.textContent).toContain("none");
    expect(screen.queryByText(/cooling until/)).toBeNull();
  });

  it("sums what a zone holds rather than listing every machine", async () => {
    renderZones([cell(), cell({ id: "cell-b", name: "cell-b" }), cell({ id: "cell-c", zone: "us-east5-a" })]);

    const rows = await screen.findAllByRole("row");
    const east5b = rows.find((row) => row.textContent?.includes("us-east5-b"));
    expect(east5b?.textContent).toContain("2");
  });

  it("names a failed history read rather than showing a quiet fleet", async () => {
    // An empty history reads as "nothing has happened here", which is the
    // opposite of what a failed read means on the afternoon somebody looks.
    renderZones([cell()], new Error("Eyvan API request failed with 403"));

    expect(await screen.findByText(/403/)).toBeTruthy();
  });

  it("drops old fleet history immediately when its permission or loader changes", async () => {
    const history: AuditEvent[] = [
      {
        id: "history-old",
        timestamp: "2026-08-17T09:00:00Z",
        action: "capacity.granted",
        actor: "operator",
        target: "old-fleet-cell",
        category: "capacity",
        severity: "info",
        detail: "",
      },
    ];
    const quota = async () => snapshot({ status: "unconfigured", entries: [] });
    const page = render(<CapacityPage loadQuota={quota} loadCapacityEvents={async () => history} />);
    await screen.findByText("old-fleet-cell");

    page.rerender(<CapacityPage loadQuota={quota} />);
    expect(screen.queryByText("old-fleet-cell")).toBeNull();
    expect(screen.queryByRole("region", { name: "Zone capacity history" })).toBeNull();

    page.rerender(<CapacityPage loadQuota={quota} loadCapacityEvents={() => new Promise<AuditEvent[]>(() => {})} />);
    expect(screen.queryByText("old-fleet-cell")).toBeNull();
    expect(screen.queryByText("No capacity has been granted or released recently.")).toBeNull();

    page.rerender(
      <CapacityPage
        loadQuota={quota}
        loadCapacityEvents={async () => {
          throw new Error("new fleet history unavailable");
        }}
      />,
    );
    await screen.findByText("new fleet history unavailable");
    expect(screen.queryByText("old-fleet-cell")).toBeNull();
  });

  it("says a fleet has been quiet only when the read succeeded", async () => {
    renderZones([cell()], []);

    expect(await screen.findByText(/No capacity has been granted or released recently/)).toBeTruthy();
  });

  it("lays each capacity event out as time, rail and body, like the workload timeline", async () => {
    // The list is a three-column grid. A row with only two cells put the
    // marker in the time column and squeezed the whole event into the 22px
    // rail column, one word per line.
    const events: AuditEvent[] = [
      {
        id: "drain-1",
        timestamp: "2026-08-20T13:19:00Z",
        action: "capacity.drain",
        actor: "controller",
        target: "research-southamerica-west1-a-v6e-8-ca0244-0008",
        category: "capacity",
        severity: "warning",
        detail: "",
      },
    ];
    renderZones([cell()], events);

    const target = await screen.findByText("research-southamerica-west1-a-v6e-8-ca0244-0008");
    const row = target.closest("li");
    expect(row).not.toBeNull();
    const cells = Array.from(row!.children);
    expect(cells.map((child) => child.tagName.toLowerCase())).toEqual(["time", "span", "div"]);
    expect(cells[1].classList.contains("timeline-rail")).toBe(true);
    expect(cells[1].querySelector(".timeline-marker")).not.toBeNull();
    expect(cells[2].classList.contains("timeline-body")).toBe(true);
    expect(cells[2].textContent).toContain("capacity.drain");
  });
});

describe("one row per quota, not per slice size", () => {
  it("collapses the sizes that draw on a single granted quota", async () => {
    // Cloud TPU quota is granted against a generation: one metric, one limit,
    // one usage reading. The provider then offers ninety-odd slice sizes that
    // all draw on it, and every one got a row -- fifty-two for a single v4
    // zone, identical but for a size the Fits column already derives.
    const sizes = [16, 128, 1024, 2048];
    render(
      <CapacityPage
        loadQuota={async () =>
          snapshot({
            entries: sizes.map((chips) =>
              entry({
                generation: "v4",
                accelerator_type: `v4-${chips}`,
                chips_per_slice: chips,
                metric: "tpu.googleapis.com/tpu-v4s-pod-preemptible",
                limit_chips: 80000,
                in_use_chips: 0,
                free_chips: 80000,
                free_slices: Math.floor(80000 / chips),
              }),
            ),
          })
        }
      />,
    );

    await waitFor(() => expect(within(measuredTable()).getAllByRole("row").length).toBeGreaterThan(1));
    const rows = within(measuredTable()).getAllByRole("row").slice(1);

    expect(rows).toHaveLength(1);
    expect(rows[0].textContent).toContain("v4");
    // The sizes are not lost, they are described.
    expect(rows[0].textContent).toContain("4 slice sizes");
  });

  it("keeps quotas apart when they really are different grants", async () => {
    render(
      <CapacityPage
        loadQuota={async () =>
          snapshot({
            entries: [
              entry({ generation: "v4", accelerator_type: "v4-16", chips_per_slice: 16 }),
              entry({ generation: "v6e", accelerator_type: "v6e-8", chips_per_slice: 8 }),
              entry({ generation: "v6e", accelerator_type: "v6e-8", chips_per_slice: 8, capacity_class: "on_demand" }),
            ],
          })
        }
      />,
    );

    await waitFor(() => expect(within(measuredTable()).getAllByRole("row").length).toBeGreaterThan(1));

    expect(within(measuredTable()).getAllByRole("row").slice(1)).toHaveLength(3);
  });

  it("counts a zone's free chips once rather than once per size", async () => {
    // The summary summed over slice sizes, so one v4 zone with 80000 free
    // reported four million.
    render(
      <CapacityPage
        loadQuota={async () =>
          snapshot({
            entries: [16, 128, 1024].map((chips) =>
              entry({
                generation: "v4",
                accelerator_type: `v4-${chips}`,
                chips_per_slice: chips,
                limit_chips: 80000,
                in_use_chips: 0,
                free_chips: 80000,
                free_slices: Math.floor(80000 / chips),
              }),
            ),
          })
        }
      />,
    );

    await waitFor(() => expect(within(measuredTable()).getAllByRole("row").length).toBeGreaterThan(1));

    expect(document.body.textContent).not.toContain("240000");
  });
});
