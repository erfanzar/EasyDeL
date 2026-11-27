// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { formatReclaimCountdown, ReclaimBadge, ReclaimBanner } from "./ReclaimBanner";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("ReclaimBanner and ReclaimBadge", () => {
  it("formats countdown durations cleanly across hours, minutes, and seconds", () => {
    expect(formatReclaimCountdown(0)).toBe("0s");
    expect(formatReclaimCountdown(-5_000)).toBe("0s");
    expect(formatReclaimCountdown(42_000)).toBe("42s");
    expect(formatReclaimCountdown(522_000)).toBe("8m 42s");
    expect(formatReclaimCountdown(545_000)).toBe("9m 05s");
    expect(formatReclaimCountdown(3_665_000)).toBe("1h 01m 05s");
  });

  it("ticks live every second for a checkpointing victim and shows requester and auto-resume policy", () => {
    vi.useFakeTimers();
    const base = new Date("2026-11-27T14:00:00Z").getTime();
    vi.setSystemTime(base);
    const deadlineAt = new Date(base + (8 * 60 + 42) * 1000).toISOString();

    render(<ReclaimBanner deadlineAt={deadlineAt} requestedBy="bob" policy="requeue" graceSeconds={600} />);

    const banner = screen.getByTestId("reclaim-banner");
    expect(banner.textContent).toContain("Reclaiming · 8m 42s remaining");
    expect(banner.textContent).toContain("bob");
    expect(banner.textContent).toContain("Auto-resume elsewhere");
    const bar = screen.getByRole("progressbar", { name: "Reclaim checkpoint window remaining" });
    expect(Number(bar.getAttribute("aria-valuenow"))).toBe(87);

    act(() => {
      vi.advanceTimersByTime(1_000);
    });
    expect(banner.textContent).toContain("Reclaiming · 8m 41s remaining");

    act(() => {
      vi.advanceTimersByTime(2_000);
    });
    expect(banner.textContent).toContain("Reclaiming · 8m 39s remaining");
  });

  it("renders stop-after-checkpoint policy, switches to urgent under 60s, and stops cleanly at expiry", () => {
    vi.useFakeTimers();
    const base = new Date("2026-11-27T14:00:00Z").getTime();
    vi.setSystemTime(base);
    const deadlineAt = new Date(base + 3_000).toISOString();

    render(<ReclaimBanner deadlineAt={deadlineAt} requestedBy="admin" policy="stop" graceSeconds={600} />);

    const banner = screen.getByTestId("reclaim-banner");
    expect(banner.className).toContain("reclaim-banner-urgent");
    expect(banner.textContent).toContain("Reclaiming · 3s remaining");
    expect(banner.textContent).toContain("Stop after checkpoint");

    act(() => {
      vi.advanceTimersByTime(3_000);
    });
    expect(banner.textContent).toContain("Reclaim deadline reached · finalizing release");
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("0");
  });

  it("renders waiting-owner copy and reason for a queued owner waiting on borrowed capacity", () => {
    vi.useFakeTimers();
    const base = new Date("2026-11-27T14:00:00Z").getTime();
    vi.setSystemTime(base);
    const deadlineAt = new Date(base + 125_000).toISOString();

    render(
      <ReclaimBanner
        deadlineAt={deadlineAt}
        role="waiting"
        reason="Reclaiming borrowed capacity from alice"
        graceSeconds={600}
      />,
    );

    const banner = screen.getByTestId("reclaim-banner");
    expect(banner.className).toContain("reclaim-banner-waiting");
    expect(banner.textContent).toContain("Reclaiming borrowed capacity · 2m 05s remaining");
    expect(banner.textContent).toContain("Reclaiming borrowed capacity from alice");
  });

  it("renders a live ticking compact ReclaimBadge with optional inline policy", () => {
    vi.useFakeTimers();
    const base = new Date("2026-11-27T14:00:00Z").getTime();
    vi.setSystemTime(base);
    const deadlineAt = new Date(base + 95_000).toISOString();

    render(<ReclaimBadge deadlineAt={deadlineAt} requestedBy="bob" policy="requeue" showPolicy />);

    const badge = screen.getByTestId("reclaim-badge");
    expect(badge.textContent).toContain("Reclaiming · 1m 35s");
    expect(badge.getAttribute("title")).toContain("Requested by bob");
    expect(screen.getByText("Auto-resume elsewhere")).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(badge.textContent).toContain("Reclaiming · 1m 30s");
  });
});
