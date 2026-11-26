// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import { formatChipHours, formatChipHoursAgainst, knownPercent, summarizeShapes } from "./format";

describe("summarizeShapes", () => {
  it("never hides a generation behind a count", () => {
    // The regression: shapes sort by generation, so "first three and a count"
    // rendered a fleet's whole v6e half as part of "+14 more".
    const fleet = [
      "v5p-16",
      "v5p-32",
      "v5p-64",
      "v5p-128",
      "v5p-256",
      "v5p-384",
      "v5p-512",
      "v5p-640",
      "v5p-768",
      "v5p-896",
      "v5p-1024",
      "v6e-8",
      "v6e-16",
      "v6e-32",
      "v6e-64",
      "v6e-128",
      "v6e-256",
    ];
    const summary = summarizeShapes(fleet);
    expect(summary).toBe("v5p 16-1024, v6e 8-256");
    expect(summary).toContain("v6e");
  });

  it("keeps a lone shape exact rather than turning it into a range", () => {
    expect(summarizeShapes(["v5p-16"])).toBe("v5p-16");
  });

  it("passes through a name that is not generation-size", () => {
    expect(summarizeShapes(["cpu"])).toBe("cpu");
  });

  it("renders nothing for no shapes", () => {
    expect(summarizeShapes([])).toBe("");
  });
});

describe("formatChipHours", () => {
  it("renders a non-finite value as unknown, never as zero", () => {
    expect(formatChipHours(Number.NaN)).toBe("unknown chip-hours");
    expect(formatChipHours(Number.POSITIVE_INFINITY, false)).toBe("unknown");
  });

  it("never rounds a figure just under a whole number up to it", () => {
    expect(formatChipHours(0.996, false)).toBe("0.996");
    expect(formatChipHours(0.9999, false)).not.toBe("1");
    expect(formatChipHours(0.25)).toBe("0.25 chip-hours");
    expect(formatChipHours(1)).toBe("1 chip-hour");
  });
});

describe("knownPercent", () => {
  it("is null, not 0 or NaN, when either side is not finite", () => {
    expect(knownPercent(Number.NaN, 1)).toBeNull();
    expect(knownPercent(1, Number.NaN)).toBeNull();
    expect(knownPercent(0.5, 1)).toBe(50);
    expect(knownPercent(1, 0)).toBe(0);
  });

  it("does not round a figure just under the limit up to 100", () => {
    expect(knownPercent(0.996, 1)).toBe(99);
    expect(knownPercent(1, 1)).toBe(100);
  });
});

describe("formatChipHoursAgainst", () => {
  it("shows enough precision that spend under a limit never reads as equal to it", () => {
    expect(formatChipHoursAgainst(0.996, 1)).toEqual(["0.996", "1"]);
    expect(formatChipHoursAgainst(1.496, 1.5)).toEqual(["1.496", "1.5"]);
    expect(formatChipHoursAgainst(0.5, 1)).toEqual(["0.5", "1"]);
    expect(formatChipHoursAgainst(Number.NaN, 1)).toEqual(["unknown", "1"]);
  });
});
