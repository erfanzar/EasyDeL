// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

// Vitest serves CSS imports empty, so read the file itself -- without adding
// Node typings to the browser-only type configuration (as writer-013 does).
const fsModule = "node:fs";
const { readFileSync }: { readFileSync: (path: URL, encoding: "utf8") => string } = await import(
  /* @vite-ignore */ fsModule
);
const stylesheet = readFileSync(new URL("../styles.css", import.meta.url), "utf8");

/** Every rule whose selector list names exactly `selector`, across all media blocks. */
function rulesFor(selector: string): string[] {
  const bodies: string[] = [];
  const source = stylesheet.replace(/\/\*[\s\S]*?\*\//g, "");
  for (const match of source.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    const selectors = match[1].split(",").map((part) => part.trim());
    if (selectors.includes(selector)) bodies.push(match[2]);
  }
  return bodies;
}

describe("topbar stylesheet", () => {
  it("no longer centres the topbar or the top navigation inside a max-width", () => {
    const topbar = rulesFor(".topbar");
    expect(topbar.length).toBeGreaterThan(1);
    for (const body of topbar) {
      expect(body).not.toMatch(/max-width/);
      expect(body).not.toMatch(/margin:\s*0 auto/);
    }
    const sidebar = rulesFor(".sidebar");
    expect(sidebar.length).toBeGreaterThan(1);
    for (const body of sidebar) expect(body).not.toMatch(/1440px/);
    // The breadcrumb grows to fill the row, which pushes the actions to the right edge.
    expect(rulesFor(".page-title").some((body) => /flex:\s*1\b/.test(body))).toBe(true);
  });
});
