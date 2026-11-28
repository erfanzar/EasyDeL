// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { Shell } from "./Shell";

afterEach(() => {
  cleanup();
  localStorage.clear();
});

// The stylesheet half of this check is Shell.topbar.css.test.ts: jsdom has no
// file: URL to read the CSS from.
describe("topbar structure", () => {
  it("puts the breadcrumb and the actions directly in the topbar header", () => {
    render(
      <Shell
        route={{ section: "overview" }}
        mobileNavigationOpen={false}
        onMobileNavigationChange={() => undefined}
        onNavigate={() => undefined}
        onSubmit={() => undefined}
        sectionAvailability={{}}
        scopes={[]}
        onScopeChange={() => undefined}
        onToggleAssistant={() => undefined}
        assistantOpen
        assistantDock={<aside aria-label="Assistant panel" />}
      >
        <p>page</p>
      </Shell>,
    );
    const crumb = screen.getByRole("navigation", { name: "Breadcrumb" });
    const header = crumb.parentElement as HTMLElement;
    expect(header.tagName).toBe("HEADER");
    expect(header.classList.contains("topbar")).toBe(true);
    const actions = header.querySelector(":scope > .topbar-actions");
    expect(actions).not.toBeNull();
    expect(actions?.contains(screen.getByRole("button", { name: "Assistant" }))).toBe(true);
    // The header is the main column's own child, so it spans that column even with the dock open.
    expect(header.parentElement?.classList.contains("main-column")).toBe(true);
  });
});
