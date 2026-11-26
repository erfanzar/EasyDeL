// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import guide from "../README.md?raw";
import { CALLBACK_PATH } from "./auth/oidc";

const authentication = guide.split("## Browser authentication")[1]?.split("Local identity headers")[0] ?? "";

describe("dashboard deployment authentication guide", () => {
  it("registers the server callback rather than the dashboard root", () => {
    const publicUrl = "https://eyvan.example.com";
    // An exact-match identity provider accepts only the URI its operator
    // registered. This must be the server callback, not the SPA landing page.
    expect(authentication).toContain(`\`${publicUrl}${CALLBACK_PATH}\``);
    expect(authentication).toContain(`\`<EYVAN_PUBLIC_URL>${CALLBACK_PATH}\``);
    expect(authentication).toMatch(/identity provider must register/i);
  });

  it("configures sign-in on the control plane rather than through unused build variables", () => {
    for (const setting of [
      "EYVAN_PUBLIC_URL",
      "EYVAN_OIDC_ISSUER",
      "EYVAN_OIDC_DASHBOARD_CLIENT_ID",
      "EYVAN_OIDC_AUDIENCE",
    ]) {
      expect(authentication).toContain(setting);
    }
    expect(authentication).not.toMatch(/VITE_EYVAN_OIDC_(?:REDIRECT_URI|SCOPE)/);
  });

  it("describes server-side token custody and the HttpOnly session cookie", () => {
    expect(authentication).toMatch(/tokens stay server-side/i);
    expect(authentication).toContain("HttpOnly");
    expect(authentication).not.toMatch(/tokens are held in `sessionStorage`/);
    expect(authentication).not.toMatch(/permit browser access to discovery and token/);
  });
});
