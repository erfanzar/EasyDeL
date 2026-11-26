// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AuthGate } from "./AuthGate";
import { LOGIN_PATH, __resetRuntimeOidcConfigForTests } from "./oidc";

function planeAnswering(document: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(document), { status: 200 })),
  );
}

/** The config endpoint and the session probe answer differently. */
function planeWithSession(
  config: Record<string, unknown>,
  sessionStatus: number,
  sessionBody: Record<string, unknown>,
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: unknown) => {
      const url = String(input);
      if (url.endsWith("/api/v1/session")) {
        return new Response(JSON.stringify(sessionBody), { status: sessionStatus });
      }
      return new Response(JSON.stringify(config), { status: 200 });
    }),
  );
}

describe("the sign-in screen", () => {
  afterEach(() => {
    cleanup();
    __resetRuntimeOidcConfigForTests();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    sessionStorage.removeItem("eyvan.local.credential");
  });

  it("renders the fleet once its owner's credential reached the browser", async () => {
    // The bundle is served by the plane, but being its page is not being its
    // owner: the plane demands the owner-only credential from every caller,
    // and the browser cannot read the file it lives in. `eyvan dashboard`
    // bridges it through the URL fragment; only once that value has arrived
    // does the page have anything to say to its own plane.
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    sessionStorage.setItem("eyvan.local.credential", "owner-secret");
    planeAnswering({ issuer: null, client_id: null, development_auth: "loopback_default" });

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );

    await waitFor(() => expect(screen.getByText("fleet")).toBeTruthy());
  });

  it("names the command that bridges the credential when the page was opened bare", async () => {
    // Typing the plane's URL by hand is ordinary, not a misconfiguration, and
    // the answer is one command in the owner's terminal. "Unable to load Eyvan
    // state" on every panel is the alternative this screen exists to prevent.
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    planeAnswering({ issuer: null, client_id: null, development_auth: "loopback_default" });

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );

    await waitFor(() => expect(screen.getByText("This plane requires its local credential")).toBeTruthy());
    expect(screen.getByText(/eyvan dashboard/)).toBeTruthy();
    expect(screen.queryByText("fleet")).toBeNull();
  });

  it("still calls a plane with no OIDC and no development auth misconfigured", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    planeAnswering({ issuer: null, client_id: null, development_auth: "disabled" });

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );

    await waitFor(() => expect(screen.getByText("Authentication is not configured")).toBeTruthy());
    expect(screen.getByRole("alert")).toBeTruthy();
  });

  it("offers sign-in when the plane does have an identity provider", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    planeWithSession(
      { issuer: "https://idp.example.com", client_id: "eyvan", development_auth: "loopback_default" },
      401,
      { detail: "authentication required" },
    );

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );

    await waitFor(() => expect(screen.getByText("Sign in to Eyvan")).toBeTruthy());
  });

  it("renders the fleet when the session cookie already authenticates the browser", async () => {
    // The cookie is unreadable from JavaScript by design, so the gate's only
    // witness to a live session is the authority endpoint itself.
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    planeWithSession({ issuer: "https://idp.example.com", client_id: "eyvan", development_auth: "disabled" }, 200, {
      subject: "alice",
      display_name: "Alice Example",
    });

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );

    await waitFor(() => expect(screen.getByText("fleet")).toBeTruthy());
    expect(screen.queryByText("Sign in to Eyvan")).toBeNull();
  });

  it("sends the sign-in button to the server login endpoint", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    planeWithSession({ issuer: "https://idp.example.com", client_id: "eyvan", development_auth: "disabled" }, 401, {
      detail: "authentication required",
    });
    const assign = vi.fn();
    vi.stubGlobal("location", { ...globalThis.location, assign });

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );

    await waitFor(() => expect(screen.getByText("Sign in to Eyvan")).toBeTruthy());
    screen.getByRole("button", { name: /Sign in with OIDC/ }).click();

    expect(assign).toHaveBeenCalledWith(LOGIN_PATH);
  });

  it("shows the provider's refusal once, without leaving it in the address bar", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    planeWithSession({ issuer: "https://idp.example.com", client_id: "eyvan", development_auth: "disabled" }, 401, {
      detail: "authentication required",
    });
    const replaceState = vi.fn();
    vi.stubGlobal("history", { ...globalThis.history, replaceState });
    vi.stubGlobal("location", { search: "?auth_error=access_denied", pathname: "/", hash: "" });

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );

    await waitFor(() => expect(screen.getByText("Sign-in failed")).toBeTruthy());
    expect(screen.getByText(/access_denied/)).toBeTruthy();
    expect(replaceState).toHaveBeenCalledWith(null, "", "/");
  });
});

describe("the gate after the plane refuses a held credential", () => {
  afterEach(() => {
    cleanup();
    __resetRuntimeOidcConfigForTests();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    sessionStorage.removeItem("eyvan.local.credential");
  });

  it("closes on a fleet that was already rendered, naming the way back in", async () => {
    // The credential was valid when the page loaded, and the plane has since
    // refused it -- rotated, reset, or a session carried between machines.
    // Presence is all the gate checked at render time, so the refusal itself
    // must retire the value and close the gate; a page of dead panels that
    // never names the working command is the failure this prevents.
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    sessionStorage.setItem("eyvan.local.credential", "stale-value");
    planeAnswering({ issuer: null, client_id: null, development_auth: "loopback_default" });

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );
    await waitFor(() => expect(screen.getByText("fleet")).toBeTruthy());

    // What the API client does when the plane refuses the value it held.
    sessionStorage.removeItem("eyvan.local.credential");
    globalThis.dispatchEvent(new Event("eyvan:auth-changed"));

    await waitFor(() => expect(screen.getByText("This plane requires its local credential")).toBeTruthy());
    expect(screen.queryByText("fleet")).toBeNull();
  });
});
