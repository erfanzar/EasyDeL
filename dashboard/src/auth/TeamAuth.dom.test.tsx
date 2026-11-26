// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseRoute } from "../routing";
import { AuthGate } from "./AuthGate";
import { LOGIN_PATH, TEAM_LOGOUT_PATH, __resetRuntimeOidcConfigForTests, logout, setTeamState } from "./oidc";

type Reply = [number, unknown];
type Handler = (init?: RequestInit) => Reply;

/** A team-mode plane: each `METHOD /path` answers from the table; anything else is a 404. */
function teamPlane(routes: Record<string, Handler>) {
  const calls: Array<{ key: string; body: unknown }> = [];
  const fetchStub = vi.fn(async (input: unknown, init?: RequestInit) => {
    const path = String(input).split("?")[0];
    const key = `${(init?.method ?? "GET").toUpperCase()} ${path}`;
    calls.push({ key, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    const handler = routes[key];
    const [status, body] = handler ? handler(init) : [404, { detail: "Not Found" }];
    return new Response(JSON.stringify(body), { status });
  });
  vi.stubGlobal("fetch", fetchStub);
  return calls;
}

const NO_IDP = { issuer: null, client_id: null, development_auth: "disabled" };
const WITH_IDP = { issuer: "https://idp.example.com", client_id: "eyvan", development_auth: "disabled" };
const ALICE = { subject: "alice", email: "alice@example.com", display_name: "Alice" };

function baseRoutes(config: Record<string, unknown>): Record<string, Handler> {
  return {
    "GET /api/v1/dashboard/config": () => [200, config],
    "GET /api/v1/team/status": () => [200, { team_mode: true }],
    "GET /api/v1/team/me": () => [401, { detail: "authentication required" }],
    "GET /api/v1/session": () => [401, { detail: "authentication required" }],
  };
}

function gate() {
  return render(
    <AuthGate>
      <p>fleet</p>
    </AuthGate>,
  );
}

afterEach(() => {
  cleanup();
  __resetRuntimeOidcConfigForTests();
  setTeamState(false, false);
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("team sign-in", () => {
  it("shows the password form, and no OIDC button when the plane has no identity provider", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    teamPlane(baseRoutes(NO_IDP));

    gate();

    await waitFor(() => expect(screen.getByRole("form", { name: "Sign in" })).toBeTruthy());
    expect(screen.queryByRole("button", { name: /Sign in with OIDC/ })).toBeNull();
    expect(screen.queryByText("fleet")).toBeNull();
  });

  it("offers both the password form and OIDC when the plane names an identity provider", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    teamPlane(baseRoutes(WITH_IDP));
    const assign = vi.fn();
    vi.stubGlobal("location", { ...globalThis.location, assign });

    gate();

    await waitFor(() => expect(screen.getByRole("form", { name: "Sign in" })).toBeTruthy());
    expect(screen.getByLabelText("User name or email")).toBeTruthy();
    expect(screen.getByLabelText("Password")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Sign in with OIDC/ }));
    expect(assign).toHaveBeenCalledWith(LOGIN_PATH);
  });

  it("signs in with a password and opens the dashboard", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const calls = teamPlane({
      ...baseRoutes(NO_IDP),
      "POST /api/v1/team/login": () => [200, { user: ALICE, expires_at: null }],
    });

    gate();
    await waitFor(() => expect(screen.getByRole("form", { name: "Sign in" })).toBeTruthy());
    fireEvent.change(screen.getByLabelText("User name or email"), { target: { value: " alice@example.com " } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "correct horse battery" } });
    fireEvent.submit(screen.getByRole("form", { name: "Sign in" }));

    await waitFor(() => expect(screen.getByText("fleet")).toBeTruthy());
    const login = calls.find((call) => call.key === "POST /api/v1/team/login");
    // The browser never asks for a token: the session is the HttpOnly cookie.
    expect(login?.body).toEqual({ email: "alice@example.com", password: "correct horse battery" });
  });

  it("shows the server's generic refusal for a wrong password", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    teamPlane({
      ...baseRoutes(NO_IDP),
      "POST /api/v1/team/login": () => [401, { detail: "wrong user name or password" }],
    });

    gate();
    await waitFor(() => expect(screen.getByRole("form", { name: "Sign in" })).toBeTruthy());
    fireEvent.change(screen.getByLabelText("User name or email"), { target: { value: "alice" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "not the password" } });
    fireEvent.submit(screen.getByRole("form", { name: "Sign in" }));

    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("wrong user name or password"));
    expect(screen.queryByText("fleet")).toBeNull();
  });

  it("a plane without team mode never shows the password form", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    teamPlane({ ...baseRoutes(WITH_IDP), "GET /api/v1/team/status": () => [200, { team_mode: false }] });

    gate();

    await waitFor(() => expect(screen.getByRole("button", { name: /Sign in with OIDC/ })).toBeTruthy());
    expect(screen.queryByRole("form", { name: "Sign in" })).toBeNull();
  });
});

describe("accepting an invite", () => {
  function openedFromInvite(token: string) {
    const replaceState = vi.fn();
    const assign = vi.fn();
    vi.stubGlobal("history", { ...globalThis.history, replaceState });
    vi.stubGlobal("location", { hash: `#/accept-invite/${token}`, pathname: "/", search: "", assign });
    return { replaceState, assign };
  }

  it("strips the token from the address bar, names the invitee and signs in", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const { replaceState } = openedFromInvite("inv-secret-123");
    const calls = teamPlane({
      ...baseRoutes(NO_IDP),
      "POST /api/v1/team/invites/preview": () => [
        200,
        { email: "bob@example.com", subject: "bob", expires_at: "2026-11-03T00:00:00+00:00" },
      ],
      "POST /api/v1/team/invites/accept": () => [200, { user: { ...ALICE, subject: "bob" }, expires_at: null }],
    });

    gate();

    await waitFor(() => expect(screen.getByRole("heading", { name: "Join Eyvan" })).toBeTruthy());
    expect(replaceState).toHaveBeenCalledWith(null, "", "/");
    expect(screen.getByText(/bob@example\.com/)).toBeTruthy();
    // The token went in a POST body, never a URL a server logs.
    expect(calls.find((call) => call.key === "POST /api/v1/team/invites/preview")?.body).toEqual({
      token: "inv-secret-123",
    });
    expect(calls.some((call) => call.key.includes("inv-secret-123"))).toBe(false);

    const create = screen.getByRole("button", { name: "Create account" });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "short" } });
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "short" } });
    expect((create as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "bob password 4567" } });
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "bob password 456" } });
    expect(screen.getByText("The passwords do not match.")).toBeTruthy();
    expect((create as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "bob password 4567" } });
    fireEvent.submit(screen.getByRole("form", { name: "Accept invite" }));

    await waitFor(() => expect(screen.getByText("fleet")).toBeTruthy());
    expect(calls.find((call) => call.key === "POST /api/v1/team/invites/accept")?.body).toEqual({
      token: "inv-secret-123",
      password: "bob password 4567",
      display_name: "",
    });
  });

  it("explains a spent or expired invite instead of showing a form", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    openedFromInvite("inv-used");
    teamPlane({
      ...baseRoutes(NO_IDP),
      "POST /api/v1/team/invites/preview": () => [404, { detail: "this invite was already used" }],
    });

    gate();

    await waitFor(() => expect(screen.getByRole("heading", { name: "This invite cannot be used" })).toBeTruthy());
    expect(screen.getByText(/already used/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Go to sign-in" }));
    await waitFor(() => expect(screen.getByRole("form", { name: "Sign in" })).toBeTruthy());
  });

  it("does not reopen a spent invite after signing out", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    openedFromInvite("inv-once");
    const calls = teamPlane({
      ...baseRoutes(NO_IDP),
      "POST /api/v1/team/invites/preview": () => [
        200,
        { email: "bob@example.com", subject: "bob", expires_at: "2026-11-03T00:00:00+00:00" },
      ],
      "POST /api/v1/team/invites/accept": () => [200, { user: { ...ALICE, subject: "bob" }, expires_at: null }],
      [`POST ${TEAM_LOGOUT_PATH}`]: () => [200, { signed_out: true }],
    });

    gate();
    await waitFor(() => expect(screen.getByRole("heading", { name: "Join Eyvan" })).toBeTruthy());
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "bob password 4567" } });
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "bob password 4567" } });
    fireEvent.submit(screen.getByRole("form", { name: "Accept invite" }));
    await waitFor(() => expect(screen.getByText("fleet")).toBeTruthy());

    await logout();

    await waitFor(() => expect(screen.getByRole("form", { name: "Sign in" })).toBeTruthy());
    expect(screen.queryByRole("heading", { name: "Join Eyvan" })).toBeNull();
    expect(calls.filter((call) => call.key === "POST /api/v1/team/invites/preview")).toHaveLength(1);
    expect(calls.some((call) => call.key === `POST ${TEAM_LOGOUT_PATH}`)).toBe(true);
  });
});

describe("the Settings route", () => {
  it("serves team accounts under Settings and keeps old #/users links working", () => {
    expect(parseRoute("#/settings")).toEqual({ section: "settings" });
    expect(parseRoute("#/users")).toEqual({ section: "settings" });
  });
});
