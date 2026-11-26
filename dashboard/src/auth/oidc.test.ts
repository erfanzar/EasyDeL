// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  LOGIN_PATH,
  LOGOUT_PATH,
  __resetRuntimeOidcConfigForTests,
  absorbLocalCredentialFromLocation,
  apiAuthHeaders,
  beginLogin,
  capabilitiesForPermissions,
  consumeSessionNotice,
  currentIdentity,
  invalidateRefusedLocalCredential,
  isOidcConfigured,
  configuredScope,
  loadRuntimeOidcConfig,
  localAttachmentsAvailable,
  logout,
  planeDevelopmentAuthMode,
  planeRequiresLocalCredential,
  serverIdentity,
} from "./oidc";

function memoryStorage(initial: Record<string, string> = {}): Storage {
  const values = new Map(Object.entries(initial));
  return {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    removeItem: (key) => {
      values.delete(key);
    },
    setItem: (key, value) => {
      values.set(key, value);
    },
  };
}

function unsignedToken(claims: Record<string, unknown>): string {
  const encode = (value: unknown) =>
    btoa(JSON.stringify(value)).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
  return `${encode({ alg: "none" })}.${encode(claims)}.`;
}

describe("OIDC sign-in is server-side", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("hands the browser to the server login endpoint instead of building an authorize URL", () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "https://identity.example.test");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "eyvan-dashboard");
    const assign = vi.fn();
    vi.stubGlobal("location", { assign });
    const fetchImpl = vi.fn();
    vi.stubGlobal("fetch", fetchImpl);

    beginLogin();

    // No discovery round trip, no PKCE material in this tab: the challenge is
    // minted where the tokens will live, on the server.
    expect(assign).toHaveBeenCalledWith(LOGIN_PATH);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("refuses to begin sign-in when OIDC is not configured", () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");

    expect(() => beginLogin()).toThrow("OIDC client configuration is incomplete");
  });

  it("never builds an Authorization header from something JavaScript can read", () => {
    // The regression this pins: tokens lived in sessionStorage and were copied
    // into a bearer header on every request, readable by any cross-site script.
    vi.stubGlobal(
      "sessionStorage",
      memoryStorage({
        "eyvan.oidc.access_token": "stolen-token",
        "eyvan.oidc.id_token": unsignedToken({ sub: "alice" }),
        "eyvan.oidc.expires_at": String(Date.now() + 60_000),
      }),
    );

    expect(apiAuthHeaders()).toEqual({});
  });

  it("asks the server whether a session cookie already authenticates this browser", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "https://identity.example.test");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "eyvan-dashboard");
    const fetchImpl = vi.fn(
      async () => new Response(JSON.stringify({ subject: "alice", display_name: "Alice Example" }), { status: 200 }),
    );

    const identity = await serverIdentity(fetchImpl as unknown as typeof fetch);

    expect(fetchImpl).toHaveBeenCalledWith("/api/v1/session", expect.objectContaining({ headers: {} }));
    expect(identity).toMatchObject({ mode: "oidc", displayName: "Alice Example" });
  });

  it("treats a 401 session probe as signed out, not as an error", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "https://identity.example.test");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "eyvan-dashboard");
    const fetchImpl = vi.fn(
      async () => new Response(JSON.stringify({ detail: "authentication required" }), { status: 401 }),
    );

    await expect(serverIdentity(fetchImpl as unknown as typeof fetch)).resolves.toBeUndefined();
  });

  it("skips the session probe entirely when OIDC is not configured", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const fetchImpl = vi.fn();

    await expect(serverIdentity(fetchImpl as unknown as typeof fetch)).resolves.toBeUndefined();
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("signs out through the server and lands back on the sign-in screen", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "https://identity.example.test");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "eyvan-dashboard");
    const assign = vi.fn();
    vi.stubGlobal("location", { assign });
    const fetchImpl = vi.fn(async () => new Response(null, { status: 204 }));

    await logout(fetchImpl as unknown as typeof fetch);

    expect(fetchImpl).toHaveBeenCalledWith(LOGOUT_PATH, { method: "POST" });
    expect(assign).toHaveBeenCalledWith("/");
    expect(consumeSessionNotice()).toMatch(/signed out/i);
  });

  it.each(["network", "http"])("reports an unconfirmed logout instead of claiming success (%s)", async (failure) => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "https://identity.example.test");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "eyvan-dashboard");
    const assign = vi.fn();
    vi.stubGlobal("location", { assign });
    const fetchImpl = vi.fn(async () => {
      if (failure === "network") throw new Error("network down");
      return new Response(null, { status: 503 });
    });

    await logout(fetchImpl as unknown as typeof fetch);

    expect(assign).not.toHaveBeenCalled();
    expect(consumeSessionNotice()).toMatch(/could not confirm sign-out.*may still be active/i);
  });
});

describe("dashboard permission capabilities", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it.each([
    [
      ["read", "submit", "operate", "manage_project", "manage_own_capacity", "manage_fleet"],
      { submitWorkloads: true, operateWorkloads: true, manageProject: true, manageFleet: true },
    ],
    [
      ["read", "submit", "operate", "manage_project", "manage_own_capacity"],
      { submitWorkloads: true, operateWorkloads: true, manageProject: true, manageFleet: false },
    ],
    [
      ["read", "submit", "operate", "manage_project"],
      { submitWorkloads: true, operateWorkloads: true, manageProject: true, manageFleet: false },
    ],
    [
      ["read", "submit", "operate"],
      { submitWorkloads: true, operateWorkloads: true, manageProject: false, manageFleet: false },
    ],
    [["submit"], { submitWorkloads: true, operateWorkloads: false, manageProject: false, manageFleet: false }],
    [["read"], { submitWorkloads: false, operateWorkloads: false, manageProject: false, manageFleet: false }],
  ] as const)("maps server permissions to presentation controls", (permissions, expected) => {
    expect(capabilitiesForPermissions(permissions)).toEqual(expected);
  });

  it("reports no local OIDC identity: the tab holds no token and no claims", () => {
    // Claims used to be decoded here for display. The session is a cookie the
    // page cannot read, so identity display comes from the server's authority
    // response -- never from something in this tab's storage.
    vi.stubEnv("VITE_EYVAN_ORG", "acme");
    vi.stubEnv("VITE_EYVAN_PROJECT", "research");
    vi.stubGlobal(
      "sessionStorage",
      memoryStorage({
        "eyvan.oidc.access_token": unsignedToken({
          sub: "alice",
          eyvan_grants: [{ role: "fleet_admin" }],
        }),
      }),
    );

    expect(currentIdentity()).toBeUndefined();
  });

  it("does not elevate a development role when a session exists only in legacy storage", () => {
    vi.stubEnv("VITE_EYVAN_DEV_AUTH", "true");
    vi.stubEnv("VITE_EYVAN_DEV_ROLE", "fleet_admin");
    vi.stubGlobal(
      "sessionStorage",
      memoryStorage({
        "eyvan.oidc.access_token": unsignedToken({ sub: "alice" }),
      }),
    );

    expect(currentIdentity()).toMatchObject({
      mode: "development",
      roles: [],
      capabilities: {
        submitWorkloads: false,
        operateWorkloads: false,
        manageProject: false,
        manageFleet: false,
      },
    });
  });
});

describe("runtime OIDC configuration", () => {
  afterEach(() => {
    __resetRuntimeOidcConfigForTests();
    vi.unstubAllEnvs();
  });

  it("asks the control plane when the bundle carries no issuer", async () => {
    // The gap this closes: the issuer and client id were compiled in by VITE_
    // build arguments, so one image served exactly one identity provider.
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const fetchImpl = vi.fn(
      async () =>
        new Response(JSON.stringify({ issuer: "https://idp.example.com", client_id: "eyvan-dashboard" }), {
          status: 200,
        }),
    );

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(fetchImpl).toHaveBeenCalledWith("/api/v1/dashboard/config");
    expect(isOidcConfigured()).toBe(true);
  });

  it("reads runtime capabilities even when identity configuration is baked in", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "https://baked.example.com");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "baked-client");
    vi.stubEnv("VITE_EYVAN_OIDC_AUDIENCE", "baked-audience");
    const fetchImpl = vi.fn(async () => Response.json({ local_attachments_available: true }));

    await loadRuntimeOidcConfig(fetchImpl);

    expect(fetchImpl).toHaveBeenCalledWith("/api/v1/dashboard/config");
    expect(localAttachmentsAvailable()).toBe(true);
    expect(isOidcConfigured()).toBe(true);
  });

  it.each([
    {},
    null,
    { local_attachments_available: false },
    { local_attachments_available: "true" },
    { local_attachments_available: 1 },
  ])("requires an explicit boolean capability: %j", async (config) => {
    expect(localAttachmentsAvailable()).toBe(false);
    await loadRuntimeOidcConfig(async () => Response.json(config));
    expect(localAttachmentsAvailable()).toBe(false);
  });

  it("treats an unreachable control plane as unconfigured rather than as an auth error", async () => {
    // An unreachable control plane is not an auth misconfiguration. The failure
    // belongs at the first API call, not as "OIDC is not configured".
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const fetchImpl = vi.fn(async () => {
      throw new Error("network down");
    });

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(isOidcConfigured()).toBe(false);
  });

  it("ignores a control plane that answers without an issuer", async () => {
    // A plane running with no OIDC answers with nulls rather than a 404, so the
    // dashboard can tell "not configured" from "too old to answer".
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const fetchImpl = vi.fn(
      async () => new Response(JSON.stringify({ issuer: null, client_id: null }), { status: 200 }),
    );

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(isOidcConfigured()).toBe(false);
  });
});

describe("what the plane says about its own auth", () => {
  afterEach(() => {
    __resetRuntimeOidcConfigForTests();
    vi.unstubAllEnvs();
  });

  it("reports a loopback plane as needing a credential the browser cannot read", async () => {
    // The screen this drives: a laptop plane has no OIDC by design, and used to
    // be greeted as misconfigured -- with two build variables named at somebody
    // who never built the bundle.
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const fetchImpl = vi.fn(
      async () =>
        new Response(JSON.stringify({ issuer: null, client_id: null, development_auth: "loopback_default" }), {
          status: 200,
        }),
    );

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(planeDevelopmentAuthMode()).toBe("loopback_default");
    expect(planeRequiresLocalCredential()).toBe(true);
    expect(isOidcConfigured()).toBe(false);
  });

  it("does not call a plane with development auth disabled a local one", async () => {
    // That plane really is misconfigured, and the local screen would send its
    // operator to run a development command against production.
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const fetchImpl = vi.fn(
      async () =>
        new Response(JSON.stringify({ issuer: null, client_id: null, development_auth: "disabled" }), { status: 200 }),
    );

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(planeRequiresLocalCredential()).toBe(false);
  });

  it("treats a plane too old to answer as not local", async () => {
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");
    const fetchImpl = vi.fn(
      async () => new Response(JSON.stringify({ issuer: null, client_id: null }), { status: 200 }),
    );

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(planeDevelopmentAuthMode()).toBeUndefined();
    expect(planeRequiresLocalCredential()).toBe(false);
  });
});

describe("which project the dashboard asks about", () => {
  afterEach(() => {
    __resetRuntimeOidcConfigForTests();
    vi.unstubAllEnvs();
  });

  it("uses the scope the control plane reports rather than a compiled-in guess", async () => {
    // The defect this closes: the literal was "acme"/"research" while the CLI
    // defaults to "default"/"default". Both are free-form scopes a plane
    // answers for, so nothing errored -- a job submitted with `eyvan run`
    // simply never appeared on the page.
    vi.stubEnv("VITE_EYVAN_ORG", "");
    vi.stubEnv("VITE_EYVAN_PROJECT", "");
    const fetchImpl = vi.fn(
      async () => new Response(JSON.stringify({ default_org: "default", default_project: "default" }), { status: 200 }),
    );

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(configuredScope()).toEqual({ orgId: "default", projectId: "default" });
  });

  it("lets a build pin its own scope over the plane's default", async () => {
    // A hosted deployment builds one bundle per environment; the plane's
    // default must not silently move it to another project.
    vi.stubEnv("VITE_EYVAN_ORG", "acme");
    vi.stubEnv("VITE_EYVAN_PROJECT", "research");
    const fetchImpl = vi.fn(
      async () => new Response(JSON.stringify({ default_org: "default", default_project: "default" }), { status: 200 }),
    );

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(configuredScope()).toEqual({ orgId: "acme", projectId: "research" });
  });

  it("falls back to the CLI's default against a plane too old to report one", async () => {
    vi.stubEnv("VITE_EYVAN_ORG", "");
    vi.stubEnv("VITE_EYVAN_PROJECT", "");
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ issuer: null }), { status: 200 }));

    await loadRuntimeOidcConfig(fetchImpl as unknown as typeof fetch);

    expect(configuredScope()).toEqual({ orgId: "default", projectId: "default" });
  });
});

describe("the loopback development credential bridge", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("absorbs the credential from the URL fragment and strips it from the address bar", () => {
    const store = memoryStorage();
    vi.stubGlobal("sessionStorage", store);
    const replaceState = vi.fn();
    const location = { hash: "#eyvan-local-credential=owner-secret", pathname: "/", search: "" };

    const absorbed = absorbLocalCredentialFromLocation(location, { replaceState });

    expect(absorbed).toBe("owner-secret");
    expect(store.getItem("eyvan.local.credential")).toBe("owner-secret");
    // The value must not linger in history or in URLs that get copied around.
    expect(replaceState).toHaveBeenCalledWith(null, "", "/");
  });

  it("keeps unrelated fragment parameters while stripping the credential", () => {
    const store = memoryStorage();
    vi.stubGlobal("sessionStorage", store);
    const replaceState = vi.fn();
    const location = { hash: "#panel=cells&eyvan-local-credential=s&tab=gang", pathname: "/", search: "" };

    absorbLocalCredentialFromLocation(location, { replaceState });

    expect(replaceState).toHaveBeenCalledWith(null, "", "/#panel=cells&tab=gang");
  });

  it("leaves a URL that carries no credential exactly as it was", () => {
    const store = memoryStorage();
    vi.stubGlobal("sessionStorage", store);
    const replaceState = vi.fn();

    expect(
      absorbLocalCredentialFromLocation({ hash: "", pathname: "/", search: "" }, { replaceState }),
    ).toBeUndefined();
    expect(
      absorbLocalCredentialFromLocation({ hash: "#panel=cells", pathname: "/", search: "" }, { replaceState }),
    ).toBeUndefined();
    expect(replaceState).not.toHaveBeenCalled();
    expect(store.length).toBe(0);
  });

  it("presents the bridged credential without claiming a build-time identity", () => {
    // A bundle served by its plane sends no identity headers: the plane fills
    // its own page's identity. The one thing only the owner can supply is the
    // credential, and that is the one header this page insists on.
    vi.stubGlobal("sessionStorage", memoryStorage({ "eyvan.local.credential": "owner-secret" }));

    expect(apiAuthHeaders()).toEqual({ "X-Eyvan-Local-Credential": "owner-secret" });
  });

  it("keeps sending the credential alongside a bundle that claims a development identity", () => {
    vi.stubEnv("VITE_EYVAN_DEV_AUTH", "true");
    vi.stubGlobal("sessionStorage", memoryStorage({ "eyvan.local.credential": "owner-secret" }));

    expect(apiAuthHeaders()).toMatchObject({
      "X-Eyvan-User": "dashboard-dev",
      "X-Eyvan-Local-Credential": "owner-secret",
    });
  });

  it("adds no bearer header when a legacy token shares the tab with the credential", () => {
    // The cookie session replaces what the legacy key used to mean; the local
    // credential header still travels because the loopback plane demands it.
    vi.stubGlobal(
      "sessionStorage",
      memoryStorage({
        "eyvan.local.credential": "owner-secret",
        "eyvan.oidc.access_token": unsignedToken({ sub: "colleague" }),
        "eyvan.oidc.expires_at": String(Date.now() + 60_000),
      }),
    );

    expect(apiAuthHeaders()).toEqual({ "X-Eyvan-Local-Credential": "owner-secret" });
  });
});

describe("retiring a refused loopback credential", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("drops a stored credential the plane refused, and announces the change", () => {
    const store = memoryStorage({ "eyvan.local.credential": "stale-value" });
    vi.stubGlobal("sessionStorage", store);

    const retired = invalidateRefusedLocalCredential(": local Eyvan development credential is required");

    expect(retired).toBe(true);
    // The value is gone from storage; the gate re-render it triggers is
    // covered where a DOM exists, by the AuthGate tests.
    expect(store.getItem("eyvan.local.credential")).toBeNull();
  });

  it("keeps the credential for any other refusal", () => {
    const store = memoryStorage({ "eyvan.local.credential": "stale-value" });
    vi.stubGlobal("sessionStorage", store);

    expect(invalidateRefusedLocalCredential(": invalid bearer token")).toBe(false);
    expect(store.getItem("eyvan.local.credential")).toBe("stale-value");
  });

  it("has nothing to retire when no credential was held", () => {
    vi.stubGlobal("sessionStorage", memoryStorage());

    expect(invalidateRefusedLocalCredential(": local Eyvan development credential is required")).toBe(false);
  });
});
