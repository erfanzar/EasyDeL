// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { SessionPermission } from "../types";

/**
 * The loopback development credential this browser was handed, if it was.
 *
 * The plane demands its owner-only credential from every caller once it has
 * issued one, this page included: the headers that identify the plane's own
 * dashboard are forgeable by any local process, so they cannot authenticate
 * anything. The browser cannot read the 0600 file the value lives in -- the
 * owner's terminal can, which is why `eyvan dashboard` opens this page with
 * the value in the URL fragment and {@link absorbLocalCredentialFromLocation}
 * moves it here before anything else runs.
 *
 * This is the one credential that still lives in this tab. The OIDC session
 * used to live here too, in the same storage any cross-site script can read;
 * it moved server-side behind an HttpOnly cookie (see the backend-for-frontend
 * endpoints in ``eyvan/server/api/dashboard_sessions.py``). The local
 * credential stays because its whole model is this-machine-only: it is handed
 * through a URL fragment precisely so no server ever sees it, and on loopback
 * there is no transport to protect it from.
 */
export const LOCAL_CREDENTIAL_KEY = "eyvan.local.credential";
export const LOCAL_CREDENTIAL_FRAGMENT_PARAM = "eyvan-local-credential";
/**
 * The refusal a loopback plane answers with when the credential it demands
 * was not presented correctly. Mirrors the server's
 * ``LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL`` in
 * ``eyvan/server/local/auth.py`` -- it is a wire contract, so it is stated
 * once per side and each side names the other.
 */
export const LOCAL_CREDENTIAL_REQUIRED_DETAIL = "local Eyvan development credential is required";
export const AUTH_CHANGED_EVENT = "eyvan:auth-changed";

/**
 * The server-side sign-in surface. The browser never sees an authorization
 * code, a PKCE verifier, or a token: ``beginLogin`` navigates to the login
 * endpoint, the identity provider redirects back to the server's callback,
 * and the server opens the session this page's cookies then carry.
 */
export const LOGIN_PATH = "/api/v1/dashboard/auth/login";
export const LOGOUT_PATH = "/api/v1/dashboard/auth/logout";
/**
 * The callback the identity provider redirects back to. No dashboard code
 * requests it — the server completes the code exchange there and redirects
 * to "/" — but the path is stated here, next to the flow it completes, so
 * the browser-routed surface is written down in the client that lives it.
 */
export const CALLBACK_PATH = "/api/v1/dashboard/auth/callback";

export interface AuthCapabilities {
  submitWorkloads: boolean;
  operateWorkloads: boolean;
  manageProject: boolean;
  manageFleet: boolean;
}

export interface AuthScope {
  orgId: string;
  projectId: string;
}

export interface AuthIdentity {
  displayName: string;
  detail: string;
  mode: "oidc" | "development" | "demo" | "team";
  /** Server-resolved roles applicable to the selected dashboard scope. */
  roles: string[];
  scope: AuthScope;
  /** Presentation derived only from the authenticated server authority response. */
  capabilities: AuthCapabilities;
}

export const NO_AUTH_CAPABILITIES: AuthCapabilities = Object.freeze({
  submitWorkloads: false,
  operateWorkloads: false,
  manageProject: false,
  manageFleet: false,
});

/** Derive presentation controls from permissions evaluated by the server Authorizer. */
export function capabilitiesForPermissions(
  permissionNames: Iterable<SessionPermission>,
  acceptsNewWork = true,
): AuthCapabilities {
  const permissions = new Set(permissionNames);
  return {
    // A grant can outlive tenant registration. Server-reported permissions say
    // what the principal may do; acceptsNewWork says whether this registered
    // scope currently accepts any tenant mutation. Reads remain independent so
    // disabled projects keep their durable history visible.
    submitWorkloads: acceptsNewWork && permissions.has("submit"),
    operateWorkloads: acceptsNewWork && permissions.has("operate"),
    manageProject: acceptsNewWork && permissions.has("manage_project"),
    manageFleet: permissions.has("manage_fleet"),
  };
}

/**
 * The org and project every scoped request is made in.
 *
 * The build-time values win, then what the control plane reports, and only then
 * a literal. That order matters: the literal used to be `acme`/`research` while
 * the CLI defaults to `default`/`default`, so a job submitted with `eyvan run`
 * never appeared on this page. Nothing errored, because both are free-form
 * scopes a plane answers for -- the dashboard was simply asking about a
 * different project than the one the operator was using.
 *
 * The remaining literal is reached only by a build with no configured scope
 * talking to a plane too old to report one, and it now matches the CLI.
 */
export function configuredScope(): AuthScope {
  return {
    orgId: import.meta.env.VITE_EYVAN_ORG || runtimeConfig?.default_org || "default",
    projectId: import.meta.env.VITE_EYVAN_PROJECT || runtimeConfig?.default_project || "default",
  };
}

export function isDemoMode(): boolean {
  return (
    import.meta.env.VITE_EYVAN_DEMO === "true" ||
    new URLSearchParams(globalThis.location?.search ?? "").get("demo") === "1"
  );
}

export function isDevelopmentAuthEnabled(): boolean {
  return import.meta.env.VITE_EYVAN_DEV_AUTH === "true";
}

/**
 * Whether the control plane itself accepts development identity headers.
 *
 * Distinct from {@link isDevelopmentAuthEnabled}, which asks whether *this
 * bundle* was built to send them. A plane can accept them while the bundle
 * serving its dashboard was never built to send any -- which is exactly the
 * bundle a laptop plane serves -- and the difference is what turns the sign-in
 * screen from a recital of build variables into the one command that works.
 */
export function planeDevelopmentAuthMode(): string | undefined {
  return runtimeConfig?.development_auth ?? undefined;
}

/** Whether the plane requires a credential the browser is unable to read. */
export function planeRequiresLocalCredential(): boolean {
  return planeDevelopmentAuthMode() === "loopback_default";
}

function storage(): Storage | undefined {
  try {
    return globalThis.sessionStorage;
  } catch {
    return undefined;
  }
}

/**
 * Identity configuration served by the control plane at runtime.
 *
 * The build arguments below compile an issuer and client id into the bundle, so
 * one image could serve exactly one identity provider -- and an operator with a
 * staging and a production realm had to build two images that differed in
 * nothing else.
 *
 * Build-time values still win when present, so every existing image behaves
 * identically. This only fills the gap they leave.
 */
let runtimeConfig:
  | {
      issuer?: string | null;
      client_id?: string | null;
      audience?: string | null;
      development_auth?: string | null;
      local_attachments_available?: boolean;
      default_org?: string | null;
      default_project?: string | null;
    }
  | undefined;

export function localAttachmentsAvailable(): boolean {
  return runtimeConfig?.local_attachments_available === true;
}

export async function loadRuntimeOidcConfig(fetchImpl: typeof fetch = fetch): Promise<void> {
  if (runtimeConfig !== undefined) return;
  try {
    const response = await fetchImpl("/api/v1/dashboard/config");
    runtimeConfig = response.ok ? await response.json() : {};
  } catch {
    // An unreachable control plane is not an auth misconfiguration. Falling back
    // to the build-time values keeps the failure where it belongs -- the first
    // API call -- rather than turning it into "OIDC is not configured".
    runtimeConfig = {};
  }
}

export function __resetRuntimeOidcConfigForTests(): void {
  runtimeConfig = undefined;
}

/**
 * A declared-but-unset build argument becomes the empty string, not undefined.
 *
 * That is the shape the shipped Dockerfile produces: `ARG VITE_EYVAN_OIDC_ISSUER`
 * with no value. Using `??` here would treat `""` as configured, keep it, and
 * defeat the runtime fallback in exactly the deployment it exists for -- which
 * is what a test caught before this shipped.
 */
function bakedIssuer(): string | undefined {
  return import.meta.env.VITE_EYVAN_OIDC_ISSUER || undefined;
}

function bakedClientId(): string | undefined {
  return import.meta.env.VITE_EYVAN_OIDC_CLIENT_ID || undefined;
}

function issuer(): string | undefined {
  return (bakedIssuer() ?? runtimeConfig?.issuer ?? undefined)?.replace(/\/$/, "");
}

export function isOidcConfigured(): boolean {
  return Boolean(issuer() && (bakedClientId() ?? runtimeConfig?.client_id));
}

let sessionNotice: string | undefined;

/**
 * Monotonic counter of auth-surface changes in this tab.
 *
 * Every place that clears or retires identity -- a rejected session, a
 * refused local credential, a sign-out -- bumps this and fires
 * {@link AUTH_CHANGED_EVENT}. The gate captures the value before an async
 * auth probe starts and discards the probe's result if the value moved,
 * because a probe that began under a live session can resolve after that
 * session was revoked, and its answer then describes a session that no
 * longer exists.
 */
let authChangeCounter = 0;

/** The current auth revision; compare before applying async auth results. */
export function authRevision(): number {
  return authChangeCounter;
}

function emitAuthChanged(): void {
  authChangeCounter += 1;
  if ("dispatchEvent" in globalThis) globalThis.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

/** Ends a team password session: drops the HttpOnly `eyvan_team_session` cookie. */
export const TEAM_LOGOUT_PATH = "/api/v1/team/logout";

// Team accounts (`eyvan-server --team` or `team: true`). The gate records what the plane said, so
// the app can label the session and route sign-out without asking again.
let teamModeEnabled = false;
let teamSessionActive = false;
let teamDisplayName: string | undefined;

export function isTeamMode(): boolean {
  return teamModeEnabled;
}

export function isTeamSession(): boolean {
  return teamSessionActive;
}

export function setTeamState(mode: boolean, session: boolean, displayName?: string): void {
  teamModeEnabled = mode;
  teamSessionActive = mode && session;
  teamDisplayName = teamSessionActive ? displayName : undefined;
}

interface TeamUserBody {
  subject: string;
  email: string;
  display_name: string;
}

async function teamError(response: Response, fallback: string): Promise<Error> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail) return new Error(body.detail);
    if (Array.isArray(body.detail) && body.detail.length) {
      const first = body.detail[0] as { msg?: unknown };
      if (typeof first?.msg === "string") return new Error(first.msg.replace(/^Value error, /, ""));
    }
  } catch {
    // Non-JSON error body: the fallback names the action that failed.
  }
  return new Error(`${fallback} (HTTP ${response.status})`);
}

/**
 * Ask the plane whether it runs team accounts and whether this browser holds a
 * live team session. Only the HttpOnly cookie is sent -- never the local
 * credential -- so a plane owner's development identity is not mistaken for a
 * team account. Never throws: an unreachable plane is "not team mode".
 */
export async function loadTeamState(fetchImpl: typeof fetch = fetch): Promise<void> {
  try {
    const status = await fetchImpl("/api/v1/team/status");
    const body = status.ok ? ((await status.json()) as { team_mode?: boolean }) : {};
    if (!body.team_mode) {
      setTeamState(false, false);
      return;
    }
    const me = await fetchImpl("/api/v1/team/me");
    const account = me.ok ? ((await me.json()) as { user?: TeamUserBody | null }).user : null;
    setTeamState(true, Boolean(account), account ? account.display_name || account.subject : undefined);
  } catch {
    setTeamState(false, false);
  }
}

function enterTeamSession(user: TeamUserBody): void {
  setTeamState(true, true, user.display_name || user.subject);
  sessionNotice = undefined;
  emitAuthChanged();
}

/** Sign in with a team email and password; the server sets the session cookie. */
export async function teamLogin(email: string, password: string, fetchImpl: typeof fetch = fetch): Promise<void> {
  const response = await fetchImpl("/api/v1/team/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) throw await teamError(response, "Sign-in failed");
  enterTeamSession(((await response.json()) as { user: TeamUserBody }).user);
}

/** Who an invite is for. The token travels in the body, never in a URL a server logs. */
export async function previewTeamInvite(
  token: string,
  fetchImpl: typeof fetch = fetch,
): Promise<{ email: string; subject: string; expires_at: string }> {
  const response = await fetchImpl("/api/v1/team/invites/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  if (!response.ok) throw await teamError(response, "This invite could not be read");
  return (await response.json()) as { email: string; subject: string; expires_at: string };
}

/** Create the account from an invite, set its password, and sign in. */
export async function acceptTeamInvite(
  token: string,
  password: string,
  displayName: string,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  const response = await fetchImpl("/api/v1/team/invites/accept", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, password, display_name: displayName }),
  });
  if (!response.ok) throw await teamError(response, "Could not accept the invite");
  enterTeamSession(((await response.json()) as { user: TeamUserBody }).user);
}

function endSession(notice: string): void {
  sessionNotice = notice;
  teamSessionActive = false;
  emitAuthChanged();
}

export function consumeSessionNotice(): string | undefined {
  const notice = sessionNotice;
  sessionNotice = undefined;
  return notice;
}

export function rejectSession(): void {
  endSession("The control plane rejected your session. Sign in again to continue.");
}

/** The credential this browser holds for its loopback plane, if any. */
export function localDevelopmentCredential(): string | undefined {
  const bridged = storage()?.getItem(LOCAL_CREDENTIAL_KEY);
  if (bridged) return bridged;
  // A declared-but-unset build argument becomes the empty string, not undefined.
  return import.meta.env.VITE_EYVAN_LOCAL_CREDENTIAL || undefined;
}

/**
 * Drop a bridged credential the plane has refused, and say so.
 *
 * A credential is written once and read forever, but it is not immortal:
 * deleting the profile's ``.eyvan`` directory, or moving between machines
 * that share a browser session, leaves this tab holding a value the plane
 * will never accept again. Every call would then fail with the same 401 the
 * bare page used to die on -- behind a gate that had already waved it
 * through, because presence was all the gate checked. So the refusal itself
 * retires the value and re-opens the gate, and the screen that knows the
 * one working command replaces a wall of dead panels.
 *
 * The build-time ``VITE_EYVAN_LOCAL_CREDENTIAL`` of a source-checkout dev
 * server cannot be un-baked from here; that flow re-reads the file when the
 * developer restarts it, which is the loop that value is part of.
 */
export function invalidateRefusedLocalCredential(refusalDetail: string): boolean {
  if (!refusalDetail.includes(LOCAL_CREDENTIAL_REQUIRED_DETAIL)) return false;
  const store = storage();
  if (!store?.getItem(LOCAL_CREDENTIAL_KEY)) return false;
  store.removeItem(LOCAL_CREDENTIAL_KEY);
  emitAuthChanged();
  return true;
}

/**
 * Move a bridged development credential out of the URL and into this tab.
 *
 * `eyvan dashboard` puts the value in the fragment -- the one part of a URL
 * no server, proxy, or `Referer` header ever sees -- because that is the only
 * way the owner's terminal can hand it to a browser that cannot read the
 * file it lives in. A fragment still lives in history and in URLs people
 * copy, so it is moved to session storage and stripped from the address bar
 * immediately; the tab keeps it for its lifetime and no longer than that.
 *
 * Returns the absorbed value so the caller can act on the transition.
 */
export function absorbLocalCredentialFromLocation(
  location: Pick<Location, "hash" | "pathname" | "search"> = globalThis.location,
  historyLike: Pick<History, "replaceState"> = globalThis.history,
): string | undefined {
  const hash = location?.hash ?? "";
  if (!hash.startsWith("#")) return undefined;
  const params = new URLSearchParams(hash.slice(1));
  const value = params.get(LOCAL_CREDENTIAL_FRAGMENT_PARAM);
  if (!value) return undefined;
  storage()?.setItem(LOCAL_CREDENTIAL_KEY, value);
  params.delete(LOCAL_CREDENTIAL_FRAGMENT_PARAM);
  const remaining = params.toString();
  historyLike?.replaceState(null, "", `${location.pathname}${location.search}${remaining ? `#${remaining}` : ""}`);
  return value;
}

export function apiAuthHeaders(): Record<string, string> {
  // OIDC adds nothing here on purpose: the session is the `eyvan_session`
  // cookie the browser sends on same-origin requests, and the token itself
  // never leaves the server (see `eyvan/server/api/dashboard_sessions.py`).
  // No Authorization header is ever built from something readable by
  // JavaScript -- that was the vulnerability this flow replaced.
  const localCredential = localDevelopmentCredential();
  if (!isDevelopmentAuthEnabled() && !localCredential) return {};
  const headers: Record<string, string> = {};
  if (isDevelopmentAuthEnabled()) {
    headers["X-Eyvan-User"] = import.meta.env.VITE_EYVAN_DEV_USER ?? "dashboard-dev";
    headers["X-Eyvan-Org"] = import.meta.env.VITE_EYVAN_ORG ?? "acme";
    headers["X-Eyvan-Project"] = import.meta.env.VITE_EYVAN_PROJECT ?? "research";
    headers["X-Eyvan-Roles"] = import.meta.env.VITE_EYVAN_DEV_ROLE ?? "viewer";
  }
  // The credential travels without the identity headers above: those are for
  // a bundle *built* to claim a development identity, while a plane serving
  // its own bundle fills that page's identity itself -- this machine's owner,
  // fleet_admin over the default scope, exactly what the CLI elects on
  // loopback. What the page cannot get from anywhere but its owner is the
  // credential, so that is the one thing this header set insists on.
  //
  // For the source-checkout dev server the value still arrives build-time:
  //   VITE_EYVAN_LOCAL_CREDENTIAL=$(cat ~/.eyvan/local-development-credential) npm run dev
  if (localCredential) headers["X-Eyvan-Local-Credential"] = localCredential;
  return headers;
}

/**
 * Begin sign-in by handing the browser to the server-side login endpoint.
 *
 * The PKCE challenge, the state, and the nonce are minted and remembered on
 * the server; this page never holds them, so nothing here can be tricked
 * into leaking them. The identity provider brings the browser back to the
 * server's callback, which opens the session and redirects here again.
 */
export function beginLogin(): void {
  if (!isOidcConfigured()) throw new Error("OIDC client configuration is incomplete");
  globalThis.location.assign(LOGIN_PATH);
}

/**
 * Ask the server whether this browser already holds a live dashboard session.
 *
 * The gate cannot read the session cookie -- that is the point of it -- so
 * the only way to know a signed-in user is sitting here is to ask the
 * authority endpoint and let the cookie speak. A 401 means "no session"; it
 * is not an error and it is never narrated to the user.
 */
export async function serverIdentity(fetchImpl: typeof fetch = fetch): Promise<AuthIdentity | undefined> {
  if (!isOidcConfigured()) return undefined;
  try {
    const response = await fetchImpl("/api/v1/session", { headers: apiAuthHeaders() });
    if (!response.ok) return undefined;
    const authority = (await response.json()) as { subject?: string; display_name?: string | null };
    if (!authority.subject) return undefined;
    return {
      displayName: authority.display_name ?? authority.subject,
      detail: "OIDC session",
      mode: "oidc",
      roles: [],
      scope: configuredScope(),
      capabilities: NO_AUTH_CAPABILITIES,
    };
  } catch {
    return undefined;
  }
}

export function currentIdentity(): AuthIdentity | undefined {
  const scope = configuredScope();
  if (isDemoMode()) {
    const permissions: SessionPermission[] = [
      "read",
      "submit",
      "operate",
      "manage_project",
      "manage_own_capacity",
      "manage_fleet",
    ];
    return {
      displayName: "Demo dataset",
      detail: "Local sample data",
      mode: "demo",
      roles: ["fleet_admin"],
      scope,
      capabilities: capabilitiesForPermissions(permissions),
    };
  }

  if (isDevelopmentAuthEnabled()) {
    return {
      displayName: import.meta.env.VITE_EYVAN_DEV_USER ?? "dashboard-dev",
      detail: "Development identity; authority pending server verification",
      mode: "development",
      roles: [],
      scope,
      capabilities: NO_AUTH_CAPABILITIES,
    };
  }
  if (teamSessionActive) {
    // The HttpOnly cookie carries the session; authority still comes from the
    // server's own response, which the app merges over these placeholders.
    return {
      displayName: teamDisplayName ?? "Team member",
      detail: "Team session",
      mode: "team",
      roles: [],
      scope,
      capabilities: NO_AUTH_CAPABILITIES,
    };
  }
  // OIDC is intentionally absent: the tab holds no token and no claims, so
  // there is nothing local identity can truthfully report. The gate asks
  // {@link serverIdentity} instead, and what the user sees comes from the
  // server's own authority response.
  return undefined;
}

/**
 * Sign out: drop the server-side session, then land back on the sign-in
 * screen. Idempotent -- a session that is already gone is not an error.
 */
export async function logout(fetchImpl: typeof fetch = fetch): Promise<void> {
  if (isDemoMode()) {
    const url = new URL(globalThis.location.href);
    url.searchParams.delete("demo");
    globalThis.location.assign(`${url.pathname}${url.search}${url.hash}`);
    return;
  }
  if (teamSessionActive) {
    try {
      const response = await fetchImpl(TEAM_LOGOUT_PATH, { method: "POST" });
      if (!response.ok) throw new Error(`Sign-out returned ${response.status}`);
    } catch {
      endSession(
        "Could not confirm sign-out. Your session may still be active; retry when the control plane is reachable.",
      );
      return;
    }
    endSession("You have been signed out.");
    globalThis.location.assign("/");
    return;
  }
  if (isOidcConfigured()) {
    try {
      const response = await fetchImpl(LOGOUT_PATH, { method: "POST" });
      if (!response.ok) throw new Error(`Sign-out returned ${response.status}`);
    } catch {
      // JavaScript cannot delete an HttpOnly cookie. Without a confirmed
      // server response, neither the session nor its cookie is known to be gone.
      endSession(
        "Could not confirm sign-out. Your session may still be active; retry when the control plane is reachable.",
      );
      return;
    }
    endSession("You have been signed out.");
  }
  emitAuthChanged();
  globalThis.location.assign("/");
}
