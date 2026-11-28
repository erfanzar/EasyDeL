# Eyvan dashboard — release reference

This reference describes the dashboard as implemented in `dashboard/`. The
source files named below are the source of truth; this document intentionally
does not describe API behavior that is not exercised or consumed by the UI.

## Architecture

The dashboard is a private, ESM React 19 + TypeScript application built with
Vite 6 (`dashboard/package.json`). It has no router or state-management library.
`App` owns the dashboard snapshot, selected project scope, route, workload
detail state, modal/toast state, and refresh lifecycles. Small hooks and the API
client provide the data flow. `lucide-react` supplies icons; styling is in
`dashboard/src/styles.css`.

The data model consumed by pages is declared in `dashboard/src/types.ts`.
`dashboard/src/api/client.ts` converts control-plane JSON (snake_case) into
those UI types and also contains overview derivation (`computeOverview`).
Demo mode uses `dashboard/src/api/mockData.ts`; it is not a live control-plane
connection.

## Entrypoint and App shell

`dashboard/index.html` supplies `#root`, metadata, the `Eyvan Console` title,
Google Fonts links, and the `src/main.tsx` module. `dashboard/src/main.tsx`
creates a React root under `StrictMode`, imports the stylesheet, and renders
`<AuthGate><App /></AuthGate>`.

`dashboard/src/App.tsx` initially parses the hash, loads authority and a
coherent dashboard snapshot, and selects the page. It supplies page callbacks
for detail loading, workload operations/submission, fleet actions, billing and
capacity panels, policy updates, and retry/error handling. A changed project
scope remounts page-local content (`Fragment` keyed by scope). Forbidden
sections are removed from navigation and inaccessible routes project back to
Overview; submit-only users do not mount read-only detail loaders.

`dashboard/src/components/Shell.tsx` implements the persistent layout: desktop
rail/mobile drawer, project-scope selector, health/live status, page title and
workload-detail breadcrumb, refresh and submit controls, identity/sign-out
area, and section counts. The rail collapse preference is stored under
`eyvan.rail.collapsed` in local storage. `PageState.tsx` provides the loading
and retryable error states.

## Hash routing (without a router)

`dashboard/src/routing.ts` defines the route union and the sections
`overview`, `workloads`, `runs`, `research`, `fleet`, `capacity`, `billing`,
`budgets`, `audit`, `assistant`, and `settings`. Detail routes include
`#/workloads/<id>`, `#/research/<id>`, and `#/runs/<id>` with optional
`episodes`, `histograms`, `rollouts`, `completions`, or `media` tabs (charts
are the default). IDs are decoded on read and encoded on navigation.
`#/users` is a compatibility alias for Settings. `routeHash`/`navigate` write
the corresponding hash. Unknown sections fall
back to Overview. `App` listens for `hashchange`; `accessibleRoute` applies
server-derived section availability before rendering. Workload IDs are
`encodeURIComponent`-encoded when navigated.

The behavior and edge cases are covered by `dashboard/src/routing.test.ts`.
This is hash routing, not history/API-path routing; the server therefore does
not need a separate fallback for each page URL.

## API client

`EyvanApiClient` defaults to `/api/v1` and appends the selected `org_id` and
`project_id` to scoped requests (`dashboard/src/api/client.ts`). JSON requests
send `Accept`, `Content-Type`, and `apiAuthHeaders()`; requests have an abort
signal and a control-plane timeout, and non-success responses become
`ApiError`. A 401 rejects the browser session and can invalidate a refused
loopback credential. Text requests are used for the one-shot workload-log
fallback.

The client exposes health, session authority, workload lists/details/progress,
explanations, cost and process status, workload operations and submission,
leases/cells, available submission shapes, budgets, policies, project audit
events, fleet capacity events, billing, warmth, launch latency, and TPU quota.
The exact endpoint paths and mapping are in the client; `App.loadDashboardData`
(`dashboard/src/App.tsx`) deliberately avoids known-forbidden reads and joins
fleet metadata only when authorized. The dashboard is a consumer of the
control-plane contracts, not an independent source of truth.

## Authentication and authorization

`AuthGate` (`dashboard/src/auth/AuthGate.tsx`) absorbs a loopback credential
from the URL fragment, loads team-mode state and `/api/v1/dashboard/config`
when baked OIDC settings are incomplete, then calls `serverIdentity()` in
`dashboard/src/auth/oidc.ts` to probe `/api/v1/session`. A team password session
uses its own HttpOnly cookie identity. Invite links open the acceptance screen
before the application; signed-out team users see password login and, when
configured, an OIDC sign-in option. Otherwise a refused or missing OIDC session
shows the OIDC sign-in screen.

OIDC uses a backend-for-frontend boundary. `beginLogin()` navigates to
`/api/v1/dashboard/auth/login`; the server performs issuer discovery and
creates the PKCE state, verifier, and nonce. The identity provider returns to
`/api/v1/dashboard/auth/callback`, where the server validates the transaction,
exchanges the code, and stores the access and ID tokens, session state, and
expiry server-side (`eyvan/server/api/dashboard_sessions.py`). The browser
receives an opaque HttpOnly, Secure, SameSite=Lax session cookie, not OIDC
tokens; JavaScript cannot read that cookie. Same-origin API requests carry it
automatically, and `apiAuthHeaders()` does not construct an OIDC bearer header.
Sign-out posts to `/api/v1/dashboard/auth/logout` to remove the server session
and clear its cookie; an unconfirmed logout is reported rather than presented
as success.

The loopback-only development credential is a separate exception, not OIDC
token storage: `eyvan dashboard` bridges it through the URL fragment,
`absorbLocalCredentialFromLocation()` strips it from the address bar and stores
it in `sessionStorage` under `eyvan.local.credential`, and API requests present
it as `X-Eyvan-Local-Credential`. `?demo=1` or `VITE_EYVAN_DEMO=true` enables
demo mode; explicit development auth is controlled by `VITE_EYVAN_DEV_AUTH`.

The server-resolved `/session` response is authoritative for subject, scopes,
roles, permissions, enabled state, and `accepts_new_work`. The UI derives
presentation capabilities from those permissions; it does not decode an
unverified JWT to grant access. `App` gates navigation and mutation callbacks
by `read`, `submit`, `operate`, `manage_project`, and `manage_fleet`. Disabled
readable projects may remain visible as history-only scopes. A stream or API
401/403 triggers session/authority reconciliation rather than an endless
forbidden-read loop. OIDC and capability rules are covered by
`dashboard/src/auth/oidc.test.ts`, `AuthGate.dom.test.tsx`, and
`dashboard/src/pages/authorization.dom.test.tsx`.

## Live updates: SSE and polling

There are two distinct stream uses:

* `EyvanApiClient.followEvents` follows the scoped `/events/stream` SSE endpoint
  as a change **notification**, not a data channel. `useLiveRefresh` coalesces
  bursts for 400 ms and refreshes the dashboard/detail instead of interpreting
  event payloads as the complete model.
* `followWorkloadLogs` follows the workload log stream as a data channel and
  delivers log records; retry-attempt change events reset the relevant log
  view. It is separate from the audit stream.

`dashboard/src/lib/eventStream.ts` parses SSE frames, ignores comments and
malformed JSON frames, remembers `Last-Event-ID`, reconnects with capped
1/2/5/10/30-second backoff, and stops on abort. Authorization failures are
surfaced. In demo mode the client does not claim a live connection.

`useVisibilityPolling` in `dashboard/src/lib/useVisibilityPolling.ts` runs the
fallback sweep every 30 seconds while the tab is visible, pauses when hidden,
and ticks immediately on return. `App` uses it for the dashboard and workload
detail. Polling remains necessary for changes not represented in the selected
tenant stream (for example shared-fleet changes) and for proxies that buffer
`text/event-stream`. Relevant tests are
`dashboard/src/lib/eventStream.test.ts` and
`dashboard/src/lib/useLiveRefresh.test.tsx`.

## Pages

`App.tsx` selects these page components:

* `OverviewPage.tsx`: derived scheduling/capacity/spend overview, queue trend,
  alerts, workload shortcuts, and audit shortcut when available.
* `WorkloadsPage.tsx`: scoped workload rows and submission entry point.
* `WorkloadDetailPage.tsx`: one workload's lifecycle, allocation, timeline,
  progress/explanation/cost/process information, operations, and logs.
* `FleetPage.tsx`: cells, leases, resource/topology state, warmth, billing and
  launch-latency panels, pools, and fleet actions when permitted.
* `CapacityPage.tsx`: provider TPU quota plus capacity history where fleet
  authority permits it.
* `BillingPage.tsx`: billing and usage reports loaded through page callbacks.
* `BudgetsPage.tsx`: budgets and scheduling policies, with policy mutations
  gated by project-management capability.
* `AuditPage.tsx`: project-scoped durable audit events.

The page files are the source of truth for displayed fields, empty states,
filters, and action affordances. DOM coverage includes workload rows, panels,
logs, capacity, billing, fleet lifecycle/actions, spend, and authorization in
`dashboard/src/pages/*dom.test.tsx` and related `App`/submit tests. A detail
route is still a workload route even when a typed authorization denial is being
shown; it does not silently retry the denied read.

## Components and utilities

Reusable UI is in `dashboard/src/components/`: `MetricCard` displays a labeled
figure/detail and optional qualifier; `StagePipeline` derives the eight
Submitted → Admitted → Capacity → Provisioning → Dispatched → Workers ready →
Running → Settled stages from workload state/phase/timeline; `StatusBadge`,
`ProgressBar`, `EmptyState`, `ConfirmDialog`, `SubmitWorkloadModal`, and
`PageState` cover common status, progress, empty, confirmation, submission, and
loading/error UI. `Shell` is the layout component.

`dashboard/src/lib/` contains `eventStream.ts`, the refresh hooks,
`format.ts` (display formatting), and `accelerator-request.ts` (submission
request handling/validation). Their unit/DOM tests are colocated. These
utilities do not add a cache or persistence layer for control-plane data.

## Build, development, and tests

From `dashboard/`:

```text
npm run dev       # Vite development server
npm run build     # tsc --noEmit, then vite build
npm run preview   # Vite preview server
npm test          # vitest run
npm run test:watch
```

`dashboard/vite.config.ts` uses port 4173 for development and 4174 for
preview. The dev server proxies `/api` to `EYVAN_DASHBOARD_API`, defaulting to
`http://127.0.0.1:8080`. Vitest's default environment is Node; DOM suites opt
in per file with `@vitest-environment jsdom`. `package-lock.json` records the
resolved toolchain. The scripts describe available checks; this reference
makes no claim that a particular run has passed.

## Static serving by the control plane

`eyvan/server/api/app.py` is the server-side source of truth. If
`EYVAN_DASHBOARD_DIR` names an existing directory, that directory is served;
otherwise a checkout's `dashboard/dist` is used when it contains `index.html`,
and then the copy packaged inside the wheel as `eyvan/_dashboard`. The
selected directory is mounted at `/` with FastAPI `StaticFiles(..., html=True)`,
which serves the built SPA and its static assets.

Release builds put the dashboard in the package: after
`npm --prefix dashboard run build`, `python3 scripts/stage_dashboard.py` copies
`dashboard/dist` to the gitignored `eyvan/_dashboard`, `uv build` packs it into
the wheel and sdist, and `python3 scripts/stage_dashboard.py --check dist/*`
refuses an artifact missing `index.html` or any asset it loads. The CI package
job and `publish.yaml` run all three, so `pip install eyvan` serves the UI.

If no bundle is available, `/` returns a small self-contained explanation that
the plane serves the API but not the dashboard, with `eyvan dashboard-dev`,
`/healthz`, and `/docs` pointers. A wheel built without staging is in that
state.

The static mount is intentionally at `/`; API routes remain under `/api/...`.
Because the dashboard uses hash routes, deep navigation stays at the same
served document and does not require history-mode rewrite rules. The control
plane's missing-bundle response and bundle selection are implemented in
`dashboard_bundle()` and the final mount block in `eyvan/server/api/app.py`.
