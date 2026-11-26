# Part 13 — The dashboard

Source: `dashboard/` (React 19 + TypeScript, Vite 6, ESM). Reference:
`docs/reference/dashboard.md`. No router library, no state-management library —
`App.tsx` owns the snapshot, scope, route, and refresh lifecycles.

## Architecture in one paragraph

`main.tsx` renders `<AuthGate><App /></AuthGate>` under StrictMode. `App` parses
the hash route, loads authority (`/session`) and a coherent dashboard snapshot,
and selects the page; a changed project scope remounts page content.
`Shell.tsx` is the persistent layout (desktop rail / mobile drawer, scope
selector, health/live status, submit control, identity area). Data flows through
small hooks (`useLiveRefresh`, `useVisibilityPolling`) and `api/client.ts`, which
converts control-plane snake_case JSON into the UI types of `types.ts`.
`lucide-react` supplies icons; styling is `styles.css`.

## Hash routing without a router

`routing.ts` defines sections `overview`, `workloads`, `runs`, `research`,
`fleet`, `capacity`, `billing`, `budgets`, `audit`, `assistant`, and `settings`.
Detail routes are `#/workloads/<id>`, `#/runs/<id>[/<tab>]`, and
`#/research/<id>`; IDs are `encodeURIComponent`-encoded. Run tabs include charts,
episodes, histograms, rollouts, completions, and media. Unknown sections fall back
to Overview; `App` listens to `hashchange`; `accessibleRoute` applies server-derived
section availability before rendering, so forbidden sections are removed from
navigation and inaccessible routes project back to Overview. Hash routing means
the server needs no per-page rewrite rules (Part 9).

## Authentication and authorization

`AuthGate` absorbs a loopback credential from the URL fragment, loads
`/api/v1/dashboard/config` when baked OIDC settings are incomplete, and asks
`/api/v1/session` whether a server session exists before rendering `App`.
OIDC sign-in uses a backend-for-frontend (BFF), server-side authorization-code/PKCE
flow: `auth/oidc.ts` navigates to `/api/v1/dashboard/auth/login`, and the server
performs discovery with the configured issuer and creates the login transaction.
The identity provider returns the browser to `/api/v1/dashboard/auth/callback`;
the server exchanges the authorization code and opens the session. OIDC access/ID
tokens, PKCE verifier/nonce, and expiry stay server-side. The browser holds an
`__Host-eyvan_session` cookie (HttpOnly, Secure, SameSite=Lax), not OIDC tokens.
The temporary login-transaction cookie is `__Host-eyvan_oidc_tx`; both cookies
use `Path=/` and no `Domain`. Their `__Host-` prefix prevents sibling-subdomain
parent-domain cookie planting; legacy unprefixed cookies are not accepted, so
existing browsers must sign in again.
Only the loopback development credential lives in `sessionStorage`; it is removed
from the URL fragment when absorbed.

Deploy OIDC behind HTTPS (TLS may terminate at the reverse proxy): the cookies
are unconditionally Secure. Set `EYVAN_PUBLIC_URL` to the browser-facing HTTPS
URL; the callback URI and cookie-write Origin checks use this configuration,
not Host or forwarded headers. A proxy must preserve the browser's Origin and
cookies, and must route the login and callback paths to the API, not to the
dashboard SPA. Proxy headers do not override the configured public origin or
waive the network listener's explicit plaintext acknowledgement requirement;
see `docs/security.md` for that separate boundary.
Login transactions and sessions use an in-process store;
with multiple API replicas, sticky sessions must keep login, callback, and
subsequent API requests on the same process. Cookie-authenticated writes require
an Origin matching the configured public URL. See `docs/security.md` for the
session lifetime, store limits, and CSRF boundary.

The server's `/session` response is authoritative for subject, scopes, roles,
permissions, and `accepts_new_work`. The UI derives presentation capabilities from
those permissions — it does **not** decode an unverified JWT to grant access.
`App` gates navigation and mutations by `read`, `submit`, `operate`,
`manage_project`, `manage_fleet`; submit-only users do not mount read-only detail
loaders. A 401/403 on a stream or API call triggers session/authority
reconciliation rather than an endless forbidden-read loop.

Demo mode: `?demo=1` or `VITE_EYVAN_DEMO=true` uses `api/mockData.ts` (not a live
connection). Development auth: `VITE_EYVAN_DEV_AUTH`.

## Live updates: two different stream uses

- **SSE as notification**: `client.followEvents` follows the scoped
  `/events/stream`; `useLiveRefresh` coalesces bursts for 400 ms and *refreshes*
  the snapshot instead of interpreting event payloads as the model.
- **SSE as data**: `followWorkloadLogs` delivers the workload log stream, resetting
  the log view on retry-attempt change events.

`lib/eventStream.ts` parses frames, ignores comments/malformed JSON, remembers
`Last-Event-ID`, reconnects with capped 1/2/5/10/30 s backoff, and stops on abort.
`useVisibilityPolling` sweeps every 30 s while the tab is visible (polling covers
changes outside the selected tenant stream and proxies that buffer SSE).

## Pages

| page | shows |
|---|---|
| `OverviewPage` | derived scheduling/capacity/spend overview, queue trend, alerts, shortcuts |
| `WorkloadsPage` | scoped workload rows, submission entry point |
| `WorkloadDetailPage` | lifecycle, allocation, timeline, progress/explanation/cost/process info, operations, logs |
| `FleetPage` | cells, leases, topology, warmth, billing and launch-latency panels, pools, fleet actions |
| `CapacityPage` | provider TPU quota + capacity history where authority permits |
| `BillingPage` | billing and usage reports |
| `BudgetsPage` | budgets and scheduling policies (policy mutations gated by project-management capability) |
| `AuditPage` | project-scoped durable audit events |

Reusable components (`components/`): `MetricCard`, `StagePipeline` (derives the
eight stages Submitted → Admitted → Capacity → Provisioning → Dispatched → Workers
ready → Running → Settled from state/phase/timeline), `StatusBadge`,
`ProgressBar`, `EmptyState`, `ConfirmDialog`, `SubmitWorkloadModal`, `PageState`.

## Build, dev, and tests

```bash
cd dashboard
npm run dev        # Vite dev server, port 4173, proxies /api to EYVAN_DASHBOARD_API (default http://127.0.0.1:8080)
npm run build      # tsc --noEmit, then vite build
npm run preview    # port 4174
npm test           # vitest run
```

Vitest's default environment is Node; DOM suites opt in per file with
`@vitest-environment jsdom`. Colocated tests cover routing, event streams, live
refresh, auth (oidc/AuthGate), workload rows, panels, logs, capacity, billing,
fleet lifecycle/actions, spend, submit, and the authorization matrix.
