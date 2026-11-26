# Eyvan dashboard

The dashboard uses the canonical `/api/v1` API. Sample data is disabled by
default and is available only with `?demo=1` or an explicit
`VITE_EYVAN_DEMO=true` development build.

## Distribution contract

The Python wheel intentionally contains neither these Node sources nor built
static assets. The production `Dockerfile` builds them into the control-plane
image and sets `EYVAN_DASHBOARD_DIR`; `eyvan dashboard` from any installed wheel
opens that hosted UI. `eyvan dashboard-dev` is the explicitly source-only Vite
launcher and fails with a source-checkout message when those files are absent.

## Browser authentication

Production uses a backend-for-frontend (BFF) with an OIDC public client and
authorization-code PKCE. Configure these settings on the control plane, not
in the dashboard build:

```bash
EYVAN_PUBLIC_URL=https://eyvan.example.com
EYVAN_OIDC_ISSUER=https://identity.example.com
EYVAN_OIDC_DASHBOARD_CLIENT_ID=eyvan-dashboard
EYVAN_OIDC_AUDIENCE=eyvan-api
```

The identity provider must register the exact redirect URI
`<EYVAN_PUBLIC_URL>/api/v1/dashboard/auth/callback` (with no trailing slash on
`EYVAN_PUBLIC_URL`). For the example above, register
`https://eyvan.example.com/api/v1/dashboard/auth/callback`, **not** the dashboard
root URL. Allow authorization-code PKCE for this public client with the
`openid profile email` scopes. The control plane must be able to access the
provider's discovery and token endpoints; the browser does not perform the
code exchange.

Sign in navigates to `/api/v1/dashboard/auth/login`. The server constructs the
provider redirect from its configuration, exchanges the returned code at the
callback, and redirects back to `/`. Access and ID tokens stay server-side;
the browser carries only an HttpOnly, Secure, SameSite=Lax session cookie for
OIDC authentication. No client secret or static bearer token is compiled into
the dashboard.

The hosted bundle reads issuer, client ID and audience from
`/api/v1/dashboard/config`. This keeps one signed image digest promotable
between environments. Optional `VITE_EYVAN_OIDC_ISSUER`,
`VITE_EYVAN_OIDC_CLIENT_ID` and `VITE_EYVAN_OIDC_AUDIENCE` settings override that
discovery for an intentional custom build, but do not configure the server's
sign-in request or callback. Keep any such overrides aligned with the control
plane settings.

Set `VITE_EYVAN_ORG` and `VITE_EYVAN_PROJECT` to select the API scope; they
default to `acme` and `research` for local development.

Local identity headers are opt-in and should be paired with
`EYVAN_DEV_AUTH_ENABLED=1` on the API. From the repository root:

```bash
npm --prefix dashboard install
uv run eyvan dashboard-dev
```
