# esurge-dashboard

Dashboard for the eSurge OpenAI-compatible API server, styled after px0.ai's light workbench (JetBrains Mono,
`#f4f5f8` background, square panels, amber accent). It is one static binary that embeds its own stylesheet and a
dependency-free script, and it reads only the API server's `GET /metrics`. No Grafana or Prometheus is needed.

## What it shows

Sixteen live charts, all inline SVG with hover tooltips, over a selectable range (5m, 15m, 1h, 6h, 24h, all):

- **Throughput:** generated tok/s (30 s trailing rate of the server counter), engine output tok/s, per-request
  decode tok/s, prefill tok/s and request/failure rate.
- **Load:** running, waiting and HTTP in-flight requests; prompt against output tokens per request; decode tok/s
  distribution.
- **Latency:** TTFT and end-to-end latency p50/p99 over time (log scale); TTFT against prompt length; latency against
  output length; TTFT distribution.
- **Prefix cache:** token and request hit rate over time, a cache summary, and cached against prefilled tokens for
  each recent request.
- **Requests:** prompt and output length distributions, finish reasons, HTTP outcomes and a request log.
- **Access:** key counts and tokens per second across keys.

## Build

```bash
cargo build --release          # target/release/esurge-dashboard
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
```

## Run

```bash
esurge-dashboard --upstream http://<api-host>:8000 --listen <internal-ip>:9090
```

| flag | default | meaning |
|---|---|---|
| `--upstream` | `http://127.0.0.1:8000` | API server base URL; the dashboard calls `{upstream}/metrics` |
| `--listen` | `127.0.0.1:9090` | listen address |
| `--refresh-seconds` | `5` | browser poll interval |
| `--upstream-timeout-seconds` | `5` | timeout for one metrics request |
| `--session-ttl-seconds` | `43200` | browser session lifetime |
| `--history-samples` | `43200` | metrics snapshots kept in memory for the charts (at most one per second) |
| `--history-requests` | `5000` | completed requests kept in memory for the log and per-request charts |
| `--secure-cookie` | off | mark the session cookie `Secure` (use behind HTTPS) |

## Chart history

The dashboard holds no API key, so it cannot poll the API server by itself. History grows from viewers' polls: each
`/api/history` call is first checked upstream with the viewer's own key, and only then recorded and answered. With
a browser open, the charts therefore cover the time the page has been open, and anyone opening the page later sees
that history too. History lives in memory and is lost when the dashboard restarts. Snapshots closer than one second
apart are recorded once, so several open browsers do not multiply it.

## Authentication

The dashboard has no key and reads no key file. Each request's API key is forwarded to the API server's
`/metrics`, and the API server alone decides whether it is valid. A rejected key gives 401, and a key that can't be
read gives a login redirect, so access is never left open.

- Scripts send `Authorization: Bearer <key>` or `X-API-Key: <key>`.
- Browsers sign in at `/login`. The key is checked against the API server and held in memory under a random
  256-bit session id. The cookie (`HttpOnly`, `SameSite=Strict`) carries only that id, never the key.
- Keys in the query string are ignored, because URLs end up in logs and browser history.
- Every response carries a strict Content-Security-Policy (`script-src 'self'`, no inline script or style,
  `frame-ancestors 'none'`), `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY`. The only external load is
  the JetBrains Mono webfont from Google Fonts; without it the page falls back to the system monospace font.

## Routes

| route | auth | returns |
|---|---|---|
| `/` | key or session | dashboard page |
| `/api/history?after=<seq>&limit=<n>` | key or session | current `/metrics` plus the samples and requests newer than `seq` |
| `/api/metrics` | key or session | the API server's `/metrics` JSON, unchanged |
| `/assets/dashboard.css`, `/assets/dashboard.js` | none | embedded static assets |
| `/login`, `/logout` | none | browser sign-in and sign-out |
| `/healthz` | none | `ok` |
