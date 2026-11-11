# esurge-dashboard

Server-rendered dashboard for the eSurge OpenAI-compatible API server. It is one static binary with no
JavaScript and no Grafana or Prometheus. It reads only the API server's `GET /metrics`.

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
| `--refresh-seconds` | `5` | page refresh interval |
| `--upstream-timeout-seconds` | `5` | timeout for one metrics request |
| `--session-ttl-seconds` | `43200` | browser session lifetime |
| `--secure-cookie` | off | mark the session cookie `Secure` (use behind HTTPS) |

## Authentication

The dashboard has no key and reads no key file. Each request's API key is forwarded to the API server's
`/metrics`, and the API server alone decides whether it is valid. A rejected key gives 401, and a key that can't be
read gives a login redirect, so access is never left open.

- Scripts send `Authorization: Bearer <key>` or `X-API-Key: <key>`.
- Browsers sign in at `/login`. The key is checked against the API server and held in memory under a random
  256-bit session id. The cookie (`HttpOnly`, `SameSite=Strict`) carries only that id, never the key.
- Keys in the query string are ignored, because URLs end up in logs and browser history.

Routes: `/` (dashboard), `/api/metrics` (upstream JSON passthrough), `/login`, `/logout`, `/healthz`.
