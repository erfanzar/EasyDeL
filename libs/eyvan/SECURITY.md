# Security Policy

## Reporting a Vulnerability

Do **not** open a public issue for security problems. Report them privately to:

- **Email:** Erfanzare810@gmail.com

Include, where possible:

- The affected component (`eyvan-server`, `eyvan-agent`, `eyvan-native-worker`, SDK, dashboard, CI/release pipeline).
- Reproduction steps or a proof of concept.
- The impact you believe it has (confidentiality, integrity, availability, supply chain).

You should receive an acknowledgment within 72 hours. Please allow a reasonable
window for a fix before public disclosure; we will coordinate an advisory and a
release with you.

## Scope

In scope:

- The control plane HTTP API (authentication, authorization, webhook delivery).
- The cell agent and native worker bootstrap (token derivation, secret
  delivery, startup script integrity).
- The artifact store and capability tokens.
- Provider drivers (credential handling, IAM bindings they create).
- The dashboard's handling of operator-visible data.
- Release integrity: hash-pinned installs (`requirements-*.txt`), lockfile
  consistency, and OCI image provenance.

Out of scope:

- Deployments with authentication disabled outside local development.
- Compromise of operator-controlled infrastructure (control plane hosts,
  provider projects) — though we still want to hear about weaknesses that
  made it easy.

## Supported Versions

Security fixes land on `main` and the most recent tagged release. Older
releases are not maintained; upgrade rather than backport.

## Supply Chain

- Python installs used in CI, Docker, and OCI images are hash-pinned via
  `requirements-*.txt` exports, which must stay in sync with `uv.lock`.
- Never commit real credentials, tokens, or private keys. Tests use
  generated or placeholder material only.
- Every source file carries the AGPL-3.0-or-later header; keep it that way.
