# Eyvan Tutorial

A ground-up tutorial for [Eyvan](../../README.md) — a capacity plane for research
compute. You state the shape of work you need (`2 × v5p-16, spot, this fleet`) as a
digest-verified, unsigned Python callable from an explicitly trusted project. The
plane acquires machines, dispatches your callable, and returns its result.

Every part of this tutorial was verified against the repository source. When a
section makes a claim about behavior, it names the file that implements it, so you
can always go one level deeper.

## The parts

| Part | File | What it covers |
|---|---|---|
| 1 | [01-what-is-eyvan.md](01-what-is-eyvan.md) | The mental model, the five core ideas, architecture, concepts |
| 2 | [02-getting-started.md](02-getting-started.md) | Install, the laptop fleet, your first callable end to end |
| 3 | [03-sdk.md](03-sdk.md) | `init`, `@execute`, shape configs, `EyvanJob`, workspaces, secrets |
| 4 | [04-cli.md](04-cli.md) | Core `eyvan` command families and what they are for |
| 5 | [05-domain-and-scheduling.md](05-domain-and-scheduling.md) | Domain policy: scheduler, DRF fairness, pricing, spend, availability |
| 6 | [06-capacity-and-fleets.md](06-capacity-and-fleets.md) | Fleet configs, pools, planning, autoscale generation, fleet state |
| 7 | [07-providers.md](07-providers.md) | Drivers, capabilities, fencing, conformance, writing a provider |
| 8 | [08-control-plane.md](08-control-plane.md) | The server: controller loops, state machines, persistence, invariants |
| 9 | [09-api-and-auth.md](09-api-and-auth.md) | The REST API, authentication paths, SSE streams, idempotency |
| 10 | [10-workers.md](10-workers.md) | The cell agent, native workers, execution, logs, progress |
| 11 | [11-artifacts-and-secrets.md](11-artifacts-and-secrets.md) | Digest-addressed artifacts, managed lifecycle, capability tokens |
| 12 | [12-security.md](12-security.md) | Trust boundaries, the secret table, what jobs can and cannot do |
| 13 | [13-dashboard.md](13-dashboard.md) | The React console: routing, auth, live streams, pages |
| 14 | [14-operations.md](14-operations.md) | The daily runbook, zone racing, troubleshooting, proofs |
| 15 | [15-testing.md](15-testing.md) | Test layout, suites, the provider-boundary rule, dashboard tests |
| 16 | [16-deployment.md](16-deployment.md) | The production prerequisites, `head up`, Helm and raw manifests |
| 17 | [17-spot-priority-demo.md](17-spot-priority-demo.md) | Explicit Spot TPU priority/checkpoint example, receipts and owned cleanup |

## Suggested paths through it

- **"I just want to run something":** parts 1 → 2 → 3. That is enough to submit a
  callable to the loopback fleet on your laptop.
- **"I will run this on real accelerators":** add parts 6 (fleets) and 14
  (operations — especially zone racing before you spend).
- **"I am deploying it for a team":** parts 8, 9, 12, 16, then 14 for the runbook.
- **"I am adding a cloud":** parts 5 (what the core expects), 6, 7.
- **"I want to work on Eyvan itself":** parts 5, 8, 15.

## Where the authoritative docs live

This tutorial teaches; the docs under `docs/` specify. The two are consistent, but
when they disagree the source wins:

- `docs/eyvan-explained.md` — the mental model
- `docs/install.md`, `docs/operate.md` — setup and the runbook
- `docs/reference/*.md` — per-subsystem references (SDK, control plane, REST, …)
- `docs/env-surface.md` — every `EYVAN_*` variable
- `docs/design/` — architecture decision records with their real status
