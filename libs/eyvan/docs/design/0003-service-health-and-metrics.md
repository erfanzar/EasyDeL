# DR-0003: Service health and metrics ownership

**Status:** Managed-service health records, probes and internal autoscaling are
implemented. Controller-progress readiness and an external autoscaling-signal
feed remain open. Eyvan is a control plane, not an application metrics pipeline.

## Current health boundary

- `/healthz` (also `/api/v1/health`) is unauthenticated liveness. It reports
  process responsiveness and the controller mode selected at startup; it does
  not probe the database or prove that the controller loop is advancing.
- `/readyz` is unauthenticated readiness. It calls `repository.ping()` and
  returns 503 when the durable backend cannot be reached. It does not test
  agent reachability or controller progress.
- `/metrics` checks `Authorization: Bearer <EYVAN_METRICS_TOKEN>` when the
  token is configured. Network-bound startup requires that token; it is not
  authenticated through the ordinary user identity flow.

These contracts are implemented in `eyvan/server/api/app.py` and the startup
requirements in `eyvan/server/api/server_startup.py`.

## Managed services and telemetry

Managed-service health is separate from plane liveness: typed
`ServiceReplicaHealthRecord` observations in `eyvan/records.py` feed
`replica_health_state()` and `plan_autoscale()` in
`eyvan/server/service/services.py`. The controller probes replicas and applies
bounded lifecycle/autoscaling decisions.

Plane telemetry describes control-plane facts such as workload state, cell
transitions and scheduling outcomes. Request bodies and raw application
telemetry do not belong in this metrics surface. An external scaler feed is
not supplied by the health probes.

## Acceptance criteria still open

- Detect a controller loop that stops advancing without tying process liveness
  to a shared database outage.
- Define any additional component readiness gates explicitly; do not infer
  agent health from database reachability.
- Specify and qualify an authenticated external autoscaling-signal contract.

A successful health probe alone is not workload, provider or deployment proof.
