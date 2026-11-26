# Part 15 — Testing

Layout: `tests/` (Python, pytest), `dashboard/src/**/*.test.ts(x)` (Vitest),
`eyvan/capacity/conformance.py` (provider ABI suite). Config lives in
`pyproject.toml` (`testpaths = ["tests"]`, `asyncio_mode = "auto"`); dev
dependencies include `pytest`, `pytest-asyncio`, and `aiosqlite`. SQLite tests
exercise SQLAlchemy persistence, but do not establish PostgreSQL locking or
transaction semantics.

## Running

Build the dashboard first: Python tests check that the plane serves
`dashboard/dist/index.html`.

```bash
npm --prefix dashboard ci
npm --prefix dashboard run build

uv run pytest tests -q             # full Python selection; inspect PostgreSQL skips
uv run pytest tests/test_cli.py -q # one file
uv run ruff check eyvan tests      # lint (line-length 121, target py311)
npm --prefix dashboard test
```

PostgreSQL tests use `EYVAN_TEST_POSTGRES_URL`, or start a disposable cluster with
`initdb`/`pg_ctl` from `EYVAN_TEST_POSTGRES_BINDIR` or `PATH`. Without either they
skip. To require PostgreSQL, select a disposable test database and run:

```bash
EYVAN_REQUIRE_POSTGRES=1 EYVAN_TEST_POSTGRES_URL=postgresql+asyncpg://eyvan@127.0.0.1:5432/eyvan_test \
  uv run pytest tests -q -rs
```

Never use a production database. `EYVAN_REQUIRE_POSTGRES=1` turns unavailable
PostgreSQL into a failure instead of a skip; memory/SQLite results alone do not
qualify PostgreSQL behavior.

## The rules the tests enforce

- **Provider boundary**: `tests/test_provider_boundary.py` fails if anything that
  decides (scheduler, planner, durable model) names a provider. This is the
  repo's central architectural rule, and it is a test, not a convention.
- **Tests assert public behavior**: state transitions, API outputs, parity — not
  private helper calls or log strings.
- **Diagnostics travel with failures**: `GcsCommandError` carries command + stderr;
  the agent's error taxonomy distinguishes provider conclusions from inconclusive
  agent failures (Part 10).
- **Grounded docs**: `tests/test_docs_are_grounded.py` keeps documentation honest
  against the code.

## What the suites cover (selected, all in `tests/`)

- **Lifecycle and scheduling**: `test_domain.py`, `test_models.py`, `test_dag.py`,
  `test_dag_waves.py`, `test_groups.py`, `test_groups_atomic_placement.py`,
  `test_gang_preemption_is_atomic.py`, `test_elastic_gang_worlds.py`,
  `test_heterogeneous_group_placement.py`, `test_cold_gang_initialization.py`.
- **Control plane**: `test_controller.py`, `test_housekeeping_phases_actually_run.py`,
  `test_outbox_claim_ordering.py`, `test_outbox_retry_pacing.py`,
  `test_persistence_leader.py`, `test_local_persistence.py`,
  `test_fleet_invariants.py`, `test_invariants_are_exported.py`,
  `test_chaos.py`, `test_chaos_on_postgres.py`.
- **Postgres-specific**: files ending in `_postgres.py`
  (`test_artifact_lifecycle_postgres.py`, `test_batch_backpressure_postgres.py`,
  `test_chaos_on_postgres.py`, …) run against a disposable PostgreSQL
  (`postgres_fixture.py`).
- **Capacity and providers**: `test_capacity_planning.py`,
  `test_capacity_ownership.py`, `test_fleet_state.py`,
  `test_gcs_rest_fleet_state.py`, `test_aws_capacity.py`, `test_aws_signing.py`,
  `test_azure_capacity.py`, `test_gcp_driver.py`, `test_gcp_compute_driver.py`,
  `test_kubernetes_capacity.py`, `test_kueue_*`, `test_hardware_catalog.py`,
  `test_autoscale*.py`, `test_local_capacity_driver.py`,
  `test_manual_capacity_driver.py`.
- **SDK/CLI**: `test_client.py`, `test_cli.py`, `test_callable_execution.py`
  (includes the concurrent-identical-submission pins), `test_contexts.py`,
  `test_agent_cli.py`, `test_head_commands.py`, `test_operations_commands.py`.
- **Auth and security**: `test_oidc_authentication.py`, `test_local_auth.py`,
  `test_authorization_matrix.py`, `test_bootstrap_identity.py`,
  `test_identity_mtls.py`, `test_identity_rotation.py`,
  `test_metrics_endpoint_is_not_public.py`, `test_artifact_tenant_isolation.py`.
- **Artifacts**: `test_managed_artifacts.py`, `test_artifact_holds.py`,
  `test_artifact_inventory.py`, `test_artifact_lifecycle*.py`,
  `test_artifact_collection.py`.
- **API integrity**: `test_api.py`, `test_every_route_is_reachable.py`,
  `test_dashboard_calls_routes_that_exist.py` — the dashboard's client cannot
  drift from the server's routes unnoticed.
- **Packaging/deployment**: `test_packaging.py`, `test_helm_chart.py`,
  `test_deployment_manifests.py`, `test_docker_build_context.py`,
  `test_oci_release.py`, `test_docs_are_grounded.py`.

## Dashboard tests

Vitest, colocated with source: `routing.test.ts`, `eventStream.test.ts`,
`useLiveRefresh.test.tsx`, `format.test.ts`, `accelerator-request.test.ts`,
`oidc.test.ts`, `AuthGate.dom.test.tsx`, `submit.dom.test.tsx`, and the page-level
DOM suites (`workload-rows`, `panels`, `logs`, `capacity`, `billing`,
`fleet-lifecycle`, `fleet-actions`, `overview.spend`, `authorization`). Default
environment is Node; DOM suites opt in with a `@vitest-environment jsdom` comment.

## Qualification vs tests

Host and disposable-Postgres suites exercise the core with fake providers; they
are **not** cloud qualification. The conformance suite (`capacity/conformance.py`)
qualifies the provider contract against a harness; real infrastructure needs a
recorded spend and an immutable exact-candidate receipt (Part 14, proofs).
