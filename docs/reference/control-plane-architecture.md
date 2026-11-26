# Control-plane architecture tour

Eyvan has one application service shared by HTTP requests and controller loops.
The stable import remains `eyvan.server.service.ControlPlaneService`; callers
need not know which private module implements an operation.

```text
SDK / browser
     |
     v
HTTP API ----------------------> controller lifecycle wiring
     |                                    |
     v                                    v
ControlPlaneService <----------- controller loops
     |
     +-- transactional subsystem methods
     |       |
     |       +--> repository sessions --> durable records / outbox
     |       +--> domain decisions
     |
     +-- external artifact operations outside transaction boundaries

outbox delivery --> cell agent --> capacity provider / worker execution
```

## Entry points

- `eyvan/server/api/app.py` is the HTTP composition root. It authenticates
  requests, creates the application service, and wires the controller lifecycle.
- `eyvan/server/controller/loops.py` runs reconciliation phases. Controllers
  invoke application-service operations; the service does not import controller
  loops or HTTP transport.
- `eyvan/server/service/control_plane.py` retains the public facade,
  constructor, compatibility exports, and the small class-referencing lease
  helper. Its static mixin bases assemble the private implementations. There
  is no runtime method injection or generated dispatch table.
- `eyvan/server/service/_control_plane/shared.py` contains shared imports,
  constants and module-level helpers. It does not import the facade.
- `eyvan/server/persistence/base.py` defines the repository/session boundary.
- `eyvan/domain/scheduler.py` consumes domain snapshots and returns scheduling
  decisions. Revalidation and durable writes belong to the application and
  controller layers, not this pure decision function.

## Finding an operation

The private `_control_plane` package groups intact methods by subsystem:

- `eyvan/server/service/_control_plane/submission.py`: ordinary submission.
- `eyvan/server/service/_control_plane/callable_submission.py`: callable and
  pool admission support.
- `eyvan/server/service/_control_plane/workload_lifecycle.py`: cancellation,
  retry and related lifecycle actions.
- `eyvan/server/service/_control_plane/leases.py`: lease creation and groups.
- `eyvan/server/service/_control_plane/lease_settlement.py`: settlement helpers.
- `eyvan/server/service/_control_plane/fleets.py`: pool/fleet administration.
- `eyvan/server/service/_control_plane/tenancy.py`: tenants and durable grants.
- `eyvan/server/service/_control_plane/reporting.py`: usage, spend and related
  operator reports.
- Artifact admission, I/O, recovery, inventory and collection are separate
  modules within that same private package.

Subsystems call shared service methods through `self`, preserving the existing
transactional workflow. Import independence is not a claim that business
operations are unrelated: settlement still updates budgets and audit records.
The boundary prevents modules from importing one another or importing the public
facade back into their implementation.

## Rules enforced by checks

`.importlinter` enforces four contracts: implementations cannot import the
facade; subsystem modules are mutually independent; application services cannot
import API/controller layers; domain modules cannot import server/SDK layers.
CI runs the locked `lint-imports` tool. `tests/test_control_plane_subsystems.py`
checks the 50,000-byte module ceiling, duplicate method definitions, contract
coverage for every subsystem, and this tour's concrete file paths.

The extraction preserved all 192 original method ASTs, signatures and descriptor
types in a one-time comparison against the pre-extraction source. That comparison
is not a permanent method-body checksum gate: future behavior changes still need
their own regressions. The ordinary service, API, artifact, tenant and lease tests
remain the behavior checks. File size and import contracts alone do not prove
correctness, database isolation, failover, or hardware behavior.
