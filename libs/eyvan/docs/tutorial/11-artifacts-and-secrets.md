# Part 11 — Artifacts and secrets

Source: `eyvan/artifacts/` (`managed.py`, `_capability_tokens.py`,
`secret_capability.py`, `workload_capability.py`, `transfers.py`),
`eyvan/secrets.py`. Contract: `docs/artifact-safety.md`; design record:
`docs/design/0001-scoped-artifact-lifecycle.md`.

## The client-side store: digest addressing

The SDK stages callables and workspaces to **digest-named** objects — e.g.
`gs://…/callables/<sha256>.pkl` — describe-checking first, then uploading with
`--no-clobber`. Two consequences:

- A pre-existing object under a digest name is fetched and **re-hashed**; a
  corrupted or foreign object is rejected, never trusted.
- Identical callables deduplicate to one object. If a concurrent upload wins the
  race and `--no-clobber` returns `HTTPError 412`, the losing submitter fetches
  and verifies the stored digest before continuing. A mismatch is rejected.
  Tests in `tests/test_callable_execution.py` cover concurrent submissions.

CLI failures travel with their own words: `GcsCommandError` carries the failing
command and its stderr.

## The managed artifact lifecycle

`artifacts/managed.py` is the durable ledger. Key rules:

1. **Identity is scoped**: `(org_id, project_id, uri)`. Rows, references, holds,
   usage, and API authorization use the complete scope — a URI is not globally
   unique.
2. **States**: `PENDING_UPLOAD → ACTIVE → DELETING → DELETED`, with `QUARANTINED`
   for legacy/incomplete/contradicted evidence. Only `ACTIVE` rows may be read,
   held, or referenced.
3. **Uploads** reserve quota and a record generation in a transaction, write
   outside it, and activate only after URI, size, backend-observed SHA-256, and
   exact provider object version all match the reservation.
4. **Deletion** commits a claim that revalidates retention, holds, and references;
   physical deletion is conditional on the recorded provider version; an
   uncertain provider result is retryable, **never success**.
5. **Inventory** is bounded, scope-explicit, current-object-only, and
   report-only by default. A failed listing is `ManagedArtifactInventoryError` —
   a failed listing is never an empty scope.
6. **Ambiguity** raises an unavailable/inconclusive result (HTTP 503 at the
   service boundary) — never converted into not-found, conflict, or success by
   parsing diagnostics.

Kinds: `callables`, `results`, `workspaces`, `logs`. `callables` and `workspaces`
are content-addressed (object id = SHA-256 of stored bytes, verified on every
write/read). Logs are special: the managed copy is the only durable copy once
journald dies with the cell.

Destructive collection stays disabled unless
`EYVAN_ARTIFACT_DESTRUCTIVE_COLLECTION_ENABLED` is explicitly set. `eyvan
artifacts collectable` output is a report, not a deletion claim.

## Capability tokens

Artifact and secret access is granted through HMAC capability tokens
(`artifacts/_capability_tokens.py`), signed with `EYVAN_ARTIFACT_TOKEN_KEY`
(≥32 bytes, identical on every replica — a per-process key yields intermittent 403s
tracking replica count). Jobs receive capability **URLs** built from
`EYVAN_PUBLIC_URL`, so a workload can fetch its callable/inputs and write its
result without holding any plane credential.

Specific capability surfaces:

- `workload_capability.py` — what one workload may read/write (its callable,
  workspace, inputs, result manifest).
- `secret_capability.py` — secret *references*: `secret_mounts` on a shape config
  are references resolved at dispatch into the attempt's environment
  (`secret_materialization.py` on the worker). The reference is persisted; the
  value never enters the record, the fleet config, or the repository.
- `transfers.py` — the content-transfer paths between plane and workers.

## Secret rotation

`POST /api/v1/secrets/rotate` (`server/api/app.py`) requires fleet-management
permission but returns **503**: deployment-wide signing credential rotation is
not implemented. It changes neither signing material nor the rotation ledger.
The missing shared credential distribution and startup recovery cannot be
replaced by minting a new key in one replica. `eyvan/identity/` contains low-level
rotation and mTLS helpers, not a working deployment-wide rotation procedure.

## Tenancy guarantees

- A callable may: use its leased hardware, read its inputs from the artifact store
  by digest, and write its result manifest.
- It may not: name other tenants' artifacts (the scoped identity forbids it) or
  hold plane credentials. The fleet's idle timeout governs eligible unleased
  surplus capacity, subject to demand, protection, and minimum-capacity policy;
  it is not a workload runtime or cost deadline. Active lease expiry/quarantine
  and fenced retirement are separate lifecycle controls. Use explicit workload
  cancellation to request termination of a valid leased workload.
- Old clients receive `426` on the scoped artifact API — the transition to scoped
  identity was forced, not silently widened.
