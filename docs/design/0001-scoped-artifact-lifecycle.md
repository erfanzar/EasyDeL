# DR-0001: Scoped managed-artifact identity and fenced lifecycle

**Status:** Implemented at the host/PostgreSQL boundary. Live migration,
object-store, and destructive-collection evidence still required — this
record is not authorization for a live cutover.

## Context

Artifacts (callables, workspaces, results) are shared, billed, and deleted
across tenants. The original single-namespace key made "which tenant owns
this object" a matter of trust in whoever wrote the row, and deletion a
matter of trust in a listing. Both were unacceptable once artifacts became
the dispatch path itself: a workload runs what an artifact row says.

## Decision

1. **Durable identity is the full scope** — `(org_id, project_id, uri)`.
   Every read, hold, reference, and authorization check uses the complete
   triple; no API answers in a single namespace.
2. **Generation-fenced lifecycle.** Rows move
   `PENDING_UPLOAD → ACTIVE → DELETING → DELETED`, with `QUARANTINED` for
   legacy, incomplete, or contradicted evidence. Upload activates a row only
   after URI, size, backend-observed SHA-256, and the exact provider object
   **version/generation** all match the reservation; deletion commits a
   claim, revalidates retention/holds/references, deletes physically
   conditional on the recorded generation, and finalizes by revalidating the
   claim. Uncertainty is retryable state, never success.
3. **Inventory is additive evidence.** Bounded, scope-explicit,
   current-object-only, report-only by default; reconciliation exact-reads
   an object before activating a row for it. A listing is never a deletion
   claim.
4. **Old clients receive 426** on the scoped surface: the transition is
   forced, not silently widened.

## Invariants

- Only `ACTIVE` rows may be read, held, or referenced.
- Provider/filesystem ambiguity raises unavailable (503) — it is never
  converted into not-found, conflict, or success by parsing diagnostics.
- The local backend deletes through a per-object, descriptor-bound
  generation transaction; it never pathname-unlinks a generation while
  validation or deletion is active.
- Destructive collection stays disabled until
  `EYVAN_ARTIFACT_DESTRUCTIVE_COLLECTION_ENABLED` is set, and only after the
  qualification matrix in [artifact-safety](../artifact-safety.md).

## Client-side corollary

The SDK store writes digest-named objects (`<sha256>.pkl`) with
describe-then-`--no-clobber`. On a create-precondition loss (HTTP 412), the
submitter fetches and re-hashes the existing object and continues only on an
exact digest match. Existence alone is neither proof of identity nor a reason
to reject an identical submission.

## Maintenance, rollout, and rollback

Migrations 0012–0014 are a maintenance-only cutover — schema first, verified
inventory second, destructive collection last — each behind a rehearsed
window with database and object-store backups. Rollback restores from
matching snapshots; recovery is exact lifecycle replay or additive repair,
never mutation by inference. Compatibility, rollout, and rollback details
for the historical sequence live with the migration code.
