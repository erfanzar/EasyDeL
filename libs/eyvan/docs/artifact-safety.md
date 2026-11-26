# Managed artifact tenancy, migration, and inventory safety

Status: M1.6 lifecycle and local-schema-v2 implementation contract. This page
does **not** authorize a live migration, object deletion, provider rollout,
or recovery action without a rehearsed maintenance window and matching
database/object-store backups.

## Implemented safety contract

1. The durable identity of an artifact is `(org_id, project_id, uri)`.
   Managed rows, references, holds, usage, service reads, and API
   authorization use the complete scope.
2. Lifecycle transitions are `PENDING_UPLOAD -> ACTIVE -> DELETING ->
   DELETED`, with `QUARANTINED` for legacy, incomplete, or contradicted
   evidence. Only `ACTIVE` rows may be read, held, or referenced.
3. Upload reserves quota and a logical record generation in a transaction,
   writes outside the transaction, then activates only after URI, size,
   backend-observed SHA-256, and exact provider object version all match the
   reservation. Failed or uncertain writes remain charged until exact
   cleanup or recovery proves an outcome.
4. Deletion first commits a claim that revalidates retention, holds, and
   references. Physical deletion happens outside the transaction and is
   conditional on the recorded provider version. Finalization revalidates
   the claim and generation; an uncertain provider result is retryable, not
   success.
5. Physical inventory is bounded, scope-explicit, current-object-only, and
   report-only by default. Reconciliation exact-reads each object before
   activating a row. It is additive and never treats a listing, size, or URI
   alone as proof for deletion. GCS orders filenames differently from logical
   URIs when result IDs have punctuation or prefix relationships. Each public
   GCS inventory call therefore scans the exact live scope with at most 32
   provider requests, each limited to 1,000 objects and a 2 MiB response, retaining
   only the smallest `limit+1` logical candidates after the cursor. It fails
   inconclusively if that budget is exhausted or provider pages do not advance;
   no partial result is returned. Repeated public pages repeat this bounded
   scan, rather than translating the logical cursor to a physical filename.
6. Recovery scans lifecycle rows and artifact-owned scopes with explicit
   caps. It resumes stranded uploads/deletions, quarantines contradictions,
   and reports when its finite scan was capped. Destructive collection
   remains disabled unless `EYVAN_ARTIFACT_DESTRUCTIVE_COLLECTION_ENABLED`
   is explicitly set.
7. The local backend performs deletion through a per-object,
   descriptor-bound generation transaction beneath the trusted artifact
   root. It never pathname-unlinks a generation while validation/deletion is
   active; bounded crash receipts are durably retired only after
   cleanup-safe completion, while unvalidated payload evidence is retained.
   Fresh local roots use the versioned `layout-v2` control namespace with at most
   16 permanent mutation stripe locks, shared across processes. These lock files
   are never unlinked by collection. Successful mutations retire an empty
   per-object generation directory only after closing its descriptor, while
   holding the stripe, and fsync the parent. Nonempty directories (including
   unknown files, anchors, and incomplete receipts) remain untouched. Legacy
   lock layouts are refused, not silently migrated; see the maintenance boundary
   below.
   GCS uses generation preconditions for create, exact read, and delete,
   with bounded subprocess timeouts.
8. Provider or filesystem ambiguity raises an unavailable/inconclusive
   result (HTTP 503 at the service boundary). It is never converted into a
   not-found, conflict, or successful deletion by parsing diagnostics.

## The client-side store: digest addressing and the no-clobber race

The SDK's `GCSArtifactStore` writes callable/workspace payloads to
**digest-named** objects — `gs://…/callables/<sha256>.pkl` — describe-checks
first, then uploads with `--no-clobber`. Two consequences:

- A pre-existing object under a digest name is **fetched and re-hashed**; a
  corrupted or foreign object is rejected, never trusted.
- A create-precondition loser (`HTTPError 412` from `--no-clobber`) fetches
  the existing object and verifies its digest. Submission proceeds only on an
  exact match; a foreign or corrupt object is refused. This invariant is pinned by
  `test_a_concurrent_upload_of_the_same_digest_wins_instead_of_crashing`,
  `test_a_precondition_loss_still_rejects_a_foreign_object`, and
  `test_any_other_upload_failure_still_names_itself` in
  `tests/test_callable_execution.py`.

CLI failures still travel with the CLI's own words (`GcsCommandError`
carries the failing command and its stderr) — a swallowed diagnostic turns
every upload failure into a mystery.

## Qualification boundary

The lifecycle, migration, race, and recovery paths have host and
disposable-PostgreSQL tests. Local-filesystem adversary tests and mocked GCS
command tests are not qualification of every production filesystem, GCS IAM
policy, bucket versioning/soft-delete policy, visibility delay, timeout
mode, or live concurrent rollout. Run the immutable release candidate
against the approved PostgreSQL, object-store, Kubernetes, backup/restore,
and rollback matrix before enabling destructive collection.

Do not describe `collectable` or inventory output as a deletion claim.
Missing bytes, physical-only objects, and size discrepancies remain evidence
to reconcile, not permission to delete. Recovery is exact lifecycle replay,
additive metadata repair, a forward code/database repair, or restoration
from matching snapshots — not mutation based on inference.

## Local lock-layout maintenance boundary

The bounded stripe protocol is initialized automatically **only for an empty
local artifact root**. A populated root without `.locks/layout-v2`, an
unversioned `.locks` directory (even if empty), or unexpected entries in the
versioned lock directory is refused with a quiesced-maintenance error. Existing
operator files and legacy per-object lock files are preserved. An interrupted
initialization before the version directory is durable is likewise refused.

Do not create the version marker over legacy locks, remove live locks, or run
old and new local-store binaries against the same root. That would split the
cross-process flock domain; a marker alone is not a safe migration. All old
artifact readers/writers and controllers must be stopped before any cutover,
and remain stopped until an explicitly reviewed migration/restore procedure
and matching backups have been validated. This repair supplies no automatic
cutover or operator-data cleanup command and does not authorize one. Existing
roots therefore need a separately approved, rehearsed maintenance procedure;
fresh disposable roots exercise the new protocol without migrating data.

## Migrations and API transitions

Old clients receive **426** on the scoped API surface — the transition is
forced, not silently widened. The 0012–0014 migration sequence is
maintenance-only cutover: schema first, verified inventory second, destructive
collection last, each behind its own rehearsal and backup gate. See
[design/0001](design/0001-scoped-artifact-lifecycle.md) for the record.
