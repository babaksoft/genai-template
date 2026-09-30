# ADR-0014: Persist Destructive Index-Build Attempts

**Status:** Accepted

## Context

Rebuilding deletes and replaces a deterministic vector collection. Collection
existence alone cannot reveal whether a later rebuild failed after destroying a
previously usable index.

## Decision

Persist and commit every build attempt as `building` before resetting its collection.
Record the source, index configuration, collection, pinned corpus fingerprint,
status, timing, verified counts, and a bounded sanitized failure summary. Complete an
attempt once as `succeeded` or `failed`; a process death may conservatively leave it
`building`.

Serialize same-process rebuilds with a non-waiting per-collection lock. Allow a later
explicit rebuild to create a newer attempt; do not use persisted `building` rows as
distributed locks or leases.

## Consequences

- Every destructive rebuild has a durable intent and outcome.
- A later failed or abandoned attempt cannot be hidden by an older success.
- Successful records identify the pinned corpus and verified stored chunk count.
- Crash recovery and cross-process locking remain operational work.
