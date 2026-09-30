# ADR-0006: Normalize and Fingerprint Selected Source Files

**Status:** Accepted

## Context

A committed repository snapshot can contain files that are irrelevant, unsafe to
interpret as source text, or unstable across platforms. Summary planning needs one
canonical set of inputs independent of filesystem location and Git output order.

## Decision

Apply configured include patterns before excludes to normalized repository-relative
POSIX paths. Accept only regular UTF-8 text blobs within the configured size limit;
reject selected symlinks, submodules, Git LFS pointers, binary content, and unsafe
paths. Normalize line endings and terminal newlines, then sort files by path.

Fingerprint the selected source from stable selection settings, paths, and normalized
content hashes. Resolve explicit, non-empty logical summary units against that source;
a file may deliberately belong to more than one unit.

## Consequences

- Equivalent selected content produces the same identity across repository locations
  and platforms.
- Selection-rule or selected-content changes invalidate the relevant fingerprints.
- Unsupported entries outside the selected set do not prevent inspection.
- Unit configuration remains curated and must be updated as repository architecture
  changes.
