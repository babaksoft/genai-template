# ADR-0012: Pin and Validate Portfolio Releases at Ingestion

**Status:** Accepted

## Context

Atomic publication may switch the public corpus pointer while an index build is
reading. Generic directory ingestion also lacks the manifest validation needed for
Portfolio provenance.

## Decision

Resolve the Portfolio publication pointer once and use that pinned immutable release
for manifest validation and document loading. Validate canonical manifest bytes, the
safe flat file set, document sizes and hashes, project ownership, and the corpus
fingerprint without reopening source repositories.

Layer a manifest-aware Portfolio loader around `TextReader` rather than adding
Portfolio domain rules to the generic reader. Keep non-manifest Markdown and text
sources on the existing generic ingestion path.

## Consequences

- One build cannot combine files from different releases.
- Ingestion trusts validated published artifacts rather than requiring repository
  access.
- Portfolio-specific validation remains isolated from reusable file reading.
- Generic sources retain their existing ingestion behavior and have no manifest
  freshness guarantee.
