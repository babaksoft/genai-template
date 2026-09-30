# ADR-0010: Publish Immutable Corpus Releases Atomically

**Status:** Accepted

## Context

Corpus readers must never observe a partial build, and a failed generation or
publication must not damage the last usable corpus. Published bytes also need stable,
machine-readable provenance.

## Decision

Build a canonical, timestamp-free `manifest.json` beside the deterministic Markdown
files and fingerprint the corpus from stable manifest fields and actual document
bytes. Validate the complete staged corpus before publication.

Store releases under `data/.portfolio-releases/<corpus-fingerprint>` and atomically
replace the relative `data/portfolio` symlink to select a release. Reuse an existing
release only when it validates and matches byte-for-byte. Never replace a real file
or directory at the publication path, overwrite conflicting release contents, or
delete superseded releases automatically.

## Consequences

- Readers see either the previous complete corpus or the new complete corpus.
- The manifest is the stable audit and ingestion contract for a release.
- Failed publication leaves the previous release available.
- Release retention and cleanup are manual operational responsibilities.
