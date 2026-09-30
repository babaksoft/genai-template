# ADR-0013: Use Portable Identities and Controlled Provenance

**Status:** Accepted

## Context

Reader paths and vector-store identifiers are machine- or backend-specific. They
cannot provide stable chunk identity or enough provenance to distinguish retrieved
documents from several projects.

## Decision

Use the globally unique project-prefixed filename as the Portfolio document ID and a
documented filename-plus-ordinal format as the chunk ID. Exclude absolute paths,
release locations, timestamps, and backend-native identifiers from both identities.

Attach only validated manifest metadata to documents and chunks: project identity,
document type and component, repository URL and resolved commit, corpus fingerprint,
generation fingerprint, filename, and Markdown header path. Preserve this provenance
through retrieval, model context, and citations; reject partial or malformed
Portfolio metadata.

## Consequences

- Document and chunk identities are stable across machines and vector backends.
- Cross-project context and citations remain distinguishable and auditable.
- Splitters and stores must preserve the controlled metadata contract.
- Generic sources may omit the complete Portfolio provenance group.
