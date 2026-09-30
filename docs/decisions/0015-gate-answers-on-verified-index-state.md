# ADR-0015: Gate Answers on Verified Index State

**Status:** Accepted

## Context

An index may exist while representing an older corpus, a failed rebuild, or an
incomplete external collection. Answering from that state would make provenance and
freshness claims unreliable.

## Decision

Derive one typed index status in the service layer. A manifest-backed index is
available only when its latest successful build matches the currently published
corpus, no newer attempt is building or failed, the deterministic collection exists,
and its count equals the persisted successful chunk count. Keep collection names
derived from source identity and index-affecting configuration; a corpus change makes
the collection stale rather than selecting or deleting another collection.

Gate answering before creating a RAG run, retrieving, or invoking an LLM. Report an
actionable unavailable reason and require an explicit rebuild. For generic sources,
report freshness as `untracked` and retain collection existence as the compatibility
rule.

## Consequences

- Publishing a different corpus immediately makes its existing index stale.
- SQL history, current manifest identity, and vector-store state must agree before
  Portfolio answers run.
- Failed, building, missing, and count-mismatched indexes fail closed.
- Automatic rebuilds and historical collection retention remain out of scope.
