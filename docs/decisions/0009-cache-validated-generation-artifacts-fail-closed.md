# ADR-0009: Cache Validated Generation Artifacts Fail Closed

**Status:** Accepted

## Context

Provider output is nondeterministic and billable. Repeated runs must reuse completed
work without allowing stale or corrupt cache data to alter a corpus or silently cause
another provider request.

## Decision

Use the generation fingerprint as the cache lookup identity and atomically store a
canonical envelope only after parsing and validation. The envelope retains stable
provenance, the typed summary, its output hash, provider usage metadata, response
hash, and recovery warnings. Project-summary fingerprints also include the ordered
output hashes of the component artifacts they consume.

Validate the key, artifact kind, schemas, canonical encoding, provenance, and output
hash on every read. Treat an absent entry as a miss, but fail on a corrupt or
mismatched entry instead of regenerating implicitly.

## Consequences

- An unchanged valid cache entry causes no provider request.
- A changed component invalidates dependent project summaries through its output
  hash.
- Cache corruption is visible and cannot create an unplanned billable call.
- Cache format changes may require explicit migration or deliberate cache disposal.
