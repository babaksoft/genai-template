# ADR-0004: Use Artifact Fingerprints for LLM Reproducibility

**Status:** Accepted

## Context

LLM output is not guaranteed to be byte-identical for identical prompts and model
parameters. Index freshness and evaluation provenance cannot assume deterministic
generation.

## Decision

Identify each boundary explicitly:

- snapshot revision by resolved commit SHA;
- selected source by normalized paths and content hashes;
- summary input by source, unit, prompt, and model settings;
- generated document by all generation inputs and output hash; and
- published corpus by its actual Markdown bytes and stable manifest fields.

Cache validated structured summaries by their input fingerprint. Use the published
corpus fingerprint as the authority for index freshness and evaluation provenance.

## Consequences

- Repeated unchanged work can reuse cached artifacts without another LLM call.
- Bypassing the cache may produce a different corpus, but the difference is visible
  through its fingerprint.
- Volatile fields such as timestamps must not participate in content fingerprints.
- Manifests and index-build records must retain the relevant fingerprints.
