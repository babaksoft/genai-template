# ADR-0011: Use a Versioned Multi-Project Corpus Manifest

**Status:** Accepted

## Context

The original manifest describes one generated project, while one flat Portfolio
corpus must eventually contain documents from several projects with different
repositories and revisions.

## Decision

Introduce manifest schema v2 with an ordered project registry and an explicit owning
project on every document. Cover the ordered project and document projections plus
the Markdown byte hashes with one corpus fingerprint. Require globally unique,
project-prefixed filenames.

Record repository and source provenance per project. Record the canonical generation
configuration once for the corpus because it applies to every project.

Normalize v1 and v2 manifests into one immutable consumer model. New publications
use v2, while existing v1 releases remain readable without changing their bytes or
fingerprints.

## Consequences

- One flat corpus can represent several projects without ambiguous ownership.
- Projects retain distinct repository and source provenance while sharing generation
  settings.
- Consumers need explicit schema dispatch and normalization.
- Multi-project generation or release composition remains a separate concern.
- Per-project generation configurations can be added later if they provide enough
  value.
