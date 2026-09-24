# ADR-0002: Use a Flat Manifest-Backed Portfolio Corpus

**Status:** Accepted

## Context

The existing text reader consumes Markdown and text files from one immediate corpus
directory. A portfolio corpus needs project and generation provenance, but recursive
ingestion is not required for the initial experiment.

## Decision

Publish generated Markdown directly under `data/portfolio` with deterministic,
project-prefixed filenames. Publish `manifest.json` beside them as the authoritative
mapping from each document to its project, component, repository revision, source
paths, generation inputs, and fingerprints.

Enrich documents from the manifest before splitting so metadata propagates to every
chunk.

## Consequences

- The initial experiment reuses the existing flat-directory ingestion behavior.
- Project-prefixed names prevent document and chunk identity collisions.
- Consumers must validate and read the manifest to obtain provenance.
- Logical hierarchy is represented by filenames and metadata rather than
  directories; nested corpus support can be reconsidered later.
