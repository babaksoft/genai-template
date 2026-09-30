# ADR-0007: Generate One Balanced Project Corpus per Run

**Status:** Accepted

## Context

Configuration supports several projects, but initial corpus generation needs a small,
auditable execution boundary and a predictable document set. Multi-project failure,
concurrency, and publication semantics are not yet required.

## Decision

Require each generation command to select exactly one configured project and one
immutable snapshot. The `balanced` profile produces an overview, architecture, and
testing/operations document plus one component document for every configured logical
summary unit. Name them `<project>--overview.md`, `<project>--architecture.md`,
`<project>--testing-operations.md`, and
`<project>--component--<unit-id>.md`.

## Consequences

- A run has one source identity, one corpus identity, and a bounded failure domain.
- The expected document set can be validated before publication.
- Several projects require separate commands and produce separate releases.
- Minimal, deep, and multi-project profiles remain future decisions.
