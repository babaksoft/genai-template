# ADR-0008: Use Single-Call Markdown Generation with Deterministic Recovery

**Status:** Accepted

## Context

Provider prose must be converted into a predictable corpus while keeping request
counts, recovery behavior, and evidence handling explicit.

## Decision

Use a provider-neutral plain-text generation port and make at most one provider
request for each component or project artifact on a cache miss. Prompts define an
exact ordered Markdown section and evidence contract and treat repository text as
untrusted data. Application code parses the response into the existing typed summary
models and renders the published Markdown; provider SDK retries and workflow retries
are disabled.

Recover deterministically from missing, duplicate, reordered, or unexpected sections.
Discard and report out-of-scope evidence. When a section has no valid cited path,
assign the call's allowed evidence scope and record an `evidence_scope_fallback`
warning. Empty, refused, truncated, transport-failed, or provider-failed responses
remain hard failures.

## Consequences

- A cache miss has a predictable upper bound of one billable provider request.
- Formatting defects can produce a usable, typed, auditable artifact without repair
  calls.
- Fallback evidence denotes the available input scope, not a model-supplied citation
  or proof that the source entails a claim.
- Providers must support complete plain-text responses and expose failures clearly.
