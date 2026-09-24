# ADR-0003: Generate Explicit Logical Summary Units

**Status:** Accepted

## Context

One LLM call per Python package could improve technical detail, but package size and
value vary significantly. Literal package discovery would create unnecessary calls
for trivial packages and make corpus cost difficult to control.

## Decision

Configure explicit logical summary units such as splitters, embeddings, vector
stores, API, or evaluation. Use focused prompts and validated structured output for
each unit, then render Markdown in application code. Begin with a balanced profile
containing project-level documents and selected high-value units.

## Consequences

- Corpus scope and LLM cost are deliberate and reviewable.
- Structured summaries provide consistent sections for validation and retrieval.
- Configuration requires some curation as repository architecture changes.
- Automatic unit discovery and minimal or deep generation profiles remain possible
  later.
