# Portfolio RAG Stage 1 Reliability Remediation

**Status:** Proposed

## Goal

Produce one correct, usable Portfolio corpus for one configured project with either
Ollama Cloud or OpenAI, using exactly one provider call per component or project
summary and no workflow-level retries.

This remediation replaces the current structured-output generation path before
Stage 2 begins. Multi-project orchestration, ingestion, index freshness, evaluation,
and conversations remain out of scope.

## Decisions Fixed for This Work

- One command continues to process exactly one configured project and one immutable
  Git snapshot.
- Each component, overview, architecture, and testing/operations summary uses one
  provider call.
- Providers return plain Markdown text. Workflow completion no longer depends on an
  LLM producing Pydantic-compatible JSON.
- Application-owned prompts define an exact ordered Markdown section contract for
  each artifact kind. Application code parses the response into the existing typed
  summary models and renders the final corpus Markdown.
- The existing minimal v1 prompts are replaced in place. They are not retained as a
  legacy prompt family because they have not produced an accepted complete corpus.
- Invalid evidence paths are discarded and reported. A section is accepted when at
  least one cited path belongs to its allowed input scope.
- When a section has no valid cited path, the parser assigns that call's allowed
  evidence scope as an explicit fallback and records an
  `evidence_scope_fallback` warning. This preserves a usable corpus without
  pretending that the model supplied precise evidence.
- Missing, duplicate, or unexpected sections are recovered deterministically and
  recorded as warnings rather than causing a retry. Empty provider responses,
  refusals, truncation, transport failures, and provider failures remain hard
  failures.
- Provider SDK automatic retries and workflow semantic retries are disabled. One
  cache miss causes at most one provider request.
- The generation fingerprint remains the cache lookup identity. It continues to
  cover the immutable source, selected files, project/component configuration,
  prompt identity, provider/model, and content-affecting inference settings.
- The output hash remains in the cached artifact as an integrity check and as the
  dependency identity consumed by project summaries. It is not a component cache
  lookup key.
- The existing cache schema is changed in place. No cache schema version bump or
  compatibility path is required because implementation starts with an empty cache
  and no accepted published corpus.
- A valid cache hit avoids every provider call. Corrupt or mismatched cache entries
  continue to fail closed rather than becoming implicit misses.

## Response Contract

Every artifact kind owns a fixed ordered set of headings. A component response, for
example, follows this shape:

```markdown
## Responsibilities

Concise factual summary text.

### Evidence

- src/example.py

## Important Abstractions

Concise factual summary text.

### Evidence

- src/example.py
```

Prompts must include the complete response skeleton, the exact allowed evidence
paths, one short valid example, canonical source-file delimiters, and explicit
instructions to emit no preamble or epilogue. Repository text remains untrusted
evidence and cannot override the response contract.

The parser normalizes line endings, recognizes only configured headings, preserves
section prose as content, filters evidence against the exact allowed scope,
deduplicates and sorts retained paths, and produces typed warnings for every
recovery.

## Slice 1 — Markdown Contracts, Prompts, and Parser

**Status:** Complete

### Outcome

Provider text can be converted deterministically into the existing component and
project summary models without JSON extraction or a provider call.

### Implementation

- Replace the current v1 prompt definitions with complete Markdown response
  contracts for component, overview, architecture, and testing/operations
  summaries.
- Add immutable summary specifications mapping exact Markdown headings to existing
  summary fields.
- Implement a provider-neutral Markdown parser that handles canonical responses and
  deterministic recovery for missing, duplicate, reordered, or unknown sections.
- Preserve otherwise unassigned prose in the artifact's primary section and give a
  missing or empty section an explicit deterministic placeholder so formatting
  defects cannot make a non-empty provider response unparsable.
- Add typed generation warnings including:
  - `missing_section`;
  - `duplicate_section`;
  - `unexpected_section`;
  - `unassigned_content`;
  - `invalid_evidence_path`; and
  - `evidence_scope_fallback`.
- Filter invalid evidence paths while retaining every valid path. Apply the allowed
  call scope as the fallback only when a section otherwise has no valid evidence.
- Keep final Pydantic validation and the existing evidence-scope validation after
  parsing.

### Verification

- Test every canonical artifact response.
- Test mixed valid and invalid paths, all-invalid paths, absent evidence blocks,
  duplicate headings, reordered headings, missing headings, preambles, epilogues,
  and empty section bodies.
- Test stable parsed output and warning order across repeated runs.
- Test prompt hashes, complete skeletons, exact allowed-path catalogs, source
  delimiters, and untrusted repository instructions.

### Acceptance

- Every non-empty fixture response produces a validated typed summary plus any
  applicable recovery warnings.
- Formatting recovery never invokes a provider or retry.
- The old minimal prompt text is no longer selectable.

## Slice 2 — Native Plain-Text Provider Boundary

**Status:** Complete

### Outcome

Ollama Cloud and OpenAI expose the same plain-text generation contract without
LlamaIndex output parsing.

### Implementation

- Replace `StructuredSummaryGenerator` with a provider-neutral text-generation port
  returning response text, provider/model identity, token usage, and safe audit
  metadata.
- Implement the OpenAI adapter with the official SDK Responses API and plain text
  output.
- Implement the Ollama adapter with the official Python SDK chat API and plain text
  output, supporting the configured Cloud base URL.
- Add `ollama` as a direct project dependency instead of relying on a transitive
  LlamaIndex dependency.
- Disable provider SDK automatic retries where supported and add no workflow retry
  loop.
- Reject empty text, refusal, incomplete/truncated completion, identity mismatch,
  and provider transport failures with stable provider-neutral reasons.
- Preserve current token accounting and safe request/finish metadata where the
  provider exposes them.

### Verification

- Unit-test both adapters with mocked SDK responses for success, refusal,
  truncation, empty output, usage variations, timeout, and provider error.
- Prove one adapter invocation makes exactly one SDK request.
- Prove no adapter adds JSON-schema instructions, Markdown extraction, or repair
  retries.

### Acceptance

- Both providers satisfy the same text-generation port.
- A cache miss can cause no more than one provider request.
- LlamaIndex is absent from the Portfolio generation boundary.

## Slice 3 — Component Generation and Cache Reuse

**Status:** Complete

### Outcome

Each configured component is generated with one text call, parsed into the existing
domain model, validated, cached, and reused without another provider call.

### Implementation

- Assemble the component prompt from the new v1 contract and exact planned files.
- On a cache miss, call the text generator once, parse the Markdown, apply evidence
  filtering/fallback, validate the resulting `ComponentSummary`, and cache it.
- Store generation warnings with the component artifact so later cache hits retain
  the original audit result.
- Continue calculating the component output hash from canonical validated summary
  JSON.
- Keep cache-hit behavior and generation fingerprints unchanged except for the new
  prompt hashes and generation-boundary contract.

### Verification

- Test first-run generation, write, hit, and repeated-hit behavior with a counting
  text generator.
- Test selective invalidation after source, unit, prompt, model, or inference
  changes.
- Test that invalid evidence is reported but does not discard otherwise usable
  component content.
- Test that provider failures and empty responses are not cached.

### Acceptance

- Every component requires at most one provider call.
- A valid unchanged component cache entry causes zero provider calls.
- Cached component output remains fully validated and integrity-checked.

## Slice 4 — Project Synthesis, Rendering, and Publication

**Status:** Complete

### Outcome

Cached or newly generated components produce all three project summaries and one
complete atomically published corpus.

### Implementation

- Apply the same one-call Markdown contract and tolerant parser to overview,
  architecture, and testing/operations synthesis.
- Continue including ordered component output hashes in project generation
  fingerprints because project prompts consume the actual component summaries.
- Preserve application-owned deterministic Markdown rendering, filenames, manifest
  construction, whole-corpus validation, immutable releases, and atomic pointer
  switching.
- Include generation-warning summaries in the run report while keeping volatile
  warning counts out of corpus identity unless they alter validated artifact
  content.

### Verification

- Test project cache hits and invalidation after one component output changes.
- Test all parser warning paths for each project artifact kind.
- Re-run renderer, manifest, corpus-validation, publisher, workflow, and CLI tests.
- Prove a failed provider call leaves the previous publication untouched.

### Acceptance

- A successful run publishes the complete configured filename set and a valid
  manifest.
- A fully unchanged rerun makes zero provider calls.
- Warning-bearing but recoverable responses remain inspectable and publishable.

## Slice 5 — Cache Envelope and Reporting Update

**Status:** Complete

### Outcome

The existing cache format records text-generation audit information and recovery
warnings without retaining compatibility code for unused structured-output cache
entries.

### Implementation

- Overwrite the current cache-envelope model and serializer in place; retain the
  existing cache schema version value.
- Store the validated structured summary, canonical output hash, original token
  usage/cost, provider metadata, raw response hash, and ordered generation warnings.
- Do not store raw provider prose separately once it has been parsed; the response
  hash is sufficient to correlate diagnostics without duplicating generated text.
- Preserve strict canonical encoding, full read validation, and atomic writes.
- Extend CLI text and JSON reports with warning counts grouped by artifact and code.
- Keep cache hits non-billable and carry original warnings and usage into artifact
  audit reporting.

### Verification

- Test canonical envelope round trips with and without warnings.
- Test response-hash, output-hash, fingerprint, artifact-kind, and canonical-byte
  corruption.
- Test warning persistence and stable report rendering on cache hits.
- Start tests with an empty cache; add no legacy-cache fixture or migration test.

### Acceptance

- Cache lookup remains based on generation fingerprint rather than output content.
- Output hashes continue to detect corruption and identify project dependencies.
- No cache schema migration or backward-compatibility branch exists.

## Slice 6 — Real-Provider Completion Gate

**Status:** Complete

### Outcome

The postponed Stage 2 work can resume only after one complete corpus has been
published through each supported provider path.

### Implementation

- Add opt-in integration tests for one representative component and project prompt
  against Ollama Cloud and OpenAI.
- Run the complete one-project workflow from an empty cache with Ollama Cloud.
- Clear the cache and run the complete workflow with OpenAI.
- Inspect warning reports and improve the shared prompt contracts when warnings
  reveal systematic formatting or citation problems; do not add retries.
- Run each unchanged configuration again and verify zero provider calls through
  cache-hit reporting.

### Verification

Run the complete local quality gate:

```bash
PHOENIX_ENABLED=false ./scripts/check.sh
```

Then run the explicitly configured provider integration commands and preserve their
JSON run reports as implementation evidence without committing credentials or raw
source prompts.

### Acceptance

- Ollama Cloud publishes one complete, validated corpus from an empty cache.
- OpenAI publishes one complete, validated corpus from an empty cache.
- Neither successful first run performs a workflow retry.
- Each unchanged second run makes zero provider calls.
- Stage 2 remains postponed until all four conditions hold.

## Completion Criteria

This remediation is complete when:

- Portfolio generation no longer depends on model-authored JSON;
- every component and project artifact uses at most one provider call;
- invalid evidence is filtered, preserved as audit warnings, and never admitted as
  validated evidence;
- generation fingerprints, output integrity hashes, dependency invalidation, and
  cache reuse remain intact;
- one full corpus has been generated and published with Ollama Cloud;
- one full corpus has been generated and published with OpenAI; and
- the full local quality gate passes.
