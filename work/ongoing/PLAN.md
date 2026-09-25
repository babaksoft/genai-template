# Portfolio RAG Stage 1 — Local Portfolio Corpus Workflow

**Status:** In Progress (Slices 1–3 complete)

## Goal

Generate, validate, and atomically publish one reproducible flat Markdown corpus
from the immutable local-Git snapshot delivered by Stage 0.

Stage 1 ends at `data/portfolio`: it does not ingest the generated documents,
rebuild a vector index, write SQL records, run evaluation trials, add GitHub
access, or schedule corpus refreshes.

## Decisions Fixed for This Stage

- One command processes exactly one configured project. The configuration may
  continue to describe multiple projects, but multi-project orchestration is
  deferred.
- The first generation profile is `balanced`: overview, architecture, and
  testing/operations documents plus one component document for every explicitly
  configured Stage 0 summary unit.
- Repository text is untrusted prompt data. Prompts delimit it as source material,
  instruct the model not to follow instructions found in it, and never execute
  repository code or tools.
- Structured generation has a dedicated port. The existing RAG
  `LanguageModel.generate()` contract remains unchanged because it cannot expose
  validated Pydantic output or provider usage metadata.
- Every generated artifact is cached after validation. Component keys include the
  Stage 0 unit input fingerprint; project-document keys also include the hashes of
  the validated component artifacts they consume.
- A generation fingerprint includes the relevant source or artifact identities,
  prompt-template identity and hash, structured-output schema version, model
  provider/name, and content-affecting inference settings. It excludes credentials,
  timeouts, absolute paths, timestamps, latency, current-run cache status, and
  pricing rates.
- Evidence validation proves exact path membership in the selected immutable
  snapshot and the artifact's actual generation input scope. It does not claim
  that a cited path semantically entails every model statement; factual quality is
  measured in later evaluation stages.
- Markdown is rendered by application code from validated structured summaries.
  Model-produced Markdown is not published directly.
- Published filenames are deterministic:
  - `<project>--overview.md`;
  - `<project>--architecture.md`;
  - `<project>--testing-operations.md`; and
  - `<project>--component--<unit-id>.md`.
- `manifest.json` is canonical JSON and contains no publication timestamp.
  Volatile execution metrics are returned in a separate run report.
- Atomic publication uses immutable release directories and an atomically replaced
  `data/portfolio` symlink. Previous releases are retained and are never deleted
  automatically.
- Provider-backed end-to-end tests are marked `integration`; the normal suite uses
  deterministic fakes and requires no network or model service.

## Proposed Package Layout

```text
src/genai_template/workflow/portfolio/
  artifacts/
    cache.py
    fingerprints.py
  corpus/
    manifest.py
    publisher.py
    renderers.py
    validation.py
  domain/
    generation.py
    manifest.py
    reports.py
  generation/
    prompts.py
    service.py
  ports/
    artifact_cache.py
    structured_generator.py
  workflow/
    events.py
    corpus.py
  cli/generate.py

tests/workflow/portfolio/
  artifacts/
  corpus/
  generation/
  workflow/
  cli/test_generate_corpus.py
```

Names may be consolidated during implementation when a module would otherwise be
trivial, but domain, provider, rendering, publication, and CLI responsibilities
must remain independently testable.

## Slice 1 — Generation Configuration and Artifact Contracts

**Status: Complete**

### Outcome

A caller can load a typed balanced-generation profile and calculate stable input
identities without invoking an LLM or writing files.

### Implementation

- Extend the Portfolio configuration additively with an optional generation
  section so the existing Stage 0 inspection profile remains valid.
- Add immutable Pydantic configuration for:
  - profile name (`balanced` only in Stage 1);
  - structured-generation provider and model;
  - content-affecting inference settings such as temperature and optional seed;
  - prompt and output-schema versions;
  - explicit repository-path patterns supplying context to each project-level
    document, plus per-call input limits;
  - cache and publication locations, resolved from `settings.REPO_ROOT`; and
  - explicit input/output token rates used only for estimated-cost reporting.
- Reject unknown fields, unsafe output paths, unsupported profiles/providers,
  negative token rates, and model settings that the selected provider cannot honor.
- Do not serialize API keys or other credentials into configuration, fingerprints,
  cache entries, reports, or manifests.
- Add immutable domain types for generation requests/results, token usage, artifact
  provenance, cached artifacts, rendered documents, corpus builds, and run reports.
- Define canonical JSON helpers and SHA-256 generation fingerprints. Keep component
  and project synthesis fingerprint construction separate so their dependencies are
  explicit and testable.
- Expand the checked-in local profile to a realistic balanced example with several
  high-value summary units while preserving its usefulness for Stage 0 inspection.

### Verification

- Test backward-compatible loading of the existing Stage 0 shape.
- Test valid balanced configuration and immutable models.
- Test unknown fields, invalid paths, unsupported inference settings, missing
  generation settings for the generation command, and malformed pricing data.
- Test that stable inputs produce the same fingerprint regardless of dictionary
  ordering, repository location, timeout, credentials, pricing, or timestamp.
- Test that source, unit configuration, prompt, schema, model, or inference changes
  alter the relevant generation fingerprint.

Run:

```bash
uv run pytest tests/workflow/portfolio/config \
  tests/workflow/portfolio/domain \
  tests/workflow/portfolio/artifacts/test_fingerprints.py -v
```

### Acceptance

- Generation inputs have a canonical, secret-free identity before any provider
  call.
- Stage 0 inspection continues to accept snapshot-only configurations.
- No LLM, cache, corpus, database, or vector-store operation occurs in this slice.

## Slice 2 — Structured Summary Generation and Evidence Validation

**Status: Complete**

### Outcome

Focused prompts produce typed component, architecture, overview, and
testing/operations summaries whose evidence paths are valid for their exact
snapshot scope.

### Implementation

- Define strict, versioned Pydantic output models for:
  - component responsibilities, important abstractions, behavior, constraints, and
    testing evidence;
  - architecture boundaries, dependencies, and principal flows;
  - project purpose, capabilities, entry points, and technology choices; and
  - testing strategy, local operation, configuration, observability, and known
    operational constraints.
- Give every output field a description and every model a complete `Attributes:`
  docstring.
- Represent evidence as normalized repository-relative paths, attached to the
  relevant structured sections rather than as unconstrained prose citations.
- Add versioned prompt definitions with stable identifiers and hashes. Assemble
  source files in canonical path order with explicit path/content boundaries and
  an instruction that repository content is evidence, not executable directions.
- Define a `StructuredSummaryGenerator` port returning the validated value, provider
  identity, input/output token counts when available, and provider metadata needed
  for auditing. Do not return or log credentials, prompts, source text, or hidden
  reasoning.
- Implement LlamaIndex-backed OpenAI and Ollama adapters behind that port. Normalize
  provider validation and usage failures into focused workflow exceptions.
- Add evidence validators:
  - component evidence must be an exact member of that component's planned files;
  - project-document evidence must occur in the component evidence or selected
    repository context actually supplied for that synthesis call; and
  - paths must be non-empty, normalized, deduplicated, and deterministically sorted.
- Treat absent provider token usage as `unknown`, not zero. Estimate cost only when
  both usage and configured rates are available.

### Verification

- Use a deterministic fake generator to test each prompt request and structured
  result without network access.
- Test prompt IDs/hashes, canonical file ordering, input delimiters, and untrusted
  source instructions.
- Test invalid structured output, missing evidence, paths outside the snapshot,
  evidence from another component, duplicate/non-normalized paths, and provider
  failures.
- Test complete, partial, and unavailable token usage and cost calculations.
- Add provider smoke tests under `@pytest.mark.integration`; do not include them in
  the Stage 1 exit gate unless the relevant service and credentials are available.

Run:

```bash
uv run pytest tests/workflow/portfolio/generation -v
```

### Acceptance

- Downstream code receives validated domain values rather than model-authored
  Markdown or unchecked JSON.
- Every accepted evidence path belongs to the exact immutable input scope.
- Source and prompt contents do not appear in normal logs or custom trace
  attributes.

## Slice 3 — Validated Artifact Cache and Component Summaries

**Status: Complete**

### Outcome

Every configured logical unit can be summarized once and then reused by exact
generation fingerprint without another LLM call.

### Implementation

- Define an artifact-cache port with `get` and atomic `put` operations keyed by the
  full generation fingerprint.
- Implement a filesystem cache outside `data/portfolio`, using versioned canonical
  JSON envelopes containing:
  - artifact kind and schema version;
  - generation fingerprint and all non-secret provenance;
  - validated structured output;
  - output hash; and
  - original provider usage and estimated cost, when available.
- Write cache entries through a temporary file followed by same-directory
  `os.replace`; never expose a partially written cache entry.
- On lookup, validate the envelope, expected artifact kind, fingerprint, output
  schema, and output hash. A corrupt or mismatched entry fails clearly instead of
  becoming an implicit cache miss and unexpected billable call.
- Implement component-summary generation in Stage 0 plan order:
  - calculate the generation fingerprint;
  - return a validated cache hit when present;
  - otherwise call the structured generator exactly once;
  - validate evidence and provenance; and
  - cache only the validated result.
- Keep cache-hit status and current-run latency in the run report, not in the cached
  artifact identity.

### Verification

- Test miss, write, hit, and repeated-hit behavior with a counting fake generator.
- Test that an unchanged second run makes zero generator calls.
- Test selective invalidation: changing one unit changes only that component's key;
  prompt/model/schema changes invalidate every affected artifact.
- Test truncated JSON, schema mismatch, key mismatch, altered output, wrong artifact
  kind, and failed atomic writes.
- Test that invalid model output is neither returned nor cached.
- Test that cache entries and logs contain no source text, prompt text, local
  repository path, or credential.

Run:

```bash
uv run pytest tests/workflow/portfolio/artifacts \
  tests/workflow/portfolio/generation/test_component_summaries.py -v
```

### Acceptance

- A validated cache hit is behaviorally equivalent to its original component
  result and causes no provider request.
- Cache corruption cannot silently change the published corpus or trigger an
  unplanned provider call.
- A failure before validation leaves no reusable artifact.

## Slice 4 — Project Synthesis and Deterministic Markdown Rendering

**Status: Planned**

### Outcome

Validated component artifacts become the balanced profile's project documents and
deterministic flat Markdown files.

### Implementation

- Implement cached architecture, overview, and testing/operations synthesis. Their
  input fingerprints include the snapshot identity, relevant prompt/model/schema
  settings, and ordered hashes of the validated component artifacts they consume.
- Resolve each project document's configured repository-context patterns against
  the snapshot, then provide synthesis with those files and bounded structured
  component summaries. Fail before a provider call when a context pattern is empty
  or configured input limits are exceeded; do not silently truncate a file or
  discard a component.
- Reuse the cache contract from Slice 3 for all project-level artifacts so a fully
  unchanged rerun makes no LLM calls.
- Add pure Markdown renderers with fixed heading order, whitespace, list ordering,
  evidence formatting, UTF-8/LF output, and exactly one terminal newline.
- Render the title and prose safely; model values cannot inject arbitrary front
  matter or control filenames.
- Calculate each document's SHA-256 hash from its actual UTF-8 Markdown bytes.
- Generate and validate the exact project-prefixed filename set for the balanced
  profile. Reject collisions before any publication work.

### Verification

- Add golden tests for all four renderer types.
- Test stable rendering across repeated runs and shuffled input collections.
- Test headings, escaping, line endings, terminal newline, deterministic evidence
  ordering, filename construction, and collision rejection.
- Test synthesis cache hits, dependency invalidation after one component output
  changes, and zero LLM calls on a fully unchanged rerun.
- Test explicit failure for over-budget synthesis inputs.

Run:

```bash
uv run pytest tests/workflow/portfolio/generation/test_project_synthesis.py \
  tests/workflow/portfolio/corpus/test_renderers.py -v
```

### Acceptance

- The balanced profile produces exactly three project documents plus one document
  per configured summary unit.
- Identical validated artifacts produce byte-identical Markdown.
- All published prose originates from validated structured fields and all evidence
  remains traceable to the snapshot.

## Slice 5 — Manifest, Corpus Validation, and Atomic Publication

**Status: Planned**

### Outcome

A complete staged corpus is validated as a unit, assigned a stable corpus
fingerprint, and made visible at `data/portfolio` in one atomic pointer switch.

### Implementation

- Define a versioned manifest model containing:
  - project slug/display name and repository URL when available;
  - requested ref, resolved commit SHA, and source fingerprint;
  - generation profile plus prompt, schema, provider/model, and configuration
    fingerprints;
  - one record per document with filename, document type, optional component ID,
    evidence paths, generation fingerprint, output artifact hash, Markdown content
    hash, and byte size; and
  - the resulting corpus fingerprint.
- Serialize `manifest.json` as sorted, compact UTF-8 JSON with one terminal newline.
  Omit timestamps, absolute paths, secrets, current-run latency, and current-run
  cache-hit flags.
- Define the corpus fingerprint over the ordered Markdown filenames and byte hashes
  plus a canonical manifest projection with `corpus_fingerprint` omitted. Document
  this non-circular calculation in code and tests.
- Before publication, validate the complete staged corpus:
  - exact expected file set and no nested files;
  - safe unique filenames;
  - manifest/document one-to-one correspondence;
  - byte size and content-hash agreement;
  - document type/component agreement;
  - evidence membership in the supplied snapshot; and
  - recomputed corpus fingerprint agreement.
- Build an immutable release in a temporary directory under
  `data/.portfolio-releases/`, then rename it to
  `data/.portfolio-releases/<corpus-fingerprint>` on the same filesystem.
- Publish through a temporary relative symlink followed by `os.replace` onto
  `data/portfolio`. Readers therefore observe either the previous complete release
  or the new complete release, never a partial directory.
- Refuse to replace an existing real file or directory at `data/portfolio`; only an
  absent target or a workflow-managed symlink is eligible for pointer switching.
- If a release fingerprint already exists, validate it byte-for-byte and reuse it;
  never overwrite conflicting contents.
- Retain superseded releases. Cleanup, retention, and deletion remain manual and
  out of scope.

### Verification

- Test canonical manifest bytes and golden manifest content.
- Test corpus fingerprint stability and changes caused by Markdown or stable
  provenance changes; confirm timestamps and run metrics cannot affect it.
- Test every corpus validation rule with focused malformed fixtures.
- Inject failures during generation, staging, release rename, and symlink switch;
  verify the previously published corpus remains readable and unchanged.
- Test first publication, replacement, same-release reuse, conflicting existing
  release, relative symlink correctness, and absence of partial files.
- Test that no old release is deleted.

Run:

```bash
uv run pytest tests/workflow/portfolio/corpus -v
```

### Acceptance

- `data/portfolio` always resolves to one fully validated immutable release.
- A failed run or publication attempt leaves the prior published corpus intact.
- The manifest contains enough stable provenance to reproduce or audit every
  generated document without exposing secrets or machine-local paths.

## Slice 6 — LlamaIndex Workflow, CLI, Metrics, and Stage Integration

**Status: Planned**

### Outcome

One command runs the complete Stage 1 lifecycle with typed workflow events and a
clear report of LLM work, cache reuse, latency, tokens, cost, and publication.

### Implementation

- Add typed LlamaIndex Workflow events and steps for:
  1. configuration and project selection;
  2. repository ref resolution and committed snapshot selection;
  3. logical summary planning;
  4. component artifact generation or cache reuse;
  5. project synthesis generation or cache reuse;
  6. deterministic rendering;
  7. manifest construction and full-corpus validation; and
  8. atomic publication and final reporting.
- Pass immutable domain objects through events. Use workflow context only for
  bounded run state and aggregation, not as durable artifact storage.
- Keep each side effect behind an injected port so workflow tests can use fake
  repository readers, generators, caches, clocks, and publishers.
- Execute generation sequentially in the initial implementation. Provider
  concurrency, retries, and rate-limit policy remain future optimizations; caching
  already makes command retry safe after completed artifacts.
- Emit observability spans around workflow steps and provider calls with IDs,
  fingerprints, counts, durations, and cache status. Do not attach source content,
  rendered prompts, generated prose, credentials, or full structured outputs.
- Produce a final typed run report with:
  - commit, source, corpus, and generation identities;
  - per-artifact cache hit/miss, token usage, provider latency, and estimated cost;
  - current-run provider-call count and billed-token/cost totals;
  - end-to-end and per-step latency; and
  - published path and release path.
- Add a CLI requiring `--config` and `--project`. Return non-zero status for config,
  snapshot, generation, cache, validation, or publication failures. Print no source
  or prompt contents.
- Document the command, provider prerequisites, cache behavior, publication layout,
  reproducibility boundaries, and manual release cleanup.

Proposed command:

```bash
uv run python -m genai_template.workflow.portfolio.cli.generate \
  --config src/genai_template/workflow/portfolio/config/profiles/local.yml \
  --project genai-template
```

### Verification

- Unit-test event routing and every step with injected deterministic fakes.
- Run an end-to-end temporary-Git test through the real workflow, filesystem cache,
  renderers, manifest validator, publisher, and CLI with only the generator faked.
- Run that test twice and assert the second execution makes zero generator calls,
  publishes the same corpus fingerprint, and reports all artifacts as cache hits.
- Inject a failure in every workflow phase and verify no incomplete corpus is
  published and the prior release remains unchanged.
- Test stable CLI output fields, JSON report mode, project selection, exit codes,
  and redaction boundaries.
- Run the complete Stage 0 and Stage 1 tests, followed by the non-integration quality
  suite.

Run:

```bash
uv run pytest tests/workflow/portfolio -v
./scripts/check.sh
```

### Acceptance

- The first successful fake-provider end-to-end run publishes a valid corpus and
  manifest for one committed repository snapshot.
- Repeating unchanged inputs makes zero LLM calls and publishes byte-identical
  Markdown and manifest content.
- A failed run leaves the prior corpus intact.
- Reports distinguish original artifact usage from current-run billed usage and do
  not claim zero usage when a provider omitted usage metadata.
- Every accepted evidence path exists in the exact selected snapshot.
- `./scripts/check.sh` passes.

## Stage 1 Completion Criteria

Stage 1 is complete only when all six slices are implemented and verified, and:

- the master plan's Stage 1 deliverables and exit criteria are satisfied;
- the checked-in balanced profile produces project overview, architecture,
  testing/operations, and selected high-value component documents;
- source, prompt, model, schema, artifact, document, and corpus identities are
  individually auditable;
- unchanged work is served entirely from validated cache artifacts;
- publication cannot expose a partial corpus or destroy the previous release;
- source text, prompts, generated prose, credentials, and secrets are absent from
  routine logs and custom trace attributes;
- all new functions, classes, and methods have complete type hints and Google-style
  docstrings; and
- all normal quality checks pass without requiring an external model service.

## Explicitly Deferred

- minimal and deep generation profiles;
- processing several configured projects in one workflow run;
- GitHub repository/archive readers and authentication;
- automatic scheduling, concurrency locks, retries, and provider rate limiting;
- automatic cache or release retention and cleanup;
- automatic index rebuilds and manifest-aware ingestion;
- SQL corpus-build history;
- evaluation datasets and trials;
- semantic verification that evidence entails each generated claim; and
- agentic repository exploration, tool execution, or dirty-working-tree inputs.

## Commit and Review Guidance

- Keep one reviewable commit per slice.
- Each slice must leave its focused tests passing and must not require an unfinished
  later slice.
- Preserve the user's existing `PLAN.md` edit that points Stage 1 to this file.
- Do not commit generated caches, release directories, the `data/portfolio` symlink,
  or provider-backed corpus output unless a later data-versioning decision
  explicitly adds them.
- Run `./scripts/check.sh` after Slice 6 before declaring Stage 1 complete.
