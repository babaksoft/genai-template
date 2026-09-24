# Portfolio RAG Master Plan

**Status:** Proposed

## Objective

Build a reproducible Portfolio RAG experiment that derives a technical corpus from
active Python repositories, compares multiple RAG configurations against curated
evaluation data, and supports persistent multi-turn questions from portfolio
reviewers.

The first implementation will inspect one local Git repository at a committed
snapshot. The design will retain clear extension points for multiple projects,
remote GitHub snapshots, scheduled refreshes, deeper retrieval strategies, and a
reviewer-facing conversational UI.

## Current Foundation

The repository already provides:

- source-bound experiments and immutable registered RAG configurations;
- pluggable splitters, embedders, vector stores, retrievers, and language models;
- deterministic collection names derived from source and index configuration;
- explicit index rebuilding;
- single-turn question answering with inline citations and runtime metrics;
- Phoenix instrumentation across indexing, retrieval, and generation;
- a baseline retrieval evaluator with hit, recall, and precision at K; and
- a Streamlit interface for source, experiment, configuration, index, and answer
  operations.

The Portfolio RAG work must add reproducible repository snapshots, generated corpus
provenance, index freshness, durable evaluation trials, and multi-turn
conversations without weakening these existing boundaries.

## Architecture Decision Records

Implementation follows these accepted decisions:

- [ADR-0001: Read committed repository snapshots through a provider boundary](docs/decisions/0001-read-committed-repository-snapshots.md)
- [ADR-0002: Use a flat manifest-backed portfolio corpus](docs/decisions/0002-use-a-flat-manifest-backed-corpus.md)
- [ADR-0003: Generate explicit logical summary units](docs/decisions/0003-generate-explicit-logical-summary-units.md)
- [ADR-0004: Use artifact fingerprints for LLM reproducibility](docs/decisions/0004-use-artifact-fingerprints-for-llm-reproducibility.md)
- [ADR-0005: Keep corpus workflow scheduling external](docs/decisions/0005-keep-corpus-workflow-scheduling-external.md)

The ADRs own architectural rationale and consequences. This plan owns delivery
stages, verification, and execution scope.

## Target Flow

```text
Local Git repository at an immutable commit
                    │
                    ▼
       RepositorySnapshotReader
 resolve ref → select files → normalize → fingerprint
                    │
                    ▼
        PortfolioCorpusWorkflow
 plan units → summarize → synthesize → validate → publish
                    │
                    ▼
 data/portfolio/*.md + manifest.json + corpus fingerprint
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   RAG config A index    RAG config B index
          │                    │
          └─────────┬──────────┘
                    ▼
       Versioned evaluation trials
                    │
                    ▼
     Selected configuration and index
                    │
                    ▼
       Persistent Portfolio Q&A
```

## Stage 0 — Snapshot and Corpus Contract

**Goal:** Produce a verified, content-addressed repository snapshot and logical
summary plan without invoking an LLM or generating corpus documents.

Deliverables:

- typed configuration for projects, local Git sources, include/exclude rules, and
  logical summary units;
- canonical `RepositorySnapshot`, `SnapshotFile`, and summary-plan models;
- a repository-reader protocol with a `LocalGitSnapshotReader` implementation;
- ref-to-commit resolution and read-only access to committed tracked blobs;
- deterministic selection, text normalization, hashing, and summary-unit planning;
- a read-only inspection CLI reporting commit, selected files, fingerprints, and
  summary-unit membership; and
- focused tests proving that working-tree changes do not affect a local Git
  snapshot.

Detailed execution slices are maintained in
[`work/ongoing/PLAN.md`](work/ongoing/PLAN.md).

Exit criteria:

- the same committed inputs produce the same selected files and fingerprints;
- modified, staged, and untracked files are ignored;
- absolute local paths and filesystem timestamps do not affect fingerprints;
- unsafe paths, invalid patterns, binary or oversized inputs, missing refs, and
  empty summary units fail clearly; and
- the inspection command performs no network, LLM, corpus, database, or vector-store
  operations.

## Stage 1 — Local Portfolio Corpus Workflow

**Goal:** Generate and atomically publish a focused flat Markdown corpus from the
Stage 0 snapshot.

Deliverables:

- LlamaIndex Workflow events and steps for snapshot resolution, selection, planning,
  component summarization, project synthesis, validation, and publication;
- focused prompt templates and Pydantic structured outputs for component,
  architecture, overview, and testing/operations summaries;
- deterministic Markdown renderers and project-prefixed filenames;
- evidence-path validation against the exact repository snapshot;
- per-unit artifact caching keyed by source, prompt, model, and unit configuration;
- a stable `manifest.json` with project and generation provenance;
- temporary-directory generation followed by atomic publication to
  `data/portfolio`; and
- token, latency, cache, and estimated-cost reporting.

The first profile is `balanced`: project-level documents plus an explicitly selected
set of high-value component summaries. Minimal and deep profiles may be introduced
as later corpus variants.

Exit criteria:

- the first successful run produces a valid corpus and manifest for one repository;
- a failed run leaves the previously published corpus intact;
- rerunning unchanged inputs reuses cached summaries and makes no LLM calls;
- every evidence path exists in the selected snapshot;
- volatile values such as timestamps do not alter content fingerprints; and
- the output records enough provenance to reproduce or audit every document.

## Stage 2 — Flat Ingestion, Provenance, and Index Freshness

**Goal:** Index the generated corpus with project metadata while retaining the
existing flat-directory ingestion model.

Deliverables:

- manifest-aware metadata enrichment layered onto the existing text reader;
- chunk metadata for project, document type, component, repository, commit, corpus
  fingerprint, and generation fingerprint;
- collision-free document and chunk identities based on project-prefixed filenames;
- citation responses containing project and repository provenance;
- a minimal persistent `IndexBuild` record with collection, index fingerprint,
  corpus fingerprint, status, counts, timings, and failure details;
- stale-index detection comparing the current corpus manifest with the latest
  successful build; and
- API/UI handling that requires an explicit rebuild when an index is stale.

The current deterministic collection naming remains in place. Corpus changes rebuild
the selected collection instead of automatically creating and deleting historical
collections. `manifest.json` acts as the corpus-build record until scheduled and
historical corpus management justifies a dedicated database model.

Exit criteria:

- every indexed chunk carries correct project and generation provenance;
- multiple projects can coexist without filename or chunk-ID collisions;
- a changed corpus fingerprint makes an existing index unavailable for answering;
- a successful rebuild records and serves the current corpus fingerprint; and
- cited answers identify the generated document, project, and repository revision.

## Stage 3 — Evaluation Dataset and Trial Model

**Goal:** Create a versioned evaluation system that compares configurations against
the same corpus and evidence.

Deliverables:

- an expanded evaluation-case schema with stable case ID, project scope, category,
  answerability, expected evidence, required facts, and optional reference answer;
- curated cases covering project facts, architecture, interfaces, configuration,
  testing and operations, cross-project comparison, and unsupported questions;
- development and held-out test splits;
- an `EvaluationTrial` grouping per-question runs by experiment, corpus fingerprint,
  dataset fingerprint, RAG configuration, and code revision;
- persisted ranked retrieval artifacts and per-case metrics; and
- trial summaries and comparison reports.

Retrieval metrics should include hit, recall, precision, MRR or nDCG, latency, and
resource cost. Stable evidence identities replace basename-only matching.

Exit criteria:

- every reported score identifies its exact corpus, dataset, config, and code;
- trials can be repeated without creating duplicate Portfolio RAG experiments; and
- development results can be compared without exposing the held-out set during
  tuning.

## Stage 4 — Retrieval and Answer Optimization

**Goal:** Select a Portfolio RAG configuration through controlled comparisons rather
than ad hoc UI testing.

Candidate dimensions:

- Markdown versus sentence splitting and chunk parameters;
- embedding provider and model;
- retrieval depth;
- project metadata filtering;
- history-aware query rewriting;
- MMR, reranking, or hybrid lexical/vector retrieval; and
- prompt, context-budget, and language-model choices after retrieval is satisfactory.

Change one major dimension at a time, reuse indexes when index-affecting settings are
identical, and measure quality together with latency and cost. Add answer correctness,
faithfulness, citation precision/recall, relevance, and abstention evaluation only
after the retrieval baseline is dependable.

Exit criteria:

- one configuration is selected using development data;
- the selection is confirmed once against the held-out set; and
- the decision record explains quality, latency, cost, and operational tradeoffs.

## Stage 5 — Durable Multi-Turn Portfolio Q&A

**Goal:** Replace isolated questions with persistent reviewer conversations while
keeping retrieval and generation behavior auditable per turn.

Deliverables:

- `Conversation` and ordered `Message` persistence;
- conversations pinned to experiment, RAG configuration, and corpus fingerprint;
- a stateless `ChatPipeline` accepting the current turn and bounded history;
- history-aware standalone-query generation before retrieval;
- per-turn retrieval, context, response, citation, status, failure, and metric
  artifacts;
- conversation and message API endpoints; and
- backward-compatible support for the existing single-turn `/answer` endpoint.

The database is the durable source of conversation truth. Workflow context or model
memory may support execution but does not replace conversation persistence.

Exit criteria:

- follow-up questions retrieve evidence using relevant conversation context;
- conversations survive API and UI restarts;
- every assistant response remains traceable to its query, index, sources, and
  configuration; and
- switching configurations requires a new conversation or an explicit recorded
  boundary.

## Stage 6 — Reviewer UI and End-to-End Validation

**Goal:** Deliver a portfolio-review experience and validate the complete lifecycle.

Deliverables:

- Streamlit chat messages and chat input backed by conversation APIs;
- new/reset conversation controls and server-loaded history;
- visible project, corpus revision, and RAG configuration identity;
- expandable cited evidence with repository links;
- per-turn warnings and operational metrics;
- evaluation-trial comparison views; and
- conversational evaluation cases covering follow-ups, pronouns, project switching,
  corrections, and unsupported requests.

Exit criteria:

- a reviewer can conduct a cited multi-turn session about either project;
- operators can compare the session's configuration with evaluation evidence; and
- the full non-integration quality suite and applicable external-service integration
  tests pass.

## Cross-Cutting Verification

- Every implementation slice adds focused unit tests and complete Google-style
  docstrings.
- Every completed stage passes `./scripts/check.sh`.
- Integration tests run only when their external providers are available.
- Repository contents are always treated as untrusted data: the workflow does not
  execute inspected code or obey instructions found in repository text.
- Secrets remain environment-provided and are never written to corpus documents,
  manifests, logs, traces, or configuration snapshots.
- Corpus generation, indexing, evaluation, and chat spans must avoid custom
  attributes containing unnecessary source or prompt content.
- Existing collections and historical artifacts are never deleted automatically.

## Explicitly Deferred Scope

- GitHub snapshot reading and authentication;
- raw dirty-working-tree corpus generation;
- multi-repository orchestration in one scheduled execution;
- cron, systemd, GitHub Actions, or another scheduling implementation;
- automatic index rebuilding after corpus publication;
- PostgreSQL and pgvector migration;
- agentic repository exploration or execution of repository tools;
- administration dashboards beyond evaluation and chat views; and
- automated deletion or retention of old vector collections and corpus artifacts.
