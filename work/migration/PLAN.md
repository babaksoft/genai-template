# Portfolio RAG Package Migration Plan

**Status:** Proposed

## Objective

Organize the current Portfolio RAG work so it can continue developing inside this
repository and later become the core of a standalone production system without a
large rewrite. The promoted system will include a FastAPI API, a Streamlit UI,
command-line utilities, durable persistence, Phoenix tracing, Prometheus metrics,
Grafana dashboards, and operational deployment assets.

The migration must also establish a strict dependency boundary for a privately
installable RAG package. Portfolio RAG will consume that package; the reusable RAG
package must never depend on Portfolio-specific concepts.

## Context and Constraints

- Stage 0 is implemented under `genai_template.workflow.portfolio`.
- Portfolio RAG Stages 1 through 6 will initially be delivered in this repository.
- Before promotion, Portfolio Q&A will be exposed as a separate page or clearly
  isolated section of the current Streamlit UI and will use the FastAPI boundary.
- The promoted repository is an application, not merely a workflow library. CLI
  commands remain useful interfaces but are not its primary architectural root.
- The future private RAG package will be installed from a private Git repository,
  private package index, or private artifact registry, never from a public index.
- Current public behavior, deterministic fingerprints, and committed-snapshot
  guarantees must survive structural refactoring unchanged.

## Architectural Decisions

### 1. Treat Portfolio RAG as a bounded context

The subtree `genai_template.workflow.portfolio` owns Portfolio-specific domain
language and policy: repository snapshots, summary units, generated corpus
artifacts, manifests, Portfolio evaluation cases, and Portfolio conversations.

Generic splitters, embedders, retrievers, rerankers, language-model adapters,
indexing pipelines, retrieval pipelines, generation primitives, and telemetry hooks
do not belong in this subtree. They remain reusable infrastructure and are eventual
candidates for the private RAG package.

### 2. Use ports and adapters around application services

The dependency direction is:

```text
FastAPI / Streamlit / CLI / scheduled invocation
                    |
                    v
        Portfolio application services -----> Portfolio domain
                    |
                    v
                  ports <---------------- concrete adapters
                                               |
                                               v
                                      private RAG package
                                               |
                                               v
                                      provider dependencies
```

Application services depend on protocols, never Git, LlamaIndex, SQLAlchemy,
filesystem publication, Phoenix, or provider clients directly. A composition root
constructs concrete adapters and injects them into services.

### 3. Keep runtime interfaces outside the core

FastAPI routes, Streamlit pages, and CLI argument parsing are inbound adapters. They
may invoke application use cases but must not contain snapshot, corpus, retrieval,
or conversation policy. Streamlit talks to FastAPI through a typed client rather
than importing workflow services.

### 4. Separate telemetry from operational monitoring

- OpenTelemetry/Phoenix records traces for corpus generation, indexing, retrieval,
  generation, and conversation turns.
- Prometheus records bounded-cardinality operational metrics such as request rate,
  error rate, latency, workflow duration, cache usage, index freshness, token usage,
  and provider failures.
- Grafana visualizes Prometheus metrics and operational service health.
- Source content, prompts, generated corpus text, secrets, repository-local paths,
  queries, and answers are excluded from metric labels. Trace content follows an
  explicit privacy configuration.

### 5. Preserve CLI utilities as first-class inbound adapters

Inspection, corpus generation, validation, indexing, and evaluation may expose CLI
commands. CLI modules perform parsing, dependency construction, exit-code mapping,
and report rendering only; application behavior remains callable without a process
boundary.

## Pre-Promotion Package Structure

The following is the target structure while Portfolio RAG remains in this
repository. Directories should be introduced only when their stage begins; this is a
boundary map, not a requirement to create empty packages.

```text
src/genai_template/workflow/portfolio/
  __init__.py

  domain/
    __init__.py
    snapshot.py             # SnapshotFile, RepositorySnapshot, read results
    summary.py              # Summary plans and generated summary identities
    corpus.py               # Corpus documents and manifest domain values
    provenance.py           # Stable artifact and generation provenance
    errors.py               # Provider-neutral domain/application failures

  config/
    __init__.py
    models.py               # Strict immutable Portfolio configuration models
    loader.py               # YAML parsing and path-independent validation
    profiles/
      local.yml

  ports/
    __init__.py
    repository.py           # RepositoryReader
    summarizer.py           # Structured summary generation
    artifact_cache.py       # Content-addressed summary cache
    corpus_store.py         # Validate/stage/atomically publish corpus
    rag.py                  # Narrow Portfolio-facing RAG capabilities
    telemetry.py            # Optional tracing/metric abstractions where needed

  snapshot/
    __init__.py
    selection.py            # Allowlist/exclude and text acceptance policy
    normalization.py        # Canonical text/path normalization
    fingerprinting.py       # Stable source and unit identities
    planning.py             # Explicit logical summary-unit planning
    service.py              # Snapshot application use case

  generation/
    __init__.py
    service.py              # Generate/reuse/validate/publish use case
    workflow.py             # LlamaIndex Workflow orchestration
    events.py               # Workflow event contracts
    schemas.py              # Structured LLM outputs
    prompts.py              # Portfolio-specific prompt templates
    cache_keys.py           # Prompt/model/unit generation fingerprints

  corpus/
    __init__.py
    manifest.py             # Stable manifest construction and fingerprinting
    rendering.py            # Deterministic Markdown rendering
    validation.py           # Evidence and artifact validation
    publication.py          # Staging and atomic publication policy

  ingestion/
    __init__.py
    metadata.py             # Manifest-aware Portfolio metadata enrichment
    freshness.py            # Corpus/index freshness policy

  evaluation/
    __init__.py
    cases.py                # Portfolio evaluation case schema
    datasets.py             # Dataset loading and fingerprinting
    trials.py               # Portfolio trial application service
    reporting.py            # Stable comparison reports

  conversation/
    __init__.py
    models.py               # Portfolio conversation and turn domain values
    service.py              # Conversation and turn use cases
    query_rewriting.py      # Portfolio history-aware query policy

  inspection/
    __init__.py
    service.py              # Read/select/plan orchestration
    reports.py              # Stable text and JSON reports

  adapters/
    repositories/
      local_git.py
      github.py             # Deferred until remote sources are in scope
    summarizers/
      rag_package.py        # Adapter over private RAG/LLM capabilities
    caches/
      filesystem.py
    corpus_stores/
      filesystem.py
    persistence/
      sqlalchemy.py         # Portfolio persistence adapters when introduced
    telemetry/
      opentelemetry.py
      prometheus.py

  cli/
    __init__.py
    inspect.py
    generate.py
    validate.py
    evaluate.py
```

Tests mirror package ownership:

```text
tests/workflow/portfolio/
  domain/
  config/
  snapshot/
  generation/
  corpus/
  ingestion/
  evaluation/
  conversation/
  inspection/
  adapters/
  cli/
```

Unit tests cover pure domain and application behavior. Adapter tests exercise Git,
filesystem, persistence, telemetry, and provider boundaries separately. End-to-end
tests compose real application services with controlled adapters.

## Current-Repository Application Integration

Portfolio-specific workflow code remains under `workflow/portfolio`, but runtime
interfaces integrate through existing application layers:

```text
src/genai_template/
  api/
    routes/
      portfolio_corpus.py
      portfolio_conversations.py
      portfolio_evaluations.py
  services/
    portfolio_service.py       # Thin API-facing facade if needed
  schemas/
    portfolio_api.py           # Request/response transport models
  ui/
    pages/
      portfolio_qa.py          # Preferred Streamlit multipage interface
    portfolio_views.py
    api_client.py
```

If converting the existing Streamlit application to multipage navigation would be
disruptive, the first Q&A interface may be a clearly isolated section in
`streamlit_app.py`. Its state, rendering helpers, and API-client methods must still
be separated so moving it to `ui/pages/portfolio_qa.py` is mechanical.

The current UI must not import Portfolio workflow modules or database sessions. It
uses conversation/corpus/evaluation HTTP endpoints through `ApiClient`, matching the
future standalone deployment boundary.

## Promoted Standalone Project Structure

At promotion, the Portfolio subtree becomes the core of a complete application. The
generic name `workflow` is dropped rather than carried into the new project.

```text
portfolio-rag/
  pyproject.toml
  README.md
  src/portfolio_rag/
    domain/                    # Promoted Portfolio domain modules
    application/               # Use cases and orchestration
      snapshot/
      generation/
      ingestion/
      evaluation/
      conversation/
    ports/                     # Repository, RAG, storage, cache, telemetry ports
    infrastructure/
      repositories/
      rag/                     # Adapter over the private RAG package
      persistence/
      cache/
      corpus/
      observability/           # Phoenix/OpenTelemetry and Prometheus adapters
    interfaces/
      api/
        main.py
        lifespan.py
        dependencies.py
        routes/
        schemas/
      ui/
        streamlit_app.py
        pages/
        api_client.py
        views/
      cli/
        inspect.py
        generate.py
        index.py
        evaluate.py
    config/
      settings.py
      logging.py
      profiles/
    bootstrap.py               # Shared dependency composition

  migrations/                  # Database migrations
  tests/
    unit/
    integration/
    end_to_end/
  deploy/
    docker/
    prometheus/
      prometheus.yml
      alerts.yml
    grafana/
      dashboards/
      provisioning/
    compose.yml
  scripts/
```

FastAPI, Streamlit, and CLI entry points use the same bootstrap/composition
functions. API lifespan owns startup and shutdown of database engines, HTTP/provider
clients, tracing providers, and metric exporters. Readiness checks verify required
dependencies; liveness checks report process health without calling external
providers.

Prometheus scraping, Grafana provisioning, alert rules, container definitions, and
deployment configuration remain outside the Python package under `deploy/`.

## Private RAG Package Boundary

The private package should expose reusable capabilities, not this application's
storage or product model.

Candidates for the private package:

- chunk, retrieved-document, citation, and execution-metric value objects;
- splitter, embedder, retriever, reranker, language-model, and vector-store
  protocols;
- provider implementations and factories;
- indexing, retrieval, context construction, prompt assembly, and stateless chat
  pipelines;
- configuration primitives and deterministic index fingerprints;
- generic evaluation metrics and runners; and
- provider-neutral tracing hooks and safe metric instruments.

Items that remain in Portfolio RAG:

- Git snapshot selection and repository configuration;
- summary units and Portfolio prompts;
- generated Markdown formats, evidence validation, and corpus manifests;
- Portfolio dataset contents and trial-selection policy;
- corpus/index freshness policy and Portfolio persistence records;
- Portfolio conversation behavior, APIs, UI, and operations configuration; and
- concrete dashboards, alerts, and deployment manifests.

The final dependency chain is one-way:

```text
Portfolio RAG system -> private RAG package -> provider libraries
```

The private RAG package is versioned and pinned by an immutable tag or commit and a
lock file. Credentials are supplied by the consuming application. Package builds,
wheels, source distributions, documentation, and CI artifacts remain private.

## Migration Sequence

### Slice 1 — Reorganize the Completed Stage 0 Package

**Assessment:** This refactor can and should be completed as one slice. Stage 0 has
a small dependency surface, focused tests, and no API, database, vector-store, or LLM
integration. A single slice avoids maintaining two competing layouts while Stage 1
is developed.

**Scope:** Structural refactoring only. Do not change snapshot behavior,
fingerprints, validation rules, report fields, or CLI results.

Move the current implementation as follows:

```text
models.py
  -> domain/snapshot.py
  -> domain/summary.py

config.py
  -> config/models.py
  -> config/loader.py

readers.py:RepositoryReader
  -> ports/repository.py

readers.py:LocalGitSnapshotReader
  -> adapters/repositories/local_git.py

selection.py
  -> snapshot/selection.py

planning.py
  -> snapshot/planning.py

inspect_snapshot.py:orchestration and report models
  -> inspection/service.py
  -> inspection/reports.py

inspect_snapshot.py:argument parsing and process exit behavior
  -> cli/inspect.py

workflow/configs/portfolio-local.yml
  -> portfolio/config/profiles/local.yml
```

Keep `genai_template.workflow.portfolio.__init__` as the curated public import
surface. Preserve
`python -m genai_template.workflow.portfolio.inspect_snapshot` as a thin compatibility
entry point delegating to `portfolio.cli.inspect.main`; document the canonical new
entry point as `python -m genai_template.workflow.portfolio.cli.inspect`. Do not keep
duplicate implementations in compatibility modules.

Retain the current configuration path-resolution semantics during this slice. Moving
repository-root injection out of `genai_template.config.settings` is valuable, but it
is a behavioral/API change and belongs in Slice 2.

Update tests to mirror the new ownership boundaries while retaining all Stage 0
acceptance coverage. Add a compatibility test for the old CLI module and verify that
old curated imports still resolve.

Verification:

```bash
uv run pytest tests/workflow/portfolio -v
PHOENIX_ENABLED=false ./scripts/check.sh
```

Acceptance:

- all existing Stage 0 reports and fingerprints are unchanged;
- both CLI module paths execute the same implementation;
- the example profile loads from its new location;
- the old configuration file is removed rather than duplicated;
- no new dependency on LLM, API, database, vector store, or external services is
  introduced; and
- formatting, linting, typing, and non-integration tests pass.

### Slice 2 — Decouple Configuration and Composition

- Make the core YAML loader accept an explicit base directory rather than importing
  application-global settings.
- Move concrete dependency construction into CLI/API composition functions.
- Keep domain and application modules free of `genai_template.config.settings`.
- Introduce focused ports only when a use case needs them; do not create speculative
  empty abstractions.
- Verify loading from arbitrary working directories and from a future package root.

### Slice 3 — Implement Stage 1 Within the New Boundaries

- Add summary-generation, cache, and corpus-store ports.
- Add structured generation schemas, prompts, events, and application workflow.
- Implement filesystem cache/publication and LlamaIndex/private-RAG adapters.
- Keep deterministic rendering and validation independent of provider code.
- Preserve atomic publication and content-addressed caching requirements from the
  master plan.

### Slice 4 — Integrate Portfolio Runtime Interfaces

- Add Portfolio corpus, evaluation, and conversation API endpoints as their stages
  are delivered.
- Add a Portfolio Q&A Streamlit page or isolated section backed only by the API.
- Add CLI commands as thin application adapters.
- Reuse the current API lifespan and observability facilities without putting them
  inside Portfolio domain code.

### Slice 5 — Extract and Consume the Private RAG Package

- Inventory generic modules and classify their dependencies before moving code.
- Define and test the smallest stable public API needed by both repositories.
- Build and install the package from private infrastructure with immutable version
  pins.
- Replace Portfolio's direct imports of current generic modules with a narrow RAG
  adapter implementing Portfolio ports.
- Run contract tests against both fake and real private-package implementations.

### Slice 6 — Promote Portfolio RAG to Its Standalone System

- Move Portfolio domain/application modules with history where practical.
- Add standalone FastAPI, Streamlit, CLI, configuration, persistence, and migration
  composition.
- Add Phoenix, Prometheus, Grafana, health/readiness, container, and deployment
  configuration.
- Migrate Portfolio database records and generated corpus data explicitly; do not
  silently reuse incompatible state.
- Keep compatibility or redirect documentation in this repository until consumers
  have moved.

## Refactoring Rules

- Do not combine structural moves with new Stage behavior in one review.
- Prefer `git mv`-equivalent history-preserving moves and small compatibility
  wrappers over copied implementations.
- Keep public exports deliberate; internal modules should not become public merely
  because they are imported in tests.
- Domain models must not contain FastAPI, Streamlit, SQLAlchemy, Git subprocess,
  provider SDK, Phoenix, Prometheus, or filesystem-publication logic.
- Application services return domain results or focused result models, not HTTP or
  Streamlit values.
- Provider and persistence failures are translated at adapter boundaries into
  contextual application errors.
- Every new function, class, method, and Pydantic field follows the repository's
  docstring and schema-description standards.
- No migration automatically deletes legacy collections, corpus artifacts, caches,
  or database records.

## Completion Criteria

The migration is complete when:

- Portfolio RAG runs as a standalone API/UI/CLI system;
- core domain and application tests run without FastAPI, Streamlit, Git, provider,
  database, Phoenix, Prometheus, or Grafana services;
- infrastructure adapters are replaceable through tested ports;
- the private RAG package is the only source of shared RAG components used by the
  promoted application;
- the current repository no longer owns duplicate Portfolio or reusable RAG
  implementations;
- telemetry and operational monitoring are configured without leaking secrets or
  unbounded/content-bearing labels; and
- end-to-end tests prove snapshot-to-corpus, corpus-to-index, evaluation, and durable
  conversational Q&A flows.
