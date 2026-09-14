# Indexing Pipeline Instrumentation Plan

## Scope and Slicing Decision

This feature should be implemented as one cohesive slice. The service-level span,
pipeline stage spans, observability tests, and documentation together define one
observable contract and are small enough to review and verify as a unit.

This plan assumes that "instrumentation" means OpenTelemetry/OpenInference tracing.
It does not add an OpenTelemetry metrics exporter; indexing durations and counts are
captured as span attributes. Phoenix export remains initialized by the FastAPI
lifespan, so command-line evaluation will continue to use the no-op tracer unless
entry-point-independent observability initialization is added separately.

## Current Flow

```text
PUT /sources/{source_id}/indexes/{rag_config_id}
└── SourceService.rebuild_index()
    ├── resolve source and configuration
    ├── acquire collection-specific lock
    ├── delete existing collection
    └── IndexingPipeline.run()
        ├── load documents
        ├── split documents
        ├── embed chunks
        └── upsert chunks or create an empty collection
```

Relevant findings:

- `src/genai_template/pipelines/indexing_pipeline.py` explicitly wraps the build in
  `suppress_instrumentation()`, preventing configured LlamaIndex instrumentation
  from observing loading, splitting, and embedding.
- `src/genai_template/services/source_service.py` owns the useful contextual
  metadata: source and configuration IDs, fingerprints, collection name, component
  types, and the collection-specific rebuild lock.
- `src/genai_template/observability.py` already provides no-op-safe,
  OpenInference-compatible spans and automatic exception recording through the
  OpenTelemetry span context manager.
- `src/genai_template/api/observability.py` already initializes Phoenix, FastAPI,
  and LlamaIndex tracing when tracing is enabled.
- `tests/services/test_rag_observability.py` establishes the testing pattern with an
  in-memory OpenTelemetry exporter and assertions over span hierarchy and
  attributes.

An existing edge case is intentionally outside this instrumentation change: the
empty-corpus path calls `embed_query("")`, while both concrete embedders reject blank
queries. The current empty-directory test uses a mock and therefore does not expose
the mismatch. Empty-corpus behavior should be corrected separately unless it is
explicitly brought into this feature's scope.

## Proposed Trace Structure

```text
FastAPI PUT request
└── rag.index.rebuild                  CHAIN
    ├── rag.index.delete               CHAIN
    └── rag.index.build                CHAIN
        ├── rag.index.load             CHAIN
        ├── rag.index.split            CHAIN
        ├── rag.index.embed            EMBEDDING
        │   └── LlamaIndex provider spans
        └── rag.index.write            CHAIN
```

The service-level span distinguishes rebuild lifecycle work from pipeline execution
and captures lock wait and collection-deletion overhead. The pipeline-level span
also remains useful when `IndexingPipeline` is invoked directly.

### Root Attributes

- `rag.source.id`
- `rag.config.id`
- `rag.config.fingerprint`
- `rag.index.fingerprint`
- `rag.index.collection`
- `rag.splitter.type`
- `rag.embedding.provider`
- `rag.embedding.model`
- `rag.vector_store.type`
- `rag.vector_store.distance`

### Result and Stage Attributes

- Document count.
- Chunk count.
- Embedding count and vector dimension.
- Write operation: `upsert` or `create`.
- Number of records written.

Directory paths, document text, chunk text, API keys, and serialized configuration
must not be placed in custom attributes. LlamaIndex instrumentation may still retain
model inputs when Phoenix is enabled, so the existing local-use privacy warning must
be expanded accordingly.

## Implementation

### 1. Generalize the Observability Helper Documentation

Update `src/genai_template/observability.py` so it describes application-wide RAG
workflow spans rather than only the answer path. Preserve its no-op behavior when no
provider is configured. Keep the current tracer identity unless changing it is
required for compatibility or Phoenix presentation.

### 2. Instrument Rebuild Orchestration

Update `SourceService.rebuild_index()` in
`src/genai_template/services/source_service.py`:

- Start `rag.index.rebuild` once the source, persisted configuration, and collection
  identity have been resolved.
- Include lock acquisition inside the root span so contention is reflected in its
  duration.
- Add safe configuration and deterministic-index attributes to the root span.
- Wrap collection deletion in `rag.index.delete`.
- Attach final document and chunk counts to the rebuild span after a successful
  pipeline run.
- Preserve exception propagation and the current destructive rebuild semantics.

### 3. Instrument Pipeline Stages

Update `IndexingPipeline.run()` in
`src/genai_template/pipelines/indexing_pipeline.py`:

- Add the `rag.index.build` parent span.
- Add `rag.index.load`, `rag.index.split`, `rag.index.embed`, and
  `rag.index.write` child spans around the existing operations.
- Attach counts only after each stage completes successfully.
- Record vector dimension without recording embedding values.
- Mark whether the write branch used `upsert` or empty-index `create`.
- Remove the pipeline-wide `suppress_instrumentation()` so configured LlamaIndex
  instrumentation can create provider-level spans beneath the application stages.
- Preserve the existing definition of `IndexingResult.indexing_time`; native span
  duration represents the wider rebuild time.

### 4. Add Focused Observability Tests

Add `tests/services/test_indexing_observability.py`, following the in-memory exporter
pattern in the answer-path observability tests. Cover:

- The exact set of manual spans and their parent-child relationships.
- Source, configuration, and deterministic-index metadata on the rebuild span.
- Document counts, chunk counts, vector dimension, and write operation on the stage
  spans.
- The absence of corpus paths and document or chunk contents from custom
  attributes.
- Empty-index branch instrumentation using mocks, without changing current
  empty-corpus behavior.
- A failing embed or store operation producing error spans while propagating the
  original exception.

Retain the existing pipeline and source-service behavior tests rather than replacing
them.

### 5. Update Operational Documentation

Update the local Phoenix section in `README.md`:

- List index rebuilds and their stages among the operations visible in Phoenix.
- Explain that enabling LlamaIndex instrumentation can expose corpus content.
- Preserve the current guidance that tracing is intended for appropriately
  protected local experimentation.

## Verification

Run the focused tests first:

```bash
uv run pytest tests/pipelines/test_indexing_pipeline.py \
  tests/services/test_source_service.py \
  tests/services/test_indexing_observability.py -v
```

Then run the complete formatting, import, lint, type, and unit-test suite:

```bash
./scripts/check.sh
```

Integration tests are unnecessary for the instrumentation contract because the
trace hierarchy and attributes can be verified deterministically with mocked
providers and an in-memory exporter.

## Acceptance Criteria

- An API-triggered index rebuild appears as one coherent nested trace beneath its
  FastAPI request span when Phoenix tracing is enabled.
- The trace separates rebuild lifecycle, deletion, loading, splitting, embedding,
  and vector-store writing.
- Successful traces expose source/config/index identity, component selection,
  document and chunk counts, and embedding dimension without adding corpus content
  to custom attributes.
- Failed stages are marked as errors and retain the original exception behavior.
- Tracing remains a no-op when Phoenix is disabled.
- Existing indexing results, collection naming, locking, API responses, and logging
  behavior remain unchanged.
- Focused tests and `./scripts/check.sh` pass.
