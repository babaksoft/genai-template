# Sliced Data-Model Redesign

## Summary

Replace the prototype schema with:

```text
Source 1 ─── * Experiment 1 ─── * Run * ─── 1 RagConfig
```

Experiments and configs use database IDs as canonical identities. Vector collections are derived deterministically from the source ID and the config’s index-affecting fingerprint; they are not represented in SQL.

No compatibility code or data migration will be retained. Full quality checks run only after the final slice.

## Public interfaces

- Remove `experiment` and `vector_store.collection_name` from `RagConfig` and YAML files. Keep backend connection/base-path settings in the vector-store config.
- Change answer requests to:

  ```json
  {
    "query": "...",
    "experiment_id": 1,
    "rag_config_id": 2
  }
  ```

- Add:
  - `POST /experiments`, `GET /experiments`, and `GET /experiments/{id}`
  - `POST /rag-configs`, `GET /rag-configs`, and `GET /rag-configs/{id}`
  - `PUT /sources/{source_id}/indexes/{rag_config_id}` to explicitly rebuild the deterministic index
- Source registration will only register the corpus directory. Source responses will no longer expose collection or indexing statistics.
- `RagService` will load and execute the persisted config selected by `rag_config_id`; it will no longer be constructed around one fixed config and language-model instance.

## Migration slices

### Slice 1 — Separate portable RAG configuration

- Remove experiment metadata and collection names from the Pydantic `RagConfig`.
- Update settings and bundled YAML files accordingly.
- Keep vector-store backend location settings such as Chroma’s base persistence directory or Qdrant’s path/URL.
- Change vector-store factories to accept the runtime collection name separately from configuration.
- Define fingerprints as:
  - full config fingerprint for `rag_configs` identity;
  - index fingerprint from splitter, embedder, vector-store type, distance, dimensions, and other index-affecting settings only.
- Update focused configuration and factory tests. Skip the full check script.

### Slice 2 — Replace the persistence schema

- Replace all existing Alembic revisions with one squashed initial migration.
- Recreate the local database rather than upgrading or backfilling it.
- Define:
  - `sources`: `id`, unique `name`, unique `directory`, `created_at`;
  - `experiments`: `id`, `source_id`, `name`, optional `description`, `created_at`;
  - `rag_configs`: `id`, unique non-null `config_fingerprint`, non-null canonical `config_json`, `created_at`;
  - `runs`: `id`, `experiment_id`, `rag_config_id`, `query`, timestamps, and runtime/result metrics.
- Remove from runs: `source_id`, `embedding_model`, `vector_store`, `llm_model`, and `top_k`.
- Make run metrics nullable until completion instead of storing misleading zero defaults. An unfinished or failed run has `finished_at = NULL` and null result metrics.
- Add indexes for all foreign keys. Use restrictive deletion for referenced sources/configs and cascading deletion from experiments to their runs.
- Add ORM relationships with `back_populates`.
- Replace persistence and migration tests. Skip unrelated checks.

### Slice 3 — Config and experiment registries

- Replace configuration-as-experiment registration with an idempotent config registry keyed by fingerprint.
- On fingerprint reuse, verify that canonical JSON matches before returning the existing row.
- Implement experiment creation and lookup by ID. Experiment names remain display metadata and are not used as identity.
- Validate the source when creating an experiment. Permit multiple experiments for the same source and duplicate names.
- Add the experiment and config endpoints and schemas.
- Register the application’s default YAML/settings config at startup so at least one config is available; other YAML configs can be registered through the same service/API.
- Remove legacy name-only experiment handling and fingerprint-based experiment resolution.
- Run focused registry/API tests only.

### Slice 4 — Source and deterministic index lifecycle

- Make source creation register only the source directory.
- Remove `collection_name`, document/chunk counts, indexing timestamp, and indexing duration from the source model and API.
- Derive collection names with a backend-safe fixed-length hash, for example:

  ```text
  idx-{first 56 hex characters of SHA256(source_id + ":" + index_fingerprint)}
  ```

- Refactor indexing to load the selected persisted config and rebuild that deterministic collection through `PUT /sources/{source_id}/indexes/{rag_config_id}`.
- Add a vector-store existence operation so execution can distinguish an explicitly built empty index from a missing index.
- Return indexing counts and duration from the build response without persisting them.
- If two configs have the same index fingerprint, they intentionally share the same source index.
- Serialize rebuilds for the same deterministic collection within the service. A failed rebuild returns an error and may require the endpoint to be retried; no SQL metadata can become inconsistent.
- Replace source ingestion/refresh tests with registration, deterministic naming, shared-index, and rebuild tests. Skip the full suite.

### Slice 5 — Run execution and summaries

- Change `start_run` to accept `experiment_id`, `rag_config_id`, and `query`.
- Resolve the source through the experiment and reject the request before creating a run if the deterministic index does not exist.
- Dynamically construct the embedder, vector store, retriever, and language model from the persisted config.
- Persist both foreign keys on the run, then populate metrics and `finished_at` on successful completion.
- Leave failed executions as unfinished runs only when failure occurs after run creation.
- Summarize by canonical experiment ID across all its configurations; optionally include per-config grouping without requiring a fingerprint to identify the experiment.
- Update the UI to select an experiment and registered config, and update evaluator code to create/use real experiment and run records rather than registering configs as experiments.
- Remove all remaining legacy tests and terminology.

### Slice 6 — Integration and cleanup

- Add end-to-end coverage for:
  - multiple configs used by runs in one experiment;
  - one config reused across different experiments;
  - multiple experiments referencing one source;
  - configs with identical index settings sharing a collection;
  - configs with different embedder or distance settings using different collections;
  - missing-index rejection;
  - unfinished versus completed run metrics;
  - summaries spanning multiple configs.
- Update API client, Streamlit UI, documentation, examples, and logging attributes.
- Run `./scripts/check.sh` once for the completed redesign.
- Recreate `db/genai_template.db` with the squashed migration.
- Do not automatically delete existing Chroma/Qdrant storage; document it as optional cleanup because old collections will become orphaned.

## Acceptance criteria and assumptions

- Every run has exactly one experiment and one immutable persisted config.
- Every experiment references exactly one source, while a source may have any number of experiments.
- Source and config are never duplicated on a run.
- Config identity excludes experiment metadata and source-specific collection identity.
- Index identity excludes retrieval-only and generation-only settings.
- Experiments/configs may temporarily have zero runs because SQL foreign keys cannot enforce minimum child cardinality and the selected registry endpoints create them before execution.
- Experiment and config IDs—not names or fingerprints—are used by execution-facing APIs.
- Intermediate slices may use focused tests or intentionally skip the full quality suite; the final slice must pass the complete non-integration check suite and the new applicable integration tests.
