# YAML RAG Configuration — Vertical Slices

## Summary

Implement the feature in six sequential, reviewable slices. Every slice leaves the repository passing `./scripts/check.sh` and delivers complete behavior without depending on unfinished code in the same diff.

After approval, save this complete plan to `work/ongoing/PLAN.md` and begin only Slice 1.

## Slice 1 — Validated YAML Configuration

Deliver a standalone configuration API without changing runtime behavior.

- Add immutable, typed Pydantic models for experiment, splitter, embedder, vector store, retrieval, and LLM configuration.
- Add `load_rag_config(path: Path | None) -> RagConfig`:
  - Start with values from `settings.py`.
  - Recursively apply partial YAML overrides.
  - Safely parse YAML and reject unknown keys, unsupported component types, and invalid values.
  - Resolve relative filesystem paths against `settings.REPO_ROOT`.
- Add PyYAML to project dependencies and update the lockfile.
- Add a documented baseline example under `src/genai_template/experiments/configs/`.
- Test default equivalence, partial overrides, validation failures, missing/malformed files, and path resolution.
- Acceptance: loading no file reproduces current RAG settings exactly; no API, ingestion, or evaluation behavior changes.

## Slice 2 — Configurable Components and Factories

Make a validated configuration capable of constructing the existing RAG components.

- Update splitter, FastEmbed embedder, Chroma store, and Ollama model constructors to accept explicit values while retaining their existing `settings.py` defaults for direct callers.
- Implement the existing factory modules for splitter, embedder, vector store, retrieval pipeline, and LLM creation.
- Keep provider selection typed and explicit; initially support only `sentence`, `fastembed`, `chroma`, and `ollama`.
- Ensure the Chroma distance setting and Ollama timeout/base URL are injected instead of read from unrelated globals.
- Test factory dispatch and constructor arguments with external model and storage initialization mocked.
- Acceptance: callers can build a complete configured indexing/retrieval/generation component graph, while all existing callers remain compatible.

## Slice 3 — YAML-Driven Retrieval Evaluation

Deliver the first end-to-end YAML experiment workflow.

- Extend the existing baseline evaluation entry point with:
  - `--config PATH`
  - `--corpus PATH`
  - `--dataset PATH`
- Keep every argument optional, preserving the existing baseline locations and `settings.py` defaults.
- Load the YAML, construct indexing and retrieval pipelines through the factories, build a dedicated evaluation collection, and evaluate using the configured `top_k`.
- Rebuild that dedicated collection for every invocation in this slice, ensuring an embedding or chunking override can never query an incompatible pre-existing index.
- Include the resolved experiment name, configuration fingerprint, collection name, and evaluation metrics in logs.
- Test argument parsing and orchestration with mocked storage/models; retain the Ollama/Chroma suite as optional integration coverage.
- Acceptance: a user can run a retrieval experiment from YAML without modifying Python settings, and running without YAML produces the current baseline configuration.

## Slice 4 — Reusable Configuration-Specific Indexes

Avoid unnecessary re-indexing without sacrificing correctness.

- Add canonical serialization and separate SHA-256 fingerprints for:
  - The fully resolved experiment configuration.
  - Index-affecting settings only.
  - Supported corpus file paths and contents.
- Derive a valid, length-bounded Chroma collection name from the index and corpus fingerprints.
- Reuse a populated matching collection by default.
- Add `--reindex` to delete and rebuild only the resolved experiment collection.
- Ensure answer-only changes such as experiment name, LLM model, or `top_k` do not create another index.
- Do not automatically delete older fingerprinted collections; cleanup policy remains outside this feature.
- Test hash stability, meaningful hash changes, collection naming, reuse, forced rebuilding, and collection isolation.
- Acceptance: identical corpus/indexing configurations reuse an index, while corpus or indexing changes select a different collection.

## Slice 5 — Experiment Configuration Provenance

Persist the exact resolved configuration associated with experiments.

- Add nullable `config_json` and `config_fingerprint` columns to `Experiment` through Alembic so legacy rows remain readable.
- Store canonical resolved JSON rather than the partial source YAML.
- Extend `ExperimentService` to register or resolve experiments by name and fingerprint.
- Permit legacy and differently configured records to coexist; if a name maps to multiple configurations, summary lookup without a fingerprint fails clearly instead of combining incomparable runs.
- Register the active experiment configuration when the evaluation CLI starts and when configured RAG runs are created.
- Test migration behavior, canonical snapshot persistence, idempotent registration, same-name/different-config handling, legacy records, and ambiguous summaries.
- Acceptance: each new experiment can be tied to an immutable configuration fingerprint and reconstructed from its stored snapshot.

## Slice 6 — Default Application Adoption

Move the API and standard ingestion paths onto the same configuration mechanism without exposing YAML through the HTTP API.

- Add a reusable `create_rag_service(config: RagConfig)` composition function.
- Have FastAPI dependencies construct the default configuration from `settings.py` and inject it into `RagService`.
- Make `RagService` use injected experiment identity, `top_k`, model/store names, metrics, and trace attributes.
- Route standard ingestion and source indexing through the shared default-config factories while preserving existing source collection behavior.
- Keep request/response schemas, routes, UI behavior, and server startup commands unchanged.
- Update service, dependency, source-ingestion, metrics, and observability tests.
- Acceptance: the application behaves as before by default, programmatic experiment callers can supply a loaded YAML configuration, and experiment-sensitive components no longer depend on mutable module globals.

## Overall Verification and Assumptions

- Run focused tests during each slice and `./scripts/check.sh` before considering that slice complete.
- All new functions, methods, and classes receive complete Google-style docstrings.
- YAML files are trusted local inputs; secrets continue to come from `.env`.
- YAML selection is explicit through the evaluation CLI or programmatic construction. Per-request configuration and an API-wide YAML environment variable are out of scope.
- Evaluation-result persistence remains unchanged; this feature persists configuration provenance, not a new metrics model.
- Commit boundaries should align with the six slices so each diff can be reviewed and reverted independently.
