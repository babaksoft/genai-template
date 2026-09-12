# Second Adapter Implementation Plan

## Summary

Add four independently selectable adapters while preserving the existing FastEmbed, Ollama, Chroma, and sentence-splitter defaults:

- Embeddings: OpenAI
- Language models: OpenAI
- Vector stores: Qdrant, supporting local and server modes
- Splitters: `MarkdownNodeParser`-based splitting splitter

## Implementation Slices

### 1. Provider-aware configuration foundation

- Convert each component section into a discriminated union while retaining the existing configuration classes for backward compatibility.
- Add provider-specific models:
  - `MarkdownSplitterConfig`
  - `OpenAIEmbedderConfig`
  - `OpenAILLMConfig`
  - `QdrantVectorStoreConfig`
- When a YAML override changes a section's `type`, replace that section rather than inheriting incompatible fields from the default provider.
- Resolve both Chroma and local-Qdrant paths relative to `settings.REPO_ROOT`.
- Keep credentials outside serialized experiment configuration:
  - OpenAI integrations read `OPENAI_API_KEY`.
  - Qdrant server mode reads optional `QDRANT_API_KEY`.
- Preserve baseline configuration and fingerprint behavior; new provider types and content-affecting options participate in index fingerprints.
- Add validation tests for provider switching, forbidden mixed-provider fields, Qdrant local/server requirements, path resolution, and unchanged baseline defaults.

### 2. Markdown splitter adapter

- Add `MarkdownDocumentSplitter`, wrapping LlamaIndex `MarkdownNodeParser`.
- Configure `header_path_separator`, defaulting to `/`.
- Convert parser nodes into the existing `DocumentChunk` schema using the same deterministic `<document-id>-<index>` identifiers.
- Preserve source metadata and the parser-generated `header_path`.
- Do not impose sentence-style chunk size or overlap; sections without Markdown headers remain one chunk.
- Export the adapter and add `markdown` factory dispatch.
- Test header hierarchy, fenced-code headings, metadata, deterministic IDs, empty input, and factory argument forwarding.

### 3. OpenAI embedding and language-model adapters

- Add the direct runtime dependencies `llama-index-embeddings-openai` and `llama-index-llms-openai`.
- Implement `OpenAIEmbeddingModel` with the existing `Embedder` contract:
  - Batch-embed chunk text and mutate the supplied chunks consistently with FastEmbed.
  - Embed queries separately.
  - Reject blank queries and propagate provider errors.
  - Accept `model_name`, optional embedding `dimensions`, and `request_timeout`.
- Implement `OpenAILanguageModel` with the existing `LanguageModel.generate(prompt) -> str` contract:
  - Accept `model_name` and `request_timeout`.
  - Return normalized response text and propagate provider errors.
- Add package exports and `openai` factory branches without changing either protocol.
- Unit-test construction, batching, dimensions, empty inputs, response extraction, errors, and factory dispatch with mocked provider clients.

### 4. Qdrant vector-store adapter

- Add `qdrant-client` as a direct dependency and implement `QdrantStore` against the existing `VectorStore` protocol.
- Support:
  - Local mode through a repository-resolved filesystem path.
  - Server mode through an HTTP(S) URL and optional environment-provided API key.
- Create collections lazily during the first non-empty upsert, deriving vector size from the embeddings. Qdrant requires one vector dimensionality and metric per collection. [Qdrant collection documentation](https://qdrant.tech/documentation/manage-data/collections/)
- Validate all chunks and dimensions before writing anything.
- Map distances as:
  - `cosine` -> Qdrant `COSINE`, returned distance `1 - score`
  - `ip` -> Qdrant `DOT`, returned distance `-score`
  - `l2` -> Qdrant `EUCLID`, returned distance unchanged
- This normalization preserves the project contract that lower values represent closer matches; Qdrant returns cosine similarity directly. [Qdrant migration guidance](https://qdrant.tech/documentation/migration-guidance/diagnosing-discrepancies/)
- Convert arbitrary chunk IDs into deterministic UUID point IDs and retain the canonical chunk ID, document ID, text, and metadata in the payload.
- Make `count`, `search`, and `delete` safe when the collection does not yet exist.
- Add `rag.qdrant.search` observability matching the Chroma adapter.
- Unit-test both connection modes, collection creation, distance mappings, deterministic IDs, payload round-tripping, upsert validation, missing collections, result ordering, deletion, factory dispatch, and error propagation.

### 5. Combined profile, documentation, and integration checks

- Add an example experiment profile combining Markdown, OpenAI embeddings/LLM, and local Qdrant; document the equivalent server-mode configuration.
- Document `OPENAI_API_KEY`, optional `QDRANT_API_KEY`, Qdrant URL/path behavior, and the need to re-index when changing splitter, embedder, vector-store type, dimensions, or distance.
- Add opt-in `integration` tests:
  - Local Qdrant lifecycle using `tmp_path`.
  - OpenAI embedding and generation smoke tests, skipped without `OPENAI_API_KEY`.
  - Qdrant server lifecycle using a unique temporary collection, skipped unless a server URL is explicitly configured and always cleaned up.
  - One composition test covering Markdown -> OpenAI embedding -> Qdrant retrieval.
- Run `./scripts/check.sh`, followed by `uv run pytest -m integration -v` when credentials and services are available.

## Public Interfaces

- Existing `Splitter`, `Embedder`, `LanguageModel`, and `VectorStore` protocols remain unchanged.
- Existing provider configuration classes and defaults remain valid.
- `RagConfig` and factory parameters accept provider-specific discriminated unions.
- New adapters are exported from their component packages:
  - `MarkdownDocumentSplitter`
  - `OpenAIEmbeddingModel`
  - `OpenAILanguageModel`
  - `QdrantStore`
- No database migration is required because experiment configuration is already stored as canonical JSON.

## Assumptions

- OpenAI credentials are environment-only and never included in logs, YAML examples, canonical configuration, or database records.
- Qdrant API keys are likewise environment-only; the non-secret server URL remains part of configuration.
- Existing adapters and the baseline experiment remain the application defaults.
- Markdown splitting applies to both `.md` and `.txt` inputs; headerless documents become a single section.
- Semantic splitting and hybrid/sparse Qdrant retrieval are out of scope.
