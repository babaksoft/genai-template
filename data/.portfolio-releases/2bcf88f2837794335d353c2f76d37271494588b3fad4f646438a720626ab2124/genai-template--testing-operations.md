# GenAI Template — Testing and Operations

## Testing Strategy

- The project employs both unit and integration tests to ensure correctness and robustness\. Unit tests cover core components such as API endpoints, dependency injection, source registration, experiment and RAG configuration registries, and document splitters\. Integration tests are run conditionally when external services like Qdrant, OpenAI, or Ollama are available, verifying end\-to\-end workflows including real\-provider completions\. Tests validate expected behaviors, error handling \(e\.g\., missing experiments, unbuilt indexes\), and observability instrumentation\. Code quality is maintained with formatting, linting, and type checking tools\.

Evidence:
- `README.md`
- `tests/api/test_answer.py`
- `tests/api/test_dependencies.py`
- `tests/api/test_health.py`
- `tests/api/test_observability.py`
- `tests/api/test_registries.py`
- `tests/api/test_sources.py`
- `tests/components/splitters/test_markdown_splitter.py`
- `tests/components/splitters/test_sentence_splitter.py`
- `tests/services/test_experiment_service.py`
- `tests/services/test_indexing_observability.py`
- `tests/services/test_rag_config_service.py`
- `tests/services/test_rag_observability.py`
- `tests/services/test_rag_service.py`
- `tests/services/test_source_service.py`

## Local Operation

- Locally, the API server is run via Uvicorn with the FastAPI app exposed on a configurable host and port \(default 0\.0\.0\.0:8000\)\. The API URL prefix is configurable \(default \`/api/v1\`\)\. Environment variables such as \`OPENAI\_API\_KEY\`, \`QDRANT\_API\_KEY\`, \`QDRANT\_URL\`, \`OLLAMA\_BASE\_URL\`, and \`PHOENIX\_ENABLED\` control external service credentials and observability features\. Phoenix tracing can be enabled locally by running a Phoenix server and setting \`PHOENIX\_ENABLED=true\` before starting the API, which sends detailed request and pipeline traces to the local Phoenix instance\. A Streamlit UI is optionally launched separately and expects the API at \`http://localhost:8000\`\. Corpus directories are placed under a configured root \(default \`data/\`\), and sources are registered explicitly before indexing\. Index rebuilding is an explicit operation triggered via API or UI\.

Evidence:
- `README.md`

## Configuration

- Configuration is primarily managed via environment variables and YAML profiles\. The project requires Python 3\.12\. Key environment variables include \`OPENAI\_API\_KEY\` for OpenAI access, \`QDRANT\_API\_KEY\` and \`QDRANT\_URL\` for Qdrant server authentication and location, \`OLLAMA\_BASE\_URL\` for Ollama endpoint specification, and \`PHOENIX\_ENABLED\` to toggle Phoenix tracing\. The API URL prefix and default models for embedding and LLM are configurable in code settings\. RAG configurations are immutable and registered via API; they specify components such as splitter type, embedder model, vector store type, and retrieval parameters\. Corpus directories must be immediate children of the configured root directory and are registered by name\. Vector store collections are named deterministically by hashing the source ID and index\-affecting configuration fingerprint to ensure backend\-safe fixed\-length names\.

Evidence:
- `README.md`
- `pyproject.toml`

## Observability

- Observability is optionally enabled via Phoenix tracing integrated with OpenTelemetry\. When enabled \(\`PHOENIX\_ENABLED=true\`\), the API initializes a Phoenix tracer provider that instruments FastAPI request handling, LlamaIndex embedding and LLM spans, and internal pipeline stages\. Traces include detailed attributes such as source and config IDs, fingerprints, document and chunk counts, embedding and vector store metadata, and retrieval metrics\. The tracing setup is dynamically imported and applied only if enabled to avoid side effects\. The API lifespan context manager cleanly shuts down the tracer provider on application shutdown\. Phoenix traces are intended for local experimentation and include sensitive data such as user queries and retrieved chunk contents, so enabling it requires appropriate local security\.

Evidence:
- `README.md`
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
- `tests/api/test_observability.py`
- `tests/services/test_indexing_observability.py`
- `tests/services/test_rag_observability.py`

## Operational Constraints

- The system enforces several operational constraints to ensure correctness and security\. Corpus directories must be immediate children of the configured root directory; relative or nested paths are rejected\. Source names must be unique\. Index rebuilds for the same deterministic collection are serialized using process\-local locks to prevent concurrent rebuilds\. Vector store collection names are fixed\-length SHA\-256 hash prefixes derived from source ID and index configuration fingerprint, ensuring backend\-safe naming and avoiding collisions\. Phoenix tracing is opt\-in and must be explicitly enabled; it should not be used with sensitive data unless local storage is secured\. The API expects a consistent URL prefix for routing\. Observability instrumentation imports are deferred to avoid side effects when disabled\. Index rebuilding is an explicit operation; answering against a missing index returns HTTP 409 Conflict\. The lifespan context manager must yield control to allow the app to run between startup and shutdown\.

Evidence:
- `README.md`
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
- `tests/api/test_observability.py`
- `tests/services/test_indexing_observability.py`
- `tests/services/test_source_service.py`
