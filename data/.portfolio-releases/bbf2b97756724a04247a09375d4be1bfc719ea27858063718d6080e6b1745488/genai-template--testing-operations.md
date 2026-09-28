# GenAI Template — Testing and Operations

## Testing Strategy

- The test suite is split into unit and integration tests\. Unit tests run fast without external services and are marked with \`not integration\`\. Integration tests require configured external services such as OpenAI, Ollama, or Qdrant and are marked with \`integration\`\. The tests cover API endpoints, dependency injection, health checks, observability initialization, registry CRUD operations, source registration, index rebuilding, and the core RAG service logic\. They also validate splitters, configuration registration, experiment lifecycle, and observability spans for both indexing and answering\.

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

- The FastAPI application is started with \`uvicorn genai\_template\.api\.main:app\`\. The API is prefixed by \`/api/v1\` as defined in the settings\. Local Phoenix tracing can be enabled by setting \`PHOENIX\_ENABLED=true\` and running \`uv run phoenix serve\` in a separate terminal; the API will then send traces to the local Phoenix collector\. The Streamlit UI can be launched with \`uv run streamlit run src/genai\_template/ui/streamlit\_app\.py\`\. The application reads environment variables for OpenAI, Qdrant, and Ollama credentials via \`python‑dotenv\`\.

Evidence:
- `README.md`
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/answer.py`
- `src/genai_template/api/routes/experiments.py`
- `src/genai_template/api/routes/health.py`
- `src/genai_template/api/routes/rag_configs.py`
- `src/genai_template/api/routes/sources.py`

## Configuration

- Configuration is driven by \`pyproject\.toml\` for package metadata and dependencies, and by environment variables loaded at runtime\. The project requires Python 3\.12, uses FastAPI, LlamaIndex, and optional providers such as OpenAI, Ollama, Chroma, and Qdrant\. The \`load\_rag\_config\` function loads a canonical RAG configuration from YAML files, which is used for idempotent registration and deterministic index naming\. The configuration fingerprint and canonical JSON are used to enforce uniqueness in the \`RagConfigService\`\.

Evidence:
- `README.md`
- `pyproject.toml`
- `tests/services/test_rag_config_service.py`

## Observability

- Observability is optional and enabled via the \`PHOENIX\_ENABLED\` flag\. When enabled, \`initialize\_observability\` registers an OpenTelemetry tracer provider with Phoenix, instruments FastAPI, and instruments LlamaIndex\. The lifespan context manager ensures the tracer provider is shut down on application exit\. OpenInference spans are emitted during indexing \(\`rag\.index\.\*\`\) and answering \(\`rag\.answer\`, \`rag\.retrieval\`, \`rag\.context\.build\`, \`rag\.prompt\.build\`, etc\.\), capturing attributes such as collection names, model names, and citation counts\. Tests verify that tracing is correctly configured, that spans are nested appropriately, and that error status codes are set on failed spans\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/observability.py`
- `tests/api/test_observability.py`
- `tests/services/test_indexing_observability.py`
- `tests/services/test_rag_observability.py`

## Operational Constraints

- \- \*\*Tracing\*\* is opt‑in; it only activates when \`settings\.PHOENIX\_ENABLED\` is true and the application state flag is set\.  
  \- \*\*Logging\*\* is configured at startup via \`configure\_logging\`\.  
  \- \*\*RAG configuration registration\*\* is idempotent; duplicate fingerprints with differing JSON raise a conflict\.  
  \- \*\*Source registration\*\* requires the directory to exist and be an immediate child of the corpus root; duplicate names raise a conflict\.  
  \- \*\*Index rebuild\*\* requires both source and RAG configuration to exist; missing records result in a 404\.  
  \- \*\*Answer generation\*\* rejects empty queries \(validation error\) and propagates service‑level errors as HTTP errors\.  
  \- \*\*Run lifecycle\*\*: a run is only marked finished if all stages succeed; failures leave the run unfinished\.  
  \- \*\*Deterministic index naming\*\* uses a SHA‑256 hash of the source ID and configuration fingerprint, producing a 60‑character collection name\.
- These constraints are enforced in the API, services, and tests\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/answer.py`
- `src/genai_template/api/routes/rag_configs.py`
- `src/genai_template/api/routes/sources.py`
- `tests/api/test_answer.py`
- `tests/api/test_sources.py`
- `tests/services/test_rag_service.py`
- `tests/services/test_source_service.py`
