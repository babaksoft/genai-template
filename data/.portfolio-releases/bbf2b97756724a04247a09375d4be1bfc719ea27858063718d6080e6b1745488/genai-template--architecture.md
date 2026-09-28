# GenAI Template — Architecture

## Boundaries

- The GenAI Template project is split into three primary layers: the \*\*API layer\*\*, the \*\*services layer\*\*, and the \*\*components/persistence layer\*\*\.  
  The API layer \(\`src/genai\_template/api\`\) exposes FastAPI routers that translate HTTP requests into calls to service objects\.  
  The services layer \(\`src/genai\_template/services\`\) contains business logic for experiments, RAG configurations, source management, and RAG execution, and it interacts with the database via SQLAlchemy sessions\.  
  The components layer \(\`src/genai\_template/components\`\) provides reusable utilities such as document splitters, context builders, and prompt builders\.  
  Observability and lifecycle management are handled by dedicated modules \(\`lifespan\.py\`, \`observability\.py\`\) that are invoked during FastAPI startup and shutdown\.  
  These boundaries keep the web interface, business logic, and low‑level utilities decoupled, enabling independent testing and replacement of each part\.

Evidence:
- `src/genai_template/api/dependencies.py`
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/routes/__init__.py`
- `src/genai_template/components/splitters/__init__.py`
- `src/genai_template/services/__init__.py`

## Dependencies

- The API layer depends on the services layer via dependency injection functions defined in \`src/genai\_template/api/dependencies\.py\`\.  
  Each service \(\`ExperimentService\`, \`RagConfigService\`, \`SourceService\`, \`RagService\`\) relies on a SQLAlchemy session factory \(\`SessionLocal\`\) and on factory functions \(\`create\_embedder\`, \`create\_llm\`, \`create\_vector\_store\`, \`create\_retrieval\_pipeline\`, \`create\_splitter\`\) to instantiate external components\.  
  \`RagService\` additionally imports the \`ContextBuilder\` and \`PromptBuilder\` from the components package\.  
  The \`SourceService\` uses the \`IndexingPipeline\` and the \`application\_span\` observability helper\.  
  Routes in \`src/genai\_template/api/routes\` call the corresponding service methods and translate exceptions into HTTP status codes\.  
  The observability module \(\`src/genai\_template/api/observability\.py\`\) conditionally imports Phoenix instrumentation only when tracing is enabled, avoiding side effects during normal startup\.

Evidence:
- `src/genai_template/api/dependencies.py`
- `src/genai_template/api/routes/answer.py`
- `src/genai_template/api/routes/experiments.py`
- `src/genai_template/api/routes/rag_configs.py`
- `src/genai_template/api/routes/sources.py`
- `src/genai_template/services/experiment_service.py`
- `src/genai_template/services/rag_config_service.py`
- `src/genai_template/services/rag_service.py`
- `src/genai_template/services/source_service.py`

## Principal Flows

- 1\. \*\*Health Check\*\* – \`GET /health\` returns a static \`HealthResponse\` indicating the API is running\.  
  2\. \*\*Experiment Registration\*\* – \`POST /experiments\` creates an experiment linked to an existing source; \`GET /experiments\` lists all experiments; \`GET /experiments/\{id\}\` retrieves a single experiment\.  
  3\. \*\*RAG Configuration Registration\*\* – \`POST /rag\-configs\` idempotently registers a configuration, computing a fingerprint and storing a canonical JSON snapshot; \`GET /rag\-configs\` lists all configurations; \`GET /rag\-configs/\{id\}\` retrieves a specific configuration\.  
  4\. \*\*Source Management\*\* – \`GET /sources/candidates\` lists available corpus directories; \`POST /sources\` registers a new source; \`GET /sources\` lists registered sources; \`PUT /sources/\{id\}/indexes/\{config\_id\}\` rebuilds the deterministic vector‑store index for a source and configuration, returning transient indexing metrics\.  
  5\. \*\*Answer Generation\*\* – \`POST /answer\` triggers \`RagService\.answer\`, which loads the experiment, source, and configuration, verifies the deterministic index exists, starts a run, constructs a retrieval pipeline, builds a prompt, calls the LLM, resolves citations, records run metrics, completes the run, and returns a \`RagResult\`\.  
  6\. \*\*Observability\*\* – During startup, \`lifespan\` configures logging and, if enabled, initializes Phoenix tracing via \`initialize\_observability\`\. Each major operation \(index rebuild, answer generation\) creates OpenTelemetry spans with attributes describing the request, configuration, and metrics\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/answer.py`
- `src/genai_template/api/routes/experiments.py`
- `src/genai_template/api/routes/health.py`
- `src/genai_template/api/routes/rag_configs.py`
- `src/genai_template/api/routes/sources.py`
- `src/genai_template/services/rag_service.py`
