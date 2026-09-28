# GenAI Template — Architecture

## Boundaries

- The project is architected as a modular Retrieval\-Augmented Generation \(RAG\) experimentation framework exposing a FastAPI\-based HTTP API\. The API component manages lifecycle events, logging, and optional Phoenix\-based observability tracing\. It exposes multiple domain\-specific routers for health checks, answer generation, experiment management, RAG configuration management, and corpus source management\. The services layer encapsulates core business logic for experiments, RAG configurations, RAG execution, and source lifecycle management, including deterministic index building and retrieval\. Document splitting is handled by specialized components that transform documents into chunks suitable for embedding and indexing\. The architecture enforces clear separation between API routing, service logic, and document processing components, with persistent state managed via SQLAlchemy ORM models\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/__init__.py`
- `src/genai_template/components/splitters/__init__.py`
- `src/genai_template/components/splitters/markdown_splitter.py`
- `src/genai_template/components/splitters/sentence_splitter.py`
- `src/genai_template/services/experiment_service.py`
- `src/genai_template/services/rag_config_service.py`
- `src/genai_template/services/rag_service.py`
- `src/genai_template/services/source_service.py`

## Dependencies

- The API layer depends on service classes for core functionality, which are injected via FastAPI&\#x27;s dependency system\. The main dependencies include:
- \- \*\*ExperimentService\*\*: Manages creation, retrieval, and summarization of experiments and runs, backed by a SQLAlchemy session factory\.  
  \- \*\*RagConfigService\*\*: Provides immutable RAG configuration registration and retrieval, ensuring fingerprint consistency and canonical JSON storage\.  
  \- \*\*SourceService\*\*: Manages corpus source registration, listing, retrieval, and deterministic index rebuilding, relying on a configured corpus root directory and the RagConfigService\.  
  \- \*\*RagService\*\*: Coordinates RAG execution by loading persisted experiments, sources, and configurations, creating retrieval pipelines, and invoking language models and embedders\. It depends on ContextBuilder and PromptBuilder components for constructing query context and prompts\.  
  \- \*\*Document Splitters\*\*: MarkdownDocumentSplitter and DocumentSplitter provide document chunking functionality used during indexing\.  
  \- \*\*Observability\*\*: Optional Phoenix tracing instrumentation is conditionally initialized based on configuration, with deferred imports to avoid side effects when disabled\.
- The services rely on SQLAlchemy ORM models for persistence and use factories to create embedders, language models, vector stores, and retrieval pipelines\. The SourceService uses process\-local locks to serialize index rebuilds\.

Evidence:
- `src/genai_template/api/dependencies.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/answer.py`
- `src/genai_template/api/routes/experiments.py`
- `src/genai_template/api/routes/rag_configs.py`
- `src/genai_template/api/routes/sources.py`
- `src/genai_template/components/splitters/__init__.py`
- `src/genai_template/components/splitters/markdown_splitter.py`
- `src/genai_template/components/splitters/sentence_splitter.py`
- `src/genai_template/services/experiment_service.py`
- `src/genai_template/services/rag_config_service.py`
- `src/genai_template/services/rag_service.py`
- `src/genai_template/services/source_service.py`

## Principal Flows

- 1\. \*\*API Application Startup and Shutdown\*\*  
  On startup, logging is configured and, if enabled, Phoenix OpenTelemetry tracing is initialized with instrumentation for FastAPI and LlamaIndex\. On shutdown, the tracer provider is cleanly shut down\. The FastAPI app is created with metadata and routers mounted under a configured API prefix\.
- 2\. \*\*Answer Generation Flow\*\*  
  The \`/answer\` endpoint receives a query with experiment and RAG configuration IDs\. The RagService loads the experiment, source, and configuration, verifies the deterministic index exists, and starts a run record\. It creates a retrieval pipeline and language model, retrieves relevant document chunks, builds context and prompt, generates a response, resolves citations, collects runtime metrics, completes the run, and returns the answer with metadata\.
- 3\. \*\*Experiment Management Flow\*\*  
  The \`/experiments\` endpoints allow clients to create experiments linked to registered sources, list all experiments, and retrieve individual experiments by ID\. The ExperimentService validates source existence and persists experiment metadata\.
- 4\. \*\*RAG Configuration Management Flow\*\*  
  The \`/rag\-configs\` endpoints support idempotent registration of immutable RAG configurations, listing all configurations, and retrieving configurations by ID\. The RagConfigService ensures fingerprint uniqueness and JSON consistency\.
- 5\. \*\*Corpus Source Management Flow\*\*  
  The \`/sources\` endpoints list available corpus directories not yet registered, list registered sources, register new sources by validating directory existence and uniqueness, and rebuild deterministic indexes for a source and RAG configuration\. The SourceService serializes index rebuilds using process\-local locks, deletes existing vector store collections, runs indexing pipelines that split documents and embed chunks, and returns indexing metrics\.
- 6\. \*\*Document Splitting\*\*  
  Document splitting is performed by either MarkdownDocumentSplitter \(splitting on Markdown headers with header path metadata\) or DocumentSplitter \(splitting on sentence boundaries with chunk size and overlap constraints\)\. Both produce deterministic chunk IDs and preserve metadata for downstream embedding and indexing\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/answer.py`
- `src/genai_template/api/routes/experiments.py`
- `src/genai_template/api/routes/rag_configs.py`
- `src/genai_template/api/routes/sources.py`
- `src/genai_template/components/splitters/markdown_splitter.py`
- `src/genai_template/components/splitters/sentence_splitter.py`
- `src/genai_template/services/experiment_service.py`
- `src/genai_template/services/rag_config_service.py`
- `src/genai_template/services/rag_service.py`
- `src/genai_template/services/source_service.py`
