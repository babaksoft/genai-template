# GenAI Template — api Component

## Responsibilities

- The configured component is responsible for creating and configuring a FastAPI application that serves as the API for the GenAI Template RAG experimentation framework\. It manages the application lifecycle, including startup and shutdown events, and conditionally enables Phoenix\-based observability tracing\. The component also registers multiple API routers that handle different aspects of the system, such as health checks, answer generation, experiment management, RAG configuration management, and corpus source management\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/__init__.py`

## Important Abstractions

- Key abstractions include:
- \- \*\*FastAPI application instance\*\*: The main web application object configured with metadata and lifecycle management\.  
  \- \*\*Lifespan context manager\*\*: Manages startup and shutdown logic, including logging configuration and observability initialization\.  
  \- \*\*Phoenix observability/tracing\*\*: Optional instrumentation for request tracing using OpenTelemetry and Phoenix\.  
  \- \*\*API routers\*\*: Modular route groups for different API domains:  
  \- \`health\_router\` for health checks  
  \- \`answer\_router\` for answer generation  
  \- \`experiments\_router\` for experiment registry operations  
  \- \`rag\_configs\_router\` for RAG configuration registry  
  \- \`sources\_router\` for corpus source management  
  \- \*\*Dependency injection\*\*: Services like RAG service, source service, experiment service, and RAG config service are injected into routes via FastAPI&\#x27;s dependency system\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/__init__.py`

## Behavior

- The component behaves as follows:
- \- On startup, it configures logging and optionally initializes Phoenix tracing if enabled\.  
  \- On shutdown, it logs the shutdown event and cleanly shuts down the tracer provider if it was initialized\.  
  \- It creates a FastAPI app with metadata and includes multiple routers under a common API URL prefix\.  
  \- It sets a flag in the app state to indicate whether Phoenix tracing is enabled, based on either the passed parameter or configuration settings\.  
  \- The API routes handle HTTP requests for various functionalities, delegating to injected service instances\.  
  \- Observability initialization dynamically imports and instruments tracing only when enabled, avoiding side effects otherwise\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`

## Constraints

- \- Phoenix tracing is opt\-in and only initialized if explicitly enabled via configuration or parameter\.  
  \- The application expects a consistent API URL prefix from configuration to mount all routers\.  
  \- Observability instrumentation imports are deferred to avoid side effects when tracing is disabled\.  
  \- The lifespan context manager must yield control to allow the app to run between startup and shutdown\.  
  \- The app state is used to store runtime flags and tracer provider references, requiring careful management\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`

## Testing Evidence

- \- Tests verify that the health endpoint returns a healthy status\.  
  \- Observability tests confirm that Phoenix tracing is only initialized when enabled and that the tracer provider is properly shut down during lifespan exit\.  
  \- Dependency tests ensure that service dependencies are correctly composed and injected\.  
  \- Registry tests validate that experiment and RAG configuration endpoints correctly create, list, and retrieve records, and handle not\-found errors\.  
  \- Source API tests confirm that corpus source endpoints list candidates, register sources, rebuild indexes, and handle error conditions appropriately\.  
  \- Answer API tests verify that the answer endpoint returns generated responses, rejects invalid queries, and handles missing experiments or unbuilt indexes with proper HTTP status codes\.

Evidence:
- `tests/api/test_answer.py`
- `tests/api/test_dependencies.py`
- `tests/api/test_health.py`
- `tests/api/test_observability.py`
- `tests/api/test_registries.py`
- `tests/api/test_sources.py`
