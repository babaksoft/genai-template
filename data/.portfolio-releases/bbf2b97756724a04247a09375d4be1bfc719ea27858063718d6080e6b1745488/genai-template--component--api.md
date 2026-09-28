# GenAI Template — api Component

## Responsibilities

- The GenAI Template API module is responsible for configuring a FastAPI application that exposes endpoints for health checks, answer generation, experiment registration, RAG configuration management, and corpus source handling\. It sets up the application lifespan to configure logging and optional Phoenix tracing, injects service dependencies via FastAPI’s dependency injection system, and includes routers that map HTTP routes to business logic in the services layer\.

Evidence:
- `src/genai_template/api/dependencies.py`
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/routes/answer.py`
- `src/genai_template/api/routes/experiments.py`
- `src/genai_template/api/routes/health.py`
- `src/genai_template/api/routes/rag_configs.py`
- `src/genai_template/api/routes/sources.py`

## Important Abstractions

- The API layer abstracts the following key concepts:
- \- \*\*FastAPI application\*\* – the central object that holds configuration, routers, and lifecycle events\.  
  \- \*\*APIRouter\*\* – modular route groups for answer, experiments, health, rag‑configs, and sources\.  
  \- \*\*Dependency injection functions\*\* \(\`get\_rag\_service\`, \`get\_source\_service\`, \`get\_experiment\_service\`, \`get\_rag\_config\_service\`\) that construct and wire together service objects\.  
  \- \*\*Service classes\*\* \(\`RagService\`, \`SourceService\`, \`ExperimentService\`, \`RagConfigService\`\) that encapsulate business logic and data persistence\.  
  \- \*\*Observability initialization\*\* – optional Phoenix/OpenTelemetry tracing that is enabled via configuration\.  
  \- \*\*Lifespan context manager\*\* – handles startup and shutdown, including tracer provider shutdown\.

Evidence:
- `src/genai_template/api/dependencies.py`
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/__init__.py`

## Behavior

- \- \*\*Endpoint responses\*\*: Each router returns Pydantic models \(\`AnswerResponse\`, \`ExperimentResponse\`, \`RagConfigResponse\`, \`SourceResponse\`, etc\.\) and translates service exceptions into appropriate HTTP status codes \(e\.g\., \`IndexNotBuiltError\` → 409, \`ValueError\` → 404\)\.  
  \- \*\*Dependency overrides\*\*: Tests replace dependencies with mocks to isolate endpoint logic\.  
  \- \*\*Observability\*\*: When Phoenix tracing is enabled, \`initialize\_observability\` registers a tracer provider and instruments FastAPI and LlamaIndex; otherwise it performs no action\.  
  \- \*\*Lifespan\*\*: On startup, logging is configured and optional tracing is initialized; on shutdown, any tracer provider is gracefully shut down\.  
  \- \*\*Idempotent config registration\*\*: The rag‑config endpoint registers a configuration only if its fingerprint is new; otherwise it returns the existing record\.  
  \- \*\*Source registration\*\*: Validates directory existence, prevents duplicate names, and returns only metadata without indexing details\.  
  \- \*\*Index rebuild\*\*: Rebuilds a source’s deterministic index and returns transient metrics; missing source or config results in a 404\.

Evidence:
- `tests/api/test_answer.py`
- `tests/api/test_dependencies.py`
- `tests/api/test_health.py`
- `tests/api/test_observability.py`
- `tests/api/test_registries.py`
- `tests/api/test_sources.py`

## Constraints

- \- \*\*Phoenix tracing\*\* is opt‑in; it is only activated when \`settings\.PHOENIX\_ENABLED\` is true and the application state flag is set\.  
  \- \*\*Logging\*\* is configured at application startup via \`configure\_logging\`\.  
  \- \*\*RAG configuration registration\*\* must be idempotent; duplicate fingerprints with differing JSON raise a conflict\.  
  \- \*\*Source registration\*\* requires the directory to exist and be a directory; duplicate names raise a conflict\.  
  \- \*\*Index rebuild\*\* requires both source and RAG configuration to exist; missing records raise a not‑found error\.  
  \- \*\*Answer generation\*\* rejects empty queries \(validation error\) and propagates service‑level errors as HTTP errors\.

Evidence:
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/observability.py`
- `src/genai_template/api/routes/answer.py`
- `src/genai_template/api/routes/rag_configs.py`
- `src/genai_template/api/routes/sources.py`

## Testing Evidence

- The test suite verifies:
- \- Correct construction of service dependencies \(\`test\_get\_rag\_service\_injects\_registries\`, \`test\_get\_source\_service\_injects\_config\_registry\`\)\.  
  \- Health endpoint returns a healthy status \(\`test\_health\_check\_returns\_healthy\_status\`\)\.  
  \- Answer endpoint handles success, validation errors, missing experiments, and missing indices \(\`test\_answer\_returns\_generated\_response\`, \`test\_answer\_rejects\_empty\_query\`, \`test\_answer\_rejects\_missing\_experiment\`, \`test\_answer\_rejects\_missing\_index\_with\_conflict\`\)\.  
  \- Observability initialization respects the enabled flag and correctly registers instruments \(\`test\_initialize\_observability\_skips\_disabled\_tracing\`, \`test\_initialize\_observability\_configures\_phoenix\`, \`test\_lifespan\_shuts\_down\_initialized\_tracer\_provider\`, \`test\_lifespan\_skips\_shutdown\_without\_tracer\_provider\`\)\.  
  \- Experiment and RAG configuration registries expose create, list, get, and error handling \(\`test\_experiment\_endpoints\_create\_list\_and\_get\`, \`test\_experiment\_creation\_rejects\_missing\_source\`, \`test\_rag\_config\_endpoints\_register\_list\_and\_get\`, \`test\_registry\_get\_endpoints\_return\_not\_found\`\)\.  
  \- Source endpoints list candidates, list registered sources, register new sources, reject duplicates, rebuild indexes, and handle missing records \(\`test\_list\_source\_candidates\`, \`test\_list\_sources\_exposes\_registration\_metadata\_only\`, \`test\_create\_source\_registers\_directory\`, \`test\_create\_source\_rejects\_duplicate\_name\`, \`test\_rebuild\_index\_returns\_transient\_metrics\`, \`test\_rebuild\_index\_rejects\_missing\_source\_or\_config\`\)\.

Evidence:
- `tests/api/test_answer.py`
- `tests/api/test_dependencies.py`
- `tests/api/test_health.py`
- `tests/api/test_observability.py`
- `tests/api/test_registries.py`
- `tests/api/test_sources.py`
