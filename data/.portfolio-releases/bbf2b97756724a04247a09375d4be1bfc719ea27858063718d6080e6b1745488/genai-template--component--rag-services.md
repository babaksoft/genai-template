# GenAI Template — rag\-services Component

## Responsibilities

- The \`genai\_template\.services\` package exposes four core services that together manage the lifecycle of experiments, RAG configurations, source corpora, and RAG execution\.  
  \* \`ExperimentService\` registers experiments tied to a source, records runs, and aggregates run metrics\.  
  \* \`RagConfigService\` stores immutable RAG configurations, ensuring idempotent registration and fingerprint‑based deduplication\.  
  \* \`SourceService\` registers document corpora, derives deterministic index collection names, and rebuilds vector‑store indexes for a given configuration\.  
  \* \`RagService\` loads a persisted configuration, validates the existence of the corresponding deterministic index, orchestrates retrieval, prompt construction, and LLM generation, and records run metrics\.
- These services coordinate through SQLAlchemy sessions and rely on factory functions for embedder, LLM, and vector‑store creation\.

Evidence:
- `src/genai_template/services/__init__.py`
- `src/genai_template/services/experiment_service.py`
- `src/genai_template/services/rag_config_service.py`
- `src/genai_template/services/rag_service.py`
- `src/genai_template/services/source_service.py`

## Important Abstractions

- \* \*\*Domain models\*\* – \`Experiment\`, \`Run\`, \`RagConfigRecord\`, and \`Source\` represent persisted entities\.  
  \* \*\*Configuration objects\*\* – \`RagConfig\` \(immutable\) and its JSON snapshot/fingerprint are used for idempotent registration\.  
  \* \*\*Service interfaces\*\* – Each service exposes CRUD‑like methods \(\`create\`, \`list\`, \`get\`, \`start\_run\`, \`complete\_run\`, \`summarize\_experiment\`, \`rebuild\_index\`\)\.  
  \* \*\*Component factories\*\* – \`create\_embedder\`, \`create\_llm\`, \`create\_vector\_store\`, and \`create\_retrieval\_pipeline\` produce the runtime components needed for RAG execution\.  
  \* \*\*Observability\*\* – OpenInference spans \(\`rag\.answer\`, \`rag\.index\.rebuild\`, etc\.\) capture execution stages and attributes\.
- These abstractions are exercised by the unit tests and by the implementation code\.

Evidence:
- `src/genai_template/services/experiment_service.py`
- `src/genai_template/services/rag_config_service.py`
- `src/genai_template/services/rag_service.py`
- `src/genai_template/services/source_service.py`
- `tests/services/test_experiment_service.py`
- `tests/services/test_rag_config_service.py`
- `tests/services/test_rag_service.py`
- `tests/services/test_source_service.py`

## Behavior

- \* \*\*ExperimentService\*\*  
  \* \`create\_experiment\` validates the source exists, persists the experiment, and logs creation\.  
  \* \`start\_run\` ensures both experiment and configuration exist, creates an unfinished run, and records foreign keys\.  
  \* \`complete\_run\` records metrics, sets \`finished\_at\`, and commits the run\.  
  \* \`summarize\_experiment\` aggregates only finished runs, optionally filtering by configuration, and returns an \`ExperimentSummary\`\.
- \* \*\*RagConfigService\*\*  
  \* \`register\_config\` computes a canonical JSON snapshot and fingerprint; if a record with the same fingerprint exists, it verifies the JSON matches, otherwise it inserts a new row\.  
  \* \`get\_config\` retrieves a record by ID, raising if missing\.  
  \* \`parse\_config\` deserializes the JSON snapshot into a \`RagConfig\`\.
- \* \*\*SourceService\*\*  
  \* \`register\` validates the directory is an immediate child of the corpus root, ensures uniqueness, persists the source, and does not build an index\.  
  \* \`rebuild\_index\` loads the persisted configuration, derives a deterministic collection name, deletes any existing index, runs the indexing pipeline, and records metrics\.  
  \* \`index\_collection\_name\` hashes the source ID and configuration fingerprint to produce a fixed‑length collection name\.  
  \* Rebuilds are serialized per collection via a class‑level lock\.
- \* \*\*RagService\*\*  
  \* \`answer\` loads experiment, source, and configuration, verifies the deterministic index exists, starts a run, constructs the retrieval pipeline, generates a prompt, calls the LLM, resolves citations, records metrics, completes the run, and returns a \`RagResult\`\.  
  \* Raises \`IndexNotBuiltError\` if the index is missing, and propagates any runtime errors without completing the run\.
- These behaviors are validated by the corresponding unit tests\.

Evidence:
- `src/genai_template/services/experiment_service.py`
- `src/genai_template/services/rag_config_service.py`
- `src/genai_template/services/rag_service.py`
- `src/genai_template/services/source_service.py`
- `tests/services/test_experiment_service.py`
- `tests/services/test_rag_config_service.py`
- `tests/services/test_rag_service.py`
- `tests/services/test_source_service.py`

## Constraints

- \* \*\*Uniqueness &amp; Idempotency\*\* – Experiments are identified by database ID; names are not unique\.  
  \* \*\*Fingerprint Integrity\*\* – \`RagConfigService\.register\_config\` rejects a fingerprint collision if the canonical JSON differs, raising \`ValueError\`\.  
  \* \*\*Deterministic Index Naming\*\* – \`SourceService\.index\_collection\_name\` produces a 60‑character hash, ensuring the same source/config pair always maps to the same collection\.  
  \* \*\*Index Existence\*\* – \`RagService\.answer\` refuses to start a run if the deterministic index does not exist, raising \`IndexNotBuiltError\`\.  
  \* \*\*Locking\*\* – Rebuilds for the same collection are serialized across service instances via a shared \`Lock\`\.  
  \* \*\*Error Propagation\*\* – Any exception during retrieval, prompt building, or LLM generation leaves the run unfinished; metrics are not persisted\.
- These constraints are enforced in the implementation and asserted in the tests\.

Evidence:
- `src/genai_template/services/rag_config_service.py`
- `src/genai_template/services/rag_service.py`
- `src/genai_template/services/source_service.py`
- `tests/services/test_rag_config_service.py`
- `tests/services/test_rag_service.py`
- `tests/services/test_source_service.py`

## Testing Evidence

- Unit tests cover all public methods and edge cases:  
  \* Experiment creation, run lifecycle, and summarization \(\`test\_experiment\_service\.py\`\)\.  
  \* Configuration registration idempotency, fingerprint collision handling, and retrieval \(\`test\_rag\_config\_service\.py\`\)\.  
  \* Source registration, collection naming, rebuild logic, and locking \(\`test\_source\_service\.py\`\)\.  
  \* RAG answer flow, index existence checks, and error handling \(\`test\_rag\_service\.py\`\)\.  
  \* Observability spans for indexing and answering, including nested spans, attributes, and error status \(\`test\_indexing\_observability\.py\`, \`test\_rag\_observability\.py\`\)\.
- These tests exercise the services in isolation with mocked collaborators, ensuring deterministic behavior and correct observability instrumentation\.

Evidence:
- `tests/services/test_experiment_service.py`
- `tests/services/test_indexing_observability.py`
- `tests/services/test_rag_config_service.py`
- `tests/services/test_rag_observability.py`
- `tests/services/test_rag_service.py`
- `tests/services/test_source_service.py`
