# GenAI Template Project

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://github.com/babaksoft/genai-template/raw/refs/heads/master/pyproject.toml)
![Category: RAG](https://img.shields.io/badge/category-RAG-orange)
![Category: Agentic](https://img.shields.io/badge/category-Agentic-orange)
![Framework: LlamaIndex](https://img.shields.io/badge/framework-LlamaIndex-orange)
![License](https://img.shields.io/github/license/babaksoft/genai-template)
![CI Status](https://img.shields.io/github/actions/workflow/status/babaksoft/genai-template/ci.yml)

---

## Overview

`genai-template` is a starter kit for building **retrieval‑augmented generation (RAG)** and **agentic** applications. It wires together a FastAPI backend, a Streamlit UI, configurable Chroma or Qdrant vector storage, and a flexible component system for readers, splitters, embedders, language models, prompts, and context building.

### Core Packages
- **`src/genai_template/config`** – Global settings (`settings.py`) and logging configuration.
- **`src/genai_template/components`** – Pluggable building blocks:
  - `readers` (e.g., `TextReader`)
  - `splitters` (sentence and Markdown header splitters)
  - `embeddings` (FastEmbed and OpenAI wrappers)
  - `language_models` (Ollama and OpenAI wrappers)
  - `prompt` (PromptBuilder)
  - `context` (ContextBuilder)
- **`src/genai_template/pipelines`** – Orchestrate workflows:
  - `IndexingPipeline` – read → split → embed → store.
  - `RetrievalPipeline` – embed query → retrieve chunks.
  - `ChatPipeline` – combines retrieval and synthesis for chat use‑cases.
- **`src/genai_template/services`** – High‑level services used by the API:
  - `RagService` – loads a registered configuration and executes a run.
  - `SourceService` – registers corpora and builds deterministic indexes.
  - `RagConfigService` – stores immutable portable configurations.
  - `ExperimentService` – persists experiments, runs, and summaries.
- **`src/genai_template/stores`** – Persistence layers:
  - `vector` – Chroma and local/server Qdrant vector stores.
  - `kv`, `document`, `index` (placeholders for future stores).
- **`src/genai_template/api`** – FastAPI app (`main.py`) with routers for:
  - `answer` – POST `/answer` returns a generated answer with inline source labels,
    metrics, structured sources, and non-fatal citation warnings.
  - `sources` – register corpora and explicitly rebuild selected indexes.
  - `experiments` – create and inspect source-bound experiments.
  - `rag-configs` – register and inspect immutable RAG configurations.
  - `health` – GET `/health` health‑check endpoint.
- **`src/genai_template/ui`** – Streamlit front‑end (`streamlit_app.py`) that talks to the API.
- **`src/genai_template/evaluation`** – Baseline evaluation script and metrics.
- **`src/genai_template/experiments`** – Experiment configuration utilities.

The persisted model is `Source 1—* Experiment 1—* Run *—1 RagConfig`. A run
always names an experiment ID and configuration ID. Vector collections are not
SQL records: their names are deterministic hashes of the source ID and the
configuration's index-affecting settings.

## Portfolio RAG Experiment

Portfolio RAG is a proposed experiment for generating a technical corpus from active
Python repositories, comparing RAG configurations against curated evaluation data,
and supporting cited multi-turn questions from portfolio reviewers.

The first planned stage reads one committed snapshot from a local Git repository,
ignoring staged, modified, and untracked working-tree files. Later stages generate a
flat, manifest-backed corpus under `data/portfolio`, track index freshness, add
versioned evaluation trials, optimize retrieval, and introduce persistent Portfolio
Q&A conversations.

This experiment is not implemented yet. Its design and delivery documents are:

- [Portfolio RAG master plan](PLAN.md), currently **Proposed**;
- [Stage 0 sliced implementation plan](work/ongoing/PLAN.md), currently **Planned**;
  and
- [Architecture decision records](docs/decisions/).

## Getting Started

### 1️⃣ Install dependencies
```bash
uv sync --all-groups
```
> **Note:** Python 3.12 is required (see `pyproject.toml`).

### 2️⃣ Configure environment
Create a ``.env`` at the repository root (or edit the existing one) with the required keys, e.g.:
```
OPENAI_API_KEY=…
# Optional: authenticate to a Qdrant server selected in an experiment profile.
QDRANT_API_KEY=…
# Optional: opt into the Qdrant server integration test.
QDRANT_URL=https://qdrant.example.com
# Optional: use a specific Ollama service instead of automatic discovery.
OLLAMA_BASE_URL=http://ollama.example:11434
# Optional: send FastAPI request traces to a locally running Phoenix instance.
PHOENIX_ENABLED=true
```
The project reads these variables via ``python‑dotenv``.

### 3️⃣ Run the API server
```bash
uv run uvicorn genai_template.api.main:app \
    --host 0.0.0.0 --port 8000
```
The API lives under the prefix defined in ``settings.API_URL_PREFIX`` (default `/api/v1`).

### Local Phoenix tracing

Phoenix is included as a project dependency. In a separate terminal, start the local
server with:

```bash
uv run phoenix serve
```

Then set ``PHOENIX_ENABLED=true`` before starting the API. FastAPI request traces,
index rebuild stages, retrieval, vector-store operations, context/prompt construction,
and LlamaIndex embedding/LLM spans are sent to
``http://localhost:6006/v1/traces`` and appear at ``http://localhost:6006`` under the
``genai-template`` project. Override the defaults with
``PHOENIX_COLLECTOR_ENDPOINT`` and ``PHOENIX_PROJECT_NAME`` if needed.

Phoenix tracing is intended for local experimentation: traces retain full user queries,
corpus content processed by LlamaIndex, retrieved chunk contents, prompts, and generated
answers. Do not enable it with sensitive data unless the local Phoenix storage is
appropriately protected.

### 4️⃣ (Optional) Launch the UI
```bash
uv run streamlit run src/genai_template/ui/streamlit_app.py
```
The UI expects the API at ``http://localhost:8000``.

### 5️⃣ Register a corpus, configuration, and experiment

Place each corpus in its own directory under ``data/`` (the default
``CORPORA_DIR``). Documents cannot be placed directly in the corpus root:

```text
data/
  baseline/
  product-docs/
  handbook/
```

Use the Streamlit sidebar to register a directory, create an experiment for the
source, select a registered configuration, and choose **Rebuild selected
index**. Source registration does not index documents. Rebuilding is an
explicit operation for one source/configuration pair; answering against a
missing index returns HTTP 409 without creating a run.

The same lifecycle is available directly through the API:

```bash
curl -X POST http://localhost:8000/api/v1/sources \
  -H 'Content-Type: application/json' \
  -d '{"directory":"baseline"}'

curl -X POST http://localhost:8000/api/v1/experiments \
  -H 'Content-Type: application/json' \
  -d '{"source_id":1,"name":"Baseline comparison"}'

curl -X PUT http://localhost:8000/api/v1/sources/1/indexes/1

curl -X POST http://localhost:8000/api/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is RAG?","experiment_id":1,"rag_config_id":1}'
```

The answer response associates inline labels such as `[S1]` with the exact retrieved
chunks supplied to the model. Every context source is returned, including uncited
sources, and labels remain distinct when several chunks come from the same document:

```json
{
  "answer": "RAG grounds a generated response in retrieved evidence [S1].",
  "metrics": {
    "query": "What is RAG?",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "vector_store": "Chroma",
    "llm_model": "gpt-oss:20b-cloud",
    "top_k": 5,
    "retrieved_chunks": 1,
    "best_distance": 0.12,
    "worst_distance": 0.12,
    "context_length": 128,
    "prompt_length": 640,
    "response_length": 65,
    "retrieval_time": 0.08,
    "generation_time": 0.45,
    "total_time": 0.53
  },
  "sources": [
    {
      "label": "S1",
      "chunk_id": "guide.md-001",
      "document_name": "guide.md",
      "section": "/Introduction/",
      "content": "Exact retrieved chunk text...",
      "distance": 0.12,
      "cited": true
    }
  ],
  "citation_warnings": [
    {
      "code": "unsupported_citation_labels",
      "message": "The answer references labels not present in its context.",
      "labels": ["S9"]
    }
  ]
}
```

Unsupported labels do not fail or rewrite the generated answer. They are reported in
`citation_warnings` so callers can make them visible. Answers, structured sources, and
citation warnings are response-only data and are not retained in the run database.

The application registers the resolved default configuration at startup. Post
another fully resolved configuration to `POST /api/v1/rag-configs`; repeating
the same configuration returns the existing registry row.

Run a retrieval evaluation with the defaults, or override the resolved RAG
configuration, corpus, and dataset from the command line:

```bash
uv run python -m genai_template.evaluation.evaluators.baseline_eval
uv run python -m genai_template.evaluation.evaluators.baseline_eval \
  --config src/genai_template/experiments/configs/baseline.yml \
  --corpus data/baseline \
  --dataset src/genai_template/evaluation/datasets/baseline-eval.json
```

Evaluation collections are keyed by source ID and index configuration. Pass
``--reindex`` to rebuild only that deterministic collection before calculating
the metrics. Retrieval-only and generation-only changes share an index;
splitter, embedder, vector-store type, embedding-dimension, or distance changes
select a different one.

### Combined OpenAI and Qdrant profile

The example profile
[`openai-markdown-qdrant.yml`](src/genai_template/experiments/configs/openai-markdown-qdrant.yml)
combines Markdown header splitting, OpenAI embeddings and generation, and local
Qdrant persistence. Set ``OPENAI_API_KEY`` in the environment before using it;
credentials are never stored in the profile or serialized experiment configuration.

Relative local Qdrant paths are resolved from the repository root. To use a Qdrant
server instead, replace the profile's ``vector_store`` section with:

```yaml
vector_store:
  type: qdrant
  distance: cosine
  location: server
  url: https://qdrant.example.com
```

The non-secret server URL belongs in YAML. Set ``QDRANT_API_KEY`` in the environment
when the server requires authentication; it is optional for an unauthenticated server.
Do not put either provider's API key in a RAG configuration file.

Re-index the corpus after changing the splitter, embedder, vector-store type,
embedding dimensions, or distance metric. These options affect chunk or vector
compatibility, so an index created with the previous configuration must not be reused.

## Testing

- **Unit tests** (fast, no external services):
  ```bash
  uv run pytest -m "not integration" -v
  ```
- **Integration tests** (run only with the configured external services available):
  ```bash
  uv run pytest -m integration -v
  ```
  Local Qdrant checks use temporary storage. OpenAI smoke and composition tests run
  when ``OPENAI_API_KEY`` is set. The server lifecycle test runs when ``QDRANT_URL``
  is set, uses ``QDRANT_API_KEY`` when present, creates a unique collection, and cleans
  it up. The provider endpoints must be reachable, and the end-to-end workflow requires
  Ollama and its configured model.

## Code Quality Checks
```bash
black --check .
isort --check-only .
ruff check .
mypy .
```
Or run the complete formatting, import, lint, type, and non-integration suite:

```bash
./scripts/check.sh
```

## Data & Persistence
- **Corpora** – Each immediate subdirectory of ``CORPORA_DIR`` is a corpus;
  Markdown and text documents must live inside that directory.
- **Vector store** – Chroma defaults to ``storage/chroma``. Collection names are
  derived at runtime and never configured or stored in SQL. A local Qdrant profile uses
  its configured repository-relative or absolute path; server mode uses its configured
  HTTP(S) URL and optional environment-provided ``QDRANT_API_KEY``.
- **SQLite DB** – Sources, experiments, immutable configs, and runs are stored at
  ``db/genai_template.sqlite3`` (`settings.DATABASE_URL`).

Upgrading from the prototype schema does not migrate existing data. Old
Chroma/Qdrant collections are no longer referenced because their names do not
match the deterministic scheme. They are intentionally left in place; delete
them manually only when their data is no longer needed.

## Configuration Highlights (`src/genai_template/config/settings.py`)
- ``API_BASE_URL = "http://127.0.0.1:8000"``
- ``API_URL_PREFIX = "/api/v1"``
- ``LLM_MODEL = "gpt-oss:20b-cloud"`` – default Ollama model.
- ``OLLAMA_BASE_URL`` – optional explicit Ollama endpoint. When unset, the app
  uses a reachable localhost service or, in WSL NAT mode, discovers the current
  Windows-host gateway automatically.
- ``EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"``
- Other tunable knobs: vector-store base location, chunk size/overlap, top‑k
  retrieval, and distance metric.

### Ollama from WSL NAT

No host IP needs to be committed or kept in ``.env``. When localhost does not
serve Ollama, the app derives WSL's current default-route gateway and connects
to that Windows-host address. Set ``OLLAMA_BASE_URL`` only to select a custom
endpoint or bypass discovery.

The Windows Ollama service must be reachable from WSL on port 11434. If you
change its ``OLLAMA_HOST`` binding to expose it beyond localhost, restrict the
Windows firewall to trusted networks; do not expose the service publicly.

## Extending the Template
The repository is deliberately modular:
- Add new **vector store factories** in ``src/genai_template/factories/vector_store_factory.py``.
- Plug in alternative **language‑model wrappers** via ``src/genai_template/factories/llm_factory.py``.
- Extend the **prompt templates** in ``src/genai_template/prompts/``.
- Add portable **RAG configurations** under
  ``src/genai_template/experiments/configs`` and register their resolved values
  through the API or evaluation script.

---

For deeper details, see the automatically generated `AGENTS.md` which lists non‑obvious commands, settings, and gotchas for coding agent harnesses.
