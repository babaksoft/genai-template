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
  - `RagService` – end‑to‑end answer generation.
  - `ExperimentService` – persists experiment metadata.
- **`src/genai_template/stores`** – Persistence layers:
  - `vector` – Chroma and local/server Qdrant vector stores.
  - `kv`, `document`, `index` (placeholders for future stores).
- **`src/genai_template/api`** – FastAPI app (`main.py`) with routers for:
  - `answer` – POST `/answer` returns generated answer + metrics.
  - `sources` – browse, ingest, and refresh isolated document corpora.
  - `health` – GET `/health` health‑check endpoint.
- **`src/genai_template/ui`** – Streamlit front‑end (`streamlit_app.py`) that talks to the API.
- **`src/genai_template/evaluation`** – Baseline evaluation script and metrics.
- **`src/genai_template/experiments`** – Experiment configuration utilities.
- **`src/genai_template/ingest.py`** – CLI entry point for corpus ingestion.

## Getting Started

### 1️⃣ Install dependencies
```bash
# Core package (editable) and runtime deps
pip install -c constraints.txt -e .

# Development extras (lint, type‑check, tests)
pip install -c constraints.txt -e ".[dev]"
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
python -m uvicorn genai_template.api.main:app \
    --host 0.0.0.0 --port 8000
```
The API lives under the prefix defined in ``settings.API_URL_PREFIX`` (default `/api/v1`).

### Local Phoenix tracing

Phoenix is included as a project dependency. In a separate terminal, start the local
server with:

```bash
uv run phoenix serve
```

Then set ``PHOENIX_ENABLED=true`` before starting the API. FastAPI request traces, retrieval,
Chroma search, context/prompt construction, and LlamaIndex embedding/LLM spans are sent
to ``http://localhost:6006/v1/traces`` and appear at
``http://localhost:6006`` under the ``genai-template`` project. Override the defaults
with ``PHOENIX_COLLECTOR_ENDPOINT`` and ``PHOENIX_PROJECT_NAME`` if needed.

Phoenix tracing is intended for local experimentation: traces retain full user queries,
retrieved chunk contents, prompts, and generated answers. Do not enable it with sensitive
data unless the local Phoenix storage is appropriately protected.

### 4️⃣ (Optional) Launch the UI
```bash
streamlit run src/genai_template/ui/streamlit_app.py
```
The UI expects the API at ``http://localhost:8000``.

### 5️⃣ Prepare and ingest a corpus

Place each corpus in its own directory under ``data/`` (the default
``CORPORA_DIR``). Documents cannot be placed directly in the corpus root:

```text
data/
  baseline/
  product-docs/
  handbook/
```

Use the **Sources** expander in the Streamlit sidebar to ingest a prepared
directory and select it for questions. Each source has its own Chroma
collection, and **Refresh active source** rebuilds it from the directory.

The baseline CLI utility remains available for evaluation setup:

```bash
python -m genai_template.ingest
```
It indexes ``data/baseline/`` into the baseline Chroma collection.

Run a retrieval evaluation with the defaults, or override the resolved RAG
configuration, corpus, and dataset from the command line:

```bash
uv run python -m genai_template.evaluation.evaluators.baseline_eval
uv run python -m genai_template.evaluation.evaluators.baseline_eval \
  --config src/genai_template/experiments/configs/baseline.yml \
  --corpus data/baseline \
  --dataset src/genai_template/evaluation/datasets/baseline-eval.json
```

Evaluation collections are keyed by the indexing configuration and corpus
contents, so a populated matching index is reused. Pass ``--reindex`` to delete
and rebuild only that resolved collection before calculating the metrics.

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
  collection_name: openai-markdown-documents
  distance: cosine
  location: server
  url: https://qdrant.example.com
```

The non-secret server URL belongs in YAML. Set ``QDRANT_API_KEY`` in the environment
when the server requires authentication; it is optional for an unauthenticated server.
Do not put either provider's API key in an experiment profile.

Re-index the corpus after changing the splitter, embedder, vector-store type,
embedding dimensions, or distance metric. These options affect chunk or vector
compatibility, so an index created with the previous configuration must not be reused.

## Testing

- **Unit tests** (fast, no external services):
  ```bash
  pytest -m "not integration" -v
  ```
- **Integration tests** (individual tests skip when their provider is unavailable):
  ```bash
  pytest -m integration -v
  ```
  Local Qdrant checks use temporary storage. OpenAI smoke and composition tests require
  ``OPENAI_API_KEY``. The server lifecycle test runs only when ``QDRANT_URL`` is set,
  uses ``QDRANT_API_KEY`` when present, creates a unique collection, and cleans it up.
  The existing end-to-end workflow test requires Ollama to be reachable.

## Code Quality Checks
```bash
black --check .
isort --check-only .
ruff check .
mypy .
```
Run these before committing.

## Data & Persistence
- **Corpora** – Each immediate subdirectory of ``CORPORA_DIR`` is a corpus;
  Markdown and text documents must live inside that directory.
- **Vector store** – Chroma defaults to ``storage/chroma``. A local Qdrant profile uses
  its configured repository-relative or absolute path; server mode uses its configured
  HTTP(S) URL and optional environment-provided ``QDRANT_API_KEY``.
- **SQLite DB** – Experiment metadata stored at ``db/genai_template.sqlite3`` (`settings.DATABASE_URL`).

## Configuration Highlights (`src/genai_template/config/settings.py`)
- ``API_BASE_URL = "http://127.0.0.1:8000"``
- ``API_URL_PREFIX = "/api/v1"``
- ``LLM_MODEL = "gpt-oss:20b-cloud"`` – default Ollama model.
- ``OLLAMA_BASE_URL`` – optional explicit Ollama endpoint. When unset, the app
  uses a reachable localhost service or, in WSL NAT mode, discovers the current
  Windows-host gateway automatically.
- ``EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"``
- ``CHROMA_COLLECTION = "documents"``
- Other tunable knobs: chunk size/overlap, top‑k retrieval, distance metric.

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
- Define new **experiments** under ``src/genai_template/experiments`` and run them with the evaluation script.

---

For deeper details, see the automatically generated `AGENTS.md` which lists non‑obvious commands, settings, and gotchas for OpenCode agents.
