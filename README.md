# GenAI Template Project

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://github.com/babaksoft/genai-template/raw/refs/heads/master/pyproject.toml)
![Category: RAG](https://img.shields.io/badge/category-RAG-orange)
![Category: Agentic](https://img.shields.io/badge/category-Agentic-orange)
![Framework: LlamaIndex](https://img.shields.io/badge/framework-LlamaIndex-orange)
![License](https://img.shields.io/github/license/babaksoft/genai-template)
![CI Status](https://img.shields.io/github/actions/workflow/status/babaksoft/genai-template/ci.yml)

---

## Overview

`genai-template` is a starter kit for building **retrieval‑augmented generation (RAG)** and **agentic** applications.  It wires together a FastAPI backend, a Streamlit UI, a configurable vector store (default = Chroma), and a flexible component system for readers, splitters, embedders, language models, prompts, and context building.

### Core Packages
- **`src/genai_template/config`** – Global settings (`settings.py`) and logging configuration.
- **`src/genai_template/components`** – Pluggable building blocks:
  - `readers` (e.g., `TextReader`)
  - `splitters` (sentence splitter)
  - `embeddings` (FastEmbed wrapper)
  - `language_models` (Ollama wrapper)
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
  - `vector/chroma_store.py` – Chroma vector DB.
  - `kv`, `document`, `index` (placeholders for future stores).
- **`src/genai_template/api`** – FastAPI app (`main.py`) with routers for:
  - `answer` – POST `/answer` returns generated answer + metrics.
  - `ingest` – POST `/ingest` runs baseline ingestion.
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
LANGFUSE_SECRET_KEY=…
TAVILY_API_KEY=…
```
The project reads these variables via ``python‑dotenv``.

### 3️⃣ Run the API server
```bash
python -m uvicorn genai_template.api.main:app \
    --host 0.0.0.0 --port 8000
```
The API lives under the prefix defined in ``settings.API_URL_PREFIX`` (default `/api/v1`).

### 4️⃣ (Optional) Launch the UI
```bash
streamlit run src/genai_template/ui/streamlit_app.py
```
The UI expects the API at ``http://127.0.0.1:8000``.

### 5️⃣ Ingest a corpus (baseline)
```bash
python -m genai_template.ingest
```
This indexes all markdown files under ``data/`` into the default Chroma store (`storage/chroma`).

## Testing

- **Unit tests** (fast, no external services):
  ```bash
  pytest -m "not integration" -v
  ```
- **Integration tests** (require Ollama and a vector store):
  ```bash
  pytest -m integration -v
  ```
  The integration suite creates a temporary Chroma collection and uses ``OllamaLanguageModel``; ensure Ollama is reachable at ``settings.OLLAMA_BASE_URL`` (default ``http://localhost:11434``).

## Code Quality Checks
```bash
black --check .
isort --check-only .
ruff check .
mypy .
```
Run these before committing.

## Data & Persistence
- **Corpus** – Markdown files under the top‑level ``data/`` directory.
- **Vector store** – Chroma files under ``storage/chroma`` (configurable via ``settings.CHROMA_PERSIST_DIR``).
- **SQLite DB** – Experiment metadata stored at ``db/genai_template.db`` (`settings.DATABASE_URL`).

## Configuration Highlights (`src/genai_template/config/settings.py`)
- ``API_BASE_URL = "http://127.0.0.1:8000"``
- ``API_URL_PREFIX = "/api/v1"``
- ``LLM_MODEL = "llama3.2:3b"`` – default Ollama model.
- ``EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"``
- ``CHROMA_COLLECTION = "documents"``
- Other tunable knobs: chunk size/overlap, top‑k retrieval, distance metric.

## Extending the Template
The repository is deliberately modular:
- Add new **vector store factories** in ``src/genai_template/factories/vector_store_factory.py``.
- Plug in alternative **language‑model wrappers** via ``src/genai_template/factories/llm_factory.py``.
- Extend the **prompt templates** in ``src/genai_template/prompts/``.
- Define new **experiments** under ``src/genai_template/experiments`` and run them with the evaluation script.

---

For deeper details, see the automatically generated `AGENTS.md` which lists non‑obvious commands, settings, and gotchas for OpenCode agents.
