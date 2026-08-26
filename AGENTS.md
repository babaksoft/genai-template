# AGENTS.md – Quick‑Reference for OpenCode Agents

**Only the facts that an agent would otherwise miss**

---

## Project Setup
- Install the package **and** development extras in one step (required for linting, type‑checking and tests):
  ```bash
  pip install -c constraints.txt -e .
  pip install -c constraints.txt -e ".[dev]"
  ```
- The project requires Python 3.12 (declared in `pyproject.toml`).
- Environment variables are loaded from a top‑level ``.env`` file (e.g. `OPENAI_API_KEY`, `LANGFUSE_*`, `TAVILY_API_KEY`).
- Log files are written to `logs/genai_template.log`; the directory is created automatically via the `settings` module.

---

## Running the API
- The FastAPI application lives in `src/genai_template/api/main.py` and is mounted under the prefix stored in `settings.API_URL_PREFIX` (default `/api/v1`).
- Start the server with **uvicorn** (the canonical command used in the CI and docs):
  ```bash
  python -m uvicorn genai_template.api.main:app --host 0.0.0.0 --port 8000
  ```
- The UI (`src/genai_template/ui/streamlit_app.py`) expects the API at `http://127.0.0.1:8000`; it can be launched with:
  ```bash
  streamlit run src/genai_template/ui/streamlit_app.py
  ```

---

## Test Suite
- Unit tests are run in CI with:
  ```bash
  pytest -m "not integration" -v
  ```
  (excludes the `@pytest.mark.integration` suite which needs external services.)
- To run **only** the integration tests (requires a running Ollama instance and any vector store you configure):
  ```bash
  pytest -m integration -v
  ```
- The integration test `test_rag_service_workflow.py` creates a temporary Chroma store (`tmp_path`) and uses `OllamaLanguageModel`. Ensure the Ollama server is reachable at the URL defined in `settings.OLLAMA_BASE_URL` (default `http://localhost:11434`).

---

## Data & Storage
- Corpus files are placed under the top‑level `data/` directory; the ingestion script (`src/genai_template/ingest.py`) indexes everything in `settings.DATA_DIR`.
- Vector store persistence defaults to `storage/chroma/` (see `settings.CHROMA_PERSIST_DIR`).
- SQLite database for experiment metadata lives at `db/genai_template.db` (path from `settings.DATABASE_URL`).

---

## Command‑line Utilities
- **Baseline ingestion** (used for evaluation):
  ```bash
  python -m genai_template.ingest
  ```
  (Runs the `main()` function defined in `ingest.py` after configuring logging.)

---

## Conventional Workflow
1. **Install** core + dev deps (see *Project Setup*).
2. **Run lint / type checks** before committing:
   ```bash
   black --check .
   isort --check-only .
   ruff check .
   mypy .
   ```
3. **Run unit tests** (`pytest -m "not integration"`).
4. **Start API** (`uvicorn …`) and, optionally, **UI** (`streamlit run …`).
5. **Run integration tests** only when external services (Ollama, Chroma, etc.) are available.

---

## Gotchas & Agent‑Specific Tips
- The API base URL and prefix are *hard‑coded* in `settings`; agents must use `settings.API_BASE_URL + settings.API_URL_PREFIX` when constructing request URLs.
- The UI imports `ApiClient` from `src/genai_template/ui/api_client.py`; the client expects the same base URL.
- Integration tests create a **temporary Chroma collection**; they do **not** touch the persistent `CHROMA_PERSIST_DIR`.
- The `settings.REPO_ROOT` is calculated relative to this file (`config/settings.py`); any path manipulations that assume the repo root must use that constant.
- `OllamaLanguageModel` reads the model name from `settings.LLM_MODEL`; changing the model requires updating that setting **and** restarting any long‑running processes.

---

*This file is intentionally terse – it contains only the non‑obvious commands, paths, and conventions that an OpenCode agent would otherwise have to infer.*