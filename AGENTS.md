# AGENTS.md – Quick‑Reference for Coding Agents

**Only the facts that an agent would otherwise miss**

---

## Project Setup
- This project is developed in **Ubuntu 24.04 on WSL** and managed with `uv`.
- Install the locked project environment, including development tools, with:
  ```bash
  uv sync --all-groups
  ```
- The project requires Python 3.12 (declared in `pyproject.toml`).
- Environment variables are loaded from a top‑level ``.env`` file (e.g. `OPENAI_API_KEY`, `QDRANT_API_KEY`, `QDRANT_URL`, `PHOENIX_ENABLED`).
- Log files are written to `logs/genai_template.log`; the directory is created automatically via the `settings` module.

---

## Quality standards
- Use four-space indentation.
- Add complete type hints to every function.
- Add complete Google-style docstrings to every class, method, and function.
- In `Args:` and `Raises:` sections, put each item name on its own line and indent
  its description beneath it.
- Omit `Args:` when there are no arguments. Omit `Returns:` when a function or
  method returns `None`.
- Use absolute package-level imports everywhere; do not use relative imports.
- Name modules and functions in `snake_case`, classes in `PascalCase`, test modules
  as `test_<behavior>.py`, and test cases as `test_<expected_behavior>()`.
- Give every field on a persistence model a concise `doc` description, including
  mapped columns and relationships.
- Give every new Pydantic model a complete Google-style `Attributes:` section and
  every field a `description` argument.

---

## Running the API
- The FastAPI application lives in `src/genai_template/api/main.py` and is mounted under the prefix stored in `settings.API_URL_PREFIX` (default `/api/v1`).
- Start the server with **uvicorn**:
  ```bash
  uv run uvicorn genai_template.api.main:app --host 0.0.0.0 --port 8000
  ```
- Start local Phoenix tracing with `uv run phoenix serve`; enable API request tracing with `PHOENIX_ENABLED=true`.
- The UI (`src/genai_template/ui/streamlit_app.py`) expects the API at `http://localhost:8000`; it can be launched with:
  ```bash
  uv run streamlit run src/genai_template/ui/streamlit_app.py
  ```

---

## Quality Checks & Test Suite
- Run all formatting, import, lint, type, and unit-test checks with:
  ```bash
  PHOENIX_ENABLED=false ./scripts/check.sh
  ```
  The script runs Black, isort, Ruff, mypy, and `pytest -m "not integration" -v` through `uv`; it excludes the `@pytest.mark.integration` suite.
- API tests require a running Phoenix server when tracing is enabled. In local
  environments without Phoenix, either skip `tests/api/` or run the full suite
  with `PHOENIX_ENABLED=false`.
- To run **only** the integration tests (requires a running Ollama instance and any vector store you configure):
  ```bash
  uv run pytest -m integration -v
  ```
- The integration test `test_rag_service_workflow.py` creates a temporary Chroma store (`tmp_path`) and uses `OllamaLanguageModel`. Ensure the Ollama server is reachable at the URL defined in `settings.OLLAMA_BASE_URL` (default `http://localhost:11434`).

---

## Data & Storage
- Corpus files are placed in immediate subdirectories of top-level `data/` (`settings.CORPORA_DIR`). Registering a source does not index it; rebuild its selected deterministic index through the API.
- Vector store persistence defaults to `storage/chroma/` (see `settings.CHROMA_PERSIST_DIR`). Collection names are derived from source IDs and index fingerprints, not configured manually.
- SQLite storage for sources, experiments, RAG configs, and runs lives at `db/genai_template.sqlite3` (path from `settings.DATABASE_URL`).
- Legacy Chroma/Qdrant collections may be orphaned by the deterministic naming scheme. Never delete them automatically; cleanup is optional and manual.

---

## Command‑line Utilities
- **Baseline evaluation** (registers the source/config/experiment and builds a missing index):
  ```bash
  uv run python -m genai_template.evaluation.evaluators.baseline_eval
  ```
- Add `--reindex` to explicitly rebuild the selected deterministic collection.

---

## Conventional Workflow
1. **Sync** dependencies with `uv sync --all-groups`.
2. **Run quality checks and unit tests** with
   `PHOENIX_ENABLED=false ./scripts/check.sh` before committing unless a Phoenix
   server is already running.
3. **Start API** (`uv run uvicorn …`) and, optionally, **UI** (`uv run streamlit …`). Register a source, create an experiment, select/register a config, then explicitly rebuild that source/config index.
4. **Run integration tests** only when external services (Ollama, Chroma, etc.) are available.

---

## Gotchas & Agent‑Specific Tips
- The API base URL and prefix are *hard‑coded* in `settings`; agents must use `settings.API_BASE_URL + settings.API_URL_PREFIX` when constructing request URLs.
- The UI imports `ApiClient` from `src/genai_template/ui/api_client.py`; the client expects the same base URL.
- The RAG workflow integration test creates a **temporary Chroma collection**; it does **not** touch the persistent `CHROMA_PERSIST_DIR`.
- Execution APIs take canonical `experiment_id` and `rag_config_id`. Resolve a source only through the selected experiment.
- The `settings.REPO_ROOT` is calculated relative to this file (`config/settings.py`); any path manipulations that assume the repo root must use that constant.
- `OllamaLanguageModel` reads the model name from `settings.LLM_MODEL`; changing the model requires updating that setting **and** restarting any long‑running processes.

---

*This file is intentionally terse – it contains only the non‑obvious commands, paths, and conventions that an OpenCode agent would otherwise have to infer.*
