# GenAI Template — Overview

## Purpose

- The project &quot;genai\-template&quot; is a starter kit designed for building retrieval\-augmented generation \(RAG\) and agentic applications\. It integrates a FastAPI backend with a Streamlit UI, supports configurable vector storage backends like Chroma and Qdrant, and provides a modular component system for readers, splitters, embedders, language models, prompts, and context building\. The system facilitates indexing, retrieval, and chat pipelines to enable generation grounded in retrieved evidence, supporting experiments and evaluations with immutable RAG configurations and persistent experiment tracking\.

Evidence:
- `README.md`

## Capabilities

- The project offers capabilities including:
- \- Document corpus registration and management as sources\.  
  \- Deterministic vector index building and rebuilding for registered corpora using configurable RAG configurations\.  
  \- Document splitting into chunks via two main splitters: a Markdown header\-based splitter and a sentence\-based splitter, preserving metadata and generating deterministic chunk IDs\.  
  \- Embedding and vector storage integration supporting Chroma and Qdrant backends\.  
  \- Retrieval pipelines that embed queries and retrieve relevant chunks\.  
  \- Chat pipelines combining retrieval and synthesis for multi\-turn conversations\.  
  \- API endpoints for answering queries with generated responses linked to inline source labels, managing sources, experiments, and RAG configurations\.  
  \- Observability support with optional Phoenix tracing for detailed request and operation tracing\.  
  \- Experimentation and evaluation frameworks to run and compare different RAG configurations and corpora\.  
  \- A Streamlit UI front\-end that interacts with the API for user\-friendly operations\.

Evidence:
- `README.md`
- `src/genai_template/components/splitters/__init__.py`
- `src/genai_template/components/splitters/markdown_splitter.py`
- `src/genai_template/components/splitters/sentence_splitter.py`
- `src/genai_template/services/source_service.py`

## Entry Points

- The primary entry points for the project are:
- \- The FastAPI application defined in \`src/genai\_template/api/main\.py\`, which creates the API app, configures logging, optionally enables Phoenix tracing, and mounts routers for health checks, answer generation, experiments, RAG configurations, and source management under a common API URL prefix\.  
  \- The lifespan context manager in \`src/genai\_template/api/lifespan\.py\` that manages startup and shutdown events, including observability initialization and tracer shutdown\.  
  \- The API routes defined in \`src/genai\_template/api/routes/\_\_init\_\_\.py\` that handle HTTP requests and delegate to injected service dependencies\.  
  \- The Streamlit UI application \(referenced in README\.md\) located at \`src/genai\_template/ui/streamlit\_app\.py\` \(not in allowed evidence but mentioned in README\)\.  
  \- The source service in \`src/genai\_template/services/source\_service\.py\` which provides programmatic interfaces for source registration, listing, retrieval, and index rebuilding\.

Evidence:
- `README.md`
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/routes/__init__.py`
- `src/genai_template/services/source_service.py`

## Technology Choices

- The project uses the following technology choices:
- \- Python 3\.12 as the required Python version\.  
  \- FastAPI as the web framework for the backend API\.  
  \- Streamlit for the user interface frontend\.  
  \- LlamaIndex for document parsing, splitting, embedding, and language model integration\.  
  \- Vector stores: Chroma and Qdrant for vector persistence and retrieval\.  
  \- Embedding providers: FastEmbed and OpenAI embeddings\.  
  \- Language models: Ollama and OpenAI wrappers\.  
  \- OpenTelemetry and Phoenix for optional observability and tracing instrumentation\.  
  \- SQLAlchemy and SQLite for metadata persistence \(sources, experiments, configurations\)\.  
  \- Python\-dotenv for environment variable configuration\.  
  \- Alembic for database migrations\.  
  \- HTTPX for HTTP client operations\.  
  \- Pytest, black, isort, mypy, and ruff for testing and code quality\.  
  \- Hatchling as the build backend\.
- These choices enable modular, pluggable components and support both local and server\-based vector stores and language model providers, with optional tracing instrumentation that is dynamically imported only when enabled\.

Evidence:
- `README.md`
- `pyproject.toml`
- `src/genai_template/api/lifespan.py`
- `src/genai_template/api/main.py`
- `src/genai_template/api/observability.py`
