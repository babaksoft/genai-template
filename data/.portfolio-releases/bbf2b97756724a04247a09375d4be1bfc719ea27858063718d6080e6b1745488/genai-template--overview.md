# GenAI Template — Overview

## Purpose

- \`genai\-template\` is a starter kit for building retrieval‑augmented generation \(RAG\) and agentic applications\. It wires together a FastAPI backend, a Streamlit UI, configurable Chroma or Qdrant vector storage, and a flexible component system for readers, splitters, embedders, language models, prompts, and context building\. The project provides services for registering corpora, experiments, and immutable RAG configurations, and exposes endpoints for answering queries, managing experiments, and health checks\.

Evidence:
- `README.md`

## Capabilities

- The template supports end‑to‑end RAG workflows: reading documents, splitting them into chunks, embedding with FastEmbed or OpenAI, storing vectors in Chroma or Qdrant, retrieving relevant chunks, building prompts, and generating answers with Ollama or OpenAI LLMs\. It offers a modular pipeline architecture, experiment tracking, deterministic index naming, and observability via Phoenix/OpenTelemetry\. The API returns structured answers with inline citation labels, metrics, and source metadata, while the Streamlit UI provides a user interface for corpus registration and query interaction\.

Evidence:
- `README.md`
- `pyproject.toml`

## Entry Points

- The FastAPI application is instantiated in \`src/genai\_template/api/main\.py\`\. It mounts routers for \`/answer\`, \`/experiments\`, \`/rag\-configs\`, \`/sources\`, and \`/health\`, and configures lifecycle events for logging and optional Phoenix tracing\. The API serves as the primary entry point for programmatic interaction, while the Streamlit script \`src/genai\_template/ui/streamlit\_app\.py\` offers a graphical interface that communicates with this backend\.

Evidence:
- `src/genai_template/api/main.py`

## Technology Choices

- The project is built in Python 3\.12 and relies on a curated set of libraries: FastAPI for the web framework, LlamaIndex for indexing and retrieval, FastEmbed and OpenAI for embeddings, Ollama and OpenAI for language models, Chroma and Qdrant for vector stores, and Phoenix/OpenTelemetry for observability\. Dependency management is handled via \`pyproject\.toml\` with Hatchling, and the codebase follows PEP 621 metadata conventions\.

Evidence:
- `pyproject.toml`
