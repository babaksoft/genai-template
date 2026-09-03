"""Application-wide configuration."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from genai_template.common.types import VectorDistance

# Global settings
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PKG_ROOT = Path(__file__).resolve().parent.parent

# Shell environment values take precedence over values kept in the local .env.
load_dotenv(REPO_ROOT / ".env")

# API settings
API_BASE_URL = "http://localhost:8000"
API_URL_PREFIX = "/api/v1"

# Observability settings
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
)
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "genai-template")

# Logging settings
LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "genai_template.log"
LOG_LEVEL = logging.DEBUG

# Database settings
DATABASE_URL = "sqlite:///db/genai_template.db"

# Corpora settings
CORPORA_DIR = REPO_ROOT / "data"

# LLM settings
LLM_MODEL = "gpt-oss:20b-cloud"
OLLAMA_BASE_URL = "http://localhost:11434"

# HTTP client settings
REQUEST_TIMEOUT = 180

# Embedding settings
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Chunking settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20

# Retrieval settings
TOP_K = 5

# Vector store settings
VECTOR_STORE = "Chroma"
CHROMA_COLLECTION = "documents"
CHROMA_PERSIST_DIR = REPO_ROOT / "storage" / "chroma"
CHROMA_DISTANCE = VectorDistance.COSINE

# Experiment settings
EXPERIMENT_NAME = "Baseline RAG"

# Evaluation settings
EVALUATION_DATA_DIR = PKG_ROOT / "evaluation" / "datasets"
