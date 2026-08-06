import logging
from pathlib import Path

from genai_template.common.types import VectorDistance

# Global settings
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PKG_ROOT = Path(__file__).resolve().parent.parent

# API settings
URL_PREFIX = "/api/v1"

# Logging settings
LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "genai_template.log"
LOG_LEVEL = logging.DEBUG

# Database settings
DATABASE_URL = "sqlite:///db/genai_template.db"

# Document settings
DATA_DIR = REPO_ROOT / "data"

# LLM settings
LLM_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

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
