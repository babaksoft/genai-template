import logging
from pathlib import Path

# Global settings
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PKG_ROOT = Path(__file__).resolve().parent.parent

# Logging settings
LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "genai_template.log"
LOG_LEVEL = logging.DEBUG

# Document settings
DATA_DIR = REPO_ROOT / "data"

# LLM settings
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding settings
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Chunking settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20

# Retrieval settings
TOP_K = 5

# Chroma settings
CHROMA_COLLECTION = "documents"
CHROMA_PERSIST_DIR = "./storage/chroma"
