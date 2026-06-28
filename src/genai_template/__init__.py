from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PKG_ROOT = Path(__file__).resolve().parent.parent

DOC_DIR = REPO_ROOT / "data"
DATA_DIR = PKG_ROOT / "chroma_db"

LLM_NAME = "gemma4:e4b"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

SIMILARITY_CUTOFF = 0.75
