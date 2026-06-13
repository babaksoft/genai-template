from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = PKG_ROOT / "data"
DATA_DIR = PKG_ROOT / "chroma_db"

LLM_NAME = "gpt-4o-mini"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
