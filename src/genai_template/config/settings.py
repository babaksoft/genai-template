import logging
from pathlib import Path

# Global settings
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PKG_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "genai_template.log"
LOG_LEVEL = logging.DEBUG

# LLM settings
OLLAMA_ENDPOINT_URL = "http://localhost:11434"
