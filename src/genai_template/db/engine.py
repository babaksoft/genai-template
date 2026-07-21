"""Database engine configuration."""

from sqlalchemy import Engine, create_engine

from genai_template.config import settings

engine: Engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
)
