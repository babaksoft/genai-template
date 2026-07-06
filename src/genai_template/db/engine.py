"""Database engine configuration."""

from sqlalchemy import Engine, create_engine

from genai_template.config.settings import DATABASE_URL

engine: Engine = create_engine(
    DATABASE_URL,
    echo=False,
)
