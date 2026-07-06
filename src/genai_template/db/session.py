"""Database session factory."""

from sqlalchemy.orm import Session, sessionmaker

from genai_template.db.engine import engine

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def create_session() -> Session:
    """Create a new database session.

    Returns:
        A SQLAlchemy session.
    """

    return SessionLocal()
