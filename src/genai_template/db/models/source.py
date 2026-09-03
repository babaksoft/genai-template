"""Corpus source database model."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from genai_template.db.base import Base
from genai_template.utils.datetime import utc_now


class Source(Base):
    """Represents an ingested document corpus."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    directory: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)

    collection_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
    )

    documents_indexed: Mapped[int] = mapped_column(Integer, nullable=False)

    chunks_indexed: Mapped[int] = mapped_column(Integer, nullable=False)

    indexed_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    indexing_time: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
