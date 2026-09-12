"""Persisted RAG configuration database model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_template.db.base import Base
from genai_template.utils.datetime import utc_now

if TYPE_CHECKING:
    from genai_template.db.models.run import Run


class RagConfig(Base):
    """Represent an immutable canonical RAG configuration."""

    __tablename__ = "rag_configs"

    id: Mapped[int] = mapped_column(primary_key=True)

    config_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )

    config_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        default=utc_now,
        nullable=False,
    )

    runs: Mapped[list[Run]] = relationship(
        back_populates="rag_config",
        passive_deletes=True,
    )
