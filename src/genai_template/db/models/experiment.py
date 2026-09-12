"""Experiment database model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_template.db.base import Base
from genai_template.utils.datetime import utc_now

if TYPE_CHECKING:
    from genai_template.db.models.run import Run
    from genai_template.db.models.source import Source


class Experiment(Base):
    """Represents an experiment."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)

    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        default=utc_now,
        nullable=False,
    )

    source: Mapped[Source] = relationship(
        back_populates="experiments",
    )

    runs: Mapped[list[Run]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
