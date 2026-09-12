"""Corpus source database model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_template.db.base import Base
from genai_template.utils.datetime import utc_now

if TYPE_CHECKING:
    from genai_template.db.models.experiment import Experiment


class Source(Base):
    """Represent a registered document corpus."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    directory: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    experiments: Mapped[list[Experiment]] = relationship(
        back_populates="source",
        passive_deletes=True,
    )
