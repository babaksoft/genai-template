"""Run database model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_template.db.base import Base
from genai_template.utils.datetime import utc_now

if TYPE_CHECKING:
    from genai_template.db.models.experiment import Experiment
    from genai_template.db.models.rag_config import RagConfig


class Run(Base):
    """Represents a single execution of an experiment."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rag_config_id: Mapped[int] = mapped_column(
        ForeignKey("rag_configs.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        default=utc_now,
        nullable=False,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        nullable=True,
    )

    retrieved_chunks: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    best_distance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    worst_distance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    context_length: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    prompt_length: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    retrieval_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    generation_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    total_time: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    response_length: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    experiment: Mapped[Experiment] = relationship(
        back_populates="runs",
    )

    rag_config: Mapped[RagConfig] = relationship(
        back_populates="runs",
    )
