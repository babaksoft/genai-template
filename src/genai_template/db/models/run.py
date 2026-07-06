"""Run database model."""

from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from genai_template.db.base import Base
from genai_template.utils.datetime import utc_now


class Run(Base):
    """Represents a single execution of an experiment."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"),
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        default=utc_now,
        nullable=False,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        nullable=True,
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embedding_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    vector_store: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    llm_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    top_k: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    retrieved_chunks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    best_distance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    worst_distance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    context_length: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    prompt_length: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    retrieval_time: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    generation_time: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    total_time: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    response_length: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
