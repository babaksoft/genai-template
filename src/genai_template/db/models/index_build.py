"""Durable index-build attempt database model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_template.db import Base
from genai_template.schemas import IndexBuildStatus
from genai_template.utils import utc_now

if TYPE_CHECKING:
    from genai_template.db.models.rag_config import RagConfig
    from genai_template.db.models.source import Source


class IndexBuild(Base):
    """Represent one durable attempt to rebuild a deterministic index."""

    __tablename__ = "index_builds"
    __table_args__ = (
        CheckConstraint(
            "status IN ('building', 'succeeded', 'failed')",
            name="ck_index_builds_status",
        ),
        Index(
            "ix_index_builds_lookup",
            "source_id",
            "collection_name",
            "index_fingerprint",
            "started_at",
            "id",
        ),
        Index(
            "ix_index_builds_success_lookup",
            "source_id",
            "collection_name",
            "index_fingerprint",
            "status",
            "started_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Unique index-build attempt identifier.",
    )
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        doc="Registered source rebuilt by this attempt.",
    )
    rag_config_id: Mapped[int] = mapped_column(
        ForeignKey("rag_configs.id", ondelete="RESTRICT"),
        nullable=False,
        doc="Immutable RAG configuration requesting this build.",
    )
    collection_name: Mapped[str] = mapped_column(
        String(60),
        nullable=False,
        doc="Deterministic vector collection selected by the source and index.",
    )
    index_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Complete fingerprint of index-affecting configuration.",
    )
    corpus_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="Validated manifest corpus fingerprint, absent for generic sources.",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=IndexBuildStatus.BUILDING.value,
        doc="Current building, succeeded, or failed lifecycle state.",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        doc="UTC timestamp at which the durable attempt started.",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        doc="UTC timestamp at which the attempt reached a terminal state.",
    )
    indexing_duration: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Measured indexing duration in seconds for a successful attempt.",
    )
    document_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of loaded documents verified on successful completion.",
    )
    chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of stored chunks verified on successful completion.",
    )
    failure_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="Bounded machine-readable category for a failed attempt.",
    )
    failure_detail: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        doc="Bounded sanitized operator-facing failure summary.",
    )

    source: Mapped[Source] = relationship(
        back_populates="index_builds",
        doc="Registered source associated with this attempt.",
    )
    rag_config: Mapped[RagConfig] = relationship(
        back_populates="index_builds",
        doc="Immutable RAG configuration associated with this attempt.",
    )
