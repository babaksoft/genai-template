"""Add durable index-build attempts.

Revision ID: f4a8c2d19e63
Revises: e7c4b2a19d6f
Create Date: 2026-09-30 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f4a8c2d19e63"
down_revision: str | Sequence[str] | None = "e7c4b2a19d6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the durable index-build attempt table and lookup indexes."""

    op.create_table(
        "index_builds",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("rag_config_id", sa.Integer(), nullable=False),
        sa.Column("collection_name", sa.String(length=60), nullable=False),
        sa.Column("index_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("corpus_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("indexing_duration", sa.Float(), nullable=True),
        sa.Column("document_count", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_detail", sa.String(length=512), nullable=True),
        sa.CheckConstraint(
            "status IN ('building', 'succeeded', 'failed')",
            name="ck_index_builds_status",
        ),
        sa.ForeignKeyConstraint(
            ["rag_config_id"],
            ["rag_configs.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["sources.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_index_builds_lookup",
        "index_builds",
        [
            "source_id",
            "collection_name",
            "index_fingerprint",
            "started_at",
            "id",
        ],
        unique=False,
    )
    op.create_index(
        "ix_index_builds_success_lookup",
        "index_builds",
        [
            "source_id",
            "collection_name",
            "index_fingerprint",
            "status",
            "started_at",
            "id",
        ],
        unique=False,
    )


def downgrade() -> None:
    """Remove durable index-build attempts."""

    op.drop_index("ix_index_builds_success_lookup", table_name="index_builds")
    op.drop_index("ix_index_builds_lookup", table_name="index_builds")
    op.drop_table("index_builds")
