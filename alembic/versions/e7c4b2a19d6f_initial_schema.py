"""Create the initial persistence schema.

Revision ID: e7c4b2a19d6f
Revises:
Create Date: 2026-09-12 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7c4b2a19d6f"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create sources, experiments, RAG configs, and runs."""

    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("directory", sa.String(length=1024), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("directory"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "rag_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "config_fingerprint",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("config_fingerprint"),
    )
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["sources.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_experiments_source_id",
        "experiments",
        ["source_id"],
        unique=False,
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("rag_config_id", sa.Integer(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("retrieved_chunks", sa.Integer(), nullable=True),
        sa.Column("best_distance", sa.Float(), nullable=True),
        sa.Column("worst_distance", sa.Float(), nullable=True),
        sa.Column("context_length", sa.Integer(), nullable=True),
        sa.Column("prompt_length", sa.Integer(), nullable=True),
        sa.Column("retrieval_time", sa.Float(), nullable=True),
        sa.Column("generation_time", sa.Float(), nullable=True),
        sa.Column("total_time", sa.Float(), nullable=True),
        sa.Column("response_length", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["rag_config_id"],
            ["rag_configs.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_runs_experiment_id",
        "runs",
        ["experiment_id"],
        unique=False,
    )
    op.create_index(
        "ix_runs_rag_config_id",
        "runs",
        ["rag_config_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the complete persistence schema."""

    op.drop_index("ix_runs_rag_config_id", table_name="runs")
    op.drop_index("ix_runs_experiment_id", table_name="runs")
    op.drop_table("runs")
    op.drop_index("ix_experiments_source_id", table_name="experiments")
    op.drop_table("experiments")
    op.drop_table("rag_configs")
    op.drop_table("sources")
