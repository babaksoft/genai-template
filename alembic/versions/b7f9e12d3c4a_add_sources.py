"""add sources

Revision ID: b7f9e12d3c4a
Revises: a4421c217bea
Create Date: 2026-09-02 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7f9e12d3c4a"
down_revision: Union[str, Sequence[str], None] = "a4421c217bea"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the corpus sources table."""

    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("directory", sa.String(length=1024), nullable=False),
        sa.Column("collection_name", sa.String(length=128), nullable=False),
        sa.Column("documents_indexed", sa.Integer(), nullable=False),
        sa.Column("chunks_indexed", sa.Integer(), nullable=False),
        sa.Column("indexed_at", sa.DateTime(), nullable=False),
        sa.Column("indexing_time", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_name"),
        sa.UniqueConstraint("directory"),
        sa.UniqueConstraint("name"),
    )


def downgrade() -> None:
    """Remove the corpus sources table."""

    op.drop_table("sources")
