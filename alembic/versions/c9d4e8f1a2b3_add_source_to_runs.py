"""add source to runs

Revision ID: c9d4e8f1a2b3
Revises: b7f9e12d3c4a
Create Date: 2026-09-03 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c9d4e8f1a2b3"
down_revision: Union[str, Sequence[str], None] = "b7f9e12d3c4a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Require every run to reference an ingested source."""

    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(sa.Column("source_id", sa.Integer(), nullable=False))
        batch_op.create_foreign_key(
            "fk_runs_source_id_sources",
            "sources",
            ["source_id"],
            ["id"],
        )


def downgrade() -> None:
    """Remove the mandatory source reference from runs."""

    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_constraint("fk_runs_source_id_sources", type_="foreignkey")
        batch_op.drop_column("source_id")
