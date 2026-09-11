"""Add experiment configuration provenance.

Revision ID: d2a6f38e9b41
Revises: c9d4e8f1a2b3
Create Date: 2026-09-11 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d2a6f38e9b41"
down_revision: Union[str, Sequence[str], None] = "c9d4e8f1a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable resolved configuration data to experiments."""

    with op.batch_alter_table("experiments") as batch_op:
        batch_op.add_column(sa.Column("config_json", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("config_fingerprint", sa.String(length=64), nullable=True)
        )
        batch_op.create_index(
            "ix_experiments_config_fingerprint",
            ["config_fingerprint"],
            unique=False,
        )


def downgrade() -> None:
    """Remove resolved configuration data from experiments."""

    with op.batch_alter_table("experiments") as batch_op:
        batch_op.drop_index("ix_experiments_config_fingerprint")
        batch_op.drop_column("config_fingerprint")
        batch_op.drop_column("config_json")
