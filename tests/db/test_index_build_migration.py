"""Tests for the durable index-build migration and model metadata."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from genai_template.db.models import IndexBuild


def create_alembic_config(database_path: Path) -> Config:
    """Create an Alembic configuration for a temporary SQLite database.

    Args:
        database_path:
            Path to the temporary database file.

    Returns:
        Alembic configuration targeting the temporary database.
    """

    config = Config()
    config.set_main_option("script_location", "alembic")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config


def test_index_build_migration_defines_lifecycle_contract(tmp_path: Path) -> None:
    """The migration should expose the planned fields, references, and indexes."""

    database_path = tmp_path / "index-builds.sqlite3"
    config = create_alembic_config(database_path)
    command.upgrade(config, "head")
    inspector = inspect(create_engine(f"sqlite:///{database_path}"))

    columns = {
        column["name"]: column for column in inspector.get_columns("index_builds")
    }
    foreign_keys = {
        tuple(foreign_key["constrained_columns"]): foreign_key
        for foreign_key in inspector.get_foreign_keys("index_builds")
    }
    indexes = {
        index["name"]: tuple(index["column_names"])
        for index in inspector.get_indexes("index_builds")
    }
    checks = inspector.get_check_constraints("index_builds")

    assert columns["corpus_fingerprint"]["nullable"] is True
    assert columns["finished_at"]["nullable"] is True
    assert columns["document_count"]["nullable"] is True
    assert columns["chunk_count"]["nullable"] is True
    assert columns["failure_code"]["nullable"] is True
    assert columns["failure_detail"]["nullable"] is True
    assert columns["source_id"]["nullable"] is False
    assert columns["status"]["nullable"] is False
    assert foreign_keys[("source_id",)]["options"]["ondelete"] == "CASCADE"
    assert foreign_keys[("rag_config_id",)]["options"]["ondelete"] == "RESTRICT"
    assert indexes["ix_index_builds_lookup"] == (
        "source_id",
        "collection_name",
        "index_fingerprint",
        "started_at",
        "id",
    )
    assert indexes["ix_index_builds_success_lookup"] == (
        "source_id",
        "collection_name",
        "index_fingerprint",
        "status",
        "started_at",
        "id",
    )
    assert checks[0]["name"] == "ck_index_builds_status"
    assert "succeeded" in checks[0]["sqltext"]


def test_index_build_migration_downgrades_independently(tmp_path: Path) -> None:
    """Downgrade should remove only the Slice 4 table."""

    database_path = tmp_path / "downgrade.sqlite3"
    config = create_alembic_config(database_path)
    command.upgrade(config, "head")
    command.downgrade(config, "e7c4b2a19d6f")

    tables = set(inspect(create_engine(f"sqlite:///{database_path}")).get_table_names())
    assert "index_builds" not in tables
    assert {"sources", "rag_configs", "experiments", "runs"} <= tables


def test_index_build_model_documents_every_persisted_field() -> None:
    """Every persisted build field should carry concise model documentation."""

    assert all(column.doc for column in IndexBuild.__table__.columns)
