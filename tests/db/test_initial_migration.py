"""Tests for the squashed initial database migration."""

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def create_alembic_config(database_path: Path) -> Config:
    """Create an Alembic configuration for a temporary SQLite database.

    Args:
        database_path:
            Path to the temporary database file.

    Returns:
        Alembic configuration targeting the temporary database.
    """

    alembic_config = Config()
    alembic_config.set_main_option("script_location", "alembic")
    alembic_config.set_main_option(
        "sqlalchemy.url",
        f"sqlite:///{database_path}",
    )
    return alembic_config


def columns_by_name(inspector: Any, table_name: str) -> dict[str, dict[str, Any]]:
    """Return reflected columns keyed by column name.

    Args:
        inspector:
            SQLAlchemy database inspector.
        table_name:
            Name of the table to reflect.

    Returns:
        Reflected column metadata keyed by name.
    """

    return {column["name"]: column for column in inspector.get_columns(table_name)}


def test_initial_migration_creates_redesigned_schema(tmp_path: Path) -> None:
    """The squashed migration should create only the redesigned tables."""

    database_path = tmp_path / "migration.sqlite3"
    alembic_config = create_alembic_config(database_path)

    command.upgrade(alembic_config, "head")

    inspector = inspect(create_engine(f"sqlite:///{database_path}"))
    assert set(inspector.get_table_names()) == {
        "alembic_version",
        "experiments",
        "rag_configs",
        "runs",
        "sources",
    }
    assert set(columns_by_name(inspector, "sources")) == {
        "id",
        "name",
        "directory",
        "created_at",
    }
    assert set(columns_by_name(inspector, "experiments")) == {
        "id",
        "source_id",
        "name",
        "description",
        "created_at",
    }
    assert set(columns_by_name(inspector, "rag_configs")) == {
        "id",
        "config_fingerprint",
        "config_json",
        "created_at",
    }
    assert set(columns_by_name(inspector, "runs")) == {
        "id",
        "experiment_id",
        "rag_config_id",
        "query",
        "started_at",
        "finished_at",
        "retrieved_chunks",
        "best_distance",
        "worst_distance",
        "context_length",
        "prompt_length",
        "retrieval_time",
        "generation_time",
        "total_time",
        "response_length",
    }


def test_initial_migration_enforces_identity_and_references(tmp_path: Path) -> None:
    """The schema should expose required unique, foreign-key, and index rules."""

    database_path = tmp_path / "constraints.sqlite3"
    alembic_config = create_alembic_config(database_path)
    command.upgrade(alembic_config, "head")

    inspector = inspect(create_engine(f"sqlite:///{database_path}"))
    source_uniques = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("sources")
    }
    config_uniques = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("rag_configs")
    }
    experiment_foreign_keys = inspector.get_foreign_keys("experiments")
    run_foreign_keys = {
        tuple(foreign_key["constrained_columns"]): foreign_key
        for foreign_key in inspector.get_foreign_keys("runs")
    }
    experiment_indexes = {
        tuple(index["column_names"]) for index in inspector.get_indexes("experiments")
    }
    run_indexes = {
        tuple(index["column_names"]) for index in inspector.get_indexes("runs")
    }

    assert source_uniques == {("directory",), ("name",)}
    assert config_uniques == {("config_fingerprint",)}
    assert experiment_foreign_keys[0]["options"]["ondelete"] == "RESTRICT"
    assert run_foreign_keys[("experiment_id",)]["options"]["ondelete"] == "CASCADE"
    assert run_foreign_keys[("rag_config_id",)]["options"]["ondelete"] == "RESTRICT"
    assert ("source_id",) in experiment_indexes
    assert {("experiment_id",), ("rag_config_id",)} <= run_indexes


def test_initial_migration_makes_incomplete_run_metrics_nullable(
    tmp_path: Path,
) -> None:
    """Every result metric should remain null until a run completes."""

    database_path = tmp_path / "nullable.sqlite3"
    alembic_config = create_alembic_config(database_path)
    command.upgrade(alembic_config, "head")

    inspector = inspect(create_engine(f"sqlite:///{database_path}"))
    columns = columns_by_name(inspector, "runs")
    metric_names = {
        "retrieved_chunks",
        "best_distance",
        "worst_distance",
        "context_length",
        "prompt_length",
        "retrieval_time",
        "generation_time",
        "total_time",
        "response_length",
    }

    assert all(columns[name]["nullable"] for name in metric_names)
    assert columns["finished_at"]["nullable"] is True
    assert columns["query"]["nullable"] is False
