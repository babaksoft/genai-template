"""Tests for experiment configuration provenance migration."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def test_migration_preserves_legacy_experiments(tmp_path: Path) -> None:
    """Nullable provenance columns should leave legacy rows readable."""

    database_path = tmp_path / "migration.sqlite3"
    database_url = f"sqlite:///{database_path}"
    alembic_config = Config()
    alembic_config.set_main_option("script_location", "alembic")
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_config, "c9d4e8f1a2b3")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO experiments (name, description, created_at) "
                "VALUES (:name, NULL, :created_at)"
            ),
            {"name": "legacy", "created_at": "2026-09-11 00:00:00"},
        )

    command.upgrade(alembic_config, "head")

    columns = {
        column["name"]: column for column in inspect(engine).get_columns("experiments")
    }
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT name, config_json, config_fingerprint "
                "FROM experiments WHERE name = :name"
            ),
            {"name": "legacy"},
        ).one()

    assert columns["config_json"]["nullable"] is True
    assert columns["config_fingerprint"]["nullable"] is True
    assert row._tuple() == ("legacy", None, None)
