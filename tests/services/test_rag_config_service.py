"""Unit tests for the immutable RAG configuration registry."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from genai_template.config import (
    canonical_config_json,
    config_fingerprint,
    load_rag_config,
)
from genai_template.db.base import Base
from genai_template.db.models import RagConfig as RagConfigRecord
from genai_template.services import RagConfigService


def create_service() -> tuple[RagConfigService, sessionmaker[Session]]:
    """Create a config registry backed by an in-memory database.

    Returns:
        Service and its session factory.
    """

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return RagConfigService(factory), factory


def test_registration_is_idempotent_and_round_trips_config() -> None:
    """Repeated canonical configs should resolve to one database row."""

    service, _ = create_service()
    config = load_rag_config()

    first = service.register_config(config)
    second = service.register_config(config)

    assert first.id == second.id
    assert first.config_json == canonical_config_json(config)
    assert first.config_fingerprint == config_fingerprint(config)
    assert service.parse_config(first) == config
    assert [record.id for record in service.list_configs()] == [first.id]


def test_distinct_configs_receive_distinct_ids() -> None:
    """A changed full config should create a separate registry entry."""

    service, _ = create_service()
    config = load_rag_config()
    changed = config.model_copy(
        update={"retrieval": config.retrieval.model_copy(update={"top_k": 9})}
    )

    assert service.register_config(config).id != service.register_config(changed).id


def test_fingerprint_reuse_verifies_canonical_json() -> None:
    """A fingerprint collision must not silently return mismatched JSON."""

    service, factory = create_service()
    config = load_rag_config()
    with factory() as session:
        session.add(
            RagConfigRecord(
                config_fingerprint=config_fingerprint(config),
                config_json="{}",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="does not match its fingerprint"):
        service.register_config(config)


def test_get_config_rejects_missing_id() -> None:
    """Lookup should report an unknown canonical identifier."""

    service, _ = create_service()

    with pytest.raises(ValueError, match="RAG config 8 does not exist"):
        service.get_config(8)
