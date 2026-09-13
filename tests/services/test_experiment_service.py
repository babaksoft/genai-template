"""Unit tests for the experiment registry service."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from genai_template.db.base import Base
from genai_template.db.models import Source
from genai_template.services import ExperimentService


def create_service() -> tuple[ExperimentService, sessionmaker[Session]]:
    """Create an experiment registry backed by an in-memory database.

    Returns:
        Service and its session factory.
    """

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return ExperimentService(factory), factory


def create_source(factory: sessionmaker[Session]) -> Source:
    """Persist a source for experiment tests.

    Args:
        factory:
            Test database session factory.

    Returns:
        Persisted source.
    """

    with factory() as session:
        source = Source(name="docs", directory="/corpora/docs")
        session.add(source)
        session.commit()
        session.refresh(source)
        return source


def test_create_and_get_experiment_by_id() -> None:
    """Experiments should use their database identifier as identity."""

    service, factory = create_service()
    source = create_source(factory)

    created = service.create_experiment(source.id, "baseline", "Initial trial")

    assert service.get_experiment(created.id).id == created.id
    assert created.source_id == source.id
    assert created.description == "Initial trial"


def test_create_experiment_rejects_missing_source() -> None:
    """Creation should validate the selected source before insertion."""

    service, _ = create_service()

    with pytest.raises(ValueError, match="Source 99 does not exist"):
        service.create_experiment(99, "baseline")


def test_duplicate_experiment_names_are_allowed() -> None:
    """Display names should not act as experiment identity."""

    service, factory = create_service()
    source = create_source(factory)

    first = service.create_experiment(source.id, "trial")
    second = service.create_experiment(source.id, "trial")

    assert first.id != second.id
    assert [item.id for item in service.list_experiments()] == [first.id, second.id]


def test_get_experiment_rejects_missing_id() -> None:
    """Lookup should report an unknown canonical identifier."""

    service, _ = create_service()

    with pytest.raises(ValueError, match="Experiment 7 does not exist"):
        service.get_experiment(7)
