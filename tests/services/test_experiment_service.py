"""Unit tests for the experiment registry service."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from genai_template.config import load_rag_config
from genai_template.db.base import Base
from genai_template.db.models import Run, Source
from genai_template.schemas import RunMetrics
from genai_template.services import ExperimentService, RagConfigService


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


def create_metrics(query: str = "Question", total_time: float = 0.5) -> RunMetrics:
    """Create successful execution metrics for persistence tests.

    Args:
        query:
            Query represented by the metrics.
        total_time:
            Total execution duration.

    Returns:
        Valid run metrics.
    """

    return RunMetrics(
        query=query,
        embedding_model="embedder",
        vector_store="Chroma",
        llm_model="llm",
        top_k=5,
        retrieved_chunks=2,
        best_distance=0.1,
        worst_distance=0.4,
        context_length=100,
        prompt_length=120,
        response_length=30,
        retrieval_time=0.1,
        generation_time=0.4,
        total_time=total_time,
    )


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


def test_start_and_complete_run_persist_canonical_foreign_keys() -> None:
    """Run lifecycle should persist both IDs and nullable-to-complete metrics."""

    service, factory = create_service()
    source = create_source(factory)
    experiment = service.create_experiment(source.id, "trial")
    config = RagConfigService(factory).register_config(load_rag_config())

    run = service.start_run(experiment.id, config.id, "Question")
    with factory() as session:
        unfinished = session.get(Run, run.id)
        assert unfinished is not None
        assert unfinished.experiment_id == experiment.id
        assert unfinished.rag_config_id == config.id
        assert unfinished.finished_at is None
        assert unfinished.total_time is None

    completed = service.complete_run(run, create_metrics())

    assert completed.finished_at is not None
    assert completed.retrieved_chunks == 2
    assert completed.total_time == 0.5


def test_summary_spans_configs_and_excludes_unfinished_runs() -> None:
    """Experiment summaries should aggregate completed runs across configs."""

    service, factory = create_service()
    source = create_source(factory)
    experiment = service.create_experiment(source.id, "trial")
    config_service = RagConfigService(factory)
    first_config = config_service.register_config(load_rag_config())
    changed = load_rag_config().model_copy(
        update={
            "retrieval": load_rag_config().retrieval.model_copy(update={"top_k": 9})
        }
    )
    second_config = config_service.register_config(changed)
    first = service.start_run(experiment.id, first_config.id, "First")
    second = service.start_run(experiment.id, second_config.id, "Second")
    service.start_run(experiment.id, second_config.id, "Failed")
    service.complete_run(first, create_metrics("First", 0.5))
    service.complete_run(second, create_metrics("Second", 1.5))

    combined = service.summarize_experiment(experiment.id)
    filtered = service.summarize_experiment(experiment.id, first_config.id)

    assert combined.experiment_id == experiment.id
    assert combined.rag_config_id is None
    assert combined.run_count == 2
    assert combined.average_total_time == 1.0
    assert filtered.rag_config_id == first_config.id
    assert filtered.run_count == 1
    assert filtered.average_total_time == 0.5
