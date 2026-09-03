"""Unit tests for the experiment service."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from genai_template.db.base import Base
from genai_template.db.models import Experiment, Run, Source
from genai_template.schemas import RunMetrics
from genai_template.services import ExperimentService


def create_service() -> ExperimentService:
    """Create an experiment service backed by an in-memory database."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )

    return ExperimentService(session_factory)


def create_metrics() -> RunMetrics:
    """Create sample run metrics."""

    return RunMetrics(
        query="What is the capital of France?",
        embedding_model="BAAI/bge-base-en-v1.5",
        vector_store="Chroma",
        llm_model="llama3.2",
        top_k=5,
        retrieved_chunks=2,
        best_distance=0.11,
        worst_distance=0.29,
        context_length=250,
        prompt_length=420,
        response_length=120,
        retrieval_time=0.015,
        generation_time=0.850,
        total_time=0.875,
    )


def create_source(service: ExperimentService) -> Source:
    """Create a source that can be assigned to test runs.

    Args:
        service:
            Experiment service with the target database session factory.

    Returns:
        Persisted source.
    """

    with service._session_factory() as session:
        source = Source(
            name="test-source",
            directory="/corpora/test-source",
            collection_name="source-test-source",
            documents_indexed=1,
            chunks_indexed=1,
            indexing_time=0.1,
        )
        session.add(source)
        session.commit()
        session.refresh(source)

    return source


def test_start_run_creates_experiment_when_missing() -> None:
    """Starting the first run should create the configured experiment."""

    service = create_service()
    source = create_source(service)
    run = service.start_run(experiment_name="default", source_id=source.id)

    with service._session_factory() as session:
        experiments = list(session.scalars(select(Experiment)))
        runs = list(session.scalars(select(Run)))

    assert len(experiments) == 1
    assert len(runs) == 1

    assert run.id == runs[0].id
    assert runs[0].experiment_id == experiments[0].id
    assert runs[0].source_id == source.id
    assert runs[0].started_at is not None
    assert runs[0].finished_at is None


def test_start_run_reuses_existing_experiment() -> None:
    """Starting multiple runs should reuse the same experiment."""

    service = create_service()
    source = create_source(service)
    first = service.start_run(experiment_name="default", source_id=source.id)
    second = service.start_run(experiment_name="default", source_id=source.id)

    with service._session_factory() as session:
        experiments = list(session.scalars(select(Experiment)))
        runs = list(session.scalars(select(Run)))

    assert len(experiments) == 1
    assert len(runs) == 2

    assert first.experiment_id == second.experiment_id


def test_complete_run_updates_metrics() -> None:
    """Completing a run should persist collected metrics."""

    service = create_service()
    metrics = create_metrics()
    source = create_source(service)

    run = service.start_run(experiment_name="default", source_id=source.id)
    service.complete_run(run, metrics)

    with service._session_factory() as session:
        persisted = session.get(Run, run.id)

        assert persisted is not None

        assert persisted.finished_at is not None

        assert persisted.query == metrics.query
        assert persisted.embedding_model == metrics.embedding_model
        assert persisted.vector_store == metrics.vector_store
        assert persisted.llm_model == metrics.llm_model

        assert persisted.top_k == metrics.top_k
        assert persisted.retrieved_chunks == metrics.retrieved_chunks

        assert persisted.best_distance == metrics.best_distance
        assert persisted.worst_distance == metrics.worst_distance

        assert persisted.context_length == metrics.context_length
        assert persisted.prompt_length == metrics.prompt_length
        assert persisted.response_length == metrics.response_length

        assert persisted.retrieval_time == metrics.retrieval_time
        assert persisted.generation_time == metrics.generation_time
        assert persisted.total_time == metrics.total_time


def test_summarize_empty_experiment() -> None:
    """Summarize an experiment without runs."""

    service = create_service()
    with service._session_factory() as session:
        session.add(Experiment(name="default"))
        session.commit()

    summary = service.summarize_experiment("default")

    assert summary.experiment_name == "default"
    assert summary.run_count == 0

    assert summary.average_retrieval_time == 0.0
    assert summary.average_generation_time == 0.0
    assert summary.average_total_time == 0.0

    assert summary.average_retrieved_chunks == 0.0

    assert summary.average_context_length == 0.0
    assert summary.average_prompt_length == 0.0
    assert summary.average_response_length == 0.0

    assert summary.best_distance is None
    assert summary.worst_distance is None


def test_summarize_experiment() -> None:
    """Summarize multiple completed runs."""

    service = create_service()
    source = create_source(service)
    run1 = service.start_run(experiment_name="default", source_id=source.id)
    service.complete_run(
        run1,
        create_metrics(),
    )

    metrics = create_metrics().model_copy(
        update={
            "retrieved_chunks": 4,
            "best_distance": 0.20,
            "worst_distance": 0.40,
            "retrieval_time": 0.030,
            "generation_time": 1.000,
            "total_time": 1.030,
            "context_length": 300,
            "prompt_length": 500,
            "response_length": 150,
        }
    )

    run2 = service.start_run(experiment_name="default", source_id=source.id)
    service.complete_run(
        run2,
        metrics,
    )

    summary = service.summarize_experiment("default")

    assert summary.run_count == 2

    assert summary.average_retrieved_chunks == 3.0

    assert summary.average_retrieval_time == 0.0225
    assert summary.average_generation_time == 0.925
    assert summary.average_total_time == 0.9525

    assert summary.average_context_length == 275.0
    assert summary.average_prompt_length == 460.0
    assert summary.average_response_length == 135.0

    assert summary.best_distance == 0.11
    assert summary.worst_distance == 0.40
