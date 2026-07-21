"""Unit tests for the experiment service."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from genai_template.db.base import Base
from genai_template.db.models import Experiment, Run
from genai_template.schemas import RunMetrics
from genai_template.services import ExperimentService


def create_service() -> ExperimentService:
    """Create an experiment service backed by an in-memory database."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
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


def test_start_run_creates_experiment_when_missing() -> None:
    """Starting the first run should create the configured experiment."""

    service = create_service()
    run = service.start_run()

    with service._session_factory() as session:
        experiments = session.query(Experiment).all()
        runs = session.query(Run).all()

    assert len(experiments) == 1
    assert len(runs) == 1

    assert run.id == runs[0].id
    assert runs[0].experiment_id == experiments[0].id
    assert runs[0].started_at is not None
    assert runs[0].finished_at is None


def test_start_run_reuses_existing_experiment() -> None:
    """Starting multiple runs should reuse the same experiment."""

    service = create_service()
    first = service.start_run()
    second = service.start_run()

    with service._session_factory() as session:
        experiments = session.query(Experiment).all()
        runs = session.query(Run).all()

    assert len(experiments) == 1
    assert len(runs) == 2

    assert first.experiment_id == second.experiment_id


def test_complete_run_updates_metrics() -> None:
    """Completing a run should persist collected metrics."""

    service = create_service()
    metrics = create_metrics()

    run = service.start_run()
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
