"""Smoke tests for the persistence layer."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from genai_template.db import create_session
from genai_template.db.models import Experiment, Run


@pytest.mark.integration
def test_can_persist_experiment() -> None:
    """Persist and retrieve an experiment."""

    session = create_session()
    transaction = session.begin()

    try:
        experiment = Experiment(
            name="Smoke Test",
            description="Persistence smoke test.",
        )

        session.add(experiment)
        session.flush()

        experiment_id = experiment.id

        retrieved = session.scalar(
            select(Experiment).where(
                Experiment.id == experiment_id,
            )
        )

        assert retrieved is not None
        assert retrieved.name == experiment.name
        assert retrieved.description == experiment.description
        assert retrieved.created_at is not None

    finally:
        transaction.rollback()
        session.close()


@pytest.mark.integration
def test_can_persist_run() -> None:
    """Persist and retrieve a run."""

    session = create_session()
    transaction = session.begin()

    try:
        experiment = Experiment(
            name="Smoke Test",
        )

        session.add(experiment)
        session.flush()

        run = Run(
            experiment_id=experiment.id,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            query="What is the capital of France?",
            embedding_model="BAAI/bge-base-en-v1.5",
            vector_store="Chroma",
            llm_model="llama3.2",
            top_k=5,
            retrieved_chunks=2,
            best_distance=0.14,
            worst_distance=0.33,
            context_length=842,
            prompt_length=1045,
            retrieval_time=0.021,
            generation_time=0.941,
            total_time=0.962,
            response_length=173,
        )

        session.add(run)
        session.flush()

        run_id = run.id

        retrieved = session.scalar(
            select(Run).where(
                Run.id == run_id,
            )
        )

        assert retrieved is not None
        assert retrieved.experiment_id == experiment.id
        assert retrieved.query == run.query
        assert retrieved.retrieved_chunks == run.retrieved_chunks
        assert retrieved.total_time == run.total_time

    finally:
        transaction.rollback()
        session.close()
