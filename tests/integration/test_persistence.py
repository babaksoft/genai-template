"""Smoke tests for the persistence layer."""

import pytest
from sqlalchemy import select

from genai_template.db import create_session
from genai_template.db.models import Experiment, RagConfig, Run, Source


@pytest.mark.integration
def test_can_persist_experiment() -> None:
    """Persist and retrieve an experiment."""

    session = create_session()
    transaction = session.begin()

    try:
        source = Source(
            name="Smoke Test Source",
            directory="/corpora/smoke-test",
        )
        experiment = Experiment(
            source=source,
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
        assert retrieved.source_id == source.id
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
        source = Source(
            name="Smoke Test Source",
            directory="/corpora/smoke-test",
        )
        experiment = Experiment(
            source=source,
            name="Smoke Test",
        )
        rag_config = RagConfig(
            config_fingerprint="a" * 64,
            config_json='{"version":1}',
        )
        session.add_all([experiment, rag_config])
        session.flush()

        run = Run(
            experiment_id=experiment.id,
            rag_config_id=rag_config.id,
            query="What is the capital of France?",
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
        assert retrieved.rag_config_id == rag_config.id
        assert retrieved.query == run.query
        assert retrieved.finished_at is None
        assert retrieved.retrieved_chunks is None
        assert retrieved.total_time is None

    finally:
        transaction.rollback()
        session.close()
