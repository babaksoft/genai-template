"""Experiment tracking service."""

import logging
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from genai_template.config import settings
from genai_template.db.models import Experiment, Run
from genai_template.schemas.run_metrics import RunMetrics
from genai_template.utils.datetime import utc_now

logger = logging.getLogger(__name__)


class ExperimentService:
    """Tracks RAG experiments and runs."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
    ) -> None:
        """Initialize the experiment service.

        Args:
            session_factory:
                SQLAlchemy session factory.
        """

        self._session_factory = session_factory

    def start_run(self) -> Run:
        """Start a new run for the configured experiment.

        Returns:
            Newly created run.

        Note:
            Most required fields in the newly created run record are
            intentionally initialized from ORM defaults. These fields
            are later populated by ``complete_run`` method.
        """

        with self._session_factory() as session:
            experiment = self._get_or_create_experiment(session)

            run = Run(
                experiment_id=experiment.id,
                started_at=utc_now(),
            )

            session.add(run)
            session.commit()

            logger.info(
                "Started run %d for experiment '%s'.",
                run.id,
                experiment.name,
            )

            return run

    def complete_run(
        self,
        run: Run,
        metrics: RunMetrics,
    ) -> None:
        """Complete a run.

        Args:
            run:
                Run to update.

            metrics:
                Collected run metrics.
        """

        with self._session_factory() as session:
            persisted_run = session.get(Run, run.id)

            if persisted_run is None:
                raise ValueError(f"Run {run.id} does not exist.")

            persisted_run.finished_at = utc_now()

            persisted_run.query = metrics.query

            persisted_run.embedding_model = metrics.embedding_model
            persisted_run.vector_store = metrics.vector_store
            persisted_run.llm_model = metrics.llm_model

            persisted_run.top_k = metrics.top_k
            persisted_run.retrieved_chunks = metrics.retrieved_chunks
            persisted_run.best_distance = metrics.best_distance
            persisted_run.worst_distance = metrics.worst_distance

            persisted_run.context_length = metrics.context_length
            persisted_run.prompt_length = metrics.prompt_length
            persisted_run.response_length = metrics.response_length

            persisted_run.retrieval_time = metrics.retrieval_time
            persisted_run.generation_time = metrics.generation_time
            persisted_run.total_time = metrics.total_time

            session.commit()

            logger.info(
                "Completed run %d.",
                persisted_run.id,
            )

    def _get_or_create_experiment(
        self,
        session: Session,
    ) -> Experiment:
        """Get or create the configured experiment."""

        statement = select(Experiment).where(
            Experiment.name == settings.EXPERIMENT_NAME,
        )
        experiment = session.scalar(statement)
        if experiment is not None:
            return experiment

        experiment = Experiment(
            name=settings.EXPERIMENT_NAME,
        )

        session.add(experiment)
        session.commit()
        session.refresh(experiment)

        logger.info(
            "Created experiment '%s'.",
            experiment.name,
        )

        return experiment
