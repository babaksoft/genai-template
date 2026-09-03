"""Experiment tracking service."""

import logging
import statistics
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from genai_template.db.models import Experiment, Run
from genai_template.schemas import ExperimentSummary, RunMetrics
from genai_template.utils import utc_now

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
                Factory that creates database sessions.
        """

        self._session_factory = session_factory

    def start_run(
        self,
        experiment_name: str,
    ) -> Run:
        """Start a new run for the configured experiment.

        Args:
            experiment_name:
                Experiment name for the new run.

        Returns:
            Newly created run.

        Note:
            Most required fields in the newly created run record are
            intentionally initialized from ORM defaults. These fields
            are later populated by ``complete_run`` method.
        """

        with self._session_factory() as session:
            experiment = self._get_or_create_experiment(
                session=session,
                experiment_name=experiment_name,
            )

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

    def summarize_experiment(
        self,
        experiment_name: str,
    ) -> ExperimentSummary:
        """Summarize all runs for an experiment.

        Args:
            experiment_name:
                Experiment name.

        Returns:
            Summary statistics for the experiment.

        Raises:
            ValueError:
                If the experiment does not exist.
        """

        with self._session_factory() as session:
            experiment = session.scalar(
                select(Experiment).where(Experiment.name == experiment_name)
            )

            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' does not exist.")

            runs = session.scalars(
                select(Run).where(Run.experiment_id == experiment.id)
            ).all()

        if not runs:
            return ExperimentSummary(
                experiment_name=experiment_name,
                run_count=0,
                average_retrieval_time=0.0,
                average_generation_time=0.0,
                average_total_time=0.0,
                average_retrieved_chunks=0.0,
                average_context_length=0.0,
                average_prompt_length=0.0,
                average_response_length=0.0,
                best_distance=None,
                worst_distance=None,
            )

        distances = [
            distance for run in runs if (distance := run.best_distance) is not None
        ]
        distances.extend(
            distance for run in runs if (distance := run.worst_distance) is not None
        )

        return ExperimentSummary(
            experiment_name=experiment_name,
            run_count=len(runs),
            average_retrieval_time=statistics.fmean(run.retrieval_time for run in runs),
            average_generation_time=statistics.fmean(
                run.generation_time for run in runs
            ),
            average_total_time=statistics.fmean(run.total_time for run in runs),
            average_retrieved_chunks=statistics.fmean(
                run.retrieved_chunks for run in runs
            ),
            average_context_length=statistics.fmean(run.context_length for run in runs),
            average_prompt_length=statistics.fmean(run.prompt_length for run in runs),
            average_response_length=statistics.fmean(
                run.response_length for run in runs
            ),
            best_distance=min(distances) if distances else None,
            worst_distance=max(distances) if distances else None,
        )

    def _get_or_create_experiment(
        self,
        session: Session,
        experiment_name: str,
    ) -> Experiment:
        """Get or create the configured experiment.

        Args:
            session:
                SQLAlchemy session for persistence.
            experiment_name:
                Name of experiment to get or create.

        Returns:
            An existing or a newly created experiment.
        """

        experiment = session.scalar(
            select(Experiment).where(Experiment.name == experiment_name)
        )
        if experiment is not None:
            return experiment

        experiment = Experiment(name=experiment_name)

        session.add(experiment)
        session.commit()
        session.refresh(experiment)

        logger.info(
            "Created experiment '%s'.",
            experiment.name,
        )

        return experiment
