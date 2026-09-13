"""Experiment registry service."""

import logging
from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from genai_template.db.models import Experiment, RagConfig, Run, Source
from genai_template.schemas import ExperimentSummary, RunMetrics
from genai_template.utils import utc_now

logger = logging.getLogger(__name__)


class ExperimentService:
    """Create and retrieve source-bound experiments."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Initialize the experiment registry.

        Args:
            session_factory:
                Factory that creates database sessions.
        """

        self._session_factory = session_factory

    def create_experiment(
        self,
        source_id: int,
        name: str,
        description: str | None = None,
    ) -> Experiment:
        """Create an experiment for an existing source.

        Args:
            source_id:
                Identifier of the source used by the experiment.
            name:
                Display name for the experiment.
            description:
                Optional human-readable description.

        Returns:
            Newly created experiment.

        Raises:
            ValueError:
                If the selected source does not exist.
        """

        with self._session_factory() as session:
            if session.get(Source, source_id) is None:
                raise ValueError(f"Source {source_id} does not exist.")

            experiment = Experiment(
                source_id=source_id,
                name=name,
                description=description,
            )
            session.add(experiment)
            session.commit()
            session.refresh(experiment)

        logger.info("Created experiment %d for source %d.", experiment.id, source_id)
        return experiment

    def list_experiments(self) -> list[Experiment]:
        """List registered experiments in identifier order.

        Returns:
            All registered experiments.
        """

        with self._session_factory() as session:
            return list(session.scalars(select(Experiment).order_by(Experiment.id)))

    def get_experiment(self, experiment_id: int) -> Experiment:
        """Get an experiment by its canonical identifier.

        Args:
            experiment_id:
                Identifier of the requested experiment.

        Returns:
            Matching experiment.

        Raises:
            ValueError:
                If the experiment does not exist.
        """

        with self._session_factory() as session:
            experiment = session.get(Experiment, experiment_id)

        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} does not exist.")
        return experiment

    def start_run(
        self,
        experiment_id: int,
        rag_config_id: int,
        query: str,
    ) -> Run:
        """Create an unfinished run for canonical registry identifiers.

        Args:
            experiment_id:
                Identifier of the experiment being executed.
            rag_config_id:
                Identifier of the immutable configuration being executed.
            query:
                User query associated with the execution.

        Returns:
            Newly persisted unfinished run.

        Raises:
            ValueError:
                If the experiment or configuration does not exist.
        """

        with self._session_factory() as session:
            if session.get(Experiment, experiment_id) is None:
                raise ValueError(f"Experiment {experiment_id} does not exist.")
            if session.get(RagConfig, rag_config_id) is None:
                raise ValueError(f"RAG config {rag_config_id} does not exist.")

            run = Run(
                experiment_id=experiment_id,
                rag_config_id=rag_config_id,
                query=query,
            )
            session.add(run)
            session.commit()
            session.refresh(run)

        logger.info(
            "Started run %d for experiment %d with RAG config %d.",
            run.id,
            experiment_id,
            rag_config_id,
        )
        return run

    def complete_run(self, run: Run, metrics: RunMetrics) -> Run:
        """Persist successful result metrics and finish a run.

        Args:
            run:
                Previously persisted unfinished run.
            metrics:
                Metrics collected by the successful execution.

        Returns:
            Updated completed run.

        Raises:
            ValueError:
                If the run no longer exists or is already complete.
        """

        with self._session_factory() as session:
            persisted = session.get(Run, run.id)
            if persisted is None:
                raise ValueError(f"Run {run.id} does not exist.")
            if persisted.finished_at is not None:
                raise ValueError(f"Run {run.id} is already complete.")

            persisted.finished_at = utc_now()
            persisted.retrieved_chunks = metrics.retrieved_chunks
            persisted.best_distance = metrics.best_distance
            persisted.worst_distance = metrics.worst_distance
            persisted.context_length = metrics.context_length
            persisted.prompt_length = metrics.prompt_length
            persisted.retrieval_time = metrics.retrieval_time
            persisted.generation_time = metrics.generation_time
            persisted.total_time = metrics.total_time
            persisted.response_length = metrics.response_length
            session.commit()
            session.refresh(persisted)

        logger.info("Completed run %d.", persisted.id)
        return persisted

    def summarize_experiment(
        self,
        experiment_id: int,
        rag_config_id: int | None = None,
    ) -> ExperimentSummary:
        """Summarize completed runs for an experiment across configurations.

        Args:
            experiment_id:
                Canonical experiment identifier.
            rag_config_id:
                Optional configuration identifier used to restrict the summary.

        Returns:
            Aggregate metrics for matching completed runs.

        Raises:
            ValueError:
                If the experiment or requested configuration does not exist.
        """

        with self._session_factory() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment {experiment_id} does not exist.")
            if (
                rag_config_id is not None
                and session.get(RagConfig, rag_config_id) is None
            ):
                raise ValueError(f"RAG config {rag_config_id} does not exist.")

            statement = select(
                func.count(Run.id),
                func.avg(Run.retrieval_time),
                func.avg(Run.generation_time),
                func.avg(Run.total_time),
                func.avg(Run.retrieved_chunks),
                func.avg(Run.context_length),
                func.avg(Run.prompt_length),
                func.avg(Run.response_length),
                func.min(Run.best_distance),
                func.max(Run.worst_distance),
            ).where(
                Run.experiment_id == experiment_id,
                Run.finished_at.is_not(None),
            )
            if rag_config_id is not None:
                statement = statement.where(Run.rag_config_id == rag_config_id)
            row = session.execute(statement).one()

        return ExperimentSummary(
            experiment_id=experiment_id,
            experiment_name=experiment.name,
            rag_config_id=rag_config_id,
            run_count=row[0],
            average_retrieval_time=row[1] or 0.0,
            average_generation_time=row[2] or 0.0,
            average_total_time=row[3] or 0.0,
            average_retrieved_chunks=row[4] or 0.0,
            average_context_length=row[5] or 0.0,
            average_prompt_length=row[6] or 0.0,
            average_response_length=row[7] or 0.0,
            best_distance=row[8],
            worst_distance=row[9],
        )
