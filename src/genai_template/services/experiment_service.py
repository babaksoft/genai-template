"""Experiment tracking service."""

import logging
import statistics
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from genai_template.config import RagConfig, canonical_config_json, config_fingerprint
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
        source_id: int,
        config: RagConfig | None = None,
    ) -> Run:
        """Start a new run for the supplied experiment.

        Args:
            experiment_name:
                Experiment name for the new run.
            source_id:
                Identifier of the source used by the new run.
            config:
                Optional resolved configuration to associate with the run's
                experiment. When omitted, a legacy name-only experiment is
                used.

        Returns:
            Newly created run.

        Note:
            Most required fields in the newly created run record are
            intentionally initialized from ORM defaults. These fields
            are later populated by ``complete_run`` method.
        """

        with self._session_factory() as session:
            experiment = (
                self._register_config(session, config, experiment_name)
                if config
                else None
            )
            if experiment is None:
                experiment = self._get_or_create_legacy_experiment(
                    session=session,
                    experiment_name=experiment_name,
                )

            run = Run(
                experiment_id=experiment.id,
                source_id=source_id,
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

    def register_experiment(
        self,
        config: RagConfig,
        experiment_name: str,
    ) -> Experiment:
        """Register or resolve an experiment's immutable configuration.

        Args:
            config:
                Fully resolved RAG configuration to persist.
            experiment_name:
                Display name of the experiment associated with the config.

        Returns:
            Existing or newly registered experiment for the configuration.
        """

        with self._session_factory() as session:
            return self._register_config(session, config, experiment_name)

    def resolve_experiment(
        self,
        experiment_name: str,
        fingerprint: str | None = None,
    ) -> Experiment:
        """Resolve an experiment by name and optional configuration fingerprint.

        Args:
            experiment_name:
                Experiment name to resolve.
            fingerprint:
                Optional exact configuration fingerprint. Name-only resolution
                is rejected when more than one matching record exists.

        Returns:
            The matching experiment.

        Raises:
            ValueError:
                If no experiment matches or name-only resolution is ambiguous.
        """

        with self._session_factory() as session:
            return self._resolve_experiment(session, experiment_name, fingerprint)

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
        fingerprint: str | None = None,
    ) -> ExperimentSummary:
        """Summarize all runs for an experiment.

        Args:
            experiment_name:
                Experiment name.
            fingerprint:
                Optional exact configuration fingerprint.

        Returns:
            Summary statistics for the experiment.

        Raises:
            ValueError:
                If the experiment does not exist or name-only lookup is
                ambiguous.
        """

        with self._session_factory() as session:
            experiment = self._resolve_experiment(
                session,
                experiment_name,
                fingerprint,
            )

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

    def _register_config(
        self,
        session: Session,
        config: RagConfig,
        experiment_name: str,
    ) -> Experiment:
        """Register a resolved configuration within an existing session.

        Args:
            session:
                SQLAlchemy session for persistence.
            config:
                Fully resolved configuration to register.
            experiment_name:
                Display name of the experiment associated with the config.

        Returns:
            Existing or newly registered experiment.
        """

        snapshot = canonical_config_json(config)
        fingerprint = config_fingerprint(config)
        experiment = session.scalar(
            select(Experiment).where(
                Experiment.name == experiment_name,
                Experiment.config_fingerprint == fingerprint,
            )
        )
        if experiment is not None:
            if experiment.config_json != snapshot:
                raise ValueError(
                    "Stored experiment configuration does not match its fingerprint."
                )
            return experiment

        experiment = Experiment(
            name=experiment_name,
            config_json=snapshot,
            config_fingerprint=fingerprint,
        )
        session.add(experiment)
        session.commit()
        session.refresh(experiment)

        logger.info(
            "Registered experiment '%s' with config fingerprint %s.",
            experiment.name,
            fingerprint,
        )

        return experiment

    def _resolve_experiment(
        self,
        session: Session,
        experiment_name: str,
        fingerprint: str | None,
    ) -> Experiment:
        """Resolve one experiment within an existing session.

        Args:
            session:
                SQLAlchemy session for persistence.
            experiment_name:
                Experiment name to resolve.
            fingerprint:
                Optional exact configuration fingerprint.

        Returns:
            Matching experiment.

        Raises:
            ValueError:
                If no experiment matches or name-only resolution is ambiguous.
        """

        statement = select(Experiment).where(Experiment.name == experiment_name)
        if fingerprint is not None:
            statement = statement.where(
                Experiment.config_fingerprint == fingerprint,
            )

        experiments = list(session.scalars(statement))
        if not experiments:
            identity = (
                f" with config fingerprint '{fingerprint}'"
                if fingerprint is not None
                else ""
            )
            raise ValueError(
                f"Experiment '{experiment_name}'{identity} does not exist."
            )
        if len(experiments) > 1:
            raise ValueError(
                f"Experiment '{experiment_name}' is ambiguous; provide a "
                "config fingerprint."
            )

        return experiments[0]

    def _get_or_create_legacy_experiment(
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
            select(Experiment).where(
                Experiment.name == experiment_name,
                Experiment.config_fingerprint.is_(None),
            )
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
