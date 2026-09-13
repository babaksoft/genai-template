"""Experiment registry service."""

import logging
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from genai_template.db.models import Experiment, Source

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
