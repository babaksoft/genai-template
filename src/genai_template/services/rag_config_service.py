"""Immutable RAG configuration registry service."""

import logging
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from genai_template.config import RagConfig, canonical_config_json, config_fingerprint
from genai_template.db.models import RagConfig as RagConfigRecord

logger = logging.getLogger(__name__)


class RagConfigService:
    """Register and retrieve immutable RAG configurations."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Initialize the configuration registry.

        Args:
            session_factory:
                Factory that creates database sessions.
        """

        self._session_factory = session_factory

    def register_config(self, config: RagConfig) -> RagConfigRecord:
        """Idempotently register a canonical RAG configuration.

        Args:
            config:
                Fully resolved configuration to register.

        Returns:
            Existing or newly persisted configuration record.

        Raises:
            ValueError:
                If a stored row reuses the fingerprint with different JSON.
        """

        snapshot = canonical_config_json(config)
        fingerprint = config_fingerprint(config)

        with self._session_factory() as session:
            existing = self._find_by_fingerprint(session, fingerprint)
            if existing is not None:
                self._verify_snapshot(existing, snapshot)
                return existing

            record = RagConfigRecord(
                config_fingerprint=fingerprint,
                config_json=snapshot,
            )
            session.add(record)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = self._find_by_fingerprint(session, fingerprint)
                if existing is None:
                    raise
                self._verify_snapshot(existing, snapshot)
                return existing

            session.refresh(record)

        logger.info("Registered RAG config %d (%s).", record.id, fingerprint)
        return record

    def list_configs(self) -> list[RagConfigRecord]:
        """List registered configurations in identifier order.

        Returns:
            All registered configuration records.
        """

        with self._session_factory() as session:
            return list(
                session.scalars(select(RagConfigRecord).order_by(RagConfigRecord.id))
            )

    def get_config(self, rag_config_id: int) -> RagConfigRecord:
        """Get a registered configuration by canonical identifier.

        Args:
            rag_config_id:
                Identifier of the requested configuration.

        Returns:
            Matching configuration record.

        Raises:
            ValueError:
                If the configuration does not exist.
        """

        with self._session_factory() as session:
            record = session.get(RagConfigRecord, rag_config_id)

        if record is None:
            raise ValueError(f"RAG config {rag_config_id} does not exist.")
        return record

    @staticmethod
    def parse_config(record: RagConfigRecord) -> RagConfig:
        """Deserialize a persisted canonical configuration.

        Args:
            record:
                Persisted configuration record.

        Returns:
            Validated RAG configuration.
        """

        return RagConfig.model_validate_json(record.config_json)

    @staticmethod
    def _find_by_fingerprint(
        session: Session, fingerprint: str
    ) -> RagConfigRecord | None:
        """Find a configuration record by fingerprint.

        Args:
            session:
                Active database session.
            fingerprint:
                Full configuration fingerprint.

        Returns:
            Matching record, if present.
        """

        return session.scalar(
            select(RagConfigRecord).where(
                RagConfigRecord.config_fingerprint == fingerprint
            )
        )

    @staticmethod
    def _verify_snapshot(record: RagConfigRecord, snapshot: str) -> None:
        """Verify that fingerprint reuse represents identical canonical JSON.

        Args:
            record:
                Existing configuration record.
            snapshot:
                Canonical JSON being registered.

        Raises:
            ValueError:
                If the canonical JSON differs.
        """

        if record.config_json != snapshot:
            raise ValueError("Stored RAG configuration does not match its fingerprint.")
