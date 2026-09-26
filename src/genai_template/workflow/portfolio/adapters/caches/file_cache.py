"""Filesystem-backed validated structured-artifact cache."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from pydantic import ValidationError

from genai_template.workflow.portfolio.artifacts.fingerprints import (
    canonical_json_bytes,
    sha256_canonical_json,
)
from genai_template.workflow.portfolio.domain.errors import ArtifactCacheError
from genai_template.workflow.portfolio.domain.generation import (
    ArtifactKind,
    CachedArtifact,
)
from genai_template.workflow.portfolio.domain.summaries import StructuredSummary

CACHE_SCHEMA_VERSION = "v1"
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FilesystemArtifactCache:
    """Store canonical cache envelopes under a dedicated filesystem directory.

    Attributes:
        root:
            Directory containing entries named by full generation fingerprint.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the filesystem cache.

        Args:
            root:
                Dedicated cache directory outside the published corpus.
        """

        self.root = root

    def get[SummaryT: StructuredSummary](
        self,
        generation_fingerprint: str,
        *,
        expected_kind: ArtifactKind,
        output_schema_version: str,
        output_type: type[SummaryT],
    ) -> CachedArtifact | None:
        """Load and fully validate one existing canonical envelope.

        Args:
            generation_fingerprint:
                Full generation fingerprint used as the cache key.
            expected_kind:
                Artifact kind required by the caller.
            output_schema_version:
                Structured-output schema version required by the caller.
            output_type:
                Pydantic summary model used to validate stored output.

        Returns:
            A validated artifact, or null when the key has never been stored.

        Raises:
            ArtifactCacheError:
                If an existing entry cannot be read or validated.
        """

        path = self._entry_path(generation_fingerprint)
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise self._error(
                generation_fingerprint,
                "cache entry could not be read",
                "read-failure",
            ) from error

        try:
            value = json.loads(payload)
            artifact = CachedArtifact.model_validate(value)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            raise self._error(
                generation_fingerprint,
                "cache entry is not a valid artifact envelope",
                "invalid-envelope",
            ) from error

        if payload != canonical_json_bytes(artifact):
            raise self._error(
                generation_fingerprint,
                "cache entry is not canonically encoded",
                "noncanonical-envelope",
            )
        self._validate_artifact(
            artifact,
            generation_fingerprint=generation_fingerprint,
            expected_kind=expected_kind,
            output_schema_version=output_schema_version,
            output_type=output_type,
        )
        return artifact

    def put[SummaryT: StructuredSummary](
        self,
        artifact: CachedArtifact,
        *,
        output_type: type[SummaryT],
    ) -> None:
        """Validate and atomically persist one canonical envelope.

        Args:
            artifact:
                Complete validated cache envelope to persist.
            output_type:
                Pydantic summary model used to validate stored output.

        Raises:
            ArtifactCacheError:
                If validation or the atomic write fails.
        """

        fingerprint = artifact.provenance.generation_fingerprint
        self._validate_artifact(
            artifact,
            generation_fingerprint=fingerprint,
            expected_kind=artifact.provenance.artifact_kind,
            output_schema_version=artifact.provenance.output_schema_version,
            output_type=output_type,
        )
        payload = canonical_json_bytes(artifact)
        temporary_path: Path | None = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=self.root,
                prefix=f".{fingerprint}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._entry_path(fingerprint))
            temporary_path = None
        except OSError as error:
            raise self._error(
                fingerprint,
                "cache entry could not be atomically written",
                "write-failure",
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _entry_path(self, generation_fingerprint: str) -> Path:
        """Resolve a validated fingerprint to its flat cache filename.

        Args:
            generation_fingerprint:
                Candidate cache key.

        Returns:
            Filesystem path for the cache entry.

        Raises:
            ArtifactCacheError:
                If the key is not a full lowercase SHA-256 digest.
        """

        if not _FINGERPRINT_PATTERN.fullmatch(generation_fingerprint):
            raise self._error(
                generation_fingerprint,
                "cache key must be a lowercase SHA-256 fingerprint",
                "invalid-key",
            )

        return self.root / f"{generation_fingerprint}.json"

    @staticmethod
    def _validate_artifact[SummaryT: StructuredSummary](
        artifact: CachedArtifact,
        *,
        generation_fingerprint: str,
        expected_kind: ArtifactKind,
        output_schema_version: str,
        output_type: type[SummaryT],
    ) -> None:
        """Validate every identity and structured-output invariant.

        Args:
            artifact:
                Parsed cache envelope.
            generation_fingerprint:
                Cache key required by the caller.
            expected_kind:
                Artifact kind required by the caller.
            output_schema_version:
                Structured-output schema version required by the caller.
            output_type:
                Pydantic summary model used to validate stored output.

        Raises:
            ArtifactCacheError:
                If any cached identity, schema, or hash is invalid.
        """

        if artifact.cache_schema_version != CACHE_SCHEMA_VERSION:
            raise FilesystemArtifactCache._error(
                generation_fingerprint,
                "cache envelope schema version does not match",
                "cache-schema-mismatch",
            )
        provenance = artifact.provenance
        if provenance.generation_fingerprint != generation_fingerprint:
            raise FilesystemArtifactCache._error(
                generation_fingerprint,
                "cache entry fingerprint does not match its key",
                "key-mismatch",
            )
        if provenance.artifact_kind != expected_kind:
            raise FilesystemArtifactCache._error(
                generation_fingerprint,
                "cache entry artifact kind does not match",
                "kind-mismatch",
            )
        if provenance.output_schema_version != output_schema_version:
            raise FilesystemArtifactCache._error(
                generation_fingerprint,
                "cache entry output schema version does not match",
                "output-schema-mismatch",
            )

        try:
            validated_output = output_type.model_validate(artifact.structured_output)
        except ValidationError as error:
            raise FilesystemArtifactCache._error(
                generation_fingerprint,
                "cache entry structured output is invalid",
                "invalid-output",
            ) from error

        if validated_output.model_dump(mode="json") != artifact.structured_output:
            raise FilesystemArtifactCache._error(
                generation_fingerprint,
                "cache entry structured output is not canonical",
                "noncanonical-output",
            )
        if sha256_canonical_json(artifact.structured_output) != artifact.output_hash:
            raise FilesystemArtifactCache._error(
                generation_fingerprint,
                "cache entry structured output hash does not match",
                "output-hash-mismatch",
            )

    @staticmethod
    def _error(
        generation_fingerprint: str,
        message: str,
        reason: str,
    ) -> ArtifactCacheError:
        """Build a cache error containing stable, non-sensitive context.

        Args:
            generation_fingerprint:
                Cache key involved in the failure.
            message:
                Safe human-readable description.
            reason:
                Stable short failure reason.

        Returns:
            Contextual cache error.
        """

        return ArtifactCacheError(
            message,
            generation_fingerprint=generation_fingerprint,
            reason=reason,
        )
