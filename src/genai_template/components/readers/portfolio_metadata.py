"""Stable metadata contract for manifest-backed Portfolio documents."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Final

PROJECT_SLUG: Final = "project_slug"
PROJECT_DISPLAY_NAME: Final = "project_display_name"
DOCUMENT_TYPE: Final = "document_type"
COMPONENT_ID: Final = "component_id"
REPOSITORY_URL: Final = "repository_url"
RESOLVED_COMMIT_SHA: Final = "resolved_commit_sha"
CORPUS_FINGERPRINT: Final = "corpus_fingerprint"
GENERATION_FINGERPRINT: Final = "generation_fingerprint"
FILE_NAME: Final = "file_name"
HEADER_PATH: Final = "header_path"

PORTFOLIO_REQUIRED_METADATA_KEYS: Final = frozenset(
    {
        PROJECT_SLUG,
        PROJECT_DISPLAY_NAME,
        DOCUMENT_TYPE,
        RESOLVED_COMMIT_SHA,
        CORPUS_FINGERPRINT,
        GENERATION_FINGERPRINT,
        FILE_NAME,
    }
)
PORTFOLIO_OPTIONAL_METADATA_KEYS: Final = frozenset(
    {COMPONENT_ID, REPOSITORY_URL, HEADER_PATH}
)
PORTFOLIO_METADATA_KEYS: Final = (
    PORTFOLIO_REQUIRED_METADATA_KEYS | PORTFOLIO_OPTIONAL_METADATA_KEYS
)
PORTFOLIO_PROVENANCE_KEYS: Final = PORTFOLIO_METADATA_KEYS.difference(
    {FILE_NAME, HEADER_PATH}
)


def has_portfolio_metadata(metadata: dict[str, Any]) -> bool:
    """Return whether metadata contains any Portfolio-specific key.

    Args:
        metadata:
            Metadata to inspect.

    Returns:
        Whether the metadata claims any part of the Portfolio contract.
    """

    return bool(PORTFOLIO_PROVENANCE_KEYS.intersection(metadata))


def validate_portfolio_metadata(
    metadata: dict[str, Any], *, allow_header_path: bool
) -> None:
    """Validate one controlled Portfolio metadata projection.

    Args:
        metadata:
            Document or chunk metadata to validate.
        allow_header_path:
            Whether splitter-generated Markdown header metadata is allowed.

    Raises:
        ValueError:
            If required provenance is absent, malformed, or contains an
            uncontrolled field.
    """

    missing = PORTFOLIO_REQUIRED_METADATA_KEYS.difference(metadata)
    if missing:
        raise ValueError(
            "Portfolio metadata is missing required keys: " + ", ".join(sorted(missing))
        )

    allowed = PORTFOLIO_METADATA_KEYS
    if not allow_header_path:
        allowed = allowed.difference({HEADER_PATH})
    unexpected = set(metadata).difference(allowed)
    if unexpected:
        raise ValueError(
            "Portfolio metadata contains unsupported keys: "
            + ", ".join(sorted(unexpected))
        )

    for key in PORTFOLIO_REQUIRED_METADATA_KEYS:
        value = metadata[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Portfolio metadata '{key}' must be a non-empty string")

    filename = metadata[FILE_NAME]
    pure_name = PurePosixPath(filename)
    if pure_name.name != filename or pure_name.suffix != ".md":
        raise ValueError("Portfolio file_name must be a safe Markdown basename")

    document_type = metadata[DOCUMENT_TYPE]
    if document_type not in {
        "overview",
        "architecture",
        "testing_operations",
        "component",
    }:
        raise ValueError("Portfolio document_type is invalid")

    component_id = metadata.get(COMPONENT_ID)
    if (document_type == "component") != (component_id is not None):
        raise ValueError("Portfolio component identity is inconsistent")

    for key in (COMPONENT_ID, REPOSITORY_URL, HEADER_PATH):
        value = metadata.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"Portfolio metadata '{key}' must be a non-empty string")
