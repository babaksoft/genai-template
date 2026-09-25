"""Canonical serialization and stable generated-artifact identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel

from genai_template.workflow.portfolio.config.models import (
    StructuredGenerationConfig,
)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value to canonical UTF-8 bytes.

    Pydantic models are first converted in JSON mode. Mapping keys are sorted,
    insignificant whitespace is omitted, non-finite floats are rejected, and no
    terminal newline is added.

    Args:
        value:
            JSON-compatible value or Pydantic model to serialize.

    Returns:
        Canonical UTF-8 JSON bytes.

    Raises:
        TypeError:
            If the value is not JSON serializable.
        ValueError:
            If the value contains a non-finite float.
    """

    serializable = (
        value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    )
    return json.dumps(
        serializable,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_canonical_json(value: object) -> str:
    """Calculate a SHA-256 hash from canonical JSON bytes.

    Args:
        value:
            JSON-compatible value or Pydantic model to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _generation_settings(
    structured_generation: StructuredGenerationConfig,
) -> dict[str, Any]:
    """Project only content-affecting provider settings into an identity.

    Args:
        structured_generation:
            Validated provider configuration, including volatile timeout settings.

    Returns:
        Secret-free stable provider, model, and inference values.
    """

    return {
        "provider": structured_generation.provider,
        "model": structured_generation.model,
        "inference": structured_generation.inference.model_dump(mode="json"),
    }


def component_generation_fingerprint(
    *,
    source_fingerprint: str,
    unit_id: str,
    unit_input_fingerprint: str,
    prompt_id: str,
    prompt_hash: str,
    output_schema_version: str,
    structured_generation: StructuredGenerationConfig,
) -> str:
    """Calculate one component artifact's stable generation identity.

    Credentials, local paths, timeout, pricing, timestamps, latency, and run cache
    status cannot affect this identity because none are admitted to the projection.

    Args:
        source_fingerprint:
            Stable selected-snapshot identity.
        unit_id:
            Stable configured summary-unit identifier.
        unit_input_fingerprint:
            Stable identity of the unit configuration and selected files.
        prompt_id:
            Stable prompt-template identifier.
        prompt_hash:
            SHA-256 hash of the exact prompt template.
        output_schema_version:
            Validated structured-output schema version.
        structured_generation:
            Validated provider and inference configuration.

    Returns:
        Lowercase hexadecimal SHA-256 generation fingerprint.
    """

    identity = {
        "fingerprint_schema": "portfolio-component-generation-v1",
        "source_fingerprint": source_fingerprint,
        "unit": {
            "id": unit_id,
            "input_fingerprint": unit_input_fingerprint,
        },
        "prompt": {"id": prompt_id, "hash": prompt_hash},
        "output_schema_version": output_schema_version,
        "structured_generation": _generation_settings(structured_generation),
    }
    return sha256_canonical_json(identity)


def project_generation_fingerprint(
    *,
    document_type: Literal["overview", "architecture", "testing_operations"],
    source_fingerprint: str,
    repository_context_fingerprint: str,
    component_artifact_hashes: Mapping[str, str],
    prompt_id: str,
    prompt_hash: str,
    output_schema_version: str,
    structured_generation: StructuredGenerationConfig,
) -> str:
    """Calculate a project synthesis artifact's stable generation identity.

    Component dependencies are sorted by unit identifier, making equivalent
    mappings independent of insertion order while retaining explicit unit-to-hash
    associations.

    Args:
        document_type:
            Balanced project document being synthesized.
        source_fingerprint:
            Stable selected-snapshot identity.
        repository_context_fingerprint:
            Identity of exact repository files supplied to synthesis.
        component_artifact_hashes:
            Mapping from component unit identifiers to validated artifact hashes.
        prompt_id:
            Stable prompt-template identifier.
        prompt_hash:
            SHA-256 hash of the exact prompt template.
        output_schema_version:
            Validated structured-output schema version.
        structured_generation:
            Validated provider and inference configuration.

    Returns:
        Lowercase hexadecimal SHA-256 generation fingerprint.
    """

    components = [
        {"unit_id": unit_id, "artifact_hash": artifact_hash}
        for unit_id, artifact_hash in sorted(component_artifact_hashes.items())
    ]
    identity = {
        "fingerprint_schema": "portfolio-project-generation-v1",
        "document_type": document_type,
        "source_fingerprint": source_fingerprint,
        "repository_context_fingerprint": repository_context_fingerprint,
        "components": components,
        "prompt": {"id": prompt_id, "hash": prompt_hash},
        "output_schema_version": output_schema_version,
        "structured_generation": _generation_settings(structured_generation),
    }
    return sha256_canonical_json(identity)
