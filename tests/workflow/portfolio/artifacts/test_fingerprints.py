"""Tests for canonical Portfolio artifact generation identities."""

from __future__ import annotations

from genai_template.workflow.portfolio.artifacts.fingerprints import (
    canonical_json_bytes,
    component_generation_fingerprint,
    project_generation_fingerprint,
)
from genai_template.workflow.portfolio.config import (
    InferenceConfig,
    StructuredGenerationConfig,
)


def _provider(
    *,
    model: str = "llama3.2",
    temperature: float = 0.1,
    seed: int | None = 7,
    timeout_seconds: float = 30,
) -> StructuredGenerationConfig:
    """Build structured-generation settings for fingerprint tests.

    Args:
        model:
            Provider model name.
        temperature:
            Content-affecting sampling temperature.
        seed:
            Optional content-affecting sampling seed.
        timeout_seconds:
            Volatile provider timeout.

    Returns:
        Validated provider configuration.
    """

    return StructuredGenerationConfig(
        provider="ollama",
        model=model,
        inference=InferenceConfig(temperature=temperature, seed=seed),
        timeout_seconds=timeout_seconds,
    )


def _component(**changes: object) -> str:
    """Calculate a component fingerprint with optional input changes.

    Args:
        **changes:
            Keyword values overriding stable fingerprint inputs.

    Returns:
        Component generation fingerprint.
    """

    values: dict[str, object] = {
        "source_fingerprint": "1" * 64,
        "unit_id": "core",
        "unit_input_fingerprint": "2" * 64,
        "prompt_id": "component-v1",
        "prompt_hash": "3" * 64,
        "output_schema_version": "v1",
        "structured_generation": _provider(),
    }
    values.update(changes)
    return component_generation_fingerprint(**values)  # type: ignore[arg-type]


def test_canonical_json_is_independent_of_mapping_order() -> None:
    """Equivalent mappings serialize to identical canonical bytes."""

    assert canonical_json_bytes({"b": 2, "a": {"d": 4, "c": 3}}) == (
        canonical_json_bytes({"a": {"c": 3, "d": 4}, "b": 2})
    )


def test_component_identity_excludes_timeout_and_unrelated_values() -> None:
    """Transport settings and values outside the projection do not affect keys."""

    first = _component(structured_generation=_provider(timeout_seconds=10))
    second = _component(structured_generation=_provider(timeout_seconds=999))

    assert first == second


def test_each_component_generation_input_changes_identity() -> None:
    """Every stable source, unit, prompt, schema, model, and inference input matters."""

    baseline = _component()
    variants = [
        _component(source_fingerprint="a" * 64),
        _component(unit_id="api"),
        _component(unit_input_fingerprint="b" * 64),
        _component(prompt_id="component-v2"),
        _component(prompt_hash="c" * 64),
        _component(output_schema_version="v2"),
        _component(structured_generation=_provider(model="qwen3")),
        _component(structured_generation=_provider(temperature=0.2)),
        _component(structured_generation=_provider(seed=8)),
    ]

    assert all(variant != baseline for variant in variants)
    assert len(set(variants)) == len(variants)


def test_project_identity_is_stable_across_component_mapping_order() -> None:
    """Project dependencies retain unit associations without insertion-order noise."""

    first = project_generation_fingerprint(
        document_type="architecture",
        source_fingerprint="1" * 64,
        repository_context_fingerprint="2" * 64,
        component_artifact_hashes={"api": "4" * 64, "core": "5" * 64},
        prompt_id="architecture-v1",
        prompt_hash="3" * 64,
        output_schema_version="v1",
        structured_generation=_provider(),
    )
    second = project_generation_fingerprint(
        document_type="architecture",
        source_fingerprint="1" * 64,
        repository_context_fingerprint="2" * 64,
        component_artifact_hashes={"core": "5" * 64, "api": "4" * 64},
        prompt_id="architecture-v1",
        prompt_hash="3" * 64,
        output_schema_version="v1",
        structured_generation=_provider(),
    )
    changed = project_generation_fingerprint(
        document_type="architecture",
        source_fingerprint="1" * 64,
        repository_context_fingerprint="2" * 64,
        component_artifact_hashes={"core": "6" * 64, "api": "4" * 64},
        prompt_id="architecture-v1",
        prompt_hash="3" * 64,
        output_schema_version="v1",
        structured_generation=_provider(),
    )

    assert first == second
    assert changed != first
