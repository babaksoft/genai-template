"""Golden and invariant tests for deterministic portfolio Markdown rendering."""

from __future__ import annotations

import hashlib

import pytest

from genai_template.workflow.portfolio.domain import (
    ArtifactKind,
    ArtifactProvenance,
    ArtifactRunReport,
    CachedArtifact,
    ComponentSummary,
    ComponentSummaryArtifact,
    EvidenceSection,
    StructuredSummary,
    TokenUsage,
)
from genai_template.workflow.portfolio.infrastructure.corpus import (
    build_document_file_name,
    render_balanced_documents,
    render_component_document,
)
from genai_template.workflow.portfolio.infrastructure.fingerprints import (
    sha256_canonical_json,
)


def _artifact(
    kind: ArtifactKind,
    summary: StructuredSummary,
    fingerprint_character: str,
) -> CachedArtifact:
    """Build artifact metadata consistent with a structured summary.

    Args:
        kind:
            Artifact kind represented by the summary.
        summary:
            Structured summary used as cached output.
        fingerprint_character:
            Character repeated to form deterministic test fingerprints.

    Returns:
        Internally consistent cached artifact.
    """

    output = summary.model_dump(mode="json")
    is_component = kind == "component"
    return CachedArtifact(
        cache_schema_version="v1",
        provenance=ArtifactProvenance(
            artifact_kind=kind,
            generation_fingerprint=fingerprint_character * 64,
            source_fingerprint="a" * 64,
            unit_input_fingerprint="b" * 64 if is_component else None,
            component_artifact_hashes=() if is_component else ("c" * 64,),
            prompt_id=f"portfolio-{kind}-v1",
            prompt_hash="d" * 64,
            output_schema_version="v1",
            provider="ollama",
            model="test-model",
            temperature=0,
        ),
        summary=output,
        output_hash=sha256_canonical_json(output),
        raw_response_hash="e" * 64,
        original_token_usage=TokenUsage(),
    )


def _report(kind: ArtifactKind, fingerprint_character: str) -> ArtifactRunReport:
    """Build current-run metadata for a rendered artifact.

    Args:
        kind:
            Artifact kind represented by the report.
        fingerprint_character:
            Character repeated to form the generation fingerprint.

    Returns:
        Immutable artifact run report.
    """

    return ArtifactRunReport(
        artifact_kind=kind,
        generation_fingerprint=fingerprint_character * 64,
        cache_hit=True,
        token_usage=TokenUsage(),
        latency_seconds=0,
    )


def _section(content: str = "Uses **validated** input.") -> EvidenceSection:
    """Build a section containing Markdown-sensitive prose.

    Args:
        content:
            Statement included in the section.

    Returns:
        Valid evidence section.
    """

    return EvidenceSection(
        content=(content,),
        evidence_paths=("README.md", "src/api.py"),
    )


def _component(unit_id: str = "api") -> ComponentSummaryArtifact:
    """Build a renderable component artifact.

    Args:
        unit_id:
            Stable component identifier.

    Returns:
        Valid component artifact.
    """

    section = _section()
    summary = ComponentSummary(
        responsibilities=section,
        important_abstractions=section,
        behavior=section,
        constraints=section,
        testing_evidence=section,
    )
    artifact = _artifact("component", summary, "1")
    return ComponentSummaryArtifact(
        unit_id=unit_id,
        summary=summary,
        artifact=artifact,
        run_report=_report("component", "1"),
    )


def test_component_renderer_matches_golden_markdown_and_hashes_actual_bytes() -> None:
    """Component headings, escaping, evidence, whitespace, and hash are fixed."""

    document = render_component_document("sample", "Sample --- Project", _component())

    expected = r"""# Sample \-\-\- Project — api Component

## Responsibilities

- Uses \*\*validated\*\* input\.

Evidence:
- `README.md`
- `src/api.py`

## Important Abstractions

- Uses \*\*validated\*\* input\.

Evidence:
- `README.md`
- `src/api.py`

## Behavior

- Uses \*\*validated\*\* input\.

Evidence:
- `README.md`
- `src/api.py`

## Constraints

- Uses \*\*validated\*\* input\.

Evidence:
- `README.md`
- `src/api.py`

## Testing Evidence

- Uses \*\*validated\*\* input\.

Evidence:
- `README.md`
- `src/api.py`
"""
    assert document.content == expected
    assert document.content_hash == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert document.filename == "sample--component--api.md"
    assert document.evidence_paths == ("README.md", "src/api.py")
    assert "\r" not in document.content
    assert document.content.endswith("\n") and not document.content.endswith("\n\n")


def test_balanced_rendering_is_stable_for_shuffled_inputs_and_all_four_types() -> None:
    """The exact balanced set has canonical order independent of input order."""

    components = (_component("web"), _component("api"))

    first = render_balanced_documents("sample", "Sample", components)
    second = render_balanced_documents("sample", "Sample", tuple(reversed(components)))

    assert first == second
    assert [document.document_type for document in first] == [
        "component",
        "component",
    ]
    assert [document.filename for document in first] == [
        "sample--component--api.md",
        "sample--component--web.md",
    ]
    assert "&lt;script&gt;" in first[0].content


def test_filename_construction_rejects_unsafe_values_and_collisions() -> None:
    """Model values cannot control paths and duplicate units fail as a set."""

    with pytest.raises(ValueError, match="lowercase kebab case"):
        build_document_file_name("../sample", "overview")
    with pytest.raises(ValueError, match="require a component"):
        build_document_file_name("sample", "component")
    with pytest.raises(ValueError, match="identifiers must be unique"):
        render_balanced_documents(
            "sample",
            "Sample",
            (_component("api"), _component("api")),
        )
