"""Pure deterministic Markdown renderers for validated portfolio summaries."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Sequence

from genai_template.workflow.portfolio.artifacts.fingerprints import (
    sha256_canonical_json,
)
from genai_template.workflow.portfolio.domain.components import ComponentSummaryArtifact
from genai_template.workflow.portfolio.domain.generation import (
    ArtifactKind,
    CachedArtifact,
    RenderedDocument,
)
from genai_template.workflow.portfolio.domain.projects import (
    ProjectArtifactKind,
    ProjectSummaryArtifact,
)
from genai_template.workflow.portfolio.domain.summaries import (
    EvidenceSection,
    StructuredSummary,
    summary_evidence_paths,
)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PROJECT_ORDER: dict[ProjectArtifactKind, int] = {
    "overview": 0,
    "architecture": 1,
    "testing_operations": 2,
}
_PROJECT_TITLES: dict[ProjectArtifactKind, str] = {
    "overview": "Overview",
    "architecture": "Architecture",
    "testing_operations": "Testing and Operations",
}
_SECTION_TITLES: dict[str, str] = {
    "responsibilities": "Responsibilities",
    "important_abstractions": "Important Abstractions",
    "behavior": "Behavior",
    "constraints": "Constraints",
    "testing_evidence": "Testing Evidence",
    "boundaries": "Boundaries",
    "dependencies": "Dependencies",
    "principal_flows": "Principal Flows",
    "purpose": "Purpose",
    "capabilities": "Capabilities",
    "entry_points": "Entry Points",
    "technology_choices": "Technology Choices",
    "testing_strategy": "Testing Strategy",
    "local_operation": "Local Operation",
    "configuration": "Configuration",
    "observability": "Observability",
    "operational_constraints": "Operational Constraints",
}
_MARKDOWN_PUNCTUATION = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|])")


def build_document_filename(
    project_slug: str,
    document_type: ArtifactKind,
    *,
    component_id: str | None = None,
) -> str:
    """Construct one safe deterministic flat Markdown filename.

    Args:
        project_slug:
            Normalized project identifier.
        document_type:
            Logical type of document to name.
        component_id:
            Normalized summary-unit identifier for a component document.

    Returns:
        Project-prefixed flat Markdown filename.

    Raises:
        ValueError:
            If identifiers are unsafe or component identity does not match type.
    """

    _validate_identifier(project_slug, "project slug")
    if document_type == "component":
        if component_id is None:
            raise ValueError("component documents require a component identifier")
        _validate_identifier(component_id, "component identifier")
        return f"{project_slug}--component--{component_id}.md"
    if component_id is not None:
        raise ValueError("project documents cannot have a component identifier")
    suffixes: dict[ArtifactKind, str] = {
        "overview": "overview",
        "architecture": "architecture",
        "testing_operations": "testing-operations",
        "component": "component",
    }
    return f"{project_slug}--{suffixes[document_type]}.md"


def render_component_document(
    project_slug: str,
    display_name: str,
    component: ComponentSummaryArtifact,
) -> RenderedDocument:
    """Render one validated component artifact as deterministic Markdown.

    Args:
        project_slug:
            Normalized project identifier used only for the filename.
        display_name:
            Human-readable project name used in the owned title template.
        component:
            Validated component summary and artifact metadata.

    Returns:
        Immutable rendered document with a hash of its actual UTF-8 bytes.
    """

    title = f"{display_name} — {component.unit_id} Component"
    return _render_document(
        project_slug=project_slug,
        title=title,
        document_type="component",
        component_id=component.unit_id,
        summary=component.summary,
        artifact=component.artifact,
    )


def render_project_document(
    project_slug: str,
    display_name: str,
    project: ProjectSummaryArtifact,
) -> RenderedDocument:
    """Render one validated project artifact as deterministic Markdown.

    Args:
        project_slug:
            Normalized project identifier used only for the filename.
        display_name:
            Human-readable project name used in the owned title template.
        project:
            Validated project summary and artifact metadata.

    Returns:
        Immutable rendered document with a hash of its actual UTF-8 bytes.
    """

    title = f"{display_name} — {_PROJECT_TITLES[project.artifact_kind]}"
    return _render_document(
        project_slug=project_slug,
        title=title,
        document_type=project.artifact_kind,
        component_id=None,
        summary=project.summary,
        artifact=project.artifact,
    )


def render_balanced_documents(
    project_slug: str,
    display_name: str,
    components: Sequence[ComponentSummaryArtifact],
    projects: Sequence[ProjectSummaryArtifact],
) -> tuple[RenderedDocument, ...]:
    """Render and validate the exact balanced-profile document set.

    Input collection order does not affect output bytes or ordering. Project
    documents precede component documents, whose order is their stable unit ID.
    The complete filename set is checked for collisions before any documents are
    returned.

    Args:
        project_slug:
            Normalized project identifier used for every filename.
        display_name:
            Human-readable project name used in owned title templates.
        components:
            One validated artifact for every configured summary unit.
        projects:
            Exactly one artifact for each balanced project-document type.

    Returns:
        Deterministically ordered rendered documents.

    Raises:
        ValueError:
            If artifacts are missing, duplicated, or produce filename collisions.
    """

    ordered_projects = tuple(
        sorted(projects, key=lambda value: _PROJECT_ORDER[value.artifact_kind])
    )
    project_kinds = [project.artifact_kind for project in ordered_projects]
    if project_kinds != ["overview", "architecture", "testing_operations"]:
        raise ValueError(
            "balanced rendering requires one overview, architecture, and "
            "testing/operations artifact"
        )
    ordered_components = tuple(sorted(components, key=lambda value: value.unit_id))
    component_ids = [component.unit_id for component in ordered_components]
    if not component_ids:
        raise ValueError("balanced rendering requires at least one component")
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("component document identifiers must be unique")

    filenames = [
        build_document_filename(project_slug, project.artifact_kind)
        for project in ordered_projects
    ] + [
        build_document_filename(
            project_slug, "component", component_id=component.unit_id
        )
        for component in ordered_components
    ]
    if len(filenames) != len(set(filenames)):
        raise ValueError("balanced document filenames collide")

    return tuple(
        render_project_document(project_slug, display_name, project)
        for project in ordered_projects
    ) + tuple(
        render_component_document(project_slug, display_name, component)
        for component in ordered_components
    )


def _render_document(
    *,
    project_slug: str,
    title: str,
    document_type: ArtifactKind,
    component_id: str | None,
    summary: StructuredSummary,
    artifact: CachedArtifact,
) -> RenderedDocument:
    """Render a structured summary using only application-owned Markdown syntax.

    Args:
        project_slug:
            Normalized project identifier.
        title:
            Untrusted title value inserted into an owned heading.
        document_type:
            Logical rendered-document type.
        component_id:
            Component identifier when rendering a component.
        summary:
            Validated structured fields to render.
        artifact:
            Cache artifact carrying stable generation identities.

    Returns:
        Complete deterministic Markdown document.

    Raises:
        ValueError:
            If summary, artifact, and requested document type disagree.
    """

    if artifact.provenance.artifact_kind != document_type:
        raise ValueError("rendered document type does not match artifact provenance")
    expected_output = summary.model_dump(mode="json")
    if artifact.structured_output != expected_output:
        raise ValueError("rendered summary does not match its artifact output")
    if artifact.output_hash != sha256_canonical_json(expected_output):
        raise ValueError("rendered artifact output hash does not match its summary")

    lines = [f"# {_escape_markdown(title)}", ""]
    for field_name in type(summary).model_fields:
        section = getattr(summary, field_name)
        if not isinstance(section, EvidenceSection):
            raise TypeError("summary fields must be evidence sections")
        lines.extend(_render_section(_SECTION_TITLES[field_name], section))
    content = "\n".join(lines).rstrip("\n") + "\n"
    content_bytes = content.encode("utf-8")
    return RenderedDocument(
        filename=build_document_filename(
            project_slug, document_type, component_id=component_id
        ),
        document_type=document_type,
        component_id=component_id,
        generation_fingerprint=artifact.provenance.generation_fingerprint,
        artifact_hash=artifact.output_hash,
        content=content,
        content_hash=hashlib.sha256(content_bytes).hexdigest(),
        evidence_paths=summary_evidence_paths(summary),
    )


def _render_section(title: str, section: EvidenceSection) -> list[str]:
    """Render one evidence-bearing section with fixed whitespace and ordering.

    Args:
        title:
            Application-owned section title.
        section:
            Validated statements and evidence paths.

    Returns:
        Markdown lines ending in one blank separator line.
    """

    lines = [f"## {title}", ""]
    lines.extend(f"- {_escape_markdown(statement)}" for statement in section.content)
    lines.extend(("", "Evidence:"))
    lines.extend(f"- {_render_code_span(path)}" for path in section.evidence_paths)
    lines.append("")
    return lines


def _escape_markdown(value: str) -> str:
    """Escape untrusted text so it cannot introduce Markdown structure or HTML.

    Args:
        value:
            Model- or configuration-supplied text.

    Returns:
        Safe deterministic inline Markdown text.
    """

    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    escaped = html.escape(normalized, quote=True)
    escaped = _MARKDOWN_PUNCTUATION.sub(r"\\\1", escaped)
    return escaped.replace("\n", "  \n  ")


def _render_code_span(value: str) -> str:
    """Render arbitrary evidence paths in a safe Markdown code span.

    Args:
        value:
            Repository-relative evidence path.

    Returns:
        Code span using a delimiter longer than any run in the value.
    """

    longest_run = max((len(run) for run in re.findall(r"`+", value)), default=0)
    delimiter = "`" * (longest_run + 1)
    padding = " " if value.startswith("`") or value.endswith("`") else ""
    return f"{delimiter}{padding}{value}{padding}{delimiter}"


def _validate_identifier(value: str, label: str) -> None:
    """Reject identifiers that could escape the flat deterministic namespace.

    Args:
        value:
            Candidate project or component identifier.
        label:
            Human-readable field name for validation errors.

    Raises:
        ValueError:
            If the value is not lowercase kebab case.
    """

    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be lowercase kebab case")
