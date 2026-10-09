"""Stable prompt definitions and deterministic prompt assembly."""

from __future__ import annotations

from collections.abc import Mapping

from genai_template.workflow.portfolio.domain import (
    PromptDefinition,
    SnapshotFile,
    prompt_template,
)
from genai_template.workflow.portfolio.infrastructure.fingerprints import (
    canonical_json_bytes,
)


def assemble_component_prompt(
    definition: PromptDefinition,
    files: tuple[SnapshotFile, ...],
) -> str:
    """Assemble a deterministic component prompt from snapshot files.

    Args:
        definition:
            Component prompt definition.
        files:
            Exact repository files supplied as component evidence.

    Returns:
        Fully assembled prompt with explicit source boundaries.

    Raises:
        ValueError:
            If the definition is not a component prompt or paths are duplicated.
    """

    if definition.artifact_kind != "component":
        raise ValueError("component prompt assembly requires a component definition")

    ordered_files = _order_files(files)
    allowed_paths = tuple(file.path for file in ordered_files)
    _require_evidence_scope(allowed_paths)

    return (
        prompt_template(definition)
        .replace("{allowed_evidence_paths}", _render_allowed_paths(allowed_paths))
        .replace("{example_evidence_path}", allowed_paths[0])
        .replace("{repository_files}", _render_files(ordered_files))
    )


def assemble_project_prompt(
    definition: PromptDefinition,
    repository_files: tuple[SnapshotFile, ...],
    component_outputs: Mapping[str, object],
) -> str:
    """Assemble a deterministic project synthesis prompt.

    Component output is canonical JSON, making mapping insertion order irrelevant.
    Repository evidence remains delimited independently from generated component
    material so the provider cannot confuse their provenance.

    Args:
        definition:
            Project-document prompt definition.
        repository_files:
            Selected repository context actually supplied to synthesis.
        component_outputs:
            Validated component summaries keyed by stable unit identifier.

    Returns:
        Fully assembled synthesis prompt.

    Raises:
        ValueError:
            If a component definition is supplied or paths are duplicated.
    """

    if definition.artifact_kind == "component":
        raise ValueError("project prompt assembly requires a project definition")

    ordered_files = _order_files(repository_files)
    components = {
        unit_id: _json_value(output)
        for unit_id, output in sorted(component_outputs.items())
    }
    component_json = canonical_json_bytes(components).decode("utf-8")
    allowed_paths = tuple(
        sorted(
            {file.path for file in ordered_files}
            | _collect_component_evidence_paths(components)
        )
    )
    _require_evidence_scope(allowed_paths)

    template = prompt_template(definition)
    template = template.replace(
        "{allowed_evidence_paths}", _render_allowed_paths(allowed_paths)
    ).replace("{example_evidence_path}", allowed_paths[0])
    before_files, separator, after_files = template.partition("{repository_files}")
    before_components, component_separator, after_components = after_files.partition(
        "{component_summaries}"
    )
    if not separator or not component_separator:
        raise ValueError("project prompt template is missing input placeholders")

    return (
        before_files
        + _render_files(ordered_files)
        + before_components
        + component_json
        + after_components
    )


def _order_files(files: tuple[SnapshotFile, ...]) -> tuple[SnapshotFile, ...]:
    """Order files canonically and reject ambiguous duplicate paths.

    Args:
        files:
            Snapshot files to order.

    Returns:
        Files sorted by repository path.

    Raises:
        ValueError:
            If a path occurs more than once.
    """

    paths = [file.path for file in files]
    if len(paths) != len(set(paths)):
        raise ValueError("prompt input paths must be unique")

    return tuple(sorted(files, key=lambda file: file.path))


def _render_files(files: tuple[SnapshotFile, ...]) -> str:
    """Render files with explicit path and content boundaries.

    Args:
        files:
            Canonically ordered snapshot files.

    Returns:
        Delimited source material.
    """

    blocks = []
    for file in files:
        blocks.append(
            f'<repository-file path="{file.path}">\n'
            f"<content>\n{file.text}\n</content>\n"
            "</repository-file>"
        )

    return "\n".join(blocks)


def _render_allowed_paths(paths: tuple[str, ...]) -> str:
    """Render the exact allowed evidence catalog.

    Args:
        paths:
            Canonically ordered repository-relative paths.

    Returns:
        Markdown list containing every exact allowed path.
    """

    return "\n".join(f"- {path}" for path in paths)


def _require_evidence_scope(paths: tuple[str, ...]) -> None:
    """Require at least one path for evidence-bearing generation.

    Args:
        paths:
            Exact evidence paths for a generation call.

    Raises:
        ValueError:
            If no path is available for required section evidence.
    """

    if not paths:
        raise ValueError("prompt input must provide at least one evidence path")


def _collect_component_evidence_paths(value: object) -> set[str]:
    """Collect evidence paths recursively from serialized component summaries.

    Args:
        value:
            Canonical JSON-compatible component-summary input.

    Returns:
        Exact string paths found under evidence-path fields.
    """

    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_paths" and isinstance(child, (list, tuple)):
                paths.update(path for path in child if isinstance(path, str))
            else:
                paths.update(_collect_component_evidence_paths(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            paths.update(_collect_component_evidence_paths(child))
    return paths


def _json_value(value: object) -> object:
    """Convert a Pydantic value while retaining plain JSON-compatible inputs.

    Args:
        value:
            Validated component output or JSON-compatible object.

    Returns:
        JSON-compatible representation.
    """

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")

    return value
