"""Stable prompt definitions and deterministic prompt assembly."""

from __future__ import annotations

from genai_template.workflow.portfolio.domain import (
    PromptDefinition,
    SnapshotFile,
    prompt_template,
)


def assemble_component_prompt(
    definition: PromptDefinition,
    files: tuple[SnapshotFile, ...],
) -> str:
    """Assemble a deterministic unit prompt from snapshot files.

    Args:
        definition:
            Unit prompt definition.
        files:
            Exact repository files supplied as unit evidence.

    Returns:
        Fully assembled prompt with explicit source boundaries.

    Raises:
        ValueError:
            If paths are duplicated.
    """

    if not files:
        raise ValueError("prompt input must provide at least one evidence path")
    allowed_paths = tuple(sorted(file.path for file in files))
    if len(allowed_paths) != len(set(allowed_paths)):
        raise ValueError("prompt input paths must be unique")

    ordered_files = tuple(sorted(files, key=lambda file: file.path))
    allowed_evidence_paths = "\n".join(f"- {path}" for path in allowed_paths)
    repository_context = _render_files(ordered_files)

    return (
        prompt_template(definition)
        .replace("{allowed_evidence_paths}", allowed_evidence_paths)
        .replace("{example_evidence_path}", allowed_paths[0])
        .replace("{repository_files}", repository_context)
    )


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
