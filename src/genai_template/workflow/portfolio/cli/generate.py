"""Command-line entry point for complete Portfolio corpus generation."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from genai_template.workflow.portfolio.adapters.generation import (
    OllamaStructuredSummaryGenerator,
    OpenAIStructuredSummaryGenerator,
)
from genai_template.workflow.portfolio.adapters.repositories import (
    LocalGitSnapshotReader,
)
from genai_template.workflow.portfolio.artifacts.fingerprints import (
    canonical_json_bytes,
)
from genai_template.workflow.portfolio.config.models import GenerationConfig
from genai_template.workflow.portfolio.domain.reports import GenerationRunReport
from genai_template.workflow.portfolio.ports.structured_generator import (
    StructuredSummaryGenerator,
)
from genai_template.workflow.portfolio.workflow.corpus import (
    PortfolioCorpusWorkflow,
    run_portfolio_corpus_workflow,
)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the corpus-generation command parser.

    Returns:
        Parser requiring one configuration and project selection.
    """

    parser = argparse.ArgumentParser(
        description="Generate and atomically publish one Portfolio corpus."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Portfolio YAML configuration path",
    )
    parser.add_argument(
        "--project",
        dest="project_slug",
        required=True,
        help="configured project slug",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit the typed run report as canonical JSON",
    )
    return parser


def create_structured_generator(
    generation: GenerationConfig,
) -> StructuredSummaryGenerator:
    """Create the configured LlamaIndex structured-generation adapter.

    Args:
        generation:
            Validated provider and inference settings.

    Returns:
        Configured structured-summary generator.
    """

    settings = generation.structured_generation
    if settings.provider == "openai":
        return OpenAIStructuredSummaryGenerator(
            settings.model,
            temperature=settings.inference.temperature,
            timeout_seconds=settings.timeout_seconds,
        )

    return OllamaStructuredSummaryGenerator(
        settings.model,
        temperature=settings.inference.temperature,
        seed=settings.inference.seed,
        timeout_seconds=settings.timeout_seconds,
    )


def render_text_report(report: GenerationRunReport) -> str:
    """Render a concise stable successful-run report without generated content.

    Args:
        report:
            Completed typed generation report.

    Returns:
        Newline-delimited human-readable report fields.
    """

    input_tokens = _optional_number(report.billed_token_usage.input_tokens)
    output_tokens = _optional_number(report.billed_token_usage.output_tokens)
    estimated_cost = _optional_number(report.estimated_cost)
    cache_hits = sum(artifact.cache_hit for artifact in report.artifacts)
    generation_identity = report.generation_configuration_fingerprint
    return "\n".join(
        (
            f"project: {report.project_slug}",
            f"commit: {report.resolved_commit_sha}",
            f"source_fingerprint: {report.source_fingerprint}",
            f"generation_configuration_fingerprint: {generation_identity}",
            f"corpus_fingerprint: {report.corpus_fingerprint}",
            f"artifacts: {len(report.artifacts)}",
            f"cache_hits: {cache_hits}",
            f"provider_calls: {report.provider_call_count}",
            f"billed_input_tokens: {input_tokens}",
            f"billed_output_tokens: {output_tokens}",
            f"estimated_cost: {estimated_cost}",
            f"elapsed_seconds: {report.elapsed_seconds:.6f}",
            f"published_path: {report.published_path}",
            f"release_path: {report.release_path}",
        )
    )


def render_json_report(report: GenerationRunReport) -> str:
    """Render the typed report as stable compact canonical JSON.

    Args:
        report:
            Completed typed generation report.

    Returns:
        Canonical JSON without a trailing newline.
    """

    return canonical_json_bytes(report).decode("utf-8")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run corpus generation and return a process-compatible exit status.

    Args:
        argv:
            Optional arguments excluding the executable name.
        stdout:
            Optional standard-output replacement for tests.
        stderr:
            Optional standard-error replacement for tests.

    Returns:
        Zero after successful publication, otherwise one for workflow failures.
    """

    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    arguments = _build_argument_parser().parse_args(argv)
    workflow = PortfolioCorpusWorkflow(
        repository_reader=LocalGitSnapshotReader(),
        generator_factory=create_structured_generator,
    )

    try:
        report = asyncio.run(
            run_portfolio_corpus_workflow(
                workflow,
                config_path=arguments.config,
                project_slug=arguments.project_slug,
            )
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=error_output)
        return 1

    rendered = (
        render_json_report(report)
        if arguments.json_output
        else render_text_report(report)
    )
    print(rendered, file=output)
    return 0


def _optional_number(value: object | None) -> str:
    """Render an optional numeric metric without implying an unavailable zero.

    Args:
        value:
            Optional metric value.

    Returns:
        String value or the explicit ``unknown`` marker.
    """

    return "unknown" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
