"""Command-line entry point for complete Portfolio corpus generation."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from genai_template.workflow.portfolio.adapters.generators import (
    OllamaStructuredSummaryGenerator,
    OpenAIStructuredSummaryGenerator,
)
from genai_template.workflow.portfolio.adapters.repositories import (
    LocalGitSnapshotReader,
)
from genai_template.workflow.portfolio.cli.generation import (
    render_json_report,
    render_text_report,
)
from genai_template.workflow.portfolio.config.models import GenerationConfig
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


if __name__ == "__main__":
    raise SystemExit(main())
