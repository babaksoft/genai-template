"""Command-line interface for immutable Portfolio snapshot inspection."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from genai_template.workflow.portfolio.cli.inspection import (
    inspect_portfolio,
    render_json_report,
    render_text_report,
)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Parser for snapshot inspection command arguments.
    """

    parser = argparse.ArgumentParser(
        description="Inspect immutable Portfolio repository snapshots."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Portfolio YAML configuration path",
    )
    parser.add_argument(
        "--project",
        "--project-slug",
        dest="project_slug",
        help="inspect only the project with this slug",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit the stable JSON report",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the snapshot inspection command.

    Args:
        argv:
            Optional command arguments excluding the executable name.
        stdout:
            Optional standard-output replacement for callers and tests.
        stderr:
            Optional standard-error replacement for callers and tests.

    Returns:
        Zero after a successful report, otherwise one for inspection failures.
    """

    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    arguments = _build_argument_parser().parse_args(argv)
    try:
        report = inspect_portfolio(
            arguments.config,
            project_slug=arguments.project_slug,
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
