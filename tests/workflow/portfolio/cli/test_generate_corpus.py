"""Tests for the Portfolio corpus generation CLI."""

from __future__ import annotations

import importlib
import json
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any

from genai_template.workflow.portfolio.domain import GenerationRunReport, TokenUsage

generate_cli = importlib.import_module("genai_template.workflow.portfolio.cli.generate")


def _report() -> GenerationRunReport:
    """Build a minimal successful CLI report.

    Returns:
        Typed completed generation report.
    """

    return GenerationRunReport(
        project_slug="sample",
        resolved_commit_sha="a" * 40,
        source_fingerprint="b" * 64,
        generation_configuration_fingerprint="c" * 64,
        corpus_fingerprint="d" * 64,
        artifacts=(),
        provider_call_count=0,
        billed_token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        estimated_cost=Decimal(0),
        elapsed_seconds=1,
        published_path=Path("/safe/data/portfolio"),
        release_path=Path("/safe/data/.portfolio-releases/release"),
    )


def test_json_report_contains_stable_metrics_without_generation_content() -> None:
    """JSON output exposes identities and metrics but no prompts or source text."""

    rendered = generate_cli.render_json_report(_report())
    value = json.loads(rendered)

    assert value["project_slug"] == "sample"
    assert value["provider_call_count"] == 0
    assert value["billed_token_usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
    }
    assert "prompt" not in rendered
    assert "structured_output" not in rendered


def test_cli_returns_one_and_uses_stderr_for_workflow_failure(
    monkeypatch: Any,
) -> None:
    """A workflow failure returns one without writing a success report."""

    class _Workflow:
        """Accept CLI composition arguments without creating provider clients."""

        def __init__(self, **kwargs: object) -> None:
            """Accept ignored injected dependencies.

            Args:
                **kwargs:
                    CLI workflow dependencies.
            """

    async def fail(*args: object, **kwargs: object) -> GenerationRunReport:
        """Simulate a focused workflow failure.

        Args:
            *args:
                Ignored positional values.
            **kwargs:
                Ignored keyword values.

        Raises:
            ValueError:
                Always, to simulate invalid configuration.
        """

        raise ValueError("invalid portfolio configuration")

    monkeypatch.setattr(generate_cli, "PortfolioCorpusWorkflow", _Workflow)
    monkeypatch.setattr(generate_cli, "run_portfolio_corpus_workflow", fail)
    stdout = StringIO()
    stderr = StringIO()

    status = generate_cli.main(
        ("--config", "config.yml", "--project", "sample"),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "error: invalid portfolio configuration\n"
