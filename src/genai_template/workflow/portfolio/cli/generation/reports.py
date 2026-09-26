from genai_template.workflow.portfolio.artifacts.fingerprints import (
    canonical_json_bytes,
)
from genai_template.workflow.portfolio.domain.reports import GenerationRunReport


def _optional_number(value: object | None) -> str:
    """Render an optional numeric metric without implying an unavailable zero.

    Args:
        value:
            Optional metric value.

    Returns:
        String value or the explicit ``unknown`` marker.
    """

    return "unknown" if value is None else str(value)


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
