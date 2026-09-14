"""Resolve inline source labels in generated answers."""

import re

from genai_template.schemas import CitationSource, CitationWarning

CITATION_PATTERN = re.compile(r"\[S[1-9][0-9]*\]")
UNSUPPORTED_CITATION_CODE = "unsupported_citation_labels"
UNSUPPORTED_CITATION_MESSAGE = (
    "The answer references labels not present in its context."
)


def resolve_citations(
    answer: str,
    sources: list[CitationSource],
) -> tuple[list[CitationSource], list[CitationWarning]]:
    """Resolve valid inline labels against the supplied context sources.

    Args:
        answer:
            Generated answer, which remains unchanged by resolution.
        sources:
            Ordered context sources available to the model.

    Returns:
        Sources with citation flags and any unsupported-label warning.
    """

    cited_labels = set(CITATION_PATTERN.findall(answer))
    available_labels = {f"[{source.label}]" for source in sources}
    resolved_sources = [
        source.model_copy(update={"cited": f"[{source.label}]" in cited_labels})
        for source in sources
    ]
    unsupported_labels = sorted(
        (label[1:-1] for label in cited_labels - available_labels),
        key=lambda label: int(label[1:]),
    )
    warnings = []
    if unsupported_labels:
        warnings.append(
            CitationWarning(
                code=UNSUPPORTED_CITATION_CODE,
                message=UNSUPPORTED_CITATION_MESSAGE,
                labels=unsupported_labels,
            )
        )

    return resolved_sources, warnings
