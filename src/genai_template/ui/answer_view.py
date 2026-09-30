"""Streamlit rendering helpers for citation-aware answers."""

import streamlit as st

from genai_template.schemas import AnswerResponse, CitationSource


def render_answer(answer_result: AnswerResponse) -> None:
    """Render an answer, its citation warnings, and retrieved sources.

    Args:
        answer_result:
            Citation-aware response returned by the answer API.
    """

    st.subheader("Answer")
    st.markdown(answer_result.answer)

    for warning in answer_result.citation_warnings:
        labels = ", ".join(warning.labels)
        st.warning(f"{warning.message} Unsupported labels: {labels}")

    st.subheader("Sources")
    if not answer_result.sources:
        st.info("No sources were retrieved for this answer.")
        return

    for source in answer_result.sources:
        status = "Cited" if source.cited else "Uncited"
        section = f" · {source.section}" if source.section else ""
        title = (
            f"[{source.label}] {status} · {source.document_name}{section} "
            f"· distance {source.distance:.4f}"
        )
        with st.expander(title):
            provenance = _source_provenance(source)
            if provenance:
                st.caption(" · ".join(provenance))
            st.text(source.content)


def _source_provenance(source: CitationSource) -> list[str]:
    """Build display-safe Portfolio provenance for one citation source.

    Args:
        source:
            Citation source with optional Portfolio metadata.

    Returns:
        Ordered human-readable provenance values.
    """

    project_display_name = source.project_display_name
    project_slug = source.project_slug
    project = project_display_name or project_slug
    if project_display_name and project_slug:
        project = f"{project_display_name} ({project_slug})"

    document_type = source.document_type
    component_id = source.component_id
    generated_document = document_type
    if document_type and component_id:
        generated_document = f"{document_type}: {component_id}"

    values = [project, generated_document]
    repository_url = source.repository_url
    if repository_url:
        values.append(f"Repository: {repository_url}")
    resolved_commit_sha = source.resolved_commit_sha
    if resolved_commit_sha:
        values.append(f"Commit: {resolved_commit_sha}")
    return [value for value in values if value]
