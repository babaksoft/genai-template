"""Streamlit rendering helpers for deterministic index status."""

import streamlit as st

from genai_template.schemas import IndexBuildResponse, IndexStatus, IndexStatusReason

_STATUS_LABELS = {
    IndexStatusReason.CURRENT: "Current",
    IndexStatusReason.UNBUILT: "Not built",
    IndexStatusReason.STALE: "Stale",
    IndexStatusReason.BUILDING: "Building",
    IndexStatusReason.FAILED: "Failed",
    IndexStatusReason.COLLECTION_MISSING: "Collection missing",
    IndexStatusReason.COUNT_MISMATCH: "Collection count mismatch",
    IndexStatusReason.BACKEND_UNAVAILABLE: "Backend unavailable",
    IndexStatusReason.CORPUS_INVALID: "Corpus invalid",
    IndexStatusReason.UNTRACKED: "Untracked legacy source",
}


def render_index_status(index_status: IndexStatus) -> None:
    """Render an actionable index availability summary.

    Args:
        index_status:
            Verified status returned by the API.
    """

    label = _STATUS_LABELS[index_status.reason]
    message = f"Index status: {label}"
    if index_status.available:
        st.success(message)
    elif index_status.reason == IndexStatusReason.BUILDING:
        st.info(message)
    else:
        st.warning(f"{message}. Rebuild the selected index before asking.")

    current = _abbreviate_fingerprint(index_status.current_corpus_fingerprint)
    built = _abbreviate_fingerprint(index_status.built_corpus_fingerprint)
    if current is not None or built is not None:
        st.caption(
            f"Published corpus: {current or 'N/A'} · Indexed corpus: {built or 'N/A'}"
        )

    if index_status.latest_build_id is not None:
        details = [f"Build {index_status.latest_build_id}"]
        if index_status.document_count is not None:
            details.append(f"{index_status.document_count} document(s)")
        if index_status.chunk_count is not None:
            details.append(f"{index_status.chunk_count} chunk(s)")
        if index_status.indexing_duration is not None:
            details.append(f"{index_status.indexing_duration:.3f} s")
        st.caption(" · ".join(details))

    if index_status.latest_failure_detail:
        st.error(index_status.latest_failure_detail)


def render_build_result(build_result: IndexBuildResponse) -> None:
    """Render the durable outcome of a successful explicit rebuild.

    Args:
        build_result:
            Successful rebuild response returned by the API.
    """

    st.success(
        f"Build {build_result.build.id} indexed "
        f"{build_result.documents_indexed} document(s) into "
        f"{build_result.chunks_indexed} chunk(s) in "
        f"{build_result.indexing_time:.3f} s."
    )


def _abbreviate_fingerprint(value: str | None) -> str | None:
    """Abbreviate a corpus fingerprint for operator display.

    Args:
        value:
            Full SHA-256 fingerprint, when available.

    Returns:
        Twelve-character display form or ``None``.
    """

    return value[:12] if value is not None else None
