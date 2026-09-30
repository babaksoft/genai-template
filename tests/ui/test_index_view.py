"""Tests for Streamlit index lifecycle rendering."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from genai_template.schemas import IndexStatus, IndexStatusReason
from genai_template.ui.index_view import render_index_status


def create_status(reason: IndexStatusReason) -> IndexStatus:
    """Create an index status for presentation tests.

    Args:
        reason:
            Status reason to render.

    Returns:
        Complete typed status.
    """

    return IndexStatus(
        source_id=1,
        rag_config_id=2,
        collection_name="idx-test",
        index_fingerprint="a" * 64,
        current_corpus_fingerprint="b" * 64,
        built_corpus_fingerprint="c" * 64,
        latest_build_id=9,
        build_started_at=datetime(2026, 9, 30, tzinfo=UTC),
        document_count=3,
        chunk_count=12,
        indexing_duration=0.25,
        latest_failure_detail=(
            "Index rebuild failed during indexing (RuntimeError)."
            if reason == IndexStatusReason.FAILED
            else None
        ),
        available=reason in {IndexStatusReason.CURRENT, IndexStatusReason.UNTRACKED},
        reason=reason,
    )


@pytest.mark.parametrize("reason", list(IndexStatusReason))
@patch("genai_template.ui.index_view.st")
def test_render_index_status_distinguishes_every_reason(
    st: MagicMock,
    reason: IndexStatusReason,
) -> None:
    """Every freshness reason should have an explicit operator-facing label.

    Args:
        st:
            Mocked Streamlit module.
        reason:
            Status reason under test.
    """

    render_index_status(create_status(reason))

    assert st.success.called or st.info.called or st.warning.called
    st.caption.assert_any_call(
        "Published corpus: bbbbbbbbbbbb · Indexed corpus: cccccccccccc"
    )
    if reason == IndexStatusReason.FAILED:
        st.error.assert_called_once_with(
            "Index rebuild failed during indexing (RuntimeError)."
        )
