"""Unit tests for the indexing pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from genai_template.pipeline.indexing_pipeline import (
    IndexingPipeline,
)
from genai_template.schemas.chunk import DocumentChunk


def test_run(
    tmp_path: Path,
) -> None:
    """The indexing pipeline should index document chunks."""

    document = tmp_path / "document.md"
    document.write_text(
        "# Title\n\nThis is a test document.",
        encoding="utf-8",
    )

    chunk = DocumentChunk(
        id="chunk-001",
        document_id="document.md",
        text="This is a test document.",
        metadata={},
        embedding=[0.1, 0.2, 0.3],
    )

    mock_store = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [chunk]

    pipeline = IndexingPipeline(
        embedder=mock_embedder,
        store=mock_store,
    )
    result = pipeline.run(tmp_path)

    assert result == [chunk]

    mock_embedder.embed.assert_called_once()
    mock_store.upsert.assert_called_once_with([chunk])


def test_run_empty_directory(
    tmp_path: Path,
) -> None:
    """An empty directory should produce no chunks."""

    mock_store = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = []

    pipeline = IndexingPipeline(
        embedder=mock_embedder,
        store=mock_store,
    )
    result = pipeline.run(tmp_path)

    assert result == []

    mock_embedder.embed.assert_called_once_with([])
    mock_store.upsert.assert_called_once_with([])
