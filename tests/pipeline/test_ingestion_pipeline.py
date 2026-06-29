"""End-to-end tests for the ingestion pipeline."""

from pathlib import Path

from genai_template.pipeline.ingestion_pipeline import (
    IngestionPipeline,
)


def test_ingestion_pipeline(tmp_path: Path) -> None:
    """Pipeline should produce canonical document chunks."""

    document = tmp_path / "document.md"

    document.write_text(
        ("# Sample Document\n\n" "This is a sample document. " * 300),
        encoding="utf-8",
    )

    pipeline = IngestionPipeline()

    chunks = pipeline.run(tmp_path)

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.id
        assert chunk.document_id == "document.md"
        assert chunk.text
        assert chunk.metadata
        assert chunk.embedding is None
