"""Tests for standard corpus ingestion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from genai_template.config import load_rag_config
from genai_template.ingest import main


@patch("genai_template.ingest.IndexingPipeline")
@patch("genai_template.ingest.create_vector_store")
@patch("genai_template.ingest.create_embedder")
@patch("genai_template.ingest.create_splitter")
@patch("genai_template.ingest.load_rag_config")
def test_main_uses_default_config_factories(
    mock_load_config: MagicMock,
    mock_create_splitter: MagicMock,
    mock_create_embedder: MagicMock,
    mock_create_store: MagicMock,
    mock_pipeline: MagicMock,
    tmp_path: Path,
) -> None:
    """Standard ingestion should use factories and retain its collection name."""

    config = load_rag_config()
    mock_load_config.return_value = config

    with patch("genai_template.ingest.settings.CORPORA_DIR", tmp_path):
        main()

    mock_create_splitter.assert_called_once_with(config.splitter)
    mock_create_embedder.assert_called_once_with(config.embedder)
    store_config = mock_create_store.call_args.args[0]
    assert store_config.collection_name == "baseline_corpus"
    assert store_config.persist_directory == config.vector_store.persist_directory
    mock_pipeline.return_value.run.assert_called_once_with(tmp_path / "baseline")
