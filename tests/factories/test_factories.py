"""Tests for configured RAG component factories."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from genai_template.common.types import VectorDistance
from genai_template.config.rag import (
    EmbedderConfig,
    LLMConfig,
    MarkdownSplitterConfig,
    RetrievalConfig,
    SplitterConfig,
    VectorStoreConfig,
)
from genai_template.factories import (
    create_embedder,
    create_llm,
    create_retrieval_pipeline,
    create_splitter,
    create_vector_store,
)


@patch("genai_template.factories.splitter_factory.DocumentSplitter")
def test_create_sentence_splitter(mock_splitter: MagicMock) -> None:
    """The splitter factory should dispatch all configured arguments."""

    config = SplitterConfig(
        type="sentence",
        chunk_size=256,
        chunk_overlap=32,
    )

    result = create_splitter(config)

    assert result is mock_splitter.return_value
    mock_splitter.assert_called_once_with(chunk_size=256, chunk_overlap=32)


@patch("genai_template.factories.splitter_factory.MarkdownDocumentSplitter")
def test_create_markdown_splitter(mock_splitter: MagicMock) -> None:
    """The splitter factory should forward the header path separator."""

    config = MarkdownSplitterConfig(
        type="markdown",
        header_path_separator=" > ",
    )

    result = create_splitter(config)

    assert result is mock_splitter.return_value
    mock_splitter.assert_called_once_with(header_path_separator=" > ")


@patch("genai_template.factories.embedder_factory.FastEmbedEmbeddingModel")
def test_create_fastembed_embedder(mock_embedder: MagicMock) -> None:
    """The embedder factory should dispatch the configured model."""

    config = EmbedderConfig(type="fastembed", model_name="custom-embedder")

    result = create_embedder(config)

    assert result is mock_embedder.return_value
    mock_embedder.assert_called_once_with(model_name="custom-embedder")


@patch("genai_template.factories.vector_store_factory.ChromaStore")
def test_create_chroma_store(mock_store: MagicMock, tmp_path: Path) -> None:
    """The vector store factory should inject every storage setting."""

    config = VectorStoreConfig(
        type="chroma",
        collection_name="experiment",
        persist_directory=tmp_path,
        distance=VectorDistance.INNER_PRODUCT,
    )

    result = create_vector_store(config)

    assert result is mock_store.return_value
    mock_store.assert_called_once_with(
        persist_directory=tmp_path,
        collection_name="experiment",
        distance=VectorDistance.INNER_PRODUCT,
    )


@patch("genai_template.factories.llm_factory.OllamaLanguageModel")
def test_create_ollama_llm(mock_llm: MagicMock) -> None:
    """The LLM factory should inject model, endpoint, and timeout."""

    config = LLMConfig(
        type="ollama",
        model_name="custom-llm",
        base_url="http://ollama.example:11434",
        request_timeout=45,
    )

    result = create_llm(config)

    assert result is mock_llm.return_value
    mock_llm.assert_called_once_with(
        model_name="custom-llm",
        base_url="http://ollama.example:11434",
        request_timeout=45.0,
    )


@patch("genai_template.factories.retriever_factory.RetrievalPipeline")
def test_create_retrieval_pipeline(mock_pipeline: MagicMock) -> None:
    """The retrieval factory should connect components and configured top-k."""

    embedder = MagicMock()
    store = MagicMock()
    config = RetrievalConfig(top_k=8)

    result = create_retrieval_pipeline(config, embedder, store)

    assert result is mock_pipeline.return_value
    mock_pipeline.assert_called_once_with(
        embedder=embedder,
        store=store,
        top_k=8,
    )
