"""Tests for the RagService."""

from unittest.mock import MagicMock, patch

from genai_template.config import load_rag_config
from genai_template.db.models import Source
from genai_template.schemas import RetrievedChunk
from genai_template.services import RagService


@patch("genai_template.services.rag_service.create_vector_store")
@patch("genai_template.services.rag_service.create_embedder")
@patch("genai_template.services.rag_service.create_retrieval_pipeline")
def test_answer_orchestrates_rag_workflow(
    mock_create_retrieval: MagicMock,
    mock_create_embedder: MagicMock,
    mock_create_store: MagicMock,
) -> None:
    """The service should orchestrate the complete RAG workflow."""

    retrieval_pipeline = MagicMock()
    context_builder = MagicMock()
    prompt_builder = MagicMock()
    language_model = MagicMock()
    experiment_service = MagicMock()
    source_service = MagicMock()

    retrieved_chunks: list[RetrievedChunk] = []

    retrieval_pipeline.retrieve.return_value = retrieved_chunks
    context_builder.build.return_value = "context"
    prompt_builder.build.return_value = "prompt"
    language_model.generate.return_value = "final answer"
    source_service.get_source.return_value = Source(
        id=7,
        name="product-docs",
        directory="/corpora/product-docs",
        collection_name="source-product-docs",
        documents_indexed=1,
        chunks_indexed=2,
        indexing_time=0.1,
    )
    mock_create_retrieval.return_value = retrieval_pipeline
    default_config = load_rag_config()
    config = default_config.model_copy(
        update={
            "experiment": default_config.experiment.model_copy(
                update={"name": "Configured experiment"}
            ),
            "embedder": default_config.embedder.model_copy(
                update={"model_name": "configured-embedder"}
            ),
            "retrieval": default_config.retrieval.model_copy(update={"top_k": 9}),
            "llm": default_config.llm.model_copy(
                update={"model_name": "configured-llm"}
            ),
        }
    )

    service = RagService(
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        language_model=language_model,
        experiment_service=experiment_service,
        source_service=source_service,
        config=config,
    )

    result = service.answer("What is RAG?", source_id=7)

    retrieval_pipeline.retrieve.assert_called_once_with(
        "What is RAG?",
        9,
    )
    context_builder.build.assert_called_once_with(retrieved_chunks)
    prompt_builder.build.assert_called_once_with(
        query="What is RAG?",
        context="context",
    )
    language_model.generate.assert_called_once_with("prompt")
    source_service.get_source.assert_called_once_with(7)
    mock_create_embedder.assert_called_once_with(config.embedder)
    store_config = mock_create_store.call_args.args[0]
    assert store_config.collection_name == "source-product-docs"
    assert store_config.persist_directory == config.vector_store.persist_directory
    mock_create_retrieval.assert_called_once_with(
        config.retrieval,
        mock_create_embedder.return_value,
        mock_create_store.return_value,
    )
    experiment_service.start_run.assert_called_once_with(
        experiment_name="Configured experiment",
        source_id=7,
        config=config,
    )

    assert result.answer == "final answer"
    assert result.metrics.embedding_model == "configured-embedder"
    assert result.metrics.vector_store == "Chroma"
    assert result.metrics.llm_model == "configured-llm"
    assert result.metrics.top_k == 9
