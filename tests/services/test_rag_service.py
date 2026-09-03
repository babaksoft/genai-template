"""Tests for the RagService."""

from unittest.mock import MagicMock

from genai_template.config import settings
from genai_template.db.models import Source
from genai_template.schemas import RetrievedChunk
from genai_template.services import RagService


def test_answer_orchestrates_rag_workflow() -> None:
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
    retrieval_pipeline_factory = MagicMock(return_value=retrieval_pipeline)

    service = RagService(
        retrieval_pipeline_factory=retrieval_pipeline_factory,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        language_model=language_model,
        experiment_service=experiment_service,
        source_service=source_service,
    )

    result = service.answer("What is RAG?", source_id=7)

    retrieval_pipeline.retrieve.assert_called_once_with(
        "What is RAG?",
        settings.TOP_K,
    )
    context_builder.build.assert_called_once_with(retrieved_chunks)
    prompt_builder.build.assert_called_once_with(
        query="What is RAG?",
        context="context",
    )
    language_model.generate.assert_called_once_with("prompt")
    source_service.get_source.assert_called_once_with(7)
    retrieval_pipeline_factory.assert_called_once_with("source-product-docs")
    experiment_service.start_run.assert_called_once_with(
        experiment_name=settings.EXPERIMENT_NAME,
        source_id=7,
    )

    assert result.answer == "final answer"
