"""Tests for the RagService."""

from unittest.mock import MagicMock

from genai_template.services.rag_service import RagService


def test_answer_orchestrates_rag_workflow() -> None:
    """The service should orchestrate the complete RAG workflow."""

    retrieval_pipeline = MagicMock()
    context_builder = MagicMock()
    prompt_builder = MagicMock()
    language_model = MagicMock()

    retrieved_chunks = object()

    retrieval_pipeline.retrieve.return_value = retrieved_chunks
    context_builder.build.return_value = "context"
    prompt_builder.build.return_value = "prompt"
    language_model.generate.return_value = "final answer"

    service = RagService(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        language_model=language_model,
    )

    answer = service.answer("What is RAG?")

    retrieval_pipeline.retrieve.assert_called_once_with("What is RAG?")
    context_builder.build.assert_called_once_with(retrieved_chunks)
    prompt_builder.build.assert_called_once_with(
        query="What is RAG?",
        context="context",
    )
    language_model.generate.assert_called_once_with("prompt")

    assert answer == "final answer"
