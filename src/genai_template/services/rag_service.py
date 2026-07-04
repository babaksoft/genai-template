"""Retrieval-Augmented Generation service."""

from genai_template.components.context.context_builder import ContextBuilder
from genai_template.components.language_models.ollama_language_model import (
    OllamaLanguageModel,
)
from genai_template.components.prompt.prompt_builder import PromptBuilder
from genai_template.pipeline.retrieval_pipeline import RetrievalPipeline


class RagService:
    """Orchestrate the Retrieval-Augmented Generation workflow."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        language_model: OllamaLanguageModel,
    ) -> None:
        """Initialize the RAG service.

        Args:
            retrieval_pipeline:
                Retrieval pipeline.
            context_builder:
                Context builder.
            prompt_builder:
                Prompt builder.
            language_model:
                Language model.
        """
        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._language_model = language_model

    def answer(self, query: str) -> str:
        """Answer a user query using Retrieval-Augmented Generation.

        Args:
            query:
                User query.

        Returns:
            Generated answer.
        """

        retrieved_chunks = self._retrieval_pipeline.retrieve(query)
        context = self._context_builder.build(retrieved_chunks)
        prompt = self._prompt_builder.build(
            query=query,
            context=context,
        )

        return self._language_model.generate(prompt)
