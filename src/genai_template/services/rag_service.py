"""Retrieval-Augmented Generation service."""

import logging

from genai_template.components.context import ContextBuilder
from genai_template.components.language_models import (
    OllamaLanguageModel,
)
from genai_template.components.prompt import PromptBuilder
from genai_template.config import settings
from genai_template.pipelines import RetrievalPipeline
from genai_template.schemas import RagResult, RunMetrics
from genai_template.services import ExperimentService
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class RagService:
    """Orchestrate the Retrieval-Augmented Generation workflow."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        language_model: OllamaLanguageModel,
        experiment_service: ExperimentService,
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
            experiment_service:
                Experiment tracking service.
        """
        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._language_model = language_model
        self._experiment_service = experiment_service

    def answer(self, query: str) -> RagResult:
        """Answer a user query using Retrieval-Augmented Generation.

        Args:
            query:
                User query.

        Returns:
            RAG result that includes generated answer, run metrics, etc.
        """

        run = self._experiment_service.start_run(
            experiment_name=settings.EXPERIMENT_NAME,
        )

        with Timer() as total_timer:
            with Timer() as retrieval_timer:
                retrieved_chunks = self._retrieval_pipeline.retrieve(
                    query, settings.TOP_K
                )

            context = self._context_builder.build(retrieved_chunks)
            prompt = self._prompt_builder.build(
                query=query,
                context=context,
            )

            with Timer() as generation_timer:
                response = self._language_model.generate(prompt)

        distances = [chunk.distance for chunk in retrieved_chunks]

        metrics = RunMetrics(
            query=query,
            embedding_model=settings.EMBEDDING_MODEL,
            vector_store=settings.VECTOR_STORE,
            llm_model=settings.LLM_MODEL,
            top_k=settings.TOP_K,
            retrieved_chunks=len(retrieved_chunks),
            best_distance=min(distances) if distances else None,
            worst_distance=max(distances) if distances else None,
            context_length=len(context),
            prompt_length=len(prompt),
            response_length=len(response),
            retrieval_time=retrieval_timer.elapsed,
            generation_time=generation_timer.elapsed,
            total_time=total_timer.elapsed,
        )

        self._experiment_service.complete_run(
            run=run,
            metrics=metrics,
        )

        logger.info(
            "RAG request completed in %.3f second(s).",
            total_timer.elapsed,
        )

        return RagResult(
            answer=response,
            metrics=metrics,
            retrieved_chunks=retrieved_chunks,
        )
