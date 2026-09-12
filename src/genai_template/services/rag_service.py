"""Retrieval-Augmented Generation service."""

import logging

from genai_template.components.context import ContextBuilder
from genai_template.components.prompt import PromptBuilder
from genai_template.config import RagConfig, config_fingerprint, settings
from genai_template.factories import (
    create_embedder,
    create_retrieval_pipeline,
    create_vector_store,
)
from genai_template.observability import INPUT_VALUE, OUTPUT_VALUE, application_span
from genai_template.protocols import LanguageModel, Retriever
from genai_template.schemas import RagResult, RunMetrics
from genai_template.services.experiment_service import ExperimentService
from genai_template.services.source_service import SourceService
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class RagService:
    """Orchestrate the Retrieval-Augmented Generation workflow."""

    def __init__(
        self,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        language_model: LanguageModel,
        experiment_service: ExperimentService,
        source_service: SourceService,
        config: RagConfig,
    ) -> None:
        """Initialize the RAG service.

        Args:
            context_builder:
                Context builder.
            prompt_builder:
                Prompt builder.
            language_model:
                Language model.
            experiment_service:
                Experiment tracking service.
            source_service:
                Service used to resolve active sources.
            config:
                Fully resolved configuration for this service instance.
        """

        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._language_model = language_model
        self._experiment_service = experiment_service
        self._source_service = source_service
        self._config = config
        self._config_fingerprint = config_fingerprint(config)

    def answer(self, query: str, source_id: int) -> RagResult:
        """Answer a user query using Retrieval-Augmented Generation.

        Args:
            query:
                User query.
            source_id:
                Identifier of the source to retrieve from.

        Returns:
            RAG result that includes generated answer, run metrics, etc.
        """

        source = self._source_service.get_source(source_id)
        retrieval_pipeline = self._get_retrieval_pipeline(source.collection_name)

        run = self._experiment_service.start_run(
            experiment_name=settings.EXPERIMENT_NAME,
            source_id=source.id,
            config=self._config,
        )

        with application_span(
            "rag.answer",
            "CHAIN",
            {
                INPUT_VALUE: query,
                "rag.top_k": self._config.retrieval.top_k,
                "rag.experiment.name": settings.EXPERIMENT_NAME,
                "rag.config.fingerprint": self._config_fingerprint,
                "rag.embedding.model": self._config.embedder.model_name,
                "rag.vector_store.type": self._config.vector_store.type,
                "rag.llm.model": self._config.llm.model_name,
            },
        ) as span:
            with Timer() as total_timer:
                with Timer() as retrieval_timer:
                    retrieved_chunks = retrieval_pipeline.retrieve(
                        query, self._config.retrieval.top_k
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
                embedding_model=self._config.embedder.model_name,
                vector_store=self._config.vector_store.type.capitalize(),
                llm_model=self._config.llm.model_name,
                top_k=self._config.retrieval.top_k,
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
            span.set_attribute(OUTPUT_VALUE, response)

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

    def _get_retrieval_pipeline(self, collection_name: str) -> Retriever:
        """Create a retrieval pipeline for a source-owned collection.

        Args:
            collection_name:
                Name of the source collection to query.

        Returns:
            Retrieval pipeline using the configured embedder and store.
        """

        return create_retrieval_pipeline(
            self._config.retrieval,
            create_embedder(self._config.embedder),
            create_vector_store(self._config.vector_store, collection_name),
        )
