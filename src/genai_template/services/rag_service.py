"""Retrieval-Augmented Generation execution service."""

import logging

from genai_template.components.context import ContextBuilder, resolve_citations
from genai_template.components.prompt import PromptBuilder
from genai_template.config import RagConfig, index_config_fingerprint
from genai_template.factories import (
    create_embedder,
    create_llm,
    create_retrieval_pipeline,
    create_vector_store,
)
from genai_template.observability import INPUT_VALUE, OUTPUT_VALUE, application_span
from genai_template.protocols import Retriever, VectorStore
from genai_template.schemas import RagResult, RunMetrics
from genai_template.services.experiment_service import ExperimentService
from genai_template.services.rag_config_service import RagConfigService
from genai_template.services.source_service import SourceService
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class IndexNotBuiltError(ValueError):
    """Raised when execution selects an index that has not been built."""


class RagService:
    """Load persisted configuration and execute RAG runs."""

    def __init__(
        self,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        experiment_service: ExperimentService,
        rag_config_service: RagConfigService,
        source_service: SourceService,
    ) -> None:
        """Initialize the RAG execution service.

        Args:
            context_builder:
                Context builder.
            prompt_builder:
                Prompt builder.
            experiment_service:
                Experiment and run persistence service.
            rag_config_service:
                Immutable configuration registry.
            source_service:
                Source registry and deterministic index service.
        """

        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._experiment_service = experiment_service
        self._rag_config_service = rag_config_service
        self._source_service = source_service

    def answer(
        self,
        query: str,
        experiment_id: int,
        rag_config_id: int,
    ) -> RagResult:
        """Execute a persisted configuration for a source-bound experiment.

        Args:
            query:
                User query.
            experiment_id:
                Canonical experiment identifier.
            rag_config_id:
                Canonical RAG configuration identifier.

        Returns:
            Generated answer, runtime metrics, sources, and citation warnings.

        Raises:
            IndexNotBuiltError:
                If the selected deterministic source index does not exist.
            ValueError:
                If the experiment, source, or configuration does not exist.
        """

        experiment = self._experiment_service.get_experiment(experiment_id)
        source = self._source_service.get_source(experiment.source_id)
        config_record = self._rag_config_service.get_config(rag_config_id)
        config = self._rag_config_service.parse_config(config_record)
        collection_name = self._source_service.index_collection_name(source.id, config)
        store = create_vector_store(config.vector_store, collection_name)
        if not store.exists():
            raise IndexNotBuiltError(
                f"Index for source {source.id} and RAG config {rag_config_id} "
                "has not been built."
            )

        run = self._experiment_service.start_run(
            experiment_id=experiment.id,
            rag_config_id=config_record.id,
            query=query,
        )
        retrieval_pipeline = self._create_retrieval_pipeline(config, store)
        language_model = create_llm(config.llm)

        with application_span(
            "rag.answer",
            "CHAIN",
            {
                INPUT_VALUE: query,
                "rag.top_k": config.retrieval.top_k,
                "rag.experiment.id": experiment.id,
                "rag.source.id": source.id,
                "rag.config.id": config_record.id,
                "rag.config.fingerprint": config_record.config_fingerprint,
                "rag.index.fingerprint": index_config_fingerprint(config),
                "rag.index.collection": collection_name,
                "rag.embedding.model": config.embedder.model_name,
                "rag.vector_store.type": config.vector_store.type,
                "rag.llm.model": config.llm.model_name,
            },
        ) as span:
            with Timer() as total_timer:
                with Timer() as retrieval_timer:
                    retrieved_chunks = retrieval_pipeline.retrieve(
                        query, config.retrieval.top_k
                    )

                citation_context = self._context_builder.build(retrieved_chunks)
                prompt = self._prompt_builder.build(
                    query=query, context=citation_context.text
                )

                with Timer() as generation_timer:
                    response = language_model.generate(prompt)

                sources, citation_warnings = resolve_citations(
                    response, citation_context.sources
                )

            distances = [chunk.distance for chunk in retrieved_chunks]
            metrics = RunMetrics(
                query=query,
                embedding_model=config.embedder.model_name,
                vector_store=config.vector_store.type.capitalize(),
                llm_model=config.llm.model_name,
                top_k=config.retrieval.top_k,
                retrieved_chunks=len(retrieved_chunks),
                best_distance=min(distances) if distances else None,
                worst_distance=max(distances) if distances else None,
                context_length=len(citation_context.text),
                prompt_length=len(prompt),
                response_length=len(response),
                retrieval_time=retrieval_timer.elapsed,
                generation_time=generation_timer.elapsed,
                total_time=total_timer.elapsed,
            )
            span.set_attribute(OUTPUT_VALUE, response)

        self._experiment_service.complete_run(run=run, metrics=metrics)
        logger.info(
            "RAG run %d completed for experiment %d, source %d, and RAG config "
            "%d using index '%s' in %.3f second(s).",
            run.id,
            experiment.id,
            source.id,
            config_record.id,
            collection_name,
            total_timer.elapsed,
        )
        return RagResult(
            answer=response,
            metrics=metrics,
            sources=sources,
            citation_warnings=citation_warnings,
        )

    @staticmethod
    def _create_retrieval_pipeline(config: RagConfig, store: VectorStore) -> Retriever:
        """Construct retrieval components from a persisted configuration.

        Args:
            config:
                Selected persisted RAG configuration.
            store:
                Vector store bound to the deterministic source index.

        Returns:
            Configured retrieval pipeline.
        """

        return create_retrieval_pipeline(
            config.retrieval,
            create_embedder(config.embedder),
            store,
        )
