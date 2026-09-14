"""End-to-end integration coverage for the redesigned registry data model."""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from genai_template.common.types import VectorDistance
from genai_template.components.context import ContextBuilder
from genai_template.components.prompt import PromptBuilder
from genai_template.config import load_rag_config
from genai_template.db.base import Base
from genai_template.db.models import Run
from genai_template.schemas import IndexingResult
from genai_template.services import (
    ExperimentService,
    IndexNotBuiltError,
    RagConfigService,
    RagService,
    SourceService,
)


class InMemoryVectorStore:
    """Minimal vector-store lifecycle implementation shared by the workflow."""

    def __init__(self, collection_name: str, built: set[str]) -> None:
        """Initialize an in-memory collection handle.

        Args:
            collection_name:
                Deterministic collection name.
            built:
                Shared set of existing collection names.
        """

        self.collection_name = collection_name
        self._built = built

    def create(self, vector_size: int) -> None:
        """Mark the collection as existing.

        Args:
            vector_size:
                Unused vector dimensionality.
        """

        del vector_size
        self._built.add(self.collection_name)

    def exists(self) -> bool:
        """Return whether the collection was explicitly built.

        Returns:
            Whether the collection exists, including as an empty index.
        """

        return self.collection_name in self._built

    def count(self) -> int:
        """Return the number of stored records.

        Returns:
            Zero because this lifecycle fake represents empty indexes.
        """

        return 0

    def delete(self) -> None:
        """Delete the collection if it exists."""

        self._built.discard(self.collection_name)

    def upsert(self, chunks: list[Any]) -> None:
        """Mark the collection as existing after an upsert.

        Args:
            chunks:
                Unused chunks supplied to the fake store.
        """

        del chunks
        self._built.add(self.collection_name)

    def search(
        self,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
    ) -> list[Any]:
        """Return no chunks from the explicitly empty index.

        Args:
            embedding:
                Unused query embedding.
            top_k:
                Unused result limit.
            query:
                Unused query text.

        Returns:
            An empty result list.
        """

        del embedding, top_k, query
        return []


class EmptyIndexingPipeline:
    """Indexing pipeline that explicitly creates an empty collection."""

    def __init__(self, store: InMemoryVectorStore) -> None:
        """Initialize the pipeline.

        Args:
            store:
                Collection handle to create.
        """

        self._store = store

    def run(self, data_dir: Path) -> IndexingResult:
        """Build an empty collection for a registered source directory.

        Args:
            data_dir:
                Existing source directory.

        Returns:
            Empty-index build metrics.
        """

        assert data_dir.is_dir()
        self._store.create(1)
        return IndexingResult(
            documents_indexed=0,
            chunks_indexed=0,
            indexing_time=0.0,
        )


class EmptyRetriever:
    """Retriever that executes successfully against an empty index."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[Any]:
        """Return an empty retrieval result.

        Args:
            query:
                User query.
            top_k:
                Optional result limit.

        Returns:
            An empty result list.
        """

        del query, top_k
        return []


class FixedLanguageModel:
    """Language model returning a deterministic integration-test response."""

    def generate(self, prompt: str) -> str:
        """Generate a fixed response.

        Args:
            prompt:
                Rendered RAG prompt.

        Returns:
            Deterministic response text.
        """

        assert prompt
        return "Integration answer"


@pytest.mark.integration
def test_registry_index_and_run_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise shared identities, index selection, runs, and summaries together."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    config_service = RagConfigService(factory)
    source_service = SourceService(factory, tmp_path, config_service)
    experiment_service = ExperimentService(factory)

    corpus = tmp_path / "docs"
    corpus.mkdir()
    source = source_service.register("docs")
    first_experiment = experiment_service.create_experiment(source.id, "First")
    second_experiment = experiment_service.create_experiment(source.id, "Second")

    default = load_rag_config()
    shared_index = default.model_copy(
        update={
            "retrieval": default.retrieval.model_copy(update={"top_k": 9}),
            "llm": default.llm.model_copy(update={"model_name": "other-llm"}),
        }
    )
    changed_embedder = default.model_copy(
        update={
            "embedder": default.embedder.model_copy(
                update={"model_name": "other-embedder"}
            )
        }
    )
    changed_distance = default.model_copy(
        update={
            "vector_store": default.vector_store.model_copy(
                update={"distance": VectorDistance.L2}
            )
        }
    )
    default_record = config_service.register_config(default)
    shared_record = config_service.register_config(shared_index)
    embedder_record = config_service.register_config(changed_embedder)
    distance_record = config_service.register_config(changed_distance)

    assert config_service.register_config(default).id == default_record.id
    default_collection = source_service.index_collection_name(source.id, default)
    assert default_collection == source_service.index_collection_name(
        source.id, shared_index
    )
    assert default_collection != source_service.index_collection_name(
        source.id, changed_embedder
    )
    assert default_collection != source_service.index_collection_name(
        source.id, changed_distance
    )

    built_collections: set[str] = set()

    def create_store(_config: Any, collection_name: str) -> InMemoryVectorStore:
        """Create a handle backed by the shared collection registry.

        Args:
            _config:
                Unused vector-store configuration.
            collection_name:
                Runtime deterministic collection name.

        Returns:
            In-memory collection handle.
        """

        return InMemoryVectorStore(collection_name, built_collections)

    def create_indexing_pipeline(
        _config: Any, store: InMemoryVectorStore
    ) -> EmptyIndexingPipeline:
        """Create the empty-index pipeline used by this workflow.

        Args:
            _config:
                Unused persisted RAG configuration.
            store:
                Selected deterministic collection.

        Returns:
            Empty indexing pipeline.
        """

        return EmptyIndexingPipeline(store)

    monkeypatch.setattr(
        "genai_template.services.source_service.create_vector_store", create_store
    )
    monkeypatch.setattr(
        source_service, "_create_indexing_pipeline", create_indexing_pipeline
    )
    source_service.rebuild_index(source.id, default_record.id)
    assert default_collection in built_collections

    monkeypatch.setattr(
        "genai_template.services.rag_service.create_vector_store", create_store
    )
    monkeypatch.setattr(
        "genai_template.services.rag_service.create_embedder", lambda _config: object()
    )
    monkeypatch.setattr(
        "genai_template.services.rag_service.create_retrieval_pipeline",
        lambda _config, _embedder, _store: EmptyRetriever(),
    )
    monkeypatch.setattr(
        "genai_template.services.rag_service.create_llm",
        lambda _config: FixedLanguageModel(),
    )
    rag_service = RagService(
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        experiment_service=experiment_service,
        rag_config_service=config_service,
        source_service=source_service,
    )

    rag_service.answer("Default config", first_experiment.id, default_record.id)
    rag_service.answer("Shared index", first_experiment.id, shared_record.id)
    rag_service.answer("Reused config", second_experiment.id, default_record.id)
    unfinished = experiment_service.start_run(
        first_experiment.id, shared_record.id, "Provider failed"
    )

    with factory() as session:
        runs_before_missing = session.scalar(select(func.count(Run.id)))
    with pytest.raises(IndexNotBuiltError, match="has not been built"):
        rag_service.answer("Missing index", first_experiment.id, embedder_record.id)
    with factory() as session:
        runs_after_missing = session.scalar(select(func.count(Run.id)))
    assert runs_after_missing == runs_before_missing

    summary = experiment_service.summarize_experiment(first_experiment.id)
    filtered = experiment_service.summarize_experiment(
        first_experiment.id, default_record.id
    )
    assert summary.run_count == 2
    assert summary.rag_config_id is None
    assert filtered.run_count == 1

    with factory() as session:
        runs = list(session.scalars(select(Run).order_by(Run.id)))
    completed = [run for run in runs if run.finished_at is not None]
    persisted_unfinished = next(run for run in runs if run.id == unfinished.id)
    assert len(completed) == 3
    assert persisted_unfinished.finished_at is None
    assert persisted_unfinished.retrieved_chunks is None
    assert persisted_unfinished.total_time is None
    assert all(
        run.experiment_id in {first_experiment.id, second_experiment.id} for run in runs
    )
    assert distance_record.id != embedder_record.id
