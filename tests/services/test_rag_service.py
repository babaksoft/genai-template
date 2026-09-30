"""Tests for persisted-config RAG execution."""

from unittest.mock import MagicMock, patch

import pytest

from genai_template.config import load_rag_config
from genai_template.db.models import Experiment
from genai_template.db.models import RagConfig as RagConfigRecord
from genai_template.db.models import Source
from genai_template.schemas import (
    CitationContext,
    CitationSource,
    DocumentChunk,
    IndexStatus,
    IndexStatusReason,
    RetrievedChunk,
)
from genai_template.services import IndexUnavailableError, RagService


def create_service() -> tuple[RagService, dict[str, MagicMock]]:
    """Create a RAG service with mocked collaborators.

    Returns:
        Service and collaborator mocks keyed by role.
    """

    collaborators = {
        "context": MagicMock(),
        "prompt": MagicMock(),
        "experiments": MagicMock(),
        "configs": MagicMock(),
        "sources": MagicMock(),
    }
    service = RagService(
        context_builder=collaborators["context"],
        prompt_builder=collaborators["prompt"],
        experiment_service=collaborators["experiments"],
        rag_config_service=collaborators["configs"],
        source_service=collaborators["sources"],
    )
    return service, collaborators


def make_index_status(
    config_fingerprint: str,
    *,
    available: bool = True,
    reason: IndexStatusReason = IndexStatusReason.UNTRACKED,
) -> IndexStatus:
    """Create a typed status for RAG service tests.

    Args:
        config_fingerprint:
            Selected index fingerprint.
        available:
            Whether the status permits execution.
        reason:
            Machine-readable availability reason.

    Returns:
        Consistent test status.
    """

    return IndexStatus(
        source_id=7,
        rag_config_id=5,
        collection_name="idx-selected",
        index_fingerprint=config_fingerprint,
        available=available,
        reason=reason,
    )


@patch("genai_template.services.rag_service.create_llm")
@patch("genai_template.services.rag_service.create_vector_store")
@patch("genai_template.services.rag_service.create_embedder")
@patch("genai_template.services.rag_service.create_retrieval_pipeline")
def test_answer_loads_config_and_completes_canonical_run(
    mock_create_retrieval: MagicMock,
    mock_create_embedder: MagicMock,
    mock_create_store: MagicMock,
    mock_create_llm: MagicMock,
) -> None:
    """Execution should resolve IDs and dynamically compose every component."""

    service, mocks = create_service()
    config = load_rag_config().model_copy(
        update={
            "embedder": load_rag_config().embedder.model_copy(
                update={"model_name": "configured-embedder"}
            ),
            "retrieval": load_rag_config().retrieval.model_copy(update={"top_k": 9}),
            "llm": load_rag_config().llm.model_copy(
                update={"model_name": "configured-llm"}
            ),
        }
    )
    experiment = Experiment(id=3, source_id=7, name="Trial")
    source = Source(id=7, name="docs", directory="/corpora/docs")
    record = RagConfigRecord(
        id=5, config_fingerprint="a" * 64, config_json=config.model_dump_json()
    )
    run = MagicMock(id=11)
    store = mock_create_store.return_value
    store.exists.return_value = True
    retrieval = mock_create_retrieval.return_value
    retrieved_chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                id="chunk-1",
                document_id="guide.md",
                text="RAG uses retrieval.",
            ),
            distance=0.25,
        )
    ]
    retrieval.retrieve.return_value = retrieved_chunks
    mocks["experiments"].get_experiment.return_value = experiment
    mocks["experiments"].start_run.return_value = run
    mocks["sources"].get_source.return_value = source
    mocks["configs"].get_config.return_value = record
    mocks["configs"].parse_config.return_value = config
    mocks["sources"].get_index_status.return_value = make_index_status(
        config_fingerprint="a" * 64
    )
    citation_source = CitationSource(
        label="S1",
        chunk_id="chunk-1",
        document_name="guide.md",
        section=None,
        content="RAG uses retrieval.",
        distance=0.25,
        cited=False,
    )
    mocks["context"].build.return_value = CitationContext(
        text="[S1] labeled context", sources=[citation_source]
    )
    mocks["prompt"].build.return_value = "prompt"
    mock_create_llm.return_value.generate.return_value = "final answer [S1] [S9]"

    result = service.answer("What is RAG?", experiment_id=3, rag_config_id=5)

    mocks["experiments"].start_run.assert_called_once_with(
        experiment_id=3, rag_config_id=5, query="What is RAG?"
    )
    mock_create_store.assert_called_once_with(config.vector_store, "idx-selected")
    mock_create_embedder.assert_called_once_with(config.embedder)
    mock_create_retrieval.assert_called_once_with(
        config.retrieval, mock_create_embedder.return_value, store
    )
    mock_create_llm.assert_called_once_with(config.llm)
    retrieval.retrieve.assert_called_once_with("What is RAG?", 9)
    mocks["context"].build.assert_called_once_with(retrieved_chunks)
    mocks["prompt"].build.assert_called_once_with(
        query="What is RAG?", context="[S1] labeled context"
    )
    mocks["experiments"].complete_run.assert_called_once_with(
        run=run, metrics=result.metrics
    )
    assert result.answer == "final answer [S1] [S9]"
    assert result.sources[0].cited is True
    assert result.citation_warnings[0].labels == ["S9"]
    assert result.metrics.retrieved_chunks == 1
    assert result.metrics.context_length == len("[S1] labeled context")
    assert result.metrics.embedding_model == "configured-embedder"
    assert result.metrics.llm_model == "configured-llm"


@patch("genai_template.services.rag_service.create_vector_store")
@pytest.mark.parametrize(
    "reason",
    [
        IndexStatusReason.UNBUILT,
        IndexStatusReason.STALE,
        IndexStatusReason.BUILDING,
        IndexStatusReason.FAILED,
        IndexStatusReason.COLLECTION_MISSING,
        IndexStatusReason.COUNT_MISMATCH,
        IndexStatusReason.BACKEND_UNAVAILABLE,
        IndexStatusReason.CORPUS_INVALID,
        IndexStatusReason.UNTRACKED,
    ],
)
def test_answer_rejects_missing_index_before_creating_run(
    mock_create_store: MagicMock,
    reason: IndexStatusReason,
) -> None:
    """Every unavailable state must reject before creating a run or store."""

    service, mocks = create_service()
    config = load_rag_config()
    mocks["experiments"].get_experiment.return_value = Experiment(
        id=3, source_id=7, name="Trial"
    )
    mocks["sources"].get_source.return_value = Source(
        id=7, name="docs", directory="/corpora/docs"
    )
    mocks["configs"].get_config.return_value = RagConfigRecord(
        id=5, config_fingerprint="a" * 64, config_json=config.model_dump_json()
    )
    mocks["configs"].parse_config.return_value = config
    mocks["sources"].get_index_status.return_value = make_index_status(
        config_fingerprint="a" * 64,
        available=False,
        reason=reason,
    )

    with pytest.raises(IndexUnavailableError, match=f"reason: {reason.value}"):
        service.answer("Question", 3, 5)

    mocks["experiments"].start_run.assert_not_called()
    mock_create_store.assert_not_called()


@patch("genai_template.services.rag_service.create_llm")
@patch("genai_template.services.rag_service.create_vector_store")
@patch("genai_template.services.rag_service.create_embedder")
@patch("genai_template.services.rag_service.create_retrieval_pipeline")
def test_execution_failure_leaves_started_run_unfinished(
    mock_create_retrieval: MagicMock,
    _mock_create_embedder: MagicMock,
    mock_create_store: MagicMock,
    mock_create_llm: MagicMock,
) -> None:
    """Failures after run creation should not write completion metrics."""

    service, mocks = create_service()
    config = load_rag_config()
    mocks["experiments"].get_experiment.return_value = Experiment(
        id=3, source_id=7, name="Trial"
    )
    mocks["experiments"].start_run.return_value = MagicMock(id=11)
    mocks["sources"].get_source.return_value = Source(
        id=7, name="docs", directory="/corpora/docs"
    )
    mocks["configs"].get_config.return_value = RagConfigRecord(
        id=5, config_fingerprint="a" * 64, config_json=config.model_dump_json()
    )
    mocks["configs"].parse_config.return_value = config
    mocks["sources"].get_index_status.return_value = make_index_status(
        config_fingerprint="a" * 64
    )
    mock_create_retrieval.return_value.retrieve.return_value = []
    mocks["context"].build.return_value = CitationContext(text="context", sources=[])
    mocks["prompt"].build.return_value = "prompt"
    mock_create_llm.return_value.generate.side_effect = RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        service.answer("Question", 3, 5)

    mocks["experiments"].start_run.assert_called_once()
    mocks["experiments"].complete_run.assert_not_called()
