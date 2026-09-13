"""Tests for persisted-config RAG execution."""

from unittest.mock import MagicMock, patch

import pytest

from genai_template.config import load_rag_config
from genai_template.db.models import Experiment
from genai_template.db.models import RagConfig as RagConfigRecord
from genai_template.db.models import Source
from genai_template.schemas import RetrievedChunk
from genai_template.services import IndexNotBuiltError, RagService, SourceService


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
    retrieved_chunks: list[RetrievedChunk] = []
    retrieval.retrieve.return_value = retrieved_chunks
    mocks["experiments"].get_experiment.return_value = experiment
    mocks["experiments"].start_run.return_value = run
    mocks["sources"].get_source.return_value = source
    mocks["sources"].index_collection_name.return_value = "idx-selected"
    mocks["configs"].get_config.return_value = record
    mocks["configs"].parse_config.return_value = config
    mocks["context"].build.return_value = "context"
    mocks["prompt"].build.return_value = "prompt"
    mock_create_llm.return_value.generate.return_value = "final answer"

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
    mocks["experiments"].complete_run.assert_called_once_with(
        run=run, metrics=result.metrics
    )
    assert result.answer == "final answer"
    assert result.metrics.embedding_model == "configured-embedder"
    assert result.metrics.llm_model == "configured-llm"


@patch("genai_template.services.rag_service.create_vector_store")
def test_answer_rejects_missing_index_before_creating_run(
    mock_create_store: MagicMock,
) -> None:
    """A missing deterministic index must not leave an unfinished run."""

    service, mocks = create_service()
    config = load_rag_config()
    mocks["experiments"].get_experiment.return_value = Experiment(
        id=3, source_id=7, name="Trial"
    )
    mocks["sources"].get_source.return_value = Source(
        id=7, name="docs", directory="/corpora/docs"
    )
    mocks["sources"].index_collection_name.side_effect = (
        SourceService.index_collection_name
    )
    mocks["configs"].get_config.return_value = RagConfigRecord(
        id=5, config_fingerprint="a" * 64, config_json=config.model_dump_json()
    )
    mocks["configs"].parse_config.return_value = config
    mock_create_store.return_value.exists.return_value = False

    with pytest.raises(IndexNotBuiltError, match="has not been built"):
        service.answer("Question", 3, 5)

    mocks["experiments"].start_run.assert_not_called()


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
    mocks["sources"].index_collection_name.return_value = "idx-selected"
    mocks["configs"].get_config.return_value = RagConfigRecord(
        id=5, config_fingerprint="a" * 64, config_json=config.model_dump_json()
    )
    mocks["configs"].parse_config.return_value = config
    mock_create_store.return_value.exists.return_value = True
    mock_create_retrieval.return_value.retrieve.return_value = []
    mocks["context"].build.return_value = "context"
    mocks["prompt"].build.return_value = "prompt"
    mock_create_llm.return_value.generate.side_effect = RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        service.answer("Question", 3, 5)

    mocks["experiments"].start_run.assert_called_once()
    mocks["experiments"].complete_run.assert_not_called()
