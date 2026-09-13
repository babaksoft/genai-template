"""Integration test for the complete RAG workflow."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from genai_template.components.context import ContextBuilder
from genai_template.components.embeddings import FastEmbedEmbeddingModel
from genai_template.components.prompt import PromptBuilder
from genai_template.components.readers import TextReader
from genai_template.components.splitters import DocumentSplitter
from genai_template.config import load_rag_config
from genai_template.db.models import Experiment
from genai_template.db.models import RagConfig as RagConfigRecord
from genai_template.db.models import Source
from genai_template.pipelines import IndexingPipeline
from genai_template.services import RagService, SourceService
from genai_template.stores.vector import ChromaStore

LLM_MODEL = "llama3"


@pytest.mark.integration
def test_rag_service_answers_question(tmp_path: Path) -> None:
    """The complete RAG workflow should produce a non-empty answer."""

    documents_dir = Path(__file__).parent / "resources"
    default_config = load_rag_config()
    config = default_config.model_copy(
        update={
            "llm": default_config.llm.model_copy(update={"model_name": LLM_MODEL}),
            "vector_store": default_config.vector_store.model_copy(
                update={"persist_directory": tmp_path}
            ),
        }
    )
    collection_name = SourceService.index_collection_name(1, config)
    vector_store = ChromaStore(
        persist_directory=tmp_path,
        collection_name=collection_name,
    )
    IndexingPipeline(
        reader=TextReader(),
        splitter=DocumentSplitter(),
        embedder=FastEmbedEmbeddingModel(),
        store=vector_store,
    ).run(documents_dir)

    experiment_service = MagicMock()
    experiment_service.get_experiment.return_value = Experiment(
        id=2, source_id=1, name="Integration"
    )
    experiment_service.start_run.return_value = MagicMock(id=3)
    config_service = MagicMock()
    config_service.get_config.return_value = RagConfigRecord(
        id=4,
        config_fingerprint="a" * 64,
        config_json=config.model_dump_json(),
    )
    config_service.parse_config.return_value = config
    source_service = MagicMock()
    source_service.get_source.return_value = Source(
        id=1,
        name="resources",
        directory=str(documents_dir),
    )
    source_service.index_collection_name.return_value = collection_name

    rag_service = RagService(
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        experiment_service=experiment_service,
        rag_config_service=config_service,
        source_service=source_service,
    )

    answer = rag_service.answer(
        "What is the capital of France?", experiment_id=2, rag_config_id=4
    )

    assert answer
    assert "Paris" in answer.answer


if __name__ == "__main__":
    from genai_template.config.logging import configure_logging
    from genai_template.config.settings import LOG_DIR

    configure_logging()
    test_rag_service_answers_question(LOG_DIR)
