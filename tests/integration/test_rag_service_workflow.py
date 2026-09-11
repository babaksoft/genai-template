"""Integration test for the complete RAG workflow."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from genai_template.components.context import ContextBuilder
from genai_template.components.embeddings import FastEmbedEmbeddingModel
from genai_template.components.language_models import (
    OllamaLanguageModel,
)
from genai_template.components.prompt import PromptBuilder
from genai_template.components.readers import TextReader
from genai_template.components.splitters import DocumentSplitter
from genai_template.config import load_rag_config
from genai_template.db.models import Source
from genai_template.pipelines import IndexingPipeline
from genai_template.services import RagService
from genai_template.stores.vector import ChromaStore

LLM_MODEL = "llama3"


@pytest.mark.integration
def test_rag_service_answers_question(tmp_path: Path) -> None:
    """The complete RAG workflow should produce a non-empty answer."""

    documents_dir = Path(__file__).parent / "resources"
    vector_store = ChromaStore(
        persist_directory=tmp_path,
    )

    indexing_pipeline = IndexingPipeline(
        reader=TextReader(),
        splitter=DocumentSplitter(),
        embedder=FastEmbedEmbeddingModel(),
        store=vector_store,
    )

    indexing_pipeline.run(documents_dir)

    default_config = load_rag_config()
    config = default_config.model_copy(
        update={
            "llm": default_config.llm.model_copy(update={"model_name": LLM_MODEL}),
            "vector_store": default_config.vector_store.model_copy(
                update={"persist_directory": tmp_path}
            ),
        }
    )

    rag_service = RagService(
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        language_model=OllamaLanguageModel(
            model_name=LLM_MODEL,
        ),
        experiment_service=MagicMock(),
        source_service=MagicMock(
            get_source=MagicMock(
                return_value=Source(
                    id=1,
                    name="resources",
                    directory=str(documents_dir),
                    collection_name="documents",
                    documents_indexed=1,
                    chunks_indexed=1,
                    indexing_time=0.1,
                )
            )
        ),
        config=config,
    )

    answer = rag_service.answer("What is the capital of France?", source_id=1)

    assert answer
    assert "Paris" in answer.answer


if __name__ == "__main__":
    from genai_template.config.logging import configure_logging
    from genai_template.config.settings import LOG_DIR

    configure_logging()
    test_rag_service_answers_question(LOG_DIR)
