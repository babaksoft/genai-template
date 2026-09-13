"""Tests for OpenInference application spans in the RAG answer path."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from genai_template.components.context import ContextBuilder
from genai_template.components.prompt import PromptBuilder
from genai_template.config import config_fingerprint, load_rag_config
from genai_template.db.models import Experiment
from genai_template.db.models import RagConfig as RagConfigRecord
from genai_template.db.models import Source
from genai_template.observability import trace as observability_trace
from genai_template.services import RagService


@patch("genai_template.stores.vector.chroma_store.chromadb.PersistentClient")
def test_answer_emits_nested_rag_spans(mock_client_class: MagicMock) -> None:
    """The answer path records safe, useful stages in one trace."""

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["chunk-001"]],
        "documents": [["FastAPI is a Python web framework."]],
        "metadatas": [[{"document_id": "fastapi.md", "metadata": "{}"}]],
        "distances": [[0.12]],
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    client.get_collection.return_value = collection
    mock_client_class.return_value = client

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]
    language_model = MagicMock()
    language_model.generate.return_value = "FastAPI is a web framework."
    default_config = load_rag_config()
    config = default_config.model_copy(
        update={
            "embedder": default_config.embedder.model_copy(
                update={"model_name": "trace-embedder"}
            ),
            "retrieval": default_config.retrieval.model_copy(update={"top_k": 3}),
            "llm": default_config.llm.model_copy(update={"model_name": "trace-llm"}),
        }
    )

    experiment_service = MagicMock()
    experiment_service.get_experiment.return_value = Experiment(
        id=4, source_id=1, name="Trace experiment"
    )
    experiment_service.start_run.return_value = MagicMock(id=8)
    config_service = MagicMock()
    config_service.get_config.return_value = RagConfigRecord(
        id=5,
        config_fingerprint=config_fingerprint(config),
        config_json=config.model_dump_json(),
    )
    config_service.parse_config.return_value = config
    source_service = MagicMock()
    source_service.get_source.return_value = Source(
        id=1,
        name="fastapi",
        directory="/corpora/fastapi",
    )
    source_service.index_collection_name.return_value = "source-fastapi"
    service = RagService(
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        experiment_service=experiment_service,
        rag_config_service=config_service,
        source_service=source_service,
    )

    with (
        patch(
            "genai_template.services.rag_service.create_embedder", return_value=embedder
        ),
        patch(
            "genai_template.services.rag_service.create_llm",
            return_value=language_model,
        ),
        patch.object(
            observability_trace,
            "get_tracer",
            side_effect=lambda name: provider.get_tracer(name),
        ),
    ):
        service.answer("What is FastAPI?", experiment_id=4, rag_config_id=5)

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {
        "rag.answer",
        "rag.retrieval",
        "rag.chroma.search",
        "rag.context.build",
        "rag.prompt.build",
    }

    answer_span = spans["rag.answer"]
    retrieval_parent = spans["rag.retrieval"].parent
    chroma_parent = spans["rag.chroma.search"].parent
    context_parent = spans["rag.context.build"].parent
    prompt_parent = spans["rag.prompt.build"].parent
    assert retrieval_parent is not None
    assert chroma_parent is not None
    assert context_parent is not None
    assert prompt_parent is not None
    assert retrieval_parent.span_id == answer_span.context.span_id
    assert chroma_parent.span_id == spans["rag.retrieval"].context.span_id
    assert context_parent.span_id == answer_span.context.span_id
    assert prompt_parent.span_id == answer_span.context.span_id

    retrieval_attributes = spans["rag.retrieval"].attributes
    chroma_attributes = spans["rag.chroma.search"].attributes
    context_attributes = spans["rag.context.build"].attributes
    prompt_attributes = spans["rag.prompt.build"].attributes
    assert retrieval_attributes is not None
    assert chroma_attributes is not None
    assert context_attributes is not None
    assert prompt_attributes is not None
    retrieval_documents = retrieval_attributes["retrieval.documents"]
    context_output = context_attributes["output.value"]
    prompt_output = prompt_attributes["output.value"]
    assert isinstance(retrieval_documents, str)
    assert isinstance(context_output, str)
    assert isinstance(prompt_output, str)
    documents = json.loads(retrieval_documents)
    assert retrieval_attributes["input.value"] == "What is FastAPI?"
    assert retrieval_attributes["rag.top_k"] == 3
    assert documents == [
        {
            "id": "chunk-001",
            "document_id": "fastapi.md",
            "content": "FastAPI is a Python web framework.",
            "distance": 0.12,
        }
    ]
    assert chroma_attributes["rag.result_count"] == 1
    assert context_output.startswith("Chunk 1")
    assert prompt_output
    answer_attributes = answer_span.attributes
    assert answer_attributes is not None
    assert answer_attributes["rag.top_k"] == 3
    assert answer_attributes["rag.experiment.id"] == 4
    assert answer_attributes["rag.config.id"] == 5
    assert answer_attributes["rag.config.fingerprint"] == config_fingerprint(config)
    assert answer_attributes["rag.embedding.model"] == "trace-embedder"
    assert answer_attributes["rag.vector_store.type"] == "chroma"
    assert answer_attributes["rag.llm.model"] == "trace-llm"
