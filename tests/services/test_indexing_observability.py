"""Tests for OpenInference application spans in the indexing path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from genai_template.config import (
    config_fingerprint,
    index_config_fingerprint,
    load_rag_config,
)
from genai_template.db.models import RagConfig as RagConfigRecord
from genai_template.db.models import Source
from genai_template.observability import trace as observability_trace
from genai_template.pipelines import IndexingPipeline
from genai_template.schemas import DocumentChunk
from genai_template.services import SourceService


def _trace_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Create a tracer provider that exports completed spans in memory.

    Returns:
        Configured tracer provider and its in-memory exporter.
    """

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_rebuild_emits_nested_indexing_spans(tmp_path: Path) -> None:
    """A rebuild should record safe lifecycle and pipeline stage attributes."""

    directory = tmp_path / "docs"
    directory.mkdir()
    document_text = "# Instrumentation\n\nVisible only to the indexing components."
    (directory / "document.md").write_text(document_text, encoding="utf-8")

    config = load_rag_config()
    config_record = RagConfigRecord(
        id=5,
        config_fingerprint=config_fingerprint(config),
        config_json=config.model_dump_json(),
    )
    config_service = MagicMock()
    config_service.get_config.return_value = config_record
    config_service.parse_config.return_value = config
    service = SourceService(MagicMock(), tmp_path, config_service)
    source = Source(id=1, name="docs", directory=str(directory.resolve()))

    chunk = DocumentChunk(
        id="document.md-000",
        document_id="document.md",
        text=document_text,
        metadata={},
        embedding=[0.1, 0.2, 0.3],
    )
    splitter = MagicMock()
    splitter.split.return_value = [chunk]
    embedder = MagicMock()
    embedder.embed.return_value = [chunk]
    store = MagicMock()
    provider, exporter = _trace_exporter()

    with (
        patch.object(service, "get_source", return_value=source),
        patch(
            "genai_template.services.source_service.create_vector_store",
            return_value=store,
        ),
        patch(
            "genai_template.services.source_service.create_splitter",
            return_value=splitter,
        ),
        patch(
            "genai_template.services.source_service.create_embedder",
            return_value=embedder,
        ),
        patch.object(
            observability_trace,
            "get_tracer",
            side_effect=lambda name: provider.get_tracer(name),
        ),
    ):
        result = service.rebuild_index(source.id, config_record.id)

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {
        "rag.index.rebuild",
        "rag.index.delete",
        "rag.index.build",
        "rag.index.load",
        "rag.index.split",
        "rag.index.embed",
        "rag.index.write",
    }

    rebuild_span = spans["rag.index.rebuild"]
    rebuild_id = rebuild_span.context.span_id
    build_span = spans["rag.index.build"]
    build_id = build_span.context.span_id
    delete_parent = spans["rag.index.delete"].parent
    build_parent = build_span.parent
    assert delete_parent is not None
    assert build_parent is not None
    assert delete_parent.span_id == rebuild_id
    assert build_parent.span_id == rebuild_id
    for stage_name in (
        "rag.index.load",
        "rag.index.split",
        "rag.index.embed",
        "rag.index.write",
    ):
        stage_parent = spans[stage_name].parent
        assert stage_parent is not None
        assert stage_parent.span_id == build_id

    collection_name = SourceService.index_collection_name(source.id, config)
    rebuild_attributes = rebuild_span.attributes
    assert rebuild_attributes is not None
    assert rebuild_attributes["rag.source.id"] == source.id
    assert rebuild_attributes["rag.config.id"] == config_record.id
    assert (
        rebuild_attributes["rag.config.fingerprint"] == config_record.config_fingerprint
    )
    assert rebuild_attributes["rag.index.fingerprint"] == index_config_fingerprint(
        config
    )
    assert rebuild_attributes["rag.index.collection"] == collection_name
    assert rebuild_attributes["rag.splitter.type"] == config.splitter.type
    assert rebuild_attributes["rag.embedding.provider"] == config.embedder.type
    assert rebuild_attributes["rag.embedding.model"] == config.embedder.model_name
    assert rebuild_attributes["rag.vector_store.type"] == config.vector_store.type
    assert (
        rebuild_attributes["rag.vector_store.distance"]
        == config.vector_store.distance.value
    )
    assert rebuild_attributes["rag.document.count"] == 1
    assert rebuild_attributes["rag.chunk.count"] == 1

    load_attributes = spans["rag.index.load"].attributes
    split_attributes = spans["rag.index.split"].attributes
    embed_attributes = spans["rag.index.embed"].attributes
    write_attributes = spans["rag.index.write"].attributes
    assert load_attributes is not None
    assert split_attributes is not None
    assert embed_attributes is not None
    assert write_attributes is not None
    assert load_attributes["rag.document.count"] == 1
    assert split_attributes["rag.chunk.count"] == 1
    assert embed_attributes["rag.embedding.count"] == 1
    assert embed_attributes["rag.embedding.dimension"] == 3
    assert write_attributes["rag.index.write.operation"] == "upsert"
    assert write_attributes["rag.index.write.count"] == 1

    custom_attributes = str(
        {
            key: value
            for span in spans.values()
            for key, value in (span.attributes or {}).items()
        }
    )
    assert str(directory) not in custom_attributes
    assert document_text not in custom_attributes
    assert result.documents_indexed == 1
    assert result.chunks_indexed == 1
    store.delete.assert_called_once_with()
    store.upsert.assert_called_once_with([chunk])


def test_empty_index_records_collection_creation() -> None:
    """An empty build should identify its collection-creation write branch."""

    reader = MagicMock()
    reader.load.return_value = []
    splitter = MagicMock()
    splitter.split.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = []
    embedder.embed_query.return_value = [0.0, 0.0, 0.0]
    store = MagicMock()
    pipeline = IndexingPipeline(
        reader=reader,
        splitter=splitter,
        embedder=embedder,
        store=store,
    )
    provider, exporter = _trace_exporter()

    with patch.object(
        observability_trace,
        "get_tracer",
        side_effect=lambda name: provider.get_tracer(name),
    ):
        pipeline.run(Path("unused"))

    spans = {span.name: span for span in exporter.get_finished_spans()}
    write_attributes = spans["rag.index.write"].attributes
    build_attributes = spans["rag.index.build"].attributes
    assert write_attributes is not None
    assert build_attributes is not None
    assert write_attributes["rag.index.write.operation"] == "create"
    assert write_attributes["rag.index.write.count"] == 0
    assert write_attributes["rag.embedding.dimension"] == 3
    assert build_attributes["rag.document.count"] == 0
    assert build_attributes["rag.chunk.count"] == 0
    store.upsert.assert_not_called()
    store.create.assert_called_once_with(3)


def test_failed_embedding_marks_stage_and_build_as_errors() -> None:
    """An indexing failure should be traced and preserve its original exception."""

    document = MagicMock()
    chunk = DocumentChunk(
        id="chunk-001",
        document_id="document.md",
        text="Failure fixture.",
        metadata={},
    )
    reader = MagicMock()
    reader.load.return_value = [document]
    splitter = MagicMock()
    splitter.split.return_value = [chunk]
    embedder = MagicMock()
    embedder.embed.side_effect = RuntimeError("embedding unavailable")
    pipeline = IndexingPipeline(
        reader=reader,
        splitter=splitter,
        embedder=embedder,
        store=MagicMock(),
    )
    provider, exporter = _trace_exporter()

    with (
        patch.object(
            observability_trace,
            "get_tracer",
            side_effect=lambda name: provider.get_tracer(name),
        ),
        pytest.raises(RuntimeError, match="embedding unavailable"),
    ):
        pipeline.run(Path("unused"))

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {
        "rag.index.build",
        "rag.index.load",
        "rag.index.split",
        "rag.index.embed",
    }
    assert spans["rag.index.embed"].status.status_code is StatusCode.ERROR
    assert spans["rag.index.build"].status.status_code is StatusCode.ERROR
