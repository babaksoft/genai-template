"""End-to-end API test for the manifest index freshness lifecycle."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_index.core import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from genai_template.api.dependencies import (
    get_experiment_service,
    get_rag_config_service,
    get_rag_service,
    get_source_service,
)
from genai_template.components.context import ContextBuilder
from genai_template.components.prompt import PromptBuilder
from genai_template.config import load_rag_config
from genai_template.db.base import Base
from genai_template.schemas import DocumentChunk, RetrievedChunk
from genai_template.services import (
    ExperimentService,
    RagConfigService,
    RagService,
    SourceService,
)
from genai_template.workflow.portfolio.config import load_portfolio_config
from genai_template.workflow.portfolio.domain import (
    ArtifactKind,
    CorpusManifestV2,
    RenderedDocument,
    RepositorySnapshot,
    SnapshotFile,
)
from genai_template.workflow.portfolio.infrastructure.corpus import (
    build_corpus_manifest,
    publish_corpus,
)


class FakeEmbedder:
    """Deterministic embedder with injectable build failure."""

    def __init__(self) -> None:
        """Initialize an available deterministic embedder."""

        self.fail = False

    def embed(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Attach one deterministic vector to every chunk.

        Args:
            chunks:
                Chunks to embed.

        Returns:
            Chunks containing deterministic embeddings.

        Raises:
            RuntimeError:
                If failure injection is enabled.
        """

        if self.fail:
            raise RuntimeError("injected provider failure with secret source text")
        return [chunk.model_copy(update={"embedding": [1.0]}) for chunk in chunks]

    def embed_query(self, query: str) -> list[float]:
        """Return a deterministic query vector.

        Args:
            query:
                Query text, intentionally ignored.

        Returns:
            One-dimensional query embedding.
        """

        return [1.0]


class FakeSplitter:
    """Produce one provenance-preserving chunk per Portfolio document."""

    def split(self, documents: list[Document]) -> list[DocumentChunk]:
        """Convert documents to deterministic chunks.

        Args:
            documents:
                Ordered Portfolio documents.

        Returns:
            One canonical chunk per document.
        """

        return [
            DocumentChunk(
                id=f"{document.id_}-000",
                document_id=document.id_,
                text=document.text,
                metadata=dict(document.metadata),
            )
            for document in documents
        ]


class FakeStore:
    """In-memory vector collection shared by indexing and retrieval."""

    def __init__(self) -> None:
        """Initialize an absent empty collection."""

        self.chunks: list[DocumentChunk] | None = None

    def create(self, vector_size: int) -> None:
        """Create an empty collection.

        Args:
            vector_size:
                Vector size, intentionally ignored by the fake.
        """

        self.chunks = []

    def exists(self) -> bool:
        """Return whether the fake collection exists.

        Returns:
            Collection existence.
        """

        return self.chunks is not None

    def count(self) -> int:
        """Return the number of stored chunks.

        Returns:
            Stored chunk count.
        """

        return len(self.chunks or [])

    def delete(self) -> None:
        """Delete the fake collection."""

        self.chunks = None

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Replace the stored chunks.

        Args:
            chunks:
                Embedded chunks to retain.
        """

        self.chunks = list(chunks)

    def search(
        self,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return stored chunks in deterministic order.

        Args:
            embedding:
                Query vector, intentionally ignored.
            top_k:
                Maximum number of results.
            query:
                Optional query, intentionally ignored.

        Returns:
            Stored chunks with stable distances.
        """

        return [
            RetrievedChunk(chunk=chunk, distance=index / 100)
            for index, chunk in enumerate((self.chunks or [])[:top_k], start=1)
        ]


class FakeLanguageModel:
    """Deterministic citation-producing language model."""

    def generate(self, prompt: str) -> str:
        """Return a stable cited answer.

        Args:
            prompt:
                Prompt text, intentionally ignored.

        Returns:
            Cited deterministic answer.
        """

        return "Deterministic answer [S1]."


def test_portfolio_index_lifecycle_through_api(
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API should expose and enforce the complete Portfolio lifecycle."""

    corpora_dir = tmp_path / "corpora"
    publication = corpora_dir / "portfolio"
    snapshot, documents, manifest = _corpus("Original content.")
    publish_corpus(
        publication_path=publication,
        manifest=manifest,
        documents=documents,
        snapshot=snapshot,
    )

    engine = create_engine(
        f"sqlite:///{tmp_path / 'lifecycle.sqlite3'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    config_service = RagConfigService(session_factory)
    source_service = SourceService(session_factory, corpora_dir, config_service)
    experiment_service = ExperimentService(session_factory)
    rag_service = RagService(
        ContextBuilder(),
        PromptBuilder(),
        experiment_service,
        config_service,
        source_service,
    )
    embedder = FakeEmbedder()
    stores: dict[str, FakeStore] = {}

    def create_store(config: object, collection_name: str) -> FakeStore:
        """Return the shared fake collection for one deterministic name.

        Args:
            config:
                Vector-store configuration, intentionally ignored.
            collection_name:
                Deterministic collection name.

        Returns:
            Shared fake vector store.
        """

        return stores.setdefault(collection_name, FakeStore())

    monkeypatch.setattr(
        "genai_template.services.source_service.create_vector_store", create_store
    )
    monkeypatch.setattr(
        "genai_template.services.source_service.create_embedder",
        lambda config: embedder,
    )
    monkeypatch.setattr(
        "genai_template.services.source_service.create_splitter",
        lambda config: FakeSplitter(),
    )
    monkeypatch.setattr(
        "genai_template.services.rag_service.create_vector_store", create_store
    )
    monkeypatch.setattr(
        "genai_template.services.rag_service.create_embedder", lambda config: embedder
    )
    monkeypatch.setattr(
        "genai_template.services.rag_service.create_llm",
        lambda config: FakeLanguageModel(),
    )
    app.dependency_overrides[get_source_service] = lambda: source_service
    app.dependency_overrides[get_rag_config_service] = lambda: config_service
    app.dependency_overrides[get_experiment_service] = lambda: experiment_service
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    client = TestClient(app, raise_server_exceptions=False)

    source = client.post("/api/v1/sources", json={"directory": "portfolio"}).json()
    config = load_rag_config()
    config_record = client.post(
        "/api/v1/rag-configs", json=config.model_dump(mode="json")
    ).json()
    experiment = client.post(
        "/api/v1/experiments",
        json={"source_id": source["id"], "name": "Lifecycle"},
    ).json()
    index_url = f"/api/v1/sources/{source['id']}/indexes/{config_record['id']}"
    answer_payload = {
        "query": "What changed?",
        "experiment_id": experiment["id"],
        "rag_config_id": config_record["id"],
    }

    assert client.get(index_url).json()["reason"] == "unbuilt"
    rebuilt = client.put(index_url)
    assert rebuilt.status_code == 200, rebuilt.text
    assert rebuilt.json()["status"]["reason"] == "current"
    assert rebuilt.json()["build"]["id"] >= 1
    answer = client.post("/api/v1/answer", json=answer_payload)
    assert answer.status_code == 200
    assert answer.json()["sources"][0]["project_slug"] == "sample"
    assert answer.json()["sources"][0]["resolved_commit_sha"] == "a" * 40

    changed_snapshot, changed_documents, changed_manifest = _corpus("Changed content.")
    publish_corpus(
        publication_path=publication,
        manifest=changed_manifest,
        documents=changed_documents,
        snapshot=changed_snapshot,
    )
    assert client.get(index_url).json()["reason"] == "stale"
    rejected = client.post("/api/v1/answer", json=answer_payload)
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["reason"] == "stale"

    assert client.put(index_url).json()["status"]["reason"] == "current"
    assert client.post("/api/v1/answer", json=answer_payload).status_code == 200
    embedder.fail = True
    assert client.put(index_url).status_code == 500
    failed = client.get(index_url).json()
    assert failed["reason"] == "failed"
    assert "secret source text" not in failed["latest_failure_detail"]
    final_answer = client.post("/api/v1/answer", json=answer_payload)
    assert final_answer.status_code == 409
    assert final_answer.json()["detail"]["reason"] == "failed"
    app.dependency_overrides.clear()


def _corpus(
    body: str,
) -> tuple[
    RepositorySnapshot,
    tuple[RenderedDocument, ...],
    CorpusManifestV2,
]:
    """Build one valid single-document Portfolio corpus.

    Args:
        body:
            Distinguishing generated document content.

    Returns:
        Snapshot, rendered documents, and derived manifest.
    """

    source = b"source\n"
    snapshot = RepositorySnapshot(
        project_slug="sample",
        repository_url="https://example.test/sample.git",
        requested_ref="main",
        resolved_commit_sha="a" * 40,
        source_fingerprint=hashlib.sha256(source + body.encode()).hexdigest(),
        files=(
            SnapshotFile(
                path="README.md",
                text=source.decode(),
                content_hash=hashlib.sha256(source).hexdigest(),
                byte_size=len(source),
            ),
        ),
    )
    specifications: tuple[tuple[str, ArtifactKind, str | None], ...] = (
        ("sample--architecture.md", "architecture", None),
        ("sample--component--api.md", "component", "api"),
        ("sample--overview.md", "overview", None),
        ("sample--testing-operations.md", "testing_operations", None),
    )
    documents = tuple(
        _rendered_document(filename, document_type, component_id, body)
        for filename, document_type, component_id in specifications
    )
    generation = load_portfolio_config(
        Path("src/genai_template/workflow/portfolio/config/profiles/local.yml")
    ).require_generation()
    manifest = build_corpus_manifest(
        project_display_name="Sample",
        snapshot=snapshot,
        generation=generation,
        documents=documents,
    )
    return snapshot, documents, manifest


def _rendered_document(
    filename: str,
    document_type: ArtifactKind,
    component_id: str | None,
    body: str,
) -> RenderedDocument:
    """Build one deterministic generated document.

    Args:
        filename:
            Canonical generated filename.
        document_type:
            Balanced project document type.
        component_id:
            Component identifier for a component document.
        body:
            Distinguishing generated content.

    Returns:
        Complete rendered document.
    """

    content = f"# {document_type}\n\n{body}\n\nEvidence: `README.md`\n"
    identity = f"{document_type}:{component_id}:{body}"
    return RenderedDocument(
        filename=filename,
        document_type=document_type,
        component_id=component_id,
        generation_fingerprint=hashlib.sha256(identity.encode()).hexdigest(),
        artifact_hash=hashlib.sha256((identity + "artifact").encode()).hexdigest(),
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        evidence_paths=("README.md",),
    )
