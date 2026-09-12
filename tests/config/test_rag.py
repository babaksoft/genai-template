"""Tests for validated RAG configuration loading."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from genai_template.common.types import VectorDistance
from genai_template.config import (
    ChromaVectorStoreConfig,
    FastEmbedEmbedderConfig,
    MarkdownSplitterConfig,
    OllamaLLMConfig,
    OpenAIEmbedderConfig,
    OpenAILLMConfig,
    QdrantVectorStoreConfig,
    SentenceSplitterConfig,
    canonical_config_json,
    config_fingerprint,
    index_config_fingerprint,
    load_rag_config,
    settings,
)


def test_defaults_match_application_settings() -> None:
    """Loading without YAML reproduces the current RAG settings."""

    config = load_rag_config()

    assert isinstance(config.splitter, SentenceSplitterConfig)
    assert isinstance(config.embedder, FastEmbedEmbedderConfig)
    assert isinstance(config.vector_store, ChromaVectorStoreConfig)
    assert isinstance(config.llm, OllamaLLMConfig)
    assert config.splitter.type == "sentence"
    assert config.splitter.chunk_size == settings.CHUNK_SIZE
    assert config.splitter.chunk_overlap == settings.CHUNK_OVERLAP
    assert config.embedder.type == "fastembed"
    assert config.embedder.model_name == settings.EMBEDDING_MODEL
    assert config.vector_store.type == settings.VECTOR_STORE.lower()
    assert config.vector_store.persist_directory == settings.CHROMA_PERSIST_DIR
    assert config.vector_store.distance == settings.CHROMA_DISTANCE
    assert config.retrieval.top_k == settings.TOP_K
    assert config.llm.type == "ollama"
    assert config.llm.model_name == settings.LLM_MODEL
    assert config.llm.base_url == settings.OLLAMA_BASE_URL
    assert config.llm.request_timeout == settings.REQUEST_TIMEOUT


def test_partial_overrides_are_recursively_merged(tmp_path: Path) -> None:
    """A partial nested section preserves all unspecified defaults."""

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "splitter:\n  chunk_size: 256\n",
        encoding="utf-8",
    )

    config = load_rag_config(config_path)

    assert isinstance(config.splitter, SentenceSplitterConfig)
    assert config.splitter.chunk_size == 256
    assert config.splitter.chunk_overlap == settings.CHUNK_OVERLAP
    assert config.embedder.model_name == settings.EMBEDDING_MODEL


def test_provider_sections_are_replaced_when_type_changes(tmp_path: Path) -> None:
    """Changing a provider should not retain fields from the default provider."""

    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        "splitter:\n"
        "  type: markdown\n"
        "embedder:\n"
        "  type: openai\n"
        "  model_name: text-embedding-3-small\n"
        "vector_store:\n"
        "  type: qdrant\n"
        "  distance: cosine\n"
        "  location: local\n"
        "  path: storage/qdrant\n"
        "llm:\n"
        "  type: openai\n"
        "  model_name: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    config = load_rag_config(config_path)

    assert isinstance(config.splitter, MarkdownSplitterConfig)
    assert config.splitter.header_path_separator == "/"
    assert isinstance(config.embedder, OpenAIEmbedderConfig)
    assert config.embedder.dimensions is None
    assert config.embedder.request_timeout == settings.REQUEST_TIMEOUT
    assert isinstance(config.vector_store, QdrantVectorStoreConfig)
    assert config.vector_store.path == settings.REPO_ROOT / "storage" / "qdrant"
    assert isinstance(config.llm, OpenAILLMConfig)
    assert config.llm.request_timeout == settings.REQUEST_TIMEOUT


def test_qdrant_server_configuration_does_not_require_a_path(
    tmp_path: Path,
) -> None:
    """Server Qdrant should require a URL and omit local storage settings."""

    config_path = tmp_path / "qdrant-server.yaml"
    config_path.write_text(
        "vector_store:\n"
        "  type: qdrant\n"
        "  distance: cosine\n"
        "  location: server\n"
        "  url: https://qdrant.example.com\n",
        encoding="utf-8",
    )

    config = load_rag_config(config_path)

    assert isinstance(config.vector_store, QdrantVectorStoreConfig)
    assert config.vector_store.url == "https://qdrant.example.com"
    assert config.vector_store.path is None


@pytest.mark.parametrize(
    "yaml_text",
    [
        "unknown: true\n",
        "experiment:\n  name: legacy\n",
        "vector_store:\n  collection_name: legacy\n",
        "splitter:\n  type: tokens\n",
        "splitter:\n  type: markdown\n  chunk_size: 256\n",
        "embedder:\n  type: openai\n",
        "embedder:\n  type: openai\n  model_name: embed\n  dimensions: 0\n",
        "vector_store:\n  type: pinecone\n",
        ("vector_store:\n  type: qdrant\n" "  distance: cosine\n  location: local\n"),
        ("vector_store:\n  type: qdrant\n" "  distance: cosine\n  location: server\n"),
        (
            "vector_store:\n  type: qdrant\n"
            "  distance: cosine\n  location: server\n  path: storage/qdrant\n"
            "  url: https://qdrant.example.com\n"
        ),
        "llm:\n  type: openai\n",
        "retrieval:\n  top_k: 0\n",
        "splitter:\n  chunk_size: 20\n  chunk_overlap: 20\n",
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, yaml_text: str) -> None:
    """Unknown keys, providers, and invalid values fail validation."""

    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_rag_config(config_path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    """A missing YAML file raises a file-not-found error."""

    with pytest.raises(FileNotFoundError):
        load_rag_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("yaml_text", ["splitter: [", "- one\n- two\n"])
def test_malformed_or_non_mapping_yaml_is_rejected(
    tmp_path: Path, yaml_text: str
) -> None:
    """Malformed YAML and non-mapping roots raise clear value errors."""

    config_path = tmp_path / "malformed.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_rag_config(config_path)


def test_relative_storage_path_is_resolved_from_repository_root(
    tmp_path: Path,
) -> None:
    """Relative persistence paths use the repository root, not YAML location."""

    config_path = tmp_path / "paths.yaml"
    config_path.write_text(
        "vector_store:\n  persist_directory: custom/index\n", encoding="utf-8"
    )

    config = load_rag_config(config_path)

    assert isinstance(config.vector_store, ChromaVectorStoreConfig)
    assert config.vector_store.persist_directory == (
        settings.REPO_ROOT / "custom" / "index"
    )


def test_relative_qdrant_path_is_resolved_from_repository_root(
    tmp_path: Path,
) -> None:
    """Relative local Qdrant paths should use the repository root."""

    config_path = tmp_path / "qdrant-path.yaml"
    config_path.write_text(
        "vector_store:\n"
        "  type: qdrant\n"
        "  distance: cosine\n"
        "  location: local\n"
        "  path: custom/qdrant\n",
        encoding="utf-8",
    )

    config = load_rag_config(config_path)

    assert isinstance(config.vector_store, QdrantVectorStoreConfig)
    assert config.vector_store.path == settings.REPO_ROOT / "custom" / "qdrant"


def test_configuration_models_are_immutable() -> None:
    """Resolved configuration cannot be mutated after validation."""

    config = load_rag_config()

    with pytest.raises(ValidationError):
        config.retrieval.top_k = 10


@pytest.mark.parametrize(
    ("config_type", "data"),
    [
        (
            SentenceSplitterConfig,
            {"type": "markdown", "chunk_size": 100, "chunk_overlap": 10},
        ),
        (MarkdownSplitterConfig, {"type": "sentence"}),
        (
            FastEmbedEmbedderConfig,
            {"type": "openai", "model_name": "embedder"},
        ),
        (
            OpenAIEmbedderConfig,
            {"type": "fastembed", "model_name": "embedder"},
        ),
        (
            ChromaVectorStoreConfig,
            {
                "type": "qdrant",
                "distance": "cosine",
                "persist_directory": "storage/chroma",
            },
        ),
        (
            QdrantVectorStoreConfig,
            {
                "type": "chroma",
                "distance": "cosine",
                "location": "local",
                "path": "storage/qdrant",
            },
        ),
        (
            OllamaLLMConfig,
            {
                "type": "openai",
                "model_name": "model",
                "base_url": "http://localhost:11434",
            },
        ),
        (OpenAILLMConfig, {"type": "ollama", "model_name": "model"}),
    ],
)
def test_concrete_config_type_cannot_be_overridden(
    config_type: type[BaseModel],
    data: dict[str, object],
) -> None:
    """Concrete provider configurations reject another provider's type."""

    with pytest.raises(ValidationError):
        config_type.model_validate(data)


def test_documented_baseline_matches_defaults() -> None:
    """The documented baseline file remains equivalent to application defaults."""

    baseline_path = settings.PKG_ROOT / "experiments" / "configs" / "baseline.yml"

    assert load_rag_config(baseline_path) == load_rag_config()
    assert load_rag_config().vector_store.distance is VectorDistance.COSINE


def test_documented_openai_qdrant_profile_is_valid() -> None:
    """The combined provider example should load as a resolved configuration."""

    profile_path = (
        settings.PKG_ROOT / "experiments" / "configs" / "openai-markdown-qdrant.yml"
    )

    config = load_rag_config(profile_path)

    assert isinstance(config.splitter, MarkdownSplitterConfig)
    assert isinstance(config.embedder, OpenAIEmbedderConfig)
    assert config.embedder.dimensions == 1536
    assert isinstance(config.vector_store, QdrantVectorStoreConfig)
    assert config.vector_store.location == "local"
    assert config.vector_store.path == settings.REPO_ROOT / "storage" / "qdrant"
    assert isinstance(config.llm, OpenAILLMConfig)


def test_canonical_serialization_and_fingerprint_are_stable() -> None:
    """Equivalent resolved configurations have identical canonical identities."""

    first = load_rag_config()
    second = load_rag_config()

    assert canonical_config_json(first) == canonical_config_json(second)
    assert config_fingerprint(first) == config_fingerprint(second)
    assert len(config_fingerprint(first)) == 64


def test_answer_only_changes_do_not_change_index_fingerprint() -> None:
    """Retrieval and generation settings should not select another index."""

    config = load_rag_config()
    changed = config.model_copy(
        update={
            "retrieval": config.retrieval.model_copy(update={"top_k": 12}),
            "llm": config.llm.model_copy(update={"model_name": "another-model"}),
        }
    )

    assert config_fingerprint(config) != config_fingerprint(changed)
    assert index_config_fingerprint(config) == index_config_fingerprint(changed)


def test_indexing_changes_change_index_fingerprint() -> None:
    """Chunking and embedding changes should select a different index."""

    config = load_rag_config()
    assert isinstance(config.splitter, SentenceSplitterConfig)
    changed_splitter = config.model_copy(
        update={
            "splitter": config.splitter.model_copy(
                update={"chunk_size": config.splitter.chunk_size + 1}
            )
        }
    )
    changed_embedder = config.model_copy(
        update={"embedder": config.embedder.model_copy(update={"model_name": "other"})}
    )
    changed_distance = config.model_copy(
        update={
            "vector_store": config.vector_store.model_copy(
                update={"distance": VectorDistance.L2}
            )
        }
    )

    assert index_config_fingerprint(config) != index_config_fingerprint(
        changed_splitter
    )
    assert index_config_fingerprint(config) != index_config_fingerprint(
        changed_embedder
    )
    assert index_config_fingerprint(config) != index_config_fingerprint(
        changed_distance
    )


def test_openai_request_timeout_does_not_change_index_fingerprint(
    tmp_path: Path,
) -> None:
    """Transport timeout should not identify embedding content."""

    config_path = tmp_path / "openai.yaml"
    config_path.write_text(
        "embedder:\n" "  type: openai\n" "  model_name: text-embedding-3-small\n",
        encoding="utf-8",
    )
    config = load_rag_config(config_path)
    assert isinstance(config.embedder, OpenAIEmbedderConfig)
    changed = config.model_copy(
        update={"embedder": config.embedder.model_copy(update={"request_timeout": 12})}
    )

    assert config_fingerprint(config) != config_fingerprint(changed)
    assert index_config_fingerprint(config) == index_config_fingerprint(changed)
