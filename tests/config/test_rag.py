"""Tests for validated RAG configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from genai_template.common.types import VectorDistance
from genai_template.config import settings
from genai_template.config.rag import load_rag_config


def test_defaults_match_application_settings() -> None:
    """Loading without YAML reproduces the current RAG settings."""

    config = load_rag_config()

    assert config.experiment.name == settings.EXPERIMENT_NAME
    assert config.splitter.type == "sentence"
    assert config.splitter.chunk_size == settings.CHUNK_SIZE
    assert config.splitter.chunk_overlap == settings.CHUNK_OVERLAP
    assert config.embedder.type == "fastembed"
    assert config.embedder.model_name == settings.EMBEDDING_MODEL
    assert config.vector_store.type == settings.VECTOR_STORE.lower()
    assert config.vector_store.collection_name == settings.CHROMA_COLLECTION
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
        "experiment:\n  name: Small chunks\nsplitter:\n  chunk_size: 256\n",
        encoding="utf-8",
    )

    config = load_rag_config(config_path)

    assert config.experiment.name == "Small chunks"
    assert config.splitter.chunk_size == 256
    assert config.splitter.chunk_overlap == settings.CHUNK_OVERLAP
    assert config.embedder.model_name == settings.EMBEDDING_MODEL


@pytest.mark.parametrize(
    "yaml_text",
    [
        "unknown: true\n",
        "splitter:\n  type: tokens\n",
        "embedder:\n  type: openai\n",
        "vector_store:\n  type: pinecone\n",
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

    assert config.vector_store.persist_directory == (
        settings.REPO_ROOT / "custom" / "index"
    )


def test_configuration_models_are_immutable() -> None:
    """Resolved configuration cannot be mutated after validation."""

    config = load_rag_config()

    with pytest.raises(ValidationError):
        config.retrieval.top_k = 10


def test_documented_baseline_matches_defaults() -> None:
    """The documented baseline file remains equivalent to application defaults."""

    baseline_path = settings.PKG_ROOT / "experiments" / "configs" / "baseline.yml"

    assert load_rag_config(baseline_path) == load_rag_config()
    assert load_rag_config().vector_store.distance is VectorDistance.COSINE
