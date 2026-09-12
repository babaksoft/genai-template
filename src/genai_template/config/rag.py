"""Validated configuration for RAG experiments."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from genai_template.common.types import VectorDistance
from genai_template.config import settings


class _ImmutableConfig(BaseModel):
    """Base model for immutable configuration sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitterConfig(_ImmutableConfig):
    """Common document splitting configuration."""


class SentenceSplitterConfig(SplitterConfig):
    """Sentence splitting configuration."""

    type: Literal["sentence"] = "sentence"
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> SentenceSplitterConfig:
        """Ensure overlap leaves some unique content in every chunk.

        Returns:
            The validated splitter configuration.

        Raises:
            ValueError:
                If chunk overlap is not smaller than chunk size.
        """

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class MarkdownSplitterConfig(SplitterConfig):
    """Markdown header-based splitting configuration."""

    type: Literal["markdown"] = "markdown"
    header_path_separator: str = Field(default="/", min_length=1)


AnySplitterConfig = Annotated[
    SentenceSplitterConfig | MarkdownSplitterConfig,
    Field(discriminator="type"),
]


class EmbedderConfig(_ImmutableConfig):
    """Common embedding model configuration."""

    model_name: str = Field(min_length=1)


class FastEmbedEmbedderConfig(EmbedderConfig):
    """FastEmbed embedding model configuration."""

    type: Literal["fastembed"] = "fastembed"


class OpenAIEmbedderConfig(EmbedderConfig):
    """OpenAI embedding model configuration."""

    type: Literal["openai"] = "openai"
    dimensions: int | None = Field(default=None, gt=0)
    request_timeout: float = Field(default=settings.REQUEST_TIMEOUT, gt=0)


AnyEmbedderConfig = Annotated[
    FastEmbedEmbedderConfig | OpenAIEmbedderConfig,
    Field(discriminator="type"),
]


class VectorStoreConfig(_ImmutableConfig):
    """Common vector store configuration."""

    distance: VectorDistance


class ChromaVectorStoreConfig(VectorStoreConfig):
    """Chroma local vector store configuration."""

    type: Literal["chroma"] = "chroma"
    persist_directory: Path


class QdrantVectorStoreConfig(VectorStoreConfig):
    """Qdrant local or server vector-store configuration."""

    type: Literal["qdrant"] = "qdrant"
    location: Literal["local", "server"]
    path: Path | None = None
    url: str | None = Field(default=None, pattern=r"^https?://")

    @model_validator(mode="after")
    def validate_location(self) -> QdrantVectorStoreConfig:
        """Ensure connection fields match the selected Qdrant location.

        Returns:
            The validated Qdrant vector-store configuration.

        Raises:
            ValueError:
                If local and server connection fields are missing or mixed.
        """

        if self.location == "local":
            if self.path is None:
                raise ValueError("path is required for local Qdrant")
            if self.url is not None:
                raise ValueError("url is not supported for local Qdrant")
        else:
            if self.url is None:
                raise ValueError("url is required for server Qdrant")
            if self.path is not None:
                raise ValueError("path is not supported for server Qdrant")

        return self


AnyVectorStoreConfig = Annotated[
    ChromaVectorStoreConfig | QdrantVectorStoreConfig,
    Field(discriminator="type"),
]


class RetrievalConfig(_ImmutableConfig):
    """Retrieval pipeline configuration."""

    top_k: int = Field(gt=0)


class LLMConfig(_ImmutableConfig):
    """Common language model configuration."""

    model_name: str = Field(min_length=1)
    request_timeout: float = Field(default=settings.REQUEST_TIMEOUT, gt=0)


class OllamaLLMConfig(LLMConfig):
    """Ollama language model configuration."""

    type: Literal["ollama"] = "ollama"
    base_url: str = Field(min_length=1, pattern=r"^https?://")


class OpenAILLMConfig(LLMConfig):
    """OpenAI language model configuration."""

    type: Literal["openai"] = "openai"


AnyLLMConfig = Annotated[
    OllamaLLMConfig | OpenAILLMConfig,
    Field(discriminator="type"),
]


class RagConfig(_ImmutableConfig):
    """Fully resolved, portable RAG configuration."""

    splitter: AnySplitterConfig
    embedder: AnyEmbedderConfig
    vector_store: AnyVectorStoreConfig
    retrieval: RetrievalConfig
    llm: AnyLLMConfig


def canonical_config_json(config: RagConfig) -> str:
    """Serialize a resolved RAG configuration deterministically.

    Args:
        config:
            Fully resolved RAG configuration.

    Returns:
        Compact JSON with stable key ordering and JSON-compatible values.
    """

    return json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def config_fingerprint(config: RagConfig) -> str:
    """Calculate a stable SHA-256 fingerprint for a resolved configuration.

    Args:
        config:
            Fully resolved RAG configuration.

    Returns:
        Hexadecimal SHA-256 configuration fingerprint.
    """

    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def index_config_fingerprint(config: RagConfig) -> str:
    """Fingerprint settings that determine the contents of a vector index.

    Backend connection and storage-location settings identify where an index
    lives rather than how its vectors are produced, so they are intentionally
    excluded. Transport settings and retrieval/generation settings likewise do
    not affect indexed vectors.

    Args:
        config:
            Fully resolved RAG configuration.

    Returns:
        Hexadecimal SHA-256 fingerprint of index-affecting settings.
    """

    embedder_config = config.embedder.model_dump(mode="json")
    embedder_config.pop("request_timeout", None)

    index_config = {
        "embedder": embedder_config,
        "splitter": config.splitter.model_dump(mode="json"),
        "vector_store": {
            "distance": config.vector_store.distance.value,
            "type": config.vector_store.type,
        },
    }
    canonical_json = json.dumps(index_config, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _settings_config() -> dict[str, Any]:
    """Build configuration data from the current application settings.

    Returns:
        Nested configuration data matching :class:`RagConfig`.
    """

    return {
        "splitter": {
            "type": "sentence",
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        },
        "embedder": {
            "type": "fastembed",
            "model_name": settings.EMBEDDING_MODEL,
        },
        "vector_store": {
            "type": settings.VECTOR_STORE.lower(),
            "persist_directory": settings.CHROMA_PERSIST_DIR,
            "distance": settings.CHROMA_DISTANCE,
        },
        "retrieval": {"top_k": settings.TOP_K},
        "llm": {
            "type": "ollama",
            "model_name": settings.LLM_MODEL,
            "base_url": settings.OLLAMA_BASE_URL,
            "request_timeout": settings.REQUEST_TIMEOUT,
        },
    }


def _merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge YAML overrides into default configuration data.

    Args:
        base:
            Default configuration data.
        overrides:
            User-provided partial configuration data.

    Returns:
        Merged configuration data.
    """

    provider_sections = {"embedder", "llm", "splitter", "vector_store"}
    result = deepcopy(base)
    for key, value in overrides.items():
        existing = result.get(key)
        provider_changed = (
            key in provider_sections
            and isinstance(existing, dict)
            and isinstance(value, dict)
            and "type" in value
            and value["type"] != existing.get("type")
        )
        if provider_changed:
            result[key] = deepcopy(value)
            continue
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _merge_config(existing, value)
        else:
            result[key] = value
    return result


def _resolve_paths(data: dict[str, Any]) -> None:
    """Resolve relative filesystem values in merged configuration data.

    Args:
        data:
            Mutable merged configuration data.
    """

    vector_store = data.get("vector_store")
    if not isinstance(vector_store, dict):
        return

    path_key = "path" if vector_store.get("type") == "qdrant" else "persist_directory"
    path_value = vector_store.get(path_key)
    if path_value is None:
        return

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = settings.REPO_ROOT / path
    vector_store[path_key] = path.resolve()


def load_rag_config(path: Path | None = None) -> RagConfig:
    """Load a fully resolved RAG configuration.

    Application settings provide the defaults. When ``path`` is supplied, its
    YAML mapping is recursively overlaid on those defaults before validation.
    Relative paths are interpreted from the repository root.

    Args:
        path:
            Optional YAML configuration file. A relative path is resolved from
            the repository root.

    Returns:
        An immutable, validated RAG configuration.

    Raises:
        FileNotFoundError:
            If the configuration file does not exist.
        ValueError:
            If the file is malformed YAML or its root is not a mapping.
        pydantic.ValidationError:
            If configuration keys or values are invalid.
    """

    data = _settings_config()
    if path is not None:
        config_path = path.expanduser()
        if not config_path.is_absolute():
            config_path = settings.REPO_ROOT / config_path

        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"Malformed YAML configuration: {config_path}") from error

        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError("RAG configuration must be a YAML mapping")
        data = _merge_config(data, loaded)

    _resolve_paths(data)
    return RagConfig.model_validate(data)
