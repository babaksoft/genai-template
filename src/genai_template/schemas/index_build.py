"""Index-build lifecycle and availability contracts."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class IndexBuildStatus(StrEnum):
    """Allowed durable states for an index-build attempt."""

    BUILDING = "building"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IndexStatusReason(StrEnum):
    """Machine-readable reasons for the current index availability decision."""

    CURRENT = "current"
    UNBUILT = "unbuilt"
    STALE = "stale"
    BUILDING = "building"
    FAILED = "failed"
    COLLECTION_MISSING = "collection_missing"
    COUNT_MISMATCH = "count_mismatch"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    CORPUS_INVALID = "corpus_invalid"
    UNTRACKED = "untracked"


class IndexStatus(BaseModel):
    """Describe the verified state of one deterministic source index.

    Attributes:
        source_id:
            Registered source identifier.
        rag_config_id:
            Selected immutable RAG configuration identifier.
        collection_name:
            Deterministic vector collection name.
        index_fingerprint:
            Fingerprint of all index-affecting configuration.
        current_corpus_fingerprint:
            Currently published validated corpus identity, when manifest-backed.
        built_corpus_fingerprint:
            Corpus identity recorded by the latest successful build.
        latest_build_id:
            Identifier of the newest rebuild attempt.
        latest_build_status:
            Lifecycle state of the newest rebuild attempt.
        build_started_at:
            Start time of the newest rebuild attempt.
        build_finished_at:
            Finish time of the newest rebuild attempt.
        document_count:
            Document count from the latest successful build.
        chunk_count:
            Chunk count from the latest successful build.
        collection_count:
            Current vector collection count when it could be inspected.
        indexing_duration:
            Indexing duration from the latest successful build.
        available:
            Whether answer execution may use the selected collection.
        reason:
            Machine-readable explanation of the availability decision.
    """

    source_id: int = Field(..., ge=1, description="Registered source identifier.")
    rag_config_id: int = Field(
        ..., ge=1, description="Selected immutable RAG configuration identifier."
    )
    collection_name: str = Field(
        ..., min_length=1, description="Deterministic vector collection name."
    )
    index_fingerprint: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="Fingerprint of all index-affecting configuration.",
    )
    current_corpus_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Currently published validated manifest identity.",
    )
    built_corpus_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Corpus identity from the latest successful build.",
    )
    latest_build_id: int | None = Field(
        default=None, ge=1, description="Newest rebuild attempt identifier."
    )
    latest_build_status: IndexBuildStatus | None = Field(
        default=None, description="Lifecycle state of the newest rebuild attempt."
    )
    build_started_at: datetime | None = Field(
        default=None, description="Start time of the newest rebuild attempt."
    )
    build_finished_at: datetime | None = Field(
        default=None, description="Finish time of the newest rebuild attempt."
    )
    document_count: int | None = Field(
        default=None,
        ge=0,
        description="Document count from the latest successful build.",
    )
    chunk_count: int | None = Field(
        default=None,
        ge=0,
        description="Chunk count from the latest successful build.",
    )
    collection_count: int | None = Field(
        default=None,
        ge=0,
        description="Current vector collection count when inspected.",
    )
    indexing_duration: float | None = Field(
        default=None,
        ge=0,
        description="Duration from the latest successful build.",
    )
    available: bool = Field(
        ..., description="Whether answer execution may use this index."
    )
    reason: IndexStatusReason = Field(
        ..., description="Machine-readable availability decision."
    )
