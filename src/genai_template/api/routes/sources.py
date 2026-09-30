"""Corpus source API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from genai_template.api.dependencies import get_source_service
from genai_template.schemas import (
    CreateSourceRequest,
    IndexBuildAttemptResponse,
    IndexBuildResponse,
    IndexStatus,
    IndexStatusReason,
    SourceCandidateResponse,
    SourceResponse,
)
from genai_template.services import IndexBuildInProgressError, SourceService
from genai_template.workflow.portfolio.corpus import CorpusValidationError

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/candidates", response_model=list[SourceCandidateResponse])
async def list_source_candidates(
    source_service: Annotated[SourceService, Depends(get_source_service)],
) -> list[SourceCandidateResponse]:
    """List directories available to register as sources.

    Args:
        source_service:
            Configured source service.

    Returns:
        Available corpus directory names.
    """

    return [
        SourceCandidateResponse(name=name) for name in source_service.list_candidates()
    ]


@router.get("", response_model=list[SourceResponse])
async def list_sources(
    source_service: Annotated[SourceService, Depends(get_source_service)],
) -> list[SourceResponse]:
    """List registered corpus sources.

    Args:
        source_service:
            Configured source service.

    Returns:
        Persisted sources.
    """

    return [_source_response(source) for source in source_service.list_sources()]


@router.post("", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def register_source(
    request: CreateSourceRequest,
    source_service: Annotated[SourceService, Depends(get_source_service)],
) -> SourceResponse:
    """Register a selected prepared corpus directory.

    Args:
        request:
            Selected corpus directory request.
        source_service:
            Configured source service.

    Returns:
        Persisted source.

    Raises:
        HTTPException:
            If the corpus directory is invalid, missing, or already registered.
    """

    try:
        source = source_service.register(request.directory)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        status_code = (
            status.HTTP_409_CONFLICT
            if "already exists" in str(exc)
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return _source_response(source)


@router.get(
    "/{source_id}/indexes/{rag_config_id}",
    response_model=IndexStatus,
)
async def get_source_index_status(
    source_id: int,
    rag_config_id: int,
    source_service: Annotated[SourceService, Depends(get_source_service)],
) -> IndexStatus:
    """Inspect freshness and availability for a deterministic source index.

    Args:
        source_id:
            Identifier of the registered source.
        rag_config_id:
            Identifier of the persisted RAG configuration.
        source_service:
            Configured source service.

    Returns:
        Current verified index status.

    Raises:
        HTTPException:
            If the source or configuration does not exist.
    """

    try:
        return source_service.get_index_status(source_id, rag_config_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.put("/{source_id}/indexes/{rag_config_id}", response_model=IndexBuildResponse)
async def rebuild_source_index(
    source_id: int,
    rag_config_id: int,
    source_service: Annotated[SourceService, Depends(get_source_service)],
) -> IndexBuildResponse:
    """Rebuild a source's deterministic index with a registered config.

    Args:
        source_id:
            Identifier of the source to rebuild.
        rag_config_id:
            Identifier of the persisted RAG configuration.
        source_service:
            Configured source service.

    Returns:
        Persisted build details, compatibility metrics, and resulting status.

    Raises:
        HTTPException:
            If the source or config cannot be found, the corpus is invalid, or a
            rebuild is already running.
    """

    try:
        result = source_service.rebuild_index(source_id, rag_config_id)
    except IndexBuildInProgressError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(exc),
                "reason": IndexStatusReason.BUILDING.value,
                "action": "retry_status",
                "status_endpoint": _index_endpoint(source_id, rag_config_id),
            },
        ) from exc
    except CorpusValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": str(exc),
                "reason": IndexStatusReason.CORPUS_INVALID.value,
                "action": "repair_corpus",
            },
        ) from exc
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    index_status = source_service.get_index_status(source_id, rag_config_id)
    if (
        index_status.latest_build_id is None
        or index_status.latest_build_status is None
        or index_status.build_started_at is None
        or index_status.build_finished_at is None
        or index_status.document_count is None
        or index_status.chunk_count is None
        or index_status.indexing_duration is None
    ):
        raise RuntimeError("Successful rebuild did not produce complete build state.")

    build = IndexBuildAttemptResponse(
        id=index_status.latest_build_id,
        status=index_status.latest_build_status,
        corpus_fingerprint=index_status.built_corpus_fingerprint,
        started_at=index_status.build_started_at,
        finished_at=index_status.build_finished_at,
        document_count=index_status.document_count,
        chunk_count=index_status.chunk_count,
        indexing_duration=index_status.indexing_duration,
    )

    return IndexBuildResponse(
        documents_indexed=result.documents_indexed,
        chunks_indexed=result.chunks_indexed,
        indexing_time=result.indexing_time,
        build=build,
        status=index_status,
    )


def _index_endpoint(source_id: int, rag_config_id: int) -> str:
    """Return the API-relative endpoint for one deterministic index.

    Args:
        source_id:
            Registered source identifier.
        rag_config_id:
            Registered RAG configuration identifier.

    Returns:
        API-relative status and rebuild endpoint.
    """

    return f"/sources/{source_id}/indexes/{rag_config_id}"


def _source_response(source: object) -> SourceResponse:
    """Convert a source ORM instance to its API response.

    Args:
        source:
            Persisted source object.

    Returns:
        Source metadata response.
    """

    return SourceResponse.model_validate(source, from_attributes=True)
