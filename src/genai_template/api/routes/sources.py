"""Corpus source API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from genai_template.api.dependencies import get_source_service
from genai_template.schemas import (
    CreateSourceRequest,
    IndexBuildResponse,
    SourceCandidateResponse,
    SourceResponse,
)
from genai_template.services import SourceService

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
        Transient indexing counts and duration.

    Raises:
        HTTPException:
            If the source or its directory cannot be found.
    """

    try:
        result = source_service.rebuild_index(source_id, rag_config_id)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return IndexBuildResponse.model_validate(result, from_attributes=True)


def _source_response(source: object) -> SourceResponse:
    """Convert a source ORM instance to its API response.

    Args:
        source:
            Persisted source object.

    Returns:
        Source metadata response.
    """

    return SourceResponse.model_validate(source, from_attributes=True)
