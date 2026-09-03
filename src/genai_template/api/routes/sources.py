"""Corpus source API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from genai_template.api.dependencies import get_source_service
from genai_template.schemas import (
    CreateSourceRequest,
    SourceCandidateResponse,
    SourceResponse,
)
from genai_template.services import SourceService

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/candidates", response_model=list[SourceCandidateResponse])
async def list_source_candidates(
    source_service: Annotated[SourceService, Depends(get_source_service)],
) -> list[SourceCandidateResponse]:
    """List directories available to ingest as sources.

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
    """List ingested corpus sources.

    Args:
        source_service:
            Configured source service.

    Returns:
        Persisted sources.
    """

    return [_source_response(source) for source in source_service.list_sources()]


@router.post("", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def ingest_source(
    request: CreateSourceRequest,
    source_service: Annotated[SourceService, Depends(get_source_service)],
) -> SourceResponse:
    """Ingest a selected prepared corpus directory.

    Args:
        request:
            Selected corpus directory request.
        source_service:
            Configured source service.

    Returns:
        Persisted source.

    Raises:
        HTTPException:
            If the corpus directory is invalid, missing, or already ingested.
    """

    try:
        source = source_service.ingest(request.directory)
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


@router.post("/{source_id}/refresh", response_model=SourceResponse)
async def refresh_source(
    source_id: int,
    source_service: Annotated[SourceService, Depends(get_source_service)],
) -> SourceResponse:
    """Rebuild an existing source from its prepared directory.

    Args:
        source_id:
            Identifier of the source to refresh.
        source_service:
            Configured source service.

    Returns:
        Refreshed source metadata.

    Raises:
        HTTPException:
            If the source or its directory cannot be found.
    """

    try:
        source = source_service.refresh(source_id)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return _source_response(source)


def _source_response(source: object) -> SourceResponse:
    """Convert a source ORM instance to its API response.

    Args:
        source:
            Persisted source object.

    Returns:
        Source metadata response.
    """

    return SourceResponse.model_validate(source, from_attributes=True)
