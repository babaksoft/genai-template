"""RAG configuration registry API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from genai_template.api.dependencies import get_rag_config_service
from genai_template.config import RagConfig
from genai_template.db.models import RagConfig as RagConfigRecord
from genai_template.schemas import RagConfigResponse
from genai_template.services import RagConfigService

router = APIRouter(prefix="/rag-configs", tags=["rag-configs"])


def _response(record: RagConfigRecord) -> RagConfigResponse:
    """Convert a persisted configuration to its API representation.

    Args:
        record:
            Persisted configuration record.

    Returns:
        Configuration API response.
    """

    return RagConfigResponse(
        id=record.id,
        config_fingerprint=record.config_fingerprint,
        config=RagConfig.model_validate_json(record.config_json),
        created_at=record.created_at,
    )


@router.post("", response_model=RagConfigResponse, status_code=status.HTTP_201_CREATED)
async def register_config(
    config: RagConfig,
    service: Annotated[RagConfigService, Depends(get_rag_config_service)],
) -> RagConfigResponse:
    """Idempotently register a resolved RAG configuration.

    Args:
        config:
            Validated configuration to register.
        service:
            RAG configuration registry service.

    Returns:
        Existing or newly registered configuration.

    Raises:
        HTTPException:
            If fingerprint reuse does not match the stored JSON.
    """

    try:
        record = service.register_config(config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return _response(record)


@router.get("", response_model=list[RagConfigResponse])
async def list_configs(
    service: Annotated[RagConfigService, Depends(get_rag_config_service)],
) -> list[RagConfigResponse]:
    """List all registered RAG configurations.

    Args:
        service:
            RAG configuration registry service.

    Returns:
        Registered configurations.
    """

    return [_response(record) for record in service.list_configs()]


@router.get("/{rag_config_id}", response_model=RagConfigResponse)
async def get_config(
    rag_config_id: int,
    service: Annotated[RagConfigService, Depends(get_rag_config_service)],
) -> RagConfigResponse:
    """Get a registered RAG configuration by identifier.

    Args:
        rag_config_id:
            Canonical configuration identifier.
        service:
            RAG configuration registry service.

    Returns:
        Requested configuration.

    Raises:
        HTTPException:
            If the configuration does not exist.
    """

    try:
        record = service.get_config(rag_config_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return _response(record)
