from fastapi import APIRouter

from genai_template.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """Return the current API health status.

    Returns:
        Current application health information.
    """

    return HealthResponse(
        status="healthy",
    )
