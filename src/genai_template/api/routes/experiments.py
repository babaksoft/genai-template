"""Experiment registry API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from genai_template.api.dependencies import get_experiment_service
from genai_template.schemas import CreateExperimentRequest, ExperimentResponse
from genai_template.services import ExperimentService

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    request: CreateExperimentRequest,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
) -> ExperimentResponse:
    """Create an experiment for a registered source.

    Args:
        request:
            Experiment metadata and source identifier.
        service:
            Experiment registry service.

    Returns:
        Newly created experiment.

    Raises:
        HTTPException:
            If the selected source does not exist.
    """

    try:
        experiment = service.create_experiment(
            source_id=request.source_id,
            name=request.name,
            description=request.description,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return ExperimentResponse.model_validate(experiment, from_attributes=True)


@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
) -> list[ExperimentResponse]:
    """List all registered experiments.

    Args:
        service:
            Experiment registry service.

    Returns:
        Registered experiments.
    """

    return [
        ExperimentResponse.model_validate(item, from_attributes=True)
        for item in service.list_experiments()
    ]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    service: Annotated[ExperimentService, Depends(get_experiment_service)],
) -> ExperimentResponse:
    """Get an experiment by identifier.

    Args:
        experiment_id:
            Canonical experiment identifier.
        service:
            Experiment registry service.

    Returns:
        Requested experiment.

    Raises:
        HTTPException:
            If the experiment does not exist.
    """

    try:
        experiment = service.get_experiment(experiment_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return ExperimentResponse.model_validate(experiment, from_attributes=True)
