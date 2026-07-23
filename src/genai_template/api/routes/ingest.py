from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from genai_template.api.dependencies import get_indexing_pipeline
from genai_template.pipeline import IndexingPipeline
from genai_template.schemas import IngestRequest, IngestResponse

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
)
async def ingest(
    request: IngestRequest,
    indexing_pipeline: Annotated[
        IndexingPipeline,
        Depends(get_indexing_pipeline),
    ],
) -> IngestResponse:
    """Index documents from a directory.

    Args:
        request:
            Document indexing request.

        indexing_pipeline:
            Configured indexing pipeline.

    Returns:
        Summary of the indexing operation.
    """

    result = indexing_pipeline.run(Path(request.directory))

    return IngestResponse(
        documents_indexed=result.documents_indexed,
        chunks_indexed=result.chunks_indexed,
        indexing_time=result.indexing_time,
    )
