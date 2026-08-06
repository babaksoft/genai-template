"""Baseline corpus ingestion utility."""

import logging

from genai_template.config import settings
from genai_template.pipelines import IndexingPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """Run baseline ingestion for the evaluation corpus."""

    pipeline = IndexingPipeline()
    result = pipeline.run(settings.DATA_DIR)

    logger.info(
        "Baseline ingestion completed",
        extra={
            "document_count": result.documents_indexed,
            "chunk_count": result.chunks_indexed,
            "indexing_time": result.indexing_time,
        },
    )


if __name__ == "__main__":
    main()
