"""Baseline corpus ingestion utility."""

import logging

from genai_template.components.readers import TextReader
from genai_template.config import load_rag_config, settings
from genai_template.config.logging import configure_logging
from genai_template.factories import (
    create_embedder,
    create_splitter,
    create_vector_store,
)
from genai_template.pipelines import IndexingPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """Run baseline ingestion for the evaluation corpus."""

    config = load_rag_config()
    store_config = config.vector_store.model_copy(
        update={"collection_name": "baseline_corpus"}
    )
    pipeline = IndexingPipeline(
        reader=TextReader(),
        splitter=create_splitter(config.splitter),
        embedder=create_embedder(config.embedder),
        store=create_vector_store(store_config),
    )
    pipeline.run(settings.CORPORA_DIR / "baseline")

    logger.info("Baseline ingestion completed.")


if __name__ == "__main__":
    configure_logging()
    main()
