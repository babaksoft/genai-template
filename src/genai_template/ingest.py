"""Baseline corpus ingestion utility."""

import logging

from genai_template.components.embeddings import FastEmbedEmbeddingModel
from genai_template.components.readers import TextReader
from genai_template.components.splitters import DocumentSplitter
from genai_template.config import settings
from genai_template.config.logging import configure_logging
from genai_template.pipelines import IndexingPipeline
from genai_template.stores.vector import ChromaStore

logger = logging.getLogger(__name__)


def main() -> None:
    """Run baseline ingestion for the evaluation corpus."""

    pipeline = IndexingPipeline(
        reader=TextReader(),
        splitter=DocumentSplitter(),
        embedder=FastEmbedEmbeddingModel(),
        store=ChromaStore(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name="baseline_corpus",
        ),
    )
    pipeline.run(settings.CORPORA_DIR / "baseline")

    logger.info("Baseline ingestion completed.")


if __name__ == "__main__":
    configure_logging()
    main()
