"""Run baseline retrieval evaluation against the evaluation dataset."""

import json
import logging
from pathlib import Path

from genai_template.components.embeddings import FastEmbedEmbeddingModel
from genai_template.config import settings
from genai_template.config.logging import configure_logging
from genai_template.evaluation.metrics.baseline_metrics import (
    calculate_hit_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from genai_template.pipelines import RetrievalPipeline
from genai_template.schemas import RetrievalTest
from genai_template.stores.vector import ChromaStore

logger = logging.getLogger(__name__)


def load_evaluation_tests(path: Path) -> list[RetrievalTest]:
    """Load and validate retrieval evaluation tests.

    Args:
        path:
            Path to the evaluation dataset.

    Returns:
        Validated retrieval evaluation tests.
    """

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [RetrievalTest.model_validate(item) for item in data]


def evaluate_test(
    pipeline: RetrievalPipeline,
    test: RetrievalTest,
    k: int,
) -> tuple[bool, float, float]:
    """Evaluate retrieval for a single test case.

    Args:
        pipeline:
            RAG retrieval pipeline used to execute the test query.
        test:
            Retrieval evaluation case.
        k:
            Number of top-ranked chunks to evaluate.

    Returns:
        Tuple containing Hit@K, Recall@K, and Precision@K.
    """

    logger.info("Evaluating RAG retrieval: query='%s'", test.question)

    retrieved_chunks = pipeline.retrieve(query=test.question, top_k=k)
    retrieved_documents = [
        Path(str(retrieved_chunk.chunk.metadata["file_path"])).name
        for retrieved_chunk in retrieved_chunks
    ]

    logger.info("Retrieved %d chunk(s).", len(retrieved_chunks))

    return (
        calculate_hit_at_k(
            retrieved_documents,
            test.expected_documents,
            k,
        ),
        calculate_recall_at_k(
            retrieved_documents,
            test.expected_documents,
            k,
        ),
        calculate_precision_at_k(
            retrieved_documents,
            test.expected_documents,
            k,
        ),
    )


def main() -> None:
    """Run baseline retrieval evaluation."""

    tests = load_evaluation_tests(
        settings.EVALUATION_DATA_DIR / "baseline-eval.json"
    )
    k = settings.TOP_K

    retrieval_pipeline = RetrievalPipeline(
        embedder=FastEmbedEmbeddingModel(),
        store=ChromaStore(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name="baseline_corpus",
        ),
    )
    results = [
        evaluate_test(
            pipeline=retrieval_pipeline,
            test=test,
            k=k,
        )
        for test in tests
    ]

    hit_rate = sum(result[0] for result in results) / len(results)
    recall = sum(result[1] for result in results) / len(results)
    precision = sum(result[2] for result in results) / len(results)

    logger.info(
        "Baseline retrieval evaluation completed: "
        "tests=%d, k=%d, hit@%d=%.3f, recall@%d=%.3f, precision@%d=%.3f",
        len(results),
        k,
        k,
        hit_rate,
        k,
        recall,
        k,
        precision,
    )


if __name__ == "__main__":
    configure_logging()
    main()
