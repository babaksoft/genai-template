"""Run baseline retrieval evaluation against the evaluation dataset."""

import json
import logging
from pathlib import Path

from genai_template.components.context import ContextBuilder
from genai_template.components.language_models import OllamaLanguageModel
from genai_template.components.prompt import PromptBuilder
from genai_template.config import settings
from genai_template.config.logging import configure_logging
from genai_template.db import SessionLocal
from genai_template.pipelines import RetrievalPipeline
from genai_template.schemas import RetrievalTest
from genai_template.services import ExperimentService, RagService

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


def calculate_hit_at_k(
    retrieved_documents: list[str],
    expected_documents: list[str],
    k: int,
) -> bool:
    """Calculate whether at least one relevant document appears in top K.

    Args:
        retrieved_documents:
            Document names returned by retrieval in ranked order.

        expected_documents:
            Document names considered relevant.

        k:
            Number of top-ranked documents to evaluate.

    Returns:
        True when at least one relevant document appears in top K.
    """

    retrieved = set(retrieved_documents[:k])
    expected = set(expected_documents)

    return bool(retrieved & expected)


def calculate_recall_at_k(
    retrieved_documents: list[str],
    expected_documents: list[str],
    k: int,
) -> float:
    """Calculate retrieval recall at K.

    Args:
        retrieved_documents:
            Document names returned by retrieval in ranked order.

        expected_documents:
            Document names considered relevant.

        k:
            Number of top-ranked documents to evaluate.

    Returns:
        Proportion of relevant documents retrieved in top K.
    """

    expected = set(expected_documents)
    if not expected:
        return 0.0

    retrieved = set(retrieved_documents[:k])

    return len(retrieved & expected) / len(expected)


def calculate_precision_at_k(
    retrieved_documents: list[str],
    expected_documents: list[str],
    k: int,
) -> float:
    """Calculate retrieval precision at K.

    Args:
        retrieved_documents:
            Document names returned by retrieval in ranked order.

        expected_documents:
            Document names considered relevant.

        k:
            Number of top-ranked documents to evaluate.

    Returns:
        Proportion of top-K retrieved documents that are relevant.
    """

    retrieved = retrieved_documents[:k]
    if not retrieved:
        return 0.0

    expected = set(expected_documents)

    return sum(document in expected for document in retrieved) / len(retrieved)


def extract_document_name(source: str) -> str:
    """Extract document filename from a chunk source path.

    Args:
        source:
            Source path associated with a retrieved chunk.

    Returns:
        Document filename.
    """

    return Path(source).name


def evaluate_test(
    rag_service: RagService,
    test: RetrievalTest,
    k: int,
) -> tuple[bool, float, float]:
    """Evaluate retrieval for a single test case.

    Args:
        rag_service:
            RAG service used to execute the test query.

        test:
            Retrieval evaluation case.

        k:
            Number of top-ranked chunks to evaluate.

    Returns:
        Tuple containing Hit@K, Recall@K, and Precision@K.
    """

    logger.info("Evaluating RAG pipeline: query='%s'", test.question)

    result = rag_service.answer(test.question)
    retrieved_documents = [
        extract_document_name(str(retrieved_chunk.chunk.metadata["file_path"]))
        for retrieved_chunk in result.retrieved_chunks
    ]

    logger.info("Retrieved %d chunk(s).", len(result.retrieved_chunks))

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

    tests = load_evaluation_tests(settings.EVALUATION_DATA_PATH)

    rag_service = RagService(
        retrieval_pipeline=RetrievalPipeline(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        language_model=OllamaLanguageModel(settings.LLM_MODEL),
        experiment_service=ExperimentService(
            session_factory=SessionLocal,
        ),
    )

    k = settings.TOP_K

    results = [
        evaluate_test(
            rag_service=rag_service,
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
