"""Run YAML-configurable retrieval evaluation against a corpus and dataset."""

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from genai_template.config import settings
from genai_template.config.logging import configure_logging
from genai_template.config.rag import RagConfig, load_rag_config
from genai_template.evaluation.metrics.baseline_metrics import (
    calculate_hit_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from genai_template.factories import (
    create_embedder,
    create_retrieval_pipeline,
    create_splitter,
    create_vector_store,
)
from genai_template.pipelines import IndexingPipeline, RetrievalPipeline
from genai_template.schemas import RetrievalTest

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = settings.EXPERIMENT_CONFIG_DIR / "baseline.yml"
DEFAULT_CORPUS_DIR = settings.CORPORA_DIR / "baseline"
DEFAULT_DATASET_PATH = settings.EVALUATION_DATA_DIR / "baseline-eval.json"


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse retrieval evaluation command-line arguments.

    Args:
        argv:
            Optional argument sequence. When omitted, arguments are read from
            the process command line.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Index a corpus and run retrieval evaluation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Optional YAML RAG configuration path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help=f"Corpus directory (default: {DEFAULT_CORPUS_DIR}).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Evaluation dataset (default: {DEFAULT_DATASET_PATH}).",
    )

    return parser.parse_args(argv)


def _config_fingerprint(config: RagConfig) -> str:
    """Calculate a stable fingerprint for a resolved configuration.

    Args:
        config:
            Fully resolved RAG configuration.

    Returns:
        Hexadecimal SHA-256 configuration fingerprint.
    """

    canonical_json = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _evaluation_collection_name(config_fingerprint: str) -> str:
    """Derive a dedicated collection name for an evaluation configuration.

    Args:
        config_fingerprint:
            SHA-256 fingerprint of the resolved configuration.

    Returns:
        Chroma collection name reserved for the evaluation run.
    """

    return f"evaluation_{config_fingerprint[:16]}"


def run_evaluation(
    config: RagConfig,
    corpus_path: Path,
    dataset_path: Path,
) -> None:
    """Rebuild an evaluation index and calculate retrieval metrics.

    Args:
        config:
            Fully resolved RAG configuration.
        corpus_path:
            Directory containing documents to index.
        dataset_path:
            JSON retrieval evaluation dataset.

    Raises:
        ValueError:
            If the evaluation dataset contains no tests.
    """

    fingerprint = _config_fingerprint(config)
    collection_name = _evaluation_collection_name(fingerprint)
    store_config = config.vector_store.model_copy(
        update={"collection_name": collection_name}
    )

    logger.info(
        "Starting retrieval evaluation: experiment='%s', "
        "config_fingerprint=%s, collection='%s'",
        config.experiment.name,
        fingerprint,
        collection_name,
    )

    # Slice 3 deliberately rebuilds the dedicated collection so changes to
    # embeddings or chunking can never query incompatible stored vectors.
    existing_store = create_vector_store(store_config)
    existing_store.delete()
    store = create_vector_store(store_config)

    embedder = create_embedder(config.embedder)
    indexing_pipeline = IndexingPipeline(
        splitter=create_splitter(config.splitter),
        embedder=embedder,
        store=store,
    )
    indexing_pipeline.run(corpus_path)

    retrieval_pipeline = create_retrieval_pipeline(
        config.retrieval,
        embedder,
        store,
    )
    tests = load_evaluation_tests(dataset_path)
    if not tests:
        raise ValueError(f"Evaluation dataset contains no tests: {dataset_path}")

    k = config.retrieval.top_k
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
        "Retrieval evaluation completed: experiment='%s', "
        "config_fingerprint=%s, collection='%s', "
        "tests=%d, k=%d, hit@%d=%.3f, recall@%d=%.3f, precision@%d=%.3f",
        config.experiment.name,
        fingerprint,
        collection_name,
        len(results),
        k,
        k,
        hit_rate,
        k,
        recall,
        k,
        precision,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run retrieval evaluation from optional YAML and path overrides.

    Args:
        argv:
            Optional argument sequence. When omitted, arguments are read from
            the process command line.
    """

    args = parse_args(argv)
    run_evaluation(
        config=load_rag_config(args.config),
        corpus_path=args.corpus,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    configure_logging()
    main()
