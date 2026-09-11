"""Run YAML-configurable retrieval evaluation against a corpus and dataset."""

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from genai_template.config import (
    RagConfig,
    config_fingerprint,
    index_config_fingerprint,
    load_rag_config,
    settings,
)
from genai_template.config.logging import configure_logging
from genai_template.db import SessionLocal
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
from genai_template.pipelines import IndexingPipeline
from genai_template.protocols import Retriever
from genai_template.schemas import RetrievalTest
from genai_template.services import ExperimentService

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
    pipeline: Retriever,
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
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete and rebuild the matching configuration-specific index.",
    )

    return parser.parse_args(argv)


def _corpus_fingerprint(corpus_path: Path) -> str:
    """Fingerprint supported corpus file paths and contents deterministically.

    Args:
        corpus_path:
            Directory containing the corpus.

    Returns:
        Hexadecimal SHA-256 corpus fingerprint.

    Raises:
        FileNotFoundError:
            If the corpus directory does not exist.
        NotADirectoryError:
            If the corpus path is not a directory.
    """

    if not corpus_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {corpus_path}")
    if not corpus_path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {corpus_path}")

    digest = hashlib.sha256()
    supported_files = sorted(
        (
            path
            for path in corpus_path.iterdir()
            if path.is_file() and path.suffix in {".md", ".txt"}
        ),
        key=lambda path: path.name,
    )
    for path in supported_files:
        relative_path = path.relative_to(corpus_path).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)

    return digest.hexdigest()


def _evaluation_collection_name(
    index_fingerprint: str,
    corpus_fingerprint: str | None = None,
) -> str:
    """Derive a dedicated collection name for an index and corpus.

    Args:
        index_fingerprint:
            SHA-256 fingerprint of index-affecting configuration.
        corpus_fingerprint:
            SHA-256 fingerprint of supported corpus files. If omitted, the
            index fingerprint is also used for backward compatibility.

    Returns:
        Chroma collection name reserved for the evaluation run.
    """

    corpus_identity = corpus_fingerprint or index_fingerprint
    return f"evaluation-{index_fingerprint[:16]}-{corpus_identity[:16]}"


def run_evaluation(
    config: RagConfig,
    corpus_path: Path,
    dataset_path: Path,
    reindex: bool = False,
) -> None:
    """Reuse or build an evaluation index and calculate retrieval metrics.

    Args:
        config:
            Fully resolved RAG configuration.
        corpus_path:
            Directory containing documents to index.
        dataset_path:
            JSON retrieval evaluation dataset.
        reindex:
            Whether to delete and rebuild the resolved experiment collection.

    Raises:
        ValueError:
            If the evaluation dataset contains no tests.
    """

    fingerprint = config_fingerprint(config)
    index_fingerprint = index_config_fingerprint(config)
    corpus_fingerprint = _corpus_fingerprint(corpus_path)
    collection_name = _evaluation_collection_name(
        index_fingerprint,
        corpus_fingerprint,
    )
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

    store = create_vector_store(store_config)
    if reindex:
        store.delete()
        store = create_vector_store(store_config)

    embedder = create_embedder(config.embedder)
    if store.count() == 0:
        indexing_pipeline = IndexingPipeline(
            splitter=create_splitter(config.splitter),
            embedder=embedder,
            store=store,
        )
        indexing_pipeline.run(corpus_path)
    else:
        logger.info(
            "Reusing populated evaluation collection '%s'.",
            collection_name,
        )

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
    config = load_rag_config(args.config)
    ExperimentService(SessionLocal).register_experiment(config)
    run_evaluation(
        config=config,
        corpus_path=args.corpus,
        dataset_path=args.dataset,
        reindex=args.reindex,
    )


if __name__ == "__main__":
    configure_logging()
    main()
