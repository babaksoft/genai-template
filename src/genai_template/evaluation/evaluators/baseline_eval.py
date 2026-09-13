"""Run YAML-configurable retrieval evaluation with persisted run records."""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from genai_template.config import RagConfig, load_rag_config, settings
from genai_template.config.logging import configure_logging
from genai_template.db import SessionLocal
from genai_template.db.models import Source
from genai_template.evaluation.metrics.baseline_metrics import (
    calculate_hit_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from genai_template.factories import (
    create_embedder,
    create_retrieval_pipeline,
    create_vector_store,
)
from genai_template.protocols import Retriever
from genai_template.schemas import RetrievalTest, RunMetrics
from genai_template.services import ExperimentService, RagConfigService, SourceService
from genai_template.utils import Timer

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = settings.EXPERIMENT_CONFIG_DIR / "baseline.yml"
DEFAULT_CORPUS_DIR = settings.CORPORA_DIR / "baseline"
DEFAULT_DATASET_PATH = settings.EVALUATION_DATA_DIR / "baseline-eval.json"


@dataclass(frozen=True)
class EvaluationResult:
    """Retrieval quality and runtime measurements for one evaluation query."""

    hit: bool
    recall: float
    precision: float
    retrieved_chunks: int
    best_distance: float | None
    worst_distance: float | None
    retrieval_time: float


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
) -> EvaluationResult:
    """Evaluate retrieval quality and runtime for one test case.

    Args:
        pipeline:
            Retrieval pipeline used to execute the test query.
        test:
            Retrieval evaluation case.
        k:
            Number of top-ranked chunks to evaluate.

    Returns:
        Quality scores and execution metrics for the query.
    """

    logger.info("Evaluating RAG retrieval: query='%s'", test.question)
    with Timer() as timer:
        retrieved_chunks = pipeline.retrieve(query=test.question, top_k=k)
    retrieved_documents = [
        Path(str(item.chunk.metadata["file_path"])).name for item in retrieved_chunks
    ]
    distances = [item.distance for item in retrieved_chunks]
    logger.info("Retrieved %d chunk(s).", len(retrieved_chunks))
    return EvaluationResult(
        hit=calculate_hit_at_k(retrieved_documents, test.expected_documents, k),
        recall=calculate_recall_at_k(retrieved_documents, test.expected_documents, k),
        precision=calculate_precision_at_k(
            retrieved_documents, test.expected_documents, k
        ),
        retrieved_chunks=len(retrieved_chunks),
        best_distance=min(distances) if distances else None,
        worst_distance=max(distances) if distances else None,
        retrieval_time=timer.elapsed,
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
        description="Build a registered source index and run retrieval evaluation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the deterministic source/config index before evaluation.",
    )
    return parser.parse_args(argv)


def _resolve_source(service: SourceService, corpus_path: Path) -> Source:
    """Find or register the source represented by an evaluation corpus.

    Args:
        service:
            Source lifecycle service rooted at the corpus parent directory.
        corpus_path:
            Evaluation corpus directory.

    Returns:
        Existing or newly registered source.
    """

    resolved = corpus_path.resolve()
    existing = next(
        (
            source
            for source in service.list_sources()
            if source.directory == str(resolved)
        ),
        None,
    )
    return existing or service.register(resolved.name)


def run_evaluation(
    config: RagConfig,
    corpus_path: Path,
    dataset_path: Path,
    reindex: bool = False,
    *,
    source_service: SourceService | None = None,
    config_service: RagConfigService | None = None,
    experiment_service: ExperimentService | None = None,
) -> None:
    """Evaluate retrieval and persist an experiment and completed runs.

    Args:
        config:
            Fully resolved RAG configuration.
        corpus_path:
            Directory containing documents to index.
        dataset_path:
            JSON retrieval evaluation dataset.
        reindex:
            Whether to rebuild an existing deterministic index.
        source_service:
            Optional source service override for testing or composition.
        config_service:
            Optional configuration registry override.
        experiment_service:
            Optional experiment/run service override.

    Raises:
        ValueError:
            If the evaluation dataset contains no tests.
    """

    config_service = config_service or RagConfigService(SessionLocal)
    source_service = source_service or SourceService(
        SessionLocal, corpus_path.resolve().parent, config_service
    )
    experiment_service = experiment_service or ExperimentService(SessionLocal)
    source = _resolve_source(source_service, corpus_path)
    config_record = config_service.register_config(config)
    experiment = experiment_service.create_experiment(
        source.id,
        settings.EXPERIMENT_NAME,
        "Baseline retrieval evaluation.",
    )
    collection_name = source_service.index_collection_name(source.id, config)
    store = create_vector_store(config.vector_store, collection_name)
    if reindex or not store.exists():
        source_service.rebuild_index(source.id, config_record.id)
        store = create_vector_store(config.vector_store, collection_name)
    else:
        logger.info("Reusing deterministic collection '%s'.", collection_name)

    retrieval_pipeline = create_retrieval_pipeline(
        config.retrieval,
        create_embedder(config.embedder),
        store,
    )
    tests = load_evaluation_tests(dataset_path)
    if not tests:
        raise ValueError(f"Evaluation dataset contains no tests: {dataset_path}")

    results: list[EvaluationResult] = []
    for test in tests:
        run = experiment_service.start_run(
            experiment.id, config_record.id, test.question
        )
        result = evaluate_test(retrieval_pipeline, test, config.retrieval.top_k)
        metrics = RunMetrics(
            query=test.question,
            embedding_model=config.embedder.model_name,
            vector_store=config.vector_store.type.capitalize(),
            llm_model=config.llm.model_name,
            top_k=config.retrieval.top_k,
            retrieved_chunks=result.retrieved_chunks,
            best_distance=result.best_distance,
            worst_distance=result.worst_distance,
            context_length=0,
            prompt_length=0,
            response_length=0,
            retrieval_time=result.retrieval_time,
            generation_time=0.0,
            total_time=result.retrieval_time,
        )
        experiment_service.complete_run(run, metrics)
        results.append(result)

    hit_rate = sum(result.hit for result in results) / len(results)
    recall = sum(result.recall for result in results) / len(results)
    precision = sum(result.precision for result in results) / len(results)
    k = config.retrieval.top_k
    logger.info(
        "Retrieval evaluation completed: experiment_id=%d, rag_config_id=%d, "
        "collection='%s', tests=%d, k=%d, hit@%d=%.3f, recall@%d=%.3f, "
        "precision@%d=%.3f",
        experiment.id,
        config_record.id,
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
            Optional command-line arguments.
    """

    args = parse_args(argv)
    run_evaluation(
        config=load_rag_config(args.config),
        corpus_path=args.corpus,
        dataset_path=args.dataset,
        reindex=args.reindex,
    )


if __name__ == "__main__":
    configure_logging()
    main()
