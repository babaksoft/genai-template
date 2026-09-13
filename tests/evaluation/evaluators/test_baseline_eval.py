"""Tests for persisted baseline retrieval evaluation."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from genai_template.config import load_rag_config, settings
from genai_template.db.models import Experiment, RagConfig, Source
from genai_template.evaluation.evaluators.baseline_eval import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CORPUS_DIR,
    DEFAULT_DATASET_PATH,
    EvaluationResult,
    evaluate_test,
    main,
    parse_args,
    run_evaluation,
)
from genai_template.schemas import DocumentChunk, RetrievalTest, RetrievedChunk


def test_parse_args_uses_defaults_and_accepts_overrides() -> None:
    """The CLI should expose stable defaults and explicit path overrides."""

    defaults = parse_args([])
    assert defaults.config == DEFAULT_CONFIG_PATH
    assert defaults.corpus == DEFAULT_CORPUS_DIR
    assert defaults.dataset == DEFAULT_DATASET_PATH
    assert defaults.reindex is False

    supplied = parse_args(
        [
            "--config",
            "experiment.yml",
            "--corpus",
            "corpus",
            "--dataset",
            "dataset.json",
            "--reindex",
        ]
    )
    assert supplied.config == Path("experiment.yml")
    assert supplied.corpus == Path("corpus")
    assert supplied.dataset == Path("dataset.json")
    assert supplied.reindex is True


def test_evaluate_test_returns_quality_and_runtime_metrics() -> None:
    """A retrieval case should report scores plus run-persistence metrics."""

    pipeline = MagicMock()
    pipeline.retrieve.return_value = [
        RetrievedChunk(
            chunk=DocumentChunk(
                id="one",
                document_id="one.md",
                text="Relevant",
                metadata={"file_path": "/corpus/one.md"},
            ),
            distance=0.2,
        )
    ]
    test = RetrievalTest(question="Question", expected_documents=["one.md"])

    result = evaluate_test(pipeline, test, 5)

    assert result.hit is True
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.retrieved_chunks == 1
    assert result.best_distance == 0.2
    assert result.worst_distance == 0.2
    assert result.retrieval_time >= 0


@patch("genai_template.evaluation.evaluators.baseline_eval.evaluate_test")
@patch("genai_template.evaluation.evaluators.baseline_eval.load_evaluation_tests")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_embedder")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_retrieval_pipeline")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_vector_store")
def test_run_evaluation_uses_real_registry_and_run_records(
    mock_create_store: MagicMock,
    mock_create_retrieval: MagicMock,
    _mock_create_embedder: MagicMock,
    mock_load_tests: MagicMock,
    mock_evaluate_test: MagicMock,
    tmp_path: Path,
) -> None:
    """Evaluation should bind source, config, experiment, and completed runs."""

    config = load_rag_config()
    corpus = tmp_path / "baseline"
    corpus.mkdir()
    source = Source(id=2, name="baseline", directory=str(corpus.resolve()))
    config_record = RagConfig(
        id=3, config_fingerprint="a" * 64, config_json=config.model_dump_json()
    )
    experiment = Experiment(id=4, source_id=2, name="Baseline")
    source_service = MagicMock()
    source_service.list_sources.return_value = [source]
    source_service.index_collection_name.return_value = "idx-selected"
    config_service = MagicMock()
    config_service.register_config.return_value = config_record
    experiment_service = MagicMock()
    experiment_service.create_experiment.return_value = experiment
    runs = [MagicMock(id=10), MagicMock(id=11)]
    experiment_service.start_run.side_effect = runs
    store = mock_create_store.return_value
    store.exists.return_value = True
    tests = [
        RetrievalTest(question="First", expected_documents=["one.md"]),
        RetrievalTest(question="Second", expected_documents=["two.md"]),
    ]
    mock_load_tests.return_value = tests
    mock_evaluate_test.side_effect = [
        EvaluationResult(True, 1.0, 0.5, 2, 0.1, 0.4, 0.01),
        EvaluationResult(False, 0.0, 0.0, 0, None, None, 0.02),
    ]

    run_evaluation(
        config,
        corpus,
        tmp_path / "dataset.json",
        source_service=source_service,
        config_service=config_service,
        experiment_service=experiment_service,
    )

    config_service.register_config.assert_called_once_with(config)
    experiment_service.create_experiment.assert_called_once_with(
        2, settings.EXPERIMENT_NAME, "Baseline retrieval evaluation."
    )
    assert experiment_service.start_run.call_args_list == [
        call(4, 3, "First"),
        call(4, 3, "Second"),
    ]
    assert experiment_service.complete_run.call_count == 2
    first_metrics = experiment_service.complete_run.call_args_list[0].args[1]
    assert first_metrics.retrieved_chunks == 2
    assert first_metrics.generation_time == 0
    source_service.rebuild_index.assert_not_called()
    mock_create_retrieval.assert_called_once()


@patch("genai_template.evaluation.evaluators.baseline_eval.load_evaluation_tests")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_vector_store")
def test_run_evaluation_builds_missing_deterministic_index(
    mock_create_store: MagicMock,
    mock_load_tests: MagicMock,
    tmp_path: Path,
) -> None:
    """A missing index should be built through the source lifecycle service."""

    config = load_rag_config()
    corpus = tmp_path / "baseline"
    corpus.mkdir()
    source = Source(id=2, name="baseline", directory=str(corpus.resolve()))
    source_service = MagicMock()
    source_service.list_sources.return_value = [source]
    source_service.index_collection_name.return_value = "idx-selected"
    config_service = MagicMock()
    config_service.register_config.return_value = RagConfig(
        id=3, config_fingerprint="a" * 64, config_json=config.model_dump_json()
    )
    experiment_service = MagicMock()
    experiment_service.create_experiment.return_value = Experiment(
        id=4, source_id=2, name="Baseline"
    )
    missing_store = MagicMock()
    rebuilt_store = MagicMock()
    missing_store.exists.return_value = False
    mock_create_store.side_effect = [missing_store, rebuilt_store]
    mock_load_tests.return_value = []

    with pytest.raises(ValueError, match="contains no tests"):
        run_evaluation(
            config,
            corpus,
            tmp_path / "dataset.json",
            source_service=source_service,
            config_service=config_service,
            experiment_service=experiment_service,
        )

    source_service.rebuild_index.assert_called_once_with(2, 3)


@patch("genai_template.evaluation.evaluators.baseline_eval.run_evaluation")
@patch("genai_template.evaluation.evaluators.baseline_eval.load_rag_config")
def test_main_loads_config_and_forwards_paths(
    mock_load_config: MagicMock,
    mock_run_evaluation: MagicMock,
) -> None:
    """The CLI should delegate registry and run creation to evaluation."""

    config = mock_load_config.return_value
    main(
        [
            "--config",
            "experiment.yml",
            "--corpus",
            "corpus",
            "--dataset",
            "dataset.json",
            "--reindex",
        ]
    )

    mock_load_config.assert_called_once_with(Path("experiment.yml"))
    mock_run_evaluation.assert_called_once_with(
        config=config,
        corpus_path=Path("corpus"),
        dataset_path=Path("dataset.json"),
        reindex=True,
    )
