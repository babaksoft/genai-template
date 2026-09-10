"""Tests for the YAML-driven retrieval evaluation workflow."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from genai_template.config import settings
from genai_template.config.rag import load_rag_config
from genai_template.evaluation.evaluators.baseline_eval import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CORPUS_DIR,
    DEFAULT_DATASET_PATH,
    _config_fingerprint,
    _evaluation_collection_name,
    main,
    parse_args,
    run_evaluation,
)
from genai_template.schemas import RetrievalTest


def test_parse_args_uses_baseline_defaults() -> None:
    """Omitted CLI arguments should retain the baseline paths."""

    args = parse_args([])

    assert args.config == settings.EXPERIMENT_CONFIG_DIR / "baseline.yml"
    assert args.config == DEFAULT_CONFIG_PATH
    assert args.corpus == settings.CORPORA_DIR / "baseline"
    assert args.corpus == DEFAULT_CORPUS_DIR
    assert args.dataset == settings.EVALUATION_DATA_DIR / "baseline-eval.json"
    assert args.dataset == DEFAULT_DATASET_PATH


def test_parse_args_accepts_all_path_overrides() -> None:
    """Every Slice 3 path option should be accepted by the CLI."""

    args = parse_args(
        [
            "--config",
            "experiment.yml",
            "--corpus",
            "corpus",
            "--dataset",
            "dataset.json",
        ]
    )

    assert args.config == Path("experiment.yml")
    assert args.corpus == Path("corpus")
    assert args.dataset == Path("dataset.json")


@patch("genai_template.evaluation.evaluators.baseline_eval.run_evaluation")
@patch("genai_template.evaluation.evaluators.baseline_eval.load_rag_config")
def test_main_loads_config_and_forwards_paths(
    mock_load_config: MagicMock,
    mock_run_evaluation: MagicMock,
) -> None:
    """The CLI should load YAML and forward all resolved arguments."""

    config = MagicMock()
    mock_load_config.return_value = config

    main(
        [
            "--config",
            "experiment.yml",
            "--corpus",
            "corpus",
            "--dataset",
            "dataset.json",
        ]
    )

    mock_load_config.assert_called_once_with(Path("experiment.yml"))
    mock_run_evaluation.assert_called_once_with(
        config=config,
        corpus_path=Path("corpus"),
        dataset_path=Path("dataset.json"),
    )


@patch("genai_template.evaluation.evaluators.baseline_eval.evaluate_test")
@patch("genai_template.evaluation.evaluators.baseline_eval.load_evaluation_tests")
@patch("genai_template.evaluation.evaluators.baseline_eval.IndexingPipeline")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_retrieval_pipeline")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_splitter")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_embedder")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_vector_store")
def test_run_evaluation_rebuilds_and_uses_configured_components(
    mock_create_store: MagicMock,
    mock_create_embedder: MagicMock,
    mock_create_splitter: MagicMock,
    mock_create_retrieval: MagicMock,
    mock_indexing_pipeline: MagicMock,
    mock_load_tests: MagicMock,
    mock_evaluate_test: MagicMock,
    tmp_path: Path,
) -> None:
    """Evaluation should rebuild its collection and use factory components."""

    config = load_rag_config()
    old_store = MagicMock()
    rebuilt_store = MagicMock()
    mock_create_store.side_effect = [old_store, rebuilt_store]
    embedder = mock_create_embedder.return_value
    splitter = mock_create_splitter.return_value
    retrieval = mock_create_retrieval.return_value
    test_one = RetrievalTest(
        question="First question",
        expected_documents=["one.md"],
    )
    test_two = RetrievalTest(
        question="Second question",
        expected_documents=["two.md"],
    )
    mock_load_tests.return_value = [test_one, test_two]
    mock_evaluate_test.side_effect = [(True, 1.0, 0.5), (False, 0.5, 0.25)]
    corpus_path = tmp_path / "corpus"
    dataset_path = tmp_path / "dataset.json"

    run_evaluation(config, corpus_path, dataset_path)

    fingerprint = _config_fingerprint(config)
    expected_collection = _evaluation_collection_name(fingerprint)
    store_calls = mock_create_store.call_args_list
    assert len(store_calls) == 2
    assert store_calls[0].args[0].collection_name == expected_collection
    assert store_calls[1].args[0].collection_name == expected_collection
    assert config.vector_store.collection_name == settings.CHROMA_COLLECTION
    old_store.delete.assert_called_once_with()
    mock_create_embedder.assert_called_once_with(config.embedder)
    mock_create_splitter.assert_called_once_with(config.splitter)
    mock_indexing_pipeline.assert_called_once_with(
        splitter=splitter,
        embedder=embedder,
        store=rebuilt_store,
    )
    mock_indexing_pipeline.return_value.run.assert_called_once_with(corpus_path)
    mock_create_retrieval.assert_called_once_with(
        config.retrieval,
        embedder,
        rebuilt_store,
    )
    mock_load_tests.assert_called_once_with(dataset_path)
    assert mock_evaluate_test.call_args_list == [
        call(pipeline=retrieval, test=test_one, k=config.retrieval.top_k),
        call(pipeline=retrieval, test=test_two, k=config.retrieval.top_k),
    ]


@patch("genai_template.evaluation.evaluators.baseline_eval.evaluate_test")
@patch("genai_template.evaluation.evaluators.baseline_eval.load_evaluation_tests")
@patch("genai_template.evaluation.evaluators.baseline_eval.IndexingPipeline")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_retrieval_pipeline")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_splitter")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_embedder")
@patch("genai_template.evaluation.evaluators.baseline_eval.create_vector_store")
def test_run_evaluation_logs_identity_and_metrics(
    mock_create_store: MagicMock,
    _mock_create_embedder: MagicMock,
    _mock_create_splitter: MagicMock,
    _mock_create_retrieval: MagicMock,
    _mock_indexing_pipeline: MagicMock,
    mock_load_tests: MagicMock,
    mock_evaluate_test: MagicMock,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Completion logs should identify the experiment, index, and metrics."""

    mock_create_store.side_effect = [MagicMock(), MagicMock()]
    test = RetrievalTest(question="Question", expected_documents=["one.md"])
    mock_load_tests.return_value = [test]
    mock_evaluate_test.return_value = (True, 0.5, 0.25)
    config = load_rag_config()
    fingerprint = _config_fingerprint(config)
    collection_name = _evaluation_collection_name(fingerprint)
    caplog.set_level(
        logging.INFO,
        logger="genai_template.evaluation.evaluators.baseline_eval",
    )

    run_evaluation(config, tmp_path / "corpus", tmp_path / "dataset.json")

    completion_message = caplog.messages[-1]
    assert f"experiment='{config.experiment.name}'" in completion_message
    assert f"config_fingerprint={fingerprint}" in completion_message
    assert f"collection='{collection_name}'" in completion_message
    assert "hit@5=1.000, recall@5=0.500, precision@5=0.250" in completion_message
