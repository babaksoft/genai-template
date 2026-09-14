"""Streamlit UI for registered experiments and RAG configurations."""

import streamlit as st
from httpx import HTTPError

from genai_template.config import load_rag_config, settings
from genai_template.schemas import ExperimentResponse, RagConfigResponse
from genai_template.ui.api_client import ApiClient

st.set_page_config(page_title="GenAI Template", page_icon="🤖", layout="wide")
st.title("GenAI Template")
st.write("RAG experimentation playground")

api_client = ApiClient(base_url=settings.API_BASE_URL)
active_experiment: ExperimentResponse | None = None
active_config: RagConfigResponse | None = None

with st.sidebar:
    st.header("Experiment setup")
    try:
        candidates = api_client.list_source_candidates()
        sources = api_client.list_sources()
        experiments = api_client.list_experiments()
        rag_configs = api_client.list_rag_configs()
    except HTTPError as exc:
        st.error(f"Unable to load registry data from the API: {exc}")
        candidates, sources, experiments, rag_configs = [], [], [], []

    with st.expander("Register source"):
        if candidates:
            directory = st.selectbox(
                "Corpus directory",
                [candidate.name for candidate in candidates],
                key="source_directory",
            )
            if st.button("Register source"):
                try:
                    registered = api_client.register_source(directory)
                except HTTPError as exc:
                    st.error(f"Unable to register source: {exc}")
                else:
                    st.success(f"Registered {registered.name}.")
                    st.rerun()
        else:
            st.info("No unregistered corpus directories are available.")

    with st.expander("Create experiment"):
        if sources:
            sources_by_label = {
                f"{source.id}: {source.name}": source for source in sources
            }
            source_label = st.selectbox("Source", list(sources_by_label))
            experiment_name = st.text_input("Experiment name")
            description = st.text_area("Description (optional)")
            if st.button("Create experiment", disabled=not experiment_name.strip()):
                try:
                    created = api_client.create_experiment(
                        sources_by_label[source_label].id,
                        experiment_name.strip(),
                        description.strip() or None,
                    )
                except HTTPError as exc:
                    st.error(f"Unable to create experiment: {exc}")
                else:
                    st.session_state.active_experiment_id = created.id
                    st.success(f"Created experiment {created.id}.")
                    st.rerun()
        else:
            st.info("Register a source before creating an experiment.")

    with st.expander("Register RAG configuration"):
        config_paths = sorted(settings.EXPERIMENT_CONFIG_DIR.glob("*.yml"))
        if config_paths:
            configs_by_name = {path.name: path for path in config_paths}
            config_name = st.selectbox(
                "Configuration file",
                list(configs_by_name),
                key="rag_config_file",
            )
            if st.button("Register RAG configuration"):
                try:
                    config = load_rag_config(configs_by_name[config_name])
                    registered_config = api_client.register_rag_config(config)
                except (OSError, ValueError, HTTPError) as exc:
                    st.error(f"Unable to register RAG configuration: {exc}")
                else:
                    st.session_state.active_rag_config_id = registered_config.id
                    st.success(f"Registered RAG configuration {registered_config.id}.")
                    st.rerun()
        else:
            st.info("No RAG configuration files are available.")

    if experiments:
        experiments_by_label = {f"{item.id}: {item.name}": item for item in experiments}
        selected_experiment = st.selectbox(
            "Experiment",
            list(experiments_by_label),
            index=next(
                (
                    index
                    for index, item in enumerate(experiments_by_label.values())
                    if item.id == st.session_state.get("active_experiment_id")
                ),
                0,
            ),
        )
        active_experiment = experiments_by_label[selected_experiment]
        st.session_state.active_experiment_id = active_experiment.id
    else:
        st.info("Create an experiment before asking questions.")

    if rag_configs:
        configs_by_label = {
            f"{item.id}: {item.config.embedder.model_name} / "
            f"{item.config.llm.model_name}": item
            for item in rag_configs
        }
        selected_config = st.selectbox(
            "RAG configuration",
            list(configs_by_label),
            index=next(
                (
                    index
                    for index, item in enumerate(configs_by_label.values())
                    if item.id == st.session_state.get("active_rag_config_id")
                ),
                0,
            ),
        )
        active_config = configs_by_label[selected_config]
        st.session_state.active_rag_config_id = active_config.id
    else:
        st.info("Register a RAG configuration before asking questions.")

    if (
        active_experiment is not None
        and active_config is not None
        and st.button("Rebuild selected index")
    ):
        with st.spinner("Rebuilding index..."):
            try:
                build_result = api_client.rebuild_index(
                    active_experiment.source_id,
                    active_config.id,
                )
            except HTTPError as exc:
                st.error(f"Unable to rebuild index: {exc}")
            else:
                st.success(
                    f"Indexed {build_result.documents_indexed} document(s) into "
                    f"{build_result.chunks_indexed} chunk(s)."
                )

st.header("Ask a question")
query = st.text_area("Question", placeholder="Enter your question...", height=100)
can_ask = active_experiment is not None and active_config is not None

if st.button("Ask", type="primary", disabled=not can_ask):
    if active_experiment is None or active_config is None:
        st.warning("Select an experiment and RAG configuration first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer_result = api_client.answer(
                    query.strip(), active_experiment.id, active_config.id
                )
            except HTTPError as exc:
                st.error(f"Unable to get an answer from the API: {exc}")
            else:
                st.subheader("Answer")
                st.write(answer_result.answer)
                with st.sidebar:
                    st.header("Execution Metrics")
                    with st.expander("Timing", expanded=True):
                        st.metric(
                            "Retrieval",
                            f"{answer_result.metrics.retrieval_time:.3f} s",
                        )
                        st.metric(
                            "Generation",
                            f"{answer_result.metrics.generation_time:.3f} s",
                        )
                        st.metric("Total", f"{answer_result.metrics.total_time:.3f} s")
                    with st.expander("Retrieval"):
                        st.metric(
                            "Retrieved chunks", answer_result.metrics.retrieved_chunks
                        )
                        st.metric(
                            "Best distance",
                            (
                                "N/A"
                                if answer_result.metrics.best_distance is None
                                else f"{answer_result.metrics.best_distance:.4f}"
                            ),
                        )
                        st.metric(
                            "Worst distance",
                            (
                                "N/A"
                                if answer_result.metrics.worst_distance is None
                                else f"{answer_result.metrics.worst_distance:.4f}"
                            ),
                        )
                    with st.expander("Request"):
                        st.metric(
                            "Context length", answer_result.metrics.context_length
                        )
                        st.metric("Prompt length", answer_result.metrics.prompt_length)
                        st.metric(
                            "Response length", answer_result.metrics.response_length
                        )
