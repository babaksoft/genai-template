import httpx
import streamlit as st

from genai_template.config import settings
from genai_template.schemas import SourceResponse
from genai_template.ui.api_client import ApiClient

st.set_page_config(
    page_title="GenAI Template",
    page_icon="🤖",
    layout="wide",
)

st.title("GenAI Template")
st.write("RAG experimentation playground")

api_client = ApiClient(
    base_url=settings.API_BASE_URL,
)

active_source: SourceResponse | None = None

with st.sidebar:
    st.header("Sources")

    with st.expander("Sources", expanded=True):
        try:
            candidates = api_client.list_source_candidates()
            sources = api_client.list_sources()
        except httpx.HTTPError as exc:
            st.error(f"Unable to load sources from the API: {exc}")
            candidates = []
            sources = []

        if candidates:
            candidate_names = [candidate.name for candidate in candidates]
            directory = st.selectbox(
                "Corpus directory",
                candidate_names,
                key="source_directory",
            )
            if st.button("Ingest source"):
                with st.spinner("Indexing corpus..."):
                    try:
                        ingested = api_client.ingest_source(directory)
                    except httpx.HTTPError as exc:
                        st.error(f"Unable to ingest source: {exc}")
                    else:
                        st.session_state.active_source_name = ingested.name
                        st.success(
                            f"Ingested {ingested.name}: "
                            f"{ingested.documents_indexed} document(s), "
                            f"{ingested.chunks_indexed} chunk(s)."
                        )
                        st.rerun()
        else:
            st.info("No prepared corpus directories are available.")

        if sources:
            sources_by_name = {source.name: source for source in sources}
            if st.session_state.get("active_source_name") not in sources_by_name:
                st.session_state.active_source_name = sources[0].name

            selected_source_name = st.selectbox(
                "Active source",
                options=list(sources_by_name),
                key="active_source_name",
            )
            active_source = sources_by_name[selected_source_name]
            st.caption(
                f"{active_source.documents_indexed} document(s) · "
                f"{active_source.chunks_indexed} chunk(s)"
            )
        else:
            st.info("Ingest a corpus to make it an active source.")

st.header("Ask a question")

query = st.text_area(
    "Question",
    placeholder="Enter your question...",
    height=100,
)

if st.button("Ask", type="primary", disabled=active_source is None):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                result = api_client.answer(query.strip())
            except httpx.HTTPError as exc:
                st.error(f"Unable to get an answer from the API: {exc}")
            else:
                st.subheader("Answer")
                st.write(result.answer)

                with st.sidebar:
                    st.header("Execution Metrics")

                    with st.expander("Timing", expanded=True):
                        st.metric(
                            "Retrieval",
                            f"{result.metrics.retrieval_time:.3f} s",
                        )
                        st.metric(
                            "Generation",
                            f"{result.metrics.generation_time:.3f} s",
                        )
                        st.metric(
                            "Total",
                            f"{result.metrics.total_time:.3f} s",
                        )

                    with st.expander("Retrieval"):
                        st.metric(
                            "Retrieved chunks",
                            result.metrics.retrieved_chunks,
                        )
                        st.metric(
                            "Best distance",
                            f"{result.metrics.best_distance:.4f}",
                        )
                        st.metric(
                            "Worst distance",
                            f"{result.metrics.worst_distance:.4f}",
                        )

                    with st.expander("Request"):
                        st.metric(
                            "Context length",
                            result.metrics.context_length,
                        )
                        st.metric(
                            "Prompt length",
                            result.metrics.prompt_length,
                        )
                        st.metric(
                            "Response length",
                            result.metrics.response_length,
                        )
