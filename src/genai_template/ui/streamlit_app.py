import httpx
import streamlit as st

from genai_template.config import settings
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

st.header("Ask a question")

query = st.text_area(
    "Question",
    placeholder="Enter your question...",
    height=100,
)

if st.button("Ask", type="primary"):
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
