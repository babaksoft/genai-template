import streamlit as st

st.set_page_config(
    page_title="GenAI Template",
    page_icon="🤖",
    layout="wide",
)

st.title("GenAI Template")
st.write("RAG experimentation playground")

st.header("Ask a question")

query = st.text_area(
    "Question",
    placeholder="Enter your question...",
    height=100,
)

if st.button("Ask", type="primary"):
    st.info("Answer submission will be implemented in the next slice.")
